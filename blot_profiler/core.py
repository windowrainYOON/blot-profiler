"""Image processing, band detection and quantification for western blots.

Everything here is GUI-independent so it can be tested and scripted.

Conventions
-----------
* The *signal* image is a float array where bands are bright (high values) and
  the membrane is dark. Typical chemiluminescence scans (dark bands on a light
  background) are inverted automatically.
* A band box is ``(x, y, w, h)`` in image pixel coordinates (top-left origin).
* Bands that sit on the same horizontal line form a *row* (one protein). All
  boxes in a row share the same width and height so every lane is measured
  with an identical area.
* Rows (protein groups) and lanes (lane groups) carry the user's names; a band's
  ID is built from them, e.g. ``Kras_IgG_12h``. Several physical lanes may share
  a lane name: they are replicates of that lane group.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------
# Image loading / preprocessing
# --------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load a TIF (8/16/32 bit, grayscale or RGB) as a 2-D float64 array."""
    arr = None
    try:
        import tifffile

        arr = tifffile.imread(path)
    except Exception:
        arr = None
    if arr is None:
        from PIL import Image

        with Image.open(path) as im:
            arr = np.asarray(im)
    arr = np.asarray(arr)
    # Multi-page stacks: take the first plane.
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):  # RGB(A), channels last
            arr = arr[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        elif arr.shape[0] in (3, 4):  # channels first
            arr = np.tensordot(np.array([0.299, 0.587, 0.114]), arr[:3].astype(np.float64), axes=1)
        else:  # stack of planes
            arr = arr[0]
    return arr.astype(np.float64)


def needs_inversion(img: np.ndarray) -> bool:
    """True when bands are dark on a light background (the usual film/ECL look)."""
    return float(np.median(img)) > float(img.min() + img.max()) / 2.0


def to_signal(img: np.ndarray, invert: Optional[bool] = None) -> np.ndarray:
    """Return an image where bands are bright, keeping the original intensity scale."""
    if invert is None:
        invert = needs_inversion(img)
    if invert:
        top = _dtype_max(img)
        return top - img
    return img.copy()


def _dtype_max(img: np.ndarray) -> float:
    m = float(img.max())
    for top in (255.0, 65535.0):
        if m <= top:
            return top
    return m


def rolling_ball_background(signal: np.ndarray, radius: int) -> np.ndarray:
    """Estimate a smooth background with the rolling-ball algorithm."""
    from skimage.restoration import rolling_ball

    radius = max(1, int(radius))
    # Work on a lightly smoothed copy so the ball does not ride on noise minima
    # (which would leave a positive bias in empty areas).
    signal = ndi.gaussian_filter(signal, 2.0)
    # Rolling ball is slow on big images; work on a downscaled copy if needed.
    scale = 1
    while max(signal.shape) / scale > 1200 and radius / scale > 8:
        scale *= 2
    if scale > 1:
        small = signal[::scale, ::scale]
        bg_small = rolling_ball(small, radius=radius / scale)
        bg = ndi.zoom(bg_small, (signal.shape[0] / small.shape[0], signal.shape[1] / small.shape[1]), order=1)
        bg = bg[: signal.shape[0], : signal.shape[1]]
        bg = np.minimum(bg, signal)
    else:
        bg = rolling_ball(signal, radius=radius)
    return bg


BACKGROUND_METHODS = ("none", "rolling_ball", "local_border", "row_median")


@dataclass
class BackgroundSettings:
    method: str = "rolling_ball"  # one of BACKGROUND_METHODS
    radius: int = 50  # rolling-ball radius (px)
    border: int = 3  # thickness of the strips above/below each box for local_border (px)


def corrected_signal(signal: np.ndarray, bg: BackgroundSettings) -> np.ndarray:
    """Apply image-wide background removal (rolling ball). Others are per box."""
    if bg.method == "rolling_ball":
        return signal - rolling_ball_background(signal, bg.radius)
    return signal


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

ROLE_TARGET = "target"
ROLE_CONTROL = "control"
ROLE_IGNORE = "ignore"
ROLES = (ROLE_TARGET, ROLE_CONTROL, ROLE_IGNORE)


@dataclass
class Row:
    """A horizontal band row = one protein. All its boxes share width/height."""

    name: str
    width: int
    height: int
    role: str = ROLE_TARGET


@dataclass
class Band:
    row: int  # index into rows
    lane: int  # 0-based physical lane index
    cx: float  # box centre x
    cy: float  # box centre y

    def box(self, rows: Sequence[Row]):
        r = rows[self.row]
        x = int(round(self.cx - r.width / 2))
        y = int(round(self.cy - r.height / 2))
        return x, y, int(r.width), int(r.height)


@dataclass
class Analysis:
    rows: List[Row] = field(default_factory=list)  # protein groups
    bands: List[Band] = field(default_factory=list)
    lane_labels: List[str] = field(default_factory=list)  # lane group name of each physical lane
    named_lanes: List[str] = field(default_factory=list)  # lane group names the user gave (kept by auto lanes)
    formulas: List[Dict[str, str]] = field(default_factory=list)  # {name, expr, normalize}

    @property
    def n_lanes(self) -> int:
        n = max((b.lane for b in self.bands), default=-1) + 1
        return max(n, len(self.lane_labels))

    def ensure_lane_labels(self):
        n = self.n_lanes
        used = set(self.lane_labels)
        k = 1
        while len(self.lane_labels) < n:
            while f"L{k}" in used:
                k += 1
            self.lane_labels.append(f"L{k}")
            used.add(f"L{k}")

    def lane_groups(self) -> List[str]:
        """Lane group names in left-to-right order (each once)."""
        out = []
        for name in self.lane_labels:
            if name not in out:
                out.append(name)
        return out

    def band_ids(self) -> Dict[int, str]:
        """``id(band) -> ID`` built from the protein and lane group names.

        Bands sharing a protein and lane group (replicate lanes, or several
        bands of one protein in a lane) get ``_1``, ``_2`` … suffixes.
        """
        self.ensure_lane_labels()
        base = {}
        for b in self.bands:
            base[id(b)] = f"{_safe_name(self.rows[b.row].name)}_{_safe_name(self.lane_labels[b.lane])}"
        count: Dict[str, int] = {}
        for v in base.values():
            count[v] = count.get(v, 0) + 1
        # unique IDs first, then numbered duplicates (skipping numbers already taken)
        out = {k: v for k, v in base.items() if count[v] == 1}
        used = set(out.values())
        nxt: Dict[str, int] = {}
        for b in sorted(self.bands, key=lambda b: (b.row, b.lane, b.cy)):
            v = base[id(b)]
            if count[v] == 1:
                continue
            k = nxt.get(v, 1)
            while f"{v}_{k}" in used:
                k += 1
            out[id(b)] = f"{v}_{k}"
            used.add(out[id(b)])
            nxt[v] = k + 1
        return out

    def band_id(self, band: Band) -> str:
        return self.band_ids()[id(band)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Analysis":
        a = cls()
        a.rows = [Row(**r) for r in d.get("rows", [])]
        a.bands = [Band(**{k: b[k] for k in ("row", "lane", "cx", "cy")}) for b in d.get("bands", [])]
        a.lane_labels = list(d.get("lane_labels", []))
        a.named_lanes = list(d.get("named_lanes", []))
        a.formulas = [dict(f) for f in d.get("formulas", [])]
        return a

    def remove_row(self, idx: int):
        self.bands = [b for b in self.bands if b.row != idx]
        for b in self.bands:
            if b.row > idx:
                b.row -= 1
        del self.rows[idx]

    def sort(self):
        """Sort rows top->bottom and renumber lanes left->right."""
        order = sorted(range(len(self.rows)), key=lambda i: self._row_y(i))
        remap = {old: new for new, old in enumerate(order)}
        self.rows = [self.rows[i] for i in order]
        for b in self.bands:
            b.row = remap[b.row]
        self.bands.sort(key=lambda b: (b.row, b.lane, b.cx))

    def _row_y(self, idx):
        ys = [b.cy for b in self.bands if b.row == idx]
        return float(np.mean(ys)) if ys else 1e9


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

@dataclass
class DetectSettings:
    invert: Optional[bool] = None  # None = auto
    smooth_sigma: float = 1.5
    sensitivity: float = 1.0  # >1 finds fainter bands, <1 only strong ones
    min_area: int = 30
    pad: int = 3  # padding added around detected bands (px)
    exclude_edge: int = 0  # ignore objects within this many px of left/right edge (ladder)
    fill_missing_lanes: bool = True  # add boxes in rows where a lane has no detected band
    snap_to_lanes: bool = True  # centre every box horizontally on its lane
    refine_vertical: bool = False  # slide each box up/down onto its band
    # Detection works on a contrast-adjusted copy (percentile stretch + gamma) so
    # faint bands are found and boxed; measurement never uses this copy.
    lut_gamma: float = 0.6  # 1.0 = linear (no LUT)
    lut_low: float = 0.5  # percentile mapped to 0
    lut_high: float = 99.8  # percentile mapped to 1


def detection_image(signal: np.ndarray, settings: "DetectSettings") -> np.ndarray:
    """Contrast-stretched working copy used only to find bands (gamma < 1 lifts faint bands)."""
    lo, hi = np.percentile(signal, [settings.lut_low, settings.lut_high])
    x = np.clip((signal - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    g = float(settings.lut_gamma)
    return x if abs(g - 1.0) < 1e-9 else x ** g


def _threshold(img: np.ndarray, sensitivity: float) -> float:
    """Noise-based threshold: membrane median + k * robust noise, k shrinks with sensitivity."""
    from skimage.filters import threshold_otsu

    vals = img[np.isfinite(img)]
    if vals.size == 0 or vals.max() <= vals.min():
        return float("inf")
    t_otsu = threshold_otsu(vals)
    low = vals[vals < t_otsu]
    if low.size < 10:
        return float(t_otsu)
    med = float(np.median(low))
    mad = float(np.median(np.abs(low - med))) * 1.4826
    span = float(np.percentile(vals, 99.9)) - med
    noise = max(mad, 0.01 * span, 1e-9)
    return med + 8.0 * noise / max(sensitivity, 1e-3)


def _remove_lines(mask: np.ndarray) -> np.ndarray:
    """Remove straight lines that cross (almost) the whole image, e.g. panel borders."""
    H, W = mask.shape
    mask = mask.copy()
    # include the blurred edges of each line (+-2 px)
    rows = ndi.binary_dilation(mask.mean(axis=1) > 0.92, iterations=2)
    cols = ndi.binary_dilation(mask.mean(axis=0) > 0.92, iterations=2)
    mask[rows, :] = False
    mask[:, cols] = False
    return mask


def detect_bands(signal: np.ndarray, settings: DetectSettings, row_names: Optional[List[str]] = None) -> Analysis:
    """Find bands, group them into rows and lanes and give each row a common box size."""
    from skimage.measure import label, regionprops
    from skimage.morphology import opening

    work = detection_image(signal, settings)
    img = ndi.gaussian_filter(work, settings.smooth_sigma) if settings.smooth_sigma > 0 else work
    t = _threshold(img, settings.sensitivity)
    mask = _remove_lines(img > t)
    # Remove thin specks and scratches.
    mask = opening(mask, np.ones((3, 3), bool))
    lab = label(mask)
    H, W = signal.shape
    regs = []
    for rp in regionprops(lab, intensity_image=img):
        if rp.area < max(1, settings.min_area):
            continue
        y0, x0, y1, x1 = rp.bbox
        w, h = x1 - x0, y1 - y0
        if h <= max(6, 0.02 * H) and w > 0.6 * W:  # thin line across the membrane: panel border
            continue
        if h > 0.5 * H:
            continue
        # Bands of neighbouring lanes often touch and form one long object: split it
        # at the valleys of its column intensity profile.
        sub = np.where(lab[y0:y1, x0:x1] == rp.label, img[y0:y1, x0:x1], 0.0)
        for sx0, sx1 in _split_columns(sub, h):
            ssub = sub[:, sx0:sx1]
            npx = int((ssub > 0).sum())
            if npx < max(1, settings.min_area) // 2:
                continue
            ys = np.nonzero(ssub.sum(axis=1) > 0)[0]
            wy = ssub.sum(axis=1)
            cy = y0 + float((np.arange(h) * wy).sum() / max(wy.sum(), 1e-12))
            ax0, ax1 = x0 + sx0, x0 + sx1
            if settings.exclude_edge and (ax0 < settings.exclude_edge or ax1 > W - settings.exclude_edge):
                continue
            regs.append(dict(x0=ax0, x1=ax1, y0=y0 + int(ys.min()), y1=y0 + int(ys.max()) + 1, cx=(ax0 + ax1) / 2, cy=cy, area=npx))
    regs = _split_by_pitch(regs)
    return _build_analysis(regs, settings, row_names, signal)


def _estimate_pitch(regs) -> float:
    """Typical lane spacing: median gap between horizontally adjacent bands of one row."""
    if len(regs) < 2:
        return 0.0
    heights = np.array([r["y1"] - r["y0"] for r in regs], float)
    rows = _cluster_1d([r["cy"] for r in regs], max(2.0, 0.6 * float(np.median(heights))))
    gaps = []
    for c in rows:
        xs = sorted(regs[i]["cx"] for i in c)
        gaps += list(np.diff(xs))
    return float(np.median(gaps)) if gaps else 0.0


def _split_by_pitch(regs):
    """Split objects spanning several lanes into equal lane-sized pieces."""
    pitch = _estimate_pitch(regs)
    if pitch <= 0:
        return regs
    out = []
    for r in regs:
        w = r["x1"] - r["x0"]
        n = int(round(w / pitch))
        if n >= 2 and w > 1.5 * pitch:
            step = w / n
            for k in range(n):
                x0 = r["x0"] + k * step
                out.append(dict(r, x0=x0, x1=x0 + step, cx=x0 + step / 2, area=r["area"] / n))
        else:
            out.append(r)
    return out


def _split_columns(sub: np.ndarray, h: int):
    """Split a region horizontally at intensity valleys; returns [(x0, x1), ...]."""
    from scipy.signal import find_peaks

    w = sub.shape[1]
    prof = ndi.gaussian_filter1d(sub.sum(axis=0), max(1.0, h / 4))
    if w < 2.5 * max(h, 4) or prof.max() <= 0:
        return [(0, w)]
    peaks, _ = find_peaks(prof, distance=max(3, int(1.2 * h)), prominence=0.05 * prof.max())
    if len(peaks) < 2:
        return [(0, w)]
    cuts = [0]
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        cuts.append(int(p0 + np.argmin(prof[p0:p1])))
    cuts.append(w)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _cluster_1d(values: Sequence[float], gap: float) -> List[List[int]]:
    order = np.argsort(values)
    clusters: List[List[int]] = []
    last = None
    for i in order:
        v = values[i]
        if last is None or v - last > gap:
            clusters.append([int(i)])
        else:
            clusters[-1].append(int(i))
        last = v
    return clusters


def _linear_extent(signal: np.ndarray, m, frac: float = 0.1) -> int:
    """Vertical extent of one band on the linear signal: first to last row whose
    lane-mean rises above ``frac`` of the band's peak (over the local baseline)."""
    H, W = signal.shape
    x0, x1 = int(max(0, m["x0"])), int(min(W, m["x1"]))
    hh = m["y1"] - m["y0"]
    y0, y1 = int(max(0, m["y0"] - hh // 2)), int(min(H, m["y1"] + hh // 2))
    if x1 - x0 < 2 or y1 - y0 < 3:
        return 0
    # middle half of the band's width avoids the rounded band ends
    q = (x1 - x0) // 4
    prof = signal[y0:y1, x0 + q:x1 - q if x1 - q > x0 + q else x1].mean(axis=1)
    base = float(np.percentile(prof, 10))
    peak_i = int(np.argmax(prof))
    peak = float(prof[peak_i]) - base
    if peak <= 0:
        return 0
    # first to last row above the level: keeps both bands of a doublet together
    idx = np.nonzero(prof - base > frac * peak)[0]
    return int(idx.max() - idx.min() + 1)


def _build_analysis(regs, settings: DetectSettings, row_names, signal: np.ndarray, _refiltered: bool = False) -> Analysis:
    a = Analysis()
    if not regs:
        return a
    # ---- rows: cluster by vertical centre
    heights = np.array([r["y1"] - r["y0"] for r in regs], float)
    row_gap = max(2.0, 0.6 * float(np.median(heights)))
    row_clusters = _cluster_1d([r["cy"] for r in regs], row_gap)
    row_clusters.sort(key=lambda c: np.mean([regs[i]["cy"] for i in c]))

    # ---- lanes: cluster by horizontal centre across all rows
    widths = np.array([r["x1"] - r["x0"] for r in regs], float)
    lane_gap = max(2.0, 0.5 * float(np.median(widths)))
    lane_clusters = _cluster_1d([r["cx"] for r in regs], lane_gap)
    lane_clusters.sort(key=lambda c: np.mean([regs[i]["cx"] for i in c]))
    lane_of = {}
    lane_x = []
    for li, c in enumerate(lane_clusters):
        lane_x.append(float(np.median([regs[i]["cx"] for i in c])))
        for i in c:
            lane_of[i] = li
    # rows present in only a few lanes (specks, smears) while others span many lanes
    # are noise: drop them and cluster the lanes again without them
    nl = [len({lane_of[i] for i in c}) for c in row_clusters]
    if not _refiltered and max(nl) >= 4:
        keep = [c for c, n in zip(row_clusters, nl) if n >= max(2, 0.3 * max(nl))]
        if len(keep) < len(row_clusters):
            return _build_analysis([regs[i] for c in keep for i in c], settings, row_names, signal, _refiltered=True)

    for ri, c in enumerate(row_clusters):
        members = [regs[i] for i in c]
        w = int(np.ceil(np.median([m["x1"] - m["x0"] for m in members]))) + 2 * settings.pad
        # height from the *linear* signal (the detection LUT exaggerates the faint tails)
        hs = [_linear_extent(signal, m) for m in members]
        hs = [x for x in hs if x] or [m["y1"] - m["y0"] for m in members]
        h = int(np.ceil(np.median(hs))) + 2 * settings.pad
        # Boxes cover most of a lane but never overlap the neighbouring lane.
        if len(lane_x) > 1:
            pitch = float(np.median(np.diff(lane_x)))
            w = int(min(max(w, 0.7 * pitch), 0.95 * pitch))
        name = row_names[ri] if row_names and ri < len(row_names) else f"P{ri + 1}"
        a.rows.append(Row(name=name, width=max(w, 4), height=max(h, 4)))
        seen = {}
        for i in c:
            r = regs[i]
            lane = lane_of[i]
            # If two objects of the same row fall in one lane (rare), keep the larger one.
            if lane in seen and seen[lane]["area"] >= r["area"]:
                continue
            seen[lane] = r
        for lane, r in seen.items():
            a.bands.append(Band(row=ri, lane=lane, cx=float(r["cx"]), cy=float(r["cy"])))
    if settings.snap_to_lanes:
        for b in a.bands:
            b.cx = lane_x[b.lane]
    a.ensure_lane_labels()
    if settings.fill_missing_lanes:
        fill_missing_lanes(a)
    if settings.refine_vertical:
        refine_vertical(signal, a)
    a.bands.sort(key=lambda b: (b.row, b.lane))
    return a


def fill_missing_lanes(a: Analysis) -> int:
    """Add a box to every row for each lane it lacks (so empty lanes are measured as ~0).

    The new box's centre follows the row's tilt by interpolating neighbouring bands.
    Returns the number of boxes added.
    """
    lane_x = {}
    for lane in range(a.n_lanes):
        xs = [b.cx for b in a.bands if b.lane == lane]
        if xs:
            lane_x[lane] = float(np.median(xs))
    added = 0
    for ri in range(len(a.rows)):
        members = sorted((b for b in a.bands if b.row == ri), key=lambda b: b.cx)
        if not members:
            continue
        have = {b.lane for b in members}
        mx = [b.cx for b in members]
        my = [b.cy for b in members]
        for lane, x in lane_x.items():
            if lane in have:
                continue
            cy = float(np.interp(x, mx, my))
            a.bands.append(Band(row=ri, lane=lane, cx=x, cy=cy))
            added += 1
    a.bands.sort(key=lambda b: (b.row, b.lane))
    return added


def add_band(a: Analysis, x: float, y: float) -> Band:
    """Add a band at (x, y): it joins the nearest row (by y) and nearest lane (by x)."""
    if not a.rows:
        a.rows.append(Row(name="P1", width=40, height=16))
        row = 0
    else:
        dists = []
        for i, r in enumerate(a.rows):
            ys = [b.cy for b in a.bands if b.row == i]
            cy = float(np.mean(ys)) if ys else 1e9
            dists.append((abs(cy - y), i))
        d, row = min(dists)
        if d > a.rows[row].height * 1.5:
            ref = a.rows[row]
            a.rows.append(Row(name=unique_name(f"P{len(a.rows) + 1}", [r.name for r in a.rows]), width=ref.width, height=ref.height))
            row = len(a.rows) - 1
    lane_x = lane_centers(a)
    if lane_x:
        dist, lane = min((abs(x - lx), i) for i, lx in enumerate(lane_x))
        if dist > a.rows[row].width * 0.75:
            lane = len(lane_x) if x > max(lane_x) else -1
    else:
        lane = 0
    if lane == -1:  # new leftmost lane: shift everything right
        for b in a.bands:
            b.lane += 1
        a.lane_labels.insert(0, unique_name("L0", a.lane_labels))
        lane = 0
    # replace an existing band in the same row/lane
    a.bands = [b for b in a.bands if not (b.row == row and b.lane == lane)]
    band = Band(row=row, lane=lane, cx=float(x), cy=float(y))
    a.bands.append(band)
    a.ensure_lane_labels()
    return band


def refine_vertical(img: np.ndarray, a: Analysis, search: float = 0.75) -> int:
    """Move each box up/down (within +-search*height) to where it captures the most signal.

    Boxes in lanes without a clear band keep their position. Returns the number moved.
    """
    H, W = img.shape
    moved = 0
    for b in a.bands:
        r = a.rows[b.row]
        x, y, w, h = b.box(a.rows)
        x0, x1 = max(0, x), min(W, x + w)
        rng = int(max(1, search * h))
        y0, y1 = max(0, y - rng), min(H, y + h + rng)
        if x1 <= x0 or y1 - y0 <= h:
            continue
        prof = img[y0:y1, x0:x1].mean(axis=1)
        sums = np.convolve(prof, np.ones(h), mode="valid")  # box sum at each top offset
        if sums.size < 2:
            continue
        best = int(np.argmax(sums))
        noise = float(np.median(np.abs(prof - np.median(prof)))) * 1.4826 + 1e-9
        if (prof.max() - np.median(prof)) < 4 * noise:
            continue
        new_cy = y0 + best + h / 2.0
        if abs(new_cy - b.cy) >= 0.5:
            b.cy = float(new_cy)
            moved += 1
    return moved


def unique_name(name: str, existing: Sequence[str]) -> str:
    """``name``, or ``name_2``, ``name_3`` … if it is already taken."""
    name = name.strip()
    taken = set(existing)
    if name not in taken:
        return name
    k = 2
    while f"{name}_{k}" in taken:
        k += 1
    return f"{name}_{k}"


def remove_empty(a: Analysis):
    """Drop rows without bands and empty lanes; renumber lanes left to right."""
    for ri in reversed(range(len(a.rows))):
        if not any(b.row == ri for b in a.bands):
            a.remove_row(ri)
    a.ensure_lane_labels()
    used = sorted({b.lane for b in a.bands})
    xs = {l: float(np.median([b.cx for b in a.bands if b.lane == l])) for l in used}
    order = sorted(used, key=lambda l: xs[l])
    remap = {old: new for new, old in enumerate(order)}
    a.lane_labels = [a.lane_labels[old] for old in order]
    for b in a.bands:
        b.lane = remap[b.lane]
    a.bands.sort(key=lambda b: (b.row, b.lane))


def make_row_group(a: Analysis, bands: Sequence[Band], name: str) -> str:
    """Move ``bands`` into a new protein group (row) called ``name``.

    The row's box size is the largest of the rows the bands came from. If the
    name is taken by another row it gets a numeric suffix. Returns the name used.
    """
    bands = list(bands)
    if not bands:
        return ""
    ids = {id(b) for b in bands}
    src = {b.row for b in bands}
    # a row that is entirely replaced may keep (or give up) its name
    others = [r.name for i, r in enumerate(a.rows) if any(b.row == i and id(b) not in ids for b in a.bands)]
    final = unique_name(name or "Protein", others)
    w = max(a.rows[i].width for i in src)
    h = max(a.rows[i].height for i in src)
    role = a.rows[min(src)].role
    a.rows.append(Row(name=final, width=w, height=h, role=role))
    new = len(a.rows) - 1
    for b in bands:
        b.row = new
    remove_empty(a)
    a.sort()
    return final


def make_lane_group(a: Analysis, bands: Sequence[Band], name: str) -> str:
    """Name the lane(s) of ``bands`` as one lane group.

    * One band per protein (a single column): the bands become one lane.
    * Several bands of the same protein (several columns): every lane they are
      in gets the name; those lanes are replicates of the group.
    If another lane group already uses the name it gets a numeric suffix.
    Returns the name used.
    """
    bands = list(bands)
    if not bands:
        return ""
    a.ensure_lane_labels()
    ids = {id(b) for b in bands}
    rows = [b.row for b in bands]
    multi = len(rows) != len(set(rows))
    if multi:
        lanes = {b.lane for b in bands}
    else:
        lanes = {min(b.lane for b in bands)}
        # move the selected bands into one lane; bands of the same protein that
        # were there already (not selected) move to a new lane of their own
        target = next(iter(lanes))
        spare = a.n_lanes
        for b in a.bands:
            if b.lane == target and id(b) not in ids and b.row in set(rows):
                b.lane = spare
        if any(b.lane == spare for b in a.bands):
            k = 1
            while f"L{k}" in a.lane_labels:
                k += 1
            a.lane_labels.append(f"L{k}")
        for b in bands:
            b.lane = target
    others = [lab for i, lab in enumerate(a.lane_labels) if i not in lanes]
    final = unique_name(name or "Lane", others)
    for l in lanes:
        a.lane_labels[l] = final
    if final not in a.named_lanes:
        a.named_lanes.append(final)
    remove_empty(a)
    a.named_lanes = [n for n in a.named_lanes if n in a.lane_labels]
    return final


def auto_assign_lanes(a: Analysis) -> bool:
    """Re-derive lanes from box positions: bands are clustered left to right by
    their centre x (a new lane starts where the gap exceeds half a box width).

    A lane keeps a lane group name the user gave (most common one among its
    bands); other lanes are numbered L1, L2 … by position. Returns True if the
    lanes or their names changed.
    """
    if not a.bands:
        return False
    a.ensure_lane_labels()
    named = set(a.named_lanes)
    gap = 0.5 * float(np.median([a.rows[b.row].width for b in a.bands]))
    order = sorted(a.bands, key=lambda b: b.cx)
    clusters: List[List[Band]] = [[order[0]]]
    for b in order[1:]:
        if b.cx - clusters[-1][-1].cx > gap:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    labels = []
    for k, c in enumerate(clusters):
        names = [a.lane_labels[b.lane] for b in c if a.lane_labels[b.lane] in named]
        labels.append(max(set(names), key=names.count) if names else None)
    used = {l for l in labels if l}
    for k, l in enumerate(labels):
        if l is None:
            labels[k] = unique_name(f"L{k + 1}", used)
            used.add(labels[k])
    new_lane = {id(b): k for k, c in enumerate(clusters) for b in c}
    changed = labels != a.lane_labels or any(b.lane != new_lane[id(b)] for b in a.bands)
    if changed:
        for b in a.bands:
            b.lane = new_lane[id(b)]
        a.lane_labels = labels
        a.bands.sort(key=lambda b: (b.row, b.lane))
    return changed


def lane_centers(a: Analysis) -> List[float]:
    out = []
    for lane in range(a.n_lanes):
        xs = [b.cx for b in a.bands if b.lane == lane]
        out.append(float(np.median(xs)) if xs else float("nan"))
    return [x for x in out if not math.isnan(x)]


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

@dataclass
class Measurement:
    band_id: str
    row: int
    lane: int
    raw: float  # integrated intensity in the box before per-box background
    background: float  # integrated background removed from the box
    net: float  # raw - background (can be slightly negative for empty lanes)
    mean: float
    area: int


def measure(signal: np.ndarray, a: Analysis, bg: BackgroundSettings, corrected: Optional[np.ndarray] = None) -> List[Measurement]:
    """Integrated density of every band box with the chosen background removal."""
    img = corrected if corrected is not None else corrected_signal(signal, bg)
    H, W = img.shape
    out = []
    row_bg = {}
    if bg.method == "row_median":
        # Background per row = median of the strip spanning the row, excluding the
        # band boxes themselves (i.e. the membrane between lanes).
        for ri, r in enumerate(a.rows):
            members = [b for b in a.bands if b.row == ri]
            if not members:
                continue
            cy = float(np.mean([b.cy for b in members]))
            y0 = max(0, int(cy - r.height / 2))
            y1 = min(H, int(cy + r.height / 2) + 1)
            if y1 <= y0:
                continue
            strip = img[y0:y1, :]
            keep = np.ones(W, bool)
            for b in members:
                bx, _, bw, _ = b.box(a.rows)
                keep[max(0, bx):max(0, min(W, bx + bw))] = False
            vals = strip[:, keep]
            row_bg[ri] = float(np.median(vals)) if vals.size else float(np.percentile(strip, 10))
    for b in a.bands:
        x, y, w, h = b.box(a.rows)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            out.append(Measurement(a.band_id(b), b.row, b.lane, 0.0, 0.0, 0.0, 0.0, 0))
            continue
        patch = img[y0:y1, x0:x1]
        area = patch.size
        raw = float(patch.sum())
        if bg.method == "local_border":
            # Strips just above and below the box (the sides would sit on the
            # neighbouring part of the same band when bands are wider than the box).
            k = max(1, int(bg.border))
            above = img[max(0, y0 - k):y0, x0:x1]
            below = img[y1:min(H, y1 + k), x0:x1]
            vals = np.concatenate([above.ravel(), below.ravel()])
            per_px = float(np.median(vals)) if vals.size else 0.0
        elif bg.method == "row_median":
            per_px = row_bg.get(b.row, 0.0)
        else:
            per_px = 0.0
        bgsum = per_px * area
        net = raw - bgsum
        out.append(Measurement(a.band_id(b), b.row, b.lane, raw, bgsum, net, net / area, area))
    return out


def lane_profile(img: np.ndarray, a: Analysis, band: Band) -> np.ndarray:
    """Vertical intensity profile through a band box (mean over box width)."""
    x, y, w, h = band.box(a.rows)
    H, W = img.shape
    pad = h
    y0, y1 = max(0, y - pad), min(H, y + h + pad)
    x0, x1 = max(0, x), min(W, x + w)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(0)
    return img[y0:y1, x0:x1].mean(axis=1)


# --------------------------------------------------------------------------
# Formula evaluation
# --------------------------------------------------------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sum": lambda x: np.nansum(x),
    "mean": lambda x: np.nanmean(x),
    "max": lambda x: np.nanmax(x),
    "min": lambda x: np.nanmin(x),
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
}


class FormulaError(ValueError):
    pass


def _safe_name(name: str) -> str:
    """Make a group name usable in a formula: spaces and symbols -> underscore."""
    out = "".join(ch if ch.isalnum() else "_" for ch in name.strip())
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    if not out or out[0].isdigit():
        out = "_" + out
    return out


def evaluate(expr: str, variables: Dict[str, object]):
    """Evaluate an arithmetic expression (+ - * / ** and parentheses) over named values.

    Values can be scalars or numpy arrays (per-lane vectors); numpy broadcasting applies.
    Division by zero gives NaN instead of an error.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"수식 문법 오류: {e.msg}") from e

    def value(v, name):
        if isinstance(v, dict):
            raise FormulaError(f"'{name}'은 이미지 이름입니다. {name}.변수 형태로 쓰세요 (예: {name}.Vinculin)")
        return np.asarray(v, float) if not np.isscalar(v) else float(v)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return value(variables[node.id], node.id)
            raise FormulaError(f"알 수 없는 이름: {node.id}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # <image>.<name>: a variable of another image, e.g. I2.Vinculin
            ns = variables.get(node.value.id)
            if not isinstance(ns, dict):
                raise FormulaError(f"알 수 없는 이미지: {node.value.id} (I1, I2 … 또는 이미지 별칭)")
            if node.attr not in ns:
                raise FormulaError(f"알 수 없는 이름: {node.value.id}.{node.attr}")
            return value(ns[node.attr], f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            with np.errstate(divide="ignore", invalid="ignore"):
                left, right = ev(node.left), ev(node.right)
                if np.ndim(left) and np.ndim(right) and np.size(left) != np.size(right):
                    raise FormulaError(f"레인 수가 다른 값끼리 계산할 수 없습니다 ({np.size(left)}개 vs {np.size(right)}개 레인)")
                if isinstance(node.op, ast.Div):
                    res = np.divide(left, right)
                    return np.where(np.asarray(right) == 0, np.nan, res) if np.ndim(res) else (float("nan") if right == 0 else float(res))
                return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and len(node.args) == 1:
            return _FUNCS[node.func.id](ev(node.args[0]))
        raise FormulaError("허용되지 않는 수식입니다 (+ - * / ** ( ) 와 sum, mean, max, min, log2, log10, sqrt만 사용 가능)")

    return ev(tree)


def row_vector(a: Analysis, meas: List[Measurement], row: int) -> np.ndarray:
    """Net intensity of one protein group per physical lane (bands in one lane are summed)."""
    vec = np.zeros(a.n_lanes)
    has = np.zeros(a.n_lanes, bool)
    for m in meas:
        if m.row == row:
            vec[m.lane] += m.net
            has[m.lane] = True
    vec[~has] = np.nan
    return vec


def formula_variables(a: Analysis, meas: List[Measurement]) -> Dict[str, object]:
    """Names usable in formulas.

    * Protein group (row) name, e.g. ``Vinculin`` -> per-lane vector of net intensities.
    * Band ID, e.g. ``Kras_IgG_12h`` -> one band's net intensity.
    """
    v: Dict[str, object] = {}
    for ri, r in enumerate(a.rows):
        v[_safe_name(r.name)] = row_vector(a, meas, ri)
    for m in meas:
        v.setdefault(m.band_id, m.net)
    return v


def project_variables(image_vars: Sequence[Dict[str, object]], aliases: Sequence[str] = (), base: int = 0) -> Dict[str, object]:
    """Variables for a formula spanning several images.

    Names of the *base* image are available directly (``Vinculin``); every image
    is also reachable as a namespace: ``I1.Vinculin``, ``I2.G_IgG`` or, when the
    image has an alias, ``<alias>.Vinculin``.
    """
    v: Dict[str, object] = dict(image_vars[base]) if 0 <= base < len(image_vars) else {}
    for k, iv in enumerate(image_vars):
        v[f"I{k + 1}"] = iv
    for k, iv in enumerate(image_vars):
        alias = aliases[k].strip() if k < len(aliases) and aliases[k] else ""
        if alias:
            v.setdefault(_safe_name(alias), iv)
    return v


def normalize_to_control(a: Analysis, meas: List[Measurement], target_row: int, control_row: int, reference_lane: Optional[int] = 0) -> np.ndarray:
    """target / control lane by lane, then scaled so the reference lane is 1."""
    t = row_vector(a, meas, target_row)
    c = row_vector(a, meas, control_row)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(c > 0, t / c, np.nan)
    return scale_to_reference(ratio, reference_lane)


def scale_to_reference(values, reference_lane: Optional[int]):
    values = np.asarray(values, float)
    if reference_lane is None or values.ndim == 0 or reference_lane >= values.size:
        return values
    ref = values[reference_lane]
    if not np.isfinite(ref) or ref == 0:
        return values
    return values / ref


def condition_stats(values: np.ndarray, conditions: Sequence[str], labels: Sequence[str]):
    """Average lanes that share a condition name (replicates). Returns (names, mean, sd, n)."""
    names, means, sds, ns = [], [], [], []
    order = []
    for i, c in enumerate(conditions[: len(values)]):
        key = c.strip() or labels[i]
        if key not in order:
            order.append(key)
    for key in order:
        idx = [i for i in range(len(values)) if ((conditions[i].strip() if i < len(conditions) else "") or labels[i]) == key]
        vals = np.array([values[i] for i in idx], float)
        vals = vals[np.isfinite(vals)]
        names.append(key)
        means.append(float(vals.mean()) if vals.size else float("nan"))
        sds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        ns.append(int(vals.size))
    return names, np.array(means), np.array(sds), ns


# --------------------------------------------------------------------------
# Several images: lanes aligned by lane group name
# --------------------------------------------------------------------------

def common_lane_keys(lane_keys: Sequence[Sequence[str]]) -> List[str]:
    """Ordered union of the lane keys of several images."""
    out: List[str] = []
    for keys in lane_keys:
        for k in keys:
            if k not in out:
                out.append(k)
    return out


def to_common(values, keys: Sequence[str], common: Sequence[str]) -> np.ndarray:
    """Map a per-physical-lane vector onto ``common`` keys (replicate lanes are averaged)."""
    values = np.asarray(values, float)
    out = np.full(len(common), np.nan)
    for ci, key in enumerate(common):
        vals = [values[i] for i, k in enumerate(keys) if k == key and i < values.size and np.isfinite(values[i])]
        if vals:
            out[ci] = float(np.mean(vals))
    return out


def combine_members(members: Sequence[np.ndarray], ref: Optional[int] = None):
    """Mean, SD and n over member vectors (one per image), optionally each scaled so
    element ``ref`` is 1 first. Returns (mean, sd, n, scaled_members)."""
    if not members:
        return np.zeros(0), np.zeros(0), np.zeros(0, int), []
    scaled = [scale_to_reference(m, ref) for m in members]
    M = np.vstack(scaled)
    n = np.isfinite(M).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(M, axis=0) / np.maximum(n, 1), np.nan)
        dev = np.where(np.isfinite(M), (M - mean) ** 2, 0.0).sum(axis=0)
        sd = np.where(n > 1, np.sqrt(dev / np.maximum(n - 1, 1)), 0.0)
    return mean, sd, n, scaled
