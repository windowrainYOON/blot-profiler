"""Lane-first band detection ("레인 먼저 찾기").

1. Find lanes: horizontal-edge energy (sharp top/bottom of bands) summed per
   column gives a lane profile; its autocorrelation gives the lane pitch; lane
   centres are peaks fitted to an evenly spaced comb and refined per lane.
2. Each lane is a slanted strip (centre line tilted up to ±10°, fitted to the
   band centroids) a little wider than the lane (±10% of the pitch), because
   lanes are rarely perfectly vertical and bands drift sideways a bit.
3. Bands inside each lane: peaks of the lane's own vertical profile over a
   local baseline, with a lane-local noise threshold (faint ladder bands next
   to dark sample bands are kept); ladder-like lanes use a lower threshold.
4. Rows: band heights matched across lanes; every candidate gets a score
   (signal-to-noise × how many lanes share the row × shape), no rejection
   rules for artifacts. Candidates above the threshold become boxes, those a
   little below are offered as suggestions.

Works on a detection copy of the image only; measurement uses raw values.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks, peak_widths

from . import core

MAX_LANE_ANGLE = 10.0  # degrees
LANE_SLACK = 0.10  # sideways drift of bands within a lane, fraction of the pitch
SUGGEST_FACTOR = 0.6  # candidates scoring >= this × threshold are offered as suggestions


def edge_energy(work: np.ndarray) -> np.ndarray:
    """Positive part of −∂²/∂y² of the lightly smoothed image: large only for horizontal
    structures with sharp top and bottom (bands)."""
    d2 = ndi.gaussian_filter(work, sigma=(1.5, 1.0), order=(2, 0))
    return np.maximum(0.0, -d2)


def lane_profile(work: np.ndarray) -> np.ndarray:
    """Per-column sum of the band edge energy above its noise level."""
    E = edge_energy(work)
    E = np.maximum(0.0, E - float(np.percentile(E, 90)))
    return ndi.gaussian_filter1d(E.sum(axis=0), 2.0)


def lane_pitch(profile: np.ndarray) -> float:
    """Lane spacing: the strongest autocorrelation peak of the (high-passed) lane profile."""
    W = profile.size
    p = profile - ndi.gaussian_filter1d(profile, max(5.0, W / 12.0))
    p = p - p.mean()
    ac = np.correlate(p, p, mode="full")[W - 1:]
    lo, hi = max(3, W // 40), max(W // 40 + 3, W // 3)
    pk, _ = find_peaks(ac[:hi])
    pk = pk[pk >= lo]
    if len(pk):
        best = int(pk[np.argmax(ac[pk])])
        # a few strong lanes (ladders) repeat at a multiple of the pitch: prefer the
        # smallest whole fraction of the best lag that still shows a clear peak
        for n in range(6, 1, -1):
            cand = [p_ for p_ in pk if abs(p_ - best / n) <= max(2, 0.12 * best / n)]
            if cand and best / n >= lo:
                c = max(cand, key=lambda p_: ac[p_])
                if ac[c] >= 0.25 * ac[best]:
                    return float(c)
        return float(best)
    return float(lo + int(np.argmax(ac[lo:hi]))) if hi > lo + 1 else float(max(8, W // 8))


def find_lanes(work: np.ndarray, sensitivity: float = 1.0) -> Tuple[List[float], float]:
    """Lane centres (x) and the lane pitch."""
    prof = lane_profile(work)
    pitch = lane_pitch(prof)
    # lanes are plateaus: a box filter about a lane wide turns each into one peak
    box = ndi.uniform_filter1d(prof, size=max(3, int(0.6 * pitch)))
    base = float(np.percentile(box, 10))
    top = float(np.percentile(box, 95))
    if top <= base:
        return [], pitch
    level = base + 0.1 * (top - base) / max(sensitivity, 1e-3)
    peaks, _ = find_peaks(box, distance=max(2, int(0.6 * pitch)), height=level)
    if len(peaks) == 0:
        return [], pitch
    # fit an evenly spaced comb through the peaks, keep comb positions that carry signal
    c0 = float(peaks[0])
    idx = np.round((peaks - c0) / pitch).astype(int)
    if len(set(idx.tolist())) >= 2:
        b, a = np.polyfit(idx, peaks.astype(float), 1)
        if 0.7 * pitch < b < 1.3 * pitch:
            pitch = float(b)
        else:
            a = c0
    else:
        a = c0
    centres = []
    r = max(1, int(0.25 * pitch))
    for i in range(int(idx.min()), int(idx.max()) + 1):
        c = a + i * pitch
        lo, hi = max(0, int(c) - r), min(box.size, int(c) + r + 1)
        if hi <= lo:
            continue
        x = lo + int(np.argmax(box[lo:hi]))
        if box[x] >= base + 0.3 * (level - base):  # an empty lane between two loaded ones is kept
            centres.append(float(x))
    return centres, pitch


def line_x(line, y, H: float):
    """x of a lane line at height ``y``: a slanted line (x0 at mid-height, slope) or a
    polyline [[x, y], …] drawn by the user (straight beyond its ends)."""
    if isinstance(line, dict):
        line = line.get("points") or (line["x0"], line["slope"])
    if isinstance(line, tuple):
        x0, slope = line
        return np.asarray(x0 + slope * (np.asarray(y, float) - H / 2.0), float)
    return core.poly_x(line, y)


def _strip(work: np.ndarray, line, half: int) -> np.ndarray:
    """Straightened lane: for each row y the columns around the lane line x(y)."""
    H, W = work.shape
    ys = np.arange(H)
    xc = line_x(line, ys, H)
    offs = np.arange(-half, half + 1)
    cols = np.clip(np.round(xc[:, None] + offs[None, :]).astype(int), 0, W - 1)
    return work[ys[:, None], cols]


def _lane_profile(strip: np.ndarray, pitch: float):
    w = strip.shape[1]
    q = w // 4
    prof = np.median(strip[:, q:w - q] if w - 2 * q >= 1 else strip, axis=1)
    sm = ndi.gaussian_filter1d(prof, 1.0)
    base = ndi.minimum_filter1d(ndi.gaussian_filter1d(prof, 3.0), size=max(9, int(0.5 * pitch)))
    rel = sm - base
    noise = float(np.median(np.abs(rel - np.median(rel)))) * 1.4826 + 1e-9
    return rel, base, noise


def _lane_bands(strip: np.ndarray, k: float, pitch: float, noise_cap: Optional[float] = None) -> List[dict]:
    """Band candidates of one straightened lane, with their signal-to-noise and shape.

    The lane's own noise is capped by ``noise_cap`` (the typical lane's): a ladder lane is
    full of bands, which would otherwise inflate its noise estimate."""
    w = strip.shape[1]
    rel, base, noise = _lane_profile(strip, pitch)
    if noise_cap is not None:
        noise = min(noise, noise_cap)
    peaks, props = find_peaks(rel, distance=3, prominence=k * noise)
    if not len(peaks):
        return []
    _, _, left, right = peak_widths(rel, peaks, rel_height=0.5)
    # one band, one peak: drop a peak lying inside the half-max extent of a stronger one
    keep = np.ones(len(peaks), bool)
    for i in np.argsort(-rel[peaks]):
        if not keep[i]:
            continue
        for j in range(len(peaks)):
            if j != i and keep[j] and left[i] <= peaks[j] <= right[i]:
                keep[j] = False
    peaks, left, right = peaks[keep], left[keep], right[keep]
    proms = props["prominences"][keep]
    out = []
    for pk, a_, b_, prom in zip(peaks, left, right, proms):
        h = float(min(max(b_ - a_, 2.0), 0.9 * pitch))
        row = strip[pk]
        thr = base[pk] + 0.5 * rel[pk]
        fill = float((row > thr).mean())
        out.append({"y": float(pk), "h": h, "snr": float(prom / noise), "fill": fill,
                    "aspect": float(w / max(h, 1.0)), "noise": noise})
    return out


def _fit_line(points, x_mid, H, pitch):
    """Lane centre line x = x0 + s·(y − H/2) through band centroids, |angle| ≤ 10°."""
    if len(points) < 2:
        return x_mid, 0.0
    ys = np.array([p[1] for p in points], float)
    xs = np.array([p[0] for p in points], float)
    if np.ptp(ys) < 0.1 * H:
        return float(np.median(xs)), 0.0
    s, x0 = np.polyfit(ys - H / 2.0, xs, 1)
    lim = math.tan(math.radians(MAX_LANE_ANGLE))
    s = float(np.clip(s, -lim, lim))
    x0 = float(np.clip(x0, x_mid - 0.4 * pitch, x_mid + 0.4 * pitch))
    return x0, s


def _centroid_x(work, y, x_line, half):
    H, W = work.shape
    yy = int(np.clip(round(y), 0, H - 1))
    a, b = max(0, int(x_line - half)), min(W, int(x_line + half) + 1)
    row = work[yy, a:b]
    if row.size == 0 or row.max() <= row.min():
        return x_line
    wgt = row - row.min()
    return float(a + (np.arange(row.size) * wgt).sum() / wgt.sum())


def detect(signal: np.ndarray, settings: Optional[core.DetectSettings] = None,
           mode: str = core.MODE_WESTERN, lane_lines=None, protein_lines=None,
           only_on_lines: bool = False) -> core.Analysis:
    """Lane-first detection. Candidates and lane geometry are kept on the analysis
    (``a.candidates``, ``a.lane_geom``) for suggestions and display.

    ``lane_lines`` (polylines drawn or edited by the user) replace the lanes found
    automatically.  ``protein_lines`` ({row, x0, y0, x1, y1}) give the rows: one row per red
    line, named after it, its box in each lane snapped to the band nearest the line (within a
    window that never reaches halfway to the next line).  With ``only_on_lines`` (default for the
    app) those are the only rows; without it the rows found automatically away from every red
    line are added as well."""
    st = settings or core.DetectSettings()
    work = core.detection_image(signal, st)
    H, W = work.shape
    user_lines = [sorted([list(map(float, p)) for p in ln], key=lambda p: p[1]) for ln in (lane_lines or []) if len(ln) >= 2]
    if user_lines:
        user_lines.sort(key=lambda ln: float(np.mean([p[0] for p in ln])))
        centres = [float(np.mean([p[0] for p in ln])) for ln in user_lines]
        pitch = float(np.median(np.diff(centres))) if len(centres) > 1 else max(16.0, W / 8.0)
    else:
        centres, pitch = find_lanes(work, st.sensitivity)
    a = core.Analysis()
    a.method = "lane"
    if not centres:
        return a
    half = int(round(0.5 * pitch * (1 + 2 * LANE_SLACK) * 0.9))  # lane half-width incl. ±10% drift
    k = 4.0 / max(st.sensitivity, 1e-3)

    # typical lane noise (median over lanes) caps each lane's own estimate
    noise_cap = float(np.median([_lane_profile(_strip(work, user_lines[i] if user_lines else (cx, 0.0), half), pitch)[2]
                                 for i, cx in enumerate(centres)]))
    lanes = []
    counts = []
    for li, cx in enumerate(centres):
        if user_lines:  # the user's lane line as it is
            line = user_lines[li]
            cands = _lane_bands(_strip(work, line, half), k, pitch, noise_cap)
            counts.append(sum(1 for c in cands if c["snr"] >= k))
            lanes.append({"line": line, "cands": cands})
            continue
        line = (cx, 0.0)
        cands = _lane_bands(_strip(work, line, half), k, pitch, noise_cap)
        # slanted strip: fit the centre line to the band centroids, then search again along it
        pts = [(_centroid_x(work, c["y"], cx, half), c["y"]) for c in cands if c["snr"] >= 2 * k]
        line = _fit_line(pts, cx, H, pitch)
        cands = _lane_bands(_strip(work, line, half), k, pitch, noise_cap)
        counts.append(sum(1 for c in cands if c["snr"] >= k))
        lanes.append({"line": line, "cands": cands})
    # ladder-like lanes: clearly more bands than a typical lane; search them with a lower threshold
    med = float(np.median(counts)) if counts else 0.0
    for ln, n in zip(lanes, counts):
        if n >= core.LADDER_MIN_BANDS and n >= 1.6 * max(med, 1.0):
            ln["cands"] = _lane_bands(_strip(work, ln["line"], half), 0.6 * k, pitch, noise_cap)
            for c in ln["cands"]:
                c["ladder_like"] = True

    # rows: band heights matched across lanes
    allc = [(li, c) for li, ln in enumerate(lanes) for c in ln["cands"]]
    if not allc:
        a.lane_geom = [{"x0": ln["line"][0], "slope": ln["line"][1], "half": half} for ln in lanes]
        return a
    med_h = float(np.median([c["h"] for _, c in allc]))
    clusters = core._cluster_1d([c["y"] for _, c in allc], max(2.0, 0.6 * med_h))
    n_lanes = len(lanes)
    scored = []
    for cl in clusters:
        members = [allc[i] for i in cl]
        cover = len({li for li, _ in members}) / n_lanes
        row_snr = float(np.mean([c["snr"] for _, c in members]))
        for li, c in members:
            shape = min(1.0, c["fill"] / 0.4) * min(1.0, c["aspect"] / 1.5)
            score = (0.5 * c["snr"] + 0.5 * row_snr) * (0.5 + 0.5 * cover) * (0.5 + 0.5 * shape)
            scored.append((li, c, score))

    # boxes: candidates above the threshold (one per row and lane); suggestions a little below
    thr = k
    a = core.Analysis()
    a.method = "lane"
    rows: Dict[int, Dict[int, Tuple[dict, float]]] = {}
    row_of = {}
    for ci, cl in enumerate(clusters):
        for i in cl:
            row_of[id(allc[i][1])] = ci
    for li, c, s_ in scored:
        if s_ < thr:
            continue
        ri = row_of[id(c)]
        cur = rows.setdefault(ri, {}).get(li)
        if cur is None or s_ > cur[1]:
            rows[ri][li] = (c, s_)
    order = sorted(rows, key=lambda ri: np.mean([c["y"] for c, _ in rows[ri].values()]))
    # rows whose boxes would overlap are one band row (thick or doublet bands split by the peak finder)
    merged: List[int] = []
    for ri in order:
        if merged:
            pv = rows[merged[-1]]
            y_prev = np.mean([c["y"] for c, _ in pv.values()])
            y_cur = np.mean([c["y"] for c, _ in rows[ri].values()])
            h_avg = np.mean([c["h"] for c, _ in list(pv.values()) + list(rows[ri].values())])
            if y_cur - y_prev < 0.9 * h_avg:
                for li, (c, s_) in rows[ri].items():
                    if li not in pv or s_ > pv[li][1]:
                        pv[li] = (c, s_)
                continue
        merged.append(ri)
    order = merged
    # western: a protein row runs across most lanes; rows seen in few lanes become suggestions
    weak_rows = []
    if mode != core.MODE_GEL and n_lanes >= 4:
        weak_rows = [ri for ri in order if len(rows[ri]) < 0.5 * n_lanes]
        order = [ri for ri in order if ri not in weak_rows]

    def lane_x(li, y):
        return float(line_x(lanes[li]["line"], y, H))

    line_a = None
    if protein_lines:  # the red lines give the rows; the boxes follow the bands next to them
        line_a = core.Analysis()
        line_a.method = "lane"
        line_a = _boxes_on_lines(line_a, lanes, protein_lines, lane_x, half, pitch, st, mode, H, k)
        if only_on_lines:
            return line_a

    box_w = max(4, int(round(0.8 * pitch)))
    for new_ri, ri in enumerate(order):
        members = rows[ri]
        h = int(np.ceil(np.median([c["h"] for c, _ in members.values()]))) + 2 * st.pad
        a.rows.append(core.Row(name=f"P{new_ri + 1}", width=box_w, height=max(4, h)))
        for li, (c, _) in members.items():
            a.bands.append(core.Band(row=new_ri, lane=li, cx=lane_x(li, c["y"]), cy=float(c["y"])))
        if st.fill_missing_lanes and mode != core.MODE_GEL:
            ys = sorted((lane_x(li, c["y"]), c["y"]) for li, (c, _) in members.items())
            for li in range(n_lanes):
                if li not in members:
                    xa = lane_x(li, float(np.mean([y for _, y in ys])))
                    y = float(np.interp(xa, [x for x, _ in ys], [y for _, y in ys]))
                    a.bands.append(core.Band(row=new_ri, lane=li, cx=lane_x(li, y), cy=y))
    a.lane_labels = [f"L{i + 1}" for i in range(n_lanes)]
    a.bands.sort(key=lambda b: (b.row, b.lane))
    a.lane_geom = [_geom(ln["line"], half) for ln in lanes]
    a.candidates = []
    for ri in weak_rows:
        for li, (c, s_) in rows[ri].items():
            a.candidates.append({"x": lane_x(li, c["y"]), "y": float(c["y"]), "h": float(c["h"]),
                                 "w": float(box_w), "score": float(min(s_ / thr, 0.99))})
    for li, c, s_ in scored:
        if SUGGEST_FACTOR * thr <= s_ < thr:
            a.candidates.append({"x": lane_x(li, c["y"]), "y": float(c["y"]), "h": float(c["h"]),
                                 "w": float(box_w), "score": float(s_ / thr)})
    if mode == core.MODE_GEL:
        lad = [li for li, ln in enumerate(lanes) if any(c.get("ladder_like") for c in ln["cands"])]
        for i, r in enumerate(a.rows):
            r.name = f"B{i + 1}"
        if lad:
            core.set_ladder_lanes(a, [l for l in lad if any(b.lane == l for b in a.bands)])
        others = [r for r in a.rows if r.name != a.ladder_row]
        for i, r in enumerate(sorted(others, key=lambda r: a._row_y(a.rows.index(r)))):
            r.name = f"B{i + 1}"
        ladset = set(core.ladder_lanes(a))
        for i in range(a.n_lanes):
            if i not in ladset and i < len(a.lane_labels):
                a.lane_labels[i] = f"L{i + 1}"
    _match_geometry(a, H)
    if line_a is not None:
        return _add_rows_off_lines(line_a, a)
    return a


def _add_rows_off_lines(line_a: core.Analysis, auto: core.Analysis) -> core.Analysis:
    """The red-line rows plus the automatic rows that lie away from every red-line row."""
    def row_y(an, ri):
        ys = [b.cy for b in an.bands if b.row == ri]
        return float(np.median(ys)) if ys else None

    taken = [(row_y(line_a, ri), r.height) for ri, r in enumerate(line_a.rows)]
    used = [r.name for r in line_a.rows]
    for ri, r in enumerate(auto.rows):
        y = row_y(auto, ri)
        if y is None or any(ty is not None and abs(y - ty) <= max(4.0, 0.75 * max(h, r.height)) for ty, h in taken):
            continue
        name = core.unique_name(r.name, used)
        used.append(name)
        line_a.rows.append(core.Row(name=name, width=r.width, height=r.height))
        new = len(line_a.rows) - 1
        for b in auto.bands:
            if b.row == ri:
                line_a.bands.append(core.Band(row=new, lane=b.lane, cx=b.cx, cy=b.cy))
    line_a.candidates = list(auto.candidates)
    line_a.bands.sort(key=lambda b: (b.row, b.lane))
    return line_a


def _geom(line, half) -> dict:
    if isinstance(line, tuple):
        return {"x0": float(line[0]), "slope": float(line[1]), "half": int(half)}
    return {"points": [[float(x), float(y)] for x, y in line], "half": int(half)}


def _boxes_on_lines(a, lanes, protein_lines, lane_x, half, pitch, st, mode, H, k: float = 0.0) -> core.Analysis:
    """One box per lane on every red protein line: the band next to the line in that lane
    (searched within about 1.5 band heights, never past halfway to the neighbouring red line;
    a strong band a little off the line wins over a faint one on it), or the crossing itself
    when the lane shows no band there."""
    all_h = [c["h"] for ln in lanes for c in ln["cands"]]
    med_h = float(np.median(all_h)) if all_h else max(6.0, 0.03 * H)
    reach = max(6.0, 1.5 * med_h)
    box_w = max(4, int(round(0.8 * pitch)))
    plines = sorted(protein_lines, key=lambda p: (p["y0"] + p["y1"]) / 2.0)

    def line_y_at(p, li):
        y = core.protein_line_y(p, lane_x(li, H / 2.0))
        return core.protein_line_y(p, lane_x(li, y))

    used = []
    for pi, p in enumerate(plines):
        name = core.unique_name(str(p.get("row") or f"P{pi + 1}"), used)
        used.append(name)
        hs, spots = [], []
        for li in range(len(lanes)):
            y0 = line_y_at(p, li)
            gaps = [abs(line_y_at(q, li) - y0) for q in plines if q is not p]
            tol = min([reach] + [0.5 * g for g in gaps if g > 0])
            near = [c for c in lanes[li]["cands"] if abs(c["y"] - y0) <= tol and c.get("snr", 1.0) >= 0.5 * k]
            c = max(near, key=lambda c: c.get("snr", 1.0) * (1.0 - 0.5 * abs(c["y"] - y0) / max(tol, 1e-6))) if near else None
            cy = float(c["y"]) if c else float(y0)
            if c:
                hs.append(c["h"])
            spots.append((li, cy))
        h = int(np.ceil(np.median(hs) if hs else med_h)) + 2 * st.pad
        a.rows.append(core.Row(name=name, width=box_w, height=max(4, h)))
        for li, cy in spots:
            a.bands.append(core.Band(row=len(a.rows) - 1, lane=li, cx=lane_x(li, cy), cy=cy))
    a.lane_labels = [f"L{i + 1}" for i in range(len(lanes))]
    a.lane_geom = [_geom(ln["line"], half) for ln in lanes]
    a.bands.sort(key=lambda b: (b.row, b.lane))
    if mode == core.MODE_GEL:
        for i, r in enumerate(a.rows):
            if r.name.startswith("P"):
                r.name = f"B{i + 1}"
    return a


def name_rows_by_lines(a: core.Analysis, protein_lines) -> None:
    """Rows crossed by a named red line take its name (the nearest row within half a box)."""
    used = set()
    for p in protein_lines:
        name = str(p.get("row") or "")
        if not name:
            continue
        best, dist = None, None
        for ri, r in enumerate(a.rows):
            pts = [(b.cx, b.cy) for b in a.bands if b.row == ri]
            if not pts or ri in used:
                continue
            d_ = float(np.median([abs(y - core.protein_line_y(p, x)) for x, y in pts]))
            if d_ <= max(4.0, 0.75 * r.height) and (dist is None or d_ < dist):
                best, dist = ri, d_
        if best is not None:
            used.add(best)
            a.rows[best].name = core.unique_name(name, [r.name for i, r in enumerate(a.rows) if i != best])


def lane_line_x(g: dict, y: float) -> float:
    if g.get("points"):
        return float(core.poly_x(g["points"], y))
    return float(g["x0"] + g["slope"] * (y - g.get("ymid", 0.0)))


def _match_geometry(a: core.Analysis, H: int):
    """Keep one lane line per lane, in lane order (lanes without boxes may have been dropped)."""
    for g in a.lane_geom:
        if not g.get("points"):
            g["ymid"] = H / 2.0
    if not a.bands or not a.lane_geom:
        return
    out = []
    for lane in range(a.n_lanes):
        pts = [(b.cx, b.cy) for b in a.bands if b.lane == lane]
        if not pts:
            continue
        x, y = np.median([p[0] for p in pts]), np.median([p[1] for p in pts])
        out.append(min(a.lane_geom, key=lambda g: abs(lane_line_x(g, y) - x)))
    if len(out) == a.n_lanes:
        a.lane_geom = [dict(g) for g in out]


def tilt_angle(a: core.Analysis) -> float:
    """Global tilt (degrees) from the band rows: the median slope of rows spanning several
    lanes. Rotating the image by this angle makes the rows horizontal."""
    slopes = []
    for ri in range(len(a.rows)):
        pts = [(b.cx, b.cy) for b in a.bands if b.row == ri]
        if len(pts) >= 3:
            xs, ys = np.array(pts, float).T
            if np.ptp(xs) > 0:
                slopes.append(np.polyfit(xs, ys, 1)[0])
    if not slopes:
        return 0.0
    return float(math.degrees(math.atan(float(np.median(slopes)))))
