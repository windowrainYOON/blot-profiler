"""Publication-style blot figures (dark bands on white) built from an analysis.

Everything is drawn with matplotlib in physical units (inches) so the preview
and the exported file are identical. Pixel values are only *displayed* here
(contrast/LUT is a display choice); measurements never come from this module.

Layout, top to bottom::

            group text            group text
            ━━━━━━━━━━━━         ━━━━━━━━━━━━
    lane    12h    24h            12h    24h          <- lane headers
    IB: X  [ladder strip][  crop of protein X  ]      <- one panel per protein
    IB: Y  [ladder strip][  crop of protein Y  ]
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
from matplotlib.lines import Line2D

from . import core

FIGURE_DEFAULTS = {
    "width_in": 3.2,  # width of the blot panels
    "font": "Arial",
    "font_size": 8.0,
    "label_bold": True,
    "panel_gap_in": 0.06,
    "margin_px": 8,  # extra blot pixels left/right of the outer lanes
    "lo_pct": 0.5,  # display contrast: percentile shown as white
    "hi_pct": 99.8,  # percentile shown as black
    "gamma": 1.0,
    "per_panel": True,  # contrast per panel (else one for the whole image)
    "use_corrected": False,  # show the background-corrected image
    "frame_lw": 0.8,
    "lane_rotation": 0,
    "panels": [],  # {row, show, label, pad_top, pad_bottom}
    "lanes": [],  # {label, text}   ("|" in text = line break)
    "groups": [],  # {text, first, last}  lanes are 0-based indices
    "marker_path": None,
    "show_strip": False,
    "marker_invert": True,  # show the ladder dark on white (white-light photos show it bright on dark)
    "strip_x0": 0,
    "strip_x1": 0,
    "show_kda": False,
    "kda": [],  # {kda, y}  y in blot pixel rows
}


def default_settings() -> dict:
    return copy.deepcopy(FIGURE_DEFAULTS)


def _row_extent(a: core.Analysis, ri: int):
    boxes = [b.box(a.rows) for b in a.bands if b.row == ri]
    if not boxes:
        return None
    return min(y for _, y, _, _ in boxes), max(y + h for _, y, _, h in boxes)


def sync_settings(s: dict, a: core.Analysis, shape) -> dict:
    """Bring figure settings in line with the current proteins and lanes.

    Existing per-protein / per-lane edits are kept (matched by name / index);
    new proteins and lanes get defaults.
    """
    for k, v in FIGURE_DEFAULTS.items():
        s.setdefault(k, copy.deepcopy(v))
    old = {p["row"]: p for p in s.get("panels", [])}
    panels = []
    for ri, r in enumerate(a.rows):
        p = old.get(r.name)
        if p is None:
            pad = int(round(0.6 * r.height))
            p = {"row": r.name, "show": True, "label": f"IB: {r.name}", "pad_top": pad, "pad_bottom": pad}
        panels.append(p)
    s["panels"] = panels
    a.ensure_lane_labels()
    lanes = []
    prev = s.get("lanes", [])
    for i, lab in enumerate(a.lane_labels):
        if i < len(prev) and prev[i].get("label") == lab:
            lanes.append(prev[i])
        else:
            lanes.append({"label": lab, "text": lab})
    s["lanes"] = lanes
    n = len(lanes)
    s["groups"] = [g for g in s.get("groups", []) if 0 <= int(g["first"]) <= int(g["last"]) < n]
    if shape is not None and not s.get("strip_x1"):
        W = shape[1]
        s["strip_x0"], s["strip_x1"] = 0, max(4, int(0.06 * W))
    return s


def _stretch(img: np.ndarray, lo_pct: float, hi_pct: float, gamma: float, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Map ``img`` to 0..1 (1 = strongest band) using percentiles of ``ref``."""
    ref = img if ref is None else ref
    lo, hi = np.percentile(ref, [lo_pct, hi_pct])
    x = np.clip((img - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-9:
        x = x ** gamma
    return x


def _text_w(text: str, fs: float) -> float:
    """Rough text width in inches (good enough for layout)."""
    longest = max((len(t) for t in text.split("\n")), default=0)
    return longest * fs * 0.56 / 72.0


def lane_centres(a: core.Analysis) -> List[float]:
    out = []
    for lane in range(a.n_lanes):
        xs = [b.cx for b in a.bands if b.lane == lane]
        out.append(float(np.median(xs)) if xs else float("nan"))
    return out


def crop_x(a: core.Analysis, s: dict, width: int):
    lefts = [b.box(a.rows)[0] for b in a.bands]
    rights = [b.box(a.rows)[0] + b.box(a.rows)[2] for b in a.bands]
    m = int(s.get("margin_px", 0))
    return max(0, min(lefts) - m), min(width, max(rights) + m)


def panel_rows(a: core.Analysis, s: dict, height: int):
    """[(panel settings, row index, y0, y1)] for the panels that are shown."""
    out = []
    names = {r.name: i for i, r in enumerate(a.rows)}
    for p in s["panels"]:
        ri = names.get(p["row"])
        if ri is None or not p.get("show", True):
            continue
        ext = _row_extent(a, ri)
        if ext is None:
            continue
        y0 = max(0, ext[0] - int(p.get("pad_top", 0)))
        y1 = min(height, ext[1] + int(p.get("pad_bottom", 0)))
        if y1 > y0:
            out.append((p, ri, y0, y1))
    return out


def render(fig, signal: np.ndarray, a: core.Analysis, s: dict, marker: Optional[np.ndarray] = None):
    """Draw the figure into the matplotlib ``fig`` (it is cleared and resized)."""
    fig.clear()
    H, W = signal.shape
    if not a.bands:
        fig.set_size_inches(3, 1)
        fig.text(0.5, 0.5, "밴드 박스가 없습니다", ha="center", va="center")
        return
    fs = float(s["font_size"])
    family = [s.get("font") or "Arial", "AppleGothic", "DejaVu Sans"]
    weight = "bold" if s.get("label_bold", True) else "normal"
    line_h = fs * 1.25 / 72.0
    x0px, x1px = crop_x(a, s, W)
    panels = panel_rows(a, s, H)
    if not panels:
        fig.set_size_inches(3, 1)
        fig.text(0.5, 0.5, "표시할 단백질 패널이 없습니다", ha="center", va="center")
        return
    scale = float(s["width_in"]) / max(x1px - x0px, 1)  # inches per pixel
    panel_w = (x1px - x0px) * scale

    show_strip = bool(s.get("show_strip")) and marker is not None and s["strip_x1"] > s["strip_x0"]
    strip_w = (s["strip_x1"] - s["strip_x0"]) * scale if show_strip else 0.0
    kda = sorted(s.get("kda", []), key=lambda k: k["y"])
    show_kda = bool(s.get("show_kda")) and bool(kda)
    tick = 0.05
    kda_w = (max(_text_w(_fmt_kda(k["kda"]), fs) for k in kda) + tick + 0.04) if show_kda else 0.0
    label_w = max(_text_w(p["label"], fs) for p, *_ in panels) + 0.1

    # ---- header (lane texts and group bars)
    lanes = s["lanes"]
    texts = [str(l.get("text", "")).replace("|", "\n") for l in lanes]
    rot = int(s.get("lane_rotation", 0))
    nlines = max((t.count("\n") + 1 for t in texts if t), default=1)
    if rot == 0:
        lane_h = nlines * line_h
    else:
        longest = max((_text_w(t, fs) for t in texts), default=0.0)
        lane_h = longest * (1.0 if rot == 90 else 0.75) + line_h * 0.5
    groups = s.get("groups", [])
    glines = max((str(g["text"]).count("|") + 1 for g in groups), default=0)
    bar_gap, bar_lw = 0.035, 2.2
    group_h = glines * line_h + 2 * bar_gap if groups else 0.0
    header_h = lane_h + group_h + 0.04

    left = 0.04 + label_w + kda_w + strip_w  # x of the blot panels
    total_w = left + panel_w + 0.08
    heights = [(y1 - y0) * scale for _, _, y0, y1 in panels]
    gap = float(s["panel_gap_in"])
    total_h = 0.04 + header_h + sum(heights) + gap * (len(panels) - 1) + 0.04
    fig.set_size_inches(total_w, total_h)
    tr = fig.dpi_scale_trans

    def text(x, y, t, **kw):
        kw.setdefault("fontsize", fs)
        fig.text(x, total_h - y, t, transform=tr, family=family, **kw)

    def hline(xa, xb, y, lw):
        fig.add_artist(Line2D([xa, xb], [total_h - y, total_h - y], transform=tr, color="k", lw=lw, solid_capstyle="butt"))

    centres = lane_centres(a)
    xs = [left + (c - x0px) * scale for c in centres]
    finite = [x for x in xs if np.isfinite(x)]
    pitch = float(np.median(np.diff(sorted(finite)))) if len(finite) > 1 else panel_w / 4
    base = 0.04 + header_h - 0.04  # bottom of the lane text block (from top)
    for x, t in zip(xs, texts):
        if not np.isfinite(x) or not t:
            continue
        if rot == 0:
            text(x, base, t, ha="center", va="bottom", linespacing=1.1)
        else:
            text(x, base, t, ha="left" if rot != 90 else "center", va="bottom", rotation=rot,
                 rotation_mode="anchor", linespacing=1.1)
    if groups:
        bar_y = 0.04 + glines * line_h + bar_gap
        for g in groups:
            a_, b_ = int(g["first"]), int(g["last"])
            if not (np.isfinite(xs[a_]) and np.isfinite(xs[b_])):
                continue
            xa, xb = xs[a_] - 0.42 * pitch, xs[b_] + 0.42 * pitch
            hline(xa, xb, bar_y, bar_lw)
            text((xa + xb) / 2, bar_y - bar_gap * 0.6, str(g["text"]).replace("|", "\n"),
                 ha="center", va="bottom", linespacing=1.1)

    # ---- panels
    src = signal
    ref_all = src[:, x0px:x1px]
    y = 0.04 + header_h
    for (p, ri, y0, y1), h in zip(panels, heights):
        crop = src[y0:y1, x0px:x1px]
        disp = _stretch(crop, s["lo_pct"], s["hi_pct"], s["gamma"], None if s.get("per_panel", True) else ref_all)
        ax = fig.add_axes([left / total_w, (total_h - y - h) / total_h, panel_w / total_w, h / total_h])
        ax.imshow(1.0 - disp, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="antialiased")
        _frame(ax, s["frame_lw"])
        if show_strip:
            mcrop = marker[y0:y1, s["strip_x0"]:s["strip_x1"]]
            sx = fig.add_axes([(left - strip_w) / total_w, (total_h - y - h) / total_h, strip_w / total_w, h / total_h])
            md = _stretch(mcrop, 1, 99.5, 1.0, marker[:, s["strip_x0"]:s["strip_x1"]])
            sx.imshow(1.0 - md if s.get("marker_invert", True) else md, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      interpolation="antialiased")
            _frame(sx, s["frame_lw"])
        text(0.04 + label_w - 0.08, y + h / 2, p["label"], ha="right", va="center", weight=weight)
        if show_kda:
            xr = left - strip_w
            for k in kda:
                if y0 <= k["y"] <= y1:
                    ky = y + (k["y"] - y0) * scale
                    hline(xr - tick, xr, ky, 0.6)
                    text(xr - tick - 0.02, ky, _fmt_kda(k["kda"]), ha="right", va="center", fontsize=fs * 0.85)
        y += h + gap


def _fmt_kda(v) -> str:
    v = float(v)
    return f"{v:g}"


def _frame(ax, lw):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(lw)
        sp.set_color("k")


def match_marker(marker: np.ndarray, shape) -> np.ndarray:
    """Return the marker image at the blot's pixel size.

    Raises ValueError if the aspect ratio differs (the images would not line up).
    """
    if marker.shape == tuple(shape):
        return marker
    mh, mw = marker.shape
    H, W = shape
    if abs(mw / mh - W / H) > 0.01 * (W / H):
        raise ValueError(f"마커 이미지({mw}×{mh})와 블롯 이미지({W}×{H})의 가로세로 비율이 다릅니다")
    from scipy import ndimage as ndi

    out = ndi.zoom(marker, (H / mh, W / mw), order=1)
    return out[:H, :W]


def guess_ladder_x(marker: np.ndarray, a: core.Analysis):
    """Guess the ladder's x-range on the marker image: the most structured
    columns left of the first lane box (ladder lanes are full of sharp bands)."""
    H, W = marker.shape
    if not a.bands:
        return 0, max(4, int(0.06 * W))
    first = min(b.box(a.rows)[0] for b in a.bands)
    widths = [a.rows[b.row].width for b in a.bands]
    win = max(6, int(0.6 * float(np.median(widths))))
    if first < win + 2:
        return 0, max(4, int(0.06 * W))
    from scipy import ndimage as ndi

    ys = [b.cy for b in a.bands]
    y0, y1 = int(max(0, min(ys) - 0.1 * H)), int(min(H, max(ys) + 0.1 * H))
    region = marker[y0:y1, :first].astype(float)
    # vertical detail per column: ladder bands make strong row-to-row changes
    detail = np.abs(np.diff(ndi.gaussian_filter(region, 1.5), axis=0)).mean(axis=0)
    score = np.convolve(detail, np.ones(win) / win, mode="same")
    c = int(np.argmax(score))
    return max(0, c - win // 2), min(first, c + win // 2 + 1)
