"""Publication-style blot figures (dark bands on white) built from an analysis.

Everything is drawn with matplotlib in physical units (inches) so the preview
and the exported file are identical. Pixel values are only *displayed* here
(contrast/LUT is a display choice); measurements never come from this module.

Layout, top to bottom::

            group text            group text           <- header tiers (any number):
            ━━━━━━━━━━━━         ━━━━━━━━━━━━             group bars or one label per lane,
    cond    12h    24h            12h    24h              each with an optional condition text
    IB: X  [ladder strip][  crop of protein X  ]      <- one panel per protein
    IB: Y  [ladder strip][  crop of protein Y  ]
    label   1.00   0.85  ...                          <- number rows (optional)
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
    "margin_px": 8,  # extra blot pixels left/right of the outer lanes (old files)
    "margin_left_px": None,  # crop margins (None = margin_px); shared by all panels
    "margin_right_px": None,
    "lo_pct": 0.5,  # display contrast: percentile shown as white
    "hi_pct": 99.8,  # percentile shown as black
    "gamma": 1.0,
    "per_panel": True,  # contrast per panel (else one for the whole image)
    "use_corrected": False,  # show the background-corrected image
    "frame_lw": 0.8,
    "group_bar_lw": 2.2,  # line under a group of lanes
    "export_dpi": 600,
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
    "numbers": [],  # {source, label, pos, decimals, rel_lane}: quantified values under the lanes
    "protein_side": "left",  # protein labels left or right of the panels
    "kda_side": "left",
    "strip_gap_in": 0.0,  # space between the marker strip and the blot
    "strip_dy": 0,  # vertical offset of the marker crop (marker pixels), if the marker photo is shifted
    "image_gap_pt": 6.0,  # extra space where the next panel comes from another image (multi-blot)
    "lane_match": "name",  # multi-blot: other images' lanes by lane name ("name") or position ("order")
    "show_src": True,  # multi-blot: small grey source tag (I1, I2 …) under each protein name
    "layout": "rows",  # "rows": one crop per protein (western) · "full": the whole gel as one panel (DNA gel)
    "gel_crop": None,  # [x0, y0, x1, y1] of the full-gel panel (None = around all bands)
    "gel_label": "",  # text left of the full-gel panel
    "unit": "kDa",  # ladder unit shown next to the ticks' title ("kDa" or "bp")
    # "fonts" (size per text category), "tiers" (header rows) and "overrides"
    # (per-text size/angle/offset) are filled in by _migrate()
}


# look of new figures chosen in File → 설정 (font, sizes, line widths, …); set by the app
LOOK_KEYS = ("font", "fonts", "label_bold", "frame_lw", "group_bar_lw", "width_in", "panel_gap_in", "export_dpi")
USER_DEFAULTS: dict = {}


def set_user_defaults(look: Optional[dict]) -> None:
    USER_DEFAULTS.clear()
    USER_DEFAULTS.update({k: copy.deepcopy(v) for k, v in (look or {}).items() if k in LOOK_KEYS})


def apply_look(s: dict, look: Optional[dict] = None) -> dict:
    """Give an existing figure the default look (File → 설정 → 'Figure에도 지금 적용')."""
    for k, v in (USER_DEFAULTS if look is None else look).items():
        if k in LOOK_KEYS:
            s[k] = copy.deepcopy(v)
    return s


def default_settings() -> dict:
    s = copy.deepcopy(FIGURE_DEFAULTS)
    return apply_look(s)


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
    prev = s.get("panels", [])
    old = {p["row"]: p for p in prev if not p.get("doc")}
    panels = []
    for ri, r in enumerate(a.rows):
        p = old.get(r.name)
        if p is None:
            pad = int(round(0.6 * r.height))
            p = {"row": r.name, "show": True, "label": f"IB: {r.name}", "pad_top": pad, "pad_bottom": pad}
        panels.append(p)
    # panels from other images (multi-blot) keep their place among this image's panels
    if any(p.get("doc") for p in prev):
        result, used = [], 0
        for p in prev:
            if p.get("doc"):
                result.append(p)
            elif used < len(panels):
                result.append(panels[used])
                used += 1
        at = max((i for i, q in enumerate(result) if not q.get("doc")), default=-1) + 1
        result[at:at] = panels[used:]
        panels = result
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
    _migrate(s)
    _sync_tiers(s, n)
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
    ml = s.get("margin_left_px")
    mr = s.get("margin_right_px")
    ml = int(s.get("margin_px", 0) if ml is None else ml)
    mr = int(s.get("margin_px", 0) if mr is None else mr)
    x0 = max(0, min(lefts) - ml)
    x1 = min(width, max(rights) + mr)
    return x0, max(x1, x0 + 4)


GEL_ROW = "__gel__"


def default_gel_crop(a: core.Analysis, shape) -> List[int]:
    """Around every band box, with a margin of a band pitch."""
    H, W = shape
    boxes = [b.box(a.rows) for b in a.bands]
    x0 = min(x for x, _, _, _ in boxes)
    x1 = max(x + w for x, _, w, _ in boxes)
    y0 = min(y for _, y, _, _ in boxes)
    y1 = max(y + h for _, y, _, h in boxes)
    m = int(0.05 * max(x1 - x0, y1 - y0)) + 10
    return [max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)]


def layout_panels(a: core.Analysis, s: dict, shape, sources: Optional[dict] = None):
    """(panels [(panel settings, row index, y0, y1)], x0px, x1px) for either layout.

    ``sources`` ({doc id: {"signal", "analysis", "tag"}}) are the other open images a
    multi-blot figure takes panels from; without it those panels are left out."""
    H, W = shape
    if s.get("layout") == "full":
        x0, y0, x1, y1 = s.get("gel_crop") or default_gel_crop(a, shape)
        x0, x1 = max(0, int(x0)), min(W, int(x1))
        y0, y1 = max(0, int(y0)), min(H, int(y1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            x0, y0, x1, y1 = default_gel_crop(a, shape)
        return [({"row": GEL_ROW, "label": s.get("gel_label", ""), "show": True}, -1, y0, y1)], x0, x1
    x0, x1 = crop_x(a, s, W)
    return panel_rows(a, s, H, sources), x0, x1


def _full_blot_names(a: core.Analysis, s: dict):
    """Full blot of a Western: (panel settings, y) of each shown protein, to label its height."""
    if a.ladder_row:  # a DNA gel: rows are band positions, not proteins
        return []
    names = {r.name: i for i, r in enumerate(a.rows)}
    out = []
    for p in s.get("panels", []):
        ri = names.get(p["row"])
        if ri is None or not p.get("show", True):
            continue
        ys = [b.cy for b in a.bands if b.row == ri]
        if ys:
            out.append((p, float(np.mean(ys))))
    return out


MISSING = -2  # row index of a panel whose image is not open


def panel_rows(a: core.Analysis, s: dict, height: int, sources: Optional[dict] = None):
    """[(panel settings, row index, y0, y1)] for the panels that are shown.  A panel from
    another image (``doc``) is measured in that image; one whose image is not open gets
    the row index MISSING (drawn as a grey placeholder)."""
    out = []
    names = {r.name: i for i, r in enumerate(a.rows)}
    for p in s["panels"]:
        if p.get("doc"):
            if sources is None or not p.get("show", True):
                continue
            src = sources.get(p["doc"])
            if src is None:
                out.append((p, MISSING, 0, 0))
                continue
            sa = src["analysis"]
            ri = next((i for i, r in enumerate(sa.rows) if r.name == p.get("src_row")), None)
            ext = _row_extent(sa, ri) if ri is not None else None
            if ext is None:
                out.append((p, MISSING, 0, 0))
                continue
            sh = src["signal"].shape[0]
            y0 = max(0, ext[0] - int(p.get("pad_top", 0)))
            y1 = min(sh, ext[1] + int(p.get("pad_bottom", 0)))
            if y1 > y0:
                out.append((p, ri, y0, y1))
            continue
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


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_numbers(values, decimals: int = 2, rel_lane: Optional[int] = None):
    """Texts for a row of numbers and a label suffix.

    ``rel_lane`` scales the row so that lane is 1. Large values (>= 10 000)
    share one power of ten, given in the suffix, e.g. ``(×10⁶)``, so the
    numbers stay narrow enough to sit under their lanes.
    """
    v = np.asarray(values, float)
    if rel_lane is not None and 0 <= rel_lane < v.size and np.isfinite(v[rel_lane]) and v[rel_lane] != 0:
        v = v / v[rel_lane]
    suffix = ""
    finite = np.abs(v[np.isfinite(v)])
    if finite.size and finite.max() >= 1e4:
        e = int(np.floor(np.log10(finite.max())))
        v = v / 10.0 ** e
        suffix = f" (×10{str(e).translate(_SUP)})"
    texts = ["–" if not np.isfinite(x) else f"{x:.{int(decimals)}f}" for x in v]
    return texts, suffix


FONT_CATEGORIES = {  # key: (label, factor of the base font size)
    "lane": ("레인 이름", 1.0),
    "group": ("상위 그룹", 1.0),
    "cond": ("조건 설명", 1.0),
    "protein": ("단백질 이름", 1.0),
    "kda": ("마커(kDa)", 0.85),
    "number": ("정량 숫자", 0.9),
}


NEW_CELL = "–"  # a new name tier starts with this in every lane (visible, so it can be clicked)

# patterns of 이 층 채우기 (key, menu text)
TIER_FILLS = [("dash", "– 모두"), ("alt", "– + 번갈아"), ("pairs", "– – + + (두 칸씩)"), ("numbers", "1, 2, 3 …"),
              ("copy", "아래 층 복사"), ("empty", "비우기")]


def fill_texts(pattern: str, n: int, below: Optional[List[str]] = None) -> List[str]:
    """Texts of ``n`` lanes for a fill pattern of a name tier."""
    if pattern == "alt":
        return ["–" if i % 2 == 0 else "+" for i in range(n)]
    if pattern == "pairs":
        return ["–" if (i // 2) % 2 == 0 else "+" for i in range(n)]
    if pattern == "numbers":
        return [str(i + 1) for i in range(n)]
    if pattern == "copy":
        below = list(below or [])
        return (below + [""] * n)[:n]
    if pattern == "empty":
        return [""] * n
    return [NEW_CELL] * n


def default_tier(kind: str, n_lanes: int = 0, pattern: str = "dash") -> dict:
    t = {"kind": kind, "condition": "", "cond_side": "left", "rotation": 0, "italic": False}
    if kind == "lanes":
        t["texts"] = fill_texts(pattern, n_lanes)
        t["condition"] = "Condition"
    else:
        t["groups"] = []
    return t


def _migrate(s: dict):
    """Older settings: one lane-label row + one group row, one font size."""
    base = float(s.get("font_size", 8.0))
    fonts = s.setdefault("fonts", {})
    for k, (_, f) in FONT_CATEGORIES.items():
        fonts.setdefault(k, round(base * f, 1))
    if "tiers" not in s:
        tiers = []
        if s.get("groups"):
            g = default_tier("groups")
            g["groups"] = s["groups"]
            tiers.append(g)
        lane = default_tier("lanes")
        lane.update(auto=True, rotation=int(s.get("lane_rotation", 0)), condition="")
        lane.pop("texts", None)
        tiers.append(lane)
        s["tiers"] = tiers
        s["groups"] = []
    s.setdefault("overrides", {})


def _sync_tiers(s: dict, n: int):
    for t in s["tiers"]:
        for k, v in default_tier(t.get("kind", "lanes")).items():
            if k not in ("texts", "groups", "condition"):
                t.setdefault(k, v)
        t.setdefault("condition", "")
        if t["kind"] == "lanes" and not t.get("auto"):
            texts = list(t.get("texts", []))[:n]
            t["texts"] = texts + [NEW_CELL] * (n - len(texts))
        elif t["kind"] == "groups":
            t["groups"] = [g for g in t.get("groups", []) if 0 <= int(g["first"]) <= int(g["last"]) < n]


def tier_texts(s: dict, t: dict) -> List[str]:
    """Per-lane texts of a lane tier (the auto tier follows the lane groups)."""
    if t.get("auto"):
        return [str(l.get("text", "")) for l in s["lanes"]]
    return [str(x) for x in t.get("texts", [])]


def _text_h(text: str, fs: float, rot: float) -> float:
    """Rough height (inches) of a possibly rotated text."""
    lines = text.count("\n") + 1
    h = lines * fs * 1.2 / 72.0
    if not rot:
        return h
    import math

    r = math.radians(abs(rot))
    return _text_w(text, fs) * math.sin(r) + h * math.cos(r)


def default_lane_map(n_fig: int, n_src: int) -> List[Optional[int]]:
    """순서대로: figure column i shows the other image's lane i (None = 빈 칸)."""
    return [i if i < n_src else None for i in range(n_fig)]


def match_lanes(a: core.Analysis, src: core.Analysis, how: str = "name") -> List[Optional[int]]:
    """Source lane of every figure column.  "name": the lane with the same lane name (the
    k-th 'DMSO' of the figure takes the k-th 'DMSO' of the other image); "order": by position."""
    a.ensure_lane_labels()
    src.ensure_lane_labels()
    if how == "order":
        return default_lane_map(a.n_lanes, src.n_lanes)
    pools = {}
    for j, lab in enumerate(src.lane_labels[:src.n_lanes]):
        pools.setdefault(lab.strip().lower(), []).append(j)
    out = []
    for lab in a.lane_labels[:a.n_lanes]:
        pool = pools.get(lab.strip().lower())
        out.append(pool.pop(0) if pool else None)
    return out


def panel_lane_map(a: core.Analysis, s: dict, p: dict, src: core.Analysis) -> List[Optional[int]]:
    return p.get("lane_map") or match_lanes(a, src, s.get("lane_match", "name"))


def lane_pitch(a: core.Analysis) -> float:
    c = [x for x in lane_centres(a) if np.isfinite(x)]
    return float(np.median(np.diff(sorted(c)))) if len(c) > 1 else 0.0


def foreign_strip(src_sig: np.ndarray, src_a: core.Analysis, y0: int, y1: int, lane_map, own_centres,
                  x0px: int, x1px: int, k: float) -> np.ndarray:
    """A panel cut from another image, laid out on this figure's columns: every figure column
    shows its mapped source lane (scaled by ``k`` = source pitch / own pitch, so the lane
    spacing matches); columns without a lane are NaN (drawn white)."""
    width = max(1, x1px - x0px)
    out = np.full((max(1, y1 - y0), width), np.nan)
    own = np.asarray(own_centres, float)
    valid = [i for i, c in enumerate(own) if np.isfinite(c)]
    if not valid:
        return out
    src_c = lane_centres(src_a)
    xs = np.arange(width, dtype=float) + x0px
    cv = own[valid]
    col = np.asarray(valid)[np.argmin(np.abs(xs[:, None] - cv[None, :]), axis=1)]
    block = src_sig[y0:y1].astype(float)
    W2 = block.shape[1]
    for x in range(width):
        i = int(col[x])
        j = lane_map[i] if i < len(lane_map) else None
        if j is None or j >= len(src_c) or not np.isfinite(src_c[j]):
            continue
        sx = src_c[j] + (xs[x] - own[i]) * k
        if 0 <= sx < W2 - 1:
            f = int(sx)
            t = sx - f
            out[:, x] = (1 - t) * block[:, f] + t * block[:, f + 1]
    return out


def render(fig, signal: np.ndarray, a: core.Analysis, s: dict, marker: Optional[np.ndarray] = None,
           numbers: Optional[List[dict]] = None, sources: Optional[dict] = None, preview: bool = False) -> dict:
    """Draw the figure into the matplotlib ``fig`` (it is cleared and resized).

    ``numbers`` are resolved number rows: ``{label, texts (one per lane), pos}``
    with ``pos`` "top", "bottom", "above:<protein>" or "below:<protein>".
    Returns the layout (inches from the top-left): panels, the marker strip and
    every text with its key and bounding box, for interactive editing.

    ``sources`` are the other open images for panels taken from them (multi-blot figure);
    ``preview`` adds their grey source tags (I1, I2 …), which are never exported.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    fig.clear()
    _migrate(s)
    H, W = signal.shape
    layout = {"panels": [], "texts": [], "strips": [], "size": (3.0, 1.0)}
    if not a.bands:
        fig.set_size_inches(3, 1)
        fig.text(0.5, 0.5, "밴드 박스가 없습니다", ha="center", va="center")
        return layout
    fonts = s["fonts"]
    ov = s.get("overrides", {})
    family = [s.get("font") or "Arial", "AppleGothic", "DejaVu Sans"]
    weight = "bold" if s.get("label_bold", True) else "normal"
    panels, x0px, x1px = layout_panels(a, s, (H, W), sources)
    if not panels:
        fig.set_size_inches(3, 1)
        fig.text(0.5, 0.5, "표시할 단백질 패널이 없습니다", ha="center", va="center")
        return layout
    _sync_tiers(s, len(s["lanes"]))
    scale = float(s["width_in"]) / max(x1px - x0px, 1)  # inches per pixel
    panel_w = (x1px - x0px) * scale
    numbers = [n for n in (numbers or []) if n.get("texts")]
    tiers = s["tiers"]

    def fsize(key, cat):
        o = ov.get(key, {})
        if cat == "src":  # the source tag: a little smaller than the protein name
            return float(o.get("size") or 0.7 * fonts.get("protein", s["font_size"]))
        return float(o.get("size") or fonts.get(cat, s["font_size"]))

    show_strip = bool(s.get("show_strip")) and marker is not None and s["strip_x1"] > s["strip_x0"]
    strip_w = (s["strip_x1"] - s["strip_x0"]) * scale if show_strip else 0.0
    strip_gap = float(s.get("strip_gap_in", 0.0)) if show_strip else 0.0
    kda = sorted((k for k in s.get("kda", []) if k.get("kda") not in (None, "")), key=lambda k: k["y"])
    show_kda = bool(s.get("show_kda")) and bool(kda)
    kda_side = s.get("kda_side", "left")
    prot_side = s.get("protein_side", "left")
    tick = 0.05
    kda_w = (max(_text_w(_fmt_kda(k["kda"]), fonts["kda"]) for k in kda) + tick + 0.04) if show_kda else 0.0

    # widths of the label columns left and right of the blot
    left_items, right_items = [], []
    full_names = _full_blot_names(a, s) if s.get("layout") == "full" else []
    for p, *_ in panels:
        (left_items if prot_side == "left" else right_items).append(_text_w(p["label"], fsize(f"protein:{p['row']}", "protein")))
    for p, _ in full_names:
        (left_items if prot_side == "left" else right_items).append(_text_w(p["label"], fsize(f"protein:{p['row']}", "protein")))
    for i, n in enumerate(numbers):
        left_items.append(_text_w(n["label"], fsize(f"numlabel:{i}", "number")))
    for ti, t in enumerate(tiers):
        if t.get("condition"):
            w_ = _text_w(str(t["condition"]).replace("|", "\n"), fsize(f"cond:{ti}", "cond"))
            (left_items if t.get("cond_side", "left") == "left" else right_items).append(w_)
    label_w = (max(left_items) + 0.1) if left_items else 0.04
    right_w = (max(right_items) + 0.12) if right_items else 0.0
    kda_l = kda_w if kda_side == "left" else 0.0
    kda_r = kda_w if kda_side == "right" else 0.0
    left = 0.04 + label_w + kda_l + strip_w + strip_gap  # x of the blot panels
    total_w = left + panel_w + kda_r + right_w + 0.08
    gap = float(s["panel_gap_in"])
    own_pitch = lane_pitch(a)
    multi = bool(sources) and any(p.get("doc") for p, *_ in panels)
    image_gap = float(s.get("image_gap_pt", 6.0)) / 72.0
    MISSING_H = 0.22

    def k_of(p, ri):
        """Source pitch / own pitch of a panel from another image (1 for this image's own)."""
        if not p.get("doc") or ri == MISSING or not sources or own_pitch <= 0:
            return 1.0
        sp = lane_pitch(sources[p["doc"]]["analysis"])
        return sp / own_pitch if sp > 0 else 1.0

    def panel_h(it):
        p, ri, y0, y1 = it
        if ri == MISSING:
            return MISSING_H
        return (y1 - y0) * scale / k_of(p, ri) * float(p.get("height_k", 1.0))

    def panel_gap(prev_p, p):
        if prev_p is None:
            return gap
        return gap + (image_gap if (prev_p.get("doc") or "") != (p.get("doc") or "") else 0.0)

    # header tier heights
    bar_gap, bar_lw = 0.035, float(s.get("group_bar_lw", 2.2))
    tier_h = []
    for ti, t in enumerate(tiers):
        rot = float(t.get("rotation", 0))
        if t["kind"] == "lanes":
            hs = [_text_h(x.replace("|", "\n"), fsize(f"lane:{ti}:{li}", "lane"), rot) for li, x in enumerate(tier_texts(s, t)) if x]
            h = (max(hs) if hs else fonts["lane"] * 1.2 / 72.0) + 0.03
        else:
            hs = [_text_h(str(g["text"]).replace("|", "\n"), fsize(f"group:{ti}:{gi}", "group"), rot)
                  for gi, g in enumerate(t.get("groups", [])) if g.get("text")]
            h = (max(hs) if hs else 0.0) + 2 * bar_gap + 0.02
        ch = _text_h(str(t.get("condition", "")).replace("|", "\n"), fsize(f"cond:{ti}", "cond"), 0) if t.get("condition") else 0
        tier_h.append(max(h, ch + 0.02))
    header_h = sum(tier_h) + 0.04
    num_h = fonts["number"] * 1.3 / 72.0

    # vertical sequence: number rows and panels
    seq = [("num", (i, n)) for i, n in enumerate(numbers) if n.get("pos") == "top"]
    for item in panels:
        name = item[0]["row"]
        seq += [("num", (i, n)) for i, n in enumerate(numbers) if n.get("pos") == f"above:{name}"]
        seq.append(("panel", item))
        seq += [("num", (i, n)) for i, n in enumerate(numbers) if n.get("pos") == f"below:{name}"]
    shown = {p["row"] for p, *_ in panels}
    seq += [("num", (i, n)) for i, n in enumerate(numbers)
            if n.get("pos") == "bottom" or (n.get("pos", "").split(":", 1)[-1] not in shown and n.get("pos") != "top")]
    body = 0.0
    prev = None
    prev_p = None
    for kind, it in seq:
        if kind == "panel":
            body += (panel_gap(prev_p, it[0]) if prev == "panel" else 0.02 if prev == "num" else 0.0) + panel_h(it)
            prev_p = it[0]
        else:
            body += num_h + (0.02 if prev == "panel" else 0.0)
        prev = kind
    total_h = 0.04 + header_h + body + 0.04
    fig.set_size_inches(total_w, total_h)
    tr = fig.dpi_scale_trans
    arts = []

    def text(key, cat, x, y, t, **kw):
        """Text at (x, y) inches from the top-left, with the per-item override applied."""
        o = ov.get(key, {})
        kw["fontsize"] = fsize(key, cat)
        if o.get("rot") is not None:
            kw["rotation"] = float(o["rot"])
            kw.setdefault("rotation_mode", "anchor")
        x += float(o.get("dx", 0.0))
        y += float(o.get("dy", 0.0))
        if o.get("bold") is not None:
            kw["weight"] = "bold" if o["bold"] else "normal"
        if o.get("italic") is not None:
            kw["style"] = "italic" if o["italic"] else "normal"
        art = fig.text(x, total_h - y, t, transform=tr, family=family, **kw)
        arts.append((key, cat, art))
        return art

    def hline(xa, xb, y, lw):
        fig.add_artist(Line2D([xa, xb], [total_h - y, total_h - y], transform=tr, color="k", lw=lw, solid_capstyle="butt"))

    centres = lane_centres(a)
    xs = [left + (c - x0px) * scale for c in centres]
    finite = [x for x in xs if np.isfinite(x)]
    pitch = float(np.median(np.diff(sorted(finite)))) if len(finite) > 1 else panel_w / 4
    label_x = 0.04 + label_w - 0.08  # right edge of the left label column
    right_x = left + panel_w + kda_r + 0.08  # left edge of the right label column

    def side_text(key, cat, side, y, t, **kw):
        if side == "left":
            return text(key, cat, label_x, y, t, ha="right", va="center", **kw)
        return text(key, cat, right_x, y, t, ha="left", va="center", **kw)

    # ---- header tiers
    y = 0.04
    layout["tiers"] = []
    layout["bars"] = []
    for ti, (t, h) in enumerate(zip(tiers, tier_h)):
        layout["tiers"].append({"ti": ti, "kind": t["kind"], "y0": y, "y1": y + h, "auto": bool(t.get("auto"))})
        rot = float(t.get("rotation", 0))
        style = "italic" if t.get("italic") else "normal"
        bottom = y + h
        if t["kind"] == "lanes":
            base = bottom - 0.02
            for li, (x, tx) in enumerate(zip(xs, tier_texts(s, t))):
                if not np.isfinite(x) or not tx:
                    continue
                if rot:
                    text(f"lane:{ti}:{li}", "lane", x, base, tx.replace("|", "\n"), ha="left", va="bottom",
                         rotation=rot, rotation_mode="anchor", linespacing=1.1, style=style)
                else:
                    text(f"lane:{ti}:{li}", "lane", x, base, tx.replace("|", "\n"), ha="center", va="bottom",
                         linespacing=1.1, style=style)
            cy = base - (h - 0.03) / 2  # middle of this tier's text block
        else:
            bar_y = bottom - bar_gap
            for gi, g in enumerate(t.get("groups", [])):
                a_, b_ = int(g["first"]), int(g["last"])
                if not (np.isfinite(xs[a_]) and np.isfinite(xs[b_])):
                    continue
                xa, xb = xs[a_] - 0.42 * pitch, xs[b_] + 0.42 * pitch
                layout["bars"].append({"ti": ti, "gi": gi, "xa": xa, "xb": xb, "y": bar_y})
                if t.get("bar", True):
                    hline(xa, xb, bar_y, bar_lw)
                gx = (xa + xb) / 2
                if rot:
                    text(f"group:{ti}:{gi}", "group", gx, bar_y - bar_gap * 0.8, str(g["text"]).replace("|", "\n"),
                         ha="left", va="bottom", rotation=rot, rotation_mode="anchor", linespacing=1.1, style=style)
                else:
                    text(f"group:{ti}:{gi}", "group", gx, bar_y - bar_gap * 0.6, str(g["text"]).replace("|", "\n"),
                         ha="center", va="bottom", linespacing=1.1, style=style)
            cy = bar_y - bar_gap * 0.6 - max(h - 2 * bar_gap - 0.02, fonts["group"] / 72.0) / 2
        if t.get("condition"):
            side_text(f"cond:{ti}", "cond", t.get("cond_side", "left"), cy, str(t["condition"]).replace("|", "\n"),
                      linespacing=1.1)
        y = bottom

    # ---- panels and number rows
    src = signal
    ref_all = src[:, x0px:x1px]
    y = 0.04 + header_h
    prev = None
    prev_p = None
    for kind, it in seq:
        if kind == "num":
            ni, n = it
            y += 0.02 if prev == "panel" else 0.0
            cy = y + num_h / 2
            idx = n.get("idx", ni)  # index in the settings' number rows
            text(f"numlabel:{idx}", "number", label_x, cy, n["label"], ha="right", va="center")
            for li, (x, t) in enumerate(zip(xs, n["texts"])):
                if np.isfinite(x):
                    text(f"num:{idx}:{li}", "number", x, cy, t, ha="center", va="center")
            y += num_h
            prev = kind
            continue
        p, ri, y0, y1 = it
        y += panel_gap(prev_p, p) if prev == "panel" else 0.02 if prev == "num" else 0.0
        prev_p = p
        h = panel_h(it)
        foreign = bool(p.get("doc"))
        sx0 = left - strip_gap - strip_w
        if ri == MISSING:  # its image is not open: a grey placeholder keeps the place
            ax = fig.add_axes([left / total_w, (total_h - y - h) / total_h, panel_w / total_w, h / total_h])
            ax.set_facecolor("#ececec")
            ax.text(0.5, 0.5, "이미지를 열면 보입니다", ha="center", va="center", fontsize=6, color="#6e6e73",
                    transform=ax.transAxes, family=family)
            _frame(ax, s["frame_lw"])
            side_text(f"protein:{p['row']}", "protein", prot_side, y + h / 2, p["label"], weight=weight)
            layout["panels"].append({"row": p["row"], "doc": p.get("doc"), "x": left, "x_blot": left, "y": y,
                                     "w": panel_w, "h": h, "y0": y0, "y1": y1})
            y += h
            prev = kind
            continue
        if foreign:
            src_d = sources[p["doc"]]
            lmap = panel_lane_map(a, s, p, src_d["analysis"])
            crop = foreign_strip(src_d["signal"], src_d["analysis"], y0, y1, lmap, centres, x0px, x1px, k_of(p, ri))
            ok = np.isfinite(crop)
            ref = crop[ok] if ok.any() else np.zeros(1)
            if not p.get("auto_range", True):
                ref = src_d["signal"]
            disp = _stretch(np.where(ok, crop, np.median(ref)), s["lo_pct"], s["hi_pct"],
                            float(p.get("gamma", s["gamma"])), ref)
            disp[~ok] = 0.0  # 빈 칸: white
        else:
            crop = src[y0:y1, x0px:x1px]
            disp = _stretch(crop, s["lo_pct"], s["hi_pct"], s["gamma"], None if s.get("per_panel", True) else ref_all)
        px0, pw_ = left, panel_w
        if foreign and p.get("missing") == "cut":  # 잘라내기: the frame spans only the matched lanes
            lmap = panel_lane_map(a, s, p, sources[p["doc"]]["analysis"])
            cols = [i for i, j in enumerate(lmap) if j is not None and i < len(xs) and np.isfinite(xs[i])]
            if cols:
                xa = max(left, xs[cols[0]] - 0.5 * pitch)
                xb = min(left + panel_w, xs[cols[-1]] + 0.5 * pitch)
                c0, c1 = int(round((xa - left) / scale)), int(round((xb - left) / scale))
                disp = disp[:, max(0, c0):max(c0 + 1, c1)]
                px0, pw_ = xa, max(xb - xa, 0.01)
        ax = fig.add_axes([px0 / total_w, (total_h - y - h) / total_h, pw_ / total_w, h / total_h])
        ax.imshow(1.0 - disp, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="antialiased")
        _frame(ax, s["frame_lw"])
        if multi and s.get("show_src", True):  # source tag under the protein name (출처 표시)
            tag = sources[p["doc"]]["tag"] if foreign else sources.get("__own__", {}).get("tag", "")
            if tag:
                side_text(f"src:{p['row']}", "src", prot_side, y + h / 2 + fsize(f"protein:{p['row']}", "protein") / 72.0 * 0.95,
                          tag, color="#8e8e93")
        if foreign:  # marker strip and kDa ticks belong to the top image only
            side_text(f"protein:{p['row']}", "protein", prot_side, y + h / 2, p["label"], weight=weight)
            layout["panels"].append({"row": p["row"], "doc": p.get("doc"), "x": left, "x_blot": left, "y": y,
                                     "w": panel_w, "h": h, "y0": y0, "y1": y1})
            y += h
            prev = kind
            continue
        if show_strip:
            mcrop = marker_crop(marker, y0 + int(s.get("strip_dy", 0)), y1 + int(s.get("strip_dy", 0)),
                                s["strip_x0"], s["strip_x1"])
            sx = fig.add_axes([sx0 / total_w, (total_h - y - h) / total_h, strip_w / total_w, h / total_h])
            md = _stretch(mcrop, 1, 99.5, 1.0, marker[:, s["strip_x0"]:s["strip_x1"]])
            sx.imshow(1.0 - md if s.get("marker_invert", True) else md, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      interpolation="antialiased")
            _frame(sx, s["frame_lw"])
            layout["strips"].append({"x": sx0, "y": y, "w": strip_w, "h": h})
        side_text(f"protein:{p['row']}", "protein", prot_side, y + h / 2, p["label"], weight=weight)
        for fp, ry in full_names:  # Full blot (Western): every protein's name at its own height
            if y0 <= ry <= y1:
                side_text(f"protein:{fp['row']}", "protein", prot_side, y + (ry - y0) * scale, fp["label"], weight=weight)
        if show_kda:
            for ki, k in enumerate(kda):
                if not y0 <= k["y"] <= y1:
                    continue
                ky = y + (k["y"] - y0) * scale
                if kda_side == "left":
                    xr = sx0 if show_strip else left
                    hline(xr - tick, xr, ky, 0.6)
                    text(f"kda:{ki}", "kda", xr - tick - 0.02, ky, _fmt_kda(k["kda"]), ha="right", va="center")
                else:
                    xr = left + panel_w
                    hline(xr, xr + tick, ky, 0.6)
                    text(f"kda:{ki}", "kda", xr + tick + 0.02, ky, _fmt_kda(k["kda"]), ha="left", va="center")
        layout["panels"].append({"row": p["row"], "x": sx0 if show_strip else left, "x_blot": left, "y": y,
                                 "w": panel_w + strip_w + strip_gap, "h": h, "y0": y0, "y1": y1})
        y += h
        prev = kind

    # text boxes for clicking/dragging in the preview
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    for key, cat, art in arts:
        bb = art.get_window_extent(renderer)
        layout["texts"].append({"key": key, "cat": cat, "text": art.get_text(),
                                "x0": bb.x0 / dpi, "x1": bb.x1 / dpi,
                                "y0": total_h - bb.y1 / dpi, "y1": total_h - bb.y0 / dpi})
    layout.update(size=(total_w, total_h), scale=scale, left=left, panel_w=panel_w, x0px=x0px, x1px=x1px,
                  lane_xs=[float(x) for x in xs], pitch=float(pitch))
    return layout


def marker_crop(marker: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """``marker[y0:y1, x0:x1]``; rows outside the image (after an offset) are filled with the median."""
    H, W = marker.shape
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    out = np.full((max(0, y1 - y0), max(0, x1 - x0)), float(np.median(marker)))
    a, b = max(0, y0), min(H, y1)
    if b > a and x1 > x0:
        out[a - y0:b - y0] = marker[a:b, x0:x1]
    return out


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
    if abs(mw / mh - W / H) > core.MARKER_ASPECT_TOL * (W / H):
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


def apply_mode_defaults(s: dict, a: core.Analysis, mode: str) -> dict:
    """Figure defaults for the experiment type: a DNA gel is shown whole with bp ticks
    taken from the ladder lane(s); a western blot is cut per protein with kDa ticks."""
    if mode == core.MODE_GEL:
        s["layout"] = "full"
        s["unit"] = "bp"
        ys = core.ladder_band_ys(a)
        if ys and not any(k.get("kda") not in (None, "") for k in s.get("kda", [])):
            s["kda"] = [{"kda": None, "y": float(y)} for y in ys]  # sizes are filled in by the user / a preset
            s["show_kda"] = True
    else:
        s["layout"] = "rows"
        if s.get("unit") == "bp":
            s["unit"] = "kDa"
    return s
