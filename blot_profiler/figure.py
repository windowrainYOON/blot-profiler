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
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.lines import Line2D

from . import core
from .i18n import _t  # noqa: E402

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
        own_at = next((i for i, p in enumerate(s.get("panels", [])) if not p.get("doc")), -1)
        out = [({"row": GEL_ROW, "label": s.get("gel_label", ""), "show": True, "at": own_at}, -1, y0, y1)]
        out += _foreign_gels(s, sources)
        out.sort(key=lambda it: it[0]["at"])  # 1.4.8: the images in the order of their panels (draggable)
        return out, x0, x1
    x0, x1 = crop_x(a, s, W)
    return panel_rows(a, s, H, sources), x0, x1


def _foreign_gels(s: dict, sources: Optional[dict]):
    """1.4.7 Full blot with several images: each ticked image (one with panels in this
    figure) as one whole-blot panel under this image's, in the order of its first panel."""
    if not sources:
        return []
    out, seen = [], set()
    for at, p in enumerate(s.get("panels", [])):
        did = p.get("doc")
        if not did or did in seen or not p.get("show", True):
            continue
        seen.add(did)
        row = f"{GEL_ROW}:{did}"
        src = sources.get(did)
        if src is None:
            out.append(({"row": row, "doc": did, "label": "", "show": True, "gel": True, "at": at}, MISSING, 0, 0))
            continue
        sa, (sh, sw) = src["analysis"], src["signal"].shape
        fs = src.get("figure") or {}
        crop = fs.get("gel_crop") or default_gel_crop(sa, (sh, sw))
        y0, y1 = max(0, int(crop[1])), min(sh, int(crop[3]))
        if y1 - y0 < 4:
            _, y0, _, y1 = default_gel_crop(sa, (sh, sw))
        out.append(({"row": row, "doc": did, "label": "", "show": True, "gel": True, "at": at}, -1, y0, y1))
    return out


def _foreign_full_names(s: dict, did: str, sa: core.Analysis):
    """Full blot of another image: (panel settings, y in that image) of its proteins in this figure."""
    names = {r.name: i for i, r in enumerate(sa.rows)}
    out = []
    for p in s.get("panels", []):
        if p.get("doc") != did or not p.get("show", True):
            continue
        ri = names.get(p.get("src_row"))
        ys = [b.cy for b in sa.bands if b.row == ri] if ri is not None else []
        if ys:
            out.append((p, float(np.mean(ys))))
    return out


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
    "lane": (_t("레인 이름"), 1.0),
    "group": (_t("상위 그룹"), 1.0),
    "cond": (_t("조건 설명"), 1.0),
    "protein": (_t("단백질 이름"), 1.0),
    "kda": (_t("마커(kDa)"), 0.85),
    "number": (_t("정량 숫자"), 0.9),
}


NEW_CELL = "–"  # a new name tier starts with this in every lane (visible, so it can be clicked)

# patterns of 이 층 채우기 (key, menu text)
TIER_FILLS = [("dash", _t("– 모두")), ("alt", _t("– + 번갈아")), ("pairs", _t("– – + + (두 칸씩)")), ("numbers", "1, 2, 3 …"),
              ("copy", _t("아래 층 복사")), ("empty", _t("비우기"))]


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
    t = {"kind": kind, "condition": "", "cond_side": "left", "rotation": 0, "italic": False, "pos": "top"}
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


def foreign_place(src_a: core.Analysis, own_centres) -> Tuple[float, float, float, float]:
    """How a blot from another image sits under this figure's lanes (never cut per lane).

    1.4.8: it is scaled so its lane spacing equals this figure's, with its aspect ratio kept
    (the same factor across and down), and moved sideways so its first and last lanes sit as
    near as they can to this figure's first and last lanes.
    Returns (f, offset, a, b): own x = f * source x + offset; the source's lane span is [a, b)."""
    own = sorted(c for c in own_centres if np.isfinite(c))
    src = sorted(c for c in lane_centres(src_a) if np.isfinite(c))
    if not own or not src:
        return 1.0, 0.0, -np.inf, np.inf
    f = 1.0
    if len(own) > 1 and len(src) > 1:
        sp, op = float(np.median(np.diff(src))), float(np.median(np.diff(own)))
        if sp > 0 and op > 0:
            f = float(min(5.0, max(0.2, op / sp)))
    off = (own[0] + own[-1]) / 2.0 - f * (src[0] + src[-1]) / 2.0
    pitch = float(np.median(np.diff(src))) if len(src) > 1 else (
        float(np.median(np.diff(own))) / f if len(own) > 1 else 20.0)
    return f, off, src[0] - pitch / 2.0, src[-1] + pitch / 2.0


def foreign_shifted(src_sig: np.ndarray, src_a: core.Analysis, y0: int, y1: int, own_centres,
                    x0px: int, x1px: int, span=None):
    """The other image's rows y0:y1 as one piece, scaled and moved by ``foreign_place``, on this
    figure's columns x0px:x1px; its rows are scaled by the same factor (aspect ratio kept).
    Columns outside its lane span (or ``span``, in source pixels) are NaN.
    Returns (image, frame start, frame end in this figure's pixels, factor)."""
    f, off, a, b = foreign_place(src_a, own_centres)
    if span is not None:  # Full blot: the other image's whole-blot crop instead of its lane span
        a, b = span
    width = max(1, x1px - x0px)
    n = max(1, int(round((y1 - y0) * f)))
    block = src_sig[y0:y1].astype(float)
    H2, W2 = block.shape
    lo, hi = max(0.0, float(np.floor(a))), min(float(W2), float(np.ceil(b)))
    xs = (np.arange(width) + x0px - off) / f  # source column of every figure column
    ok = (xs >= lo) & (xs <= hi - 1)
    out = np.full((n, width), np.nan)
    if ok.any() and H2:
        from scipy import ndimage as ndi

        ys = np.clip((np.arange(n) + 0.5) / f - 0.5, 0, H2 - 1)
        yy, xx = np.meshgrid(ys, xs[ok], indexing="ij")
        out[:, ok] = ndi.map_coordinates(block, [yy, xx], order=1, mode="nearest")
    return out, lo * f + off, hi * f + off, f


def stack_sequence(s: dict, rows: List[str], numbers: Optional[List[dict]] = None) -> List[Tuple[str, object]]:
    """The figure from top to bottom (1.4.9): ("tier", tier index), ("num", (i, number row)) and
    ("panel", row name), for the shown panel ``rows`` in their order.

    A tier's ``pos`` (like a number row's) is "top", "above:<row>", "below:<row>" or "bottom";
    a tier whose panel is not shown stays on top.  Around one panel: tiers, number rows, the
    panel, number rows, tiers."""
    tiers = s.get("tiers", [])
    numbers = list(enumerate(numbers or []))
    shown = set(rows)

    def tpos(t):
        p = str(t.get("pos") or "top")
        return p if p in ("top", "bottom") or p.split(":", 1)[-1] in shown else "top"

    def npos(n):
        p = str(n.get("pos") or "bottom")
        return p if p in ("top", "bottom") or p.split(":", 1)[-1] in shown else "bottom"

    T = lambda p: [("tier", ti) for ti, t in enumerate(tiers) if tpos(t) == p]  # noqa: E731
    N = lambda p: [("num", (i, n)) for i, n in numbers if npos(n) == p]  # noqa: E731
    seq = T("top") + N("top")
    for r in rows:
        seq += T(f"above:{r}") + N(f"above:{r}") + [("panel", r)] + N(f"below:{r}") + T(f"below:{r}")
    return seq + N("bottom") + T("bottom")


def stack_positions(order: List[Tuple[str, object]]) -> List[Optional[str]]:
    """``pos`` of every tier and number row for a top-to-bottom ``order`` of ("tier", …),
    ("num", …) and ("panel", row) (a drag in 그림 안의 것 or on the canvas); None for panels.
    Between two panels, number rows before the first tier stay under the upper panel; tiers and
    the rest go above the lower panel.  Before the first panel: "top"; after the last: "bottom"
    (number rows right under the last panel: "below:<it>")."""
    out: List[Optional[str]] = [None] * len(order)
    prev_panel = None
    gap: List[int] = []

    def flush(next_panel):
        seen_tier = False
        for i in gap:
            kind = order[i][0]
            seen_tier = seen_tier or kind == "tier"
            first_nums = kind == "num" and not seen_tier
            if prev_panel is None:
                pos = "top"
            elif next_panel is None:
                pos = f"below:{prev_panel}" if first_nums else "bottom"
            else:
                pos = f"below:{prev_panel}" if first_nums else f"above:{next_panel}"
            out[i] = pos

    for i, (kind, ident) in enumerate(order):
        if kind == "panel":
            flush(ident)
            gap = []
            prev_panel = ident
        else:
            gap.append(i)
    flush(None)
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
        fig.text(0.5, 0.5, _t("밴드 박스가 없습니다"), ha="center", va="center")
        return layout
    fonts = s["fonts"]
    ov = s.get("overrides", {})
    family = [s.get("font") or "Arial", "AppleGothic", "DejaVu Sans"]
    weight = "bold" if s.get("label_bold", True) else "normal"
    panels, x0px, x1px = layout_panels(a, s, (H, W), sources)
    if not panels:
        fig.set_size_inches(3, 1)
        fig.text(0.5, 0.5, _t("표시할 단백질 패널이 없습니다"), ha="center", va="center")
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

    def marker_of(p):
        """(marker image, its figure settings) of the image a panel comes from; each image
        keeps its own marker strip and kDa ticks (1.4.7: also the other images in the figure)."""
        if p.get("doc"):
            src_d = (sources or {}).get(p["doc"]) or {}
            return src_d.get("marker"), src_d.get("figure") or {}
        return marker, s

    def strip_of(p):  # this figure's switch turns every image's strip on or off
        m, fs = marker_of(p)
        if m is None or not s.get("show_strip") or int(fs.get("strip_x1", 0)) <= int(fs.get("strip_x0", 0)):
            return None
        return m, fs

    def kda_of(p):
        _, fs = marker_of(p)
        if not s.get("show_kda"):
            return []
        return sorted((k for k in fs.get("kda", []) if k.get("kda") not in (None, "")), key=lambda k: k["y"])

    own_c = lane_centres(a)
    _fac = {}

    def fac(p):
        """Size factor of a panel's blot: 1 for this image; another image's blot is scaled to
        this figure's lane spacing (1.4.8, aspect ratio kept)."""
        did = p.get("doc")
        if not did or did not in (sources or {}):
            return 1.0
        if did not in _fac:
            _fac[did] = foreign_place(sources[did]["analysis"], own_c)[0]
        return _fac[did]

    shown_p = [p for p, ri, *_ in panels if ri != MISSING]
    strips = [(strip_of(p), fac(p)) for p in shown_p]
    show_strip = any(st is not None for st, _ in strips)
    strip_w = max(((int(st[1]["strip_x1"]) - int(st[1]["strip_x0"])) * scale * k for st, k in strips if st),
                  default=0.0)
    strip_gap = float(s.get("strip_gap_in", 0.0)) if show_strip else 0.0
    all_kda = [k for p in shown_p for k in kda_of(p)]
    show_kda = bool(all_kda)
    kda_side = s.get("kda_side", "left")
    prot_side = s.get("protein_side", "left")
    tick = 0.05
    kda_w = (max(_text_w(_fmt_kda(k["kda"]), fonts["kda"]) for k in all_kda) + tick + 0.04) if show_kda else 0.0

    # widths of the label columns left and right of the blot
    left_items, right_items = [], []
    full_names = _full_blot_names(a, s) if s.get("layout") == "full" else []
    for p, *_ in panels:
        if p.get("gel"):
            for fp, _ in _foreign_full_names(s, p["doc"], sources[p["doc"]]["analysis"]) if p["doc"] in (sources or {}) else []:
                (left_items if prot_side == "left" else right_items).append(
                    _text_w(fp["label"], fsize(f"protein:{fp['row']}", "protein")))
            continue
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

    multi = bool(sources) and any(p.get("doc") for p, *_ in panels)
    image_gap = float(s.get("image_gap_pt", 6.0)) / 72.0
    MISSING_H = 0.22

    def k_of(p, ri):
        """Source pixels per figure pixel of a panel (another image's blot is scaled, 1.4.8)."""
        return 1.0 if ri == MISSING else 1.0 / fac(p)

    def panel_h(it):
        p, ri, y0, y1 = it
        if ri == MISSING:
            return MISSING_H
        return (y1 - y0) * scale / k_of(p, ri) * float(p.get("height_k", 1.0))

    def panel_gap(prev_p, p):
        if prev_p is None:
            return gap
        return gap + (image_gap if (prev_p.get("doc") or "") != (p.get("doc") or "") else 0.0)

    # tier heights
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
            # 1.4.9: an empty group tier keeps one line of room (a blank space until it is deleted)
            h = (max(hs) if hs else fonts["group"] * 1.2 / 72.0) + 2 * bar_gap + 0.02
        ch = _text_h(str(t.get("condition", "")).replace("|", "\n"), fsize(f"cond:{ti}", "cond"), 0) if t.get("condition") else 0
        tier_h.append(max(h, ch + 0.02))
    num_h = fonts["number"] * 1.3 / 72.0

    seq = stack_sequence(s, [it[0]["row"] for it in panels], numbers)
    by_row = {it[0]["row"]: it for it in panels}
    seq = [(k, by_row[v] if k == "panel" else v) for k, v in seq]

    def step(prev, kind):
        """Room between two neighbours of the vertical stack."""
        if prev is None:
            return 0.0
        if prev == "tier":
            return 0.0 if kind == "tier" else 0.04
        if kind == "tier":
            return 0.03 if prev == "panel" else 0.02
        if kind == "num":
            return 0.02 if prev == "panel" else 0.0
        return 0.02 if prev == "num" else 0.0  # a panel after a number row (panel after panel: panel_gap)

    body = 0.0
    prev = None
    prev_p = None
    for kind, it in seq:
        if kind == "panel":
            body += (panel_gap(prev_p, it[0]) if prev == "panel" else step(prev, kind)) + panel_h(it)
            prev_p = it[0]
        elif kind == "tier":
            body += step(prev, kind) + tier_h[it]
        else:
            body += step(prev, kind) + num_h
        prev = kind
    total_h = 0.04 + body + 0.04
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

    # ---- tiers (1.4.9: anywhere in the stack; under a blot they hang upside down: bar on top)
    layout["tiers"] = []
    layout["bars"] = []
    layout["nums"] = []

    def draw_tier(ti, y):
        t, h = tiers[ti], tier_h[ti]
        pos = str(t.get("pos") or "top")
        down = pos == "bottom" or pos.startswith("below:")
        layout["tiers"].append({"ti": ti, "kind": t["kind"], "y0": y, "y1": y + h, "auto": bool(t.get("auto")),
                                "pos": pos, "down": down, "empty": t["kind"] == "groups" and not t.get("groups")})
        rot = float(t.get("rotation", 0))
        style = "italic" if t.get("italic") else "normal"
        bottom = y + h
        va = "top" if down else "bottom"
        if t["kind"] == "lanes":
            base = y + 0.02 if down else bottom - 0.02
            for li, (x, tx) in enumerate(zip(xs, tier_texts(s, t))):
                if not np.isfinite(x) or not tx:
                    continue
                if rot:
                    text(f"lane:{ti}:{li}", "lane", x, base, tx.replace("|", "\n"), ha="left" if not down else "right",
                         va=va, rotation=rot, rotation_mode="anchor", linespacing=1.1, style=style)
                else:
                    text(f"lane:{ti}:{li}", "lane", x, base, tx.replace("|", "\n"), ha="center", va=va,
                         linespacing=1.1, style=style)
            cy = base + (h - 0.03) / 2 if down else base - (h - 0.03) / 2  # middle of this tier's text block
        else:
            bar_y = y + bar_gap if down else bottom - bar_gap
            for gi, g in enumerate(t.get("groups", [])):
                a_, b_ = int(g["first"]), int(g["last"])
                if not (np.isfinite(xs[a_]) and np.isfinite(xs[b_])):
                    continue
                xa, xb = xs[a_] - 0.42 * pitch, xs[b_] + 0.42 * pitch
                layout["bars"].append({"ti": ti, "gi": gi, "xa": xa, "xb": xb, "y": bar_y})
                if t.get("bar", True):
                    hline(xa, xb, bar_y, bar_lw)
                gx = (xa + xb) / 2
                ty = bar_y + bar_gap * 0.6 if down else bar_y - bar_gap * (0.8 if rot else 0.6)
                if rot:
                    text(f"group:{ti}:{gi}", "group", gx, ty, str(g["text"]).replace("|", "\n"),
                         ha="left" if not down else "right", va=va, rotation=rot, rotation_mode="anchor",
                         linespacing=1.1, style=style)
                else:
                    text(f"group:{ti}:{gi}", "group", gx, ty, str(g["text"]).replace("|", "\n"),
                         ha="center", va=va, linespacing=1.1, style=style)
            half = max(h - 2 * bar_gap - 0.02, fonts["group"] / 72.0) / 2
            cy = bar_y + bar_gap * 0.6 + half if down else bar_y - bar_gap * 0.6 - half
        if t.get("condition"):
            side_text(f"cond:{ti}", "cond", t.get("cond_side", "left"), cy, str(t["condition"]).replace("|", "\n"),
                      linespacing=1.1)

    # ---- panels and number rows
    src = signal
    ref_all = src[:, x0px:x1px]
    y = 0.04
    prev = None
    prev_p = None
    for kind, it in seq:
        if kind == "tier":
            y += step(prev, kind)
            draw_tier(it, y)
            y += tier_h[it]
            prev = kind
            continue
        if kind == "num":
            ni, n = it
            y += step(prev, kind)
            cy = y + num_h / 2
            idx = n.get("idx", ni)  # index in the settings' number rows
            layout["nums"].append({"idx": idx, "y0": y, "y1": y + num_h})
            text(f"numlabel:{idx}", "number", label_x, cy, n["label"], ha="right", va="center")
            for li, (x, t) in enumerate(zip(xs, n["texts"])):
                if np.isfinite(x):
                    text(f"num:{idx}:{li}", "number", x, cy, t, ha="center", va="center")
            y += num_h
            prev = kind
            continue
        p, ri, y0, y1 = it
        y += panel_gap(prev_p, p) if prev == "panel" else step(prev, kind)
        prev_p = p
        h = panel_h(it)
        foreign = bool(p.get("doc"))
        sx0 = left - strip_gap - strip_w
        if ri == MISSING:  # its image is not open: a grey placeholder keeps the place
            ax = fig.add_axes([left / total_w, (total_h - y - h) / total_h, panel_w / total_w, h / total_h])
            ax.set_facecolor("#ececec")
            ax.text(0.5, 0.5, _t("이미지를 열면 보입니다"), ha="center", va="center", fontsize=6, color="#6e6e73",
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
            span = None
            if p.get("gel"):
                sa_ = src_d["analysis"]
                gc = (src_d.get("figure") or {}).get("gel_crop") or default_gel_crop(sa_, src_d["signal"].shape)
                span = (float(gc[0]), float(gc[2]))
            crop, fa, fb, _ = foreign_shifted(src_d["signal"], src_d["analysis"], y0, y1, centres, x0px, x1px, span)
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
        if foreign:  # the frame spans the other blot's lanes only (first to last lane)
            c0 = int(max(0, min(fa - x0px, x1px - x0px - 1)))
            c1 = int(max(c0 + 1, min(fb - x0px, x1px - x0px)))
            disp = disp[:, c0:c1]
            px0, pw_ = left + c0 * scale, max((c1 - c0) * scale, 0.01)
        ax = fig.add_axes([px0 / total_w, (total_h - y - h) / total_h, pw_ / total_w, h / total_h])
        ax.imshow(1.0 - disp, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="antialiased")
        _frame(ax, s["frame_lw"])
        if multi and s.get("show_src", True):  # source tag under the protein name (출처 표시)
            tag = sources[p["doc"]]["tag"] if foreign else sources.get("__own__", {}).get("tag", "")
            if tag:
                side_text(f"src:{p['row']}", "src", prot_side, y + h / 2 + fsize(f"protein:{p['row']}", "protein") / 72.0 * 0.95,
                          tag, color="#8e8e93")
        st = strip_of(p)
        if st is not None:  # each image's own marker strip, right-aligned next to its blot
            m_, fs = st
            sw_ = (int(fs["strip_x1"]) - int(fs["strip_x0"])) * scale * fac(p)
            dy_ = int(fs.get("strip_dy", 0))
            mcrop = marker_crop(m_, y0 + dy_, y1 + dy_, int(fs["strip_x0"]), int(fs["strip_x1"]))
            sx = fig.add_axes([(left - strip_gap - sw_) / total_w, (total_h - y - h) / total_h, sw_ / total_w, h / total_h])
            md = _stretch(mcrop, 1, 99.5, 1.0, m_[:, int(fs["strip_x0"]):int(fs["strip_x1"])])
            sx.imshow(1.0 - md if fs.get("marker_invert", True) else md, cmap="gray", vmin=0, vmax=1, aspect="auto",
                      interpolation="antialiased")
            _frame(sx, s["frame_lw"])
            layout["strips"].append({"x": left - strip_gap - sw_, "y": y, "w": sw_, "h": h, "doc": p.get("doc")})
        if p.get("gel"):  # Full blot of another image: its proteins' names at their own heights
            names_ = _foreign_full_names(s, p["doc"], sources[p["doc"]]["analysis"])
        else:
            side_text(f"protein:{p['row']}", "protein", prot_side, y + h / 2, p["label"], weight=weight)
            names_ = [] if foreign else full_names
        for fp, ry in names_:  # Full blot (Western): every protein's name at its own height
            if y0 <= ry <= y1:
                side_text(f"protein:{fp['row']}", "protein", prot_side, y + (ry - y0) * scale * fac(p), fp["label"], weight=weight)
        for ki, k in enumerate(kda_of(p)):
            if not y0 <= k["y"] <= y1:
                continue
            ky = y + (k["y"] - y0) * scale * fac(p)
            key = f"kda:{ki}" if not foreign else f"xkda:{p['doc']}:{ki}"
            if kda_side == "left":
                xr = sx0 if show_strip else left
                hline(xr - tick, xr, ky, 0.6)
                text(key, "kda", xr - tick - 0.02, ky, _fmt_kda(k["kda"]), ha="right", va="center")
            else:
                xr = left + panel_w
                hline(xr, xr + tick, ky, 0.6)
                text(key, "kda", xr + tick + 0.02, ky, _fmt_kda(k["kda"]), ha="left", va="center")
        if foreign:  # fx/fw: the frame as drawn (its edges are the resize handles)
            layout["panels"].append({"row": p["row"], "doc": p.get("doc"), "x": left, "x_blot": left, "y": y,
                                     "w": panel_w, "h": h, "y0": y0, "y1": y1, "fx": px0, "fw": pw_})
        else:
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
        raise ValueError(_t("마커 이미지({0}×{1})와 블롯 이미지({2}×{3})의 가로세로 비율이 다릅니다").format(mw, mh, W, H))
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
        s["unit"] = "bp" if s.get("ladder") in core.LADDER_PRESETS else "kDa"  # 1.3: the unit follows the ladder
        ys = core.ladder_band_ys(a)
        if ys and not any(k.get("kda") not in (None, "") for k in s.get("kda", [])):
            s["kda"] = [{"kda": None, "y": float(y)} for y in ys]  # sizes are filled in by the user / a preset
            s["show_kda"] = True
    else:
        s["layout"] = "rows"
        if s.get("unit") == "bp":
            s["unit"] = "kDa"
    return s
