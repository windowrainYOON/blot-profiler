"""Result graphs in the style of GraphPad PRISM, and the data behind them.

PRISM look: white background, no top/right axis lines, thick black axes with ticks
outside, bold labels, black-outlined bars (white → grey → black for several data
sets), mean ± SD (or SEM) error bars with caps, and the individual values as dots.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

from . import prefs

PALETTES = {
    "bw": ["#FFFFFF", "#9E9E9E", "#000000", "#D9D9D9", "#616161"],
    "gray": ["#F2F2F2", "#BDBDBD", "#7A7A7A", "#424242", "#DADADA"],
    "color": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"],
}
FALLBACK_FONTS = ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]


def style(graph_prefs: Optional[dict] = None, **over) -> dict:
    """Graph style: the defaults of File → 설정, with the choices of the graph sidebar on top."""
    s = dict(prefs.DEFAULTS["graph"])
    s.update({"bar_width": 0.8, "x_rot": "auto", "points": True, "legend": True})
    s.update(graph_prefs or {})
    s.update({k: v for k, v in over.items() if v is not None})
    return s


def _dark(color: str) -> bool:
    r, g, b = matplotlib.colors.to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.45


def x_groups(n_values: int, labels: Sequence[str], conditions: Sequence[str], by_condition: bool) -> List[Tuple[str, List[int]]]:
    """Positions on the x axis: lanes, or lane groups (replicate lanes averaged)."""
    if not by_condition:
        return [(labels[i] if i < len(labels) else str(i + 1), [i]) for i in range(n_values)]
    order: Dict[str, List[int]] = {}
    for i in range(n_values):
        key = ((conditions[i].strip() if i < len(conditions) else "") or (labels[i] if i < len(labels) else str(i + 1)))
        order.setdefault(key, []).append(i)
    return list(order.items())


def summary(values, err: str = "sd") -> Tuple[float, float, int]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan"), 0
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    e = sd / np.sqrt(v.size) if err == "sem" else sd
    return float(np.mean(v)), e, int(v.size)


MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # a different dot shape per series (PRISM)


def _draw(fig, xlabels: List[str], data: List[Tuple[str, List[np.ndarray]]], ylabel: str, st: dict, title: str = ""):
    """data = [(series name, [values at each x position])]. Style keys: see prefs.DEFAULTS['graph']
    plus the editor's: bar_width, x_rot, tick_dir, tick_len, y_min, y_max, y_title, title, legend,
    legend_pos (axes fraction), labels {series: shown name}, series {series: {color, marker}}, texts."""
    fig.clear()
    fig.patch.set_facecolor("white")
    font = [st.get("font") or "Arial"] + FALLBACK_FONTS
    weight = "bold" if st.get("bold", True) else "normal"
    with matplotlib.rc_context({"font.family": font, "axes.unicode_minus": False}):
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        if not data or not xlabels:
            ax.text(0.5, 0.5, "표시할 결과가 없습니다", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return ax
        pal = PALETTES.get(st.get("colors"), PALETTES["bw"])
        k = len(data)
        width = float(st.get("bar_width", 0.8)) / k
        rng = np.random.default_rng(7)
        kind = st.get("kind", "bar_points")
        points = st.get("points", True)
        lw_bar, lw_err, cap = float(st["bar_lw"]), float(st["err_lw"]), float(st["cap"])
        own = st.get("series") or {}
        shown = st.get("labels") or {}
        all_vals = []
        for si, (name, groups) in enumerate(data):
            fill = (own.get(name) or {}).get("color") or pal[si % len(pal)]
            marker = (own.get(name) or {}).get("marker") or MARKERS[si % len(MARKERS)]
            dot_face = "white" if _dark(fill) and kind != "points_mean" else ("black" if kind != "points_mean" else fill)
            label = shown.get(name, name)
            for xi, vals in enumerate(groups):
                vals = np.asarray(vals, float)
                vals = vals[np.isfinite(vals)]
                all_vals += list(vals)
                if not vals.size:
                    continue
                x = xi + (si - (k - 1) / 2) * width
                m, e, n = summary(vals, st.get("err", "sd"))
                if kind in ("bar_points", "bars"):
                    ax.bar(x, m, width * 0.82, color=fill, edgecolor="black", linewidth=lw_bar,
                           label=label if xi == 0 else None, zorder=2, gid=f"series:{name}")
                    if n > 1:  # PRISM: error bar above the bar only
                        ax.errorbar(x, m, yerr=[[0], [e]], fmt="none", ecolor="black", elinewidth=lw_err,
                                    capsize=cap, capthick=lw_err, zorder=3)
                else:  # scatter dot plot: mean line with error bars both ways
                    half = width * 0.32
                    ax.plot([x - half, x + half], [m, m], color="black", lw=lw_err * 1.3, zorder=3,
                            label=label if xi == 0 else None, solid_capstyle="butt")
                    if n > 1:
                        ax.errorbar(x, m, yerr=e, fmt="none", ecolor="black", elinewidth=lw_err,
                                    capsize=cap * 0.8, capthick=lw_err, zorder=3)
                if kind != "bars" and points and (n > 1 or kind == "points_mean" or st.get("single_points", False)):
                    jit = rng.uniform(-0.22, 0.22, vals.size) * width if vals.size > 1 else np.zeros(1)
                    ax.scatter(x + jit, vals, s=float(st["point_size"]), marker=marker, facecolor=dot_face,
                               edgecolor="black", linewidth=0.9, zorder=4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(float(st["axis_lw"]))
        ax.tick_params(direction=st.get("tick_dir", "out"), length=float(st.get("tick_len", 6)),
                       width=float(st["axis_lw"]), labelsize=float(st["tick_size"]))
        ax.set_xticks(range(len(xlabels)))
        rot = st.get("x_rot", "auto")
        rot = (45 if max(len(str(s)) for s in xlabels) > 4 else 0) if rot in ("auto", None) else int(rot)
        ax.set_xticklabels(xlabels, rotation=rot, ha="right" if rot and rot < 90 else "center",
                           fontsize=float(st["tick_size"]) + 1, fontweight=weight)
        for lab in ax.get_yticklabels():
            lab.set_fontweight(weight)
        ax.set_xlim(-0.6, len(xlabels) - 0.4)
        yl = ax.set_ylabel(st.get("y_title") or ylabel, fontsize=float(st["label_size"]), fontweight=weight)
        yl.set_gid("ylabel")
        finite = [v for v in all_vals if np.isfinite(v)]
        if st.get("y0", True) and finite and min(finite) >= 0:
            ax.set_ylim(bottom=0)
        if st.get("y_min") is not None:
            ax.set_ylim(bottom=float(st["y_min"]))
        if st.get("y_max") is not None:
            ax.set_ylim(top=float(st["y_max"]))
        t = st.get("title") if st.get("title") is not None else title
        if t:
            ax.set_title(t, fontsize=float(st.get("title_size", st["label_size"])), fontweight=weight).set_gid("title")
        if st.get("legend", True) and (k > 1 or (data and data[0][0])):
            pos = st.get("legend_pos")
            if pos:
                leg = ax.legend(frameon=False, fontsize=float(st["legend_size"]), loc="upper left", bbox_to_anchor=tuple(pos))
            else:
                leg = ax.legend(frameon=False, fontsize=float(st["legend_size"]), loc="upper left", bbox_to_anchor=(1.0, 1.0))
            leg.set_gid("legend")
        for i, tb in enumerate(st.get("texts") or []):
            ax.text(float(tb.get("x", 0.5)), float(tb.get("y", 0.9)), tb.get("text", ""), transform=ax.transAxes,
                    fontsize=float(tb.get("size", st["legend_size"])), gid=f"text:{i}", zorder=6)
        fig.tight_layout()
        return ax


def pool(per_image: List[Tuple[str, Dict[str, np.ndarray], List[str]]], by_condition: bool, split_images: bool):
    """Values of several images in one graph: ``per_image`` = [(image key, {series: values per lane},
    lane names)]. Lanes of the same name are one x position (replicates, all images together);
    with ``split_images`` every image gets its own series. Returns (x labels, [(series, [arrays])])."""
    order: List[str] = []
    cells: Dict[str, Dict[str, List[float]]] = {}
    for key, series, labels in per_image:
        for name, v in series.items():
            v = np.atleast_1d(np.asarray(v, float))
            if v.size <= 1:
                continue
            sname = f"{key}: {name}" if split_images else name
            for i, val in enumerate(v):
                lab = labels[i] if i < len(labels) else str(i + 1)
                x = lab if by_condition else (f"{key} {lab}" if len(per_image) > 1 else lab)
                if not by_condition and len(per_image) == 1:
                    x = f"{i + 1}:{lab}" if labels.count(lab) > 1 else lab
                if x not in order:
                    order.append(x)
                cells.setdefault(sname, {}).setdefault(x, []).append(float(val))
    data = [(s, [np.array(c.get(x, []), float) for x in order]) for s, c in cells.items()]
    return order, data


def plot_series(fig, series: Dict[str, np.ndarray], labels, conditions, by_condition: bool, ylabel: str, st: dict,
                title: str = ""):
    """Results of one image: one value per lane (or group of a grouping set) for each series."""
    series = {k: np.atleast_1d(np.asarray(v, float)) for k, v in series.items()}
    if series and all(v.size == 1 for v in series.values()):
        return _draw(fig, ["값"], [(k, [v]) for k, v in series.items()], ylabel, st, title)
    series = {k: v for k, v in series.items() if v.size > 1}
    n = max((v.size for v in series.values()), default=0)
    xg = x_groups(n, labels, conditions, by_condition)
    data = [(k, [v[[i for i in idx if i < v.size]] for _, idx in xg]) for k, v in series.items()]
    return _draw(fig, [x for x, _ in xg], data, ylabel, st, title)


def plot_summary(fig, keys: List[str], series: Dict[str, Dict[str, object]], ylabel: str, st: dict):
    """Several images (tab 5): per lane key the values of the member images."""
    if series and all(np.size(s["mean"]) == 1 for s in series.values()):
        keys = ["값"]
    data = []
    for name, s in series.items():
        mean = np.atleast_1d(np.asarray(s["mean"], float))
        pts = s.get("points")
        groups = []
        for i in range(len(keys)):
            if pts:
                groups.append(np.array([p[i] for p in pts if i < len(p)], float))
            else:
                groups.append(np.array([mean[i]] if i < mean.size else [], float))
        data.append((name, groups))
    return _draw(fig, list(keys), data, ylabel, st)


def series_table(series: Dict[str, np.ndarray], labels, conditions, by_condition: bool, err: str = "sd"):
    """The numbers of a graph: values per lane, then mean / SD (SEM) / n per lane group.
    Returns (header, rows); summary rows start with an empty first cell."""
    names = list(series)
    vals = [np.atleast_1d(np.asarray(series[k], float)) for k in names]
    n = max((v.size for v in vals), default=0)
    header = ["레인", "레인 그룹"] + names
    rows = []
    for i in range(n):
        rows.append([str(i + 1), (conditions[i] if i < len(conditions) and conditions[i] else
                                  (labels[i] if i < len(labels) else ""))]
                    + [_fmt(v[i]) if i < v.size else "" for v in vals])
    if by_condition and n:
        lab_err = "SEM" if err == "sem" else "SD"
        for key, idx in x_groups(n, labels, conditions, True):
            stats = [summary(v[[i for i in idx if i < v.size]], err) for v in vals]
            rows.append(["", f"{key} 평균"] + [_fmt(m) for m, _, _ in stats])
            rows.append(["", f"{key} {lab_err}"] + [_fmt(e) if c > 1 else "" for _, e, c in stats])
            rows.append(["", f"{key} n"] + [str(c) for _, _, c in stats])
    return header, rows


def summary_table(keys: List[str], series: Dict[str, Dict[str, object]], err: str = "sd"):
    """Tab 5: per lane key the mean, SD/SEM and n of each series."""
    lab_err = "SEM" if err == "sem" else "SD"
    header = ["레인"]
    for name in series:
        header += [f"{name} 평균", f"{name} {lab_err}", f"{name} n"]
    rows = []
    for i, key in enumerate(keys):
        row = [key]
        for s in series.values():
            pts = s.get("points")
            if pts:
                m, e, c = summary([p[i] for p in pts if i < len(p)], err)
            else:
                mean = np.atleast_1d(np.asarray(s["mean"], float))
                m, e, c = (float(mean[i]) if i < mean.size else float("nan")), float("nan"), 1
            row += [_fmt(m), _fmt(e) if c > 1 else "", str(c)]
        rows.append(row)
    return header, rows


def _fmt(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    return f"{x:.6g}" if np.isfinite(x) else ""
