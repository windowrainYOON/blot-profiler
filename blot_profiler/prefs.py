"""Defaults for the look of graphs and figures and for exporting (File → 설정).

Kept per computer in a small JSON file (not in projects), so every project starts
from the same defaults. ``BLOT_PROFILER_PREFS`` points elsewhere (tests).
"""

from __future__ import annotations

import copy
import json
import os
import sys
from .i18n import _t, prefs_path  # noqa: E402

DEFAULTS = {
    "general": {
        "language": "system",  # system · ko · en (1.4; applied at the next start)
    },
    "graph": {
        "font": "Arial",
        "label_size": 14.0,  # axis titles
        "tick_size": 12.0,  # tick labels
        "legend_size": 11.0,
        "bold": True,
        "axis_lw": 1.6,
        "bar_lw": 1.6,  # bar outlines
        "err_lw": 1.6,  # error bars
        "point_size": 34,
        "cap": 7,  # error bar cap length (points)
        "tick_len": 6.0,  # tick length (points)
        "tick_dir": "out",  # out · in
        "title_size": 14.0,
        "kind": "bar_points",  # bar_points · points_mean · bars
        "err": "sd",  # sd · sem
        "colors": "bw",  # bw (PRISM default) · gray · color
        "y0": True,  # y axis from 0
    },
    "figure": {
        "font": "Arial",
        "fonts": {"lane": 8.0, "group": 8.0, "cond": 8.0, "protein": 8.0, "kda": 6.8, "number": 7.2},
        "label_bold": True,
        "frame_lw": 0.8,
        "group_bar_lw": 2.2,
        "width_in": 3.2,
        "panel_gap_in": 0.06,
        "export_dpi": 600,
    },
    "export": {
        "image_format": "png",  # png · pdf · svg · tif
        "folder": "image",  # image (next to the blot) · last · desktop
        "last_folder": "",
    },
}

GRAPH_KINDS = [("bar_points", _t("막대 + 개별 점 (Column bar + scatter)")), ("points_mean", _t("점 + 평균선 (Scatter dot plot)")),
               ("bars", _t("막대만"))]
ERRORS = [("sd", _t("평균 ± SD")), ("sem", _t("평균 ± SEM"))]
COLOR_SCHEMES = [("bw", _t("흑백 (PRISM 기본)")), ("gray", _t("회색 단계")), ("color", _t("컬러"))]
IMAGE_FORMATS = [("png", "PNG"), ("pdf", "PDF"), ("svg", "SVG"), ("tif", "TIFF")]
FOLDERS = [("image", _t("이미지 파일 옆")), ("last", _t("마지막으로 저장한 폴더")), ("desktop", _t("바탕화면"))]


def path() -> str:
    return prefs_path()


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif k in out or not isinstance(base, dict) or not base:
            out[k] = v
    return out


def load() -> dict:
    try:
        with open(path(), encoding="utf-8") as f:
            return _merge(DEFAULTS, json.load(f))
    except (OSError, ValueError):
        return copy.deepcopy(DEFAULTS)


def save(p: dict) -> None:
    fp = path()
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp + ".saving"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fp)


def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)


def export_folder(p: dict, image_path: str = "") -> str:
    """Where save dialogs start."""
    e = p.get("export", {})
    if e.get("folder") == "last" and e.get("last_folder") and os.path.isdir(e["last_folder"]):
        return e["last_folder"]
    if e.get("folder") == "desktop":
        return os.path.expanduser("~/Desktop")
    if image_path:
        return os.path.dirname(os.path.abspath(image_path))
    return os.path.expanduser("~")
