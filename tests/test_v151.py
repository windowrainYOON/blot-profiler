"""1.5.1: Full blot shows the whole selected blot region (tilt-corrected), every image in its own frame."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from blot_profiler import core, figure, lanes  # noqa: E402


def _blot(angle=0.0):
    raw, _ = synth.chemi_blot()
    raw = core.apply_transform(raw, dict(core.default_transform(), angle=angle))
    sig = core.to_signal(raw.astype(float), invert=core.needs_inversion(raw))
    corr = core.corrected_signal(sig, core.BackgroundSettings())
    return corr, lanes.detect(corr)


def test_full_blot_is_the_whole_region():
    corr, a = _blot(angle=1.5)
    H, W = corr.shape
    s = figure.sync_settings(figure.default_settings(), a, corr.shape)
    s["layout"] = "full"
    panels, x0, x1 = figure.layout_panels(a, s, corr.shape)
    (_, _, y0, y1), = panels
    assert (x0, x1, y0, y1) == (0, W, 0, H)


def test_old_band_trimmed_crop_is_dropped():
    corr, a = _blot()
    s = figure.default_settings()
    s.update(layout="full", gel_crop=[20, 20, 60, 60])  # saved by 1.5.0
    figure.sync_settings(s, a, corr.shape)
    assert "gel_crop" not in s
    _, x0, x1 = figure.layout_panels(a, s, corr.shape)
    assert (x0, x1) == (0, corr.shape[1])
    s["gel_box"] = [10, 5, 100, 80]  # trimmed by hand: kept
    panels, x0, x1 = figure.layout_panels(a, s, corr.shape)
    assert (x0, x1, panels[0][2], panels[0][3]) == (10, 100, 5, 80)


def test_other_blot_keeps_its_whole_height_in_its_own_frame():
    from matplotlib.figure import Figure

    corr, a = _blot()
    corr2, b = _blot(angle=2.0)  # another blot (another region of the photo), tilt-corrected
    s = figure.sync_settings(figure.default_settings(), a, corr.shape)
    s["layout"] = "full"
    s["panels"].append({"row": "x · o", "src_row": b.rows[0].name, "doc": "o", "show": True, "label": "IB: x"})
    figure.sync_settings(s, a, corr.shape)
    sources = {"o": {"signal": corr2, "analysis": b, "tag": "I2", "figure": figure.default_settings()},
               "__own__": {"tag": "I1"}}
    panels, _, _ = figure.layout_panels(a, s, corr.shape, sources)
    other = next(p for p in panels if p[0].get("gel"))
    assert (other[2], other[3]) == (0, corr2.shape[0])  # top to bottom, nothing cut
    lay = figure.render(Figure(), corr, a, s, sources=sources, preview=True)
    assert [p["row"] for p in lay["panels"]] == [figure.GEL_ROW, f"{figure.GEL_ROW}:o"]
    assert s["frame_lw"] > 0  # each blot drawn in its own black frame
