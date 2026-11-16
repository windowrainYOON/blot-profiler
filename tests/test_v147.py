"""1.4.7: Full blot and the marker (strip + kDa ticks) with several images in one Figure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from blot_profiler import core, figure, lanes  # noqa: E402


def _setup():
    raw, _ = synth.chemi_blot()
    sig = core.to_signal(raw.astype(float), invert=core.needs_inversion(raw))
    corr = core.corrected_signal(sig, core.BackgroundSettings())
    a = lanes.detect(corr)
    b = lanes.detect(corr)
    mk = synth.marker_photo().astype(float)
    s = figure.sync_settings(figure.default_settings(), a, corr.shape)
    s["panels"].append({"row": f"{b.rows[1].name} · other", "src_row": b.rows[1].name, "doc": "o", "show": True,
                        "label": "IB: other", "pad_top": 4, "pad_bottom": 4})
    figure.sync_settings(s, a, corr.shape)
    fs = figure.default_settings()
    ky = float(np.mean([q.cy for q in b.bands if q.row == 1]))  # inside the other image's panel
    fs.update(strip_x0=10, strip_x1=60, kda=[{"kda": 75, "y": ky}])
    sources = {"o": {"signal": corr, "analysis": b, "tag": "I2", "figure": fs, "marker": mk}, "__own__": {"tag": "I1"}}
    return corr, a, s, mk, sources


def test_full_blot_shows_every_ticked_image():
    from matplotlib.figure import Figure

    corr, a, s, mk, sources = _setup()
    s["layout"] = "full"
    lay = figure.render(Figure(), corr, a, s, sources=sources, preview=True)
    rows = [p["row"] for p in lay["panels"]]
    assert rows == [figure.GEL_ROW, f"{figure.GEL_ROW}:o"]
    assert any(t["text"] == "IB: other" for t in lay["texts"])  # its protein named at its height
    # its image closed: a grey place keeps the spot
    lay = figure.render(Figure(), corr, a, s, sources={"__own__": {"tag": "I1"}}, preview=True)
    assert [p["row"] for p in lay["panels"]] == [figure.GEL_ROW, f"{figure.GEL_ROW}:o"]


def test_other_image_keeps_its_marker_and_kda():
    from matplotlib.figure import Figure

    corr, a, s, mk, sources = _setup()
    s["show_strip"], s["show_kda"] = True, True
    lay = figure.render(Figure(), corr, a, s, sources=sources, preview=True)
    assert any(st.get("doc") == "o" for st in lay["strips"])  # the other image's own marker strip
    assert any(t["key"].startswith("xkda:o:") and t["text"] == "75" for t in lay["texts"])
    s["show_strip"], s["show_kda"] = False, False  # the figure's switches turn them off for every image
    lay = figure.render(Figure(), corr, a, s, sources=sources, preview=True)
    assert not lay["strips"] and not any(t["cat"] == "kda" for t in lay["texts"])
