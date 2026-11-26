"""1.5.2: 마커 정의 step: marker strip and sizes for every image, also on a marker lane of the blot."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from blot_profiler import core, figure  # noqa: E402


def test_find_marker_bands_on_marker_photo_and_gel_lane():
    ys = core.find_marker_bands(synth.marker_photo(), 60, 132)
    assert [round(y) for y in ys] == list(synth.LADDER_Y.values())
    rgb, lanes_, lad = synth.gel_photo()
    cx = int(lanes_[0])
    ys = core.find_marker_bands(rgb, cx - 20, cx + 20)
    for y in lad:
        assert min(abs(y - v) for v in ys) <= 2


def test_copy_marker_sizes_by_band_order():
    src = {"ladder": "PageRuler (180–10 kDa)", "unit": "kDa", "kda": [{"kda": 180.0, "y": 10}, {"kda": 130.0, "y": 40}]}
    dst = {"kda": [{"kda": None, "y": 15}, {"kda": None, "y": 50}, {"kda": None, "y": 90}]}
    core.copy_marker_sizes(src, dst)
    assert [k["kda"] for k in dst["kda"]] == [180.0, 130.0, None]
    assert [k["y"] for k in dst["kda"]] == [15, 50, 90] and dst["ladder"] == src["ladder"]


def test_marker_step_in_the_app(tmp_path):
    from PySide6 import QtWidgets
    import tifffile
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    tifffile.imwrite(str(tmp_path / "a-Ch.tif"), raw)
    tifffile.imwrite(str(tmp_path / "a-Marker.tif"), synth.marker_photo())
    tifffile.imwrite(str(tmp_path / "b.tif"), raw)
    w = gui.MainWindow()
    w._auto_accept = True
    w.ask_marker.setChecked(False)
    w.load_image(str(tmp_path / "a-Ch.tif"))
    assert w.set_marker(w.doc, str(tmp_path / "a-Marker.tif"))
    w.load_image(str(tmp_path / "b.tif"))
    w.switch_doc(0)
    w.set_step(w.STEP_MARKER)
    assert w.step_list.count() == 3 and w.view.tool == "marker"
    assert w.img_list.count() == 4  # every image with its marker row under it
    assert not w.marker_in_blot()
    s = w._marker_s()
    s.update(strip_x0=60, strip_x1=132)
    w.auto_marker_bands()
    assert len(s["kda"]) == 10
    w.ladder_combo.setCurrentIndex(w.ladder_combo.findData("PageRuler (180–10 kDa)"))
    w.set_marker_size(0, "180")
    w.fill_marker_sizes()
    assert [k["kda"] for k in s["kda"]] == core.LADDER_PRESETS_KDA["PageRuler (180–10 kDa)"]
    assert w.marker_status(w.doc)[1] == "ok"
    # the second blot has no marker photo: its strip goes on a lane of the blot itself
    w.switch_doc(1)
    assert w.marker_in_blot() and w.view.marker_in_blot
    w.detect(auto=True)
    a = w.analysis
    cx = float(np.median([b.cx for b in a.bands if b.lane == 0]))
    s2 = w._marker_s()
    s2.update(strip_x0=int(cx - 30), strip_x1=int(cx + 30))
    s2["kda"] = [{"kda": None, "y": 180.0}, {"kda": None, "y": 370.0}]
    w.on_marker_edited("x")
    assert a.lane_roles[0] == core.LANE_MARKER and a.lane_roles[1] == core.LANE_SAMPLE
    w.switch_doc(0)
    w.copy_marker_sizes()
    assert [k["kda"] for k in w.docs[1].figure["kda"]] == [180.0, 130.0]
    # the Figure takes the strip from the blot's own marker lane
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    d = w.docs[1]
    d.figure.update(show_strip=True, show_kda=True)
    fs = figure.sync_settings(d.figure, a, d.signal.shape)
    lay = figure.render(Figure(), d.signal, a, fs, None)
    assert lay["strips"]
