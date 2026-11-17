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


def test_detect_inspector_fits_its_column(tmp_path):
    """1.4.8: the 검출 card grew wider than the inspector (cut off on the right) after the
    '단백질 선으로 수평 맞추기' row came in; long row titles wrap now."""
    from PySide6 import QtWidgets
    from PySide6.QtTest import QTest
    import tifffile
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    tifffile.imwrite(str(tmp_path / "a.tif"), raw)
    w = gui.MainWindow()
    w.resize(1280, 800)
    w.show()
    w.ask_marker.setChecked(False)
    w.load_image(str(tmp_path / "a.tif"))
    w.detect(auto=True)
    QTest.qWait(100)
    sa = w.kind_seg
    while not isinstance(sa, QtWidgets.QScrollArea):
        sa = sa.parentWidget()
    assert sa.widget().minimumSizeHint().width() <= 300  # room for a scroll bar and wider Mac fonts


def test_full_blot_images_move_and_resize(tmp_path):
    """1.4.8: in Full blot the images can be dragged up/down and another image's whole blot
    can be cropped by its edges (it has its own crop)."""
    from PySide6.QtTest import QTest
    from test_v141 import _boxes, _window

    w, ed = _window(tmp_path, n=2)
    _boxes(ed)["I1"].click()
    QTest.qWait(30)
    ed.s["layout"] = "full"
    ed.load_from_settings()
    ed.render()
    rows = [p["row"] for p in ed.preview.layout_["panels"]]
    assert rows[0] == figure.GEL_ROW and rows[1].startswith(figure.GEL_ROW + ":")
    ed.move_panel_to(rows[1], 0)
    assert [p["row"] for p in ed.preview.layout_["panels"]] == [rows[1], rows[0]]
    h0 = next(p["h"] for p in ed.preview.layout_["panels"] if p["row"] == rows[1])
    ed.on_crop_dragged("bottom", rows[1], 0.1)
    h1 = next(p["h"] for p in ed.preview.layout_["panels"] if p["row"] == rows[1])
    assert h1 > h0 + 0.05
    other = w.loaded_docs()[0]
    assert other.figure and other.figure.get("gel_crop")  # its own crop changed, not this image's
