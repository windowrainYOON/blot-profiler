"""1.3: ] zooms in, the import window for every image (kind = marker image or not),
the canvas crop button reopens it, no glass off macOS."""

import os
import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402
from blot_profiler import core  # noqa: E402


def _window(tmp_path, raw):
    import tifffile
    from PySide6 import QtWidgets
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    tifffile.imwrite(str(tmp_path / "a.tif"), raw)
    w = gui.MainWindow()
    w.resize(1400, 900)
    w.show()
    w.ask_marker.setChecked(False)
    w.load_image(str(tmp_path / "a.tif"))
    return w


def _accept(gui, dlg, fn):
    from PySide6 import QtWidgets

    dlg.exec = lambda: QtWidgets.QDialog.Accepted
    orig = gui.MarkerStepDialog
    gui.MarkerStepDialog = lambda *a, **k: dlg
    try:
        fn()
    finally:
        gui.MarkerStepDialog = orig


def test_kind_follows_marker_and_crop_button_reopens(tmp_path):
    from PySide6 import QtCore
    from PySide6.QtTest import QTest
    from blot_profiler import gui
    from blot_profiler.ui import glass

    raw, _ = synth.chemi_blot()
    w = _window(tmp_path, raw)
    d = w.doc
    assert d.mode == core.MODE_GEL  # no marker image next to it: 마커 따로 없음
    assert not glass.active()
    k = w.view.transform().m11()
    w.view.setFocus()
    QTest.keyClick(w.view, QtCore.Qt.Key_BracketRight)
    assert w.view.transform().m11() > k
    QTest.keyClick(w.view, QtCore.Qt.Key_BracketLeft)
    QTest.keyClick(w.view, QtCore.Qt.Key_BracketLeft)
    assert w.view.transform().m11() < k
    # reopen from the crop button: draw a region -> only that part, lines cleared
    H, W = raw.shape
    dlg = gui.MarkerStepDialog(w, d, None, initial={"blot": None, "marker": None, "marker_tf": {}})
    dlg.blot_thumb.regions = [[0, 0, W // 2, H]]
    dlg.refresh_regions()
    assert w.region_btn.isVisible()
    _accept(gui, dlg, w.region_btn.click)
    assert d.raw.shape == (H, W // 2)
    assert d.mode == core.MODE_GEL
    # reopen again: the region is shown; clear it -> the whole image
    dlg = gui.MarkerStepDialog(w, d, None, initial={"blot": d.transform["region"], "marker": None, "marker_tf": {}})
    assert dlg.blot_thumb.regions
    dlg.blot_thumb.regions = []
    dlg.refresh_regions()
    _accept(gui, dlg, w.edit_regions)
    assert d.raw.shape == (H, W)


def test_marker_attached_means_separate_marker_kind(tmp_path):
    import tifffile
    from blot_profiler import gui

    raw, _ = synth.chemi_blot()
    w = _window(tmp_path, raw)
    m = (raw.max() - raw).astype(raw.dtype)
    tifffile.imwrite(str(tmp_path / "a-Marker.tif"), m)
    dlg = gui.MarkerStepDialog(w, w.doc, str(tmp_path / "a-Marker.tif"))
    assert dlg.path
    _accept(gui, dlg, lambda: w.ask_marker_for(w.doc))
    assert w.doc.mode == core.MODE_WESTERN and w.doc.marker is not None
    assert w.kind_seg.value() == core.MODE_WESTERN if hasattr(w.kind_seg, "value") else True
