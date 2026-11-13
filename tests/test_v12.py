"""1.2: the window waits for drawn lines, PowerPoint-style Figure editing (drag lanes → group,
new name tiers prefilled, tiers moved whole), blot/marker regions in the import window."""

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402
from blot_profiler import core, figure  # noqa: E402


def _window(tmp_path, raw=None, name="a.tif"):
    import tifffile
    from PySide6 import QtWidgets
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    if raw is None:
        raw, _ = synth.chemi_blot()
    tifffile.imwrite(str(tmp_path / name), raw)
    w = gui.MainWindow()
    w.resize(1400, 900)
    w.show()
    w.ask_marker.setChecked(False)
    w.load_image(str(tmp_path / name))
    return w


def test_nothing_is_found_before_the_user_draws_lines(tmp_path):
    w = _window(tmp_path)
    a = w.analysis
    assert not a.bands and not a.lane_lines and not a.protein_lines
    assert not w.find_btn.isEnabled()
    w.detect()  # still nothing without lines
    assert not w.analysis.bands
    w.find_lane_lines()
    H = w.doc.raw.shape[0]
    assert not w.find_btn.isEnabled()  # lanes only: still waiting for a protein line
    auto = core.Analysis()
    w.add_protein_line(5, H * 0.5, w.doc.raw.shape[1] - 5, H * 0.5)
    assert w.find_btn.isEnabled()
    del auto


def test_fill_patterns_and_new_tier_prefilled():
    assert figure.fill_texts("alt", 4) == ["–", "+", "–", "+"]
    assert figure.fill_texts("pairs", 5) == ["–", "–", "+", "+", "–"]
    assert figure.fill_texts("numbers", 3) == ["1", "2", "3"]
    assert figure.fill_texts("copy", 2, below=["a", "b", "c"]) == ["a", "b"]
    t = figure.default_tier("lanes", 3)
    assert t["texts"] == ["–", "–", "–"]


def test_figure_drag_group_type_and_move_tier(tmp_path):
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    w = _window(tmp_path)
    w.detect(auto=True)
    w.open_figure_window()
    ed = w.figure_editor
    ed.render()
    pv = ed.preview
    L = pv.layout_
    k, ox, oy = pv._k()

    def scr(x, y):
        return QtCore.QPoint(int(ox + x * k), int(oy + y * k))

    t0 = L["tiers"][0]
    xs = L["lane_xs"]
    ym = (t0["y0"] + t0["y1"]) / 2
    QTest.mousePress(pv, QtCore.Qt.LeftButton, pos=scr(xs[1], ym))
    QTest.mouseMove(pv, scr(xs[3], ym))
    QTest.mouseRelease(pv, QtCore.Qt.LeftButton, pos=scr(xs[3], ym))
    assert pv.selected["kind"] == "cells" and list(pv.selected["lanes"]) == [1, 2, 3]
    ed.make_group(pv.selected["ti"], pv.selected["lanes"])
    QTest.keyClicks(pv.editor, "Drug")
    QTest.keyClick(pv.editor, QtCore.Qt.Key_Return)
    groups = [t for t in ed.s["tiers"] if t["kind"] == "groups"]
    assert groups and groups[0]["groups"][0]["text"] == "Drug"
    n = len(ed.s["tiers"])
    ed.add_from_list("lanes")
    assert len(ed.s["tiers"]) == n + 1
    new = [t for t in ed.s["tiers"] if t["kind"] == "lanes" and not t.get("auto")]
    assert new and all(x == figure.NEW_CELL for x in new[-1]["texts"])
    kinds = [t["kind"] for t in ed.s["tiers"]]
    ed.move_tier_to(0, 1)
    assert [t["kind"] for t in ed.s["tiers"]] == [kinds[1], kinds[0]] + kinds[2:]


def test_marker_region_lands_on_blot_region():
    marker = np.zeros((100, 200))
    marker[10:30, 20:60] = 1.0
    out = core.place_marker_region(marker, [20, 10, 60, 30], [100, 50, 180, 90], (100, 200))
    assert out.shape == (100, 200)
    assert out[60:80, 110:170].mean() > 0.9 and out[:40].mean() < 0.1
    tf = dict(core.default_transform(), region=[100, 50, 180, 90])
    assert core.apply_transform(out, tf).shape == (40, 80)


def test_import_regions_open_as_tabs(tmp_path):
    from PySide6 import QtWidgets
    from blot_profiler import gui

    raw, _ = synth.chemi_blot()
    two = np.concatenate([raw, raw], axis=0)
    w = _window(tmp_path, two, "two.tif")
    H, W = two.shape
    dlg = gui.MarkerStepDialog(w, w.doc, None)
    dlg.blot_thumb.regions = [[0, 0, W, H // 2], [0, H // 2, W, H]]
    dlg.refresh_regions()
    assert dlg.ok.isEnabled() and "2" in dlg.ok.text()
    dlg.exec = lambda: QtWidgets.QDialog.Accepted
    orig = gui.MarkerStepDialog
    gui.MarkerStepDialog = lambda *a: dlg
    try:
        w.ask_marker_for(w.doc)
    finally:
        gui.MarkerStepDialog = orig
    assert len(w.docs) == 2
    assert [d.raw.shape for d in w.docs] == [(H // 2, W), (H - H // 2, W)]
    assert w.docs[1].name.endswith("#2")
