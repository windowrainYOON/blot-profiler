"""1.0: lane lines and protein lines, ROI fill, ladder, lane roles, canvas tools, the quantification
and Figure windows."""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

import synth  # noqa: E402
from blot_profiler import core, lanes  # noqa: E402


def _blot():
    raw, _ = synth.chemi_blot()
    sig = core.to_signal(raw.astype(float), invert=core.needs_inversion(raw))
    return core.corrected_signal(sig, core.BackgroundSettings())


def test_detect_along_user_lines_and_on_red_lines_only():
    corr = _blot()
    H = corr.shape[0]
    auto = lanes.detect(corr)
    core.lines_from_analysis(auto, corr.shape)
    assert len(auto.lane_lines) == auto.n_lanes and len(auto.protein_lines) == len(auto.rows)
    # a bent line: the middle point moved sideways; detection follows it
    lines = [[list(p) for p in ln] for ln in auto.lane_lines]
    x_mid = core.poly_x(lines[0], H / 2)
    lines[0] = [lines[0][0], [float(x_mid) + 3.0, H / 2], lines[0][-1]]
    a = lanes.detect(corr, lane_lines=lines, protein_lines=auto.protein_lines)
    assert a.n_lanes == len(lines)
    assert [r.name for r in a.rows][:1] == [auto.protein_lines[0]["row"]]
    # only on the red lines: one box per line and lane, named after the line
    pl = [dict(auto.protein_lines[0], row="Vinculin")]
    b = lanes.detect(corr, lane_lines=lines, protein_lines=pl, only_on_lines=True)
    assert [r.name for r in b.rows] == ["Vinculin"] and len(b.bands) == len(lines)
    ys = [bd.cy for bd in b.bands]
    assert max(abs(y - core.protein_line_y(pl[0], bd.cx)) for y, bd in zip(ys, b.bands)) < b.rows[0].height


def test_fill_row_ladder_and_lane_roles():
    a = core.Analysis(rows=[core.Row("C", 10, 6, role=core.ROLE_CONTROL), core.Row("T", 10, 6)])
    a.lane_lines = [[[20.0 + 30 * i, 0.0], [20.0 + 30 * i + 6, 100.0]] for i in range(4)]  # slanted lanes
    a.bands = [core.Band(0, l, 20 + 30 * l + 3, 50) for l in range(4)] + [core.Band(1, 0, 23, 70)]
    a.ensure_lane_labels()
    a.lane_labels = ["L1", "L2", "L3", "L4"]
    assert core.fill_row(a, 1) == 3
    t = [b for b in a.bands if b.row == 1]
    assert sorted(b.lane for b in t) == [0, 1, 2, 3]
    assert all(abs(b.cx - core.poly_x(a.lane_lines[b.lane], 70)) < 1e-6 for b in t if b.user)
    # ladder: one size fills the others in order
    assert core.fill_ladder(5, 2, 70, core.DEFAULT_LADDER) == [130, 100, 70, 55, 40]
    assert core.fill_ladder(3, 0, 15, core.DEFAULT_LADDER) == [15, 10, None]
    # a marker lane leaves the results
    core.sync_lane_roles(a)
    a.lane_roles[3] = core.LANE_MARKER
    sig = np.ones((120, 140))
    meas = core.measure(sig, a, core.BackgroundSettings(method="none"))
    v, vm = core.basis_view(a, meas, None)
    assert v.lane_labels == ["L1", "L2", "L3"] and {m.lane for m in vm} == {0, 1, 2}


def test_windows_and_canvas_tools(tmp_path):
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtTest import QTest

    import tifffile
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    for n in ("a.tif", "b.tif"):
        tifffile.imwrite(str(tmp_path / n), raw)
    w = gui.MainWindow()
    w.resize(1400, 900)
    w.show()
    w.ask_marker.setChecked(False)
    w.load_image(str(tmp_path / "a.tif"))
    a = w.analysis
    assert a.lane_lines and a.protein_lines
    v = w.view
    QTest.qWait(50)
    # lane tool: click on empty blot -> new line; double-click -> new point; ⌫ removes the point
    w.tool_seg.set_value("lane", emit=True)
    n = len(a.lane_lines)
    xs = sorted(float(np.mean([p[0] for p in ln])) for ln in a.lane_lines)
    x_new = xs[-1] + (xs[-1] - xs[-2]) * 0.5
    pos = v.mapFromScene(QtCore.QPointF(x_new, 40))
    QTest.mouseClick(v.viewport(), QtCore.Qt.LeftButton, pos=pos)
    assert len(a.lane_lines) == n + 1
    i = v.sel["i"]
    QTest.mouseDClick(v.viewport(), QtCore.Qt.LeftButton, pos=v.mapFromScene(QtCore.QPointF(x_new, 120)))
    assert len(a.lane_lines[i]) == 3
    v.setFocus()
    QTest.keyClick(v, QtCore.Qt.Key_Backspace)
    assert len(a.lane_lines[i]) == 2
    QTest.keyClick(v, QtCore.Qt.Key_Backspace)  # no point selected now: the line goes
    assert len(a.lane_lines) == n
    # protein tool: drag sideways -> a red line
    w.tool_seg.set_value("protein", emit=True)
    m = len(a.protein_lines)
    p0, p1 = v.mapFromScene(QtCore.QPointF(xs[0], 300)), v.mapFromScene(QtCore.QPointF(xs[-1], 304))
    QTest.mousePress(v.viewport(), QtCore.Qt.LeftButton, pos=p0)
    QTest.mouseMove(v.viewport(), p1)
    QTest.mouseRelease(v.viewport(), QtCore.Qt.LeftButton, pos=p1)
    assert len(a.protein_lines) == m + 1
    # detect again along the lines
    w.only_on_lines.setChecked(True)
    w.detect(use_lines=True)
    assert len(w.analysis.rows) == m + 1
    # [ and ] zoom
    k = v.transform().m11()
    QTest.keyClick(v, QtCore.Qt.Key_BracketLeft)
    assert v.transform().m11() > k
    # second image, quantification window: two images pooled by lane name
    w.load_image(str(tmp_path / "b.tif"))
    w.open_quant_window()
    pooled = w._pooled
    assert len(pooled) == 2
    xl, data = w.graph_xy()
    assert data and max(len(g) for _, groups in data for g in groups) >= 2
    w.table_seg.set_value("bands", emit=True)
    assert w.results_table.rowCount() == sum(len(d.measurements) for d in w.quant_docs())
    w.results_table.selectRow(0)
    assert w.view.selected_bands()
    # Figure window: Full blot keeps protein names
    w.open_figure_window()
    w.fig_win.layout_seg.set_value("full", emit=True)
    w.figure_editor.render()
    texts = [t["key"] for t in w.figure_editor.preview.layout_["texts"]]
    assert any(k_.startswith("protein:") for k_ in texts)
