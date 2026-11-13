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
    w.detect(auto=True)  # 1.2: the window waits for drawn lines
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
    w.extra_bands.setChecked(False)  # default: the red lines are the rows
    w.detect(use_lines=True)
    assert len(w.analysis.rows) == m + 1
    assert {p["row"] for p in w.analysis.protein_lines} == {r.name for r in w.analysis.rows}
    # [ and ] zoom
    k = v.transform().m11()
    QTest.keyClick(v, QtCore.Qt.Key_BracketLeft)
    assert v.transform().m11() > k
    # second image, quantification window: two images pooled by lane name
    w.load_image(str(tmp_path / "b.tif"))
    w.detect(auto=True)  # 1.2: the window waits for drawn lines
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
    # several images in one figure (Board B): tick I2 in 그림에 넣을 블롯, its panel joins the figure
    ed = w.figure_editor
    w.fig_win.layout_seg.set_value("rows", emit=True)
    ed.s["layout"] = "rows"
    other = next(d for d in w.loaded_docs() if d is not ed.doc)
    ed.toggle_image(other, True)
    ed.render()
    lay = ed.preview.layout_
    assert any(p_.get("doc") == ed.doc_id(other) for p_ in lay["panels"])
    assert any(k_.startswith("src:") for k_ in (t["key"] for t in lay["texts"]))  # source tags I1 / I2
    fp = next(p_ for p_ in ed.s["panels"] if p_.get("doc"))
    ed.preview.select({"kind": "panel", "row": fp["row"]})
    assert ed.props_layout.count() > 2  # the other image's panel cards
    assert any(k_.startswith("src:") for k_, _, _ in ed.number_sources())
    ed.write_figure(str(tmp_path / "multi.png"), 100)
    assert (tmp_path / "multi.png").exists()
    ed.toggle_image(other, False)
    assert not any(p_.get("doc") for p_ in ed.s["panels"])


def test_figure_numbers_old_names_and_dropped_lanes():
    from blot_profiler.figure_window import _number_key

    values = {"res:P2 ÷ P1": 1, "net:P1": 2}
    assert _number_key("res:P2/P1", values) == "res:P2 ÷ P1"
    assert _number_key("net:P1", values) == "net:P1"
    a = core.Analysis(rows=[core.Row("C", 10, 6)])
    a.bands = [core.Band(0, l, 20 + 30 * l, 50) for l in range(4)]
    a.ensure_lane_labels()
    core.sync_lane_roles(a)
    a.lane_roles[1] = core.LANE_EXCLUDE
    out = core.lanes_from_results(a, [1.0, 3.0, 4.0])
    assert np.isnan(out[1]) and list(out[[0, 2, 3]]) == [1.0, 3.0, 4.0]


def test_graph_fits_small_canvas():
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from blot_profiler import graph

    labels = ["DMSO", "DMSO", "Drug 1 µM", "Drug 1 µM", "Drug 10 µM", "Drug 10 µM"]
    fig = Figure(figsize=(4.6, 3.4), dpi=100)
    FigureCanvasAgg(fig)
    graph.plot_series(fig, {"A ÷ B": np.arange(6) + 1.0, "C ÷ B": np.arange(6) * 0.5 + 1}, labels, labels, True,
                      "Relative intensity", graph.style())
    r = fig.canvas.get_renderer()
    fig.draw(r)
    fb = fig.bbox
    ax = fig.axes[0]
    for art in [ax.get_legend(), ax.yaxis.label] + list(ax.get_xticklabels()):
        e = art.get_window_extent(r)
        assert e.x0 >= fb.x0 - 1 and e.x1 <= fb.x1 + 1 and e.y0 >= fb.y0 - 1 and e.y1 <= fb.y1 + 1
    assert matplotlib  # (imported for the backend)


def test_red_lines_lead_detection():
    """A red line a few px off a band row: the boxes snap to the bands; rows come from the lines
    (named after them), and 'extra' adds the automatic rows away from every line."""
    corr = _blot()
    auto = lanes.detect(corr)
    core.lines_from_analysis(auto, corr.shape)
    assert len(auto.rows) >= 2
    p = dict(auto.protein_lines[0], row="Vinculin")
    true_y = [b.cy for b in auto.bands if b.row == 0]
    off = 0.4 * auto.rows[0].height
    p["y0"] += off
    p["y1"] += off
    a = lanes.detect(corr, protein_lines=[p], only_on_lines=True)
    assert [r.name for r in a.rows] == ["Vinculin"]
    ys = sorted(b.cy for b in a.bands)
    assert abs(float(np.median(ys)) - float(np.median(true_y))) < 0.5 * off + 1.0  # snapped to the band, not the line
    b = lanes.detect(corr, protein_lines=[p], only_on_lines=False)
    assert b.rows[0].name == "Vinculin" and len(b.rows) == len(auto.rows)


def test_multi_blot_figure_panels():
    """Panels from another image stack under this image's panels on the same lane columns;
    a missing lane is a white cell; a panel whose image is not open keeps a grey place."""
    from matplotlib.figure import Figure
    from blot_profiler import figure

    corr = _blot()
    a = lanes.detect(corr)
    b = lanes.detect(corr)
    b_sig = corr[:, : int(corr.shape[1] * 0.8)]  # the other image: fewer lanes
    b = lanes.detect(b_sig)
    s = figure.sync_settings(figure.default_settings(), a, corr.shape)
    n_own = len(s["panels"])
    other = {"row": f"{b.rows[0].name} · other.tif", "src_row": b.rows[0].name, "doc": "/x/other.tif",
             "show": True, "label": "IB: other", "pad_top": 4, "pad_bottom": 4}
    gone = dict(other, row="gone", doc="/x/gone.tif")
    s["panels"] += [other, gone]
    figure.sync_settings(s, a, corr.shape)  # keeps the other images' panels in place
    assert [p.get("doc") for p in s["panels"][n_own:]] == ["/x/other.tif", "/x/gone.tif"]
    sources = {"/x/other.tif": {"signal": b_sig, "analysis": b, "tag": "I2"}, "__own__": {"tag": "I1"}}
    lay = figure.render(Figure(), corr, a, s, sources=sources, preview=True)
    docs = [p.get("doc") for p in lay["panels"]]
    assert docs.count("/x/other.tif") == 1 and docs.count("/x/gone.tif") == 1
    own_c = figure.lane_centres(a)
    strip = figure.foreign_strip(b_sig, b, 0, 10, figure.default_lane_map(len(own_c), b.n_lanes), own_c,
                                 int(min(c for c in own_c) - 5), int(max(own_c) + 5), 1.0)
    if b.n_lanes < a.n_lanes:
        assert np.isnan(strip[:, -1]).all()  # the last figure column has no lane in the other image
    # without sources (export of an old project): the other images' panels are left out
    lay2 = figure.render(Figure(), corr, a, s)
    assert all(not p.get("doc") for p in lay2["panels"])
