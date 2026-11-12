"""PRISM-style graphs, the data behind them, defaults (File → 설정) and exporting."""

import csv
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

from matplotlib.figure import Figure  # noqa: E402

from blot_profiler import figure, graph, prefs  # noqa: E402

LABELS = ["Veh", "Veh", "A", "A", "A"]


def test_series_table_has_only_the_graph_numbers():
    s = {"P2/P1": np.array([1.0, 0.8, 1.5, 1.3, 1.2])}
    header, rows = graph.series_table(s, LABELS, LABELS, True, "sd")
    assert header == ["레인", "레인 그룹", "P2/P1"]
    assert [r[0] for r in rows[:5]] == ["1", "2", "3", "4", "5"]
    summ = {r[1]: r[2] for r in rows[5:]}
    assert float(summ["Veh 평균"]) == pytest.approx(0.9)
    assert float(summ["A SD"]) == pytest.approx(np.std([1.5, 1.3, 1.2], ddof=1), rel=1e-5)
    assert summ["A n"] == "3"
    _, rows = graph.series_table(s, LABELS, LABELS, True, "sem")
    assert {r[1]: r[2] for r in rows[5:]}["A SEM"] == f"{np.std([1.5, 1.3, 1.2], ddof=1) / np.sqrt(3):.6g}"


@pytest.mark.parametrize("kind", ["bar_points", "points_mean", "bars"])
def test_prism_plot_draws(kind):
    fig = Figure()
    st = graph.style(kind=kind)
    ax = graph.plot_series(fig, {"a": np.array([1, 2, 3, 4, 5.0]), "b": np.array([2, 1, 3, 2, 2.0])},
                           LABELS, LABELS, True, "Relative intensity", st)
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_linewidth() == pytest.approx(st["axis_lw"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Veh", "A"]
    assert ax.get_ylim()[0] == 0
    fig.savefig(os.devnull, format="png")


def test_prefs_round_trip_and_figure_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOT_PROFILER_PREFS", str(tmp_path / "p.json"))
    p = prefs.load()
    assert p == prefs.DEFAULTS
    p["graph"]["axis_lw"] = 2.5
    p["figure"]["frame_lw"] = 1.5
    p["figure"]["fonts"]["lane"] = 10.0
    prefs.save(p)
    q = prefs.load()
    assert q["graph"]["axis_lw"] == 2.5 and q["graph"]["bar_lw"] == prefs.DEFAULTS["graph"]["bar_lw"]
    figure.set_user_defaults(q["figure"])
    try:
        s = figure.default_settings()
        assert s["frame_lw"] == 1.5 and s["fonts"]["lane"] == 10.0
        old = {"frame_lw": 0.3}
        figure.apply_look(old)
        assert old["frame_lw"] == 1.5
    finally:
        figure.set_user_defaults(None)


def test_export_dialog_writes_only_graph_data(tmp_path):
    from PySide6 import QtWidgets

    import synth
    import tifffile
    from blot_profiler import core, gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    p = tmp_path / "exp.tif"
    tifffile.imwrite(str(p), raw)
    w = gui.MainWindow()
    w.ask_marker.setChecked(False)
    w.load_image(str(p))
    for i in range(1, len(w.analysis.rows)):
        w.set_row_role(i, core.ROLE_TARGET)
    names = list(w.results)
    assert len(names) >= 2
    w.series_on[names[1]] = False  # this one is not in the graph
    w.replot()
    dlg = gui.ExportDialog(w)
    dlg.folder.setText(str(tmp_path / "out"))
    dlg.prefix.setText("t")
    dlg.checks["results_csv"].setChecked(True)
    dlg.run()
    out = tmp_path / "out"
    assert sorted(os.listdir(out)) == ["t_graph.png", "t_graph_data.csv", "t_results.csv"]
    rows = list(csv.reader(open(out / "t_graph_data.csv", encoding="utf-8-sig")))
    header = next(r for r in rows if r and r[0] == "레인")
    assert names[0] in header and names[1] not in header
    # the inline formula card adds a graph item
    w.open_formula_card()
    ed = w.formula_editor
    ed.push(core._safe_name(w.analysis.rows[1].name))
    ed.push("/")
    ed.push(core._safe_name(w.analysis.rows[0].name))
    ed.name.setText("my ratio")
    assert ed.valid
    w.accept_formula_card()
    assert "my ratio" in w.results and w.series_on["my ratio"]
    assert not w.formula_card.isVisible()
