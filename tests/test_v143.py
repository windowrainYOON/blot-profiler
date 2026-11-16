"""1.4.3: lists rewritten in place (crash fix), long image names fit the Figure sidebar."""
import synth


def test_lists_are_rewritten_in_place():
    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from blot_profiler import gui
    from blot_profiler.ui import controls

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    lst = controls.SideList()
    lst.set_rows([("a", ""), ("b", "I1"), ("c", "")])
    first = lst.item(0)
    lst.set_rows([("x", ""), ("y", "")])  # shorter: no clear(), the first item stays the same object
    assert lst.count() == 2 and lst.item(0) is first and lst.item(0).text() == "x"
    lst.set_rows([("x", ""), ("y", ""), ("z", "I2")])
    assert [lst.item(i).text() for i in range(3)] == ["x", "y", "z    I2"]
    # check-box rows keep their state by name
    ser = QtWidgets.QListWidget()
    gui.MainWindow._fill_series_list(ser, ["p", "q"])
    ser.item(1).setCheckState(Qt.Unchecked)
    gui.MainWindow._fill_series_list(ser, ["q", "r", "p"])
    assert [(ser.item(i).text(), ser.item(i).checkState() == Qt.Checked) for i in range(3)] == \
        [("q", False), ("r", True), ("p", True)]
    gui.MainWindow._fill_series_list(ser, [])
    assert ser.count() == 0


def test_long_image_names_fit_the_sidebar(tmp_path):
    from PySide6 import QtWidgets
    from PySide6.QtTest import QTest
    import tifffile
    from blot_profiler import gui
    from blot_profiler.figure_window import ElideCheckBox

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    p = str(tmp_path / ("ikras, hela, vinculin, Erk raw 2026.11.08_04.42.19-03_" + "Chemiluminescence" * 3 + ".tif"))
    tifffile.imwrite(p, raw)
    w = gui.MainWindow()
    w.show()
    w.ask_marker.setChecked(False)
    for i in range(2):
        w.load_image(p)
        w.detect(auto=True)
        w.doc.transform["region_tag"] = f"#{i + 1}"
    w.docs[0].analysis.rows[0].name = "HeLa phospho-ERK1/2 (Thr202/Tyr204) with a very long antibody name"
    w.open_figure_window()
    ed = w.figure_editor
    ed.load_from_settings()
    w.fig_win.resize(1300, 900)
    w.fig_win.show()
    QTest.qWait(100)
    boxes = ed.blot_box.findChildren(ElideCheckBox)
    assert boxes
    side = ed.blot_box.parentWidget()
    scroll = side.parentWidget().parentWidget()  # viewport -> scroll area
    assert side.width() <= scroll.viewport().width() + 1  # nothing wider than the sidebar
    for cb in boxes:
        right = cb.mapTo(side, cb.rect().topRight()).x()
        assert right <= side.width()
        assert cb.text() != cb.full_text() and "…" in cb.text()  # cut with …
        assert cb.text().endswith(f"#{1 + boxes.index(cb)}")  # the region tag stays visible
    for b in ed.blot_box.findChildren(QtWidgets.QPushButton):
        assert b.mapTo(side, b.rect().topRight()).x() <= side.width()
