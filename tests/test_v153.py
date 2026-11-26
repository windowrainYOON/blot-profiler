"""1.5.3: the Figure inspector has one fixed width that fits its pages; inside, long check
boxes wrap, long choices are cut with … and nothing is wider than the box (also in English)."""
import test_v141 as T
from blot_profiler import i18n
from blot_profiler.ui import controls, theme


def _overflow(scroll):
    """How much wider the page needs to be than the room the fixed box gives it (box margins
    12 + 4, scroll bar); measured from the sizes, so it holds before the window is painted."""
    from PySide6 import QtWidgets

    bar = scroll.style().pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
    room = theme.FIGURE_INSPECTOR_W - 16 - bar
    return scroll.widget().minimumSizeHint().width() - room


def _widest(page):
    """The widest pieces of a page (shown when it does not fit)."""
    from PySide6 import QtWidgets

    rows = []
    for c in page.findChildren(QtWidgets.QWidget):
        if c.isVisibleTo(page):
            t = c.text() if hasattr(c, "text") and callable(c.text) else (c.title() if hasattr(c, "title") else "")
            rows.append((c.minimumSizeHint().width(), type(c).__name__, str(t)[:30], c.font().family(), c.font().pointSizeF()))
    return sorted(rows, reverse=True)[:12]


def test_inspector_fixed_width_and_nothing_cut(tmp_path):
    from PySide6.QtTest import QTest

    old = i18n.LANG
    i18n.set_language("en")  # the longer texts
    try:
        w, ed = T._window(tmp_path, n=1)
        w.fig_win.resize(1280, 800)
        insp = ed.split.widget(2)
        assert insp.width() == theme.FIGURE_INSPECTOR_W
        ed.props_tabs.setCurrentIndex(1)
        QTest.qWait(30)
        assert _overflow(ed.props_tabs.widget(1)) <= 0, _widest(ed.props_tabs.widget(1).widget())  # was 382 px in a 330 px box
        ed.preview.select({"kind": "blot"})
        ed.on_object_selected({"kind": "blot"})
        QTest.qWait(30)
        assert _overflow(ed.props_tabs.widget(0)) <= 0
        assert isinstance(ed.w_minv, controls.WrapCheckBox)
        assert ed.panel_table.horizontalHeaderItem(2).toolTip() == "Top margin"
    finally:
        i18n.set_language(old)


def test_wrap_check_box_takes_more_lines_when_narrow():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    c = controls.WrapCheckBox("Invert marker strip (dark ladder / white background) and more words")
    assert c.minimumSizeHint().width() < c.sizeHint().width()
    assert c.heightForWidth(140) > c.heightForWidth(800)
    lab = controls.ElideLabel("● a long message " * 10)
    assert lab.minimumSizeHint().width() == 0 and lab.text().startswith("● a long")
