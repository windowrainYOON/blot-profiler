"""1.4.5: blots dragged up/down in the Figure preview (grip ⋮⋮ or the blot itself)."""
import test_v141 as T


def _order(ed):
    ed.render()
    return [p["row"] for p in ed.preview.layout_["panels"]]


def _drag(pv, a, b):
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    QTest.mousePress(pv, QtCore.Qt.LeftButton, pos=a)
    for t in range(1, 6):
        QTest.mouseMove(pv, a + (b - a) * t / 5)
    QTest.mouseRelease(pv, QtCore.Qt.LeftButton, pos=b)
    QTest.qWait(30)


def _y(pv, pn, frac):
    k, ox, oy = pv._k()
    return int(oy + (pn["y"] + pn["h"] * frac) * k)


def test_drag_blot_by_grip_and_body(tmp_path):
    from PySide6 import QtCore

    w, ed = T._window(tmp_path, n=3)
    T._named(w)
    T._boxes(ed)["I1"].click()  # one blot from another image joins at the bottom
    before = _order(ed)
    assert len(before) >= 3
    pv = ed.preview
    ps = pv.layout_["panels"]
    last = ps[-1]
    # grip of the last blot → above the first one
    g = pv.panel_grip_rect_screen(last).center().toPoint()
    _drag(pv, g, QtCore.QPoint(g.x(), _y(pv, ps[0], 0.2)))
    after = _order(ed)
    assert after[0] == last["row"] and sorted(after) == sorted(before)
    assert ed.preview.selected == {"kind": "panel", "row": last["row"]}
    # the blot itself, dragged down below the second one
    ps = pv.layout_["panels"]
    top = ps[0]
    k, ox, oy = pv._k()
    x = int(ox + (top["x_blot"] + pv.layout_["panel_w"] * 0.5) * k)
    _drag(pv, QtCore.QPoint(x, _y(pv, top, 0.5)), QtCore.QPoint(x, _y(pv, ps[1], 0.9)))
    again = _order(ed)
    assert again[1] == top["row"] and again[0] == after[1]
    # a short click only selects
    ps = pv.layout_["panels"]
    _drag(pv, QtCore.QPoint(x, _y(pv, ps[2], 0.5)), QtCore.QPoint(x, _y(pv, ps[2], 0.5) + 2))
    assert _order(ed) == again
    # hidden panels keep their place: hide one, move another, show it again
    hid = ed.s["panels"][1]["row"]
    ed.s["panels"][1]["show"] = False
    ed.load_from_settings()
    shown = _order(ed)
    ed.move_panel_to(shown[-1], 0)
    assert _order(ed)[0] == shown[-1]
    assert [p["row"] for p in ed.s["panels"]].index(hid) >= 1
