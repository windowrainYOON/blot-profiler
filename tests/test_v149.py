"""1.4.9: one group bar dragged to another tier (or a new one), empty group tiers, tiers anywhere
in the stack (between / under the blots), 그림 안의 것 in the figure's order, 내용 in the inspector."""
import test_v141 as T
import test_v145 as D
from blot_profiler import figure


def _tiers(ed, groups):
    """Two group tiers over the lane names: [[(text, first, last)], …]."""
    lane = [t for t in ed.s["tiers"] if t["kind"] == "lanes"][-1]
    ts = []
    for gs in groups:
        t = figure.default_tier("groups", len(ed.s["lanes"]))
        t["groups"] = [{"text": a, "first": b, "last": c} for a, b, c in gs]
        ts.append(t)
    ed.s["tiers"] = ts + [lane]
    ed.load_from_settings()
    ed.render()


def _texts(ed, ti):
    return [g["text"] for g in ed.s["tiers"][ti].get("groups", [])]


def test_stack_sequence_and_positions():
    s = {"tiers": [{"kind": "groups", "pos": "top"}, {"kind": "lanes", "pos": "above:B"}, {"kind": "groups", "pos": "bottom"},
                   {"kind": "groups", "pos": "above:gone"}]}
    nums = [{"pos": "below:A"}, {"pos": "bottom"}]
    seq = figure.stack_sequence(s, ["A", "B"], nums)
    kinds = [(k, v if k != "num" else v[0]) for k, v in seq]
    assert kinds == [("tier", 0), ("tier", 3), ("panel", "A"), ("num", 0), ("tier", 1), ("panel", "B"), ("num", 1), ("tier", 2)]
    order = [("tier", 0), ("panel", "A"), ("num", 0), ("tier", 1), ("num", 1), ("panel", "B"), ("tier", 2)]
    assert figure.stack_positions(order) == ["top", None, "below:A", "above:B", "above:B", None, "bottom"]


def test_bar_drag_between_tiers_and_new_tier(tmp_path):
    from PySide6 import QtCore

    w, ed = T._window(tmp_path, n=1)
    _tiers(ed, [[("siCtrl", 0, 1), ("siKRAS", 2, 3)], [("DMSO", 0, 0), ("Drug", 1, 1), ("DMSO", 2, 2), ("Drug", 3, 3)]])
    pv = ed.preview

    def bar_pt(ti, gi):
        k, ox, oy = pv._k()
        b = next(b for b in pv.layout_["bars"] if b["ti"] == ti and b["gi"] == gi)
        return QtCore.QPoint(int(ox + (b["xa"] + b["xb"]) / 2 * k), int(oy + b["y"] * k))

    def mid(ti):
        k, ox, oy = pv._k()
        t = pv.tier(ti)
        return int(oy + (t["y0"] + t["y1"]) / 2 * k)

    # add an empty group tier from 추가 and drag "Drug" (lane 2) into it; the others stay
    ed.add_from_list("gtier")
    assert ed.s["tiers"][0]["kind"] == "groups" and ed.s["tiers"][0]["groups"] == []
    assert any(t.get("empty") for t in pv.layout_["tiers"])
    a = bar_pt(2, 1)
    D._drag(pv, a, QtCore.QPoint(a.x(), mid(0)))
    assert _texts(ed, 0) == ["Drug"] and _texts(ed, 2) == ["DMSO", "DMSO", "Drug"]
    assert ed.preview.selected == {"kind": "text", "key": "group:0:0"}
    # overlap: siCtrl covers lanes 0–1, Drug (lane 1) cannot go there
    a = bar_pt(0, 0)
    D._drag(pv, a, QtCore.QPoint(a.x(), mid(1)))
    assert _texts(ed, 0) == ["Drug"] and _texts(ed, 1) == ["siCtrl", "siKRAS"]
    # between two tiers: a new tier holding only that bar; the tier it leaves stays (empty)
    n_before = len(ed.s["tiers"])
    a = bar_pt(1, 1)  # siKRAS
    y = (pv.tier(1)["y1"] + pv.tier(2)["y0"]) / 2
    ed.on_bar_moved(1, 1, ("new", pv.stack_drop_index(y)))
    assert len(ed.s["tiers"]) == n_before + 1
    assert _texts(ed, 2) == ["siKRAS"] and _texts(ed, 1) == ["siCtrl"]
    # the menu way: 다른 층으로 옮기기 → back into the empty first tier is refused (Drug sits on lane 1? no: lanes 2–3 free)
    ed.on_bar_moved(2, 0, ("tier", 0))
    assert _texts(ed, 0) == ["Drug", "siKRAS"] and _texts(ed, 2) == []
    # sideways only: the bar shifts lanes in its tier
    ed.render()
    a = bar_pt(1, 0)
    D._drag(pv, a, QtCore.QPoint(int(a.x() + pv.pitch() * pv._k()[0]), a.y()))
    g = ed.s["tiers"][1]["groups"][0]
    assert (g["first"], g["last"]) == (1, 2)


def test_tiers_under_and_between_blots_and_list(tmp_path):
    w, ed = T._window(tmp_path, n=1)
    _tiers(ed, [[("siCtrl", 0, 1), ("siKRAS", 2, 3)]])
    pv = ed.preview
    rows = [p["row"] for p in pv.layout_["panels"]]
    assert len(rows) >= 2
    # the list shows the figure's order: tier, its bars (not draggable), lane names, blots …, blot settings last
    texts = [ed.obj_list.item(i).text() for i in range(ed.obj_list.count())]
    assert "그룹 층 1" in texts[0] and "siCtrl" in texts[1] and "레인 이름" in texts[3]
    from PySide6.QtCore import Qt
    assert not ed.obj_list.item(1).flags() & Qt.ItemIsDragEnabled
    assert ed.obj_list.item(0).flags() & Qt.ItemIsDragEnabled
    assert ed._objects[-1][2] == {"kind": "blot"} or any(o[2] == {"kind": "blot"} for o in ed._objects[-3:])
    # drag the group tier under the last blot (canvas grip): it hangs upside down, bar on top
    n = len(pv.stack_items())
    ed.on_stack_moved("tier", 0, n)
    t = next(t for t in ed.s["tiers"] if t["kind"] == "groups")
    assert t["pos"] == "bottom"
    lt = next(x for x in pv.layout_["tiers"] if x["kind"] == "groups")
    assert lt["down"] and lt["y0"] > max(p["y"] + p["h"] for p in pv.layout_["panels"])
    bar = next(b for b in pv.layout_["bars"])
    txt = next(x for x in pv.layout_["texts"] if x["key"].startswith("group:"))
    assert bar["y"] < txt["y0"] + 0.01  # text under the bar
    # list order follows; drag it in the list (simulated) between the first two blots
    texts = [ed.obj_list.item(i).text() for i in range(ed.obj_list.count())]
    gi = next(i for i, x in enumerate(texts) if "그룹 층" in x)
    pi = [i for i, x in enumerate(texts) if rows[0] in x or ed.s["panels"][0].get("label", "") in x]
    assert gi > pi[0]
    order = ed.stack_order()
    tier = next(v for k, v in order if k == "tier" and v is t)
    order.remove(("tier", tier))
    order.insert(next(i for i, (k, v) in enumerate(order) if k == "panel") + 1, ("tier", tier))
    ed.apply_stack(order)
    assert t["pos"] == f"above:{rows[1]}"
    lt = next(x for x in pv.layout_["tiers"] if x["kind"] == "groups")
    p0, p1 = pv.layout_["panels"][0], pv.layout_["panels"][1]
    assert p0["y"] + p0["h"] <= lt["y0"] and lt["y1"] <= p1["y"] and not lt["down"]
    # export still works
    ed.write_figure(str(tmp_path / "f.png"), 100)
    assert (tmp_path / "f.png").exists()


def test_inspector_content_card(tmp_path):
    from PySide6 import QtWidgets

    w, ed = T._window(tmp_path, n=1)
    _tiers(ed, [[("siCtrl", 0, 1), ("siKRAS", 2, 3)]])
    ed.select_from_list(0)  # the group tier row
    boxes = [b.title() for b in ed.props_layout.parentWidget().findChildren(QtWidgets.QGroupBox)]
    assert any(b.startswith("내용") for b in boxes)
    edits = ed.props_layout.parentWidget().findChildren(QtWidgets.QLineEdit)
    e = next(e for e in edits if e.text() == "siKRAS")
    e.setText("KRAS KD")
    e.editingFinished.emit()
    assert ed.s["tiers"][0]["groups"][1]["text"] == "KRAS KD"
    buttons = [b.text() for b in ed.props_layout.parentWidget().findChildren(QtWidgets.QPushButton)]
    assert "한 층 위" not in buttons and "복제" in buttons
