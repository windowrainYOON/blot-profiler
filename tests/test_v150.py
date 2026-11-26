"""1.5.0: lane names loaded from another blot, linked lanes lined up in Figure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from blot_profiler import core, figure, lanes  # noqa: E402


def _two():
    raw, _ = synth.chemi_blot()
    sig = core.to_signal(raw.astype(float), invert=core.needs_inversion(raw))
    corr = core.corrected_signal(sig, core.BackgroundSettings())
    a = lanes.detect(corr)
    corr2 = np.ascontiguousarray(corr[:, 330:])  # lanes 2-8 of the first
    b = lanes.detect(corr2)
    return corr, a, corr2, b


def test_lane_pairs_directions_and_shared_parent():
    _, a, _, b = _two()
    assert a.n_lanes == 8 and b.n_lanes == 7
    assert figure.lane_pairs(a, "A", b, "B") == []
    b.lane_link = {"doc": "A", "start": 1}
    assert figure.lane_pairs(a, "A", b, "B") == [(i, i - 1) for i in range(1, 8)]
    assert figure.lane_pairs(b, "B", a, "A") == [(j, j + 1) for j in range(7)]
    c = core.Analysis.from_dict(b.to_dict())  # saved and loaded
    assert c.lane_link == {"doc": "A", "start": 1}
    a.lane_link = {"doc": "Z", "start": 0}
    c.lane_link = {"doc": "Z", "start": 1}  # both linked to a third blot
    assert figure.lane_pairs(a, "A", c, "C")[0] == (1, 0)


def test_linked_blot_sits_under_its_lanes():
    _, a, _, b = _two()
    own = figure.lane_centres(a)
    src = figure.lane_centres(b)
    f, off, *_ = figure.foreign_place(b, own)  # not linked: centred, half a lane off
    assert abs(f * src[0] + off - own[1]) > 30
    b.lane_link = {"doc": "A", "start": 1}
    f, off, *_ = figure.foreign_place(b, own, figure.lane_pairs(a, "A", b, "B"))
    for j in range(7):
        assert abs(f * src[j] + off - own[j + 1]) < 5


def test_number_rows_follow_linked_lanes():
    _, a, _, b = _two()
    b.lane_link = {"doc": "A", "start": 1}
    lm = figure.panel_lane_map(a, {"lane_match": "order"}, {}, b, figure.lane_pairs(a, "A", b, "B"))
    assert lm == [None, 0, 1, 2, 3, 4, 5, 6]


def test_load_lanes_from_other_blot_in_the_app(tmp_path):
    from PySide6 import QtWidgets
    import tifffile
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    tifffile.imwrite(str(tmp_path / "a.tif"), raw)
    tifffile.imwrite(str(tmp_path / "b.tif"), np.ascontiguousarray(raw[:, 330:]))
    w = gui.MainWindow()
    w.ask_marker.setChecked(False)
    w.load_image(str(tmp_path / "a.tif"))
    w.detect(auto=True)
    names = ["M", "DMSO", "DMSO", "1h", "6h", "24h", "48h", "72h"]
    for i, n in enumerate(names):
        w.rename_lane(i, n)
    w.set_lane_role(0, core.LANE_MARKER)
    w.copy_lanes()
    w.load_image(str(tmp_path / "b.tif"))
    w.detect(auto=True)
    src = w.docs[0]
    dlg = gui.LaneLoadDialog(w, [src], src)
    dlg.start.setCurrentIndex(dlg.start.findData(1))
    assert dlg.table.item(0, 2).text() == "DMSO" and dlg.table.item(6, 2).text() == "72h"
    d, start, tn, tr, link = dlg.result_values()
    assert (start, tn, tr, link) == (1, True, True, True)
    w.apply_lane_info(names, [core.LANE_MARKER] + [core.LANE_SAMPLE] * 7, start)
    w.analysis.lane_link = {"doc": src.uid, "start": start, "name": src.name}
    w.refresh_lane_link()
    assert w.analysis.lane_labels[:7] == names[1:]
    assert core.LANE_MARKER not in w.analysis.lane_roles
    assert "I1" in w.lane_link_label.text() and "L2–L8" in w.lane_link_label.text()
    w.unlink_lanes()
    assert w.analysis.lane_link is None and not w.lane_link_chip.isVisibleTo(w.lane_link_chip.parentWidget())


def test_paste_excel_column():
    from blot_profiler import gui

    assert gui.MainWindow.parse_lane_text("DMSO\nDrug\t마커\n\n") == [("DMSO", None), ("Drug", core.LANE_MARKER)]
    assert [n for n, _ in gui.MainWindow.parse_lane_text("A\tB\tC")] == ["A", "B", "C"]  # one Excel row
