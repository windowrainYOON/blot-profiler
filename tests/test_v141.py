"""1.4.1: 그림에 넣을 블롯 — several images of one file (blot regions #1, #2 …) in one Figure."""
import json
import os
import zipfile

import synth


def _window(tmp_path, n=4):
    from PySide6 import QtWidgets
    import tifffile
    from blot_profiler import gui

    app = QtWidgets.QApplication.instance() or gui.App([])  # noqa: F841
    raw, _ = synth.chemi_blot()
    path = str(tmp_path / "blot.tif")
    tifffile.imwrite(path, raw)
    w = gui.MainWindow()
    w.resize(1400, 900)
    w.show()
    w.ask_marker.setChecked(False)
    for i in range(n):  # the same file in every tab, as blot regions of one picture are
        w.load_image(path)
        w.detect(auto=True)
        w.doc.transform["region_tag"] = f"#{i + 1}"
    w.refresh_doc_tabs()
    w.switch_doc(n - 1)
    w.open_figure_window()
    ed = w.figure_editor
    ed.s["layout"] = "rows"
    ed.load_from_settings()
    return w, ed


def _boxes(ed):
    """{image key (I1 …): check box} of the 그림에 넣을 블롯 list as it is shown now."""
    from PySide6 import QtCore, QtWidgets

    QtCore.QCoreApplication.processEvents()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    out = {}
    for i in range(ed.blot_lay.count()):
        wd = ed.blot_lay.itemAt(i).widget()
        if wd is None:
            continue
        for cb in wd.findChildren(QtWidgets.QCheckBox):
            out[cb.text().split(" · ")[0]] = cb
    return out


def _in_figure(ed):
    ed.render()
    return {p["doc"] for p in ed.preview.layout_["panels"] if p.get("doc")}


def test_each_region_is_its_own_image(tmp_path):
    w, ed = _window(tmp_path)
    docs = w.loaded_docs()
    assert len({ed.doc_id(d) for d in docs}) == len(docs)  # one file, four ids
    box = _boxes(ed)
    assert set(box) == {"I1", "I2", "I3", "I4"}
    assert box["I4"].isChecked() and not box["I4"].isEnabled()  # the figure's own image
    # tick I1, I2, I3 one after the other: each joins, none of the others changes
    for n, key in enumerate(("I1", "I2", "I3"), 1):
        _boxes(ed)[key].click()
        box = _boxes(ed)
        assert [box[k].isChecked() for k in ("I1", "I2", "I3")] == [i < n for i in range(3)]
        assert _in_figure(ed) == {ed.doc_id(d) for d in docs[:n]}
    # untick I2: only I2 leaves the figure
    _boxes(ed)["I2"].click()
    box = _boxes(ed)
    assert [box[k].isChecked() for k in ("I1", "I2", "I3")] == [True, False, True]
    assert _in_figure(ed) == {ed.doc_id(docs[0]), ed.doc_id(docs[2])}
    # tick it again, then take everything out
    _boxes(ed)["I2"].click()
    assert _in_figure(ed) == {ed.doc_id(d) for d in docs[:3]}
    for key in ("I3", "I1", "I2"):
        _boxes(ed)[key].click()
    assert not _in_figure(ed)
    assert not any(b.isChecked() for k, b in _boxes(ed).items() if k != "I4")


def test_protein_chips_follow_their_image(tmp_path):
    from PySide6 import QtCore, QtWidgets

    w, ed = _window(tmp_path, n=3)
    docs = w.loaded_docs()
    _boxes(ed)["I1"].click()
    _boxes(ed)["I2"].click()
    names = [r.name for r in docs[0].analysis.rows]
    assert len(names) >= 2

    def chips():
        _boxes(ed)
        out, key = {}, None
        for i in range(ed.blot_lay.count()):
            wd = ed.blot_lay.itemAt(i).widget()
            cbs = wd.findChildren(QtWidgets.QCheckBox)
            if cbs:
                key = cbs[0].text().split(" · ")[0]
                continue
            out[key] = {b.text(): b.isChecked() for b in wd.findChildren(QtWidgets.QPushButton)}
        return out

    before = chips()
    assert sum(before["I1"].values()) == 1 and sum(before["I2"].values()) == 1
    # a second protein of I1 only
    extra = next(n for n, on in before["I1"].items() if not on)
    btn = next(b for b in ed.blot_lay.parentWidget().findChildren(QtWidgets.QPushButton)
               if b.text() == extra and b.isVisible() and _owner(ed, b) == "I1")
    btn.click()
    QtCore.QCoreApplication.processEvents()
    after = chips()
    assert after["I1"][extra] and sum(after["I1"].values()) == 2
    assert after["I2"] == before["I2"]  # I2 untouched
    src = [p.get("src_row") for p in ed.s["panels"] if p.get("doc") == ed.doc_id(docs[0])]
    assert sorted(src) == sorted(n for n, on in after["I1"].items() if on)


def _owner(ed, button):
    key = None
    for i in range(ed.blot_lay.count()):
        wd = ed.blot_lay.itemAt(i).widget()
        from PySide6 import QtWidgets

        cbs = wd.findChildren(QtWidgets.QCheckBox)
        if cbs:
            key = cbs[0].text().split(" · ")[0]
        elif button in wd.findChildren(QtWidgets.QPushButton):
            return key
    return None


def test_image_without_bands_stays_listed(tmp_path):
    w, ed = _window(tmp_path, n=3)
    w.docs[0].analysis.bands = []
    ed.refresh_blots()
    box = _boxes(ed)
    assert set(box) == {"I1", "I2", "I3"}
    assert not box["I1"].isEnabled() and not box["I1"].isChecked()
    assert box["I2"].isEnabled()


def test_project_keeps_which_image(tmp_path):
    from blot_profiler import gui

    w, ed = _window(tmp_path, n=3)
    docs = w.loaded_docs()
    _boxes(ed)["I2"].click()
    want = ed.doc_id(docs[1])
    proj = str(tmp_path / "p.blotproj")
    w.write_project(proj, embed=True)
    w2 = gui.MainWindow()
    w2.load_project(proj, ask_save=False)
    d2 = w2.loaded_docs()
    assert [d.uid for d in d2] == [d.uid for d in docs]
    refs = [p["doc"] for p in d2[2].figure["panels"] if p.get("doc")]
    assert refs == [want]

    # a project saved by 1.4.0 named the file instead: it is moved onto the image itself
    old = str(tmp_path / "old.blotproj")
    with zipfile.ZipFile(proj) as z:
        items = {n: z.read(n) for n in z.namelist()}
    jname = next(n for n in items if n.endswith(".json"))
    data = json.loads(items[jname])
    for im in data["images"]:
        im.pop("uid", None)
        for p in (im.get("figure") or {}).get("panels", []):
            if p.get("doc"):
                p["doc"] = os.path.abspath(im["image_path"])
    items[jname] = json.dumps(data).encode()
    with zipfile.ZipFile(old, "w") as z:
        for n, b in items.items():
            z.writestr(n, b)
    w3 = gui.MainWindow()
    w3.load_project(old, ask_save=False)
    d3 = w3.loaded_docs()
    assert len({d.uid for d in d3}) == 3
    refs = [p["doc"] for p in d3[2].figure["panels"] if p.get("doc")]
    assert len(refs) == 1 and refs[0] in {d.uid for d in d3}

    # the same project added twice: the second copy gets new ids and its panels follow them
    w2.load_project(proj, ask_save=False, append=True)
    d4 = w2.loaded_docs()
    assert len(d4) == 6 and len({d.uid for d in d4}) == 6
    refs = [p["doc"] for p in d4[5].figure["panels"] if p.get("doc")]
    assert refs == [d4[4].uid]
