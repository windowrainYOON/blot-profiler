"""Project save/open round trip through the real main window (offscreen Qt)."""

import io
import json
import os
import shutil
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(__file__))

PySide6 = pytest.importorskip("PySide6")
tifffile = pytest.importorskip("tifffile")

import synth  # noqa: E402
from blot_profiler import core, figure, gui  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets

    a = QtWidgets.QApplication.instance() or gui.App([])
    yield a


def _png(doc, numbers=None):
    from matplotlib.figure import Figure

    fig = Figure()
    figure.render(fig, doc.signal, doc.analysis, doc.figure, doc.marker, numbers)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return buf.getvalue()


def _normalized(d):
    """Project dict without file locations (they change when moved/embedded)."""
    d = json.loads(json.dumps(d, default=str))
    for im in d["images"]:
        for k in ("image_path", "image", "marker"):
            im.pop(k, None)
        if im.get("figure"):
            im["figure"].pop("marker_path", None)
    return d


def _make_session(w, folder):
    raw, _ = synth.chemi_blot()
    blot = os.path.join(folder, "exp1.tif")
    tifffile.imwrite(blot, raw)
    tifffile.imwrite(os.path.join(folder, "exp1-Marker.tif"), synth.marker_photo())
    tifffile.imwrite(os.path.join(folder, "exp2.tif"), raw)
    w.ask_marker.setChecked(False)
    w.load_image(blot)
    assert w.set_marker(w.doc, os.path.join(folder, "exp1-Marker.tif"))
    a = w.analysis
    core.make_row_group(a, [b for b in a.bands if b.row == 0], "Vinculin")
    core.make_row_group(a, [b for b in a.bands if b.row == 2], "Kras G12D")
    core.make_lane_group(a, [b for b in a.bands if b.lane == 1], "DOX")
    w.set_analysis(a)
    w.set_row_role(0, core.ROLE_CONTROL)
    a.formulas.append({"name": "ratio", "expr": "Kras_G12D / Vinculin", "normalize": "1"})
    s = figure.sync_settings(w.doc.figure, a, w.doc.raw.shape)
    tier = figure.default_tier("groups")
    tier["groups"] = [{"text": "IgG", "first": 2, "last": 3}]
    tier["condition"] = "Antibody"
    s["tiers"].insert(0, tier)
    s["overrides"]["protein:Vinculin"] = {"size": 11.0, "dx": 0.05}
    s["strip_x0"], s["strip_x1"], s["strip_dy"] = 60, 130, 4
    s["show_kda"] = True
    s["kda"] = [{"kda": 100, "y": 165.0}, {"kda": 40, "y": 390.0}]
    s["numbers"] = [{"source": "res:Kras G12D/Vinculin", "label": "Kras/Vin", "pos": "bottom", "decimals": 2, "rel_lane": 1}]
    w.load_image(os.path.join(folder, "exp2.tif"))
    w.refresh_multi()
    w.big_groups.append({"name": "Kras_all", "kind": "protein", "members": [[0, "Kras G12D"], [1, "P3"]], "value": "net"})
    w.multi_formulas.append({"name": "x", "expr": "I1.Vinculin / I2.P1", "normalize": "1"})
    w.by_condition.setChecked(True)
    w.switch_doc(0)


def test_project_round_trip_embedded(app, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    w = gui.MainWindow()
    _make_session(w, str(src))
    assert w.is_dirty()
    before = _normalized(w.project_dict())
    png_before = _png(w.docs[0])
    proj = str(tmp_path / "out" / "study.blotproj")
    os.makedirs(os.path.dirname(proj))
    w.write_project(proj, embed=True)
    assert not w.is_dirty()
    shutil.rmtree(src)  # the project must not need the original files any more

    w2 = gui.MainWindow()
    w2.load_project(proj, ask_save=False)
    assert len(w2.docs) == 2
    assert w2.docs[0].marker is not None
    assert _normalized(w2.project_dict()) == before
    assert _png(w2.docs[0]) == png_before
    assert w2.by_condition.isChecked()
    assert not w2.is_dirty()
    # an edit marks the project as changed
    w2.analysis.rows[0].width += 2
    assert w2.is_dirty()


def test_old_json_project_still_opens(app, tmp_path):
    raw, _ = synth.chemi_blot()
    blot = tmp_path / "old.tif"
    tifffile.imwrite(str(blot), raw)
    sig = core.to_signal(raw.astype(float))
    a = core.detect_bands(core.corrected_signal(sig, core.BackgroundSettings()), core.DetectSettings())
    old = {"app": "Blot Profiler", "version": "0.6.0", "format": 3, "detect": {"sensitivity": 1.2},
           "images": [{"image_path": "/somewhere/else/old.tif", "invert": 0, "background": {"method": "none"},
                       "reference_lane": 0, "analysis": a.to_dict(), "figure": None}],
           "big_groups": [], "multi_formulas": [], "multi_reference": None}
    p = tmp_path / "old_analysis.json"
    p.write_text(json.dumps(old), encoding="utf-8")
    w = gui.MainWindow()
    w.load_project(str(p), ask_save=False)  # the image is found next to the project file
    assert w.doc.image_path == str(blot)
    assert len(w.analysis.bands) == len(a.bands)
    assert w.sens.value() == pytest.approx(1.2)


def test_two_projects_together(app, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    w = gui.MainWindow()
    _make_session(w, str(src))
    p1 = str(tmp_path / "a.blotproj")
    w.write_project(p1, embed=True)
    w.write_project(str(tmp_path / "b.blotproj"), embed=True)
    w2 = gui.MainWindow()
    w2.open_projects([p1, str(tmp_path / "b.blotproj")], ask_save=False)
    assert len(w2.docs) == 4
    assert [d.project for d in w2.docs] == ["a", "a", "b", "b"]
    # the second project's formula now points at its own images (I3, I4) and has its own name
    names = [f["name"] for f in w2.multi_formulas]
    assert len(names) == 2 and len(set(names)) == 2
    assert w2.multi_formulas[1]["expr"] == "I3.Vinculin / I4.P1"
    assert [m[0] for m in w2.big_groups[1]["members"]] == [2, 3]
    assert w2.big_groups[1]["name"] != w2.big_groups[0]["name"]
    assert w2.project_path is None
    # a formula across the two projects
    w2.multi_formulas.append({"name": "ab", "expr": "I1.Vinculin / I3.Vinculin", "normalize": "1"})
    w2.recompute_multi()
    assert "ab" in w2.multi_results
