import os

import numpy as np
import pytest

from blot_profiler import core

DATA = os.path.join(os.path.dirname(__file__), "data")


def synthetic_blot(lanes=6, intens=(1.0, 0.5, 2.0, 1.0, 0.0, 1.5), ctrl=1.0, noise=2.0, seed=0):
    """Dark bands on a light background, two rows (control on top, target below)."""
    rng = np.random.default_rng(seed)
    H, W = 160, 40 + lanes * 50
    img = np.full((H, W), 50.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(lanes):
        cx = 45 + i * 50
        img += 150 * ctrl * np.exp(-(((xx - cx) / 14) ** 6) - ((yy - 40) / 4) ** 2)
        img += 150 * intens[i] * np.exp(-(((xx - cx) / 14) ** 6) - ((yy - 110) / 5) ** 2)
    img += rng.normal(0, noise, img.shape)
    return np.clip(65535 - img * 200, 0, 65535)  # 16-bit, dark bands


def test_detects_rows_and_lanes():
    img = synthetic_blot()
    sig = core.to_signal(img)
    a = core.detect_bands(core.corrected_signal(sig, core.BackgroundSettings()), core.DetectSettings())
    assert len(a.rows) == 2
    assert a.n_lanes == 6
    # empty lane 5 of the target row still gets a box
    assert sum(1 for b in a.bands if b.row == 1) == 6
    # same box size within a row
    for ri in range(2):
        boxes = {b.box(a.rows)[2:] for b in a.bands if b.row == ri}
        assert len(boxes) == 1


@pytest.mark.parametrize("method", ["rolling_ball", "local_border", "row_median"])
def test_normalization_recovers_ratios(method):
    intens = (1.0, 0.5, 2.0, 1.0, 0.0, 1.5)
    img = synthetic_blot(intens=intens)
    sig = core.to_signal(img)
    bg = core.BackgroundSettings(method=method)
    corr = core.corrected_signal(sig, bg)
    a = core.detect_bands(corr, core.DetectSettings())
    a.rows[0].role = core.ROLE_CONTROL
    m = core.measure(sig, a, bg, corr)
    norm = core.normalize_to_control(a, m, 1, 0, reference_lane=0)
    np.testing.assert_allclose(norm, np.array(intens) / intens[0], atol=0.12)


def test_formulas_and_band_ids():
    a = core.Analysis(rows=[core.Row("Vinculin", 10, 10, core.ROLE_CONTROL), core.Row("p Erk", 10, 10)])
    a.bands = [core.Band(r, l, 10 * l, 10 * r) for r in range(2) for l in range(3)]
    a.lane_labels = ["DMSO (Off)", "IgG 12h", "IgG 24h"]
    ids = [a.band_id(b) for b in a.bands]
    assert ids[0] == "Vinculin_DMSO_Off" and ids[5] == "p_Erk_IgG_24h"
    meas = [core.Measurement(a.band_id(b), b.row, b.lane, 0, 0, float(10 * (b.row + 1) + b.lane), 0, 1) for b in a.bands]
    v = core.formula_variables(a, meas)
    np.testing.assert_allclose(core.evaluate("p_Erk / Vinculin", v), [2.0, 21 / 11, 22 / 12])
    assert core.evaluate("p_Erk_IgG_24h - Vinculin_DMSO_Off * 2", v) == 2.0
    assert np.isnan(core.evaluate("Vinculin_IgG_12h / 0", v))
    with pytest.raises(core.FormulaError):
        core.evaluate("__import__('os')", v)
    with pytest.raises(core.FormulaError):
        core.evaluate("nope + 1", v)


def grid(rows=3, lanes=4):
    a = core.Analysis(rows=[core.Row(f"R{r + 1}", 10, 6) for r in range(rows)])
    a.bands = [core.Band(r, l, 20 + 30 * l, 20 + 20 * r) for r in range(rows) for l in range(lanes)]
    a.ensure_lane_labels()
    return a


def test_make_row_group_names_and_merges():
    a = grid()
    sel = [b for b in a.bands if b.row == 0]
    assert core.make_row_group(a, sel, "Vinculin") == "Vinculin"
    assert [r.name for r in a.rows] == ["Vinculin", "R2", "R3"]
    # doublet: rows 2 and 3 become one protein group; bands in one lane are summed
    sel = [b for b in a.bands if b.row in (1, 2)]
    assert core.make_row_group(a, sel, "Erk") == "Erk"
    assert [r.name for r in a.rows] == ["Vinculin", "Erk"]
    assert a.band_id(a.bands[4]).startswith("Erk_L1_")
    meas = core.measure(np.ones((100, 200)), a, core.BackgroundSettings(method="none"))
    np.testing.assert_allclose(core.row_vector(a, meas, 1), [120.0] * 4)
    # duplicate name gets a suffix
    sel = [b for b in a.bands if b.row == 1 and b.cy > 50]
    assert core.make_row_group(a, sel, "Vinculin") == "Vinculin_2"


def test_make_lane_group_single_and_replicates():
    a = grid(rows=2, lanes=4)
    col = [b for b in a.bands if b.lane == 1]
    assert core.make_lane_group(a, col, "IgG 12h") == "IgG 12h"
    assert a.lane_labels == ["L1", "IgG 12h", "L3", "L4"]
    reps = [b for b in a.bands if b.lane in (2, 3)]
    assert core.make_lane_group(a, reps, "DOX") == "DOX"
    assert a.lane_labels == ["L1", "IgG 12h", "DOX", "DOX"]
    assert core.make_lane_group(a, [b for b in a.bands if b.lane == 0], "DOX") == "DOX_2"
    # lane group "DOX_2" owns R1_DOX_2, so the DOX replicates are numbered around it
    assert sorted(a.band_id(b) for b in a.bands if b.row == 0) == ["R1_DOX_1", "R1_DOX_2", "R1_DOX_3", "R1_IgG_12h"]
    # a column assembled from bands of two detected lanes becomes one lane
    b = grid(rows=2, lanes=3)
    mixed = [b.bands[0], b.bands[4]]  # row 0 lane 0, row 1 lane 1
    core.make_lane_group(b, mixed, "X")
    assert b.bands[0].lane == b.bands[4].lane and b.lane_labels[b.bands[0].lane] == "X"


def test_common_lanes_and_combine():
    common = core.common_lane_keys([["A", "B", "B"], ["B", "C"]])
    assert common == ["A", "B", "C"]
    np.testing.assert_allclose(core.to_common([1.0, 2.0, 4.0], ["A", "B", "B"], common), [1.0, 3.0, np.nan])
    mean, sd, n, _ = core.combine_members([np.array([1.0, 3.0, np.nan]), np.array([np.nan, 2.0, 4.0])], ref=1)
    np.testing.assert_allclose(mean, [1 / 3, 1.0, 2.0])
    assert list(n) == [1, 2, 1]


def test_condition_stats():
    names, mean, sd, n = core.condition_stats(np.array([1.0, 2.0, 4.0]), ["a", "b", "b"], ["L1", "L2", "L3"])
    assert names == ["a", "b"]
    np.testing.assert_allclose(mean, [1.0, 3.0])
    assert n == [1, 2]


def test_example_tif_loads_and_detects():
    img = core.load_image(os.path.join(DATA, "example_blot.tif"))
    assert core.needs_inversion(img)
    sig = core.to_signal(img)
    a = core.detect_bands(core.corrected_signal(sig, core.BackgroundSettings()), core.DetectSettings())
    assert len(a.rows) >= 2
    assert a.n_lanes >= 6


def test_add_band_and_roundtrip():
    a = core.Analysis(rows=[core.Row("A", 20, 10)], bands=[core.Band(0, 0, 20, 20), core.Band(0, 1, 60, 20)])
    a.ensure_lane_labels()
    core.add_band(a, 100, 21)
    assert a.n_lanes == 3
    core.add_band(a, 61, 80)  # far below -> new row
    assert len(a.rows) == 2
    b = core.Analysis.from_dict(a.to_dict())
    assert len(b.bands) == len(a.bands) and b.rows[1].width == 20


def test_cross_image_formulas():
    v1 = {"Vinculin": np.array([1.0, 2.0]), "Erk": np.array([4.0, 6.0])}
    v2 = {"Vinculin": np.array([2.0, 2.0]), "Kras": np.array([3.0, 5.0])}
    v = core.project_variables([v1, v2], ["", "blot B"], base=0)
    np.testing.assert_allclose(core.evaluate("I2.Kras / Vinculin", v), [3.0, 2.5])
    np.testing.assert_allclose(core.evaluate("blot_B.Kras / I1.Erk", v), [0.75, 5 / 6])
    with pytest.raises(core.FormulaError):
        core.evaluate("I2 / Vinculin", v)
    with pytest.raises(core.FormulaError):
        core.evaluate("I3.Kras", v)
    with pytest.raises(core.FormulaError):
        core.evaluate("I1.Vinculin + np.array([1,2,3])", v)
    v3 = core.project_variables([v1, {"X": np.array([1.0, 2.0, 3.0])}], base=0)
    with pytest.raises(core.FormulaError):
        core.evaluate("Vinculin + I2.X", v3)


def test_auto_assign_lanes_keeps_user_names():
    a = grid(rows=2, lanes=4)
    core.make_lane_group(a, [b for b in a.bands if b.lane == 1], "DOX")
    # drag the row-1 box of lane 3 next to lane 2 and the row-0 box of lane 0 far right
    b03 = next(b for b in a.bands if b.row == 1 and b.lane == 3)
    b03.cx = 81  # lane 2 sits at x=80
    b00 = next(b for b in a.bands if b.row == 0 and b.lane == 0)
    b00.cx = 160
    assert core.auto_assign_lanes(a)
    assert a.lane_labels == ["L1", "DOX", "L3", "L4", "L5"]
    assert b03.lane == 2 and b00.lane == 4
    assert not core.auto_assign_lanes(a)  # stable


def test_measurement_uses_raw_intensities():
    """Net intensity = sum of (linear) inverted raw pixels in the box; neither the
    detection LUT nor any display contrast enters the measurement."""
    rng = np.random.default_rng(1)
    raw = rng.integers(20000, 60000, size=(60, 80)).astype(np.float64)  # 16-bit, dark-band polarity
    sig = core.to_signal(raw, invert=True)
    np.testing.assert_array_equal(sig, 65535.0 - raw)
    a = core.Analysis(rows=[core.Row("P", 10, 6)], bands=[core.Band(0, 0, 20, 30), core.Band(0, 1, 50, 30)])
    m = core.measure(sig, a, core.BackgroundSettings(method="none"))
    for meas, b in zip(m, a.bands):
        x, y, w, h = b.box(a.rows)
        assert meas.net == pytest.approx(float((65535.0 - raw[y:y + h, x:x + w]).sum()))
    # detection LUT is a separate copy and leaves the signal untouched
    before = sig.copy()
    core.detect_bands(sig, core.DetectSettings(lut_gamma=0.5))
    np.testing.assert_array_equal(sig, before)
    # non-inverted (fluorescence) images are measured as is
    m2 = core.measure(core.to_signal(raw, invert=False), a, core.BackgroundSettings(method="none"))
    x, y, w, h = a.bands[0].box(a.rows)
    assert m2[0].net == pytest.approx(float(raw[y:y + h, x:x + w].sum()))


def test_figure_renders_chemi_blot():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from blot_profiler import figure
    import synth
    raw, _ = synth.chemi_blot()
    sig = core.to_signal(raw.astype(float))
    assert not core.needs_inversion(raw.astype(float))  # bright bands on black
    a = core.detect_bands(core.corrected_signal(sig, core.BackgroundSettings()), core.DetectSettings())
    assert len(a.rows) == 3 and a.n_lanes == 8
    s = figure.sync_settings(figure.default_settings(), a, raw.shape)
    s["groups"] = [{"text": "IgG", "first": 2, "last": 3}]
    s["show_strip"], s["show_kda"] = True, True
    s["kda"] = [{"kda": k, "y": y} for k, y in synth.LADDER_Y.items()]
    fig = Figure()
    figure.render(fig, sig, a, s, synth.marker_photo().astype(float))
    w, h = fig.get_size_inches()
    assert w > 3 and h > 1 and len(fig.axes) == 6  # 3 panels + 3 ladder strips


def test_figure_numbers_and_layout():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from blot_profiler import figure
    import synth
    assert figure.format_numbers([2.0, 4.0, np.nan], 2, rel_lane=0) == (["1.00", "2.00", "–"], "")
    assert figure.format_numbers([6.6e6, 1.25e7], 2) == (["0.66", "1.25"], " (×10⁷)")
    assert figure.format_numbers([123.4], 0) == (["123"], "")
    raw, _ = synth.chemi_blot()
    sig = core.to_signal(raw.astype(float))
    a = core.detect_bands(core.corrected_signal(sig, core.BackgroundSettings()), core.DetectSettings())
    s = figure.sync_settings(figure.default_settings(), a, raw.shape)
    nums = [{"label": "Kras/Vin", "pos": "bottom", "texts": ["1.00"] * 8},
            {"label": "x", "pos": f"below:{a.rows[0].name}", "texts": ["1"] * 8}]
    fig = Figure()
    lay0 = figure.render(fig, sig, a, s)
    h0 = fig.get_size_inches()[1]
    lay = figure.render(fig, sig, a, s, numbers=nums)
    assert fig.get_size_inches()[1] > h0 + 0.2  # two number rows add height
    assert [p["row"] for p in lay["panels"]] == [r.name for r in a.rows]
    # widening the crop on the left keeps the panel width (inches) but lowers the scale
    s["margin_left_px"] = 200
    lay2 = figure.render(fig, sig, a, s)
    assert lay2["x0px"] < lay0["x0px"] and lay2["scale"] < lay0["scale"]
    # a taller crop of one panel
    s["panels"][0]["pad_bottom"] += 40
    lay3 = figure.render(fig, sig, a, s)
    assert lay3["panels"][0]["y1"] == lay2["panels"][0]["y1"] + 40


def _figure_setup():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import matplotlib
    matplotlib.use("Agg")
    from blot_profiler import figure
    import synth
    raw, _ = synth.chemi_blot()
    sig = core.to_signal(raw.astype(float))
    a = core.detect_bands(core.corrected_signal(sig, core.BackgroundSettings()), core.DetectSettings())
    return figure, raw, sig, a


def test_figure_old_settings_migrate_to_tiers():
    figure, raw, sig, a = _figure_setup()
    old = figure.default_settings()
    old.update(font_size=10.0, lane_rotation=45, groups=[{"text": "IgG", "first": 2, "last": 3}])
    old.pop("tiers", None)
    s = figure.sync_settings(old, a, raw.shape)
    assert [t["kind"] for t in s["tiers"]] == ["groups", "lanes"]
    assert s["tiers"][0]["groups"][0]["text"] == "IgG" and s["tiers"][1]["auto"] and s["tiers"][1]["rotation"] == 45
    assert s["fonts"]["lane"] == 10.0 and s["fonts"]["kda"] == 8.5


def test_figure_tiers_conditions_overrides_and_sides():
    from matplotlib.figure import Figure
    figure, raw, sig, a = _figure_setup()
    s = figure.sync_settings(figure.default_settings(), a, raw.shape)
    n = len(s["lanes"])
    g1 = figure.default_tier("groups")
    g1["groups"] = [{"text": "Brain", "first": 0, "last": 3}, {"text": "Microglia", "first": 4, "last": n - 1}]
    g2 = figure.default_tier("groups")
    g2.update(groups=[{"text": "Con", "first": 0, "last": 1}], rotation=60)
    lanes = figure.default_tier("lanes", n)
    lanes.update(texts=["+", "-"] * (n // 2), condition="Dox", cond_side="right", italic=True)
    s["tiers"] = [g1, g2, lanes] + s["tiers"]
    s["protein_side"] = "right"
    s["kda_side"] = "right"
    s["show_kda"] = True
    s["kda"] = [{"kda": 40, "y": 370}]
    fig = Figure()
    lay = figure.render(fig, sig, a, s)
    keys = {t["key"]: t for t in lay["texts"]}
    assert {"group:0:0", "group:0:1", "group:1:0", "lane:2:0", "cond:2", "kda:0"} <= set(keys)
    blot_right = lay["left"] + lay["panel_w"]
    # condition "Dox" and protein labels on the right of the blot, kDa label too
    assert keys["cond:2"]["x0"] > blot_right
    assert all(t["x0"] > blot_right for k, t in keys.items() if k.startswith("protein:"))
    assert keys["kda:0"]["x0"] > blot_right
    # tiers stack top to bottom
    assert keys["group:0:0"]["y1"] <= keys["group:1:0"]["y1"] <= keys["lane:2:0"]["y1"]
    # per-text override: bigger font and moved
    before = keys["group:0:0"]
    s["overrides"]["group:0:0"] = {"size": 16.0, "dx": 0.3}
    lay2 = figure.render(fig, sig, a, s)
    after = {t["key"]: t for t in lay2["texts"]}["group:0:0"]
    assert (after["x1"] - after["x0"]) > 1.5 * (before["x1"] - before["x0"])
    # category font size
    s["fonts"]["protein"] = 14.0
    lay3 = figure.render(fig, sig, a, s)
    p3 = [t for t in lay3["texts"] if t["key"].startswith("protein:")][0]
    p2 = [t for t in lay2["texts"] if t["key"].startswith("protein:")][0]
    assert (p3["y1"] - p3["y0"]) > (p2["y1"] - p2["y0"])


def test_transform_and_line_straightening():
    img = np.zeros((40, 60))
    img[5, 10] = 1.0
    r = core.apply_transform(img, {"rot90": 1, "angle": 0.0, "crop": None})
    assert r.shape == (60, 40) and r[49, 5] == 1.0  # counter-clockwise quarter turn
    c = core.apply_transform(img, {"rot90": 0, "angle": 0.0, "crop": [5, 2, 30, 20]})
    assert c.shape == (18, 25) and c[3, 5] == 1.0
    # a line that rises 3 degrees to the right becomes horizontal after rotating by its angle
    from scipy import ndimage as ndi
    line = np.zeros((200, 300))
    line[100, 50:250] = 1.0
    tilted = ndi.rotate(line, 3.0, reshape=False, order=1)
    ys, xs = np.nonzero(tilted > 0.3)
    i0, i1 = np.argmin(xs), np.argmax(xs)
    ang = core.line_angle(xs[i0], ys[i0], xs[i1], ys[i1])
    fixed = core.apply_transform(tilted, {"rot90": 0, "angle": ang, "crop": None})
    ys2, xs2 = np.nonzero(fixed > 0.3)
    slope = np.polyfit(xs2, ys2, 1)[0]
    assert abs(np.degrees(np.arctan(slope))) < 0.5


def test_gel_mode_detects_ladders_and_samples():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import synth
    rgb, lanes, ladder_y = synth.gel_photo()
    raw = rgb[..., 0].astype(float)
    sig = core.to_signal(raw)
    corr = core.corrected_signal(sig, core.BackgroundSettings())
    assert core.looks_like_gel(corr)
    a = core.detect_gel(corr)
    assert a.n_lanes == 8 and core.ladder_lanes(a) == [0, 4]
    assert [round(y) for y in core.ladder_band_ys(a)] == [int(y) for y in ladder_y]
    assert sorted(r.name for r in a.rows if r.name != a.ladder_row) == ["B1", "B2", "B3"]
    # a western blot is not mistaken for a gel
    w = core.to_signal(core.load_image(os.path.join(DATA, "example_blot.tif")))
    assert not core.looks_like_gel(core.corrected_signal(w, core.BackgroundSettings()))


def test_marker_orientation_candidates():
    blot = (1024, 1408)
    # same aspect: unchanged first
    c = core.marker_orientations((1024, 1408), blot)
    assert c[0] == {"rot90": 0, "flip": False, "label": "원래대로"}
    # swapped: only quarter turns fit
    c = core.marker_orientations((1408, 1024), blot)
    assert {x["rot90"] for x in c} == {1, 3}
    img = np.arange(12.0).reshape(3, 4)
    assert core.orient(core.orient(img, {"rot90": 1}), {"rot90": 3}).tolist() == img.tolist()
    assert core.orient(img, {"rot90": 0, "flip": True})[0, 0] == img[0, -1]
    # a marker turned by 90 degrees fits the blot again after one of the candidates
    m = np.random.default_rng(0).random((40, 60))
    turned = np.rot90(m, 1)
    fits = [core.orient(turned, tf) for tf in core.marker_orientations(turned.shape, m.shape)]
    assert any(f.shape == m.shape and np.allclose(f, m) for f in fits)


def test_merge_bands_lane_by_lane():
    a = core.Analysis(rows=[core.Row(name="P1", width=20, height=10), core.Row(name="P2", width=20, height=10)])
    for lane in range(3):
        a.bands.append(core.Band(row=0, lane=lane, cx=50 + 40 * lane, cy=100))
        a.bands.append(core.Band(row=1, lane=lane, cx=50 + 40 * lane, cy=112))
    # one band split into two rows: select both rows in every lane -> one box per lane
    assert core.merge_bands(a, list(a.bands)) == 3
    assert len(a.rows) == 1 and len(a.bands) == 3
    assert a.rows[0].height == 22 and all(abs(b.cy - 106) <= 1 for b in a.bands)
    # two neighbouring boxes of one row -> one wide box
    assert core.merge_bands(a, a.bands[:2]) == 1
    assert len(a.bands) == 2 and a.rows[0].width == 60


def test_far_apart_bands_pinned_to_one_lane():
    """Bands far apart chosen by hand as one lane stay one lane (moves, re-derived lanes,
    lane-first lines, save/open, re-detection) and the control of that lane is used."""
    a = grid(rows=2, lanes=4)  # lanes at x = 20, 50, 80, 110; row 0 = control, row 1 = target
    a.rows[0].role = core.ROLE_CONTROL
    c0 = next(b for b in a.bands if b.row == 0 and b.lane == 0)
    t3 = next(b for b in a.bands if b.row == 1 and b.lane == 3)
    assert core.make_lane_group(a, [c0, t3], "X") == "X"
    assert c0.lane == t3.lane and c0.pin == t3.pin == "X"

    def lane_of(b):
        return a.lane_labels[b.lane]

    def check():
        assert lane_of(c0) == lane_of(t3) == "X"
        assert sum(1 for b in a.bands if b.lane == c0.lane and b.row == 1) == 1  # one target box in X
        # measurement: X's target value is divided by X's control value
        sig = np.zeros((80, 160))
        for b in a.bands:
            x, y, w, h = b.box(a.rows)
            sig[y:y + h, x:x + w] = 1.0 + b.cx / 10.0 + 5 * b.row
        meas = core.measure(sig, a, core.BackgroundSettings(method="none"))
        ratio = core.normalize_to_control(a, meas, 1, 0, None)
        area = a.rows[0].width * a.rows[0].height
        assert ratio[c0.lane] == pytest.approx((1 + t3.cx / 10 + 5) * area / ((1 + c0.cx / 10) * area))

    check()
    t3.cx += 4  # the user drags the far box a little
    core.auto_assign_lanes(a)
    check()
    assert not core.auto_assign_lanes(a)  # stable
    # lane-first results re-derive lanes from slanted lane lines
    a.method = "lane"
    a.lane_geom = [{"x0": 20.0 + 30 * i, "slope": 0.0, "half": 12, "ymid": 30.0} for i in range(4)]
    core.auto_assign_lanes(a)
    check()
    # save / open
    b2 = core.Analysis.from_dict(a.to_dict())
    assert sorted(b.pin for b in b2.bands if b.pin) == ["X", "X"]
    # detecting again: new boxes at the same places get the pins back
    new = grid(rows=2, lanes=4)
    pins = [(b.cx, b.cy, b.pin) for b in a.bands if b.pin]
    assert core.restore_pins(new, pins) == 2
    pinned = [b for b in new.bands if b.pin]
    assert len({b.lane for b in pinned}) == 1 and new.lane_labels[pinned[0].lane] == "X"
