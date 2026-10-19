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


def test_formulas_and_groups():
    a = core.Analysis(rows=[core.Row("Vinculin", 10, 10, core.ROLE_CONTROL), core.Row("p Erk", 10, 10)])
    a.bands = [core.Band(r, l, 0, 0) for r in range(2) for l in range(3)]
    a.groups = ["ctrl"]
    a.bands[3].groups.append("ctrl")
    a.bands[4].groups.append("ctrl")
    meas = [core.Measurement(a.band_id(b), b.row, b.lane, 0, 0, float(10 * (b.row + 1) + b.lane), 0, 1) for b in a.bands]
    v = core.formula_variables(a, meas)
    np.testing.assert_allclose(core.evaluate("p_Erk / Vinculin", v), [2.0, 21 / 11, 22 / 12])
    assert core.evaluate("ctrl", v) == 41.0
    assert core.evaluate("R2L3 - R1L1 * 2", v) == 2.0
    np.testing.assert_allclose(core.evaluate("G_ctrl", v)[:2], [20.0, 21.0])
    assert np.isnan(core.evaluate("G_ctrl", v)[2])
    assert np.isnan(core.evaluate("R1L1 / 0", v))
    with pytest.raises(core.FormulaError):
        core.evaluate("__import__('os')", v)
    with pytest.raises(core.FormulaError):
        core.evaluate("nope + 1", v)


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
