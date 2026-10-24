import importlib
import numpy as np
import pandas as pd
import pytest

bn = importlib.import_module("108_bdsp_null")


def _morgoth(n, **cols):
    base = {"t_s": np.arange(n), "valid": np.ones(n, bool), "valid_sleep": np.ones(n, bool), "normal_p": 0.9,
            "slowing_none": 0.95, "slowing_focal": 0.0, "slowing_generalized": 0.05, "bs_p": 0.0,
            "iiic_other": 1.0, "iiic_seizure": 0.0, "iiic_lpd": 0.0, "iiic_gpd": 0.0, "iiic_lrda": 0.0,
            "iiic_grda": 0.0, "sleep_awake": 0.9, "sleep_n1": 0.05, "sleep_n2": 0.05, "sleep_n3": 0.0,
            "sleep_rem": 0.0}
    base.update(cols)
    return pd.DataFrame({k: (np.full(n, v) if np.isscalar(v) else v) for k, v in base.items()})


def test_drift_index_higher_for_slow_drift_than_white_noise():
    rng = np.random.default_rng(0)
    white = rng.normal(size=(48, 4))
    ar = np.zeros((48, 4))
    for t in range(1, 48):
        ar[t] = 0.95 * ar[t - 1] + rng.normal(size=4)
    assert bn.drift_index(ar) > bn.drift_index(white) + 0.2


def test_drift_index_ignores_constant_offset_and_scale():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(48, 4)).cumsum(0)
    assert bn.drift_index(3.0 * F + 7.0) == pytest.approx(bn.drift_index(F))


def test_row_labels_follow_the_hierarchy():
    m = _morgoth(6, bs_p=[0.9, 0, 0, 0, 0, 0], iiic_other=[1, 0.2, 1, 1, 1, 1],
                 sleep_n2=[0, 0, 0.8, 0, 0, 0], sleep_awake=[0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
                 slowing_generalized=[0, 0, 0, 0.8, 0, 0], normal_p=[0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    assert list(bn.row_labels(m)) == ["burst_suppression", "iiic_periodic", "sleep", "diffuse_slowing",
                                      "normal_awake", "other_awake"]


def test_row_labels_mark_invalid_rows_and_respect_normal_direction():
    m = _morgoth(2, valid=np.array([False, True]), normal_p=0.1)
    assert list(bn.row_labels(m)) == ["invalid", "other_awake"]
    assert list(bn.row_labels(m, normal_is_p_normal=False)) == ["invalid", "normal_awake"]


def test_run_stratum_mode_mixed_unlabelled_and_transition():
    lab = np.array(["sleep"] * 400 + ["normal_awake"] * 71)
    assert bn.run_stratum(lab) == ("sleep", False)
    half = np.array(["sleep"] * 236 + ["normal_awake"] * 235)
    assert bn.run_stratum(half) == ("mixed", True)
    assert bn.run_stratum(np.array(["invalid"] * 300 + ["sleep"] * 171))[0] == "unlabelled"
    tr = np.array(["sleep"] * 200 + ["normal_awake"] * 30 + ["normal_awake"] * 241)
    s, t = bn.run_stratum(tr)
    assert t is True


def test_eligible_starts_exclude_zero_padding_with_margin_but_keep_large_artifacts():
    fs = 128
    rng = np.random.default_rng(2)
    x = rng.normal(0, 10, size=(4, fs * 1300)).astype(np.float32)
    x[:, : fs * 600] = 0.0
    x[:, fs * 900: fs * 905] *= 50
    s = bn.eligible_starts(x, fs, run_s=480)
    assert s.min() >= 600 + 30
    assert np.any((s <= 900) & (s + 480 >= 905))


def test_eligible_starts_exclude_long_noise_floor_but_keep_brief_suppression():
    fs = 128
    rng = np.random.default_rng(4)
    x = rng.normal(0, 10, size=(4, fs * 2000)).astype(np.float32)
    x[:, : fs * 900] = rng.normal(0, 1.0, size=(4, fs * 900))
    x[:, fs * 1400: fs * 1420] = rng.normal(0, 1.0, size=(4, fs * 20))
    s = bn.eligible_starts(x, fs, run_s=480)
    assert s.min() >= 900 + 30
    assert np.any((s <= 1400) & (s + 480 >= 1420))


def test_stratum_rates_and_fixed_pair_bootstrap():
    rows = pd.DataFrame({"seg": np.repeat(np.arange(60), 10),
                         "stratum": np.repeat(["a"] * 30 + ["b"] * 30, 10),
                         "call": np.r_[np.ones(300, bool), np.zeros(300, bool)]})
    r = bn.stratum_rates(rows)
    assert r["a"] == 1.0 and r["b"] == 0.0
    d, lo, hi = bn.pair_diff_ci(rows, "a", "b", draws=200, seed=0)
    assert d == 1.0 and lo == 1.0 and hi == 1.0


def test_gate_c_fails_only_when_lower_bound_exceeds_alpha():
    ok = pd.DataFrame({"seg": np.arange(400), "stratum": "a", "call": np.r_[np.ones(20, bool), np.zeros(380, bool)]})
    bad = pd.DataFrame({"seg": np.arange(400), "stratum": "a", "call": np.r_[np.ones(80, bool), np.zeros(320, bool)]})
    assert bn.not_above_alpha(ok, "a", draws=500, seed=0)
    assert not bn.not_above_alpha(bad, "a", draws=500, seed=0)


def test_overlay_returns_every_design():
    fs = 128
    rng = np.random.default_rng(3)
    x = rng.normal(0, 3, size=(4, fs * 600)).astype(np.float32)
    out = bn.overlay(x, fs, start_s=10, rng=rng, n_perm=50)
    want = {f"B1_b{b}" for b in bn.BLOCKS} | {f"B2_b{b}" for b in bn.BLOCKS} | {"B4", "R", "Pcol_a", "Pcol_b"}
    assert set(out["calls"]) == want
    assert np.isfinite(out["drift"])
