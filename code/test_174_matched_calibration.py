"""Tests of 174_matched_calibration.py on synthetic rows only."""
import importlib

import numpy as np
import pandas as pd
import pytest

mcal = importlib.import_module("174_matched_calibration")


def _null(n=2000, seed=0):
    g = np.random.default_rng(seed)
    return pd.DataFrame({"unit": [f"p{i}" for i in range(n)], "stat": g.standard_normal(n)})


def test_per_assessment_alpha_gives_alpha_over_k():
    for k in (1, 2, 5, 10):
        a = mcal.per_assessment_alpha(0.05, k)
        assert 1 - (1 - a) ** k == pytest.approx(0.05)
    assert mcal.per_assessment_alpha(0.05, 1) == pytest.approx(0.05)


def test_threshold_holds_size_on_held_out_draws():
    rates = []
    for s in range(20):
        nd = _null(4000, s)
        res = mcal.crossfit(nd, pd.DataFrame({"unit": [], "stat": []}), 0.05, 1)
        rates.append(res["false_call_heldout"])
    assert abs(np.mean(rates) - 0.05) < 0.006


def test_halves_split_by_patient_not_by_row():
    nd = pd.DataFrame({"unit": ["a", "a", "b", "b", "c", "c"], "stat": [1, 2, 3, 4, 5, 6.0]})
    h = mcal.half_of(nd["unit"])
    assert h[0] == h[1] and h[2] == h[3] and h[4] == h[5]


def test_detection_falls_as_assessments_rise():
    nd = _null(3000, 1)
    g = np.random.default_rng(9)
    task = pd.DataFrame({"unit": [f"t{i}" for i in range(400)], "stat": 1.8 + g.standard_normal(400)})
    det = [mcal.crossfit(nd, task, 0.05, k)["detection"] for k in (1, 2, 5, 10)]
    assert all(a >= b for a, b in zip(det, det[1:]))
    assert det[0] > 0.3


def test_missing_statistic_is_not_a_negative():
    nd = _null(2000, 2)
    task = pd.DataFrame({"unit": ["t0", "t1", "t2"], "stat": [10.0, np.nan, 10.0]})
    res = mcal.crossfit(nd, task, 0.05, 1)
    assert res["n_task"] == 2 and res["n_task_missing"] == 1
    assert res["detection"] == pytest.approx(1.0)


def test_p_based_comparator_uses_p_at_or_below_level():
    rows = pd.DataFrame({"unit": ["a", "b", "c", "d"], "p": [0.004, 0.02, 0.05, 0.3]})
    assert mcal.p_rule_rate(rows, mcal.per_assessment_alpha(0.05, 1)) == pytest.approx(3 / 4)
    assert mcal.p_rule_rate(rows, mcal.per_assessment_alpha(0.05, 5)) == pytest.approx(1 / 4)


def test_published_patient_rate_under_independence():
    assert mcal.any_positive(0.233, 2) == pytest.approx(1 - 0.767 ** 2)
    assert mcal.any_positive(0.0, 5) == 0.0


def test_curve_is_monotone_in_level():
    nd = _null(3000, 3)
    g = np.random.default_rng(4)
    task = pd.DataFrame({"unit": [f"t{i}" for i in range(300)], "stat": 1.0 + g.standard_normal(300)})
    cv = mcal.curve(nd, task, levels=(0.01, 0.02, 0.05, 0.1, 0.2))
    fp = [c["false_call_heldout"] for c in cv]
    det = [c["detection"] for c in cv]
    assert all(a <= b + 1e-12 for a, b in zip(fp, fp[1:]))
    assert all(a <= b + 1e-12 for a, b in zip(det, det[1:]))


def test_halves_follow_the_patient_column_when_present():
    nd = pd.DataFrame({"unit": ["r1", "r2", "r3", "r4"], "patient": ["a", "a", "b", "b"], "stat": [1.0, 2, 3, 4]})
    h = mcal.half_of(nd["patient"])
    assert h[0] == h[1] and h[2] == h[3]


def test_ties_at_the_threshold_cannot_exceed_the_level():
    nd = pd.DataFrame({"unit": [f"p{i}" for i in range(400)], "stat": np.r_[np.zeros(300), np.full(100, 5.0)]})
    res = mcal.crossfit(nd, pd.DataFrame({"unit": [], "stat": []}), 0.05, 1)
    assert res["false_call_heldout"] <= 0.05


def test_infinite_statistic_is_capped_not_missing():
    nd = _null(2000, 5)
    task = pd.DataFrame({"unit": ["t0", "t1"], "stat": [np.inf, np.nan]})
    res = mcal.crossfit(nd, task, 0.05, 1)
    assert res["n_task"] == 1 and res["detection"] == 1.0


def test_task_units_are_judged_by_the_other_halfs_threshold():
    g = np.random.default_rng(7)
    pts = [f"p{i}" for i in range(6000)]
    nd = pd.DataFrame({"unit": pts, "patient": pts, "stat": g.standard_normal(6000)})
    task = pd.DataFrame({"unit": pts, "patient": pts, "stat": g.standard_normal(6000)})
    res = mcal.crossfit(nd, task, 0.05, 1)
    assert abs(res["detection"] - 0.05) < 0.012


def _runs(n=4000, k=10, rho=0.5, seed=7):
    g = np.random.default_rng(seed)
    u = g.standard_normal(n) * np.sqrt(rho)
    e = g.standard_normal((n, k)) * np.sqrt(1 - rho)
    x = u[:, None] + e
    return pd.DataFrame({"patient": np.repeat([f"p{i}" for i in range(n)], k), "j": np.tile(np.arange(k), n),
                         "stat": x.ravel()})


def test_direct_threshold_holds_patient_level_rate():
    runs = _runs()
    task = pd.DataFrame({"unit": ["t0", "t1"], "stat": [10.0, -10.0]})
    r = mcal.direct_patient_level(runs, task, 5)
    assert abs(r["direct"]["rate"] - 0.05) < 0.02
    assert r["sidak"]["rate"] <= 0.05 + 0.01
    assert r["direct"]["detection"] == pytest.approx(0.5)
    lo, hi = r["direct"]["ci95"]
    assert lo <= r["direct"]["rate"] <= hi


def test_patients_with_fewer_runs_are_excluded_and_counted():
    runs = _runs(n=200, k=3)
    runs = runs[~((runs["patient"] == "p0") & (runs["j"] == 2))]
    r = mcal.direct_patient_level(runs, pd.DataFrame({"unit": [], "stat": []}), 3)
    assert r["n_patients"] == 199 and r["n_excluded"] == 1


def test_crossfit_reports_pooled_heldout_counts_with_ci():
    res = mcal.crossfit(_null(2000, 11), pd.DataFrame({"unit": [], "stat": []}), 0.05, 1)
    k, n = res["false_call_heldout_k"], res["false_call_heldout_n"]
    assert n == 2000 and 0 < k < n
    lo, hi = res["false_call_heldout_ci95"]
    assert lo <= k / n <= hi
