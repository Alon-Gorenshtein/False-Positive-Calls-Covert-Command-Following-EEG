"""Tests for code/163_stroke_acute_control.py: the 40-trial chance bound, label shuffles, the patient-cluster bootstrap
(which must RELABEL resampled patients), the per-patient detection / false-call run and the summary numbers.
cd code && /usr/bin/python3 -m pytest test_163_stroke_acute_control.py -q"""
import importlib

import numpy as np
import pytest

sc = importlib.import_module("163_stroke_acute_control")
kc = importlib.import_module("162_stroke_check")


def planted(n, n_ch, rng, effect=3.0, T=250):
    X = rng.standard_normal((2 * n, n_ch, T))
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    X[y == 1, 0] *= effect
    X[y == 0, 1] *= effect
    return X, y



def test_bound_reproduces_63_19_at_60_and_66_20_at_40_from_one_formula():
    assert sc.bound(60) == pytest.approx(0.631938, abs=5e-7)
    assert sc.bound(40) == pytest.approx(0.661982, abs=5e-7)
    assert sc.bound(40) == kc.chance_bound(40)


def test_a_call_needs_accuracy_strictly_above_the_bound():
    b = sc.bound(40)
    assert sc.call(b + 1e-9, 40) and not sc.call(b, 40) and not sc.call(0.5, 40)



def test_shuffles_are_reproducible_per_patient_keep_the_label_counts_and_differ_between_patients():
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    a, b = sc.shuffles(y, 7, 20), sc.shuffles(y, 7, 20)
    assert (a == b).all() and a.shape == (20, 40)
    assert (a.sum(1) == 20).all()
    assert len({tuple(r) for r in a}) == 20
    assert not (sc.shuffles(y, 8, 20) == a).all()
    assert not (a == y).all(axis=1).any()
    r = np.random.default_rng([20261022, 7])
    assert (a[0] == r.permutation(y)).all()



def test_bootstrap_relabels_resampled_patients_and_a_non_relabelled_version_merges_them():
    rows = {"pid": np.repeat(np.arange(4), 5), "called": np.r_[[1, 1, 0, 0, 0], np.zeros(15)].astype(bool)}
    pat = {"pid": np.arange(4), "true_call": np.array([1, 1, 0, 0], bool)}
    draw = np.array([0, 0, 0, 1])
    good = sc.stats_of_draw(rows, pat, draw, relabel=True)
    bad = sc.stats_of_draw(rows, pat, draw, relabel=False)
    assert good["n_clusters"] == 4 and bad["n_clusters"] == 2
    assert good["any_false_share"] == pytest.approx(3 / 4)
    assert bad["any_false_share"] == pytest.approx(1 / 2)
    assert good["false_rate"] == pytest.approx((0.4 * 3 + 0) / 4)
    assert good["detection"] == pytest.approx(1.0)
    assert good["diff"] == pytest.approx(good["detection"] - good["false_rate"])


def test_bootstrap_is_seeded_and_its_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    n = 30
    calls = rng.random((n, 20)) < 0.1
    rows = {"pid": np.repeat(np.arange(n), 20), "called": calls.ravel()}
    pat = {"pid": np.arange(n), "true_call": rng.random(n) < 0.3}
    a = sc.cluster_bootstrap(rows, pat, B=300, seed=1)
    b = sc.cluster_bootstrap(rows, pat, B=300, seed=1)
    assert a == b
    pt = sc.stats_of_draw(rows, pat, np.arange(n), relabel=True)
    for k in ("false_rate", "detection", "diff", "any_false_share"):
        lo, hi = a[k]["ci95"]
        assert lo <= pt[k] <= hi and a[k]["estimate"] == pytest.approx(pt[k])



def test_run_patient_calls_a_planted_patient_and_rarely_calls_shuffles():
    X, y = planted(20, 8, np.random.default_rng(3))
    names = ["FCz", "FC3", "FC4", "Cz", "C3", "C4", "CP3", "CP4"]
    lab = list(np.where(y == 1, "right", "left"))
    r = sc.run_patient(X, lab, names, idx=5, n_shuf=6, reps=2)
    assert r["sm"]["true_acc"] > 0.9 and r["sm"]["true_call"] is True
    assert len(r["sm"]["shuf_acc"]) == 6 and np.mean(r["sm"]["shuf_calls"]) < 0.5
    assert r["all"]["n_channels"] == 8 and r["sm"]["channels"] == names


def test_run_patient_uses_only_the_named_sensorimotor_channels():
    X, y = planted(20, 9, np.random.default_rng(4))
    names = ["FCz", "FC3", "FC4", "Cz", "C3", "C4", "CP3", "CP4", "Oz"]
    r = sc.run_patient(X, list(np.where(y == 1, "right", "left")), names, idx=1, n_shuf=2, reps=1)
    assert r["sm"]["channels"] == names[:8] and r["all"]["n_channels"] == 9
    with pytest.raises(ValueError, match="FCz"):
        sc.run_patient(X[:, 1:], list(np.where(y == 1, "right", "left")), names[1:], idx=1, n_shuf=2, reps=1)



def test_summary_numbers_on_a_hand_checkable_example():
    pats = {i: {"n_trials": 40, "sm": {"true_acc": acc, "true_call": acc > sc.bound(40),
                                       "shuf_acc": [0.5] * 20, "shuf_calls": [False] * 20}}
            for i, acc in zip((1, 2, 3, 4), (0.8, 0.7, 0.5, 0.45))}
    pats[3]["sm"]["shuf_calls"][0] = True
    pats[3]["sm"]["shuf_acc"][0] = 0.7
    pats[4]["sm"]["shuf_calls"][:2] = [True, True]
    s = sc.summarize(pats, "sm", B=200)
    assert s["detection"]["k"] == 2 and s["detection"]["n"] == 4 and s["detection"]["rate"] == 0.5
    assert s["false_call_rate"]["estimate"] == pytest.approx((1 / 20 + 2 / 20) / 4)
    assert s["share_patients_with_a_false_call"]["k"] == 2
    assert s["detection_minus_false_rate"]["estimate"] == pytest.approx(0.5 - 3 / 80)
    assert s["median_true_acc"] == pytest.approx(0.6)
    lo, hi = s["detection"]["exact_ci95"]
    assert lo == pytest.approx(0.0676, abs=5e-4) and hi == pytest.approx(0.9324, abs=5e-4)


def test_distinct_patients_drop_the_later_member_of_each_duplicate_pair():
    arrays = {1: np.random.default_rng(1).standard_normal((4, 3, 20)), 2: np.random.default_rng(2).standard_normal((4, 3, 20))}
    arrays[3] = arrays[1] * 1.0
    assert sc.distinct(arrays) == [1, 2]
