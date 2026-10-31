"""Tests of 170_dubois_reconcile.py on synthetic recordings only."""
import importlib
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "blind_impl")]

import dubois_blind as db
from bench import pipe_dubois as pdb
from bench.tests import _synth as S

rc = importlib.import_module("170_dubois_reconcile")


@pytest.fixture(scope="module")
def window():
    ov = pdb.schedule(30.0, 0, "assessment_blocked_phase1")
    r = S.rec(pdb.extent(ov)[1] + 10.0, seed=3, drift=0.5)
    return r, ov


def test_ladder_order_and_singles():
    assert [s for s, _ in rc.LADDER] == ["L0_blind", "L1_folds", "L2_channels", "L3_peak", "L4_step", "L5_bands",
                                         "L6_csp_pairs", "L7_mi", "L8_lda"]
    for k, (name, ch) in enumerate(rc.LADDER[1:], 1):
        prev = rc.LADDER[k - 1][1]
        assert len(set(ch.items()) - set(prev.items())) == 1, name
    assert set(rc.SINGLES) == {s for s, _ in rc.LADDER[1:]}
    for name, ch in rc.SINGLES.items():
        assert len(ch) == 1


def test_l0_equals_blind_as_written(window):
    r, ov = window
    xc, a = rc.crop(r, ov)
    ev = rc.events(ov, a)
    ref = db.assess(xc, r.fs, r.names, [e[:2] for e in ev], seed=0)
    got = rc.run_blind(r, ov, {}, seed=0)
    assert got["call"] == ref["call"]
    assert got["peak_da"] == pytest.approx(ref["peak_da"], abs=0)
    assert got["p_t"] == pytest.approx(ref["p_t"], abs=0)


def test_patches_restore_module_state(window):
    r, ov = window
    before = {k: getattr(db, k) for k in ("_folds", "_mutual_info", "_rlda_predict", "BANDS", "N_PAIRS", "STEP_S")}
    rc.run_blind(r, ov, dict(rc.LADDER[-1][1]), seed=0)
    for k, v in before.items():
        assert getattr(db, k) is v, k


def test_patches_restore_after_error(window):
    r, ov = window
    before = db._folds
    with pytest.raises(ValueError):
        rc.run_blind(r, ov, {"folds": "blocked"}, seed=0, blocks_override=np.zeros(3, int))
    assert db._folds is before


def test_blocked_folds_are_the_blocks(window):
    r, ov = window
    seen = {}
    orig = db._rlda_predict

    def spy(Xtr, ytr, Xte, *a, **k):
        seen.setdefault("n_tr", set()).add(len(ytr))
        return orig(Xtr, ytr, Xte, *a, **k)
    db._rlda_predict = spy
    try:
        rc.run_blind(r, ov, {"folds": "blocked"}, seed=0)
    finally:
        db._rlda_predict = orig
    assert seen["n_tr"] == {75}


def test_gaussian_mi_matches_study():
    g = np.random.default_rng(0)
    X = g.standard_normal((4, 40, 6))
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    X[:, y == 1, 2] += 1.5
    got = rc.mi_gaussian(X, y)
    for t in range(4):
        ref = pdb.mi_scores(X[t], np.where(y == 1, pdb.CLASS_B, pdb.CLASS_A))
        np.testing.assert_allclose(got[t], ref, rtol=1e-12, atol=1e-12)
    assert np.argmax(got[0]) == 2


def test_lw_lda_matches_study():
    g = np.random.default_rng(1)
    Xtr = g.standard_normal((3, 36, 5))
    ytr = np.r_[np.zeros(14, int), np.ones(22, int)]
    Xtr[:, ytr == 1, 0] += 1.0
    Xte = g.standard_normal((3, 10, 5))
    got = rc.rlda_lw_predict(Xtr, ytr, Xte)
    for t in range(3):
        w, b0 = pdb.rlda(Xtr[t], np.where(ytr == 1, pdb.CLASS_B, pdb.CLASS_A))
        np.testing.assert_array_equal(got[t], (Xte[t] @ w + b0 > 0).astype(np.int8))


def test_reading_b_matches_130():
    fr = importlib.import_module("130_fidelity_readings")
    g = np.random.default_rng(2)
    T = len(pdb.TIMES)
    res = {("full", (0,), 1.0, 4): 0.5 + 0.05 * g.standard_normal((6, T)),
           ("full", (1,), 1.0, 4): 0.5 + 0.05 * g.standard_normal((6, T))}
    res[("full", (1,), 1.0, 4)][:, 40:45] += 0.15
    out = rc.reading_b_rule(res, pdb.TIMES, 0.1)
    sm = {k: pdb.smooth(v) for k, v in res.items()}
    task, _ = pdb._masks()
    best = max(sm, key=lambda k: sm[k].mean(0)[task].max())
    tv, p, ti, bi = fr.reading_b(sm[best].mean(0), sm[best])
    assert out["config"] == best
    assert out["p_t"] == pytest.approx(p)
    assert out["peak_da"] == pytest.approx(sm[best].mean(0)[task].max())
    assert out["call"] == bool(out["peak_da"] > 0.7 or p < 0.05)


def test_channels_factor_drops_frontopolar(window):
    r, ov = window
    x, names = rc.select_channels(r, "study17")
    assert names == list(pdb.CHANSETS["full"])
    assert x.shape[0] == 17
    x2, n2 = rc.select_channels(r, "all")
    assert x2.shape[0] == 19 and n2 == list(r.names)


def test_summary_counts_and_gap_share():
    rows = []
    for i in range(10):
        rows.append({"unit": f"u{i}", "step": "L0_blind", "call": True})
        rows.append({"unit": f"u{i}", "step": "L1_folds", "call": i < 4})
        rows.append({"unit": f"u{i}", "step": "S_L1_folds", "call": i < 4})
        rows.append({"unit": f"u{i}", "step": "study_blocked", "call": i < 2})
    s = rc.summarize(rows)
    assert s["steps"]["L0_blind"]["k"] == 10 and s["steps"]["L1_folds"]["k"] == 4
    assert s["gap_share"]["S_L1_folds"] == pytest.approx((1.0 - 0.4) / (1.0 - 0.2))
    assert s["mcnemar"]["L0_blind->L1_folds"]["b"] == 6 and s["mcnemar"]["L0_blind->L1_folds"]["c"] == 0
    assert s["n_units"] == 10


def test_blind_selection_reproduces_the_blind_peak(window):
    r, ov = window
    out, res, step, _ = rc._assess(r, ov, {}, 0)
    cfg, ti, peak = rc.blind_selection(res["results"], res["times"], r.fs, step)
    assert peak == pytest.approx(out["peak_da"], abs=0)
    assert cfg in res["results"]


def test_fixed_config_runs_one_configuration(window):
    r, ov = window
    _, res, _, _ = rc._assess(r, ov, {"config": "fixed"}, 0)
    assert len(res["results"]) == 1
    assert db.N_FEATURES == tuple(range(4, 11)) and len(db.BAND_SETS) == 3 and db.MOTOR_SET


def test_permutation_evaluation_small(window):
    r, ov = window
    for folds in ("random", "blocked"):
        out = rc.run_blind_perm(r, ov, {"folds": folds}, seed=0, n_perm=3)
        assert out["p_perm"] in (0.25, 0.5, 0.75, 1.0)
        assert out["call"] is False
        assert out["adv_and_perm"] == (out["adv_call"] and out["p_perm"] < 0.05)
    assert db.WINDOWS == (1.0, 2.0)


def test_factorial_has_eight_cells():
    assert len(rc.FACTORIAL) == 8
    assert rc.FACTORIAL["F_Rxs"] == {"folds": "random", "peak": "refine", "config": "search"}
    assert rc.FACTORIAL["F_Bbf"] == {"folds": "blocked", "peak": "reading_b", "config": "fixed"}


def test_failed_fit_does_not_reuse_the_previous_result(window):
    r, ov = window
    rc.run_blind(r, ov, {}, seed=0)
    bad = type(r)(r.source, r.uid + "nan", r.patient, r.x.copy(), r.fs, list(r.names), {})
    bad.x[0, :] = np.nan
    out = rc.run_blind(bad, ov, {"peak": "reading_b"}, seed=0)
    assert out["call"] is None and out["status"].startswith("no_fit")


def test_reading_b_smoothing_width_is_rate_independent():
    T = 200
    for fs in (128.0, 125.0, 250.0):
        step = max(1, int(round(0.04 * fs))) / fs
        times = np.arange(T) * step - 1.0
        res = {("full", (0,), 1.0, 4): np.full((6, T), 0.5)}
        res[("full", (0,), 1.0, 4)][:, 100] = 1.0
        out = rc.reading_b_rule(res, times, step)
        assert out["peak_da"] == pytest.approx(0.5 + 0.5 / 5)
