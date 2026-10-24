"""pipe_columbia wraps 112_columbia_exact (bit-exactness vs the authors' code is tested in code/test_columbia_exact.py).
Here: schedule timing, the per-trial randomized order, the planted-effect call, a noise call rate and the validity of
run_randomized. Speed-ups (documented): N_PERM is lowered for the planted/noise tests, and the 200-draw validity test
uses features of 20 noise recordings x 10 randomized orders each (the randomization test is exact conditional on the
data, so re-using data across orders keeps every draw valid) with B = 20 (exact level 1/21 = 4.8%)."""
import time

import numpy as np
from scipy.stats import binom

from bench import pipe_columbia as pc
from bench.tests import _synth as sy

ce = pc.ce


def test_schedule_timing_matches_the_coded_protocol():
    ov = pc.schedule(30.0, None)
    on, lab = ov.onsets_s, ov.labels
    assert len(on) == 96 and on[0] == 30.0
    assert np.isin(lab[0::2], (ce.KEEP_L, ce.KEEP_R)).all() and (lab[1::2] == lab[0::2] + 1).all()
    assert np.allclose(on[1::2] - on[0::2], ce.CMD_S + 10.0)
    assert np.isclose(on[-1] + ce.CMD_S + ce.WIN_S - on[0], 1502.2)
    assert on[-1] + ce.CMD_S + ce.WIN_S + ce.TMAX - 30.0 <= pc.RUN_S
    assert (lab[0:16:2] == ce.KEEP_R).all() and (lab[16:32:2] == ce.KEEP_L).all()


def test_randomized_schedule_flips_whole_trials_and_keeps_timing():
    a, b = pc.schedule(30.0, None), pc.schedule_randomized(30.0, np.random.default_rng(3))
    assert np.array_equal(a.onsets_s, b.onsets_s)
    pa, pb = a.labels.reshape(-1, 2), b.labels.reshape(-1, 2)
    f = b.meta['flipped']
    assert np.array_equal(pb[f], pa[f][:, ::-1]) and np.array_equal(pb[~f], pa[~f])
    assert 12 < f.sum() < 36


def _planted(seed):
    ov = pc.schedule(20.0, None)
    r = sy.rec(1540, seed=seed)
    keep_on = ov.onsets_s[0::2]
    return sy.plant(r, keep_on, ['C3', 'Cz', 'C4'], ce.CMD_S, ce.CMD_S + 10.0, depth=0.6), ov


def test_planted_effect_is_called(monkeypatch):
    monkeypatch.setattr(pc, 'N_PERM', 100)
    r, ov = _planted(1)
    t = time.time()
    res = pc.run(r, ov, 'as_coded', np.random.default_rng(0))
    print(f"columbia planted: auc {res['stat']:.3f} p {res['p']:.4f} ({time.time() - t:.1f} s)")
    assert res['call'] and res['stat'] > 0.6


def test_noise_call_rate(monkeypatch):
    monkeypatch.setattr(pc, 'N_PERM', 100)
    calls = []
    for s in range(8):
        r = sy.rec(1540, seed=100 + s)
        calls.append(pc.run(r, pc.schedule(20.0, None), 'as_coded', np.random.default_rng(s))['call'])
    print('columbia noise call rate (8 stationary draws):', np.mean(calls))
    assert np.mean(calls) <= 0.5


def _validity_task(s):
    r = sy.rec(1540, seed=500 + s, drift=0.5)
    ov0 = pc.schedule(20.0, None)
    X, _, g = ce.features(r.x, r.fs, ov0.onsets_s, ov0.labels, r.names)
    calls = []
    for k in range(10):
        ov = pc.schedule_randomized(20.0, np.random.default_rng(1000 * s + k))
        y = np.repeat(np.isin(ov.labels, (ce.KEEP_L, ce.KEEP_R)), ce.N_SEG)
        res = ce.trialflip_test(X, y, g, np.random.RandomState(7 * s + k), n_perm=20,
                                stop_at=pc.mc.stop_count(20, 0.05))
        calls.append(res['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t0 = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'columbia run_randomized: {n}/200 false calls ({time.time() - t0:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi


def test_patient_rule_is_bh_over_recordings():
    mk = lambda p: {'stat': 0.6, 'p': p, 'call': p < 0.05, 'extra': {}}
    assert pc.patient_rule([mk(0.01), mk(0.5)])
    assert not pc.patient_rule([mk(0.03), mk(0.04), mk(0.9)])
    assert not pc.patient_rule([mk(0.049), mk(0.6), mk(0.7)])
    assert pc.patient_rule([mk(0.02), mk(0.03)])
