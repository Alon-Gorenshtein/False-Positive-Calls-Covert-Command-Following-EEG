"""pipe_dubois (Du Bois 2026, cat. §9.2): schedule timing, the 84-configuration list, the planted-effect call, a noise
call rate and run_randomized validity. Speed-ups (documented): the noise test runs the full 84-configuration search;
the 200-draw validity test uses a 4-configuration search (motor channels, one band set, both windows, k = 4 and 6),
20 noise recordings x 10 re-drawn orders (exact conditional on the data), B = 40 (exact level 2/41 = 4.9%) and a
process pool. The randomization test's validity does not depend on the number of configurations searched."""
import time

import numpy as np
from scipy.stats import binom

from bench import pipe_dubois as pm
from bench.tests import _synth as sy

SMALL = [('motor', 'mu-lowgamma', 1.0, 4), ('motor', 'mu-lowgamma', 1.0, 6), ('motor', 'mu-lowgamma', 2.0, 4),
         ('motor', 'mu-lowgamma', 2.0, 6)]


def _dur(ov):
    return float(ov.onsets_s.max() + pm.TIMES[-1] + 15.0)


def test_schedule_timing_and_configs():
    ov = pm.schedule(10.0, np.random.default_rng(0), 'assessment_random_phase1')
    blk = ov.meta['block']
    assert len(ov.onsets_s) == 90 and (np.bincount(blk) == 15).all()
    assert [ov.labels[blk == b][0] for b in range(6)] == [1, 2, 1, 2, 1, 2]
    for b in range(6):
        assert np.allclose(np.diff(ov.onsets_s[blk == b]), 8.0)
    span = ov.onsets_s[-1] + 8.0 - 10.0
    assert 16 * 60 < span < 18 * 60 and _dur(ov) - 10.0 <= pm.RUN_S
    tr = pm.schedule(10.0, np.random.default_rng(0), 'training_random_phase1')
    assert len(tr.labels) == 60 and (tr.labels == 1).sum() == 30 and np.allclose(np.diff(tr.onsets_s), 8.0)
    assert len(pm.CONFIGS) == 84 and len(set(pm.CONFIGS)) == 84
    rz = pm.schedule_randomized(10.0, np.random.default_rng(1), 'assessment_random_phase1')
    assert (rz.labels == 1).sum() == 45 and any(len(set(rz.labels[rz.meta['block'] == b])) == 2 for b in range(6))


def _planted(seed, v):
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    r = sy.rec(_dur(ov), seed=seed)
    r = sy.plant(r, ov.onsets_s[ov.labels == pm.CLASS_A], ['C3'], 0.5, 5.5, depth=0.8)
    return sy.plant(r, ov.onsets_s[ov.labels == pm.CLASS_B], ['C4'], 0.5, 5.5, depth=0.8), ov


def test_planted_effect_is_called():
    v = 'training_random_phase1'
    r, ov = _planted(3, v)
    t = time.time()
    res = pm.run(r, ov, v, np.random.default_rng(0))
    print(f"dubois {v} planted: peak DA {res['extra']['peak_da']:.3f} t {res['extra']['t']:.2f} "
          f"p {res['p']:.4f} config {res['extra']['config']} ({time.time() - t:.1f} s)")
    assert res['call'] and res['extra']['calls']['t_only']
    vp = dict(pm.VARIANTS['training_random_perm'], n_perm_labels=20)
    t = time.time()
    assert pm.run(r, ov, vp, np.random.default_rng(0))['call']
    print(f'dubois perm (20 label permutations): {time.time() - t:.1f} s')
    t = time.time()
    rr = pm.run_randomized(r, ov, dict(pm.VARIANTS['training_random_t_only'], n_perm_rand=40),
                           np.random.default_rng(1))
    print(f"dubois run_randomized planted (B = 40): p {rr['p']:.4f} ({time.time() - t:.1f} s)")
    assert rr['call']


def test_noise_call_rate():
    rates, t = {}, time.time()
    for v in ('assessment_random_phase1', 'training_random_phase1'):
        calls, tonly = [], []
        for s in range(4):
            ov = pm.schedule(10.0, np.random.default_rng(40 + s), v)
            res = pm.run(sy.rec(_dur(ov), seed=40 + s), ov, v, np.random.default_rng(s))
            calls.append(res['call'])
            tonly.append(res['extra']['calls']['t_only'])
        rates[v] = (float(np.mean(calls)), float(np.mean(tonly)))
    print('dubois noise (phase1, t_only) over 4 stationary draws, full search:', rates, f'{time.time() - t:.0f} s')
    assert all(0 <= a <= 1 for r in rates.values() for a in r)


def _validity_task(s):
    v = dict(pm.VARIANTS['assessment_random_t_only'], n_perm_rand=40, configs=SMALL)
    ov = pm.schedule_randomized(10.0, np.random.default_rng(10 ** 6 + s), v)
    r = sy.rec(_dur(ov), seed=1700 + s, drift=0.5)
    calls = []
    for k in range(10):
        ov.labels = pm.mc.balanced_labels((1, 2), (45, 45), np.random.default_rng(1000 * s + k))
        calls.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'dubois run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi
