"""pipe_mbmi (mindBEAGLE-type MI, cat. §6): schedule timing, the 63.19% threshold, the planted-effect call, a noise
call rate and run_randomized validity. Speed-up (documented): the 200-draw validity test uses 20 noise recordings x
10 re-drawn orders (exact conditional on the data), B = 40 (exact level 2/41 = 4.9%) and a process pool."""
import time

import numpy as np
import pytest
from scipy.stats import binom

from bench import pipe_mbmi as pm
from bench.tests import _synth as sy


def _dur(ov):
    return float(ov.onsets_s.max() + pm.T_END + 10.0)


def test_schedule_timing():
    for v, step in (('gu17_cspfold_peak', (6.5, 8.0)), ('ch18_cspfold_peak', (9.0, 9.0)),
                    ('or17_60_cspfold_peak', (9.0, 9.0)), ('or17_120_cspfold_peak', (9.0, 9.0))):
        ov = pm.schedule(10.0, np.random.default_rng(0), v)
        n = pm.VARIANTS[v]['n']
        assert len(ov.labels) == n and (ov.labels == pm.LEFT).sum() == n // 2
        d = np.diff(ov.onsets_s)
        if v.startswith('or17_60'):
            assert np.isclose(d[29], 69.0) and np.allclose(np.delete(d, 29), 9.0)
        else:
            assert d.min() >= step[0] - 1e-9 and d.max() <= step[1] + 1e-9
        assert _dur(ov) - 20.0 <= pm.RUN_S


def test_threshold_is_the_clopper_pearson_line():
    assert abs(pm.threshold(pm.VARIANTS['ch18_cspfold_peak'], 60) - 0.6319) < 1e-4
    assert pm.threshold(pm.VARIANTS['gu17_cspfold_peak'], 52) == 0.6319
    assert 0.59 < pm.threshold(pm.VARIANTS['or17_120_cspfold_peak'], 120) < 0.60


def _planted(seed, v):
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    r = sy.rec(_dur(ov), seed=seed)
    imag = pm.VARIANTS[v]['imag']
    r = sy.plant(r, ov.onsets_s[ov.labels == pm.RIGHT], ['C3'], pm.CUE_S, pm.CUE_S + imag, depth=0.8)
    return sy.plant(r, ov.onsets_s[ov.labels == pm.LEFT], ['C4'], pm.CUE_S, pm.CUE_S + imag, depth=0.8), ov


def test_planted_effect_is_called():
    for v in ('gu17_cspfold_peak', 'ch18_cspfold_mean57'):
        r, ov = _planted(3, v)
        t = time.time()
        res = pm.run(r, ov, v, np.random.default_rng(0))
        print(f"mbmi {v} planted: acc {res['stat']:.3f} thr {res['extra']['threshold']:.4f} "
              f"({time.time() - t:.2f} s)")
        assert res['call']
    t = time.time()
    rr = pm.run_randomized(r, ov, 'gu17_cspfold_peak', np.random.default_rng(1))
    print(f"mbmi run_randomized planted: p {rr['p']:.4f} ({time.time() - t:.1f} s, {rr['extra']['n_perm_run']} draws)")
    assert rr['call']


def test_noise_call_rate():
    rates, t = {}, time.time()
    for v in ('gu17_cspfold_peak', 'gu17_cspfold_mean57', 'ch18_cspglobal_peak'):
        calls = []
        for s in range(30):
            ov = pm.schedule(10.0, np.random.default_rng(80 + s), v)
            calls.append(pm.run(sy.rec(_dur(ov), seed=80 + s), ov, v, np.random.default_rng(s))['call'])
        rates[v] = float(np.mean(calls))
    print('mbmi noise call rates (30 stationary draws each):', rates, f'{time.time() - t:.0f} s')
    assert rates['gu17_cspfold_mean57'] <= 0.3


def _validity_task(s):
    v = dict(pm.VARIANTS['gu17_cspfold_peak'], n_perm_rand=40)
    ov = pm.schedule(10.0, np.random.default_rng(10 ** 6 + s), v)
    r = sy.rec(_dur(ov), seed=1300 + s, drift=0.5)
    calls = []
    for k in range(10):
        ov.labels = pm.mc.balanced_labels((pm.LEFT, pm.RIGHT), (30, 30), np.random.default_rng(1000 * s + k))
        calls.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'mbmi run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi



def _surviving(seed, v, n_keep, drift=0.0):
    """ch18-timed run where only the first n_keep[class] trials of each class escape an artefact burst at C3."""
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    keep = np.concatenate([np.nonzero(ov.labels == c)[0][:n_keep[c]] for c in (pm.LEFT, pm.RIGHT)])
    bad = np.setdiff1d(np.arange(len(ov.labels)), keep)
    return sy.burst(sy.rec(_dur(ov), seed=seed, drift=drift), ov.onsets_s[bad] + 3.0, 1.0, ['C3']), ov


def test_all_trials_rejected_raises_no_usable_trials():
    v = 'ch18_cspfold_peak'
    r, ov = _surviving(51, v, {pm.LEFT: 0, pm.RIGHT: 0})
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run(r, ov, v, np.random.default_rng(0))
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run_randomized(r, ov, v, np.random.default_rng(0))


def test_exactly_enough_trials_runs_and_one_fewer_raises():
    v = 'ch18_cspfold_peak'
    r, ov = _surviving(52, v, {pm.LEFT: pm.N_FOLD, pm.RIGHT: pm.N_FOLD})
    res = pm.run(r, ov, v, np.random.default_rng(0))
    assert res['extra']['n_kept'] == 2 * pm.N_FOLD
    assert np.isfinite(pm.run_randomized(r, ov, dict(pm.VARIANTS[v], n_perm_rand=20), np.random.default_rng(1))['stat'])
    r2, ov2 = _surviving(52, v, {pm.LEFT: pm.N_FOLD - 1, pm.RIGHT: pm.N_FOLD})
    with pytest.raises(pm.mc.NoUsableTrials, match='left'):
        pm.run(r2, ov2, v, np.random.default_rng(0))


def _heavy_task(s):
    v = dict(pm.VARIANTS['ch18_cspfold_peak'], n_perm_rand=40)
    ov0 = pm.schedule(10.0, np.random.default_rng(10 ** 6 + s), v)
    keep = np.random.default_rng(6 * 10 ** 6 + s).choice(60, 24, replace=False)
    bad = np.setdiff1d(np.arange(60), keep)
    r = sy.burst(sy.rec(_dur(ov0), seed=2400 + s, drift=0.5), ov0.onsets_s[bad] + 3.0, 1.0, ['C3'])

    def one(k):
        ov0.labels = pm.mc.balanced_labels((pm.LEFT, pm.RIGHT), (30, 30), np.random.default_rng(1000 * s + k))
        return pm.run_randomized(r, ov0, v, np.random.default_rng(7 * s + k))['call']
    return sy.usable_calls(one, range(10 ** 4), 10, pm.mc)


def test_heavy_rejection_validity():
    t = time.time()
    calls = sum(sy.pool_map(_heavy_task, range(30)), [])[:200]
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'mbmi heavy rejection (24/60 trials kept): {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi
