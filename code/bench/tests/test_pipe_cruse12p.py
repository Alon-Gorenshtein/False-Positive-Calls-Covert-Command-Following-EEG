"""pipe_cruse12p (Cruse 2012 PLoS One, cat. §3): schedule timing, the planted-effect call, a noise call rate and
run_randomized validity. Speed-up (documented): the 200-draw validity test uses 20 noise recordings x 10 re-drawn
orders (exact conditional on the data) and B = 40 (exact level 2/41 = 4.9%)."""
import time

import numpy as np
import pytest
from scipy.stats import binom

from bench import pipe_cruse12p as pm
from bench.tests import _synth as sy


def test_schedule_timing_and_order():
    ov = pm.schedule(10.0, np.random.default_rng(0))
    on, lab, blk = ov.onsets_s, ov.labels, ov.meta['block']
    assert len(on) == 180 and on[0] == 10.0
    for b in range(5):
        lb = lab[blk == b]
        assert len(lb) == 36 and all((lb == c).sum() == 12 for c in (pm.RIGHT, pm.LEFT, pm.RELAX))
        assert pm.mc.max_run(lb) <= 3
        d = np.diff(on[blk == b])
        assert d.min() >= 6.0 and d.max() <= 9.0
    gaps = on[1:] - on[:-1]
    assert np.sum(gaps > 60.0) == 4
    assert on[-1] + pm.EPOCH_S - on[0] <= pm.RUN_S


def test_features_shape_and_time_axis():
    r = sy.rec(400, seed=1)
    ov = pm.schedule(10.0, np.random.default_rng(1))
    ov.onsets_s, ov.labels = ov.onsets_s[:30], ov.labels[:30]
    F, keep, which = pm.features(r, ov)
    assert F.shape == (30, 100, 8) and which == '1020' and keep.all()
    assert (pm.N_T - 1) * pm.STEP + pm.WIN == 595


def _planted(seed):
    ov = pm.schedule(10.0, np.random.default_rng(seed))
    r = sy.rec(pm.RUN_S + 20, seed=seed)
    r = sy.plant(r, ov.onsets_s[ov.labels == pm.RIGHT], ['F3'], 2.5, 5.5, depth=0.8)
    r = sy.plant(r, ov.onsets_s[ov.labels == pm.LEFT], ['F4'], 2.5, 5.5, depth=0.8)
    return r, ov


def test_planted_effect_is_called():
    r, ov = _planted(2)
    t = time.time()
    res = pm.run(r, ov, 'random_perclass', np.random.default_rng(0))
    print(f"cruse12p planted: acc {res['stat']:.3f} p {res['p']:.4f} ({time.time() - t:.1f} s)")
    assert res['call']
    rr = pm.run_randomized(r, ov, 'random_perclass', np.random.default_rng(1))
    assert rr['call']


def test_noise_call_rate_all_variants():
    rates = {}
    t = time.time()
    for v in pm.VARIANTS:
        calls = []
        for s in range(10):
            r = sy.rec(pm.RUN_S + 20, seed=50 + s)
            ov = pm.schedule(10.0, np.random.default_rng(50 + s))
            calls.append(pm.run(r, ov, v, np.random.default_rng(s))['call'])
        rates[v] = np.mean(calls)
    print('cruse12p noise call rates (10 stationary draws each):', rates, f'{time.time() - t:.0f} s')
    assert max(rates.values()) <= 0.3


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = []
    v = dict(pm.VARIANTS['random_perclass'], n_perm_rand=40)
    for s in range(20):
        r = sy.rec(pm.RUN_S + 20, seed=300 + s, drift=0.5)
        for k in range(10):
            ov = pm.schedule(10.0, np.random.default_rng(1000 * s + k))
            calls.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'cruse12p run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert lo <= n <= hi



def _surviving(seed, n_keep):
    """Recording where only the first n_keep[class] trials of each class escape an artefact burst at F3/F4."""
    ov = pm.schedule(10.0, np.random.default_rng(seed))
    keep = np.concatenate([np.nonzero(ov.labels == c)[0][:n_keep[c]] for c in (pm.RIGHT, pm.LEFT, pm.RELAX)])
    bad = np.setdiff1d(np.arange(len(ov.labels)), keep)
    return sy.burst(sy.rec(pm.RUN_S + 20, seed=seed), ov.onsets_s[bad] + 2.0, 1.0, ['F3', 'F4']), ov


def test_all_trials_rejected_raises_no_usable_trials():
    r, ov = _surviving(21, {pm.RIGHT: 0, pm.LEFT: 0, pm.RELAX: 0})
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run(r, ov, 'random_perclass', np.random.default_rng(0))
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run_randomized(r, ov, 'random_perclass', np.random.default_rng(0))


def test_exactly_enough_trials_runs_and_one_fewer_raises():
    r, ov = _surviving(22, {pm.RIGHT: pm.MIN_PER_CLASS, pm.LEFT: pm.MIN_PER_CLASS, pm.RELAX: pm.MIN_PER_CLASS})
    for v in pm.VARIANTS:
        res = pm.run(r, ov, v, np.random.default_rng(0))
        assert res['extra']['n_kept'] == 3 * pm.MIN_PER_CLASS and np.isfinite(res['stat'])
        assert res['extra']['n_nonfinite_null'] == 0
    rr = pm.run_randomized(r, ov, dict(pm.VARIANTS['random_perclass'], n_perm_rand=20), np.random.default_rng(1))
    assert np.isfinite(rr['stat'])
    r2, ov2 = _surviving(23, {pm.RIGHT: pm.MIN_PER_CLASS, pm.LEFT: pm.MIN_PER_CLASS - 1, pm.RELAX: pm.MIN_PER_CLASS})
    with pytest.raises(pm.mc.NoUsableTrials, match='left'):
        pm.run(r2, ov2, 'random_perclass', np.random.default_rng(0))



_ORIG = pm.mc.admissible_stat
N_HEAVY = 10


def _heavy_task(args):
    s, broken = args
    tally = {'draws': 0, 'rejected': 0}
    pm.mc.admissible_stat = sy.zero_fill if broken else sy.counting(_ORIG, tally)
    ov0 = pm.schedule(10.0, np.random.default_rng(10 ** 6 + s))
    keep = np.random.default_rng(3 * 10 ** 6 + s).choice(len(ov0.labels), N_HEAVY, replace=False)
    bad = np.setdiff1d(np.arange(len(ov0.labels)), keep)
    r = sy.burst(sy.rec(pm.RUN_S + 20, seed=2100 + s, drift=0.5), ov0.onsets_s[bad] + 2.0, 1.0, ['F3', 'F4'])
    v = dict(pm.VARIANTS['random_perclass'], n_perm_rand=40)

    def one(k):
        g = np.random.default_rng(1000 * s + k)
        lab = np.concatenate([pm.mc.constrained_order((pm.RIGHT, pm.LEFT, pm.RELAX), pm.PER_CLASS, 3, g)
                              for _ in range(pm.N_BLOCKS)])
        ov = pm.mc.Overlay(onsets_s=ov0.onsets_s, labels=lab, kind='cmd', meta=dict(ov0.meta))
        return pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call']
    return sy.usable_calls(one, range(10 ** 4), 10, pm.mc), tally


def test_heavy_rejection_validity_and_the_zero_fill_bias():
    t = time.time()
    res = sy.pool_map(_heavy_task, [(s, False) for s in range(30)])
    calls = sum([c for c, _ in res], [])[:200]
    rej = sum(tl['rejected'] for _, tl in res) / max(1, sum(tl['draws'] for _, tl in res))
    broken = sum([c for c, _ in sy.pool_map(_heavy_task, [(s, True) for s in range(30)])], [])[:200]
    nz = int(np.sum(broken))
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'cruse12p heavy rejection: {n}/200 false calls with re-drawing ({rej:.0%} of null draws re-drawn), '
          f'{nz}/200 with zero-filling ({time.time() - t:.0f} s)')
    assert len(calls) == len(broken) == 200 and lo <= n <= hi and rej > 0.3
    assert nz > hi
