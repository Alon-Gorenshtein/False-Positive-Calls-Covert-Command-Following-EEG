"""pipe_edlow17 (Edlow 2017, cat. §5): schedule timing, the batched SVM against liblinear, the planted-effect call, a
noise call rate and run_randomized validity. Speed-ups (documented): the noise test lowers the permutation count to
100; the 200-draw validity test uses 20 noise recordings x 10 re-drawn block orders (exact conditional on the data),
B = 20 (exact level 1/21 = 4.8%) and a process pool."""
import time

import numpy as np
import pytest
from scipy.stats import binom
from sklearn.svm import LinearSVC

from bench import pipe_edlow17 as pm
from bench.tests import _synth as sy


def test_schedule_timing():
    ov = pm.schedule(10.0, np.random.default_rng(0), 'permsecond_cvsecond_alt')
    assert len(ov.onsets_s) == 96 and np.allclose(np.diff(ov.onsets_s), 6.0)
    assert np.array_equal(ov.meta['block_class'], np.tile([0, 1], 12))
    assert (np.bincount(ov.meta['block']) == 4).all()
    r = pm.schedule(10.0, np.random.default_rng(0), 'permblock_cvblock_random')
    assert r.meta['block_class'].sum() == 12
    assert ov.onsets_s[-1] + 6.0 - 10.0 == 576.0 <= pm.RUN_S


def test_features_drop_instruction_seconds():
    ov = pm.schedule(10.0, None, 'permsecond_cvsecond_alt')
    rec = sy.rec(600, seed=1)
    X, blocks, keep = pm.features(rec, ov)
    assert X.shape == (384, 76) and keep.sum() == 384
    assert (np.bincount(blocks) == 16).all()


def test_batched_svm_matches_liblinear():
    g = np.random.default_rng(2)
    X = g.standard_normal((300, 20))
    s = np.where(g.random(300) < 0.5, 1.0, -1.0)
    masks = np.ones((5, 300), bool)
    for f in range(5):
        masks[f, f * 60:(f + 1) * 60] = False
    W = pm.l2svm_fit(np.c_[X, np.ones(300)], s, masks)
    for f in range(5):
        m = LinearSVC(C=1.0, dual=False, tol=1e-12, max_iter=100000).fit(X[masks[f]], s[masks[f]])
        assert np.allclose(W[f], np.r_[m.coef_[0], m.intercept_], atol=1e-5)


def _planted(seed, v='permblock_cvblock_alt'):
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    r = sy.rec(600, seed=seed)
    sq = ov.onsets_s[ov.meta['block_start'] & (ov.labels == pm.SQUEEZE)]
    return sy.plant(r, sq, ['C3', 'C4'], 0.5, 23.5, depth=0.7), ov


def test_planted_effect_is_called():
    r, ov = _planted(3)
    t = time.time()
    res = pm.run(r, ov, 'permblock_cvblock_alt', np.random.default_rng(0))
    print(f"edlow17 planted: acc {res['stat']:.3f} p {res['p']:.4f} ({time.time() - t:.1f} s, "
          f"{res['extra']['n_perm_run']} perms)")
    assert res['call']
    assert pm.run_randomized(r, ov, 'permblock_cvblock_alt', np.random.default_rng(1))['call']


def test_noise_call_rate():
    rates, t = {}, time.time()
    for v in ('permsecond_cvsecond_alt', 'permblock_cvblock_alt'):
        calls = []
        for s in range(6):
            ov = pm.schedule(10.0, np.random.default_rng(s), v)
            vv = dict(pm.VARIANTS[v], n_perm=100)
            calls.append(pm.run(sy.rec(600, seed=70 + s, drift=0.5), ov, vv, np.random.default_rng(s))['call'])
        rates[v] = float(np.mean(calls))
    print('edlow17 noise call rates (6 drifting draws each, 100 perms):', rates, f'{time.time() - t:.0f} s')
    assert rates['permblock_cvblock_alt'] <= 0.5


def _validity_task(s):
    v = dict(pm.VARIANTS['permsecond_cvsecond_alt'], n_perm_rand=20)
    r = sy.rec(600, seed=1100 + s, drift=0.5)
    calls = []
    for k in range(10):
        ov = pm.schedule_randomized(10.0, np.random.default_rng(1000 * s + k), v)
        calls.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'edlow17 run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi



def _blocks_kept(seed, v, kept_blocks, drift=0.0):
    """Recording where every second of the blocks not in kept_blocks carries an artefact burst at C3."""
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    starts = ov.onsets_s[ov.meta['block_start']]
    bad = [starts[b] for b in range(pm.N_BLOCKS) if b not in set(kept_blocks)]
    return sy.burst(sy.rec(600, seed=seed, drift=drift), np.array(bad), float(pm.BLOCK_S), ['C3']), ov


def test_all_seconds_rejected_raises_no_usable_trials():
    v = 'permblock_cvblock_alt'
    r, ov = _blocks_kept(61, v, [])
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run(r, ov, v, np.random.default_rng(0))
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run_randomized(r, ov, v, np.random.default_rng(0))


def test_exactly_enough_blocks_runs_and_one_fewer_raises():
    v = dict(pm.VARIANTS['permblock_cvblock_alt'], n_perm=20)
    r, ov = _blocks_kept(62, v, [0, 2, 1, 3])
    res = pm.run(r, ov, v, np.random.default_rng(0))
    assert res['extra']['n_cols'] == 4 * 16
    r2, ov2 = _blocks_kept(62, v, [0, 1, 3])
    with pytest.raises(pm.mc.NoUsableTrials, match='rest'):
        pm.run(r2, ov2, v, np.random.default_rng(0))


def test_column_minimum():
    blocks = np.repeat([0, 1, 2, 3], 5)
    pm.require(np.array([0, 1, 0, 1] + [0, 1] * 10), blocks)
    with pytest.raises(pm.mc.NoUsableTrials, match='column'):
        pm.require(np.array([0, 1, 0, 1] + [0, 1] * 10), blocks[:-1])


def _heavy_task(s):
    v = dict(pm.VARIANTS['permsecond_cvsecond_alt'], n_perm_rand=20)
    kept = np.random.default_rng(7 * 10 ** 6 + s).choice(pm.N_BLOCKS, 5, replace=False)
    r, ov0 = _blocks_kept(2500 + s, v, list(kept), drift=0.5)

    def one(k):
        ov = pm.schedule_randomized(10.0, np.random.default_rng(1000 * s + k), v)
        return pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call']
    return sy.usable_calls(one, range(10 ** 4), 10, pm.mc)


def test_heavy_rejection_validity():
    t = time.time()
    calls = sum(sy.pool_map(_heavy_task, range(30)), [])[:200]
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'edlow17 heavy rejection (5/24 blocks kept): {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi
