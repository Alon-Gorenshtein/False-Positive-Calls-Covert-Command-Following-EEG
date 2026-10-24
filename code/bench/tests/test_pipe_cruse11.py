"""pipe_cruse11 (Cruse 2011 / 2012 Neurology / Goldfine 2013, cat. §1-2): schedule timing, the planted-effect call for
each method, a noise call rate and run_randomized validity. Speed-up (documented): the 200-draw validity test uses 20
noise recordings x 10 re-drawn orders (exact conditional on the data) with B = 40 (exact level 2/41 = 4.9%), run in
a process pool."""
import time

import numpy as np
import pytest
from scipy.stats import binom

from bench import pipe_cruse11 as pm
from bench.tests import _synth as sy


def _dur(ov):
    return float(ov.onsets_s.max() + pm.EPOCH[1] + 20.0)


def test_schedule_timing_and_blocks():
    for v, isi in (('cr11_b4_hilbert_ci99', (4.5, 9.5)), ('cr12n_b6_sq250', (3.0, 6.5))):
        ov = pm.schedule(10.0, np.random.default_rng(0), v)
        nb = pm.VARIANTS[v]['n_blocks']
        blk, lab = ov.meta['block'], ov.labels
        assert len(ov.onsets_s) == 2 * nb * 15 and (np.bincount(blk) == 15).all()
        assert all(len(set(lab[blk == b])) == 1 for b in range(2 * nb))
        assert pm.mc.max_run(ov.meta['block_class']) <= 2
        assert (np.bincount(ov.meta['block_class'])[1:] == nb).all()
        for b in range(2 * nb):
            d = np.diff(ov.onsets_s[blk == b])
            assert d.min() >= isi[0] and d.max() <= isi[1]
        gaps = np.diff(ov.onsets_s)[np.diff(blk) == 1]
        assert gaps.min() >= isi[0] + pm.RELAX_S + 60 + pm.INSTR_S
        for f in np.unique(ov.meta['fold']):
            assert sorted(set(lab[ov.meta['fold'] == f])) == [pm.HAND, pm.TOE]
        assert _dur(ov) - 10.0 <= pm.RUN_S


def test_randomized_schedule_is_balanced_per_tone():
    ov = pm.schedule_randomized(10.0, np.random.default_rng(1), 'cr11_b4_hilbert_ci99')
    assert len(ov.labels) == 120 and (ov.labels == pm.HAND).sum() == 60
    assert any(len(set(ov.labels[ov.meta['block'] == b])) == 2 for b in range(8))
    assert (np.bincount(ov.meta['fold']) == 30).all()


def _planted(seed, v):
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    r = sy.rec(_dur(ov), seed=seed)
    r = sy.plant(r, ov.onsets_s[ov.labels == pm.HAND], ['C3'], 0.5, 3.5, depth=0.8)
    r = sy.plant(r, ov.onsets_s[ov.labels == pm.TOE], ['Cz'], 0.5, 3.5, depth=0.8)
    return r, ov


def test_planted_effect_is_called():
    for v in ('cr11_b4_hilbert_ci99', 'cr12n_b4_sq250', 'gf13_b4_hilbert'):
        r, ov = _planted(3, v)
        t = time.time()
        res = pm.run(r, ov, v, np.random.default_rng(0))
        print(f"cruse11 {v} planted: acc {res['stat']:.3f} p {res['p']:.4g} call {res['call']} "
              f"({time.time() - t:.1f} s)")
        assert res['call']
    assert pm.run_randomized(r, ov, 'cr11_b4_hilbert_ci99', np.random.default_rng(1))['call']


def test_gf13_minimum_p_with_four_blocks_each_is_one_in_35():
    r, ov = _planted(4, 'gf13_b4_hilbert')
    res = pm.run(r, ov, 'gf13_b4_hilbert', np.random.default_rng(0))
    assert res['extra']['n_labelings'] == 70 and res['extra']['exhaustive']
    assert np.isclose(res['p'], 2 / 70)


def test_noise_call_rate():
    rates, t = {}, time.time()
    for v in ('cr11_b4_hilbert_ci99', 'cr12n_b4_hilbert', 'gf13_b4_hilbert'):
        calls = []
        for s in range(8):
            ov = pm.schedule(10.0, np.random.default_rng(60 + s), v)
            calls.append(pm.run(sy.rec(_dur(ov), seed=60 + s), ov, v, np.random.default_rng(s))['call'])
        rates[v] = float(np.mean(calls))
    print('cruse11 noise call rates (8 stationary draws each):', rates, f'{time.time() - t:.0f} s')
    assert max(rates.values()) <= 0.5


def _validity_task(s):
    v = dict(pm.VARIANTS['cr11_b4_hilbert_ci99'], n_perm_rand=40)
    ov = pm.schedule_randomized(10.0, np.random.default_rng(10 ** 6 + s), v)
    r = sy.rec(_dur(ov), seed=900 + s, drift=0.5)
    calls = []
    for k in range(10):
        ov.labels = pm.mc.balanced_labels((pm.HAND, pm.TOE), (60, 60), np.random.default_rng(1000 * s + k))
        calls.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'cruse11 run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi



def _surviving(seed, v, keep_fn):
    """Recording where only the trials keep_fn(overlay) selects escape an artefact burst at C3."""
    ov = pm.schedule(10.0, np.random.default_rng(seed), v)
    bad = np.setdiff1d(np.arange(len(ov.labels)), keep_fn(ov))
    return sy.burst(sy.rec(_dur(ov), seed=seed), ov.onsets_s[bad] + 1.0, 1.0, ['C3']), ov


def _first(ov, cls, fold, k):
    return list(np.nonzero((ov.labels == cls) & (ov.meta['fold'] == fold))[0][:k])


def test_all_trials_rejected_raises_no_usable_trials():
    for v in ('cr11_b4_hilbert_ci99', 'cr12n_b4_sq250', 'gf13_b4_hilbert'):
        r, ov = _surviving(31, v, lambda o: [])
        with pytest.raises(pm.mc.NoUsableTrials):
            pm.run(r, ov, v, np.random.default_rng(0))
    ovr = pm.schedule_randomized(10.0, np.random.default_rng(32), 'cr11_b4_hilbert_ci99')
    rr = sy.burst(sy.rec(_dur(ovr), seed=32), ovr.onsets_s + 1.0, 1.0, ['C3'])
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run_randomized(rr, ovr, 'cr11_b4_hilbert_ci99', np.random.default_rng(0))


def test_exactly_enough_trials_runs_and_one_fewer_raises():
    v = 'cr11_b4_hilbert_ci99'
    r, ov = _surviving(33, v, lambda o: _first(o, pm.HAND, 0, 2) + _first(o, pm.TOE, 1, 2))
    res = pm.run(r, ov, v, np.random.default_rng(0))
    assert res['extra']['n_kept'] == 4 and res['extra']['n_tested'] == 4
    r1, ov1 = _surviving(33, v, lambda o: _first(o, pm.HAND, 0, 1) + _first(o, pm.TOE, 1, 2))
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run(r1, ov1, v, np.random.default_rng(0))
    r2, ov2 = _surviving(33, v, lambda o: _first(o, pm.HAND, 0, 2) + _first(o, pm.TOE, 0, 2))
    with pytest.raises(pm.mc.NoUsableTrials, match='fold'):
        pm.run(r2, ov2, v, np.random.default_rng(0))
    g = 'gf13_b4_hilbert'
    blocks = lambda o, cls, k: [int(np.nonzero((o.labels == cls) & (o.meta['block'] == b))[0][0])
                                for b in np.unique(o.meta['block'][o.labels == cls])[:k]]
    r3, ov3 = _surviving(34, g, lambda o: blocks(o, pm.HAND, 2) + blocks(o, pm.TOE, 2))
    res = pm.run(r3, ov3, g, np.random.default_rng(0))
    assert res['extra']['n_kept'] == 4 and res['extra']['n_labelings'] == 6
    r4, ov4 = _surviving(34, g, lambda o: blocks(o, pm.HAND, 2) + blocks(o, pm.TOE, 1))
    with pytest.raises(pm.mc.NoUsableTrials, match='block'):
        pm.run(r4, ov4, g, np.random.default_rng(0))



N_HEAVY = 20


def _heavy_task(s):
    v = dict(pm.VARIANTS['cr11_b4_hilbert_ci99'], n_perm_rand=40)
    ov0 = pm.schedule_randomized(10.0, np.random.default_rng(10 ** 6 + s), v)
    keep = np.random.default_rng(5 * 10 ** 6 + s).choice(len(ov0.labels), N_HEAVY, replace=False)
    bad = np.setdiff1d(np.arange(len(ov0.labels)), keep)
    r = sy.burst(sy.rec(_dur(ov0), seed=2300 + s, drift=0.5), ov0.onsets_s[bad] + 1.0, 1.0, ['C3'])

    def one(k):
        ov0.labels = pm.mc.balanced_labels((pm.HAND, pm.TOE), (60, 60), np.random.default_rng(1000 * s + k))
        return pm.run_randomized(r, ov0, v, np.random.default_rng(7 * s + k))['call']
    return sy.usable_calls(one, range(10 ** 4), 10, pm.mc)


def test_heavy_rejection_validity():
    t = time.time()
    calls = sum(sy.pool_map(_heavy_task, range(30)), [])[:200]
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'cruse11 heavy rejection ({N_HEAVY}/120 trials kept): {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi
