"""pipe_cornell (Goldfine 2011 / Curley 2018, cat. §4): schedule timing, the Two Group Test under its own null, the
contiguity rule, the planted-effect call, a noise call rate and run_randomized validity. Speed-up (documented): the
200-draw validity test uses 20 noise recordings x 10 re-drawn flips (exact conditional on the data), B = 40 (exact
level 2/41 = 4.9%) and a process pool."""
import time

import numpy as np
import pytest
from scipy.stats import binom

from bench import pipe_cornell as pm
from bench.tests import _synth as sy


def test_schedule_timing():
    for nr in (2, 4):
        v = f'theoretical_bh_keep40_r{nr}'
        ov = pm.schedule(10.0, None, v)
        assert len(ov.onsets_s) == 16 * nr
        assert (ov.labels[0::2] == pm.TASK).all() and (ov.labels[1::2] == pm.REST).all()
        for r in range(nr):
            on = ov.onsets_s[ov.meta['run'] == r]
            assert np.allclose(np.diff(on), 15.0) and on[-1] + 15.0 - on[0] == 240.0
        assert ov.onsets_s[-1] + pm.CMD_S + pm.RESP[1] - 10.0 <= pm.RUN_S
    rv = pm.schedule_randomized(10.0, np.random.default_rng(1), 'theoretical_bh_keep40_r4')
    pairs = rv.labels.reshape(-1, 2)
    assert (pairs.sum(1) == 1).all() and 6 < rv.meta['flipped'].sum() < 26


def test_tgt_is_standard_normal_under_its_model():
    g = np.random.default_rng(0)
    dz, vdz = [], []
    for _ in range(400):
        S1 = g.exponential(1.0, (24, 5, 1, 3))
        S2 = g.exponential(1.0, (24, 5, 1, 3))
        a, b = pm.tgt(S1, S2)
        dz.append(a.ravel())
        vdz.append(b.ravel())
    dz, vdz = np.concatenate(dz), np.concatenate(vdz)
    assert abs(dz.mean()) < 0.1 and 0.85 < dz.var() < 1.15 and 0.85 < vdz.mean() < 1.15


def test_regions_and_contiguity():
    f = np.arange(61) / 3 + 4
    sig = np.zeros((1, 61), bool)
    sig[0, 3:10] = True
    sig[0, 20:29] = True
    reg = pm.regions(sig, np.ones((1, 61)), f)
    assert [(r[1], r[2]) for r in reg] == [(3, 9), (20, 28)]
    assert reg[0][4] <= pm.MIN_RANGE_HZ < reg[1][4]


def _planted(seed, v):
    ov = pm.schedule(10.0, None, v)
    r = sy.rec(ov.onsets_s.max() + 40.0, seed=seed)
    on = ov.onsets_s[ov.labels == pm.TASK] + pm.CMD_S
    return sy.plant(r, on, ['C3', 'C4'], 0.5, 10.5, depth=0.6), ov


def test_planted_effect_is_called():
    for v in ('theoretical_bh_keep40_r4', 'jackknife_by_thresh_r2'):
        r, ov = _planted(5, v)
        t = time.time()
        res = pm.run(r, ov, v, np.random.default_rng(0))
        print(f"cornell {v} planted: zmax {res['stat']:.2f} q {res['p']:.3g} om1 {res['extra']['om1']} "
              f"om2 {res['extra']['om2']} ({time.time() - t:.1f} s)")
        assert res['call']
    t = time.time()
    rr = pm.run_randomized(r, ov, 'jackknife_by_thresh_r2', np.random.default_rng(1))
    print(f"cornell run_randomized planted: p {rr['p']:.4f} ({time.time() - t:.1f} s)")
    assert rr['call']


def test_noise_call_rate():
    rates, t = {}, time.time()
    for v in ('theoretical_bh_keep40_r4', 'jackknife_bh_keep40_r4', 'theoretical_by_thresh_r2'):
        calls, ind, reg = [], [], []
        for s in range(12):
            ov = pm.schedule(10.0, None, v)
            res = pm.run(sy.rec(ov.onsets_s.max() + 40.0, seed=90 + s), ov, v, np.random.default_rng(s))
            calls.append(res['call'])
            ind.append(res['extra']['indeterminate'])
            reg.append(res['extra']['call_regrouped'])
        rates[v] = (float(np.mean(calls)), float(np.mean(ind)), float(np.mean(reg)))
    print('cornell noise (positive, indeterminate, regrouped) over 12 stationary draws:', rates,
          f'{time.time() - t:.0f} s')
    assert all(0.0 <= r[0] <= r[2] <= 1.0 for r in rates.values())


def _validity_task(s):
    v = dict(pm.VARIANTS['jackknife_bh_keep40_r4'], n_perm_rand=40)
    ov = pm.schedule(10.0, None, v)
    r = sy.rec(ov.onsets_s.max() + 40.0, seed=1500 + s, drift=0.5)
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
    print(f'cornell run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi



def _surviving(seed, v, keep):
    """Recording where only the (run, command-within-run, epoch-within-command) triples in `keep` escape a saturating
    (600 µV) artefact burst at C3 in the middle of the epoch."""
    ov = pm.schedule(10.0, None, v)
    t = []
    for c, (on, run) in enumerate(zip(ov.onsets_s, ov.meta['run'])):
        for e in range(pm.N_EP):
            if (int(run), c % 16, e) not in keep:
                t.append(on + pm.CMD_S + pm.RESP[0] + e * pm.EPOCH_S + 1.0)
    return sy.burst(sy.rec(ov.onsets_s.max() + 40.0, seed=seed), np.array(t), 1.0, ['C3'], amp=600.0), ov


def _enough(runs, n_task=2, n_rest=2):
    """n_task epochs of the first task command and n_rest of the first rest command in each run."""
    return {(r, 0, e) for r in runs for e in range(n_task)} | {(r, 1, e) for r in runs for e in range(n_rest)}


def test_all_epochs_rejected_raises_no_usable_trials():
    v = 'theoretical_bh_thresh_r4'
    r, ov = _surviving(41, v, set())
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run(r, ov, v, np.random.default_rng(0))
    with pytest.raises(pm.mc.NoUsableTrials):
        pm.run_randomized(r, ov, v, np.random.default_rng(0))


def test_exactly_enough_epochs_runs_and_one_fewer_raises():
    v = 'jackknife_bh_keep40_r4'
    r, ov = _surviving(42, v, _enough((0, 1)))
    res = pm.run(r, ov, v, np.random.default_rng(0))
    assert res['extra']['n_kept'] == 8 and np.isfinite(res['stat'])
    rr = pm.run_randomized(r, ov, dict(pm.VARIANTS[v], n_perm_rand=20), np.random.default_rng(1))
    assert np.isfinite(rr['stat']) and rr['extra']['n_nonfinite_null'] == 0
    r2, ov2 = _surviving(42, v, _enough((0,)) | _enough((1,), n_rest=1))
    with pytest.raises(pm.mc.NoUsableTrials, match='runs'):
        pm.run(r2, ov2, v, np.random.default_rng(0))


def test_an_emptied_run_no_longer_breaks_the_regrouping():
    v = 'theoretical_bh_thresh_r4'
    r, ov = _surviving(43, v, _enough((0, 1, 2), 3, 3))
    res = pm.run(r, ov, v, np.random.default_rng(0))
    assert res['extra']['regroup'] is not None and len(res['extra']['regroup']) == 3


def test_keep40_never_keeps_flat_or_saturated_epochs():
    v = 'theoretical_bh_keep40_r2'
    ov = pm.schedule(10.0, None, v)
    r = sy.rec(ov.onsets_s.max() + 40.0, seed=44)
    x = np.asarray(r.x).copy()
    t0 = ov.onsets_s[0] + pm.CMD_S + pm.RESP[0]
    x[:, int(t0 * r.fs):int((t0 + 9.0) * r.fs)] = 0.0
    r = sy.FakeRec(r.source, r.uid + '+flat', r.patient, x, r.fs, r.names, {})
    F = pm.features(r, ov)
    lab = np.asarray(ov.labels)[F['cmd']]
    keep = pm.kept(F, lab, 'keep40')
    assert not keep[F['cmd'] == 0].any()
    assert keep[(F['run'] == 0) & (lab == pm.TASK)].sum() == 10



_ORIG = pm.mc.admissible_stat


def _heavy_task(s):
    """Each run keeps the 3 epochs of 2 commands from different trials (chosen independently of the labels; 56 of 64
    commands carry a saturating burst). A run is usable only if its 2 kept commands differ, so some flip sets fall
    below the minimum and are re-drawn. (A heavier gate leaves too few distinct statistic values for a 5% test: the
    zero-fill bias is shown in test_pipe_cruse12p.)"""
    tally = {'draws': 0, 'rejected': 0}
    pm.mc.admissible_stat = sy.counting(_ORIG, tally)
    v = dict(pm.VARIANTS['theoretical_bh_thresh_r4'], n_perm_rand=40)
    g = np.random.default_rng(4 * 10 ** 6 + s)
    keep = set()
    for run in range(4):
        for tr in g.choice(pm.TRIALS_PER_RUN, 2, replace=False):
            keep |= {(run, 2 * int(tr) + int(g.integers(2)), e) for e in range(pm.N_EP)}
    r, _ = _surviving(2200 + s, v, keep)

    def one(k):
        ov = pm.schedule_randomized(10.0, np.random.default_rng(1000 * s + k), v)
        return pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call']
    return sy.usable_calls(one, range(10 ** 4), 10, pm.mc), tally


def test_heavy_rejection_validity():
    t = time.time()
    res = sy.pool_map(_heavy_task, range(30))
    calls = sum([c for c, _ in res], [])[:200]
    rej = sum(tl['rejected'] for _, tl in res) / max(1, sum(tl['draws'] for _, tl in res))
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'cornell heavy rejection: {n}/200 false calls ({rej:.0%} of null draws re-drawn, {time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi and rej > 0.03
