"""pipe_mbvt (cat. §4): schedule timing per paradigm and timing variant; the derived binomial thresholds; the
accuracy curve's granularity; a planted P3 is called; noise call rates (the per-run accuracy tail); the patient
rules; run_randomized validity on 200 noise draws (20 recordings x 10 re-drawn randomizations, B = 40)."""
import time

import numpy as np
from scipy.stats import binom

from bench import engine as en
from bench import pipe_mbvt as pm
from bench.tests import _synth as sy


def test_schedule_timing():
    for v, soa, pause, run in (('vt3_mur20_seg8_full_gt23', 0.2, 13.5, 150.0),
                               ('vt2_ann18_ds7_full_gt23', 0.3, 0.0, 144.0)):
        ov = pm.schedule(0.0, np.random.default_rng(1), v)
        assert len(ov.onsets_s) == 480 and ov.kind == 'stim'
        grp, site, lab = ov.meta['group'], ov.meta['site'], ov.labels
        for gi in range(pm.N_GROUPS):
            m = grp == gi
            assert np.sum(lab[m] == pm.TARGET) == 1
            if v.startswith('vt3'):
                assert sorted(site[m]) == [pm.OTHER] * 6 + [pm.LEFT, pm.RIGHT]
            else:
                assert sorted(site[m]) == [pm.OTHER] * 0 + [pm.LEFT] * 7 + [pm.RIGHT]
        d = np.diff(ov.onsets_s)
        assert np.allclose(np.delete(d, np.nonzero(d > soa + 1e-6)[0]), soa)
        big = d[d > soa + 1e-6]
        assert len(big) == (3 if pause else 0) and np.allclose(big, soa + pause)
        span = ov.onsets_s.max() - ov.onsets_s.min() + soa + pause
        assert abs(span - run) < 1e-6 and ov.onsets_s.max() + 0.6 <= pm.run_s(v)
    ov = pm.schedule(0.0, np.random.default_rng(2), 'vt3_mur20_seg8_full_gt23')
    tw = ov.meta['target_wrist']
    assert (tw.reshape(4, 15) == tw.reshape(4, 15)[:, :1]).all() and np.sum(tw[::15] == pm.LEFT) == 2
    r = pm.schedule_randomized(0.0, np.random.default_rng(2), 'vt3_mur20_seg8_full_gt23')
    assert np.sum(r.meta['target_wrist'] == pm.LEFT) == 30


def test_thresholds():
    t = {k: v[1] for k, v in pm.THRESHOLDS.items()}
    assert t['binom_n10'] == 0.4 and t['binom_n20'] == 0.3
    assert abs(t['binom_n30'] - 8 / 30) < 1e-12 and abs(t['binom_n60'] - 13 / 60) < 1e-12
    assert len(pm.VARIANTS) == 120


def _planted(seed, v, effect, randomized=False):
    ov = (pm.schedule_randomized if randomized else pm.schedule)(0.0, np.random.default_rng(seed), v)
    r = sy.rec(pm.run_s(v), seed=seed)
    x = en.inject(r, ov, ov.labels == pm.TARGET, 'erp', effect, np.random.default_rng(seed + 1))
    return sy.FakeRec('synthetic', f'p3-{seed}', 'p', x, r.fs, list(r.names), {}), ov


def test_planted_p3_is_called():
    for v in ('vt3_mur20_seg8_full_gt23', 'vt2_ann18_ds7_median_binom_n10'):
        r, ov = _planted(3, v, 8.0)
        res = pm.run(r, ov, v, np.random.default_rng(0))
        c = res['extra']['curve']
        assert np.allclose(c * 10, np.round(c * 10))
        print(f"mbvt {v} planted 8 µV: acc {res['stat']:.2f} (curve {np.round(c[:5], 1)})")
        assert res['call']
        r, ov = _planted(4, v, 8.0, randomized=True)
        rr = pm.run_randomized(r, ov, v, np.random.default_rng(1))
        print(f"mbvt {v} run_randomized planted: p {rr['p']:.4f}")
        assert rr['call']


def test_noise_call_rate():
    vs = [f'vt3_mur20_seg8_{s}_{t}' for s in pm.SUMMARIES for t in ('gt23', 'binom_n10')] + \
         ['vt2_ann18_ds7_full_gt23', 'vt2_ann18_ds7_max_gt23']
    calls, accs, t = {v: [] for v in vs}, [], time.time()
    for s in range(40):
        for v in vs:
            ov = pm.schedule(0.0, np.random.default_rng(100 + s), v)
            r = sy.rec(pm.run_s(v), seed=100 + s)
            res = pm.run(r, ov, v, np.random.default_rng(s))
            calls[v].append(res['call'])
            if v == 'vt3_mur20_seg8_full_gt23':
                accs.append(res['stat'])
    rates = {v: float(np.mean(c)) for v, c in calls.items()}
    print('mbvt noise call rates (40 draws per variant):', rates, f'{time.time() - t:.0f} s')
    print('mbvt null accuracy (vt3 full) quantiles 50/90/95/99%:', np.quantile(accs, [0.5, 0.9, 0.95, 0.99]))
    assert rates['vt3_mur20_seg8_max_gt23'] >= rates['vt3_mur20_seg8_full_gt23']


def test_patient_rules():
    def r(par, acc, thr=0.23):
        return {'call': acc > thr, 'extra': {'paradigm': par, 'acc': acc}}
    seq = [r('vt2', 0.2), r('vt3', 0.3), r('vt3', 0.1)]
    assert pm.patient_rule(seq) and pm.patient_rule(seq, 'best_run') and not pm.patient_rule(seq, 'best60')
    assert not pm.patient_rule(seq, 'ann18_gate70')
    seq = [r('vt2', 0.75), r('vt3', 0.3)]
    assert pm.patient_rule(seq, 'ann18_gate_thr') and not pm.patient_rule(seq, 'ann18_gate70')
    assert pm.patient_rule([r('vt2', 0.8), r('vt3', 0.8)], 'ann18_gate70')
    assert pm.patient_rule([r('vt3', 0.6)], 'best60')


def _validity_task(s):
    out = []
    for v0, rnd in (('vt3_mur20_seg8_full_gt23', 'vt3'), ('vt2_mur20_ds7_full_gt23', 'vt2')):
        v = dict(pm.VARIANTS[v0], n_perm_rand=40)
        ov = pm.schedule_randomized(0.0, np.random.default_rng(10 ** 6 + s), v)
        r = sy.rec(pm.run_s(v0), seed=5300 + s, drift=0.5)
        for k in range(5):
            g = np.random.default_rng(1000 * s + k)
            grp, site = ov.meta['group'], ov.meta['site']
            if rnd == 'vt3':
                tw = pm.mc.balanced_labels((pm.LEFT, pm.RIGHT), (30, 30), g)
                ov.meta['target_wrist'] = tw
                ov.labels = np.where(site == pm.OTHER, pm.DISTRACTOR,
                                     np.where(site == tw[grp], pm.TARGET, pm.NONTARGET))
            else:
                tp = g.integers(0, 8, pm.N_GROUPS)
                ov.labels = np.where(ov.meta['pos'] == tp[grp], pm.TARGET, pm.NONTARGET)
            out.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return out


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'mbvt run_randomized: {n}/200 false calls (VT3 100 + VT2 100; {time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi


def test_seconds_per_assessment():
    v = pm.DEFAULT
    ov = pm.schedule(0.0, np.random.default_rng(9), v)
    r = sy.rec(pm.run_s(v), seed=9)
    t0 = time.time()
    pm.run(r, ov, v, np.random.default_rng(0))
    t1 = time.time()
    ovr = pm.schedule_randomized(0.0, np.random.default_rng(9), v)
    rr = sy.rec(pm.run_s(v), seed=10)
    t2 = time.time()
    res = pm.run_randomized(rr, ovr, v, np.random.default_rng(0))
    t3 = time.time()
    print(f'mbvt seconds per assessment ({pm.run_s(v):.0f}-s run, 7 ch, 128 Hz): run {t1 - t0:.2f} s, '
          f"run_randomized {t3 - t2:.2f} s ({res['extra']['n_perm_run']} draws; features included)")


def _truncate(ov, G):
    """The first G groups of an overlay (real passive overlays carry 49-60 groups)."""
    keep = ov.meta['group'] < G
    meta = {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
            for k, v in ov.meta.items()}
    meta['target_wrist'] = np.asarray(ov.meta['target_wrist'])[:G]
    return pm.mc.Overlay(onsets_s=ov.onsets_s[keep], labels=ov.labels[keep], kind=ov.kind, meta=meta)


def test_run_randomized_uses_the_overlay_group_count():
    v = 'vt3_mur20_seg8_full_gt23'
    ov = _truncate(pm.schedule_randomized(0.0, np.random.default_rng(6), v), 52)
    assert len(ov.meta['target_wrist']) == 52 and ov.meta['group'].max() == 51
    r = sy.rec(pm.run_s(v), seed=6)
    F = pm.features(r, ov, 'seg8')
    n_left = int(np.sum(ov.meta['target_wrist'] == pm.LEFT))
    t = pm._redraw(F, pm.mc.cfg(pm.VARIANTS, v), np.random.default_rng(1), ov)
    assert len(t) == len(F['E']) == 52
    assert np.sum(F['site'][np.arange(52), t] == pm.LEFT) == n_left
    res = pm.run_randomized(r, ov, dict(pm.VARIANTS[v], n_perm_rand=40), np.random.default_rng(2))
    assert 0 < res['p'] <= 1
    vt2 = 'vt2_mur20_seg8_full_gt23'
    ov2 = _truncate(pm.schedule_randomized(0.0, np.random.default_rng(7), vt2), 52)
    assert 0 < pm.run_randomized(r, ov2, dict(pm.VARIANTS[vt2], n_perm_rand=40), np.random.default_rng(3))['p'] <= 1


def test_feature_cache_distinguishes_site_layouts():
    """"""
    v = 'vt3_mur20_seg8_full_gt23'
    ov = pm.schedule_randomized(0.0, np.random.default_rng(8), v)
    r = sy.rec(pm.run_s(v), seed=8)
    meta2 = dict(ov.meta)
    site2 = np.asarray(ov.meta['site']).copy()
    g0 = ov.meta['group'] == 0
    site2[g0] = site2[g0][::-1]
    meta2['site'] = site2
    ov2 = pm.mc.Overlay(onsets_s=ov.onsets_s.copy(), labels=ov.labels.copy(), kind='stim', meta=meta2)
    F1, F2 = pm._feat(r, ov, 'seg8'), pm._feat(r, ov2, 'seg8')
    assert (F1['site'] == pm.features(r, ov, 'seg8')['site']).all()
    assert (F2['site'] == pm.features(r, ov2, 'seg8')['site']).all() and not (F1['site'] == F2['site']).all()
    meta3 = dict(ov.meta, pos=np.asarray(ov.meta['pos'])[::-1].copy())
    ov3 = pm.mc.Overlay(onsets_s=ov.onsets_s.copy(), labels=ov.labels.copy(), kind='stim', meta=meta3)
    assert pm._feat(r, ov3, 'seg8') is not F1



def test_unusable_recording_raises_and_exactly_enough_runs():
    import pytest
    v = 'vt3_mur20_seg8_full_gt23'
    full = pm.schedule_randomized(0.0, np.random.default_rng(16), v)
    r = sy.rec(pm.run_s(v), seed=16)
    few = _truncate(full, pm.MIN_GROUPS - 1)
    with pytest.raises(pm.mc.NoUsableTrials, match='mbvt'):
        pm.run(r, few, v, np.random.default_rng(0))
    with pytest.raises(pm.mc.NoUsableTrials, match='mbvt'):
        pm.run_randomized(r, few, v, np.random.default_rng(0))
    ok = _truncate(full, pm.MIN_GROUPS)
    assert pm.run(r, ok, v, np.random.default_rng(0))['extra']['n_groups'] == pm.MIN_GROUPS
    assert 0 < pm.run_randomized(r, ok, dict(pm.VARIANTS[v], n_perm_rand=10), np.random.default_rng(0))['p'] <= 1


def _size_task(s):
    out = []
    for v in ('vt3_mur20_seg8_full_gt23', 'vt2_ann18_ds7_max_gt23'):
        ov = pm.schedule(0.0, np.random.default_rng(50000 + s), v)
        res = pm.run(sy.rec(pm.run_s(v), seed=50000 + s), ov, v, np.random.default_rng(s))
        out.append(pm.stat_for_size_correction(res))
        assert res['extra']['size_stat_cont'] == out[-1] and res['extra']['size_stat'] == res['stat']
    return out


def test_size_statistic_is_continuous_on_200_null_draws():
    vals = np.array([_size_task(s) for s in range(200)])
    for j in range(vals.shape[1]):
        assert len(np.unique(vals[:, j])) == 200, f'ties in variant column {j}'


def test_size_statistic_is_the_score_margin_of_the_same_splits():
    v = 'vt3_mur20_seg8_full_gt23'
    r, ov = _planted(3, v, 8.0)
    res = pm.run(r, ov, v, np.random.default_rng(0))
    F = pm._feat(r, ov, 'seg8')
    curve, margin = pm.split_eval(F['E'], pm._tpos(ov, F['keep']), np.random.default_rng(0), 10)
    assert np.array_equal(curve, res['extra']['curve']) and margin == res['extra']['size_stat_cont']
    noise = pm.run(sy.rec(pm.run_s(v), seed=3), ov, v, np.random.default_rng(0))['extra']['size_stat_cont']
    assert margin > noise


N_KEEP_GROUPS = 10


def _heavy_task(s):
    out = []
    for v0, rnd in (('vt3_mur20_seg8_full_gt23', 'vt3'), ('vt2_mur20_ds7_full_gt23', 'vt2')):
        v = dict(pm.VARIANTS[v0], n_perm_rand=40)
        ov = pm.schedule_randomized(0.0, np.random.default_rng(10 ** 6 + s), v)
        keep = np.random.default_rng(4 * 10 ** 6 + s).choice(pm.N_GROUPS, N_KEEP_GROUPS, replace=False)
        first = np.array([ov.onsets_s[ov.meta['group'] == gi].min() for gi in range(pm.N_GROUPS)])
        bad = np.setdiff1d(np.arange(pm.N_GROUPS), keep)
        r = sy.burst(sy.rec(pm.run_s(v0), seed=8300 + s, drift=0.5), first[bad] + 0.6, 0.2, ['Cz'])
        kept = np.nonzero(pm._feat(r, ov, v['feat'])['keep'])[0]
        assert set(kept) <= set(keep) and len(kept) >= N_KEEP_GROUPS - 3
        for k in range(5):
            g = np.random.default_rng(1000 * s + k)
            grp, site = ov.meta['group'], ov.meta['site']
            if rnd == 'vt3':
                tw = pm.mc.balanced_labels((pm.LEFT, pm.RIGHT), (30, 30), g)
                ov.meta['target_wrist'] = tw
                ov.labels = np.where(site == pm.OTHER, pm.DISTRACTOR,
                                     np.where(site == tw[grp], pm.TARGET, pm.NONTARGET))
            else:
                tp = g.integers(0, 8, pm.N_GROUPS)
                ov.labels = np.where(ov.meta['pos'] == tp[grp], pm.TARGET, pm.NONTARGET)
            out.append(pm.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return out


def test_run_randomized_validity_under_heavy_rejection():
    """Whole groups are rejected and each kept group keeps exactly one target, so every re-draw is admissible
    (no re-drawing needed); validity with 10 of 60 groups left."""
    t = time.time()
    calls = sum([_heavy_task(s) for s in range(20)], [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'mbvt run_randomized, <= {N_KEEP_GROUPS}/60 groups kept: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi
