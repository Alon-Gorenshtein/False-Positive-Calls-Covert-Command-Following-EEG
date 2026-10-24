"""pipe_gibson16 (cat. §6): schedule timing; the SSEP F-test is positive on stimulus-free noise with a strictly
periodic 200-ms marker train (documented property, cat. §6 [mine]) and much less so with jittered markers; the
FieldTrip pieces (pooled t, clusters over the channel-time graph); planted P3b / P3a are called; noise call rates;
run_randomized validity on 200 noise draws (20 recordings x 10 re-drawn target-wrist orders, B = 40)."""
import time

import numpy as np
from scipy.stats import binom, ttest_ind

from bench import engine as en
from bench import pipe_gibson16 as pg
from bench.tests import _synth as sy


def test_schedule_timing():
    ov = pg.schedule(0.0, np.random.default_rng(0))
    lab, site, blk = ov.labels, ov.meta['site'], ov.meta['block']
    assert list(ov.meta['target_wrist'][:4]) == [pg.RIGHT, pg.LEFT, pg.RIGHT, pg.LEFT] and ov.kind == 'stim'
    assert ov.onsets_s.max() + pg.EPOCH_S <= pg.RUN_S
    share = np.bincount(lab, minlength=4)[1:] / len(lab)
    assert abs(share[2] - 0.8) < 0.03 and abs(share[0] - 0.1) < 0.02
    for b in range(pg.N_BLOCKS):
        m = blk == b
        d = np.diff(ov.onsets_s[m])
        assert np.allclose(d, pg.SOA_S)
        assert 50.0 <= d.sum() <= 75.0
        s = site[m]
        for w in (pg.LEFT, pg.RIGHT):
            assert 28 <= np.sum(s == w) <= 32
        wi = np.nonzero(s != pg.UPPER_BACK)[0]
        gaps = np.diff(wi) - 1
        assert gaps.min() >= 3 and gaps.max() <= 21
        tw = ov.meta['target_wrist'][b]
        assert ((lab[m] == pg.TARGET) == (s == tw)).all()
    r = pg.schedule_randomized(0.0, np.random.default_rng(0))
    assert np.sum(r.meta['target_wrist'] == pg.RIGHT) == 7


def test_fieldtrip_pieces():
    g = np.random.default_rng(1)
    a, b = g.standard_normal((30, 5)), g.standard_normal((40, 5)) + 0.3
    t = pg.indep_t(a.sum(0), (a * a).sum(0), 30, b.sum(0), (b * b).sum(0), 40)
    assert np.allclose(t, ttest_ind(a, b).statistic)
    adj = np.zeros((3, 3), bool)
    adj[0, 1] = adj[1, 0] = True
    A = pg._graph(3, 4, adj)
    tv = np.zeros(12)
    mask = np.zeros(12, bool)
    for node, v in ((0, 2.0), (1, 2.0), (5, 3.0), (8, 4.0), (9, 4.0)):
        tv[node], mask[node] = v, True
    assert pg._max_cluster(tv, mask, A, 1.0) == 8.0
    nb = pg.ec.triangulation_neighbours(pg.SUB_1020)
    assert (nb == nb.T).all() and nb.sum(1).min() >= 2


def _ssep_rate(jitter, n=8):
    calls = []
    for s in range(n):
        ov = pg.schedule(0.0, np.random.default_rng(40 + s))
        if jitter:
            g = np.random.default_rng(s)
            on = ov.onsets_s.copy()
            for b in range(pg.N_BLOCKS):
                m = np.nonzero(ov.meta['block'] == b)[0]
                on[m] = on[m[0]] + np.r_[0.0, np.cumsum(g.uniform(0.15, 0.25, len(m) - 1))]
            ov.onsets_s = on
        r = sy.rec(pg.RUN_S + 20.0, seed=400 + s)
        calls.append([pg.run(r, ov, v, np.random.default_rng(s))['stat'] for v in pg.VARIANTS if v.startswith('ssep')])
    return np.array(calls)


def test_ssep_is_positive_on_a_periodic_marker_train():
    t = time.time()
    per, jit = _ssep_rate(False), _ssep_rate(True)
    print(f'gibson SSEP F on noise: periodic 200 ms median {np.median(per):.1f}, share >= 3.49 '
          f'{np.mean(per >= pg.F_CRIT):.2f}; jittered 150-250 ms median {np.median(jit):.2f}, share '
          f'{np.mean(jit >= pg.F_CRIT):.2f} ({time.time() - t:.0f} s)')
    assert (per >= pg.F_CRIT).all()
    assert np.median(jit) < np.median(per) / 3


def test_planted_p3_is_called():
    ov = pg.schedule(0.0, np.random.default_rng(3))
    r = sy.rec(pg.RUN_S, seed=3)
    x = en.inject(r, ov, ov.labels == pg.TARGET, 'erp', 6.0, np.random.default_rng(1))
    rp = sy.FakeRec('synthetic', 'p3b', 'p', x, r.fs, list(r.names), {})
    for v in ('p3b_two025', 'p3b_two05', 'p3b_onepos', 'p3a_two025'):
        t = time.time()
        res = pg.run(rp, ov, v, np.random.default_rng(0))
        print(f"gibson {v} planted 6 µV at targets: p {res['p']:.4f} ({time.time() - t:.1f} s)")
        assert res['call']
    ovr = pg.schedule_randomized(0.0, np.random.default_rng(3))
    x = en.inject(r, ovr, ovr.labels == pg.TARGET, 'erp', 6.0, np.random.default_rng(1))
    rq = sy.FakeRec('synthetic', 'p3b-r', 'p', x, r.fs, list(r.names), {})
    rr = pg.run_randomized(rq, ovr, pg.DEFAULT, np.random.default_rng(0))
    print(f"gibson run_randomized planted: p {rr['p']:.4f}")
    assert rr['call']


def test_noise_call_rate():
    vs = ('p3a_two025', 'p3a_onepos', 'p3b_two025', 'p3b_two05', 'p3b_onepos')
    calls, t = {v: [] for v in vs}, time.time()
    for s in range(20):
        ov = pg.schedule(0.0, np.random.default_rng(100 + s))
        r = sy.rec(pg.RUN_S, seed=100 + s)
        for v in vs:
            calls[v].append(pg.run(r, ov, v, np.random.default_rng(s))['call'])
    rates = {v: float(np.mean(c)) for v, c in calls.items()}
    print('gibson noise call rates (20 draws, 13 independent 1/f channels, 128 Hz):', rates, f'{time.time() - t:.0f} s')
    assert rates['p3b_two025'] <= 0.3


def _validity_task(s):
    v = dict(pg.VARIANTS[pg.DEFAULT], n_perm_rand=40)
    ov = pg.schedule_randomized(0.0, np.random.default_rng(10 ** 6 + s))
    r = sy.rec(pg.RUN_S, seed=4300 + s, drift=0.5)
    calls = []
    for k in range(10):
        tw = pg.mc.balanced_labels((pg.RIGHT, pg.LEFT), (7, 7), np.random.default_rng(1000 * s + k))
        ov.meta['target_wrist'] = tw
        site = ov.meta['site']
        ov.labels = np.where(site == pg.UPPER_BACK, pg.BACK,
                             np.where(site == tw[ov.meta['block']], pg.TARGET, pg.NONTARGET))
        calls.append(pg.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'gibson run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi


def test_seconds_per_assessment():
    ov = pg.schedule(0.0, np.random.default_rng(9))
    r = sy.rec(pg.RUN_S, seed=9)
    t0 = time.time()
    pg.run(r, ov, 'ssep_avg_nonharm', np.random.default_rng(0))
    t1 = time.time()
    pg.run(r, ov, 'p3b_two025', np.random.default_rng(0))
    t2 = time.time()
    pg.run(r, ov, 'p3a_two025', np.random.default_rng(0))
    t3 = time.time()
    ovr = pg.schedule_randomized(0.0, np.random.default_rng(9))
    rr = sy.rec(pg.RUN_S, seed=10)
    t4 = time.time()
    pg.run_randomized(rr, ovr, pg.DEFAULT, np.random.default_rng(0))
    t5 = time.time()
    print(f'gibson seconds per assessment ({pg.RUN_S:.0f}-s overlay, 13 ch, 128 Hz): features + SSEP {t1 - t0:.1f} s, '
          f'P3b cluster (1000) {t2 - t1:.1f} s, P3a cluster (1000) {t3 - t2:.1f} s, run_randomized {t5 - t4:.1f} s '
          f'(features included)')



EEG19_EDGES = [('C3', 'Cz'), ('C3', 'F7'), ('C3', 'P3'), ('C3', 'T3'), ('C3', 'T5'), ('C4', 'F8'), ('C4', 'P4'),
               ('C4', 'T4'), ('C4', 'T6'), ('Cz', 'C4'), ('Cz', 'F4'), ('Cz', 'P4'), ('Cz', 'Pz'), ('F3', 'C3'),
               ('F3', 'Cz'), ('F3', 'F7'), ('F3', 'Fz'), ('F4', 'C4'), ('F4', 'F8'), ('F7', 'T3'), ('F8', 'T4'),
               ('Fp1', 'F3'), ('Fp1', 'F7'), ('Fp1', 'Fp2'), ('Fp1', 'Fz'), ('Fp2', 'F4'), ('Fp2', 'F8'), ('Fz', 'Cz'),
               ('Fz', 'F4'), ('Fz', 'Fp2'), ('O1', 'O2'), ('O1', 'Pz'), ('P3', 'Cz'), ('P3', 'O1'), ('P3', 'Pz'),
               ('P3', 'T5'), ('P4', 'O2'), ('P4', 'T6'), ('Pz', 'O2'), ('Pz', 'P4'), ('T3', 'T5'), ('T4', 'T6'),
               ('T5', 'O1'), ('T6', 'O2')]
SUB_EDGES = [('C3', 'Cz'), ('C3', 'P3'), ('C3', 'T5'), ('C4', 'P4'), ('C4', 'T6'), ('Cz', 'C4'), ('Cz', 'P3'),
             ('Cz', 'P4'), ('Cz', 'Pz'), ('F3', 'C3'), ('F3', 'Cz'), ('F3', 'F4'), ('F3', 'Fz'), ('F3', 'T5'),
             ('F4', 'C4'), ('F4', 'Cz'), ('F4', 'T6'), ('Fz', 'Cz'), ('Fz', 'F4'), ('O1', 'O2'), ('O1', 'T5'),
             ('O2', 'T6'), ('P3', 'O1'), ('P3', 'Pz'), ('P3', 'T5'), ('P4', 'O2'), ('P4', 'Pz'), ('P4', 'T6'),
             ('Pz', 'O1'), ('Pz', 'O2')]
ERPCORE30 = ['FP1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz', 'FP2',
             'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2']


def _edges(names):
    A = pg.ec.triangulation_neighbours(names)
    assert (A == A.T).all() and not A.diagonal().any()
    return {frozenset((names[i], names[j])) for i, j in zip(*np.nonzero(np.triu(A, 1)))}, A


def test_neighbours_regression_on_10_20_sets():
    """The 19-site results computed before the switch to MNE standard_1005 positions (old: 112's 10-20 unit
    sphere) stay identical."""
    for names, want in ((pg.ec.EEG19, EEG19_EDGES), (pg.SUB_1020, SUB_EDGES)):
        got, _ = _edges(list(names))
        assert got == {frozenset(e) for e in want}


def test_neighbours_for_any_montage():
    for names in (pg.NATIVE_CH, ERPCORE30):
        got, A = _edges(list(names))
        assert A.sum(1).min() >= 2
        import scipy.sparse.csgraph as cg
        assert cg.connected_components(A, directed=False)[0] == 1
    got, _ = _edges(list(pg.NATIVE_CH))
    assert {frozenset(('FC1', 'Fz')), frozenset(('CP1', 'Pz')), frozenset(('Pz', 'Oz'))} <= got
    lower, _ = _edges([c.lower() for c in pg.NATIVE_CH])
    assert len(lower) == len(got)


def test_neighbours_unknown_channel_raises():
    import pytest
    with pytest.raises(ValueError, match=r"EOG1.*XYZ|XYZ.*EOG1"):
        pg.ec.triangulation_neighbours(['Cz', 'Pz', 'C3', 'XYZ', 'EOG1'])


def test_runs_on_the_native_montage():
    """ds003061 / kojimaB carry the native 10-10 sites (FC1, CP1, Oz, PO7, ...): run and run_randomized use them."""
    ov = pg.schedule_randomized(0.0, np.random.default_rng(5))
    r = sy.rec(pg.RUN_S, seed=5, names=list(pg.NATIVE_CH))
    res = pg.run(r, ov, 'p3b_two025', np.random.default_rng(0))
    assert res['extra']['channels'] == 'native' and res['p'] is not None
    rr = pg.run_randomized(r, ov, pg.DEFAULT, np.random.default_rng(0))
    assert 0 < rr['p'] <= 1



def lit_fieldtrip_neighbours(names):
    """Literal transliteration of FieldTrip ft_prepare_neighbours.m @ 7f5a1c60 (2018-01-19), 'triangulation':
    l. 198-211 (elproj 'polar', delaunay of prj, of prj with x halved and with y halved, stacked) and
    compneighbstructfromtri l. 313-347 (mark neighbours per triangle; per channel the 3-D distances to its
    neighbours; neighbdist = mean(alldist) + 3 * std(alldist); neighbours farther than neighbdist removed)."""
    from scipy.spatial import Delaunay
    chanpos = pg.ec.std_positions(names)
    x, y, z = chanpos.T
    az, el = np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))
    prj = np.c_[(np.pi / 2 - el) * np.cos(az), (np.pi / 2 - el) * np.sin(az)]
    tri = np.r_[Delaunay(prj).simplices, Delaunay(np.c_[prj[:, 0] / 2, prj[:, 1]]).simplices,
                Delaunay(np.c_[prj[:, 0], prj[:, 1] / 2]).simplices]
    n = len(names)
    mat = np.zeros((n, n))
    for t in tri:
        mat[t[0], t[1]] = mat[t[0], t[2]] = mat[t[1], t[0]] = mat[t[2], t[0]] = mat[t[1], t[2]] = mat[t[2], t[1]] = 1
    neigh, alldist = [], []
    for i in range(n):
        idx = np.nonzero(mat[i])[0]
        d = np.sqrt(((chanpos[i] - chanpos[idx]) ** 2).sum(1))
        neigh.append((idx, d))
        alldist += list(d)
    lim = np.mean(alldist) + 3 * np.std(alldist, ddof=1)
    A = np.zeros((n, n), bool)
    for i, (idx, d) in enumerate(neigh):
        A[i, idx[d <= lim]] = True
    return A


def test_fieldtrip2018_neighbours_equal_the_literal_transliteration():
    for names in (pg.ec.EEG19, pg.SUB_1020, pg.NATIVE_CH, ERPCORE30):
        names = list(names)
        ft = pg.ec.triangulation_neighbours(names, 'fieldtrip2018')
        assert (ft == lit_fieldtrip_neighbours(names)).all() and (ft == ft.T).all()
    d19 = pg.ec.triangulation_neighbours(list(pg.ec.EEG19), 'delaunay')
    f19 = pg.ec.triangulation_neighbours(list(pg.ec.EEG19), 'fieldtrip2018')
    dsub = pg.ec.triangulation_neighbours(list(pg.SUB_1020))
    fsub = pg.ec.triangulation_neighbours(list(pg.SUB_1020), 'fieldtrip2018')
    print(f'gibson neighbours fieldtrip2018 vs delaunay: 19 sites {int((d19 != f19).sum() // 2)} edges differ, '
          f'13-site set {int((dsub != fsub).sum() // 2)}')
    assert (d19 != f19).any() and (dsub != fsub).any()
    import pytest
    with pytest.raises(ValueError, match='method'):
        pg.ec.triangulation_neighbours(list(pg.SUB_1020), 'template')


def test_neighbours_is_a_variant_axis_of_the_cluster_tests():
    cl = {k: c for k, c in pg.VARIANTS.items() if c['call_on'] != 'ssep'}
    assert {c['neighbours'] for c in cl.values()} == {'delaunay', 'fieldtrip2018'}
    for k, c in cl.items():
        base = k.replace('_fieldtrip2018', '')
        assert pg.VARIANTS[base]['neighbours'] == 'delaunay' and base + '_fieldtrip2018' in pg.VARIANTS
        assert '7f5a1c60' in c['cite'] if c['neighbours'] == 'fieldtrip2018' else True
    assert pg.VARIANTS[pg.DEFAULT]['neighbours'] == 'delaunay'
    assert all('neighbours' not in c for k, c in pg.VARIANTS.items() if c['call_on'] == 'ssep')


def test_run_uses_the_variant_neighbours(monkeypatch):
    seen = []
    orig = pg.ec.triangulation_neighbours
    monkeypatch.setattr(pg.ec, 'triangulation_neighbours', lambda names, method='delaunay': seen.append(method)
                        or orig(names, method))
    ov = pg.schedule_randomized(0.0, np.random.default_rng(8))
    r = sy.rec(pg.RUN_S, seed=8)
    pg.run(r, ov, 'p3b_onepos_fieldtrip2018', np.random.default_rng(0))
    pg.run(r, ov, 'p3b_onepos', np.random.default_rng(0))
    pg.run_randomized(r, ov, dict(pg.VARIANTS['p3b_two025_fieldtrip2018'], n_perm_rand=10), np.random.default_rng(0))
    pg.run_randomized(r, ov, dict(pg.VARIANTS[pg.DEFAULT], n_perm_rand=10), np.random.default_rng(0))
    assert seen == ['fieldtrip2018', 'delaunay', 'fieldtrip2018', 'delaunay']



def _size_task(s):
    pg.NUMRAND = 50
    ov = pg.schedule(0.0, np.random.default_rng(30000 + s))
    r = sy.rec(pg.RUN_S, seed=30000 + s)
    out = []
    for v in ('p3b_two025', 'p3b_onepos', 'p3a_onepos'):
        res = pg.run(r, ov, v, np.random.default_rng(s))
        out.append(pg.stat_for_size_correction(res))
        assert res['extra']['size_stat_cont'] == out[-1]
    return out


def test_size_statistic_is_continuous_on_200_null_draws():
    vals = np.array(sy.pool_map(_size_task, range(200)))
    for j in range(vals.shape[1]):
        assert len(np.unique(vals[:, j])) == 200, f'ties in variant column {j}'


def test_cluster_evidence_is_monotone():
    """Positive cluster mass when a supra-threshold cluster exists (>= crit), else the largest t (< crit)."""
    adj = np.zeros((2, 2), bool)
    A = pg._graph(2, 3, adj)
    crit = 2.0
    none = np.array([1.0, 1.5, 0.2, -3.0, 0.0, 1.9])
    one = np.array([1.0, 2.5, 2.1, 0.0, 0.0, 0.0])
    more = np.array([2.2, 2.5, 2.1, 0.0, 0.0, 0.0])
    e = [pg.cluster_evidence(t, crit, A, 1.0) for t in (none, one, more)]
    assert e[0] == 1.9 and e[1] == 4.6 and np.isclose(e[2], 6.8) and e[0] < crit <= e[1] < e[2]
    assert pg.cluster_evidence(-one, crit, A, -1.0) == 4.6 and pg.cluster_evidence(-none, crit, A, -1.0) == 1.9



def _artefacted(seed, clean=None):
    """Noise plus a 500-µV 20-Hz artefact everywhere except the clean (t0, t1) span, so that every epoch touching
    the artefact fails the +-100 µV rule."""
    r = sy.rec(pg.RUN_S, seed=seed)
    t = np.arange(r.x.shape[1]) / r.fs
    art = 500.0 * np.sin(2 * np.pi * 20.0 * t)
    if clean is not None:
        art[(t >= clean[0]) & (t <= clean[1])] = 0.0
    return sy.FakeRec(r.source, f'{r.uid}-art', r.patient, (r.x + art[None]).astype(np.float32), r.fs,
                      list(r.names), {})


def _window_with(ov, n_t, n_nt, start=0):
    """A clean span covering consecutive wrist stimuli of one block with exactly n_t targets and n_nt non-targets."""
    lab, blk, on = ov.labels, ov.meta['block'], ov.onsets_s
    wr = np.nonzero(lab != pg.BACK)[0]
    for i in range(start, len(wr)):
        for j in range(i, len(wr)):
            seg = wr[i:j + 1]
            if blk[seg[0]] != blk[seg[-1]]:
                break
            k_t, k_n = np.sum(lab[seg] == pg.TARGET), np.sum(lab[seg] == pg.NONTARGET)
            if (k_t, k_n) == (n_t, n_nt) and lab[seg[0]] != pg.BACK:
                yield i, (on[seg[0]] - pg.PRE_S - 0.15, on[seg[-1]] + pg.EPOCH_S - pg.PRE_S + 0.15)
            if k_t > n_t or k_n > n_nt:
                break


def _rec_keeping(ov, n_t, n_nt, seed=11):
    """A recording whose kept trials are exactly n_t targets and n_nt non-targets (searched; the kurtosis rule may
    drop an epoch, so the counts are checked through features())."""
    for _, span in _window_with(ov, n_t, n_nt):
        r = _artefacted(seed, span)
        F = pg.features(r, ov)
        lab = ov.labels[F['keep']]
        if (np.sum(lab == pg.TARGET), np.sum(lab == pg.NONTARGET)) == (n_t, n_nt):
            return r, F
    raise AssertionError('no window found')


def test_all_rejected_recording_raises_no_usable_trials():
    import pytest
    assert pg.NoUsableTrials is pg.mc.NoUsableTrials and issubclass(pg.NoUsableTrials, ValueError)
    ov = pg.schedule_randomized(0.0, np.random.default_rng(12))
    r = _artefacted(12)
    assert pg.features(r, ov)['keep'].sum() == 0
    for v in ('ssep_avg_nonharm', 'p3a_two025', 'p3b_onepos'):
        with pytest.raises(pg.NoUsableTrials, match='gibson16'):
            pg.run(r, ov, v, np.random.default_rng(0))
    with pytest.raises(pg.NoUsableTrials, match='gibson16'):
        pg.run_randomized(r, ov, pg.DEFAULT, np.random.default_rng(0))


def test_exactly_enough_trials_run_and_one_fewer_raises():
    import pytest
    ov = pg.schedule_randomized(0.0, np.random.default_rng(13))
    r, F = _rec_keeping(ov, pg.MIN_PER_CLASS, pg.MIN_PER_CLASS)
    res = pg.run(r, ov, 'p3b_onepos', np.random.default_rng(0))
    assert res['extra']['n_a'] == res['extra']['n_b'] == pg.MIN_PER_CLASS
    rr = pg.run_randomized(r, ov, dict(pg.VARIANTS[pg.DEFAULT], n_perm_rand=10), np.random.default_rng(0))
    assert 0 < rr['p'] <= 1
    assert np.isfinite(pg.run(r, ov, 'ssep_f5_plain', np.random.default_rng(0))['stat'])
    r2, _ = _rec_keeping(ov, pg.MIN_PER_CLASS - 1, pg.MIN_PER_CLASS)
    with pytest.raises(pg.NoUsableTrials, match='gibson16'):
        pg.run(r2, ov, 'p3b_onepos', np.random.default_rng(0))
    with pytest.raises(pg.NoUsableTrials, match='gibson16'):
        pg.run_randomized(r2, ov, pg.DEFAULT, np.random.default_rng(0))



N_HEAVY_WRISTS = 30


def _heavy_task(s):
    """Every draw of the target wrists meets MIN_PER_CLASS whenever the observed one can be tested (run_randomized
    raises NoUsableTrials unless sum_b min(n_left_b, n_right_b) >= MIN_PER_CLASS), so no re-draw is ever undefined."""
    v = dict(pg.VARIANTS[pg.DEFAULT], n_perm_rand=40)
    ov = pg.schedule_randomized(0.0, np.random.default_rng(10 ** 6 + s))
    wr = np.nonzero(ov.meta['site'] != pg.UPPER_BACK)[0]
    keep = np.random.default_rng(5 * 10 ** 6 + s).choice(wr, N_HEAVY_WRISTS, replace=False)
    r = sy.rec(pg.RUN_S, seed=9300 + s, drift=0.5)
    t = np.arange(r.x.shape[1]) / r.fs
    art = np.ones_like(t, bool)
    for i in keep:
        art[(t >= ov.onsets_s[i] - pg.PRE_S - 0.3) & (t <= ov.onsets_s[i] + pg.EPOCH_S - pg.PRE_S + 0.3)] = False
    x = r.x + (500.0 * np.sin(2 * np.pi * 10.0 * t) * art)[None]
    r = sy.FakeRec(r.source, r.uid + '+heavy', r.patient, x.astype(np.float32), r.fs, list(r.names), {})

    def one(k):
        tw = pg.mc.balanced_labels((pg.RIGHT, pg.LEFT), (7, 7), np.random.default_rng(1000 * s + k))
        ov.meta['target_wrist'] = tw
        site = ov.meta['site']
        ov.labels = np.where(site == pg.UPPER_BACK, pg.BACK,
                             np.where(site == tw[ov.meta['block']], pg.TARGET, pg.NONTARGET))
        return pg.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call']
    return sy.usable_calls(one, range(10 ** 4), 10, pg.mc, max_tries=10), int(pg._feat(r, ov)['keep'].sum())


def test_run_randomized_validity_under_heavy_rejection():
    t = time.time()
    res = [_heavy_task(s) for s in range(30)]
    calls = sum([c for c, _ in res], [])[:200]
    kept = np.median([k for _, k in res])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'gibson run_randomized, median {kept:.0f} of ~1000 epochs kept: {n}/200 false calls '
          f'({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi and kept < 150
