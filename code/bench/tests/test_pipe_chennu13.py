"""pipe_chennu13 (cat. §2): EQUIVALENCE with the authors' MATLAB (srivaschennu/P300Words @ 1d716d48) by literal
transliteration: no MATLAB/Octave runtime exists here, so each routine is re-written below loop for loop from the
.m file (1-based indices kept as comments) and compared with the module's vectorized port on the same random draws
(compgfp.m, corrclust.m, calctct.m + corrtct.m, the base condition, calcgfp's rmbase). Plus: the paradigm generator
(p300words.m setupblock), a planted P3b is called, noise call rates, the documented channel-count dependence of the
condition-vs-own-baseline test, and run_randomized validity on 200 noise draws."""
import time

import numpy as np
from scipy.stats import binom

from bench import engine as en
from bench import pipe_chennu13 as pc
from bench import erp_common as ec
from bench.tests import _synth as sy

V12 = 'p1_samples10_none_b12'



def lit_calcgfp(data, n0):
    """compgfp.m l. 301-307: eeg_gfp(data', 0)' then rmbase(gfp, [], 1:find(times == 0))."""
    gfp = np.array([np.std(data[:, t]) for t in range(data.shape[1])])
    return gfp - np.mean(gfp[0:n0 + 1])


def lit_compgfp(c1, c2, n0, numrand, g):
    """compgfp.m l. 193-199, 248-269 (statmode 'trial'), data as (epochs, C, T)."""
    inddata1, inddata2 = c1.copy(), c2.copy()
    mergedata = np.concatenate([c1, c2])
    gfpdiff = np.zeros((numrand + 1, c1.shape[2]))
    for n in range(1, numrand + 2):
        if n > 1:
            mergedata = mergedata[g.permutation(mergedata.shape[0])]
            inddata1 = mergedata[:c1.shape[0]]
            inddata2 = mergedata[c1.shape[0]:]
        gfpdiff[n - 1] = lit_calcgfp(inddata1.mean(0), n0) - lit_calcgfp(inddata2.mean(0), n0)
    return gfpdiff


def lit_corrclust(gfpdiff, corrwin, alpha=0.05, clustsize=10):
    """corrclust.m l. 13-90 (ttesttail 1). corrwin: 0-based sample indices."""
    R, T = gfpdiff.shape
    pdist = gfpdiff[:, corrwin].max(1)
    tstats = np.zeros(R)
    for n in range(R):
        valu, pprob = np.zeros(T), np.ones(T)
        for p in corrwin:
            valu[p] = (gfpdiff[n, p] - gfpdiff[:, p].mean()) / (gfpdiff[:, p].std(ddof=1) / np.sqrt(R))
            pprob[p] = np.sum(pdist >= gfpdiff[n, p]) / R
        pmask = (pprob < alpha).astype(int)
        best, pstart = 0.0, None
        for p in range(1, T):
            if pmask[p] == 1 and pmask[p - 1] == 0:
                pstart = p
            elif (pmask[p] == 0 or p == T - 1) and pmask[p - 1] == 1:
                pend = p
                ts = valu[pstart:pend].sum()
                if ts > best and len(range(pstart, pend)) >= clustsize:
                    best = ts
        tstats[n] = best
    prob = np.sum(tstats >= tstats[0]) / len(tstats)
    return tstats[0], prob, tstats


def lit_tct(c1, corrwin, n0, numrand, g, alpha=0.05):
    """calctct.m l. 173-211 + corrtct.m l. 27-58 (trial mode). The channel permutation of each epoch is drawn as
    argsort(uniform) in the same order as the module, so both see identical draws."""
    inddata = c1.copy()
    n, C, T = c1.shape
    condgfp = np.zeros((numrand + 1, T))
    for k in range(1, numrand + 2):
        if k > 1:
            perms = np.argsort(g.random((n, C)), axis=1)
            for s in range(n):
                inddata[s] = inddata[s][perms[s]]
        condgfp[k - 1] = lit_calcgfp(inddata.mean(0), n0)
    R = numrand + 1
    phits = np.zeros(R)
    for k in range(R):
        pmask = np.zeros(T)
        for p in corrwin:
            pmask[p] = (np.sum(condgfp[:, p] >= condgfp[k, p]) / R) < alpha
        phits[k] = pmask.sum()
    return np.sum(phits[0] <= phits) / R, phits


def _epochs(seed, n=60, C=8, T=60, bump=0.0):
    g = np.random.default_rng(seed)
    x = np.cumsum(g.standard_normal((n, C, T)), axis=2) * 0.3 + g.standard_normal((n, C, T))
    if bump:
        x[:, :3, 35:50] += bump
    x = x - x[..., :21].mean(-1, keepdims=True)
    return x - x.mean(1, keepdims=True)


def test_equivalence_compgfp_and_base_condition():
    c1 = _epochs(1)
    W = np.arange(30, 50)
    c2 = pc.base_condition(c1, W)
    lit_c2 = c1.copy()
    lit_c2[:, :, W] = c1[:, :, 0:len(W)]
    assert np.array_equal(c2, lit_c2)
    a = pc.compgfp(c1, c2, 20, 50, np.random.default_rng(5))
    b = lit_compgfp(c1, c2, 20, 50, np.random.default_rng(5))
    print(f'chennu equivalence compgfp: max |diff| {np.abs(a - b).max():.2e}')
    assert np.allclose(a, b, rtol=0, atol=1e-10)


def test_equivalence_corrclust():
    worst = 0.0
    for seed in range(12):
        g = np.random.default_rng(seed)
        R, T = 101, 60
        gd = g.standard_normal((R, T))
        if seed % 3 == 0:
            gd[0, 32:46] += 3.0
        if seed % 3 == 1:
            gd[0, 32:36] += 3.0
        if seed % 4 == 2:
            gd[0, 40:59] += 3.0
        W = np.arange(30, 58)
        t1, p1, rows1 = lit_corrclust(gd, W, clustsize=10)
        cc = pc.corrclust(gd, W, clustsize=10)
        worst = max(worst, np.abs(rows1 - cc['best']).max())
        assert np.isclose(cc['tstat'], t1, atol=1e-9) and cc['prob'] == p1
        assert np.allclose(cc['best'], rows1, atol=1e-9)
    print(f'chennu equivalence corrclust: 12 matrices, max |row cluster sum diff| {worst:.2e}, probs identical')


def test_equivalence_tct():
    c1 = _epochs(2, n=40, bump=1.5)
    W = np.arange(30, 55)
    p_lit, hits_lit = lit_tct(c1, W, 20, 60, np.random.default_rng(9))
    r = pc.tct(c1, W, 20, 60, np.random.default_rng(9))
    print(f'chennu equivalence TCT: p {r["prob"]:.4f} (literal {p_lit:.4f})')
    assert r['prob'] == p_lit and np.array_equal(r['hits'], hits_lit)


def test_schedule_follows_setupblock():
    for v, nb in ((pc.DEFAULT, 20), (V12, 12)):
        ov = pc.schedule(0.0, np.random.default_rng(4), v)
        ex = ov.meta['explicit_word']
        assert len(ex) == nb and (ex == pc.YES).sum() == nb // 2 and pc.mc.max_run(ex) <= 4
        assert ov.onsets_s.max() + pc.EPOCH[1] + 9.0 <= pc.run_s(v) and ov.kind == 'stim'
        for b in range(nb):
            m = ov.meta['block'] == b
            w, lab, t = ov.meta['word'][m], ov.labels[m], ov.onsets_s[m]
            for word in (pc.YES, pc.NO):
                pos = np.nonzero(w == word)[0] + 1
                assert 13 <= len(pos) <= 16 and pos[0] >= 3
                assert np.diff(pos).min() >= 3 and np.diff(pos).max() <= 9
            assert len(w) == max(np.nonzero(w)[0]) + 1 + pc.TREPINT
            assert ((lab == pc.EXPL) == (w == ex[b])).all() and ((lab == pc.DIST) == (w == 0)).all()
            d = np.diff(t)
            assert d.min() >= 0.9 - 1e-9 and d.max() <= 1.1 + 1e-9
        first = np.array([ov.onsets_s[ov.meta['block'] == b][0] for b in range(nb)])
        assert np.isclose(first[0], 1.0 + 13.0)


def _planted(seed, effect, v=V12):
    ov = pc.schedule(0.0, np.random.default_rng(seed), v)
    r = sy.rec(pc.run_s(v), seed=seed)
    x = en.inject(r, ov, ov.labels == pc.EXPL, 'erp', effect, np.random.default_rng(seed + 1))
    return sy.FakeRec(r.source, r.uid + '+p3', r.patient, x, r.fs, list(r.names), {}), ov


def test_planted_p3b_is_called():
    r, ov = _planted(3, 12.0)
    for v in (V12, 'p3b_base_ms40_zvar_b12'):
        t = time.time()
        res = pc.run(r, ov, v, np.random.default_rng(0))
        print(f"chennu {v} planted 12 µV: call {res['call']} parts {res['extra']['parts']} ({time.time() - t:.1f} s)")
        assert res['call']
    rr = pc.run_randomized(r, ov, V12, np.random.default_rng(1))
    print(f"chennu run_randomized planted: p {rr['p']:.4f}")
    assert rr['call']


def test_noise_call_rate():
    rates, t = {v: [] for v in (V12, 'p3b_base_samples10_none_b12', 'p3_any_ms40_none_b12')}, time.time()
    for s in range(10):
        ov = pc.schedule(0.0, np.random.default_rng(100 + s), V12)
        r = sy.rec(pc.run_s(V12), seed=100 + s)
        for v in rates:
            rates[v].append(pc.run(r, ov, v, np.random.default_rng(s))['call'])
    rates = {v: float(np.mean(c)) for v, c in rates.items()}
    print('chennu noise call rates (10 draws, 19 independent 1/f channels, 128 Hz):', rates, f'{time.time() - t:.0f} s')
    assert rates[V12] <= 0.2


def _base_sig(C, fs, seed, n=280):
    """The condition-vs-own-baseline P3b cluster on C independent 1/f channels with CH13 preprocessing."""
    g = np.random.default_rng(seed)
    N = int((n + 20) * fs)
    f = np.fft.rfftfreq(N, 1 / fs)
    shape = np.where(f > 0, f.clip(0.5) ** -0.5, 0)
    X = np.fft.irfft(np.fft.rfft(g.standard_normal((C, N)), axis=1) * shape, N, axis=1)
    X = ec.eegfilt_fir1(ec.eegfilt_fir1(X, fs, 0, 20.0), fs, 0.5, 0)
    on = 10 + np.cumsum(0.9 + 0.2 * g.random(n))
    n_pre, n_post = int(round(0.3 * fs)), int(round(0.8 * fs))
    ep = ec.sample_epochs(X, fs, on, n_pre, n_post)
    ep = ep - ep[..., :n_pre + 1].mean(-1, keepdims=True)
    ep = ep - ep.mean(1, keepdims=True)
    W = pc.corrwin(np.arange(-n_pre, n_post) / fs * 1000, pc.WINDOWS['p3b'])
    return pc.corrclust(pc.compgfp(ep, pc.base_condition(ep, W), n_pre, pc.NUMRAND, g), W)['sig']


def test_base_test_depends_on_channel_count():
    """Documented property (cat. §2 [mine] predicted 95-100% on spatially independent 91-channel noise): reproduced
    at 91 channels; at the benchmark's 19 channels the same noise gives no calls."""
    r91 = np.mean([_base_sig(91, 250.0, s) for s in range(8)])
    r19 = np.mean([_base_sig(19, 250.0, s) for s in range(8)])
    print(f'chennu P3b-vs-own-baseline null call rate, independent 1/f channels: 91 ch {r91:.2f}, 19 ch {r19:.2f}')
    assert r91 >= 0.75 and r19 <= 0.25


def _validity_task(s):
    v = dict(pc.VARIANTS[V12])
    ov = pc.schedule(0.0, np.random.default_rng(10 ** 6 + s), v)
    r = sy.rec(pc.run_s(V12), seed=3300 + s, drift=0.5)
    calls = []
    for k in range(10):
        ex = pc._explicit_order(12, np.random.default_rng(1000 * s + k))
        ov.meta['explicit_word'] = ex
        w = ov.meta['word']
        ov.labels = np.where(w == 0, pc.DIST, np.where(w == ex[ov.meta['block']], pc.EXPL, pc.IMPL))
        calls.append(pc.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'chennu run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi


def test_seconds_per_assessment():
    for v in (pc.DEFAULT, V12):
        ov = pc.schedule(0.0, np.random.default_rng(9), v)
        r = sy.rec(pc.run_s(v), seed=9)
        t0 = time.time()
        pc.run(r, ov, v, np.random.default_rng(0))
        t1 = time.time()
        pc.run_randomized(r, ov, v, np.random.default_rng(0))
        t2 = time.time()
        print(f'chennu seconds per assessment {v} ({pc.run_s(v):.0f}-s overlay, 19 ch, 128 Hz): run {t1 - t0:.1f} s '
              f'(features + 3 x 1000 randomisations), run_randomized {t2 - t1:.2f} s (B = 500, features cached)')



def _size_task(s):
    pc.NUMRAND = 50
    ov = pc.schedule(0.0, np.random.default_rng(20000 + s), V12)
    r = sy.rec(pc.run_s(V12), seed=20000 + s)
    out = []
    for v in (V12, 'p3b_base_samples10_none_b12', 'p3_any_ms40_none_b12'):
        res = pc.run(r, ov, v, np.random.default_rng(s))
        out.append(pc.stat_for_size_correction(res))
        assert res['extra']['size_stat_cont'] == out[-1]
    return out


def test_size_statistic_is_continuous_on_200_null_draws():
    vals = np.array(sy.pool_map(_size_task, range(200)))
    for j in range(vals.shape[1]):
        assert len(np.unique(vals[:, j])) == 200, f'ties in variant column {j}'


def test_size_statistic_is_deterministic_and_tracks_evidence():
    r, ov = _planted(3, 12.0)
    a = pc.run(r, ov, V12, np.random.default_rng(0))['extra']['size_stat_cont']
    b = pc.run(r, ov, V12, np.random.default_rng(99))['extra']['size_stat_cont']
    noise = pc.run(sy.rec(pc.run_s(V12), seed=3), ov, V12, np.random.default_rng(0))['extra']['size_stat_cont']
    assert a == b and a > noise


def test_feature_cache_distinguishes_label_layouts():
    """features() epochs the non-distractor words, so overlays with equal onsets but different labels must not share
    cached features."""
    ov = pc.schedule(0.0, np.random.default_rng(4), V12)
    r = sy.rec(pc.run_s(V12), seed=4)
    lab2 = ov.labels.copy()
    d = np.nonzero(lab2 == pc.DIST)[0][:5]
    lab2[d] = pc.IMPL
    ov2 = pc.ec.Overlay(onsets_s=ov.onsets_s.copy(), labels=lab2, kind='stim', meta=dict(ov.meta))
    n1 = len(pc._feat(r, ov)['sel'])
    n2 = len(pc._feat(r, ov2)['sel'])
    assert n2 == n1 + 5



def _relabel(ov, n_expl, n_impl):
    """Keep only the first n_expl explicit and n_impl implicit words as targets (the rest become distractors)."""
    lab = ov.labels.copy()
    for code, n in ((pc.EXPL, n_expl), (pc.IMPL, n_impl)):
        idx = np.nonzero(lab == code)[0]
        lab[idx[n:]] = pc.DIST
    meta = dict(ov.meta)
    meta['word'] = np.where(lab == pc.DIST, 0, ov.meta['word'])
    return pc.ec.Overlay(onsets_s=ov.onsets_s.copy(), labels=lab, kind='stim', meta=meta)


def test_unusable_recording_raises_and_exactly_enough_runs():
    import pytest
    ov = pc.schedule(0.0, np.random.default_rng(14), V12)
    r = sy.rec(pc.run_s(V12), seed=14)
    few = _relabel(ov, pc.MIN_EPOCHS - 1, 50)
    for v in (V12, 'p3b_base_samples10_none_b12'):
        with pytest.raises(pc.mc.NoUsableTrials, match='chennu13'):
            pc.run(r, few, v, np.random.default_rng(0))
    with pytest.raises(pc.mc.NoUsableTrials, match='chennu13'):
        pc.run_randomized(r, few, V12, np.random.default_rng(0))
    ok = _relabel(ov, pc.MIN_EPOCHS, pc.MIN_EPOCHS)
    assert isinstance(pc.run(r, ok, V12, np.random.default_rng(0))['call'], bool)
    assert 0 < pc.run_randomized(r, ok, dict(pc.VARIANTS[V12], n_perm_rand=10), np.random.default_rng(0))['p'] <= 1
    short_impl = _relabel(ov, 50, pc.MIN_EPOCHS - 1)
    assert isinstance(pc.run(r, short_impl, 'p3b_base_samples10_none_b12', np.random.default_rng(0))['call'], bool)
    with pytest.raises(pc.mc.NoUsableTrials, match='chennu13'):
        pc.run(r, short_impl, V12, np.random.default_rng(0))


_ORIG = pc.mc.admissible_stat
KEEP_SHARE = 0.025


def _heavy_task(s):
    tally = {'draws': 0, 'rejected': 0}
    keep0 = pc._keep
    pc.mc.admissible_stat = sy.counting(_ORIG, tally)
    pc._keep = lambda ep, reject, _s=s: np.random.default_rng(777 + _s).random(len(ep)) < KEEP_SHARE
    try:
        v = dict(pc.VARIANTS[V12], n_perm_rand=40)
        ov = pc.schedule(0.0, np.random.default_rng(10 ** 6 + s), V12)
        r = sy.rec(pc.run_s(V12), seed=6300 + s, drift=0.5)

        def one(k):
            ex = pc._explicit_order(12, np.random.default_rng(1000 * s + k))
            ov.meta['explicit_word'] = ex
            w = ov.meta['word']
            ov.labels = np.where(w == 0, pc.DIST, np.where(w == ex[ov.meta['block']], pc.EXPL, pc.IMPL))
            return pc.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call']
        return sy.usable_calls(one, range(10 ** 4), 10, pc.mc), tally
    finally:
        pc._keep, pc.mc.admissible_stat = keep0, _ORIG


def test_run_randomized_validity_under_heavy_rejection():
    t = time.time()
    res = [_heavy_task(s) for s in range(25)]
    calls = sum([c for c, _ in res], [])[:200]
    rej = sum(tl['rejected'] for _, tl in res) / max(1, sum(tl['draws'] for _, tl in res))
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'chennu run_randomized, {KEEP_SHARE:.1%} of epochs kept: {n}/200 false calls ({rej:.0%} of null draws '
          f're-drawn; {time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi and rej > 0.05
