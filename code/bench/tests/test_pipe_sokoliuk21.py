"""pipe_sokoliuk21 (cat. §8): schedule timing; EQUIVALENCE with the authors' Script_2 output (their saved ITPC from
their own preprocessed trials, P03 at 256 Hz and P15 at 512 Hz through the resampling path; fixture built by
make_sokoliuk_ref.py, skipped when absent); Script_3 frequency bins (MATLAB mod round-off compensation); the
band-stop in Script_1 as coded, visible in their preprocessed data; the two surrogate code quirks; a planted
sentence-rate response is called; the noise call rate; run_randomized validity on 200 noise draws (20 recordings x
10 re-drawn label sets, B = 40: exact level 2/41 = 4.9%, process pool)."""
import os
import time

import numpy as np
import pytest
from scipy.signal import freqz
from scipy.stats import binom

from bench import engine as en
from bench import pipe_sokoliuk21 as ps
from bench.tests import _synth as sy

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                   'output', 'intermediate', 'sokoliuk_equiv_ref.npz')
need_ref = pytest.mark.skipif(not os.path.exists(REF), reason='run make_sokoliuk_ref.py with an h5py interpreter')


def _dur(ov):
    return float(ov.onsets_s.max() + ps.TRIAL_S + 20.0)


def test_schedule_timing():
    ov = ps.schedule(5.0, np.random.default_rng(0))
    d = np.diff(ov.onsets_s)
    brk = np.isin(np.arange(1, ps.N_TRIALS), ps.BREAK_AFTER)
    gap = d - ps.TRIAL_S - np.where(brk, ps.BREAK_S, 0.0)
    assert len(ov.onsets_s) == 72 and (ov.labels == ps.SENT).all() and ov.kind == 'stim'
    assert gap.min() >= 1.2 - 1e-9 and gap.max() <= 2.2 + 1e-9
    assert np.allclose(gap, ov.meta['isi_s'][1:])
    assert ps.extent(ov)[0] >= 5.0 and ps.extent(ov)[1] <= 5.0 + ps.RUN_S
    r = ps.schedule_randomized(5.0, np.random.default_rng(0))
    assert np.allclose(r.onsets_s, ov.onsets_s) and (r.labels == ps.SENT).sum() == 36
    assert abs(ps.TRIAL_S - 15.36) < 1e-12


@need_ref
def test_equivalence_with_their_script2_output():
    z = np.load(REF)
    for pid, tol in (('P03', 1e-9), ('P15', 1e-9)):
        it, f = ps.script2_itpc(z[f'{pid}_trials'], z[f'{pid}_time'], float(z[f'{pid}_fs']), list(z[f'{pid}_labels']))
        d = np.abs(it - z[f'{pid}_itpc']).max()
        print(f'sokoliuk equivalence {pid} ({float(z[pid + "_fs"]):.0f} Hz, {z[pid + "_trials"].shape[0]} trials): '
              f'max |ITPC ours - theirs| = {d:.2e} over {it.size} values')
        assert np.array_equal(f, z[f'{pid}_f']) and d < tol


@need_ref
def test_script3_bins_and_matlab_mod():
    f = np.load(REF)['P03_f']
    fois, chance = ps.script3_bins(f)
    assert list(fois) == [11, 22, 44] and np.allclose(f[fois], [0.78135, 1.56271, 3.12542], atol=1e-5)
    lim = int(np.nonzero(f <= 39)[0][-1])
    naive = [k for k in range(lim + 1) if np.mod(f[k], f[11]) != 0]
    assert len(chance) == 500 and not (set(chance) & set(range(0, lim + 1, 11)))
    assert len(naive) > len(chance)


@need_ref
def test_their_band_pass_is_a_band_stop():
    b, edges, cut, df = ps.ec.eegfiltnew_design(256.0, 0.01, 100, 1)
    assert len(b) - 1 == 68 and np.allclose(cut, [6.2218, 93.7882], atol=1e-3)
    w, h = freqz(b, worN=[0.78, 3.125, 30.0, 115.0], fs=256.0)
    g = np.abs(h)
    assert g[0] > 0.95 and g[2] < 0.01 and g[3] > 0.95
    z = np.load(REF)
    x = z['P03_trials']
    P = (np.abs(np.fft.rfft(x, axis=-1)) ** 2).mean((0, 1))
    fr = np.fft.rfftfreq(x.shape[-1], 1 / 256.0)
    hi, mid = P[(fr >= 102) & (fr < 120)].mean(), P[(fr >= 55) & (fr < 80)].mean()
    print(f'their preprocessed P03: mean power 102-120 Hz / 55-80 Hz = {hi / mid:.0f}')
    assert hi > 100 * mid
    assert ps.ec.eegfiltnew_design(256.0, 0.01, 100, 0)[0].size - 1 == 84480


def test_surrogate_quirks():
    g = np.random.default_rng(0)
    ch = np.linspace(0.2, 0.3, 50)
    p_coded = ps.surrogate_p(0.1, ch, g, code='as_coded')[0]
    p_fixed = ps.surrogate_p(0.1, ch, g, code='fixed')[0]
    assert p_coded == 0.5 / ps.REPS and p_fixed == 1.0
    p_top = ps.surrogate_p(0.31, ch, g, code='as_coded')[0]
    assert p_top == 0.0
    chance = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    vals = np.array([0., 10., 20., 30., 40., 50., 60., 70.])
    _, s = ps.surrogate_p(0.0, vals, np.random.default_rng(1), pair=True, code='as_coded', reps=2000)
    assert set(np.unique(s)) <= {0.5 * (vals[i] + vals[2 * i]) for i in range(4)}
    harm = ps.harmonic_entries(chance)
    _, s = ps.surrogate_p(0.0, vals, np.random.default_rng(1), pair=True, code='fixed', harm=harm, reps=2000)
    assert set(np.unique(s)) <= {0.5 * (vals[j] + vals[harm[j]]) for j in range(8) if harm[j] >= 0}


def _planted(seed, effect):
    ov = ps.schedule(5.0, np.random.default_rng(seed))
    r = sy.rec(_dur(ov), seed=seed)
    sent = (ov.onsets_s[:, None] + np.arange(ps.SENT_PER_TRIAL) * ps.WORD_S * ps.WORDS_PER_SENT).ravel()
    so = en.Overlay(onsets_s=sent, labels=np.ones(len(sent), int), kind='stim')
    x = en.inject(r, so, np.ones(len(sent), bool), 'erp', effect, np.random.default_rng(seed + 1))
    return sy.FakeRec(r.source, r.uid + '+trk', r.patient, x, r.fs, list(r.names), {}), ov


def test_planted_tracking_is_called():
    r, ov = _planted(3, 4.0)
    for v in ('roc116_as_coded_as_coded_none', 'surr_ps_fixed_as_coded_none', 'surr_ps_as_coded_as_described_amp'):
        t = time.time()
        res = ps.run(r, ov, v, np.random.default_rng(0))
        e = res['extra']
        print(f"sokoliuk {v} planted: ITPC ps {e['itpc_ps']:.3f} word {e['itpc_word']:.3f} p {res['p']} "
              f"K {e['n_trials']} ({time.time() - t:.1f} s)")
        assert res['call']


def test_noise_call_rate():
    calls = {v: [] for v in ('roc116_as_coded_as_coded_none', 'surr_ps_as_coded_as_coded_none',
                             'surr_word_as_coded_as_coded_none', 'surr_fdr_fixed_as_coded_none')}
    ps_vals, t = [], time.time()
    for s in range(20):
        ov = ps.schedule(5.0, np.random.default_rng(50 + s))
        r = sy.rec(_dur(ov), seed=500 + s)
        for v in calls:
            res = ps.run(r, ov, v, np.random.default_rng(s))
            calls[v].append(res['call'])
        ps_vals.append(res['extra']['itpc_ps'])
    rates = {v: float(np.mean(c)) for v, c in calls.items()}
    print('sokoliuk noise call rates (20 draws, K = 72, independent channels):', rates,
          f'mean ITPC ps {np.mean(ps_vals):.4f} (floor {np.sqrt(np.pi / (4 * 72)):.4f})', f'{time.time() - t:.0f} s')
    assert abs(np.mean(ps_vals) - np.sqrt(np.pi / (4 * 72))) < 0.01
    assert rates['surr_ps_as_coded_as_coded_none'] <= 0.5


def test_band_stop_lowers_the_null_itpc_of_high_chance_bins():
    """Documented property: with Script_1's band-stop as coded, chance bins above ~13 Hz lie in the stop band
    (with the 25-Hz low-pass), where the trial-edge leakage (fixed phase, random sign) weighs in and the null ITPC
    drops toward sqrt(2 / (pi K)) = 0.094 from sqrt(pi / (4 K)) = 0.104 (K = 72). The surrogate is then shifted down
    relative to the FOIs (anti-conservative: 30 noise draws gave p < .05 in 13% (word) and 13% (phrase + sentence)
    as coded, 3% and 7% with the paper's band-pass). Same recordings, both filter readings."""
    lo_c, hi_c, hi_d = [], [], []
    for s in range(8):
        ov = ps.schedule(5.0, np.random.default_rng(50 + s))
        r = sy.rec(_dur(ov), seed=900 + s)
        for filt in ('as_coded', 'as_described'):
            avg, f = ps.itpc_from_features(ps.features(r, ov, filt), np.ones(72, bool))
            _, ch = ps.script3_bins(f)
            low = avg[[k for k in ch if 1 <= f[k] < 8]].mean()
            high = avg[[k for k in ch if 20 <= f[k] <= 39]].mean()
            if filt == 'as_coded':
                lo_c.append(low)
                hi_c.append(high)
            else:
                hi_d.append(high)
    lo_c, hi_c, hi_d = map(np.mean, (lo_c, hi_c, hi_d))
    print(f'sokoliuk null channel-mean ITPC at chance bins: as coded 1-8 Hz {lo_c:.4f}, 20-39 Hz {hi_c:.4f}; '
          f'band-pass 20-39 Hz {hi_d:.4f}')
    assert hi_c < hi_d - 0.0015 and hi_c < lo_c - 0.0015


def _validity_task(s):
    v = dict(ps.VARIANTS['roc116_as_coded_as_coded_none'], n_perm_rand=40)
    ov = ps.schedule_randomized(5.0, np.random.default_rng(10 ** 6 + s), v)
    r = sy.rec(_dur(ov), seed=2300 + s, drift=0.5)
    calls = []
    for k in range(10):
        ov.labels = ps.mc.balanced_labels((ps.SENT, ps.LIST), (36, 36), np.random.default_rng(1000 * s + k))
        calls.append(ps.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call'])
    return calls


def test_run_randomized_validity_on_noise():
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, range(20)), [])
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'sokoliuk run_randomized: {n}/200 false calls ({time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi


def test_seconds_per_assessment():
    ov = ps.schedule(5.0, np.random.default_rng(9))
    r = sy.rec(_dur(ov), seed=9)
    t0 = time.time()
    ps.run(r, ov, 'surr_fdr_as_coded_as_coded_none', np.random.default_rng(0))
    t1 = time.time()
    ovr = ps.schedule_randomized(5.0, np.random.default_rng(9))
    ps.run_randomized(r, ovr, 'roc116_as_coded_as_coded_none', np.random.default_rng(0))
    t2 = time.time()
    print(f'sokoliuk seconds per assessment ({ps.RUN_S:.0f}-s overlay, 19 ch, 128 Hz): run {t1 - t0:.1f} s '
          f'(features included), run_randomized {t2 - t1:.1f} s (features cached)')



SURR_VARIANTS = ('surr_ps_as_coded_as_coded_none', 'surr_word_fixed_as_coded_none', 'surr_fdr_as_coded_as_coded_none')


def _size_task(s):
    ov = ps.schedule(5.0, np.random.default_rng(40000 + s))
    r = sy.rec(_dur(ov), seed=40000 + s)
    out = []
    for v in SURR_VARIANTS:
        res = ps.run(r, ov, v, np.random.default_rng(s))
        out.append(ps.stat_for_size_correction(res))
        assert res['extra']['size_stat_cont'] == out[-1]
    return out


def test_size_statistic_is_continuous_on_200_null_draws():
    vals = np.array(sy.pool_map(_size_task, range(200)))
    for j in range(vals.shape[1]):
        assert len(np.unique(vals[:, j])) == 200, f'ties in {SURR_VARIANTS[j]}'


def test_size_statistic_is_deterministic_and_tracks_evidence():
    r, ov = _planted(3, 4.0)
    noise = sy.rec(_dur(ov), seed=3)
    for v in SURR_VARIANTS:
        a = ps.run(r, ov, v, np.random.default_rng(0))['extra']['size_stat_cont']
        b = ps.run(r, ov, v, np.random.default_rng(99))['extra']['size_stat_cont']
        n = ps.run(noise, ov, v, np.random.default_rng(0))['extra']['size_stat_cont']
        assert a == b and a > n, v



AMP = 'roc116_as_coded_as_described_amp'


def _clean_trials(ov, keep_idx, seed=21):
    """A recording whose only artefact-free trials are keep_idx: a 600-µV 2-Hz burst at F3 and F4 (a burst common to
    all channels would cancel in the average reference) over every other trial, so the 'amp' rule rejects it."""
    r = sy.rec(_dur(ov), seed=seed)
    bad = np.setdiff1d(np.arange(len(ov.onsets_s)), keep_idx)
    return sy.burst(r, ov.onsets_s[bad] + 1.5, 12.0, ['F3', 'F4'], amp=600.0, f_hz=2.0)


def test_unusable_recording_raises_and_exactly_enough_runs():
    import pytest
    ov = ps.schedule_randomized(5.0, np.random.default_rng(15))
    sent, lst = np.nonzero(ov.labels == ps.SENT)[0], np.nonzero(ov.labels == ps.LIST)[0]
    none = _clean_trials(ov, [])
    for v in (AMP, 'surr_ps_fixed_as_described_amp'):
        with pytest.raises(ps.mc.NoUsableTrials, match='sokoliuk21'):
            ps.run(none, ov, v, np.random.default_rng(0))
    with pytest.raises(ps.mc.NoUsableTrials, match='sokoliuk21'):
        ps.run_randomized(none, ov, AMP, np.random.default_rng(0))
    m = ps.MIN_TRIALS
    ok = _clean_trials(ov, list(sent[:m]) + list(lst[:m]))
    assert ps.run(ok, ov, AMP, np.random.default_rng(0))['extra']['n_trials'] == 2 * m
    assert 0 < ps.run_randomized(ok, ov, dict(ps.VARIANTS[AMP], n_perm_rand=10), np.random.default_rng(0))['p'] <= 1
    short = _clean_trials(ov, list(sent[:m]) + list(lst[:m - 1]))
    with pytest.raises(ps.mc.NoUsableTrials, match='sokoliuk21'):
        ps.run_randomized(short, ov, AMP, np.random.default_rng(0))


_ORIG = ps.mc.admissible_stat
N_HEAVY = 6


def _heavy_task(s):
    tally = {'draws': 0, 'rejected': 0}
    ps.mc.admissible_stat = sy.counting(_ORIG, tally)
    try:
        v = dict(ps.VARIANTS[AMP], n_perm_rand=40)
        ov = ps.schedule_randomized(5.0, np.random.default_rng(10 ** 6 + s))
        keep = np.random.default_rng(3 * 10 ** 6 + s).choice(72, N_HEAVY, replace=False)
        r = _clean_trials(ov, keep, seed=7300 + s)

        def one(k):
            ov.labels = ps.mc.balanced_labels((ps.SENT, ps.LIST), (36, 36), np.random.default_rng(1000 * s + k))
            return ps.run_randomized(r, ov, v, np.random.default_rng(7 * s + k))['call']
        return sy.usable_calls(one, range(10 ** 4), 10, ps.mc), tally
    finally:
        ps.mc.admissible_stat = _ORIG


def test_run_randomized_validity_under_heavy_rejection():
    t = time.time()
    res = [_heavy_task(s) for s in range(20)]
    calls = sum([c for c, _ in res], [])[:200]
    rej = sum(tl['rejected'] for _, tl in res) / max(1, sum(tl['draws'] for _, tl in res))
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, 200, 0.05)
    print(f'sokoliuk run_randomized, {N_HEAVY}/72 trials kept: {n}/200 false calls ({rej:.0%} of null draws '
          f're-drawn; {time.time() - t:.0f} s)')
    assert len(calls) == 200 and lo <= n <= hi and rej > 0.05
