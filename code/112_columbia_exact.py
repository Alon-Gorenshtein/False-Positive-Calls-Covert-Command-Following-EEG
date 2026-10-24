"""Exact re-implementation of the Columbia CMD analysis (NEJM 2019, the 'psd' pipeline) from the authors' public code,
kwdoyle/CMD-Detection @ 38d013c (the Columbia pipeline notes cites every line), for the no-command null.

- features(): command onsets -> instruction offsets (their shift_events), five 2-s sub-epochs per command (2 * fs + 1
  samples each), 1-30 Hz zero-phase FIR (MNE Raw.filter defaults), pycsd spherical-spline CSD (10-5 template, m = 4,
  50 Legendre terms, lambda 1e-5, unit sphere), MNE 1.1.1 multitaper PSD (NW = 4, symmetric DPSS, tapers with
  eigenvalue > 0.9), mean PSD in 1-3/4-7/8-13/14-30 Hz -> 76 features (19 channels x 4 bands, channel-major).
- exact_test(): StandardScaler + LinearSVC(C=1, dual=True, max_iter=1000), leave-one-trial-out CV (a trial = one keep
  plus one stop, 10 epochs), AUC = mean of the 48 per-fold AUCs; null = 500 joint shuffles of (label, fold id) over
  epochs with X fixed; p = (#{perm >= obs} + 1) / (n_perm + 1); call = their Benjamini-Hochberg reject for one
  recording (mne fdr_correction: p < 0.05).
- trialflip_test(): the same statistic; the null flips each trial's keep/stop labels as a unit (prob 0.5).
- schedule(): 6 blocks x 8 trials, keep then stop, hands alternating by block, 3.2-s commands, 10 s after keep and
  15 s after stop (the code's timing assumption).
Library behaviour reproduced is the repo's requirements.txt (mne 1.1.1, scikit-learn 1.2.0, numpy 1.23.3): newer
versions changed two defaults (LinearSVC dual -> 'auto'; MNE multitaper DPSS sym -> False), which are pinned here.
The RNG: one np.random.RandomState drives the shuffles and every liblinear seed in the same order as their global
np.random state, so a RandomState(s) here reproduces np.random.seed(s) + run_pipeline exactly. Serial by design;
parallelise across overlays.
python 112_columbia_exact.py --smoke   (one overlay: features + both tests at 500 permutations)"""
import warnings

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.linalg import inv
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal.windows import dpss as sp_dpss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from mne.filter import filter_data
from mne.stats import fdr_correction

ALPHA = 0.05
USE_CH = ['C3', 'C4', 'O1', 'O2', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz',
          'Fp1', 'Fp2', 'P3', 'P4', 'Pz', 'T7', 'T8', 'P7', 'P8']
ALIAS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
FMIN, FMAX = 1.0, 30.0
BANDS = ((1, 3), (4, 7), (8, 13), (14, 30))
N_SEG, TMAX, WIN_S = 5, 2.0, 10.0
KEEP_L, STOP_L, KEEP_R, STOP_R = 1, 2, 3, 4
N_BLOCKS, N_TRIALS = 6, 8
CMD_S, AFTER_KEEP_S, AFTER_STOP_S = 3.2, 10.0, 15.0
MAX_ITER = 1000

_POS = {'C3': (-0.7071, 0.0, 0.7071), 'C4': (0.7071, 0.0, 0.7071), 'O1': (-0.3090, -0.9511, 0.0),
        'O2': (0.3090, -0.9511, 0.0), 'Cz': (0.0, 0.0, 1.0), 'F3': (-0.5489, 0.6717, 0.4975),
        'F4': (0.5489, 0.6717, 0.4975), 'F7': (-0.8090, 0.5878, 0.0), 'F8': (0.8090, 0.5878, 0.0),
        'Fz': (0.0, 0.7071, 0.7071), 'Fp1': (-0.3090, 0.9511, 0.0), 'Fp2': (0.3090, 0.9511, 0.0),
        'P3': (-0.5489, -0.6717, 0.4975), 'P4': (0.5489, -0.6717, 0.4975), 'Pz': (0.0, -0.7071, 0.7071),
        'T7': (-1.0, 0.0, 0.0), 'T8': (1.0, 0.0, 0.0), 'P7': (-0.8090, -0.5878, 0.0), 'P8': (0.8090, -0.5878, 0.0)}



def csd_matrices(names=USE_CH, m=4, n_terms=50, lambda2=1e-5):
    pos = np.array([_POS[c] for c in names], np.float32).astype(np.float64)
    cosang = np.dot(pos, pos.T)
    g = [(2 * n + 1) / (n ** m * (n + 1) ** m * 4 * np.pi) for n in range(1, n_terms + 1)]
    h = [(2 * n + 1) / (n ** (m - 1) * (n + 1) ** (m - 1) * 4 * np.pi) for n in range(1, n_terms + 1)]
    G, H = legval(cosang, [0] + g), legval(cosang, [0] + h)
    G.flat[::len(G) + 1] += lambda2
    Gi = inv(G)
    TC = Gi.sum(0)
    return Gi, TC, np.sum(TC), H


_CSD = csd_matrices()


def _csd(epoch, Gi, TC, sgi, H):
    Z = epoch - epoch.mean(0)[None]
    Cp2 = np.dot(Gi, Z)
    c02 = np.sum(Cp2, axis=0) / sgi
    C2 = Cp2 - np.dot(TC[:, None], c02[None, :])
    return np.dot(C2.T, H).T / 1.0



def _next_fast_len(target):
    """MNE 1.1.1 filter.next_fast_len: smallest 2,3,5-smooth number >= target."""
    if target <= 6 or not (target & (target - 1)):
        return target
    n = target
    while True:
        k = n
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return n
        n += 1


def dpss_111(n_times, half_nbw=4.0, low_bias=True):
    """MNE 1.1.1 dpss_windows: symmetric DPSS, eigenvalues from the taper autocorrelation, keep eigenvalue > 0.9."""
    N, Kmax = int(n_times), int(2 * half_nbw)
    W = float(half_nbw) / N
    dpss = sp_dpss(N, half_nbw, Kmax)
    n_fft = _next_fast_len(2 * N - 1)
    dpss_fft = rfft(dpss, n_fft)
    rxx = irfft(dpss_fft * dpss_fft.conj(), n_fft)[:, :N]
    r = 4 * W * np.sinc(2 * W * np.arange(N, dtype='d'))
    r[0] = 2 * W
    eigvals = np.dot(rxx, r)
    if low_bias:
        idx = eigvals > 0.9
        if not idx.any():
            idx = [np.argmax(eigvals)]
        dpss, eigvals = dpss[idx], eigvals[idx]
    return dpss, eigvals


def psd_multitaper_111(x, sfreq, fmin=FMIN, fmax=FMAX):
    """x: (..., n_times) -> (psd (..., n_freqs), freqs); normalization='length', adaptive=False."""
    x = np.asarray(x, float)
    shape, n_times = x.shape[:-1], x.shape[-1]
    x = x.reshape(-1, n_times)
    dpss, eigvals = dpss_111(n_times)
    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
    freqs = rfftfreq(n_times, 1. / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    x = x - np.mean(x, axis=-1, keepdims=True)
    x_mt = rfft(x[:, np.newaxis, :] * dpss, n=n_times)
    x_mt[..., 0] /= np.sqrt(2.)
    if n_times % 2 == 0:
        x_mt[..., -1] /= np.sqrt(2.)
    psd = weights * x_mt[:, :, mask]
    psd *= psd.conj()
    psd = psd.real.sum(axis=-2)
    psd *= 2 / (weights * weights.conj()).real.sum(axis=-2)
    return psd.reshape(shape + (int(mask.sum()),)), freqs[mask]



def _reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.mean(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def shift_events(samp, ids, fs):
    """Command onsets (samples) -> instruction offsets: keep shifted by the median of (next stop - keep) - 10 s, stop
    by the outlier-trimmed median of (next keep - stop) - 15 s, per hand."""
    samp, ids = np.asarray(samp, np.int64), np.asarray(ids, int)
    if not (np.any(ids == STOP_L) and np.any(ids == STOP_R)):
        raise ValueError('their read_data skips recordings without both hands')
    out = samp.copy()
    for keep, stop in ((KEEP_R, STOP_R), (KEEP_L, STOP_L)):
        km, sm = samp[ids == keep], samp[ids == stop]
        k = round(np.median([(sm[i] - km[i]) - (10 * fs) for i in range(len(sm))]))
        s = round(np.median(_reject_outliers(np.array([(km[i] - sm[i - 1]) - (15 * fs) for i in range(1, len(km))]))))
        out[ids == keep] = km + np.int64(k)
        out[ids == stop] = sm + np.int64(s)
    return out


def epoch_events(cmd_onsets_s, labels, fs):
    """-> (sub-epoch start samples, event ids, groups). Labels are their event ids (1/2 keep/stop left, 3/4 right) in
    keep-stop pairs; under that pattern their clean_events removes nothing."""
    ids = np.asarray(labels, int)
    if len(ids) % 2 or not (np.isin(ids[0::2], (KEEP_L, KEEP_R)).all() and (ids[1::2] == ids[0::2] + 1).all()):
        raise ValueError('labels must be keep/stop pairs of one hand (1, 2 or 3, 4)')
    samp = np.round(np.asarray(cmd_onsets_s, float) * fs).astype(np.int64)
    off = shift_events(samp, ids, fs)
    ev = np.array([[s + r * fs * 10. / N_SEG, 0, i] for s, i in zip(off, ids) for r in range(N_SEG)], int)
    trial = np.cumsum(np.diff(np.r_[ev[0, 2], ev[:, 2]]) != 0)
    return ev[:, 0], ev[:, 2], np.array(trial / 2., int)



def _canon(names):
    return [ALIAS.get(str(c), str(c)) for c in names]


def features(x19, fs, cmd_onsets_s, labels, names):
    """x19: (channels, samples) in µV, rows named by `names` (T3-T6 accepted for T7/T8/P7/P8); onsets in s from the
    first sample; labels: event ids per command. -> X (n_epochs, 76), y (bool, keep = True), groups (trial index)."""
    nm = _canon(names)
    xV = np.asarray(x19, float)[[nm.index(c) for c in USE_CH]] * 1e-6
    xf = filter_data(xV, float(fs), FMIN, FMAX, filter_length='auto', l_trans_bandwidth='auto',
                     h_trans_bandwidth='auto', method='fir', phase='zero', fir_window='hamming',
                     fir_design='firwin', pad='reflect_limited', verbose=False)
    start, ids, groups = epoch_events(cmd_onsets_s, labels, fs)
    n_t = int(round(TMAX * fs)) + 1
    if start.min() < 0 or start.max() + n_t > xf.shape[1]:
        raise ValueError('an epoch falls outside the data (their pipeline would drop it)')
    ep = np.array([_csd(xf[:, s:s + n_t], *_CSD) for s in start])
    psd, freqs = psd_multitaper_111(ep, float(fs))
    X = np.zeros((len(ep), len(USE_CH), len(BANDS)))
    for ii, (lo, hi) in enumerate(BANDS):
        X[:, :, ii] = psd[:, :, np.where(np.logical_and(freqs >= lo, freqs <= hi))[0]].mean(2)
    return X.reshape(len(ep), -1), np.isin(ids, (KEEP_L, KEEP_R)), groups



def _as_rs(rng):
    if isinstance(rng, np.random.RandomState):
        return rng
    if isinstance(rng, np.random.Generator):
        return np.random.RandomState(int(rng.integers(2 ** 32)))
    return np.random.RandomState(rng)


def cv_scores(X, y, groups, rs):
    """Per-fold AUCs of StandardScaler + LinearSVC under LeaveOneGroupOut (folds in sorted group order); each fit draws
    its liblinear seed from rs, as their cross_val_score fits draw from the global RNG."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for g in np.unique(groups):
            te = groups == g
            sc = StandardScaler().fit(X[~te])
            m = LinearSVC(dual=True, max_iter=MAX_ITER, random_state=rs).fit(sc.transform(X[~te]), y[~te])
            out.append(roc_auc_score(y[te], m.decision_function(sc.transform(X[te]))))
    return np.array(out)


STOP_AT = 25


def _result(scores, perm, n_perm=None, stopped=False):
    """p from the permutations computed. When stopped early (stop_at reached), the full-run p is at least the
    returned p, which is already >= alpha, so the call is identical to the full run's."""
    score = scores.mean(0)
    perm = np.asarray(perm)
    n = len(perm) if not stopped else n_perm
    p = (np.sum(perm >= score) + 1.0) / (n + 1)
    return {'auc': float(score), 'se': float(scores.std(0) / np.sqrt(len(scores))), 'p': float(p),
            'call': bool(p < ALPHA), 'call_paper': bool(score > 0.5 and p < ALPHA), 'perm': perm,
            'stopped_early': bool(stopped), 'n_perm_run': int(len(perm))}


def _run_perms(scores, draw, n_perm, stop_at):
    """Permutation AUCs in RNG order; with stop_at, stop once stop_at of them reach the observed AUC."""
    score, perm, ge = scores.mean(0), [], 0
    for _ in range(n_perm):
        v = draw()
        perm.append(v)
        ge += v >= score
        if stop_at is not None and ge >= stop_at and len(perm) < n_perm:
            return perm, True
    return perm, False


def exact_test(X, y, groups, rng, n_perm=500, stop_at=None):
    """Their statistic and null: each permutation shuffles epoch order once and applies it to labels and fold ids
    together (X fixed). rng: RandomState (bit-exact with their np.random.seed), int seed or Generator."""
    rs = _as_rs(rng)
    X, y, groups = np.asarray(X, float), np.asarray(y, bool), np.asarray(groups)
    scores = cv_scores(X, y, groups, rs)

    def draw():
        order = list(range(len(y)))
        rs.shuffle(order)
        return cv_scores(X, y[order], groups[order], rs).mean(0)

    perm, stopped = _run_perms(scores, draw, n_perm, stop_at)
    return _result(scores, perm, n_perm, stopped)


def trialflip_labels(y, groups, rs):
    """Flip each trial's labels as a unit with probability 0.5."""
    _, g = np.unique(groups, return_inverse=True)
    flip = rs.randint(0, 2, size=g.max() + 1).astype(bool)
    return y ^ flip[g]


def trialflip_test(X, y, groups, rng, n_perm=500, stop_at=None):
    """The same statistic; null = trial-level keep/stop flips. Folds and the 5-epoch runs are kept."""
    rs = _as_rs(rng)
    X, y, groups = np.asarray(X, float), np.asarray(y, bool), np.asarray(groups)
    scores = cv_scores(X, y, groups, rs)
    perm, stopped = _run_perms(scores, lambda: cv_scores(X, trialflip_labels(y, groups, rs), groups, rs).mean(0),
                               n_perm, stop_at)
    return _result(scores, perm, n_perm, stopped)


def bh_calls(p, alpha=ALPHA):
    """Their multiple-recording rule (parse_cmd.read_file2): Benjamini-Hochberg over one patient's recordings."""
    return fdr_correction(np.asarray(p, float), alpha=alpha, method='indep')[0]



def schedule(start_s, cmd_s=CMD_S, block_gap_s=AFTER_STOP_S, first_hand='right'):
    """Command onsets (s) and event ids for one overlay: 6 blocks x 8 trials of keep then stop, hands alternating by
    block. Keep onset -> stop onset = cmd_s + 10 s; stop onset -> next keep onset = cmd_s + 15 s within a block and
    cmd_s + block_gap_s across blocks. Span (first onset to end of the last 10-s window) = 47 * (2 * cmd_s + 25) +
    2 * cmd_s + 20 s with the default block gap (1502.2 s)."""
    hands = [(KEEP_R, STOP_R), (KEEP_L, STOP_L)]
    if first_hand == 'left':
        hands = hands[::-1]
    t, on, lab = float(start_s), [], []
    for b in range(N_BLOCKS):
        keep, stop = hands[b % 2]
        for k in range(N_TRIALS):
            if k == 0 and b > 0:
                t += block_gap_s - AFTER_STOP_S
            on.append(t)
            lab.append(keep)
            t += cmd_s + AFTER_KEEP_S
            on.append(t)
            lab.append(stop)
            t += cmd_s + AFTER_STOP_S
    return np.array(on), np.array(lab, int)


def _smoke():
    import glob
    import os
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    on, lab = schedule(30.0)
    need_s = on[-1] + CMD_S + WIN_S + TMAX
    src = None
    for f in sorted(glob.glob(os.path.join(here, '..', 'output', 'bdsp', 'sig', '*.npz'))):
        z = np.load(f)
        if 'x19' in z.files and z['x19'].shape[1] / float(z['fs']) > need_s:
            src = f
            break
    if src is not None:
        x, fs, names = z['x19'], float(z['fs']), list(z['names19'])
        print('segment', os.path.basename(src), x.shape, fs)
    else:
        fs, names = 128.0, USE_CH
        x = np.random.RandomState(0).standard_normal((19, int(1800 * fs))) * 20.0
        print(f'no x19 segment longer than {need_s:.0f} s: synthetic white noise, 19 x 1800 s at 128 Hz')
    t0 = time.time()
    X, y, g = features(x, fs, on, lab, names)
    t1 = time.time()
    r1 = exact_test(X, y, g, np.random.RandomState(1))
    t2 = time.time()
    r2 = trialflip_test(X, y, g, np.random.RandomState(1))
    t3 = time.time()
    print(f'features {t1 - t0:.1f} s  X {X.shape}')
    print(f"exact_test {t2 - t1:.1f} s  auc {r1['auc']:.4f} p {r1['p']:.4f} call {r1['call']}")
    print(f"trialflip_test {t3 - t2:.1f} s  auc {r2['auc']:.4f} p {r2['p']:.4f} call {r2['call']}")


if __name__ == '__main__':
    import sys
    if '--smoke' in sys.argv:
        _smoke()
