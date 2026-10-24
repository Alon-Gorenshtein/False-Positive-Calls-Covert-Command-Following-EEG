"""Blind implementation of the Phase I (assessment) offline single-run
analysis and advancement criterion of du Bois et al. 2026 (Commun Med).

Written from the published Methods text only. DA and the 70 % threshold are fractions in [0, 1].

Pipeline (per run):
  6 band-pass bands -> epochs -> per-fold CSP (3 pairs/band) -> log-variance
  in a 1 s or 2 s sliding window (40 ms step) -> mutual-information ranking
  (3 quantisation levels) -> top 4..10 features -> regularised LDA, trained
  and tested separately at every time point -> six-fold CV -> time-resolved
  DA -> 200 ms smoothing -> peak DA in baseline [-1, 0] s and task
  [0.4, 7] s -> one-tailed paired t-test across folds.
  The configuration (channel set x band set x window x n_features) with the
  highest task peak DA is the one reported.
"""
import itertools

import numpy as np
from scipy import linalg as sla
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 18.0), (18.0, 28.0),
         (28.0, 40.0)]
BAND_SETS = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
MOTOR_SET = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
WINDOWS = (1.0, 2.0)
N_FEATURES = tuple(range(4, 11))
N_PAIRS = 3
N_FOLDS = 6
STEP_S = 0.040
BASELINE = (-1.0, 0.0)
TASK = (0.4, 7.0)
SMOOTH_S = 0.200
REFINE_S = 0.150
MI_LEVELS = 3
LDA_SHRINK = 0.1
PEAK_THRESHOLD = 0.70
ALPHA = 0.05
MIN_PER_CLASS = 6
_LAST = {}
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def _canon(name):
    n = str(name).strip()
    n = n[:1].upper() + n[1:].lower()
    return ALIASES.get(n.upper(), n)


def _fail(msg, peak=0.0, p=1.0):
    return {"call": False, "peak_da": float(peak), "p_t": float(p), "notes": msg}


def _bandpass(x, fs, lo, hi):
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _csp(C1, C2, n_pairs):
    """Rows = spatial filters: n_pairs from each end of the spectrum."""
    d = C1.shape[0]
    reg = 1e-6 * np.eye(d)
    evals, evecs = sla.eigh(C1 + reg, C1 + C2 + 2 * reg)
    order = np.argsort(evals)
    pick = np.r_[order[:n_pairs], order[-n_pairs:]]
    return evecs[:, pick].T


def _mutual_info(X, y):
    """MI between each feature and the binary label after quantising the
    feature into MI_LEVELS equal-frequency bins.
    X: (T, n, F); y: (n) in {0,1}.  Returns (T, F)."""
    T, n, F = X.shape
    ranks = np.argsort(np.argsort(X, axis=1), axis=1)
    bins = np.minimum((ranks * MI_LEVELS) // n, MI_LEVELS - 1)
    mi = np.zeros((T, F))
    py = np.array([np.mean(y == 0), np.mean(y == 1)])
    for b in range(MI_LEVELS):
        ind = (bins == b)
        for c in (0, 1):
            pbc = ind[:, y == c, :].sum(axis=1) / n
            pb = ind.sum(axis=1) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                term = pbc * np.log(pbc / (pb * py[c]))
            mi += np.nan_to_num(term)
    return mi


def _rlda_predict(Xtr, ytr, Xte, shrink=LDA_SHRINK):
    """Batched regularised LDA.  Xtr (T,ntr,k), Xte (T,nte,k) -> (T,nte) in
    {0,1}."""
    m0 = Xtr[:, ytr == 0].mean(axis=1)
    m1 = Xtr[:, ytr == 1].mean(axis=1)
    A0 = Xtr[:, ytr == 0] - m0[:, None]
    A1 = Xtr[:, ytr == 1] - m1[:, None]
    S = (np.einsum("tni,tnj->tij", A0, A0) + np.einsum("tni,tnj->tij", A1, A1))
    S /= max(len(ytr) - 2, 1)
    k = S.shape[1]
    tr = np.trace(S, axis1=1, axis2=2) / k
    S = (1 - shrink) * S + shrink * (tr[:, None, None] + 1e-12) * np.eye(k)
    w = np.linalg.solve(S, (m1 - m0)[:, :, None])[:, :, 0]
    b = -np.einsum("tk,tk->t", w, (m0 + m1) / 2)
    score = np.einsum("tnk,tk->tn", Xte, w) + b[:, None]
    return (score > 0).astype(np.int8)


def _folds(y, k, rng):
    fold = np.empty(len(y), dtype=int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def _peak(curve, lo, hi, step_n, smooth_n, refine_n):
    """Peak extraction on curve[lo:hi+1]; returns (index, value)."""
    seg = curve[lo:hi + 1]
    sm = uniform_filter1d(seg, size=smooth_n, mode="nearest")
    i = int(np.argmax(sm))
    a, b = max(0, i - refine_n), min(len(seg) - 1, i + refine_n)
    j = a + int(np.argmax(seg[a:b + 1]))
    return lo + j, float(seg[j]), float(sm[i])


def assess(eeg, fs, ch_names, events, seed=0):
    eeg = np.asarray(eeg)
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2-D (n_channels, n_samples)")
    if eeg.shape[0] != len(ch_names):
        raise ValueError("eeg rows (%d) != len(ch_names) (%d)"
                         % (eeg.shape[0], len(ch_names)))
    if not fs or fs <= 0:
        raise ValueError("fs must be positive")
    eeg = eeg.astype(np.float64)
    if not np.all(np.isfinite(eeg)):
        return _fail("EEG contains non-finite values; no call.")
    rng = np.random.default_rng(seed)
    n_ch, n_samp = eeg.shape
    canon = [_canon(c) for c in ch_names]
    motor_idx = [canon.index(c) for c in MOTOR_SET if c in canon]
    chsets = {"full": list(range(n_ch))}
    if 2 <= len(motor_idx) < n_ch:
        chsets["motor"] = motor_idx
    if n_ch < 2:
        return _fail("Need at least 2 channels for CSP.")

    nyq = fs / 2.0
    valid_bands = [i for i, (lo, hi) in enumerate(BANDS) if hi <= 0.95 * nyq]
    band_sets = [bs for bs in BAND_SETS if all(b in valid_bands for b in bs)]
    if not band_sets:
        return _fail("Sampling rate %.1f Hz too low for the filter bank." % fs)
    used_bands = sorted({b for bs in band_sets for b in bs})

    step = max(1, int(round(STEP_S * fs)))
    step_t = step / fs
    off_lo = int(round(BASELINE[0] * fs))
    off_hi = int(round(TASK[1] * fs))
    offs = np.arange(off_lo, off_hi + 1, step)
    times = offs / fs
    base_mask = (times >= BASELINE[0] - 1e-9) & (times <= BASELINE[1] + 1e-9)
    task_mask = (times >= TASK[0] - 1e-9) & (times <= TASK[1] + 1e-9)
    max_win = int(round(max(WINDOWS) * fs))
    pre_n = max_win - off_lo
    ep_start = off_lo - max_win + 1
    ep_len = off_hi - ep_start + 1

    onsets, labels = [], []
    n_bad = 0
    for ev in events:
        if len(ev) < 2 or ev[1] not in ("left", "right"):
            continue
        s0 = int(round(ev[0] * fs))
        if s0 + ep_start < 0 or s0 + off_hi >= n_samp:
            n_bad += 1
            continue
        onsets.append(s0)
        labels.append(0 if ev[1] == "left" else 1)
    onsets = np.asarray(onsets, dtype=int)
    y = np.asarray(labels, dtype=int)
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    if min(n0, n1) < MIN_PER_CLASS:
        return _fail("Too few trials per class (left=%d, right=%d, "
                     "dropped-out-of-range=%d); need >= %d each."
                     % (n0, n1, n_bad, MIN_PER_CLASS))
    n_tr = len(y)
    k_folds = min(N_FOLDS, n0, n1)
    fold = _folds(y, k_folds, rng)

    c_a, c_b = int(round(TASK[0] * fs)), int(round(TASK[1] * fs))

    ep = {}
    cov = {}
    idx_ep = onsets[:, None] + ep_start + np.arange(ep_len)[None, :]
    for b in used_bands:
        lo, hi = BANDS[b]
        xb = _bandpass(eeg, fs, lo, hi)
        E = xb[:, idx_ep]
        E = np.ascontiguousarray(E.transpose(1, 0, 2))
        ep[b] = E
        seg = E[:, :, c_a - ep_start:c_b - ep_start + 1]
        cov[b] = np.einsum("tcl,tdl->tcd", seg, seg) / seg.shape[2]

    end_pos = offs - ep_start

    results = {}
    cfgs = []
    for cs, bs, win, nf in itertools.product(chsets, band_sets, WINDOWS,
                                             N_FEATURES):
        cfgs.append((cs, bs, win, nf))
        results[(cs, bs, win, nf)] = np.zeros((k_folds, len(offs)))

    for f in range(k_folds):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        ytr, yte = y[tr], y[te]
        feat = {}
        for cs, cidx in chsets.items():
            ci = np.asarray(cidx)
            for b in used_bands:
                Cb = cov[b][:, ci][:, :, ci]
                Cb = Cb / np.trace(Cb, axis1=1, axis2=2)[:, None, None]
                C1 = Cb[tr][ytr == 0].mean(axis=0)
                C2 = Cb[tr][ytr == 1].mean(axis=0)
                W = _csp(C1, C2, N_PAIRS)
                Y = np.einsum("fc,tcl->tfl", W, ep[b][:, ci, :])
                cs1 = np.concatenate([np.zeros(Y.shape[:2] + (1,)),
                                      np.cumsum(Y, axis=2)], axis=2)
                cs2 = np.concatenate([np.zeros(Y.shape[:2] + (1,)),
                                      np.cumsum(Y * Y, axis=2)], axis=2)
                for win in WINDOWS:
                    wn = int(round(win * fs))
                    e = end_pos + 1
                    s1 = cs1[:, :, e] - cs1[:, :, e - wn]
                    s2 = cs2[:, :, e] - cs2[:, :, e - wn]
                    var = s2 / wn - (s1 / wn) ** 2
                    feat[(cs, b, win)] = np.log(np.maximum(var, 1e-12)
                                                ).transpose(0, 2, 1)
        for cs in chsets:
            for bs in band_sets:
                for win in WINDOWS:
                    F = np.concatenate([feat[(cs, b, win)] for b in bs],
                                       axis=2)
                    Xtr = np.ascontiguousarray(F[tr].transpose(1, 0, 2))
                    Xte = np.ascontiguousarray(F[te].transpose(1, 0, 2))
                    mi = _mutual_info(Xtr, ytr)
                    rank = np.argsort(-mi, axis=1)[:, :max(N_FEATURES)]
                    Str = np.take_along_axis(Xtr, rank[:, None, :], axis=2)
                    Ste = np.take_along_axis(Xte, rank[:, None, :], axis=2)
                    for nf in N_FEATURES:
                        pred = _rlda_predict(Str[:, :, :nf], ytr,
                                             Ste[:, :, :nf])
                        results[(cs, bs, win, nf)][f] = \
                            (pred == yte[None, :]).mean(axis=1)

    _LAST.clear()
    _LAST.update(results=results, times=times)

    smooth_n = max(1, int(round(SMOOTH_S / step_t)))
    refine_n = max(0, int(round(REFINE_S / step_t)))
    b_lo, b_hi = np.flatnonzero(base_mask)[[0, -1]]
    t_lo, t_hi = np.flatnonzero(task_mask)[[0, -1]]
    best = None
    for key in cfgs:
        da = results[key]
        mean_curve = da.mean(axis=0)
        ti, tv, ts = _peak(mean_curve, t_lo, t_hi, step, smooth_n, refine_n)
        if best is None or (tv, ts) > (best[1], best[2]):
            best = (key, tv, ts, ti)
    key, task_peak, _, ti = best
    da = results[key]
    bi, base_peak, _ = _peak(da.mean(axis=0), b_lo, b_hi, step, smooth_n,
                             refine_n)
    task_f, base_f = da[:, ti], da[:, bi]
    diff = task_f - base_f
    if np.allclose(diff, diff[0]) and np.isclose(diff[0], 0.0):
        p = 1.0
    else:
        t_stat, p = stats.ttest_rel(task_f, base_f, alternative="greater")
        p = 1.0 if not np.isfinite(p) else float(p)
    call = bool((task_peak > PEAK_THRESHOLD) or (p < ALPHA))
    cs, bs, win, nf = key
    notes = ("n_trials=%d (left %d / right %d), folds=%d, dropped-out-of-range=%d; "
             "selected: chans=%s bands=%s window=%.0fs n_feat=%d; "
             "task peak DA=%.3f at %.2fs, baseline peak DA=%.3f at %.2fs; "
             "one-tailed paired t over folds p=%.4f; "
             "rule: peak>%.2f or p<%.2f"
             % (n_tr, n0, n1, k_folds, n_bad, cs,
                "+".join(["%g-%g" % BANDS[b] for b in (bs[0], bs[-1])]),
                win, nf, task_peak, times[ti], base_peak, times[bi], p,
                PEAK_THRESHOLD, ALPHA))
    return {"call": call, "peak_da": float(task_peak), "p_t": float(p),
            "notes": notes}
