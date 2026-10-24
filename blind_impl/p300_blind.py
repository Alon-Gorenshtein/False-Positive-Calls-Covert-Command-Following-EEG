"""Blind implementation of the vibrotactile P300 (VT3) assessment run.

Written from the method descriptions in Guger et al. 2018 (Front Neurosci 12:423) and Murovec et al. 2020 (Front
Neurosci 14:294) only.

Accuracy and the 23 % threshold are expressed as fractions in [0, 1].
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

PREFERRED_CHANNELS = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4"]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
SITES = ("left", "right", "distractor")
N_SPLITS = 10
N_SEGMENTS = 8
EPOCH = (-0.100, 0.600)
REJECT_UV = 100.0
THRESHOLD = 0.23
RATIO = 7
MIN_TARGETS = 4
RIDGE = 1e-6


def _canon(name):
    n = str(name).strip()
    n = n[:1].upper() + n[1:].lower()
    return ALIASES.get(n.upper(), n)


def _fail(msg, **extra):
    out = {"call": False, "accuracy": 0.0, "notes": msg}
    out.update(extra)
    return out


def _bandpass(x, fs, lo=0.1, hi=30.0):
    hi = min(hi, 0.45 * fs)
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _fisher_lda(Xt, Xn):
    """Fisher LDA direction (pooled within-class covariance)."""
    mt, mn = Xt.mean(0), Xn.mean(0)
    A = np.vstack([Xt - mt, Xn - mn])
    Sw = np.einsum("ni,nj->ij", A, A) / max(len(A) - 2, 1)
    d = Sw.shape[0]
    Sw = Sw + RIDGE * np.trace(Sw) / d * np.eye(d)
    w = np.linalg.solve(Sw, mt - mn)
    return w


def assess(eeg, fs, ch_names, events, seed=0, target="left"):
    eeg = np.asarray(eeg)
    if eeg.ndim != 2:
        raise ValueError("eeg must be 2-D (n_channels, n_samples)")
    if eeg.shape[0] != len(ch_names):
        raise ValueError("eeg rows (%d) != len(ch_names) (%d)"
                         % (eeg.shape[0], len(ch_names)))
    if target not in ("left", "right"):
        raise ValueError("target must be 'left' or 'right'")
    if not fs or fs <= 0:
        raise ValueError("fs must be positive")
    eeg = eeg.astype(np.float64)
    if not np.all(np.isfinite(eeg)):
        return _fail("EEG contains non-finite values; no call.")

    canon = [_canon(c) for c in ch_names]
    sel = [canon.index(c) for c in PREFERRED_CHANNELS if c in canon]
    if not sel:
        raise ValueError("none of %s found in ch_names" % PREFERRED_CHANNELS)
    used = [canon[i] for i in sel]

    x = _bandpass(eeg[sel], fs)

    pre = int(round(-EPOCH[0] * fs))
    n_ep = int(round((EPOCH[1] - EPOCH[0]) * fs))
    bounds = np.round(np.linspace(0, n_ep, N_SEGMENTS + 1)).astype(int)

    feats, is_target = [], []
    n_dropped_range = n_rejected = 0
    for ev in events:
        onset, site = ev[0], ev[1]
        if site not in SITES:
            continue
        i = int(round(onset * fs))
        a = i - pre
        b = a + n_ep
        if a < 0 or b > x.shape[1]:
            n_dropped_range += 1
            continue
        seg = x[:, a:b]
        seg = seg - seg[:, :pre].mean(axis=1, keepdims=True)
        if np.max(np.abs(seg)) > REJECT_UV:
            n_rejected += 1
            continue
        f = np.stack([seg[:, bounds[k]:bounds[k + 1]].mean(axis=1)
                      for k in range(N_SEGMENTS)], axis=1)
        feats.append(f.ravel())
        is_target.append(site == target)
    info = ("channels=%s; out-of-range=%d; artifact-rejected=%d"
            % (",".join(used), n_dropped_range, n_rejected))
    if not feats:
        return _fail("No usable epochs. " + info)
    F = np.asarray(feats)
    is_target = np.asarray(is_target)

    rng = np.random.default_rng(seed)
    t_idx = np.flatnonzero(is_target)
    n_idx = np.flatnonzero(~is_target)
    n_t = min(len(t_idx), len(n_idx) // RATIO)
    if n_t < MIN_TARGETS:
        return _fail("Too few usable trials (targets=%d, non-targets=%d). %s"
                     % (len(t_idx), len(n_idx), info))
    t_idx = rng.permutation(t_idx)[:n_t]
    n_idx = rng.permutation(n_idx)[:RATIO * n_t]

    curves = []
    for _ in range(N_SPLITS):
        tp = rng.permutation(t_idx)
        npool = rng.permutation(n_idx)
        h_t, h_n = n_t // 2, (RATIO * n_t) // 2
        tr_t, te_t = tp[:h_t], tp[h_t:]
        tr_n, te_n = npool[:h_n], npool[h_n:]
        w = _fisher_lda(F[tr_t], F[tr_n])
        s_t = np.einsum("nf,f->n", F[te_t], w)
        s_n = np.einsum("nf,f->n", F[te_n], w)
        K = min(len(s_t), len(s_n) // RATIO)
        if K < 1:
            continue
        ct = np.cumsum(s_t[:K]) / np.arange(1, K + 1)
        slots = np.stack([np.cumsum(s_n[j::RATIO][:K]) / np.arange(1, K + 1)
                          for j in range(RATIO)])
        curves.append((ct > slots.max(axis=0)).astype(float))
    if not curves:
        return _fail("Test pool too small. " + info)
    C = np.asarray(curves)
    mean_curve = C.mean(axis=0)
    acc = float(C.mean())
    final = float(mean_curve[-1])
    med = float(np.median(mean_curve))
    notes = ("%s; targets=%d non-targets=%d; accuracy = mean over curve points "
             "1..%d and %d splits (alt: end-of-curve %.3f, median-of-curve %.3f)"
             % (info, n_t, RATIO * n_t, C.shape[1], C.shape[0], final, med))
    return {"call": bool(acc > THRESHOLD), "accuracy": acc, "notes": notes}
