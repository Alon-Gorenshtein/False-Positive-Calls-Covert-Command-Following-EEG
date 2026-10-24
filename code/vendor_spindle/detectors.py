"""Six published spindle-detector families, one signature.

Each keeps its own published decision rule. They are deliberately NOT tuned to
agree; their disagreement is the measurement being reported. Every detector
takes a single-channel signal in microvolts and returns [onset_s, duration_s].

Rules implemented here:
  fixed_amplitude     Molle-like: fixed microvolt threshold on the sigma envelope.
  relative_threshold  Wamsley-like: percentile of the envelope within the record.
  rms_ferrarelli      Ferrarelli-like: mean-multiple threshold on a sliding RMS.
  a7_lacourse         Lacourse A7: four coupled criteria on log sigma power,
                      relative sigma power, and sigma covariance/correlation.
  wavelet             Complex Morlet magnitude at the sigma centre frequency.
  sigma_index         Peak sigma amplitude over surrounding-band amplitude.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, morlet2, welch

SIGMA_LO, SIGMA_HI = 11.0, 16.0
MIN_DUR, MAX_DUR = 0.4, 3.0


def sigma_filter(sig, fs, lo=SIGMA_LO, hi=SIGMA_HI):
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.95)
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def envelope(sig_filtered):
    return np.abs(hilbert(sig_filtered))


def _moving(x, w):
    """Centred moving average over w samples, length preserved."""
    w = max(int(w), 1)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def runs_to_events(mask, fs, min_dur=MIN_DUR, max_dur=MAX_DUR):
    """Contiguous True runs -> [onset_s, duration_s], duration-filtered."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros((0, 2))
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    out = []
    for s, e in zip(starts, ends):
        dur = (e - s) / fs
        if min_dur <= dur <= max_dur:
            out.append((s / fs, dur))
    return np.array(out, dtype=float).reshape(-1, 2)


def _merge_close(events, gap_s=0.1):
    """Merge events separated by less than gap_s, then re-apply duration bounds."""
    if len(events) < 2:
        return np.asarray(events, dtype=float).reshape(-1, 2)
    events = np.asarray(events, dtype=float)
    events = events[np.argsort(events[:, 0])]
    merged = [list(events[0])]
    for onset, dur in events[1:]:
        prev_end = merged[-1][0] + merged[-1][1]
        if onset - prev_end < gap_s:
            merged[-1][1] = (onset + dur) - merged[-1][0]
        else:
            merged.append([onset, dur])
    arr = np.array(merged, dtype=float)
    keep = (arr[:, 1] >= MIN_DUR) & (arr[:, 1] <= MAX_DUR)
    return arr[keep].reshape(-1, 2)


def fixed_amplitude(sig, fs, thresh_uv=8.0):
    env = envelope(sigma_filter(sig, fs))
    env = _moving(env, int(0.1 * fs))
    return _merge_close(runs_to_events(env > thresh_uv, fs))


def relative_threshold(sig, fs, pct=95.0):
    env = envelope(sigma_filter(sig, fs))
    env = _moving(env, int(0.1 * fs))
    thr = np.percentile(env, pct)
    if not np.isfinite(thr) or thr <= 0:
        return np.zeros((0, 2))
    return _merge_close(runs_to_events(env > thr, fs))


def rms_ferrarelli(sig, fs, mult=4.5):
    f = sigma_filter(sig, fs)
    w = max(int(0.25 * fs), 1)
    rms = np.sqrt(np.maximum(_moving(f ** 2, w), 0.0))
    base = np.mean(rms)
    if base <= 0:
        return np.zeros((0, 2))
    return _merge_close(runs_to_events(rms > mult * base, fs))


def a7_lacourse(sig, fs, scale=1.0, z_abs=1.25, z_rel=1.6, z_cov=1.3,
                corr_min=0.69):
    """A7: absolute sigma power, relative sigma power, sigma covariance and
    sigma correlation must all clear their thresholds simultaneously.

    `scale` multiplies the three Z-SCORE thresholds, preserving their ratios. It
    is the single tunable used by the DREAMS calibration, because loosening one
    criterion alone leaves the others binding.

    `corr_min` is deliberately NOT scaled. It is a Pearson correlation bounded in
    [-1, 1], not a z-score, so multiplying it by a z-threshold scale factor is a
    category error: at scale 0.625 the published 0.69 criterion silently became
    0.431, which is a different operating point rather than a rescaling of the
    published one.
    """
    z_abs = z_abs * scale
    z_rel = z_rel * scale
    z_cov = z_cov * scale
    f = sigma_filter(sig, fs)
    w = max(int(0.3 * fs), 1)
    p_sigma = _moving(f ** 2, w)
    with np.errstate(divide="ignore"):
        log_abs = np.log10(np.maximum(p_sigma, 1e-12))
    nyq = fs / 2.0
    b, a = butter(4, [4.5 / nyq, min(30.0, nyq * 0.95) / nyq], btype="band")
    broad = filtfilt(b, a, sig)
    p_broad = _moving(broad ** 2, w)
    rel = p_sigma / np.maximum(p_broad, 1e-12)
    sig_c = sig - _moving(sig, w)
    f_c = f - _moving(f, w)
    cov = _moving(sig_c * f_c, w)
    denom = np.sqrt(np.maximum(_moving(sig_c ** 2, w), 1e-12) *
                    np.maximum(_moving(f_c ** 2, w), 1e-12))
    corr = cov / denom

    def _z(x):
        mu, sd = np.mean(x), np.std(x)
        return (x - mu) / sd if sd > 0 else np.zeros_like(x)

    mask = (_z(log_abs) > z_abs) & (_z(rel) > z_rel) & \
           (_z(cov) > z_cov) & (corr > corr_min)
    return _merge_close(runs_to_events(mask, fs))


def wavelet(sig, fs, centre=13.5, n_cycles=7.0, mult=3.0):
    w = n_cycles
    s = w * fs / (2.0 * np.pi * centre)
    width = int(min(max(8 * s, 32), max(sig.size // 2, 32)))
    if width % 2 == 0:
        width += 1
    kernel = morlet2(width, s, w=w)
    mag = np.abs(np.convolve(sig, kernel, mode="same"))[:sig.size]
    mag = _moving(mag, int(0.1 * fs))
    base = np.median(mag)
    if base <= 0:
        return np.zeros((0, 2))
    return _merge_close(runs_to_events(mag > mult * base, fs))


def sigma_index(sig, fs, thresh=4.5, win_s=1.0):
    """Sliding sigma index: peak 11-16 Hz amplitude divided by the mean of the
    4-10 Hz and 20-40 Hz peak amplitudes in the same window."""
    n = sig.size
    step = max(int(0.1 * fs), 1)
    w = max(int(win_s * fs), int(fs))
    idx = np.zeros(n)
    nperseg = min(w, max(int(fs), 64))
    for a in range(0, max(n - w, 0) + 1, step):
        seg = sig[a:a + w]
        if seg.size < nperseg:
            break
        fr, P = welch(seg, fs=fs, nperseg=nperseg)
        amp = np.sqrt(np.maximum(P, 0))
        s_band = amp[(fr >= SIGMA_LO) & (fr <= SIGMA_HI)]
        lo_band = amp[(fr >= 4.0) & (fr <= 10.0)]
        hi_band = amp[(fr >= 20.0) & (fr <= 40.0)]
        if s_band.size == 0 or lo_band.size == 0 or hi_band.size == 0:
            continue
        ref = (lo_band.max() + hi_band.max()) / 2.0
        val = s_band.max() / ref if ref > 0 else 0.0
        idx[a:a + step] = val
    return _merge_close(runs_to_events(idx > thresh, fs))


DETECTORS = {
    "fixed_amplitude": fixed_amplitude,
    "relative_threshold": relative_threshold,
    "rms_ferrarelli": rms_ferrarelli,
    "a7_lacourse": a7_lacourse,
    "wavelet": wavelet,
    "sigma_index": sigma_index,
}


# ---------------------------------------------------------------------------
# Calibrated detectors
# ---------------------------------------------------------------------------
# code/05_calibrate.py tunes exactly one constant per detector so its event rate
# on the eight DREAMS excerpts matches expert 1's. Those constants are frozen
# here and applied unchanged to comatose EEG, so any disagreement observed in
# coma cannot be attributed to thresholds that were mis-set for real sleep.

import json as _json
import os as _os
import functools as _functools

_CAL_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "calibration.json")


def load_calibration():
    if not _os.path.exists(_CAL_PATH):
        return {}
    return _json.load(open(_CAL_PATH))


def calibrated_detectors():
    """DETECTORS with the frozen DREAMS-calibrated constant bound in."""
    cal = load_calibration()
    out = {}
    for name, fn in DETECTORS.items():
        if name in cal:
            out[name] = _functools.partial(
                fn, **{cal[name]["param"]: cal[name]["value"]})
        else:
            out[name] = fn
    return out
