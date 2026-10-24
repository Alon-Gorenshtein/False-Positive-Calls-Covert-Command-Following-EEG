"""Pieces shared by the ERP / stimulus pipeline modules of the null benchmark (pipe_mbvt, pipe_chennu13,
pipe_gibson16, pipe_sokoliuk21). Pipeline facts live in the modules and cite
the erp catalogue as 'cat. §n'; this file holds generic signal handling and Python ports of the toolbox
routines the authors' code calls (re-implemented from the toolbox sources, nothing imported from them):
- Generic helpers (Overlay, seeds, results, channel aliases, Monte Carlo tests, BH) come from motor_common.
- fir1 / eegfilt_fir1: MATLAB fir1 (Hamming, scaled) + filtfilt, EEGLAB eegfilt(..., 'fir1') default order
  3 * fix(fs / cutoff) (CH13 code dataimport.m l. 67-72).
- eegfiltnew: EEGLAB firfilt pop_eegfiltnew (Hamming windowed sinc via firws, transition-width heuristic, zero-phase
  by group-delay compensation with the first/last sample repeated, firfilt.m) (SOK21 code Script_1 l. 38-49).
- filtfilt_ba: MATLAB filtfilt(b, a) (odd reflection of 3 * (n - 1) samples, steady-state initial conditions).
- matlab_resample: MATLAB resample(x, p, q) (firls x Kaiser(beta 5), N = 10, upfirdn, delay trim) as FieldTrip
  ft_resampledata calls it (trial mean removed first and added back).
- Sample-exact epochs (t = 0 falls on a sample), GFP (spatial SD), triangulation neighbours for any montage."""
import math

import numpy as np
from scipy.signal import fftconvolve, firwin, lfilter, lfilter_zi, firls, upfirdn
from scipy.signal.windows import kaiser

from . import motor_common as mc

Overlay = mc.Overlay
EEG19 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
         'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']



def pick(rec, native, sub1020):
    """Rows of rec.x for the native montage when every native site is present, else the stated 10-20 substitute.
    Names are compared after the T3/T4/T5/T6 aliasing. -> (x float64 µV, names used, 'native' | '1020')."""
    return mc.pick(rec, mc.canon(native), mc.canon(sub1020))


_STD1005 = None


def std_positions(names):
    """MNE standard_1005 positions (m, head frame) of the channels: names matched case-insensitively, T3/T4/T5/T6
    read as T7/T8/P7/P8. Raises ValueError listing every channel without a position."""
    global _STD1005
    if _STD1005 is None:
        import mne
        pos = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
        _STD1005 = {k.lower(): np.asarray(v, float) for k, v in pos.items()}
    alias = {k.lower(): v.lower() for k, v in mc.ALIAS.items()}
    key = [alias.get(str(c).strip().lower(), str(c).strip().lower()) for c in names]
    missing = [str(c) for c, k in zip(names, key) if k not in _STD1005]
    if missing:
        raise ValueError(f"no standard_1005 position for channel(s): {', '.join(missing)}")
    return np.array([_STD1005[k] for k in key])


NEIGHBOUR_METHODS = ('delaunay', 'fieldtrip2018')
FIELDTRIP_COMMIT = '7f5a1c60fef80b457e2e1802ebf14e3fd6fe7c77'


def triangulation_neighbours(names, method='delaunay'):
    """Spatial neighbours for any montage from FieldTrip's 'triangulation' geometry: elproj's 'polar' projection of
    the MNE standard_1005 positions about the head-frame origin (cart2sph, then radius pi/2 - elevation).
    - 'delaunay': the edges of the Delaunay triangulation of that 2-D layout.
    - 'fieldtrip2018': FieldTrip ft_prepare_neighbours.m @ 7f5a1c60 (2018-01-19) as written: l. 206-211 stack the
      Delaunay triangles of the layout, of the layout with x halved and with y halved; compneighbstructfromtri
      (l. 313-347) marks every triangle edge, collects each channel's 3-D distances to its neighbours, and removes the
      neighbours farther than mean + 3 SD of all those distances (MATLAB std, n - 1).
    -> boolean (C, C) adjacency."""
    from scipy.spatial import Delaunay
    if method not in NEIGHBOUR_METHODS:
        raise ValueError(f'method must be one of {NEIGHBOUR_METHODS}, not {method!r}')
    pos = std_positions(names)
    x, y, z = pos.T
    az, el = np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))
    rho = np.pi / 2 - el
    xy = np.c_[rho * np.cos(az), rho * np.sin(az)]
    layouts = [xy] if method == 'delaunay' else [xy, xy * [0.5, 1.0], xy * [1.0, 0.5]]
    A = np.zeros((len(names), len(names)), bool)
    for L in layouts:
        for s in Delaunay(L).simplices:
            A[np.ix_(s, s)] = True
    np.fill_diagonal(A, False)
    if method == 'fieldtrip2018':
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        A &= d <= d[A].mean() + 3 * d[A].std(ddof=1)
    return A



def sample_epochs(x, fs, onsets_s, n_pre, n_post):
    """(n, channels, n_pre + n_post) epochs whose sample n_pre is the onset sample round(onset * fs)."""
    idx = np.round(np.asarray(onsets_s, float) * fs).astype(int)
    if idx.min() - n_pre < 0 or idx.max() + n_post > x.shape[-1]:
        raise ValueError('an epoch falls outside the data')
    off = np.arange(-n_pre, n_post)
    return x[:, idx[:, None] + off[None, :]].transpose(1, 0, 2)


def gfp(data):
    """Global field power: spatial standard deviation over channels (axis -2) at each sample."""
    return data.std(axis=-2)



def _const_pad_conv(x, b, left, right, n_left, n_right):
    xp = np.concatenate([np.repeat(left, n_left, axis=-1), x, np.repeat(right, n_right, axis=-1)], axis=-1)
    return fftconvolve(xp, np.asarray(b, float)[(None,) * (x.ndim - 1)], mode='valid', axes=-1)


def fir_filter_zi(b, x):
    """MATLAB filter(b, 1, x, zi * x(1)) with zi the step steady state: x preceded by len(b) - 1 copies of x[0]."""
    L = len(b)
    return _const_pad_conv(x, b, x[..., :1], x[..., -1:], L - 1, 0)


def filtfilt_fir(b, x):
    """MATLAB filtfilt(b, 1, x) along the last axis (FFT convolution; identical up to rounding)."""
    x = np.asarray(x, float)
    L = len(b)
    nfact = max(1, 3 * (L - 1))
    if x.shape[-1] <= nfact:
        raise ValueError('data shorter than 3 x filter order (MATLAB filtfilt restriction)')
    y = np.concatenate([2 * x[..., :1] - x[..., nfact:0:-1], x, 2 * x[..., -1:] - x[..., -2:-nfact - 2:-1]], axis=-1)
    y = fir_filter_zi(b, y)[..., ::-1]
    y = fir_filter_zi(b, y)[..., ::-1]
    return y[..., nfact:nfact + x.shape[-1]]


def filtfilt_ba(b, a, x):
    """MATLAB filtfilt(b, a, x) along the last axis (IIR): odd padding of 3 * (max(len) - 1), zi * first value."""
    x = np.asarray(x, float)
    nf = max(len(a), len(b))
    nfact = max(1, 3 * (nf - 1))
    y = np.concatenate([2 * x[..., :1] - x[..., nfact:0:-1], x, 2 * x[..., -1:] - x[..., -2:-nfact - 2:-1]], axis=-1)
    zi = lfilter_zi(b, a)
    y = lfilter(b, a, y, axis=-1, zi=zi * y[..., :1])[0][..., ::-1]
    y = lfilter(b, a, y, axis=-1, zi=zi * y[..., :1])[0][..., ::-1]
    return y[..., nfact:nfact + x.shape[-1]]


def fir1(order, wn, kind='low'):
    """MATLAB fir1(order, wn[, 'high']) with its default Hamming window and scaling; wn normalised to Nyquist."""
    return firwin(order + 1, wn, window='hamming', pass_zero=(kind == 'low'), scale=True, fs=2.0)


def eegfilt_fir1(x, fs, locutoff, hicutoff):
    """EEGLAB eegfilt(data, srate, locutoff, hicutoff, 0, 0, 0, 'fir1'): one of low-pass (locutoff 0) or high-pass
    (hicutoff 0); order minfac * fix(srate / cutoff) with minfac 3, minimum 15; filtfilt."""
    cut = locutoff if locutoff > 0 else hicutoff
    order = max(15, 3 * int(math.trunc(fs / cut)))
    if locutoff > 0 and hicutoff > 0:
        b = firwin(order + 1, [locutoff / (fs / 2), hicutoff / (fs / 2)], window='hamming', pass_zero=False, fs=2.0)
    elif locutoff > 0:
        order += order % 2
        b = fir1(order, locutoff / (fs / 2), 'high')
    else:
        b = fir1(order, hicutoff / (fs / 2), 'low')
    return filtfilt_fir(b, x)


def _hamming_firfilt(m):
    """firfilt windows('hamming', m): symmetric, built from its first half."""
    if m % 2:
        x = np.arange((m - 1) // 2 + 1) / (m - 1)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * x)
        return np.r_[w, w[-2::-1]]
    x = np.arange(m // 2) / (m - 1)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * x)
    return np.r_[w, w[::-1]]


def _fkernel(m, f, w):
    n = np.arange(-m / 2, m / 2 + 1)
    b = np.empty(len(n))
    b[n == 0] = 2 * np.pi * f
    nz = n != 0
    b[nz] = np.sin(2 * np.pi * f * n[nz]) / n[nz]
    b = b * w
    return b / b.sum()


def _fspecinv(b):
    b = -b
    b[(len(b) - 1) // 2] += 1
    return b


def firws(m, f, t='', w=None):
    """firfilt firws(m, f, t, w): windowed sinc of even order m, f normalised to Nyquist (1 or 2 edges)."""
    f = np.atleast_1d(np.asarray(f, float)) / 2
    w = _hamming_firfilt(m + 1) if w is None else w
    b = _fkernel(m, f[0], w)
    if len(f) == 1 and t == 'high':
        b = _fspecinv(b)
    if len(f) == 2:
        b = b + _fspecinv(_fkernel(m, f[1], w))
        if t != 'stop':
            b = _fspecinv(b)
    return b


def eegfiltnew_design(fs, locutoff, hicutoff, revfilt=0):
    """pop_eegfiltnew's default design (filtorder []). -> (b, edges, cutoffs, df)."""
    fny = fs / 2.0
    lo = None if not locutoff else locutoff
    hi = None if not hicutoff else hicutoff
    if hi is None:
        hi, lo, revfilt = lo, None, int(not revfilt)
    edges = np.sort([e for e in (lo, hi) if e is not None]).astype(float)
    if np.any((edges < 0) | (edges >= fny)):
        raise ValueError('Cutoff frequency out of range')
    maxtbw = edges.copy()
    if revfilt == 0:
        maxtbw[-1] = fny - edges[-1]
    elif len(edges) == 2:
        maxtbw = np.array([np.diff(edges)[0] / 2])
    maxdf = maxtbw.min()
    if revfilt == 1:
        df = min(max(maxdf * 0.25, 2.0), maxdf)
    else:
        df = min(max(edges[0] * 0.25, 2.0), maxdf)
    order = 3.3 / (df / fs)
    order = int(math.ceil(order / 2) * 2)
    df = 3.3 / order * fs
    dfa = {(0, 1): np.array([df]), (0, 2): np.array([-df, df]), (1, 1): np.array([-df]), (1, 2): np.array([df, -df])}
    cut = edges + dfa[(revfilt, len(edges))] / 2
    w = _hamming_firfilt(order + 1)
    if revfilt == 1:
        b = firws(order, cut / fny, 'high' if len(cut) == 1 else 'stop', w)
    else:
        b = firws(order, cut / fny, '', w)
    return b, edges, cut, df


def firfilt(x, b):
    """firfilt.m on continuous data without boundaries: zero-phase by group-delay compensation, the first and last
    samples repeated groupDelay times."""
    gd = (len(b) - 1) // 2
    x = np.asarray(x, float)
    return _const_pad_conv(x, b, x[..., :1], x[..., -1:], gd, gd)


def eegfiltnew(x, fs, locutoff, hicutoff, revfilt=0):
    b = eegfiltnew_design(fs, locutoff, hicutoff, revfilt)[0]
    return firfilt(x, b)


def matlab_resample(x, p, q, n=10, beta=5.0):
    """MATLAB resample(x, p, q) along the last axis (classic algorithm, default N = 10, Kaiser beta 5)."""
    g = math.gcd(int(p), int(q))
    p, q = int(p) // g, int(q) // g
    if p == q:
        return np.asarray(x, float)
    pq = max(p, q)
    fc = 1.0 / 2 / pq
    L = 2 * n * pq + 1
    h = firls(L, [0, 2 * fc, 2 * fc, 1], [1, 1, 0, 0], fs=2.0) * kaiser(L, beta)
    h = p * h / h.sum()
    lhalf = (L - 1) / 2
    nz = int(math.floor(q - (lhalf % q)))
    h = np.r_[np.zeros(nz), h]
    lhalf += nz
    delay = int(math.floor(math.ceil(lhalf) / q))
    lx = x.shape[-1]
    ly = int(math.ceil(lx * p / q))
    nz1 = 0
    while math.ceil(((lx - 1) * p + len(h) + nz1) / q) - delay < ly:
        nz1 += 1
    h = np.r_[h, np.zeros(nz1)]
    y = upfirdn(h, np.asarray(x, float), p, q, axis=-1)
    return y[..., delay:delay + ly]


def ft_resample(x, fs, fs_new):
    """FieldTrip ft_resampledata (cfg.method 'resample', detrend/demean 'no', baselinewindow 'all'): the trial mean
    is removed, MATLAB resample(x, fsres, fsorig) with [fsorig, fsres] = rat(fs / fs_new), the mean added back."""
    from fractions import Fraction
    fr = Fraction(fs / fs_new).limit_denominator(10 ** 6)
    bsl = np.asarray(x, float).mean(-1, keepdims=True)
    return matlab_resample(np.asarray(x, float) - bsl, fr.denominator, fr.numerator) + bsl



def bh_threshold(p, q=0.05):
    """Benjamini-Hochberg threshold (the largest p_(i) <= i q / V; 0 when none), as fdr.m's pthr (cV = 1)."""
    p = np.sort(np.asarray(p, float).ravel())
    V = len(p)
    ok = p <= np.arange(1, V + 1) / V * q
    return float(p[np.nonzero(ok)[0].max()]) if ok.any() else 0.0
