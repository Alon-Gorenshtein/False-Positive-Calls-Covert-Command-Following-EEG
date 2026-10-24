"""Pieces shared by the motor/command pipeline modules of the null benchmark. Pipeline facts live in the modules and cite the motor catalogue as
'cat. §n'; this file holds only generic signal handling and inference.
- Overlay: the engine's dataclass when code/bench/engine.py imports, else a field-identical stand-in (the engine is
  written in parallel; nothing here depends on its internals).
- Channels: 10-20 unit-sphere positions and the T3/T4/T5/T6 -> T7/T8/P7/P8 aliases of 112_columbia_exact (BDSP x19
  uses the old names); Hjorth Laplacian by great-circle arc.
- Randomization tests with decision-exact early stopping: 112.STOP_AT generalised to any B and alpha.
- Seeds: every module accepts a numpy Generator, a RandomState or an int and draws sklearn/libsvm seeds from it."""
import importlib.util
import itertools
import math
import os
import sys
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.stats import beta

try:
    from .engine import Overlay
except ImportError:
    @dataclass
    class Overlay:
        onsets_s: np.ndarray
        labels: np.ndarray
        kind: str
        meta: dict = field(default_factory=dict)

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_112():
    """code/112_columbia_exact.py as a module (its file name is not an identifier); one instance per process."""
    name = '112_columbia_exact'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(CODE, name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_CE = load_112()
ALIAS = dict(_CE.ALIAS)
POS = {k: np.asarray(v, float) for k, v in _CE._POS.items()}
EEG19 = list(_CE.USE_CH)



def cfg(variants, variant):
    """A variant key of VARIANTS, or a dict (tests pass reduced settings; every key the module reads must be set)."""
    if isinstance(variant, dict):
        return dict(variant)
    return dict(variants[variant])


def gen(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(int(rng.randint(2 ** 31)))
    return np.random.default_rng(rng)


def seed_of(rng):
    return int(gen(rng).integers(2 ** 31 - 1))


def result(stat, p, call, **extra):
    return {'stat': float(stat), 'p': None if p is None else float(p), 'call': bool(call), 'extra': extra}


class NoUsableTrials(ValueError):
    """Too few trials/epochs survive a module's artefact rule to compute its statistic (the module's documented
    minimum). The assessment is unusable, which is not the same as negative."""



def canon(names):
    return [ALIAS.get(str(c), str(c)) for c in names]


def pick(rec, native, sub1020):
    """Rows of rec.x for the native montage when every native site is present, else the stated 10-20 substitute
    (BDSP x19 has 10-20 sites only). -> (x float64 µV, names used, 'native' | '1020')."""
    nm = canon(rec.names)
    if all(c in nm for c in native):
        use, which = list(native), 'native'
    else:
        use, which = list(sub1020), '1020'
    return np.asarray(rec.x, float)[[nm.index(c) for c in use]], use, which


def hjorth_matrix(names, k_interior=4, k_edge=3, weights='inverse_arc'):
    """Hjorth Laplacian over 10-20 sites: row c = e_c - sum_j w_j e_j / sum_j w_j over the k nearest sites by
    great-circle arc; k_edge for the outer ring (z = 0 on the unit sphere), k_interior otherwise. w = 1/arc or 1."""
    nm = canon(names)
    P = np.array([POS[c] for c in nm])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    arc = np.arccos(np.clip(P @ P.T, -1.0, 1.0))
    L = np.eye(len(nm))
    for i in range(len(nm)):
        k = k_edge if abs(P[i, 2]) < 1e-6 else k_interior
        nb = [j for j in np.argsort(arc[i], kind='stable') if j != i][:k]
        w = 1.0 / arc[i, nb] if weights == 'inverse_arc' else np.ones(len(nb))
        L[i, nb] -= w / w.sum()
    return L



def bandpass(x, fs, lo, hi, order=4):
    """Zero-phase Butterworth along the last axis; lo None = low-pass, hi None = high-pass. hi is capped at 0.45 fs."""
    nyq = fs / 2.0
    if hi is not None:
        hi = min(hi, 0.45 * fs)
    if lo is None:
        sos = butter(order, hi / nyq, 'lowpass', output='sos')
    elif hi is None:
        sos = butter(order, lo / nyq, 'highpass', output='sos')
    else:
        sos = butter(order, [lo / nyq, hi / nyq], 'bandpass', output='sos')
    return sosfiltfilt(sos, x, axis=-1)


def resample(x, fs, fs_new):
    """Polyphase resampling (anti-aliased) along the last axis."""
    if abs(fs - fs_new) < 1e-9:
        return np.asarray(x, float)
    fr = Fraction(fs_new / fs).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def span(x, fs, t0, t1):
    """Continuous segment [t0, t1) s of x (channels x samples) and its first sample index."""
    i0, i1 = int(math.floor(t0 * fs)), int(math.ceil(t1 * fs))
    if i0 < 0 or i1 > x.shape[-1]:
        raise ValueError(f'overlay span {t0:.1f}-{t1:.1f} s falls outside the recording')
    return x[..., i0:i1], i0


def epochs(x, fs, onsets_s, t0, t1):
    """(n, channels, samples) epochs [onset + t0, onset + t1) with onsets in s from x's first sample."""
    n0, n = int(round(t0 * fs)), int(round((t1 - t0) * fs))
    idx = np.round(np.asarray(onsets_s, float) * fs).astype(int) + n0
    if idx.min() < 0 or idx.max() + n > x.shape[-1]:
        raise ValueError('an epoch falls outside the data')
    return np.stack([x[:, i:i + n] for i in idx])



def cp_interval(k, n, conf):
    """Clopper-Pearson interval (MATLAB binofit's method, cat. §1.4 [ext]); k may be non-integer (chance = n/2)."""
    a = 1.0 - conf
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def fdr_reject(p, alpha=0.05, method='bh'):
    """Benjamini-Hochberg ('bh') or Benjamini-Yekutieli ('by') rejections over a flat family."""
    p = np.asarray(p, float).ravel()
    m = len(p)
    c = 1.0 if method == 'bh' else np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p)
    ok = p[order] <= alpha * np.arange(1, m + 1) / (m * c)
    rej = np.zeros(m, bool)
    if ok.any():
        rej[order[:np.max(np.nonzero(ok)[0]) + 1]] = True
    return rej



def stop_count(n_perm, alpha, plus_one=True):
    """Number of null draws >= observed after which p >= alpha whatever the remaining draws give (p < alpha calls):
    p = (c + 1) / (B + 1) with plus_one, else c / B. B = 500, alpha = 0.05 -> 25 (= 112.STOP_AT)."""
    if plus_one:
        return int(math.ceil(alpha * (n_perm + 1) - 1e-9)) - 1
    return int(math.ceil(alpha * n_perm - 1e-9))


def perm_test(obs, draw, n_perm, alpha, plus_one=True, early_stop=True, tol=1e-12):
    """One-sided Monte Carlo test of obs against draw() values; draws run in RNG order. With early_stop the run
    ends once stop_count draws reach obs, which fixes the call at p >= alpha; the returned p is then a lower bound
    (as in 112._result). Fails safe: a non-finite null draw (NaN, +-inf) counts as an exceedance, so it can never
    lower p (NaN >= obs is False, which would count it as below), and is reported in n_nonfinite_null; a non-finite
    observed statistic raises NoUsableTrials. -> dict(p, null, n_run, stopped, n_nonfinite_null)."""
    obs = float(obs)
    if not np.isfinite(obs):
        raise NoUsableTrials(f'non-finite observed statistic ({obs})')
    stop = stop_count(n_perm, alpha, plus_one) if early_stop else None
    null, ge, bad = [], 0, 0
    for _ in range(n_perm):
        v = float(draw())
        null.append(v)
        if np.isfinite(v):
            ge += v >= obs - tol
        else:
            ge += 1
            bad += 1
        if stop is not None and ge >= stop and len(null) < n_perm:
            break
    stopped = len(null) < n_perm
    p = (ge + 1.0) / (n_perm + 1) if plus_one else ge / float(n_perm)
    return {'p': float(p), 'null': np.asarray(null), 'n_run': len(null), 'stopped': bool(stopped),
            'n_nonfinite_null': bad}


MAX_REDRAWS = 10000


def admissible_stat(draw_labels, ok, stat, cap=MAX_REDRAWS):
    """One null draw of a conditional randomization test: re-draw the labelling until ok(labels) (the module's
    minimum after artefact rejection), then return stat(labels). The rejected trials are fixed by the data, not by the
    labels, and the observed labelling meets the minimum, so the test is exact over the admissible labellings.
    Zero-filling an inadmissible draw instead biases p downward whenever the observed statistic is > 0."""
    for _ in range(cap):
        lab = draw_labels()
        if ok(lab):
            return stat(lab)
    raise NoUsableTrials(f'randomization space too small: no admissible labelling in {cap} draws')


def balanced_labels(classes, counts, rng):
    """Complete randomization: counts[i] copies of classes[i] in a uniformly random order."""
    return gen(rng).permutation(np.repeat(np.asarray(classes), counts))


def max_run(seq):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return 0
    change = np.r_[True, seq[1:] != seq[:-1], True]
    return int(np.max(np.diff(np.nonzero(change)[0])))


def constrained_order(classes, counts, max_same, rng, tries=100000):
    """Uniform over orders with at most max_same consecutive equal classes (rejection sampling)."""
    g = gen(rng)
    base = np.repeat(np.asarray(classes), counts)
    for _ in range(tries):
        s = g.permutation(base)
        if max_run(s) <= max_same:
            return s
    raise RuntimeError('no admissible order found')



_CACHE = {}


def memo(key, fn, size=4):
    """Cache fn() under key, scoped to fn's call site (file and line), so two modules that pass equal keys
    (same recording and onsets) can never read each other's features."""
    key = (fn.__code__.co_filename, fn.__code__.co_firstlineno, key)
    if key in _CACHE:
        return _CACHE[key]
    if len(_CACHE) >= size:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = fn()
    return _CACHE[key]


_OBJ_TOKENS = itertools.count(1)


def rec_token(rec):
    """The recording object's unique token: engine.Recording.token, else one attached on first use. Never id(): a
    freed injected copy's address is reused by the next copy."""
    t = getattr(rec, 'token', None)
    if t is not None:
        return ('rec', t)
    t = getattr(rec, '_bench_token', None)
    if t is None:
        t = next(_OBJ_TOKENS)
        setattr(rec, '_bench_token', t)
    return ('obj', t)


def rec_key(rec, *parts):
    """Recording identity: uid, the object's token (engine.Recording.token), shape and fs; then the parts."""
    return (getattr(rec, 'uid', None), rec_token(rec), np.shape(rec.x), float(rec.fs)) + tuple(
        p.tobytes() if isinstance(p, np.ndarray) else p for p in parts)
