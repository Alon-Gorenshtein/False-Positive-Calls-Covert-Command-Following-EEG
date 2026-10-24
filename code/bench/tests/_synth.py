"""Synthetic recordings for the motor-module tests: a stand-in Recording with the engine's fields (so the tests do
not depend on engine.py), 1/f noise with a 10-Hz mu rhythm at central sites, optional slow amplitude drift, and a
planted ERD (8-30 Hz power drop after chosen onsets at chosen sites). Nothing here touches real EEG."""
from dataclasses import dataclass, field

import numpy as np

from bench import motor_common as mc

NAMES19 = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
           "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]


@dataclass
class FakeRec:
    source: str
    uid: str
    patient: str
    x: np.ndarray
    fs: float
    names: list
    meta: dict = field(default_factory=dict)


def noise(dur_s, fs=128.0, seed=0, drift=0.0, amp=10.0, names=NAMES19):
    """channels x samples µV: independent 1/f noise per channel, plus a 10-Hz rhythm at C3/Cz/C4 and their
    neighbours; drift > 0 multiplies each channel by exp(drift * slow Gaussian process, ~60-s scale)."""
    g = np.random.default_rng(seed)
    n = int(round(dur_s * fs))
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = np.where(f > 0, 1 / np.sqrt(np.maximum(f, 0.5)), 0.0)
    X = np.fft.irfft(np.fft.rfft(g.standard_normal((len(names), n)), axis=1) * shape, n, axis=1)
    X /= X.std(1, keepdims=True)
    mu = mc.bandpass(g.standard_normal((len(names), n)), fs, 9.0, 11.0, order=2)
    mu /= mu.std(1, keepdims=True)
    central = np.isin(mc.canon(names), ["C3", "Cz", "C4", "F3", "F4", "P3", "P4"])
    X += 1.0 * mu * central[:, None]
    if drift > 0:
        slow = mc.bandpass(g.standard_normal((len(names), n)), fs, None, 1 / 60.0, order=2)
        slow /= slow.std(1, keepdims=True)
        X *= np.exp(drift * slow)
    return (amp * X).astype(np.float32)


def rec(dur_s, fs=128.0, seed=0, drift=0.0, names=NAMES19):
    return FakeRec("synthetic", f"syn{seed}", f"p{seed}", noise(dur_s, fs, seed, drift, names=names), fs,
                   list(names), {})


def plant(r, onsets_s, sites, t0, t1, depth=0.7, band=(8.0, 30.0), ramp_s=0.3):
    """Copy of r with the band component at `sites` scaled by (1 - depth) over [onset + t0, onset + t1] (raised-
    cosine ramps), i.e. an ERD."""
    x = np.asarray(r.x, float).copy()
    nm = mc.canon(r.names)
    rows = [nm.index(s) for s in mc.canon(sites)]
    comp = mc.bandpass(x[rows], r.fs, band[0], band[1])
    env = np.zeros(x.shape[1])
    nr = max(1, int(ramp_s * r.fs))
    win = np.ones(int(round((t1 - t0) * r.fs)))
    win[:nr] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nr))
    win[-nr:] = win[:nr][::-1]
    for o in np.atleast_1d(onsets_s):
        i = int(round((o + t0) * r.fs))
        seg = slice(i, min(i + len(win), len(env)))
        env[seg] = np.maximum(env[seg], win[:seg.stop - seg.start])
    x[rows] -= depth * comp * env
    return FakeRec(r.source, r.uid + "+erd", r.patient, x.astype(np.float32), r.fs, list(r.names), dict(r.meta))


def pool_map(fn, args, workers=8):
    """Process-parallel map for the 200-draw validity tests (spawned workers re-import bench from sys.path)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, args))


def burst(r, t0s, dur_s, sites, amp=400.0, f_hz=10.0):
    """Copy of r with a large f_hz sinusoid (amp µV) added at `sites` over [t0, t0 + dur_s] for each t0: an artefact
    that every module's amplitude rule rejects."""
    x = np.asarray(r.x, float).copy()
    nm = mc.canon(r.names)
    rows = [nm.index(s) for s in mc.canon(sites)]
    n = int(round(dur_s * r.fs))
    w = amp * np.sin(2 * np.pi * f_hz * np.arange(n) / r.fs)
    for t0 in np.atleast_1d(t0s):
        i = int(round(t0 * r.fs))
        x[rows, i:i + n] += w[:max(0, min(n, x.shape[1] - i))]
    return FakeRec(r.source, r.uid + "+art", r.patient, x.astype(np.float32), r.fs, list(r.names), dict(r.meta))


def zero_fill(draw_labels, ok, stat, cap=None):
    """The rejected design (team-decision): an inadmissible re-draw contributes a statistic of 0 instead of being
    re-drawn. For the bias demonstrations only."""
    lab = draw_labels()
    return stat(lab) if ok(lab) else 0.0


def counting(orig, tally):
    """Wrap motor_common.admissible_stat so a worker can report the share of inadmissible draws it rejected."""
    def wrapped(draw_labels, ok, stat, cap=10000):
        def ok2(lab):
            r = ok(lab)
            tally['draws'] += 1
            tally['rejected'] += not r
            return r
        return orig(draw_labels, ok2, stat, cap)
    return wrapped


def usable_calls(run_one, seeds, n_target, mc_mod, max_tries=300):
    """Calls of run_one(seed) over successive seeds until n_target usable ones (NoUsableTrials = observed labelling
    below the minimum: skipped, the test is conditional on usability). Gives up after max_tries seeds: drifting noise
    can push extra trials over an artefact rule, leaving a recording that never meets the minimum."""
    out = []
    for s in list(seeds)[:max_tries]:
        try:
            out.append(run_one(s))
        except mc_mod.NoUsableTrials:
            continue
        if len(out) == n_target:
            break
    return out
