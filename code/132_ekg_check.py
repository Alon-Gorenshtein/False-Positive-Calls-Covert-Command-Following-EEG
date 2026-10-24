"""Arousal check (go/no-go only; design data): does slow EEG power non-stationarity track slow heart-rate fluctuation?
Built test-first on synthetic ECG and EEG (code/test_ekg_check.py).

Measures, per recording, on the same 1800-s window [start, start + 1800 s):
- A_HR (a_hr): R peaks by one fixed detector (r_peaks: EKG band-passed 5-25 Hz, zero-phase Butterworth order 4;
  absolute first difference; local maxima at least REFRACTORY_S = 0.3 s apart above a threshold fixed on its
  quantiles, PEAK_FRAC = 0.5 times its PEAK_Q = 0.995 quantile. A bare quantile cut is not used: the share of samples
  in the QRS grows with heart rate, so a bare 98% cut drops most beats above 120 bpm on clean synthetic ECG).
  RR intervals in 0.3-2.0 s give the instantaneous heart rate (60 / RR, bpm, at the closing beat); intervals outside
  are dropped. Linear interpolation onto a 4-Hz grid from the first to the last kept beat, mean removed, band-passed
  0.01-0.1 Hz (zero-phase Butterworth order 4); A_HR = log SD. NA, by the first failing rule in this order: fewer than
  90% of RR intervals in 0.3-2.0 s (or fewer than 2 beats) "rr_implausible"; the signal-quality guard of the doc's
  Clarification 2 (e4010bf): mean Pearson correlation of the beats (the 5-25 Hz detector signal, +/- 100 ms around each
  R peak, beats whose window leaves the signal not used) with their sample-wise median template below 0.6
  "low_template_correlation"; median heart rate (60 / median RR over all detected intervals) outside 30-200 bpm
  "median_hr_out_of_range".
- log VER_ref (ver_ref): C3 and C4 of the cached x19 window; 8-30 Hz (zero-phase Butterworth order 4 over the window)
  log mean power in consecutive 2-s epochs; per channel the variance over epochs (ddof 1) divided by its mean over 10
  multivariate phase-randomised surrogates of the raw C3/C4 window (128.surrogates: 117.phase_randomise, one random
  phase per frequency shared by both channels, seeded [20261014, crc32(uid), 0, window samples]); log, mean over the
  two channels (a channel with no variance left out; none: NA).
Recordings (select_rows, the doc's 2026-10-16 clarification, recomputed here and required to equal the extraction
manifest output/bdsp/ekg_design/manifest.csv, check_manifest): pool = the 500 sample_500 recordings; order =
default_rng([20261016, 3]).permutation of the sorted uids; a recording is taken if its cached npz lists exactly one
EKG/ECG channel (all_names) and its cached x19 holds an 1800-s window by 108.eligible_starts; start =
default_rng([20261016, 3, position]).choice(eligible starts); failures counted by reason; stop at 100. The EKG npz
(ekg, fs, start_s, channel, uid) must match its manifest row and hold exactly 1800 s; the x19 window is the cached
x19 over the same [start, start + 1800 s) (recording_measures).
Rule (fixed in the doc): Spearman rho between A_HR and log VER_ref over usable recordings, percentile 95% bootstrap CI
over recordings (2000 draws, default_rng(20261016)). NOGO if rho < 0.10 or the CI includes 0; GO if rho >= 0.10 and the
CI excludes 0. Only rho, CI, N usable and NA counts are reported.
-> output/tables/ekg_check.json; per recording output/intermediate/_ekg_check_rows.csv.
python3 132_ekg_check.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import importlib
import json
import math
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

ns = importlib.import_module("128_nonstationarity")
mc, en = ns.mc, ns.en
bn = importlib.import_module("108_bdsp_null")
LOG_FLOOR = ns.mech.LOG_FLOOR

R_BAND = (5.0, 25.0)
REFRACTORY_S = 0.3
PEAK_Q, PEAK_FRAC = 0.995, 0.5
RR_OK = (0.3, 2.0)
RR_MIN_SHARE = 0.9
HR_FS = 4.0
TEMPLATE_HALF_S, TEMPLATE_MIN = 0.1, 0.6
MEDIAN_HR = (30.0, 200.0)
HR_BAND = (0.01, 0.1)
EEG_CH = ("C3", "C4")
EEG_BAND = (8.0, 30.0)
EPOCH_S = 2.0
N_SURR = 10
BOOT_B, BOOT_SEED = 2000, 20261016
RHO_MIN = 0.10
SEED, STREAM, N_REC, WINDOW_S = 20261016, 3, 100, 1800
EKG_NAME = re.compile(r"EKG|ECG", re.I)
META = os.path.join(ROOT, "output", "bdsp", "meta")
SIG = os.path.join(ROOT, "output", "bdsp", "sig")
EKG_DIR = os.path.join(ROOT, "output", "bdsp", "ekg_design")
OUT_JSON = os.path.join(ROOT, "output", "tables", "ekg_check.json")
OUT_ROWS = os.path.join(ROOT, "output", "intermediate", "_ekg_check_rows.csv")



def _detect(ekg, fs):
    """(R-peak sample indices, the 5-25 Hz band-passed EKG)."""
    x = mc.bandpass(np.asarray(ekg, float), fs, *R_BAND)
    d = np.abs(np.diff(x))
    thr = PEAK_FRAC * np.quantile(d, PEAK_Q)
    if not thr > 0:
        return np.array([], int), x
    p, _ = find_peaks(d, height=thr, distance=max(1, int(math.ceil(REFRACTORY_S * fs))))
    return p, x


def r_peaks(ekg, fs):
    """Sample indices of the R peaks (module docstring)."""
    return _detect(ekg, fs)[0]


def template_corr(x, peaks, fs):
    """(mean Pearson correlation of the beats, +/- 100 ms of x around each R peak, with their sample-wise median
    template; beats used). Beats whose window leaves the signal are not used."""
    h = int(round(TEMPLATE_HALF_S * fs))
    P = [p for p in np.asarray(peaks, int) if p - h >= 0 and p + h + 1 <= len(x)]
    if len(P) < 2:
        return float("nan"), len(P)
    B = np.array([x[p - h:p + h + 1] for p in P], float)
    tpl = np.median(B, axis=0)
    Bc, tc = B - B.mean(1, keepdims=True), tpl - tpl.mean()
    den = np.sqrt((Bc ** 2).sum(1) * (tc ** 2).sum())
    r = np.where(den > 0, (Bc @ tc) / np.where(den > 0, den, 1.0), 0.0)
    return float(r.mean()), len(P)


def quality_reason(rr_ok_share, template_r, median_hr):
    """The first failing rule, in the order written (RR rule, beat template, median heart rate), or None."""
    if not rr_ok_share >= RR_MIN_SHARE:
        return "rr_implausible"
    if not template_r >= TEMPLATE_MIN:
        return "low_template_correlation"
    if not MEDIAN_HR[0] <= median_hr <= MEDIAN_HR[1]:
        return "median_hr_out_of_range"
    return None


def hr_series(peaks, fs):
    """(t, hr) on the 4-Hz grid from the RR intervals in 0.3-2.0 s, and the share of RR intervals in that range."""
    tb = np.asarray(peaks, float) / fs
    rr = np.diff(tb)
    if len(rr) == 0:
        return np.array([]), np.array([]), 0.0
    ok = (rr >= RR_OK[0]) & (rr <= RR_OK[1])
    share = float(ok.mean())
    t_k, hr_k = tb[1:][ok], 60.0 / rr[ok]
    if len(t_k) < 2:
        return np.array([]), np.array([]), share
    t = np.arange(t_k[0], t_k[-1] + 1e-9, 1.0 / HR_FS)
    return t, np.interp(t, t_k, hr_k), share


def a_hr(ekg, fs):
    """{"a_hr": log SD of the 0.01-0.1 Hz heart rate (bpm) or None, "reason", "n_beats", "rr_ok_share",
    "template_corr", "median_hr"}."""
    p, x = _detect(ekg, fs)
    t, hr, share = hr_series(p, fs)
    tc, _ = template_corr(x, p, fs)
    mhr = float(60.0 / np.median(np.diff(p) / fs)) if len(p) > 1 else float("nan")
    out = {"n_beats": int(len(p)), "rr_ok_share": share, "template_corr": tc, "median_hr": mhr}
    if len(hr) < 2:
        return dict(out, a_hr=None, reason="rr_implausible")
    why = quality_reason(share, tc, mhr)
    if why:
        return dict(out, a_hr=None, reason=why)
    h = mc.bandpass(hr - hr.mean(), HR_FS, *HR_BAND)
    sd = float(h.std())
    if not sd > 0:
        return dict(out, a_hr=None, reason="no_variance")
    return dict(out, a_hr=math.log(sd), reason=None)



def c3c4_log_power(x, fs):
    """(channels, epochs) 8-30 Hz log mean power in consecutive 2-s epochs of a (channels, samples) window."""
    xf = mc.bandpass(np.asarray(x, float), fs, *EEG_BAND)
    m = int(round(EPOCH_S * fs))
    k = xf.shape[1] // m
    return np.log(np.mean(xf[:, :k * m].reshape(xf.shape[0], k, m) ** 2, axis=2) + LOG_FLOOR)


def ver_ref(x19, names, fs, uid):
    """log VER_ref of one window (module docstring); x19: (19, samples) of the window."""
    seg = np.asarray(x19, float)[[list(names).index(c) for c in EEG_CH]]
    F = c3c4_log_power(seg, fs)
    v = F.var(1, ddof=1)
    vs = np.mean([c3c4_log_power(s, fs).var(1, ddof=1) for s in ns.surrogates(seg, uid, 0, seg.shape[1], N_SURR)], 0)
    ok = np.isfinite(v) & np.isfinite(vs) & (v > ns.MIN_VAR) & (vs > ns.MIN_VAR)
    out = {"n_epochs": int(F.shape[1]), "n_ch_used": int(ok.sum())}
    if not ok.any():
        return dict(out, log_ver_ref=None, reason="no_variance")
    return dict(out, log_ver_ref=float(np.mean(np.log(v[ok] / vs[ok]))), reason=None)



def spearman_ci(a, b, B=BOOT_B, seed=BOOT_SEED):
    """Spearman rho over the pairs with both values, and its percentile 95% bootstrap CI over recordings."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b)
    a, b = a[k], b[k]
    n = len(a)
    rho = float(spearmanr(a, b).correlation)
    idx = np.random.default_rng(seed).integers(0, n, (B, n))
    draws = np.array([spearmanr(a[i], b[i]).correlation for i in idx])
    draws = draws[np.isfinite(draws)]
    return {"rho": rho, "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], "n": int(n),
            "B": int(B), "seed": int(seed), "n_draws_finite": int(len(draws))}


def verdict(out):
    lo, hi = out["ci95"]
    go = out["rho"] >= RHO_MIN and (lo > 0 or hi < 0)
    return {"verdict": "GO" if go else "NOGO",
            "rule": "GO: rho >= 0.10 with the 95% CI excluding 0; otherwise NOGO"}


def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()



def select_rows(uids, load, n=N_REC):
    """([{position, uid, start_s, n_eligible_starts}], failure counts) by the clarified rule (module docstring)."""
    order = np.random.default_rng([SEED, STREAM]).permutation(np.array(sorted(uids)))
    rows, fail = [], Counter()
    for pos, uid in enumerate(order):
        if len(rows) == n:
            break
        z = load(str(uid))
        ekg = [str(c) for c in z["all_names"] if EKG_NAME.search(str(c))]
        if len(ekg) != 1:
            fail["no_ekg" if not ekg else "several_ekg"] += 1
            continue
        el = bn.eligible_starts(np.asarray(z["x19"], np.float32), float(z["fs"]), run_s=WINDOW_S)
        if not len(el):
            fail["no_30min_window"] += 1
            continue
        start = int(np.random.default_rng([SEED, STREAM, pos]).choice(el))
        rows.append({"position": pos, "uid": str(uid), "start_s": start, "n_eligible_starts": int(len(el))})
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} usable recordings in the pool")
    return rows, dict(fail)


def check_manifest(man, rows):
    """The extraction manifest must list exactly the recomputed recordings, positions and starts."""
    want = [(int(r["position"]), str(r["uid"]), int(r["start_s"])) for r in rows]
    got = [(int(p), str(u), int(s)) for p, u, s in zip(man["position"], man["uid"], man["start_s"])]
    if got != want:
        bad = [w for w, g in zip(want, got) if w != g][:3]
        raise RuntimeError(f"the EKG manifest differs from the recomputed selection ({len(got)} vs {len(want)} rows; "
                           f"first differences {bad})")


def recording_measures(row, ekg, sig):
    """A_HR and log VER_ref of one manifest row; ekg: its EKG npz; sig: its cached design npz."""
    uid, start, fs = str(row["uid"]), int(row["start_s"]), float(row["fs"])
    for k, want in (("uid", uid), ("start_s", start), ("fs", fs), ("channel", str(row["channel"]))):
        got = type(want)(np.asarray(ekg[k]).item() if np.ndim(ekg[k]) == 0 else ekg[k])
        if got != want:
            raise RuntimeError(f"{uid}: EKG npz {k} {got!r} differs from the manifest's {want!r}")
    e = np.asarray(ekg["ekg"], float)
    if len(e) != int(round(WINDOW_S * fs)):
        raise RuntimeError(f"{uid}: the EKG holds {len(e)} samples, not 1800 s at {fs} Hz")
    fx = float(np.asarray(sig["fs"]).item())
    i0, m = int(round(start * fx)), int(round(WINDOW_S * fx))
    x = np.asarray(sig["x19"])[:, i0:i0 + m]
    if x.shape[1] != m:
        raise RuntimeError(f"{uid}: the cached x19 does not hold 1800 s from {start} s")
    h, v = a_hr(e, fs), ver_ref(x, [str(c) for c in sig["names19"]], fx, uid)
    reason = ";".join(f"{k}:{r}" for k, r in (("a_hr", h["reason"]), ("ver", v["reason"])) if r) or None
    return {"position": int(row["position"]), "uid": uid, "a_hr": h["a_hr"], "log_ver_ref": v["log_ver_ref"],
            "n_beats": h["n_beats"], "rr_ok_share": h["rr_ok_share"], "template_corr": h["template_corr"],
            "median_hr": h["median_hr"], "n_epochs": v["n_epochs"], "reason": reason}


def _npz(path, keys=None):
    """The npz members named in keys (all if None); only those are read."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (keys or z.files)}


def main():
    man_path = os.path.join(EKG_DIR, "manifest.csv")
    man = pd.read_csv(man_path, dtype={"uid": str, "channel": str})
    s = pd.read_csv(os.path.join(META, "sample_500.csv"))
    uids = s["file"].str.replace(".mat", "", regex=False).tolist()
    rows, fail = select_rows(uids, lambda u: _npz(os.path.join(SIG, f"{u}.npz"), ("all_names", "x19", "fs")))
    check_manifest(man, rows)
    res = []
    for r in man.itertuples(index=False):
        res.append(recording_measures(r._asdict(), _npz(os.path.join(EKG_DIR, f"{r.uid}.npz")),
                                      _npz(os.path.join(SIG, f"{r.uid}.npz"), ("x19", "fs", "names19"))))
        print(f"{len(res)}/{len(man)} {r.uid}", flush=True)
    d = pd.DataFrame(res)
    os.makedirs(os.path.dirname(OUT_ROWS), exist_ok=True)
    d.to_csv(OUT_ROWS, index=False)
    out = spearman_ci(d["a_hr"].astype(float), d["log_ver_ref"].astype(float))
    out.update(verdict(out))
    out.update(n_recordings=int(len(d)), na_counts=d["reason"].dropna().value_counts().to_dict(),
               selection_failures=fail, manifest_sha256=hashlib.sha256(open(man_path, "rb").read()).hexdigest(),
               sha256_132=own_sha256(), label="arousal check (added later, go/no-go only; design data)")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"132: N usable {out['n']} of {len(d)} -> {OUT_JSON}")


if __name__ == "__main__":
    main()
