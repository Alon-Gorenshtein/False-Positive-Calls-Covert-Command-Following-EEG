"""Arousal coupling. Built test-first on synthetic recordings, ECG and rows (code/test_arousal.py). The heart-rate
marker and the EEG index are the arousal check's own functions (132_ekg_check: R-peak detector, signal-quality
guard, log VER_ref), imported, not copied.

Arousal coupling:, part D (52c27ea; added later). Built test-first on
synthetic recordings, ECG and rows (code/test_arousal.py). The heart-rate marker and the EEG index are the check's
own functions (132_ekg_check: R-peak detector, signal-quality guard of the check doc's Clarification 2, log
VER_ref), imported, not copied.

Per window (window_measures), on [start, start + dur) of an engine Recording (x19 at 128 Hz, Morgoth labels) and the
EKG over the same span (dur x its rate samples, refused otherwise):
- A_HR = 132.a_hr (NA by reason); log VER_ref = 132.ver_ref (C3/C4, 8-30 Hz log power in 2-s epochs, over 10
  multivariate phase-randomised surrogates seeded by uid).
- Covariates: Morgoth majority stratum and transitioning = engine.stratum (108.run_stratum over the window); artifact
  share = engine.artifact_index (flat, saturated or high-frequency seconds; recording-wide median as in 114).
- EEG-derived secondary markers (circular, labelled so): Morgoth transitions per 10 min (morgoth_transitions: label
  changes between consecutive valid labels at non-overlapping 10-s steps start, start + 10, ..., x 600 / dur); O1/O2
  alpha envelope variance (alpha_marker: mean of O1 and O2; 8-12 Hz zero-phase band-pass, Hilbert amplitude averaged
  to 4 Hz, band-passed 0.01-0.1 Hz, log variance; NA "no_posterior_alpha_peak" unless the Welch spectrum (4-s Hann
  segments) of mean(O1, O2) has a local maximum in 8-12 Hz at least 0.1 log10 units above a least-squares log-log line
  over 2-30 Hz fitted with 7-14 Hz left out).
- HR-envelope coherence (hr_envelope_coherence, descriptive): the 4-Hz heart rate of 132 (RR in 0.3-2.0 s) read at
  the 2-s epoch centres and the C3/C4-mean 8-30 Hz log power per 2-s epoch (132.c3c4_log_power), over the epochs
  inside the heart-rate span, both demeaned; Welch magnitude-squared coherence (256-s segments, 50% overlap), mean
  over 0.01-0.1 Hz; null: 100 circular shifts of the heart-rate series by 60 epochs (120 s) to n - 60, generator
  default_rng([20261016, crc32(uid)]); observed, null mean and null 95th percentile. Only where A_HR is usable.

D1 model (d1_model): complete-case OLS, z(log VER_ref) ~ z(marker) + Morgoth stratum dummies (reference: the most
frequent level, ties alphabetical; "mixed" and "unlabelled" are levels) + artifact share, z by the model sample (ddof
0), so the marker's coefficient is the standardised beta; HC3 robust SE; one-sided lower 95% bound beta - 1.645 SE.
Bar (d1_bar, primary marker A_HR on the fresh 400): beta >= 0.15 and the one-sided lower bound > 0.
Mediation (mediation, descriptive): on the C2 rows (call, pipeline-variant, patient; 116._prep weights), the total
model call ~ z(A_HR) + FE and the direct model call ~ z(A_HR) + z(log VER) + FE, z weighted (116.zscore), each fit
penalised as 116.fe_model (116.fe_logit, C = 100 on covariates / weighted SD); share mediated = 1 - beta_direct /
beta_total; both models on the same 2,000 relabelled patient draws (116.boot_sample, seed [20261014, crc32(name)]);
percentile CIs.

Inputs and modes (guard_pinned; each summary is
written once):
- --d1-fresh --root DIR --manifest c2_fresh_400.csv --ekg-dir DIR [--rows DIR]: the fresh 400 (C2 draw). Windows: the
  whole fresh npz (128.fresh_items, the list's sha256 checked; engine._load); EKG: <ekg-dir>/<uid>.npz, the EKG
  manifest checked against FRESH_EKG_MANIFEST_SHA256, each file against its listed sha256, uid, start_s (= the list's
  window_start_s), fs and channel; a non-ok status leaves A_HR NA with that reason. D1 bar applied. Mediation on 128's
  fresh rows and per-row log VER. -> output/tables/arousal_d1_fresh.json, _arousal/d1_fresh_windows.csv.
- --d1-phase1 (HYPOTHESIS-GENERATING): the analysis recordings (engine._items minus 114's touched list) in sorted-uid
  order; per position the EKG channel rule (exactly one EKG/ECG channel in the cached all_names: else no_ekg /
  several_ekg), then 108.eligible_starts (run_s 1800, cached x19 as float32: else no_window), start =
  default_rng([20261016, 4, position]).choice; recomputed here and required to equal output/bdsp/ekg_phase1/
  manifest.csv row for row (check_phase1_manifest); status counts reported. Mediation on 128's Phase-1 rows whose
  overlay lies inside the heart-rate window. -> arousal_d1_phase1.json, _arousal/d1_phase1_windows.csv.
- --d2-measures (no outcome or covariate read): per analysis patient of 133_icare_refetch.planned_patients, log
  VER_ref on keep_windows/<id>.npz (x at 128 Hz, uid = patient id; a refetched window that failed the refetch SD check
  or no longer matches its recorded sha256 is refused) and A_HR on keep_ecg/<id>.npz at the ECG file's own rate: the
  status of keep_ecg/<id>.json if not ok; the lead (icare_ecg_signal): of the channels named EKG/ECG, one is used as
  is, two give the bipolar difference first minus second in header order, none "no_ecg_lead", more than two
  "more_than_two_ecg_leads"; any invalid (NaN) sample in a channel used -> NA "ecg_invalid_samples"; the ECG window
  must equal the EEG window (segment, start). Refetched windows: the last refetch-manifest line per patient decides.
  -> _arousal/d2_measures.csv.
- --d2-analyse (once, at the arm's completion): 122.patient_table of 122.analysis_order (unfinished refused), the
  included patients joined to the D2 measures (missing refused) and to the CPC column of the arm's outcome file;
  D2b's patients by d2_model_set (complete on log VER_ref and the B3 covariates and outcome, X_ref not required). D2a:
  OLS z(log VER_ref) ~ z(A_HR) + hours since ROSC + hospital dummies (HC3), on S < 0.50 (primary) and all; positive
  point estimate flag. D2b: 122.primary with z(log VER_ref) in place of z(X_ref) (name icare_d2b; S < 0.50) and on all
  (icare_d2b_full); proportional-odds CPC 1-5 on the same covariates plus hospital dummies (constant columns dropped;
  OR per SD for a better CPC); the primary model with z(A_HR) added (116.fe_model, descriptive). -> arousal_d2.json.
- --d3 (descriptive): the 38 pDoC analysis recordings (sorted uids, k = position); log VER_ref on the 1800-s window
  (pdoc_window: 108.eligible_starts on the 19 EEG sites at the file's rate, default_rng([20261016, 5, k])); A_HR where
  the EDF header lists exactly one EKG/ECG channel (its signal and rate only; no subject or date field is read);
  joined to 116.a3_table (fixed mapping and outcome hashes); medians, IQR and Spearman rho by diagnosis, outcome
  and CRS-R arousal, no tests. -> arousal_d3.json.

python 135_arousal.py --d1-fresh --root R --manifest M --ekg-dir E | --d1-phase1 | --d2-measures | --d2-analyse | --d3
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import importlib
import math
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.signal import coherence, hilbert, welch

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

kt = importlib.import_module("132_ekg_check")
ns = kt.ns
mech, rn, en, mc = ns.mech, ns.rn, ns.en, ns.mc
bn = kt.bn

SEED = 20261016
B_BOOT = 2000
BAR_BETA = 0.15
Z_ONE_SIDED = 1.6448536269514722
STEP_S = 10.0
ALPHA_BAND, ALPHA_FIT, ALPHA_EXCL, ALPHA_PEAK_MIN = (8.0, 12.0), (2.0, 30.0), (7.0, 14.0), 0.1
ENV_FS, SLOW_BAND = 4.0, (0.01, 0.1)
COH_SEG_S, N_SHIFTS, MIN_SHIFT_S = 256.0, 100, 120.0
WINDOW_S, PHASE1_STREAM, PDOC_STREAM = 1800, 4, 5
FRESH_EKG_MANIFEST_SHA256 = "11134096ea529a32435483cb3a999cfe3e9e49b08fe2100ef09f9949f3ca7923"
PHASE1_EKG = os.path.join(ROOT, "output", "bdsp", "ekg_phase1")



def morgoth_transitions(rec, start_s, dur_s):
    """Label changes per 10 min between consecutive valid Morgoth labels at non-overlapping 10-s steps."""
    t, lab = np.asarray(rec.meta["morgoth_t"], float), np.asarray(rec.meta["morgoth_lab"], object)
    by_t = dict(zip(np.round(t, 6), lab))
    steps = np.arange(start_s, start_s + dur_s - STEP_S + 1e-9, STEP_S)
    seq = [by_t.get(round(float(s), 6)) for s in steps]
    seq = [s for s in seq if s is not None and s != "invalid"]
    if len(seq) < 2:
        return float("nan")
    return sum(a != b for a, b in zip(seq, seq[1:])) * 600.0 / dur_s


def alpha_marker(x, names, fs):
    """{"alpha_peak", "alpha_env_logvar" (None without a posterior alpha peak), "reason"}."""
    m = np.asarray(x, float)[[list(names).index("O1"), list(names).index("O2")]].mean(0)
    f, p = welch(m, fs=fs, nperseg=int(4 * fs))
    fit = (f >= ALPHA_FIT[0]) & (f <= ALPHA_FIT[1]) & ~((f >= ALPHA_EXCL[0]) & (f <= ALPHA_EXCL[1])) & (p > 0)
    lf, lp = np.log10(f[fit]), np.log10(p[fit])
    k, c = np.polyfit(lf, lp, 1)
    with np.errstate(divide="ignore"):
        r = np.log10(np.where(p > 0, p, np.nan)) - (k * np.log10(np.where(f > 0, f, np.nan)) + c)
    idx = np.flatnonzero((f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1]))
    peak = any(r[i] >= ALPHA_PEAK_MIN and r[i] > r[i - 1] and r[i] > r[i + 1] for i in idx if 0 < i < len(r) - 1)
    if not peak:
        return {"alpha_peak": False, "alpha_env_logvar": None, "reason": "no_posterior_alpha_peak"}
    env = np.abs(hilbert(mc.bandpass(m, fs, *ALPHA_BAND)))
    k4 = int(round(fs / ENV_FS))
    e4 = env[:len(env) // k4 * k4].reshape(-1, k4).mean(1)
    s = mc.bandpass(e4 - e4.mean(), ENV_FS, *SLOW_BAND)
    v = float(s.var())
    return {"alpha_peak": True, "alpha_env_logvar": math.log(v) if v > 0 else None,
            "reason": None if v > 0 else "no_variance"}


def _coh(a, b, nper):
    f, c = coherence(a, b, fs=1.0 / kt.EPOCH_S, nperseg=nper, noverlap=nper // 2)
    return float(np.mean(c[(f >= SLOW_BAND[0]) & (f <= SLOW_BAND[1])]))


def hr_envelope_coherence(x, names, fs, ekg, fs_e, uid):
    """Observed 0.01-0.1 Hz coherence of heart rate and C3/C4 power, and its circular-shift null (module docstring)."""
    t, hr, _ = kt.hr_series(kt.r_peaks(ekg, fs_e), fs_e)
    P = kt.c3c4_log_power(np.asarray(x, float)[[list(names).index(c) for c in kt.EEG_CH]], fs).mean(0)
    tc = kt.EPOCH_S * (np.arange(len(P)) + 0.5)
    nper, shift0 = int(round(COH_SEG_S / kt.EPOCH_S)), int(round(MIN_SHIFT_S / kt.EPOCH_S))
    if len(t) < 2:
        return {"coh": None, "reason": "no_heart_rate"}
    k = (tc >= t[0]) & (tc <= t[-1])
    a, b = np.interp(tc[k], t, hr), P[k]
    n = len(a)
    if n < max(nper, 2 * shift0 + 1):
        return {"coh": None, "reason": "too_short"}
    a, b = a - a.mean(), b - b.mean()
    rng = np.random.default_rng([SEED, zlib.crc32(str(uid).encode())])
    null = np.array([_coh(np.roll(a, s), b, nper) for s in rng.integers(shift0, n - shift0 + 1, N_SHIFTS)])
    return {"coh": _coh(a, b, nper), "coh_null_mean": float(null.mean()),
            "coh_null_p95": float(np.quantile(null, 0.95)), "n_shifts": int(len(null)), "n_epochs_coh": int(n),
            "reason": None}


def window_measures(rec, ekg, start_s, dur_s, ekg_reason=None):
    """All per-window quantities (module docstring); ekg: {"ekg", "fs"} over the same span, or None (A_HR and the
    coherence NA with ekg_reason)."""
    fs = float(rec.fs)
    if ekg is not None:
        fe = float(np.asarray(ekg["fs"]).item())
        e = np.asarray(ekg["ekg"], float)
        if len(e) != int(round(dur_s * fe)):
            raise ValueError(f"{rec.uid}: the EKG holds {len(e)} samples, not {dur_s:g} s at {fe:g} Hz")
    i0, m = int(round(start_s * fs)), int(round(dur_s * fs))
    x = np.asarray(rec.x)[:, i0:i0 + m]
    if x.shape[1] != m:
        raise ValueError(f"{rec.uid}: the recording does not hold {dur_s:g} s from {start_s:g} s")
    h = kt.a_hr(e, fe) if ekg is not None else {"a_hr": None, "reason": ekg_reason or "no_ekg", "n_beats": None,
                                                 "rr_ok_share": None, "template_corr": None, "median_hr": None}
    v = kt.ver_ref(x, rec.names, fs, rec.uid)
    st, tr = en.stratum(rec, start_s, dur_s)
    art = en.artifact_index(rec, start_s, dur_s)[0]
    al = alpha_marker(x, rec.names, fs)
    coh = (hr_envelope_coherence(x, rec.names, fs, e, fe, rec.uid) if h["a_hr"] is not None
           else {"coh": None, "reason": "a_hr_na"})
    return {"uid": rec.uid, "patient": rec.patient, "start_s": float(start_s), "a_hr": h["a_hr"],
            "a_hr_reason": h["reason"], "n_beats": h["n_beats"], "rr_ok_share": h["rr_ok_share"],
            "template_corr": h["template_corr"], "median_hr": h["median_hr"], "log_ver_ref": v["log_ver_ref"],
            "ver_reason": v["reason"], "stratum": st, "transitioning": tr, "artifact_share": art,
            "transitions_per_10min": morgoth_transitions(rec, start_s, dur_s), "alpha_peak": al["alpha_peak"],
            "alpha_env_logvar": al["alpha_env_logvar"], "alpha_reason": al["reason"],
            **{k if k.startswith("coh") or k.startswith("n_") else "coh_" + k: val for k, val in coh.items()}}



def _z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


def d1_model(d, marker):
    """OLS of z(log VER_ref) on z(marker) + stratum dummies + artifact share, HC3 (module docstring)."""
    import statsmodels.api as sm
    cols = [marker, "log_ver_ref", "stratum", "artifact_share"]
    na = {c: int(d[c].isna().sum()) for c in cols if d[c].isna().any()}
    e = d.dropna(subset=cols)
    if len(e) < 5:
        return {"n": int(len(e)), "n_na": na, "note": "fewer than 5 complete windows: not estimated"}
    st = e["stratum"].astype(str)
    cnt = st.value_counts()
    ref = sorted(cnt[cnt == cnt.max()].index)[0]
    X = pd.DataFrame({marker: _z(e[marker]), "artifact_share": e["artifact_share"].astype(float).to_numpy()},
                     index=e.index)
    for lev in sorted(set(st) - {ref}):
        X[f"stratum[{lev}]"] = (st == lev).astype(float).to_numpy()
    fit = sm.OLS(_z(e["log_ver_ref"]), sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")
    b, se = float(fit.params[marker]), float(fit.bse[marker])
    return {"marker": marker, "n": int(len(e)), "n_na": na, "beta": b, "se": se, "se_type": "HC3",
            "ci95": [b - 1.959963984540054 * se, b + 1.959963984540054 * se],
            "lower95_one_sided": b - Z_ONE_SIDED * se, "p_one_sided": float(fit.pvalues[marker] / 2 if b > 0
                                                                            else 1 - fit.pvalues[marker] / 2),
            "covariates": list(X.columns), "stratum_reference": ref,
            "stratum_counts": {str(k): int(v) for k, v in cnt.items()}}


def d1_bar(model):
    if "beta" not in model:
        return {"confirmed": False, "note": "not estimated", "bar": "beta >= 0.15 and one-sided lower 95% bound > 0"}
    return {"confirmed": bool(model["beta"] >= BAR_BETA and model["lower95_one_sided"] > 0),
            "bar": "beta >= 0.15 and one-sided lower 95% bound > 0"}



def mediation(d, n_variants, B=B_BOOT, name="d1_mediation"):
    """Total and direct effects of z(A_HR) on calls and the share mediated through z(log VER) (module docstring)."""
    e = mech._prep(d.dropna(subset=["a_hr", "log_ver"]), n_variants)
    y, fe, w = e["call"].astype(float).to_numpy(), e["fe"].to_numpy(), e["w"].to_numpy()
    za = mech.zscore(e["a_hr"].astype(float), w)[0]
    zv = mech.zscore(e["log_ver"].astype(float), w)[0]
    Zs = {"total": za[:, None], "direct": np.column_stack([za, zv])}
    sds = {k: np.where((s := np.sqrt(np.average((Z - np.average(Z, 0, w)) ** 2, 0, w))) > 0, s, 1.0)
           for k, Z in Zs.items()}

    def fit(idx):
        out = {}
        for k, Z in Zs.items():
            r = mech.fe_logit(y[idx], Z[idx] / sds[k], fe[idx], w[idx], lam=mech.LAM)
            out[k] = r["beta"] / sds[k] if r["converged"] else None
        return out

    full = fit(np.arange(len(e)))
    ci = mech.cluster_index(e["patient"].to_numpy())
    draws = mech.mg.boot_idx(len(ci["ids"]), B, mech._seed(name))
    bt, bd, share, failed = [], [], [], 0
    for b in range(B):
        r = fit(mech.boot_sample(ci, draws[b])[0])
        if r["total"] is None or r["direct"] is None:
            failed += 1
            continue
        bt.append(r["total"][0])
        bd.append(r["direct"][0])
        share.append(1.0 - r["direct"][0] / r["total"][0] if r["total"][0] != 0 else np.nan)
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] if len(v) else None
    bt_full, bd_full = full["total"][0], full["direct"][0]
    return {"n_rows": int(len(e)), "n_patients": int(e["patient"].nunique()), "B": int(B), "n_boot_failed": failed,
            "total": {"beta": float(bt_full), "ci95": q(bt)}, "direct": {"beta": float(bd_full), "ci95": q(bd)},
            "log_ver_in_direct": float(full["direct"][1]),
            "share_mediated": float(1.0 - bd_full / bt_full) if bt_full != 0 else None, "share_ci95": q(share)}



def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def phase1_selection(uids, load):
    """The Phase-1 windows by the fixed rule: positions over the sorted uids; status no_ekg / several_ekg
    (exactly one EKG/ECG channel in the cached all_names), else no_window (no 1800-s window by 108.eligible_starts on
    the cached x19 as float32), else ok with start = default_rng([20261016, 4, position]).choice(eligible starts)."""
    rows = []
    for pos, uid in enumerate(sorted(uids)):
        z = load(uid)
        e = [str(c) for c in z["all_names"] if kt.EKG_NAME.search(str(c))]
        row = {"position": pos, "uid": uid, "status": None, "start_s": None}
        if len(e) != 1:
            row["status"] = "no_ekg" if not e else "several_ekg"
        else:
            el = bn.eligible_starts(np.asarray(z["x19"], np.float32), float(z["fs"]), run_s=WINDOW_S)
            if not len(el):
                row["status"] = "no_window"
            else:
                row.update(status="ok", start_s=int(np.random.default_rng([SEED, PHASE1_STREAM, pos]).choice(el)))
        rows.append(row)
    return rows


def check_phase1_manifest(man, rows):
    """The Phase-1 EKG manifest must equal the recomputed selection: position, uid, status and (ok rows) start."""
    key = lambda p, u, s, st: (int(p), str(u), str(s), int(st) if s == "ok" and pd.notna(st) else None)
    want = [key(r["position"], r["uid"], r["status"], r["start_s"]) for r in rows]
    got = [key(p, u, s, st) for p, u, s, st in zip(man["position"], man["uid"], man["status"], man["start_s"])]
    if got != want:
        bad = [w for w, g in zip(want, got) if w != g][:3]
        raise RuntimeError(f"the Phase-1 EKG manifest differs from the recomputed selection ({len(got)} vs "
                           f"{len(want)} rows; first differences {bad})")


def load_ekg(path, row):
    """An EKG npz checked against its manifest row (where listed, the sha256 of the ekg array's bytes, as the
    extractors record it; uid, start_s, fs, channel)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if row.get("ekg_sha256") and isinstance(row["ekg_sha256"], str) and \
            hashlib.sha256(np.asarray(d["ekg"]).tobytes()).hexdigest() != row["ekg_sha256"]:
        raise RuntimeError(f"{path}: the sha256 of its ekg array differs from the manifest's")
    for k, cast in (("uid", str), ("start_s", int), ("fs", float), ("channel", str)):
        if k in row and cast(np.asarray(d[k]).item()) != cast(row[k]):
            raise RuntimeError(f"{path}: {k} {np.asarray(d[k]).item()!r} differs from the manifest's {row[k]!r}")
    return {"ekg": d["ekg"], "fs": float(np.asarray(d["fs"]).item())}



def icare_ecg_signal(ecg, names):
    """(the ECG signal, None) or (None, reason), by the D2 lead rule fixed before any ECG was read: of the channels
    named EKG/ECG, one is used as is; two give the bipolar difference, the first minus the second in header order;
    none: "no_ecg_lead"; more than two: "more_than_two_ecg_leads"; any invalid (NaN) sample in a channel used:
    "ecg_invalid_samples". No quality-based choice."""
    idx = [i for i, c in enumerate(names) if kt.EKG_NAME.search(str(c))]
    if not idx:
        return None, "no_ecg_lead"
    if len(idx) > 2:
        return None, "more_than_two_ecg_leads"
    x = np.asarray(ecg, float)[idx]
    if np.isnan(x).any():
        return None, "ecg_invalid_samples"
    return (x[0] if len(idx) == 1 else x[0] - x[1]), None


def ecg_lead_name(names):
    idx = [str(c) for c in names if kt.EKG_NAME.search(str(c))]
    return "-".join(idx[:2]) if 1 <= len(idx) <= 2 else None


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def refetch_flagged(keep_windows):
    """Patients whose refetched window (the LAST manifest line per patient: the file as it stands) failed
    133_icare_refetch's SD verification, or whose file no longer matches the sha256 recorded there. Windows linked by
    the sidecar have no manifest line (the arm's own staged window)."""
    import json
    fn = os.path.join(keep_windows, "refetch_manifest.jsonl")
    last = {}
    if os.path.exists(fn):
        with open(fn) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    last[str(r["patient"])] = r
    out = set()
    for pid, r in last.items():
        f = os.path.join(keep_windows, r["file"])
        if not r["verified"] or not os.path.exists(f) or sha256_file(f) != r["sha256"]:
            out.add(pid)
    return out


def d2_measures(pid, keep_windows, keep_ecg, flagged):
    """log VER_ref of the patient's kept 30-min window and A_HR of the ECG over the same span (no outcome read)."""
    import json
    if pid in flagged:
        raise RuntimeError(f"{pid}: the refetched window is flagged (SD verification or sha256): not used")
    w = _npz(os.path.join(keep_windows, f"{pid}.npz"))
    win = json.loads(str(w["window"]))
    names = [str(c) for c in w["names"]]
    v = kt.ver_ref(w["x"], names, 128.0, pid)
    out = {"patient": pid, "segment": win["segment"]["seg"], "start_s": int(win["start_s"]),
           "rosc_hours": win.get("rosc_hours"), "log_ver_ref": v["log_ver_ref"], "ver_reason": v["reason"]}
    with open(os.path.join(keep_ecg, f"{pid}.json")) as fh:
        st = json.load(fh)["status"]
    if st != "ok":
        return dict(out, a_hr=None, a_hr_reason=st)
    e = _npz(os.path.join(keep_ecg, f"{pid}.npz"))
    ew = json.loads(str(e["window"]))
    if ew["segment"]["seg"] != win["segment"]["seg"] or int(ew["start_s"]) != int(win["start_s"]):
        raise RuntimeError(f"{pid}: the ECG window {ew['segment']['seg']}/{ew['start_s']} differs from the EEG "
                           "window's")
    names = [str(c) for c in e["names"]]
    sig, why = icare_ecg_signal(e["ecg"], names)
    if sig is None:
        return dict(out, a_hr=None, a_hr_reason=why)
    h = kt.a_hr(sig, float(np.asarray(e["fs"]).item()))
    return dict(out, a_hr=h["a_hr"], a_hr_reason=h["reason"], ecg_lead=ecg_lead_name(names),
                template_corr=h["template_corr"])


def _ols(y, X, name):
    import statsmodels.api as sm
    fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")
    b, se = float(fit.params[name]), float(fit.bse[name])
    return {"beta": b, "se": se, "se_type": "HC3", "ci95": [b - 1.959963984540054 * se, b + 1.959963984540054 * se],
            "covariates": list(X.columns)}


def d2a(t):
    """D2a: OLS z(log VER_ref) ~ z(A_HR) + hours since ROSC + hospital dummies (reference: the most frequent, ties
    alphabetical), HC3; on S < 0.50 (primary) and on all patients; a positive point estimate is required for "in
    coma"."""
    out = {}
    for label, d in (("primary_s_below_0.5", t[t["acns_S"] < 0.5]), ("full_sample", t)):
        e = d.dropna(subset=["a_hr", "log_ver_ref", "rosc_hours", "hospital"])
        if len(e) < 5:
            out[label] = {"n": int(len(e)), "note": "fewer than 5 patients: not estimated"}
            continue
        h = e["hospital"].astype(str)
        cnt = h.value_counts()
        ref = sorted(cnt[cnt == cnt.max()].index)[0]
        X = pd.DataFrame({"a_hr": _z(e["a_hr"]), "rosc_hours": e["rosc_hours"].astype(float).to_numpy()},
                         index=e.index)
        for lev in sorted(set(h) - {ref}):
            X[f"hospital[{lev}]"] = (h == lev).astype(float).to_numpy()
        r = _ols(_z(e["log_ver_ref"]), X, "a_hr")
        out[label] = dict(r, n=int(len(e)), hospital_reference=ref, positive_point_estimate=bool(r["beta"] > 0))
    return out


def _ordinal(d, B_name):
    """Proportional-odds CPC 1-5 ~ the primary model's covariates + hospital dummies; OR per SD of log VER_ref for a
    better (lower) CPC."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ic = importlib.import_module("122_icare_arm")
    frame, covs, _ = ic.design_matrix(d)
    X = pd.DataFrame(covs)
    h = d["hospital"].astype(str).to_numpy()
    for lev in sorted(set(h))[1:]:
        X[f"hospital[{lev}]"] = (h == lev).astype(float)
    X = X.loc[:, X.std() > 0]
    y = pd.Series(pd.Categorical(d["cpc"].astype(int).to_numpy(), categories=[1, 2, 3, 4, 5], ordered=True))
    try:
        fit = OrderedModel(y, X.reset_index(drop=True), distr="logit").fit(method="bfgs", disp=False, maxiter=5000)
    except Exception as e:
        return {"note": f"not estimated: {type(e).__name__}: {str(e)[:120]}"}
    b, se = float(fit.params["z_xref"]), float(fit.bse["z_xref"])
    return {"n": int(len(d)), "beta_higher_cpc": b, "or_per_sd_better": float(np.exp(-b)),
            "or_ci95_better": [float(np.exp(-b - 1.959963984540054 * se)), float(np.exp(-b + 1.959963984540054 * se))],
            "converged": bool(fit.mle_retvals.get("converged", True))}


D2_REQUIRED = (("log_ver_ref", "log_ver_ref_na"), ("acns_grade", "acns_missing"), ("hospital", "hospital_missing"),
               ("age", "age_missing"), ("ohca", "ohca_missing"), ("shockable", "shockable_missing"),
               ("rosc_hours", "rosc_hours_missing"), ("outcome_good", "outcome_missing"))


def d2_model_set(t):
    """The D2b patients: included, complete on log VER_ref and the B3 covariates and outcome (X_ref is not required;
    122.model_set would drop X_ref NA). Each exclusion is counted under its first missing field in this order."""
    inc = t[t["status"] == "included"].copy() if "status" in t else t.copy()
    why = pd.Series("", index=inc.index)
    for col, name in D2_REQUIRED:
        why[inc[col].isna() & (why == "")] = name
    return inc[why == ""].copy(), why[why != ""].value_counts().to_dict()


def d2b(t, B=B_BOOT):
    """D2b: B3's primary model (122.primary: good outcome ~ z(.) + ACNS background + periodic + voltage + hours since
    ROSC + age + OHCA + shockable + TTM + hospital FE; 116.fe_model, relabelled patient bootstrap) with z(log VER_ref)
    in place of z(X_ref), on S < 0.50 (primary, name "icare_d2b") and on all patients (sensitivity,
    "icare_d2b_full"); secondaries: ordinal CPC and the primary model with z(A_HR) added (descriptive)."""
    ic = importlib.import_module("122_icare_arm")
    keep, why = d2_model_set(t)
    keep["xref"] = keep["log_ver_ref"]
    prim = keep[keep["acns_S"] < 0.5]
    out = {"predictor": "z(log VER_ref) in place of z(X_ref)", "exclusions": why,
           "primary_s_below_0.5": ic.primary(prim, B, "icare_d2b"),
           "full_sample": ic.primary(keep, B, "icare_d2b_full"),
           "ordinal_cpc": _ordinal(prim.dropna(subset=["cpc"]), "icare_d2b_ordinal")}
    e = prim.dropna(subset=["a_hr"])
    frame, covs, _ = ic.design_matrix(e)
    covs = {"z_a_hr": _z(e["a_hr"]), **covs}
    out["with_a_hr"] = {"n_patients": int(len(e)), "label": "descriptive",
                        "model": mech.fe_model(frame, covs, B, "icare_d2b_ahr", None, "fe")}
    return out



def pdoc_ecg_name(names):
    """(the one EKG/ECG channel of an EDF header channel list, None) or (None, "no_ecg" | "several_ecg")."""
    e = [str(c) for c in names if kt.EKG_NAME.search(str(c))]
    return (e[0], None) if len(e) == 1 else (None, "no_ecg" if not e else "several_ecg")


def pdoc_window(x, fs, k):
    """Start (s) of the recording's 1800-s window: 108.eligible_starts on the 19 EEG sites, start =
    default_rng([20261016, 5, k]).choice (k: the recording's position among the sorted pDoC uids); None if none."""
    el = bn.eligible_starts(np.asarray(x, np.float32), float(fs), run_s=WINDOW_S)
    return int(np.random.default_rng([SEED, PDOC_STREAM, int(k)]).choice(el)) if len(el) else None


def d3_measures(rec, ecg, fs_e, k, ecg_reason=None):
    """log VER_ref and (with an ECG) A_HR on the pDoC recording's window; the patient key as 116.a3_table forms it."""
    key = pd.Series([rec.uid]).str.extract(r"^(Patient[ _]\d+)_", expand=False).str.replace(" ", "_").iloc[0]
    base = {"uid": rec.uid, "patient_key": key, "diagnosis": rec.meta.get("diagnosis")}
    s = pdoc_window(rec.x, rec.fs, k)
    if s is None:
        return dict(base, start_s=None, log_ver_ref=None, ver_reason="no_30min_window", a_hr=None,
                    a_hr_reason="no_30min_window")
    fs = float(rec.fs)
    v = kt.ver_ref(np.asarray(rec.x)[:, int(s * fs):int((s + WINDOW_S) * fs)], rec.names, fs, rec.uid)
    out = dict(base, start_s=s, log_ver_ref=v["log_ver_ref"], ver_reason=v["reason"])
    if ecg is None:
        return dict(out, a_hr=None, a_hr_reason=ecg_reason or "no_ecg")
    e = np.asarray(ecg, float)[int(s * fs_e):int((s + WINDOW_S) * fs_e)]
    if len(e) != int(round(WINDOW_S * fs_e)):
        return dict(out, a_hr=None, a_hr_reason="ecg_shorter_than_window")
    h = kt.a_hr(e, fs_e)
    return dict(out, a_hr=h["a_hr"], a_hr_reason=h["reason"], template_corr=h["template_corr"])


def _describe(v):
    v = pd.Series(v, dtype=float).dropna()
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(v.median()), "q25": float(v.quantile(0.25)),
            "q75": float(v.quantile(0.75))}


def d3_summary(t):
    """Descriptive D3: per measure, median and IQR by diagnosis (VS vs MCS) and by outcome (recovered = Consc vs not;
    patients without a matched outcome left out and counted), and the Spearman rho with the CRS-R arousal subscale
    (pairs with both values). No test, no p, no CI."""
    from scipy.stats import spearmanr
    meas = ("log_ver_ref", "a_hr")
    dx = t["diagnosis"].astype(str).map(lambda s: "VS" if s.startswith("VS") else "MCS" if s.startswith("MCS")
                                        else "other")
    known = t["outcome"].notna()
    rec = pd.Series(np.where(t["outcome"].astype(str).isin(mech.A3_RECOVERED), "recovered", "not_recovered"),
                    index=t.index)[known]
    out = {"note": "descriptive only (D3): no tests", "n_patients": int(len(t)),
           "n_outcome_missing": int((~known).sum()), "n_arousal_missing": int(t["arousal"].isna().sum()),
           "by_diagnosis": {g: {m: _describe(t.loc[dx == g, m]) for m in meas} for g in sorted(set(dx))},
           "by_outcome": {g: {m: _describe(t.loc[rec.index[rec == g], m]) for m in meas} for g in sorted(set(rec))},
           "spearman_with_arousal": {}}
    for m in meas:
        k = t[m].notna() & t["arousal"].notna()
        out["spearman_with_arousal"][m] = {"n": int(k.sum()), "rho": float(spearmanr(t.loc[k, m], t.loc[k, "arousal"])
                                                                         .correlation) if k.sum() > 2 else None}
    return out


def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()



OUT_DIR = os.path.join(ROOT, "output", "intermediate", "_arousal")
TABLES = os.path.join(ROOT, "output", "tables")


def d1_tables(w, label, bar):
    """The D1 models on one set of windows: primary (A_HR), its bar when bar, the EEG-derived secondaries and the
    coherence summary."""
    out = {"label": label, "n_windows": int(len(w)),
           "a_hr_na_by_reason": w["a_hr_reason"].dropna().value_counts().to_dict() if "a_hr_reason" in w else {},
           "primary": d1_model(w, "a_hr")}
    if bar:
        out["bar"] = d1_bar(out["primary"])
    out["secondary_eeg_derived"] = {m: dict(d1_model(w, m), label="EEG-derived (circular)")
                                    for m in ("transitions_per_10min", "alpha_env_logvar")}
    c = w.dropna(subset=["coh"])
    out["coherence"] = {"label": "descriptive", "n": int(len(c)),
                        "mean_observed": float(c["coh"].mean()) if len(c) else None,
                        "mean_shift_null": float(c["coh_null_mean"].mean()) if len(c) else None,
                        "share_above_own_null_p95": float((c["coh"] > c["coh_null_p95"]).mean()) if len(c) else None}
    return out


def d1_fresh_windows(root, list_manifest, ekg_dir):
    """Per-window measures of the fresh 400 (C2 windows = the whole fresh npz; EKG from ekg_dir, its manifest checked
    by sha256, each file by its own sha256 and its start_s against the list's window_start_s)."""
    man_path = os.path.join(ekg_dir, "manifest.csv")
    if sha256_file(man_path) != FRESH_EKG_MANIFEST_SHA256:
        raise RuntimeError(f"{man_path}: sha256 is not the pinned {FRESH_EKG_MANIFEST_SHA256}")
    em = pd.read_csv(man_path, dtype={"uid": str, "channel": str}).set_index("uid")
    starts = pd.read_csv(list_manifest, dtype={"uid": str}).set_index("uid")["window_start_s"]
    items = ns.fresh_items(root, list_manifest)
    rows = []
    for uid in sorted(items):
        r = em.loc[uid]
        if int(r["start_s"]) != int(starts[uid]):
            raise RuntimeError(f"{uid}: EKG start_s {r['start_s']} is not the list's window_start_s {starts[uid]}")
        ekg, why = None, (None if r["status"] == "ok" else str(r["status"]))
        if why is None:
            ekg = load_ekg(os.path.join(ekg_dir, f"{uid}.npz"), dict(r.to_dict(), uid=uid))
        rec = en._load(items[uid])
        rows.append(window_measures(rec, ekg, 0.0, float(WINDOW_S), ekg_reason=why))
        print(f"{len(rows)}/{len(items)} {uid}", flush=True)
    return pd.DataFrame(rows)


def d1_phase1_windows(ekg_dir=PHASE1_EKG):
    """Per-window measures of the Phase-1 recordings with an EKG window: the selection recomputed (phase1_selection)
    and required to equal the EKG manifest; returns (windows, status counts)."""
    items = {it["uid"]: it for it in en._items("bdsp", "analysis")}
    uids = sorted(set(items) - set(mech.mg.touched_uids()))

    def load(u):
        with np.load(items[u]["path"], allow_pickle=False) as z:
            return {k: z[k] for k in ("all_names", "x19", "fs")}
    rows = phase1_selection(uids, load)
    man = pd.read_csv(os.path.join(ekg_dir, "manifest.csv"), dtype={"uid": str, "channel": str, "status": str})
    check_phase1_manifest(man, rows)
    out = []
    for r in man[man["status"] == "ok"].to_dict("records"):
        ekg = load_ekg(os.path.join(ekg_dir, f"{r['uid']}.npz"), r)
        out.append(window_measures(en._load(items[r["uid"]]), ekg, float(r["start_s"]), float(WINDOW_S)))
        print(f"{len(out)} {r['uid']}", flush=True)
    return pd.DataFrame(out), man["status"].value_counts().to_dict()


def d1_mediation(w, rows, ver, n_variants, B=B_BOOT, name="d1_mediation", within_window=False):
    """Mediation on the C2 rows: 128's per-row log VER and the window's A_HR per recording. within_window (Phase 1,
    where the heart-rate window is one 1800-s stretch of a longer recording): only rows whose overlay extent lies inside
    [start, start + 1800 s) of that window (rows without an extent, e.g. 113's Columbia rows, are left out)."""
    d = rows.merge(ver[["key", "log_ver"]], on="key", how="left", validate="one_to_one")
    hr = w[["uid", "a_hr"] + (["start_s"] if within_window else [])].rename(columns={"start_s": "hr_start_s"})
    d = d.merge(hr, on="uid", how="inner")
    if within_window:
        t0, t1 = pd.to_numeric(d["extent_t0"], errors="coerce"), pd.to_numeric(d["extent_t1"], errors="coerce")
        d = d[(t0 >= d["hr_start_s"]) & (t1 <= d["hr_start_s"] + WINDOW_S)]
    return mediation(d, n_variants, B, name)


def edf_channel_names(path):
    """The EDF header's channel list (no signal, no subject or date field is read)."""
    import mne
    return list(mne.io.read_raw_edf(path, preload=False, verbose="ERROR").ch_names)


def edf_signal(path, name):
    """(one channel's samples, its sampling rate) of an EDF; nothing else of the header is used."""
    import mne
    raw = mne.io.read_raw_edf(path, include=[name], preload=True, verbose="ERROR")
    return raw.get_data(picks=[name])[0] * 1e6, float(raw.info["sfreq"])


def d3_windows():
    """D3 per-patient measures on the pDoC analysis recordings, k = position among the sorted uids."""
    items = sorted(en._items("pdoc", "analysis"), key=lambda it: it["uid"])
    out = []
    for k, it in enumerate(items):
        rec = en._load(it)
        name, why = pdoc_ecg_name(edf_channel_names(it["path"]))
        ecg, fs_e = edf_signal(it["path"], name) if name else (None, None)
        out.append(dict(d3_measures(rec, ecg, fs_e, k, ecg_reason=why), k=k))
    return pd.DataFrame(out)


def guard_pinned(spec_path=None):
    """Not applied in this copy: the check reads files of the private working copy."""
    return None


def _write(name, out):
    import json
    path = os.path.join(TABLES, name)
    os.makedirs(TABLES, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(mech.mg._jsonable(dict(out, sha256_135=own_sha256())), fh, indent=1, default=float)
    os.replace(path + ".tmp", path)
    print(f"135 -> {path}")


def _once(name):
    path = os.path.join(TABLES, name)
    if os.path.exists(path):
        raise RuntimeError(f"{path} exists: each analysis runs once (move it away to rerun)")


def _windows_csv(w, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    w.to_csv(os.path.join(OUT_DIR, name), index=False)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for f in ("--d1-fresh", "--d1-phase1", "--d2-measures", "--d2-analyse", "--d3"):
        m.add_argument(f, action="store_true")
    ap.add_argument("--root", help="--d1-fresh: the C2 window directory (sig/, morgoth/)")
    ap.add_argument("--manifest", help="--d1-fresh: the pinned c2_fresh_400.csv")
    ap.add_argument("--ekg-dir", help="--d1-fresh: the fresh-400 EKG directory (manifest.csv, <uid>.npz)")
    ap.add_argument("--rows", default=ns.FRESH_SHARDS, help="--d1-fresh: the C2 fresh row shards")
    a = ap.parse_args(argv)
    guard_pinned()
    if a.d1_fresh:
        _once("arousal_d1_fresh.json")
        if not (a.root and a.manifest and a.ekg_dir):
            raise SystemExit("--d1-fresh needs --root, --manifest and --ekg-dir")
        w = d1_fresh_windows(a.root, a.manifest, a.ekg_dir)
        _windows_csv(w, "d1_fresh_windows.csv")
        out = d1_tables(w, "fresh 400: CONFIRMATION (D1 bar)", bar=True)
        rows = ns.fresh_rows(a.rows, outcomes=True)
        ns.require_all(rows, ns.paths("fresh")["ver"])
        out["mediation"] = dict(label="descriptive", **d1_mediation(w, rows, ns.read_results(ns.paths("fresh")["ver"]),
                                                                  ns.variants_present(rows)))
        _write("arousal_d1_fresh.json", out)
    elif a.d1_phase1:
        _once("arousal_d1_phase1.json")
        w, status = d1_phase1_windows()
        _windows_csv(w, "d1_phase1_windows.csv")
        out = d1_tables(w, "Phase-1 windows: HYPOTHESIS-GENERATING (added later)", bar=False)
        out["selection_status"] = status
        rows = ns.phase1_rows(outcomes=True)
        nv = ns.non_immune_variant_counts(sorted(set(rows["pipe"])), ns.pinned_typing())
        out["mediation"] = dict(label="descriptive; rows whose overlay lies inside the heart-rate window",
                                **d1_mediation(w, rows, ns.read_results(ns.paths("phase1")["ver"]), nv,
                                               within_window=True))
        _write("arousal_d1_phase1.json", out)
    elif a.d2_measures:
        rf = importlib.import_module("133_icare_refetch")
        kw, ke = rf.keep_dirs()
        flagged = refetch_flagged(kw)
        res = [d2_measures(pid, kw, ke, flagged) for pid, _ in rf.planned_patients()]
        _windows_csv(pd.DataFrame(res), "d2_measures.csv")
        print(f"135 d2 measures: {len(res)} patients")
    elif a.d2_analyse:
        _once("arousal_d2.json")
        ic = importlib.import_module("122_icare_arm")
        index = ic.load_index()
        order = ic.analysis_order(index, ic.dev_patients(index))
        t = ic.patient_table(order, index)
        if (t["status"] == "unfinished").any():
            raise RuntimeError("the I-CARE arm is not finished: D2 runs at completion only")
        meas = pd.read_csv(os.path.join(OUT_DIR, "d2_measures.csv"), dtype={"patient": str})
        inc = t[t["status"] == "included"]
        missing = sorted(set(inc["patient"]) - set(meas["patient"]))
        if missing:
            raise RuntimeError(f"{len(missing)} included patient(s) without D2 measures (e.g. {missing[:3]})")
        cpc = pd.read_csv(os.path.join(ic.paths("analysis")["outcome"], "outcomes.csv"), dtype={"patient": str},
                          usecols=["patient", "cpc"])
        d = inc.merge(meas[["patient", "log_ver_ref", "a_hr", "a_hr_reason"]], on="patient", how="left",
                      validate="one_to_one").merge(cpc, on="patient", how="left", validate="one_to_one")
        _write("arousal_d2.json", {"label": "D2 (added later SECONDARY; B3's X_ref primary unchanged)",
                                   "n_included": int(len(d)),
                                   "a_hr_na_by_reason": d["a_hr_reason"].dropna().value_counts().to_dict(),
                                   "d2a": d2a(d), "d2b": d2b(d)})
    elif a.d3:
        _once("arousal_d3.json")
        w = d3_windows()
        _windows_csv(w, "d3_windows.csv")
        a3, counts = mech.a3_table()
        d = w.merge(a3[["patient", "arousal", "outcome"]].rename(columns={"patient": "patient_key"}),
                    on="patient_key", how="left", validate="one_to_one")
        _write("arousal_d3.json", dict(d3_summary(d), a3_cases=counts,
                                       a_hr_na_by_reason=d["a_hr_reason"].dropna().value_counts().to_dict()))


if __name__ == "__main__":
    main()
