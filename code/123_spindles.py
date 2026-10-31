"""Spindle-like activity and false calls: do false calls co-occur with spindle-like activity, and only under designs
whose rhythm overlaps the ~0.02-Hz sigma oscillation of N2?
part B5 of the null benchmark ("### B5. Spindle-like activity and false calls"):
do false calls co-occur with spindle-like activity, and only under designs whose rhythm overlaps the ~0.02-Hz sigma
oscillation of N2? Outside the benchmark hash; built test-first on synthetic fixtures and the dev split
(code/test_spindles.py). The detectors are the author's published ones, vendored verbatim in code/vendor_spindle/ and
pinned by sha256 (VENDOR_SHA256): the code refuses to load them otherwise and never executes a modified file.

Features of one window (spindle_features, also called per window by the I-CARE arm, 122_icare_arm.py), on C3 and C4 in
uV, each averaged over the two channels:
- a7_density (S-SP1): Lacourse A7 events per minute of the window at its published operating point (scale 1.0, its
  0.69 correlation criterion untouched; NOT rate-matched: calibration.json records the A7 as un-tuned);
- sigma_slow_rel (S-SP2, detector-free): power of the 11-16 Hz amplitude envelope (detectors.sigma_filter + envelope)
  at 0.015-0.025 Hz, relative to 0.005-0.05 Hz. The envelope's periodogram (linear detrend, Hann taper, one segment)
  is summed over the bins inside each band; NA (never 0) when the window holds no bin of the narrow band or fewer than
  3 of the wide band (reason_sigma "slow_band_unresolved");
- dens_<name> for the other five calibrated detectors (descriptive sensitivity).
A window with non-finite samples or a flat channel (SD below 1e-3 uV) has NA features with its reason (missing is not
zero); a window shorter than 10 s has none; a window that runs past the recording (or starts before it) has none
(reason "window_outside_recording"): it is cut with 116's own bench.motor_common.span (floor, ceil of the unrounded
extent), which raises there, and 116 records that row's X as NA (feature_error), so the row is kept and left out of the
models, counted.

Stages (python 123_spindles.py --split dev|analysis [--stage features|stats|all] [--release]):
- features: the first-overlay windows of the BDSP rows (116's M1 rows: window = the row's extent_t0..extent_t1, for
  113's Columbia rows the rebuilt overlay's extent), one result per unique (uid, window), computed on the frozen engine's
  recording (bench.engine._load). One part file per recording (parts_<split>/, written whole; parts of another version
  of this script are refused), merged into output/intermediate/_spindles/bdsp_<split>.parquet.
- stats, per measure (S-SP1, S-SP2; the five other detectors as a descriptive sensitivity): weighted logistic
    call ~ z(measure) x design type + Morgoth stratum + pipeline-variant fixed effects,
  design type = inflated vs immune or deflated (116's x_ref_class of the frozen variant, from its design table; variants
  without a contrast have no class and are left out and counted). The design type is a function of the variant, so its
  main effect is absorbed by the fixed effects; the measure enters as z x inflated and z x other (their difference is
  the interaction). Row weight = 1 / (frozen variants of the pipeline) divided by the patient's rows of that variant
  (116._prep); z by the weighted mean and SD of the model's rows; every fit carries 116's weak L2 (C = 100 on covariates
  divided by their weighted SD; stratum dummies against normal_awake; fixed effects about their mean); patient-cluster
  bootstrap, 2000 relabelled draws (116.boot_sample), seed [20261014, crc32(model name)], percentile 95% CI.
  - Test: the measure's OR per SD within inflated designs, two-sided bootstrap p; Holm over exactly the two measures.
  - Also reported: the OR in the other designs, the interaction (OR ratio), the share mediated through X (rows with
    116's X: 1 - beta_inflated(with z(X)) / beta_inflated(without), the same rows and the same draws), and the five-
    detector sensitivity. The CLI prints counts, never an estimate.

python 123_spindles.py --split dev|analysis [--stage features|stats|all] [--release]
                       [--limit N --max-windows N --boot N]   (these three: dev smoke runs only)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.signal import periodogram

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

VENDOR = os.path.join(CODE, "vendor_spindle")
VENDOR_SHA256 = {"detectors.py": "19d15d9daa15a2c0cb505242856637c75ac1ddc97e881e7b994efe58d26038a9",
                 "calibration.json": "e1ef26f96c2f2f87b8e1cf7707bf97ed2e7579b71ae182c77762ad9f62d5e9ec"}

SEED = 20261014
B_BOOT = 2000
PRIMARY = ("a7_density", "sigma_slow_rel")
OTHER_DETECTORS = ("fixed_amplitude", "relative_threshold", "rms_ferrarelli", "wavelet", "sigma_index")
FEATURE_COLS = ("a7_density", "sigma_slow_rel") + tuple("dens_" + n for n in OTHER_DETECTORS)
NARROW_HZ, WIDE_HZ = (0.015, 0.025), (0.005, 0.05)
MIN_NARROW_BINS, MIN_WIDE_BINS = 1, 3
MIN_WINDOW_S = 10.0
FLAT_SD_UV = 1e-3
REF_STRATUM = "normal_awake"
ROUND = 6



def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def verify_vendor(vendor_dir=None):
    """The vendored files must equal their fixed sha256 (VENDOR_SHA256), or nothing runs."""
    d = vendor_dir or VENDOR
    for name, want in VENDOR_SHA256.items():
        path = os.path.join(d, name)
        if not os.path.exists(path):
            raise RuntimeError(f"vendored spindle file {path} is missing: refused (pinned sha256 {want})")
        got = _sha(path)
        if got != want:
            raise RuntimeError(f"vendored spindle file {path} has sha256 {got}, not the pinned {want}: refused")


_CACHE = {}


def _clear_cache():
    _CACHE.clear()


def load_detectors(vendor_dir=None):
    """The vendored detectors module, loaded only after the hash check (a modified file is never executed)."""
    d = os.path.abspath(vendor_dir or VENDOR)
    if d not in _CACHE:
        verify_vendor(d)
        spec = importlib.util.spec_from_file_location("vendor_spindle_detectors_pinned", os.path.join(d, "detectors.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE[d] = mod
    return _CACHE[d]



def _na(reason, reason_sigma=None):
    out = {c: np.nan for c in FEATURE_COLS}
    out.update(reason=reason, reason_sigma=reason_sigma, n_events_a7=np.nan)
    return out


def _band_bins(f, band):
    return (f >= band[0] - 1e-12) & (f <= band[1] + 1e-12)


def sigma_slow_power(x, fs, det):
    """Power of the 11-16 Hz envelope at NARROW_HZ relative to WIDE_HZ; (value, reason) with NaN and a reason where the
    window cannot resolve the bands."""
    env = det.envelope(det.sigma_filter(x, fs))
    f, P = periodogram(env, fs=fs, window="hann", detrend="linear")
    nar, wid = _band_bins(f, NARROW_HZ), _band_bins(f, WIDE_HZ)
    if nar.sum() < MIN_NARROW_BINS or wid.sum() < MIN_WIDE_BINS:
        return np.nan, "slow_band_unresolved"
    tot = P[wid].sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.nan, "no_slow_power"
    return float(P[nar].sum() / tot), None


def spindle_features(x_c3, x_c4, fs):
    """Spindle features of one window (uV, C3 and C4, same length): A7 density (S-SP1), the detector-free slow sigma
    modulation (S-SP2) and the five other calibrated detectors' densities, each averaged over the two channels. NA
    (NaN, with a reason) rather than 0 for a window that cannot be measured. -> dict (FEATURE_COLS, reason,
    reason_sigma, n_events_a7)."""
    det = load_detectors()
    fs = float(fs)
    xs = [np.asarray(x_c3, float), np.asarray(x_c4, float)]
    if xs[0].shape != xs[1].shape or xs[0].ndim != 1:
        raise ValueError("C3 and C4 must be 1-D arrays of the same length")
    dur = xs[0].size / fs
    if dur < MIN_WINDOW_S:
        return _na("window_too_short")
    if not all(np.isfinite(x).all() for x in xs):
        return _na("non_finite")
    if any(np.std(x) < FLAT_SD_UV for x in xs):
        return _na("flat_channel")
    cal = det.calibrated_detectors()
    minutes = dur / 60.0
    out = {"reason": None, "reason_sigma": None}
    ev = [det.a7_lacourse(x, fs) for x in xs]
    out["a7_density"] = float(np.mean([len(e) / minutes for e in ev]))
    out["n_events_a7"] = int(sum(len(e) for e in ev))
    sig = [sigma_slow_power(x, fs, det) for x in xs]
    if any(not np.isfinite(v) for v, _ in sig):
        out["sigma_slow_rel"], out["reason_sigma"] = np.nan, next(r for v, r in sig if not np.isfinite(v))
    else:
        out["sigma_slow_rel"] = float(np.mean([v for v, _ in sig]))
    for name in OTHER_DETECTORS:
        out["dens_" + name] = float(np.mean([len(cal[name](x, fs)) / minutes for x in xs]))
    return out



def own_sha256():
    return _sha(os.path.abspath(__file__))


def _rn():
    return importlib.import_module("114_benchmark")


def _mech():
    return importlib.import_module("116_mechanism")


def _mg():
    return importlib.import_module("115_benchmark_merge")


def guard_split(split, release=False, spec_path=None):
    """The dev split needs nothing; the analysis split needs the explicit release flag and 114's guard (the
    check of files not distributed with this code is not applied here)."""
    rn = _rn()
    if split != "analysis":
        rn.guard_split(split, spec_path)
        return
    if not release:
        raise RuntimeError("analysis split refused: it runs only with the explicit release flag (--release)")
    rn.guard_split(split, spec_path)


def paths(split, out_dir=None):
    base = out_dir or os.path.join(ROOT, "output", "intermediate", "_spindles")
    tables = base if (split == "dev" or out_dir) else os.path.join(ROOT, "output", "tables")
    return {"parts": os.path.join(base, f"parts_{split}"), "features": os.path.join(base, f"bdsp_{split}.parquet"),
            "summary": os.path.join(tables, "spindles_dev.json" if split == "dev" else "spindles.json")}



def _r(v):
    return round(float(v), ROUND)


def rows_with_windows(split, release=False):
    """(116's M1 rows of the split, with t0, t1: the first-overlay window of each row)."""
    guard_split(split, release)
    mech, rn = _mech(), _rn()
    df = mech.load_rows(split)
    m1 = mech.m1_rows(df).copy()
    t0 = m1["extent_t0"].astype(float) if "extent_t0" in m1 else pd.Series(np.nan, index=m1.index)
    t1 = m1["extent_t1"].astype(float) if "extent_t1" in m1 else pd.Series(np.nan, index=m1.index)
    miss = t0.isna() | t1.isna()
    mods = {}
    for i in m1.index[miss]:
        row = m1.loc[i]
        mod = mods.setdefault(row["pipe"], rn.load_pipe(row["pipe"]))
        ov = mech.overlay_for_row(mod, {}, row)
        t0[i], t1[i] = (float(v) for v in mod.extent(ov))
    m1["t0_raw"], m1["t1_raw"] = t0.astype(float), t1.astype(float)
    m1["t0"], m1["t1"] = t0.map(_r), t1.map(_r)
    return m1


def window_table(split, release=False):
    """One row per unique (uid, t0, t1) of the first-overlay windows."""
    m1 = rows_with_windows(split, release)
    w = m1[["uid", "patient", "t0", "t1", "t0_raw", "t1_raw"]].drop_duplicates(["uid", "t0", "t1"])
    return w.sort_values(["uid", "t0", "t1"], kind="stable").reset_index(drop=True)


def _channel_rows(rec):
    mc = importlib.import_module("bench.motor_common")
    names = mc.canon(rec.names)
    for c in ("C3", "C4"):
        if c not in names:
            raise RuntimeError(f"{rec.uid}: channel {c} missing (has {names})")
    return names.index("C3"), names.index("C4")


def recording_features(rec, windows):
    """Feature rows of one recording's windows (DataFrame with t0, t1)."""
    bn = importlib.import_module("108_bdsp_null")
    i3, i4 = _channel_rows(rec)
    fs = float(rec.fs)
    out = []
    mc = importlib.import_module("bench.motor_common")
    x2 = rec.x[[i3, i4]]
    raw0 = windows["t0_raw"] if "t0_raw" in windows else windows["t0"]
    raw1 = windows["t1_raw"] if "t1_raw" in windows else windows["t1"]
    for t0, t1, r0, r1 in zip(windows["t0"], windows["t1"], raw0, raw1):
        try:
            seg, _ = mc.span(x2, fs, float(r0), float(r1))
        except ValueError:
            out.append({"uid": rec.uid, "patient": rec.patient, "t0": float(t0), "t1": float(t1),
                        "dur_s": np.nan, "fs": fs, "pad_frac": np.nan, **_na("window_outside_recording")})
            continue
        seg = np.asarray(seg, float)
        b = seg.shape[1]
        f = spindle_features(seg[0], seg[1], fs)
        pad = float(np.mean(bn.padding_seconds(np.nan_to_num(seg), fs))) if seg.shape[1] >= fs else np.nan
        out.append({"uid": rec.uid, "patient": rec.patient, "t0": float(t0), "t1": float(t1), "dur_s": b / fs,
                    "fs": fs, "pad_frac": pad, **f})
    return out


def _part_file(path, uid):
    return os.path.join(path, f"{uid}.json")


def _read_part(path, uid):
    fn = _part_file(path, uid)
    if not os.path.exists(fn):
        return None
    with open(fn) as fh:
        part = json.load(fh)
    if part.get("sha256_123") != own_sha256():
        raise RuntimeError(f"{fn} comes from another version of 123_spindles.py: move it away before running this "
                           "version")
    return part


def _write_part(path, uid, rows):
    fn = _part_file(path, uid)
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
    with open(fn + ".tmp", "w") as fh:
        json.dump({"uid": uid, "sha256_123": own_sha256(), "rows": clean}, fh)
    os.replace(fn + ".tmp", fn)


def run_features(split, release=False, limit=None, out_dir=None, max_windows=None):
    """Features of every first-overlay window of the split's BDSP rows -> bdsp_<split>.parquet. -> number of windows."""
    guard_split(split, release)
    if split == "analysis" and (limit or max_windows):
        raise RuntimeError("the analysis split runs on every window (no --limit/--max-windows)")
    en = _rn().en
    p = paths(split, out_dir)
    win = window_table(split, release)
    items = {it["uid"]: it for it in en._items("bdsp", split)}
    uids = sorted(win["uid"].unique())[:limit]
    missing = sorted(set(uids) - set(items))
    if missing:
        raise RuntimeError(f"{len(missing)} row recording(s) not in the {split} listing: {missing[:5]}")
    os.makedirs(p["parts"], exist_ok=True)
    rn = _rn()
    rows = []
    for i, uid in enumerate(uids):
        w = win[win["uid"] == uid].iloc[:max_windows]
        part = _read_part(p["parts"], uid)
        if part is not None and [(r["t0"], r["t1"]) for r in part["rows"]] == list(zip(w["t0"], w["t1"])):
            rows += part["rows"]
            continue
        rec = en._load(items[uid])
        rn.check_member(rec, split)
        res = recording_features(rec, w)
        del rec
        _write_part(p["parts"], uid, res)
        rows += json.load(open(_part_file(p["parts"], uid)))["rows"]
        print(f"{i + 1}/{len(uids)} {uid}: {len(res)} windows, {sum(r['reason'] is not None for r in res)} NA",
              flush=True)
    df = pd.DataFrame(rows)
    for c in FEATURE_COLS + ("dur_s", "pad_frac", "n_events_a7"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["split"] = split
    df["sha256_123"] = own_sha256()
    df.to_parquet(p["features"], index=False)
    return len(df)



def holm_two(pvals):
    """Holm over exactly the two pre-specified tests (S-SP1 and S-SP2)."""
    if len(pvals) != 2:
        raise ValueError(f"Holm is applied over exactly 2 tests (S-SP1, S-SP2), not {len(pvals)}")
    if set(pvals) != set(PRIMARY):
        raise ValueError(f"the two tests are S-SP1 and S-SP2 ({PRIMARY}), not {sorted(pvals)}")
    names = list(PRIMARY)
    if any(pvals[k] is None for k in names):
        return {k: {"p": pvals[k], "p_holm": None, "reject": None} for k in names}
    adj = _mg().holm([pvals[k] for k in names])
    return {k: {"p": float(pvals[k]), "p_holm": float(a), "reject": bool(a <= 0.05)} for k, a in zip(names, adj)}


def prepare_frame(raw, n_variants):
    """Analysis frame from rows with patient, pipe, variant, stratum, call, x_ref_class and the measures: the design
    type (inflated vs immune or deflated), fixed-effect codes and row weights (116._prep). Rows without a design class
    (variants with no contrast) are left out and counted."""
    mech = _mech()
    d = raw.copy()
    cls = d["x_ref_class"]
    keep = cls.notna()
    n_no = int((~keep).sum())
    d = d[keep].copy()
    d["inflated"] = (d["x_ref_class"] == "inflated").to_numpy()
    d = mech._prep(d, n_variants)
    d["stratum"] = d["stratum"].fillna("unlabelled").astype(str)
    d.attrs["n_no_design_class"] = n_no
    return d


def _stratum_dummies(e):
    levels = sorted(set(e["stratum"]) - {REF_STRATUM})
    return {f"stratum_{s}": (e["stratum"] == s).to_numpy(float) for s in levels}


def _fit_draws(y, Z, fe, w, ci, draws):
    """Penalised point fit and bootstrap draws (original units) of y ~ Z + fixed effects; every fit as 116.fe_model."""
    mech = _mech()
    sd = np.sqrt(np.average((Z - np.average(Z, 0, w)) ** 2, 0, w))
    sd = np.where(sd > 0, sd, 1.0)
    full = mech.fe_logit(y, Z / sd, fe, w, lam=mech.LAM)
    bd = np.full((len(draws), Z.shape[1]), np.nan)
    if not full["converged"]:
        return None, bd, full
    for b in range(len(draws)):
        idx, _ = mech.boot_sample(ci, draws[b])
        r = mech.fe_logit(y[idx], Z[idx] / sd, fe[idx], w[idx], lam=mech.LAM)
        if r["converged"]:
            bd[b] = r["beta"] / sd
    return full["beta"] / sd, bd, full


def _or(beta, draws):
    t = _mech()._ptests(beta, draws)
    return {"beta": float(beta), "or": float(np.exp(beta)),
            "or_ci95": None if t["ci95"] is None else [float(np.exp(v)) for v in t["ci95"]],
            "p_two_sided": t["p_two_sided"], "p_greater": t["p_greater"]}


def fit_measure(d, measure, B=B_BOOT, name=None, mediation=False):
    """The B5 model for one measure on the frame d (prepare_frame). -> within-inflated OR (the test), other designs' OR,
    interaction (OR ratio) and, with mediation, the share mediated through X."""
    mech, mg = _mech(), _mg()
    name = name or f"spindle|{measure}"
    n_na = int(d[measure].isna().sum())
    e = d[d[measure].notna()]
    out = {"measure": measure, "B": B, "n_rows_na_measure": n_na, "n_rows": int(len(e)),
           "n_patients": int(e["patient"].nunique()) if len(e) else 0,
           "n_inflated_rows": int(e["inflated"].sum()), "n_other_rows": int((~e["inflated"]).sum()),
           "penalty": {"C": mech.PENALTY_C, "on": "covariates / weighted SD; pipeline-variant effects about their mean"}}
    if len(e) == 0 or e["patient"].nunique() < 2 or out["n_inflated_rows"] == 0:
        out["note"] = "fewer than 2 patients or no inflated-design rows: not estimated"
        return out
    z, m, sd = mech.zscore(e[measure], e["w"])
    infl = e["inflated"].to_numpy(float)
    cov = {"z_inflated": z * infl, "z_other": z * (1 - infl), **_stratum_dummies(e)}
    y, w, fe = e["call"].astype(float).to_numpy(), e["w"].to_numpy(float), e["fe"].to_numpy()
    ci = mech.cluster_index(e["patient"].to_numpy())
    draws = mg.boot_idx(len(ci["ids"]), B, mech._seed(name))
    beta, bd, full = _fit_draws(y, np.column_stack(list(cov.values())), fe, w, ci, draws)
    out.update(x_weighted_mean=float(m), x_weighted_sd=float(sd), n_boot_failed=int(np.sum(~np.isfinite(bd[:, 0]))),
               plain_mle_converged=bool(mech.fe_logit(y, np.column_stack(list(cov.values())), fe, w)["converged"]))
    if beta is None:
        out["note"] = "full-sample fit did not converge"
        return out
    out["within_inflated"] = _or(beta[0], bd[:, 0])
    out["other_designs"] = _or(beta[1], bd[:, 1]) if out["n_other_rows"] else {"note": "no rows in other designs"}
    if out["n_other_rows"]:
        inter = _or(beta[0] - beta[1], bd[:, 0] - bd[:, 1])
        out["interaction"] = {"or_ratio": inter["or"], "or_ratio_ci95": inter["or_ci95"],
                              "p_two_sided": inter["p_two_sided"]}
    if mediation:
        out["mediation"] = _mediation(e, z, y, w, fe, name, B, mg, mech)
    return out


def _mediation(e, z, y, w, fe, name, B, mg, mech):
    """Share of the within-inflated log-OR carried by X: 1 - beta(with z(X)) / beta(without), on the rows with X and the
    same bootstrap draws for both fits."""
    if "X" not in e or e["X"].notna().sum() == 0:
        return {"note": "no X results (116) for these rows", "share": None, "n_rows": 0}
    keep = e["X"].notna().to_numpy()
    s = e[keep]
    if s["patient"].nunique() < 2 or not s["inflated"].any():
        return {"note": "fewer than 2 patients or no inflated rows with X", "share": None, "n_rows": int(keep.sum())}
    infl = s["inflated"].to_numpy(float)
    zs = z[keep]
    cov = {"z_inflated": zs * infl, "z_other": zs * (1 - infl), **_stratum_dummies(s)}
    zx, _, _ = mech.zscore(s["X"], s["w"])
    base = np.column_stack(list(cov.values()))
    adj = np.column_stack([base, zx])
    ys, ws, fes = y[keep], w[keep], fe[keep]
    ci = mech.cluster_index(s["patient"].to_numpy())
    draws = mg.boot_idx(len(ci["ids"]), B, mech._seed(name + "|mediation"))
    b0, d0, _ = _fit_draws(ys, base, fes, ws, ci, draws)
    b1, d1, _ = _fit_draws(ys, adj, fes, ws, ci, draws)
    if b0 is None or b1 is None:
        return {"note": "a mediation fit did not converge", "share": None, "n_rows": int(keep.sum())}
    share = 1.0 - b1[0] / b0[0]
    sd = 1.0 - d1[:, 0] / d0[:, 0]
    sd = sd[np.isfinite(sd)]
    ci_b0 = np.quantile(d0[np.isfinite(d0[:, 0]), 0], [0.025, 0.975]) if np.isfinite(d0[:, 0]).any() else None
    return {"note": None, "n_rows": int(keep.sum()), "n_patients": int(s["patient"].nunique()),
            "beta_inflated_without_x": float(b0[0]), "beta_inflated_with_x": float(b1[0]), "share": float(share),
            "share_ci95": [float(np.quantile(sd, 0.025)), float(np.quantile(sd, 0.975))] if len(sd) else None,
            "or_without_x": float(np.exp(b0[0])), "or_with_x": float(np.exp(b1[0])),
            "ratio_stable": bool(ci_b0 is not None and (ci_b0[0] > 0 or ci_b0[1] < 0)),
            "note_ratio": "the share is a ratio of log-ORs: unstable when the unadjusted within-inflated log-OR's "
                          "bootstrap CI includes 0 (ratio_stable False)"}


def run_models(d, B=B_BOOT):
    """Both pre-specified tests (Holm over exactly these two), their interaction and mediation, and the five-detector
    sensitivity (descriptive; outside the Holm family)."""
    out = {"seed": SEED, "B": B, "n_rows": int(len(d)), "n_no_design_class": d.attrs.get("n_no_design_class"),
           "weights": "1 / (frozen variants of the pipeline), over the patient's rows of that variant",
           "measures": {m: fit_measure(d, m, B, name=f"spindle|{m}", mediation=True) for m in PRIMARY}}
    out["holm"] = holm_two({m: out["measures"][m].get("within_inflated", {}).get("p_two_sided") for m in PRIMARY})
    out["sensitivity_detectors"] = {"dens_" + n: fit_measure(d, "dens_" + n, B, name=f"spindle|dens_{n}")
                                    for n in OTHER_DETECTORS if "dens_" + n in d}
    return out


def assemble_frame(split, release=False, out_dir=None):
    """Frame of the split's M1 rows with the window features, the design class (116's design table) and X (116's X
    results, where current). -> (frame, info)."""
    mech, rn = _mech(), _rn()
    p = paths(split, out_dir)
    m1 = rows_with_windows(split, release)
    if not os.path.exists(p["features"]):
        raise RuntimeError(f"{p['features']} is missing: run --stage features first")
    feats = pd.read_parquet(p["features"])
    feats = feats.assign(t0=feats["t0"].map(_r), t1=feats["t1"].map(_r))
    drop = [c for c in feats if c in m1 and c not in ("uid", "t0", "t1")]
    d = m1.merge(feats.drop(columns=drop), on=["uid", "t0", "t1"], how="left", validate="many_to_one")
    info = {"rows_without_feature_row": int(d["reason"].isna().sum() - (d["reason"].isna() & d["a7_density"].notna()).sum())}
    tab_path = mech.paths(split)["table"]
    if not os.path.exists(tab_path):
        raise RuntimeError(f"{tab_path} is missing: run 116_mechanism.py --split {split} --stage table first")
    tab = pd.read_csv(tab_path)
    tp = tab[tab["kind"] == "published"].drop_duplicates(["pipe", "variant"]).set_index(["pipe", "variant"])
    xclass = tp["x_ref_1f"].map(mech.x_ref_class)
    d["x_ref_class"] = [xclass.get((a, b)) for a, b in zip(d["pipe"], d["variant"])]
    try:
        res = mech.read_results(mech.paths(split)["results"])
        pub = res[res["kind"] == "published"][["key", "pipe", "X"]]
        d = d.merge(pub, on=["key", "pipe"], how="left", validate="one_to_one")
        info["x_source"] = "116 x_rows"
    except (RuntimeError, KeyError, ValueError) as err:
        info["x_source"] = f"unavailable: {str(err)[:200]}"
    return d, info


def run_stats(split, release=False, B=B_BOOT, out_dir=None):
    guard_split(split, release)
    if split == "analysis" and B != B_BOOT:
        raise RuntimeError("the analysis split runs at the fixed in advance 2000 bootstrap draws only")
    mech = _mech()
    p = paths(split, out_dir)
    raw, info = assemble_frame(split, release, out_dir)
    pipes = sorted(set(raw["pipe"]))
    n_variants = {pp: mech.n_frozen_variants(pp) for pp in pipes}
    d = prepare_frame(raw, n_variants)
    out = {"split": split, "sha256_123": own_sha256(), "vendor_sha256": VENDOR_SHA256, "frozen_variants": n_variants,
           "design_classes": d.drop_duplicates(["pipe", "variant"])["x_ref_class"].value_counts().to_dict(),
           "feature_na_reasons": raw["reason"].fillna("ok").value_counts().to_dict(),
           "sigma_na_reasons": raw["reason_sigma"].fillna("ok").value_counts().to_dict(),
           "rows_with_x": int(d["X"].notna().sum()) if "X" in d else 0, **info}
    out.update(run_models(d, B))
    os.makedirs(os.path.dirname(p["summary"]), exist_ok=True)
    with open(p["summary"], "w") as fh:
        json.dump(_mg()._jsonable(out), fh, indent=1, default=float)
    print(f"spindles {split}: {len(d)} rows ({out['n_no_design_class']} without a design class), "
          f"{out['rows_with_x']} with X -> {p['summary']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "analysis"))
    ap.add_argument("--stage", default="all", choices=("features", "stats", "all"))
    ap.add_argument("--release", action="store_true", help="required for the analysis split")
    ap.add_argument("--limit", type=int, default=None, help="dev only: first N recordings")
    ap.add_argument("--max-windows", type=int, default=None, help="dev only: first N windows per recording")
    ap.add_argument("--boot", type=int, default=B_BOOT, help="dev only: bootstrap draws")
    a = ap.parse_args(argv)
    if a.split == "analysis" and not a.release:
        raise SystemExit("analysis split refused: it runs only with the explicit --release flag")
    if a.split == "analysis" and (a.limit or a.max_windows or a.boot != B_BOOT):
        raise SystemExit("the analysis split runs at the fixed in advance settings only (no --limit/--max-windows/--boot)")
    guard_split(a.split, a.release)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.stage in ("features", "all"):
        n = run_features(a.split, a.release, a.limit, None, a.max_windows)
        print(f"features: {n} windows")
    if a.stage in ("stats", "all"):
        run_stats(a.split, a.release, a.boot)


if __name__ == "__main__":
    main()
