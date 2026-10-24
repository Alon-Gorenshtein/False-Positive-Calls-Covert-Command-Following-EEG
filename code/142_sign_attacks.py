"""Attacks on the sign test's label-aligned heart-rate effect, on DISCOVERY data (137's Phase-1 sign rows, exposed
designs). Exploration. Refuses fresh, C2 and I-CARE inputs (137.assert_discovery). Built test-first
(code/test_sign_attacks.py).

1. Same heart, different schedules (recording_model): call ~ z(aligned) with RECORDING strata (fixed effects, the
   penalised fit of 116.fe_logit) and pipeline-variant indicators as covariates, 116._prep weights, patient-cluster
   bootstrap (relabelled draws, 2000). Within a recording the same heart-rate series meets the schedules of every
   pipeline-variant, so the alignment varies while the recording's drift is fixed. Also with z(placebo).
2. Placebo distribution: the run's labels shifted circularly against the heart rate, K = 40 shifts evenly from one
   run length + 1 s later to one run length + 1 s earlier (shift_grid), wrapping within the recording; heart rate per
   shifted span exactly as the aligned one (137.span_heart_rate over the extent +/- 5 s, the check's guard; a
   wrapped span in two pieces, each guarded, pieces under 10 s left out: circular_hr); |Welch t| by 137's
   trial_contrast. Per shift the pooled model call ~ z(placebo_k) (pipeline-variant FE) against call ~ z(aligned) on
   the same rows: the rows with all K shifts (common), and per shift on its own rows; empirical p = (1 + shifts with a
   placebo coefficient >= the aligned) / (K + 1).
3. Montage (descriptive): the exposed aligned model by the pipeline's spatial filter (116.SPATIAL): local (CSD or
   Hjorth: columbia, cornell, cruse11, edlow17), bipolar (cruse12p), referential (dubois, mbmi); with 140's
   contamination by class.
4. Off-feature EEG arousal markers (not the pipeline's own feature, so not circular): label-aligned |Welch t| of the
   trial-mean Morgoth awake probability (sleep_awake, 1-s steps, valid and valid_sleep seconds), and of the frontal
   log delta (1-4 Hz) / alpha (8-12 Hz) power ratio per second on the frontal channels (Fp1 Fp2 F7 F8) the pipeline
   does not read (none left for columbia, cornell and edlow17; alpha overlaps the motor and ERP bands). (a) call ~
   z(marker); (b) decomposition: a = weighted OLS slope of z(marker) on z(aligned HR) within pipeline-variant, b =
   z(marker) in call ~ z(aligned HR) + z(marker), indirect = a b, direct = z(aligned HR) in that model, total = call ~
   z(aligned HR) on the same rows; patient-cluster bootstrap; (c) the direct effect.
python 142_sign_attacks.py --pass [--shard i --nshards n] | --summary"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib
import json
import sys
import warnings

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

cc = importlib.import_module("140_cardiac_control")
ex, mech, rn, en, mc, ns = cc.ex, cc.mech, cc.rn, cc.en, cc.mc, cc.ns

SEED = 20261018
B_BOOT = 2000
K_SHIFTS = 40
FRONTAL = ("Fp1", "Fp2", "F7", "F8")
DELTA, ALPHA = (1.0, 4.0), (8.0, 12.0)
MONTAGE = {"columbia": "local", "cornell": "local", "cruse11": "local", "edlow17": "local", "cruse12p": "bipolar",
           "dubois": "referential", "mbmi": "referential"}
OUT = cc.OUT
ATT = os.path.join(OUT, "attacks")



def shift_grid(t0, t1, dur, k=K_SHIFTS):
    """K circular shifts (s) from one run length + 1 s later to one run length + 1 s earlier; [] if the recording
    is shorter than two such lengths."""
    L = (t1 - t0) + 1.0
    if dur - 2 * L < 0:
        return []
    return list(L + np.arange(k) * (dur - 2 * L) / (k - 1))


def circular_hr(ekg, fs, t0, t1, s, dur):
    """(t, hr, reason): heart rate over [t0 + s - 5, t1 + s + 5) taken circularly within the recording, times in the
    shifted frame (a wrapped piece carries + dur)."""
    a, b = t0 + s - ex.SIGN_MARGIN_S, t1 + s + ex.SIGN_MARGIN_S
    if b <= dur:
        pieces = [(a, b, 0.0)]
    elif a >= dur:
        pieces = [(a - dur, b - dur, dur)]
    else:
        pieces = [(a, dur, 0.0), (0.0, b - dur, dur)]
    ts, hs = [], []
    for p0, p1, off in pieces:
        if p1 - p0 < ex.MIN_SPAN_S:
            continue
        h = ex.span_heart_rate(ekg, fs, p0, p1)
        if h["reason"]:
            return None, None, h["reason"]
        ts.append(h["t"] + off)
        hs.append(h["hr"])
    if not ts:
        return None, None, "span_too_short"
    return np.concatenate(ts), np.concatenate(hs), None


def frontal_log_ratio(x, fs, names):
    """(4, seconds): per whole second, log delta / alpha mean power of Fp1 Fp2 F7 F8 (zero-phase band-passes over
    the whole recording)."""
    nm = mc.canon(names)
    X = np.asarray(x, float)[[nm.index(c) for c in mc.canon(list(FRONTAL))]]
    n = X.shape[1] // int(fs)
    p = lambda band: (mc.bandpass(X, fs, *band)[:, :n * int(fs)] ** 2).reshape(len(FRONTAL), n, int(fs)).mean(2)
    return np.log(p(DELTA) + mech.LOG_FLOOR) - np.log(p(ALPHA) + mech.LOG_FLOOR)


def off_feature_rows(module_names):
    """Indices into FRONTAL of the frontal channels the module does not read."""
    use = set(mc.canon(list(module_names)))
    return [i for i, c in enumerate(mc.canon(list(FRONTAL))) if c not in use]


def recording_attacks(rows, rec, ekg, fs, morgoth):
    """Circular-shift placebo contrasts and off-feature marker contrasts of every row of one recording (published
    overlay)."""
    dur = rec.x.shape[1] / fs
    fr = frontal_log_ratio(rec.x, fs, rec.names)
    tf = np.arange(fr.shape[1]) + 0.5
    ok = morgoth["valid"].astype(bool) & morgoth["valid_sleep"].astype(bool)
    tm, aw = morgoth.loc[ok, "t_s"].to_numpy(float) + 0.5, morgoth.loc[ok, "sleep_awake"].to_numpy(float)
    out, cache = [], {}
    for pipe, g in rows.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, "bdsp")
        offr = off_feature_rows([mc.canon(rec.names)[i] for i in mech.module_rows(mod, rec)])
        fser = fr[offr].mean(0) if offr else None
        for r in g.to_dict("records"):
            o = {"key": r["key"], "uid": r["uid"], "pipe": pipe, "variant": r["variant"], "n_frontal": len(offr)}
            try:
                ov = mech.overlay_for_row(mod, specs, r, "published")
            except (RuntimeError, ValueError, KeyError) as e:
                out.append(dict(o, att_reason=f"overlay: {str(e)[:80]}"))
                continue
            cfg = mech.module_cfg(mod, specs[r["variant"]])
            cons = mech.contrasts(mod, cfg)
            on, lab = np.asarray(ov.onsets_s, float), ov.labels
            t0, t1 = (float(v) for v in mod.extent(ov))
            pl, why = [], {}
            shifts = shift_grid(t0, t1, dur)
            for s in shifts:
                k = (t0, t1, round(s, 6))
                if k not in cache:
                    cache[k] = circular_hr(ekg, fs, t0, t1, s, dur)
                t, hr, w = cache[k]
                v = None if w else ex.trial_contrast(t, hr, on + s, lab, cons)["abs_t"]
                pl.append(v)
                if v is None:
                    why[w or "no_contrast"] = why.get(w or "no_contrast", 0) + 1
            o.update(shifts=shifts, abs_t_shift=pl, shift_na=why,
                     morgoth_abs_t=ex.trial_contrast(tm, aw, on, lab, cons)["abs_t"],
                     frontal_abs_t=None if fser is None else ex.trial_contrast(tf, fser, on, lab, cons)["abs_t"])
            out.append(o)
    return out



def recording_model(d, n_variants, cols, B=B_BOOT, seed=(SEED, 5)):
    """call ~ z(cols) with recording strata and pipeline-variant indicators (module docstring)."""
    e = mech._prep(d.dropna(subset=list(cols) + ["call"]).copy(), n_variants)
    y = e["call"].astype(bool).to_numpy(float)
    w = e["w"].to_numpy(float)
    codes = sorted(set(e["fe"]))
    Z = np.column_stack([mech.zscore(e[c].astype(float), w)[0] for c in cols]
                        + [(e["fe"] == c).to_numpy(float) for c in codes[1:]])
    sd = np.sqrt(np.average((Z - np.average(Z, 0, w)) ** 2, 0, w))
    sd = np.where(sd > 0, sd, 1.0)
    st = np.asarray(e["uid"].astype(str).to_numpy(), dtype=str)
    full = mech.fe_logit(y, Z / sd, st, w, lam=mech.LAM)
    ci = mech.cluster_index(e["patient"].to_numpy())
    draws = mech.mg.boot_idx(len(ci["ids"]), B, list(seed))
    k = len(cols)
    bd = np.full((B, k), np.nan)
    for b in range(B):
        idx, lab = mech.boot_sample(ci, draws[b])
        r = mech.fe_logit(y[idx], Z[idx] / sd, np.char.add(st[idx], lab.astype(str)), w[idx], lam=mech.LAM)
        if r["converged"]:
            bd[b] = (r["beta"] / sd)[:k]
    beta = (full["beta"] / sd)[:k]
    both = e.groupby("uid")["call"].agg(lambda v: v.astype(bool).nunique() == 2)
    out = {"n_rows": int(len(e)), "n_recordings": int(e["uid"].nunique()),
           "n_recordings_with_both_outcomes": int(both.sum()), "n_pipeline_variants": len(codes),
           "converged": bool(full["converged"]), "n_boot_failed": int(np.sum(~np.isfinite(bd[:, 0]))), "B": int(B),
           "coef": {}}
    for j, c in enumerate(cols):
        t = mech._ptests(beta[j], bd[:, j])
        out["coef"][f"z_{c}"] = {"beta": float(beta[j]), "or": float(np.exp(beta[j])),
                                 "or_ci95": [float(np.exp(v)) for v in t["ci95"]], "p_two_sided": t["p_two_sided"]}
    return out


def point_beta(d, n_variants, cols):
    e, y, Z, w, sd = cc._set(d, n_variants, cols, "call", None)
    r = mech.fe_logit(y, Z / sd, e["fe"].to_numpy(), w, lam=mech.LAM)
    return (r["beta"] / sd) if r["converged"] else np.full(len(cols), np.nan)


def empirical_p(aligned, placebos):
    p = np.asarray([v for v in placebos if np.isfinite(v)], float)
    return float((1 + np.sum(p >= aligned)) / (len(p) + 1))


def placebo_distribution(x, n_variants):
    """Aligned against the K circular-shift placebos (module docstring)."""
    P = np.array([[np.nan if v is None else v for v in row] if isinstance(row, list) and len(row) == K_SHIFTS
                  else [np.nan] * K_SHIFTS for row in x["abs_t_shift"]], float)
    out = {}
    have = np.isfinite(P).all(1) & x["abs_t_aligned"].notna().to_numpy()
    e = x[have].copy()
    for k in range(K_SHIFTS):
        e[f"pl{k}"] = P[have, k]
    ba = float(point_beta(e, n_variants, ("abs_t_aligned",))[0]) if len(e) else np.nan
    bp = [float(point_beta(e, n_variants, (f"pl{k}",))[0]) for k in range(K_SHIFTS)] if len(e) else []
    out["common_rows"] = {"n_rows": int(len(e)), "n_recordings": int(e["uid"].nunique()),
                          "aligned_or": float(np.exp(ba)), "placebo_or": [float(np.exp(v)) for v in bp],
                          "placebo_or_median": float(np.exp(np.nanmedian(bp))) if bp else None,
                          "placebo_or_95": [float(np.exp(np.nanquantile(bp, q))) for q in (0.025, 0.975)] if bp
                          else None, "empirical_p": empirical_p(ba, bp) if bp else None}
    per = []
    for k in range(K_SHIFTS):
        m = np.isfinite(P[:, k]) & x["abs_t_aligned"].notna().to_numpy()
        f = x[m].assign(pl=P[m, k])
        per.append({"n_rows": int(m.sum()), "aligned": float(point_beta(f, n_variants, ("abs_t_aligned",))[0]),
                    "placebo": float(point_beta(f, n_variants, ("pl",))[0])})
    dif = np.array([q["aligned"] - q["placebo"] for q in per])
    out["per_shift_rows"] = {"n_rows_range": [min(q["n_rows"] for q in per), max(q["n_rows"] for q in per)],
                             "aligned_or_range": [float(np.exp(min(q["aligned"] for q in per))),
                                                  float(np.exp(max(q["aligned"] for q in per)))],
                             "placebo_or": [float(np.exp(q["placebo"])) for q in per],
                             "share_placebo_at_or_above_aligned": float(np.mean(dif <= 0)),
                             "empirical_p": float((1 + np.sum(dif <= 0)) / (K_SHIFTS + 1))}
    na = {}
    for dct in x["shift_na"].dropna():
        for kk, v in (dct.items() if isinstance(dct, dict) else []):
            na[kk] = na.get(kk, 0) + int(v)
    out["shift_na"] = na
    out["rows_without_shifts"] = int(sum(1 for r in x["abs_t_shift"] if not (isinstance(r, list) and r)))
    return out


def decomposition(x, n_variants, marker, B=B_BOOT, seed=(SEED, 6)):
    """Aligned HR -> aligned off-feature marker -> call (module docstring, 4b-c)."""
    e, y, Z, w, sd = cc._set(x, n_variants, ("abs_t_aligned", marker), "call", None)
    fe = e["fe"].to_numpy()
    _, _, Zt, _, sdt = cc._set(e, n_variants, ("abs_t_aligned",), "call", None)

    def one(idx):
        dm = lambda v: v - (pd.Series(v * w[idx]).groupby(fe[idx]).transform("sum").to_numpy()
                            / pd.Series(w[idx]).groupby(fe[idx]).transform("sum").to_numpy())
        h, m = dm(Z[idx, 0]), dm(Z[idx, 1])
        a = float(np.sum(w[idx] * h * m) / np.sum(w[idx] * h * h))
        j = mech.fe_logit(y[idx], Z[idx] / sd, fe[idx], w[idx], lam=mech.LAM)
        t = mech.fe_logit(y[idx], Zt[idx] / sdt, fe[idx], w[idx], lam=mech.LAM)
        if not (j["converged"] and t["converged"]):
            return [np.nan] * 5
        bj, bt = j["beta"] / sd, t["beta"] / sdt
        return [a, bj[1], bj[0], bt[0], a * bj[1]]
    names = ("a_marker_on_hr", "b_marker", "direct_hr", "total_hr", "indirect")
    pt = one(np.arange(len(e)))
    ci = mech.cluster_index(e["patient"].to_numpy())
    draws = mech.mg.boot_idx(len(ci["ids"]), B, list(seed))
    bd = np.array([one(mech.boot_sample(ci, draws[b])[0]) for b in range(B)])
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
    return {"n_rows": int(len(e)), "n_recordings": int(e["uid"].nunique()), "B": int(B),
            **{n: {"est": float(pt[i]), "ci95": q(bd[:, i])} for i, n in enumerate(names)},
            "note": "a: SD of marker per SD of aligned HR; b, direct, total, indirect: log odds per SD"}



def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _path(uid):
    return os.path.join(ATT, f"{uid}.json")


def _exposed_aligned_keys():
    d = pd.read_csv(cc.SIGN_ROWS, usecols=["key", "pipe", "variant", "abs_t_aligned"], dtype={"key": str})
    typ = ns.pinned_typing()
    imm = np.array([typ.get((p, v)) == "immune" for p, v in zip(d["pipe"], d["variant"])])
    return set(d.loc[~imm & d["abs_t_aligned"].notna(), "key"])


def run_pass(shard, nshards):
    pool = ex.phase1_pool()
    rows = ex.load_sign_rows()
    rows = rows[rows["key"].isin(_exposed_aligned_keys())]
    man = pd.read_csv(os.path.join(ex.EKG_FULL, "manifest.csv"), dtype={"uid": str, "status": str, "channel": str})
    man = man.set_index("uid")
    uids = sorted(set(rows["uid"]) & set(pool))
    by = dict(tuple(rows.groupby("uid")))
    for i, uid in enumerate(uids):
        if i % nshards != shard or os.path.exists(_path(uid)):
            continue
        ex.assert_discovery(pool[uid])
        r = man.loc[uid]
        n = ex.x19_length(pool[uid]["path"])
        ekg = ex.load_full_ekg(os.path.join(ex.EKG_FULL, f"{uid}.npz"), dict(r.to_dict(), uid=uid), n)
        rec = en._load(pool[uid])
        if rec.x.shape[1] != len(ekg):
            raise RuntimeError(f"{uid}: x19 and EKG lengths differ")
        res = recording_attacks(by[uid], rec, ekg, float(r["fs"]), pd.read_parquet(pool[uid]["morgoth"]))
        del rec
        os.makedirs(ATT, exist_ok=True)
        with open(_path(uid) + ".tmp", "w") as fh:
            json.dump({"sha256_142": own_sha256(), "uid": uid, "rows": res}, fh, default=float)
        os.replace(_path(uid) + ".tmp", _path(uid))
        print(f"{i + 1}/{len(uids)} {uid}: {len(res)} rows", flush=True)


def summary(B=B_BOOT):
    with open(cc.SIGN_ROWS, "rb") as fh:
        rows_sha = hashlib.sha256(fh.read()).hexdigest()
    d = pd.read_csv(cc.SIGN_ROWS, dtype={"uid": str, "patient": str, "key": str})
    keys = _exposed_aligned_keys()
    x = d[d["key"].isin(keys)].copy()
    res = []
    for u in sorted(set(x["uid"])):
        o = json.load(open(_path(u)))
        if o["sha256_142"] != own_sha256():
            raise RuntimeError(f"{u}: attack result of another version of 142")
        res += o["rows"]
    a = pd.DataFrame(res).drop(columns=["uid", "pipe", "variant"], errors="ignore")
    if not a["key"].is_unique or set(a["key"]) != set(x["key"]):
        raise RuntimeError("attack rows do not match the exposed aligned sign rows one to one")
    x = x.merge(a, on="key", how="left", validate="one_to_one")
    typ = ns.pinned_typing()
    ve = {p: n for p, n in ns.non_immune_variant_counts(sorted(set(d["pipe"])), typ).items() if n}
    out = {"label": "attacks on the sign test's aligned heart-rate effect (discovery data, exploration)",
           "sha256_142": own_sha256(), "sign_rows_sha256": rows_sha,
           "recording_stratified": {"aligned": recording_model(x, ve, ("abs_t_aligned",), B),
                                    "aligned_and_placebo": recording_model(x, ve, ("abs_t_aligned", "abs_t_placebo"),
                                                                           B, (SEED, 7))},
           "placebo_distribution": placebo_distribution(x, ve)}
    cr = os.path.join(OUT, "cardiac_rows.csv")
    contam = (pd.read_csv(cr, usecols=["key", "contam_amp", "contam_ratio"], dtype={"key": str}).set_index("key")
              if os.path.exists(cr) else None)
    mont = {}
    for cls in sorted(set(MONTAGE.values())):
        pipes = [p for p, c in MONTAGE.items() if c == cls and p in ve]
        f = x[x["pipe"].isin(pipes)]
        if f.empty:
            continue
        mont[cls] = {"pipelines": pipes, "model": ex.sign_model(f, {p: ve[p] for p in pipes}, B, f"montage_{cls}",
                                                                ("abs_t_aligned",))}
        if contam is not None:
            mont[cls]["contam_amp_median"] = float(contam.loc[f["key"], "contam_amp"].median())
            mont[cls]["contam_ratio_median"] = float(contam.loc[f["key"], "contam_ratio"].median())
    out["montage"] = mont
    off = {}
    for m in ("morgoth_abs_t", "frontal_abs_t"):
        e = x.dropna(subset=[m])
        off[m] = {"pipelines": sorted(set(e["pipe"])),
                  "a_marker_predicts_calls": ex.sign_model(e, {p: ve[p] for p in set(e["pipe"])}, B, f"off_{m}", (m,)),
                  "decomposition": decomposition(e, {p: ve[p] for p in set(e["pipe"])}, m, B)}
    out["off_feature"] = off
    x.drop(columns=["shifts", "abs_t_shift", "shift_na"]).to_csv(os.path.join(OUT, "attack_rows.csv"), index=False)
    with open(os.path.join(OUT, "attack_summary.json"), "w") as fh:
        json.dump(mech.mg._jsonable(out), fh, indent=1, default=float)
    print("written", os.path.join(OUT, "attack_summary.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--pass", dest="run", action="store_true")
    m.add_argument("--summary", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.run:
        run_pass(a.shard, a.nshards)
    else:
        summary()


if __name__ == "__main__":
    main()
