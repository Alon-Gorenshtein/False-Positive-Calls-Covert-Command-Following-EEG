"""Sign-test follow-ups on DISCOVERY data (the Phase-1 BDSP command rows of 137's sign test): formal contrasts from
shared bootstrap draws, and the cardiac-artifact control. Exploration, not confirmation. Refuses fresh, C2 and
I-CARE inputs (137.assert_discovery). Built test-first (code/test_cardiac_control.py).

Contrasts (shared_draw_fits, contrast): the sign models refitted as 137's sign_model does (116.fe_model: 116._prep
weights, pipeline-variant FE, penalised fit on covariates over their weighted SD), every set on the SAME patient
draws (one draw of the union of patients; a set keeps the drawn patients it holds, each copy its own cluster), so a
difference of coefficients has a bootstrap CI: aligned minus placebo (exposed rows with both); exposed minus
randomised (the same rows: exposed rows with both an aligned contrast and a randomised-method one; the
randomised-method model on rand_call with the timing-group FE); exposed minus immune.

Cardiac-artifact control: a label-aligned heart-rate difference puts more QRS complexes into one label class, and
the cardiac field reaches scalp EEG, so a classifier could be reading the artifact, not arousal.
- Contamination per run (cardiac_contamination): R peaks of the full-span EKG over the run's extent +/- 5 s (132's
  detector, beats re-centred on the apex by 139.realign_peaks); the module's spatially transformed channels
  (116.spatial) of the extent (clipped to the recording), band-passed to the family band (116.MOTOR_BAND or ERP_BAND, as 128's VER reads them).
  Per channel the RMS of the QRS-locked average (+/- 100 ms) and the mean squared RMS of 20 random-locked averages
  of as many epochs (seeded by uid, extent and pipeline). contam_ratio = median over channels of
  log(RMS QRS-locked / RMS random-locked) (grows with the number of beats); contam_amp = median over channels of
  sqrt(max(0, RMS_qrs^2 - RMS_random^2)) / channel SD (the cardiac amplitude share, independent of the beat count).
- (a) Exposed aligned effect by contamination tertile (tertiles within pipeline), high minus low tertile on shared
  draws, and the aligned x contamination interaction (call ~ z(aligned) + z(contam) + z(aligned) z(contam)); both
  measures.
- (b) Beat-count contrast (count_contrast): |Welch t| of beats per second per trial between the design's label
  groups (116.CONTRASTS; trial = onset to the next onset); call ~ z(count) alone and with z(aligned); the randomised
  method's own count contrast as control.
python 140_cardiac_control.py --cardiac [--shard i --nshards n] | --summary"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib
import json
import math
import sys
import warnings
import zlib

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

ex = importlib.import_module("137_arousal_explore")
gd = importlib.import_module("139_ekg_guard")
kt, mech, rn, en, mc, ns = ex.kt, ex.mech, ex.rn, ex.en, ex.mc, ex.ns

SEED = 20261018
B_BOOT = 2000
QRS_HALF_S, N_RAND, MIN_BEATS = 0.1, 20, 10
OUT = ex.OUT
CARD = os.path.join(OUT, "cardiac")
SIGN_ROWS, SIGN_SUMMARY = os.path.join(OUT, "sign_rows.csv"), os.path.join(OUT, "sign_summary.json")



def cardiac_contamination(xs, fs, beats, rng, n_rand=N_RAND):
    """QRS-locked against random-locked averages of band-passed channels xs (channels, samples) at beat sample
    indices (module docstring)."""
    h = int(round(QRS_HALF_S * fs))
    n = xs.shape[1]
    b = np.asarray(beats, int)
    b = b[(b - h >= 0) & (b + h + 1 <= n)]
    if len(b) < MIN_BEATS:
        return {"contam_ratio": None, "contam_amp": None, "n_beats_eeg": int(len(b)), "contam_reason": "too_few_beats"}
    w = np.arange(-h, h + 1)
    rq2 = np.mean(xs[:, b[:, None] + w].mean(1) ** 2, axis=1)
    rr2 = np.mean([np.mean(xs[:, rng.integers(h, n - h, len(b))[:, None] + w].mean(1) ** 2, axis=1)
                   for _ in range(n_rand)], axis=0)
    sd = xs.std(1)
    ok = (sd > 0) & (rr2 > 0)
    if not ok.any():
        return {"contam_ratio": None, "contam_amp": None, "n_beats_eeg": int(len(b)), "contam_reason": "no_variance"}
    return {"contam_ratio": float(np.median(0.5 * np.log(rq2[ok] / rr2[ok]))),
            "contam_amp": float(np.median(np.sqrt(np.maximum(0.0, rq2[ok] - rr2[ok])) / sd[ok])),
            "n_beats_eeg": int(len(b)), "contam_reason": None}


def count_contrast(beat_t, onsets, labels, cons):
    """|Welch t| of beats per second per trial between the label groups (trial = onset to the next onset, the last
    one median-ISI long), the largest over the design's contrasts; None without two trials a side."""
    on = np.asarray(onsets, float)
    lab = np.asarray(labels)
    if len(on) < 4:
        return {"abs_t": None}
    end = np.r_[on[1:], on[-1] + np.median(np.diff(on))]
    bt = np.sort(np.asarray(beat_t, float))
    rate = (np.searchsorted(bt, end) - np.searchsorted(bt, on)) / (end - on)
    best = None
    for pos, neg in cons:
        a, b = rate[np.isin(lab, pos)], rate[np.isin(lab, neg)]
        if len(a) < 2 or len(b) < 2:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if se > 0:
            v = abs(a.mean() - b.mean()) / se
            best = v if best is None else max(best, v)
    return {"abs_t": best}


def eeg_span(x, fs, t0, t1):
    """mc.span clipped to the recording ([t0, t1) s; an extent may end a fraction of a second past the last sample):
    (segment, first sample index)."""
    i0, i1 = max(0, int(math.floor(t0 * fs))), min(x.shape[-1], int(math.ceil(t1 * fs)))
    return x[..., i0:i1], i0


def _beats(ekg, fs, t0, t1):
    """Re-aligned R-peak sample indices (recording samples) over [t0 - 5 s, t1 + 5 s) clipped to the recording."""
    a, b = max(0, int(round((t0 - ex.SIGN_MARGIN_S) * fs))), min(len(ekg), int(round((t1 + ex.SIGN_MARGIN_S) * fs)))
    if b - a < ex.MIN_SPAN_S * fs:
        return np.array([], int)
    p, xf = kt._detect(np.asarray(ekg[a:b], float), fs)
    return gd.realign_peaks(xf, p, fs) + a


def recording_cardiac(rows, rec, ekg, fs):
    """Contamination and beat-count contrast of every row of one recording (published overlay and, where the row
    has one, its randomised overlay)."""
    out, cache = [], {}
    for pipe, g in rows.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, "bdsp")
        lo, hi = mech.ERP_BAND if mod.FAMILY == "erp" else mech.MOTOR_BAND
        for r in g.to_dict("records"):
            o = {"key": r["key"], "uid": r["uid"], "pipe": pipe, "variant": r["variant"]}
            for kind, tag in (("published", ""), ("randomized", "rand_")):
                if kind == "randomized" and not isinstance(r.get("rand_variant"), str):
                    continue
                try:
                    ov = mech.overlay_for_row(mod, specs, r, kind)
                except (RuntimeError, ValueError, KeyError) as e:
                    o[tag + "contam_reason"] = f"overlay: {str(e)[:80]}"
                    continue
                cfg = mech.module_cfg(mod, specs[r["variant"] if kind == "published" else r["rand_variant"]])
                t0, t1 = (float(v) for v in mod.extent(ov))
                q = _beats(ekg, fs, t0, t1)
                k = (pipe, t0, t1)
                if k not in cache:
                    seg, i0 = eeg_span(np.asarray(rec.x), fs, t0, t1)
                    xs = mc.bandpass(mech.spatial(mod, rec, seg)[0], fs, lo, hi)
                    rng = np.random.default_rng([SEED, zlib.crc32(str(r["uid"]).encode()), int(round(t0 * fs)),
                                                 zlib.crc32(pipe.encode())])
                    cache[k] = cardiac_contamination(xs, fs, q - i0, rng)
                c = count_contrast(q / fs, ov.onsets_s, ov.labels, mech.contrasts(mod, cfg))
                o.update({tag + kk: v for kk, v in cache[k].items()})
                o[tag + "count_abs_t"] = c["abs_t"]
            out.append(o)
    return out



def _set(d, n_variants, cols, call, fe):
    e = d.dropna(subset=list(cols) + [call]).copy()
    if fe:
        e["variant"] = e[fe]
    e = mech._prep(e, n_variants)
    y = e[call].astype(bool).to_numpy(float)
    Z = np.column_stack([mech.zscore(e[c].astype(float), e["w"])[0] for c in cols])
    w = e["w"].to_numpy(float)
    sd = np.sqrt(np.average((Z - np.average(Z, 0, w)) ** 2, 0, w))
    return e, y, Z, w, np.where(sd > 0, sd, 1.0)


def shared_draw_fits(sets, B=B_BOOT, seed=(SEED, 1)):
    """{name: (rows, n_variants, covariate columns, call column, FE column or None)} -> per set the point
    coefficients (per SD) and their bootstrap draws, every set on the same patient draws (module docstring)."""
    prep = {k: _set(*v) for k, v in sets.items()}
    union = np.unique(np.concatenate([e["patient"].astype(str).to_numpy() for e, *_ in prep.values()]))
    draws = mech.mg.boot_idx(len(union), B, list(seed))
    out = {}
    for name, (e, y, Z, w, sd) in prep.items():
        cols = [f"z_{c}" for c in sets[name][2]]
        fe = e["fe"].to_numpy()
        full = mech.fe_logit(y, Z / sd, fe, w, lam=mech.LAM)
        ci = mech.cluster_index(e["patient"].to_numpy())
        pos = np.full(len(union), -1)
        pos[np.searchsorted(union, ci["ids"])] = np.arange(len(ci["ids"]))
        bd = np.full((B, Z.shape[1]), np.nan)
        for b in range(B):
            sel = pos[draws[b]]
            idx, _ = mech.boot_sample(ci, sel[sel >= 0])
            r = mech.fe_logit(y[idx], Z[idx] / sd, fe[idx], w[idx], lam=mech.LAM)
            if r["converged"]:
                bd[b] = r["beta"] / sd
        beta = full["beta"] / sd
        out[name] = {"n_rows": int(len(e)), "n_patients": int(e["patient"].nunique()),
                     "n_recordings": int(e["uid"].nunique()), "converged": bool(full["converged"]),
                     "n_boot_failed": int(np.sum(~np.isfinite(bd[:, 0]))),
                     "beta": {c: float(beta[j]) for j, c in enumerate(cols)},
                     "draws": {c: bd[:, j] for j, c in enumerate(cols)}}
        out[name]["coef"] = {c: {"beta": out[name]["beta"][c], "or": float(np.exp(out[name]["beta"][c])),
                                 "or_ci95": [float(np.exp(np.nanquantile(bd[:, j], q))) for q in (0.025, 0.975)]}
                             for j, c in enumerate(cols)}
    return out


def contrast(fits, a, b):
    """Coefficient a minus coefficient b ((set, covariate) each) on the shared draws: difference of log odds ratios
    per SD, its percentile 95% CI and bootstrap p."""
    d0 = fits[a[0]]["beta"][a[1]] - fits[b[0]]["beta"][b[1]]
    dd = fits[a[0]]["draws"][a[1]] - fits[b[0]]["draws"][b[1]]
    t = mech._ptests(d0, dd)
    return {"a": list(a), "b": list(b), "diff": float(d0), "ci95": t["ci95"], "p_two_sided": t["p_two_sided"],
            "ratio_of_or": float(np.exp(d0)), "ratio_of_or_ci95": None if t["ci95"] is None
            else [float(np.exp(v)) for v in t["ci95"]]}


def _public(fits):
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("draws", "beta")} for k, v in fits.items()}



def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _path(uid):
    return os.path.join(CARD, f"{uid}.json")


def run_cardiac(shard, nshards):
    pool = ex.phase1_pool()
    rows = ex.load_sign_rows()
    man = pd.read_csv(os.path.join(ex.EKG_FULL, "manifest.csv"), dtype={"uid": str, "status": str, "channel": str})
    man = man.set_index("uid")
    uids = sorted(set(rows["uid"]) & set(pool))
    by = dict(tuple(rows.groupby("uid")))
    for i, uid in enumerate(uids):
        if i % nshards != shard or os.path.exists(_path(uid)):
            continue
        ex.assert_discovery(pool[uid])
        r = man.loc[uid]
        if r["status"] != "ok":
            res = [{"uid": uid, "key": k, "contam_reason": "ekg_" + r["status"]} for k in by[uid]["key"]]
        else:
            n = ex.x19_length(pool[uid]["path"])
            ekg = ex.load_full_ekg(os.path.join(ex.EKG_FULL, f"{uid}.npz"), dict(r.to_dict(), uid=uid), n)
            rec = en._load(pool[uid])
            if rec.x.shape[1] != len(ekg):
                raise RuntimeError(f"{uid}: x19 and EKG lengths differ")
            res = recording_cardiac(by[uid], rec, ekg, float(r["fs"]))
            del rec
        os.makedirs(CARD, exist_ok=True)
        with open(_path(uid) + ".tmp", "w") as fh:
            json.dump({"sha256_140": own_sha256(), "sha256_137": ex.own_sha256(), "uid": uid, "rows": res}, fh,
                      default=float)
        os.replace(_path(uid) + ".tmp", _path(uid))
        print(f"{i + 1}/{len(uids)} {uid}: {len(res)} rows", flush=True)


def _tertile(d, col):
    return d.groupby("pipe")[col].transform(lambda v: pd.qcut(v, 3, labels=False, duplicates="drop")
                                            if v.notna().sum() >= 3 else pd.Series(np.nan, index=v.index))


def summary(B=B_BOOT):
    sm = json.load(open(SIGN_SUMMARY))
    with open(SIGN_ROWS, "rb") as fh:
        rows_sha = hashlib.sha256(fh.read()).hexdigest()
    d = pd.read_csv(SIGN_ROWS, dtype={"uid": str, "patient": str, "key": str})
    res = []
    for u in sorted(set(d["uid"])):
        o = json.load(open(_path(u)))
        if o["sha256_140"] != own_sha256():
            raise RuntimeError(f"{u}: cardiac result of another version of 140")
        res += o["rows"]
    c = pd.DataFrame(res).drop(columns=["uid", "pipe", "variant"], errors="ignore")
    if not c["key"].is_unique or set(c["key"]) != set(d["key"]):
        raise RuntimeError("cardiac rows do not match the sign rows one to one")
    d = d.merge(c, on="key", how="left", validate="one_to_one")
    typ = ns.pinned_typing()
    imm = np.array([typ.get((p, v)) == "immune" for p, v in zip(d["pipe"], d["variant"])])
    pipes = sorted(set(d["pipe"]))
    nv_exp = ns.non_immune_variant_counts(pipes, typ)
    nv_imm = {p: sum(typ.get((p, v)) == "immune" for v in rn.variant_specs(rn.load_pipe(p), "bdsp")) for p in pipes}
    ve = {p: n for p, n in nv_exp.items() if n}
    vi = {p: n for p, n in nv_imm.items() if n}
    x = d[~imm].copy()
    same = x.dropna(subset=["abs_t_aligned", "rand_abs_t_aligned", "rand_call"]).copy()
    same["rand_grp"] = same["rand_variant"]
    nv_r = same.groupby("pipe")["rand_grp"].nunique().to_dict()
    al, pl = "z_abs_t_aligned", "z_abs_t_placebo"
    f1 = shared_draw_fits({
        "exposed_aligned_vs_placebo": (x, ve, ("abs_t_aligned", "abs_t_placebo"), "call", None),
        "exposed_same_rows": (same, ve, ("abs_t_aligned",), "call", None),
        "randomised_same_rows": (same, nv_r, ("rand_abs_t_aligned",), "rand_call", "rand_grp"),
        "exposed_aligned": (x, ve, ("abs_t_aligned",), "call", None),
        "immune_aligned": (d[imm], vi, ("abs_t_aligned",), "call", None)}, B, (SEED, 1))
    out = {"label": "sign-test contrasts and cardiac-artifact control (discovery data, exploration)",
           "sha256_140": own_sha256(), "sign_rows_sha256": rows_sha, "sign_sha256_137": sm["sha256_137"],
           "check_point_estimates": {
               "exposed_aligned_vs_placebo": [f1["exposed_aligned_vs_placebo"]["beta"][al],
                                              sm["exposed_aligned_vs_placebo"]["coef"][al]["beta"]],
               "exposed_aligned": [f1["exposed_aligned"]["beta"][al], sm["exposed_aligned"]["coef"][al]["beta"]],
               "immune_aligned": [f1["immune_aligned"]["beta"][al], sm["immune_aligned"]["coef"][al]["beta"]]},
           "fits": _public(f1),
           "contrasts": {
               "aligned_minus_placebo_exposed": contrast(f1, ("exposed_aligned_vs_placebo", al),
                                                         ("exposed_aligned_vs_placebo", pl)),
               "exposed_minus_randomised_same_rows": contrast(f1, ("exposed_same_rows", al),
                                                              ("randomised_same_rows", "z_rand_abs_t_aligned")),
               "exposed_minus_immune": contrast(f1, ("exposed_aligned", al), ("immune_aligned", al))}}
    out["contamination_na"] = x["contam_reason"].dropna().astype(str).str.split(":").str[0].value_counts().to_dict()
    for m in ("contam_ratio", "contam_amp"):
        e = x.dropna(subset=["abs_t_aligned", m]).copy()
        e["tert"] = _tertile(e, m)
        sets = {f"t{t}": (e[e["tert"] == t], ve, ("abs_t_aligned",), "call", None) for t in (0, 1, 2)}
        zt = lambda v: (v - v.mean()) / v.std()
        e["inter"] = zt(e["abs_t_aligned"].astype(float)) * zt(e[m].astype(float))
        sets["interaction"] = (e, ve, ("abs_t_aligned", m, "inter"), "call", None)
        f = shared_draw_fits(sets, B, (SEED, 2))
        out[m] = {"describe": e[m].describe(percentiles=[0.1, 0.5, 0.9]).to_dict(),
                  "by_pipeline_median": e.groupby("pipe")[m].median().to_dict(), "fits": _public(f),
                  "high_minus_low_tertile": contrast(f, ("t2", al), ("t0", al))}
    cnt = shared_draw_fits({
        "exposed_count": (x, ve, ("count_abs_t",), "call", None),
        "exposed_count_and_aligned": (x, ve, ("count_abs_t", "abs_t_aligned"), "call", None),
        "randomised_count": (same, nv_r, ("rand_count_abs_t",), "rand_call", "rand_grp")}, B, (SEED, 3))
    out["count_contrast"] = {"fits": _public(cnt),
                             "spearman_count_vs_aligned": float(x[["count_abs_t", "abs_t_aligned"]].corr(
                                 method="spearman").iloc[0, 1])}
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "cardiac_rows.csv"), index=False)
    with open(os.path.join(OUT, "cardiac_summary.json"), "w") as fh:
        json.dump(mech.mg._jsonable(out), fh, indent=1, default=float)
    print("written", os.path.join(OUT, "cardiac_summary.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--cardiac", action="store_true")
    m.add_argument("--summary", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.cardiac:
        run_cardiac(a.shard, a.nshards)
    else:
        summary()


if __name__ == "__main__":
    main()
