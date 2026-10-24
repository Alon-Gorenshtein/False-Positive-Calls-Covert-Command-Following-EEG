"""Arousal exploration on DISCOVERY data only (the Phase-1 BDSP analysis recordings, already read): the
within-recording chain from moment-to-moment EEG non-stationarity to false calls. Exploration, not confirmation:
whatever specification wins is fixed, with its date, before it is confirmed once on the fresh 400 by separate code.
Refuses fresh, C2 and I-CARE inputs (assert_discovery). Built test-first (code/test_arousal_explore.py).

STEP 0 (check of the moment-to-moment framing, no EKG):
- Recordings: every recording of the Phase-1 pool (engine._items("bdsp", "analysis") minus 114's touched list; the
  re-powered STEP 0; the first pass on 50 seeded recordings was underpowered, 14 with >= 3 tiles).
- Tiles: per pipeline, consecutive non-overlapping overlays of the variant's own schedule (tiles: start 10 s, then each
  next start 1 s after the previous extent ends, shifted by any lead-in of the extent before its start), kept while
  the extent lies inside the recording and
  holds no padding second (108.padding_seconds); k = tile index. Schedule seed 114._rng(pipe, uid, k, 1).
- Pipelines and statistic (statistic): cornell, variants jackknife_by_keep40_r4 (its BDSP validity primary; jackknife
  variance) and jackknife_by_keep40_r2 (two runs: shorter tiles), the frozen run()'s continuous statistic (max |z| of
  the TGT); edlow17, its a-priori BDSP default, the
  observed cross-validated accuracy exactly as its run() computes it before any permutation (no permutations run).
  Run seed 114._rng(pipe, uid, k, 2, crc(variant)); module globals as 114._run applies them.
- VER_tile: 128.window_ver on the tile (the module's log feature on the 116 grid over 10 phase-randomised surrogates).
- Association (within_association), per pipeline-variant group, statistic and VER z-scored within the group: the
  PRIMARY within slope = OLS of z(statistic) on z(VER) with recording fixed effects, on recordings with >= 2 usable
  tiles of the group, recording-cluster bootstrap 95% CI (relabelled draws, 2000, seed [20261017, 1], a recording's
  groups resampled together); the pooled within-recording Spearman rho (Pearson correlation of the ranks centred
  within each recording) on recordings with >= 3 tiles; the between-recording Spearman of the recording means, side
  by side.
- NOGO (step0_verdict): pooled within slope <= 0 or rho < 0.10: non-stationarity would be a trait of the recording,
  not of moments.

SIGN TEST (label-aligned heart rate; the command pipelines are two-sided or classifier-based, so a label-locked heart-
rate difference of either sign can raise a call: the test is on its magnitude, against a placebo):
- Rows: 128's Phase-1 command rows (ncmd_bdsp, j = 0, status ok, sequence windows, 113's Columbia rows, touched
  recordings dropped; load_sign_rows) with each row's randomised-method fields.
- Heart rate: the full-span EKG (136_ekg_span; checked against its manifest by the sha256 of the array, uid, start 0
  and the cached x19 length), per span by the check's detector with its signal-quality guard (132; span_heart_rate)
  over the run's extent +/- 5 s, clipped to the recording (under 10 s left: "span_too_short").
- Exposure (trial_contrast): |Welch t| of trial-mean heart rate between the design's label groups (116.CONTRASTS;
  trial = onset to the next onset), the largest over the design's contrasts. Placebo (placebo_shift): the same onsets
  and labels one run length + 1 s later (else earlier), non-overlapping, in the same recording.
- Model (sign_model): 116.fe_model, call ~ z(aligned) on every row with an aligned contrast, and call ~ z(aligned) +
  z(placebo) on the rows with both (long runs leave no placebo span in short recordings), pipeline-variant FE,
  116._prep weights over the frozen variants of each set, patient bootstrap. Sets: exposed (not typed immune;
  primary), immune (control), each row's randomised method on its trial-randomised schedule (control; FE = the
  timing-group variant); per pipeline. Exposed set also: within one stable Morgoth state (stable_state: one label holds
  the run and the two halves share it), with run-level EMG as a covariate (run_emg: median over the 19 channels of the
  log mean 45-58 Hz power over the run's extent; the x19 is 128 Hz with a 60-Hz notch, so 45-70 Hz is not available),
  both, and A9-clean (a9_clean: 116's activation-procedure rule on 121's tables).
python 137_arousal_explore.py --step0 [--shard i --nshards n] | --step0-summary | --sign [--shard i --nshards n] |
                              --sign-summary"""
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

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

ns = importlib.import_module("128_nonstationarity")
mech, rn, en, mc = ns.mech, ns.rn, ns.en, ns.mc
bn = importlib.import_module("108_bdsp_null")
wr = importlib.import_module("bench.wrappers")

SEED = 20261017
MIN_TILES_SLOPE, MIN_TILES_RHO = 2, 3
TILE_FIRST_S, TILE_GAP_S = 10.0, 1.0
B_BOOT = 2000
RHO_MIN = 0.10
OUT = os.path.join(ROOT, "output", "intermediate", "_arousal_explore")
FRESH_LIST = os.path.join(ROOT, "output", "bdsp", "meta", "c2_fresh_400.csv")
FORBIDDEN = ("c2_fresh", "_fresh_shards", "_nonstationarity_fresh", "keep_windows", "keep_ecg", "icare", "i-care")
def step0_variants():
    """(pipeline, variant) pairs of STEP 0: cornell jackknife r4 (its BDSP validity primary) and r2 (shorter runs),
    edlow17's a-priori BDSP default."""
    return [("cornell", "jackknife_by_keep40_r4"), ("cornell", "jackknife_by_keep40_r2"),
            ("edlow17", rn.primary_variant(rn.load_pipe("edlow17"), "bdsp"))]



def fresh_uids():
    return set(pd.read_csv(FRESH_LIST, usecols=["uid"], dtype=str)["uid"]) if os.path.exists(FRESH_LIST) else set()


def assert_discovery(item):
    """Refuse anything that is not a Phase-1 discovery recording: a fresh-draw uid, or a fresh, C2 or I-CARE path."""
    path = str(item.get("path", "")).lower()
    if any(f in path for f in FORBIDDEN) or str(item["uid"]) in fresh_uids():
        raise RuntimeError(f"{item['uid']} ({item.get('path')}): not discovery data; refused")


def phase1_pool():
    items = {it["uid"]: it for it in en._items("bdsp", "analysis")}
    return {u: it for u, it in items.items() if u not in set(mech.mg.touched_uids())}


def step0_sample(pool_uids):
    """Every Phase-1 recording (the re-powered STEP 0)."""
    return sorted(pool_uids)



def tiles(rec, pipe, variant):
    """Consecutive non-overlapping clean overlays of one variant over the recording (module docstring)."""
    mod = rn.load_pipe(pipe)
    spec = rn.variant_specs(mod, "bdsp")[variant]
    pad = bn.padding_seconds(rec.x, rec.fs)
    dur = rec.x.shape[1] / rec.fs
    out, start, k = [], TILE_FIRST_S, 0
    while True:
        ov = rn._sched(mod, mod.schedule, spec, start, rn._rng(pipe, rec.uid, k, 1))
        t0, t1 = (float(v) for v in mod.extent(ov))
        if t1 > dur:
            break
        if t0 >= 0 and not pad[int(math.floor(t0)):int(math.ceil(t1))].any():
            out.append({"k": len(out), "start_s": start, "extent": (t0, t1), "overlay": ov, "seed_k": k})
        start, k = t1 + TILE_GAP_S + max(0.0, start - t0), k + 1
    return out


def statistic(pipe, mod, spec, rec, ov, rng):
    """The continuous statistic of one tile (module docstring)."""
    if pipe == "cornell":
        return float(rn._run(mod, mod.run, spec, rec, ov, rng)["stat"])
    if pipe == "edlow17":
        with wr.module_globals(mod, spec["globals"]):
            c = mod.mc.cfg(mod.VARIANTS, spec["arg"])
            g = mod.mc.gen(rng)
            X, blocks, keep = mod._feat(rec, ov)
            bcls = np.asarray(ov.meta["block_class"])
            mod.require(bcls, blocks)
            y = bcls[blocks]
            return float(mod.cv_accuracy(X, y, mod.cv_partitions(len(y), blocks, c["cv_unit"], g)))
    raise ValueError(f"no STEP 0 statistic for {pipe}")


def recording_step0(rec):
    rows = []
    for pipe, variant in step0_variants():
        mod = rn.load_pipe(pipe)
        spec = rn.variant_specs(mod, "bdsp")[variant]
        cfg = mech.module_cfg(mod, spec)
        for t in tiles(rec, pipe, variant):
            r = {"uid": rec.uid, "patient": rec.patient, "pipe": pipe, "variant": variant, "k": t["k"],
                 "t0": t["extent"][0], "t1": t["extent"][1]}
            try:
                r["stat"] = statistic(pipe, mod, spec, rec, t["overlay"],
                                      rn._rng(pipe, rec.uid, t["seed_k"], 2, rn._crc(variant)))
            except mc.NoUsableTrials as e:
                r.update(stat=None, reason=f"no_usable_trials: {str(e)[:80]}")
            v = ns.window_ver(rec, mod, cfg, t["overlay"])
            r.update(ver=v["log_ver"], ver_reason=v["reason"])
            rows.append(r)
            rn._clear_cache()
    return rows



def within_association(d, B=B_BOOT):
    """Within- and between-recording association of the statistic with VER (module docstring). Groups are
    pipeline-variants; z-scores within a group; the within slope uses recordings with >= MIN_TILES_SLOPE tiles of a
    group, the centred-rank rho >= MIN_TILES_RHO; the bootstrap resamples recordings (all their groups together)."""
    e = d.dropna(subset=["ver", "stat"]).copy()
    e["grp"] = e["pipe"].astype(str) + "|" + e["variant"].astype(str)
    e["n_g"] = e.groupby(["grp", "uid"])["ver"].transform("size")
    e = e[e["n_g"] >= MIN_TILES_SLOPE].copy()
    for c in ("ver", "stat"):
        e["z_" + c] = e.groupby("grp")[c].transform(lambda v: (v - v.mean()) / v.std())
        e["w_" + c] = e["z_" + c] - e.groupby(["grp", "uid"])["z_" + c].transform("mean")
        e["r_" + c] = e.groupby(["grp", "uid"])[c].rank()
        e["r_" + c] -= e.groupby(["grp", "uid"])["r_" + c].transform("mean")
    r3 = e[e["n_g"] >= MIN_TILES_RHO]
    rho = float(np.corrcoef(r3["r_ver"], r3["r_stat"])[0, 1]) if len(r3) > 2 else float("nan")
    slope = lambda f: float((f["w_ver"] * f["w_stat"]).sum() / (f["w_ver"] ** 2).sum())
    ci = mech.cluster_index(e["uid"].to_numpy())
    draws = mech.mg.boot_idx(len(ci["ids"]), B, [SEED, 1])
    bs = [slope(e.iloc[mech.boot_sample(ci, draws[b])[0]]) for b in range(B)]
    m = e.groupby(["grp", "uid"])[["z_ver", "z_stat"]].mean()
    from scipy.stats import spearmanr
    return {"n_tiles_slope": int(len(e)), "n_recordings_slope": int(e["uid"].nunique()),
            "n_tiles_rho": int(len(r3)), "n_recordings_rho": int(r3["uid"].nunique()),
            "n_clusters": int(len(ci["ids"])), "slope_within": slope(e),
            "slope_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "rho_within": rho,
            "rho_between": float(spearmanr(m["z_ver"], m["z_stat"]).correlation), "n_between": int(len(m)),
            "B": int(B)}


def step0_verdict(w):
    nogo = not (w["slope_within"] > 0 and w["rho_within"] >= RHO_MIN)
    return {"verdict": "NOGO" if nogo else "PASS",
            "rule": "NOGO if the pooled within-recording slope <= 0 or rho < 0.10"}



kt = importlib.import_module("132_ekg_check")
EKG_FULL = os.path.join(ROOT, "output", "bdsp", "ekg_full")
SIGN_EXTRA = ("stat", "rand_call", "rand_variant", "rand_extent_t0", "rand_extent_t1", "rand_stat")
SIGN_PIPES = ("columbia", "cornell", "cruse11", "cruse12p", "dubois", "edlow17", "mbmi")
SIGN_MARGIN_S = 5.0
MIN_SPAN_S = 10.0
EMG_BAND = (45.0, 58.0)


def x19_length(path):
    """Samples of the cached x19, read from the npz member's header only."""
    import zipfile
    with zipfile.ZipFile(path) as z, z.open("x19.npy") as fh:
        version = np.lib.format.read_magic(fh)
        shape = np.lib.format._read_array_header(fh, version)[0]
    return int(shape[1])


def load_full_ekg(path, row, n_x19):
    """A full-span EKG (136_ekg_span) checked against its manifest row and the cached x19 length."""
    assert_discovery({"uid": row["uid"], "path": path})
    if row["status"] != "ok":
        raise RuntimeError(f"{row['uid']}: EKG status {row['status']}")
    with np.load(path, allow_pickle=False) as z:
        e = z["ekg"]
        uid, s0 = str(np.asarray(z["uid"]).item()), int(np.asarray(z["start_s"]).item())
    if hashlib.sha256(np.asarray(e).tobytes()).hexdigest() != row["ekg_sha256"]:
        raise RuntimeError(f"{path}: the sha256 of its ekg array differs from the manifest's")
    if uid != row["uid"] or s0 != 0 or len(e) != int(row["n_samples"]) or len(e) != int(n_x19):
        raise RuntimeError(f"{path}: uid/start/length do not match the manifest and the cached x19")
    return np.asarray(e, float)


def span_heart_rate(ekg, fs, t0, t1):
    """Heart rate (4 Hz, recording seconds) over [t0, t1) by the check's detector, with its signal-quality guard
    (132: RR rule, beat template, median heart rate) applied to that span."""
    a, b = max(0, int(round(t0 * fs))), min(len(ekg), int(round(t1 * fs)))
    if b - a < MIN_SPAN_S * fs:
        return {"t": np.array([]), "hr": np.array([]), "reason": "span_too_short", "template_corr": float("nan"),
                "rr_ok_share": float("nan")}
    seg = np.asarray(ekg[a:b], float)
    p, xf = kt._detect(seg, fs)
    t, hr, share = kt.hr_series(p, fs)
    tc = kt.template_corr(xf, p, fs)[0]
    mhr = float(60.0 / np.median(np.diff(p) / fs)) if len(p) > 1 else float("nan")
    why = "rr_implausible" if len(hr) < 2 else kt.quality_reason(share, tc, mhr)
    return {"t": t + a / fs, "hr": hr, "reason": why, "template_corr": tc, "rr_ok_share": share}


def run_emg(x, fs, t0, t1):
    """Run-level EMG: median over channels of the log mean 45-58 Hz power (Welch, 2-s Hann segments) of x19 over
    [t0, t1) clipped to the recording; None under 2 s. The x19 is 128 Hz with a 60-Hz notch, so 45-70 Hz is not
    available."""
    from scipy.signal import welch
    a, b = max(0, int(round(t0 * fs))), min(x.shape[1], int(round(t1 * fs)))
    if b - a < 2 * fs:
        return None
    f, p = welch(np.asarray(x[:, a:b], float), fs=fs, nperseg=int(2 * fs), axis=-1)
    k = (f >= EMG_BAND[0]) & (f <= EMG_BAND[1])
    return float(np.median(np.log(p[:, k].mean(1) + mech.LOG_FLOOR)))


def stable_state(d):
    """Runs within one stable Morgoth state: one label holds the run (not mixed or unlabelled) and the two halves
    share it (not transitioning)."""
    st = d["stratum"].astype(object)
    one = st.notna() & ~st.isin(["mixed", "unlabelled"])
    return (one & ~d["transitioning"].fillna(True).astype(bool)).to_numpy()


def trial_contrast(t, hr, onsets, labels, cons):
    """|Welch t| of trial-mean heart rate between the design's label groups (116.CONTRASTS; trial = onset to the next
    onset, the last one median-ISI long); the largest over the design's contrasts; None without two trials a side."""
    on = np.asarray(onsets, float)
    lab = np.asarray(labels)
    if len(on) < 4:
        return {"abs_t": None, "n_trials": int(len(on))}
    end = np.r_[on[1:], on[-1] + np.median(np.diff(on))]
    m = np.array([np.mean(hr[(t >= a) & (t < b)]) if np.any((t >= a) & (t < b)) else np.nan for a, b in zip(on, end)])
    best = None
    for pos, neg in cons:
        a, b = m[np.isin(lab, pos) & np.isfinite(m)], m[np.isin(lab, neg) & np.isfinite(m)]
        if len(a) < 2 or len(b) < 2:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if se > 0:
            v = abs(a.mean() - b.mean()) / se
            best = v if best is None else max(best, v)
    return {"abs_t": best, "n_trials": int(len(on))}


def placebo_shift(t0, t1, dur):
    """Shift (s) of the placebo span: one run length plus 1 s later, else earlier, non-overlapping; None if neither
    fits in the recording."""
    d = (t1 - t0) + 1.0
    if t1 + d <= dur:
        return d
    if t0 - d >= 0:
        return -d
    return None


def _contrast_at(ekg, fs, ov, mod, cfg, t0, t1, shift):
    h = span_heart_rate(ekg, fs, t0 + shift - SIGN_MARGIN_S, t1 + shift + SIGN_MARGIN_S)
    if h["reason"]:
        return None, h["reason"]
    c = trial_contrast(h["t"], h["hr"], np.asarray(ov.onsets_s, float) + shift, ov.labels, mech.contrasts(mod, cfg))
    return c["abs_t"], (None if c["abs_t"] is not None else "no_contrast")


def recording_sign(rows, ekg, fs, dur, x19=None):
    """Aligned and placebo label contrasts of heart rate for every row of one recording (published overlay and, where
    the row has one, its randomised overlay), with the run's extent and run-level EMG (x19 given)."""
    out, cache = [], {}
    for pipe, g in rows.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, "bdsp")
        for r in g.to_dict("records"):
            rec = {k: r.get(k) for k in ("key", "uid", "patient", "pipe", "variant", "call", "stat", "rand_call",
                                         "rand_variant", "rand_stat", "stratum", "transitioning")}
            for kind, tag in (("published", ""), ("randomized", "rand_")):
                if kind == "randomized" and not isinstance(r.get("rand_variant"), str):
                    continue
                try:
                    ov = mech.overlay_for_row(mod, specs, r, kind)
                except (RuntimeError, ValueError, KeyError) as e:
                    rec[tag + "reason"] = f"overlay: {str(e)[:80]}"
                    continue
                cfg = mech.module_cfg(mod, specs[r["variant"] if kind == "published" else r["rand_variant"]])
                t0, t1 = (float(v) for v in mod.extent(ov))
                k = (pipe, np.asarray(ov.onsets_s, float).tobytes(), np.asarray(ov.labels).tobytes())
                if k not in cache:
                    al, why = _contrast_at(ekg, fs, ov, mod, cfg, t0, t1, 0.0)
                    d = placebo_shift(t0, t1, dur)
                    pl, why_p = ((None, "no_placebo_span") if d is None
                                 else _contrast_at(ekg, fs, ov, mod, cfg, t0, t1, d))
                    cache[k] = (al, why, pl, why_p)
                al, why, pl, why_p = cache[k]
                rec.update({tag + "abs_t_aligned": al, tag + "abs_t_placebo": pl, tag + "reason": why,
                            tag + "placebo_reason": why_p, tag + "t0": t0, tag + "t1": t1,
                            tag + "emg": None if x19 is None else run_emg(x19, fs, t0, t1)})
            out.append(rec)
    return out


def sign_model(d, n_variants, B=B_BOOT, name="sign", cols=("abs_t_aligned", "abs_t_placebo"), call="call",
               fe=None):
    """M1-style weighted FE logit (116._prep weights, pipeline-variant FE, patient bootstrap) of the call on z of the
    aligned and placebo heart-rate contrasts."""
    e = d.dropna(subset=list(cols) + [call]).copy()
    if fe:
        e["variant"] = e[fe]
    e = mech._prep(e, n_variants)
    e["call"] = e[call].astype(bool)
    covs = {f"z_{c}": mech.zscore(e[c].astype(float), e["w"])[0] for c in cols}
    return mech.fe_model(e, covs, B, name, e["w"])


def a9_clean(d):
    """Rows whose rebuilt extent overlaps no activation-procedure interval (116's A9 rule on 121's tables); a row
    without an extent is not clean."""
    ok = d["t0"].notna() & d["t1"].notna() if "t0" in d else pd.Series(False, index=d.index)
    clean = np.zeros(len(d), bool)
    if ok.any():
        e = d[ok].assign(extent_t0=d.loc[ok, "t0"].astype(float), extent_t1=d.loc[ok, "t1"].astype(float))
        clean[np.flatnonzero(ok.to_numpy())] = ~mech._a9(e, "bdsp", "analysis")["a9_flag"].to_numpy(bool)
    return clean



def load_sign_rows():
    """The Phase-1 command-pipeline rows (ncmd_bdsp, j = 0, status ok, native, no responder, sequence windows; 113's
    Columbia rows swapped in, touched recordings dropped) by 128's own loader, with the randomised-method fields kept
    (128.KEEP_FIELDS extended for the call only)."""
    keep = ns.KEEP_FIELDS
    ns.KEEP_FIELDS = keep + SIGN_EXTRA
    try:
        d = ns.first_overlay_rows(ns.analysis_rows(ns.load_bdsp_rows("analysis", outcomes=True, nulls=("ncmd_bdsp",))),
                                  nulls=("ncmd_bdsp",))
    finally:
        ns.KEEP_FIELDS = keep
    return d[d["pipe"].isin(SIGN_PIPES)]


def _sign_path(uid):
    return os.path.join(OUT, "sign", f"{uid}.json")


def run_sign(shard, nshards):
    pool = phase1_pool()
    rows = load_sign_rows()
    man = pd.read_csv(os.path.join(EKG_FULL, "manifest.csv"), dtype={"uid": str, "status": str, "channel": str})
    man = man.set_index("uid")
    uids = sorted(set(rows["uid"]) & set(pool))
    by = dict(tuple(rows.groupby("uid")))
    for i, uid in enumerate(uids):
        if i % nshards != shard or os.path.exists(_sign_path(uid)):
            continue
        assert_discovery(pool[uid])
        r = man.loc[uid]
        if r["status"] != "ok":
            res = [{"uid": uid, "key": k, "reason": "ekg_" + r["status"]} for k in by[uid]["key"]]
        else:
            n = x19_length(pool[uid]["path"])
            ekg = load_full_ekg(os.path.join(EKG_FULL, f"{uid}.npz"), dict(r.to_dict(), uid=uid), n)
            with np.load(pool[uid]["path"], allow_pickle=False) as z:
                x19 = np.asarray(z["x19"], np.float32)
            if x19.shape[1] != n:
                raise RuntimeError(f"{uid}: x19 length differs from its header")
            res = recording_sign(by[uid], ekg, float(r["fs"]), n / float(r["fs"]), x19)
            del x19
        os.makedirs(os.path.dirname(_sign_path(uid)), exist_ok=True)
        with open(_sign_path(uid) + ".tmp", "w") as fh:
            json.dump({"sha256_137": own_sha256(), "uid": uid, "rows": res}, fh, default=float)
        os.replace(_sign_path(uid) + ".tmp", _sign_path(uid))
        print(f"{i + 1}/{len(uids)} {uid}: {len(res)} rows", flush=True)


def sign_summary(B=B_BOOT):
    pool = phase1_pool()
    rows = load_sign_rows()
    uids = sorted(set(rows["uid"]) & set(pool))
    missing = [u for u in uids if not os.path.exists(_sign_path(u))]
    if missing:
        raise RuntimeError(f"{len(missing)} recording(s) without sign-test results (e.g. {missing[:3]})")
    res = []
    for u in uids:
        o = json.load(open(_sign_path(u)))
        if o["sha256_137"] != own_sha256():
            raise RuntimeError(f"{u}: sign-test result of another version of 137")
        res += o["rows"]
    d = pd.DataFrame(res)
    typ = ns.pinned_typing()
    imm = np.array([typ.get((p, v)) == "immune" for p, v in zip(d["pipe"], d["variant"])])
    pipes = sorted(set(d["pipe"]))
    nv_exp = ns.non_immune_variant_counts(pipes, typ)
    nv_imm = {p: sum(typ.get((p, v)) == "immune" for v in rn.variant_specs(rn.load_pipe(p), "bdsp")) for p in pipes}
    na = lambda c: d[c].dropna().astype(str).str.split(":").str[0].value_counts().to_dict() if c in d else {}
    out = {"label": "sign test (discovery data, exploration): label-aligned vs placebo heart-rate contrast",
           "sha256_137": own_sha256(), "n_rows": int(len(d)), "n_recordings": int(d["uid"].nunique()),
           "na_aligned": na("reason"), "na_placebo": na("placebo_reason")}
    al, both = ("abs_t_aligned",), ("abs_t_aligned", "abs_t_placebo")
    ral, rboth = ("rand_abs_t_aligned",), ("rand_abs_t_aligned", "rand_abs_t_placebo")
    ve = {p: n for p, n in nv_exp.items() if n}
    vi = {p: n for p, n in nv_imm.items() if n}
    r = d[~imm].dropna(subset=["rand_call"]).copy()
    r["rand_grp"] = r["rand_variant"]
    nv_r = r.groupby("pipe")["rand_grp"].nunique().to_dict()
    out["exposed_aligned"] = sign_model(d[~imm], ve, B, "sign_exposed", al)
    out["exposed_aligned_vs_placebo"] = sign_model(d[~imm], ve, B, "sign_exposed_pl", both)
    out["immune_aligned"] = sign_model(d[imm], vi, B, "sign_immune", al)
    out["immune_aligned_vs_placebo"] = sign_model(d[imm], vi, B, "sign_immune_pl", both)
    out["randomised_aligned"] = sign_model(r, nv_r, B, "sign_randomised", ral, call="rand_call", fe="rand_grp")
    out["randomised_aligned_vs_placebo"] = sign_model(r, nv_r, B, "sign_randomised_pl", rboth, call="rand_call",
                                                      fe="rand_grp")
    out["per_pipeline_exposed_aligned"] = {p: sign_model(g, {p: nv_exp[p]}, B, f"sign_{p}", al)
                                           for p, g in d[~imm].groupby("pipe") if nv_exp.get(p)}
    st = stable_state(d)
    out["exposed_stable_strata"] = d[~imm & st].dropna(subset=list(al))["stratum"].value_counts().to_dict()
    out["exposed_stable_aligned"] = sign_model(d[~imm & st], ve, B, "sign_exposed_stable", al)
    out["exposed_stable_aligned_vs_placebo"] = sign_model(d[~imm & st], ve, B, "sign_exposed_stable_pl", both)
    out["exposed_aligned_emg"] = sign_model(d[~imm], ve, B, "sign_exposed_emg", al + ("emg",))
    out["exposed_aligned_vs_placebo_emg"] = sign_model(d[~imm], ve, B, "sign_exposed_pl_emg", both + ("emg",))
    out["exposed_stable_aligned_emg"] = sign_model(d[~imm & st], ve, B, "sign_exposed_stable_emg", al + ("emg",))
    clean = a9_clean(d)
    out["exposed_a9_clean_aligned"] = sign_model(d[~imm & clean], ve, B, "sign_exposed_a9", al)
    out["exposed_a9_clean_aligned_emg"] = sign_model(d[~imm & clean], ve, B, "sign_exposed_a9_emg", al + ("emg",))
    out["n_rows_a9_flagged"] = int((~clean).sum())
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "sign_rows.csv"), index=False)
    with open(os.path.join(OUT, "sign_summary.json"), "w") as fh:
        json.dump(mech.mg._jsonable(out), fh, indent=1, default=float)
    print("sign summary written", os.path.join(OUT, "sign_summary.json"))


def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _result_path(uid):
    return os.path.join(OUT, "step0", f"{uid}.json")


def run_step0(shard, nshards):
    pool = phase1_pool()
    uids = step0_sample(list(pool))
    for i, uid in enumerate(uids):
        if i % nshards != shard or os.path.exists(_result_path(uid)):
            continue
        assert_discovery(pool[uid])
        rec = en._load(pool[uid])
        rows = recording_step0(rec)
        del rec
        os.makedirs(os.path.dirname(_result_path(uid)), exist_ok=True)
        with open(_result_path(uid) + ".tmp", "w") as fh:
            json.dump({"sha256_137": own_sha256(), "uid": uid, "rows": rows}, fh, default=float)
        os.replace(_result_path(uid) + ".tmp", _result_path(uid))
        print(f"{i + 1}/{len(uids)} {uid}: {len(rows)} tiles", flush=True)


def step0_summary():
    pool = phase1_pool()
    uids = step0_sample(list(pool))
    missing = [u for u in uids if not os.path.exists(_result_path(u))]
    if missing:
        raise RuntimeError(f"{len(missing)} STEP 0 recording(s) not done (e.g. {missing[:3]})")
    rows = []
    for u in uids:
        o = json.load(open(_result_path(u)))
        if o["sha256_137"] != own_sha256():
            raise RuntimeError(f"{u}: STEP 0 result of another version of 137")
        rows += o["rows"]
    d = pd.DataFrame(rows)
    out = {"label": "STEP 0 check (discovery data, exploration)", "sha256_137": own_sha256(),
           "n_tiles_total": int(len(d)), "na": {"stat": int(d["stat"].isna().sum()), "ver": int(d["ver"].isna().sum())},
           "pooled": within_association(d)}
    out["verdict"] = step0_verdict(out["pooled"])
    out["per_variant"] = {f"{p}|{v}": within_association(g) for (p, v), g in d.groupby(["pipe", "variant"])}
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "step0_tiles.csv"), index=False)
    with open(os.path.join(OUT, "step0_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("n_tiles_total", "na", "verdict")}, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--step0", action="store_true")
    m.add_argument("--step0-summary", action="store_true")
    m.add_argument("--sign", action="store_true")
    m.add_argument("--sign-summary", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.step0:
        run_step0(a.shard, a.nshards)
    elif a.step0_summary:
        step0_summary()
    elif a.sign:
        run_sign(a.shard, a.nshards)
    else:
        sign_summary()


if __name__ == "__main__":
    main()
