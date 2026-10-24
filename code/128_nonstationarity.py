"""Non-stationarity index (VER) and its false-call model.
Non-stationarity index (VER) and its false-call model:, part C, C2
(added later). Built test-first on synthetic recordings and synthetic shard rows (code/test_nonstationarity.py).

VER per window (window_ver):
- The window is a null row's overlay extent on its recording. Its log feature is the module's log band power per epoch
  of the 116 grid (116.grid_windows) on the module's spatially transformed channels (116.spatial): the family band
  (motor 8-30 Hz, ERP 1-20 Hz; zero-phase Butterworth over the extent) as 116.feature_series computes it, so for the
  motor modules it is 116's X feature series exactly (tested). ERP modules: log power of the ERP band in each epoch's
  window (no baseline), since their 116 feature (a mean amplitude) is not a log feature.
- Surrogates: 117's multivariate Fourier phase randomisation (117.phase_randomise: one random phase per frequency
  shared by every channel, so every auto- and cross-spectrum is kept and the stationarity is destroyed) of the RAW
  window (every recorded channel, before the spatial transform and the filter), N_SURR = 10 per window, generator
  default_rng([20261014, crc32(uid), first sample, end sample]): seeded by the recording and its window.
- VER_c = variance over epochs (ddof 1) of channel c's log feature / its mean over the surrogates; the window's log
  VER = the mean over channels of log VER_c (channels whose log-feature variance is <= 1e-18, round-off only, left
  out; none left: NA "no_variance").
- Windows shared by several variants (same module, extent and grid) are computed once.

Rows (load_bdsp_rows, analysis_rows, first_overlay_rows): the 116 M1 rows, i.e. BDSP null rows (ncmd_bdsp,
nstim0_bdsp; resp none, native montage) of the first overlay (j = 0) of every frozen variant, status ok, sequence
windows (115._null_seq); analysis split: touched recordings dropped and Columbia's first overlays replaced by 113's
rows (as 115.merge and 116.load_rows). Unlike 116, only the BDSP null directories are read, only j = 0 rows are kept,
and only the structural fields (KEEP_FIELDS) plus, where a model needs it, the call: the VER stage reads no outcome.
Non-immune rows only (non_immune): the variants the 116 design table (output/tables/mechanism_design_table.csv, the
fixed 116 analysis output, checked by sha256) does not type "immune" by design (|X_ref| <= 0.1); no-contrast
variants (sokoliuk21, gibson16 SSEP) are kept.

Model (ver_model): as 116's M1 (116.fe_model): weighted logistic call ~ z(log VER) + pipeline-variant fixed effects,
penalised (C = 100) with the plain-MLE failure count, row weight 1 / (variants of the pipeline) / (the patient's rows
of that variant), z by the weighted mean and SD, patient-cluster bootstrap B = 2000 (relabelled draws), seed
[20261014, crc32(model name)]; one-sided p for OR per SD of log VER > 1 and the 95% CI. Variants per pipeline (the
weight denominator): its frozen BDSP variants not typed immune (Phase 1; with the immune variants left out every
pipeline still counts equally, as in M1; only du Bois loses variants) or those present in the rows (fresh draw).

Modes (guard_real):
- --phase1: Phase-1 analysis rows, labelled hypothesis-generating. Stage ver (no outcome read), then stats.
- --fresh --root DIR --manifest CSV: the fresh BDSP draw, rows of 127_fresh_rows.py (nulls ncmd_bdsp_fresh and
  nstim0_bdsp_fresh, split c2_fresh, under output/intermediate/_fresh_shards or --rows), windows in the c2_fresh
  layout (--root: the directory holding sig/ and morgoth/; --manifest: the fixed c2_fresh_400.csv;
  fresh_items). Model rows: the validity-primary and default variants only (C2; c2_variants; a shortened twin of a
  role variant, twin_of, counts as it), weights over the variants present. Bar (fresh_bar): OR >= 1.20 per SD with
  the lower 95% bound > 1.
- --atlas-map: the C1 calibration map, no call read. Stage ver: log VER of the default and validity-primary windows
  (non-immune) of every envelope-cell unit of the C1 atlas, on the unit's own synthetic recording (129_atlas.noise,
  seeds, need and overlay start, as 129_atlas.run_unit; rows keyed pipe|cell|r|variant as its rows, with the sha256
  of 129_atlas.py) -> output/intermediate/_nonstationarity/atlas_ver. Stage stats: per (pipe, variant) of 116's
  variant choices (fixed output/tables/mechanism.json), the log VER distribution of each envelope cell and of
  the Phase-1 windows (needs the --phase1 ver stage), overall and per Morgoth stratum.
- Sharding: --stage ver --shard s --nshards n takes the recordings (or atlas units) i % n == s; --stage stats refuses
  while any row's recording lacks its result.

python 128_nonstationarity.py (--phase1 | --fresh --root DIR --manifest CSV [--rows DIR] | --atlas-map)
                              [--stage ver|stats|all] [--shard i --nshards n] [--atlas-n N]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
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

mech = importlib.import_module("116_mechanism")
m117 = importlib.import_module("117_m1_check")
rn, mg, en, mc = mech.rn, mech.mg, mech.en, mech.mc
phase_randomise = m117.phase_randomise

SEED = 20261014
N_SURR = 10
B_BOOT = 2000
BAR_OR = 1.20
MIN_VAR = 1e-18
BDSP_NULLS = mech.BDSP_NULLS
FRESH_NULLS = ("ncmd_bdsp_fresh", "nstim0_bdsp_fresh")
FRESH_SPLIT = "c2_fresh"
FRESH_SHARDS = os.path.join(ROOT, "output", "intermediate", "_fresh_shards")
FRESH_LIST_SHA256 = "216c6a2977264d2c3f5e354df6185daa83782a69a7e176de8fbb1ede92c45d39"
KEEP_FIELDS = ("key", "null", "pipe", "variant", "uid", "patient", "j", "status", "resp", "montage", "seq", "overlap",
               "split", "dev", "source", "start_s", "extent_t0", "extent_t1", "n_onsets", "stratum", "transitioning",
               "primary_variant", "family", "grade", "tile", "twin_of")
OUTCOME_FIELDS = ("call", "p", "stat", "size_stat", "prob", "rand_call", "rand_p", "rand_stat", "rand_size_stat")
TYPING_TABLE = os.path.join(ROOT, "output", "tables", "mechanism_design_table.csv")
TYPING_TABLE_SHA256 = "4827cc42a951d9c44b20fd3ae0b1d82123d62af55c6eed0f37313bd357ba190c"
MECH_SUMMARY = os.path.join(ROOT, "output", "tables", "mechanism.json")
MECH_SUMMARY_SHA256 = "e5f3030f7cf349e2638325498aeb6575319bfd34766f34dc8e2baf8ad0678267"
ATLAS_MODULE = "129_atlas"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)



def epoch_log_power(rec, mod, cfg, ov, seg, grid=None):
    """(channels, epochs) log mean power of the family band per grid epoch on the module's spatially transformed
    channels; seg: the raw extent of every recorded channel (mc.span's samples)."""
    fs = float(rec.fs)
    g = grid or mech.grid_windows(mod, cfg, ov, fs)
    i0 = int(math.floor(mod.extent(ov)[0] * fs))
    lo, hi = mech.ERP_BAND if mod.FAMILY == "erp" else mech.MOTOR_BAND
    xf = mc.bandpass(mech.spatial(mod, rec, seg)[0], fs, lo, hi)
    cs = np.concatenate([np.zeros((xf.shape[0], 1)), np.cumsum(xf ** 2, axis=1)], axis=1)
    a = np.round(g["win"][:, 0] * fs).astype(int) - i0
    b = np.round(g["win"][:, 1] * fs).astype(int) - i0
    if a.min() < 0 or b.max() > xf.shape[1] or np.any(b <= a):
        raise ValueError(f"{mod.NAME}: an epoch window falls outside the module's extent")
    return np.log((cs[:, b] - cs[:, a]) / (b - a) + mech.LOG_FLOOR)


def surrogates(seg, uid, i0, i1, n=N_SURR):
    """n multivariate phase-randomised surrogates of a raw (channels, samples) window (117.phase_randomise), seeded by
    the recording and the window's sample span."""
    rng = np.random.default_rng([SEED, zlib.crc32(str(uid).encode()), int(i0), int(i1)])
    return [phase_randomise(np.asarray(seg, float).T, rng).T for _ in range(n)]


def window_ver(rec, mod, cfg, ov, n_surr=N_SURR):
    """log VER of one window (module docstring) and its counts."""
    fs = float(rec.fs)
    seg, i0 = mc.span(np.asarray(rec.x), fs, *mod.extent(ov))
    seg = np.asarray(seg, float)
    i1 = i0 + seg.shape[1]
    g = mech.grid_windows(mod, cfg, ov, fs)
    F = epoch_log_power(rec, mod, cfg, ov, seg, g)
    v = F.var(1, ddof=1)
    vs = np.mean([epoch_log_power(rec, mod, cfg, ov, s, g).var(1, ddof=1) for s in surrogates(seg, rec.uid, i0, i1,
                                                                                               n_surr)], 0)
    ok = np.isfinite(v) & np.isfinite(vs) & (v > MIN_VAR) & (vs > MIN_VAR)
    out = {"n_ch": int(len(v)), "n_ch_used": int(ok.sum()), "n_epochs": int(F.shape[1]), "n_surr": int(n_surr),
           "span_i0": int(i0), "span_i1": int(i1)}
    if not ok.any():
        return dict(out, log_ver=None, log_ver_ch_sd=None, reason="no_variance")
    lv = np.log(v[ok] / vs[ok])
    return dict(out, log_ver=float(lv.mean()), log_ver_ch_sd=float(lv.std()), reason=None)


def recording_ver(rec, rows, n_surr=N_SURR):
    """VER of every row of one recording (rows: key, uid, patient, pipe, variant, j, start_s, extent fields); a window
    shared by several variants (same module, extent and grid) is computed once. Never carries an outcome."""
    out, cache = [], {}
    for pipe, d in rows.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, rec.source)
        for _, r in d.iterrows():
            cfg = mech.module_cfg(mod, specs[r["variant"]])
            base = {"key": r["key"], "uid": r["uid"], "patient": r["patient"], "pipe": pipe, "variant": r["variant"],
                    "j": int(r["j"])}
            try:
                ov = mech.overlay_for_row(mod, specs, r)
                g = mech.grid_windows(mod, cfg, ov, rec.fs)
                k = (pipe, tuple(round(float(t), 6) for t in mod.extent(ov)), g["win"].tobytes())
                if k not in cache:
                    with mech.source_channels(mod, rec.source):
                        cache[k] = window_ver(rec, mod, cfg, ov, n_surr)
                res = cache[k]
            except (ValueError, IndexError) as e:
                res = {"log_ver": None, "reason": f"feature_error: {str(e)[:120]}"}
            out.append({**base, **res})
    return out



def load_bdsp_rows(split, root=None, outcomes=True, nulls=BDSP_NULLS):
    """The first-overlay (j = 0) rows of the BDSP null directories (resp none, native) under root (default 114's shard
    root of the split): KEEP_FIELDS, plus the call if outcomes. Duplicate keys within a directory counted once (as
    115.load_rows); rows of another split, or development recordings among non-dev rows, are refused."""
    root = root or rn.SH[split]
    keep = KEEP_FIELDS + (("call",) if outcomes else ())
    frames = []
    for null in nulls:
        for d in sorted(glob.glob(os.path.join(root, f"{null}__*__native__none"))):
            rows = []
            for fn in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
                with open(fn) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if r.get("j") == 0:
                            rows.append({k: r.get(k) for k in keep})
            if rows:
                frames.append(pd.DataFrame(rows).drop_duplicates("key", keep="first").assign(dir=os.path.basename(d)))
    if not frames:
        return pd.DataFrame(columns=list(keep))
    df = pd.concat(frames, ignore_index=True, sort=False)
    if (df["split"] != split).any():
        raise RuntimeError(f"rows of split(s) {sorted(set(df['split']) - {split})} under the {split} shards")
    dev = df["dev"].fillna(False).astype(bool)
    if split != "dev" and dev.any():
        raise RuntimeError(f"{int(dev.sum())} development-recording rows among {split} rows: refused")
    return df


def analysis_rows(df, outcomes=True):
    """Analysis split: touched recordings dropped and Columbia's first overlays replaced by 113's rows (115.merge,
    116.load_rows); outcome fields of 113's rows removed unless outcomes."""
    df, _ = mg.drop_touched(df)
    c = rn.columbia113()
    col = (df["null"] == "ncmd_bdsp") & (df["pipe"] == "columbia") & (df["resp"] == "none") & (df["j"] == 0)
    mg.check_113(c, df[~col], mg.dev_uids(), mg.touched_uids())
    r113 = mg.rows_from_113(c)
    r113 = r113[[k for k in r113.columns if k in KEEP_FIELDS + (("call",) if outcomes else ())]]
    return pd.concat([df[~col], r113], ignore_index=True, sort=False)


def first_overlay_rows(df, nulls=BDSP_NULLS):
    """116's M1 rows: first overlay of every variant, BDSP nulls, no responder, native, status ok, sequence windows."""
    d = mg._null_seq(df)
    return d[d["null"].isin(nulls) & (d["j"] == 0)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def typing_from_table(tab):
    """{(pipe, variant): "immune" | "exposed" | None} of the published designs (116.immunity_type of X_ref)."""
    p = tab[tab["kind"] == "published"].drop_duplicates(["pipe", "variant"])
    return {(a, b): mech.immunity_type(x) for a, b, x in zip(p["pipe"], p["variant"], p["x_ref_1f"].astype(float))}


def pinned_typing():
    if sha256_file(TYPING_TABLE) != TYPING_TABLE_SHA256:
        raise RuntimeError(f"{TYPING_TABLE} is not the pinned 116 design table (sha256)")
    return typing_from_table(pd.read_csv(TYPING_TABLE))


def non_immune(d, typing):
    """Rows of the variants not typed immune by design (no-contrast variants kept). A variant absent from the typing is
    refused."""
    keys = list(zip(d["pipe"], d["variant"]))
    missing = sorted(set(k for k in keys if k not in typing))
    if missing:
        raise RuntimeError(f"variants without a design typing: {missing[:5]}")
    return d[[typing[k] != "immune" for k in keys]]


def non_immune_variant_counts(pipes, typing):
    """{pipe: frozen BDSP variants not typed immune}: the M1 weight denominators once immune variants are left out, so
    every pipeline still counts equally (116.row_weights)."""
    return {p: sum(typing[(p, v)] != "immune" for v in rn.variant_specs(rn.load_pipe(p), mech.SOURCE)) for p in pipes}


def c2_variants(d, roles):
    """Rows of the validity-primary and default variants (C2's fresh-draw variants); roles: variant_roles(). A row of
    the engine's shortened BDSP twin of a role variant (twin_of = that variant) counts as the role variant."""
    tw = d["twin_of"] if "twin_of" in d else pd.Series(None, index=d.index)
    return d[[(p, v) in roles or (pd.notna(t) and (p, t) in roles)
              for p, v, t in zip(d["pipe"], d["variant"], tw)]]


def fresh_items(root, manifest):
    """{uid: engine item} of the fresh-draw windows: <root>/sig/<uid>.npz (the fixed 30-min window only, times
    from its start) and <root>/morgoth/<uid>.parquet, listed by the fixed c2_fresh_400.csv (rank, patient_id,
    uid, ...; sha256 FRESH_LIST_SHA256, refused otherwise); no signal loaded."""
    got = sha256_file(manifest)
    if got != FRESH_LIST_SHA256:
        raise RuntimeError(f"{manifest}: sha256 {got} is not the pinned list's {FRESH_LIST_SHA256}")
    df = pd.read_csv(manifest, dtype={"patient_id": str, "uid": str}).sort_values("rank", kind="stable")
    for col in ("uid", "patient_id"):
        if df[col].duplicated().any():
            raise RuntimeError(f"{manifest}: a {col} is listed twice")
    out = {}
    for r in df.itertuples(index=False):
        it = dict(source="bdsp", path=os.path.join(str(root), "sig", f"{r.uid}.npz"),
                  morgoth=os.path.join(str(root), "morgoth", f"{r.uid}.parquet"), uid=r.uid, patient=r.patient_id,
                  meta={"dev": False, "fresh": True, "fresh_rank": int(r.rank)})
        for k in ("path", "morgoth"):
            if not os.path.exists(it[k]):
                raise FileNotFoundError(it[k])
        out[r.uid] = it
    return out



def prep(d, n_variants):
    return mech._prep(d, n_variants)


def variants_present(d):
    return {p: int(n) for p, n in d.groupby("pipe")["variant"].nunique().items()}


def ver_model(d, n_variants, B=B_BOOT, name="c2"):
    """116's M1 model with z(log VER) as the predictor (module docstring)."""
    e = d[d["log_ver"].notna()]
    if e.empty or e["patient"].nunique() < 2:
        return {"n_rows": int(len(e)), "note": "fewer than 2 patients with log VER: not estimated"}
    e = prep(e, n_variants)
    z, m, sd = mech.zscore(e["log_ver"].astype(float), e["w"])
    out = mech.fe_model(e, {"z_log_ver": z}, B, name, e["w"])
    out.update(log_ver_weighted_mean=float(m), log_ver_weighted_sd=float(sd), test="OR per SD of log VER > 1, "
               "one-sided: p_greater")
    return out


def fresh_bar(model):
    """C2 confirmation bar: OR >= 1.20 per SD with the lower 95% bound > 1."""
    c = model.get("coef", {}).get("z_log_ver")
    if not c or not c.get("or_ci95"):
        return {"confirmed": False, "note": "not estimated", "bar": f"OR >= {BAR_OR} and lower 95% bound > 1"}
    return {"confirmed": bool(c["or"] >= BAR_OR and c["or_ci95"][0] > 1), "or": c["or"], "lower95": c["or_ci95"][0],
            "bar": f"OR >= {BAR_OR} and lower 95% bound > 1"}



def own_sha256():
    return sha256_file(os.path.abspath(__file__))


def guard_real(spec_path=None):
    """Not applied in this copy: the check reads files of the private working copy."""
    return None


def write_result(d, uid, rows):
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f"{uid}.json")
    with open(fn + ".tmp", "w") as fh:
        json.dump({"sha256_128": own_sha256(), "uid": uid, "rows": rows}, fh)
    os.replace(fn + ".tmp", fn)


def done_uids(d):
    """Recordings with a result file; files of another version of this script are refused (never mixed)."""
    if not os.path.isdir(d):
        return set()
    uids = {f[:-5] for f in os.listdir(d) if f.endswith(".json") and not f.startswith("._")}
    h = own_sha256()
    stale = [u for u in sorted(uids) if json.load(open(os.path.join(d, f"{u}.json")))["sha256_128"] != h]
    if stale:
        raise RuntimeError(f"{len(stale)} result file(s) in {d} come from another version of 128_nonstationarity.py "
                           f"(e.g. {stale[:3]}): move them away before running this version")
    return uids


def read_results(d):
    return pd.DataFrame([r for u in sorted(done_uids(d))
                         for r in json.load(open(os.path.join(d, f"{u}.json")))["rows"]])


def paths(mode):
    base = os.path.join(ROOT, "output", "intermediate", "_nonstationarity" + ("_fresh" if mode == "fresh" else ""))
    return {"ver": os.path.join(base, "ver"), "atlas": os.path.join(base, "atlas_ver"),
            "summary": os.path.join(ROOT, "output", "tables", f"nonstationarity_{mode.replace('-', '_')}.json")}



def run_ver(rows, items, out_dir, n_surr=N_SURR, shard=0, nshards=1):
    """VER of every row, one result file per recording (resumable; no outcome is read or written); a shard takes the
    recordings i % nshards == shard of the sorted uids."""
    done = done_uids(out_dir)
    uids = sorted(set(rows["uid"]))
    missing = sorted(set(uids) - set(items))
    if missing:
        raise RuntimeError(f"{len(missing)} row recording(s) not in the listing: {missing[:5]}")
    by = dict(tuple(rows.groupby("uid")))
    for i, uid in enumerate(uids):
        if uid in done or i % nshards != shard:
            continue
        rec = en._load(items[uid])
        res = recording_ver(rec, by[uid], n_surr)
        del rec
        write_result(out_dir, uid, res)
        print(f"{i + 1}/{len(uids)} {uid}: {len(res)} rows, {sum(r['log_ver'] is None for r in res)} NA", flush=True)


def phase1_rows(outcomes):
    return non_immune(first_overlay_rows(analysis_rows(load_bdsp_rows("analysis", outcomes=outcomes), outcomes)),
                      pinned_typing())


def fresh_rows(root, outcomes):
    d = first_overlay_rows(load_bdsp_rows(FRESH_SPLIT, root=root, outcomes=outcomes, nulls=FRESH_NULLS),
                           nulls=FRESH_NULLS)
    return c2_variants(non_immune(d, pinned_typing()), variant_roles())


def _with_ver(rows, ver):
    v = ver.drop(columns=[c for c in ("uid", "patient", "variant", "j") if c in ver])
    d = rows.merge(v, on=["key", "pipe"], how="left", validate="one_to_one")
    return d


def _na(d):
    r = d["reason"].fillna("missing_result").astype(str).str.split(":").str[0]
    return {"n_rows": int(len(d)), "n_na": int(d["log_ver"].isna().sum()),
            "by_reason": r[d["log_ver"].isna()].value_counts().to_dict()}


def require_all(rows, out_dir):
    """Every row's recording must have its result file (a missing recording is never read as NA)."""
    missing = sorted(set(rows["uid"]) - done_uids(out_dir))
    if missing:
        raise RuntimeError(f"{len(missing)} recording(s) without a VER result in {out_dir} (e.g. {missing[:3]}): "
                           "finish every shard of the ver stage first")


def stats(rows, ver, n_variants, B, name, label):
    d = _with_ver(rows, ver)
    out = {"label": label, "sha256_128": own_sha256(), "B": B, "rows": _na(d),
           "rows_by_pipeline": d.groupby("pipe").size().to_dict(), "n_variants": n_variants}
    out["model"] = ver_model(d, n_variants, B, name)
    return out, d


def _items(split):
    return {it["uid"]: it for it in en._items("bdsp", split)}


def _distribution(v):
    v = pd.Series(v, dtype=float).dropna()
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
            "quantiles": {str(q): float(v.quantile(q)) for q in QUANTILES} if len(v) else None}


def variant_roles():
    """{(pipe, variant): [roles]} of the validity-primary and default variants of the BDSP nulls, from the fixed
    116 summary (its variant choices; no call is read)."""
    if sha256_file(MECH_SUMMARY) != MECH_SUMMARY_SHA256:
        raise RuntimeError(f"{MECH_SUMMARY} is not the pinned 116 summary (sha256)")
    vc = json.load(open(MECH_SUMMARY))["part_A"]["variant_choices"]
    out = {}
    for k, v in vc.items():
        nul, pipe = k.split("|")
        if nul in BDSP_NULLS:
            for role in ("validity_primary", "default"):
                if v.get(role):
                    out.setdefault((pipe, v[role]), []).append(role)
    return out


def atlas_module():
    return importlib.import_module(ATLAS_MODULE)


def atlas_units(A, pipes=None, n=None):
    """The C1 atlas units (pipe, cell, r) of the envelope cells (129_atlas.units, family "env")."""
    return [u for u in A.units(pipes, n=n) if A.CELLS[u[1]]["family"] == "env"]


def atlas_recording(A, pipe, cell, r, R):
    """The synthetic recording of one C1 atlas unit and its overlay start, exactly as 129_atlas.run_unit builds them
    (R: 129_atlas.roles of the pipeline)."""
    mod = rn.load_pipe(pipe)
    uid = f"atlas|{cell}|{r}"
    need = rn.need_seconds(mod, A._arg_names(R), A.SOURCE)
    rng = np.random.default_rng([A.SEED, rn._crc(pipe), rn._crc(cell), r])
    x = A.noise(A.CELLS[cell], A.PAD_S + need + A.PAD_S, rng)
    rec = en.Recording(source="atlas", uid=uid, patient=uid, x=x, fs=A.FS, names=list(A.NAMES),
                       meta={"dev": False, "cell": cell})
    return rec, A.PAD_S + rn.LEAD_S


def atlas_ver(A, units, R, typing, done, write, n_surr=N_SURR):
    """log VER of the published (default and validity-primary) non-immune variants of every atlas unit, one row per
    unit and variant, keyed as 129_atlas's rows (pipe|cell|r|variant). No call is run or read."""
    for pipe, cell, r in units:
        pub = [(v, e) for v, e in R[pipe].items() if e["kind"] == "published" and typing.get((pipe, v)) != "immune"]
        todo = [(v, e) for v, e in pub if f"{pipe}|{cell}|{r}|{v}" not in done]
        if not todo:
            continue
        mod = rn.load_pipe(pipe)
        rec, start = atlas_recording(A, pipe, cell, r, R[pipe])
        c, cache = A.CELLS[cell], {}
        for v, e in todo:
            row = {"key": f"{pipe}|{cell}|{r}|{v}", "pipe": pipe, "cell": cell, "log_sd": c["log_sd"],
                   "tau_s": c["tau_s"], "r": r, "variant": v, "roles": e["roles"], "sha256_128": own_sha256(),
                   "sha256_atlas": sha256_file(A.__file__)}
            try:
                cfg = mech.module_cfg(mod, e["spec"])
                ov = rn._sched(mod, mod.schedule, e["spec"], start, rn._rng(pipe, rec.uid, 0, 1))
                g = mech.grid_windows(mod, cfg, ov, rec.fs)
                k = (tuple(round(float(t), 6) for t in mod.extent(ov)), g["win"].tobytes())
                if k not in cache:
                    cache[k] = window_ver(rec, mod, cfg, ov, n_surr)
                row.update(cache[k])
            except (ValueError, IndexError) as err:
                row.update(log_ver=None, reason=f"feature_error: {str(err)[:120]}")
            write(row)
        print(f"atlas {pipe} {cell} r{r}: {len(todo)} variant(s)", flush=True)


def read_atlas_rows(d):
    """Atlas VER rows of every shard file; rows of another version of this script or of 129_atlas.py are refused."""
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
        with open(fn) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    h128 = own_sha256()
    h_atlas = {r["sha256_atlas"] for r in rows}
    if any(r["sha256_128"] != h128 for r in rows) or len(h_atlas) > 1:
        raise RuntimeError(f"{d}: atlas rows of another version of 128_nonstationarity.py or of 129_atlas.py")
    return pd.DataFrame(rows).drop_duplicates("key", keep="first") if rows else pd.DataFrame()


def run_atlas_ver(shard, nshards, n=None, out_dir=None, typing=None, vp=None, pipes=None):
    """One shard of the atlas VER stage (units i % nshards == shard), appended to and resumable."""
    A = atlas_module()
    out_dir = out_dir or paths("phase1")["atlas"]
    fn = os.path.join(out_dir, f"rows{shard:02d}.jsonl")
    done = set(read_atlas_rows(out_dir).get("key", pd.Series(dtype=str)))
    vp = A.validity_primaries() if vp is None else vp
    typing = pinned_typing() if typing is None else typing
    us = [u for i, u in enumerate(atlas_units(A, pipes, n)) if i % nshards == shard]
    R = {p: A.roles(rn.load_pipe(p), vp.get(p)) for p in sorted({u[0] for u in us})}
    os.makedirs(out_dir, exist_ok=True)
    with open(fn, "a") as fh:
        def write(row):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
        atlas_ver(A, us, R, typing, done, write)


def atlas_map(syn, d, roles):
    """The C1 calibration map: log VER distributions of the synthetic envelope cells per (pipe, variant) and of the
    Phase-1 windows of the same variants (overall and per Morgoth stratum). No call is read."""
    out = {}
    for (pipe, variant), rr in sorted(roles.items()):
        e = d[(d["pipe"] == pipe) & (d["variant"] == variant)]
        s = syn[(syn["pipe"] == pipe) & (syn["variant"] == variant)] if len(syn) else syn
        out[f"{pipe}|{variant}"] = {
            "roles": rr, "real": _distribution(e["log_ver"]),
            "real_by_stratum": {k: _distribution(g["log_ver"])
                                for k, g in e.groupby(e["stratum"].fillna("unlabelled"))},
            "synthetic": {c: dict(_distribution(g["log_ver"]), log_sd=float(g["log_sd"].iloc[0]),
                                  tau_s=float(g["tau_s"].iloc[0])) for c, g in s.groupby("cell")} if len(s) else {}}
    return out


def run_atlas_map():
    p = paths("phase1")
    rows = phase1_rows(outcomes=False)
    require_all(rows, p["ver"])
    d = _with_ver(rows, read_results(p["ver"]))
    syn = read_atlas_rows(p["atlas"])
    roles = {k: v for k, v in variant_roles().items() if k in set(zip(d["pipe"], d["variant"]))}
    return {"label": "C1 envelope calibration map (ADDED LATER; no call read)", "sha256_128": own_sha256(),
            "sha256_atlas": sorted(set(syn["sha256_atlas"])) if len(syn) else None, "rows": _na(d),
            "synthetic_rows": int(len(syn)), "map": atlas_map(syn, d, roles)}


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--phase1", action="store_true")
    m.add_argument("--fresh", action="store_true")
    m.add_argument("--atlas-map", action="store_true")
    ap.add_argument("--stage", default="all", choices=("ver", "stats", "all"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--rows", default=FRESH_SHARDS, help="--fresh: shard root of the fresh-draw rows")
    ap.add_argument("--root", help="--fresh: directory holding sig/ and morgoth/ of the fresh-draw windows")
    ap.add_argument("--manifest", help="--fresh: the pinned c2_fresh_400.csv")
    ap.add_argument("--atlas-n", type=int, default=None, help="--atlas-map: replicates per cell (default: the atlas's)")
    a = ap.parse_args()
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("--shard must lie in [0, --nshards)")
    if a.stage == "all" and a.nshards > 1:
        raise SystemExit("--stage all runs unsharded; run --stage ver per shard, then --stage stats")
    guard_real()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mode = "phase1" if a.phase1 else "fresh" if a.fresh else "atlas-map"
    p = paths("fresh" if mode == "fresh" else "phase1")
    if mode == "atlas-map":
        if a.stage in ("ver", "all"):
            run_atlas_ver(a.shard, a.nshards, a.atlas_n)
        if a.stage == "ver":
            return
        out = run_atlas_map()
    else:
        if mode == "phase1":
            items = _items("analysis")
            rows_s = phase1_rows(outcomes=False)
        else:
            if not (a.root and a.manifest):
                raise SystemExit("--fresh needs --root and --manifest")
            items = fresh_items(a.root, a.manifest)
            rows_s = fresh_rows(a.rows, outcomes=False)
        if a.stage in ("ver", "all"):
            run_ver(rows_s, items, p["ver"], shard=a.shard, nshards=a.nshards)
        if a.stage == "ver":
            return
        require_all(rows_s, p["ver"])
        if mode == "phase1":
            rows = phase1_rows(outcomes=True)
            nv = non_immune_variant_counts(sorted(set(rows["pipe"])), pinned_typing())
            out, _ = stats(rows, read_results(p["ver"]), nv, B_BOOT, "c2_phase1",
                           "Phase-1 rows: HYPOTHESIS-GENERATING (added later; these rows suggested the idea)")
        else:
            rows = fresh_rows(a.rows, outcomes=True)
            out, _ = stats(rows, read_results(p["ver"]), variants_present(rows), B_BOOT, "c2_fresh",
                           "fresh BDSP draw: CONFIRMATION (C2 bar)")
            out["bar"] = fresh_bar(out["model"])
    summary = paths(mode)["summary"]
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    with open(summary, "w") as fh:
        json.dump(mg._jsonable(out), fh, indent=1, default=float)
    print(f"128 {mode}: {out['rows']['n_rows']} rows ({out['rows']['n_na']} NA log VER) -> {summary}")


if __name__ == "__main__":
    main()
