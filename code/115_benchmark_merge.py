"""Merge of the common null benchmark: the first-stage Bars of
 on the rows of 114_benchmark.py.
python 115_benchmark_merge.py --split dev|analysis [--B 2000] [--freeze-eval-points]
Dev rows (output/intermediate/_bench_shards_dev) are merged into output/intermediate/_bench_dev_tables only (logged,
never reported); analysis rows (output/intermediate/_bench_shards) into output/tables. A row of the other split, or a
development recording among analysis rows, stops the merge. Analysis rows of the recordings touched before the dev
freeze (git-ignored output/bdsp/meta/touched_before_freeze.json, key
"excluded") are dropped (counted in the headline); an analysis merge without that file is refused.

Bootstrap (every CI): patient-level, B = 2000, relabelled draws. Per-patient quantities are computed first and the
patients are resampled as units, so a patient drawn twice counts as two patients (the same as giving each drawn copy a
new id before any per-patient rule is applied; test_merge checks the equivalence). Percentile intervals.

Published pipelines (per null set x pipeline x variant; rows: no responder, native montage, the first overlay j = 0,
one per recording):
- false-call rate = calls / overlays; one-sided 95% lower bound = 5th percentile of the bootstrap;
- variants that cannot call by construction (minimum attainable p > .05: rows' min_attainable_p, else
  bench.wrappers.min_attainable_p) are reported, flagged cannot_call, and never the validity primary or the comparator;
- validity primary = the variant with the FEWEST false calls on the recordings where every candidate variant ran
  (selection; each variant's own n reported; ties: the as-coded variant, then name); p_boot =
  (#{draws <= 5%} + 1) / (B + 1); Holm across pipelines within the null set; "inflated" = Holm-adjusted p_boot <= .05
  ("inflated_unadjusted" = lower bound > 5%). Strata (Morgoth background, artifact index) are descriptive.
N-passive: the published calls form the separate "sensory read as volition" family (same bar, Holm across pipelines
within the null set); a label permutation on physically different stimuli is not a volition null. The instruction-
randomization test there is exact by construction (the sham instruction is drawn independently of the EEG and the
stimulation is identical across conditions), so it is reported as a CONSTRUCTION CHECK (construction_check: per
subject, per run), never in the proposed method's validity families or pooled point estimate.
Proposed method, validity (rand_call of every randomized-use overlay, rows without a responder, native montage):
- the randomized method runs once per TIMING GROUP (114 timing_groups: variants with the same trial count, timing
  and response model), on that group's timing;
- point estimate <= 5% (all null sets and groups pooled; per null set and per group reported);
- one-sided exact binomial tests vs 5% in three families per pipeline, Holm within each family: null set (each null
  set x timing group), stratum (null set x Morgoth background; null set x artifact label), recording (null set x
  recording);
- pass = point <= 5% and no Holm-significant test.
Sensitivity (second stage: the size correction is on the STATISTIC; rows of the first overlay, native montage):
- threshold per pipeline x variant x stratum (Morgoth background, "transitioning" its own stratum) = 95th percentile
  ('higher') of the null stat_for_size_correction, CROSS-FITTED: halves by crc32(patient) parity, each half judged by
  the other half's threshold; strata with < MIN_STRATUM null overlays in the calibrating half use that half's pooled
  threshold (flagged). The corrected comparator calls if its statistic >= the threshold; sensitivity = mean of the two
  halves' call rates; corrected_size = the comparator's cross-fitted null call rate (the achieved size).
- locator (resp "grid", the 60 locator patients): per model x response probability and timing group, comparator =
  the group's variant with the highest corrected sensitivity averaged over the grid (ties: as coded, then name);
  evaluation point = the smallest grid effect at which it reaches >= 50%, else the largest grid effect, labelled "50%
  not reached". The pipeline group (headline) = the group of the overall best comparator. The points are written to
  the evaluation-point file (114 --resp eval reads it) only by --freeze-eval-points, which refuses unless the locator
  is complete (every locator patient x variant x grid point has a row, error rows counted); entries are written once,
  and a later freeze that locates a different point reports it in eval_points_moved and does not overwrite.
- evaluation (resp "eval", all patients at that point): non-inferiority = proposed (rand_call) minus corrected
  comparator, patient-paired bootstrap stratified by half (the estimate is the mean of the halves), margin -5 points,
  on the patients OUTSIDE the locator (on all as well, with overlap_with_locator); on the 100-patient subsample: the
  comparator's own permutation call (uncorrected, labelled "inflated") and the p-based size correction (null p 5th
  percentile, cross-fitted; flagged if > 5 points more sensitive than the statistic-based one).
  Columbia's uncorrected arm runs at ERD p_resp = 1 only (114.UNCORRECTED_POINTS): its other points say so in a
  note (n_patients 0).
pDoC (ncmd_pdoc, nstim0_pdoc) is the reduced-montage arm, nulls only: published, proposed validity, composites and
coverage; no sensitivity. The headline's coverage_notes["pdoc"] gives each module's channel set, Columbia N/A and the
variants not run (gibson16 SSEP reads Pz).
Columbia on BDSP (analysis): 113's rows (columbia_null_rows.parquet; test i = published, test iii = randomized; the
design sample and, part 4, the extra recordings: rows say source_113) replace the runner's first-overlay rows;
they must hold no dev and no touched recording (refused) and lie in every other BDSP pipeline's sample (reported, also
per 113 source). Columbia k = 2 bracket, both labelled bounds: independent days = BH within pairs of
DIFFERENT patients' first overlays (all pairs; patient bootstrap, copies never paired); same session = the first and
the second overlay of one recording, where both exist.
Shortened BDSP twins: the paired difference native minus short on the recordings that ran both, and coverage by
background stratum.
Composites (sequence overlays, non-overlapping, time order j; each module's patient_rule on the first k):
- Columbia BH within patient, k = 1..8 (headline k = 2, per null set); mindBEAGLE-type ge1_run / best_run over k with the per-run
  alpha and 1 - (1 - alpha)^k; du Bois k = 1..12; Cornell (Curley) patient_rule_regrouped, overall and by stratum;
  EAN/Bodien any-approach: any call over the validity-primary variants of every pipeline that ran on the same patients
  of one source (cmd and erp null sets pooled), an upper bound. The headline composite pools COMMAND-FOLLOWING
  pipelines only; any_approach_incl_covert_processing adds Sokoliuk 2021 (COVERT_PROCESSING), labelled (second stage).
Output: <tables>/benchmark_<null>.json per null set and <tables>/benchmark_headline.json."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import importlib
import inspect
import json
import sys
import types
import zlib

import numpy as np
import pandas as pd
from scipy.stats import binom

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

SH = {"dev": os.path.join(ROOT, "output", "intermediate", "_bench_shards_dev"),
      "analysis": os.path.join(ROOT, "output", "intermediate", "_bench_shards")}
TOUCHED_FILE = os.path.join(ROOT, "output", "bdsp", "meta", "touched_before_freeze.json")
OUT = {"dev": os.path.join(ROOT, "output", "intermediate", "_bench_dev_tables"),
       "analysis": os.path.join(ROOT, "output", "tables")}
B_DEFAULT, SEED = 2000, 20261026
ALPHA, NOMINAL, MARGIN, Q = 0.05, 0.05, -0.05, 0.95
MIN_STRATUM = 100
COVERAGE_FLOOR = 0.70
TYPICAL_K = {"columbia": 2, "dubois": 12}
COMPOSITE_K = {"columbia": tuple(range(1, 9)), "dubois": tuple(range(1, 13)),
               "mbvt": (1, 3, 5, 10, 20, 40, 80, 120), "mbmi": (1, 3, 5, 10, 20, 40, 80, 120)}
MB_RULES = ("ge1_run", "best_run")



def holm(p):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate(p[o] * (m - np.arange(m))))
    out = np.empty(m)
    out[o] = adj
    return out


def binom_greater(k, n, p0=NOMINAL):
    """One-sided exact binomial p-value of k or more calls in n at rate p0."""
    return float(binom.sf(k - 1, n, p0)) if n else 1.0


def bh_reject(p, q=0.05):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    rej = np.zeros(m, bool)
    if ok.any():
        rej[o[:np.nonzero(ok)[0].max() + 1]] = True
    return rej


def _seed(*parts):
    return [SEED] + [zlib.crc32(str(p).encode()) for p in parts]


def boot_idx(n, B, seed):
    """(B, n) patient indices drawn with replacement; each drawn copy is its own unit (relabelled)."""
    return np.random.default_rng(seed).integers(0, n, (B, n))


def mean_draws(values, idx):
    return np.asarray(values, float)[idx].mean(1)


def ratio_draws(s, n, idx):
    return np.asarray(s, float)[idx].sum(1) / np.asarray(n, float)[idx].sum(1)


def _per_patient(d, col):
    g = d.groupby("patient", sort=True)[col]
    return g.sum().to_numpy(float), g.size().to_numpy(float)


def _ci(draws):
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _bg(df, prefix=""):
    tr = df.get(prefix + "transitioning", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    st = df.get(prefix + "stratum", pd.Series("na", index=df.index)).fillna("na").astype(str)
    return np.where(tr, "transitioning", st)


def _pick_variant(table, key, best, as_coded):
    """The variant with the best value of key (min or max); ties: as coded first, then name."""
    sign = 1 if best == "min" else -1
    return sorted(table, key=lambda v: (sign * table[v][key], not as_coded.get(v, False), v))[0]



def load_rows(split, root=None):
    """All rows of one split's shard directories (duplicate keys within a directory counted once). Refuses rows of
    another split and development recordings among analysis rows."""
    root = root or SH[split]
    frames = []
    for d in sorted(glob.glob(os.path.join(root, "*__*__*__*"))):
        rows = []
        for fn in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
            with open(fn) as fh:
                rows += [json.loads(l) for l in fh if l.strip()]
        if rows:
            df = pd.DataFrame(rows).drop_duplicates("key", keep="first")
            df["dir"] = os.path.basename(d)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if (df["split"] != split).any():
        raise RuntimeError(f"rows of split(s) {sorted(set(df['split']) - {split})} under the {split} shards")
    dev = df["dev"].astype(bool)
    if split != "dev" and dev.any():
        raise RuntimeError(f"{int(dev.sum())} development-recording rows among {split} rows: refused")
    if split == "dev" and (~dev).any():
        raise RuntimeError("non-development recordings among dev rows: refused")
    return df


def touched_uids():
    """Recordings touched before the dev freeze (key "excluded" of the git-ignored TOUCHED_FILE; part 3)."""
    if not os.path.exists(TOUCHED_FILE):
        raise RuntimeError(f"{TOUCHED_FILE} is missing: an analysis merge needs the list of recordings touched before "
                           "the dev freeze")
    with open(TOUCHED_FILE) as fh:
        return set(json.load(fh)["excluded"])


def drop_touched(df):
    """Rows of the recordings touched before the dev freeze removed (the runner already skips them); the counts."""
    hit = df["uid"].isin(touched_uids())
    return df[~hit], {"recordings": int(df.loc[hit, "uid"].nunique()), "rows": int(hit.sum())}


def _ok(df):
    return df[df["status"] == "ok"]


def _flag(df, col, default=False):
    return df[col].fillna(default).astype(bool) if col in df else pd.Series(default, index=df.index)


def _null_seq(df, montage="native"):
    """No-responder rows of published variants on non-overlapping sequence overlays."""
    d = _ok(df)
    d = d[(d["resp"] == "none") & (d["montage"] == montage) & d["variant"].notna()]
    return d[_flag(d, "seq", True) & ~_flag(d, "overlap")]



def cannot_call_of(d, pipe, variant):
    """True if the published variant cannot call by construction (minimum attainable p > .05): from its rows'
    min_attainable_p or cannot_call, else derived from the module (bench.wrappers.min_attainable_p)."""
    if "min_attainable_p" in d and d["min_attainable_p"].notna().any():
        return bool(d["min_attainable_p"].astype(float).max() > NOMINAL)
    if "cannot_call" in d and d["cannot_call"].notna().any():
        return bool(d["cannot_call"].astype(bool).any())
    mod = _load_pipe(pipe)
    if mod is None:
        return False
    rn, wr = importlib.import_module("114_benchmark"), importlib.import_module("bench.wrappers")
    try:
        sp = rn._spec(mod, variant)
    except (KeyError, AttributeError):
        return False
    with wr.module_globals(mod, sp["globals"]):
        return bool(wr.cannot_call(mod, sp["arg"]))


def coverage_of(df, nul, pipe, variant, montage="native"):
    """(usable share, n attempted) of a published variant's first-overlay null rows: usable = status ok; attempted =
    ok + 'na: no_usable_trials' + error rows (a window that does not fit, 'no_fit', is not attempted)."""
    d = df[(df["null"] == nul) & (df["pipe"] == pipe) & (df["variant"] == variant) & (df["resp"] == "none")
           & (df["montage"] == montage)]
    d = d[_flag(d, "seq", True) & ~_flag(d, "overlap")]
    if "j" in d:
        d = d[d["j"] == 0]
    st = d["status"].astype(str)
    att = st.eq("ok") | st.eq(NA_NO_USABLE) | st.str.startswith("error")
    n = int(att.sum())
    return (float(st[att].eq("ok").sum() / n) if n else None), n


def low_coverage(cov):
    return cov is not None and cov < COVERAGE_FLOOR


def _select_primary(dp, ok, coded):
    """(validity primary, selection record): the fewest false calls on the recordings where EVERY candidate variant
    ran (like for like: native variants run only where their length fits); each variant's own n is reported in
    variants. Without such a recording, on each variant's own recordings (noted)."""
    if not ok:
        return None, {"on": None, "n_common": 0}
    common = set.intersection(*[set(dp[dp["variant"] == v]["uid"]) for v in ok])
    if not common:
        return _pick_variant(ok, "rate", "min", coded), {
            "on": "own recordings", "n_common": 0, "note": "no recording ran every candidate variant"}
    d = dp[dp["uid"].isin(common)]
    rates = {v: {"rate": float(d[d["variant"] == v]["call"].astype(float).mean())} for v in ok}
    return _pick_variant(rates, "rate", "min", coded), {"on": "common recordings", "n_common": len(common),
                                                         "rates_common": {v: r["rate"] for v, r in rates.items()}}


def published(df, B=B_DEFAULT, montage="native"):
    """Per pipeline: variants (rate, lower95, p_boot, strata), validity primary, Holm across pipelines."""
    d = _null_seq(df, montage)
    d = d[d["j"] == 0]
    out = {}
    for pipe, dp in d.groupby("pipe"):
        tab, coded = {}, {}
        for v, dv in dp.groupby("variant"):
            dv = dv.assign(c=dv["call"].astype(float))
            s, n = _per_patient(dv, "c")
            draws = ratio_draws(s, n, boot_idx(len(s), B, _seed(dv["null"].iloc[0], pipe, v, montage)))
            strata = {}
            for name, lab in (("background", _bg(dv)), ("artifact", dv.get("artifact_label", "na"))):
                g = dv.assign(lab=lab).groupby("lab")["c"]
                strata[name] = {str(k): {"n": int(sz), "rate": float(m)} for k, sz, m in
                                zip(g.size().index, g.size().to_numpy(), g.mean().to_numpy())}
            tab[v] = {"n": int(n.sum()), "n_patients": int(len(s)), "calls": int(s.sum()),
                      "rate": float(s.sum() / n.sum()), "lower95": float(np.quantile(draws, 0.05)),
                      "ci95": _ci(draws), "p_boot": float((np.sum(draws <= NOMINAL) + 1) / (B + 1)),
                      "strata": strata}
            coded[v] = bool(_flag(dv, "as_coded").any())
            tab[v]["cannot_call"] = cannot_call_of(dv, pipe, v)
            tab[v]["coverage"], tab[v]["n_attempted"] = coverage_of(df, dv["null"].iloc[0], pipe, v, montage)
            tab[v]["low_coverage"] = low_coverage(tab[v]["coverage"])
        ok = {v: t for v, t in tab.items() if not t["cannot_call"] and not t["low_coverage"]}
        prim, selection = _select_primary(dp, ok, coded)
        errors = df[(df["pipe"] == pipe) & df["status"].astype(str).str.startswith("error")]
        rates = [t["rate"] for t in ok.values()]
        out[pipe] = {"variants": tab, "validity_primary": prim, "selection": selection,
                     "p_boot": tab[prim]["p_boot"] if prim else None,
                     "inflated_unadjusted": bool(prim and tab[prim]["lower95"] > NOMINAL),
                     "variant_range": [min(rates), max(rates)] if rates else None,
                     "variant_range_all": [min(t["rate"] for t in tab.values()), max(t["rate"] for t in tab.values())],
                     "cannot_call": sorted(v for v, t in tab.items() if t["cannot_call"]),
                     "low_coverage": sorted(v for v, t in tab.items() if t["low_coverage"]),
                     "n_error_rows": int(len(errors)),
                     "n_no_fit": int((df[(df["pipe"] == pipe)]["status"] == "no_fit").sum()),
                     "n_na_no_usable_trials": int((df[(df["pipe"] == pipe)]["status"] == NA_NO_USABLE).sum())}
        if prim is None:
            out[pipe].update(inflated=False, note="every variant cannot call by construction or is usable on < 70% "
                                                  "of its recordings: no validity primary")
    pipes = sorted(p for p in out if out[p]["validity_primary"] is not None)
    adj = holm([out[p]["p_boot"] for p in pipes])
    for p, a in zip(pipes, adj):
        out[p]["p_holm"] = float(a)
        out[p]["inflated"] = bool(a <= ALPHA)
    return out



def _rand_rows(df):
    """The proposed method's no-responder draws: the randomized overlays of the engine nulls. The N-passive
    instruction-randomization rows are a construction check (construction_check), never validity draws."""
    if "rand_call" not in df:
        return df.iloc[:0].assign(rand_call=None)
    d = _ok(df)
    d = d[(d["resp"] == "none") & (d["montage"] == "native") & d["rand_call"].notna()]
    if "design" in d:
        d = d[d["design"].fillna("") != "instruction"]
    con = d["contrast"].fillna("-") if "contrast" in d else pd.Series("-", index=d.index)
    grp = d["rand_variant"].fillna("-").astype(str) if "rand_variant" in d else pd.Series("-", index=d.index)
    return d.assign(_con=con, _grp=grp).drop_duplicates(["null", "pipe", "uid", "j", "_grp", "_con"]).drop(
        columns="_con")


def proposed_validity(df):
    out = {}
    for pipe, d in _rand_rows(df).groupby("pipe"):
        c = d["rand_call"].astype(bool)
        bg = _bg(d, "rand_") if "rand_stratum" in d else _bg(d)
        art = d["rand_artifact_label"] if "rand_artifact_label" in d else d.get("artifact_label", "na")
        units = {"null_set": np.where(d["_grp"] == "-", d["null"], d["null"] + "|" + d["_grp"]),
                 "stratum": np.r_[d["null"] + "|bg:" + bg, d["null"] + "|art:" + pd.Series(art, index=d.index)
                                  .fillna("na").astype(str)],
                 "recording": (d["null"] + "|" + d["uid"]).to_numpy()}
        fam = {}
        for name, u in units.items():
            cc = np.r_[c.to_numpy(), c.to_numpy()] if name == "stratum" else c.to_numpy()
            tab = pd.DataFrame({"u": u, "c": cc}).groupby("u")["c"].agg(["sum", "size"])
            p = [binom_greater(int(k), int(n)) for k, n in zip(tab["sum"], tab["size"])]
            adj = holm(p)
            fam[name] = [{"unit": str(u_), "calls": int(k), "n": int(n), "rate": float(k / n), "p": float(pp),
                          "p_holm": float(a)} for u_, k, n, pp, a in zip(tab.index, tab["sum"], tab["size"], p, adj)]
        point = float(c.mean())
        sig = [t["unit"] for f in fam.values() for t in f if t["p_holm"] <= ALPHA]
        out[pipe] = {"n": int(len(d)), "point": point,
                     "point_by_null": {k: float(v) for k, v in d.groupby("null")["rand_call"].mean().items()},
                     "point_by_group": {k: float(v) for k, v in
                                        c.astype(float).groupby(d["_grp"]).mean().items()},
                     "families": fam, "significant": sig, "pass": bool(point <= NOMINAL and not sig)}
    return out


CONSTRUCTION_LABEL = ("construction check, not a validity test: the sham instruction is drawn independently of the "
                      "EEG and the stimulation is identical across conditions, so P(call) <= 5% by construction")


def construction_check(df):
    """N-passive: the instruction-randomization test per subject (runs pooled) and per run (rate, one-sided binomial
    vs 5%): an implementation check, outside the validity families."""
    if "unit" not in df or "rand_call" not in df:
        return {}
    d = _ok(df)
    d = d[d["unit"].isin(["subject", "run"]) & d["rand_call"].notna()]
    out = {}
    for (nul, pipe), g in d.groupby(["null", "pipe"]):
        res = {"label": CONSTRUCTION_LABEL}
        for unit, key in (("subject", "per_subject"), ("run", "per_run")):
            u = g[g["unit"] == unit]
            k, n = int(u["rand_call"].astype(bool).sum()), int(len(u))
            res[key] = {"n": n, "calls": k, "rate": k / n if n else None, "p": binom_greater(k, n)}
        out.setdefault(nul, {})[pipe] = res
    return out



def _half(patients):
    return np.array([zlib.crc32(str(p).encode()) & 1 for p in patients])


def _censored(d, col):
    """Censoring flags of a p column: <col>_censored (114), else x_stopped_early for the run's own p."""
    c = "p_censored" if col == "p" else f"{col}_censored"
    for k in (c, "x_stopped_early" if col == "p" else None):
        if k and k in d:
            return d[k].fillna(False).astype(bool)
    return pd.Series(False, index=d.index)


def _threshold(v, cens, q, method):
    """(threshold, estimable). p-based ('lower'): early-stopped p-values are censored above their pile, so the
    quantile is not estimable when it falls on a censored value (at or above the smallest pile)."""
    t = float(np.quantile(v, q, method=method))
    if method == "lower" and cens.any() and t >= float(v[cens].min()) - 1e-12:
        return None, False
    return t, True


def crossfit_calls(null, resp, stat="size_stat", resp_stat=None, q=Q, upper=True):
    """resp rows with thr, thr_half (the calibrating half, never the row's own), thr_pooled, thr_not_estimable and
    corr_call. upper: threshold = the q quantile ('higher') of the null statistic, call if the responder's statistic
    >= it (the statistic-based size correction). Otherwise (p-based): the q quantile ('lower') of the null p, call if
    the responder's p is strictly below it (as the published p < alpha); early-stopped p-values (p_censored /
    x_stopped_early) are censored above their pile, so a threshold that falls on the pile is not estimable (no call)
    and a stopped responder is never called."""
    resp_stat = resp_stat or stat
    null = null.assign(h=_half(null["patient"]), s=_bg(null), _c=_censored(null, stat))
    out = resp.copy()
    out["h"], out["s"] = _half(out["patient"]), _bg(out)
    method = "higher" if upper else "lower"
    tables = {}
    for h in (0, 1):
        nh = null[(null["h"] == h) & null[stat].notna()]
        by = {s_: _threshold(g[stat].astype(float).to_numpy(), g["_c"].to_numpy(), q, method)
              for s_, g in nh.groupby("s") if len(g) >= MIN_STRATUM}
        pool = (_threshold(nh[stat].astype(float).to_numpy(), nh["_c"].to_numpy(), q, method) if len(nh)
                else (None, True))
        tables[h] = (by, pool)
    thr, pooled, src, est = [], [], [], []
    for h, s_ in zip(out["h"], out["s"]):
        by, pool = tables[1 - h]
        t, ok_ = by.get(s_, pool)
        thr.append(t)
        est.append(ok_)
        pooled.append(s_ not in by)
        src.append(1 - h)
    out["thr"], out["thr_pooled"], out["thr_half"] = thr, pooled, src
    out["thr_not_estimable"] = [not e for e in est]
    t = pd.to_numeric(out["thr"], errors="coerce")
    x = pd.to_numeric(out[resp_stat], errors="coerce")
    ok = t.notna() & x.notna()
    corr = pd.Series(None, index=out.index, dtype=object)
    if upper:
        corr[ok] = (x >= t)[ok]
    else:
        rc = _censored(out, resp_stat).to_numpy()
        corr[ok] = ((x < t) & ~rc)[ok]
    out["corr_call"] = corr
    return out.drop(columns=["h", "s"])


def halves_mean(df, col):
    """Mean of col within each crc32-parity half of the patients, averaged over the halves present."""
    h = _half(df["patient"])
    v = df[col].astype(float).to_numpy()
    return float(np.mean([np.nanmean(v[h == k]) for k in (0, 1) if np.isfinite(v[h == k]).any()]))


def evaluation_point(sens):
    """The smallest grid effect at which the (corrected comparator's) sensitivity is >= 50%; None if none."""
    hit = [e for e in sorted(sens) if sens[e] >= 0.5]
    return hit[0] if hit else None


def ni_test(d, a, b, B=B_DEFAULT, seed=0, margin=MARGIN):
    """mean(a) - mean(b) on the same overlays: per half, then averaged over halves; patient-paired bootstrap
    stratified by half."""
    h = _half(d["patient"])
    rng = np.random.default_rng(seed)
    est, draws = [], []
    for k in (0, 1):
        p = d[h == k].reset_index(drop=True)
        if not len(p):
            continue
        g = p.assign(x=p[a].astype(float), y=p[b].astype(float)).groupby("patient", sort=False)
        sa, sb, n = g["x"].sum().to_numpy(), g["y"].sum().to_numpy(), g.size().to_numpy(float)
        idx = rng.integers(0, len(n), (B, len(n)))
        est.append(sa.sum() / n.sum() - sb.sum() / n.sum())
        draws.append((sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1))
    diff, dr = float(np.mean(est)), np.mean(draws, axis=0)
    ci = _ci(dr)
    return {"diff": diff, "ci95": ci, "margin": margin, "noninferior": bool(ci[0] > margin),
            "n_patients": int(d["patient"].nunique()), "n": int(len(d))}


def _resp_rows(df, resp):
    d = _ok(df)
    d = d[(d["resp"] == resp) & (d["montage"] == "native") & d["variant"].notna()]
    return d[_flag(d, "seq", True) & ~_flag(d, "overlap")]


NOT_REACHED = "50% not reached"


def _group_of(d):
    """Timing group of each row (114's representative variant); rows without one form a single group."""
    return d["group"].fillna("all").astype(str) if "group" in d else pd.Series("all", index=d.index)


def locate(df):
    """Evaluation-point locator (the seeded locator patients' grid rows, statistic only): per null set x pipeline x
    model|prob and TIMING GROUP (variants with the same trial count, timing and response model; 114 timing_groups),
    the comparator (highest corrected sensitivity averaged over the grid, within the group), its curve and the
    evaluation effect: the smallest grid effect reaching >= 50%, else the LARGEST grid effect, labelled "50% not
    reached". The pipeline group = the group holding the overall best comparator (charitable); its comparator and
    effect are also the top-level entries. corrected_size = the comparator's cross-fitted null call rate."""
    null_all, g_all = _null_seq(df), _resp_rows(df, "grid")
    g_all = g_all[_flag(g_all, "in_locator", True)]
    out = {}
    if not len(g_all):
        return out
    for (nul, pipe), rp in g_all.groupby(["null", "pipe"]):
        npipe = null_all[(null_all["null"] == nul) & (null_all["pipe"] == pipe)]
        gmap = rp.assign(_g=_group_of(rp)).groupby("variant")["_g"].first().to_dict()
        cc = {v: crossfit_calls(npipe[npipe["variant"] == v], rv) for v, rv in rp.groupby("variant")}
        cannot = sorted(v for v, rv in rp.groupby("variant")
                        if cannot_call_of(npipe[npipe["variant"] == v] if (npipe["variant"] == v).any() else rv,
                                          pipe, v))
        low = sorted(v for v in rp["variant"].unique() if low_coverage(coverage_of(df, nul, pipe, v)[0]))
        res = {}
        for (model, prob), _ in rp.groupby(["model", "prob"]):
            curves, coded = {}, {}
            for v, rv in cc.items():
                rv = rv[(rv["model"] == model) & (rv["prob"] == prob) & rv["corr_call"].notna()]
                if len(rv):
                    curves[v] = {float(e): halves_mean(g, "corr_call") for e, g in rv.groupby("effect")}
                    coded[v] = bool(_flag(rv, "as_coded").any())
            if not curves:
                continue
            groups = {}
            for grp in sorted({gmap[v] for v in curves}):
                ok = {v: {"mean": float(np.mean(list(curves[v].values())))} for v in curves
                      if gmap[v] == grp and v not in cannot and v not in low}
                if not ok:
                    groups[grp] = {"comparator": None, "evaluation_effect": None, "reached": False,
                                   "note": "every variant of the group cannot call or has < 70% coverage"}
                    continue
                comp = _pick_variant(ok, "mean", "max", coded)
                eff = evaluation_point(curves[comp])
                nv = npipe[npipe["variant"] == comp]
                size = crossfit_calls(nv, nv)["corr_call"].dropna()
                groups[grp] = {"comparator": comp, "evaluation_effect": eff if eff is not None else max(curves[comp]),
                               "reached": eff is not None, "mean": ok[comp]["mean"],
                               "variants": sorted(v for v in curves if gmap[v] == grp),
                               "corrected_size": float(size.astype(float).mean()) if len(size) else None}
                if eff is None:
                    groups[grp]["label"] = NOT_REACHED
            cand = {g: x for g, x in groups.items() if x["comparator"] is not None}
            pg = _pick_variant(cand, "mean", "max", {g: coded.get(x["comparator"], False) for g, x in cand.items()}) \
                if cand else None
            top = groups.get(pg, {})
            res[f"{model}|{prob:g}"] = {"comparator": top.get("comparator"),
                                        "evaluation_effect": top.get("evaluation_effect"),
                                        "reached": top.get("reached"), "pipeline_group": pg, "groups": groups,
                                        "cannot_call": [v for v in cannot if v in curves],
                                        "low_coverage": [v for v in low if v in curves],
                                        "n_patients": int(rp["patient"].nunique()),
                                        "corrected_size": top.get("corrected_size"),
                                        "curves": {v: {f"{k:g}": x for k, x in c.items()} for v, c in curves.items()}}
        out.setdefault(nul, {})[pipe] = res
    return out


def eval_points_of(loc):
    """{null: {pipe: {model|prob: {groups: {group: effect, comparator, reached}, pipeline_group}}}} (114 --resp eval)."""
    return {n: {p: {k: {"groups": {g: {"effect": x["evaluation_effect"], "comparator": x["comparator"],
                                        "reached": x["reached"]} for g, x in r["groups"].items()
                                   if x["comparator"] is not None},
                        "pipeline_group": r["pipeline_group"]} for k, r in v.items() if r["pipeline_group"] is not None}
                for p, v in d.items()} for n, d in loc.items()}


LOCATOR_PROBS = None
NO_RUN = ("no_fit", "not_in_113", "no_responder_model")


def grid_for(split, spec_path=None):
    return importlib.import_module("114_benchmark").load_grid(split, spec_path)[0]


def expected_locator(df, split):
    """{null: locator patients} for every null set with grid rows: 114's seeded subsample of the split's listing."""
    rn = importlib.import_module("114_benchmark")
    skip = touched_uids() if split != "dev" else frozenset()
    out = {}
    for n in sorted(df.loc[df["resp"] == "grid", "null"].unique()):
        spec = rn.parse_null(n)
        out[n] = rn.subsamples([it["patient"] for it in rn.list_items(spec, split, skip)], spec["source"])[0]
    return out


def locator_completeness(df, expected, grid, probs=None):
    """Per null|pipe with grid rows: every expected locator patient has a row (any status: error rows count as run and
    are counted) for each of its variants at every grid point, or a row saying it cannot run (NO_RUN)."""
    rn = importlib.import_module("114_benchmark")
    probs = probs or LOCATOR_PROBS or rn.PROBS
    rows = df[df["resp"] == "grid"]
    out = {}
    for (nul, pipe), d in rows.groupby(["null", "pipe"]):
        fam = str(d["family"].iloc[0]) if "family" in d and d["family"].notna().any() else "cmd"
        points = {(m, float(e), float(p)) for m in rn.MODELS[fam] for e in grid.get(m, []) for p in probs}
        exp = set(expected.get(nul, set()))
        missing, partial = [], []
        for pat in sorted(exp):
            pr = d[d["patient"] == pat]
            if not len(pr):
                missing.append(pat)
                continue
            vs = set(pr["variant"].dropna())
            if not vs:
                if not pr["status"].isin(NO_RUN).any():
                    partial.append(pat)
                continue
            pv = pr[pr["variant"].notna()]
            have = set(zip(pv["variant"], pv["model"], pv["effect"].astype(float), pv["prob"].astype(float)))
            if not {(v,) + pt for v in vs for pt in points} <= have:
                partial.append(pat)
        out[f"{nul}|{pipe}"] = {"complete": not missing and not partial, "n_expected": len(exp),
                                "n_complete": len(exp) - len(missing) - len(partial), "missing": missing[:20],
                                "incomplete": partial[:20],
                                "error_rows": int(d["status"].astype(str).str.startswith("error").sum())}
    return out


def write_eval_points(points, path):
    """Write the located points once: entries already in the file are never changed (new entries are added);
    returns the 'null|pipe|model|prob' entries whose new location differs from the file's."""
    old = {}
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
    moved, merged = [], json.loads(json.dumps(old))
    for n, d in points.items():
        for p, v in d.items():
            for k, e in v.items():
                cur = old.get(n, {}).get(p, {}).get(k)
                if cur is None:
                    merged.setdefault(n, {}).setdefault(p, {})[k] = e
                elif cur != e:
                    moved.append(f"{n}|{p}|{k}")
    if merged != old or not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(merged, fh, indent=1)
        os.replace(tmp, path)
    return moved


def _arm(d, a, b, B, seed):
    if not len(d):
        return {"n_patients": 0}
    return {"n_patients": int(d["patient"].nunique()), "proposed": halves_mean(d, a), "comparator": halves_mean(d, b),
            "ni": ni_test(d, a, b, B, seed)}


def evaluate(df, B=B_DEFAULT):
    """At each located point, per timing group (resp "eval" rows: the group's comparator, statistic only, paired with
    the group's own randomized method on the same timing): non-inferiority of the proposed method vs the
    statistic-corrected comparator on the patients OUTSIDE the locator (and on all, with the overlap flagged); on the
    uncorrected subsample: proposed vs the comparator's own permutation call (labelled inflated) and the p-based size
    correction vs the statistic-based one (flag if > 5 points more sensitive). Every group is reported (groups); the
    top-level entries are the pipeline group's (the group of the overall best comparator). A group whose comparator
    never reached 50% on the grid is evaluated at the largest grid effect and labelled "50% not reached"."""
    null_all, ev = _null_seq(df), _resp_rows(df, "eval")
    out = {}
    if not len(ev):
        return out
    rn = importlib.import_module("114_benchmark")
    for (nul, pipe, model, prob), rv_all in ev.groupby(["null", "pipe", "model", "prob"]):
        rv_all = rv_all.assign(_g=_group_of(rv_all))
        per = {g: _evaluate_group(null_all, rv, nul, pipe, model, prob, B, rn) for g, rv in rv_all.groupby("_g")}
        pg = sorted(set(rv_all.loc[_flag(rv_all, "pipeline_group"), "_g"]))
        pg = pg[0] if pg else (next(iter(per)) if len(per) == 1 else None)
        out.setdefault(nul, {}).setdefault(pipe, {})[f"{model}|{prob:g}"] = {**per.get(pg, {}), "pipeline_group": pg,
                                                                            "groups": per}
    return out


def _evaluate_group(null_all, rv, nul, pipe, model, prob, B, rn):
    """One timing group's evaluation (its comparator and its own randomized method)."""
    comp = sorted(rv["variant"].unique())
    nv = null_all[(null_all["null"] == nul) & (null_all["pipe"] == pipe) & (null_all["variant"] == comp[0])]
    c = crossfit_calls(nv, rv)
    c = c[c["corr_call"].notna() & c["rand_call"].notna()]
    loc = _flag(c, "in_locator")
    sd = _seed(nul, pipe, model, prob)
    r = {"comparator": comp[0], "effect": float(rv["effect"].iloc[0]), "n_comparators": len(comp),
         "eval_set": str(rv["eval_set"].iloc[0]) if "eval_set" in rv and rv["eval_set"].notna().any() else "all",
         "outside_locator": _arm(c[~loc], "rand_call", "corr_call", B, sd),
         "all": _arm(c, "rand_call", "corr_call", B, sd), "overlap_with_locator": bool(loc.any())}
    sub = c[_flag(c, "in_uncorrected") & c.get("perm_call", pd.Series(dtype=object)).notna()] \
        if "perm_call" in c else c.iloc[:0]
    r["uncorrected"] = dict(_arm(sub, "rand_call", "perm_call", B, sd), label="inflated")
    if not rn.uncorrected_point(pipe, model, prob):
        why = (f"not run (run-time cost): {pipe}'s uncorrected arm runs at "
               + ", ".join(f"{m.upper()} p_resp = {q:g}" for m, q in sorted(rn.UNCORRECTED_POINTS[pipe])) + " only")
        r["uncorrected"]["note"] = why
        r["p_based"] = {"n_patients": 0, "note": why}
    elif len(sub) and "perm_p" in sub and sub["perm_p"].notna().any() and nv["p"].notna().any():
        pb = crossfit_calls(nv, sub[sub["perm_p"].notna()].drop(columns=["thr", "thr_pooled", "thr_half",
                                                                           "thr_not_estimable", "corr_call"]),
                            stat="p", resp_stat="perm_p", q=ALPHA, upper=False)
        ne = pb["thr_not_estimable"].astype(bool)
        if ne.all():
            r["p_based"] = {"n_patients": int(pb["patient"].nunique()),
                            "note": "not estimable: the null p 5th percentile falls on the early-stopping pile "
                                    "(stopped p-values are censored above it)"}
        else:
            pb = pb[~ne]
            sp = halves_mean(pb, "corr_call")
            ss = halves_mean(sub[sub["perm_p"].notna() & sub["patient"].isin(pb["patient"])], "corr_call")
            r["p_based"] = {"n_patients": int(pb["patient"].nunique()), "sensitivity_p": sp,
                            "sensitivity_stat": ss, "flag_p_more_sensitive": bool(sp - ss > 0.05),
                            "n_not_estimable": int(ne.sum())}
    else:
        r["p_based"] = {"n_patients": 0, "note": "no permutation p on the subsample, or no null p for this module"}
    if "reached" in rv and not _flag(rv, "reached", True).all():
        r.update(reached=False, label=NOT_REACHED)
    return r



def rows_from_113(c):
    """Benchmark rows (ncmd_bdsp, columbia, as_coded, overlay j = 0) from 113's merged rows (uid, patient added by
    114.columbia113)."""
    tr = c["transition"].astype(bool)
    return pd.DataFrame({
        "key": "113|" + c["uid"].astype(str), "dir": "113", "null": "ncmd_bdsp", "pipe": "columbia",
        "family": "cmd", "grade": "EXACT", "split": "analysis", "dev": False, "source": "bdsp",
        "uid": c["uid"].astype(str), "patient": c["patient"].astype(str), "resp": "none", "montage": "native",
        "source_113": (c["source"].fillna("design").astype(str) if "source" in c else "design"),
        "j": 0, "seq": True, "overlap": False, "start_s": c["start_s"].astype(float), "variant": "as_coded",
        "as_coded": True, "model": "none", "effect": 0.0, "prob": 0.0, "status": "ok",
        "stat": c["i_auc"].astype(float), "size_stat": c["i_auc"].astype(float), "p": c["i_p"].astype(float),
        "call": c["i_call"].astype(bool), "x_call_paper": c["i_call_paper"].astype(bool),
        "x_n_perm_run": c["i_nperm"], "x_stopped_early": c["i_stopped"].astype(bool),
        "p_censored": c["i_stopped"].astype(bool),
        "rand_variant": "as_coded", "rand_stat": c["iii_auc"].astype(float), "rand_size_stat": c["iii_auc"].astype(float),
        "rand_p": c["iii_p"].astype(float), "rand_call": c["iii_call"].astype(bool), "rand_x_n_perm_run": c["iii_nperm"],
        "stratum": c["stratum"].astype(str), "transitioning": tr, "rand_stratum": c["stratum"].astype(str),
        "rand_transitioning": tr, "artifact_label": None, "rand_artifact_label": None,
        "x_primary_113": c["primary"].astype(bool), "min_attainable_p": 1.0 / 501, "cannot_call": False})


def dev_uids():
    rn = importlib.import_module("114_benchmark")
    fn = rn.en.DEV_FILE
    if not os.path.exists(fn):
        raise RuntimeError(f"{fn} is missing: the BDSP dev split must be frozen before an analysis merge")
    with open(fn) as fh:
        return {str(t) for _, _, t in json.load(fh)["bdsp"]}


def check_113(c, df, dev, touched):
    """113's recordings (design sample and extras alike) must carry the benchmark's exclusions (no dev, no touched
    recording: refused) and lie in the benchmark's BDSP sample (every other BDSP pipeline's rows, whatever their
    status: reported, also per 113 source)."""
    uids = set(c["uid"].astype(str))
    src = (c["source"].fillna("design").astype(str) if "source" in c else pd.Series("design", index=c.index))
    by_src = dict(zip(c["uid"].astype(str), src))
    if uids & set(dev) or uids & set(touched):
        raise RuntimeError(f"113 rows hold {len(uids & set(dev))} dev and {len(uids & set(touched))} touched "
                           "recording(s) that the benchmark excludes")
    bd = df[df["null"].isin(["ncmd_bdsp", "nstim0_bdsp"]) & (df["pipe"] != "columbia")]
    gone = {f"{n}|{p}": uids - set(g["uid"]) for (n, p), g in bd.groupby(["null", "pipe"])}
    missing = {k: len(v) for k, v in gone.items() if v}
    per_src = {k: dict(pd.Series([by_src[u] for u in v]).value_counts().sort_index()) for k, v in gone.items() if v}
    return {"n_113": len(uids), "n_113_by_source": {k: int(v) for k, v in src.value_counts().sort_index().items()},
            "checked_against": int(bd.groupby(["null", "pipe"]).ngroups),
            "contained": bool(len(bd) and not missing), "missing": missing,
            "missing_by_source": {k: {s: int(n) for s, n in v.items()} for k, v in per_src.items()}}



def short_twins(df, B=B_DEFAULT):
    """Per null set x pipeline x short variant: the paired difference native minus short (false-call rate) on the
    recordings that ran both (one overlay, the same start), and coverage by background stratum."""
    d = _null_seq(df)
    d = d[d["j"] == 0]
    if "twin_of" not in d:
        return {}
    out = {}
    for (nul, pipe, s_), g in d[d["twin_of"].notna()].groupby(["null", "pipe", "variant"]):
        res = {"pairs": {}, "coverage": {}}
        nat_all = d[(d["null"] == nul) & (d["pipe"] == pipe)]
        for v in str(g["twin_of"].iloc[0]).split(","):
            nat = nat_all[nat_all["variant"] == v]
            m = g.merge(nat[["uid", "call"]], on="uid", suffixes=("_s", "_n"))
            if not len(m):
                continue
            m = m.assign(dn=m["call_n"].astype(float) - m["call_s"].astype(float))
            sm, n = _per_patient(m, "dn")
            dr = ratio_draws(sm, n, boot_idx(len(sm), B, _seed(nul, pipe, s_, v)))
            res["pairs"][v] = {"n_patients": int(len(sm)), "diff_native_minus_short": float(sm.sum() / n.sum()),
                               "ci95": _ci(dr), "rate_native": float(m["call_n"].astype(float).mean()),
                               "rate_short": float(m["call_s"].astype(float).mean())}
        both = set(nat_all[nat_all["variant"].isin(str(g["twin_of"].iloc[0]).split(","))]["uid"])
        for st, gg in g.assign(bg=_bg(g)).groupby("bg"):
            res["coverage"][str(st)] = {"short": int(gg["uid"].nunique()), "both": int(gg["uid"].isin(both).sum())}
        out.setdefault(nul, {}).setdefault(pipe, {})[s_] = res
    return out



def bh2_pairs_rate(p, idx):
    """Per bootstrap draw (rows of idx, patient indices; each drawn copy a unit): the share of unit pairs from
    DIFFERENT patients in which BH over the two p-values (q = .05) rejects any: min p <= .025 or max p <= .05."""
    p = np.asarray(p, float)
    idx = np.atleast_2d(idx)
    M = np.zeros((len(idx), len(p)))
    for r, row in enumerate(idx):
        M[r] = np.bincount(row, minlength=len(p))
    hi025, mid, hi05 = p > 0.025, (p > 0.025) & (p <= 0.05), p > 0.05
    c2 = lambda x: x * (x - 1) / 2.0
    N = M.sum(1)
    same = c2(M)
    total = c2(N) - same.sum(1)
    nocall = c2(M @ hi025) - c2(M @ mid) - (same * hi05).sum(1)
    return 1.0 - nocall / total


def columbia_k2_bracket(df, B=B_DEFAULT):
    """Columbia BH within patient at k = 2, bracketed (both labelled bounds): independent days = pairs of different
    patients' first overlays, BH within the pair (all pairs; patient bootstrap); same session = the first and the
    second (non-overlapping) overlay of one recording, where both exist."""
    d = _null_seq(df)
    d = d[d["pipe"] == "columbia"]
    out = {}
    for nul, g in d.groupby("null"):
        first = g[g["j"] == 0].sort_values(["patient", "uid"]).drop_duplicates("patient")
        p = first["p"].astype(float).to_numpy()
        ind = {"label": "independent-days bound: BH within pairs of different patients' first overlays (all pairs)",
               "n_patients": int(len(p))}
        if len(p) >= 2:
            ind["rate"] = float(bh2_pairs_rate(p, np.arange(len(p))[None])[0])
            dr = bh2_pairs_rate(p, boot_idx(len(p), B, _seed(nul, "k2", "independent")))
            ind.update(ci95=_ci(dr), lower95=float(np.quantile(dr, 0.05)))
        w = g[g["j"].isin([0, 1])].pivot_table(index=["uid", "patient"], columns="j", values="p", aggfunc="first")
        w = w.dropna() if {0, 1} <= set(w.columns) else w.iloc[:0]
        n_same = int(w.index.get_level_values("patient").nunique())
        same = {"label": f"same-session bound, subsample n = {n_same}", "n_patients": n_same}
        if len(w):
            hit = ((w.min(1) <= 0.025) | (w.max(1) <= 0.05)).astype(float).groupby(level="patient").mean()
            same["rate"] = float(hit.mean())
            dr = mean_draws(hit.to_numpy(), boot_idx(len(hit), B, _seed(nul, "k2", "same")))
            same.update(ci95=_ci(dr), lower95=float(np.quantile(dr, 0.05)))
        out[nul] = {"independent_days_bound": ind, "same_session_bound": same}
    return out



def _results(g):
    xs = [c for c in g.columns if c.startswith("x_")]
    return [{"stat": r["stat"], "p": r["p"], "call": bool(r["call"]),
             "extra": {c[2:]: r[c] for c in xs if pd.notna(r[c])}} for _, r in g.iterrows()]


def patient_outcomes(rows, rule, k):
    """{patient: rule(first k windows)} for patients with >= k non-overlapping sequence windows (time order j)."""
    d = rows[_flag(rows, "seq", True) & ~_flag(rows, "overlap")]
    out = {}
    for pat, g in d.sort_values(["uid", "j"]).groupby("patient", sort=True):
        if len(g) >= k:
            out[pat] = bool(rule(_results(g.iloc[:k])))
    return out


def composite_curve(rows, rule, ks, B=B_DEFAULT, seed=None):
    out = []
    for k in ks:
        o = patient_outcomes(rows, rule, k)
        v = np.array(list(o.values()), float)
        if not len(v):
            out.append({"k": int(k), "n": 0, "rate": None, "ci95": None, "lower95": None})
            continue
        dr = mean_draws(v, boot_idx(len(v), B, seed if seed is not None else _seed("composite", k)))
        out.append({"k": int(k), "n": int(len(v)), "rate": float(v.mean()), "ci95": _ci(dr),
                    "lower95": float(np.quantile(dr, 0.05))})
    return out


def _load_pipe(name):
    try:
        return importlib.import_module(f"bench.pipe_{name}")
    except ImportError:
        return None


def _rule(mod, **kw):
    if mod is None:
        return lambda results: any(r["call"] for r in results)
    sig = inspect.signature(mod.patient_rule).parameters
    return (lambda results: mod.patient_rule(results, **kw)) if kw and "rule" in sig else mod.patient_rule


def composites(df, pub, B=B_DEFAULT):
    """Per null set and pipeline: curves over k for every variant (validity primary marked)."""
    d = _null_seq(df)
    out = {}
    for (nul, pipe), dp in d.groupby(["null", "pipe"]):
        mod = _load_pipe(pipe)
        ks = COMPOSITE_K.get(pipe, (1,))
        prim = pub.get(nul, {}).get(pipe, {}).get("validity_primary")
        res = {"validity_primary": prim, "variants": {}}
        for v, dv in dp.groupby("variant"):
            if pipe in ("mbvt", "mbmi"):
                alpha = float(dv["call"].astype(float).mean())
                cur = {rule: composite_curve(dv, _rule(mod, rule=rule), ks, B, _seed(nul, pipe, v, rule))
                       for rule in MB_RULES}
                for c in cur["ge1_run"]:
                    c["reference_independent"] = 1 - (1 - alpha) ** c["k"]
                res["variants"][v] = {"alpha_run": alpha, "n_runs": int(len(dv)), **cur}
            else:
                res["variants"][v] = {"own_rule": composite_curve(dv, _rule(mod), ks, B, _seed(nul, pipe, v))}
                if mod is not None and hasattr(mod, "patient_rule_regrouped"):
                    res["variants"][v]["curley_regrouped"] = composite_curve(dv, mod.patient_rule_regrouped, (1,), B,
                                                                             _seed(nul, pipe, v, "rg"))
                    first = dv[dv["j"] == 0].assign(bg=_bg(dv[dv["j"] == 0]))
                    res["variants"][v]["curley_regrouped_by_stratum"] = {
                        str(s): composite_curve(g, mod.patient_rule_regrouped, (1,), B, _seed(nul, pipe, v, s))[0]
                        for s, g in first.groupby("bg")}
        out.setdefault(nul, {})[pipe] = res
    return out


def any_approach(df, primary, B=B_DEFAULT):
    """Any call over the given (null, pipe) -> variant on their first overlays, over the patients covered by all of
    them: an upper bound across approaches (EAN / Bodien 'highest level revealed by any')."""
    d = _null_seq(df)
    d = d[d["j"] == 0]
    keep = [(n, p, v) for (n, p), v in primary.items()]
    parts = [d[(d["null"] == n) & (d["pipe"] == p) & (d["variant"] == v)] for n, p, v in keep]
    parts = [x for x in parts if len(x)]
    if not parts:
        return {"n_patients": 0, "rate": None, "pipelines": []}
    common = set.intersection(*[set(x["patient"]) for x in parts])
    pats = sorted(common)
    hit = np.array([any(bool(x[x["patient"] == p]["call"].any()) for x in parts) for p in pats], float)
    out = {"n_patients": len(pats), "pipelines": sorted({x["pipe"].iloc[0] for x in parts}),
           "null_sets": sorted({x["null"].iloc[0] for x in parts}), "rate": float(hit.mean()) if len(hit) else None}
    if len(hit):
        dr = mean_draws(hit, boot_idx(len(hit), B, _seed("any", *out["pipelines"])))
        out.update(ci95=_ci(dr), lower95=float(np.quantile(dr, 0.05)))
    return out


def chancount(df, B=B_DEFAULT):
    d = _ok(df)
    d = d[d["null"].str.startswith("chancount_") & d["variant"].notna()]
    out = {}
    for (nul, v, tag), g in d.groupby(["null", "variant", "montage"]):
        g = g.assign(c=g["call"].astype(float))
        s, n = _per_patient(g, "c")
        dr = ratio_draws(s, n, boot_idx(len(s), B, _seed(nul, v, tag)))
        out.setdefault(nul, {}).setdefault(v, {})[tag] = {"n": int(n.sum()), "rate": float(s.sum() / n.sum()),
                                                          "ci95": _ci(dr), "lower95": float(np.quantile(dr, 0.05))}
    return out


NA_NO_USABLE = "na: no_usable_trials"


def _na_rows(d):
    """Rows a module could not analyse (no usable trials): the published run's status or the randomized arm's."""
    hit = d["status"].astype(str) == NA_NO_USABLE
    if "rand_error" in d:
        hit |= d["rand_error"].astype(str) == NA_NO_USABLE
    return d[hit]


def _na_by_stratum(na):
    """{recordings, by_stratum}: recordings with any no-usable-trials row, by the row's background stratum (the
    randomized overlay's where the published one has none)."""
    if not len(na):
        return {"recordings": 0, "by_stratum": {}}
    lab = pd.Series(_bg(na), index=na.index)
    if "rand_stratum" in na:
        lab = lab.where(na.get("stratum", pd.Series(index=na.index, dtype=object)).notna(),
                        pd.Series(_bg(na, "rand_"), index=na.index))
    first = na.assign(_s=lab).sort_values(["uid", "j"] if "j" in na else ["uid"]).drop_duplicates("uid")
    return {"recordings": int(first["uid"].nunique()),
            "by_stratum": {str(k): int(v) for k, v in first["_s"].value_counts().sort_index().items()}}


def coverage(df):
    """Per null set and pipeline: recordings, no-fit recordings, error rows, recordings the module could not analyse
    (na: no_usable_trials; out of every denominator) by stratum, and how many recordings hold k non-overlapping
    sequence windows of the first published variant (what the composite curves can use)."""
    out = {}
    for (nul, pipe), d in df.groupby(["null", "pipe"]):
        st = d["status"].astype(str)
        seq = _null_seq(d)
        if len(seq):
            seq = seq[seq["variant"] == sorted(seq["variant"].unique())[0]]
        w = seq.groupby("uid")["j"].nunique() if len(seq) else pd.Series(dtype=int)
        out.setdefault(nul, {})[pipe] = {
            "recordings": int(d["uid"].nunique()), "no_fit": int(d[st == "no_fit"]["uid"].nunique()),
            "error_rows": int(st.str.startswith("error").sum()), "ok_rows": int((st == "ok").sum()),
            "na_no_usable_trials": _na_by_stratum(_na_rows(d)),
            "seq_windows_max": int(w.max()) if len(w) else 0,
            "recordings_with_k": {str(k): int((w >= k).sum()) for k in range(1, int(w.max()) + 1)} if len(w) else {}}
    return out



def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


COVERAGE_NOTES = {"icare": "dropped from the command pipelines (second stage): its 8-min windows fit none"}
COVERT_PROCESSING = frozenset({"sokoliuk21"})
CCP_LABEL = ("upper bound including covert-processing pipelines (not command-following): false covert-processing "
             "calls are reported separately from the CMD headline")


def pdoc_note():
    """The pDoC reduced-montage arm (second stage): what runs there, each module's channel set, what is N/A and why."""
    wr = importlib.import_module("bench.wrappers")
    gib = importlib.import_module("bench.pipe_gibson16")
    return {"arm": "reduced montage (F3 F4 C3 C4 O1 O2), nulls only: no responders, no kit8, no sensitivity",
            "channel_sets": {m: wr.pdoc_channel_set(types.SimpleNamespace(NAME=m)) for m in sorted(wr.PDOC_CH)},
            "columbia": wr.PDOC_NA_MODULES["columbia"],
            "n_a_variants": {"gibson16": sorted(v for v in gib.VARIANTS if wr.pdoc_variant_na(gib, v))}}


SECTIONS = ("coverage", "published", "sensory_read_as_volition", "construction_check", "published_kit8",
            "proposed_validity", "sensitivity.locator", "sensitivity.evaluation", "short_twins", "columbia_k2",
            "composites", "chancount")
HEAD_SECTIONS = ("excluded_touched", "columbia_113", "eval_points_moved", "columbia_k2", "null_sets", "any_approach",
                 "proposed_validity")


def is_empty(v):
    return v is None or (isinstance(v, (dict, list, tuple)) and len(v) == 0)


def _get(res, section):
    v = res
    for k in section.split("."):
        v = (v or {}).get(k)
    return v


def _no_columbia(kind, pdoc, split):
    wr = importlib.import_module("bench.wrappers")
    if kind != "ncmd":
        return "Columbia is a command pipeline; this is not a command null set"
    if pdoc:
        return wr.PDOC_NA_MODULES["columbia"]
    return ("no Columbia rows: Columbia is skipped on dev (validated by 112/113)" if split == "dev" else None)


def empty_reasons(n, res, d, split):
    """{section: reason} for every empty section of one null set's output. A section empty for no known reason says
    UNEXPECTED (plumbing to check)."""
    kind, _, src = n.partition("_")
    passive, chan, pdoc = kind == "npassive", kind == "chancount", src == "pdoc"
    pipes, resps = set(d["pipe"]), set(d["resp"])

    def resp_why(resp, what):
        if pdoc:
            return "the pDoC arm is nulls only (reduced montage, second stage): no responders"
        if passive or chan:
            return "no responders on the real-event null sets"
        return what if resp not in resps else None

    by_count = ("the channel-count arm: chennu13 rows are tagged by channel count (n<k>, never native) and reported "
                "in chancount; the full-montage calls are the matching npassive set's published chennu13")
    known = {
        "published": ("N-passive: the published calls form sensory_read_as_volition" if passive else
                      by_count if chan else None),
        "composites": by_count if chan else None,
        "sensory_read_as_volition": None if passive else "not an N-passive null set",
        "construction_check": None if passive else "not an N-passive null set (no instruction-randomization test)",
        "published_kit8": ("the pDoC arm is the reduced-montage arm itself: no kit8" if pdoc else
                           None if (d["montage"] == "kit8").any() else "no kit8 rows: the Kit-8 arm was not run"),
        "proposed_validity": ("the channel-count arm runs the published chennu13 only" if chan else
                              "N-passive: the instruction-randomization test is a construction check "
                              "(construction_check), outside the validity families" if passive else None),
        "sensitivity.locator": resp_why("grid", "no --resp grid rows: the locator was not run"),
        "sensitivity.evaluation": resp_why("eval", "no --resp eval rows: freeze the evaluation points (115 "
                                                   "--freeze-eval-points, complete locator only), then run 114 "
                                                   "--resp eval"),
        "short_twins": (None if src == "bdsp" and pipes & {"cruse11", "cruse12p"} else
                        "shortened twins exist for cruse11 / cruse12p on BDSP only"),
        "columbia_k2": None if "columbia" in pipes else _no_columbia(kind, pdoc, split),
        "chancount": None if chan else "not a channel-count null set"}
    return {s: (known.get(s) or "UNEXPECTED: empty although this null set should populate it")
            for s in SECTIONS if is_empty(_get(res, s))}


def head_empty_reasons(head, df, split):
    known = {"excluded_touched": "dev merge: the touched-before-freeze exclusion applies to analysis rows only"
             if split == "dev" else None,
             "columbia_113": "dev merge: 113's rows are analysis rows" if split == "dev" else None,
             "eval_points_moved": "no located evaluation point differs from the written file",
             "columbia_k2": None if (df["pipe"] == "columbia").any() else _no_columbia("ncmd", False, split),
             "any_approach": None if df["null"].str.startswith(("ncmd_", "nstim0_")).any()
             else "no command / no-stimulus null set merged",
             "proposed_validity": None if not df["null"].str.startswith("chancount_").all()
             else "the channel-count arm runs the published chennu13 only"}
    return {s: (known.get(s) or "UNEXPECTED: empty although the merged rows should populate it")
            for s in HEAD_SECTIONS if is_empty(head.get(s))}


def merge(split, root=None, out=None, B=B_DEFAULT, spec_path=None, eval_file=None, freeze=False, locator_sets=None):
    """Merge one split's rows into the benchmark tables. freeze: also write the located evaluation points (only on a
    complete locator: locator_completeness over locator_sets, default 114's seeded locator of the split's listing)."""
    if split != "dev":
        rn = importlib.import_module("114_benchmark")
        rn.guard_split(split, spec_path)
    df = load_rows(split, root)
    excluded = c113 = None
    rn = importlib.import_module("114_benchmark")
    if split != "dev":
        df, excluded = drop_touched(df)
        if os.path.exists(rn.COLUMBIA_ROWS):
            c = rn.columbia113()
            col = (df["null"] == "ncmd_bdsp") & (df["pipe"] == "columbia") & (df["resp"] == "none") & (df["j"] == 0) \
                if len(df) else pd.Series(dtype=bool)
            df = df[~col] if len(df) else df
            c113 = check_113(c, df, dev_uids(), touched_uids())
            df = pd.concat([df, rows_from_113(c)], ignore_index=True, sort=False)
        else:
            c113 = {"note": f"{rn.COLUMBIA_ROWS} not written yet (113 --merge): Columbia on BDSP not merged"}
    eval_file = eval_file or (os.path.join(out, "_eval_points.json") if out else rn.EVAL_FILE[split])
    out = out or OUT[split]
    os.makedirs(out, exist_ok=True)
    if df.empty:
        print("no rows")
        return {}
    pub = {n: published(d, B) for n, d in df.groupby("null")}
    kit8 = {n: published(d, B, "kit8") for n, d in df.groupby("null") if (d["montage"] == "kit8").any()}
    val = proposed_validity(df)
    cons = construction_check(df)
    loc = locate(df)
    moved, completeness = [], None
    if freeze:
        if not loc:
            raise RuntimeError("no locator (--resp grid) rows: no evaluation point to freeze")
        completeness = locator_completeness(df, locator_sets if locator_sets is not None else
                                            expected_locator(df, split), grid_for(split, spec_path))
        bad = {k: {x: c[x] for x in ("n_expected", "n_complete", "missing", "incomplete")}
               for k, c in completeness.items() if not c["complete"]}
        if bad:
            raise RuntimeError(f"locator incomplete, evaluation points not frozen: {bad}")
        moved = write_eval_points(eval_points_of(loc), eval_file)
    ev = evaluate(df, B)
    comp = composites(df, pub, B)
    twins = short_twins(df, B)
    k2 = columbia_k2_bracket(df, B)
    chan = chancount(df, B)
    cov = coverage(df)
    head = {"split": split, "B": B, "excluded_touched": excluded, "columbia_113": c113,
            "coverage_notes": dict(COVERAGE_NOTES, pdoc=pdoc_note(), kit8=dict(rn.KIT8_NA)), "eval_points_moved": moved,
            "eval_points": {"frozen": os.path.exists(eval_file), "locator_completeness": completeness}, "columbia_k2": k2,
            "null_sets": {}, "any_approach": {}, "any_approach_incl_covert_processing": {},
            "proposed_validity": {
        p: {"point": v["point"], "pass": v["pass"], "significant": v["significant"]} for p, v in val.items()}}
    for n in sorted(df["null"].unique()):
        pipes = set(df[df["null"] == n]["pipe"])
        passive = n.startswith("npassive_")
        res = {"null": n, "split": split, "coverage": cov.get(n, {}),
               "published": {} if passive else pub.get(n, {}),
               "sensory_read_as_volition": pub.get(n, {}) if passive else {},
               "construction_check": cons.get(n, {}), "published_kit8": kit8.get(n, {}),
               "proposed_validity": {p: v for p, v in val.items() if p in pipes},
               "sensitivity": {"locator": loc.get(n, {}), "evaluation": ev.get(n, {})},
               "short_twins": twins.get(n, {}), "columbia_k2": k2.get(n), "composites": comp.get(n, {}),
               "chancount": chan.get(n, {})}
        res["empty_sections"] = empty_reasons(n, res, df[df["null"] == n], split)
        with open(os.path.join(out, f"benchmark_{n}.json"), "w") as fh:
            json.dump(_jsonable(res), fh, indent=1)
        own = {}
        for p, r in pub.get(n, {}).items():
            k = TYPICAL_K.get(p, 1)
            v = r["validity_primary"]
            if v is None:
                own[p] = {"variant": None, "note": r.get("note")}
                continue
            cv = comp.get(n, {}).get(p, {}).get("variants", {}).get(v, {})
            curve = cv.get("own_rule") or cv.get("ge1_run") or []
            at = next((c for c in curve if c["k"] == k), None)
            if at is None or not at["n"]:
                at = {"k": 1, "n": r["variants"][v]["n_patients"], "rate": r["variants"][v]["rate"],
                      "ci95": r["variants"][v]["ci95"], "lower95": r["variants"][v]["lower95"]}
                if k != 1:
                    at["note"] = (f"k = {k} not available (no patient has {k} non-overlapping sequence windows): "
                                  "k = 1 shown")
            own[p] = {"variant": v, "inflated": r["inflated"], **at}
            if p == "columbia":
                if at["k"] == 2:
                    own[p]["label"] = f"same-session bound, subsample n = {at['n']}"
                own[p]["k2_bracket"] = k2.get(n)
        if n.startswith("npassive_"):
            head["null_sets"][n] = {"sensory_read_as_volition": own, "construction_check": {
                p: {"label": "construction check", "rate_per_subject": c["per_subject"]["rate"]}
                for p, c in cons.get(n, {}).items()}}
            continue
        head["null_sets"][n] = {"own_rule": own,
                                "ni": {p: {k: {"pipeline_group": x["pipeline_group"],
                                               "comparator": x["comparator"], "evaluation_effect": x["effect"],
                                               "reached": x.get("reached", True), **({"label": x["label"]}
                                                                                     if "label" in x else {}),
                                               "noninferior": x["outside_locator"].get("ni", {}).get("noninferior"),
                                               "overlap_with_locator": x["overlap_with_locator"],
                                               "p_based_flag": x["p_based"].get("flag_p_more_sensitive"),
                                               "groups_noninferior": {
                                                   g: y["outside_locator"].get("ni", {}).get("noninferior")
                                                   for g, y in x["groups"].items()}}
                                           for k, x in s.items() if "outside_locator" in x}
                                       for p, s in ev.get(n, {}).items()}}
    for src in sorted({n.split("_", 1)[1] for n in df["null"] if n.startswith(("ncmd_", "nstim0_"))}):
        prim = {(n, p): r["validity_primary"] for n in (f"ncmd_{src}", f"nstim0_{src}") for p, r in
                pub.get(n, {}).items() if r["validity_primary"] is not None}
        head["any_approach"][src] = any_approach(df, {k: v for k, v in prim.items() if k[1] not in COVERT_PROCESSING},
                                                 B)
        head["any_approach_incl_covert_processing"][src] = dict(any_approach(df, prim, B), label=CCP_LABEL)
    head["empty_sections"] = head_empty_reasons(head, df, split)
    with open(os.path.join(out, "benchmark_headline.json"), "w") as fh:
        json.dump(_jsonable(head), fh, indent=1)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "analysis"))
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--freeze-eval-points", action="store_true",
                    help="write the located evaluation points (refused unless the locator is complete)")
    a = ap.parse_args()
    head = merge(a.split, B=a.B, freeze=a.freeze_eval_points)
    print(json.dumps(_jsonable({k: head.get(k) for k in ("split", "columbia_k2", "proposed_validity")}), indent=1))


if __name__ == "__main__":
    main()
