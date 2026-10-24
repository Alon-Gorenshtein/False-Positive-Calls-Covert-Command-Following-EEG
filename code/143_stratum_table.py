"""Fidelity-graded false-call table by stratum (pre-draft, descriptive, added later). No new run and no fixed
output changed: it reads the second-stage analysis rows 115 merged (BDSP 974 patients, pDoC 38), the C3 rows of
130_fidelity_readings, 116's X results (pDoC lower-delta flags, 113's rebuilt extents), 121's A9 tables and the A5
summary. Built test-first (code/test_143_stratum_table.py).
- Rows: the command-following pipelines (locked_spec.PIPES without sokoliuk21, which C0 grades NOT A CALL RULE).
- Reading per pipeline (part C, C0 and C3), never a LIBERAL DEPARTURE default:
  - cornell, du Bois: the C3 fidelity-corrected reading (cornell jackknife + OM2 contiguity; du Bois reading B under
    random and blocked CV, a range);
  - edlow17, mbvt: the default, a FAITHFUL LITERAL READING;
  - every other pipeline (no C0 grade): the validity primary of the 115 headline merge (benchmark_<null>.json).
- Windows: the first overlay of every recording (115._null_seq, j = 0; 116.load_rows: touched recordings dropped,
  Columbia's BDSP rows from 113). N/A and error rows are left out of every denominator and counted.
- Strata (116, A1; no new cut-off): BDSP ICU-typical = Morgoth burst_suppression / iiic_periodic / diffuse_slowing
  (116.DISORGANISED) and awake = normal_awake, both non-transitioning and A9-clean (116.a1_rates); pDoC wake-like =
  the extent's mean C3/C4 log delta/alpha below the recording median (116.RecordingX.lower_delta, 91's
  lower_delta_half rule), A9-clean. An unprocessed A9 recording counts as flagged (116.a9_flags).
- Rate: per-patient ratio, patient bootstrap B = 2000 (115.boot_idx: each drawn copy its own unit), 95% percentile
  CI; a validity-primary or default cell uses 116's A1 seed (so it equals 116's cell for the same variant), every
  other cell 115._seed("143", ...).
- Randomised counterpart: the randomization-based method's call (rand_call) on the same recording, with its own
  window's labels: rand_stratum and rand_transitioning, A9 on its own extent, and on pDoC the lower-delta flag of its
  own extent (116's rule recomputed on the pDoC recording). A C3 reading takes its base variant's randomised call.
- Positive control: the A5 primary arm (output/tables/positive_control.json) for the reading's variant and its
  randomised group; the C3 readings were never run in A5, so the base variant is shown, labelled as nearest.
- Published benchmarks: each group's own null or false-positive numbers (the motor catalogue,
  the erp catalogue, the Columbia protocol notes), compared with every stratum by exact tests; each
  benchmark names the stratum closest to its population (closest).
- Checks before anything is written: 116's A1a and A1c rates (n, rate, CI) are reproduced from these rows with
  116's own seeds, the C3 first-overlay counts of c3_fidelity.json are reproduced, and the recomputed lower-delta flag
  equals 116's on every published pDoC row used. The output carries no recording or patient identifier.
python3 143_stratum_table.py [--B 2000]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import hashlib
import importlib
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
mech = importlib.import_module("116_mechanism")
fid = importlib.import_module("130_fidelity_readings")
mg, rn, en = mech.mg, mech.rn, mech.en

SPLIT = "analysis"
B_BOOT = 2000
SHARDS = mg.SH[SPLIT]
C3_SHARDS = os.path.join(ROOT, "output", "intermediate", "_c3_shards")
TABLES = os.path.join(ROOT, "output", "tables")
OUT = os.path.join(TABLES, "stratum_table.json")
PC_FILE = os.path.join(TABLES, "positive_control.json")
MECH_FILE = os.path.join(TABLES, "mechanism.json")
C3_FILE = os.path.join(TABLES, "c3_fidelity.json")

LEFT_OUT = {"sokoliuk21": "NOT A CALL RULE (C0): a prognostic ROC cut-off, reported only as an exceedance"}
CMD_PIPES = tuple(p for p in mech.PIPES if p not in LEFT_OUT)
NULLS = {"bdsp": ("ncmd_bdsp", "nstim0_bdsp"), "pdoc": ("ncmd_pdoc", "nstim0_pdoc")}
DISORGANISED = tuple(mech.DISORGANISED)
C0 = {"cornell": "LIBERAL DEPARTURE", "dubois": "LIBERAL DEPARTURE", "edlow17": "FAITHFUL LITERAL READING",
      "mbvt": "FAITHFUL LITERAL READING"}
UNGRADED = "validity primary (default not graded by C0)"
C3_GRADE = "C3 FIDELITY-CORRECTED (default: LIBERAL DEPARTURE)"
PC_ARM = {"columbia": "openbmi", "cornell": "openbmi", "cruse12p": "openbmi", "dubois": "openbmi",
          "edlow17": "openbmi", "mbmi": "openbmi", "mbvt": "erp", "chennu13": "erp", "gibson16": "erp"}
ID_PATTERNS = (re.compile(r"Bonobo\d+"), re.compile(r"Patient_\d+"), re.compile(r"sub-\d+"))

STRATUM_RULES = {
    "icu_typical": {"source": "bdsp", "label": "ICU-typical background (Morgoth burst suppression, IIIC/periodic or "
                    "diffuse slowing; non-transitioning, A9-clean)"},
    "awake": {"source": "bdsp", "label": "model-assigned awake background (Morgoth normal_awake; non-transitioning, "
              "A9-clean)"},
    "pdoc_wake_like": {"source": "pdoc", "label": "pDoC lower-delta (wake-like) extent (C3/C4 log delta/alpha below "
                       "the recording median; A9-clean)"}}


SOURCES = {
    "NEJM19": {"doi": "10.1056/NEJMoa1812757", "cite": "Claassen J, et al. N Engl J Med 2019;380:2497-505"},
    "CU18": {"doi": "10.1093/brain/awy070", "cite": "Curley WH, et al. Brain 2018;141:1404-21"},
    "CR11": {"doi": "10.1016/S0140-6736(11)61224-5", "cite": "Cruse D, et al. Lancet 2011;378:2088-94"},
    "GF13": {"doi": "10.1016/S0140-6736(13)60125-7", "cite": "Goldfine AM, et al. Lancet 2013;381:289-91"},
    "CR12P": {"doi": "10.1371/journal.pone.0049933", "cite": "Cruse D, et al. PLoS One 2012;7:e49933"},
    "DB26": {"doi": "10.1038/s43856-026-01574-x", "cite": "du Bois N, et al. Commun Med 2026;6:344"},
    "ED17": {"doi": "10.1093/brain/awx176", "cite": "Edlow BL, et al. Brain 2017;140:2399-414"},
    "GU17": {"doi": "10.3389/fnins.2017.00251", "cite": "Guger C, et al. Front Neurosci 2017;11:251"},
    "CH13": {"doi": "10.1016/j.nicl.2013.10.008", "cite": "Chennu S, et al. NeuroImage Clin 2013;3:450-61"},
    "GIB16": {"doi": "10.1002/ana.24726", "cite": "Gibson RM, et al. Ann Neurol 2016;80:412-23"}}

PUBLISHED = {
    "columbia": {"benchmarks": [
        {"source": "NEJM19", "kind": "design", "rate": 0.05, "closest": ["icu_typical", "awake"],
         "what": "the stated false-positive rate per recording of the permutation test", "where": "NEJM19-SA p. 8",
         "quote": "we expect 5% of recordings to pass the statistical threshold of significance by chance"},
        {"source": "NEJM19", "kind": "null", "k": 12, "n": 240, "closest": ["icu_typical"],
         "what": "recordings significantly BELOW chance, the mirror image of the one-tailed null [derived 5.0%]",
         "where": "NEJM19-SA p. 13", "quote": "12 of the 240 recordings in our study were significantly below chance"}],
        "differences": "Same code (the authors' public repository); their recordings are acute ICU patients, ours "
                       "BDSP clinical EEG at 19 channels (no pDoC: the 6-channel montage cannot run it)."},
    "cornell": {"benchmarks": [
        {"source": "CU18", "kind": "null", "k": 2, "n": 26, "closest": ["pdoc_wake_like", "icu_typical"],
         "what": "surrogate datasets (epochs exchanged within a trial, one surrogate per dataset) called positive",
         "where": "CU18 p. 1412", "quote": "a positive outcome in 2 (7.7%) of 26 of patient 'tennis' datasets"}],
        "differences": "CU18: 37 electrodes, visual pruning of 56-65% of epochs, sleep screening by two "
                       "neurophysiologists, the tennis paradigm only. Ours: unpruned 19-channel BDSP or 6-channel "
                       "pDoC windows; C1 atlas: cornell's calls rise with slow amplitude fluctuation."},
    "cruse11": {"benchmarks": [
        {"source": "CR11", "kind": "null", "k": 0, "n": 12, "closest": ["awake"],
         "what": "healthy controls told to mind-wander (no command followed) called positive", "where": "CR11 p. 4",
         "quote": "no healthy control participant returned EEG responses that could be classified significant"},
        {"source": "CR11", "kind": "upper_bound", "k": 3, "n": 16, "closest": ["pdoc_wake_like"],
         "what": "VS patients called positive (true and false calls together); GF13's block-aware reanalysis left "
                 "none significant after FDR", "where": "CR11 p. 4; GF13 main text",
         "quote": "none of the patients were significant at p<=0.05 (GF13)"}],
        "differences": "CR11: 25 channels, healthy volunteers awake and attending; its 4-block native variant fits "
                       "few BDSP windows (26 recordings; selection on 24)."},
    "cruse12p": {"benchmarks": [
        {"source": "CR12P", "kind": "null", "k": 0, "n": 6, "closest": ["awake"],
         "what": "healthy controls told to mind-wander called positive", "where": "CR12P Results, single-trial",
         "quote": "no healthy controls returned significant classification in any comparison"}],
        "differences": "CR12P: 4 electrodes, healthy volunteers awake; one patient."},
    "dubois": {"benchmarks": [
        {"source": "DB26", "kind": "upper_bound", "k": 138, "n": 293, "closest": ["pdoc_wake_like"],
         "what": "significant runs in UWS patients who progressed (true and false calls; mostly training and "
                 "feedback runs)", "where": "DB26 Fig. 5", "quote": "UWS 138/293"},
        {"source": "DB26", "kind": "upper_bound", "k": 31, "n": 42, "closest": ["pdoc_wake_like"],
         "what": "patients passing Phase I (true and false calls; several Phase-I runs each)",
         "where": "DB26 Results", "quote": "73.8% of patients passed assessment"}],
        "differences": "No sham or no-command data published; the significant-run fractions bound the null from "
                       "above only. DB26 runs are mostly training/feedback; ours the assessment run type."},
    "edlow17": {"benchmarks": [
        {"source": "ED17", "kind": "null", "k": 0, "n": 4, "closest": ["icu_typical"],
         "what": "acute TBI patients without behavioural language called positive (FPR against behaviour)",
         "where": "ED17 p. 2406", "quote": "none had EEG response to motor imagery (TNR = 4/4; FPR = 0/4)"}],
        "differences": "ED17: acute TBI in the ICU, 4 patients, ICA plus consensus review; no no-command null."},
    "mbmi": {"benchmarks": [
        {"source": "GU17", "kind": "null", "k": 0, "n": 3, "closest": ["awake"],
         "what": "healthy sham runs (system muted) above the 63.19% threshold", "where": "GU17 Table 3",
         "quote": "near chance level (below the significance level) for the sham run"}],
        "differences": "GU17: 3 healthy volunteers, one sham run each; mindBEAGLE montage."},
    "mbvt": {"benchmarks": [
        {"source": "GU17", "kind": "null", "k": 0, "n": 3, "closest": ["awake"],
         "what": "healthy VT3 sham runs (system muted) above 23% (VT2 shams: 1/3, 25%)", "where": "GU17 Table 3",
         "quote": "0 to 25% for sham runs"}],
        "differences": "GU17: 3 healthy volunteers, one sham run each; the n behind 23% is not stated."},
    "chennu13": {"benchmarks": [
        {"source": "CH13", "kind": "upper_bound", "k": 4, "n": 21, "closest": ["pdoc_wake_like"],
         "what": "patients with significant ERP responses to targets or distractors (true and false calls)",
         "where": "CH13 Results, individual results",
         "quote": "in all but 4 patients, targets and distractors did not generate statistically significant ERP "
                  "responses"}],
        "differences": "CH13: 91 channels, DoC patients; no null data published."},
    "gibson16": {"benchmarks": [
        {"source": "GIB16", "kind": "upper_bound", "k": 0, "n": 14, "closest": ["pdoc_wake_like"],
         "what": "patients with a top-down P3b (true and false calls); P3a was found in 8 of 14",
         "where": "GIB16 AAM p. 9", "quote": "Top-down ERP attention effects ... were not detected from any of the "
                                             "patients"}],
        "differences": "GIB16: 12 electrodes, DoC patients; no no-stimulus null. The pDoC validity primary "
                       "(p3a_onepos) is a bottom-up P3a call."}}



def rate(d, col="call", B=B_BOOT, seed=None, idx=None):
    """Per-patient ratio of col over the rows where it is not NA (counted), patient-bootstrap 95% CI. idx: (B, n)
    patient draws (115.boot_idx when None; each drawn copy is its own unit)."""
    v = d[col] if col in d else pd.Series(dtype=float)
    ok = v.notna()
    e = d[ok].assign(_c=v[ok].astype(float))
    out = {"k": int(e["_c"].sum()), "n": int(len(e)), "n_patients": int(e["patient"].nunique()) if len(e) else 0,
           "n_na": int((~ok).sum()), "rate": None, "ci95": None}
    if e.empty:
        return out
    s, n = mg._per_patient(e, "_c")
    idx = mg.boot_idx(len(s), B, seed) if idx is None else np.asarray(idx)
    draws = mg.ratio_draws(s, n, idx)
    out.update(rate=float(s.sum() / n.sum()), ci95=mg._ci(draws), draws=[float(x) for x in draws])
    return out


def _public(r):
    return {k: v for k, v in r.items() if k != "draws"}



def stratum_mask(d, name):
    """116's A1 rule for the stratum: BDSP awake / ICU-typical on non-transitioning, A9-clean windows; pDoC lower-delta
    on A9-clean windows (an A9 flag that is NA counts as flagged)."""
    tr = d["transitioning"].fillna(False).astype(bool) if "transitioning" in d else pd.Series(False, index=d.index)
    clean = ~d["a9_flag"].isna() & ~d["a9_flag"].fillna(True).astype(bool) & ~tr
    if name == "pdoc_wake_like":
        return clean & d["lower_delta"].fillna(False).astype(bool)
    st = d["stratum"].astype(str)
    if name == "awake":
        return clean & (st == "normal_awake")
    if name == "icu_typical":
        return clean & st.isin(DISORGANISED)
    raise ValueError(f"unknown stratum {name}")



def readings_for(pipe, validity_primary, default):
    """The reading(s) of a pipeline on one null set: C3 where C0 grades the default a LIBERAL DEPARTURE, the default
    where it is a FAITHFUL LITERAL READING, else the validity primary. sokoliuk21 is refused."""
    if pipe in LEFT_OUT:
        raise ValueError(f"{pipe}: {LEFT_OUT[pipe]}")
    if C0.get(pipe) == "LIBERAL DEPARTURE":
        return [{"pipe": pipe, "kind": "c3", "reading": r, "variant": fid.base(pipe, r), "grade": C3_GRADE}
                for r in fid.readings(pipe)]
    if C0.get(pipe) == "FAITHFUL LITERAL READING":
        return [{"pipe": pipe, "kind": "default", "reading": default, "variant": default, "grade": C0[pipe]}]
    return [{"pipe": pipe, "kind": "validity_primary", "reading": validity_primary, "variant": validity_primary,
             "grade": UNGRADED}]



def load_first_overlays(root=SHARDS, split=SPLIT):
    """First-overlay rows (j = 0) of the null sets' native resp-none shard directories, any status: 115.load_rows
    restricted to them (the first row of a key within a directory kept; a row of another split, or a development
    recording among analysis rows, refused, checked on every row)."""
    frames = []
    for nul in NULLS["bdsp"] + NULLS["pdoc"]:
        for d in sorted(glob.glob(os.path.join(root, f"{nul}__*__native__none"))):
            seen, rows = set(), []
            for fn in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
                with open(fn) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        if f'"split": "{split}"' not in line:
                            raise RuntimeError(f"a row of another split under the {split} shards ({fn})")
                        if split != "dev" and '"dev": false' not in line:
                            raise RuntimeError(f"a development-recording row among {split} rows ({fn})")
                        if '"j": 0,' not in line:
                            continue
                        r = json.loads(line)
                        if r["key"] in seen:
                            continue
                        seen.add(r["key"])
                        rows.append(r)
            if rows:
                frames.append(pd.DataFrame(rows).assign(dir=os.path.basename(d)))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def analysis_rows(root=SHARDS):
    """116.load_rows on the first overlays: touched recordings dropped, Columbia's BDSP first overlays from 113."""
    df, _ = mg.drop_touched(load_first_overlays(root))
    c = rn.columbia113()
    col = (df["null"] == "ncmd_bdsp") & (df["pipe"] == "columbia") & (df["resp"] == "none") & (df["j"] == 0)
    mg.check_113(c, df[~col], mg.dev_uids(), mg.touched_uids())
    return pd.concat([df[~col], mg.rows_from_113(c)], ignore_index=True, sort=False)


def published_rows(df):
    """115._null_seq first overlays (status ok, resp none, native, seq, not overlapping), the rows of 116's A1."""
    d = mg._null_seq(df)
    return d[d["j"] == 0]


def c3_rows(root=C3_SHARDS):
    rows = []
    for fn in sorted(glob.glob(os.path.join(root, "*__*", "rows*.jsonl"))):
        with open(fn) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return pd.DataFrame(rows)


def c3_on_base(c3, base):
    """C3 first-overlay rows (status ok) on their base variant's published row of the same recording: the C3 call
    with the base row's labels, extent, A9 flag, lower-delta flag and randomised counterpart. A C3 row whose window
    start differs from its base row's is refused."""
    c = c3[c3["j"] == 0]
    ok = c[c["status"] == "ok"]
    b = base.drop(columns=[x for x in ("call", "p", "stat", "status", "key", "seconds", "error", "reading")
                           if x in base]).rename(columns={"variant": "base", "start_s": "start_s_base"})
    m = ok.merge(b, on=["null", "pipe", "uid", "patient", "j", "base"], how="inner", validate="one_to_one")
    if (m["start_s"].astype(float) != m["start_s_base"].astype(float)).any():
        raise RuntimeError("a C3 row lies on another window than its base variant's row: refused")
    m = m.assign(variant=m["base"])
    counts = {"c3_rows_ok": int(len(ok)), "c3_rows_not_ok": int(len(c) - len(ok)),
              "c3_rows_without_base_row": int(len(ok) - len(m))}
    return m.reset_index(drop=True), counts


def randomised(d):
    """The randomised counterpart on its own window: call = rand_call, labels from rand_stratum / rand_transitioning,
    extent from rand_extent_* (113's rows: the published window, its test iii runs there), A9 and lower-delta flags
    from rand_a9_flag / rand_lower_delta when attached. Rows without a randomised call are dropped and counted."""
    has = d["rand_call"].notna() if "rand_call" in d else pd.Series(False, index=d.index)
    r = d[has]
    out = pd.DataFrame({"uid": r["uid"], "patient": r["patient"], "call": r["rand_call"].astype(bool)}, index=r.index)
    for c in ("stratum", "transitioning", "extent_t0", "extent_t1"):
        own = r["rand_" + c] if "rand_" + c in r else pd.Series(np.nan, index=r.index, dtype=object)
        out[c] = own.where(own.notna(), r[c]) if c in r else own
    for c in ("a9_flag", "lower_delta"):
        if "rand_" + c in r:
            out[c] = r["rand_" + c]
    return out, int((~has).sum())


A1_NAME = {"icu_typical": "icu_typical", "awake": "awake", "pdoc_wake_like": "lower_delta"}


def cell(rows, stratum, B=B_BOOT, seed_parts=(), reading_seed=None):
    """One table cell: the reading's rate and its randomised counterpart's on the same stratum (each on its own
    windows' labels). reading_seed: 116's A1 seed for a validity-primary or default cell, so the cell equals 116's."""
    keep = rows[stratum_mask(rows, stratum)]
    rd, n_no_rand = randomised(rows)
    rkeep = rd[stratum_mask(rd, stratum)]
    seed = reading_seed or mg._seed("143", *seed_parts, stratum, "reading")
    return {"reading": _public(rate(keep, "call", B, seed)),
            "randomised": _public(rate(rkeep, "call", B, mg._seed("143", *seed_parts, stratum, "randomised"))),
            "n_rows_without_randomised_call": n_no_rand}



def a9(d, source, t0="extent_t0", t1="extent_t1", tables=None):
    """116.a9_flags on the given extent columns (121's tables of the source and split)."""
    act = importlib.import_module("121_activation")
    iv, recs = tables or act.load(source, SPLIT)
    e = pd.DataFrame({"uid": d["uid"].astype(str), "extent_t0": d[t0].astype(float), "extent_t1": d[t1].astype(float)},
                     index=d.index)
    return mech.a9_flags(e, iv, recs)["a9_flag"]


def lower_delta_flags(d, extents):
    """116.RecordingX.lower_delta on each row's extent, for every (name, t0 column, t1 column) in extents; each pDoC
    recording of the analysis split is loaded once."""
    items = {it["uid"]: it for it in en._items("pdoc", SPLIT)}
    out = {name: pd.Series(None, index=d.index, dtype=object) for name, _, _ in extents}
    for uid, g in d.groupby("uid"):
        rec = en._load(items[uid])
        rn.check_member(rec, SPLIT)
        rx = mech.RecordingX(rec, None)
        for name, t0, t1 in extents:
            for i, a, b in zip(g.index, g[t0].astype(float), g[t1].astype(float)):
                out[name][i] = rx.lower_delta(a, b)
        del rec, rx
    return out


def same_flags(a, b):
    """Rows where two flag columns differ (both NA counts as equal)."""
    na_a, na_b = a.isna(), b.isna()
    return (na_a != na_b) | (~na_a & ~na_b & (a.astype(object) != b.astype(object)))



def _pc_cell(cells, key):
    c = cells.get(key)
    return None if c is None else {"k": c["k"], "n": c["n"], "rate": c["rate"], "ci95": c["ci"]}


def positive_control(pc, pipe, variant, group, exact=True):
    """A5 primary-arm detection of the variant and of its randomised group (true version)."""
    arm = PC_ARM.get(pipe)
    if arm is None or f"{arm}|{pipe}" not in pc["headline"]:
        return {"arm": None, "variant": variant, "published": None, "randomised": None, "exact_reading": exact,
                "note": "no reuse-free primary arm (A5: reuse-confounded, never passed or failed)"}
    out = {"arm": arm, "variant": variant, "group": group, "exact_reading": exact,
           "published": _pc_cell(pc["cells"], f"{arm}|{pipe}|published|{variant}|true"),
           "randomised": _pc_cell(pc["cells"], f"{arm}|{pipe}|randomized|{group}|true")}
    notes = [] if exact else ["the C3 reading was never run in A5: its base variant shown (nearest variant)"]
    if out["published"] is None:
        notes.append("variant not run in A5")
    if out["randomised"] is None:
        notes.append("randomised group not run in A5")
    out["note"] = "; ".join(notes) or None
    return out



def clopper_pearson(k, n, level=0.95):
    """Exact (Clopper-Pearson) two-sided interval of k calls in n; None when n = 0."""
    if not 0 <= k <= n:
        raise ValueError(f"k = {k} outside 0..n = {n}")
    if n == 0:
        return None
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - a, k + 1, n - k))
    return [lo, hi]


_cp = clopper_pearson


def binom_vs_nominal(k, n, p0=mg.NOMINAL):
    """Exact one-sided binomial p of k or more calls in n at the nominal rate (115.binom_greater)."""
    return mg.binom_greater(k, n, p0)


def holm(p):
    """Holm step-down adjusted p-values in input order (115.holm)."""
    return mg.holm(p)


def exact_cell(k, n, p0=mg.NOMINAL):
    """A count with its rate, exact 95% interval and exact one-sided p against the nominal rate."""
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "ci95_cp": clopper_pearson(k, n),
            "p_greater_nominal": binom_vs_nominal(k, n, p0)}


def consistency(ours, bm, alpha=0.05):
    """Our count (k of n windows) against a published number, by exact tests (windows as units; a bootstrap interval
    is degenerate at 0 calls). null: Fisher's exact two-sided test, inside if p >= alpha, else above or below; also
    P(at most the published count | our rate). upper_bound (positives among patients, true and false calls
    together, so it bounds a null rate from above only): Fisher's exact one-sided test of ours > theirs. design: exact
    binomial test against the stated nominal rate."""
    if ours.get("rate") is None:
        return {"verdict": "no data"}
    k0, n0 = ours["k"], ours["n"]
    side = lambda p, hi: "inside" if p >= alpha else ("above" if hi else "below")
    if bm["kind"] == "design":
        p = float(stats.binomtest(k0, n0, bm["rate"]).pvalue)
        return {"nominal": bm["rate"], "p_binomial": p, "verdict": side(p, k0 / n0 > bm["rate"])}
    k, n = bm["k"], bm["n"]
    out = {"published": f"{k}/{n}", "published_rate": k / n, "published_ci95": _cp(k, n),
           "p_at_most_k_at_our_rate": float(stats.binom.cdf(k, n, ours["rate"]))}
    tab = [[k0, n0 - k0], [k, n - k]]
    if bm["kind"] == "null":
        out["p_fisher"] = float(stats.fisher_exact(tab)[1])
        out["verdict"] = side(out["p_fisher"], k0 / n0 > k / n)
    else:
        out["p_fisher_greater"] = float(stats.fisher_exact(tab, alternative="greater")[1])
        out["verdict"] = ("above the published positive fraction" if out["p_fisher_greater"] < alpha else
                          "not above the published positive fraction")
    return out



def assert_no_identifiers(obj):
    """Refuse an output holding a recording or patient identifier (key or value)."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(str(k))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for p in ID_PATTERNS:
                if p.search(o):
                    raise RuntimeError(f"identifier-like string in the output ({p.pattern}): refused")
    walk(obj)



def check_116(rows_by_role, mech_out, B):
    """116's A1a (BDSP awake, ICU-typical) and A1c (pDoC lower-delta) rates for the validity primary and the
    default, recomputed from these rows with 116's _rate_ci and seeds: n, patients, rate and CI must match."""
    bad, n = [], 0
    a = mech_out["part_A"]
    for block, masks in (("A1a_awake", ("awake", "icu_typical")), ("A1c_pdoc_lower_delta", ("lower_delta",))):
        for pipe, e in a[block].items():
            if pipe in LEFT_OUT:
                continue
            nul = e["null"]
            for m in masks:
                for role in ("validity_primary", "default"):
                    want = e[m][role]
                    d = rows_by_role.get((nul, pipe, want["variant"]))
                    if d is None:
                        bad.append(f"{block}|{pipe}|{m}|{role}: no rows")
                        continue
                    sm = "pdoc_wake_like" if m == "lower_delta" else m
                    got = mech._rate_ci(d[stratum_mask(d, sm)], B, mech._seed(f"a1|{nul}|{pipe}|{m}|{role}"))
                    n += 1
                    same = (got["n"], got["n_patients"]) == (want["n"], want["n_patients"]) and (
                        (got["rate"] is None and want["rate"] is None) or
                        (np.isclose(got["rate"], want["rate"], atol=1e-12, rtol=0) and
                         np.allclose(got["ci95"], want["ci95"], atol=1e-12, rtol=0)))
                    if not same:
                        bad.append(f"{block}|{pipe}|{m}|{role}: {got} vs {want}")
    if bad:
        raise RuntimeError(f"116 A1 not reproduced ({len(bad)} of {n}): " + "; ".join(bad[:5]))
    return {"cells_reproduced": n}


def check_c3(c3m, c3_out):
    """The C3 first-overlay counts of c3_fidelity.json (k / n per null set and reading)."""
    bad, n = [], 0
    for nul in NULLS["bdsp"] + NULLS["pdoc"]:
        for pipe, e in c3_out.get(nul, {}).items():
            for reading in fid.readings(pipe):
                fo = e[reading]["first_overlay"]
                d = c3m[(c3m["null"] == nul) & (c3m["reading"] == reading)]
                got = (int(d["call"].astype(bool).sum()), int(len(d)), int(d["patient"].nunique()))
                n += 1
                if got != (fo["calls"], fo["n"], fo["patients"]):
                    bad.append(f"{nul}|{reading}: {got} vs {(fo['calls'], fo['n'], fo['patients'])}")
    if bad or not n:
        raise RuntimeError(f"C3 counts not reproduced ({n} checked): {bad}")
    return {"cells_reproduced": n}


def cells_vs_116(table, mech_out):
    """Every validity-primary or default cell whose variant is 116's for that role equals 116's A1 cell (n, rate,
    CI); the cells whose variant differs are listed."""
    a = mech_out["part_A"]
    same, other = 0, []
    for e in table:
        if e["kind"] not in ("validity_primary", "default"):
            continue
        for s, c in e["cells"].items():
            if c is None:
                continue
            block = "A1c_pdoc_lower_delta" if s == "pdoc_wake_like" else "A1a_awake"
            want = a[block][e["pipe"]][A1_NAME[s]][e["kind"]]
            if want["variant"] != c["variant"]:
                other.append(f"{c['null']}|{e['pipe']}|{s}: {c['variant']} (116: {want['variant']})")
                continue
            got = c["reading"]
            if (got["n"], got["n_patients"]) != (want["n"], want["n_patients"]) or got["rate"] != want["rate"] or \
                    got["ci95"] != want["ci95"]:
                raise RuntimeError(f"{e['pipe']}|{s}: {got} differs from 116's {want}")
            same += 1
    return {"equal": same, "variant_differs_from_116": other}



def choices_from_headline():
    """{(null, pipe): (validity primary, default)} from the 115 headline merge outputs."""
    out = {}
    for nul in NULLS["bdsp"] + NULLS["pdoc"]:
        with open(os.path.join(TABLES, f"benchmark_{nul}.json")) as fh:
            pub = json.load(fh)["published"]
        for pipe, e in pub.items():
            src = "pdoc" if nul in NULLS["pdoc"] else "bdsp"
            out[(nul, pipe)] = (e["validity_primary"], rn.primary_variant(rn.load_pipe(pipe), src))
    return out


def null_of(pipe, source):
    fam = rn.load_pipe(pipe).FAMILY
    return NULLS[source][0 if fam == "cmd" else 1]


def build(rows_by_variant, c3_by_reading, choices, pc, B=B_BOOT):
    """The table: per pipeline and reading, the three strata (reading and randomised counterpart), the positive
    control and the published-benchmark checks."""
    table = []
    for pipe in CMD_PIPES:
        per_source = {}
        for source in ("bdsp", "pdoc"):
            nul = null_of(pipe, source)
            if (nul, pipe) not in choices:
                continue
            per_source[source] = (nul, readings_for(pipe, *choices[(nul, pipe)]))
        names = []
        for source, (nul, rs) in per_source.items():
            names += [r["reading"] if r["kind"] == "c3" else None for r in rs]
        c3_names = [r for r in dict.fromkeys(names) if r]
        groups = c3_names or [None]
        for g in groups:
            entry = {"pipe": pipe, "cells": {}, "positive_control": {}, "published": []}
            for source, (nul, rs) in per_source.items():
                r = next(x for x in rs if g is None or x["reading"] == g)
                d = c3_by_reading.get((nul, r["reading"])) if r["kind"] == "c3" else \
                    rows_by_variant.get((nul, pipe, r["variant"]))
                entry.update(kind=r["kind"], grade=r["grade"], reading=r["reading"])
                entry.setdefault("variants", {})[source] = r["variant"]
                for s, rule in STRATUM_RULES.items():
                    if rule["source"] != source:
                        continue
                    a1 = mech._seed(f"a1|{nul}|{pipe}|{A1_NAME[s]}|{r['kind']}") \
                        if r["kind"] in ("validity_primary", "default") else None
                    c = cell(d, s, B, (nul, pipe, r["reading"]), a1) if d is not None else None
                    if c is not None:
                        c.update(null=nul, variant=r["variant"])
                    entry["cells"][s] = c
                group = None
                if d is not None and "rand_variant" in d and d["rand_variant"].notna().any():
                    group = d["rand_variant"].dropna().astype(str).mode().iloc[0]
                entry["positive_control"][source] = positive_control(pc, pipe, r["variant"], group,
                                                                     exact=r["kind"] != "c3")
            for bm in PUBLISHED[pipe]["benchmarks"]:
                e = {k: v for k, v in bm.items()}
                e.update(doi=SOURCES[bm["source"]]["doi"], cite=SOURCES[bm["source"]]["cite"], checks={})
                for s, c in entry["cells"].items():
                    e["checks"][s] = dict(consistency(c["reading"], bm) if c else {"verdict": "no data"},
                                          closest_population=s in bm["closest"])
                entry["published"].append(e)
            if entry.get("kind") != "c3":
                vs = entry.get("variants", {})
                entry["reading"] = (next(iter(vs.values())) if len(set(vs.values())) == 1 else
                                    "; ".join(f"{s}: {v}" for s, v in vs.items()))
            entry["differences"] = PUBLISHED[pipe]["differences"]
            table.append(entry)
    return table


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--B", type=int, default=B_BOOT)
    a = ap.parse_args(argv)
    with open(MECH_FILE) as fh:
        mech_out = json.load(fh)
    with open(C3_FILE) as fh:
        c3_out = json.load(fh)
    with open(PC_FILE) as fh:
        pc = json.load(fh)
    choices = choices_from_headline()
    vc116 = mech_out["part_A"]["variant_choices"]
    differ = {"|".join(k): {"headline": {"validity_primary": vp, "default": dflt},
                            "116": {"validity_primary": vc116["|".join(k)]["validity_primary"],
                                    "default": vc116["|".join(k)]["default"]}}
              for k, (vp, dflt) in choices.items() if "|".join(k) in vc116 and
              (vc116["|".join(k)]["validity_primary"], vc116["|".join(k)]["default"]) != (vp, dflt)}
    df = analysis_rows()
    pub = published_rows(df)
    res = mech.read_results(mech.paths(SPLIT)["results"])
    res = res[res["kind"] == "published"]
    keep = [c for c in ("key", "pipe", "lower_delta", "extent_t0_116", "extent_t1_116") if c in res]
    pub = pub.merge(res[keep], on=["key", "pipe"], how="left", validate="one_to_one")
    pub = mech._with_extent(pub)
    want = set()
    for (nul, pipe), (vp, dflt) in choices.items():
        if pipe in LEFT_OUT:
            continue
        want |= {(nul, pipe, vp), (nul, pipe, dflt)} | {(nul, pipe, r["variant"]) for r in
                                                       readings_for(pipe, vp, dflt)}
    want |= {(k.split("|")[0], k.split("|")[1], v[role]) for k, v in vc116.items() if k.split("|")[1] not in LEFT_OUT
             for role in ("validity_primary", "default")}
    sel = pub[[(n, p, v) in want for n, p, v in zip(pub["null"], pub["pipe"], pub["variant"])]].copy()
    tables = {s: importlib.import_module("121_activation").load(s, SPLIT) for s in ("bdsp", "pdoc")}
    src = np.where(sel["null"].isin(NULLS["pdoc"]), "pdoc", "bdsp")
    sel["a9_flag"], sel["rand_a9_flag"] = np.nan, np.nan
    sel["a9_flag"] = sel["a9_flag"].astype(object)
    sel["rand_a9_flag"] = sel["rand_a9_flag"].astype(object)
    for s in ("bdsp", "pdoc"):
        m = src == s
        sel.loc[m, "a9_flag"] = a9(sel[m], s, tables=tables[s])
        r = sel[m].copy()
        for c in ("extent_t0", "extent_t1"):
            own = r["rand_" + c] if "rand_" + c in r else pd.Series(np.nan, index=r.index)
            r["rand_" + c] = own.astype(float).fillna(r[c].astype(float))
        sel.loc[m, "rand_a9_flag"] = a9(r, s, "rand_extent_t0", "rand_extent_t1", tables=tables[s])
        if s == "pdoc":
            f = lower_delta_flags(r, (("check", "extent_t0_116", "extent_t1_116"),
                                      ("rand", "rand_extent_t0", "rand_extent_t1")))
            sel["ld_check"] = pd.Series(None, index=sel.index, dtype=object)
            sel["rand_lower_delta"] = pd.Series(None, index=sel.index, dtype=object)
            sel.loc[m, "ld_check"], sel.loc[m, "rand_lower_delta"] = f["check"], f["rand"]
    pd_rows = sel[src == "pdoc"]
    mism = int(same_flags(pd_rows["ld_check"], pd_rows["lower_delta"]).sum())
    if mism:
        raise RuntimeError(f"recomputed lower-delta flag differs from 116's on {mism} pDoC rows: refused")
    own_vs_116 = int(((pd_rows["extent_t0"].astype(float) != pd_rows["extent_t0_116"].astype(float)) |
                      (pd_rows["extent_t1"].astype(float) != pd_rows["extent_t1_116"].astype(float))).sum())
    rows_by_variant = {k: g for k, g in sel.groupby(["null", "pipe", "variant"])}
    checks = {"a1_116": check_116(rows_by_variant, mech_out, 2000),
              "variant_choices_116_vs_headline": differ,
              "lower_delta_recomputed": {"rows": int(len(pd_rows)), "equal_to_116": True,
                                         "rows_whose_extent_differs_from_116s_rebuilt_extent": own_vs_116}}
    c3m, c3_counts = c3_on_base(c3_rows(), sel[sel["pipe"].isin(list(fid.READINGS))])
    checks["c3_fidelity"] = check_c3(c3m, c3_out)
    checks["c3_rows"] = c3_counts
    c3_by_reading = {k: g for k, g in c3m.groupby(["null", "reading"])}
    table = build(rows_by_variant, c3_by_reading, choices, pc, a.B)
    checks["cells_equal_to_116"] = cells_vs_116(table, mech_out)
    out = {"what": __doc__.split("\n")[0], "split": SPLIT, "B": a.B, "sha256_143": _sha256(os.path.abspath(__file__)),
           "inputs": {os.path.relpath(p, ROOT): _sha256(p) for p in (MECH_FILE, C3_FILE, PC_FILE) +
                      tuple(os.path.join(TABLES, f"benchmark_{n}.json") for n in NULLS["bdsp"] + NULLS["pdoc"])},
           "strata": STRATUM_RULES, "left_out": LEFT_OUT, "sources": SOURCES, "checks": checks, "table": table}
    out = mg._jsonable(out)
    assert_no_identifiers(out)
    os.makedirs(TABLES, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"stratum table: {len(table)} rows; checks {checks} -> {OUT}")
    return out


if __name__ == "__main__":
    main()
