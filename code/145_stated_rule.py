"""Stated-rule headline (added later, labelled so wherever reported): the published pipelines read with every analysis
choice the papers state, choosing only among variants that differ in items the papers leave NOT STATED. Built
test-first (code/test_145_stated_rule.py). No new run: it reads the 115 analysis rows, 130's C3 rows, Columbia's
call_paper field (113 / 114 rows), 116's X results, 121's A9 tables and the eligibility files
docs/stated-rule-motor.json and docs/stated-rule-erp.json.
- Rows: the first overlay (115._null_seq, j = 0), one row per patient: per null set and pipeline, the patient's
  first recording (sorted uid) on which the pipeline has a first-overlay row (no analysis patient has two today;
  dropped rows are counted). Null sets ncmd_bdsp, nstim0_bdsp, ncmd_pdoc, nstim0_pdoc. sokoliuk21 is left out (C0:
  NOT A CALL RULE) gets a separate exceedance-only row, labelled covert processing, outside Holm and outside the
  count of call-rule analyses.
- Eligible set per pipeline: stated_rule_eligible true, minus cannot-call (115.cannot_call_of) and minus coverage
  below 70% (115.coverage_of / low_coverage, from the rows exactly as 115 computes them). With the second blind
  classifier in, stated_rule_eligible is the union of the two classifications (the primary, pre-declared) and
  stated_rule_eligible_strict marks the variants both classifiers admit: the flag must be on every variant or none,
  a strict variant outside the union is refused, and a strict variant (or, without strict flags, any eligible one)
  with a CONTRADICTS or SHORTENED_STRUCTURAL axis is refused; a shortened variant is never eligible. Variants with
  shortened_to_fit true form a separate "shortened to fit" row and never enter the headline or Holm. Eligibility
  reads the flags alone (constant departures never affect it).
- Stated call rule (CALL_PRINCIPLE, verbatim in the output; CALL_AXES: the axes that set each pipeline's call
  criterion or threshold, decision): a variant has a stated call rule only when every call axis it has is STATED
  in the axis classes of CLASS_REF (9a94bcc, the first blind classification, read from git; the classes changed
  since are listed in the output, a changed variant or axis list is refused); a pipeline whose call is common to
  every variant has it stated for all.
- Three readings, each with the full set below: the primary ("stated rule") = the selection among the union-eligible
  variants with a stated call rule, or, where none has one, among all union-eligible variants, labelled with the
  call status ("call rule not stated" / "call rule not fully stated": most favourable reading); the sensitivity
  "sensitivity: most favourable reading incl. unstated call rules" = all union-eligible variants; the sensitivity
  "sensitivity: strict (both blind classifiers agree)" = the primary's rule on the strict-eligible variants.
  Pseudo-variants: the C3 readings (cornell c3_jackknife_bh_keep40_r4_om2contig; du Bois c3_b_assessment_{random,
  blocked}_phase1) on their base variant's windows, and Columbia's as_coded:call_paper (AUC > 0.5 and p < 0.05).
- Each reading = 115's own selection (_select_primary: the fewest false calls on the recordings where every
  candidate ran; ties as coded, then name). Reported: its k/n on all its recordings, exact Clopper-Pearson
  95% CI, one-sided exact binomial p against 5%, Holm across pipelines within each BDSP null set (pDoC: estimation
  only, no inflated label); the cross-fit rate (select on one crc32(patient) parity half, 115._half, estimate on the
  other, swap, average; patient-bootstrap CI, B = 2000, the selection redone in every draw); the eligible range and
  its count; beside it the fixed min-rule validity primary (115 headline merge) and the C0 default.
  Held-out (valid) test, beside the full-data p: the pooled cross-fit count (each half's calls of the variant
  selected on the other half, a sum over disjoint patients), exact CP CI and one-sided exact binomial p against 5%,
  Holm across the call-rule pipelines within each BDSP null set. Each endpoint of the eligible range names its
  variant(s) with an exact CP CI, labelled "min / max over eligible readings (selected, not an estimate)".
  Deflation diagnostic for the selected variant and each endpoint: the recording-level accuracy-type statistic
  (ACCURACY_STAT: AUC, CV / peak decoding / 8-way accuracy) against its chance value, median, IQR and the share
  below chance; "no accuracy-type statistic" where the module returns none. Holm runs within each reading. With
  no eligible variant on the null set's rows the note gives the reason from the classification.
- Sensitivity: a sensitivity_eligible set is reported separately only if it names its reclassified_axes and is not
  contained in the main eligible set; one contained in it is reported as retired (identical to, or a subset of,
  main). No eligible count or variant name is fixed in the code: everything comes from the files.
- Per patient at the published k (docs/benchmark-k-table.json), on the calls of the variant the stated-rule
  selection picked (a frozen variant, a C3 reading or call_paper), exact CP CIs: Columbia's k = 2 bracket
  (115.columbia_k2_bracket's two bounds; on call_paper with the papers' AUC > 0.5 on the rejected recording); du
  Bois k = 2 (Phase I; NOT AVAILABLE only if no patient has two windows of the selected variant); mbvt k = 2 = the
  first two runs of the selected variant, with a note when the eligible set holds one paradigm only (the GUG18
  session pairs one VT2 and one VT3 run), and the descriptive best-of-k curve, read up to the largest published k
  (--mbvt-published-k-max, never below the k table's k; added to the k grid), each larger k marked "beyond
  published k"; every other pipeline its k (1).
  Composites use 115.patient_outcomes / composite_curve / _rule unchanged (each patient's first k usable windows:
  the k lowest overlay indices, a seeded random order, not the k earliest; FIRST_K in the output);
  the k = 1 composite is checked patient by patient against an independent first-usable-window reading, every
  usable first overlay must be its patient's first usable window, and the patients whose first overlay is N/A with
  a later usable window are counted (the run stops on a mismatch).
- Strata: 143's ICU-typical, awake and pDoC wake-like cells for the stated-rule readings and their randomised
  counterparts, with exact CP CIs.
- cruse11's BDSP b4 natives are read from 148's rows (output/intermediate/_bench_shards_cruse11_b4: b4-sized windows,
  so those readings replace the earlier ones on the 6-block window; counts in the output); the beside columns and
  116's A1 check keep the fixed 115 rows.
- Run guard: the eligibility files must be committed and unchanged, and their sha256 passed on the command line;
  the same variants and axes as at CLASS_REF; the strict flags present.
python3 145_stated_rule.py --motor-sha HEX --erp-sha HEX --mbvt-published-k-max K [--B 2000] | --smoke"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import hashlib
import importlib
import json
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
st = importlib.import_module("143_stratum_table")
mg, rn, mech, fid = st.mg, st.rn, st.mech, st.fid
b4m = importlib.import_module("148_cruse11_b4")

FILES = (os.path.join(ROOT, "docs", "stated-rule-motor.json"), os.path.join(ROOT, "docs", "stated-rule-erp.json"))
OUT = os.path.join(st.TABLES, "stated_rule.json")
META = ("sensitivity_eligible", "shortened_to_fit", "constant_departures", "errata", "second_classifier")
CLASS_REF = "9a94bcc"
STRICT_FLAG = "stated_rule_eligible_strict"
OK_CLASSES = ("STATED", "NOT_STATED")
SHORT = "SHORTENED_STRUCTURAL"
KNOWN = OK_CLASSES + ("CONTRADICTS", SHORT)
CALL_PAPER = "as_coded:call_paper"
COVERT_LABEL = "covert processing (not a command-following call rule): exceedance only"
HELD_OUT_LABEL = "held-out (valid) test"
MIN_LABEL = "min over eligible readings (selected, not an estimate)"
MAX_LABEL = "max over eligible readings (selected, not an estimate)"
NO_ACC = "no accuracy-type statistic"
ACCURACY_STAT = {"columbia": ("stat", 0.5, "AUC"),
                 "cruse11": ("stat", 0.5, "CV accuracy (GF13: all-pairs accuracy)"),
                 "cruse12p": ("stat", 0.5, "best smoothed accuracy over time, max over the two comparisons"),
                 "dubois": ("x_peak_da", 0.5, "peak decoding accuracy"),
                 "edlow17": ("stat", 0.5, "mean CV accuracy"),
                 "mbmi": ("stat", 0.5, "reported accuracy (peak or 5-7 s mean)"),
                 "mbvt": ("stat", 1 / 8, "8-way target accuracy")}
SOURCE_LABEL = {"bdsp": "BDSP", "pdoc": "pDoC"}
GENEROUS_LABEL = "sensitivity: most favourable reading incl. unstated call rules"
STRICT_LABEL = "sensitivity: strict (both blind classifiers agree)"
READINGS = ("primary", "sensitivity_generous", "sensitivity_strict")
STRICT_NOTES = {"sokoliuk21": "the strict set drops surr_fdr and surr_word on the second classifier's top-down "
                              "argument, which the lead ruled does not apply to sokoliuk21"}
B4_INTERIM = ("interim: 148's rows are not in yet; cruse11's b4 natives are read on the pinned 6-block window "
              "(115's rows)")
B4_NOTE = ("cruse11's b4 natives read from 148's rows on b4-sized windows: the seeded start depends on the window "
           "size, so the earlier b4 readings (on the 6-block window) are replaced, not extended")
FIRST_K = ("first k = the k lowest overlay indices j, not the k earliest windows: 114.plan_overlays numbers a "
           "recording's windows in a seeded random order (mbvt / mbmi: a seeded permutation of the time-ordered "
           "tiling; others: the acceptance order of a seeded permutation of clean starts)")
CALL_PRINCIPLE = ("A variant has a STATED call rule only when EVERY axis that sets the call criterion (which test or "
                  "contrast decides) or its threshold (alpha, CI level, cut-off, correction) is STATED. Axes that "
                  "define the statistic, the null's construction (permutation unit), cross-validation, or "
                  "preprocessing are analysis choices; they stay in the STATED/NOT_STATED eligibility rule, as now.")
CALL_AXES = {
    "columbia": {"axes": ["call"]},
    "dubois": {"axes": ["call"]},
    "mbvt": {"axes": ["thr"], "note": "summary is the statistic"},
    "mbmi": {"axes": ["threshold"]},
    "chennu13": {"axes": ["rule"]},
    "sokoliuk21": {"axes": ["rule"]},
    "cornell": {"axes": ["om2", "fdr"],
                "note": "om2 = the criterion, fdr = the threshold correction; var is the variance estimator"},
    "cruse11": {"axes": ["method", "baseline_check", "ci_base"],
                "note": "method = criterion + 99% CI, baseline_check = criterion, ci_base = threshold"},
    "edlow17": {"axes": [], "note": "the call (one-sided permutation P < 0.05) is common and STATED; perm_unit is "
                                    "the null's construction"},
    "cruse12p": {"axes": [], "note": "the call is common to every variant and STATED"},
    "gibson16": {"axes": ["tail"], "note": "tails and cluster alpha = the threshold; call_on on hold pending the "
                                           "second blind classifier"},
}
NULLS = st.NULLS
B_BOOT = st.B_BOOT
K_TABLE = os.path.join(ROOT, "docs", "benchmark-k-table.json")
assert_no_identifiers = st.assert_no_identifiers



def load_classification(paths=FILES):
    """The files merged: pipelines once each; shortened_to_fit variants, sensitivity sets and constant departures
    merged by pipeline or name (a key in two files refused); errata concatenated."""
    out = {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        for k, v in d.items():
            if k == "errata":
                out.setdefault(k, []).extend(v)
            elif k == "second_classifier":
                out.setdefault(k, {})[os.path.basename(p)] = v
            elif k in ("sensitivity_eligible", "constant_departures", "shortened_to_fit"):
                tgt = out.setdefault(k, {"variants": {}} if k == "shortened_to_fit" else {})
                src = v["variants"] if k == "shortened_to_fit" else v
                dst = tgt["variants"] if k == "shortened_to_fit" else tgt
                for kk, vv in src.items():
                    if kk in dst:
                        raise RuntimeError(f"{k}: {kk} appears in more than one classification file")
                    dst[kk] = vv
                if k == "shortened_to_fit":
                    tgt.setdefault("rules", []).append(v.get("rule"))
            elif k in out:
                raise RuntimeError(f"{k} appears in more than one classification file")
            else:
                out[k] = v
    return out


def _classes(e):
    return [a.get("class") for a in e["axes"].values()]


def classification_sets(cl):
    """{"headline": {pipe: [eligible]}, "shortened": {pipe: [...]}, "sensitivity": {name: {...}}} with the guards: unknown classes, a CONTRADICTS or SHORTENED_STRUCTURAL variant marked eligible,
    the shortened_to_fit list disagreeing with the variants whose only departure is SHORTENED_STRUCTURAL (a variant
    flagged shortened that also contradicts gets no row and is listed), a sensitivity variant departing on another
    axis."""
    head, short, flagged_contra, strict = {}, {}, {}, {}
    entries = [(pipe, v, e) for pipe, vs in cl.items() if pipe not in META for v, e in vs.items()]
    n_flag = sum(STRICT_FLAG in e for _, _, e in entries)
    if n_flag not in (0, len(entries)):
        raise RuntimeError(f"{STRICT_FLAG} is on {n_flag} of {len(entries)} variants: refused")
    union = n_flag > 0
    for pipe, v, e in entries:
        cls = _classes(e)
        bad = [c for c in cls if c not in KNOWN]
        if bad:
            raise RuntimeError(f"{pipe}|{v}: unknown axis class {bad}")
        shortened = bool(e.get("shortened_to_fit")) or SHORT in cls
        if union and e[STRICT_FLAG]:
            if not e.get("stated_rule_eligible"):
                raise RuntimeError(f"{pipe}|{v}: strict-eligible but not union-eligible: refused")
            strict.setdefault(pipe, []).append(v)
        if e.get("stated_rule_eligible"):
            if shortened:
                raise RuntimeError(f"{pipe}|{v}: a shortened variant marked stated-rule eligible (headline)")
            if (not union or e[STRICT_FLAG]) and set(cls) - set(OK_CLASSES):
                raise RuntimeError(f"{pipe}|{v}: marked eligible{' by both classifiers' if union else ''} with a "
                                   f"{sorted(set(cls) - set(OK_CLASSES))} axis (CONTRADICTS is never eligible)")
            head.setdefault(pipe, []).append(v)
        elif e.get("shortened_to_fit"):
            if set(cls) - set(OK_CLASSES) - {SHORT}:
                flagged_contra.setdefault(pipe, []).append(v)
            else:
                short.setdefault(pipe, []).append(v)
    listed = {p: sorted(v) for p, v in cl.get("shortened_to_fit", {}).get("variants", {}).items()}
    if listed != {p: sorted(v) for p, v in short.items()}:
        raise RuntimeError(f"shortened_to_fit list {listed} differs from the variants whose only departure is "
                           f"SHORTENED_STRUCTURAL {short}")
    sens = {}
    for name, s in cl.get("sensitivity_eligible", {}).items():
        pipe, vs = s["pipeline"], list(s["variants"])
        if set(vs) <= set(head.get(pipe, [])):
            sens[name] = {"pipe": pipe, "variants": vs, "status": "retired: identical to the main eligible set"
                          if set(vs) == set(head[pipe]) else "retired: a subset of the main eligible set"}
            continue
        if not s.get("reclassified_axes"):
            raise RuntimeError(f"sensitivity {name}: no reclassified_axes field, and its variants are not all "
                               "main-eligible")
        re_ax = list(s["reclassified_axes"])
        for v in vs:
            e = cl[pipe][v]
            other = {a: x.get("class") for a, x in e["axes"].items() if a not in re_ax}
            if set(other.values()) - set(OK_CLASSES):
                raise RuntimeError(f"sensitivity {name}: {pipe}|{v} departs on an axis outside {re_ax}: {other}")
        sens[name] = {"pipe": pipe, "variants": vs, "reclassified_axes": re_ax, "status": "separate"}
    return {"headline": head, "shortened": short, "sensitivity": sens,
            "shortened_flag_but_contradicting": flagged_contra, "strict": strict if union else None}


def require_strict(sets):
    if sets.get("strict") is None:
        raise RuntimeError(f"no {STRICT_FLAG} flags in the classification: the strict reading cannot be formed")
    return sets


def load_classification_at(ref=CLASS_REF, paths=FILES, repo=ROOT):
    """The first classification (the classification files as committed at ref), from its copies in docs/."""
    return load_classification([p.replace(".json", "-first.json") for p in paths])


def class_changes(ref_cl, cl):
    """[{pipe, axis, from, to, n_variants}] of the axis classes that differ from the reference classification; a
    variant or axis list that differs is refused (the call test reads the reference classes of these variants)."""
    a = {p: vs for p, vs in ref_cl.items() if p not in META}
    b = {p: vs for p, vs in cl.items() if p not in META}
    if {p: sorted(vs) for p, vs in a.items()} != {p: sorted(vs) for p, vs in b.items()}:
        raise RuntimeError("the variant list differs from the reference classification: refused")
    n = {}
    for p, vs in a.items():
        for v, e in vs.items():
            if sorted(e["axes"]) != sorted(b[p][v]["axes"]):
                raise RuntimeError(f"{p}|{v}: the axis list differs from the reference classification: refused")
            for ax, x in e["axes"].items():
                c0, c1 = x.get("class"), b[p][v]["axes"][ax].get("class")
                if c0 != c1:
                    n[(p, ax, c0, c1)] = n.get((p, ax, c0, c1), 0) + 1
    return [{"pipe": p, "axis": ax, "from": c0, "to": c1, "n_variants": k} for (p, ax, c0, c1), k in sorted(n.items())]



def call_stated(cl, pipe, eligible, axes=None):
    """(the eligible variants whose call rule is STATED under CALL_PRINCIPLE, status): every call axis a variant has
    is STATED (the classes of CLASS_REF); with no call axis the call is common to every variant and stated."""
    axes = CALL_AXES if axes is None else axes
    if pipe not in axes:
        raise RuntimeError(f"{pipe}: no call-axis decision")
    names = list(axes[pipe]["axes"])
    if not names:
        return list(eligible), "stated (common to every variant)"
    for a in names:
        if not any(a in e["axes"] for e in cl[pipe].values()):
            raise RuntimeError(f"{pipe}: call axis {a!r} is on no variant")
    ok, partial = [], False
    for v in eligible:
        have = [cl[pipe][v]["axes"][a]["class"] for a in names if a in cl[pipe][v]["axes"]]
        if not have:
            raise RuntimeError(f"{pipe}|{v}: none of the call axes {names}")
        if all(c == "STATED" for c in have):
            ok.append(v)
        elif "STATED" in have:
            partial = True
    return ok, ("stated" if ok else "call rule not fully stated" if partial else "call rule not stated")


def reading_candidates(cl, sets, pipe, reading, axes=None):
    """(candidates, label, call rule) of one reading: the primary and the strict sensitivity take the variants with a
    stated call rule among the union (strict) eligible set, else the whole set labelled with the call status; the
    generous sensitivity takes the whole union set."""
    if reading == "sensitivity_generous":
        return list(sets["headline"].get(pipe, [])), GENEROUS_LABEL, None
    pool = sets["headline"] if reading == "primary" else sets["strict"]
    el = list(pool.get(pipe, []))
    ok, status = call_stated(cl, pipe, el, axes)
    base = "stated rule" if reading == "primary" else STRICT_LABEL
    rule = {"axes": list((CALL_AXES if axes is None else axes)[pipe]["axes"]), "status": status,
            "stated_call_variants": ok}
    if ok:
        return ok, base, rule
    return el, f"{base} ({status}: most favourable reading)", rule



def one_per_patient(d):
    """Per null set and pipeline, every row of the patient's first recording (sorted uid) that carries a row of the
    pipeline; rows of later recordings dropped and counted."""
    first = d.groupby(["null", "pipe", "patient"])["uid"].transform("min")
    keep = d["uid"] == first
    return d[keep].copy(), int((~keep).sum())


def c3_as_rows(c3, base):
    """C3 first-overlay rows as pseudo-variants (variant = reading) on the base variant's recording, any status
    (coverage), with the base row's min_attainable_p (cannot-call) and labels."""
    c = c3[c3["j"].notna()].copy()
    c["j"] = c["j"].astype(int)
    b = base[["null", "pipe", "uid", "j", "variant", "min_attainable_p"]].rename(columns={"variant": "base"})
    c = c.merge(b, on=["null", "pipe", "uid", "j", "base"], how="left")
    return c.assign(variant=c["reading"], resp="none", montage="native", seq=True, overlap=False, as_coded=False)


def c3_on_base_rows(c3, base):
    """143.c3_on_base on base rows taken from the combined frame: the pseudo-variant columns (base, reading) that
    rows of other variants carry are dropped first."""
    return st.c3_on_base(c3, base.drop(columns=[c for c in ("base", "reading") if c in base]))


def call_paper_rows(d):
    """Columbia's as_coded rows read with the papers' rule (x_call_paper: AUC > 0.5 and p < 0.05)."""
    c = d[(d["pipe"] == "columbia") & (d["variant"] == "as_coded")].copy()
    ok = c["status"].astype(str) == "ok"
    if "x_call_paper" not in c or c.loc[ok, "x_call_paper"].isna().any():
        raise RuntimeError("an ok Columbia row without call_paper: refused (missing is not a negative)")
    c["call"] = c["x_call_paper"]
    c["key"] = c["key"].astype(str) + "|call_paper"
    return c.assign(variant=CALL_PAPER)


def usable(df, nul, pipe, cands):
    """The eligible candidates that can call and reach 70% coverage, computed from the rows as 115 does."""
    rec, ok = {}, []
    pub = mg._null_seq(df)
    pub = pub[(pub["null"] == nul) & (pub["pipe"] == pipe) & (pub["j"] == 0)]
    for v in cands:
        dv = pub[pub["variant"] == v]
        cov, n_att = mg.coverage_of(df, nul, pipe, v)
        cc = mg.cannot_call_of(dv, pipe, v) if len(dv) else False
        rec[v] = {"n_ok_rows": int(len(dv)), "coverage": cov, "n_attempted": n_att, "cannot_call": bool(cc),
                  "low_coverage": bool(mg.low_coverage(cov))}
        if len(dv) and not cc and not mg.low_coverage(cov):
            ok.append(v)
    return ok, rec


def own_rates(dp, ok):
    """115.published's per-variant rate (per-patient ratio over the variant's rows), the fallback of the selection."""
    out = {}
    for v in ok:
        dv = dp[dp["variant"] == v].assign(_c=lambda x: x["call"].astype(float))
        s, n = mg._per_patient(dv, "_c")
        out[v] = {"rate": float(s.sum() / n.sum())}
    return out


def select(dp, ok, coded):
    """115._select_primary on the eligible set."""
    return mg._select_primary(dp, own_rates(dp, ok), coded)


def exact_rate(d):
    """k / n over one row per patient, exact CI and one-sided p against 5%."""
    if d["patient"].duplicated().any():
        raise RuntimeError("more than one row per patient: refused")
    if d["call"].isna().any():
        raise RuntimeError("a row without a call: refused (missing is not a negative)")
    k = int(d["call"].astype(bool).sum())
    return dict(st.exact_cell(k, len(d)), n_patients=int(d["patient"].nunique()))


def is_covert(pipe):
    """115's covert-processing pipelines (not command-following call rules)."""
    return pipe in mg.COVERT_PROCESSING


def add_holm(entries, field="stated_rule"):
    """Holm across the call-rule pipelines within each BDSP null set, on the one-sided exact p of `field` (the
    stated-rule reading, or the held-out test; covert-processing entries stay outside)."""
    for nul in NULLS["bdsp"]:
        keys = [k for k, e in entries.items() if e["null"] == nul and e.get(field)
                and not e.get("covert_processing")
                and e[field].get("p_greater_nominal") is not None]
        adj = st.holm([entries[k][field]["p_greater_nominal"] for k in keys])
        for k, a in zip(keys, adj):
            entries[k][field].update(p_holm=float(a), inflated=bool(a <= mg.ALPHA))


def range_endpoints(rates):
    """Each endpoint of the eligible range: its rate and every variant at it (exact ties, by name) with k / n and
    exact CI."""
    fr_ = {v: Fraction(r["k"], r["n"]) for v, r in rates.items()}
    out = {}
    for side, label, f in (("min", MIN_LABEL, min), ("max", MAX_LABEL, max)):
        x = f(fr_.values())
        out[side] = {"label": label, "rate": float(x),
                     "readings": [{"variant": v, "k": rates[v]["k"], "n": rates[v]["n"], "ci95_cp": rates[v]["ci95_cp"]}
                                  for v in sorted(fr_) if fr_[v] == x]}
    return out


def deflation(d, pipe):
    """The recording-level accuracy-type statistic against its chance value on one row per patient: median, IQR and
    the share below chance (a row without the statistic is counted as missing, never as below)."""
    if pipe not in ACCURACY_STAT:
        return {"status": NO_ACC}
    if d["patient"].duplicated().any():
        raise RuntimeError("deflation needs one row per patient: refused")
    field, chance, what = ACCURACY_STAT[pipe]
    x = pd.to_numeric(d[field], errors="coerce") if field in d else pd.Series(np.nan, index=d.index)
    ok = x.dropna().to_numpy(float)
    out = {"statistic": what, "field": field, "chance": chance, "n": int(len(ok)), "n_missing": int(len(d) - len(ok))}
    if not len(ok):
        return dict(out, status="NOT AVAILABLE")
    q25, med, q75 = np.quantile(ok, [0.25, 0.5, 0.75])
    return dict(out, median=float(med), iqr=[float(q25), float(q75)], n_below_chance=int((ok < chance).sum()),
                share_below_chance=float((ok < chance).mean()))


def no_rows_note(cl, sets, pipe, nul):
    """Why there is no stated-rule reading when no eligible variant has rows on the null set: the stated value of
    the axis the pipeline's shortened-to-fit variants shorten, where it has such variants."""
    src = SOURCE_LABEL["pdoc" if nul in NULLS["pdoc"] else "bdsp"]
    note = f"no eligible variant has {src} rows"
    axes = {a for v in sets["shortened"].get(pipe, []) for a, x in cl[pipe][v]["axes"].items() if x["class"] == SHORT}
    vals = sorted({cl[pipe][v]["axes"][a]["value"] for v in sets["headline"].get(pipe, []) for a in axes
                   if a in cl[pipe][v]["axes"]})
    return note + (f"; the stated {' / '.join(vals)} do not fit the {src} windows" if vals else "")



def _halves_of(patients):
    return mg._half(list(patients))


def _estimation_mask(half, h):
    return half != h


def _matrix(dp, ok):
    """(patients x variants) calls, NaN where the variant has no row for the patient."""
    m = dp[dp["variant"].isin(ok)].pivot_table(index="patient", columns="variant", values="call",
                                               aggfunc=lambda x: float(x.iloc[0]))
    m = m.reindex(columns=ok)
    return m.to_numpy(float), list(m.index)


def _select_weighted(C, w, ok, coded):
    """_select_primary on weighted patients (bootstrap counts): the fewest calls on the patients where every
    variant has a row, else each variant's own rate; ties as coded, then name."""
    present = ~np.isnan(C) & (w > 0)[:, None]
    cols = present.any(0)
    C, present, ok = C[:, cols], present[:, cols], [v for v, c in zip(ok, cols) if c]
    common = present.all(1)
    X = np.nan_to_num(C)
    if common.any():
        rates = (w[common] @ X[common]) / w[common].sum()
    else:
        den = (w[:, None] * present).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates = (w[:, None] * X * present).sum(0) / den
    cand = [(float(r), not coded.get(v, False), v) for v, r in zip(ok, rates) if np.isfinite(r)]
    return sorted(cand)[0][2] if cand else None


def _weighted_rate(C, w, j):
    keep = ~np.isnan(C[:, j]) & (w > 0)
    return float((w[keep] * C[keep, j]).sum() / w[keep].sum()) if keep.any() else np.nan


def crossfit(dp, ok, coded, B=B_BOOT, seed=None):
    """Select on one crc32(patient) parity half, estimate the chosen variant on the other, swap, average; the patient
    bootstrap redoes both selections in every draw (each drawn copy keeps its patient's half)."""
    d = dp[dp["variant"].isin(ok)]
    half = np.asarray(_halves_of(d["patient"]))
    by = pd.Series(half, index=d.index).groupby(d["patient"]).nunique()
    if (by > 1).any():
        raise RuntimeError("a patient has rows in both halves: refused")
    parts = []
    for h in (0, 1):
        sel, est = half == h, _estimation_mask(half, h)
        if (sel & est).any():
            raise RuntimeError("cross-fit leak: the estimation half holds selection rows")
        v, _ = select(d[sel], [x for x in ok if (d[sel]["variant"] == x).any()], coded)
        e = d[est & (d["variant"] == v)]
        parts.append({"select_half": h, "variant": v, "k": int(e["call"].astype(bool).sum()), "n": int(len(e)),
                      "rate": float(e["call"].astype(float).mean()) if len(e) else None})
    rates = [p["rate"] for p in parts]
    out = {"halves": parts, "average": float(np.mean(rates)) if None not in rates else None}
    C, pats = _matrix(d, ok)
    ph = np.asarray(_halves_of(pats))
    draws = []
    for row in mg.boot_idx(len(pats), B, seed):
        w = np.bincount(row, minlength=len(pats)).astype(float)
        rs = []
        for h in (0, 1):
            ws, we = w * (ph == h), w * _estimation_mask(ph, h)
            if (ws * we).any():
                raise RuntimeError("cross-fit leak in a bootstrap draw")
            v = _select_weighted(C, ws, ok, coded)
            rs.append(np.nan if v is None else _weighted_rate(C, we, ok.index(v)))
        draws.append(np.mean(rs))
    draws = np.asarray(draws, float)
    fin = draws[np.isfinite(draws)]
    out.update(ci95=mg._ci(fin) if len(fin) else None, n_draws_undefined=int((~np.isfinite(draws)).sum()))
    return out


def held_out_test(cf):
    """The held-out (valid) test: the two estimation halves pooled (each half's calls of the variant selected on the
    other half; disjoint patients), exact CI and one-sided exact binomial p against 5%."""
    hv = cf["halves"]
    if sorted(h["select_half"] for h in hv) != [0, 1]:
        raise RuntimeError("the held-out test needs both halves, once each")
    by = {str(h["select_half"]): h["variant"] for h in sorted(hv, key=lambda h: h["select_half"])}
    if any(h["variant"] is None or not h["n"] for h in hv):
        return {"label": HELD_OUT_LABEL, "status": "NOT AVAILABLE", "variants_by_select_half": by,
                "note": "a half has no selected variant or no estimation rows"}
    return dict(st.exact_cell(sum(h["k"] for h in hv), sum(h["n"] for h in hv)), label=HELD_OUT_LABEL,
                variants_by_select_half=by)



def bh2_paper(p1, a1, p2, a2):
    """The papers' rule over two recordings: BH at q = .05 within the pair, and AUC > 0.5 on a rejected recording."""
    p1, a1, p2, a2 = (np.asarray(x, float) for x in (p1, a1, p2, a2))
    lo_first = p1 <= p2
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    a_lo = np.where(lo_first, a1, a2)
    both = hi <= 0.05
    first = (lo <= 0.025) & ~both
    return (both & ((a1 > 0.5) | (a2 > 0.5))) | (first & (a_lo > 0.5))


def paper_pairs_rate(p, auc, idx):
    """115.bh2_pairs_rate with the paper's AUC condition: per draw, the share of unit pairs from different patients
    that the papers' rule calls (each drawn copy a unit)."""
    p, auc = np.asarray(p, float), np.asarray(auc, float)
    P = bh2_paper(p[:, None], auc[:, None], p[None, :], auc[None, :]).astype(float)
    np.fill_diagonal(P, 0.0)
    out = []
    for row in np.atleast_2d(idx):
        M = np.bincount(row, minlength=len(p)).astype(float)
        out.append(float(M @ P @ M) / float(M.sum() ** 2 - (M ** 2).sum()))
    return np.asarray(out)


def columbia_bracket(rows, paper=True, B=B_BOOT, seed=None):
    """115.columbia_k2_bracket's two bounds: independent days = pairs of different patients' first overlays
    (bootstrap CI, pairs are not independent units); same session = the first and second overlay of one recording
    (per patient, exact CI). paper: BH within the pair and AUC > 0.5 on the rejected recording (call_paper);
    otherwise 115's p-only BH."""
    d = rows[mg._flag(rows, "seq", True) & ~mg._flag(rows, "overlap")]
    if not paper:
        d = d.assign(stat=np.inf)
    first = d[d["j"] == 0].sort_values(["patient", "uid"]).drop_duplicates("patient")
    p, a = first["p"].astype(float).to_numpy(), first["stat"].astype(float).to_numpy()
    ind = {"label": "independent-days bound: BH within pairs of different patients' first overlays (all pairs)"
                    + ("; AUC > 0.5 on the rejected recording" if paper else ""), "n_patients": int(len(p))}
    if len(p) >= 2:
        ind["rate"] = float(paper_pairs_rate(p, a, np.arange(len(p))[None])[0])
        ind["ci95_bootstrap"] = mg._ci(paper_pairs_rate(p, a, mg.boot_idx(len(p), B, seed)))
    w = d[d["j"].isin([0, 1])]
    pw = w.pivot_table(index=["uid", "patient"], columns="j", values=["p", "stat"], aggfunc="first")
    same = {"label": "same-session bound: the first and second overlay of one recording"}
    if ("p", 1) in pw.columns:
        pw = pw.dropna()
        hit = bh2_paper(pw[("p", 0)], pw[("stat", 0)], pw[("p", 1)], pw[("stat", 1)])
        per = pd.Series(hit, index=pw.index.get_level_values("patient")).groupby(level=0).max()
        same.update(st.exact_cell(int(per.sum()), int(len(per))), n_patients=int(len(per)))
    else:
        same.update(status="NOT AVAILABLE", n_patients=0)
    return {"independent_days_bound": ind, "same_session_bound": same}



def _rule_of(pipe):
    return mg._rule(mg._load_pipe(pipe))


def published_k(path=K_TABLE):
    """The published k per pipeline."""
    with open(path) as fh:
        return {p: int(e["k"]) for p, e in json.load(fh).items()}


def check_k_max(k_max, kmap, pipe="mbvt"):
    """The largest published k the best-of-k curve is read to: an integer, never below the k table's k."""
    if int(k_max) != k_max or k_max < kmap[pipe]:
        raise RuntimeError(f"{pipe}: published k max {k_max} is below the k table's k {kmap[pipe]} or not an integer")
    return int(k_max)


def eligible_paradigms(cl, pipe, eligible):
    """The paradigm values of the eligible variants (axis 'paradigm', where the classification has one)."""
    return sorted({cl[pipe][v]["axes"]["paradigm"]["value"] for v in eligible if "paradigm" in cl[pipe][v]["axes"]})


def composite_rows(nul, pipe, v, c3, frozen, columbia_first, uids):
    """Every overlay of the selected variant on the patients' chosen recordings: a C3 reading from 130's rows,
    Columbia (as_coded or call_paper) from 113's first overlays plus the frozen later overlays, any other variant
    from the frozen rows."""
    if v in fid.READINGS.get(pipe, {}):
        g = c3[(c3["null"] == nul) & (c3["reading"] == v) & (c3["status"] == "ok")].assign(
            variant=v, seq=True, overlap=False)
    elif pipe == "columbia":
        later = mg._null_seq(frozen[(frozen["null"] == nul) & (frozen["pipe"] == pipe) &
                                    (frozen["variant"] == "as_coded")])
        g = pd.concat([mg._null_seq(columbia_first), later[later["j"] != 0]], ignore_index=True, sort=False)
        if v == CALL_PAPER:
            g = g.assign(call=g["x_call_paper"], variant=CALL_PAPER)
    else:
        g = mg._null_seq(frozen[(frozen["null"] == nul) & (frozen["pipe"] == pipe) & (frozen["variant"] == v)])
    return g[g["uid"].isin(uids)]


def composite(rows, pipe, k):
    """115.patient_outcomes over each patient's first k non-overlapping windows with the module's own rule, exact
    CI; NOT AVAILABLE when no patient has k windows."""
    o = mg.patient_outcomes(rows, _rule_of(pipe), k)
    if not o:
        return {"k_runs": int(k), "status": "NOT AVAILABLE", "note": f"no patient has {k} windows of this reading"}
    v = np.array(list(o.values()), bool)
    return dict(st.exact_cell(int(v.sum()), int(len(v))), k_runs=int(k),
                n_patients_with_fewer_windows=int(rows["patient"].nunique() - len(v)))


def k1_check(g, pub, pipe):
    """The k = 1 composite (115.patient_outcomes: each patient's first usable window) against an independent reading
    of each patient's first usable window (the smallest j with a usable row), patient by patient; every patient
    with a usable first overlay must have it as the first usable window, with the first overlay's own call. Either
    failing stops the run. Reported beside the first-overlay rate, with the patients whose first overlay is N/A and
    a later window usable."""
    o = mg.patient_outcomes(g, _rule_of(pipe), 1)
    d = g[mg._flag(g, "seq", True) & ~mg._flag(g, "overlap")]
    first = d.loc[d.groupby("patient")["j"].idxmin()].set_index("patient")
    mine = {p: bool(c) for p, c in first["call"].items()}
    if o != mine:
        bad = sorted(set(o) ^ set(mine)) or [p for p in o if o[p] != mine.get(p)]
        raise RuntimeError(f"the k = 1 composite does not take each patient's first usable window ({len(bad)} "
                           "patients differ)")
    p0 = pub[pub["j"] == 0].set_index("patient")
    for p, c in p0["call"].items():
        if p not in first.index or int(first.at[p, "j"]) != 0 or bool(first.at[p, "call"]) != bool(c):
            raise RuntimeError("a usable first overlay (j = 0) is not the patient's first usable window in the "
                               "composite rows")
    k1 = sum(o.values())
    return {"composite_k1": st.exact_cell(int(k1), len(o)), "first_usable": st.exact_cell(int(sum(mine.values())),
                                                                                           len(mine)),
            "first_overlay": st.exact_cell(int(p0["call"].astype(bool).sum()), len(p0)),
            "n_first_overlay_na_later_usable": int((first["j"] > 0).sum())}


def best_of_k_curve(rows, pipe, ks, k_max, B=B_BOOT, seed=None):
    """Descriptive: 115.composite_curve with the best-run rule, each point with its exact CI, on ks plus k_max (the
    largest published k); every larger k is marked beyond published k."""
    ks = sorted(set(ks) | {int(k_max)})
    cur = mg.composite_curve(rows, mg._rule(mg._load_pipe(pipe), rule="best_run"), ks, B, seed)
    for c in cur:
        c["ci95_cp"] = None
        if c["n"]:
            c["ci95_cp"] = st.clopper_pearson(int(round(c["rate"] * c["n"])), c["n"])
        c["reading"] = "within published k" if c["k"] <= k_max else "beyond published k"
    return cur



def stratum_cell(rows, stratum, B=B_BOOT, seed_parts=()):
    """143's cell (reading and randomised counterpart, each on its own window's labels) with exact intervals."""
    c = st.cell(rows, stratum, B, ("145",) + tuple(seed_parts))
    for arm in ("reading", "randomised"):
        r = c[arm]
        r["ci95_cp"] = st.clopper_pearson(r["k"], r["n"]) if r["n"] else None
        r["p_greater_nominal"] = st.binom_vs_nominal(r["k"], r["n"])
    return c



def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_classification(paths, shas, repo=ROOT):
    """Each file with the given sha256 (the version-control checks are not applied here)."""
    for p, want in zip(paths, shas):
        got = _sha256(p)
        if got != want:
            raise RuntimeError(f"{os.path.basename(p)}: sha256 {got} is not the given {want}")



def load_all_variant_rows(want, b4set, shards=None, b4_root=None):
    """load_variant_rows over 114's shards, cruse11's BDSP b4 natives from 148's folder."""
    key = (b4m.NULL, b4m.PIPE)
    main = {k: (set(vs) - set(b4set) if k == key else set(vs)) for k, vs in want.items()}
    frames = [load_variant_rows({k: v for k, v in main.items() if v}, root=shards or st.SHARDS)]
    if set(want.get(key, ())) & set(b4set):
        frames.append(load_variant_rows({key: set(want[key]) & set(b4set)}, root=b4_root or b4m.B4_ROOT))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_variant_rows(wanted, root=None):
    """Every overlay (any j) of the wanted (null, pipe, variant) rows from the native resp-none shards, 115.load_rows
    semantics (first row of a key kept; split and dev checked on every row)."""
    frames = []
    root = root or st.SHARDS
    for (nul, pipe), vs in wanted.items():
        d = os.path.join(root, f"{nul}__{pipe}__native__none")
        tags = {f'"variant": "{v}"' for v in vs}
        seen, rows = set(), []
        for fn in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
            with open(fn) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    if f'"split": "{st.SPLIT}"' not in line or '"dev": false' not in line:
                        raise RuntimeError(f"a row of another split or a development recording ({fn})")
                    if not any(t in line for t in tags):
                        continue
                    r = json.loads(line)
                    if r["key"] in seen or r.get("variant") not in vs:
                        continue
                    seen.add(r["key"])
                    rows.append(r)
        if rows:
            frames.append(pd.DataFrame(rows).assign(dir=os.path.basename(d)))
    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return mg.drop_touched(df)[0] if len(df) else df


def attach_flags(sel, tables):
    """143's labels on published first-overlay rows: A9 on the row's extent and on the randomised extent; pDoC
    lower-delta from 116 (checked against the recomputed flag) and recomputed on the randomised extent."""
    sel = sel.copy()
    src = np.where(sel["null"].isin(NULLS["pdoc"]), "pdoc", "bdsp")
    for c in ("a9_flag", "rand_a9_flag", "ld_check", "rand_lower_delta"):
        sel[c] = pd.Series(None, index=sel.index, dtype=object)
    for s in ("bdsp", "pdoc"):
        m = src == s
        if not m.any():
            continue
        sel.loc[m, "a9_flag"] = st.a9(sel[m], s, tables=tables[s])
        r = sel[m].copy()
        for c in ("extent_t0", "extent_t1"):
            own = r["rand_" + c] if "rand_" + c in r else pd.Series(np.nan, index=r.index)
            r["rand_" + c] = own.astype(float).fillna(r[c].astype(float))
        sel.loc[m, "rand_a9_flag"] = st.a9(r, s, "rand_extent_t0", "rand_extent_t1", tables=tables[s])
        if s == "pdoc":
            f = st.lower_delta_flags(r, (("check", "extent_t0_116", "extent_t1_116"),
                                         ("rand", "rand_extent_t0", "rand_extent_t1")))
            sel.loc[m, "ld_check"], sel.loc[m, "rand_lower_delta"] = f["check"], f["rand"]
            bad = int(st.same_flags(sel.loc[m, "ld_check"], sel.loc[m, "lower_delta"]).sum())
            if bad:
                raise RuntimeError(f"recomputed lower-delta flag differs from 116's on {bad} pDoC rows: refused")
    return sel


def _assess(dp_all, nul, pipe, cands, coded, B, label):
    """Selection and report on one candidate set (headline, shortened, sensitivity or side row)."""
    ok, rec = usable(dp_all, nul, pipe, cands)
    pub = mg._null_seq(dp_all)
    pub = pub[(pub["null"] == nul) & (pub["pipe"] == pipe) & (pub["j"] == 0)]
    v, selection = select(pub, ok, coded)
    out = {"label": label, "candidates": list(cands), "eligible_after_rows": ok, "variants": rec,
           "selection": selection, "variant": v}
    if v is None:
        out["note"] = "no eligible variant can call with >= 70% coverage: no stated-rule reading"
        return out, pub
    rates = {x: exact_rate(pub[pub["variant"] == x]) for x in ok}
    out["stated_rule"] = rates[v]
    out["eligible_range"] = {"min": min(r["rate"] for r in rates.values()), "max": max(r["rate"] for r in rates.values()),
                             "n_variants": len(ok), "rates": rates, "endpoints": range_endpoints(rates)}
    out["crossfit"] = crossfit(pub, ok, coded, B, mg._seed("145", nul, pipe, label, "crossfit"))
    out["held_out_test"] = held_out_test(out["crossfit"])
    out["deflation"] = {"selected": dict(variant=v, **deflation(pub[pub["variant"] == v], pipe)),
                        "endpoints": {side: [dict(variant=r["variant"],
                                                  **deflation(pub[pub["variant"] == r["variant"]], pipe))
                                             for r in out["eligible_range"]["endpoints"][side]["readings"]]
                                      for side in ("min", "max")}}
    return out, pub


def swap_b4(df, new):
    """(df with 114's cruse11 BDSP rows of the b4 natives replaced by 148's, the replaced keys, counts): 148's rows
    must be cruse11 on the BDSP no-command null, of exactly the b4 natives; its no_fit rows are left out. new None:
    the interim reading (the fixed rows kept, labelled)."""
    if new is None:
        return df, set(), {"status": B4_INTERIM}
    if not len(new):
        raise RuntimeError("no 148 rows: run 148_cruse11_b4 first")
    if ((new["null"] != b4m.NULL) | (new["pipe"] != b4m.PIPE)).any():
        raise RuntimeError("148 rows of another null set or pipeline: refused")
    v4 = set(b4m.b4_natives(rn.load_pipe(b4m.PIPE)))
    got = set(new["variant"].dropna())
    if got != v4:
        raise RuntimeError(f"148 rows hold {sorted(got)}, not the b4 natives {sorted(v4)}: refused")
    nofit = new["variant"].isna()
    old = (df["null"] == b4m.NULL) & (df["pipe"] == b4m.PIPE) & df["variant"].isin(v4)
    ok = lambda d: d[d["status"] == "ok"]
    info = {"variants": sorted(v4), "n_recordings_old": int(ok(df[old])["uid"].nunique()),
            "n_rows_old": int(old.sum()), "n_recordings_new": int(ok(new[~nofit])["uid"].nunique()),
            "n_rows_new": int((~nofit).sum()), "n_rows_no_fit_new": int(nofit.sum())}
    return (pd.concat([df[~old], new[~nofit]], ignore_index=True, sort=False), set(df.loc[old, "key"]), info)


def build_rows(df, c3):
    rows = pd.concat([df, c3_as_rows(c3, df), call_paper_rows(df)], ignore_index=True, sort=False)
    return one_per_patient(rows)


def prepare(b4=True):
    """The rows (first overlays of the 115 rows with cruse11's b4 rows from 148, C3 pseudo-variants, call_paper; one
    row per patient), the fixed rows (115's, for the beside columns and 116's A1 check), 130's C3 rows, 116's X
    results (those of the replaced keys left out for the rows) and 121's A9 tables."""
    df = st.analysis_rows()
    c3 = st.c3_rows()
    new = mg.drop_touched(st.load_first_overlays(b4m.B4_ROOT))[0] if b4 else None
    df_b4, keys, info = swap_b4(df, new)
    rows, n_dropped = build_rows(df_b4, c3)
    rows_reg, _ = build_rows(df, c3)
    res = mech.read_results(mech.paths(st.SPLIT)["results"])
    res = res[res["kind"] == "published"][[c for c in ("key", "pipe", "lower_delta", "extent_t0_116",
                                                       "extent_t1_116") if c in res]]
    res_rows = res[~(res["key"].isin(keys) & (res["pipe"] == b4m.PIPE))]
    tables = {s: importlib.import_module("121_activation").load(s, st.SPLIT) for s in ("bdsp", "pdoc")}
    return {"df": df, "c3": c3, "rows": rows, "rows_pinned": rows_reg, "n_dropped": n_dropped, "res": res,
            "res_rows": res_rows, "tables": tables, "b4": dict(info, root=os.path.relpath(b4m.B4_ROOT, ROOT))}


def label_rows(rows_pub, res, tables):
    """Published first-overlay rows with 116's extents and lower-delta flags and 143's A9 / randomised labels."""
    b = rows_pub.merge(res, on=["key", "pipe"], how="left", validate="one_to_one")
    return attach_flags(mech._with_extent(b), tables)


def check_a1(df, res, tables):
    """116's A1 cells reproduced from the same row and label plumbing (143.check_116)."""
    with open(st.MECH_FILE) as fh:
        mech_out = json.load(fh)
    want = {(k.split("|")[0], k.split("|")[1], v[r]) for k, v in mech_out["part_A"]["variant_choices"].items()
            if k.split("|")[1] not in st.LEFT_OUT for r in ("validity_primary", "default")}
    p = mg._null_seq(df)
    p = p[(p["j"] == 0) & [t in want for t in zip(p["null"], p["pipe"], p["variant"])]]
    p = label_rows(p, res, tables)
    return st.check_116({k: g for k, g in p.groupby(["null", "pipe", "variant"])}, mech_out, 2000)


def smoke():
    """Plumbing check before the classification is final: rows loaded, the classification applied, coverage and
    cannot-call per candidate, labels attached to every candidate's first overlays (C3 readings on their base rows),
    116's A1 cells reproduced. No selection is made and no call is counted."""
    cl = load_classification()
    sets = classification_sets(cl)
    P = prepare()
    df, c3, rows, n_dropped, res, tables = P["df"], P["c3"], P["rows"], P["n_dropped"], P["res"], P["tables"]
    choices = st.choices_from_headline()
    cov, lab_keys = {}, set()
    for (nul, pipe) in sorted(choices):
        dp = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
        present = set(dp["variant"])
        for kind in ("headline", "shortened"):
            cands = [v for v in sets[kind].get(pipe, []) if v in present]
            if not cands:
                continue
            ok, rec = usable(dp, nul, pipe, cands)
            cov[f"{nul}|{pipe}|{kind}"] = {"n_candidates": len(cands), "n_usable": len(ok),
                                           "missing_here": sorted(set(sets[kind].get(pipe, [])) - present),
                                           "variants": rec}
            for v in cands:
                base = fid.base(pipe, v) if v in fid.READINGS.get(pipe, {}) else (
                    "as_coded" if v == CALL_PAPER else v)
                lab_keys.add((nul, pipe, base))
    pub = mg._null_seq(rows)
    pub = pub[(pub["j"] == 0) & [t in lab_keys for t in zip(pub["null"], pub["pipe"], pub["variant"])]]
    lab = label_rows(pub, P["res_rows"], tables)
    c3_ok = {}
    for (nul, reading), g in c3[c3["reading"].notna()].groupby(["null", "reading"]):
        pipe = g["pipe"].iloc[0]
        b = lab[(lab["null"] == nul) & (lab["pipe"] == pipe) & (lab["variant"] == fid.base(pipe, reading))]
        _, counts = c3_on_base_rows(g, b)
        c3_ok[f"{nul}|{reading}"] = counts
    na = {c: int(lab[c].isna().sum()) for c in ("a9_flag", "rand_a9_flag", "stratum")}
    pd_lab = lab[lab["null"].isin(NULLS["pdoc"])]
    na.update(pdoc_lower_delta=int(pd_lab["lower_delta"].isna().sum()),
              pdoc_rand_lower_delta=int(pd_lab["rand_lower_delta"].isna().sum()))
    out = {"rows_dropped_later_recordings": n_dropped, "coverage": cov, "labelled_rows": int(len(lab)),
           "labelled_variant_sets": len(lab_keys), "label_na_counts": na, "c3_on_base": c3_ok, "cruse11_b4": P["b4"],
           "a1_116": check_a1(df, res, tables), "classification": {os.path.relpath(p, ROOT): _sha256(p)
                                                                    for p in FILES}}
    assert_no_identifiers(mg._jsonable(out))
    return out


def reading_flag(reading, pipe):
    return STRICT_NOTES.get(pipe) if reading == "sensitivity_strict" else None


def assess_reading(reading, cl_ref, cl, sets, choices, rows, B, rows_beside=None):
    """One reading (primary, generous or strict sensitivity) over every null set and pipeline: selection, exact CI,
    held-out test, cross-fit, eligible range with endpoints, deflation; Holm within the reading. The call test reads
    the reference classes (cl_ref), eligibility the current flags (sets)."""
    entries = {}
    for (nul, pipe), (vp, dflt) in sorted(choices.items()):
        dp = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
        coded = {v: bool(mg._flag(g, "as_coded").any()) for v, g in dp.groupby("variant")}
        present = set(dp["variant"])
        cand, label, rule = reading_candidates(cl_ref, sets, pipe, reading)
        if rule and rule["stated_call_variants"] and not set(cand) & present:
            raise RuntimeError(f"{reading} {nul}|{pipe}: no variant with a stated call rule has rows")
        cands = [v for v in cand if v in present]
        e, pub = _assess(dp, nul, pipe, cands, coded, B, label)
        if not cands:
            e["note"] = no_rows_note(cl, sets, pipe, nul)
        rb = rows if rows_beside is None else rows_beside
        pb = mg._null_seq(rb[(rb["null"] == nul) & (rb["pipe"] == pipe)])
        pb = pb[pb["j"] == 0]
        e.update(null=nul, pipe=pipe, reading=reading, call_rule=rule,
                 beside={"rows": "pinned (115's rows)",
                         "validity_primary": {"variant": vp, **exact_rate(pb[pb["variant"] == vp])},
                         "c0_default": {"variant": dflt, "c0": st.C0.get(pipe, "not graded by C0"),
                                        **exact_rate(pb[pb["variant"] == dflt])}})
        if reading_flag(reading, pipe):
            e["flag"] = reading_flag(reading, pipe)
        if is_covert(pipe):
            e.update(covert_processing=True, label=COVERT_LABEL, reading_label=label)
        entries[f"{nul}|{pipe}"] = e
    add_holm(entries)
    add_holm(entries, "held_out_test")
    return entries


def wanted_rows(entries, want):
    """The frozen rows every composite of these entries reads (Columbia: as_coded; C3 readings come from 130)."""
    for key, e in entries.items():
        nul, pipe = key.split("|")
        v = e.get("variant")
        if e.get("covert_processing"):
            continue
        if pipe == "columbia":
            want.setdefault((nul, pipe), set()).add("as_coded")
        elif v and v not in fid.READINGS.get(pipe, {}):
            want.setdefault((nul, pipe), set()).add(v)
    return want


def composites_of(entries, cl, sets, rows, c3, allj, kmap, k_max, B):
    """Per patient at the published k, on the calls of the variant each entry's selection picked."""
    comp = {}
    for key, e in entries.items():
        nul, pipe = key.split("|")
        v, k = e.get("variant"), kmap.get(pipe, 1)
        if e.get("covert_processing"):
            continue
        if v is None:
            comp[key] = {"status": "no stated-rule reading"}
            continue
        mine = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
        col_first = mine[mine["variant"] == "as_coded"] if pipe == "columbia" else None
        g = composite_rows(nul, pipe, v, c3, allj, col_first, set(mine["uid"]))
        pub_v = mg._null_seq(mine)
        if pipe == "columbia":
            comp[key] = {"variant": v, "k": k, "first_k": FIRST_K, "bracket": columbia_bracket(
                g, paper=(v == CALL_PAPER), B=B, seed=mg._seed("145", nul, "columbia", v, "k2")),
                "k1_check": k1_check(g, pub_v[(pub_v["j"] == 0) & (pub_v["variant"] == v)], pipe)}
            continue
        c = {"variant": v, "k": k, "first_k": FIRST_K, "composite": composite(g, pipe, k)}
        c["k1_check"] = k1_check(g, pub_v[(pub_v["j"] == 0) & (pub_v["variant"] == v)], pipe)
        if (c["k1_check"]["first_overlay"]["k"], c["k1_check"]["first_overlay"]["n"]) != (e["stated_rule"]["k"],
                                                                                       e["stated_rule"]["n"]):
            raise RuntimeError(f"{key}: the first-overlay count of the composite rows differs from the stated rule")
        if pipe == "mbvt":
            par = eligible_paradigms(cl, pipe, sets["headline"].get(pipe, []))
            c["eligible_paradigms"] = par
            if len(par) == 1:
                c["note"] = (f"k = 2 = the first two runs of {v}: only {par[0]} variants are stated-rule eligible, "
                             "so the GUG18 session (one VT2 + one VT3 run) cannot be formed")
            c["best_of_k_descriptive"] = best_of_k_curve(g, pipe, mg.COMPOSITE_K["mbvt"], k_max, B,
                                                         mg._seed("145", nul, pipe, v, "best_run"))
            c["published_k_max"] = k_max
        comp[key] = c
    return comp


def strata_of(entries, rows, c3, res, tables, B):
    """143's strata cells on each entry's selected reading and its randomised counterpart, exact intervals."""
    strata = {}
    for key, e in entries.items():
        nul, pipe = key.split("|")
        v = e.get("variant")
        if v is None or e.get("covert_processing"):
            continue
        base_v = fid.base(pipe, v) if v in fid.READINGS.get(pipe, {}) else ("as_coded" if v == CALL_PAPER else v)
        b = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe) & (rows["variant"] == base_v)])
        b = label_rows(b[b["j"] == 0], res, tables)
        if v in fid.READINGS.get(pipe, {}):
            g, _ = c3_on_base_rows(c3[(c3["null"] == nul) & (c3["reading"] == v)], b)
        elif v == CALL_PAPER:
            g = b.assign(call=b["x_call_paper"])
        else:
            g = b
        src = "pdoc" if nul in NULLS["pdoc"] else "bdsp"
        strata[key] = {s: stratum_cell(g, s, B, (nul, pipe, v)) for s, r in st.STRATUM_RULES.items()
                       if r["source"] == src}
    return strata


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motor-sha")
    ap.add_argument("--erp-sha")
    ap.add_argument("--mbvt-published-k-max", type=int, help="largest published k the mbvt best-of-k is read to")
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--interim-without-b4", action="store_true",
                    help="before 148's rows exist: cruse11's b4 natives on the pinned rows, labelled interim")
    ap.add_argument("--smoke", action="store_true", help="plumbing check only: no selection, no call counted")
    a = ap.parse_args(argv)
    if a.smoke:
        out = smoke()
        print(json.dumps(mg._jsonable(out), indent=1, default=float))
        return out
    if not (a.motor_sha and a.erp_sha and a.mbvt_published_k_max is not None):
        ap.error("--motor-sha, --erp-sha and --mbvt-published-k-max are required for the run")
    check_classification(FILES, (a.motor_sha, a.erp_sha))
    kmap = published_k()
    k_max = check_k_max(a.mbvt_published_k_max, kmap)
    cl = load_classification()
    cl_ref = load_classification_at(CLASS_REF)
    changes = class_changes(cl_ref, cl)
    sets = require_strict(classification_sets(cl))
    choices = st.choices_from_headline()
    P = prepare(b4=not a.interim_without_b4)
    c3, rows, tables = P["c3"], P["rows"], P["tables"]
    readings = {r: assess_reading(r, cl_ref, cl, sets, choices, rows, a.B, P["rows_pinned"]) for r in READINGS}
    side = {}
    for (nul, pipe), _ in sorted(choices.items()):
        dp = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
        coded = {v: bool(mg._flag(g, "as_coded").any()) for v, g in dp.groupby("variant")}
        present, key = set(dp["variant"]), f"{nul}|{pipe}"
        sv = [v for v in sets["shortened"].get(pipe, []) if v in present]
        if sv:
            side.setdefault(key, {})["shortened"] = _assess(dp, nul, pipe, sv, coded, a.B,
                                                            "shortened to fit (never in the headline)")[0]
        for name, s in sets["sensitivity"].items():
            if s["pipe"] == pipe and s["status"] == "separate":
                sv = [v for v in s["variants"] if v in present]
                side.setdefault(key, {})[f"sensitivity:{name}"] = _assess(dp, nul, pipe, sv, coded, a.B,
                                                                          f"sensitivity: {name}")[0]
    want = {}
    for ents in readings.values():
        wanted_rows(ents, want)
    allj = load_all_variant_rows(want, set(P["b4"].get("variants", ())))
    checks = {"a1_116": check_a1(P["df"], P["res"], tables)}
    blocks = {r: {"label": "stated rule" if r == "primary" else GENEROUS_LABEL if r == "sensitivity_generous"
                  else STRICT_LABEL, "entries": ents,
                  "per_patient_at_published_k": composites_of(ents, cl, sets, rows, c3, allj, kmap, k_max, a.B),
                  "strata": strata_of(ents, rows, c3, P["res_rows"], tables, a.B)} for r, ents in readings.items()}
    prim = blocks["primary"]
    out = {"what": __doc__.split("\n")[0], "added_later": True, "split": st.SPLIT, "B": a.B,
           "sha256_145": _sha256(os.path.abspath(__file__)),
           "classification": {os.path.relpath(p, ROOT): _sha256(p) for p in FILES},
           "class_reference": {"commit": CLASS_REF, "note": "the call test reads the axis classes of this commit; "
                                                            "eligibility reads the current flags",
                               "class_changes_since": changes},
           "k_table": {os.path.relpath(K_TABLE, ROOT): _sha256(K_TABLE)}, "mbvt_published_k_max": k_max,
           "call_principle": CALL_PRINCIPLE, "call_axes": CALL_AXES, "first_k": FIRST_K,
           "rows_dropped_later_recordings": P["n_dropped"], "checks": checks,
           "cruse11_b4": dict(P["b4"], note=B4_INTERIM if a.interim_without_b4 else B4_NOTE),
           "covert_processing_rows": sorted(k for k, e in prim["entries"].items() if e.get("covert_processing")),
           "n_call_rule_analyses": sum(1 for e in prim["entries"].values() if not e.get("covert_processing")),
           "classification_sets": sets, "entries": prim["entries"], "side_rows": side,
           "per_patient_at_published_k": prim["per_patient_at_published_k"], "strata": prim["strata"],
           "sensitivity_generous": blocks["sensitivity_generous"],
           "sensitivity_strict": blocks["sensitivity_strict"]}
    out = mg._jsonable(out)
    assert_no_identifiers(out)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"stated-rule table: {len(prim['entries'])} entries x {len(READINGS)} readings -> {OUT}")
    return out


if __name__ == "__main__":
    main()
