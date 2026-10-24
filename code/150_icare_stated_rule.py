"""Stated-rule readings in post-arrest coma (added later, labelled so wherever reported). No fixed output changes and
no outcome is read: the patients are the arm's primary-model set (the denominators of 122's false-call rates), taken
from 122's own patient table.
- Rows: 122's first-overlay rows of those patients (null sets ncmd_icare, nstim0_icare), Columbia's call_paper rows
  (145.call_paper_rows) and the C3 fidelity-corrected readings of 130 (Cornell jackknife + contiguity; du Bois reading
  B, random and blocked CV) as pseudo-variants (145.c3_as_rows).
- C3 rows (--c3): run here on each patient's fixed window (keep_windows/<id>.npz of 133, verified there against
  the arm's channel SDs), rebuilt as 122 built the Recording for the code version that processed the patient
  (pre_reference_v1 / v2 windows: 122's artifact seconds and padding context; windows without a padding rule, processed
  before it existed: the engine's own), with 130's planning and seeds (130.phase1_jobs, first overlay only; 114's
  schedule and run seeds) and n_recs as 122.run_arm computes it. Reproduction check on every row: the reading's
  frozen by-products must equal the arm's stored base-variant row (Cornell: the any-bin call and z-max; du Bois: the
  frozen reading A p), else the row is marked not_reproduced and left out of every denominator (counted).
- Randomised counterpart of each selected reading: rand_call of the same rows (a C3 reading takes its base variant's);
  and per pipeline the pooled randomised rate over every overlay, as on BDSP (115.proposed_validity).
- Holm: none (estimation only, as pDoC). The output carries no patient identifier.
python3 150_icare_stated_rule.py --c3 [--workers W] [--limit N] | --merge [--B 2000] | --by-background | --primary-outcome"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import concurrent.futures as cf
import hashlib
import importlib
import json
import sys

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
ic = importlib.import_module("122_icare_arm")
rf = importlib.import_module("133_icare_refetch")
sr = importlib.import_module("145_stated_rule")
fid, mg, rn, st = sr.fid, sr.mg, sr.rn, sr.st

C3_DIR = os.path.join(ROOT, "output", "intermediate", "_icare_c3")
OUT = os.path.join(ROOT, "output", "tables", "icare_stated_rule.json")
NULL = "ncmd_icare"
PIPES = ("cornell", "dubois")
TOL = 1e-9


def model_patients():
    """The arm's primary-model patients (122.analyse's set: the denominators of its false-call rates)."""
    index = ic.load_index()
    dev = ic.dev_patients(index)
    t = ic.patient_table(ic.analysis_order(index, dev), index)
    kept, _ = ic.model_set(t)
    return sorted(kept["patient"].astype(str)), sum(bool(ic.candidates(s)) for s in index.values())


def recording(pid, x_rec, keep_windows):
    """The Recording 122 built for this patient, from the kept window (its window record must be the arm's)."""
    win = x_rec["window"]
    z = np.load(os.path.join(keep_windows, f"{pid}.npz"))
    kw = json.loads(str(z["window"]))
    if (kw["segment"], int(kw["start_s"])) != (win["segment"], int(win["start_s"])):
        raise RuntimeError(f"{pid}: the kept window is not the arm's window")
    df = pd.read_parquet(os.path.join(ic.paths("analysis")["morgoth"], f"{pid}.parquet"))
    rec = ic.make_recording(pid, z["x"], df, False, win)
    n = rec.x.shape[1] // ic.FS_OUT
    rule = win.get("padding_rule")
    if rule in (ic.PADDING_RULE, ic.V1_RULE):
        pad, flat = np.zeros(n, bool), np.zeros(n, bool)
        pad[np.asarray(win["pad_s"], int)] = True
        flat[np.asarray(win["flat_s"], int)] = True
        rec.meta["_artifact_s"] = ic.artifact_seconds_icare(rec, flat, pad)
        return rec, pad, rule
    if rule is not None:
        raise RuntimeError(f"{pid}: unknown padding rule {rule!r}")
    return rec, None, "none (engine's own artifact seconds)"


def stored_base(rows, pipe, base):
    r = [x for x in rows if x["pipe"] == pipe and x["variant"] == base and x.get("j") == 0 and x.get("seq")
         and x.get("resp") == "none" and x.get("montage") == "native"]
    return r[0] if len(r) == 1 else None


def reproduced(pipe, res, stored):
    """The reading's frozen by-products against the arm's stored base row."""
    if stored is None or stored.get("status") != "ok":
        return False, "no ok stored base row"
    ex = res.get("extra", {})
    if pipe == "cornell":
        ok = bool(ex["call_anybin"]) == bool(stored["call"]) and abs(float(res["stat"]) - float(stored["stat"])) <= TOL
        return ok, None if ok else f"call {ex['call_anybin']} vs {stored['call']}, z {res['stat']} vs {stored['stat']}"
    ok = abs(float(ex["p_t_A"]) - float(stored["p"])) <= TOL
    return ok, None if ok else f"p_A {ex['p_t_A']} vs {stored['p']}"


def c3_patient(pid, n_recs, keep_windows):
    x_rec = ic.read_data_json(os.path.join(ic.paths("analysis")["x"], f"{pid}.json"))
    arm_rows = [json.loads(ln) for ln in ic.read_lines(os.path.join(ic.paths("analysis")["rows"], f"{pid}.jsonl"))]
    rec, pad, rule = recording(pid, x_rec, keep_windows)
    out = []
    try:
        if pad is not None:
            ic.set_padding_context(rec.x, pad)
        for pipe in PIPES:
            mod = rn.load_pipe(pipe)
            spec = {"name": NULL, "kind": "ncmd", "family": mod.FAMILY, "source": "bdsp"}
            jobs, why = fid.phase1_jobs(rec, mod, spec, n_recs, first_only=True)
            base_row = {"null": NULL, "pipe": pipe, "uid": pid, "patient": pid, "recording_setup": rule}
            if not jobs:
                out.append({**base_row, "key": f"{pid}|nofit", "status": why or "no_published_overlay"})
                continue
            for jb in jobs:
                if jb["ov"] is None:
                    res, err, sec = None, jb["err"], 0.0
                else:
                    res, err, sec = rn._call_run(mod, fid.run, pipe, rec, jb["ov"], jb["reading"], jb["rng"])
                row = {**base_row, "key": f"{pid}|{jb['j']}|{jb['reading']}", "j": jb["j"], "start_s": jb["start_s"],
                       "reading": jb["reading"], "base": jb["base"], "status": rn._status(err), "error": err,
                       "call": None if res is None else bool(res["call"]), "p": None if res is None else res["p"],
                       "stat": None if res is None else res["stat"], "seconds": sec,
                       **rn.scalars(res and res.get("extra"))}
                if res is not None:
                    sb = stored_base(arm_rows, pipe, jb["base"])
                    ok, why_not = reproduced(pipe, res, sb)
                    if sb is not None and float(sb["start_s"]) != float(jb["start_s"]):
                        ok, why_not = False, f"start {jb['start_s']} vs stored {sb['start_s']}"
                    row["reproduced"] = ok
                    if not ok:
                        row.update(status="not_reproduced", reproduce_detail=why_not)
                out.append(row)
            rn._clear_cache()
    finally:
        if pad is not None:
            ic.clear_padding_context()
    return out


def window_patients():
    """Every analysis patient with a processed window (122.patient_table status "included"); the selection reads the
    status column only, never an outcome or covariate."""
    index = ic.load_index()
    t = ic.patient_table(ic.analysis_order(index, ic.dev_patients(index)), index)
    return sorted(t.loc[t["status"] == "included", "patient"].astype(str))


def run_c3(workers=1, limit=None, all_windows=False):
    pats, n_recs = model_patients()
    if all_windows:
        pats = window_patients()
    kw, _ = rf.keep_dirs()
    os.makedirs(C3_DIR, exist_ok=True)
    fn = os.path.join(C3_DIR, "rows.jsonl")
    done = set()
    if os.path.exists(fn):
        done = {json.loads(ln)["patient"] for ln in open(fn) if ln.strip()}
    todo = [p for p in pats if p not in done and os.path.exists(os.path.join(kw, f"{p}.npz"))][:limit]
    missing = [p for p in pats if not os.path.exists(os.path.join(kw, f"{p}.npz"))]
    ic.say(f"150 c3: {len(pats)} model patients, n_recs {n_recs}, {len(done)} done, {len(todo)} to run, "
           f"{len(missing)} without a kept window yet")
    with open(fn, "a") as fh, cf.ProcessPoolExecutor(workers) as ex:
        for pid, rows in zip(todo, ex.map(c3_patient, todo, [n_recs] * len(todo), [kw] * len(todo))):
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            bad = [r["key"].split("|", 1)[1] for r in rows if r.get("status") == "not_reproduced"]
            ic.say(f"{pid}: {len(rows)} C3 rows" + (f", NOT reproduced: {bad}" if bad else ""))
    return len(todo), len(missing)



def no_rows_note(cl, sets, pipe, nul):
    """145.no_rows_note for the I-CARE windows (145's joins the stated values as strings; an integer value, never
    reached on BDSP or pDoC, is converted here). 145 reads this module-level name at call time."""
    axes = {a for v in sets["shortened"].get(pipe, []) for a, x in cl[pipe][v]["axes"].items() if x["class"] == sr.SHORT}
    vals = sorted({str(cl[pipe][v]["axes"][a]["value"]) for v in sets["headline"].get(pipe, []) for a in axes
                   if a in cl[pipe][v]["axes"]})
    return "no eligible variant has I-CARE rows" + (
        f"; the stated {' / '.join(vals)} do not fit the I-CARE windows" if vals else "")


sr.no_rows_note = no_rows_note


def assess_reading(reading, cl_ref, cl, sets, choices, rows, B):
    """145.assess_reading on the arm's rows, with one difference: where the variants with a stated call rule have no
    I-CARE rows (the 30-min window holds only shortened-to-fit twins of Cruse 2011 and Cruse 2012), the entry is
    recorded with the reason instead of refused. Holm: none (estimation only)."""
    entries = {}
    for (nul, pipe), (vp, dflt) in sorted(choices.items()):
        dp = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
        coded = {v: bool(mg._flag(g, "as_coded").any()) for v, g in dp.groupby("variant")}
        present = set(dp["variant"])
        cand, label, rule = sr.reading_candidates(cl_ref, sets, pipe, reading)
        cands = [v for v in cand if v in present]
        e, _ = sr._assess(dp, nul, pipe, cands, coded, B, label)
        if not cands:
            e["note"] = no_rows_note(cl, sets, pipe, nul)
        pb = mg._null_seq(dp)
        pb = pb[pb["j"] == 0]
        e.update(null=nul, pipe=pipe, reading=reading, call_rule=rule,
                 beside={"rows": "the arm's rows (122)",
                         "validity_primary": {"variant": vp, **sr.exact_rate(pb[pb["variant"] == vp])},
                         "c0_default": {"variant": dflt, "c0": st.C0.get(pipe, "not graded by C0"),
                                        **(sr.exact_rate(pb[pb["variant"] == dflt]) if (pb["variant"] == dflt).any()
                                           else {"note": "the default has no I-CARE rows"})}})
        if sr.reading_flag(reading, pipe):
            e["flag"] = sr.reading_flag(reading, pipe)
        if sr.is_covert(pipe):
            e.update(covert_processing=True, label=sr.COVERT_LABEL, reading_label=label)
        entries[f"{nul}|{pipe}"] = e
    return entries


def side_rows(sets, choices, rows, B):
    """The shortened-to-fit readings (never in the headline), as 145.main reports them."""
    side = {}
    for (nul, pipe), _ in sorted(choices.items()):
        dp = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
        coded = {v: bool(mg._flag(g, "as_coded").any()) for v, g in dp.groupby("variant")}
        sv = [v for v in sets["shortened"].get(pipe, []) if v in set(dp["variant"])]
        if sv:
            side[f"{nul}|{pipe}"] = sr._assess(dp, nul, pipe, sv, coded, B, "shortened to fit (never in the headline)")[0]
    return side


def c3_frame():
    fn = os.path.join(C3_DIR, "rows.jsonl")
    c3 = pd.DataFrame([json.loads(ln) for ln in open(fn) if ln.strip()])
    c3["patient"], c3["uid"] = c3["patient"].astype(str), c3["uid"].astype(str)
    return c3


def icare_rows(pats):
    d = ic.load_rows(pats)
    d["patient"], d["uid"] = d["patient"].astype(str), d["uid"].astype(str)
    return d


def randomised_rate(dp_all, nul, pipe, variant):
    """rand_call of the reading's first-overlay rows (a C3 reading: its base variant's rows), exact CI."""
    base = fid.base(pipe, variant) if variant in fid.READINGS.get(pipe, {}) else (
        "as_coded" if variant == sr.CALL_PAPER else variant)
    p = mg._null_seq(dp_all)
    p = p[(p["null"] == nul) & (p["pipe"] == pipe) & (p["j"] == 0) & (p["variant"] == base)]
    r = p[p["rand_call"].notna()].drop_duplicates("patient")
    if not len(r):
        return {"variant_base": base, "note": "no randomised rows"}
    return dict(st.exact_cell(int(r["rand_call"].astype(bool).sum()), int(len(r))), variant_base=base)


def merge(B=sr.B_BOOT):
    pats, _ = model_patients()
    df = icare_rows(pats)
    c3 = c3_frame()
    c3 = c3[c3["patient"].isin(pats)]
    n_c3 = {"rows": int(len(c3)), "ok": int((c3["status"] == "ok").sum()),
            "not_reproduced": int((c3["status"] == "not_reproduced").sum()),
            "patients": int(c3["patient"].nunique()), "by_setup": c3.groupby("recording_setup")["patient"].nunique().to_dict()}
    if c3["patient"].nunique() != len(pats):
        raise RuntimeError(f"C3 rows for {c3['patient'].nunique()} of {len(pats)} patients: run --c3 to completion")
    c3 = c3[c3["status"] != "not_reproduced"]
    rows, n_dropped = sr.build_rows(df, c3)
    with open(os.path.join(ROOT, "output", "tables", "icare_arm.json")) as fh:
        arm = json.load(fh)
    fc = arm["false_call_rates"]["per_pipeline"]
    choices = {(e["null"], pipe): (e["validity_primary"]["variant"], e["default"]["variant"]) for pipe, e in fc.items()}
    cl = sr.load_classification()
    cl_ref = sr.load_classification_at(sr.CLASS_REF)
    sets = sr.require_strict(sr.classification_sets(cl))
    readings = {r: assess_reading(r, cl_ref, cl, sets, choices, rows, B) for r in sr.READINGS}
    for ents in readings.values():
        for key, e in ents.items():
            if e.get("variant"):
                e["randomised"] = randomised_rate(rows, e["null"], e["pipe"], e["variant"])
    out = {"what": __doc__.split("\n")[0], "added_later": True, "B": B, "n_patients": len(pats),
           "sha256_150": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "sha256_122": ic.own_sha256(), "classification": {os.path.relpath(p, ROOT): sr._sha256(p) for p in sr.FILES},
           "class_reference": sr.CLASS_REF, "c3_rows": n_c3, "rows_dropped_later_recordings": n_dropped,
           "randomised_pooled": {"label": "115.proposed_validity on the arm's rows (every overlay, as on BDSP)",
                                 **mg.proposed_validity(df)},
           "entries": readings["primary"], "side_rows": side_rows(sets, choices, rows, B),
           "sensitivity_generous": readings["sensitivity_generous"],
           "sensitivity_strict": readings["sensitivity_strict"]}
    out = mg._jsonable(out)
    sr.assert_no_identifiers(out)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"150: {len(readings['primary'])} entries, {len(pats)} patients -> {OUT}")
    return out


def all_windows_rates():
    """Sensitivity: each primary reading selected on the model set, read on every patient with a processed window
    (the model set plus the patients without a complete covariate or mechanism value). No selection is redone and no
    outcome is read. The model-set subset must reproduce icare_stated_rule.json exactly.
    -> output/tables/icare_stated_rule_all_windows.json."""
    pats, model = window_patients(), set(model_patients()[0])
    df = icare_rows(pats)
    c3 = c3_frame()
    c3 = c3[c3["patient"].isin(pats)]
    if c3["patient"].nunique() != len(pats):
        raise RuntimeError(f"C3 rows for {c3['patient'].nunique()} of {len(pats)} patients: run --c3 --all-windows")
    c3 = c3[c3["status"] != "not_reproduced"]
    rows, _ = sr.build_rows(df, c3)
    with open(OUT) as fh:
        prim = json.load(fh)["entries"]
    out = {}
    for key, e in prim.items():
        v = e.get("variant")
        if not v:
            continue
        nul, pipe = key.split("|")
        pb = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe)])
        sel = pb[(pb["j"] == 0) & (pb["variant"] == v)]
        sub = sr.exact_rate(sel[sel["patient"].isin(model)])
        if (sub["k"], sub["n"]) != (e["stated_rule"]["k"], e["stated_rule"]["n"]):
            raise RuntimeError(f"{key}: model-set subset {sub['k']}/{sub['n']} differs from "
                               f"{e['stated_rule']['k']}/{e['stated_rule']['n']}")
        out[key] = {"variant": v, "all_windows": sr.exact_rate(sel), "model_set": sub}
    res = mg._jsonable({"what": all_windows_rates.__doc__.split("\n")[0], "n_patients": len(pats),
                        "n_model_set": len(model), "entries": out})
    sr.assert_no_identifiers(res)
    fn = os.path.join(ROOT, "output", "tables", "icare_stated_rule_all_windows.json")
    with open(fn, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"150 all windows: {len(pats)} patients, {len(out)} readings -> {fn}")
    return res


def by_background():
    """Descriptive: each primary reading's first-overlay false calls by the ACNS continuity grade of the window
    (122.patient_table; no outcome or covariate column is read). -> output/tables/icare_by_background.json."""
    index = ic.load_index()
    t = ic.patient_table(ic.analysis_order(index, ic.dev_patients(index)), index)
    kept, _ = ic.model_set(t)
    grade = dict(zip(kept["patient"].astype(str), kept["acns_grade"]))
    pats = sorted(grade)
    c3 = c3_frame()
    c3 = c3[c3["patient"].isin(pats) & (c3["status"] != "not_reproduced")]
    rows, _ = sr.build_rows(icare_rows(pats), c3)
    with open(OUT) as fh:
        entries = json.load(fh)["entries"]
    out = {}
    for key, e in entries.items():
        if not e.get("variant"):
            continue
        nul, pipe = key.split("|")
        p = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe)])
        p = p[(p["j"] == 0) & (p["variant"] == e["variant"]) & (p["status"] == "ok")].drop_duplicates("patient")
        g = p["patient"].astype(str).map(grade)
        cells = {str(k): st.exact_cell(int(x["call"].astype(bool).sum()), int(len(x))) for k, x in p.groupby(g)}
        cells["all"] = st.exact_cell(int(p["call"].astype(bool).sum()), int(len(p)))
        if cells["all"]["k"] != e["stated_rule"]["k"] or cells["all"]["n"] != e["stated_rule"]["n"]:
            raise RuntimeError(f"{key}: by-background total does not reproduce the primary rate")
        out[pipe] = {"variant": e["variant"], "by_acns_grade": cells}
    out = mg._jsonable({"what": "primary-reading false calls by ACNS continuity grade (descriptive, added later)",
                        "sha256_150": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                        "pipelines": out})
    sr.assert_no_identifiers(out)
    fn = os.path.join(ROOT, "output", "tables", "icare_by_background.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"150: by background -> {fn}")
    return out


def background_cells(calls, grades):
    """calls: one row per patient with columns patient and call; grades: patient -> ACNS grade or None.
    -> {grade: exact cell, ..., "all": exact cell}; a patient without a grade counts only in "all"."""
    g = calls["patient"].astype(str).map(grades)
    cells = {str(k): st.exact_cell(int(x["call"].astype(bool).sum()), int(len(x))) for k, x in calls.groupby(g)}
    cells["all"] = st.exact_cell(int(calls["call"].astype(bool).sum()), int(len(calls)))
    return cells


def by_background_all_windows():
    """Descriptive: each primary reading's first-overlay false calls by the ACNS continuity grade of the window, on
    every patient with a processed window (the model set plus the patients without a complete covariate or mechanism
    value). No outcome or covariate column is used. The totals must reproduce icare_stated_rule_all_windows.json.
    -> output/tables/icare_by_background_all_windows.json."""
    index = ic.load_index()
    t = ic.patient_table(ic.analysis_order(index, ic.dev_patients(index)), index)
    inc = t[t["status"] == "included"]
    grade = {str(p): (g if isinstance(g, str) else None) for p, g in zip(inc["patient"], inc["acns_grade"])}
    pats = sorted(grade)
    c3 = c3_frame()
    c3 = c3[c3["patient"].isin(pats) & (c3["status"] != "not_reproduced")]
    rows, _ = sr.build_rows(icare_rows(pats), c3)
    with open(os.path.join(ROOT, "output", "tables", "icare_stated_rule_all_windows.json")) as fh:
        aw = json.load(fh)["entries"]
    out = {}
    for key, e in aw.items():
        nul, pipe = key.split("|")
        p = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe)])
        p = p[(p["j"] == 0) & (p["variant"] == e["variant"]) & (p["status"] == "ok")].drop_duplicates("patient")
        cells = background_cells(p, grade)
        want = e["all_windows"]
        if (cells["all"]["k"], cells["all"]["n"]) != (want["k"], want["n"]):
            raise RuntimeError(f"{key}: background total {cells['all']['k']}/{cells['all']['n']} differs from the "
                               f"all-windows rate {want['k']}/{want['n']}")
        out[pipe] = {"variant": e["variant"], "by_acns_grade": cells}
    out = mg._jsonable({"what": "primary-reading false calls by ACNS continuity grade, every patient with a "
                                "processed window (descriptive)", "n_patients": len(pats), "pipelines": out})
    sr.assert_no_identifiers(out)
    fn = os.path.join(ROOT, "output", "tables", "icare_by_background_all_windows.json")
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"150: by background, all windows ({len(pats)} patients) -> {fn}")
    return out


def primary_call_outcome(B=None):
    """Added later: 122.call_status_vs_outcome with each analysis's PRIMARY reading (this file's selection) in place of
    its a priori default: the first-overlay call against good outcome with the primary model's adjustment (no X_ref),
    OR with a patient-bootstrap CI; never ranked. Outcome and covariates are read here, on this computer only.
    -> output/tables/icare_primary_call_outcome.json."""
    B = B or ic.PRIMARY_B
    mx = ic._mx()
    index = ic.load_index()
    t = ic.patient_table(ic.analysis_order(index, ic.dev_patients(index)), index)
    kept, _ = ic.model_set(t)
    pats = sorted(kept["patient"].astype(str))
    c3 = c3_frame()
    c3 = c3[c3["patient"].isin(pats) & (c3["status"] != "not_reproduced")]
    rows, _ = sr.build_rows(icare_rows(pats), c3)
    with open(OUT) as fh:
        entries = json.load(fh)["entries"]
    d = kept.assign(patient=kept["patient"].astype(str))
    out = {}
    for key, e in entries.items():
        nul, pipe = key.split("|")
        if not e.get("variant"):
            out[pipe] = {"note": "no primary reading"}
            continue
        p = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe)])
        g = p[(p["j"] == 0) & (p["variant"] == e["variant"]) & (p["status"] == "ok")].drop_duplicates("patient")
        call = d["patient"].map(dict(zip(g["patient"].astype(str), g["call"].astype(float))))
        sel = call.notna().to_numpy()
        ed = d[sel]
        frame, covs, _ = ic.design_matrix(ed)
        covs = {"call_status": call[sel].to_numpy(float), **{k: v for k, v in covs.items() if k != "z_xref"}}
        n1 = int(covs["call_status"].sum())
        if n1 != e["stated_rule"]["k"] or len(ed) != e["stated_rule"]["n"]:
            raise RuntimeError(f"{key}: {n1}/{len(ed)} does not reproduce the primary rate")
        r = mx.fe_model(frame, covs, B, f"icare_call_primary_{pipe}", None, "fe")
        c = r.get("coef", {}).get("call_status")
        out[pipe] = {"variant": e["variant"], "n_patients": int(len(ed)), "n_false_calls": n1,
                     "sparse": bool(n1 < 10 or len(ed) - n1 < 10),
                     **({"or": c["or"], "or_ci95": c["or_ci95"], "p_two_sided": c.get("p_two_sided")} if c
                        else {"note": "fit did not converge"})}
    res = mg._jsonable({"what": "added later: primary-reading false call vs good outcome (122 adjustment, no X_ref)",
                        "B": B, "sha256_150": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                        "pipelines": out})
    sr.assert_no_identifiers(res)
    fn = os.path.join(ROOT, "output", "tables", "icare_primary_call_outcome.json")
    with open(fn, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"150: primary-reading call vs outcome -> {fn}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--c3", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--by-background", action="store_true")
    ap.add_argument("--by-background-all", action="store_true")
    ap.add_argument("--primary-outcome", action="store_true")
    ap.add_argument("--all-windows", action="store_true",
                    help="with --c3: run every patient with a processed window; alone: the all-windows sensitivity")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--B", type=int, default=sr.B_BOOT)
    a = ap.parse_args(argv)
    if a.c3:
        run_c3(a.workers, a.limit, a.all_windows)
    elif a.all_windows:
        all_windows_rates()
    if a.merge:
        merge(a.B)
    if a.by_background:
        by_background()
    if a.by_background_all:
        by_background_all_windows()
    if a.primary_outcome:
        primary_call_outcome()


if __name__ == "__main__":
    main()
