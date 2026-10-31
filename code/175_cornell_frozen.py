"""Frozen replication of the Cornell command-level jackknife (171) on windows it was not developed on: the 730
replication windows (153) and every post-arrest window (150). FROZEN below, written to
output/tables/cornell_command_frozen.json by --freeze and committed before any replication or post-arrest window is
run; nothing in it changes after a row is read.

Units, one per patient, the first published Cornell overlay of the c3 reading (130.phase1_jobs, first_only):
- replication: 153.items() (the 730 listed windows), Recording by 153.load_without_morgoth, n_recs = the listing size
  as 153.run computes it;
- post-arrest: 150.window_patients() (every patient with a processed window), Recording rebuilt from the kept window
  by 150.recording with 122's padding context, n_recs as 150.run_c3 computes it.
Each unit runs 171.run_unit (all readings on the same features). Reproduction: the sample-level reading must equal the
stored c3 call of that window (replication: 153's c3 rows; post-arrest: 150's _icare_c3 rows, status ok); a unit that
does not, or has no stored call, leaves every denominator and is counted.
Tests fixed here: per cohort, one-sided exact binomial of the primary reading's count against 5% (H1: below 5%), Holm
over the 2 cohorts, 'calibrated' if the Holm-adjusted P < .05; paired exact McNemar of the sample-level reading against
the primary. Descriptive strata: post-arrest ACNS continuity grade (122 patient table); development MORGOTH strata
(the window's MORGOTH background) on 171's development rows. The output carries no identifier.
python 175_cornell_frozen.py --freeze | --cohort rep|icare [--workers W] [--limit N] | --merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys

import numpy as np
from scipy.stats import beta, binom

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
tj = importlib.import_module("171_cornell_trialjack")

FROZEN = {"primary": "command_normal", "secondary": ["command_t", "command_t_dec"], "check": "sample_normal",
          "alpha": 0.05, "bar": 0.05, "cohorts": ["rep", "icare"], "reading": tj.READING,
          "test": "one-sided exact binomial of the primary count against the bar (H1: below), Holm over cohorts"}
FREEZE = os.path.join(ROOT, "output", "tables", "cornell_command_frozen.json")
INTER = os.path.join(ROOT, "output", "intermediate")
OUT = os.path.join(ROOT, "output", "tables", "cornell_frozen_replication.json")


def rows_path(cohort):
    return os.path.join(INTER, f"cornell_frozen_{cohort}.jsonl")


def _cp(k, n):
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def p_below(k, n, bar=FROZEN["bar"]):
    """One-sided exact binomial P of a count this low or lower if the rate were the bar."""
    return float(binom.cdf(k, n, bar))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def mcnemar(a, c):
    """Exact two-sided McNemar on paired booleans (a = first reading, c = second)."""
    a, c = np.asarray(a, bool), np.asarray(c, bool)
    b_, c_ = int(np.sum(a & ~c)), int(np.sum(~a & c))
    p = float(min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5))) if b_ + c_ else 1.0
    return {"first_only": b_, "second_only": c_, "p_exact": p}



_REP = {}


def _rep_items():
    if not _REP:
        import pandas as pd
        rp = importlib.import_module("153_replication_rows")
        _REP["rp"] = rp
        _REP["items"] = {it["uid"]: it for it in rp.items()}
        _REP["n_recs"] = len(pd.read_csv(rp.MANIFEST))
        stored = {}
        for r in rp.load_rows("cornell"):
            if r.get("reading") == tj.READING and r.get("j") == 0 and r.get("status") == "ok":
                stored[r["uid"]] = bool(r["call"])
        _REP["stored"] = stored
    return _REP


def rep_unit(uid):
    S = _rep_items()
    rp = S["rp"]
    fid = importlib.import_module("130_fidelity_readings")
    rec = rp.load_without_morgoth(S["items"][uid])
    mod = rp.b.load_pipe("cornell")
    jobs, why = fid.phase1_jobs(rec, mod, rp.rep_spec("cornell"), S["n_recs"], first_only=True)
    jb = [j for j in jobs if j["reading"] == tj.READING and j["ov"] is not None]
    if not jb:
        return [{"unit": uid, "cohort": "rep", "status": why or "no_published_overlay"}]
    return mark(tj.run_unit(rec, jb[0]["ov"], uid), S["stored"].get(uid), "rep")


_IC = {}


def _icare():
    if not _IC:
        ic = importlib.import_module("150_icare_stated_rule")
        rf = importlib.import_module("133_icare_refetch")
        _IC["ic"], _IC["kw"] = ic, rf.keep_dirs()[0]
        _IC["n_recs"] = ic.model_patients()[1]
        stored = {}
        with open(os.path.join(ic.C3_DIR, "rows.jsonl")) as fh:
            for ln in fh:
                r = json.loads(ln)
                if (r.get("pipe") == "cornell" and r.get("reading") == tj.READING and r.get("j") == 0
                        and r.get("status") == "ok"):
                    stored[r["patient"]] = bool(r["call"])
        _IC["stored"] = stored
    return _IC


def icare_unit(pid):
    S = _icare()
    ic, arm = S["ic"], S["ic"].ic
    fid = importlib.import_module("130_fidelity_readings")
    x_rec = arm.read_data_json(os.path.join(arm.paths("analysis")["x"], f"{pid}.json"))
    rec, pad, _ = ic.recording(pid, x_rec, S["kw"])
    try:
        if pad is not None:
            arm.set_padding_context(rec.x, pad)
        mod = ic.rn.load_pipe("cornell")
        spec = {"name": ic.NULL, "kind": "ncmd", "family": mod.FAMILY, "source": "bdsp"}
        jobs, why = fid.phase1_jobs(rec, mod, spec, S["n_recs"], first_only=True)
        jb = [j for j in jobs if j["reading"] == tj.READING and j["ov"] is not None]
        if not jb:
            return [{"unit": pid, "cohort": "icare", "status": why or "no_published_overlay"}]
        return mark(tj.run_unit(rec, jb[0]["ov"], pid), S["stored"].get(pid), "icare")
    finally:
        if pad is not None:
            arm.clear_padding_context()


def mark(rows, stored, cohort):
    """Attach the cohort and the reproduction check (sample-level call vs the stored c3 call) to every row."""
    base = [r for r in rows if r["reading"] == FROZEN["check"]]
    ok = bool(base and base[0].get("call") is not None and stored is not None and bool(base[0]["call"]) == stored)
    for r in rows:
        r.update(cohort=cohort, stored_call=stored, reproduced=ok)
    return rows



def usable_units(rows):
    """{unit: {reading: call}} for units that reproduced their stored call; plus counts of those left out."""
    units, left = {}, {"no_overlay_or_error": 0, "no_usable_trials": 0, "not_reproduced": 0}
    by = {}
    for r in rows:
        by.setdefault(r["unit"], []).append(r)
    for u, rs in by.items():
        if any("reading" not in r for r in rs):
            left["no_overlay_or_error"] += 1
            continue
        calls = {r["reading"]: r.get("call") for r in rs}
        if calls.get(FROZEN["primary"]) is None or calls.get(FROZEN["check"]) is None:
            left["no_usable_trials"] += 1
            continue
        if not rs[0].get("reproduced"):
            left["not_reproduced"] += 1
            continue
        units[u] = calls
    return units, left


def cohort_summary(rows):
    units, left = usable_units(rows)
    out = {"n_units": len(units), "left_out": left, "readings": {}}
    names = [FROZEN["check"], FROZEN["primary"]] + FROZEN["secondary"]
    for name in names:
        c = [bool(v[name]) for v in units.values() if v.get(name) is not None]
        k, n = int(sum(c)), len(c)
        out["readings"][name] = {"k": k, "n": n, "rate": k / n if n else None, "ci95": _cp(k, n) if n else None}
    k, n = out["readings"][FROZEN["primary"]]["k"], out["readings"][FROZEN["primary"]]["n"]
    out["p_below_bar"] = p_below(k, n) if n else None
    u = list(units)
    out["mcnemar_sample_vs_primary"] = mcnemar([units[x][FROZEN["check"]] for x in u],
                                               [units[x][FROZEN["primary"]] for x in u])
    return out, units


def strata_cells(units, lab):
    """{stratum: {reading: {k, n, ci95}}} for the check and primary readings; units without a label are left out."""
    out = {}
    for g in sorted({v for v in lab.values() if v is not None}):
        sel = [units[u] for u in units if lab.get(u) == g]
        out[g] = {}
        for name in (FROZEN["check"], FROZEN["primary"]):
            k, n = int(sum(bool(v[name]) for v in sel)), len(sel)
            out[g][name] = {"k": k, "n": n, "ci95": _cp(k, n) if n else None}
    return out


def summarize(rows_by_cohort, strata=None):
    res = {"frozen": FROZEN, "cohorts": {}}
    units_by = {}
    for c in FROZEN["cohorts"]:
        if c in rows_by_cohort:
            res["cohorts"][c], units_by[c] = cohort_summary(rows_by_cohort[c])
    ps = [res["cohorts"][c]["p_below_bar"] for c in res["cohorts"]]
    for c, a in zip(res["cohorts"], holm(ps) if ps else []):
        res["cohorts"][c]["p_below_bar_holm"] = a
        res["cohorts"][c]["calibrated"] = bool(a < FROZEN["alpha"])
    res["strata"] = {c: strata_cells(units_by[c], lab) for c, lab in (strata or {}).items() if c in units_by}
    return res


def dev_units(dev):
    """{unit: {reading: call}} of 171's development rows whose sample-level call matched the stored one."""
    by, ok = {}, {}
    for r in dev:
        by.setdefault(r["unit"], {})[r["reading"]] = r.get("call")
        if r["reading"] == FROZEN["check"]:
            ok[r["unit"]] = bool(r.get("matches_stored"))
    return {u: c for u, c in by.items() if ok.get(u) and c.get(FROZEN["primary"]) is not None
            and c.get(FROZEN["check"]) is not None}


def icare_grades():
    ic = importlib.import_module("122_icare_arm")
    index = ic.load_index()
    t = ic.patient_table(ic.analysis_order(index, ic.dev_patients(index)), index)
    inc = t[t["status"] == "included"]
    return {str(p): (g if isinstance(g, str) else None) for p, g in zip(inc["patient"], inc["acns_grade"])}


def dev_strata():
    """The window's MORGOTH background of the 171 development windows (143's first-overlay Cornell rows): awake,
    sleep, mixed or ICU-typical (143.DISORGANISED). Descriptive; 143.stratum_mask's activation filter needs its full
    pipeline and is not applied."""
    st = importlib.import_module("143_stratum_table")
    d = st.load_first_overlays()
    d = d[(d["null"] == "ncmd_bdsp") & (d["pipe"] == "cornell")].drop_duplicates("uid")
    name = {"normal_awake": "awake", "sleep": "sleep", "mixed": "mixed"}
    return {str(u): ("icu_typical" if s in st.DISORGANISED else name.get(s)) for u, s in zip(d["uid"], d["stratum"])}


def _pool(fn, items, workers):
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(fn, items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--cohort", choices=("rep", "icare"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        if os.path.exists(FREEZE):
            raise SystemExit(f"{FREEZE} exists; the freeze is written once")
        json.dump({"what": __doc__.split("\n")[0], **FROZEN}, open(FREEZE, "w"), indent=1)
        print("frozen:", FREEZE)
        return
    frozen = json.load(open(FREEZE))
    if {k: frozen[k] for k in FROZEN} != FROZEN:
        raise SystemExit("FROZEN differs from the committed freeze record")
    if a.merge:
        rows = {}
        for c in FROZEN["cohorts"]:
            if os.path.exists(rows_path(c)):
                rows[c] = [json.loads(l) for l in open(rows_path(c)) if l.strip()]
        dev = [json.loads(l) for l in open(os.path.join(INTER, "cornell_trialjack_dev.jsonl")) if l.strip()]
        res = summarize(rows, strata={"icare": icare_grades()})
        res["dev_strata"] = strata_cells(dev_units(dev), dev_strata())
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(json.dumps({c: {k: v for k, v in res["cohorts"][c].items() if k != "readings"} |
                          {"primary": res["cohorts"][c]["readings"][FROZEN["primary"]]}
                          for c in res["cohorts"]}, indent=1, default=float))
        return
    if a.cohort == "rep":
        todo = sorted(_rep_items()["items"])
        fn = rep_unit
    else:
        todo = _icare()["ic"].window_patients()
        fn = icare_unit
    path = rows_path(a.cohort)
    done = {json.loads(l)["unit"] for l in open(path)} if os.path.exists(path) else set()
    todo = [u for u in todo if u not in done][:a.limit]
    with open(path, "a") as fh:
        for k, rows in enumerate(_pool(fn, todo, a.workers)):
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            print(k + 1, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
