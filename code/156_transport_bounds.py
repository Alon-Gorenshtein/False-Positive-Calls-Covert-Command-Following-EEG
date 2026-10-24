"""Supporting tables, from existing outputs only (no new runs, no identifiers):
- transport: the clinical-EEG primary reading of each analysis applied unchanged to post-arrest coma (I-CARE rows of
  the same reading), against the reading I-CARE's own selection picked;
- usability: worst-case bounds of the primary rate when every unusable recording is counted negative or positive;
- categories: the range of eligible readings three ways: every text-consistent reading; technically sound readings
  (no flag below); and the fidelity-corrected readings;
- decision_summary: per analysis, false calls under the published per-patient rule and detection measured under that
  same reading (None where not measured, never 0);
- false_share: the illustrative share of positive calls that would be false, assuming perfect detection.
-> output/tables/transport_bounds.json."""
import importlib
import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
T = os.path.join(ROOT, "output", "tables")
OUT = os.path.join(T, "transport_bounds.json")
PREVALENCE = (0.10, 0.25, 0.40)
RATES = (0.05, 0.138, 0.222, 0.233, 0.245)

BLOCK_DESIGN_RANDOM_FOLDS = {
    "dubois": lambda v: "assessment_random" in v,
    "edlow17": lambda v: "_cvsecond" in v,
}
SPATIAL_FILTER_OUTSIDE_FOLDS = {
    "mbmi": lambda v: "cspglobal" in v,
}
SELECTION_ON_TEST_DATA = {
    "mbmi": lambda v: v.endswith("_peak"),
    "mbvt": lambda v: "_max_" in v,
}


def _load(name):
    with open(os.path.join(T, name)) as fh:
        return json.load(fh)


def _bdsp_entries():
    return {k.split("|")[1]: e for k, e in _load("stated_rule.json")["entries"].items()
            if k.split("|")[0] in ("ncmd_bdsp", "nstim0_bdsp") and e.get("variant")}


def _cell(c):
    return None if c is None else {"k": c["k"], "n": c["n"], "rate": c["rate"]}


def transport():
    ic = {k.split("|")[1]: e for k, e in _load("icare_stated_rule.json")["entries"].items()}
    out = {}
    for pipe, e in _bdsp_entries().items():
        if pipe not in ic:
            continue
        rates = (ic[pipe].get("eligible_range") or {}).get("rates", {})
        out[pipe] = {"clinical_variant": e["variant"], "icare_selected": ic[pipe].get("variant"),
                     "same": e["variant"] == ic[pipe].get("variant"),
                     "icare_cell_of_clinical_variant": _cell(rates.get(e["variant"]))}
    return out


def usability():
    out = {}
    for pipe, e in _bdsp_entries().items():
        v = e["variants"][e["variant"]]
        k, usable, att = e["stated_rule"]["k"], v["n_ok_rows"], v["n_attempted"]
        out[pipe] = {"k": k, "usable": usable, "attempted": att, "worst_low": k / att,
                     "worst_high": (k + att - usable) / att}
    return out


def flags(pipe, variant):
    f = []
    if BLOCK_DESIGN_RANDOM_FOLDS.get(pipe, lambda v: False)(variant):
        f.append("random_folds_on_block_design")
    if SPATIAL_FILTER_OUTSIDE_FOLDS.get(pipe, lambda v: False)(variant):
        f.append("spatial_filter_outside_folds")
    if SELECTION_ON_TEST_DATA.get(pipe, lambda v: False)(variant):
        f.append("selection_on_test_data")
    return f


def _range(rates):
    rs = [c["rate"] for c in rates.values() if c.get("rate") is not None]
    return {"min": min(rs) if rs else None, "max": max(rs) if rs else None, "n": len(rs)}


def categories():
    out = {}
    for pipe, e in _bdsp_entries().items():
        rates = (e.get("eligible_range") or {}).get("rates", {})
        vf = {v: flags(pipe, v) for v in rates}
        c3 = {v: c for v, c in rates.items() if v.startswith("c3_")}
        out[pipe] = {"text_consistent": _range(rates),
                     "technically_sound": _range({v: c for v, c in rates.items() if not vf[v]}),
                     "fidelity_corrected": _range(c3) if c3 else None,
                     "variant_flags": vf, "variant_cells": {v: _cell(c) for v, c in rates.items()}}
    return out


def decision_summary():
    sr = _load("stated_rule.json")
    pp = sr["per_patient_at_published_k"]
    pc = _load("positive_control_stated.json")["variants"]
    out = {}
    for pipe, e in _bdsp_entries().items():
        key = f"{'nstim0' if pipe in ('mbvt', 'gibson16', 'chennu13', 'sokoliuk21') else 'ncmd'}_bdsp|{pipe}"
        p = pp.get(key) or {}
        if "composite" in p:
            false = _cell(p["composite"])
        elif "bracket" in p:
            false = {"bracket": {name: {"k": c.get("k"), "n": c.get("n", c.get("n_patients")), "rate": c["rate"]}
                                 for name, c in p["bracket"].items() if isinstance(c, dict) and "rate" in c}}
        else:
            false = _cell(e["stated_rule"])
        hd = (pc.get(pipe) or {}).get(e["variant"])
        out[pipe] = {"variant": e["variant"], "published_k": p.get("k"), "per_patient_false": false,
                     "healthy_detection": None if hd is None else _cell(hd["true"]),
                     "healthy_shuffled_calls": None if hd is None else _cell(hd["control"]),
                     "simulated_detection": None,
                     "note": "simulated detection was measured for size-corrected comparators, not at the published "
                             "call rule (eTable 14B)"}
    return out


def false_share(rate, prevalence):
    return rate * (1 - prevalence) / (rate * (1 - prevalence) + prevalence)


def main():
    out = {"transport": transport(), "usability": usability(), "categories": categories(),
           "decision_summary": decision_summary(),
           "false_share": [{"prevalence": pv, "rate": r, "share": false_share(r, pv)}
                           for pv in PREVALENCE for r in RATES]}
    s = json.dumps(out, indent=1, default=float)
    if "Bonobo" in s:
        raise RuntimeError("identifier in output")
    with open(OUT, "w") as fh:
        fh.write(s)
    for pipe, c in out["categories"].items():
        print(pipe, {k: c[k] for k in ("text_consistent", "technically_sound", "fidelity_corrected")})


if __name__ == "__main__":
    main()
