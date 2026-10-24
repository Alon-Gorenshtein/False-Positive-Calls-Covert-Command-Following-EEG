"""Post-arrest coma: how many patients with a poor outcome (Cerebral Performance Category 3 to 5) the published call
rules would call positive for command-following, with no command given. Each analysis's primary reading, first
overlay of the window nearest 24 hours, the 490 patients of the outcome model; and the complete-case share called by
any of the 3 analyses above 5% (Cornell, du Bois, vibrotactile P300). Exposure only: how clinicians would act on such a
call is not known. Outcome is read here, on this computer only. -> output/tables/poor_outcome_exposure.json"""
import hashlib
import importlib
import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
ic = importlib.import_module("122_icare_arm")
ics = importlib.import_module("150_icare_stated_rule")
sr = importlib.import_module("145_stated_rule")
st = importlib.import_module("143_stratum_table")
mg = sr.mg
OUT = os.path.join(ROOT, "output", "tables", "poor_outcome_exposure.json")
THREE = ("cornell", "dubois", "mbvt")


def _calls():
    index = ic.load_index()
    t = ic.patient_table(ic.analysis_order(index, ic.dev_patients(index)), index)
    kept, _ = ic.model_set(t)
    pats = sorted(kept["patient"].astype(str))
    c3 = ics.c3_frame()
    c3 = c3[c3["patient"].isin(pats) & (c3["status"] != "not_reproduced")]
    rows, _ = sr.build_rows(ics.icare_rows(pats), c3)
    entries = json.load(open(ics.OUT))["entries"]
    good = dict(zip(kept["patient"].astype(str), kept["outcome_good"].astype(bool)))
    calls = {}
    for key, e in entries.items():
        nul, pipe = key.split("|")
        if not e.get("variant"):
            continue
        p = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe)])
        g = p[(p["j"] == 0) & (p["variant"] == e["variant"]) & (p["status"] == "ok")].drop_duplicates("patient")
        calls[pipe] = (nul, {str(a): bool(float(b)) for a, b in zip(g["patient"], g["call"])})
    return good, calls


def exposure():
    good, calls = _calls()
    res = {"n_good": int(sum(good.values())), "n_poor": int(sum(not v for v in good.values())), "pipes": {}}
    for pipe, (nul, c) in calls.items():
        poor = [v for pt, v in c.items() if not good[pt]]
        gd = [v for pt, v in c.items() if good[pt]]
        res["pipes"][pipe] = {"null": nul, "poor": st.exact_cell(sum(poor), len(poor)),
                              "good": st.exact_cell(sum(gd), len(gd)),
                              "per_1000_poor": 1000 * sum(poor) / len(poor) if poor else None}
    both = set.intersection(*(set(calls[p][1]) for p in THREE))
    anyc = {pt: any(calls[p][1][pt] for p in THREE) for pt in both}
    for lab, want in (("poor", False), ("good", True)):
        v = [x for pt, x in anyc.items() if good[pt] == want]
        res.setdefault("any_of_three", {})[lab] = st.exact_cell(sum(v), len(v))
    return res


def main(out=OUT):
    res = dict(exposure(), what=__doc__.split("\n")[0],
               sha256_160=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    sr.assert_no_identifiers(res)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({p: (r["poor"]["k"], r["poor"]["n"], round(r["poor"]["rate"], 3)) for p, r in res["pipes"].items()},
                     indent=1), res["any_of_three"])


if __name__ == "__main__":
    main()
