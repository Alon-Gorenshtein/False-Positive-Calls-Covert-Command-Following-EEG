"""Per-patient accumulation of the vibrotactile P300's primary reading over repeated runs in a FIXED cohort.

145 reports the best-run share at k = 1, 3, 5, 7 among the patients with at least k runs, so the population changes
with k. Here the cohort is fixed: the clinical EEG patients with at least K_MAX usable sequence windows of the primary
reading, and every k is read on those same patients (any positive run among the first k, 115's rule). Rows are 145's:
the primary variant from output/tables/stated_rule.json, every overlay of the chosen recording (145.composite_rows).
Checks: the unrestricted counts reproduce stated_rule.json (two-run composite and the best-of-k curve).
-> output/tables/fixed_cohort_runs.json (no patient identifier)."""
import importlib
import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
sr = importlib.import_module("145_stated_rule")
mg, st = sr.mg, sr.st

KEY = "nstim0_bdsp|mbvt"
KS = (1, 2, 3, 5, 7)
K_MAX = 7
OUT = os.path.join(ROOT, "output", "tables", "fixed_cohort_runs.json")


def cohort_curve(g, rule, ks=KS, k_max=K_MAX):
    """Best-run share at each k on the patients with >= k_max windows; plus the unrestricted share at each k."""
    cohort = set(mg.patient_outcomes(g, rule, k_max))
    fixed, open_ = [], []
    for k in ks:
        o = mg.patient_outcomes(g, rule, k)
        v = [o[p] for p in sorted(cohort)]
        fixed.append(dict(st.exact_cell(int(sum(v)), len(v)), runs=k))
        open_.append(dict(st.exact_cell(int(sum(o.values())), len(o)), runs=k))
    return len(cohort), fixed, open_


def main():
    with open(os.path.join(ROOT, "output", "tables", "stated_rule.json")) as fh:
        sr_out = json.load(fh)
    nul, pipe = KEY.split("|")
    v = sr_out["entries"][KEY]["variant"]
    rows, _ = sr.build_rows(st.analysis_rows(), st.c3_rows())
    mine = rows[(rows["null"] == nul) & (rows["pipe"] == pipe)]
    frozen = sr.load_all_variant_rows({(nul, pipe): {v}}, set())
    g = sr.composite_rows(nul, pipe, v, None, frozen, None, set(mine["uid"]))
    rule = mg._rule(mg._load_pipe(pipe), rule="best_run")
    n, fixed, open_ = cohort_curve(g, rule)
    pp = sr_out["per_patient_at_published_k"][KEY]
    two = next(c for c in open_ if c["runs"] == 2)
    if (pp["composite"]["k"], pp["composite"]["n"]) != (two["k"], two["n"]):
        raise RuntimeError(f"two-run check: {pp['composite']} vs {two}")
    for c in pp["best_of_k_descriptive"]:
        mine_c = next((x for x in open_ if x["runs"] == c["k"]), None)
        if mine_c and c["n"] and (c["n"], round(c["rate"], 9)) != (mine_c["n"], round(mine_c["rate"], 9)):
            raise RuntimeError(f"best-of-k check at k={c['k']}: {c['n']}, {c['rate']} vs {mine_c}")
    out = mg._jsonable({"what": __doc__.split("\n")[0], "entry": KEY, "variant": v, "k_max": K_MAX,
                        "cohort_n": n, "fixed_cohort": fixed, "patients_with_at_least_k": open_})
    sr.assert_no_identifiers(out)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
