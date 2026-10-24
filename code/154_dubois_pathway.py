"""The du Bois decision pathway on recordings without study commands. The publication advances a patient "when DA
exceeded 70% or when task-related DAs were statistically greater than baseline (one-tailed paired t-test, alpha =
0.05)"; it calls the t-test alone a "significant run"; and it describes a 100-permutation label test as an
"Evaluation against chance", without stating that the test enters advancement. For the primary (fidelity-corrected,
leave-one-block-out) reading, four calls per first-overlay window:
- advancement: peak decoding accuracy > 0.70 or the paired t-test p < .05 (the primary call);
- significant run: the t-test alone;
- permutation: the fixed permutation reading's p < .05 at the fold-mean peak time, configuration fixed;
- advancement and permutation: both.
The permutation p comes from the fixed permutation row of the same window (variant assessment_blocked_perm). Its folds are
leave-one-block-out, so the configuration and the fold-mean peak time must equal the fidelity-corrected row's; a pair
with another window start, peak time or peak accuracy (the configuration's fingerprint) is counted as unmatched and left out of the gate's denominator;
the gate is also bounded with every unmatched window counted as negative and as a call. A pair whose permutation
row carries no peak accuracy is matched on window and peak time only, and counted (matched_without_peak_check).
Datasets: clinical EEG, prolonged DoC, post-arrest coma (the 490 model patients and every patient with a window), and
the replication set once its rows exist. -> output/tables/dubois_pathway.json (no identifiers)."""
import glob
import gzip
import importlib
import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
st = importlib.import_module("143_stratum_table")
sr = importlib.import_module("145_stated_rule")

READING = "c3_b_assessment_blocked_phase1"
PERM = "assessment_blocked_perm"
OUT = os.path.join(ROOT, "output", "tables", "dubois_pathway.json")
INTER = os.path.join(ROOT, "output", "intermediate")
TOL = 1e-9


def _ok(r):
    return r.get("status") == "ok"


def _j0(r):
    try:
        return int(float(r.get("j"))) == 0
    except (TypeError, ValueError):
        return False


def advancement(r):
    return bool(float(r["x_peak_da"]) > 0.70 or float(r["p"]) < 0.05)


def pathway(c3_rows, perm_rows):
    c3 = {r["uid"]: r for r in c3_rows if _ok(r) and _j0(r)}
    pm = {r["uid"]: r for r in perm_rows if _ok(r) and _j0(r)}
    adv = [advancement(r) for r in c3.values()]
    sig = [float(r["p"]) < 0.05 for r in c3.values()]
    perm = [float(r["x_p_perm"]) < 0.05 for r in pm.values()]
    gate, unmatched, no_peak = [], 0, 0
    for uid, r in c3.items():
        q = pm.get(uid)
        if (q is None or abs(float(q["start_s"]) - float(r["start_s"])) > TOL
                or abs(float(q["x_t_peak_s"]) - float(r["x_t_peak_s"])) > TOL
                or ("x_peak_da" in q and abs(float(q["x_peak_da"]) - float(r["x_peak_da"])) > TOL)):
            unmatched += 1
            continue
        no_peak += "x_peak_da" not in q or q["x_peak_da"] is None
        gate.append(advancement(r) and float(q["x_p_perm"]) < 0.05)
    n_all = len(gate) + unmatched
    return {"advancement": st.exact_cell(sum(adv), len(adv)), "significant_run": st.exact_cell(sum(sig), len(sig)),
            "permutation": st.exact_cell(sum(perm), len(perm)),
            "advancement_and_permutation": st.exact_cell(sum(gate), len(gate)),
            "gate_bounds": {"unmatched_as_negative": st.exact_cell(sum(gate), n_all),
                            "unmatched_as_call": st.exact_cell(sum(gate) + unmatched, n_all)},
            "matched": len(gate), "unmatched": unmatched, "matched_without_peak_check": int(no_peak)}


def _jsonl(pattern, keep):
    out = []
    for fn in sorted(glob.glob(pattern)):
        if os.path.basename(fn).startswith("._"):
            continue
        op = gzip.open if fn.endswith(".gz") else open
        with op(fn, "rt") as fh:
            out += [r for r in (json.loads(l) for l in fh if l.strip()) if keep(r)]
    return out


def _bdsp(null):
    c = st.c3_rows()
    c = c[(c["null"] == null) & (c["pipe"] == "dubois") & (c["reading"] == READING)]
    pm = _jsonl(os.path.join(INTER, "_bench_shards", f"{null}__dubois__native__none", "rows*.jsonl"),
                lambda r: r.get("variant") == PERM and r.get("resp") == "none")
    return c.to_dict("records"), pm


def _icare(patients=None):
    c3 = _jsonl(os.path.join(INTER, "_icare_c3", "rows*.jsonl"),
                lambda r: r.get("pipe") == "dubois" and r.get("reading") == READING)
    pm = _jsonl(os.path.join(INTER, "_icare_rows", "*.jsonl.gz"),
                lambda r: r.get("pipe") == "dubois" and r.get("variant") == PERM)
    if patients is not None:
        keep = set(patients)
        c3 = [r for r in c3 if str(r["patient"]) in keep]
        pm = [r for r in pm if str(r["patient"]) in keep]
    return c3, pm


def _replication():
    rows = _jsonl(os.path.join(INTER, "_rep_shards", "*__dubois", "rows*.jsonl"), lambda r: True)
    return [r for r in rows if r.get("reading") == READING], [r for r in rows if r.get("variant") == PERM]


def dataset(name):
    if name == "clinical":
        c3, pm = _bdsp("ncmd_bdsp")
    elif name == "doc":
        c3, pm = _bdsp("ncmd_pdoc")
    elif name == "icare490":
        ics = importlib.import_module("150_icare_stated_rule")
        c3, pm = _icare(ics.model_patients()[0])
    elif name == "icare_all":
        c3, pm = _icare()
    elif name == "replication":
        c3, pm = _replication()
        if not c3:
            return None
    else:
        raise ValueError(name)
    for r in c3:
        if _ok(r) and _j0(r) and r.get("call") is not None and bool(r["call"]) != advancement(r):
            raise RuntimeError(f"{name}: stored call differs from the advancement rule")
    return pathway(c3, pm)


def main():
    out = {"reading": READING, "permutation_reading": PERM,
           "datasets": {n: dataset(n) for n in ("clinical", "doc", "icare490", "icare_all", "replication")}}
    sr.assert_no_identifiers(out)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
