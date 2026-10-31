"""Cluster-aware inference for the genuine-task controls, on the stored rows of 172 (stroke) and 173 (tactile); the
estimands are fixed here before any of these summaries is computed and are descriptive (no margin, no new test bar).
- Stroke: each reading's true-label detection with a patient-cluster bootstrap CI, and the paired change in detection
  between the Cornell primary reading (cornell_c3) and the command-level reading (cornell_cmd) on the same sessions
  (172.duplicate_sessions excluded), also with a patient-cluster bootstrap CI.
- Tactile: the participant is the unit. Session calls are 173.session_calls (any positive among runs A and B). For each
  reading: session-level true and sham rates and their difference with participant-cluster bootstrap CIs, run-level
  rates (runs A and B as units) with the same, participants with any true or sham call, and an exact two-sided sign
  test over participants of (true sessions called - sham sessions called), ties left out.
Bootstrap: B = 2000 resamples of whole clusters, seed [20261030, 176], percentile 95% interval.
python 176_clustered.py -> output/tables/clustered_inference.json"""
import glob
import importlib
import json
import os
import sys

import numpy as np
from scipy.stats import binom

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
INTER = os.path.join(ROOT, "output", "intermediate")
OUT = os.path.join(ROOT, "output", "tables", "clustered_inference.json")
B, SEED = 2000, [20261030, 176]
STROKE_READINGS = ("cornell_c3", "cornell_cmd", "cornell_cmd_t", "dubois_literal", "dubois_reading_b")
TACTILE_READINGS = ("mbvt_primary", "mbvt_default", "blind", "randomization")


def cluster_mean_ci(clusters, values, b=B, seed=SEED):
    """Mean of values over units and its percentile CI when whole clusters are resampled. -> (est, lo, hi)."""
    clusters, values = np.asarray(clusters), np.asarray(values, float)
    ids = np.unique(clusters)
    groups = [values[clusters == c] for c in ids]
    sums = np.array([g.sum() for g in groups])
    ns = np.array([len(g) for g in groups])
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(ids), size=(b, len(ids)))
    boot = sums[idx].sum(1) / ns[idx].sum(1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sign_test(diffs):
    """Exact two-sided sign test; zero differences are left out. -> dict."""
    d = np.asarray(diffs, float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    p = float(min(1.0, 2 * binom.cdf(min(pos, neg), n, 0.5))) if n else 1.0
    return {"positive": pos, "negative": neg, "ties": int((d == 0).sum()), "p_exact": p}


def _jsonl(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        if not os.path.basename(f).startswith("._"):
            out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def stroke(rows):
    dup = importlib.import_module("172_stroke_native").duplicate_sessions(rows)
    true = {}
    for r in rows:
        if r["version"] == "true" and r["session_id"] not in dup and r.get("call") is not None:
            true.setdefault(r["reading"], {})[r["session_id"]] = (r["subject"], bool(r["call"]))
    out = {"excluded_sessions": len(dup), "readings": {}}
    for name in STROKE_READINGS:
        d = true.get(name, {})
        cl, v = zip(*d.values()) if d else ((), ())
        est, lo, hi = cluster_mean_ci(cl, v) if d else (None, None, None)
        out["readings"][name] = {"n_sessions": len(d), "n_patients": len(set(cl)), "detection": est, "ci95": [lo, hi]}
    a, c = true["cornell_c3"], true["cornell_cmd"]
    common = sorted(set(a) & set(c))
    diff = [float(a[s][1]) - float(c[s][1]) for s in common]
    est, lo, hi = cluster_mean_ci([a[s][0] for s in common], diff)
    b_, c_ = int(sum(x > 0 for x in diff)), int(sum(x < 0 for x in diff))
    out["paired_c3_minus_cmd"] = {"n_sessions": len(common), "difference": est, "ci95": [lo, hi],
                                  "c3_only": b_, "cmd_only": c_,
                                  "mcnemar_exact_p": float(min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5)))
                                  if b_ + c_ else 1.0}
    return out


def tactile(rows):
    bp = importlib.import_module("173_braille_p300_control")
    out = {}
    for name in TACTILE_READINGS:
        t, s = bp.session_calls(rows, name, "true"), bp.session_calls(rows, name, "sham")
        keys = sorted(set(t) & set(s))
        persons = sorted({k[0] for k in keys})
        cl = [k[0] for k in keys]
        res = {"n_sessions": len(keys), "n_participants": len(persons),
               "missing_sessions": len(set(t) ^ set(s))}
        for lab, v in (("true", [float(t[k]) for k in keys]), ("sham", [float(s[k]) for k in keys]),
                       ("difference", [float(t[k]) - float(s[k]) for k in keys])):
            est, lo, hi = cluster_mean_ci(cl, v)
            res[f"session_{lab}"] = {"estimate": est, "ci95": [lo, hi]}
        runs = [r for r in rows if r["reading"] == name and r["run"] in bp.PRIMARY_RUNS and r.get("call") is not None]
        for version in ("true", "sham"):
            rr = [r for r in runs if r["version"] == version]
            est, lo, hi = cluster_mean_ci([r["subject"] for r in rr], [float(bool(r["call"])) for r in rr])
            res[f"run_{version}"] = {"n_runs": len(rr), "estimate": est, "ci95": [lo, hi]}
        per = {p: sum(t[k] for k in keys if k[0] == p) - sum(s[k] for k in keys if k[0] == p) for p in persons}
        res["participants_any_true"] = sum(any(t[k] for k in keys if k[0] == p) for p in persons)
        res["participants_any_sham"] = sum(any(s[k] for k in keys if k[0] == p) for p in persons)
        res["sign_test_participants"] = sign_test(list(per.values()))
        out[name] = res
    return out


def main():
    res = {"what": __doc__.split("\n")[0], "bootstrap": {"B": B, "seed": SEED},
           "stroke": stroke(_jsonl(os.path.join(INTER, "stroke_native_rows_s*.jsonl"))),
           "tactile": tactile(_jsonl(os.path.join(INTER, "braille_p300_rows_s*.jsonl")))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps({"stroke_paired": res["stroke"]["paired_c3_minus_cmd"],
                      "tactile_primary": res["tactile"]["mbvt_primary"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
