"""Agreement between the two blind classifications of the varied analysis choices (added later summary).

Reads the first classification as committed at 9a94bcc and the second classifier's class stored per axis in
the current docs/stated-rule-{motor,erp}.json (second_classifier.class). One row per classified choice:
(pipeline, axis, value, first-classifier class); a choice whose first class differs across variants counts
once per class, so the rows match the supplement's classification table.

Reports raw agreement and Cohen kappa for the four classes and for the eligibility split
(STATED or NOT_STATED vs CONTRADICTS or SHORTENED_STRUCTURAL), with percentile bootstrap 95% CIs over rows.

Output: output/tables/classifier_agreement.json
"""
import collections
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = ["docs/stated-rule-motor.json", "docs/stated-rule-erp.json"]
FIRST_REV = "9a94bcc"
ELIGIBLE = ("STATED", "NOT_STATED")
B = 2000
SEED = 0


def _load(rev, path):
    if rev is None:
        return json.loads((ROOT / path).read_text())
    return json.loads((ROOT / path.replace(".json", "-first.json")).read_text())


def rows(first=None, second=None):
    """(first class, second class) per classified choice."""
    out = {}
    for f in FILES:
        v1 = first[f] if first else _load(FIRST_REV, f)
        v2 = second[f] if second else _load(None, f)
        for pipe, variants in v2.items():
            if not isinstance(variants, dict):
                continue
            for var, vd in variants.items():
                if not isinstance(vd, dict) or "axes" not in vd:
                    continue
                for axis, a in vd["axes"].items():
                    sc = a.get("second_classifier")
                    if not isinstance(sc, dict) or "class" not in sc:
                        continue
                    c1 = v1.get(pipe, {}).get(var, {}).get("axes", {}).get(axis, {}).get("class")
                    if c1 is None:
                        continue
                    out[(pipe, axis, str(a.get("value")), c1)] = sc["class"]
    return [(k[3], v) for k, v in sorted(out.items())]


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    c1 = collections.Counter(a for a, _ in pairs)
    c2 = collections.Counter(b for _, b in pairs)
    pe = sum(c1[k] * c2.get(k, 0) for k in c1) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def summarize(pairs, b=B, seed=SEED):
    po, k = kappa(pairs)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    ks, ps = [], []
    for _ in range(b):
        draw = [pairs[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        p_, k_ = kappa(draw)
        ps.append(p_)
        ks.append(k_)
    return {"n": len(pairs), "agreement": po, "kappa": k,
            "agreement_ci95": [float(np.nanpercentile(ps, 2.5)), float(np.nanpercentile(ps, 97.5))],
            "kappa_ci95": [float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))]}


def main():
    pairs = rows()
    elig = [("E" if a in ELIGIBLE else "X", "E" if b in ELIGIBLE else "X") for a, b in pairs]
    res = {"what": "agreement of the two blind classifications (added later summary)",
           "first_classification_rev": FIRST_REV, "B": B, "seed": SEED,
           "four_class": summarize(pairs), "eligibility": summarize(elig),
           "classes_first": dict(collections.Counter(a for a, _ in pairs)),
           "classes_second": dict(collections.Counter(b for _, b in pairs))}
    out = ROOT / "output/tables/classifier_agreement.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    for key in ("four_class", "eligibility"):
        r = res[key]
        print(key, r["n"], f"agreement {100*r['agreement']:.1f}%", f"kappa {r['kappa']:.2f}",
              "CI", [round(x, 2) for x in r["kappa_ci95"]])


if __name__ == "__main__":
    main()
