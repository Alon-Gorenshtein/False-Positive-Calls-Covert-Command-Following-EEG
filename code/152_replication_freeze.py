"""Frozen readings of the replication set, fixed before any replication patient is drawn: the primary reading of every
command-following analysis whose schedule fits a 30-minute window, as selected on the clinical EEG development patients
(stated_rule.json), with the tests and the draw identity. A held-out internal replication: the same corpus and sampling
frame, patients never analyzed in this study (126 extend refuses any touched patient).
Tests, fixed here:
- confirmatory: Cornell, du Bois and the vibrotactile P300 (the analyses above 5% in development), one-sided exact
  binomial of the first-overlay count against 5%, Holm over the 3; the P300 two-run rule reported beside them;
- specificity controls: the other 4 analyses, one-sided exact binomial for a rate below 10%, Holm over the 4;
- reproduction: the replication rate lies inside the 95% prediction interval around the development cross-fitted
  rate, p_d +- 1.96 * sqrt(p_d (1 - p_d) (1 / n_d + 1 / n_r)), n_r the replication denominator;
- missing is not negative: error, no-fit and no-usable-trial windows leave the denominator.
-> output/tables/replication_frozen.json (no identifiers); never overwritten."""
import hashlib
import json
import math
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = os.path.join(ROOT, "output", "tables", "stated_rule.json")
LISTING = os.path.join(ROOT, "output", "bdsp", "meta", "c2_fresh_listing.csv")
OUT = os.path.join(ROOT, "output", "tables", "replication_frozen.json")
PIPES = {"columbia": "ncmd", "cornell": "ncmd", "dubois": "ncmd", "edlow17": "ncmd", "mbmi": "ncmd",
         "gibson16": "nstim0", "mbvt": "nstim0"}
CONFIRMATORY = ["cornell", "dubois", "mbvt"]
CONTROLS = ["columbia", "edlow17", "mbmi", "gibson16"]
NOT_REPLICATED = {"cruse11": "primary reading needs 2,186 s of clean EEG; replication windows are 1,800 s",
                  "cruse12p": "primary reading needs 1,891 s of clean EEG; replication windows are 1,800 s",
                  "chennu13": "no eligible reading on clinical EEG"}
EXTRA = {"dubois": ["assessment_blocked_perm"]}
MAX_OVERLAYS = {"mbvt": 2}
DRAW = {"rule": "126_c2_fresh.py extend: ranks of the pinned fresh-draw listing after the first 400, each patient "
                "with its first scanned segment that holds a clean 1,800-s window, at a start drawn uniformly from its "
                "eligible starts; touched patients refused",
        "window_s": 1800, "seed": [20261016, 2], "n_ext": 730}


def prediction_interval(k_d, n_d, n_r, z=1.959963984540054):
    p = k_d / n_d
    h = z * math.sqrt(p * (1 - p) * (1 / n_d + 1 / n_r))
    return max(0.0, p - h), min(1.0, p + h)


def frozen(sr):
    pipes = {}
    for pipe, kind in PIPES.items():
        e = sr["entries"][f"{kind}_bdsp|{pipe}"]
        if not e.get("variant"):
            raise RuntimeError(f"{pipe}: no primary reading to freeze")
        s, h = e["stated_rule"], e["held_out_test"]
        pipes[pipe] = {"null": f"{kind}_bdsp", "variant": e["variant"], "extra_variants": EXTRA.get(pipe, []),
                       "max_overlays": MAX_OVERLAYS.get(pipe, 1),
                       "development": {"k": s["k"], "n": s["n"], "rate": s["rate"]},
                       "development_crossfit": {"k": h["k"], "n": h["n"], "rate": h["rate"]}}
    two = sr["per_patient_at_published_k"]["nstim0_bdsp|mbvt"]["composite"]
    return {"pipes": pipes, "not_replicated": NOT_REPLICATED,
            "development_mbvt_two_run": {"k": two["k"], "n": two["n"], "rate": two["rate"]},
            "test": {"confirmatory": {"pipes": CONFIRMATORY, "benchmark": 0.05, "multiplicity": "Holm over 3",
                                      "test": "one-sided exact binomial, rate above 5%"},
                     "specificity_controls": {"pipes": CONTROLS, "bound": 0.10, "multiplicity": "Holm over 4",
                                              "test": "one-sided exact binomial, rate below 10%"},
                     "reproduction": "replication rate inside the 95% prediction interval around the development "
                                     "cross-fitted rate (prediction_interval)",
                     "per_patient": {"mbvt": "any positive among the first 2 runs"},
                     "unit": "patient (one window each, first overlay)",
                     "missing": "error, no-fit and no-usable-trial windows leave the denominator"},
            "draw": dict(DRAW, listing_sha256=hashlib.sha256(open(LISTING, "rb").read()).hexdigest())}


def main(out=OUT):
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: the freeze is never rewritten")
    with open(SR, "rb") as fh:
        raw = fh.read()
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(frozen(json.loads(raw)), stated_rule_sha256=hashlib.sha256(raw).hexdigest(), code_commit=head)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
