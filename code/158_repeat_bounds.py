"""Repeated recordings per patient for the Columbia and Edlow 2017 analyses, at the numbers of recordings their groups
use (Columbia: 1 to 8 recordings per patient, Benjamini-Hochberg at q = .05 across the patient's recordings and AUC above
0.5 on a rejected recording; Edlow: any positive among k = 1 to 5 sessions). The clinical EEG files carry no dates and
the stored post-arrest windows are one per patient, so true repeated recordings are not available; each recording is
treated as another patient's first overlay (independent recordings; stored as 'upper_bound'). For Edlow's any-positive
rule this is an upper bound under non-negative within-patient association of calls. For the Columbia rule it is not:
its correction depends on the number of recordings, and identical recordings give the 1-recording rate at every k.
The same-session value at k = 2 is reported in Table S9B. Monte Carlo over draws of k distinct patients; 95% CI by resampling patients.
Also the calibration of the Edlow permutation P values on command-free first overlays (share below .05 and .5, and a
Kolmogorov-Smirnov test against uniform). -> output/tables/repeat_bounds.json (no identifiers)."""
import importlib
import json
import os
import sys

import numpy as np
from scipy.stats import kstest

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
sr = importlib.import_module("145_stated_rule")
mg = sr.mg
OUT = os.path.join(ROOT, "output", "tables", "repeat_bounds.json")
NULLS = ("ncmd_bdsp", "ncmd_icare")
K_COLUMBIA, K_EDLOW = range(1, 9), range(1, 6)
N_DRAWS, B, Q = 200000, 200, 0.05
_ROWS = {}


def bh_any(p, auc, q=Q):
    """Per row of k recordings: Benjamini-Hochberg at q across the k P values, positive when a rejected recording has
    AUC above 0.5."""
    p, auc = np.atleast_2d(np.asarray(p, float)), np.atleast_2d(np.asarray(auc, float))
    k = p.shape[1]
    o = np.argsort(p, axis=1, kind="stable")
    ps, a = np.take_along_axis(p, o, 1), np.take_along_axis(auc, o, 1)
    ok = ps <= q * np.arange(1, k + 1) / k
    last = np.where(ok.any(1), k - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
    rejected = np.arange(k)[None, :] <= last[:, None]
    return (rejected & (a > 0.5)).any(1)


def all_pairs_rate(p, auc):
    p, auc = np.asarray(p, float), np.asarray(auc, float)
    return float(sr.paper_pairs_rate(p, auc, np.arange(len(p))[None])[0])


def _draw(n, k, n_draws, rng):
    return np.argsort(rng.random((n_draws, n)), axis=1)[:, :k]


def indep_rate(p, auc, k, n_draws=N_DRAWS, rng=None, chunk=20000):
    p, auc = np.asarray(p, float), np.asarray(auc, float)
    rng = rng or np.random.default_rng(0)
    hits = 0
    for s in range(0, n_draws, chunk):
        idx = _draw(len(p), k, min(chunk, n_draws - s), rng)
        hits += int(bh_any(p[idx], auc[idx]).sum())
    return hits / n_draws


def indep_any(calls, k, n_draws=N_DRAWS, rng=None, chunk=20000):
    c = np.asarray(calls, bool)
    rng = rng or np.random.default_rng(0)
    hits = 0
    for s in range(0, n_draws, chunk):
        idx = _draw(len(c), k, min(chunk, n_draws - s), rng)
        hits += int(c[idx].any(1).sum())
    return hits / n_draws


def _rows(nul):
    if nul not in _ROWS:
        if nul.endswith("icare"):
            ics = importlib.import_module("150_icare_stated_rule")
            pats = [str(x) for x in ics.model_patients()[0]]
            c3 = ics.c3_frame()
            c3 = c3[c3["patient"].astype(str).isin(pats) & (c3["status"] != "not_reproduced")]
            _ROWS[nul], _ = sr.build_rows(ics.icare_rows(pats), c3)
        else:
            _ROWS[nul] = sr.prepare()["rows"]
    return _ROWS[nul]


def _first(nul, pipe, variant):
    rows = _rows(nul)
    d = mg._null_seq(rows[(rows["null"] == nul) & (rows["pipe"] == pipe) & (rows["variant"] == variant)])
    d = d[mg._flag(d, "seq", True) & ~mg._flag(d, "overlap")]
    d = d[d["j"] == 0].sort_values(["patient", "uid"]).drop_duplicates("patient")
    return d[d["status"] == "ok"] if "status" in d else d


def columbia_first(nul):
    d = _first(nul, "columbia", "as_coded")
    return {"p": d["p"].astype(float).to_numpy(), "stat": d["stat"].astype(float).to_numpy()}


def edlow_first(nul, variant):
    d = _first(nul, "edlow17", variant)
    return {"p": d["p"].astype(float).to_numpy(), "call": d["call"].astype(bool).to_numpy(),
            "censored": d["p_censored"].astype(bool).to_numpy()}


def calibration(p, censored):
    """Share of P values below .05 and .5; a Kolmogorov-Smirnov test against uniform only when at most half are
    censored (sequential permutation stopping reports a censored P once significance is out of reach)."""
    p, cz = np.asarray(p, float), np.asarray(censored, bool)
    out = {"n": int(len(p)), "share_censored": float(cz.mean()), "share_p_below_05": float((p < 0.05).mean()),
           "share_p_below_50": float((p < 0.5).mean()), "ks_uniform_p": None}
    if cz.mean() > 0.5:
        out["note"] = "P values censored by sequential stopping; uniformity not testable"
    else:
        out["ks_uniform_p"] = float(kstest(p[~cz], "uniform").pvalue)
    return out


def _boot(fn, n, seed):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n), rng) for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main(out=OUT):
    entries = json.load(open(os.path.join(ROOT, "output", "tables", "stated_rule.json")))["entries"]
    icare = json.load(open(os.path.join(ROOT, "output", "tables", "icare_stated_rule.json")))["entries"]
    res = {"what": __doc__.split("\n")[0], "n_draws": N_DRAWS, "B": B, "nulls": {}}
    for nul in NULLS:
        col = columbia_first(nul)
        ed_v = (icare if nul.endswith("icare") else entries)[f"{nul}|edlow17"]["variant"]
        ed = edlow_first(nul, ed_v)
        r = {"columbia": {"n_patients": int(len(col["p"])), "k": {}}, "edlow17": {"variant": ed_v,
                                                                              "n_patients": int(len(ed["call"])), "k": {}}}
        for k in K_COLUMBIA:
            seed = mg._seed("158", nul, "columbia", str(k))
            rate = indep_rate(col["p"], col["stat"], k, rng=np.random.default_rng(seed))
            ci = _boot(lambda i, g: indep_rate(col["p"][i], col["stat"][i], k, n_draws=5000, rng=g), len(col["p"]),
                       list(seed) + [1])
            r["columbia"]["k"][str(k)] = {"upper_bound": rate, "ci95": ci}
        for k in K_EDLOW:
            seed = mg._seed("158", nul, "edlow17", str(k))
            rate = indep_any(ed["call"], k, rng=np.random.default_rng(seed))
            ci = _boot(lambda i, g: indep_any(ed["call"][i], k, n_draws=5000, rng=g), len(ed["call"]), list(seed) + [1])
            r["edlow17"]["k"][str(k)] = {"upper_bound": rate, "ci95": ci}
        r["edlow17"]["calibration"] = calibration(ed["p"], ed["censored"])
        res["nulls"][nul] = r
    sr.assert_no_identifiers(res)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({n: {"columbia": {k: round(v["upper_bound"], 4) for k, v in r["columbia"]["k"].items()},
                          "edlow17": {k: round(v["upper_bound"], 4) for k, v in r["edlow17"]["k"].items()},
                          "edlow_cal": r["edlow17"]["calibration"]} for n, r in res["nulls"].items()}, indent=1))


if __name__ == "__main__":
    main()
