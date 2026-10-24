"""Absolute size and dose-response of the sign test's label-aligned heart-rate effect, on DISCOVERY data (137's
Phase-1 sign rows, output/intermediate/_arousal_explore/sign_rows.csv). Exploration. Built test-first
(code/test_sign_size.py).

Attributable calls (attributable): the exposed aligned-vs-placebo model refitted as 137's sign_model does (140._set:
116._prep weights, pipeline-variant FE, penalised fit on covariates over their weighted SD). Counterfactual: z(aligned)
set to its 10th percentile over the model rows, every other term kept. Per pipeline-variant at the pipeline's
validity primary and default (116's part A variant_choices for ncmd_bdsp, read from mechanism.json) and pooled
over every exposed row: extra false calls per 100 runs = 100 x (calls - counterfactual calls) / runs, and the share
of calls, with calls taken as observed (observed minus counterfactual predicted) and as fitted (fitted minus
counterfactual: no misfit); patient-cluster bootstrap (relabelled draws, 2000; refit, 10th percentile recomputed on
each draw), percentile 95% CI.
Dose-response (dose_response): call rate by quintile of |t_aligned| within each pipeline-variant (so a quintile mixes
the variants evenly), per pipeline and pooled; exposed rows and, on the same rows, the randomised method by quintile
of its own |t_aligned| within its timing group.
python 141_sign_size.py"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib
import importlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

cc = importlib.import_module("140_cardiac_control")
ex, mech, rn, ns = cc.ex, cc.mech, cc.rn, cc.ns

SEED = 20261018
B_BOOT = 2000
Q_CF, N_Q = 0.10, 5
MECH_JSON = os.path.join(ROOT, "output", "tables", "mechanism.json")
OUT_JSON = os.path.join(cc.OUT, "sign_size.json")


def predicted(alpha, beta, Z, code):
    return expit(np.asarray(alpha, float)[code] + np.asarray(Z, float) @ np.asarray(beta, float))


def set_column(Z, j, value):
    Z = np.array(Z, float, copy=True)
    Z[:, j] = value
    return Z


def _metrics(y, p, cf):
    n, obs, fit, c = len(y), float(y.sum()), float(p.sum()), float(cf.sum())
    return {"n_runs": n, "calls": obs, "fitted": fit, "counterfactual": c,
            "extra_per_100_observed": 100 * (obs - c) / n if n else np.nan,
            "extra_per_100_model": 100 * (fit - c) / n if n else np.nan,
            "share_observed": (obs - c) / obs if obs else np.nan, "share_model": (fit - c) / fit if fit else np.nan}


def _fit_metrics(y, Zs, fe, w, groups):
    r = mech.fe_logit(y, Zs, fe, w, lam=mech.LAM)
    if not r["converged"]:
        return None
    code = np.searchsorted(r["codes"], fe)
    p = predicted(r["alpha"], r["beta"], Zs, code)
    cf = predicted(r["alpha"], r["beta"], set_column(Zs, 0, np.quantile(Zs[:, 0], Q_CF)), code)
    out = {"pooled": _metrics(y, p, cf)}
    for g in groups:
        m = fe == g
        out[g] = _metrics(y[m], p[m], cf[m]) if m.any() else None
    return out


def attributable(d, n_variants, choices, B=B_BOOT, seed=(SEED, 4)):
    """Extra false calls attributable to the label-aligned heart-rate contrast (module docstring). choices:
    {pipe: {"validity_primary": v, "default": v}}."""
    e, y, Z, w, sd = cc._set(d, n_variants, ("abs_t_aligned", "abs_t_placebo"), "call", None)
    fe, Zs = e["fe"].to_numpy(), Z / sd
    roles = {}
    for pipe, c in choices.items():
        for role in ("validity_primary", "default"):
            if c.get(role):
                roles.setdefault(f"{pipe}|{c[role]}", []).append(role)
    groups = [g for g in roles if g in set(fe)]
    point = _fit_metrics(y, Zs, fe, w, groups)
    ci = mech.cluster_index(e["patient"].to_numpy())
    draws = mech.mg.boot_idx(len(ci["ids"]), B, list(seed))
    keys = ("extra_per_100_observed", "extra_per_100_model", "share_observed", "share_model")
    bd = {g: {k: [] for k in keys} for g in ["pooled"] + groups}
    for b in range(B):
        idx, _ = mech.boot_sample(ci, draws[b])
        r = _fit_metrics(y[idx], Zs[idx], fe[idx], w[idx], groups)
        for g in bd:
            for k in keys:
                bd[g][k].append(np.nan if r is None or r.get(g) is None else r[g][k])
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
    res = {g: dict(point[g], **{k + "_ci95": q(bd[g][k]) for k in keys}) for g in ["pooled"] + groups}
    out = {"n_rows": int(len(e)), "n_patients": int(e["patient"].nunique()), "q_counterfactual": Q_CF, "B": int(B),
           "pooled": res["pooled"], "by_variant": {}}
    for g, rl in roles.items():
        out["by_variant"][g] = dict(res[g], role=rl) if g in res else {"role": rl, "na": "not in the exposed model"}
    return out


def dose_response(d, col, call, grp_col, n_q=N_Q):
    """Call rate by quintile of col within each pipeline|grp_col group, per pipeline and pooled."""
    e = d.dropna(subset=[col, call]).copy()
    e["grp"] = e["pipe"].astype(str) + "|" + e[grp_col].astype(str)
    e["q"] = e.groupby("grp")[col].transform(
        lambda v: pd.qcut(v.rank(method="first"), n_q, labels=False) if len(v) >= n_q else np.nan)
    e = e.dropna(subset=["q"])
    e["c"] = e[call].astype(bool).astype(float)
    tab = lambda f: {int(k): {"n": int(len(g)), "calls": int(g["c"].sum()), "rate": float(g["c"].mean())}
                     for k, g in f.groupby("q")}
    return {"pooled": tab(e), "by_pipeline": {p: tab(f) for p, f in e.groupby("pipe")}}


def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    with open(cc.SIGN_ROWS, "rb") as fh:
        rows_sha = hashlib.sha256(fh.read()).hexdigest()
    d = pd.read_csv(cc.SIGN_ROWS, dtype={"uid": str, "patient": str, "key": str})
    typ = ns.pinned_typing()
    imm = np.array([typ.get((p, v)) == "immune" for p, v in zip(d["pipe"], d["variant"])])
    ve = {p: n for p, n in ns.non_immune_variant_counts(sorted(set(d["pipe"])), typ).items() if n}
    vc = json.load(open(MECH_JSON))["part_A"]["variant_choices"]
    choices = {k.split("|", 1)[1]: v for k, v in vc.items() if k.startswith("ncmd_bdsp|")}
    x = d[~imm]
    same = x.dropna(subset=["abs_t_aligned", "rand_abs_t_aligned", "rand_call"])
    out = {"label": "sign test: attributable false calls and dose-response (discovery data, exploration)",
           "sha256_141": own_sha256(), "sign_rows_sha256": rows_sha,
           "variant_choices": choices,
           "attributable": attributable(x, ve, choices),
           "dose_response": {"exposed_all_rows": dose_response(x, "abs_t_aligned", "call", "variant"),
                             "exposed_same_rows": dose_response(same, "abs_t_aligned", "call", "variant"),
                             "randomised_same_rows": dose_response(same, "rand_abs_t_aligned", "rand_call",
                                                                   "rand_variant")}}
    with open(OUT_JSON, "w") as fh:
        json.dump(mech.mg._jsonable(out), fh, indent=1, default=float)
    print("written", OUT_JSON)


if __name__ == "__main__":
    main()
