"""Positive control: mindBEAGLE-type motor imagery, 40-trial variant (published run: 60 trials), on acute-stroke hand MI
(stroke_acute_io; 50 recordings, 47 distinct: sub-05 / 15 / 49 duplicate sub-04 / 11 / 47 and are excluded).
Rule = 162's CSP + LDA (8-30 Hz, the 4-s MI epoch, 2 CSP filters per end, 10 x 10-fold stratified CV, left vs right).
Per patient:
  (a) detection: the true labels give a CALL when the mean accuracy > bound(40), the Clopper-Pearson two-sided 95% upper
      limit at k = n/2 for n = 40 (the formula that gives 63.19% at n = 60);
  (b) false calls: 20 random permutations of the patient's 40 labels (default_rng([20261022, subject number])), same rule,
      same call;
  (c) secondary: all 29 EEG channels, the same true run and the same 20 shuffles.
Summary: detection k / n with exact CI; shuffled false-call rate (mean over patients of the share of their shuffles
called) with a patient-cluster bootstrap CI that relabels resampled patients (B = 2000); share of patients with >= 1
false call; detection minus false-call rate with its bootstrap CI; median accuracy, true and shuffled.
python 163_stroke_acute_control.py [--workers 4] -> output/tables/stroke_acute_control.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys

import numpy as np
from scipy.stats import beta

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
kc = importlib.import_module("162_stroke_check")
OUT = os.path.join(ROOT, "output", "tables", "stroke_acute_control.json")
LABEL = "mindBEAGLE-type motor imagery, 40-trial variant (published run: 60 trials)"
SM = ["FCz", "FC3", "FC4", "Cz", "C3", "C4", "CP3", "CP4"]
SEED = 20261022
N_SHUF, REPS, FOLDS, B = 20, 10, 10, 2000
DATA = {}



def bound(n):
    return kc.chance_bound(n)


def call(acc, n):
    return bool(acc > bound(n))


def shuffles(y, idx, n_shuf=N_SHUF):
    """n_shuf random permutations of the label vector, from default_rng([SEED, idx]) (idx = the subject number)."""
    rng = np.random.default_rng([SEED, int(idx)])
    y = np.asarray(y)
    return np.stack([rng.permutation(y) for _ in range(n_shuf)])


def _arm(X, y, perms, n, reps):
    t = kc.cv_accuracy(X, y, FOLDS, reps)
    s = [kc.cv_accuracy(X, p, FOLDS, reps) for p in perms]
    return {"true_acc": t, "true_call": call(t, n), "shuf_acc": s, "shuf_calls": [call(a, n) for a in s],
            "n_channels": int(X.shape[1])}


def run_patient(X, labels, names, idx, n_shuf=N_SHUF, reps=REPS):
    """One patient: sensorimotor-only (primary) and all-channel (secondary) arms, the same 20 shuffles for both."""
    names = list(names)
    missing = [c for c in SM if c not in names]
    if missing:
        raise ValueError(f"sensorimotor channel(s) {missing} not in the recording")
    y = (np.asarray(labels) == "right").astype(int)
    n = len(y)
    perms = shuffles(y, idx, n_shuf)
    sm = [names.index(c) for c in SM]
    out = {"n_trials": n, "sm": _arm(X[:, sm], y, perms, n, reps), "all": _arm(X, y, perms, n, reps)}
    out["sm"]["channels"] = [names[i] for i in sm]
    return out



def stats_of_draw(rows, pat, draw, relabel=True):
    """Statistics of one patient draw (indices into the patient list, with repeats). rows: one entry per shuffle
    (pid, called); pat: one per patient (pid, true_call). relabel=True gives every DRAW its own cluster id (a patient
    drawn three times is three clusters); relabel=False keeps the original id, which merges repeated draws into one."""
    draw = np.asarray(draw)
    by = {}
    for r, p in enumerate(rows["pid"]):
        by.setdefault(int(p), []).append(r)
    ppos = {int(p): k for k, p in enumerate(pat["pid"])}
    sel = [by[int(pat["pid"][i])] for i in draw]
    ridx = np.concatenate(sel)
    cl = np.repeat(np.arange(len(draw)), [len(s) for s in sel]) if relabel else rows["pid"][ridx]
    uniq, inv = np.unique(cl, return_inverse=True)
    called = np.asarray(rows["called"])[ridx].astype(float)
    n_cl = np.bincount(inv)
    calls = np.bincount(inv, weights=called)
    share = calls / n_cl
    pcl = np.arange(len(draw)) if relabel else np.asarray(pat["pid"])[draw]
    pu, first = np.unique(pcl, return_index=True)
    det = float(np.asarray(pat["true_call"], float)[draw][first].mean())
    fr = float(share.mean())
    return {"n_clusters": int(len(uniq)), "false_rate": fr, "any_false_share": float((calls > 0).mean()),
            "detection": det, "diff": det - fr}


def cluster_bootstrap(rows, pat, B=B, seed=SEED):
    """Patient-cluster bootstrap (resampled patients are RELABELLED): percentile 95% CIs of false-call rate, share with
    >= 1 false call, detection and detection minus false-call rate."""
    n = len(pat["pid"])
    rng = np.random.default_rng(seed)
    keys = ("false_rate", "any_false_share", "detection", "diff")
    boots = {k: np.empty(B) for k in keys}
    for b in range(B):
        s = stats_of_draw(rows, pat, rng.integers(0, n, n), relabel=True)
        for k in keys:
            boots[k][b] = s[k]
    pt = stats_of_draw(rows, pat, np.arange(n), relabel=True)
    return {k: {"estimate": pt[k], "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]}
            for k in keys}



def exact_ci(k, n, conf=0.95):
    a = 1 - conf
    return [float(beta.ppf(a / 2, k, n - k + 1)) if k > 0 else 0.0,
            float(beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0]


def summarize(pats, arm, B=B):
    ids = sorted(pats)
    k = int(sum(pats[i][arm]["true_call"] for i in ids))
    rows = {"pid": np.repeat(np.array(ids), len(pats[ids[0]][arm]["shuf_calls"])),
            "called": np.concatenate([np.asarray(pats[i][arm]["shuf_calls"], bool) for i in ids])}
    pat = {"pid": np.array(ids), "true_call": np.array([pats[i][arm]["true_call"] for i in ids], bool)}
    bt = cluster_bootstrap(rows, pat, B)
    sh = np.array([pats[i][arm]["shuf_acc"] for i in ids])
    any_k = int(sum(any(pats[i][arm]["shuf_calls"]) for i in ids))
    n = len(ids)
    return {"label": LABEL, "n_patients": n, "n_trials": int(np.median([pats[i]["n_trials"] for i in ids])),
            "bound": bound(int(np.median([pats[i]["n_trials"] for i in ids]))),
            "detection": {"k": k, "n": n, "rate": k / n, "exact_ci95": exact_ci(k, n)},
            "false_call_rate": bt["false_rate"],
            "share_patients_with_a_false_call": {"k": any_k, "n": n, "rate": any_k / n, "exact_ci95": exact_ci(any_k, n),
                                                 "bootstrap": bt["any_false_share"]},
            "detection_minus_false_rate": bt["diff"],
            "detection_bootstrap": bt["detection"],
            "median_true_acc": float(np.median([pats[i][arm]["true_acc"] for i in ids])),
            "median_shuffled_acc_all_shuffles": float(np.median(sh)),
            "median_of_patient_median_shuffled_acc": float(np.median(np.median(sh, axis=1))),
            "mean_shuffled_acc": float(sh.mean()), "n_shuffles_per_patient": int(sh.shape[1]), "B": B}



def distinct(arrays):
    dup = kc.duplicates(arrays)
    return [s for s in sorted(arrays) if s not in dup]


def _work(s):
    X, y, ch = DATA[s]
    return s, run_patient(X, y, ch, s)


def main(workers=4, subjects=range(1, 51), out=OUT):
    import multiprocessing as mp
    import stroke_acute_io as sa
    for s in subjects:
        DATA[s] = sa.trials(s, band=kc.BAND)
    keep = distinct({s: v[0] for s, v in DATA.items()})
    dropped = sorted(set(DATA) - set(keep))
    with mp.get_context("fork").Pool(min(workers, 4)) as pool:
        res = dict(pool.map(_work, keep, chunksize=1))
    pats = {s: res[s] for s in keep}
    prev = json.load(open(kc.OUT))["acute"]["patients"] if os.path.exists(kc.OUT) else {}
    same = {f"sub-{s:02d}": (abs(pats[s]["sm"]["true_acc"] - prev[f"sub-{s:02d}"]["sensorimotor_acc"]) < 1e-12)
            for s in keep if f"sub-{s:02d}" in prev}
    d = {"label": LABEL, "patients_analysed": len(keep), "excluded_duplicates": [f"sub-{s:02d}" for s in dropped],
         "rule": "162 CSP + LDA, 8-30 Hz, 4-s MI epoch (0-4 s after the MI marker), 10 x 10-fold CV; sensorimotor = "
                 + " ".join(SM), "call": "mean accuracy > bound(40) = Clopper-Pearson two-sided 95% upper limit at k = n/2",
         "bound_40": bound(40), "bound_60": bound(60), "shuffle_seed": f"default_rng([{SEED}, subject number])",
         "true_accuracies_equal_162_primary": bool(same) and all(same.values()),
         "sensorimotor": summarize(pats, "sm"), "all_channels": summarize(pats, "all"),
         "patients": {f"sub-{s:02d}": {a: {k: v for k, v in pats[s][a].items()} for a in ("sm", "all")} for s in keep}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(d, fh, indent=1)
    os.replace(out + ".tmp", out)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    r = main(a.workers)
    print(json.dumps({k: v for k, v in r.items() if k != "patients"}, indent=1))
