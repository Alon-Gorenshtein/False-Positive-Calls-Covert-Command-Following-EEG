"""cruse12p on synthetic noise (added later diagnostic, synthetic data only): why random_perclass calls 9/60 = 15.0% of
147's white-noise draws but 5/210 = 2.4% of BDSP recordings. Steps, fixed before the run:
1. Precision. random_perclass on 500 fresh white-noise draws: 129's generator (at.noise, cell 'white': 19 NAMES19
   channels, 128 Hz, 20 uV), 147's length (PAD_S + 1891 s cruse12p atlas need + PAD_S) and overlay start, seeds
   disjoint from 147's and 129's (r = R0 + i, R0 = 10**6, i < 500; both use r < 60). Each draw runs through 114's own
   wrappers exactly as 147.run_unit (rn._sched, rn._run, schedule seed rn._rng(pipe, uid, 0, 1), run seed
   rn._rng(pipe, uid, 0, 2, crc(arg)); uid 'diag149|r'). Rate, exact CP 95% CI (143.clopper_pearson), one-sided exact
   binomial p against 5% (115.binom_greater). The same for every other cruse12p eligible variant (classification v2:
   contiguous_perclass).
2. p-value calibration. The same white draws, random_perclass with the permutation test's early stopping off (the
   module stops once the decision is fixed, so its reported p is censored at about .025): 1000 permutations per
   comparison. KS against uniform and the share below 0.05, 0.01, 0.001, for each comparison's p (right-vs-relax,
   left-vs-relax: the module tests each at .025) and for the module's p (the smaller of the two).
3. Spectrum. The same 500 draws as pink noise (129's 'pink1', 1/f: the same seed, so the same Gaussian numbers
   shaped to 1/f, plus 129's 1-uV floor); the same schedules and run seeds. BDSP-like spectrum: only from a median BDSP
   power-spectrum aggregate already on disk. None exists (116's per-row 'spec' and A7 hold spectra of the FEATURE
   sequence over trials, 138 holds per-recording slope fits, 128 holds variance ratios): skipped.
4. Mechanism check. Candidate steps, each with its evidence:
   a. trial count: kept trials on noise vs BDSP (114's cruse12p BDSP rows, quantiles only); the white draws re-run with
      trials dropped at random to n_kept values drawn from BDSP's (arm 'drop').
   b. the permutation null: run() draws the CV groups stratified on the OBSERVED labels and keeps them for every
      permuted labelling (FIXED['perm_folds']), so permuted folds are no longer class-balanced. Arm 'regroup': this
      file's run_mirror (reproduces run() exactly, test) with the groups re-stratified on each permuted labelling.
      Evidence: rates of the two arms on the same draws (exact McNemar), test-fold class imbalance observed vs
      permuted, observed vs permuted accuracy.
   c. artefact rejection: kept trials per draw on white and pink.
   d. window / epoch: minimum inter-onset interval vs EPOCH_S (overlapping epochs share samples), extents inside the
      noise window.
   e. ADDED AFTER STEP 1's interim rows (contiguous_perclass also above 5%; its groups ignore the labels): arm
      'withinfold', the null permutes labels within each CV test fold, so every permuted labelling keeps the observed
      fold class counts (a valid null for exchangeable trials), for both variants; and, from the schedules alone (no
      noise), the test-fold class imbalance of the observed labels vs uniformly permuted ones under both group kinds.
   A step is named only if its arm moves the rate; otherwise 'cause not identified'.
Mac, one process, nice 10, caffeinate; rows resume from output/intermediate/_cruse12p_diag. Stops if swap in use grows
by more than 1 GB over its value at start.
Run order as executed: 1, 3, 2 (first 200 draws, to fit the hour), 4; within a step one draw per cell feeds all its arms.
python3 149_cruse12p_noise_diagnostic.py [--n 500] [--steps 1,3,2,4] [--full-n 200] | --summarise"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib
import json
import sys

import numpy as np
from scipy.stats import binom, kstest

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
wn = importlib.import_module("147_white_noise_eligible")
at, rn, en, sr = wn.at, wn.rn, wn.en, wn.sr
st = importlib.import_module("143_stratum_table")
bm = importlib.import_module("115_benchmark_merge")
from bench import motor_common as mc

ROOT = en.ROOT
PIPE = "cruse12p"
VARIANTS = ("random_perclass", "contiguous_perclass")
N, R0 = 500, 10 ** 6
NEED_S = 1891
NOMINAL = 0.05
SWAP_GROWTH_MB = 1024.0
DIR = os.path.join(ROOT, "output", "intermediate", "_cruse12p_diag")
OUT = os.path.join(ROOT, "output", "tables", "cruse12p_noise_diagnostic.json")
CELLS = {"white": "white", "pink": "pink1"}
N_IMBALANCE_PERM = 100



def draw_rec(cell, r):
    """129's generator; the white seed for every cell, so pink is the same Gaussian draw shaped to 1/f."""
    x = at.noise(at.CELLS[CELLS[cell]], at.PAD_S + NEED_S + at.PAD_S,
                 np.random.default_rng([at.SEED, rn._crc(PIPE), rn._crc("white"), r]))
    uid = f"diag149|{cell}|{r}"
    return en.Recording(source="atlas", uid=uid, patient=uid, x=x, fs=at.FS, names=list(at.NAMES),
                        meta={"dev": False, "cell": cell})


def overlay(mod, spec, r):
    o = {"j": 0, "start_s": at.PAD_S + rn.LEAD_S, "window": (at.PAD_S, at.PAD_S + NEED_S)}
    ov, err, _ = rn._call(rn._sched, mod, mod.schedule, spec, o["start_s"], rn._rng(PIPE, f"diag149|{r}", 0, 1))
    return ov, err, (ov is not None and rn._inside(mod, ov, o))


def run_rng(spec, r):
    return rn._rng(PIPE, f"diag149|{r}", 0, 2, rn._crc(spec["arg"]))


def drop_keep(keep, n_target, rng):
    """keep with trials removed at random until n_target remain (never adds a trial back)."""
    keep = np.asarray(keep, bool).copy()
    i = np.nonzero(keep)[0]
    if n_target < len(i):
        keep[rng.choice(i, len(i) - int(n_target), replace=False)] = False
    return keep



def run_mirror(mod, rec, ov, variant, rng, regroup=False, n_keep=None, drop_rng=None, early_stop=True,
               imbalance_rng=None, withinfold=False):
    """mod.run with four switches: regroup re-stratifies the CV groups on each permuted labelling (run() keeps the
    observed ones); withinfold permutes the labels within each test fold; n_keep drops kept trials at random (drop_rng) to n_keep; early_stop off runs all permutations.
    With the defaults it is run() line for line (test). imbalance_rng: also report the test-fold class imbalance of
    the observed groups and of N_IMBALANCE_PERM permuted labellings (its own generator; the run's is untouched)."""
    c = mc.cfg(mod.VARIANTS, variant)
    g = mc.gen(rng)
    F, keep, which = mod._feat(rec, ov)
    if n_keep is not None:
        keep = drop_keep(keep, n_keep, drop_rng)
    lab = np.asarray(ov.labels)[keep]
    F = F[keep]
    mod._require_trials(lab)
    out = {}
    for name, cls in (("right", mod.RIGHT), ("left", mod.LEFT)):
        Fc, y = mod._comparison(F, lab, cls)
        grp = mod.cv_groups(len(y), c["cv"], g, y)
        mod._require_folds(name, y, grp)
        obs = mod.smoothed_max(mod.nb_cv_accuracy(Fc, y, grp, c["nb_var"]))

        def draw(Fc=Fc, y=y, grp=grp):
            yp = permute_within(y, grp, g) if withinfold else g.permutation(y)
            gp = mod.cv_groups(len(yp), c["cv"], g, yp) if regroup else grp
            return mod.smoothed_max(mod.nb_cv_accuracy(Fc, yp, gp, c["nb_var"]))
        t = mc.perm_test(obs, draw, c.get("n_perm", mod.N_PERM), mod.ALPHA, early_stop=early_stop)
        out[name] = {"acc": obs, "p": t["p"], "n_perm_run": t["n_run"], "stopped_early": t["stopped"],
                     "n_nonfinite_null": t["n_nonfinite_null"], "n": len(y)}
        if not early_stop:
            out[name].update(null_mean=float(np.mean(t["null"])), null_sd=float(np.std(t["null"])))
        if imbalance_rng is not None:
            out[name].update(imbalance_obs=fold_imbalance(y, grp),
                             imbalance_perm=float(np.mean([fold_imbalance(imbalance_rng.permutation(y), grp)
                                                           for _ in range(N_IMBALANCE_PERM)])))
    p = min(out["right"]["p"], out["left"]["p"])
    stat = max(out["right"]["acc"], out["left"]["acc"])
    return mc.result(stat, p, p < mod.ALPHA, comparisons=out, n_kept=int(keep.sum()), channels=which,
                     n_nonfinite_null=sum(o["n_nonfinite_null"] for o in out.values()))


def permute_within(y, grp, g):
    yp = np.array(y, copy=True)
    for f in np.unique(grp):
        i = np.nonzero(grp == f)[0]
        yp[i] = y[g.permutation(i)]
    return yp


def design_imbalance(n, n_perm=N_IMBALANCE_PERM):
    """Test-fold class imbalance (fold_imbalance) of the observed labels and of uniformly permuted ones, for random
    stratified and contiguous groups, over the step-1 schedules (labels only; every trial kept, as on white noise)."""
    mod = rn.load_pipe(PIPE)
    spec = rn._spec(mod, "random_perclass", "bdsp")
    acc = {k: {"observed": [], "permuted": []} for k in ("random", "contiguous")}
    for i in range(n):
        ov, _, _ = overlay(mod, spec, R0 + i)
        lab = np.asarray(ov.labels)
        g = np.random.default_rng([at.SEED, rn._crc("design149"), R0 + i])
        for cls in (mod.RIGHT, mod.LEFT):
            _, y = mod._comparison(np.zeros((len(lab), 1)), lab, cls)
            for kind in acc:
                grp = mod.cv_groups(len(y), kind, g, y)
                acc[kind]["observed"].append(fold_imbalance(y, grp))
                acc[kind]["permuted"].append(np.mean([fold_imbalance(g.permutation(y), grp) for _ in range(n_perm)]))
    return {k: {"n": len(v["observed"]), "observed_mean": float(np.mean(v["observed"])),
                "permuted_mean": float(np.mean(v["permuted"]))} for k, v in acc.items()}


def fold_imbalance(y, grp):
    """Mean over test folds of |share of class 1 in the fold - share of class 1 overall|."""
    y = np.asarray(y)
    return float(np.mean([abs(y[grp == f].mean() - y.mean()) for f in np.unique(grp)]))



ARMS = {
    "white|random_perclass": (1, "white", "random_perclass", "module"),
    "white|contiguous_perclass": (1, "white", "contiguous_perclass", "module"),
    "pink|random_perclass": (3, "pink", "random_perclass", "module"),
    "pink|contiguous_perclass": (3, "pink", "contiguous_perclass", "module"),
    "white|contiguous_perclass|withinfold": (4, "white", "contiguous_perclass", "withinfold"),
    "white|random_perclass|withinfold": (4, "white", "random_perclass", "withinfold"),
    "white|random_perclass|regroup": (4, "white", "random_perclass", "regroup"),
    "white|random_perclass|drop": (4, "white", "random_perclass", "drop"),
    "white|random_perclass|drop|regroup": (4, "white", "random_perclass", "drop_regroup"),
    "white|random_perclass|full": (2, "white", "random_perclass", "full"),
}


def run_arm(arm, r, bdsp_kept=None, rec=None):
    """One row: the module (through rn._run, as 147) or run_mirror with the arm's switch. rec: the draw, shared by the
    arms of one cell (features cached across them, as 147.run_unit across a draw's variants); None builds it."""
    _, cell, variant, how = ARMS[arm]
    mod = rn.load_pipe(PIPE)
    spec = rn._spec(mod, variant, "bdsp")
    own = rec is None
    if own:
        rec = draw_rec(cell, r)
        rn._clear_cache()
    ov, err, inside = overlay(mod, spec, r)
    if ov is None or not inside:
        return {"arm": arm, "r": r, "status": "extent_outside_window" if ov is not None else rn._status(err)}
    kw = {}
    if how in ("drop", "drop_regroup"):
        dr = np.random.default_rng([at.SEED, rn._crc("drop149"), r])
        kw.update(n_keep=int(bdsp_kept[dr.integers(len(bdsp_kept))]), drop_rng=dr)
    if how in ("regroup", "drop_regroup"):
        kw["regroup"] = True
    if how == "withinfold":
        kw["withinfold"] = True
    if how == "full":
        kw.update(early_stop=False, imbalance_rng=np.random.default_rng([at.SEED, rn._crc("imb149"), r]))
    fn = mod.run if how == "module" else (lambda rec_, ov_, v_, g_: run_mirror(mod, rec_, ov_, v_, g_, **kw))
    res, err, sec = rn._call_run(mod, rn._run, mod, fn, spec, rec, ov, run_rng(spec, r))
    if own:
        rn._clear_cache()
    on = np.asarray(ov.onsets_s, float)
    row = {"arm": arm, "r": r, "status": rn._status(err), "error": err, "seconds": sec,
           "min_ioi_s": float(np.diff(on).min()), "n_keep_target": kw.get("n_keep")}
    if res is not None:
        row.update(call=bool(res["call"]), p=res["p"], stat=res["stat"], n_kept=res["extra"]["n_kept"],
                   comparisons=res["extra"]["comparisons"])
    return row


def rows_file(arm):
    return os.path.join(DIR, arm.replace("|", "__") + ".jsonl")


def load_rows(arm):
    f = rows_file(arm)
    if not os.path.exists(f):
        return []
    with open(f) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def bdsp_rows():
    """114's cruse12p BDSP rows of the eligible variants with a fit (result rows only; no recording is read)."""
    d = sr.load_variant_rows({("ncmd_bdsp", PIPE): set(VARIANTS)})
    return d[d["p"].notna()]


def bdsp_n_kept():
    d = bdsp_rows()
    return np.sort(d[d["variant"] == "random_perclass"]["x_n_kept"].astype(int).to_numpy())


def swap_mb():
    return wn.swap_used_mb()


def run(n, steps, full_n=None):
    os.makedirs(DIR, exist_ok=True)
    swap0 = swap_mb()
    kept = bdsp_n_kept()
    for step in steps:
        for cell in dict.fromkeys(ARMS[a][1] for a in ARMS if ARMS[a][0] == step):
            arms = [a for a in ARMS if ARMS[a][:2] == (step, cell)]
            done = {a: {x["r"] for x in load_rows(a)} for a in arms}
            fhs = {a: open(rows_file(a), "a") for a in arms}
            for i in range(full_n if step == 2 and full_n else n):
                r = R0 + i
                todo = [a for a in arms if r not in done[a]]
                if not todo:
                    continue
                if i % 50 == 0 and swap_mb() > swap0 + SWAP_GROWTH_MB:
                    raise SystemExit(f"swap grew above {swap0 + SWAP_GROWTH_MB:.0f} MB: stopped (rerun resumes)")
                rec = draw_rec(cell, r)
                rn._clear_cache()
                for a in todo:
                    fhs[a].write(json.dumps(run_arm(a, r, kept, rec), default=float) + "\n")
                    fhs[a].flush()
                rn._clear_cache()
            for fh in fhs.values():
                fh.close()
            print("step", step, cell, "done", flush=True)



def rate(rows):
    fit = [x for x in rows if x.get("status") == "ok"]
    k, m = sum(x["call"] for x in fit), len(fit)
    ci = st.clopper_pearson(k, m)
    return {"calls": k, "n": m, "n_na": len(rows) - m, "rate": k / m if m else None,
            "ci95": None if ci is None else [float(ci[0]), float(ci[1])],
            "p_binom_greater_5pct": bm.binom_greater(k, m, NOMINAL)}


def mcnemar(a, b):
    """Exact two-sided McNemar on the draws both arms fit: (calls only in a, calls only in b, p)."""
    A = {x["r"]: x["call"] for x in a if x.get("status") == "ok"}
    B = {x["r"]: x["call"] for x in b if x.get("status") == "ok"}
    both = sorted(set(A) & set(B))
    n10 = sum(A[r] and not B[r] for r in both)
    n01 = sum(B[r] and not A[r] for r in both)
    m = n10 + n01
    p = 1.0 if m == 0 else float(min(1.0, 2 * binom.cdf(min(n10, n01), m, 0.5)))
    return {"n_paired": len(both), "only_first": n10, "only_second": n01, "p_exact": p}


def calibration(rows):
    fit = [x for x in rows if x.get("status") == "ok"]
    out = {"n_draws": len(fit)}
    sets = {"right": [x["comparisons"]["right"]["p"] for x in fit],
            "left": [x["comparisons"]["left"]["p"] for x in fit],
            "module_p_min_of_two": [x["p"] for x in fit]}
    sets["right_and_left_pooled"] = sets["right"] + sets["left"]
    for k, v in sets.items():
        v = np.asarray(v, float)
        if not len(v):
            continue
        ks = kstest(v, "uniform")
        out[k] = {"n": len(v), "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
                  **{f"share_below_{a:g}": float(np.mean(v < a)) for a in (0.05, 0.025, 0.01, 0.001)}}
    out["module_p_min_of_two"]["note"] = ("min of two p values: not uniform under a valid null even if each is "
                                          "(about 2x the share below a small alpha when the two are independent)")
    acc, nm = [], []
    for x in fit:
        for c in ("right", "left"):
            acc.append(x["comparisons"][c]["acc"])
            nm.append(x["comparisons"][c]["null_mean"])
    d = np.asarray(acc) - np.asarray(nm)
    out["observed_minus_null_mean_acc"] = {"n": len(d), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                           "observed_mean": float(np.mean(acc)), "null_mean": float(np.mean(nm))}
    io = [x["comparisons"][c]["imbalance_obs"] for x in fit for c in ("right", "left")]
    ip = [x["comparisons"][c]["imbalance_perm"] for x in fit for c in ("right", "left")]
    out["test_fold_class_imbalance"] = {"observed_groups_mean": float(np.mean(io)),
                                        "permuted_labels_same_groups_mean": float(np.mean(ip))}
    return out


def q(v, qs=(0, .1, .25, .5, .75, .9, 1)):
    v = np.asarray([x for x in v if x is not None], float)
    return None if not len(v) else dict(zip([f"q{int(x * 100)}" for x in qs], np.quantile(v, qs).round(3).tolist()))


def summarise():
    R = {a: load_rows(a) for a in ARMS}
    kept = bdsp_n_kept()
    with open(os.path.abspath(__file__), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    out = {"label": "ADDED LATER diagnostic, synthetic data only", "sha256_149": sha, "pipe": PIPE,
           "draws": {"n_planned": N, "r": [R0, R0 + N - 1], "generator": "129 at.noise, white seed for every cell",
                     "need_s": NEED_S, "uid": "diag149|r"},
           "rates": {a: rate(R[a]) for a in ARMS if R[a]},
           "paired": {}, "calibration_white_random_perclass": None,
           "spectrum_bdsp_like": "skipped: no median BDSP power-spectrum aggregate on disk",
           "mechanism": {}}
    pairs = [("white|random_perclass", "pink|random_perclass"),
             ("white|contiguous_perclass", "pink|contiguous_perclass"),
             ("white|random_perclass", "white|contiguous_perclass"),
             ("white|random_perclass", "white|random_perclass|regroup"),
             ("white|random_perclass|drop", "white|random_perclass|drop|regroup"),
             ("white|random_perclass", "white|random_perclass|drop"),
             ("white|random_perclass", "white|random_perclass|withinfold"),
             ("white|contiguous_perclass", "white|contiguous_perclass|withinfold")]
    for a, b in pairs:
        if R[a] and R[b]:
            out["paired"][f"{a} vs {b}"] = mcnemar(R[a], R[b])
    if R["white|random_perclass|full"]:
        out["calibration_white_random_perclass"] = calibration(R["white|random_perclass|full"])
    from scipy.stats import fisher_exact
    b = bdsp_rows()
    out["bdsp_vs_white"] = {}
    for v in VARIANTS:
        g = b[b["variant"] == v]
        k, n_ = int((g["call"] == True).sum()), len(g)
        w = out["rates"].get(f"white|{v}")
        if w:
            out["bdsp_vs_white"][v] = {"bdsp": [k, n_], "white": [w["calls"], w["n"]],
                                       "fisher_p": float(fisher_exact([[k, n_ - k],
                                                                       [w["calls"], w["n"] - w["calls"]]])[1])}
    m = out["mechanism"]
    m["trial_count"] = {"bdsp_n_kept": {"n": len(kept), **q(kept)},
                        **{f"{a}_n_kept": q([x.get("n_kept") for x in R[a] if x.get("status") == "ok"])
                           for a in ARMS if R[a]}}
    m["epochs"] = {"epoch_s": float(rn.load_pipe(PIPE).EPOCH_S),
                   "min_inter_onset_s": float(min(x["min_ioi_s"] for a in ARMS for x in R[a] if "min_ioi_s" in x)),
                   "outside_window": sum(x.get("status") == "extent_outside_window" for a in ARMS for x in R[a])}
    m["design_fold_imbalance"] = design_imbalance(N)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--steps", default="1,3,4,2")
    ap.add_argument("--full-n", type=int, default=None, help="draws of the step-2 arm (default --n)")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args(argv)
    if a.summarise:
        print(json.dumps(summarise()["rates"], indent=1))
        return
    run(a.n, [int(s) for s in a.steps.split(",")], a.full_n)


if __name__ == "__main__":
    main()
