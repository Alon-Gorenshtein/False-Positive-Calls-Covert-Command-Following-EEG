"""du Bois reconciliation: where the two implementations part. The blind implementation (blind_impl/dubois_blind.py,
written from the publication alone; 161) and this study's reading B (130, c3_b_assessment_blocked_phase1) disagree on
the same development windows. Here the blind implementation is moved toward reading B one unstated choice at a time,
by patching its module attributes for the duration of a call (its file is never edited), on the same windows and
schedules, so each choice's share of the call-rate gap is measured rather than argued.

Ladder (cumulative, order fixed here before any real window is run; each step changes one choice):
- L0_blind      the blind implementation as written and as 161 runs it (every recorded channel, random stratified folds).
- L1_folds      leave-one-block-out folds: fold = the trial's block (6 blocks of 15 alternating A B A B A B), the
                reading-B fold rule; random folds let slow drift that follows the block order into the test folds.
- L2_channels   the study's 17-site full set (Fp1 / Fp2 out) instead of every channel passed in.
- L3_peak       reading B's peak: the configuration with the highest smoothed fold-mean peak in 0.4-7 s, its peak value
                (no +-150-ms refinement to the unsmoothed maximum), per-fold smoothed DA at the task and baseline peak
                times, one-tailed paired t with the frozen zero-variance rule (130.reading_b).
- L4_step       0.1-s time grid (3-point smoothing) instead of 40 ms (5-point).
- L5_bands      the study's band edges (1-4, 4-8, 8-13, 13-20, 20-30, 30-40 Hz).
- L6_csp_pairs  2 CSP pairs per band instead of 3.
- L7_mi         mutual information by the Gaussian plug-in estimator (pipe_dubois.mi_scores) instead of 3-level bins.
- L8_lda        Ledoit-Wolf shrinkage with the class-prior offset (pipe_dubois.rlda) instead of a fixed 0.1 shrinkage.
What still differs from reading B after L8: the CSP covariance (blind: the 0.4-7 s interval; study: trailing windows at
the 0.5-s training times) and the training scheme (blind: one classifier per time point; study: one per fold on pooled
training times). SINGLES: each factor alone on L0. Study side: reading B with blocked (primary) and random folds.

Readings fixed here (descriptive, reported whatever they show): the share of the L0-to-reading-B gap that a factor
closes alone (gap_share of its single); "fold assignment is the main driver" if the L1 single closes >= 0.5 of the gap
with an exact McNemar p < .05 against L0; "the residual choices matter" if Cohen's kappa between L8 and reading B is
< 0.6. Units: one window (first overlay) per patient, the 200 development windows of 161 (same seed and listing).

python 170_dubois_reconcile.py --synthetic N [--drift D] [--workers W]   (synthetic 1/f recordings, no real EEG)
python 170_dubois_reconcile.py --dev [--limit N] [--workers W]            (the 161 development windows)
python 170_dubois_reconcile.py --merge ROWS.jsonl [--out PATH] -> output/tables/dubois_reconcile[_synthetic].json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import importlib
import json
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import beta, binom, norm, t as tdist
from sklearn.covariance import ledoit_wolf

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
for _p in (CODE, os.path.join(ROOT, "blind_impl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dubois_blind as db
from bench import motor_common as mc
from bench import pipe_dubois as pdb

OUT_DIR = os.path.join(ROOT, "output", "tables")
MARGIN_S = 5.0
HAND = {pdb.CLASS_A: "left", pdb.CLASS_B: "right"}
STUDY_BANDS = [tuple(float(v) for v in b) for b in pdb.BANDS.values()]
STUDY_READING = "c3_b_assessment_blocked_phase1"

LADDER = [("L0_blind", {})]
for _name, _k, _v in [("L1_folds", "folds", "blocked"), ("L2_channels", "channels", "study17"),
                      ("L3_peak", "peak", "reading_b"), ("L4_step", "step", 0.1), ("L5_bands", "bands", "study"),
                      ("L6_csp_pairs", "csp_pairs", 2), ("L7_mi", "mi", "gaussian"), ("L8_lda", "lda", "ledoit_wolf")]:
    LADDER.append((_name, {**LADDER[-1][1], _k: _v}))
SINGLES = {name: dict(set(ch.items()) - set(LADDER[i - 1][1].items())) for i, (name, ch) in enumerate(LADDER) if i}
FIXED_CONFIG = ("motor", (2, 3, 4, 5), 2.0, 7)
_CODE = {"random": "R", "blocked": "B", "refine": "x", "reading_b": "b", "search": "s", "fixed": "f"}
FACTORIAL = {f"F_{_CODE[f]}{_CODE[p]}{_CODE[c]}": {"folds": f, "peak": p, "config": c}
             for f in ("random", "blocked") for p in ("refine", "reading_b") for c in ("search", "fixed")}
N_PERM = 100


def events(ov, start_s):
    """(onset s from start_s, 'left' | 'right', block) per trial, in schedule order (161.dubois_events)."""
    return [(float(t) - start_s, HAND[int(l)], int(k)) for t, l, k in zip(ov.onsets_s, ov.labels, ov.meta["block"])]


def crop(rec, ov):
    """The overlay's extent plus MARGIN_S on each side, as 161 hands the blind implementation its window."""
    t0, t1 = (float(v) for v in pdb.extent(ov))
    a = max(0.0, t0 - MARGIN_S)
    i0, i1 = int(round(a * rec.fs)), min(rec.x.shape[1], int(round((t1 + MARGIN_S) * rec.fs)))
    return np.asarray(rec.x)[:, i0:i1], a


def select_channels(rec, which):
    """'all': every channel of the recording (161); 'study17': pipe_dubois's full set in its order."""
    if which == "all":
        return np.asarray(rec.x), list(rec.names)
    nm = mc.canon(rec.names)
    use = list(pdb.CHANSETS["full"])
    return np.asarray(rec.x)[[nm.index(c) for c in use]], use


def mi_gaussian(X, y):
    """pipe_dubois.mi_scores at every time point. X (T, n, F), y in {0, 1} -> (T, F)."""
    p1 = np.mean(y == 1)
    a, b = X[:, y == 0], X[:, y == 1]
    ma, sa = a.mean(1, keepdims=True), a.std(1, keepdims=True) + 1e-9
    mb, sb = b.mean(1, keepdims=True), b.std(1, keepdims=True) + 1e-9
    l0 = norm.logpdf(X, ma, sa) + np.log(1 - p1)
    l1 = norm.logpdf(X, mb, sb) + np.log(p1)
    q = 1.0 / (1.0 + np.exp(np.clip(l0 - l1, -50, 50)))
    hc = -np.mean(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12), 1)
    h = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    return h - hc


def rlda_lw_predict(Xtr, ytr, Xte, shrink=None):
    """pipe_dubois.rlda at every time point (Ledoit-Wolf pooled covariance, class-prior offset). -> (T, n_te) 0/1."""
    out = np.zeros(Xte.shape[:2], np.int8)
    for t in range(Xtr.shape[0]):
        a, b = Xtr[t, ytr == 0], Xtr[t, ytr == 1]
        ma, mb = a.mean(0), b.mean(0)
        S = ledoit_wolf(np.r_[a - ma, b - mb], assume_centered=True)[0]
        w = np.linalg.solve(S, mb - ma)
        b0 = -w @ (ma + mb) / 2 + np.log(len(b) / len(a))
        out[t] = (Xte[t] @ w + b0 > 0).astype(np.int8)
    return out


def _paired_greater(d):
    """130._paired_greater: one-tailed paired t, frozen zero-variance rule."""
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return (np.inf if m > 0 else 0.0), (0.0 if m > 0 else 1.0)
    tv = m / (s / np.sqrt(len(d)))
    return float(tv), float(tdist.sf(tv, len(d) - 1))


def reading_b_rule(results, times, step):
    """Reading B on per-fold DA curves {config: (k, T)} on the grid `times` (s): 200-ms centred boxcar (3 points at
    0.1 s, as reading B; on the blind's ~40-ms grid its own round(200 ms / step) points, so the peak-rule step changes
    the peak rule only, not the smoothing width), configuration with the highest fold-mean peak in the task interval, its peak value, per-fold values at
    the fold-mean task and baseline peak times, one-tailed paired t; call = peak > 0.70 or p < 0.05."""
    times = np.asarray(times, float)
    n = 3 if step > 0.07 else max(1, int(round(db.SMOOTH_S / step)))
    task = (times >= pdb.TASK[0] - 1e-9) & (times <= pdb.TASK[1] + 1e-9)
    bm = (times >= pdb.BASE[0] - 1e-9) & (times <= pdb.BASE[1] + 1e-9)
    best, best_v, best_sm = None, -np.inf, None
    for key, da in results.items():
        sm = uniform_filter1d(np.asarray(da, float), n, axis=-1, mode="nearest")
        v = sm.mean(0)[task].max()
        if v > best_v:
            best, best_v, best_sm = key, v, sm
    mean = best_sm.mean(0)
    ti = int(np.nonzero(task)[0][np.argmax(mean[task])])
    bi = int(np.nonzero(bm)[0][np.argmax(mean[bm])])
    tv, p = _paired_greater(best_sm[:, ti] - best_sm[:, bi])
    return {"call": bool(best_v > pdb.DA_70 or p < pdb.ALPHA), "peak_da": float(best_v), "p_t": float(p), "t": tv,
            "config": best, "t_peak_s": float(times[ti]), "t_base_s": float(times[bi])}


def blind_selection(results, times, fs, step_s):
    """The blind implementation's own configuration choice and refined task peak, recomputed from its per-fold DA
    (dubois_blind.assess: smoothed peak, refinement on the unsmoothed curve, ties by the smoothed value).
    -> (config, task time index, task peak)."""
    step = max(1, int(round(step_s * fs)))
    smooth_n = max(1, int(round(db.SMOOTH_S / (step / fs))))
    refine_n = max(0, int(round(db.REFINE_S / (step / fs))))
    task = (times >= db.TASK[0] - 1e-9) & (times <= db.TASK[1] + 1e-9)
    t_lo, t_hi = np.flatnonzero(task)[[0, -1]]
    best = None
    for key, da in results.items():
        ti, tv, ts = db._peak(da.mean(axis=0), t_lo, t_hi, step, smooth_n, refine_n)
        if best is None or (tv, ts) > (best[1], best[2]):
            best = (key, tv, ts, ti)
    return best[0], best[3], best[1]


@contextlib.contextmanager
def patched(choices, blocks):
    """Set the blind module's attributes for `choices`; always restore them. choices['config']: 'search' (all 84),
    'fixed' (FIXED_CONFIG, chosen before any run: the motor set, mu to low gamma, 2-s window, 7 features) or a
    configuration tuple (the one selected on the true labels, for the permutation evaluation)."""
    saved = {k: getattr(db, k) for k in ("_folds", "_mutual_info", "_rlda_predict", "BANDS", "N_PAIRS", "STEP_S",
                                         "BAND_SETS", "WINDOWS", "N_FEATURES", "MOTOR_SET")}
    try:
        cfg = choices.get("config", "search")
        cfg = FIXED_CONFIG if cfg == "fixed" else cfg
        if cfg != "search":
            cs, bs, win, nf = cfg
            db.BAND_SETS, db.WINDOWS, db.N_FEATURES = [tuple(bs)], (float(win),), (int(nf),)
            if cs == "full":
                db.MOTOR_SET = []
        if choices.get("folds") == "blocked":
            def _block_folds(y, k, rng):
                if len(blocks) != len(y):
                    raise ValueError(f"{len(blocks)} block ids for {len(y)} kept trials")
                return np.asarray(blocks, int).copy()
            db._folds = _block_folds
        if "step" in choices:
            db.STEP_S = float(choices["step"])
        if choices.get("bands") == "study":
            db.BANDS = list(STUDY_BANDS)
        if "csp_pairs" in choices:
            db.N_PAIRS = int(choices["csp_pairs"])
        if choices.get("mi") == "gaussian":
            db._mutual_info = mi_gaussian
        if choices.get("lda") == "ledoit_wolf":
            db._rlda_predict = rlda_lw_predict
        yield
    finally:
        for k, v in saved.items():
            setattr(db, k, v)


def _config_rows(x, names, cfg):
    """Rows of x the blind implementation sees for a fixed configuration: the motor set for 'motor', else all."""
    cfg = FIXED_CONFIG if cfg == "fixed" else cfg
    if cfg in ("search", None) or cfg[0] != "motor":
        return x, names
    nm = [db._canon(n) for n in names]
    keep = [nm.index(c) for c in db.MOTOR_SET if c in nm]
    return x[keep], [names[i] for i in keep]


def _assess(rec, ov, choices, seed, labels=None, blocks_override=None):
    x, names = select_channels(rec, choices.get("channels", "all"))
    x, names = _config_rows(x, names, choices.get("config", "search"))
    sub = type("R", (), {"x": x, "fs": rec.fs})()
    xc, a = crop(sub, ov)
    ev = events(ov, a)
    if labels is not None:
        ev = [(t, lab, k) for (t, _, k), lab in zip(ev, labels)]
    blocks = np.array([e[2] for e in ev]) if blocks_override is None else np.asarray(blocks_override)
    with patched(choices, blocks):
        db._LAST.clear()
        out = db.assess(xc, rec.fs, names, [e[:2] for e in ev], seed=seed)
        res = dict(db._LAST) if db._LAST else {}
        step = db.STEP_S
    return out, res, step, [e[1] for e in ev]


def run_blind_perm(rec, ov, choices, seed=0, n_perm=N_PERM, refit_search=False):
    """The publication's chance evaluation on the blind implementation: the configuration selected on the true labels
    (refit_search=False, as 'the full six-fold CV procedure was repeated') or the whole 84-configuration search
    repeated in every draw (refit_search=True); n_perm free permutations of the trial labels, each with its own
    six-fold CV (random folds re-drawn per permutation; blocked folds stay the blocks); p at the observed refined peak
    time = (#{permuted DA >= observed} + 1) / (n + 1). Calls: perm (p < .05), advancement (the blind rule),
    both. -> dict."""
    out, res, step, labs = _assess(rec, ov, choices, seed)
    if not res:
        return {"call": None, "status": f"no_fit: {out.get('notes', '')}"[:160], "adv_call": bool(out["call"]),
                "adv_and_perm": None}
    cfg, ti, peak = blind_selection(res["results"], res["times"], rec.fs, step)
    obs = float(res["results"][cfg].mean(0)[ti])
    g = np.random.default_rng([20261029, 170, seed])
    pch = dict(choices) if refit_search else {**choices, "config": cfg}
    ge = 0
    for _ in range(n_perm):
        lp = list(g.permutation(labs))
        _, rp, sp, _ = _assess(rec, ov, pch, seed, labels=lp)
        if not rp:
            ge += 1
            continue
        if refit_search:
            c2, t2, _ = blind_selection(rp["results"], rp["times"], rec.fs, sp)
            v = float(rp["results"][c2].mean(0)[t2])
        else:
            if len(rp["results"]) != 1:
                raise RuntimeError(f"fixed configuration ran {len(rp['results'])} configurations")
            v = float(next(iter(rp["results"].values())).mean(0)[ti])
        ge += v >= obs - 1e-12
    p = (ge + 1.0) / (n_perm + 1)
    return {"call": bool(p < pdb.ALPHA), "p_perm": p, "peak_da": float(peak), "da_at_peak": obs,
            "adv_call": bool(out["call"]), "adv_and_perm": bool(out["call"] and p < pdb.ALPHA),
            "config": [str(c) for c in cfg], "refit_search": bool(refit_search)}


def run_blind(rec, ov, choices, seed=0, blocks_override=None):
    """The blind implementation under `choices` on the 161 crop of rec around ov. -> dict(call, peak_da, p_t, ...)."""
    out, res, step, _ = _assess(rec, ov, choices, seed, blocks_override=blocks_override)
    if not res:
        return {"call": None, "status": f"no_fit: {out.get('notes', '')}"[:160]}
    if choices.get("peak") == "reading_b":
        rb = reading_b_rule(res["results"], res["times"], max(1, int(round(step * rec.fs))) / rec.fs)
        rb["config"] = [str(c) for c in rb["config"]]
        return {**rb, "blind_rule_call": bool(out["call"])}
    return {"call": bool(out["call"]), "peak_da": float(out["peak_da"]), "p_t": float(out["p_t"])}


def run_study(rec, ov, cv, rng):
    """Reading B (130.dubois) with blocked (primary) or random folds."""
    fr = importlib.import_module("130_fidelity_readings")
    r = fr.dubois(rec, ov, f"c3_b_assessment_{cv}_phase1", rng=rng)
    return {"call": bool(r["call"]), "peak_da": float(r["extra"]["peak_da"]), "p_t": float(r["p"])}


def run_unit_v2(rec, ov, unit, seed=0, refit_search=False):
    """The literal-specification arm: the 2 x 2 x 2 factorial (folds x peak rule x configuration search) and the
    publication's permutation evaluation under each fold rule -> rows."""
    rows = []
    for name, ch in FACTORIAL.items():
        peak = ch["peak"]
        r = run_blind(rec, ov, {"folds": ch["folds"], "config": ch["config"],
                                **({"peak": "reading_b"} if peak == "reading_b" else {})}, seed)
        rows.append({"unit": unit, "step": name, **r})
    for folds in ("random", "blocked"):
        r = run_blind_perm(rec, ov, {"folds": folds}, seed, refit_search=refit_search)
        tag = "permS" if refit_search else "perm"
        rows.append({"unit": unit, "step": f"P_{folds}_{tag}", **r})
        rows.append({"unit": unit, "step": f"P_{folds}_adv_and_{tag}", **r, "call": r["adv_and_perm"]})
    return rows


def run_unit(rec, ov, unit, seed=0, study_rng=None):
    """Every ladder step, every single, both study readings on one window -> rows."""
    rows = []
    for name, ch in LADDER:
        rows.append({"unit": unit, "step": name, **run_blind(rec, ov, ch, seed)})
    for name, ch in SINGLES.items():
        if name == "L1_folds":
            rows.append({**[r for r in rows if r["step"] == "L1_folds"][0], "step": "S_L1_folds"})
            continue
        rows.append({"unit": unit, "step": "S_" + name, **run_blind(rec, ov, ch, seed)})
    for cv in ("blocked", "random"):
        rows.append({"unit": unit, "step": f"study_{cv}", **run_study(rec, ov, cv, study_rng)})
    return rows


def _cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def _kappa(a, c):
    a, c = np.asarray(a, bool), np.asarray(c, bool)
    po = float((a == c).mean())
    pe = float(a.mean() * c.mean() + (1 - a.mean()) * (1 - c.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def summarize(rows):
    """Call rate per step (Clopper-Pearson), exact McNemar between consecutive ladder steps and between L0 and each
    single, kappa against reading B (blocked), and each single's share of the L0-to-reading-B gap. Units with any
    missing step are left out of every comparison."""
    by = {}
    for r in rows:
        if r.get("call") is not None:
            by.setdefault(r["unit"], {})[r["step"]] = bool(r["call"])
    steps = sorted({r["step"] for r in rows})
    units = [u for u, d in by.items() if all(s in d for s in steps)]
    calls = {s: np.array([by[u][s] for u in units]) for s in steps}
    n = len(units)
    out = {"n_units": n, "steps": {}, "mcnemar": {}, "gap_share": {}, "kappa_vs_study_blocked": {}}
    for s in steps:
        k = int(calls[s].sum())
        out["steps"][s] = {"k": k, "n": n, "rate": k / n if n else None, "ci95": _cp(k, n) if n else None}

    def mcn(a, c):
        b_, c_ = int(np.sum(calls[a] & ~calls[c])), int(np.sum(~calls[a] & calls[c]))
        p = float(min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5))) if b_ + c_ else 1.0
        return {"b": b_, "c": c_, "p_exact": p}
    names = [s for s, _ in LADDER if s in calls]
    for a, c in zip(names, names[1:]):
        out["mcnemar"][f"{a}->{c}"] = mcn(a, c)
    ref = "study_blocked"
    for s in steps:
        if s.startswith("S_") and "L0_blind" in calls:
            out["mcnemar"][f"L0_blind->{s}"] = mcn("L0_blind", s)
            if ref in calls:
                gap = calls["L0_blind"].mean() - calls[ref].mean()
                out["gap_share"][s] = float((calls["L0_blind"].mean() - calls[s].mean()) / gap) if gap else None
        if ref in calls:
            out["kappa_vs_study_blocked"][s] = _kappa(calls[s], calls[ref])
    if "S_L1_folds" in out["gap_share"]:
        m = out["mcnemar"].get("L0_blind->S_L1_folds", {})
        out["reading_folds_main_driver"] = bool((out["gap_share"]["S_L1_folds"] or 0) >= 0.5
                                                and m.get("p_exact", 1) < 0.05)
    if "L8_lda" in calls and ref in calls:
        out["reading_residual_matters"] = bool(_kappa(calls["L8_lda"], calls[ref]) < 0.6)
    return out


def _synthetic_unit(args):
    i, drift, arm, refit = args
    from bench.tests import _synth as S
    g = np.random.default_rng([20261029, i])
    ov = pdb.schedule(30.0, g, "assessment_blocked_phase1")
    rec = S.rec(pdb.extent(ov)[1] + 10.0, seed=int(g.integers(2 ** 31)), drift=drift)
    study_rng = int(g.integers(2 ** 31))
    if arm == "v2":
        return run_unit_v2(rec, ov, f"syn{i}", seed=0, refit_search=refit)
    return run_unit(rec, ov, f"syn{i}", seed=0, study_rng=study_rng)


def _dev_unit(item):
    """One 161 development window: the same overlay (first published dubois overlay of reading B's base) and the
    stored reading-B call, which study_blocked must reproduce (matches_stored)."""
    uid, stored, arm, refit = item
    b161 = importlib.import_module("161_blind_agreement")
    rp = b161.rp
    rec = rp.b.en._load(rp.dev_item(uid))
    mod = rp.b.load_pipe("dubois")
    jobs, _ = rp.fid.phase1_jobs(rec, mod, rp.rep_spec("dubois"), 974, first_only=True)
    jb = [j for j in jobs if j["reading"] == b161.DUBOIS_READING and j["ov"] is not None]
    if not jb:
        return []
    if arm == "v2":
        return run_unit_v2(rec, jb[0]["ov"], uid, seed=0, refit_search=refit)
    rows = run_unit(rec, jb[0]["ov"], uid, seed=0, study_rng=jb[0]["rng"])
    for r in rows:
        if r["step"] == "study_blocked":
            r["stored_call"], r["matches_stored"] = bool(stored), bool(r["call"]) == bool(stored)
    return rows


def _pool(fn, items, workers):
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(fn, items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arm", choices=("ladder", "v2"), default="ladder")
    ap.add_argument("--refit-search", action="store_true")
    ap.add_argument("--rows")
    ap.add_argument("--merge")
    ap.add_argument("--out", help="merge output (default output/tables/dubois_reconcile[_synthetic].json)")
    a = ap.parse_args()
    if a.merge:
        rows = [json.loads(l) for l in open(a.merge)]
        res = {"what": __doc__.split("\n")[0], **summarize(rows)}
        tag = "_synthetic" if all(r["unit"].startswith("syn") for r in rows) else ""
        os.makedirs(OUT_DIR, exist_ok=True)
        out = a.out or os.path.join(OUT_DIR, f"dubois_reconcile{tag}.json")
        json.dump(res, open(out, "w"), indent=1, default=str)
        print(json.dumps({k: res[k] for k in ("n_units", "gap_share")}, indent=1, default=str))
        return
    if a.synthetic:
        items, fn = [(i, a.drift, a.arm, a.refit_search) for i in range(a.synthetic)], _synthetic_unit
        tag = "" if a.arm == "ladder" else f"_{a.arm}{'S' if a.refit_search else ''}"
        default = os.path.join(ROOT, "output", "intermediate", f"dubois_reconcile{tag}_syn_d{a.drift:g}.jsonl")
    else:
        b161 = importlib.import_module("161_blind_agreement")
        uids, stored = b161.windows()
        items, fn = [(u, stored[u], a.arm, a.refit_search) for u in uids], _dev_unit
        tag = "" if a.arm == "ladder" else f"_{a.arm}{'S' if a.refit_search else ''}"
        default = os.path.join(ROOT, "output", "intermediate", f"dubois_reconcile{tag}_dev.jsonl")
    items = items[:a.limit] if a.limit else items
    path = a.rows or default
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {json.loads(l)["unit"] for l in open(path)} if os.path.exists(path) else set()
    todo = [it for it in items if (f"syn{it[0]}" if a.synthetic else it[0]) not in done]
    with open(path, "a") as fh:
        for k, rows in enumerate(_pool(fn, todo, a.workers)):
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(k + 1, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
