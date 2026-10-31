"""Time-matched physiological specificity of the Cornell test: left- against right-hand motor imagery at the same time
after the cue, in acute stroke (stroke_acute_io; the 47 distinct patients of 163, sub-05 / 15 / 49 excluded as
duplicates there). Rest never enters, so a call cannot come from the difference between rest and an instructed period;
the hand-specific video shown during imagery (163) is a known visual confound, read out by where the bins lie.

Design, fixed here before any patient is analysed:
- Recording: the continuous EDF (EEG rows of stroke_acute_io.eeg_rows, FP1 / FP2 named Fp1 / Fp2), resampled to 250 Hz,
  with 10 s of reflected signal at each end (added after the first run stopped on the first patient: the first cue
  lies inside pc.extent's margin);
  one command per trial at its MI onset (stroke_acute_io.mi_onsets_grid); Cornell's own features (pc.features) with
  CMD_S 0, RESP (0.5, 3.5), N_EP 1 (one 3-s epoch inside the 4-s imagery); pseudo-runs of 10 consecutive trials (4 runs,
  5 left and 5 right each, the trials strictly alternating); labels TASK = left, REST = right.
- Readings: 171.decide with the sample-level jackknife (the primary c3 reading) and the command-level jackknife
  (normal reference), same features.
- Null: the same epochs with the labels of exactly half of the 20 consecutive left / right pairs swapped (random which,
  seed [20261030, 177, subject, draw]), 20 draws per patient; every pseudo-class keeps 20 trials, half of each hand.
- Readouts (descriptive): detection (share of patients called with true labels, exact CI); null rate (mean over
  patients of the share of draws called, patient-cluster bootstrap CI, B = 2000); among true-label calls, the share
  with any FDR-surviving bin at C3 or C4 in 8-24 Hz with the contralateral sign (left minus right log power < 0 at C4 or
  > 0 at C3) and the share whose bins lie only at posterior sites (171.POSTERIOR). Reading fixed here: the calls are
  'sensorimotor-specific' if at least half of the true-label calls carry the contralateral sign and at most a quarter
  are posterior-only.
python 177_cornell_lr.py [--workers 4] [--limit N] | --merge -> output/tables/cornell_lr_specificity.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import beta, binom

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from bench import motor_common as mc
from bench import pipe_cornell as pc
from bench import wrappers as wr

tj = importlib.import_module("171_cornell_trialjack")
GLOBALS = {"CMD_S": 0.0, "RESP": (0.5, 3.5), "N_EP": 1}
RUN_TRIALS, N_DRAWS, B, FS, PAD_S = 10, 20, 2000, 250.0, 10.0
EXCLUDED = (5, 15, 49)
READINGS = {"sample": ("sample", "normal"), "command": ("command", "normal")}
BAND = (8.0, 24.0)
ROWS = os.path.join(ROOT, "output", "intermediate", "cornell_lr_rows.jsonl")
OUT = os.path.join(ROOT, "output", "tables", "cornell_lr_specificity.json")


def overlay(onsets_s, hands):
    """One command per trial; TASK = left, REST = right; pseudo-runs of RUN_TRIALS consecutive trials."""
    n = len(onsets_s)
    run = np.arange(n) // RUN_TRIALS
    lab = np.array([pc.TASK if h == "left" else pc.REST for h in hands])
    return mc.Overlay(onsets_s=np.asarray(onsets_s, float), labels=lab, kind="cmd",
                      meta={"trial": np.arange(n), "block": run, "run": run})


def pair_swap(labels, rng):
    """Labels with exactly half of the consecutive (2i, 2i + 1) pairs swapped."""
    lab = np.array(labels, copy=True)
    n_pairs = len(lab) // 2
    for p in rng.choice(n_pairs, size=n_pairs // 2, replace=False):
        lab[2 * p], lab[2 * p + 1] = lab[2 * p + 1], lab[2 * p]
    return lab


def features(rec, ov):
    with wr.module_globals(pc, GLOBALS):
        return dict(pc.features(rec, ov))


def bins(F, lab_ep, unit):
    """The decision of 171.decide plus where its pooled FDR bins lie: (result, contralateral, posterior_only)."""
    res = tj.decide(F, lab_ep, unit, "normal", tj.BASE)
    c = mc.cfg(pc.VARIANTS, tj.BASE)
    keep = pc.kept(F, lab_ep, c["prune"])
    m1, m2 = keep & (lab_ep == pc.TASK), keep & (lab_ep == pc.REST)
    z, p, _ = tj._zp(F["S"][m1], F["S"][m2], F["cmd"][m1], F["cmd"][m2], unit, "normal")
    rej = mc.fdr_reject(p.ravel(), pc.Q_FDR, c["fdr"]).reshape(z.shape)
    nm = mc.canon(pc.NATIVE_CH)
    fb = (F["f"] >= BAND[0]) & (F["f"] <= BAND[1])
    c3, c4 = nm.index("C3"), nm.index("C4")
    contra = bool(np.any(rej[c4, fb] & (z[c4, fb] < 0)) or np.any(rej[c3, fb] & (z[c3, fb] > 0)))
    post = np.array([n in tj.POSTERIOR for n in nm])
    posterior_only = bool(rej.any() and not rej[~post].any())
    return res, contra, posterior_only


def load(subj):
    import mne
    sa = importlib.import_module("stroke_acute_io")
    raw = mne.io.read_raw_edf(sa.edf_path(subj), preload=True, verbose="ERROR")
    x = raw.get_data() * 1e6
    rows = sa.eeg_rows(raw.ch_names, x.std(axis=1))
    names = [{"FP1": "Fp1", "FP2": "Fp2"}.get(raw.ch_names[i], raw.ch_names[i]) for i in rows]
    y = sa.labels_from_events(open(os.path.join(sa.RAW, "task-motor-imagery_events.tsv")).read())
    on = sa.mi_onsets_grid(x[-1], int(sa.TRIAL_S * sa.FS_IN), int(sa.MI_S * sa.FS_IN), int(0.05 * sa.FS_IN), len(y))
    sig = pad(mc.resample(x[rows], sa.FS_IN, FS))
    rec = SimpleNamespace(x=sig, fs=FS, names=names, uid=f"sub-{subj:02d}")
    return rec, np.asarray(on, float) / sa.FS_IN + PAD_S, list(y)


def pad(sig):
    """PAD_S of reflected signal at each end: the first cue lies about 2 s into each recording, inside the margin
    pc.extent needs; reflection keeps the 1-50 Hz filter's edge transient away from the analysed epochs."""
    n = int(PAD_S * FS)
    return np.pad(sig, ((0, 0), (n, n)), mode="reflect")


def patient(subj):
    rec, on, y = load(subj)
    ov = overlay(on, y)
    F = features(rec, ov)
    true = np.asarray(ov.labels)
    rows = []
    for draw in range(N_DRAWS + 1):
        lab = true if draw == 0 else pair_swap(true, np.random.default_rng([20261030, 177, subj, draw]))
        lab_ep = lab[F["cmd"]]
        for name, (unit, _) in READINGS.items():
            try:
                res, contra, post_only = bins(F, lab_ep, unit)
                rows.append({"subject": subj, "draw": draw, "version": "true" if draw == 0 else "null",
                             "reading": name, "call": bool(res["call"]), "contralateral": contra,
                             "posterior_only": post_only, "vif": res["extra"]["vif"]})
            except mc.NoUsableTrials as e:
                rows.append({"subject": subj, "draw": draw, "reading": name, "call": None, "why": str(e)[:100]})
    return rows


def _cp(k, n):
    return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]


def summarize(rows):
    out = {}
    for name in READINGS:
        r = [x for x in rows if x["reading"] == name and x.get("call") is not None]
        true = {x["subject"]: x for x in r if x["draw"] == 0}
        null = {}
        for x in r:
            if x["draw"] > 0:
                null.setdefault(x["subject"], []).append(float(x["call"]))
        subs = sorted(set(true) & set(null))
        k = sum(true[s]["call"] for s in subs)
        rates = np.array([np.mean(null[s]) for s in subs])
        g = np.random.default_rng([20261030, 177])
        boot = rates[g.integers(0, len(rates), size=(B, len(rates)))].mean(1) if len(rates) else np.array([np.nan])
        calls = [true[s] for s in subs if true[s]["call"]]
        contra = sum(c["contralateral"] for c in calls)
        post = sum(c["posterior_only"] for c in calls)
        ncalls = [x for x in r if x["draw"] > 0 and x["call"] and x["subject"] in subs]
        p_null = float(binom.sf(int(k) - 1, len(subs), float(rates.mean()))) if len(rates) else None
        out[name] = {"n_patients": len(subs), "detection": {"k": int(k), "n": len(subs), "ci95": _cp(int(k), len(subs))},
                     "null_rate": {"estimate": float(rates.mean()) if len(rates) else None,
                                   "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
                     "true_calls_contralateral": int(contra), "true_calls_posterior_only": int(post),
                     "null_calls": len(ncalls), "null_calls_contralateral": int(sum(x["contralateral"] for x in ncalls)),
                     "null_calls_posterior_only": int(sum(x["posterior_only"] for x in ncalls)),
                     "p_detection_vs_null_rate": p_null,
                     "sensorimotor_specific": bool(calls) and contra >= len(calls) / 2 and post <= len(calls) / 4}
    return out


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
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        rows = [json.loads(l) for l in open(ROWS) if l.strip()]
        res = {"what": __doc__.split("\n")[0], "excluded_duplicates": [f"sub-{s:02d}" for s in EXCLUDED],
               "readings": summarize(rows)}
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(json.dumps(res, indent=1, default=float))
        return
    subs = [s for s in range(1, 51) if s not in EXCLUDED]
    done = {json.loads(l)["subject"] for l in open(ROWS)} if os.path.exists(ROWS) else set()
    todo = [s for s in subs if s not in done][:a.limit]
    with open(ROWS, "a") as fh:
        for k, rows in enumerate(_pool(patient, todo, a.workers)):
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            print(k + 1, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
