"""Checks for an EEG positive control in stroke patients (two open datasets, one decision rule each).
Gait (Liu 2025, Sci Data 12:314; stroke_gait_io): first 3 subjects by ID, session 'pre' only (pure MI, no stimulation).
  (a) imagery-vs-idle CSP + LDA at 8-30 Hz, leave-one-run-out; excerpt window 0.5-4.5 s after the imagery / idle onset.
  (b) 8-30 Hz log band power at C3, Cz, C4 (their mean is the subject's statistic): imagery minus idle, one-sided
      (imagery lower = desynchronization), 1,000 trial-label permutations WITHIN run (runs differ in level).
  (c) task-window minus pre-cue-fixation log power per trial, imagery and idle trials separately, one-sided sign-flip
      test (1,000); idle 'desynchronizes like imagery' in a subject if its p < .05 AND its mean log ratio is at least
      half the imagery one (imagery ratio < 0).
  NOGO if median (a) < 60%, OR no subject has (b) P < .05, OR idle is like imagery in a majority of the subjects.
Acute (stroke_acute_io, 50 patients, 40 trials): mindBEAGLE-type rule = CSP (2 filters per end, fewer if < 4 channels)
  + LDA on the 8-30 Hz 4-s MI epoch, 10 x 10-fold stratified CV, left vs right, per patient; once on sensorimotor
  channels (FC / C / CP) and once on occipital channels (O1, O2, Oz; no PO site exists). NOGO if the occipital median
  accuracy >= the sensorimotor one (a visual confound: the MI video differs by hand). Chance bound at 40 trials =
  Clopper-Pearson two-sided 95% upper limit at k = n/2, the formula that gives 63.19% at 60.
python 162_stroke_check.py [--gait] [--acute] -> output/tables/stroke_check.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from scipy.stats import beta
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
OUT = os.path.join(ROOT, "output", "tables", "stroke_check.json")

FS = 250.0
BAND = (8.0, 30.0)
TASK_WIN_S = (0.5, 4.5)
FIX_WIN_S = (0.0, 3.0)
GAIT_SUBJECTS = 3
ERD_CH = ("C3", "Cz", "C4")
B_PERM = 1000
SEED = 20261022
ACC_BAR, ALPHA = 0.60, 0.05
CSP_K = 2



def chance_bound(n, conf=0.95):
    """Upper limit of the Clopper-Pearson interval of k = n / 2 correct of n (two-sided `conf`): 63.19% at n = 60."""
    k = n / 2
    return float(beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))


def bandpass(x, fs=FS, band=BAND):
    return sosfiltfilt(butter(4, band, btype="bandpass", fs=fs, output="sos"), x, axis=-1)


def _cov(X):
    X = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", X, X)
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


def csp(X, y, k=CSP_K):
    """Common spatial patterns: k filters from each end of the generalized eigen-decomposition of the trace-normalized
    class-mean covariances (class 0 vs 1); k is reduced to n_ch // 2 when there are few channels. -> (n_ch, 2k)."""
    C = _cov(X)
    Ca, Cb = C[y == 0].mean(0), C[y == 1].mean(0)
    k = min(k, X.shape[1] // 2)
    ridge = 1e-9 * np.eye(X.shape[1])
    _, V = eigh(Ca, Ca + Cb + ridge)
    return V[:, list(range(k)) + list(range(V.shape[1] - k, V.shape[1]))]


def features(X, W):
    v = np.einsum("nct,ck->nkt", X - X.mean(-1, keepdims=True), W).var(-1)
    return np.log(v / v.sum(1, keepdims=True))


def fit_predict(Xtr, ytr, Xte, k=CSP_K, W=None):
    W = csp(Xtr, ytr, k) if W is None else W
    lda = LinearDiscriminantAnalysis().fit(features(Xtr, W), ytr)
    return lda.predict(features(Xte, W))


def cv_accuracy(X, y, folds=10, reps=10, seed=SEED, k=CSP_K, _leak=False):
    """Mean over `reps` stratified shuffled `folds`-fold partitions of the pooled out-of-fold accuracy. CSP is fitted inside
    each training fold (_leak=True fits it on all trials: a deliberately broken variant for the guard test)."""
    y = np.asarray(y)
    if min(np.bincount(y)) < folds:
        raise ValueError(f"fewer than {folds} trials in a class: cannot make {folds} stratified folds")
    Wall = csp(X, y, k) if _leak else None
    accs = []
    for r in range(reps):
        ok = np.zeros(len(y), bool)
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            ok[te] = fit_predict(X[tr], y[tr], X[te], k, W=Wall) == y[te]
        accs.append(ok.mean())
    return float(np.mean(accs))


def loro_accuracy(X, y, run, k=CSP_K):
    """Leave-one-run-out: train on the other runs, test on the held-out run. -> (pooled accuracy, per-run accuracies)."""
    y, run = np.asarray(y), np.asarray(run)
    per, ok = [], np.zeros(len(y), bool)
    for r in np.unique(run):
        te = run == r
        ok[te] = fit_predict(X[~te], y[~te], X[te], k) == y[te]
        per.append(float(ok[te].mean()))
    return float(ok.mean()), per


def permute_within(lab, run, rng):
    q = np.array(lab).copy()
    for r in np.unique(run):
        i = np.flatnonzero(run == r)
        q[i] = rng.permutation(q[i])
    return q


def perm_diff(v, lab, run, B=B_PERM, rng=None):
    """Mean(v | lab 1) - mean(v | lab 0) and its one-sided P that the difference is this negative or more (label 1 lower),
    from B label permutations within run. P = (1 + #{null <= obs}) / (B + 1)."""
    rng = rng or np.random.default_rng(SEED)
    v, lab = np.asarray(v, float), np.asarray(lab)
    obs = v[lab == 1].mean() - v[lab == 0].mean()
    null = np.empty(B)
    for b in range(B):
        q = permute_within(lab, run, rng)
        null[b] = v[q == 1].mean() - v[q == 0].mean()
    return float(obs), float((1 + (null <= obs).sum()) / (B + 1))


def signflip_p(d, B=B_PERM, rng=None):
    """One-sided sign-flip test that mean(d) < 0: P = (1 + #{null mean <= obs mean}) / (B + 1)."""
    rng = rng or np.random.default_rng(SEED)
    d = np.asarray(d, float)
    null = (rng.choice([-1.0, 1.0], size=(B, len(d))) * d).mean(1)
    return float((1 + (null <= d.mean()).sum()) / (B + 1))


def log_power(x, i0, i1):
    """ln of the mean square of x[..., i0:i1] -> (n, ch)."""
    return np.log(np.mean(np.asarray(x, float)[..., i0:i1] ** 2, axis=-1))



def gait_nogo(subjects):
    """subjects: dicts with acc, erd_p, idle_like_imagery. Fires on median acc < 60%, no subject with erd_p < .05, or idle
    like imagery in a majority of subjects."""
    acc = float(np.median([s["acc"] for s in subjects]))
    low = acc < ACC_BAR
    no_erd = not any(s["erd_p"] < ALPHA for s in subjects)
    like = sum(bool(s["idle_like_imagery"]) for s in subjects) > len(subjects) / 2
    return {"median_accuracy": acc, "low_accuracy": bool(low), "no_erd": bool(no_erd),
            "idle_like_imagery": bool(like), "nogo": bool(low or no_erd or like)}


def acute_nogo(sensorimotor_median, occipital_median):
    return {"sensorimotor_median": float(sensorimotor_median), "occipital_median": float(occipital_median),
            "nogo": bool(occipital_median >= sensorimotor_median)}



def write_section(name, d, path=OUT):
    cur = json.load(open(path)) if os.path.exists(path) else {}
    cur[name] = d
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(cur, fh, indent=1)
    os.replace(path + ".tmp", path)



def trial_table(sources):
    """Each imagery / idle excerpt with its own pre-cue fixation (same run, the baseline marker 4 s before the onset)."""
    fix = [q for q in sources if q.cls == "fixation"]
    out = []
    for t in sources:
        if t.cls not in ("mi", "idle"):
            continue
        f = [q for q in fix if q.run == t.run and q.tag.split(":")[1].startswith("fix-" + t.cls)
             and abs((t.start - q.start) - 4 * FS) <= 0.2 * FS]
        if len(f) == 0 and t.start < 4 * FS + 0.2 * FS:
            out.append({"cls": t.cls, "task": t, "fix": None})
            continue
        if len(f) != 1:
            raise ValueError(f"{t.tag}: {len(f)} matching fixation excerpts")
        out.append({"cls": t.cls, "task": t, "fix": f[0]})
    return out


def gait_subject(person, B=B_PERM, seed=SEED):
    """(a)-(c) for one gait person (pure-MI session(s)): see the module docstring."""
    fs = person.fs
    if abs(fs - FS) > 1e-9:
        raise ValueError(f"expected {FS} Hz, got {fs}")
    filt = [bandpass(r.astype(np.float64), fs) for r in person.runs]
    tt = trial_table(person.sources)
    a0, a1 = (int(round(t * fs)) for t in TASK_WIN_S)
    f0, f1 = (int(round(t * fs)) for t in FIX_WIN_S)
    X = np.stack([filt[t["task"].run][:, t["task"].start + a0: t["task"].start + a1] for t in tt])
    has_fix = np.array([t["fix"] is not None for t in tt])
    F = np.stack([filt[t["fix"].run][:, t["fix"].start + f0: t["fix"].start + f1] for t in tt if t["fix"] is not None])
    y = np.array([1 if t["cls"] == "mi" else 0 for t in tt])
    run = np.array([t["task"].run for t in tt])
    rows = [person.names.index(c) for c in ERD_CH]
    acc, per_run = loro_accuracy(X, y, run)
    regions = {"sensorimotor": r"^(FC|CP|C)(z|\d+)$", "frontal": r"^(Fp|F|AF)(z|\d+)$"}
    import re
    where = {k: [i for i, c in enumerate(person.names) if re.match(v, c)] for k, v in regions.items()}
    where["posterior_temporal"] = [i for i in range(len(person.names)) if i not in where["sensorimotor"] + where["frontal"]]
    by_region = {k: loro_accuracy(X[:, ix], y, run)[0] for k, ix in where.items()}
    by_region["n_channels"] = {k: len(ix) for k, ix in where.items()}
    lp, lf = log_power(X, 0, X.shape[-1])[:, rows], log_power(F, 0, F.shape[-1])[:, rows]
    rng = np.random.default_rng(seed)
    comp = lp.mean(1)
    diff, p_diff = perm_diff(comp, y, run, B, rng)
    ch = {}
    for j, c in enumerate(ERD_CH):
        d_c, p_c = perm_diff(lp[:, j], y, run, B, rng)
        ch[c] = {"diff_log": d_c, "p": p_c}
    d = comp[has_fix] - lf.mean(1)
    yf, runf = y[has_fix], run[has_fix]
    mi, idle = d[yf == 1], d[yf == 0]
    p_mi, p_idle = signflip_p(mi, B, rng), signflip_p(idle, B, rng)
    d_di, p_di = perm_diff(d, yf, runf, B, rng)
    like = bool(p_idle < ALPHA and mi.mean() < 0 and idle.mean() <= 0.5 * mi.mean())
    pct = lambda v: float(100 * (np.exp(v) - 1))
    return {"n_mi": int((y == 1).sum()), "n_idle": int((y == 0).sum()), "n_runs": int(len(np.unique(run))), "n_without_fixation": int((~has_fix).sum()),
            "acc": acc, "acc_per_held_out_run": per_run, "acc_by_region": by_region, "n_channels_csp": int(X.shape[1]),
            "erd_diff_log": diff, "erd_diff_pct": pct(diff), "erd_p": p_diff, "channel_erd": ch,
            "mi_vs_fixation_log": float(mi.mean()), "mi_vs_fixation_pct": pct(mi.mean()), "mi_vs_fixation_p": p_mi,
            "idle_vs_fixation_log": float(idle.mean()), "idle_vs_fixation_pct": pct(idle.mean()),
            "idle_vs_fixation_p": p_idle, "mi_minus_idle_vs_fixation_log": d_di, "mi_minus_idle_vs_fixation_p": p_di,
            "idle_like_imagery": like, "B": B}


def run_gait(n_subjects=GAIT_SUBJECTS, session="pre", fetch=True):
    import stroke_gait_io as sgi
    subs = list(range(1, n_subjects + 1))
    rows = {}
    for s in subs:
        if fetch:
            sgi.fetch(s, sessions=(session,))
        r = gait_subject(sgi.person(s, [session]))
        rows[f"sub-{s:02d}"] = r
        print(f"gait sub-{s:02d}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                      if k in ("acc", "erd_diff_log", "erd_p", "mi_vs_fixation_p", "idle_vs_fixation_p",
                                               "idle_like_imagery")}, flush=True)
    d = {"session": session, "subjects": subs, "rule": __doc__.split("Acute")[0], "chance_bound_80_trials": chance_bound(80),
         "nogo": gait_nogo(list(rows.values())), "subjects_detail": rows}
    write_section("gait", d)
    return d


def dataset_facts():
    """What was read off the hosting APIs, the dataset papers and the files themselves (sources named)."""
    return {
        "gait": {"paper": "Liu et al. 2025, Sci Data 12:314, doi 10.1038/s41597-025-04618-4",
                 "hosts": "figshare 10.6084/m9.figshare.27130299 (rar archives); Zenodo 10.5281/zenodo.18987384 (BIDS re-host, "
                          "one zip per subject, md5 from the API, used here)", "licence": "CC BY 4.0",
                 "licence_source": "figshare API license = CC BY (52) url creativecommons.org/licenses/by/4.0; Zenodo API "
                                   "license.id = cc-by-4.0",
                 "event_codes": {"1": "MI baseline (fixation) begin", "2": "MI text cue begin", "3": "MI onset (imagery begins)",
                                 "4-6": "MI gait phases 2-4 (timestamps at +1.0 / +1.6 / +3.4 s into the task, present in every "
                                        "session)", "7": "idle baseline begin", "8": "idle text cue begin", "9": "idle onset",
                                 "10-12": "idle phases 2-4 (same offsets)"},
                 "event_codes_source": "figshare events.json + README; verified on sub-01..03 pre (vmrk)",
                 "timing_s": {"baseline_to_cue": 3.0, "cue_to_onset": 1.0, "task": 5.0, "checked": "markers 7/8/9 at 0.656 / "
                              "3.652 / 4.652 s on sub-01 run 1; sources() enforces cue = onset - 1 s and baseline = onset - 4 s "
                              "within 0.15 s"},
                 "sampling_hz": 1000, "channels": "60 EEG (10-10 names incl. T7 / T8 / P7 / P8, no CPz) + ECG + 2 HEOG + 2 VEOG "
                                                   "+ trigger; reference common average, ground AFz (Zenodo description)",
                 "sessions": "pre / post / follow = pure MI; mises / miies = MI + electrical stimulation (never read)",
                 "anomalies": ["CPz (in 118 MOTOR_CHANNELS) is not in the cap: 29 of 30 channels kept",
                               "sub-03 archive has only pre, mises, miies (no post / follow); sub-03 pre run 3 starts mid-trial "
                               "(no baseline marker for its first trial: that trial has no fixation excerpt)",
                               "sub-02 pre has 5 runs (50 + 50 trials), sub-01 and sub-03 pre have 4 runs (40 + 40)",
                               "raw DC offsets up to +-0.4 V: per-run per-channel median removed before resampling"]},
        "acute": {"paper": "Liu H et al. 2023 'An EEG motor imagery dataset for brain computer interface in acute stroke patients', Sci Data, doi 10.1038/s41597-023-02787-8",
                  "host": "figshare 10.6084/m9.figshare.21679035", "licence": "CC BY 4.0",
                  "licence_source": "figshare API license = CC BY 4.0", "patients": 50,
                  "timing_s": {"instruction": 2, "motor_imagery": 4, "break": 2, "per_epoch_in_file": 8},
                  "sampling_hz": 500, "reference_channel": "CPz", "channels": "29 EEG (10-10) + 2 EOG + marker row; reference CPz (flat row, dropped), ground FPz",
                  "markers": "last EDF row: 1 instruction, 2 MI beginning, 3 break beginning; 18 of 50 files carry stray edges, "
                             "MI onset = the code-2 edge followed by a code-3 edge 4 s later",
                  "labels": "events.tsv trial_type 1 left / 2 right, strictly alternating; equal to the per-subject labels of all "
                            "50 sourcedata files",
                  "hand_specific_video": True,
                  "video_quote_paper": "In the MI stage, participants imagined performing this action, a video of gripping motion "
                                       "is played on the computer, which leads the patient imagine grabbing the ball. This video "
                                       "stays playing for 4 s. (Methods, Experimental paradigm)",
                  "video_quote_figshare": "the video of ipsilateral hand movement is displayed on the computer screen and lasts "
                                          "for 4s (figshare description)",
                  "video_files": ["stimuli/left-hand-motor-imagery/stim2.mp4 (1,551,854 B): Left-hand motor imagery stimuli "
                                  "material 4s", "stimuli/right-hand-motor-imagery/stim2.mp4 (1,524,185 B): Right-hand motor "
                                  "imagery stimuli material 4s"],
                  "video_note": "two different files, one per hand (the dataset's stimuli README); a still frame of each shows a gripping hand",
                  "anomalies": ["recordings of sub-04 / sub-05, sub-11 / sub-15 and sub-47 / sub-49 are the same signal (Pearson r "
                                ">= 0.9999 on the epochs); the later of each pair is dropped from the primary analysis",
                                "the instruction phase varies between trials, so the MI onset is not on a fixed grid"]},
    }



def acute_patient(X, labels, names, reps=10, folds=10):
    """One patient: 10 x 10-fold CSP + LDA accuracy (left 0 / right 1) on the sensorimotor and the occipital channels."""
    import stroke_acute_io as sa
    y = (np.asarray(labels) == "right").astype(int)
    sm, oc = sa.group_idx(names, "sensorimotor"), sa.group_idx(names, "occipital")
    return {"sensorimotor_acc": cv_accuracy(X[:, sm], y, folds, reps), "occipital_acc": cv_accuracy(X[:, oc], y, folds, reps),
            "n_sensorimotor": len(sm), "n_occipital": len(oc), "sensorimotor_channels": [names[i] for i in sm],
            "occipital_channels": [names[i] for i in oc]}


def duplicates(arrays, r_min=0.9999):
    """{later subject: earlier subject} for recordings that are the same signal: Pearson r of the flattened epochs >= r_min
    (a re-scaled or re-rounded copy still correlates at ~1; two different people at ~0)."""
    keys = sorted(arrays)
    V = np.stack([np.asarray(arrays[k], float).ravel() for k in keys])
    R = np.corrcoef(V)
    out = {}
    for j in range(len(keys)):
        for i in range(j):
            if keys[i] not in out and R[i, j] >= r_min:
                out[keys[j]] = keys[i]
                break
    return out


def _summ(sm, oc, bound):
    return {"n_patients": len(sm), "sensorimotor_median": float(np.median(sm)), "occipital_median": float(np.median(oc)),
            "sensorimotor_iqr": [float(np.percentile(sm, 25)), float(np.percentile(sm, 75))],
            "occipital_iqr": [float(np.percentile(oc, 25)), float(np.percentile(oc, 75))],
            "sensorimotor_mean": float(np.mean(sm)), "occipital_mean": float(np.mean(oc)),
            "n_above_bound_sensorimotor": int((sm > bound).sum()), "n_above_bound_occipital": int((oc > bound).sum()),
            "n_occipital_ge_sensorimotor": int((oc >= sm).sum()), "nogo": acute_nogo(np.median(sm), np.median(oc))}


def run_acute(subjects=range(1, 51)):
    """Primary = distinct recordings (a later subject whose signal duplicates an earlier one is dropped); the all-patients
    numbers are reported beside it."""
    import stroke_acute_io as sa
    from scipy.stats import wilcoxon
    data = {}
    for s in subjects:
        X, y, ch = sa.trials(s, band=BAND)
        data[s] = (X, y, ch)
    dup = duplicates({s: v[0] for s, v in data.items()})
    rows = {}
    for s, (X, y, ch) in data.items():
        rows[s] = dict(acute_patient(X, y, ch), n_trials=len(y), n_left=y.count("left"), duplicate_of=dup.get(s))
        print(f"acute sub-{s:02d}", {k: round(v, 3) for k, v in rows[s].items() if k.endswith("_acc")}, flush=True)
    keep = [s for s in rows if s not in dup]
    arr = lambda ss, k: np.array([rows[s][k] for s in ss])
    n_tr = int(np.median([r["n_trials"] for r in rows.values()]))
    bound = chance_bound(n_tr)
    first = rows[keep[0]]
    d = {"rule": "CSP (2 filters per end; 1 per end for the 3 occipital channels) + LDA on the 8-30 Hz 4-s MI epoch, "
                 "10 x 10-fold stratified CV, left vs right, each patient's own trials",
         "n_trials_per_patient": n_tr,
         "chance_bound_formula": "Clopper-Pearson two-sided 95% upper limit at k = n/2 correct of n",
         "chance_bound_60_trials": chance_bound(60), "chance_bound_40_trials": bound,
         "duplicate_recordings": {f"sub-{k:02d}": f"sub-{v:02d}" for k, v in dup.items()},
         "primary_distinct_recordings": _summ(arr(keep, "sensorimotor_acc"), arr(keep, "occipital_acc"), bound),
         "all_patients": _summ(arr(list(rows), "sensorimotor_acc"), arr(list(rows), "occipital_acc"), bound),
         "wilcoxon_sensorimotor_vs_occipital_p_primary": float(wilcoxon(arr(keep, "sensorimotor_acc"),
                                                                        arr(keep, "occipital_acc")).pvalue),
         "channels": {"sensorimotor": first["sensorimotor_channels"], "occipital": first["occipital_channels"]},
         "nogo": None, "patients": {f"sub-{s:02d}": r for s, r in rows.items()}}
    d["nogo"] = d["primary_distinct_recordings"]["nogo"]
    write_section("acute", d)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", action="store_true")
    ap.add_argument("--acute", action="store_true")
    a = ap.parse_args()
    if a.gait or a.acute:
        write_section("dataset_facts", dataset_facts())
    if a.gait:
        r = run_gait()
        print(json.dumps({k: v for k, v in r.items() if k != "subjects_detail"}, indent=1))
    if a.acute:
        r = run_acute()
        print(json.dumps({k: v for k, v in r.items() if k != "patients"}, indent=1))
