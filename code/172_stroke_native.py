"""Real instructed motor imagery against a matched sham in the same stroke patients, on the design as recorded (no
splicing, no reuse): the Cornell reading (130's c3, and 171's command-level jackknife) and du Bois (the literal
published specification, blind_impl/dubois_blind.py, and the study's reading B, 130.dubois) on the Wang / Liu
sixth-finger stroke recordings (wangliu_io; 24 patients 3-14 months after stroke, sessions pre / post / follow-up, 4
runs each; every trial = 7 s rest with a fixation cross, 1 s prepare, 6 s imagery of the affected hand or of a
robotic sixth finger, the imagery type randomized).

Design, fixed here before any analysis recording is read (development subject 1 is plumbing only):
- Unit = session (patient x time point); a patient's sessions are clustered in every CI. Runs as recorded.
- Cornell. Per trial two 3-s epochs from the imagery period [cue + 0.5, cue + 6.5) s (task) and two from the rest
  period [rest + 0.5, rest + 6.5) s (rest), through pipe_cornell.features with its timing globals set for the call
  (CMD_S 0, RESP (0.5, 6.5), N_EP 2; wrappers.module_globals) on each run separately; the runs' features are merged
  and the c3 decision applied (171.decide: keep40 pruning per run and condition, OM1 on the per-run tests, OM2 with
  FDR and the > 2 Hz contiguity on all runs). Readings: sample-level jackknife (= the c3 reading) and the command-level
  jackknife (normal and t reference).
  Two nulls, SHAM_DRAWS draws each per session (task structure preserved): sham (rest only) = each rest period's two 3-s epochs labelled pseudo-task and
  pseudo-rest in a random order per trial (side by side in time, as rest and imagery are in the trial; no imagery EEG;
  half the epochs per condition; half of each run's rest periods in each order); mismatch = the recorded trials with
  the task / rest labels swapped in exactly half of each run's trials (timing and pairing kept, the imagery stays in
  the EEG but cancels; it does NOT keep the recording's drift cancellation, because rest always precedes imagery in the
  recording, so it is a conservative, drift-sensitive comparator only). A free split of whole rest
  periods across the run is NOT used: it breaks the within-trial pairing that cancels slow drift, and on null
  synthetic sessions at strong drift it was called in ~48% against 0% for the recorded design.
  Trials without room for the windows (rest marker < 5.5 s into the run, cue > run end - 12.5 s) are dropped.
- du Bois. Imagery of the affected hand (class A) vs the sixth finger (class B), cue onsets as recorded, the session's
  runs demeaned and joined by a 0.25-s equal-power crossfade (118.join; the first cue of a run lies >= 8 s after its
  start). Readings: the literal specification (blind implementation, the study's 17-site full set and the 9-site motor
  set derived inside it, random stratified folds) and reading B with random folds (the imagery type was randomized,
  as in du Bois's Training runs). Sham: the trial labels redrawn (balanced, uniform order; exchangeable because the
  type was randomized), SHAM_DRAWS draws per session; the same EEG.
Readings fixed here (descriptive; reported whatever they show), per analysis reading and null (sham, mismatch; du Bois
has the relabelled sham only): detection = share of sessions called on the true labels; null rate = mean over
sessions of the share of null draws called (patient-cluster bootstrap,
B = 2000, relabelled); 'separates real from the null in patients' if the lower 95% bound of detection minus the null
rate > 0; for the Cornell readings 'calibrated' if the upper bound of the rest-only sham rate <= 10% (the sham
only). Patient level: any call over the
patient's sessions (true; and per sham draw). Descriptive: Cornell true calls whose FDR-surviving bins lie only at
posterior sites (O1 O2 P7 P8: the rest and imagery screens differ).

python 172_stroke_native.py --dev | --run [--workers W] [--limit N] | --merge -> output/tables/stroke_native.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
for _p in (CODE, os.path.join(ROOT, "blind_impl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import motor_common as mc
from bench import pipe_cornell as pc
from bench import pipe_dubois as pdb
from bench import wrappers as wr

tj = importlib.import_module("171_cornell_trialjack")
CORNELL_GLOBALS = {"CMD_S": 0.0, "RESP": (0.5, 6.5), "N_EP": 2}
CORNELL_READINGS = {"cornell_c3": ("sample", "normal", "full"), "cornell_cmd": ("command", "normal", "full"),
                    "cornell_cmd_t": ("command", "t", "full"), "cornell_c3_dec": ("sample", "normal", "dec"),
                    "cornell_cmd_t_dec": ("command", "t", "dec")}
SHAM_DRAWS = 20
XFADE_S = 0.25
CLASS_OF = {"affected_hand": pdb.CLASS_A, "sixth_finger": pdb.CLASS_B, 2: pdb.CLASS_A, 3: pdb.CLASS_B}
MARGIN_BEFORE_S, MARGIN_AFTER_S = 5.5, 12.5
ROWS = os.path.join(ROOT, "output", "intermediate", "stroke_native_rows.jsonl")
ROWS_DEV = os.path.join(ROOT, "output", "intermediate", "stroke_native_rows_dev.jsonl")
OUT = os.path.join(ROOT, "output", "tables", "stroke_native.json")


def usable(run, tr):
    """The run's trials with room for every analysis window: the rest marker >= MARGIN_BEFORE_S s after the run's start
    (Cornell's extent reads 5 s before its first onset) and the imagery cue <= the run's end - MARGIN_AFTER_S s (Cornell
    reads 11.5 s after the last cue, reading B 12 s); trials outside are dropped (counted in the session row)."""
    dur = np.asarray(run.x).shape[1] / float(run.fs)
    m = (tr["rest_s"].to_numpy() >= MARGIN_BEFORE_S) & (tr["mi_s"].to_numpy() <= dur - MARGIN_AFTER_S)
    return tr[m].reset_index(drop=True)


def _rec(run):
    return SimpleNamespace(x=np.asarray(run.x), fs=float(run.fs), names=list(run.names),
                           uid=f"S{run.subject}_{run.timepoint}_{run.run}")


def cornell_overlay(tr, run_index):
    """Rest and imagery onsets of one run's trials, interleaved (rest first); labels REST / TASK."""
    on, lab, trial = [], [], []
    for k, (r, m) in enumerate(zip(tr["rest_s"].to_numpy(), tr["mi_s"].to_numpy())):
        on += [float(r), float(m)]
        lab += [pc.REST, pc.TASK]
        trial += [k, k]
    n = len(on)
    return mc.Overlay(onsets_s=np.array(on), labels=np.array(lab), kind="cmd",
                      meta={"trial": np.array(trial), "block": np.full(n, run_index), "run": np.full(n, run_index)})


def cornell_features(runs, trials_of):
    """Merged pc.features over the session's runs -> (F, true label per epoch)."""
    parts, labs, off = [], [], 0
    for k, run in enumerate(runs):
        tr = trials_of(run)
        if len(tr) < 4:
            continue
        ov = cornell_overlay(tr, k)
        if off % 2:
            raise AssertionError("command offsets must stay even so cmd // 2 is the trial")
        with wr.module_globals(pc, CORNELL_GLOBALS):
            F = dict(pc.features(_rec(run), ov))
        labs.append(np.asarray(ov.labels)[F["cmd"]])
        F["cmd"] = F["cmd"] + off
        off += len(ov.onsets_s)
        parts.append(F)
    if not parts:
        raise mc.NoUsableTrials("no run with >= 4 trials")
    M = {k: np.concatenate([p[k] for p in parts]) for k in ("S", "cmd", "run", "score", "dead", "hard")}
    M["f"] = parts[0]["f"]
    return M, np.concatenate(labs)


def cornell_sham(F, lab, rng):
    """Rest-only sham, drift-matched within the trial: only the rest epochs; each rest period's two 3-s epochs
    ([rest + 0.5, + 3.5) and [rest + 3.5, + 6.5)) are labelled pseudo-task and pseudo-rest in a random order per trial,
    so the two conditions sit side by side in time as rest and imagery do in the recorded trial, and no imagery EEG
    enters; exactly half of each run's rest periods put their first epoch in pseudo-task (random which), so a fixed
    early-versus-late difference within the rest period cancels. Each epoch becomes its own pseudo-command (cmd * 2 + position) for the command-level jackknife. Half the
    epochs per condition of the true analysis. -> (F subset, labels)."""
    g = mc.gen(rng)
    m = lab == pc.REST
    Fs = {k: (v[m] if k != "f" else v) for k, v in F.items()}
    cmd = Fs["cmd"]
    first = np.r_[True, cmd[1:] != cmd[:-1]]
    pos = np.arange(len(cmd)) - np.maximum.accumulate(np.where(first, np.arange(len(cmd)), 0))
    if pos.max() > 1:
        raise ValueError("cornell_sham expects two epochs per rest command")
    flip = {}
    for r in np.unique(Fs["run"]):
        cs = np.unique(cmd[Fs["run"] == r])
        sw = set(g.permutation(cs)[:len(cs) // 2].tolist())
        flip.update({c: c in sw for c in cs})
    out = np.array([pc.TASK if (p == 0) != flip[c] else pc.REST for c, p in zip(cmd, pos)])
    Fs["cmd"] = cmd * 2 + pos
    return Fs, out


def cornell_mismatch(F, lab, rng):
    """Mismatched task: the recorded trials and epochs unchanged, the task / rest labels swapped in exactly half of each
    run's trials (random which; the trial's rest command is cmd 2k, its imagery cmd 2k + 1 within the run's overlay;
    merged runs keep that pairing), so the real imagery cancels in expectation and cannot leak into the labels. The
    recorded timing and pairing are kept, but NOT its drift cancellation: in the recording rest always precedes imagery,
    while the swapped half reverses that order, so slow drift adds instead of cancelling. The mismatch is therefore a
    conservative, drift-sensitive comparator, never a calibration bar. -> labels."""
    g = mc.gen(rng)
    trial = F["cmd"] // 2
    out = lab.copy()
    for r in np.unique(F["run"]):
        ts = np.unique(trial[F["run"] == r])
        m = np.isin(trial, g.permutation(ts)[:len(ts) // 2])
        out[m] = np.where(lab[m] == pc.TASK, pc.REST, pc.TASK)
    return out


def cornell_rows(runs, trials_of, base, rng):
    rows = []
    F, lab = cornell_features(runs, trials_of)
    names = list(pc.NATIVE_CH)
    for name, (unit, ref, grid) in CORNELL_READINGS.items():
        try:
            r = tj.decide(F, lab, unit, ref, names=names, grid=grid)
            rows.append({**base, "pipe": "cornell", "reading": name, "version": "true", "draw": 0, "call": r["call"],
                         "stat": r["stat"], "om1": r["extra"]["om1"], "om2_contig": r["extra"]["om2_contig"],
                         "n_sig": r["extra"]["n_sig_allruns"], "n_sig_posterior": r["extra"]["n_sig_posterior"],
                         "vif": r["extra"]["vif"]})
        except mc.NoUsableTrials as e:
            rows.append({**base, "pipe": "cornell", "reading": name, "version": "true", "draw": 0, "call": None,
                         "status": f"no_usable: {e}"[:160]})
    g = mc.gen(rng)
    for d in range(1, SHAM_DRAWS + 1):
        Fs, ls = cornell_sham(F, lab, g)
        lm = cornell_mismatch(F, lab, g)
        for version, FF, ll in (("sham", Fs, ls), ("mismatch", F, lm)):
            for name, (unit, ref, grid) in CORNELL_READINGS.items():
                try:
                    r = tj.decide(FF, ll, unit, ref, names=names, grid=grid)
                    rows.append({**base, "pipe": "cornell", "reading": name, "version": version, "draw": d,
                                 "call": r["call"], "stat": r["stat"]})
                except mc.NoUsableTrials as e:
                    rows.append({**base, "pipe": "cornell", "reading": name, "version": version, "draw": d,
                                 "call": None, "status": f"no_usable: {e}"[:160]})
    return rows


def session_signal(runs, trials_of):
    """The session's runs demeaned and joined (equal-power crossfade) -> (x, fs, names, onsets, labels, blocks)."""
    pc118 = importlib.import_module("118_positive_control")
    fs = float(runs[0].fs)
    c = int(round(XFADE_S * fs))
    pieces, on, lab, blk, pos = [], [], [], [], 0
    for k, run in enumerate(runs):
        tr = trials_of(run)
        x = np.asarray(run.x, float)
        x = x - x.mean(1, keepdims=True)
        pieces.append(x)
        on += list(pos / fs + tr["mi_s"].to_numpy())
        lab += [CLASS_OF[t] for t in tr["mi_type"]]
        blk += [k] * len(tr)
        pos += x.shape[1] - c
    return (pc118.join(pieces, c).astype(np.float32), fs, list(runs[0].names), np.array(on), np.array(lab),
            np.array(blk))


def dubois_overlay(on, lab, blk):
    return mc.Overlay(onsets_s=on, labels=lab, kind="cmd",
                      meta={"trial": np.arange(len(on)), "block": blk, "run": np.zeros(len(on), int),
                            "resp_hand": {pdb.CLASS_A: "left", pdb.CLASS_B: "right"}, "resp_delay_s": 0.0,
                            "resp_dur_s": 6.0})


def dubois_rows(runs, trials_of, base, rng):
    import dubois_blind as db
    rc = importlib.import_module("170_dubois_reconcile")
    fr = importlib.import_module("130_fidelity_readings")
    x, fs, names, on, lab, blk = session_signal(runs, trials_of)
    nm = mc.canon(names)
    full = list(pdb.CHANSETS["full"])
    rec = SimpleNamespace(x=x[[nm.index(c) for c in full]], fs=fs, names=full, uid=base["session_id"])
    g = mc.gen(rng)
    rows = []
    for d in range(0, SHAM_DRAWS + 1):
        y = lab if d == 0 else mc.balanced_labels((pdb.CLASS_A, pdb.CLASS_B),
                                                  (int(np.sum(lab == pdb.CLASS_A)), int(np.sum(lab == pdb.CLASS_B))), g)
        ov = dubois_overlay(on, y, blk)
        version = "true" if d == 0 else "sham"
        try:
            xc, a = rc.crop(rec, ov)
            out = db.assess(xc, fs, full, [e[:2] for e in rc.events(ov, a)], seed=0)
            rows.append({**base, "pipe": "dubois", "reading": "dubois_literal", "version": version, "draw": d,
                         "call": bool(out["call"]), "peak_da": float(out["peak_da"]), "p_t": float(out["p_t"])})
        except Exception as e:
            rows.append({**base, "pipe": "dubois", "reading": "dubois_literal", "version": version, "draw": d,
                         "call": None, "status": f"error: {type(e).__name__}: {e}"[:160]})
        try:
            r = fr.dubois(rec, ov, "c3_b_assessment_random_phase1", rng=mc.seed_of(g))
            rows.append({**base, "pipe": "dubois", "reading": "dubois_reading_b", "version": version, "draw": d,
                         "call": bool(r["call"]), "peak_da": float(r["extra"]["peak_da"]), "p_t": float(r["p"]),
                         "t": float(r["stat"])})
        except Exception as e:
            rows.append({**base, "pipe": "dubois", "reading": "dubois_reading_b", "version": version, "draw": d,
                         "call": None, "status": f"error: {type(e).__name__}: {e}"[:160]})
    mc._CACHE.clear()
    return rows


def session_rows(subject, timepoint, runs, trials_of):
    raw_of = trials_of

    def trials_of(run):
        return usable(run, raw_of(run))
    names = [list(r.names) for r in runs]
    if any(n != names[0] for n in names) or len({float(r.fs) for r in runs}) != 1:
        raise ValueError(f"S{subject} {timepoint}: runs differ in channels or rate")
    base = {"subject": subject, "timepoint": timepoint, "session_id": f"S{subject}_{timepoint}",
            "n_runs": len(runs), "n_trials": int(sum(len(trials_of(r)) for r in runs)),
            "n_trials_dropped_margin": int(sum(len(raw_of(r)) - len(trials_of(r)) for r in runs))}
    try:
        rows = cornell_rows(runs, trials_of, base, np.random.default_rng([20261029, 172, subject, len(timepoint)]))
    except mc.NoUsableTrials as e:
        rows = [{**base, "pipe": "cornell", "reading": n, "version": "true", "draw": 0, "call": None,
                 "status": f"no_usable: {e}"[:160]} for n in CORNELL_READINGS]
    rows += dubois_rows(runs, trials_of, base, np.random.default_rng([20261029, 172, 2, subject, len(timepoint)]))
    return rows


def _boot_ci(per_patient, B=2000, seed=20261029):
    """Percentile CI of the mean over patients of per-patient values (each a list of session values), relabelled."""
    g = np.random.default_rng(seed)
    pts = list(per_patient.values())
    est = np.mean([v for p in pts for v in p])
    bs = []
    for _ in range(B):
        pick = g.integers(0, len(pts), len(pts))
        vals = [v for i in pick for v in pts[i]]
        bs.append(np.mean(vals))
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


TP_ORDER = {"pre": 0, "post": 1, "follow": 2}


def duplicate_sessions(rows):
    """Sessions excluded for byte-identical recordings (session hash = its runs' sha256 in order), decided from the
    hashes alone: copies within one patient keep the earliest time point; copies across patients are all excluded
    (the recording cannot be attributed). -> {session_id: reason}."""
    h = {}
    for r in rows:
        h.setdefault(r["session_id"], (r["sha256"], r["subject"], r["timepoint"]))
    by = {}
    for sid, (sha, subj, tp) in h.items():
        by.setdefault(sha, []).append((sid, subj, tp))
    out = {}
    for copies in by.values():
        if len(copies) < 2:
            continue
        if len({c[1] for c in copies}) > 1:
            out.update({c[0]: f"identical to {', '.join(x[0] for x in copies if x[0] != c[0])} (other patient)"
                        for c in copies})
        else:
            keep = min(copies, key=lambda c: TP_ORDER.get(c[2], 9))
            out.update({c[0]: f"identical to {keep[0]} (same patient)" for c in copies if c[0] != keep[0]})
    return out


def summarize(rows):
    """Per reading: detection on the true labels and, per null version, the null rate and detection minus it, all as
    means over patients' sessions with patient-cluster bootstrap CIs (module docstring). Byte-identical sessions are
    excluded first (duplicate_sessions)."""
    dup = duplicate_sessions(rows)
    rows = [r for r in rows if r["session_id"] not in dup]
    out = {"excluded_duplicates": dup, "readings": {}}
    for reading in sorted({r["reading"] for r in rows}):
        rr = [r for r in rows if r["reading"] == reading and r.get("call") is not None]
        true = {r["session_id"]: float(r["call"]) for r in rr if r["version"] == "true"}
        if not true:
            continue
        det = {}
        for s_, v in true.items():
            det.setdefault(s_.split("_")[0], []).append(v)
        d_est, d_ci = _boot_ci(det)
        res = {"n_sessions": len(true), "n_patients": len(det), "detection": {"rate": d_est, "ci95": d_ci},
               "patient_any_true": float(np.mean([max(v) for v in det.values()])), "nulls": {}}
        for version in sorted({r["version"] for r in rr} - {"true"}):
            null, diff = {}, {}
            for s_, t in true.items():
                sh = [r["call"] for r in rr if r["session_id"] == s_ and r["version"] == version]
                if not sh:
                    continue
                p = s_.split("_")[0]
                null.setdefault(p, []).append(float(np.mean(sh)))
                diff.setdefault(p, []).append(t - float(np.mean(sh)))
            if not null:
                continue
            n_est, n_ci = _boot_ci(null)
            x_est, x_ci = _boot_ci(diff)
            v = {"rate": {"rate": n_est, "ci95": n_ci}, "difference": {"value": x_est, "ci95": x_ci},
                 "separates": bool(x_ci[0] > 0)}
            if reading.startswith("cornell") and version == "sham":
                v["calibrated_upper_le_10pct"] = bool(n_ci[1] <= 0.10)
            res["nulls"][version] = v
        if reading.startswith("cornell"):
            calls = [r for r in rr if r["version"] == "true" and r["call"] and r.get("n_sig")]
            res["true_calls_posterior_only"] = int(sum(1 for r in calls if r.get("n_sig_posterior", 0) == r["n_sig"]))
        out["readings"][reading] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.merge:
        import glob
        files = sorted(glob.glob(ROWS.replace(".jsonl", "_s*.jsonl"))) + ([ROWS] if os.path.exists(ROWS) else [])
        rows = [json.loads(l) for f in files for l in open(f)]
        res = {"what": __doc__.split("\n")[0], **summarize(rows)}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(json.dumps(res, indent=1, default=float))
        return
    wl = importlib.import_module("wangliu_io")
    subjects = [wl.DEV_SUBJECT] if a.dev else [s for s in wl.SUBJECTS if s != wl.DEV_SUBJECT]
    jobs = [(s, tp) for s in subjects for tp in (("pre",) if a.dev else wl.TIMEPOINTS)][:a.limit]
    jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    path = ROWS_DEV if a.dev else ROWS
    if a.nshards > 1:
        path = path.replace(".jsonl", f"_s{a.shard}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {(r["subject"], r["timepoint"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    for s, tp in jobs:
        if (s, tp) in done or tp not in wl.sessions(s):
            continue
        runs = [wl.load_run(p) for p in wl.fetch(s, tp)]
        for r in runs:
            wl.check(r)
        rows = session_rows(s, tp, runs, wl.trials)
        with open(path, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "sha256": ",".join(x.sha256 for x in runs)}, default=float) + "\n")
        print(s, tp, {k: sum(1 for r in rows if r.get("call") and r["version"] == "true" and r["reading"] == k)
                      for k in sorted({r["reading"] for r in rows})}, flush=True)


if __name__ == "__main__":
    main()
