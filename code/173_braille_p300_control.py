"""Tactile P300 positive control with genuine attention: the vibrotactile P300 analysis (bench/pipe_mbvt.py, the
frozen replication reading vt3_mur20_seg8_median_gt23, imported and never edited) and the independent implementation
(blind_impl/p300_blind.py) on the Braille BCI recordings (braille_io; Grigoryan et al., G-Node, CC0), where people
counted touches on one attended fingertip among eight stimulated ones, against a sham in the same recordings.

Design, fixed here before any analysis recording is read (development subject 6 is plumbing only):
- Groups: within each selection cycle (one attended finger; 10 touches of each of the 8 fingers in random order), the
  k-th touch of every finger forms group k (k = 0..9): 8 stimuli, one per finger, exactly one on the attended finger,
  each epoch at its recorded onset. pos = finger, so the module's 8-way decision is the BCI's own 'which finger'.
- Runs: cycles 0-5 = run A (60 groups, the module's published run length), 6-11 = run B, 12-15 = run C (40 groups,
  secondary). Person-session call = any positive among runs A and B (the replication's per-patient rule, 'any positive
  among the first 2 runs'); person call = any over the two sessions (large and small pins).
- Versions: true (the attended finger is the target) and sham (every cycle's target mapped through one random
  derangement of the 8 fingers per session: same EEG, same stimulation, never the attended finger, and the run's
  targets distinct fingers exactly as the true ones are). A sham whose fingers repeat within a run is not a null: the
  split-half LDA then learns finger-specific evoked responses (model probe 39-56% calls with no
  attention effect). Revisions during the design phase: an other-hand sham (prompted by development subject 6) and
  the per-cycle independent shams before it were both replaced for that reason; no analysis recording was read.
- Readings: mbvt primary (median of the split-half accuracy curve > 23%), mbvt DEFAULT (full test pool > 23%), the blind
  implementation (attended finger = its 'left' target, the other seven = distractors), and the randomization-based
  test: statistic = the module's LDA score margin, null = the run's labelled targets permuted across its cycles,
  B = 200, decision-exact early stopping (mc.perm_test).
Readings fixed here (descriptive; reported whatever they show):
- 'positive control passes' if the true-version person-session detection of the primary reading has its exact 95%
  upper bound >= the paper's bar for this analysis (118 BARS mbvt, 3/3) AND exceeds the sham rate by exact McNemar
  p < .05 over the same person-sessions;
- the sham rate is the analysis's positive rate under stimulation without attention to the labelled finger
  ('criterion permissiveness' when it exceeds 5%: one-sided exact binomial), read only beside its split by whether the
  sham finger stayed on the attended hand (attention spreading within that hand would make same-hand shams callable).
Caveats stated in advance: fingertip Braille pins, not wrist and back vibrators; healthy blind and sighted adults; the
attended-finger order is the same fixed sequence in every record (target identity follows time in session).

python 173_braille_p300_control.py --dev | --run [--workers W] | --merge -> output/tables/braille_p300_control.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta, binom

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
for _p in (CODE, os.path.join(ROOT, "blind_impl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import erp_common as ec
from bench import motor_common as mc
from bench import pipe_mbvt as mv

PRIMARY = "vt3_mur20_seg8_median_gt23"
READINGS = {"mbvt_primary": PRIMARY, "mbvt_default": mv.DEFAULT}
RUNS = {"A": tuple(range(0, 6)), "B": tuple(range(6, 12)), "C": tuple(range(12, 16))}
PRIMARY_RUNS = ("A", "B")
VERSIONS = ("true", "sham")
N_FINGERS, PER_FINGER = 8, 10
B_RAND = 200
ROWS = os.path.join(ROOT, "output", "intermediate", "braille_p300_rows.jsonl")
ROWS_DEV = os.path.join(ROOT, "output", "intermediate", "braille_p300_rows_dev.jsonl")
OUT = os.path.join(ROOT, "output", "tables", "braille_p300_control.json")


@dataclass
class Rec:
    source: str
    uid: str
    patient: str
    x: np.ndarray
    fs: float
    names: list
    meta: dict = field(default_factory=dict)


def derangement(n, rng):
    """A uniformly random permutation of range(n) with no fixed point (rejection sampling)."""
    g = mc.gen(rng)
    while True:
        p = g.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def sham_targets(true_by_cycle, rng):
    """{cycle: pi(true target)}, pi one random derangement of the 8 fingers per session: no cycle keeps its attended
    finger, and the run's targets stay distinct fingers in the same pattern as the true ones (a sham whose fingers
    repeat within a run lets the split-half LDA learn finger-specific evoked responses: an artefact of the sham)."""
    pi = derangement(N_FINGERS, rng)
    return {c: int(pi[t]) for c, t in sorted(true_by_cycle.items())}


def overlay(events, cycles, target_by_cycle):
    """mbvt Overlay of the stimuli of `cycles`: group = (cycle, k-th touch of each finger), pos = finger, label TARGET
    on the cycle's target finger, NONTARGET elsewhere; onsets as recorded (s from the start of the recording)."""
    ev = events[events["cycle"].isin(cycles)].sort_values("onset_s")
    on, lab, grp, pos = [], [], [], []
    for gi, (c, sub) in enumerate((c, s) for c in cycles for s in [ev[ev["cycle"] == c]]):
        k = sub.groupby("finger").cumcount().to_numpy()
        if not (np.bincount(sub["finger"], minlength=N_FINGERS) == PER_FINGER).all():
            raise ValueError(f"cycle {c}: not {PER_FINGER} touches of every finger")
        for t, f, kk in zip(sub["onset_s"].to_numpy(), sub["finger"].to_numpy(), k):
            on.append(float(t))
            pos.append(int(f))
            grp.append(gi * PER_FINGER + int(kk))
            lab.append(mv.TARGET if int(f) == int(target_by_cycle[c]) else mv.NONTARGET)
    grp, pos = np.array(grp), np.array(pos)
    n_groups = int(grp.max()) + 1
    return ec.Overlay(onsets_s=np.array(on), labels=np.array(lab, int), kind="stim",
                      meta={"group": grp, "pos": pos, "site": pos.copy(), "block": grp // PER_FINGER,
                            "trial": grp, "run": np.zeros(len(on), int), "paradigm": "vt3",
                            "target_wrist": np.full(n_groups, mv.LEFT)})


def blind_events(ov, start_s=0.0):
    """p300_blind events: the labelled target finger -> 'left' (its target), every other finger -> 'distractor'."""
    return [(float(t) - start_s, "left" if int(l) == mv.TARGET else "distractor") for t, l in zip(ov.onsets_s, ov.labels)]


def rand_test(rec, ov, cycles, rng, B=B_RAND):
    """Randomization test of the module's score margin: the run's labelled targets permuted across its cycles (the
    set of fingers and one target per cycle kept; under no attention effect the assignment of targets to cycles is
    exchangeable, finger-specific evoked responses included)."""
    g = mc.gen(rng)
    grp, lab, pos = np.asarray(ov.meta["group"]), np.asarray(ov.labels), np.asarray(ov.meta["pos"])
    tpos_by_cycle = [pos[(grp // PER_FINGER == i) & (lab == mv.TARGET)][0] for i in range(len(cycles))]
    c = mc.cfg(mv.VARIANTS, PRIMARY)
    F = mv._feat(rec, ov, c["feat"])
    mv._require(F)
    tpos = mv._tpos(ov, F["keep"])
    cyc = (np.arange(len(F["keep"])) // PER_FINGER)[F["keep"]]
    obs = mv.score_margin(F["E"], tpos, g, c["reps"])

    run_t = np.array([int(t) for t in tpos_by_cycle])

    def draw():
        tgt = g.permutation(run_t)
        return mv.score_margin(F["E"], tgt[cyc], g, c["reps"])
    t = mc.perm_test(obs, draw, B, 0.05)
    return {"stat": float(obs), "p": t["p"], "call": bool(t["p"] < 0.05), "n_perm_run": t["n_run"]}


def run_session(run, seed=0):
    """Every row of one person-session (braille_io.Run)."""
    import p300_blind
    rec = Rec("braille", f"S{run.subject}_{run.session}", f"S{run.subject}", run.x, run.fs, list(run.names))
    true_t = run.events.groupby("cycle")["target"].first().astype(int).to_dict()
    sham_t = sham_targets(true_t, np.random.default_rng([20261029, 173, run.subject, run.session == "large"]))
    rows = []
    for version, tgt in (("true", true_t), ("sham", sham_t)):
        for rname, cycles in RUNS.items():
            base = {"subject": run.subject, "session": run.session, "run": rname, "version": version}
            try:
                ov = overlay(run.events, cycles, tgt)
            except (ValueError, KeyError) as e:
                for reading in list(READINGS) + ["blind", "randomization"]:
                    rows.append({**base, "reading": reading, "call": None, "status": f"no_overlay: {e}"[:160]})
                continue
            for reading, variant in READINGS.items():
                try:
                    r = mv.run(rec, ov, variant, rng=np.random.default_rng([20261029, 173, run.subject, len(rows)]))
                    rows.append({**base, "reading": reading, "call": r["call"], "acc": r["extra"]["acc"],
                                 "margin": r["extra"]["size_stat_cont"], "n_groups": r["extra"]["n_groups"]})
                except mc.NoUsableTrials as e:
                    rows.append({**base, "reading": reading, "call": None, "status": f"no_usable: {e}"[:160]})
            t0, t1 = mv.extent(ov)
            i0 = max(0, int(np.floor(t0 * rec.fs)))
            out = p300_blind.assess(rec.x[:, i0:int(np.ceil(t1 * rec.fs))], rec.fs, rec.names,
                                    blind_events(ov, i0 / rec.fs), seed=seed, target="left")
            rows.append({**base, "reading": "blind", "call": bool(out["call"]), "acc": float(out["accuracy"])})
            try:
                rr = rand_test(rec, ov, cycles, np.random.default_rng([20261029, 173, 7, run.subject, len(rows)]))
                rows.append({**base, "reading": "randomization", **rr})
            except mc.NoUsableTrials as e:
                rows.append({**base, "reading": "randomization", "call": None, "status": f"no_usable: {e}"[:160]})
        mc._CACHE.clear()
    return rows


def _cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def session_calls(rows, reading, version, runs=PRIMARY_RUNS):
    """{(subject, session): call} = any positive among `runs`; a person-session with any unusable run is left out."""
    by = {}
    for r in rows:
        if r["reading"] == reading and r["version"] == version and r["run"] in runs:
            by.setdefault((r["subject"], r["session"]), []).append(r.get("call"))
    return {k: any(v) for k, v in by.items() if len(v) == len(runs) and all(c is not None for c in v)}


def sham_same_hand(subject, session, events):
    """{run: share of its cycles whose sham finger lies on the attended finger's hand}, recomputed from the seeded
    derangement exactly as run_session draws it (one draw per session, before anything else)."""
    true_t = events.groupby("cycle")["target"].first().astype(int).to_dict()
    sh = sham_targets(true_t, np.random.default_rng([20261029, 173, subject, session == "large"]))
    return {r: float(np.mean([(sh[c] < 4) == (true_t[c] < 4) for c in cyc if c in true_t])) for r, cyc in RUNS.items()}


def hand_split(rows, events_of, reading="mbvt_primary"):
    """Sham session calls (runs A, B) split by whether most of the session's sham cycles stayed on the attended hand
    (attention spreading within the attended hand would make same-hand shams callable)."""
    calls = session_calls(rows, reading, "sham")
    out = {"same_hand_majority": [0, 0], "other_hand_majority": [0, 0]}
    for (s, ses), c in calls.items():
        share = sham_same_hand(s, ses, events_of(s, ses))
        key = "same_hand_majority" if np.mean([share[r] for r in PRIMARY_RUNS]) > 0.5 else "other_hand_majority"
        out[key][0] += int(c)
        out[key][1] += 1
    return {k: {"k": v[0], "n": v[1]} for k, v in out.items()}


def summarize(rows, bar=(3, 3)):
    out = {"readings": {}}
    for reading in list(READINGS) + ["blind", "randomization"]:
        t, s = session_calls(rows, reading, "true"), session_calls(rows, reading, "sham")
        keys = sorted(set(t) & set(s))
        kt, ks, n = sum(t[k] for k in keys), sum(s[k] for k in keys), len(keys)
        b = sum(t[k] and not s[k] for k in keys)
        c = sum(s[k] and not t[k] for k in keys)
        p_mcn = float(min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))) if b + c else 1.0
        persons = sorted({k[0] for k in keys})
        pt = sum(any(t[k] for k in keys if k[0] == p) for p in persons)
        ps = sum(any(s[k] for k in keys if k[0] == p) for p in persons)
        d = {"n_sessions": n, "true": {"k": kt, "ci95": _cp(kt, n) if n else None},
             "sham": {"k": ks, "ci95": _cp(ks, n) if n else None, "p_above_5pct": float(binom.sf(ks - 1, n, 0.05))
                      if n else None},
             "mcnemar": {"b": b, "c": c, "p_exact": p_mcn},
             "persons": {"n": len(persons), "true_any": pt, "sham_any": ps}}
        if reading == "mbvt_primary" and n:
            d["positive_control_passes"] = bool(_cp(kt, n)[1] >= bar[0] / bar[1] and kt > ks and p_mcn < 0.05)
        out["readings"][reading] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    bio = importlib.import_module("braille_io")
    if a.merge:
        import glob
        files = sorted(glob.glob(ROWS.replace(".jsonl", "_s*.jsonl"))) + ([ROWS] if os.path.exists(ROWS) else [])
        rows = [json.loads(l) for f in files for l in open(f)]
        res = {"what": __doc__.split("\n")[0], **summarize(rows)}
        res["sham_hand_split"] = {rd: hand_split(rows, bio.events_only, rd)
                                  for rd in ("mbvt_primary", "blind")}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
        print(json.dumps(res, indent=1, default=float))
        return
    subjects = [bio.DEV_SUBJECT] if a.dev else [s for s in bio.SUBJECTS if s != bio.DEV_SUBJECT]
    jobs = [(s, ses) for s in subjects for ses in bio.SESSIONS][:a.limit]
    jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    path = ROWS_DEV if a.dev else ROWS
    if a.nshards > 1:
        path = path.replace(".jsonl", f"_s{a.shard}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {(r["subject"], r["session"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    for s, ses in jobs:
        if (s, ses) in done:
            continue
        run = bio.load(s, ses)
        bio.check(run)
        rows = run_session(run)
        with open(path, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "sha256": run.sha256}, default=float) + "\n")
        print(s, ses, sum(1 for r in rows if r.get("call")), "calls of", len(rows), flush=True)


if __name__ == "__main__":
    main()
