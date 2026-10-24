"""Checks for a P300 positive control in ALS (BigP3BCI studies B, F, L, N; bigp3_als_io), 48 patients, Train and Test runs
pooled. Per patient:
  signal   Pz, 0.1-30 Hz zero-phase (4th-order Butterworth, forward-backward) on the continuous run;
  per flash mean amplitude 250-600 ms after the flash onset minus the mean over -100-0 ms (baseline), in µV; flashes whose
           epoch leaves the recording are dropped and counted;
  statistic  mean(target flashes) - mean(far non-target flashes) pooled over the patient's runs; far = at least 1 s (4 flashes
           at the 250-ms SOA) from every target of the same sequence;
  test     one-sided (target larger) permutation test: 1,000 shuffles of the target flags WITHIN each sequence (one
           PhaseInSequence == 2 stretch of one run = one character trial), the far set recomputed in every draw,
           P = (1 + #{null >= observed}) / (B + 1).
GO if >= 34 of 48 patients have P < .05 AND the grand-average difference is positive; else NOGO.
python 165_als_p300_check.py [--extract] [--workers 2] -> output/tables/als_p300_check.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import bigp3_als_io as bio

OUT = os.path.join(ROOT, "output", "tables", "als_p300_check.json")
BAND = (0.1, 30.0)
WIN_MS, BASE_MS = (250.0, 600.0), (-100.0, 0.0)
PZ = bio.CHANNELS.index("Pz")
B_PERM, ALPHA, SEED = 1000, 0.05, 20261022
MIN_GAP_S = 1.0
GO_MIN = 34


def bandpass(x, fs, band=BAND):
    x = np.asarray(x, float)
    return sosfiltfilt(butter(4, band, btype="bandpass", fs=fs, output="sos"), x, padlen=min(len(x) - 1, int(5 * fs)))


def epoch_means(x, fs, onsets, win_ms=WIN_MS):
    """Baseline-corrected window mean per flash -> (values, ok); ok False where the epoch leaves the recording."""
    s = lambda ms: int(round(ms * fs / 1000.0))
    w0, w1, b0, b1 = s(win_ms[0]), s(win_ms[1]), s(BASE_MS[0]), s(BASE_MS[1])
    on = np.asarray(onsets, int)
    ok = (on + b0 >= 0) & (on + w1 <= len(x))
    out = np.full(len(on), np.nan)
    for i in np.flatnonzero(ok):
        o = on[i]
        out[i] = x[o + w0:o + w1].mean() - x[o + b0:o + b1].mean()
    return out, ok


def shuffle_flags(flags, seq, rng):
    """Permute the target flags among the flashes of each sequence (the count per sequence is kept)."""
    flags, seq = np.asarray(flags, bool), np.asarray(seq)
    out = flags.copy()
    for s in np.unique(seq):
        i = np.flatnonzero(seq == s)
        out[i] = rng.permutation(flags[i])
    return out


def perm_test_all(a, flags, seq, B=B_PERM, rng=None):
    """Superseded variant, kept for comparison: every non-target flash is used (no distance rule)."""
    rng = rng or np.random.default_rng(SEED)
    a, flags, seq = np.asarray(a, float), np.asarray(flags, bool), np.asarray(seq)
    n_t = int(flags.sum())
    obs = float(a[flags].mean() - a[~flags].mean())
    order0 = np.argsort(seq, kind="stable")
    seq_s, a_s, f_s = seq[order0], a[order0], flags[order0]
    _, start, cnt = np.unique(seq_s, return_index=True, return_counts=True)
    n_tg = np.add.reduceat(f_s.astype(int), start)
    grp = np.repeat(np.arange(len(start)), cnt)
    pos = np.arange(len(a)) - np.repeat(start, cnt)
    tot = a_s.sum()
    null = np.empty(B)
    for b in range(B):
        key = rng.random(len(a))
        o = np.lexsort((key, grp))
        rank = np.empty(len(a), int)
        rank[o] = pos
        sel = rank < n_tg[grp]
        null[b] = a_s[sel].mean() - (tot - a_s[sel].sum()) / (len(a) - n_t)
    return {"n_target": n_t, "n_nontarget": int(len(a) - n_t), "diff_uv": obs,
            "p": float((1 + (null >= obs - 1e-12).sum()) / (B + 1)), "B": B}


def gap_samples(fs, min_gap_s=MIN_GAP_S):
    """The distance rule in whole samples. B and L EDF headers carry 256.00005-256.00013 Hz, so min_gap_s * fs = 256.00006
    would exclude flashes exactly 4 flashes (256 samples) from a target; rounding gives the intended >= 1 s."""
    return int(round(min_gap_s * fs))


def eligible(onsets, flags, seq, fs, min_gap_s=MIN_GAP_S):
    """Flashes usable on the non-target side: non-targets at least min_gap_s from EVERY target flash of the same sequence
    (4 flashes at the 250-ms SOA). Targets are never eligible; a sequence without a target keeps all its flashes."""
    on, flags, seq = np.asarray(onsets), np.asarray(flags, bool), np.asarray(seq)
    el = ~flags
    for s in np.unique(seq):
        i = np.flatnonzero(seq == s)
        t = on[i][flags[i]]
        if len(t):
            near = (np.abs(on[i][:, None] - t[None, :]) < gap_samples(fs, min_gap_s)).any(1)
            el[i[near]] = False
    return el


def target_gaps(onsets, flags, seq, fs):
    """Smallest spacing between two targets of a sequence, in flashes (index steps) and seconds; None if no pair."""
    on, flags, seq = np.asarray(onsets), np.asarray(flags, bool), np.asarray(seq)
    g_f, g_s = [], []
    for s in np.unique(seq):
        i = np.flatnonzero(seq == s)
        pos = np.flatnonzero(flags[i])
        if len(pos) > 1:
            g_f.append(int(np.diff(pos).min()))
            g_s.append(float(np.diff(on[i][pos]).min() / fs))
    return {"min_gap_flashes": min(g_f) if g_f else None, "min_gap_s": min(g_s) if g_s else None}


def perm_test(a, flags, seq, onset, fs, B=B_PERM, rng=None, min_gap_s=MIN_GAP_S):
    """Target mean minus mean of the far non-targets, one-sided (target larger). Null: B shuffles of the target flags within
    each sequence (count per sequence kept); the far-non-target set is recomputed from the shuffled targets in every draw, as
    it was built for the observed data. P = (1 + #{null >= observed}) / (B + 1)."""
    rng = rng or np.random.default_rng(SEED)
    a, flags, seq, onset = np.asarray(a, float), np.asarray(flags, bool), np.asarray(seq), np.asarray(onset)
    el = eligible(onset, flags, seq, fs, min_gap_s)
    obs = float(a[flags].mean() - a[el].mean())
    groups = []
    for s in np.unique(seq):
        i = np.flatnonzero(seq == s)
        groups.append((a[i], int(flags[i].sum()), np.abs(onset[i][:, None] - onset[i][None, :]) < gap_samples(fs, min_gap_s)))
    K = int(flags.sum())
    null = np.empty(B)
    for b in range(B):
        ts = ns = 0.0
        nn = 0
        for ag, k, near in groups:
            if k == 0:
                ns += ag.sum()
                nn += len(ag)
                continue
            idx = rng.permutation(len(ag))[:k]
            keep = ~near[:, idx].any(1)
            ts += ag[idx].sum()
            ns += ag[keep].sum()
            nn += int(keep.sum())
        null[b] = ts / K - ns / nn
    return {"n_target": K, "n_nontarget": int(el.sum()), "n_excluded_near": int((~flags & ~el).sum()), "diff_uv": obs,
            "p": float((1 + (null >= obs - 1e-12).sum()) / (B + 1)), "B": B, "min_gap_s": min_gap_s}


def global_sequence(runs):
    """Sequence ids unique across runs (one id per run x sequence)."""
    ids, off = [], 0
    for r in runs:
        s = np.asarray(r["seq"])
        ids.append(s + off)
        off += int(s.max()) + 1
    return np.concatenate(ids), off


def patient_stat(runs, B=B_PERM, seed=(SEED,)):
    amps, flags, seqs, ons, dropped = [], [], [], [], 0
    gseq, _ = global_sequence(runs)
    k = 0
    for r in runs:
        y = bandpass(r["x"][PZ], r["fs"])
        a, ok = epoch_means(y, r["fs"], r["onset"])
        n = len(r["onset"])
        ons.append(np.asarray(r["onset"])[ok]); amps.append(a[ok]); flags.append(np.asarray(r["is_target"], bool)[ok]); seqs.append(gseq[k:k + n][ok])
        dropped += int((~ok).sum())
        k += n
    a, f, s = np.concatenate(amps), np.concatenate(flags), np.concatenate(seqs)
    fs = float(runs[0]["fs"])
    res = perm_test(a, f, s, np.concatenate(ons), fs, B, np.random.default_rng(list(seed)))
    res["all_nontargets"] = perm_test_all(a, f, s, B, np.random.default_rng(list(seed)))
    res.update({"n_runs": len(runs), "n_dropped": dropped,
                "n_runs_train": sum(1 for r in runs if r["meta"].get("phase") == "Train"),
                "n_runs_test": sum(1 for r in runs if r["meta"].get("phase") == "Test")})
    return res


def go_rule(patients):
    """GO iff (1) at least GO_MIN of the patients have P < .05 AND (2) the grand-average target-minus-non-target difference
    (mean over patients of the per-patient difference) is positive. Otherwise NOGO."""
    n = len(patients)
    k = int(sum(p["p"] < ALPHA for p in patients))
    d = np.array([p["diff_uv"] for p in patients], float)
    c1, c2 = k >= GO_MIN, bool(d.mean() > 0)
    return {"n_patients": n, "n_significant": k, "threshold_n": GO_MIN, "condition_1_met": bool(c1),
            "grand_average_diff_uv": float(d.mean()), "median_diff_uv": float(np.median(d)), "condition_2_met": c2,
            "go": bool(c1 and c2), "nogo": bool(not (c1 and c2))}



def subjects(study):
    d = os.path.join(bio.SRC, f"Study{study}")
    return sorted(s for s in os.listdir(d) if not s.startswith("._") and os.path.isdir(os.path.join(d, s)))


def _work(args):
    study, subj = args
    return study, subj, patient_stat(bio.flashes(study, subj), seed=(SEED, ord(study), int(subj.split("_")[1])))


def design_facts(runs):
    """Checks of the stated design on a list of runs (bigp3_als_io.flashes dicts): the first Train file in time order (session
    id, then file name; never chosen by result) with its SOA, flash length, flashes and target probability; and the smallest
    spacing between two targets over all the runs (flashes and seconds), with how many runs have a spacing below 4 flashes."""
    def one(r):
        g = target_gaps(r["onset"], r["is_target"], r["seq"], r["fs"])
        return dict(g, n_flashes=int(len(r["onset"])), n_targets=int(np.asarray(r["is_target"]).sum()),
                    target_probability=float(np.asarray(r["is_target"]).mean()), soa_s=r["meta"]["soa_s"],
                    flash_ms=r["meta"]["flash_ms"], session=r["meta"]["session"], file=r["meta"]["file"])
    rows = [one(r) for r in runs]
    train = sorted((x for r, x in zip(runs, rows) if r["meta"]["phase"] == "Train"), key=lambda x: (x["session"], x["file"]))
    gf = [x["min_gap_flashes"] for x in rows if x["min_gap_flashes"] is not None]
    gs = [x["min_gap_s"] for x in rows if x["min_gap_s"] is not None]
    return {"first_train_file": train[0] if train else None,
            "all_runs": {"n_runs": len(rows), "min_gap_flashes": min(gf) if gf else None, "min_gap_s": min(gs) if gs else None,
                         "n_runs_with_gap_below_4": int(sum(g < 4 for g in gf)),
                         "mean_target_probability_train": float(np.mean([x["target_probability"] for r, x in zip(runs, rows)
                                                                         if r["meta"]["phase"] == "Train"])) if train else None}}


def paradigm_facts(study, subj_list):
    """Per-study facts from the extracted runs: paradigms, phases, SOA, flash duration, flashes / targets per sequence."""
    metas = [m for s in subj_list for m in (r["meta"] for r in bio.flashes(study, s))]
    fps = sorted({f for m in metas for f in m["flashes_per_sequence"]})
    tps = sorted({t for m in metas for t in m["targets_per_sequence"]})
    by = {}
    for m in metas:
        key = f"{m['phase']}/{m['paradigm']}"
        d = by.setdefault(key, {"n_runs": 0, "flashes_per_sequence": set(), "targets_per_sequence": set()})
        d["n_runs"] += 1
        d["flashes_per_sequence"] |= set(m["flashes_per_sequence"])
        d["targets_per_sequence"] |= set(m["targets_per_sequence"])
    for d in by.values():
        d["flashes_per_sequence"] = sorted(d["flashes_per_sequence"])
        d["targets_per_sequence"] = sorted(d["targets_per_sequence"])
    return {"n_patients": len(subj_list), "n_runs": len(metas), "n_runs_train": sum(m["phase"] == "Train" for m in metas),
            "n_runs_test": sum(m["phase"] == "Test" for m in metas),
            "soa_s": sorted({round(m["soa_s"], 4) for m in metas}), "flash_ms": sorted({round(m["flash_ms"], 3) for m in metas}),
            "fs_hz": sorted({round(m["fs"], 3) for m in metas}), "n_chars_in_grid": sorted({m["n_chars"] for m in metas}),
            "flashes_per_sequence_values": fps, "targets_per_sequence_values": tps,
            "n_sequences_per_run": [int(min(m["n_sequences"] for m in metas)), int(max(m["n_sequences"] for m in metas))],
            "by_phase_paradigm": by}


def main(workers=2, do_extract=True):
    import multiprocessing as mp
    todo = [(st, s) for st in bio.STUDIES for s in subjects(st)]
    if do_extract:
        with mp.get_context("fork").Pool(min(workers, 2)) as pool:
            pool.starmap(_extract_one, todo, chunksize=1)
    with mp.get_context("fork").Pool(min(workers, 2)) as pool:
        res = pool.map(_work, todo, chunksize=1)
    rows = {f"{st}:{s}": dict(r, study=st, subject=s) for st, s, r in res}
    pats = list(rows.values())
    per_study = {st: {"n_patients": sum(p["study"] == st for p in pats),
                      "n_significant": sum(p["study"] == st and p["p"] < ALPHA for p in pats)} for st in bio.STUDIES}
    go = go_rule(pats)
    pooled = float(np.average([p["diff_uv"] for p in pats], weights=[p["n_target"] for p in pats]))
    old = [p["all_nontargets"] for p in pats]
    first_subj = {st: subjects(st)[0] for st in bio.STUDIES}
    d = {"licence": bio.LICENCE, "signal": "Pz, 0.1-30 Hz zero-phase; mean 250-600 ms minus baseline -100-0 ms, µV",
         "non_target_rule": f"only non-target flashes >= {MIN_GAP_S} s (4 flashes at the 250-ms SOA) from every target flash of the same sequence",
         "test": f"one-sided permutation, {B_PERM} within-sequence shuffles of the target flags (far-non-target set recomputed per draw); "
                 "P = (1 + #{null >= obs}) / (B + 1)",
         "seed": f"default_rng([{SEED}, ord(study letter), subject number])",
         "rule": f"GO if >= {GO_MIN} of 48 patients have P < .05 AND the grand-average target-minus-non-target difference is positive; else NOGO",
         "go_rule": go, "per_study": per_study, "n_negative_diff": int(sum(p["diff_uv"] < 0 for p in pats)),
         "n_p_above_95": int(sum(p["p"] > 0.95 for p in pats)),
         "grand_average_target_weighted_diff_uv": pooled,
         "excluded_near_nontargets_total": int(sum(p["n_excluded_near"] for p in pats)),
         "nontargets_used_total": int(sum(p["n_nontarget"] for p in pats)), "targets_total": int(sum(p["n_target"] for p in pats)),
         "superseded_all_nontargets": {"n_significant": int(sum(o["p"] < ALPHA for o in old)),
                                       "grand_average_diff_uv": float(np.mean([o["diff_uv"] for o in old])),
                                       "median_diff_uv": float(np.median([o["diff_uv"] for o in old]))},
         "design_facts": {st: design_facts(bio.flashes(st, first_subj[st])) for st in bio.STUDIES},
         "design_facts_all_patients": {st: design_facts([r for s in subjects(st) for r in bio.flashes(st, s)]) for st in bio.STUDIES},
         "paradigm_facts": {st: paradigm_facts(st, subjects(st)) for st in bio.STUDIES}, "patients": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(d, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)
    return d


def _extract_one(study, subj):
    if not os.path.exists(bio.npz_path(study, subj)):
        bio.extract(study, subj)
    return subj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    r = main(a.workers, not a.no_extract)
    print(json.dumps({k: v for k, v in r.items() if k not in ("patients", "paradigm_facts")}, indent=1))
