"""Confirmatory replication of the M1 check, with the logic and decision rule of 117_m1_check.py, on the BDSP extra
listing (engine.bdsp_extras: patient-disjoint from the 108/113 design sample; its B-design rates are unseen).
Confirmatory replication of the M1 check (part A4), with the logic
and decision rule of (117_m1_check.py), on the BDSP extra listing
(engine.bdsp_extras: patient-disjoint from the 108/113 design sample; its B-design rates are unseen).
- Segments: the extras' analysis stems, loaded with the frozen engine loader (bench.engine._load); F3, F4, C3, C4.
- Feature series: 117.grid_series (8-30 Hz log power 0.5-4.5 s after each point of one 10-s grid over the whole
  segment). Padding (108.padding_seconds, widened by 108's 30-s margin) never enters a series: the surrogates, AR
  residuals and IAAFT (117.series_set) are computed per clean span, a maximal run of clean grid slots long enough for
  a 48-trial run. Seed of a span's surrogates: 20261014 + 100 x seed index + span number.
- Runs: up to 20 per segment, 48 consecutive clean grid trials, drawn without replacement from every eligible start by
  np.random.default_rng([20261014, seed index]); a run is eligible exactly when 108.eligible_starts admits its 480 s.
  Seed index = the segment's row in bench_extra_500.csv. Morgoth stratum per run (engine.stratum).
- B2 at b2/b4/b8/b12 by 117.b2_rows; schedules by 117.schedules from np.random.default_rng([20261014, seed index,
  run]); every series of a run gets the same schedule, folds and permutations.
- Rates, R over b8 + b12, its paired recording-cluster bootstrap CI and the verdict by 117.summarize (R >= 0.5 keeps
  M1). R is reported as the share of the block-design excess explained by the recording's own spectrum.
-> output/tables/m1_confirm.parquet (per run), m1_confirm.json
python 120_m1_confirm.py [--split analysis|dev --limit N --out DIR]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import warnings
import numpy as np
import pandas as pd
from bench import engine as en

warnings.filterwarnings("ignore")
mk = importlib.import_module("117_m1_check")
bn = en.bn
CH = mk.m91.CH
TRIAL_S, RUN_TRIALS = mk.m91.TRIAL_S, mk.m91.RUN_TRIALS
N_RUNS = 20
SPAN_SEED = 100
LABEL = "confirmatory"
FRAMING = ("R: the share of the block-design excess explained by the recording's own spectrum (quantification of a "
           "known mechanism: Purdon and Weisskoff 1998; Eklund 2016)")


def items(split="analysis"):
    """[(seed index, engine item)]: the BDSP extras (seed index = csv row), or the BDSP dev split (index = rank)."""
    if split == "dev":
        return [(int(it["meta"]["rank"]), it) for it in en._items("bdsp", "dev")]
    return [(int(it["meta"]["extra_row"]), it) for it in en._items("bdsp", "analysis") if it["meta"]["extra"]]


def overlay_channels(rec):
    """F3, F4, C3, C4 of an engine Recording as (samples, 4) float64, uV."""
    return np.asarray(rec.x[[rec.names.index(c) for c in CH]], float).T


def clean_slots(x4, fs):
    """Per 10-s grid slot (117.n_grid slots): True if its 10 s hold no padding within 108's margin."""
    el = bn.eligible_starts(np.asarray(x4).T, fs, run_s=TRIAL_S)
    ok = np.zeros(mk.n_grid(len(x4), fs), bool)
    g = el[el % TRIAL_S == 0] // TRIAL_S
    ok[g[g < len(ok)]] = True
    return ok


def spans(clean, min_len=RUN_TRIALS):
    """Maximal runs [a, b) of clean slots with at least min_len slots."""
    d = np.diff(np.r_[0, np.asarray(clean, int), 0])
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def run_offsets(sp, j, n_runs=N_RUNS):
    """Up to n_runs first grid indices of 48-trial runs inside the spans, without replacement, in draw order."""
    el = np.array([g for a, b in sp for g in range(a, b - RUN_TRIALS + 1)], int)
    if not len(el):
        return []
    rng = np.random.default_rng([mk.NOGO_SEED, j])
    return [int(g) for g in rng.choice(el, min(n_runs, len(el)), replace=False)]


def segment_rows(j, rec, n_surr=mk.N_SURR, n_perm=mk.m91.N_PERM, n_runs=N_RUNS):
    """B2 calls of every series, run and block length of one segment (seed index j), and diagnostics."""
    x4 = overlay_channels(rec)
    F = mk.grid_series(x4, rec.fs)
    sp = spans(clean_slots(x4, rec.fs))
    offs = run_offsets(sp, j, n_runs)
    base = {"recording": rec.uid, "seed_index": j}
    if not offs:
        return [], {**base, "excluded": "no eligible run", "n_grid": int(len(F))}
    rows, used = [], []
    for s, (a, b) in enumerate(sp):
        mine = [(k, g) for k, g in enumerate(offs) if a <= g and g + RUN_TRIALS <= b]
        if not mine:
            continue
        S, d = mk.series_set(F[a:b], SPAN_SEED * j + s, n_surr)
        runs = [(k, g - a, mk.schedules(np.random.default_rng([mk.NOGO_SEED, j, k]))) for k, g in mine]
        strata = {k: en.stratum(rec, float(g * TRIAL_S), float(RUN_TRIALS * TRIAL_S)) for k, g in mine}
        for r in mk.b2_rows(S, runs, j, n_perm):
            st, tr = strata[r["offset"]]
            rows.append({**base, "span": s, **r, "grid_index": r["grid_index"] + a, "stratum": st, "transition": tr})
        used.append({"span": [a, b], **d})
    return rows, {**base, "n_grid": int(len(F)), "spans": [list(x) for x in sp], "offsets": offs,
                  "n_spans_used": len(used), "span_diag": used}


def summarize(df, diags, listing):
    s = mk.summarize(df, diags)
    s.update({"label": LABEL, "framing": FRAMING, "listing": listing,
              "n_excluded": int(sum("excluded" in d for d in diags))})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("analysis", "dev"), default="analysis")
    ap.add_argument("--limit", type=int, default=0, help="first N segments only (smoke test; use with --out)")
    ap.add_argument("--out", default=mk.m91.T)
    a = ap.parse_args()
    if a.split == "dev" and os.path.abspath(a.out) == os.path.abspath(mk.m91.T):
        ap.error("dev runs write outside output/tables: pass --out")
    ex = en.bdsp_extras()
    listing = {"split": a.split, "n_listed": ex["n_listed"], "n_present": ex["n_present"],
               "n_missing": len(ex["missing"]), "n_failed": len(ex["failed"]), "complete": ex["complete"]}
    rows, diags = [], []
    for j, it in items(a.split)[: a.limit or None]:
        rec = en._load(it)
        r, d = segment_rows(j, rec)
        del rec
        rows += r
        diags.append(d)
        print(j, it["uid"], len(d.get("offsets", [])), d.get("n_spans_used", 0), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{a.out}/m1_confirm.parquet", index=False)
    s = summarize(df, diags, listing)
    json.dump(s, open(f"{a.out}/m1_confirm.json", "w"), indent=2, default=float)
    print(json.dumps({k: s[k] for k in ("pooled_b8_b12", "nogo_rule", "support_prewhitened_le_6pct", "n_excluded")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
