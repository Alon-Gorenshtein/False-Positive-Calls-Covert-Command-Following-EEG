"""The published CMD pipeline on BDSP clinical EEG recorded without commands.
One overlay per eligible recording: (i) exact epoch-level permutation, (ii) trial-flip permutation, (iii) randomized
keep/stop order with the trial-flip permutation. Decision-exact early stopping (112.STOP_AT).
python 113_columbia_null.py --shard i --nshards n [--limit N] | --merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import importlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ce = importlib.import_module("112_columbia_exact")
bn = importlib.import_module("108_bdsp_null")
SEED = 20261022
SPAN_S = 1505
N_PRIMARY = 150
DEV_N = 20
SIG, MOR, META = "../output/bdsp/sig", "../output/bdsp/morgoth", "../output/bdsp/meta"
SH, T = "../output/intermediate/_columbia_null_shards", "../output/tables"
DRAWS = 2000
SEED_EXTRA = 20261026
FROZEN = f"{META}/dev_split.json"
SOURCES = {"design": (SIG, MOR, f"{META}/sample_500.csv", SEED),
           "extra": ("../output/bdsp_extra/sig", "../output/bdsp_extra/morgoth", f"{META}/bench_extra_500.csv",
                     SEED_EXTRA)}
SLEEP = ("sleep_awake", "sleep_n1", "sleep_n2", "sleep_n3", "sleep_rem")
N2_ANY = 0.10


def sample_order(n, seed=SEED):
    """Rank order over the whole design sample (indices of sample_500.csv), fixed before any data are seen."""
    return np.random.default_rng(seed).permutation(n)


def randomized_order(y, groups, seed):
    """Keep/stop order randomized per trial: each trial's labels flipped as a unit with probability 0.5."""
    return ce.trialflip_labels(np.asarray(y, bool), np.asarray(groups), np.random.RandomState(seed))


def trial_drift(X, groups):
    """Drift index of the per-trial log mean band power (lag 2-12 autocorrelation); scale-free."""
    g = np.asarray(groups)
    s = np.log(np.array([np.asarray(X)[g == t].mean() for t in np.unique(g)]))
    return bn.drift_index(s[:, None])


def eligible_list(source="design"):
    """(rank, sample index, stem) for every recording of the source with x19 and >= SPAN_S clean seconds, in seeded
    order."""
    sig, mor, lst, seed = SOURCES[source]
    s = pd.read_csv(lst)
    out = []
    for rank, i in enumerate(sample_order(len(s), seed=seed)):
        stem = s.file[i].replace(".mat", "")
        f = f"{sig}/{stem}.npz"
        if not (os.path.exists(f) and os.path.exists(f"{mor}/{stem}.parquet")):
            continue
        z = np.load(f)
        if "x19" not in z.files:
            continue
        if len(bn.eligible_starts(np.asarray(z["x19"], np.float32), float(z["fs"]), run_s=SPAN_S)):
            out.append((rank, int(i), stem))
    return out


def split_dev(elig, n=DEV_N):
    """The last n eligible recordings in the seeded order form the development split; the rest are analysed."""
    return elig[:-n], elig[-n:]


def checked_split(elig, frozen=FROZEN):
    """split_dev, refused unless its dev members equal the frozen dev_split.json."""
    ana, dev = split_dev(elig)
    if not os.path.exists(frozen):
        raise RuntimeError(f"no frozen dev split at {frozen}")
    want = [d[2] for d in json.load(open(frozen))["bdsp"]]
    if [d[2] for d in dev] != want:
        raise RuntimeError("113's dev split differs from the frozen dev_split.json")
    return ana, dev


def n2_features(m, start_s, span_s):
    """Morgoth-assigned N2 (argmax over the five sleep-head classes, scorable rows only): share over the whole
    recording, the N2_ANY flag, and share within the overlay span [start_s, start_s + span_s]. A proxy for organised
    N2 sleep; spindles themselves are not detected."""
    ok = np.asarray(m.valid_sleep, bool) & m[list(SLEEP)].notna().all(axis=1).to_numpy()
    n2 = np.asarray(m[list(SLEEP)].to_numpy().argmax(axis=1) == SLEEP.index("sleep_n2"))
    t = m.t_s.to_numpy()
    win = ok & (t >= start_s) & (t <= start_s + span_s)
    rec = float(n2[ok].mean()) if ok.any() else 0.0
    return {"n2_share_rec": rec, "n2_any": bool(rec >= N2_ANY),
            "n2_share_overlay": float(n2[win].mean()) if win.any() else 0.0}


def run_one(rank, i, stem, source="design"):
    sig, mor, _, seed = SOURCES[source]
    z = np.load(f"{sig}/{stem}.npz")
    x, fs, names = np.asarray(z["x19"], np.float32), float(z["fs"]), [str(n) for n in z["names19"]]
    st = bn.eligible_starts(x, fs, run_s=SPAN_S)
    start = int(np.random.default_rng(seed + i).choice(st)) + 1
    on, lab = ce.schedule(start)
    X, y, g = ce.features(x, fs, on, lab, names)
    r1 = ce.exact_test(X, y, g, np.random.RandomState(seed + i), stop_at=ce.STOP_AT)
    r2 = ce.trialflip_test(X, y, g, np.random.RandomState(seed + 10007 + i), stop_at=ce.STOP_AT)
    yr = randomized_order(y, g, seed + 20011 + i)
    r3 = ce.trialflip_test(X, yr, g, np.random.RandomState(seed + 30011 + i), stop_at=ce.STOP_AT)
    m = pd.read_parquet(f"{mor}/{stem}.parquet")
    lab_m = bn.row_labels(m)
    t = m.t_s.to_numpy()
    stratum, trans = bn.run_stratum(lab_m[(t >= start) & (t <= start + SPAN_S - 10)])
    row = {"rank": rank, "sample_idx": i, "start_s": start, "drift": trial_drift(X, g), "stratum": stratum,
           "transition": trans, **n2_features(m, start, SPAN_S - 10)}
    for k, r in (("i", r1), ("ii", r2), ("iii", r3)):
        row |= {f"{k}_auc": r["auc"], f"{k}_p": r["p"], f"{k}_call": r["call"], f"{k}_call_paper": r["call_paper"],
                f"{k}_stopped": r["stopped_early"], f"{k}_nperm": r["n_perm_run"]}
    return row


def touched_stems(path=None):
    """part 3: recordings a benchmark builder ran a pipeline on while provisional dev (git-ignored meta file)."""
    return set(json.load(open(path or f"{META}/touched_before_freeze.json"))["excluded"])


def drop_touched(df):
    stems = pd.read_csv(f"{META}/sample_500.csv").file.str.replace(".mat", "", regex=False)
    return df[~stems.iloc[df.sample_idx.to_numpy()].isin(touched_stems()).to_numpy()]


def assert_touched_not_primary(df):
    """part 3 claim: no touched recording is in the PRIMARY 150."""
    stems = pd.read_csv(f"{META}/sample_500.csv").file.str.replace(".mat", "", regex=False)
    hit = set(stems.iloc[df[df.primary.astype(bool)].sample_idx.to_numpy()]) & touched_stems()
    if hit:
        raise RuntimeError(f"{len(hit)} touched recording(s) in the PRIMARY sample")


def run_shard(shard, nshards, limit, source="design"):
    os.makedirs(SH, exist_ok=True)
    elig = (checked_split(eligible_list())[0] if source == "design" else eligible_list("extra"))[:limit]
    done_f = f"{SH}/{'rows' if source == 'design' else 'extra'}{shard:02d}.jsonl"
    done = set()
    if os.path.exists(done_f):
        done = {json.loads(l)["rank"] for l in open(done_f)}
    with open(done_f, "a") as fh:
        for j, (rank, i, stem) in enumerate(elig):
            if j % nshards != shard or rank in done:
                continue
            row = run_one(rank, i, stem, source)
            row |= {"primary": source == "design" and j < N_PRIMARY, "source": source, "stem": stem}
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            print(f"rank {rank} done", flush=True)


def _lower(v, draws=DRAWS, seed=SEED, q=0.05):
    v = np.asarray(v, float)
    b = v[np.random.default_rng(seed).integers(0, len(v), size=(draws, len(v)))].mean(1)
    return float(np.quantile(b, q))


def _diff_ci(a, b, draws=DRAWS, seed=SEED):
    """Two-sided 95% bootstrap CI of mean(a) - mean(b), resampling recordings within each group."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (a[rng.integers(0, len(a), (draws, len(a)))].mean(1) - b[rng.integers(0, len(b), (draws, len(b)))].mean(1))
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def h3_n2(d):
    """part 2 (secondary, two-sided, not gated): (i) false calls in recordings with vs without N2 sleep."""
    import statsmodels.api as sm
    y, n2 = d.i_call.astype(float).to_numpy(), d.n2_any.astype(bool).to_numpy()
    out = {"n_n2": int(n2.sum()), "n_no_n2": int((~n2).sum())}
    for k in ("i", "ii", "iii"):
        c = d[f"{k}_call"].astype(float).to_numpy()
        out[f"{k}_rate_n2"] = float(c[n2].mean()) if n2.any() else float("nan")
        out[f"{k}_rate_no_n2"] = float(c[~n2].mean()) if (~n2).any() else float("nan")
    if n2.sum() >= 5 and (~n2).sum() >= 5:
        out["i_diff_ci95"] = _diff_ci(y[n2], y[~n2])
        X = sm.add_constant(np.column_stack([n2.astype(float), d.drift.to_numpy(float)]))
        try:
            fit = sm.Logit(y, X).fit(disp=0)
            lo, hi = np.exp(fit.conf_int()[1])
            out["i_or_adj_drift"] = [float(np.exp(fit.params[1])), float(lo), float(hi)]
        except Exception as e:
            out["i_or_adj_drift"] = f"not estimable: {type(e).__name__}"
        out["spearman_n2_overlay_i"] = float(spearmanr(d.n2_share_overlay, y).correlation)
    return out


def summarize(df):
    out = {}
    src = df.source if "source" in df else pd.Series("design", index=df.index)
    blocks = [("primary", df[df.primary.astype(bool)]), ("all", df[src == "design"])]
    if (src == "extra").any():
        blocks.append(("all_with_extras", df))
    for name, d in blocks:
        r = {"n": int(len(d))}
        for k in ("i", "ii", "iii"):
            r[f"{k}_rate"] = float(d[f"{k}_call"].mean())
            r[f"{k}_rate_paper_rule"] = float(d[f"{k}_call_paper"].mean())
            r[f"{k}_lower95_one_sided"] = _lower(d[f"{k}_call"].astype(float))
        r["H1_pass"] = bool(r["i_lower95_one_sided"] > 0.05)
        diff = d.i_call.astype(float).values - d.ii_call.astype(float).values
        r["i_minus_ii"] = float(diff.mean())
        r["i_minus_ii_lower95_one_sided"] = _lower(diff)
        r["H2_pass"] = bool(r["i_minus_ii_lower95_one_sided"] > 0)
        r["iii_check_pass"] = bool(r["iii_lower95_one_sided"] <= 0.05)
        r["by_stratum"] = d.groupby(np.where(d.transition, "transitioning", d.stratum)).agg(
            n=("i_call", "size"), i=("i_call", "mean"), ii=("ii_call", "mean"), iii=("iii_call", "mean")
        ).round(3).to_dict(orient="index")
        r["drift_spearman_i"] = float(spearmanr(d.drift, d.i_call.astype(float)).correlation)
        r["auc_i_quantiles"] = d.i_auc.quantile([0.05, 0.5, 0.95]).round(4).tolist()
        r["H3_n2"] = h3_n2(d)
        out[name] = r
    return out


def merge():
    des = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(f"{SH}/rows*.jsonl")) for l in open(f)])
    des = drop_touched(des.drop_duplicates("rank")).assign(source="design")
    assert_touched_not_primary(des)
    ext = [json.loads(l) for f in sorted(glob.glob(f"{SH}/extra*.jsonl")) for l in open(f)]
    df = pd.concat([des, pd.DataFrame(ext).drop_duplicates("rank")], ignore_index=True) if ext else des
    df.to_parquet("../output/intermediate/columbia_null_rows.parquet", index=False)
    out = summarize(df)
    json.dump(out, open(f"{T}/columbia_null.json", "w"), indent=2)
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "i_rate", "ii_rate", "iii_rate", "H1_pass", "H2_pass",
                                               "iii_check_pass")} for k, v in out.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--source", choices=sorted(SOURCES), default="design")
    a = ap.parse_args()
    merge() if a.merge else run_shard(a.shard, a.nshards, a.limit, a.source)


if __name__ == "__main__":
    main()
