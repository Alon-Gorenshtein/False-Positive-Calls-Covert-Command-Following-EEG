"""Clinical-EEG background null.
Clinical-EEG background null, check.
De-identified clinical EEG from the BDSP (500 segments, one per patient), streamed read-only to the Extreme SSD by
bdsp_pull.py. Per segment, 40 random-offset 48-trial runs (10-s trials, F3 F4 C3 C4, 8-30 Hz log power) receive
command-following designs although no command was ever given, so every call is false:
- B1/B2 block schedules at 2/4/8/12-trial blocks (doc_null statistics; B2 = CV AUROC + trial-label permutation);
- B4 (per-trial randomized labels, B2's statistic) and R (E6 betting on fresh iid labels): implementation checks;
- Columbia-style clip analysis, alternating commands, five 2-s clips per command: as described (Pcol_a) and grouped
  by command (Pcol_b).
Moderator: drift index = mean autocorrelation of the channel-averaged trial log-power at lags 2-12 trials.
Strata: Morgoth-assigned background over the run (row_labels -> run_stratum).
python 108_bdsp_null.py --shard i --nshards n | --merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
import doc_null as dn

warnings.filterwarnings("ignore")
SIG, MOR, META = "../output/bdsp/sig", "../output/bdsp/morgoth", "../output/bdsp/meta"
T, SH = "../output/tables", "../output/intermediate/_bdsp_null_shards"
SEED = 20261019
OVERLAYS, TRIAL_S, RUN_TRIALS = 40, 10, 48
RUN_S = TRIAL_S * RUN_TRIALS
CH = ("F3", "F4", "C3", "C4")
BLOCKS = (2, 4, 8, 12)
N_PERM, N_PERM_COL = 1000, 500
STRATA = ("normal_awake", "sleep", "diffuse_slowing", "iiic_periodic", "burst_suppression")
ZERO_UV = 1e-3
FLOOR_UV, FLOOR_S = 2.0, 60
MARGIN_S = 30
MIN_SEGS = 30
NORMAL_IS_P_NORMAL = True
DRAWS = 2000


def drift_index(F, lags=range(2, 13)):
    s = np.asarray(F, float).mean(1)
    s = s - s.mean()
    v = float((s ** 2).sum())
    if v <= 0:
        return float("nan")
    return float(np.mean([(s[:-k] * s[k:]).sum() / v for k in lags]))


def row_labels(m, normal_is_p_normal=NORMAL_IS_P_NORMAL):
    """One label per Morgoth window, by a fixed hierarchy: burst suppression > IIIC/periodic > sleep > generalized
    slowing > normal awake > other awake. Invalid windows are 'invalid'."""
    sleep = m.sleep_n1 + m.sleep_n2 + m.sleep_n3 + m.sleep_rem
    normal = m.normal_p if normal_is_p_normal else 1.0 - m.normal_p
    lab = np.select([m.bs_p > 0.5, (1.0 - m.iiic_other) > 0.5, sleep > 0.5, m.slowing_generalized > 0.5,
                     normal > 0.5],
                    ["burst_suppression", "iiic_periodic", "sleep", "diffuse_slowing", "normal_awake"],
                    "other_awake").astype(object)
    lab[~np.asarray(m.valid, bool)] = "invalid"
    return lab


def _mode(lab):
    v, c = np.unique(lab, return_counts=True)
    return v[np.argmax(c)], c.max() / c.sum()


def run_stratum(lab, min_valid=0.5, min_share=0.6):
    """(stratum, transitioning) for the Morgoth windows of one run. 'unlabelled' if under half the windows are valid;
    'mixed' if no label holds 60% of the valid windows; transitioning if the two halves have different modes."""
    lab = np.asarray(lab, object)
    ok = lab != "invalid"
    if ok.mean() < min_valid:
        return "unlabelled", False
    m, share = _mode(lab[ok])
    h = len(lab) // 2
    a, b = lab[:h][ok[:h]], lab[h:][ok[h:]]
    trans = bool(len(a) and len(b) and _mode(a)[0] != _mode(b)[0])
    return (m if share >= min_share else "mixed"), trans


def padding_seconds(x, fs):
    """Per whole second: True if it is padding (part 1): exact zeros on any overlay channel, or part of a
    stretch of >= FLOOR_S s with every overlay channel below FLOOR_UV. x: channels x samples."""
    fs = int(fs)
    n = x.shape[1] // fs
    sd = np.asarray(x[:, : n * fs], float).reshape(x.shape[0], n, fs).std(2)
    pad = sd.min(0) < ZERO_UV
    low = sd.max(0) < FLOOR_UV
    d = np.diff(np.r_[0, low.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= FLOOR_S:
            pad[a:b] = True
    return pad


def eligible_starts(x, fs, run_s=RUN_S):
    """Run starts (whole seconds) with no padding within the run or MARGIN_S around it. Large artifacts are kept on
    purpose (fixed in advance: no artifact rejection)."""
    pad = padding_seconds(x, fs)
    n = len(pad)
    bad = pad.copy()
    for k in range(1, MARGIN_S + 1):
        bad[k:] |= pad[:-k]
        bad[:-k] |= pad[k:]
    c = np.r_[0, np.cumsum(bad.astype(int))]
    s = np.arange(0, n - run_s + 1)
    return s[(c[s + run_s] - c[s]) == 0]


def _bandpass(x, fs):
    sos = butter(4, [8.0 / (fs / 2), 30.0 / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float), axis=1)


def overlay(x, fs, start_s, rng, n_perm=N_PERM, xf=None):
    """All designs on one 48-trial run starting at start_s. x: overlay channels x samples (uV)."""
    xf = _bandpass(x, fs) if xf is None else xf
    fs = int(fs)
    starts = int(start_s * fs) + np.arange(RUN_TRIALS) * TRIAL_S * fs
    a, b, e = int(0.5 * fs), int(4.5 * fs), 2 * fs
    F = np.log(np.array([np.mean(xf[:, s + a: s + b] ** 2, axis=1) for s in starts]) + 1e-12)
    Fe = np.log(np.array([np.mean(xf[:, s + j * e: s + (j + 1) * e] ** 2, axis=1)
                          for s in starts for j in range(5)]) + 1e-12)
    calls = {}
    for blk in BLOCKS:
        y, _ = dn.schedule("block", rng, block=blk)
        calls[f"B1_b{blk}"] = dn.b1(F, y, rng)["call"]
        calls[f"B2_b{blk}"] = dn.b2(F, y, rng, n_perm)["call"]
    yr, _ = dn.schedule("random", rng)
    calls["B4"] = dn.b2(F, yr, rng, n_perm)["call"]
    calls["R"] = dn.r_test(F, rng.integers(0, 2, RUN_TRIALS))["call"]
    cmd = dn.alternating(RUN_TRIALS, rng)
    calls["Pcol_a"] = dn.columbia(Fe, cmd, rng, n_perm=min(n_perm, N_PERM_COL))["call"]
    calls["Pcol_b"] = dn.columbia(Fe, cmd, rng, n_perm=min(n_perm, N_PERM_COL), grouped=True)["call"]
    return {"calls": {k: bool(v) for k, v in calls.items()}, "drift": drift_index(F)}



def _counts(rows):
    segs = np.unique(rows.seg)
    strata = sorted(rows.stratum.unique())
    g = rows.groupby(["seg", "stratum"]).call.agg(["sum", "count"]).unstack(fill_value=0)
    S = g["sum"].reindex(index=segs, columns=strata, fill_value=0).to_numpy(float)
    N = g["count"].reindex(index=segs, columns=strata, fill_value=0).to_numpy(float)
    return strata, S, N


def stratum_rates(rows):
    return rows.groupby("stratum").call.mean().to_dict()


def _boot_rates(rows, draws, seed):
    """Cluster bootstrap over segments (one per patient); each resampled segment counts as its own cluster."""
    strata, S, N = _counts(rows)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(S.shape[0], np.full(S.shape[0], 1.0 / S.shape[0]), size=draws).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (W @ S) / (W @ N)
    return strata, R


def pair_diff_ci(rows, hi_stratum, lo_stratum, draws=DRAWS, seed=SEED):
    r = stratum_rates(rows)
    strata, R = _boot_rates(rows, draws, seed)
    d = R[:, strata.index(hi_stratum)] - R[:, strata.index(lo_stratum)]
    d = d[np.isfinite(d)]
    return float(r[hi_stratum] - r[lo_stratum]), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def not_above_alpha(rows, stratum, draws=DRAWS, seed=SEED, alpha=dn.ALPHA):
    """True unless the one-sided 95% lower bound of the stratum's call rate exceeds alpha."""
    strata, R = _boot_rates(rows, draws, seed)
    r = R[:, strata.index(stratum)]
    return bool(np.quantile(r[np.isfinite(r)], 0.05) <= alpha)



def sample():
    s = pd.read_csv(f"{META}/sample_500.csv")
    s["stem"] = s.file.str.replace(".mat", "", regex=False)
    s["seg"] = np.arange(len(s))
    ok = [os.path.exists(f"{SIG}/{t}.npz") and os.path.exists(f"{MOR}/{t}.parquet") for t in s.stem]
    return s[ok].reset_index(drop=True)


def segment_rows(seg, stem):
    z = np.load(f"{SIG}/{stem}.npz")
    names = [str(n) for n in z["names"]]
    x = np.asarray(z["x"], np.float32)[[names.index(c) for c in CH]]
    fs = float(z["fs"])
    m = pd.read_parquet(f"{MOR}/{stem}.parquet")
    lab = row_labels(m)
    st = eligible_starts(x, fs)
    if len(st) == 0:
        return [], {"seg": seg, "excluded": "no eligible run"}
    rng = np.random.default_rng(SEED + seg)
    offs = rng.choice(st, OVERLAYS, replace=True)
    xf = _bandpass(x, fs)
    t = m.t_s.to_numpy()
    rows = []
    for k, o in enumerate(offs):
        stratum, trans = run_stratum(lab[(t >= o) & (t <= o + RUN_S - 10)])
        out = overlay(x, fs, int(o), rng, xf=xf)
        for design, call in out["calls"].items():
            rows.append({"seg": seg, "overlay": k, "stratum": stratum, "transition": trans, "drift": out["drift"],
                         "design": design, "call": call})
    return rows, None


def run_shard(shard, nshards):
    os.makedirs(SH, exist_ok=True)
    out = f"{SH}/part{shard:02d}.parquet"
    s = sample()
    rows, excl = [], []
    for i in range(shard, len(s), nshards):
        r, e = segment_rows(int(s.seg[i]), s.stem[i])
        rows += r
        if e:
            excl.append(e)
        print(f"seg {int(s.seg[i])} {len(r)} rows", flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    json.dump(excl, open(f"{SH}/excluded{shard:02d}.json", "w"))


def gates(df):
    clean = df[df.stratum.isin(STRATA) & ~df.transition]
    nseg = clean.groupby("stratum").seg.nunique()
    gate_strata = [s for s in STRATA if nseg.get(s, 0) >= MIN_SEGS]
    out = {"segments": int(df.seg.nunique()), "segments_per_stratum": nseg.to_dict(), "gate_strata": gate_strata,
           "rates": {}, "a": {}, "b": {}, "c": {}}
    for design, g in df.groupby("design"):
        c = g[g.stratum.isin(STRATA) & ~g.transition]
        r = stratum_rates(c)
        r["transitioning"] = float(g[g.transition].call.mean()) if g.transition.any() else None
        r["mixed_or_unlabelled"] = float(g[g.stratum.isin(["mixed", "unlabelled", "other_awake"])].call.mean())
        r["all"] = float(g.call.mean())
        out["rates"][design] = r
    for blk in BLOCKS:
        d = f"B2_b{blk}"
        c = clean[(clean.design == d) & clean.stratum.isin(gate_strata)]
        r = stratum_rates(c)
        if len(r) < 2:
            out["a"][d] = {"pass": False, "reason": "fewer than 2 gate strata"}
            continue
        hi, lo = max(r, key=r.get), min(r, key=r.get)
        diff, l, u = pair_diff_ci(c, hi, lo)
        out["a"][d] = {"hi": hi, "lo": lo, "diff": diff, "ci": [l, u], "pass": bool(diff >= 0.10 and l > 0)}
        per = df[df.design == d].groupby("seg").agg(drift=("drift", "mean"), p=("call", "mean"))
        rho = float(spearmanr(per.drift, per.p).correlation)
        out["b"][d] = {"rho": rho, "pass": bool(rho >= 0.2)}
    for d in ("R", "B4"):
        g = df[df.design == d].copy()
        g["stratum"] = np.where(g.transition, "transitioning", g.stratum)
        out["c"][d] = {s: not_above_alpha(g, s) for s in sorted(g.stratum.unique())}
    out["gate_a"] = any(v["pass"] for v in out["a"].values())
    out["gate_b"] = any(v["pass"] for v in out["b"].values())
    out["gate_c"] = all(all(v.values()) for v in out["c"].values())
    out["decision"] = "scale" if out["gate_a"] and out["gate_b"] and out["gate_c"] else (
        "stop and debug" if not out["gate_c"] else "do not scale")
    return out


def merge():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SH}/part*.parquet"))], ignore_index=True)
    excl = sum((json.load(open(f)) for f in sorted(glob.glob(f"{SH}/excluded*.json"))), [])
    df.to_parquet("../output/intermediate/bdsp_null_rows.parquet", index=False)
    out = gates(df)
    out["excluded"] = len(excl)
    json.dump(out, open(f"{T}/bdsp_null.json", "w"), indent=2, default=float)
    print(json.dumps({k: out[k] for k in ("segments", "gate_a", "gate_b", "gate_c", "decision")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run_shard(a.shard, a.nshards)


if __name__ == "__main__":
    main()
