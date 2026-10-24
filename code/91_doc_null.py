"""Real-DoC-EEG null on the Mendeley pDoC polysomnograms (CC BY 4.0).

- Channels: the plain-named F3, F4, C3, C4, present in every recording (fixed from the channel lists before any
  analysis; there is no Cz). 256 Hz; 8-30 Hz log band power in 0.5-4.5 s after each 10-s trial start.
- 20 random start offsets per recording (seed 20261013 + recording + offset), each a 48-trial (8-min) run.
- Designs per run: B1, B2 (12 x 4 blocks), B1long, B2long (6 x 8), B3 (12 x 4, exact block permutation), B4 and R
  (per-trial randomized), B2alt (48 alternating 10-s periods x five 2-s epochs; epoch- and period-level permutation).
- R's prior: LR on per-person z-scored 8-30 Hz log power at F3, F4, C3, C4 from EEGMMIDB imagery vs rest
  (epochs_cue, 105 people).
- Strata: diagnosis from the file name (VS, MCS-, MCS+); a sleep/arousal transition flag: the C3/C4 log delta/alpha
  ratio (30-s windows) averages on opposite sides of its recording median in the run's two halves, and the halves
  differ by more than the recording's SD of that ratio (a sustained shift).
- Accuracy rules (descriptive): share of runs whose CV accuracy (B1 statistic) is >= 70% and >= 80%.
- Recordings: the 41 unique PSGs. Patient_42_VS.edf, a byte copy of Patient_40_VS.edf, was moved out (declared
  correction, 2026-10-13); every recording keeps the seed index it had in the 42-file listing (seed_index).
-> output/tables/doc_null.parquet, doc_null.json
python 91_doc_null.py [--shard i --nshards n | --merge] [--published] [--tag T]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import re
import warnings
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
import doc_null as dn
import kit_stats as ks
import tags

warnings.filterwarnings("ignore")
RAW = "../output/raw/doc_psg"
CUE = "../output/intermediate/epochs_cue"
T = "../output/tables"
SHARDS = "../output/intermediate/_docnull_shards"
CH = ["F3", "F4", "C3", "C4"]
SEED = 20261013
OFFSETS = 20
TRIAL_S, RUN_TRIALS = 10, 48
N_PERM = 1000
EXCLUDED = ("Patient_42_VS.edf",)


def recordings():
    return sorted(f for f in glob.glob(f"{RAW}/*.edf") if not os.path.basename(f).startswith("._"))


def seed_index(name):
    """Position of a recording in the original 42-file listing (the recordings plus EXCLUDED), so removing the
    duplicate leaves every other recording's seeds unchanged."""
    return sorted([os.path.basename(f) for f in recordings()] + list(EXCLUDED)).index(os.path.basename(name))


def indexed_recordings():
    """(seed index, path) of every recording, in listing order."""
    return [(seed_index(p), p) for p in recordings()]


def diagnosis(path):
    m = re.search(r"_(VS|MCS[+-])\.edf$", os.path.basename(path))
    return m.group(1) if m else "unknown"


def _bandpass(x, fs, lo, hi):
    return sosfiltfilt(butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos"), x, axis=0)


def cue_logpower(X_volts, sf):
    """8-30 Hz log power per channel of EEGMMIDB cue epochs (trials, channels, samples), stored in VOLTS: converted to
    microvolts first (in volts^2 the 1e-12 floor flattens differences) and filtered along time. The first version
    filtered along the trials axis, which destroyed the signal."""
    sos = butter(4, [8.0 / (sf / 2), 30.0 / (sf / 2)], btype="band", output="sos")
    x = sosfiltfilt(sos, np.asarray(X_volts, float) * 1e6, axis=-1)
    return np.log(np.mean(x ** 2, axis=-1) + 1e-12)


def prior_model():
    X, Y = [], []
    for f in sorted(glob.glob(f"{CUE}/S*.npz")):
        if os.path.basename(f).startswith("._"):
            continue
        d = np.load(f)
        ch = list(d["ch"])
        idx = [ch.index(c) for c in CH]
        F = cue_logpower(d["X"][:, idx, :], float(d["sf"]))
        X.append((F - F.mean(0)) / F.std(0))
        Y.append(d["y"])
    return LogisticRegression(C=1.0, max_iter=1000).fit(np.concatenate(X), np.concatenate(Y))


def _windows_ratio(x, fs):
    """delta (1-4 Hz) / alpha (8-12 Hz) power ratio of the C3/C4 mean in 30-s windows."""
    c = x[:, 2:4].mean(1)
    d, a = _bandpass(c, fs, 1.0, 4.0), _bandpass(c, fs, 8.0, 12.0)
    w = int(30 * fs)
    n = len(c) // w
    return np.array([np.mean(d[i * w:(i + 1) * w] ** 2) / (np.mean(a[i * w:(i + 1) * w] ** 2) + 1e-12)
                     for i in range(n)])


def run_recording(ri, path, prior):
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    x = raw.get_data(picks=CH).T * 1e6
    xf = _bandpass(x, fs, 8.0, 30.0)
    lr = np.log(_windows_ratio(x, fs) + 1e-12)
    med, sd = np.median(lr), np.std(lr)
    run_s = TRIAL_S * RUN_TRIALS
    rows = []
    for k in range(OFFSETS):
        rng = np.random.default_rng(SEED + ri * 100 + k)
        start = int(rng.integers(0, int((len(x) / fs - run_s - 5) * fs)))
        starts = start + np.arange(RUN_TRIALS) * int(TRIAL_S * fs)
        a, b = int(0.5 * fs), int(4.5 * fs)
        F = np.log(np.array([np.mean(xf[s + a: s + b] ** 2, axis=0) for s in starts]) + 1e-12)
        w0, w1 = int(start / fs // 30), int((start / fs + run_s) // 30) + 1
        wm = (w0 + w1) // 2
        h1, h2 = lr[w0:wm].mean(), lr[wm:w1].mean()
        transition = bool((h1 - med) * (h2 - med) < 0 and abs(h1 - h2) > sd)
        lower_delta_half = bool(lr[w0:w1].mean() < med)
        base = {"recording": os.path.basename(path), "diagnosis": diagnosis(path), "offset": k,
                "transition": transition, "lower_delta_half": lower_delta_half}
        yb, blocks = dn.schedule("block", rng)
        yl, _ = dn.schedule("block", rng, block=8)
        yr, _ = dn.schedule("random", rng)
        res = {"B1": dn.b1(F, yb, rng), "B2": dn.b2(F, yb, rng, N_PERM), "B1long": dn.b1(F, yl, rng),
               "B2long": dn.b2(F, yl, rng, N_PERM), "B3": dn.b3(F, yb, blocks, rng), "B4": dn.b2(F, yr, rng, N_PERM),
               "R": dn.r_test(F, yr, prior=prior), "B1rand": dn.b1(F, yr, rng)}
        e = int(2 * fs)
        Fe = np.log(np.array([np.mean(xf[s + j * e: s + (j + 1) * e] ** 2, axis=0)
                              for s in starts for j in range(5)]) + 1e-12)
        first = int(rng.integers(0, 2))
        per = np.array([(first + i) % 2 for i in range(RUN_TRIALS)])
        alt = dn.b2alt(Fe, per, rng, N_PERM)
        res["B2alt_a"], res["B2alt_b"] = alt["a_epoch"], alt["b_period"]
        for blk in (2, 12):
            yk, _ = dn.schedule("block", rng, block=blk)
            res[f"B1_b{blk}"], res[f"B2_b{blk}"] = dn.b1(F, yk, rng), dn.b2(F, yk, rng, N_PERM)
        for design, o in res.items():
            rows.append({**base, "design": design, "call": bool(o["call"]), "p": o.get("p"),
                         "acc": o.get("acc"), "auc": o.get("auc"), "max_wealth": o.get("max_wealth")})
    return rows


BANDS = ((1.0, 3.0), (4.0, 7.0), (8.0, 13.0), (14.0, 30.0))


def run_recording_published(ri, path):
    """Published-protocol designs: Columbia (a, b), Cruse, Cornell."""
    from scipy.signal import welch
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    fs = raw.info["sfreq"]
    x = raw.get_data(picks=CH).T * 1e6
    xb = [_bandpass(x, fs, lo, hi) for lo, hi in BANDS]
    xf = _bandpass(x, fs, 8.0, 30.0)
    xc = _bandpass(x, fs, 1.0, 30.0)
    rows = []
    for k in range(OFFSETS):
        rng = np.random.default_rng(SEED + 50000 + ri * 100 + k)
        start = int(rng.integers(0, int((len(x) / fs - 48 * 12 - 5) * fs)))
        base = {"recording": os.path.basename(path), "diagnosis": diagnosis(path), "offset": k}
        cmd = dn.alternating(48, rng)
        t = start + np.arange(48) * int(12 * fs)
        clip = int(2 * fs)
        Fc = np.array([[np.log(np.mean(b[s + int(2 * fs) + j * clip: s + int(2 * fs) + (j + 1) * clip] ** 2, axis=0)
                         + 1e-12) for b in xb] for s in t for j in range(5)]).reshape(48 * 5, -1)
        res = {"Pcol_a": dn.columbia(Fc, cmd, rng, 500, grouped=False),
               "Pcol_b": dn.columbia(Fc, cmd, rng, 500, grouped=True)}
        y, blocks = dn.schedule("block", rng, block=6)
        st = start + np.arange(48) * int(TRIAL_S * fs)
        F = np.log(np.array([np.mean(xf[s + int(0.5 * fs): s + int(4.5 * fs)] ** 2, axis=0) for s in st]) + 1e-12)
        res["Pcruse"] = dn.cruse(F, y, blocks)
        c16 = dn.alternating(16, rng)
        t16 = start + np.arange(16) * int(12 * fs)
        P = []
        for s in t16:
            f, pxx = welch(xc[s + int(3 * fs): s + int(12 * fs)], fs=fs, nperseg=int(fs), axis=0)
            P.append(np.log(pxx[(f >= 1) & (f <= 30)].T + 1e-12))
        P = np.array(P)
        res["Pcornell"] = dn.cornell(P[c16 == 1], P[c16 == 0])
        for design, o in res.items():
            rows.append({**base, "design": design, "call": bool(o["call"]), "p": o.get("p", o.get("min_p")),
                         "acc": o.get("acc"), "auc": o.get("auc"), "transition": None})
    return rows


def any_call_in_sessions(g, n=5, draws=2000, seed=SEED):
    """Mean over recordings of P(at least one call in n sessions), sessions drawn without replacement from that
    recording's offsets."""
    rng = np.random.default_rng(seed)
    out = []
    for _, r in g.groupby("recording"):
        c = r.call.to_numpy(bool)
        out.append(np.mean([c[rng.choice(len(c), min(n, len(c)), replace=False)].any() for _ in range(draws)]))
    return float(np.mean(out))


def per_patient(g):
    rates = g.groupby("recording").call.mean()
    return {"median_patient_rate": float(rates.median()), "max_patient_rate": float(rates.max()),
            "share_patients_any_call": float((rates > 0).mean()),
            "share_of_calls_from_top5_patients": float(rates.sort_values(ascending=False).head(5).sum()
                                                       / max(rates.sum(), 1e-12))}


def summarize(df):
    out = {"n_recordings": int(df.recording.nunique()),
           "diagnoses": df.groupby("diagnosis").recording.nunique().to_dict(), "designs": {}}
    for design, g in df.groupby("design"):
        rate, ci, draws = ks.cluster_rate(g.call, g.recording)
        out["designs"][design] = {"rate": rate, "ci": ci,
                                  "p_above_5pct": float(np.mean(draws <= 0.05)),
                                  "by_diagnosis": g.groupby("diagnosis").call.mean().to_dict(),
                                  "by_transition": g.dropna(subset=["transition"]).groupby("transition")
                                  .call.mean().to_dict()}
        out["designs"][design]["per_patient"] = per_patient(g)
        out["designs"][design]["p_any_call_5_sessions"] = any_call_in_sessions(g)
        col = "lower_delta_half" if "lower_delta_half" in g else "wake_like"
        if col in g and g[col].notna().any():
            w = g[g[col].fillna(False).astype(bool)]
            if len(w):
                wr, wci, _ = ks.cluster_rate(w.call, w.recording)
                out["designs"][design]["lower_delta_half"] = {"rate": wr, "ci": wci, "n_runs": int(len(w))}
        if g.acc.notna().any():
            out["designs"][design]["acc_ge_70"] = float((g.acc >= 0.70).mean())
            out["designs"][design]["acc_ge_80"] = float((g.acc >= 0.80).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--published", action="store_true")
    ap.add_argument("--tag", default="", help="declared rerun: tagged outputs beside the originals (tags.py)")
    a = ap.parse_args()
    shards, name = (f"{SHARDS}_published", "doc_null_published") if a.published else (SHARDS, "doc_null")
    shards = tags.name(shards, a.tag)
    if a.merge:
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{shards}/part*.parquet"))])
        df.to_parquet(tags.name(f"{T}/{name}.parquet", a.tag), index=False)
        s = summarize(df)
        json.dump(s, open(tags.name(f"{T}/{name}.json", a.tag), "w"), indent=2, default=float)
        print(json.dumps(s, indent=1, default=float))
        return
    prior = None if a.published else prior_model()
    rows = []
    for ri, path in indexed_recordings():
        if ri % a.nshards == a.shard:
            rows += run_recording_published(ri, path) if a.published else run_recording(ri, path, prior)
            print(ri, os.path.basename(path), flush=True)
    os.makedirs(shards, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f"{shards}/part{a.shard:02d}.parquet", index=False)


if __name__ == "__main__":
    main()
