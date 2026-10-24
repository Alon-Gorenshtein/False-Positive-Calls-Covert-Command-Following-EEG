"""M1 check: are block-design false calls the recording's own slow fluctuations of the trial feature series?
M1 check (committed before the run): are block-design false calls the
recording's own slow fluctuations of the trial feature series?
- Recordings, seed indices and run starts of 91_doc_null.py (41 pDoC PSGs). The feature series (F3, F4, C3, C4; 8-30 Hz
  log power 0.5-4.5 s after each trial start, doc_null.trial_features) is cut on ONE 10-s grid over the whole
  recording; a run is the 48 grid trials from the grid point at or before 91's start sample.
- Series: observed; multivariate phase-randomised surrogates (one set of random phases shared by the 4 channels, 10
  per recording); per-channel AR(p) residuals, p by AIC <= 10; multivariate IAAFT surrogates (10 per recording).
- B2 (doc_null.b2, 1000 permutations) at block lengths 2, 4, 8, 12. A run's label schedule, folds and permutations at
  one block length are the same for every series.
- R = (rate(phase-randomised) - 0.05) / (rate(observed) - 0.05) over b8 + b12: R >= 0.5 keeps M1 as the primary
  hypothesis, R < 0.5 drops it as primary.
-> output/tables/m1_check.parquet (per run), m1_check.json
python 117_m1_check.py [--limit N --out DIR]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import warnings
import mne
import numpy as np
import pandas as pd
import doc_null as dn
import kit_stats as ks

warnings.filterwarnings("ignore")
m91 = importlib.import_module("91_doc_null")
NOGO_SEED = 20261014
BLOCKS = (2, 4, 8, 12)
NOGO_BLOCKS = (8, 12)
N_SURR = 10
DRAWS = 2000
KEEP_R = 0.5
MIN_EXCESS = 0.01
SUPPORT_MAX = 0.06


def phase_randomise(X, rng):
    """Multivariate Fourier phase randomisation (Prichard & Theiler 1994) of a (time, channel) series: one random
    phase per frequency, shared by the channels, so every auto- and cross-spectrum is kept. DC (the mean) and the
    Nyquist bin keep their phase so the surrogate stays real."""
    X = np.asarray(X, float)
    n = len(X)
    ph = rng.uniform(0, 2 * np.pi, n // 2 + 1)
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.fft.rfft(X, axis=0) * np.exp(1j * ph)[:, None], n=n, axis=0)


IAAFT_MAX_ITER = 1000


def iaaft(X, rng, max_iter=IAAFT_MAX_ITER):
    """Multivariate IAAFT (Schreiber & Schmitz 1996, 2000) of a (time, channel) series: each channel keeps its exact
    values and, approximately, its power spectrum; at each frequency one phase rotation, shared by the channels and
    chosen to stay closest to the current iterate, keeps the cross-spectrum. Starts from one random shuffle of the
    time points, alternates spectrum and rank-order steps until the ranks stop changing, and ends on the rank-order
    step (exact values). Returns (surrogate, iterations)."""
    X = np.asarray(X, float)
    n = len(X)
    A = np.fft.rfft(X, axis=0)
    srt = np.sort(X, axis=0)
    S = X[rng.permutation(n)]
    prev = None
    for it in range(1, max_iter + 1):
        rot = np.exp(1j * np.angle(np.sum(np.fft.rfft(S, axis=0) * np.conj(A), axis=1)))
        r = np.argsort(np.argsort(np.fft.irfft(A * rot[:, None], n=n, axis=0), axis=0), axis=0)
        S = np.take_along_axis(srt, r, axis=0)
        if prev is not None and np.array_equal(r, prev):
            break
        prev = r
    return S, it


AR_MAX_ORDER = 10


def ar_residuals(x, max_p=AR_MAX_ORDER):
    """Residuals of an AR(p) fit to one demeaned series, p in 0..max_p by AIC (OLS, all orders on the common sample
    t >= max_p). Presample values are set to the mean, so the residual series keeps the input's length and grid.
    Returns (residuals, p)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    lags = np.column_stack([x[max_p - i: n - i] for i in range(1, max_p + 1)])
    best = None
    for p in range(max_p + 1):
        a = np.linalg.lstsq(lags[:, :p], x[max_p:], rcond=None)[0] if p else np.zeros(0)
        aic = (n - max_p) * np.log(np.mean((x[max_p:] - lags[:, :p] @ a) ** 2)) + 2 * p
        if best is None or aic < best[0]:
            best = (aic, p, a)
    _, p, a = best
    xp = np.r_[np.zeros(p), x]
    return x - sum(a[i] * xp[p - 1 - i: p - 1 - i + n] for i in range(p)), p


def prewhiten(X):
    """Per-channel AR(p) residuals of a (time, channel) series. Returns (residuals, orders)."""
    out = [ar_residuals(c) for c in np.asarray(X, float).T]
    return np.column_stack([w for w, _ in out]), [p for _, p in out]


def run_start(ri, k, n_samples, fs):
    """91_doc_null.run_recording's seed and start sample of offset k of recording ri (its seed index). Returns
    (rng, start); the rng goes on to draw the label schedules."""
    rng = np.random.default_rng(m91.SEED + ri * 100 + k)
    return rng, int(rng.integers(0, int((n_samples / fs - m91.TRIAL_S * m91.RUN_TRIALS - 5) * fs)))


def schedules(rng):
    """Block schedule per block length, drawn after the start: b4 then b8 first, as 91's B2 and B2long."""
    ys = {blk: dn.schedule("block", rng, block=blk)[0] for blk in (4, 8, 2, 12)}
    return {blk: ys[blk] for blk in BLOCKS}


def _step(fs):
    return int(m91.TRIAL_S * fs)


def n_grid(n_samples, fs):
    """Number of 10-s grid trials, from sample 0, whose 0.5-4.5 s window lies inside the recording."""
    return (n_samples - int(4.5 * fs)) // _step(fs) + 1


def grid_index(start, fs):
    """The grid trial at or before a start sample."""
    return start // _step(fs)


def grid_series(x, fs):
    """Trial log band power (doc_null.trial_features) on the 10-s grid over the whole recording: (grid trials, 4)."""
    return dn.trial_features(x, fs, np.arange(n_grid(len(x), fs)) * _step(fs))


def load(path):
    """F3, F4, C3, C4 in microvolts (samples, 4) and the sampling rate, read exactly as 91_doc_null.run_recording
    reads them: all channels preloaded, so channels stored at a lower rate are resampled to the file's highest rate
    (include=CH would change the rate; a non-preloaded full read trips an mne assertion on mixed-rate files). Only
    the signals and the sampling rate are used; no other header field is touched."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return raw.get_data(picks=m91.CH).T * 1e6, float(raw.info["sfreq"])


def _band_err(X, S, nbins=25):
    """Largest relative error of S's power, per channel, in nbins equal-width frequency bands (DC excluded)."""
    def bands(Z):
        P = np.abs(np.fft.rfft(Z - Z.mean(0), axis=0))[1:] ** 2
        return np.array([b.sum(0) for b in np.array_split(P, nbins)])
    return float(np.max(np.abs(bands(S) / bands(X) - 1)))


def _lag1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def series_set(F, si, n_surr):
    """The four series of one recording, {name: [(grid trials, 4) arrays]}, and diagnostics. Phase-randomised and
    IAAFT surrogates each draw from their own generator seeded NOGO_SEED + seed index."""
    rp, rq = np.random.default_rng(NOGO_SEED + si), np.random.default_rng(NOGO_SEED + si)
    W, orders = prewhiten(F)
    ia = [iaaft(F, rq) for _ in range(n_surr)]
    S = {"observed": [F], "phase": [phase_randomise(F, rp) for _ in range(n_surr)], "prewhitened": [W],
         "iaaft": [a for a, _ in ia]}
    diag = {"ar_orders": orders, "prewhitened_lag1": [_lag1(c) for c in W.T],
            "iaaft_iters": [it for _, it in ia], "iaaft_max_band_err": max(_band_err(F, a) for a, _ in ia)}
    return S, diag


def b2_rows(S, runs, si, n_perm=m91.N_PERM):
    """B2 calls of every series of S on each run, runs = [(offset k, first grid index, {block: labels})]. The test
    generator of a run and block length is seeded by (NOGO_SEED, si, k, block), so every series gets the same folds
    and permutations."""
    rows = []
    for k, g0, ys in runs:
        for blk, y in ys.items():
            for series, arrs in S.items():
                for j, A in enumerate(arrs):
                    o = dn.b2(A[g0:g0 + m91.RUN_TRIALS], y, np.random.default_rng([NOGO_SEED, si, k, blk]), n_perm)
                    rows.append({"offset": k, "grid_index": g0, "block": blk, "series": series, "surrogate": j,
                                 "first_label": int(y[0]), "call": bool(o["call"]), "p": o["p"], "auc": o["auc"]})
    return rows


def recording_rows(si, path, x, fs, n_surr=N_SURR, n_perm=m91.N_PERM, offsets=m91.OFFSETS):
    """B2 calls of every series, offset and block length of one recording (seed index si), on 91's runs."""
    F = grid_series(x, fs)
    S, diag = series_set(F, si, n_surr)
    name = os.path.basename(path)
    runs = []
    for k in range(offsets):
        rng, start = run_start(si, k, len(x), fs)
        runs.append((k, grid_index(start, fs), schedules(rng)))
    base = {"recording": name, "diagnosis": m91.diagnosis(path), "seed_index": si}
    rows = [{**base, **r} for r in b2_rows(S, runs, si, n_perm)]
    diag.update({"recording": name, "seed_index": si, "n_grid": int(len(F)),
                 "n_floor_trials": int(np.sum(F < np.log(1e-12) + 1))})
    return rows, diag


def summarize(df, diags=()):
    rates, pooled = {}, {}
    for (series, blk), g in df.groupby(["series", "block"]):
        r, ci, _ = ks.cluster_rate(g.call, g.recording, DRAWS, NOGO_SEED)
        rates.setdefault(series, {})[f"b{blk}"] = {"rate": r, "ci": ci, "n_runs": int(len(g))}
    ids = {}
    for series, g in df[df.block.isin(NOGO_BLOCKS)].groupby("series"):
        pooled[series] = ks.cluster_rate(g.call, g.recording, DRAWS, NOGO_SEED)
        ids[series] = tuple(np.unique(g.recording))
    assert ids["observed"] == ids["phase"]
    (ro, _, do), (rp, _, dp) = pooled["observed"], pooled["phase"]
    R, Rd = (rp - dn.ALPHA) / (ro - dn.ALPHA), (dp - dn.ALPHA) / (do - dn.ALPHA)
    verdict = "uninformative" if ro - dn.ALPHA <= MIN_EXCESS else ("M1 kept" if R >= KEEP_R else "M1 dropped")
    text = {"uninformative": "E(observed) at b8 + b12 <= 0.01: the test is uninformative",
            "M1 kept": "R >= 0.5: M1 stays the primary mechanism hypothesis for second stage",
            "M1 dropped": "R < 0.5: M1 is dropped as primary; muscle artifact (M3) is promoted to primary, M1 secondary"}
    pw = rates.get("prewhitened", {})
    return {"n_recordings": int(df.recording.nunique()), "alpha": dn.ALPHA,
            "rates": rates,
            "pooled_b8_b12": {s: {"rate": r, "ci": ci} for s, (r, ci, _) in pooled.items()},
            "nogo_rule": {"observed_rate_b8_b12": ro, "phase_rate_b8_b12": rp, "E_observed": ro - dn.ALPHA,
                          "E_phase": rp - dn.ALPHA, "R": R,
                          "R_ci": [float(np.quantile(Rd, 0.025)), float(np.quantile(Rd, 0.975))],
                          "verdict": verdict, "verdict_text": text[verdict]},
            "support_prewhitened_le_6pct": (bool(all(pw[f"b{b}"]["rate"] <= SUPPORT_MAX for b in NOGO_BLOCKS))
                                            if pw else None),
            "recordings": list(diags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N recordings only (smoke test; use with --out)")
    ap.add_argument("--out", default=m91.T)
    a = ap.parse_args()
    rows, diags = [], []
    for si, path in m91.indexed_recordings()[: a.limit or None]:
        x, fs = load(path)
        r, d = recording_rows(si, path, x, fs)
        del x
        rows += r
        diags.append(d)
        print(si, os.path.basename(path), d["n_grid"], d["ar_orders"], max(d["iaaft_iters"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{a.out}/m1_check.parquet", index=False)
    s = summarize(df, diags)
    json.dump(s, open(f"{a.out}/m1_check.json", "w"), indent=2, default=float)
    print(json.dumps({k: s[k] for k in ("pooled_b8_b12", "nogo_rule", "support_prewhitened_le_6pct")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
