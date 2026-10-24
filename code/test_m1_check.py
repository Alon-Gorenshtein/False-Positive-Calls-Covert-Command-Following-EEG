import importlib
import numpy as np

mk = importlib.import_module("117_m1_check")


def _correlated_ar(n=2000, seed=0):
    """Four channels sharing a slow AR(1) component (lag-1 0.95) plus channel noise: low-frequency power and strong
    cross-channel coherence, like the trial log-power series."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.95 * z[t - 1] + rng.normal()
    return 2.0 + np.outer(z, [1.0, 0.8, -0.5, 0.3]) + rng.normal(0, 0.5, (n, 4))


def test_phase_surrogate_keeps_each_channel_periodogram():
    X = _correlated_ar()
    S = mk.phase_randomise(X, np.random.default_rng(1))
    assert S.shape == X.shape
    PX, PS = np.abs(np.fft.rfft(X, axis=0)) ** 2, np.abs(np.fft.rfft(S, axis=0)) ** 2
    assert np.allclose(PS, PX, rtol=1e-8, atol=1e-8 * PX.max())
    assert np.isclose(S.mean(), X.mean())
    assert not np.allclose(S, X)


def test_phase_surrogate_keeps_the_cross_spectrum():
    X = _correlated_ar()
    S = mk.phase_randomise(X, np.random.default_rng(2))
    FX, FS = np.fft.rfft(X, axis=0), np.fft.rfft(S, axis=0)
    for i in range(4):
        for j in range(i + 1, 4):
            cx, cs = FX[:, i] * np.conj(FX[:, j]), FS[:, i] * np.conj(FS[:, j])
            assert np.allclose(cs, cx, rtol=1e-8, atol=1e-8 * np.abs(cx).max())
    assert np.allclose(np.corrcoef(S.T), np.corrcoef(X.T), atol=1e-8)


def test_phase_surrogates_are_seeded_and_differ_from_each_other():
    X = _correlated_ar()
    a = mk.phase_randomise(X, np.random.default_rng(3))
    b = mk.phase_randomise(X, np.random.default_rng(3))
    c = mk.phase_randomise(X, np.random.default_rng(4))
    assert np.array_equal(a, b) and not np.allclose(a, c)


def _binned_power(X, nbins=25):
    P = np.abs(np.fft.rfft(X - X.mean(0), axis=0))[1:] ** 2
    return np.array([b.sum(0) for b in np.array_split(P, nbins)])


def _log_power(n=2000, seed=0):
    """Log band-power-like series: a shared slow AR(1) component plus log-gamma (left-skewed) power noise."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.95 * z[t - 1] + rng.normal()
    return 2.0 + np.outer(z / 3, [1.0, 0.9, 0.8, 0.7]) + np.log(rng.gamma(5.0, 1.0, (n, 4)))


def test_iaaft_keeps_sorted_values_and_approximately_the_spectrum():
    X = _log_power(seed=5)
    S, iters = mk.iaaft(X, np.random.default_rng(6))
    assert S.shape == X.shape and 1 <= iters < mk.IAAFT_MAX_ITER
    assert np.array_equal(np.sort(S, axis=0), np.sort(X, axis=0))
    assert np.all(np.abs(_binned_power(S) / _binned_power(X) - 1) < 0.10)
    assert np.allclose(np.corrcoef(S.T), np.corrcoef(X.T), atol=0.05)
    assert not np.allclose(S, X)


def test_iaaft_one_strongly_skewed_channel():
    X = np.exp(_correlated_ar(seed=5)[:, [2]] / 3)
    S, iters = mk.iaaft(X, np.random.default_rng(6))
    assert iters < mk.IAAFT_MAX_ITER
    assert np.array_equal(np.sort(S, axis=0), np.sort(X, axis=0))
    assert np.all(np.abs(_binned_power(S) / _binned_power(X) - 1) < 0.10)


def test_iaaft_is_seeded():
    X = np.exp(_correlated_ar(seed=7) / 3)
    a, _ = mk.iaaft(X, np.random.default_rng(8))
    b, _ = mk.iaaft(X, np.random.default_rng(8))
    c, _ = mk.iaaft(X, np.random.default_rng(9))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _lag1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def test_prewhitened_series_has_near_zero_lag1_autocorrelation():
    X = _log_power(seed=3) + np.cumsum(np.random.default_rng(4).normal(0, 0.05, (2000, 4)), axis=0)
    assert min(_lag1(X[:, c]) for c in range(4)) > 0.5
    W, orders = mk.prewhiten(X)
    assert W.shape == X.shape and np.all(np.isfinite(W))
    assert len(orders) == 4 and all(1 <= p <= mk.AR_MAX_ORDER for p in orders)
    assert all(abs(_lag1(W[:, c])) < 0.05 for c in range(4))


def test_ar_order_by_aic_recovers_an_ar2():
    rng = np.random.default_rng(10)
    x = np.zeros(5000)
    for t in range(2, 5000):
        x[t] = 1.2 * x[t - 1] - 0.4 * x[t - 2] + rng.normal()
    w, p = mk.ar_residuals(x)
    assert 2 <= p <= 4 and abs(_lag1(w)) < 0.03
    assert np.allclose(w[100:], (x - x.mean())[100:] - 1.2 * (x - x.mean())[99:-1] + 0.4 * (x - x.mean())[98:-2],
                       atol=0.2)


def test_run_cutting_matches_91_offsets_seeds_and_schedules(monkeypatch):
    """Runs 91_doc_null.run_recording on a synthetic recording whose trial feature encodes the trial start sample,
    with the tests stubbed to record their inputs; the starts and the b4 (B2) and b8 (B2long) schedules 91 used must
    equal this script's."""
    m91 = mk.m91
    fs, n = 256.0, int(256 * 3600 * 1.5)
    x = np.column_stack([np.sqrt(np.arange(n, dtype=float) + 1)] * 4)

    class Raw:
        info = {"sfreq": fs}

        def get_data(self, picks):
            return x.T / 1e6

    seen = []

    def b2(F, y, rng, n_perm=1000):
        seen.append((F, y))
        return {"call": False}

    def stub(*a, **k):
        return {"call": False}

    monkeypatch.setattr(m91.mne.io, "read_raw_edf", lambda *a, **k: Raw())
    monkeypatch.setattr(m91, "_bandpass", lambda x, fs, lo, hi: x)
    for name, f in (("b1", stub), ("b2", b2), ("b3", stub), ("r_test", stub),
                    ("b2alt", lambda *a, **k: {"a_epoch": {"call": False}, "b_period": {"call": False}})):
        monkeypatch.setattr(m91.dn, name, f)
    ri = 7
    m91.run_recording(ri, "Patient_8_VS.edf", None)
    assert len(seen) == 5 * m91.OFFSETS
    a, b = int(0.5 * fs), int(4.5 * fs)
    step = int(m91.TRIAL_S * fs)
    for k in range(m91.OFFSETS):
        (F4, y4), (_, y8) = seen[5 * k], seen[5 * k + 1]
        start91 = int(round(np.exp(F4[0, 0]) - (a + b + 1) / 2))
        rng, start = mk.run_start(ri, k, n, fs)
        assert start == start91
        ys = mk.schedules(rng)
        assert sorted(ys) == [2, 4, 8, 12]
        assert np.array_equal(ys[4], y4) and np.array_equal(ys[8], y8)
        assert all(len(y) == m91.RUN_TRIALS and y.sum() == m91.RUN_TRIALS // 2 for y in ys.values())
        g0 = mk.grid_index(start, fs)
        assert g0 * step <= start < (g0 + 1) * step
        assert g0 + m91.RUN_TRIALS <= mk.n_grid(n, fs)


def test_grid_series_is_doc_null_trial_features_on_the_whole_recording_grid():
    fs = 256.0
    x = np.random.default_rng(0).normal(size=(int(fs * 600) + 100, 4))
    F = mk.grid_series(x, fs)
    step = int(10 * fs)
    assert len(F) == mk.n_grid(len(x), fs) == 60
    assert np.allclose(F, mk.dn.trial_features(x, fs, np.arange(60) * step))


def _synthetic_recording(minutes=60, fs=256.0, seed=0):
    """White EEG with an 8-30 Hz amplitude that drifts slowly (minutes), the same drift on all four channels."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fs)
    drift = np.exp(0.5 * np.sin(2 * np.pi * np.arange(n) / (fs * 600)))
    return rng.normal(size=(n, 4)) * drift[:, None], fs


def test_recording_rows_pair_every_series_on_one_schedule_and_one_permutation_set():
    x, fs = _synthetic_recording()
    rows, diag = mk.recording_rows(3, "Patient_x_VS.edf", x, fs, n_surr=2, n_perm=50, offsets=2)
    import pandas as pd
    df = pd.DataFrame(rows)
    assert len(df) == 2 * 4 * (1 + 2 + 1 + 2)
    assert df.groupby("series").size().to_dict() == {"iaaft": 16, "observed": 8, "phase": 16, "prewhitened": 8}
    assert sorted(df.block.unique()) == [2, 4, 8, 12] and set(df.seed_index) == {3}
    assert (df.groupby(["offset", "block"]).first_label.nunique() == 1).all()
    assert (df.groupby(["offset"]).grid_index.nunique() == 1).all()
    assert len(diag["ar_orders"]) == 4 and len(diag["iaaft_iters"]) == 2
    assert df.call.dtype == bool and df.p.between(0, 1).all()


def test_identical_series_get_identical_p_values(monkeypatch):
    """Every series of a run and block length goes through the same folds and permutations: if every series is the
    observed one, every p is the observed p."""
    x, fs = _synthetic_recording(seed=1)
    real = mk.series_set

    def same(F, si, n_surr):
        S, diag = real(F, si, n_surr)
        return {k: [F] * len(v) for k, v in S.items()}, diag

    monkeypatch.setattr(mk, "series_set", same)
    import pandas as pd
    df = pd.DataFrame(mk.recording_rows(5, "Patient_y_MCS+.edf", x, fs, n_surr=2, n_perm=50, offsets=2)[0])
    assert (df.groupby(["offset", "block"]).p.nunique() == 1).all()
    assert (df.groupby(["offset", "block"]).auc.nunique() == 1).all()


def _df(rates, n_rec=20, runs=200, seed=0):
    """Per-run rows with the given call rate per (series, block): rates = {(series, block): rate}."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    out = []
    for (series, blk), r in rates.items():
        for i in range(n_rec):
            calls = np.zeros(runs, bool)
            calls[: int(round(r * runs))] = True
            out.append(pd.DataFrame({"recording": f"r{i}", "series": series, "block": blk,
                                     "call": rng.permutation(calls)}))
    return pd.concat(out, ignore_index=True)


def test_nogo_rule():
    base = {("observed", 2): 0.03, ("observed", 4): 0.05, ("observed", 8): 0.20, ("observed", 12): 0.30,
            ("prewhitened", 8): 0.05, ("prewhitened", 12): 0.06, ("iaaft", 8): 0.2, ("iaaft", 12): 0.2}
    keep = mk.summarize(_df({**base, ("phase", 8): 0.15, ("phase", 12): 0.20}))
    assert np.isclose(keep["nogo_rule"]["observed_rate_b8_b12"], 0.25)
    assert np.isclose(keep["nogo_rule"]["R"], (0.175 - 0.05) / (0.25 - 0.05))
    assert keep["nogo_rule"]["verdict"] == "M1 kept"
    assert keep["nogo_rule"]["R_ci"][0] <= keep["nogo_rule"]["R"] <= keep["nogo_rule"]["R_ci"][1]
    assert keep["support_prewhitened_le_6pct"] is True
    assert set(keep["rates"]["observed"]) == {"b2", "b4", "b8", "b12"}
    nogoed = mk.summarize(_df({**base, ("phase", 8): 0.10, ("phase", 12): 0.12}))
    assert nogoed["nogo_rule"]["R"] < 0.5 and nogoed["nogo_rule"]["verdict"] == "M1 dropped"
    flat = mk.summarize(_df({**base, ("observed", 8): 0.05, ("observed", 12): 0.06, ("phase", 8): 0.05,
                             ("phase", 12): 0.05}))
    assert flat["nogo_rule"]["verdict"] == "uninformative"


def test_load_matches_91_loading_on_a_real_recording():
    import os
    import mne
    import pytest
    recs = mk.m91.indexed_recordings()
    if not recs:
        pytest.skip("pDoC recordings not on disk")
    path = min((p for _, p in recs), key=os.path.getsize)
    x, fs = mk.load(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    assert fs == raw.info["sfreq"]
    assert np.array_equal(x, raw.get_data(picks=mk.m91.CH).T * 1e6)


def test_b2_rows_runs_every_series_on_the_given_runs_with_shared_tests():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 4))
    S = {"observed": [F], "phase": [F, F]}
    ys = mk.schedules(np.random.default_rng(1))
    rows = mk.b2_rows(S, [(0, 10, ys), (1, 120, ys)], si=7, n_perm=50)
    assert len(rows) == 2 * 4 * 3
    assert [r["grid_index"] for r in rows[::12]] == [10, 120]
    import pandas as pd
    df = pd.DataFrame(rows)
    assert (df.groupby(["offset", "block"]).p.nunique() == 1).all()
    o = mk.dn.b2(F[120:168], ys[8], np.random.default_rng([mk.NOGO_SEED, 7, 1, 8]), 50)
    assert df[(df.offset == 1) & (df.block == 8)].p.iloc[0] == o["p"]
