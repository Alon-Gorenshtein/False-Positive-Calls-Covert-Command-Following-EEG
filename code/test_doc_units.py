import importlib
import numpy as np

nul = importlib.import_module("91_doc_null")


def test_cue_logpower_is_scale_equivariant_for_volt_inputs():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 2e-5, (6, 4, 321))
    a = nul.cue_logpower(X, 160.0)
    b = nul.cue_logpower(10 * X, 160.0)
    assert np.allclose(b - a, np.log(100.0), atol=1e-6)


def test_cue_logpower_filters_along_time_not_trials():
    rng = np.random.default_rng(1)
    t = np.arange(321) / 160.0
    X = rng.normal(0, 1e-6, (40, 4, 321))
    X[:20] += 2e-5 * np.sin(2 * np.pi * 12 * t)
    F = nul.cue_logpower(X, 160.0)
    assert F[:20, 0].mean() - F[20:, 0].mean() > 3.0


def test_trial_features_filter_along_time_and_separate_classes():
    import doc_null as dn
    fs, rng = 100.0, np.random.default_rng(3)
    x = rng.normal(0, 1, (int(40 * 10 * fs), 4))
    starts = np.arange(40) * int(10 * fs)
    t = np.arange(int(4 * fs)) / fs
    for s in starts[:20]:
        x[s + int(0.5 * fs): s + int(4.5 * fs), 0] += 5 * np.sin(2 * np.pi * 12 * t)
    F = dn.trial_features(x, fs, starts)
    assert F[:20, 0].mean() - F[20:, 0].mean() > 2.0 and abs(F[:20, 1].mean() - F[20:, 1].mean()) < 0.3


def test_null_bandpass_keeps_in_band_sinusoid():
    fs = 100.0
    t = np.arange(int(60 * fs)) / fs
    x = np.c_[np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 45 * t)]
    y = nul._bandpass(x, fs, 8.0, 30.0)
    assert y[:, 0].std() > 0.6 and y[:, 1].std() < 0.05
