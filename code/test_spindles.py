"""Tests for code/123_spindles.py (part B5: spindle-like activity and false calls):
synthetic fixtures and the DEV split only. Every guard is also run against a deliberately broken input and the test
checks that the guard catches it. Dev rows are read for structure only (no rate is computed or printed).
cd code && /usr/bin/python3 -m pytest test_spindles.py -q"""
import hashlib
import importlib
import os
import shutil

import numpy as np
import pandas as pd
import pytest

sp = importlib.import_module("123_spindles")
mg = importlib.import_module("115_benchmark_merge")

FS = 128.0



def pink(n, rng, chi=1.0):
    f = np.fft.rfftfreq(n)
    a = np.zeros(len(f))
    a[1:] = f[1:] ** (-chi / 2)
    x = np.fft.irfft(a * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))), n=n)
    return x / x.std()


def synth(T, onsets, rng, amp=25.0, bg=10.0, dur=1.5, fs=FS):
    """Pink background (bg uV SD) plus Hann-tapered 13-Hz bursts of dur s at the onsets."""
    n = int(T * fs)
    x = bg * pink(n, rng)
    t = np.arange(int(dur * fs)) / fs
    burst = amp * np.hanning(len(t)) * np.sin(2 * np.pi * 13 * t)
    for o in onsets:
        i = int(o * fs)
        if i + len(burst) < n:
            x[i:i + len(burst)] += burst
    return x


def clusters(starts, k=5, gap=1.6):
    return np.concatenate([s + np.arange(k) * gap for s in starts])



@pytest.mark.parametrize("rate", [2, 4, 8])
def test_a7_density_recovers_injected_rate(rate):
    T = 600
    k = int(rate * T / 60)
    on = np.arange(k) * (T / k) + 0.5
    rng = np.random.default_rng(rate)
    f = sp.spindle_features(synth(T, on, rng), synth(T, on, rng), FS)
    assert f["reason"] is None
    assert f["a7_density"] == pytest.approx(rate, rel=0.15)


def test_a7_density_is_near_zero_without_spindles():
    rng = np.random.default_rng(3)
    f = sp.spindle_features(synth(600, [], rng), synth(600, [], rng), FS)
    assert f["a7_density"] < 1.0


def test_a7_runs_at_its_published_operating_point_not_rate_matched():
    """The calibrated A7 is the published one (scale 1.0, never rate-matched): same events as the uncalibrated call."""
    det = sp.load_detectors()
    cal = det.load_calibration()["a7_lacourse"]
    assert cal["param"] == "scale" and cal["value"] == 1.0
    rng = np.random.default_rng(4)
    x = synth(300, np.arange(20) * 14 + 1.0, rng)
    a = det.calibrated_detectors()["a7_lacourse"](x, FS)
    b = det.a7_lacourse(x, FS)
    np.testing.assert_array_equal(a, b)
    assert len(b) > 0


def test_a7_is_called_with_the_published_thresholds_only(monkeypatch):
    """spindle_features passes no threshold of its own to A7: the published z 1.25 / 1.6 / 1.3 and r 0.69."""
    import inspect
    det = sp.load_detectors()
    sig = inspect.signature(det.a7_lacourse).parameters
    assert [sig[k].default for k in ("scale", "z_abs", "z_rel", "z_cov", "corr_min")] == [1.0, 1.25, 1.6, 1.3, 0.69]
    seen = []
    orig = det.a7_lacourse

    def spy(*a, **k):
        seen.append(k)
        return orig(*a, **k)
    monkeypatch.setattr(det, "a7_lacourse", spy)
    rng = np.random.default_rng(5)
    sp.spindle_features(synth(120, [], rng), synth(120, [], rng), FS)
    assert len(seen) == 2 and all(k == {} for k in seen)


def test_density_is_per_minute_of_the_window_and_averaged_over_channels():
    rng = np.random.default_rng(5)
    T = 300
    on = np.arange(20) * 14 + 1.0
    x3, x4 = synth(T, on, rng), synth(T, on, rng)
    det = sp.load_detectors()
    d3 = len(det.a7_lacourse(x3, FS)) / (T / 60)
    d4 = len(det.a7_lacourse(x4, FS)) / (T / 60)
    f = sp.spindle_features(x3, x4, FS)
    assert f["a7_density"] == pytest.approx((d3 + d4) / 2)


def test_five_other_detectors_reported_as_densities():
    rng = np.random.default_rng(6)
    on = np.arange(20) * 14 + 1.0
    f = sp.spindle_features(synth(300, on, rng), synth(300, on, rng), FS)
    assert set(sp.OTHER_DETECTORS) == {"fixed_amplitude", "relative_threshold", "rms_ferrarelli", "wavelet",
                                       "sigma_index"}
    for name in sp.OTHER_DETECTORS:
        v = f["dens_" + name]
        assert np.isfinite(v) and v >= 0



def test_sigma_slow_rel_is_higher_when_spindles_recur_every_50_s_than_at_random_times():
    T, k = 900, 18
    per, rnd = [], []
    for s in range(8):
        rng = np.random.default_rng(100 + s)
        on_p = clusters(np.arange(k) * 50 + rng.uniform(0, 10))
        on_r = clusters(rng.uniform(0, T - 12, k))
        per.append(sp.spindle_features(synth(T, on_p, rng), synth(T, on_p, rng), FS)["sigma_slow_rel"])
        rnd.append(sp.spindle_features(synth(T, on_r, rng), synth(T, on_r, rng), FS)["sigma_slow_rel"])
    assert np.mean(per) > 1.5 * np.mean(rnd)
    assert np.mean(np.array(per) > np.array(rnd)) >= 0.85


def test_sigma_slow_rel_tracks_the_0_02_hz_band_not_its_harmonic():
    """Recurrence every 50 s (0.02 Hz) beats recurrence every 25 s (0.04 Hz, the harmonic): the band is 0.015-0.025."""
    T, k = 900, 18
    at50, at25 = [], []
    for s in range(6):
        rng = np.random.default_rng(200 + s)
        o50 = clusters(np.arange(k) * 50 + 2.0, k=3)
        o25 = clusters(np.arange(2 * k) * 25 + 2.0, k=3)
        at50.append(sp.spindle_features(synth(T, o50, rng), synth(T, o50, rng), FS)["sigma_slow_rel"])
        at25.append(sp.spindle_features(synth(T, o25, rng), synth(T, o25, rng), FS)["sigma_slow_rel"])
    assert np.mean(at50) > 1.3 * np.mean(at25)


def test_sigma_slow_rel_is_na_not_zero_when_the_window_cannot_resolve_the_band():
    rng = np.random.default_rng(7)
    f = sp.spindle_features(synth(30, [], rng), synth(30, [], rng), FS)
    assert np.isnan(f["sigma_slow_rel"]) and f["reason_sigma"] == "slow_band_unresolved"
    assert f["reason"] is None and np.isfinite(f["a7_density"])


def test_window_shorter_than_10_s_has_no_features():
    rng = np.random.default_rng(7)
    f = sp.spindle_features(synth(5, [], rng), synth(5, [], rng), FS)
    assert f["reason"] == "window_too_short" and np.isnan(f["a7_density"]) and np.isnan(f["sigma_slow_rel"])


def test_flat_channel_is_na_not_zero():
    """Missing is not zero: a flat channel gives no density, not a density of 0."""
    rng = np.random.default_rng(8)
    x = synth(300, np.arange(20) * 14 + 1.0, rng)
    f = sp.spindle_features(x, np.zeros_like(x), FS)
    assert np.isnan(f["a7_density"]) and np.isnan(f["sigma_slow_rel"]) and f["reason"] == "flat_channel"


def test_non_finite_samples_are_na():
    rng = np.random.default_rng(9)
    x = synth(300, [], rng)
    y = x.copy()
    y[1000] = np.nan
    f = sp.spindle_features(x, y, FS)
    assert np.isnan(f["a7_density"]) and f["reason"] == "non_finite"



def test_vendored_files_match_the_pinned_sha256():
    assert sp.VENDOR_SHA256 == {
        "detectors.py": "19d15d9daa15a2c0cb505242856637c75ac1ddc97e881e7b994efe58d26038a9",
        "calibration.json": "e1ef26f96c2f2f87b8e1cf7707bf97ed2e7579b71ae182c77762ad9f62d5e9ec"}
    sp.verify_vendor()
    for name, want in sp.VENDOR_SHA256.items():
        with open(os.path.join(sp.VENDOR, name), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == want


@pytest.mark.parametrize("victim", ["detectors.py", "calibration.json"])
def test_a_modified_vendored_file_is_refused_and_never_executed(tmp_path, victim):
    d = tmp_path / "vendor_spindle"
    shutil.copytree(sp.VENDOR, d, ignore=shutil.ignore_patterns("__pycache__"))
    with open(d / victim, "a") as fh:
        fh.write("\n# tampered\n")
    marker = tmp_path / "ran"
    if victim == "detectors.py":
        with open(d / victim, "a") as fh:
            fh.write(f"open({str(marker)!r}, 'w').write('x')\n")
    with pytest.raises(RuntimeError, match="sha256"):
        sp.verify_vendor(str(d))
    with pytest.raises(RuntimeError, match="sha256"):
        sp.load_detectors(str(d))
    assert not marker.exists()


def test_spindle_features_refuses_to_run_when_the_pin_does_not_match(monkeypatch):
    monkeypatch.setattr(sp, "VENDOR_SHA256", {**sp.VENDOR_SHA256, "detectors.py": "0" * 64})
    sp._clear_cache()
    try:
        rng = np.random.default_rng(10)
        with pytest.raises(RuntimeError, match="sha256"):
            sp.spindle_features(synth(300, [], rng), synth(300, [], rng), FS)
    finally:
        monkeypatch.undo()
        sp._clear_cache()
    rng = np.random.default_rng(10)
    assert sp.spindle_features(synth(300, [], rng), synth(300, [], rng), FS)["reason"] is None



def test_analysis_split_refused_without_release():
    with pytest.raises(RuntimeError, match="release"):
        sp.guard_split("analysis", release=False)
    with pytest.raises(RuntimeError, match="release"):
        sp.run_features("analysis", release=False)
    with pytest.raises(RuntimeError, match="release"):
        sp.run_stats("analysis", release=False)
    with pytest.raises(SystemExit):
        sp.main(["--split", "analysis", "--stage", "features"])




def test_dev_split_needs_no_flag():
    sp.guard_split("dev", release=False)



def synth_frame(n_pat=300, seed=0, log_or_inf=np.log(1.6), log_or_oth=0.0, mediated=False):
    """Rows of three pipelines (inflated pa x2 variants, immune pb x1, deflated pc x1), one window per patient x
    pipeline shared by its variants. Calls depend on the window's measure only in the inflated design."""
    rng = np.random.default_rng(seed)
    rows = []
    spec = {"pa": (["v1", "v2"], "inflated", -1.0), "pb": (["v1"], "immune", -1.0),
            "pc": (["v1"], "deflated", -1.5)}
    strata = np.array(["normal_awake", "sleep", "mixed", "burst_suppression"])
    for p in range(n_pat):
        st = rng.choice(strata)
        for pipe, (variants, cls, a) in spec.items():
            m5 = rng.normal()
            m_x = rng.normal()
            for v in variants:
                lp = a + (log_or_inf if cls == "inflated" else log_or_oth) * (m_x if mediated else m5)
                rows.append(dict(patient=f"p{p}", uid=f"r{p}", pipe=pipe, variant=v, stratum=st,
                                 a7_density=m5 * 2 + 5, sigma_slow_rel=rng.normal() * 0.1 + 0.3,
                                 X=(m_x if cls == "inflated" else np.nan) if mediated else m5 + rng.normal() * 0.05,
                                 x_ref_class=cls, call=bool(rng.random() < 1 / (1 + np.exp(-lp)))))
    d = pd.DataFrame(rows)
    for name in sp.OTHER_DETECTORS:
        d["dens_" + name] = rng.normal(size=len(d)) + 3.0
    if mediated:
        d["a7_density"] = d["X"].fillna(0).to_numpy() * 2 + 5 + rng.normal(size=len(d)) * 0.05
    return d


NV = {"pa": 2, "pb": 1, "pc": 1}


def test_within_inflated_or_exceeds_one_and_other_designs_are_near_one():
    d = sp.prepare_frame(synth_frame(seed=1), NV)
    r = sp.fit_measure(d, "a7_density", B=200, name="t_a7")
    inf, oth = r["within_inflated"], r["other_designs"]
    assert inf["or"] > 1.25 and inf["or_ci95"][0] > 1.0
    assert 0.8 < oth["or"] < 1.25 and oth["or_ci95"][0] < 1.0 < oth["or_ci95"][1]
    assert r["interaction"]["or_ratio"] > 1.0
    assert r["n_inflated_rows"] > 0 and r["n_other_rows"] > 0


def test_no_dependence_gives_or_near_one_within_inflated_designs():
    d = sp.prepare_frame(synth_frame(seed=2, log_or_inf=0.0), NV)
    r = sp.fit_measure(d, "a7_density", B=200, name="t_null")
    assert r["within_inflated"]["or_ci95"][0] < 1.0 < r["within_inflated"]["or_ci95"][1]


def test_row_weights_are_one_over_the_variants_of_the_pipeline():
    d = sp.prepare_frame(synth_frame(n_pat=20, seed=3), NV)
    for pipe, n in NV.items():
        assert np.allclose(d.loc[d["pipe"] == pipe, "w"], 1.0 / n)
    per_pipe = d.groupby("pipe")["w"].sum()
    assert np.allclose(per_pipe, per_pipe.iloc[0])


def test_rows_without_a_design_class_are_excluded_and_counted():
    raw = synth_frame(n_pat=30, seed=4)
    raw.loc[raw["pipe"] == "pc", "x_ref_class"] = None
    d = sp.prepare_frame(raw, NV)
    assert set(d["pipe"]) == {"pa", "pb"}
    assert d.attrs["n_no_design_class"] == int((raw["pipe"] == "pc").sum())


def test_design_type_is_inflated_versus_immune_or_deflated():
    d = sp.prepare_frame(synth_frame(n_pat=20, seed=5), NV)
    assert set(d.loc[d["inflated"], "pipe"]) == {"pa"}
    assert set(d.loc[~d["inflated"], "pipe"]) == {"pb", "pc"}


def test_stratum_is_a_covariate_and_absorbs_a_stratum_confound():
    """Calls driven by the Morgoth stratum only, with the measure tracking the stratum: no OR once stratum enters."""
    rng = np.random.default_rng(6)
    raw = synth_frame(n_pat=400, seed=6, log_or_inf=0.0)
    shift = raw["stratum"].map({"normal_awake": 1.5, "sleep": 0.5, "mixed": -0.5, "burst_suppression": -1.5})
    raw["a7_density"] = shift * 2 + 5 + rng.normal(size=len(raw))
    lp = -1.0 + 1.2 * shift
    pat_u = raw["patient"].map(dict(zip(raw["patient"].unique(), rng.random(raw["patient"].nunique()))))
    raw["call"] = (pat_u < 1 / (1 + np.exp(-lp))).to_numpy()
    d = sp.prepare_frame(raw, NV)
    r = sp.fit_measure(d, "a7_density", B=100, name="t_strat")
    assert 0.7 < r["within_inflated"]["or"] < 1.35


def test_holm_is_over_exactly_two_tests():
    with pytest.raises(ValueError, match="exactly 2"):
        sp.holm_two({"a7_density": 0.01})
    with pytest.raises(ValueError, match="exactly 2"):
        sp.holm_two({"a7_density": 0.01, "sigma_slow_rel": 0.02, "dens_wavelet": 0.03})
    with pytest.raises(ValueError, match="S-SP1"):
        sp.holm_two({"a7_density": 0.01, "dens_wavelet": 0.02})
    h = sp.holm_two({"a7_density": 0.01, "sigma_slow_rel": 0.03})
    assert h["a7_density"]["p_holm"] == pytest.approx(0.02)
    assert h["sigma_slow_rel"]["p_holm"] == pytest.approx(0.03)
    h2 = sp.holm_two({"a7_density": 0.04, "sigma_slow_rel": 0.03})
    assert h2["a7_density"]["p_holm"] == pytest.approx(0.06) and h2["a7_density"]["reject"] is False
    assert h2["sigma_slow_rel"]["p_holm"] == pytest.approx(0.06)
    assert np.isclose(h["a7_density"]["p_holm"], mg.holm([0.01, 0.03])[0])


def test_models_report_two_tests_in_holm_and_sensitivity_outside_it():
    d = sp.prepare_frame(synth_frame(n_pat=120, seed=7), NV)
    out = sp.run_models(d, B=30)
    assert set(out["holm"]) == {"a7_density", "sigma_slow_rel"}
    assert set(out["sensitivity_detectors"]) == {"dens_" + n for n in sp.OTHER_DETECTORS}
    assert all("p_holm" not in v for v in out["sensitivity_detectors"].values())
    assert out["B"] == 30 and out["seed"] == 20261014


def test_bootstrap_is_reproducible_and_uses_the_pinned_seed():
    assert sp.SEED == 20261014 and sp.B_BOOT == 2000
    d = sp.prepare_frame(synth_frame(n_pat=80, seed=8), NV)
    a = sp.fit_measure(d, "a7_density", B=40, name="rep")
    b = sp.fit_measure(d, "a7_density", B=40, name="rep")
    assert a["within_inflated"] == b["within_inflated"]


def test_bootstrap_draws_are_relabelled():
    """A patient drawn twice counts as two clusters (116.boot_sample): the draws' cluster labels are unique per copy."""
    mech = importlib.import_module("116_mechanism")
    ci = mech.cluster_index(np.array(["a", "a", "b", "c", "c", "c"]))
    idx, lab = mech.boot_sample(ci, np.array([2, 2, 0]))
    assert len(set(lab)) == 3 and len(idx) == 3 + 3 + 2


def test_share_mediated_through_x_is_high_when_x_carries_the_effect():
    d = sp.prepare_frame(synth_frame(n_pat=500, seed=9, log_or_inf=np.log(2.0), mediated=True), NV)
    r = sp.fit_measure(d, "a7_density", B=60, name="t_med", mediation=True)
    m = r["mediation"]
    assert m["n_rows"] > 0 and m["share"] > 0.5


def test_share_mediated_through_x_is_low_when_x_is_unrelated():
    raw = synth_frame(n_pat=500, seed=10, log_or_inf=np.log(2.0))
    rng = np.random.default_rng(10)
    raw["X"] = np.where(raw["x_ref_class"] == "inflated", rng.normal(size=len(raw)), np.nan)
    d = sp.prepare_frame(raw, NV)
    r = sp.fit_measure(d, "a7_density", B=60, name="t_med0", mediation=True)
    assert r["mediation"]["share"] < 0.2


def test_mediation_is_na_without_x_results():
    raw = synth_frame(n_pat=60, seed=11).drop(columns="X")
    d = sp.prepare_frame(raw, NV)
    r = sp.fit_measure(d, "a7_density", B=20, name="t_noX", mediation=True)
    assert r["mediation"]["note"] and r["mediation"].get("share") is None


def test_missing_measure_rows_are_dropped_and_counted_never_zeroed():
    raw = synth_frame(n_pat=60, seed=12)
    raw.loc[raw.index[::7], "a7_density"] = np.nan
    d = sp.prepare_frame(raw, NV)
    r = sp.fit_measure(d, "a7_density", B=20, name="t_na")
    assert r["n_rows_na_measure"] == int(raw["a7_density"].isna().sum())
    assert r["n_rows"] == len(raw) - r["n_rows_na_measure"]



@pytest.fixture(scope="module")
def dev_windows():
    return sp.window_table("dev")







class FakeRec:
    def __init__(self, seconds=300.0, fs=FS, seed=0):
        rng = np.random.default_rng(seed)
        self.x = np.stack([10 * pink(int(seconds * fs), rng) for _ in range(19)]).astype(np.float32)
        self.fs, self.uid, self.patient = fs, "fake_0", "fake"
        self.names = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8",
                      "T4", "T6", "O2"]


@pytest.mark.parametrize("t0,t1", [(100.0, 300.2), (100.0, 300.0 + 1 / FS), (-1.0, 150.0), (299.0, 400.0)])
def test_window_past_the_recording_is_na_not_an_error_and_matches_116s_span_rule(t0, t1):
    """116 reads a window with motor_common.span, which raises past the recording end; 116 records that row's X as NA
    (feature_error), never an error and never 0. 123 does the same: NA features with a reason, the row kept."""
    mc = importlib.import_module("bench.motor_common")
    rec = FakeRec()
    with pytest.raises(ValueError):
        mc.span(rec.x, rec.fs, t0, t1)
    r = sp.recording_features(rec, pd.DataFrame({"t0": [t0], "t1": [t1]}))[0]
    assert r["reason"] == "window_outside_recording"
    assert all(np.isnan(r[c]) for c in sp.FEATURE_COLS)
    assert (r["t0"], r["t1"]) == (t0, t1)


def test_window_inside_the_recording_is_cut_by_116s_span_and_scored():
    mc = importlib.import_module("bench.motor_common")
    rec = FakeRec(seconds=300.0)
    for t0, t1 in [(10.0, 300.0), (0.0, 150.5), (10.003, 150.004)]:
        seg, i0 = mc.span(rec.x[[2, 13]], rec.fs, t0, t1)
        r = sp.recording_features(rec, pd.DataFrame({"t0": [t0], "t1": [t1]}))[0]
        assert r["reason"] is None and np.isfinite(r["a7_density"])
        assert r["dur_s"] == pytest.approx(seg.shape[1] / FS)
