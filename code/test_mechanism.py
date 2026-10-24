"""Tests for code/116_mechanism.py (second stage, "Mechanism: where false calls come from"):
synthetic fixtures and the DEV split only. Every guard is also run against a deliberately broken input or a broken
version of the code, and the test checks that the guard catches it. Dev rows are read for structure only (no rate is
computed or printed).
cd code && /usr/bin/python3 -m pytest test_mechanism.py -q"""
import glob
import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest

mech = importlib.import_module("116_mechanism")
rn = importlib.import_module("114_benchmark")
dn = importlib.import_module("doc_null")



def one_over_f(n, rng, chi=1.0, n_ch=None):
    """Gaussian series with power spectrum ~ 1 / f^chi (FFT shaping, random phases), length n (rows: channels)."""
    shape = (n_ch, n // 2 + 1) if n_ch else (n // 2 + 1,)
    f = np.fft.rfftfreq(n)
    amp = np.zeros(n // 2 + 1)
    amp[1:] = f[1:] ** (-chi / 2)
    spec = amp * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return np.fft.irfft(spec, n=n, axis=-1)


def b12_draw(g):
    y, _ = dn.schedule("block", g, block=12)
    return mech.contrast_weights(y, [((1,), (0,))])


def random_draw(g):
    y, _ = dn.schedule("random", g)
    return mech.contrast_weights(y, [((1,), (0,))])


@pytest.fixture(scope="module")
def E_b12():
    return mech.expected_spectrum(b12_draw).at(dn.N_TRIALS)


@pytest.fixture(scope="module")
def E_random():
    return mech.expected_spectrum(random_draw).at(dn.N_TRIALS)



def test_x_near_zero_for_randomised_design_and_clearly_positive_for_b12(E_b12, E_random):
    rng = np.random.default_rng(1)
    xr, xb = [], []
    for _ in range(100):
        F = one_over_f(dn.N_TRIALS, rng)
        xr.append(mech.overlap(F, E_random)["X"])
        xb.append(mech.overlap(F, E_b12)["X"])
    xr, xb = np.array(xr), np.array(xb)
    assert np.isfinite(xr).mean() > 0.9 and np.isfinite(xb).mean() > 0.9
    assert abs(np.nanmean(xr)) < 0.05 and np.nanmax(np.abs(xr)) < 0.15
    assert np.nanmean(xb) > 0.5 and np.mean(xb[np.isfinite(xb)] > 0.2) > 0.9


def test_x_is_exactly_zero_for_a_flat_label_spectrum():
    rng = np.random.default_rng(2)
    F = one_over_f(48, rng)
    assert mech.overlap(F, np.ones(24))["X"] == pytest.approx(0.0, abs=1e-12)


def test_exclusion_removes_the_alternation_and_its_harmonics(E_b12):
    f, _ = mech.periodogram(np.zeros(dn.N_TRIALS))
    k = np.round(f * dn.N_TRIALS).astype(int)
    assert set(k[mech.peak_bins(E_b12)]) == {2, 6, 10, 14, 18, 22}


def _x_shift_from_inband_signal(E, y, exclude):
    """Largest change of X when a strong label-locked response (the test statistic's own in-band power) is added to
    1/f series: with the exclusion the aperiodic fit never sees the alternation bins, so X must not move."""
    rng = np.random.default_rng(3)
    out = []
    for _ in range(20):
        F = one_over_f(dn.N_TRIALS, rng)
        a = mech.overlap(F, E, exclude=exclude)["X"]
        b = mech.overlap(F + 8 * F.std() * (y - y.mean()), E, exclude=exclude)["X"]
        if np.isfinite(a) or np.isfinite(b):
            out.append(abs(b - a) if np.isfinite(a) and np.isfinite(b) else np.inf)
    assert len(out) >= 15
    return max(out)


def test_exclusion_guard_and_it_fails_when_the_exclusion_is_disabled(E_b12):
    y, _ = dn.schedule("block", np.random.default_rng(0), block=12)
    assert _x_shift_from_inband_signal(E_b12, y, exclude=True) < 1e-9
    assert _x_shift_from_inband_signal(E_b12, y, exclude=False) > 0.1


def test_time_shuffling_the_feature_series_drives_x_to_zero(E_b12):
    rng = np.random.default_rng(4)
    xo, xs = [], []
    for _ in range(60):
        F = one_over_f(dn.N_TRIALS, rng, chi=1.5)
        xo.append(mech.overlap(F, E_b12)["X"])
        xs.append(mech.overlap(F[rng.permutation(len(F))], E_b12)["X"])
    xo, xs = np.array(xo), np.array(xs)
    assert np.isfinite(xs).mean() > 0.75
    assert np.nanmean(xo) > 0.5
    assert abs(np.nanmean(xs)) < 0.15 and abs(np.nanmean(xs)) < 0.2 * np.nanmean(xo)
    blocks = np.repeat(np.arange(4), 12)
    F = one_over_f(dn.N_TRIALS, np.random.default_rng(40), chi=1.5)
    keep_slow = np.concatenate([rng.permutation(np.flatnonzero(blocks == b)) for b in range(4)])
    assert mech.overlap(F[keep_slow], E_b12)["X"] > 0.3


def test_forced_fit_failure_gives_na_never_zero(E_b12, monkeypatch):
    r = mech.overlap(np.full(dn.N_TRIALS, 3.0), E_b12)
    assert np.isnan(r["X"]) and np.isnan(r["chi"]) and r["reason"] == "fit_failed"
    F = one_over_f(dn.N_TRIALS, np.random.default_rng(5))
    assert np.isfinite(mech.overlap(F, E_b12)["X"])

    def boom(*a, **k):
        raise RuntimeError("forced")
    monkeypatch.setattr(mech, "least_squares", boom)
    r = mech.overlap(F, E_b12)
    assert np.isnan(r["X"]) and r["X"] != 0 and r["reason"] == "fit_failed"


def test_degenerate_step_fit_is_na():
    """A periodogram whose least-squares knee fit is a step through its lowest bin (parameters diverge) is a failed
    fit: NA with its own reason, never a finite X."""
    f = np.arange(1, 13) / 24
    P = np.r_[1e-4, np.ones(11)]
    fit, why = mech.fit_aperiodic(f, P)
    assert fit is None and why == "fit_degenerate"
    r = mech.overlap_from_periodogram(f, P[None], np.ones(12))
    assert np.isnan(r["X"]) and r["reason"] == "fit_degenerate"


def test_channel_average_needs_a_strict_majority_of_fitted_channels(E_b12):
    rng = np.random.default_rng(6)
    good = []
    while len(good) < 5:
        g = one_over_f(dn.N_TRIALS, rng)
        if np.isfinite(mech.overlap(g, E_b12)["X"]):
            good.append(g)
    bad = np.full(dn.N_TRIALS, 1.0)
    assert np.isnan(mech.overlap(np.vstack([good[0], bad]), E_b12)["X"])
    one = [mech.overlap(g, E_b12)["X"] for g in good[:4]]
    r = mech.overlap(np.vstack(good[:4] + [bad]), E_b12)
    assert r["n_ch"] == 5 and r["n_ch_fit"] == 4 and r["X"] == pytest.approx(np.mean(one))
    r = mech.overlap(np.vstack(good[:2] + [bad] * 3), E_b12)
    assert np.isnan(r["X"]) and r["reason"] == "fit_failed" and r["n_ch_fit"] == 2


def test_fewer_than_8_surviving_bins_is_na():
    E = np.zeros(10)
    E[:4] = 1.0
    F = one_over_f(20, np.random.default_rng(7))
    r = mech.overlap(F, E)
    assert np.isnan(r["X"]) and r["reason"] == "few_bins" and r["n_bins"] == 6


def test_no_label_contrast_is_na_with_its_own_reason():
    F = one_over_f(48, np.random.default_rng(8))
    r = mech.overlap(F, np.zeros(24))
    assert np.isnan(r["X"]) and r["reason"] == "no_contrast"
    assert mech.contrast_weights(np.ones(48, int), [((1,), (2,))]) == []


def test_periodogram_normalisation_is_shared_by_feature_and_label():
    y = np.r_[np.ones(24), np.zeros(24)]
    f, P = mech.periodogram(y)
    assert np.allclose(P, np.abs(np.fft.rfft(y - y.mean()))[1:] ** 2 / len(y))
    assert np.allclose(mech.label_spectrum([y]), P)



def test_expected_spectrum_randomised_design_is_flat(E_random):
    n = dn.N_TRIALS
    assert not mech.peak_bins(E_random).any()
    assert E_random.std() / E_random.mean() < 0.08
    assert E_random.mean() == pytest.approx(4.0 / (n * (n - 1)), rel=0.05)


def test_expected_spectrum_of_a_fixed_schedule_equals_the_realised_one():
    mod = rn.load_pipe("cornell")
    spec = rn.variant_specs(mod, "bdsp")["theoretical_bh_keep40_r4"]
    ds = mech.design_spectrum(mod, spec, "published", n_draws=50)
    assert ds.fixed
    ov = rn._sched(mod, mod.schedule, spec, 0.0, None)
    cfg = mech.module_cfg(mod, spec)
    w = mech.contrast_weights(mech.grid_labels(mod, cfg, ov), mech.contrasts(mod, cfg))
    assert np.array_equal(ds.at(ds.n), mech.label_spectrum(w))
    assert not mech.design_spectrum(mod, spec, "randomized", n_draws=50).fixed


def test_variable_length_designs_are_evaluated_on_each_rows_grid():
    def draw(g):
        n = int(g.integers(40, 57))
        return mech.contrast_weights(g.permutation(np.arange(n) % 2), [((1,), (0,))])
    ds = mech.expected_spectrum(draw, n_draws=300)
    assert not ds.fixed and len(ds.ns) > 1
    for n in (40, 47, 56):
        E = ds.at(n)
        assert E.shape == (n // 2,) and np.isfinite(E).all() and not mech.peak_bins(E).any()


def test_design_table_classifies_known_designs():
    tab = mech.design_table(["cornell", "mbmi", "dubois", "gibson16", "sokoliuk21"], n_draws=60)
    pub = tab[tab["kind"] == "published"]
    typ = lambda pipe, pre="": pub[(pub["pipe"] == pipe) & pub["variant"].str.startswith(pre)]["design_type"]
    assert len(typ("cornell")) == 16 and (typ("cornell") == "block").all()
    assert (typ("mbmi") == "randomised").all()
    assert (typ("dubois", "assessment") == "block").all() and (typ("dubois", "training") == "randomised").all()
    assert (typ("gibson16", "ssep") == "no_contrast").all() and (typ("gibson16", "p3b") == "randomised").all()
    assert (typ("sokoliuk21") == "no_contrast").all()
    assert (tab[(tab["pipe"] == "sokoliuk21") & (tab["kind"] == "randomized")]["design_type"] == "randomised").all()
    pk = pub[pub["pipe"] == "cornell"]
    assert pk["peak_freqs"].str.len().gt(0).all()
    assert pub[pub["variant"].str.startswith("assessment")]["x_ref_1f"].gt(0.5).all()
    assert pk["x_ref_1f"].lt(0).all()
    assert pub[pub["pipe"] == "sokoliuk21"]["x_ref_1f"].isna().all()



@pytest.mark.parametrize("pipe,variant,n_epochs", [
    ("columbia", "as_coded", 96 * 5), ("cornell", "theoretical_bh_keep40_r4", 64 * 3),
    ("cornell", "theoretical_bh_keep40_r2", 32 * 3), ("edlow17", "permsecond_cvsecond_alt", 24 * 16),
    ("mbmi", "gu17_cspfold_peak", 60), ("cruse12p", "random_perclass", 180), ("mbvt", "vt3_mur20_ds7_full_gt23", 480),
    ("dubois", "assessment_random_phase1", 90), ("sokoliuk21", "roc116_as_coded_as_coded_none", 72)])
def test_grid_sizes(pipe, variant, n_epochs):
    mod = rn.load_pipe(pipe)
    spec = rn.variant_specs(mod, "bdsp")[variant]
    ov = rn._sched(mod, mod.schedule, spec, 100.0, np.random.default_rng(0))
    cfg = mech.module_cfg(mod, spec)
    lab = mech.grid_labels(mod, cfg, ov)
    g = mech.grid_windows(mod, cfg, ov, 128.0)
    assert len(lab) == n_epochs and g["win"].shape == (n_epochs, 2)
    assert (np.diff(g["win"][:, 0]) > 0).all()
    t0, t1 = mod.extent(ov)
    lo = g["win"].min() if g["base"] is None else min(g["win"].min(), g["base"].min())
    assert t0 <= lo and g["win"].max() <= t1 + 1e-9


def test_chennu13_grid_holds_the_epoched_words_only():
    mod = rn.load_pipe("chennu13")
    spec = rn.variant_specs(mod, "bdsp")["p1_samples10_none_b12"]
    ov = rn._sched(mod, mod.schedule, spec, 100.0, np.random.default_rng(0))
    lab = mech.grid_labels(mod, mech.module_cfg(mod, spec), ov)
    assert len(lab) == int(np.sum(ov.labels != mod.DIST)) and set(lab) == {mod.EXPL, mod.IMPL}


@pytest.mark.parametrize("pipe,variant,base", [
    ("mbvt", "vt3_mur20_ds7_full_gt23", (-0.1, 0.0)), ("mbvt", "vt3_mur20_seg8_full_gt23", None),
    ("chennu13", "p1_samples10_none_b12", (-0.3, 0.0)), ("gibson16", "p3b_two025", (-0.2, 0.0)),
    ("sokoliuk21", "roc116_as_coded_as_coded_none", None)])
def test_erp_windows_and_baselines_come_from_the_module(pipe, variant, base):
    mod = rn.load_pipe(pipe)
    spec = rn.variant_specs(mod, "bdsp")[variant]
    ov = rn._sched(mod, mod.schedule, spec, 100.0, np.random.default_rng(0))
    cfg = mech.module_cfg(mod, spec)
    g = mech.grid_windows(mod, cfg, ov, 128.0)
    on = np.asarray(ov.onsets_s, float)
    if pipe == "chennu13":
        on = on[np.asarray(ov.labels) != mod.DIST]
    if base is None:
        assert g["base"] is None
    else:
        assert np.allclose(g["base"] - on[:, None], base)


def test_erp_feature_is_window_mean_minus_baseline_mean():
    mod = rn.load_pipe("mbvt")
    specs = rn.variant_specs(mod, "bdsp")
    fs = 128.0
    x = np.random.default_rng(14).normal(size=(19, int(400 * fs))).astype(np.float32) * 10
    rec = rn.en.Recording(source="bdsp", uid="syn", patient="P", x=x, fs=fs, names=list(rn.en.NAMES19))
    ov = rn._sched(mod, mod.schedule, specs["vt3_mur20_ds7_full_gt23"], 100.0, np.random.default_rng(0))
    out = {}
    for v in ("vt3_mur20_ds7_full_gt23", "vt3_mur20_seg8_full_gt23"):
        cfg = mech.module_cfg(mod, specs[v])
        out[v] = (mech.feature_series(rec, mod, cfg, ov), mech.grid_windows(mod, cfg, ov, fs))
    F, g = out["vt3_mur20_ds7_full_gt23"]
    rows = mech.module_rows(mod, rec)
    seg, i0 = mech.mc.span(np.asarray(x, float)[rows], fs, *mod.extent(ov))
    xf = mech.mc.bandpass(seg, fs, *mech.ERP_BAND)
    k = 37
    s = lambda t: int(round(t * fs)) - i0
    want = xf[:, s(g["win"][k, 0]):s(g["win"][k, 1])].mean(1) - xf[:, s(g["base"][k, 0]):s(g["base"][k, 1])].mean(1)
    assert F.shape == (len(rows), 480) and np.allclose(F[:, k], want)
    F2, g2 = out["vt3_mur20_seg8_full_gt23"]
    assert np.allclose(F2[:, k], xf[:, s(g2["win"][k, 0]):s(g2["win"][k, 1])].mean(1))



def test_cluster_bootstrap_relabels_repeated_draws():
    pats = np.array(["A", "A", "B", "C", "C", "C"])
    ci = mech.cluster_index(pats)
    draw = np.array([list(ci["ids"]).index("A")] * 2 + [list(ci["ids"]).index("C")])
    idx, cl = mech.boot_sample(ci, draw)
    assert list(pats[idx]) == ["A", "A", "A", "A", "C", "C", "C"]
    assert len(np.unique(cl)) == 3 and len(np.unique(cl[:4])) == 2
    assert len(np.unique(pats[idx])) == 2


def test_row_weights_make_pipelines_count_equally():
    pipes = np.array(["big"] * 20 + ["small"] * 2)
    w = mech.row_weights(pipes, {"big": 20, "small": 2})
    assert w[pipes == "big"].sum() == pytest.approx(1.0) and w[pipes == "small"].sum() == pytest.approx(1.0)


def _two_pipeline_rows(rng, n_rec=400):
    """Pipeline 'big' (20 variants) with a strong X effect, 'small' (2 variants) with none."""
    rows = []
    for i in range(n_rec):
        x = rng.normal()
        for pipe, nv, beta in (("big", 20, 1.5), ("small", 2, 0.0)):
            for v in range(nv):
                p = 1 / (1 + np.exp(-(-1.0 + beta * x)))
                rows.append({"patient": f"P{i}", "pipe": pipe, "variant": f"{pipe}{v}", "X": x,
                             "call": float(rng.random() < p)})
    return pd.DataFrame(rows)


def test_weighted_fe_logit_gives_each_pipeline_equal_weight():
    d = _two_pipeline_rows(np.random.default_rng(9))
    w = mech.row_weights(d["pipe"].to_numpy(), {"big": 20, "small": 2})
    fe = pd.factorize(d["variant"])[0]
    bw = mech.fe_logit(d["call"].to_numpy(), d[["X"]].to_numpy(), fe, w)["beta"][0]
    bu = mech.fe_logit(d["call"].to_numpy(), d[["X"]].to_numpy(), fe, np.ones(len(d)))["beta"][0]
    assert bu > 1.2
    assert 0.4 < bw < 1.1


def test_fe_logit_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(10)
    n = 3000
    fe = rng.integers(0, 4, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(np.array([-2, -1, 0, 0.5])[fe] + 0.7 * x)))).astype(float)
    w = rng.uniform(0.2, 1.0, n)
    ours = mech.fe_logit(y, x[:, None], fe, w)
    D = np.column_stack([x, np.eye(4)[fe]])
    ref = sm.GLM(y, D, family=sm.families.Binomial(), var_weights=w).fit()
    assert ours["beta"][0] == pytest.approx(ref.params[0], abs=1e-6)


def test_fe_logit_drops_variants_without_both_outcomes():
    rng = np.random.default_rng(11)
    n = 2000
    fe = rng.integers(0, 3, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.3 + 0.2 * (x > 0)).astype(float)
    y[fe == 2] = 0.0
    r = mech.fe_logit(y, x[:, None], fe, np.ones(n))
    assert r["converged"] and r["n_fe_dropped"] == 1 and np.isfinite(r["beta"]).all()


def _s1_rows(rng, n_pat=60):
    rows = []
    for i in range(n_pat):
        for pipe, nv, nw in (("a", 3, 2), ("b", 2, 1), ("c", 4, 3)):
            for v in range(nv):
                for j in range(nw):
                    rows.append({"patient": f"P{i:03d}", "pipe": pipe, "variant": f"{pipe}{v}", "j": j,
                                 "call": float(rng.random() < 0.1 + 0.3 * (i % 7 == 0)), "X": rng.normal()})
    return pd.DataFrame(rows)


def test_s1_uses_one_row_per_patient_and_pipeline_not_per_variant():
    d = _s1_rows(np.random.default_rng(12))
    u = mech.s1_units(d)
    assert len(u) == 60 * 3 and not u.duplicated(["patient", "pipe"]).any()
    a = d[d["pipe"] == "a"]
    d2 = pd.concat([d[d["pipe"] != "a"], a, a.assign(variant=a.variant + "_copy")], ignore_index=True)
    u2 = mech.s1_units(d2)
    m = u.merge(u2, on=["patient", "pipe"], suffixes=("", "_2"))
    assert np.allclose(m["O"], m["O_2"]) and np.allclose(m["E"], m["E_2"])
    r1 = mech.s1_dispersion(u, n_perm=200, seed=1)
    r2 = mech.s1_dispersion(u2, n_perm=200, seed=1)
    assert r1["D"] == pytest.approx(r2["D"]) and r1["p"] == pytest.approx(r2["p"])


def test_s1_permutation_is_within_pipeline_and_detects_shared_patients():
    rng = np.random.default_rng(13)
    hot = set(rng.choice(80, 8, replace=False))
    rows = []
    for i in range(80):
        for pipe in "abcd":
            for v in range(2):
                p = 0.5 if i in hot else 0.03
                rows.append({"patient": f"P{i:03d}", "pipe": pipe, "variant": f"{pipe}{v}", "j": 0,
                             "call": float(rng.random() < p), "X": 0.0})
    u = mech.s1_units(pd.DataFrame(rows))
    r = mech.s1_dispersion(u, n_perm=500, seed=2)
    assert r["p"] < 0.01
    rows2 = [dict(r_, patient=f"Q{(int(r_['patient'][1:]) * 7 + ord(r_['pipe'])) % 80:03d}") for r_ in rows]
    r2 = mech.s1_dispersion(mech.s1_units(pd.DataFrame(rows2)), n_perm=500, seed=2)
    assert r2["p"] > 0.05


def _s4_windows(rng, n_pat=8, shift=0.6):
    rows = []
    for i in range(2 * n_pat):
        mcs = i < n_pat
        for k in range(30):
            rows.append({"uid": f"R{i}", "patient": f"P{i}", "diagnosis": "MCS+" if mcs else "VS", "window": k,
                         "skipped": None, "X": rng.normal(shift if mcs else 0.0, 0.5), "reason": None,
                         "aperiodic": rng.normal(1.5, 0.2), "rel_alpha": rng.uniform(0.05, 0.2),
                         "log_delta_alpha": rng.normal(1.0, 0.3), "lower_delta": bool(k % 2)})
    return pd.DataFrame(rows)


def test_s4_recovers_a_planted_mcs_shift():
    out = mech.s4(_s4_windows(np.random.default_rng(16)), B=60)
    for v in ("a_overall", "b_lower_delta", "c_state_adjusted"):
        r = out[v]
        assert r["n_mcs"] == 8 and r["n_vs"] == 8 and r["estimate"] > 0.3 and r["estimate_ci95"][0] > 0
        assert r["auroc_markers_plus_x"] > r["auroc_markers"] and np.isfinite(r["auroc_gain_ci95"]).all()
    assert out["a_overall"]["wilcoxon_p_two_sided"] < 0.01 and out["p_test"] == out["a_overall"]["wilcoxon_p_two_sided"]
    null = mech.s4(_s4_windows(np.random.default_rng(17), shift=0.0), B=60)
    assert null["a_overall"]["wilcoxon_p_two_sided"] > 0.05


def test_loo_auc_leaves_out_every_copy_of_a_patient(monkeypatch):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(12, 2))
    y = np.r_[np.ones(6), np.zeros(6)]
    groups = np.r_[0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8]
    seen = []
    real = mech._ridge_logit
    monkeypatch.setattr(mech, "_ridge_logit", lambda A, yy, **k: seen.append(len(yy)) or real(A, yy, **k))
    mech.loo_auc(X, y, groups)
    sizes = {g: 12 - int(np.sum(groups == g)) for g in np.unique(groups)}
    assert seen == [sizes[g] for g in np.unique(groups)]


def test_background_groups():
    st = pd.Series(["normal_awake", "sleep", "burst_suppression", "iiic_periodic", "diffuse_slowing", "other_awake",
                    "mixed", "unlabelled"])
    g = mech.background_group(st)
    assert list(g) == ["organised"] * 2 + ["disorganised"] * 3 + ["excluded"] * 3


def test_holm_family_has_four_tests():
    out = mech.holm_family({"S1": 0.01, "S2": 0.04, "S3": 0.03, "S4": 0.5})
    assert set(out) == {"S1", "S2", "S3", "S4"} and out["S1"]["p_holm"] == pytest.approx(0.04)





def test_results_are_whole_files_of_one_code_version(tmp_path):
    d = str(tmp_path / "x_rows")
    os.makedirs(d)
    mech._write_result(d, "U1", [{"key": "a", "uid": "U1", "X": 0.1}])
    mech._write_result(d, "U2", [{"key": "b", "uid": "U2", "X": None}])
    open(os.path.join(d, "U3.json.tmp"), "w").write("{partial")
    assert mech._done_uids(d) == {"U1", "U2"} and list(mech.read_results(d)["key"]) == ["a", "b"]
    json.dump({"uid": "U4", "sha256_116": "0" * 64, "rows": []}, open(os.path.join(d, "U4.json"), "w"))
    with pytest.raises(RuntimeError, match="another version"):
        mech._done_uids(d)


def test_no_analysis_paths_in_dev_mode():
    p = mech.paths("dev")
    for v in p.values():
        assert "_bench_shards" not in v or "_bench_shards_dev" in v
        assert "columbia_null_rows" not in v and "benchmark_" not in os.path.basename(v)



def test_analysis_rows_swap_columbias_first_overlays_for_113s(monkeypatch):
    runner = pd.DataFrame([
        {"key": "c0", "null": "ncmd_bdsp", "pipe": "columbia", "resp": "none", "j": 0, "uid": "U1"},
        {"key": "c1", "null": "ncmd_bdsp", "pipe": "columbia", "resp": "none", "j": 1, "uid": "U1"},
        {"key": "k0", "null": "ncmd_bdsp", "pipe": "cornell", "resp": "none", "j": 0, "uid": "U1"}])
    seen = {}
    monkeypatch.setattr(mech.mg, "load_rows", lambda split: runner)
    monkeypatch.setattr(mech.mg, "drop_touched", lambda df: (df, {}))
    monkeypatch.setattr(mech.rn, "columbia113", lambda: "C")
    monkeypatch.setattr(mech.mg, "dev_uids", lambda: set())
    monkeypatch.setattr(mech.mg, "touched_uids", lambda: set())
    monkeypatch.setattr(mech.mg, "check_113", lambda c, df, dev, t: seen.setdefault("checked", list(df["key"])))
    monkeypatch.setattr(mech.mg, "rows_from_113", lambda c: pd.DataFrame([{"key": "113|U1", "null": "ncmd_bdsp",
                                                                          "pipe": "columbia", "j": 0, "uid": "U1"}]))
    out = mech.load_rows("analysis")
    assert sorted(out["key"]) == ["113|U1", "c1", "k0"] and seen["checked"] == ["c1", "k0"]


def _synthetic_recording(seconds=1800, fs=128.0, seed=19):
    x = np.random.default_rng(seed).normal(size=(19, int(seconds * fs))).astype(np.float32) * 20
    return rn.en.Recording(source="bdsp", uid="SYN", patient="PSYN", x=np.cumsum(x, 1) * 0.05 + x, fs=fs,
                           names=list(rn.en.NAMES19))


def test_recording_results_on_a_113_row_and_a_runner_row():
    rec = _synthetic_recording()
    col, cor = rn.load_pipe("columbia"), rn.load_pipe("cornell")
    v = "theoretical_bh_keep40_r2"
    spec = rn.variant_specs(cor, "bdsp")[v]
    ov = rn._sched(cor, cor.schedule, spec, 30.0, rn._rng("cornell", "SYN", 0, 1))
    t0, t1 = cor.extent(ov)
    need = pd.DataFrame([
        {"key": "113|SYN", "uid": "SYN", "patient": "PSYN", "pipe": "columbia", "variant": "as_coded", "j": 0,
         "start_s": 20.0, "part": "m1", "rand_variant": "as_coded"},
        {"key": "SYN|0|cornell", "uid": "SYN", "patient": "PSYN", "pipe": "cornell", "variant": v, "j": 0,
         "start_s": 30.0, "part": "m1", "extent_t0": t0, "extent_t1": t1, "n_onsets": len(ov.onsets_s)},
        {"key": "113|SYN", "uid": "SYN", "patient": "PSYN", "pipe": "columbia", "variant": "as_coded", "j": 0,
         "start_s": 20.0, "part": "rand", "rand_variant": "as_coded"}])
    out = {(r["pipe"], r["kind"]): r for r in mech.recording_results(rec, need, mech.Spectra(None, 30), "analysis")}
    c, k, cr = out[("columbia", "published")], out[("cornell", "published")], out[("columbia", "randomized")]
    assert c["n_epochs"] == 480 and c["n_ch"] == 19 and 0.0 <= c["artifact_share_116"] <= 1.0
    assert k["n_epochs"] == 96 and "artifact_share_116" not in k
    for r in (c, k):
        assert (r["X"] is None) == (r["reason"] is not None) and {"X_whole", "X_band", "n_tiles"} <= set(r)
    assert cr["key"] == "SYN|columbia|0|~as_coded|rand" and "X_whole" not in cr
    assert {"X_flat0", "n_ch_flat", "X_mask", "n_rejected", "mask"} <= set(c) and "X_mask" not in cr
    assert c["mask"] == "none" and c["X_mask"] == c["X"] and c["n_rejected"] == 0
    assert k["mask"] == "module" and k["n_rejected"] == 96 - 4 * int(round(cor.KEEP_FRAC * 24))



DEV = rn.SH["dev"]
needs_dev = pytest.mark.skipif(not glob.glob(os.path.join(DEV, "n*_bdsp__*__native__none")), reason="no dev rows")


@needs_dev
def test_dev_rows_select_first_overlays_of_every_variant():
    df = mech.load_rows("dev")
    m1 = mech.m1_rows(df)
    assert len(m1) and (m1["j"] == 0).all() and (m1["status"] == "ok").all() and (m1["resp"] == "none").all()
    assert set(m1["null"]) <= {"ncmd_bdsp", "nstim0_bdsp"} and m1["dev"].all()
    assert not m1.duplicated(["uid", "pipe", "variant", "tile"]).any()


@needs_dev
def test_dev_overlays_are_rebuilt_exactly():
    df = mech.load_rows("dev")
    m1 = mech.m1_rows(df)
    for pipe, d in m1.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, "bdsp")
        for _, r in d.head(5).iterrows():
            ov = mech.overlay_for_row(mod, specs, r)
            assert len(ov.onsets_s) == r["n_onsets"]
            if pd.notna(r.get("rand_extent_t0")):
                mech.overlay_for_row(mod, specs, r, kind="randomized")
        broken = d.iloc[0].copy()
        broken["start_s"] += 1.0
        with pytest.raises(RuntimeError, match="does not rebuild"):
            mech.overlay_for_row(mod, specs, broken)


@needs_dev
def test_dev_pdoc_overlays_rebuild_and_read_the_reduced_montage():
    df = mech.load_rows("dev")
    pr = mech.pdoc_rows(df)
    assert len(pr) and set(pr["null"]) <= {"ncmd_pdoc", "nstim0_pdoc"} and (pr["j"] == 0).all()
    rec = rn.en.Recording(source="pdoc", uid="S", patient="P", x=np.zeros((6, 10)), fs=128.0,
                          names=["F3", "C3", "O1", "F4", "C4", "O2"])
    for pipe, d in pr.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, "pdoc")
        mech.overlay_for_row(mod, specs, d.iloc[0])
        with mech.source_channels(mod, "pdoc"):
            names = [rec.names[i] for i in mech.module_rows(mod, rec)]
        assert set(names) <= {"F3", "F4", "C3", "C4", "O1", "O2"} and names == rn.wr.PDOC_CH[pipe]["NATIVE_CH"]



def test_disjoint_set_keeps_only_extra_listing_recordings():
    listing = {"analysis": [(0, "Extra_A"), (3, "Extra_B")], "complete": True}
    m1 = pd.DataFrame({"uid": ["Extra_A", "Design_C", "Extra_B", "Design_D", "Extra_A"],
                       "pipe": ["a", "a", "b", "b", "b"]})
    stems = mech.extra_stems(listing)
    assert stems == {"Extra_A", "Extra_B"}
    sub = mech.disjoint_rows(m1, stems)
    assert list(sub["uid"]) == ["Extra_A", "Extra_B", "Extra_A"]
    assert mech.disjoint_rows(m1, set()).empty



def test_k_table_reader(tmp_path):
    good = {p: {"k": 1, "k_source": "x"} for p in mech.PIPES}
    good["dubois"]["k"] = 2
    good["mbmi"]["k_sensitivity"] = 2
    fn = tmp_path / "k.json"
    fn.write_text(json.dumps(good))
    t = mech.read_k_table(str(fn))
    assert t["dubois"]["k"] == 2 and t["mbmi"]["k_sensitivity"] == 2 and t["cornell"]["k_sensitivity"] is None
    for broken in ({**good, "dubois": {"k": 0}}, {**good, "dubois": {"k": 1.5}},
                   {k: v for k, v in good.items() if k != "mbvt"}, {**good, "nonsense": {"k": 1}}):
        fn.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            mech.read_k_table(str(fn))


def _a1_rows():
    rows = []
    for i in range(40):
        for v, base in (("vA", 0.1), ("vB", 0.3)):
            st = ["normal_awake", "sleep", "burst_suppression", "normal_awake"][i % 4]
            rows.append({"null": "ncmd_bdsp", "pipe": "p", "variant": v, "patient": f"P{i}", "uid": f"U{i}", "j": 0,
                         "stratum": st, "transitioning": i % 8 == 3, "a9_flag": i % 10 == 0,
                         "call": (i % 5 == 0) if v == "vA" else (i % 2 == 0), "primary_variant": "vB"})
    return pd.DataFrame(rows)


def test_a1_awake_rate_uses_clean_non_transitioning_awake_windows_for_both_variants():
    d = _a1_rows()
    out = mech.a1_rates(d, {("ncmd_bdsp", "p"): {"validity_primary": "vA", "default": "vB"}}, B=50)["p"]
    keep = (d["stratum"] == "normal_awake") & ~d["transitioning"] & ~d["a9_flag"]
    for role, v in (("validity_primary", "vA"), ("default", "vB")):
        e = out["awake"][role]
        want = d[keep & (d["variant"] == v)]
        assert e["variant"] == v and e["n"] == len(want) and e["rate"] == pytest.approx(want["call"].mean())
        assert e["ci95"][0] <= e["rate"] <= e["ci95"][1]
    assert out["n_a9_excluded"] == int(d["a9_flag"].sum())
    icu = ~d["transitioning"] & ~d["a9_flag"] & (d["variant"] == "vA") & (d["stratum"] == "burst_suppression")
    assert out["icu_typical"]["validity_primary"]["n"] == int(icu.sum())


def test_a9_flags_from_the_activation_tables():
    iv = pd.DataFrame({"uid": ["U1"], "kind": ["photic"], "start_s": [100.0], "end_s": [200.0], "detail": [""],
                       "method": ["detector"]})
    recs = pd.DataFrame({"uid": ["U1", "U2"]})
    rows = pd.DataFrame({"uid": ["U1", "U1", "U2", "U3"], "extent_t0": [0.0, 300.0, 100.0, 0.0],
                         "extent_t1": [150.0, 400.0, 200.0, 10.0]})
    f = mech.a9_flags(rows, iv, recs)
    assert list(f["a9_flag"]) == [True, False, False, True] and list(f["a9_processed"]) == [True, True, True, False]


def _seq_rows(pipe, variant, calls_by_patient, null="ncmd_bdsp"):
    return pd.DataFrame([{"null": null, "pipe": pipe, "variant": variant, "patient": p, "uid": p, "j": j, "seq": True,
                          "overlap": False, "call": bool(c), "stat": 0.0, "p": 0.5}
                         for p, cs in calls_by_patient.items() for j, c in enumerate(cs)])


def test_a1b_per_100_at_the_published_k():
    calls = {f"P{i}": [i % 4 == 0, i % 4 == 1, False] for i in range(40)}
    d = _seq_rows("dubois", "v", calls)
    r = mech.per100_composite(d, rn.load_pipe("dubois"), 2, B=50, seed=1)
    assert r["n_patients"] == 40 and r["per100"] == pytest.approx(50.0) and r["ci95"][0] <= 50.0 <= r["ci95"][1]
    assert mech.per100_composite(d, rn.load_pipe("dubois"), 1, B=50, seed=1)["per100"] == pytest.approx(25.0)
    mb = mech.per100_composite(_seq_rows("mbvt", "v", calls, "nstim0_pdoc"), rn.load_pipe("mbvt"), 2, B=50, seed=1,
                               rule="best_run")
    assert mb["per100"] == pytest.approx(50.0) and mb["rule"] == "best_run"


def test_a1b_at_k_uses_patients_with_k_windows_and_reports_the_excluded():
    calls = {**{f"P{i}": [i % 4 == 0, i % 4 == 1, False] for i in range(40)}, **{f"S{i}": [True] for i in range(10)}}
    d = _seq_rows("dubois", "v", calls)
    r2 = mech.per100_composite(d, rn.load_pipe("dubois"), 2, B=50, seed=1)
    assert r2["n_patients"] == 40 and r2["per100"] == pytest.approx(50.0)
    assert r2["n_patients_any_window"] == 50 and r2["n_excluded_fewer_than_k"] == 10
    assert r2["share_excluded_fewer_than_k"] == pytest.approx(0.2)
    r1 = mech.per100_composite(d, rn.load_pipe("dubois"), 1, B=50, seed=1)
    assert r1["n_patients"] == 50 and r1["n_excluded_fewer_than_k"] == 0 and r1["share_excluded_fewer_than_k"] == 0


def _calibration_data(rng, slope_pdoc=1.0, shift_pdoc=0.0, n_pat=300):
    fe_a = {"p|v1": -2.5, "p|v2": -1.5, "q|w1": -2.0}
    rows, prow = [], []
    for i in range(n_pat):
        for fe, a in fe_a.items():
            pipe, var = fe.split("|")
            x = rng.normal()
            rows.append({"patient": f"B{i}", "uid": f"B{i}", "pipe": pipe, "variant": var, "X": x,
                         "call": float(rng.random() < 1 / (1 + np.exp(-(a + 0.8 * x))))})
            x = rng.normal()
            eta = shift_pdoc + slope_pdoc * (a + 0.8 * x)
            prow.append({"patient": f"D{i}", "uid": f"D{i}", "pipe": pipe, "variant": var, "X": x,
                         "call": float(rng.random() < 1 / (1 + np.exp(-eta)))})
        prow.append({"patient": f"D{i}", "uid": f"D{i}", "pipe": "q", "variant": "pdoc_only", "X": 0.0, "call": 0.0})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def test_a2_transport_calibration():
    nv = {"p": 2, "q": 1}
    b, p = _calibration_data(np.random.default_rng(20))
    r = mech.transport(b, p, nv, {"p": 2, "q": 2}, B=40)
    assert r["n_rows_excluded"]["variant_not_in_bdsp"] == 300 and r["n_rows"] == 900
    assert abs(r["calibration_intercept"]) < 0.25 and r["calibration_slope"] == pytest.approx(1.0, abs=0.25)
    assert r["calibration_slope_ci95"][0] < 1.0 < r["calibration_slope_ci95"][1]
    b2, p2 = _calibration_data(np.random.default_rng(21), slope_pdoc=0.4, shift_pdoc=1.0)
    r2 = mech.transport(b2, p2, nv, {"p": 2, "q": 2}, B=40)
    assert r2["calibration_slope"] < 0.7 and r2["calibration_intercept"] > 0.5


def test_calibration_is_exact_for_perfect_predictions():
    rng = np.random.default_rng(22)
    lp = rng.normal(-1, 1.5, 20000)
    y = (rng.random(len(lp)) < 1 / (1 + np.exp(-lp))).astype(float)
    c = mech.calibration(y, lp, np.ones(len(y)))
    assert c["slope"] == pytest.approx(1.0, abs=0.06) and c["intercept"] == pytest.approx(0.0, abs=0.05)


def test_a3_estimates_spearman_auroc_and_adjusted():
    rng = np.random.default_rng(24)
    n = 60
    mcs = np.arange(n) % 2 == 0
    x = rng.normal(size=n) + 0.8 * mcs
    arousal = np.round(2 * x + rng.normal(size=n))
    rec = (x + rng.normal(0, 0.5, n)) > 0.5
    outcome = np.where(rec, "Consc", np.where(np.arange(n) % 3 == 0, "D", "PDOC"))
    P = pd.DataFrame({"patient": [f"P{i}" for i in range(n)], "X": x, "mcs": mcs, "arousal": arousal,
                      "outcome": outcome})
    r = mech.a3_estimates(P, B=200)
    assert r["n"] == n and r["n_recovered"] == int(rec.sum())
    assert r["spearman"]["estimate"] > 0.5 and r["spearman"]["ci95"][0] > 0.2
    assert r["auroc_recovery"]["estimate"] > 0.75 and r["auroc_recovery_no_deaths"]["n"] == int(np.sum(outcome != "D"))
    assert np.isfinite(r["spearman_vs_mcs_adjusted"]["estimate"])
    assert np.isfinite(r["auroc_recovery_vs_mcs_adjusted"]["estimate"])
    assert "not powered for prognosis" in r["wording"]


def test_adjusted_auroc_compares_within_strata_only():
    score = np.array([1, 2, 10, 11], float)
    y = np.array([0, 1, 0, 1], bool)
    strata = np.array(["VS", "VS", "MCS", "MCS"])
    assert mech.adjusted_auroc(score, y, strata) == 1.0
    assert mech._auc(score, y) == 0.75
    assert mech.partial_spearman(np.array([1, 2, 3, 4.0]), np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1.0])) \
        == pytest.approx(1.0)


def test_a7_finds_a_residual_peak_above_the_aperiodic_fit():
    f = np.arange(1, 241) / 480
    rng = np.random.default_rng(25)
    rows = []
    for i in range(40):
        st = "sleep" if i % 2 else "normal_awake"
        P = 1 / (0.01 + f ** 1.2) * np.exp(rng.normal(0, 0.3, len(f)))
        if st == "sleep":
            P = P * (1 + 5 * np.exp(-(f - 0.1) ** 2 / (2 * 0.004 ** 2)))
        rows.append({"pipe": "p", "stratum": st, "transitioning": False, "spec": list(P), "spec_dt": 2.0})
    out = mech.a7_spectra(pd.DataFrame(rows))
    pk = out["p"]["sleep"]["peaks"]
    assert out["p"]["sleep"]["n"] == 20 and abs(pk[0]["f_cycles_per_epoch"] - 0.1) < 0.01
    assert pk[0]["f_hz_approx"] == pytest.approx(pk[0]["f_cycles_per_epoch"] / 2.0)
    assert not any(abs(p["f_cycles_per_epoch"] - 0.1) < 0.01 for p in out["p"]["normal_awake"]["peaks"])
    assert pk[0]["residual_db"] == pytest.approx(10 * np.log10(6), abs=1.5)
    assert all(p["residual_db"] < 2 for p in out["p"]["normal_awake"]["peaks"])


def test_a8_pc1_of_identical_channels_is_the_channel():
    rng = np.random.default_rng(26)
    g = one_over_f(96, rng)
    F = np.vstack([g, g, g])
    assert np.allclose(np.abs(mech.pc1_series(F)), np.abs((g - g.mean()) / g.std()))
    E = np.ones(48)
    E[3] = 50
    assert mech.overlap(mech.pc1_series(F), E)["X"] == pytest.approx(mech.overlap(g, E)["X"])


def test_a8_m1_without_columbia():
    d = pd.DataFrame({"pipe": ["columbia", "cornell", "columbia", "mbmi"]})
    assert list(mech.without_columbia(d)["pipe"]) == ["cornell", "mbmi"]



def _fe_data(rng, n=4000, zero_variant=False):
    fe = rng.integers(0, 5, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(np.array([-2, -1, 0, 0.5, -1.5])[fe] + 0.7 * x)))).astype(float)
    if zero_variant:
        y[fe == 4] = 0.0
    return y, x, fe, rng.uniform(0.2, 1.0, n)


def test_penalized_fe_logit_matches_sklearn_c100():
    from sklearn.linear_model import LogisticRegression
    y, x, fe, w = _fe_data(np.random.default_rng(30), zero_variant=True)
    ours = mech.fe_logit(y, x[:, None], fe, w, lam=1 / mech.PENALTY_C)
    D = np.column_stack([x, np.eye(5)[fe]])
    sk = LogisticRegression(C=mech.PENALTY_C, tol=1e-10, max_iter=10000).fit(D, y, sample_weight=w)
    assert ours["converged"] and ours["n_fe_dropped"] == 0
    assert ours["beta"][0] == pytest.approx(sk.coef_[0][0], abs=1e-4)
    assert np.allclose(ours["alpha"], sk.intercept_[0] + sk.coef_[0][1:], atol=1e-3)


def test_fe_model_penalized_never_fails_and_reports_plain_mle_failures():
    rng = np.random.default_rng(31)
    rows = []
    for i in range(30):
        for v in ("a", "b"):
            rows.append({"patient": f"P{i}", "uid": f"P{i}", "fe": v, "call": 1.0 if i == 0 else
                         float(rng.random() < 0.3), "rare": float(i == 0)})
    d = pd.DataFrame(rows)
    out = mech.fe_model(d, {"rare": d["rare"].to_numpy()}, B=60, name="t")
    assert out["n_boot_failed"] == 0 and out["plain_mle"]["n_boot_failed"] > 0
    assert not out["plain_mle"]["converged"] and np.isfinite(out["coef"]["rare"]["beta"])


def test_weights_divide_by_rows_per_patient_and_variant():
    d = pd.DataFrame({"patient": ["P1", "P1", "P1", "P2"], "pipe": ["t", "t", "t", "t"],
                      "variant": ["v1", "v1", "v2", "v1"], "uid": ["P1"] * 3 + ["P2"]})
    w = mech._prep(d, {"t": 2})["w"].to_numpy()
    assert np.allclose(w, [0.25, 0.25, 0.5, 0.5])


def _rec19(seconds=400, fs=128.0, seed=32):
    x = np.random.default_rng(seed).normal(size=(19, int(seconds * fs))).astype(np.float32) * 10
    return rn.en.Recording(source="bdsp", uid="S", patient="P", x=x, fs=fs, names=list(rn.en.NAMES19))


def test_spatial_transforms_are_the_modules_own():
    rec = _rec19()
    x = np.asarray(rec.x, float)
    cor = rn.load_pipe("cornell")
    rows = mech.module_rows(cor, rec)
    L = mech.mc.hjorth_matrix(cor.NATIVE_CH, 4, 3, "inverse_arc")
    xt, names = mech.spatial(cor, rec, x)
    assert np.allclose(xt, L @ x[rows]) and len(names) == 19
    c12 = rn.load_pipe("cruse12p")
    xt, names = mech.spatial(c12, rec, x)
    r = mech.module_rows(c12, rec)
    assert xt.shape[0] == 2 and np.allclose(xt[0], x[r[0]] - x[r[1]]) and np.allclose(xt[1], x[r[2]] - x[r[3]])
    c11 = rn.load_pipe("cruse11")
    xt, _ = mech.spatial(c11, rec, x)
    Lall = mech.mc.hjorth_matrix(rec.names, k_interior=4, k_edge=4, weights="equal")
    assert np.allclose(xt, (Lall @ x)[mech.module_rows(c11, rec)])
    col = rn.load_pipe("columbia")
    xt, _ = mech.spatial(col, rec, x)
    seg = x[mech.module_rows(col, rec)][:, 1000:1257]
    assert np.allclose(xt[:, 1000:1257], col.ce._csd(seg, *col.ce._CSD))
    for p in ("chennu13", "sokoliuk21"):
        xt, _ = mech.spatial(rn.load_pipe(p), rec, x)
        assert np.allclose(xt.sum(0), 0, atol=1e-6)
    for p in ("mbmi", "dubois", "mbvt", "gibson16"):
        m = rn.load_pipe(p)
        assert np.allclose(mech.spatial(m, rec, x)[0], x[mech.module_rows(m, rec)])


def test_feature_series_sees_the_transformed_channels():
    rec = _rec19(seconds=700)
    cor = rn.load_pipe("cornell")
    spec = rn.variant_specs(cor, "bdsp")["theoretical_bh_keep40_r2"]
    ov = rn._sched(cor, cor.schedule, spec, 20.0, None)
    F = mech.feature_series(rec, cor, mech.module_cfg(cor, spec), ov)
    g = mech.grid_windows(cor, mech.module_cfg(cor, spec), ov, rec.fs)
    xt, _ = mech.spatial(cor, rec, np.asarray(rec.x, float))
    seg, i0 = mech.mc.span(xt, rec.fs, *cor.extent(ov))
    xf = mech.mc.bandpass(seg, rec.fs, *mech.MOTOR_BAND)
    a, b = int(round(g["win"][5, 0] * rec.fs)) - i0, int(round(g["win"][5, 1] * rec.fs)) - i0
    assert np.allclose(F[:, 5], np.log(np.mean(xf[:, a:b] ** 2, 1) + mech.LOG_FLOOR))


def test_masked_series_and_label_weights_are_zero_filled():
    rng = np.random.default_rng(33)
    F = one_over_f(48, rng, n_ch=2) + 5.0
    keep = np.ones(48, bool)
    keep[[3, 17, 30]] = False
    Fm = mech.masked_series(F, keep)
    assert np.allclose(Fm[:, ~keep], 0) and np.allclose(Fm[:, keep].mean(1), 0)
    draws = [dn.schedule("block", g, block=12)[0] for g in [np.random.default_rng(s) for s in range(4)]]
    E = mech.masked_label_spectrum(draws, [((1,), (0,))], keep)
    want = np.mean([mech.label_spectrum(mech.contrast_weights(np.where(keep, d, -999), [((1,), (0,))]))
                    for d in draws], 0)
    assert np.allclose(E, want)
    assert not np.allclose(E, mech.masked_label_spectrum(draws, [((1,), (0,))], np.ones(48, bool)))


def test_module_artefact_mask_comes_from_the_module():
    rec = _rec19(seconds=1300)
    mod = rn.load_pipe("mbmi")
    spec = rn.variant_specs(mod, "bdsp")["gu17_cspfold_peak"]
    ov = rn._sched(mod, mod.schedule, spec, 30.0, np.random.default_rng(0))
    i = int(round((ov.onsets_s[7] + 2.0) * rec.fs))
    rec.x[:, i:i + 20] += 400.0
    keep = mech.artefact_mask(mod, mech.module_cfg(mod, spec), rec, ov)
    assert keep is not None and not keep[7] and keep.sum() >= len(keep) - 2
    assert mech.artefact_mask(rn.load_pipe("dubois"), mech.module_cfg(rn.load_pipe("dubois"),
                              rn.variant_specs(rn.load_pipe("dubois"), "bdsp")["assessment_random_phase1"]),
                              rec, ov) is None


def test_flatness_check():
    rng = np.random.default_rng(34)
    flat = [mech.is_flat(mech.periodogram(rng.normal(size=96))[1]) for _ in range(200)]
    red = [mech.is_flat(mech.periodogram(np.cumsum(rng.normal(size=96)))[1]) for _ in range(50)]
    assert np.mean(flat) > 0.9 and not any(red)
    assert not mech.is_flat(np.zeros(48))


def test_failed_but_flat_channel_enters_as_zero_in_the_sensitivity():
    rng = np.random.default_rng(35)
    E = np.ones(48)
    E[2] = 40
    while True:
        wch = rng.normal(size=96)
        f, P = mech.periodogram(wch)
        if mech.fit_aperiodic(f[~mech.peak_bins(E)], P[~mech.peak_bins(E)])[1] == "fit_degenerate" and mech.is_flat(
                P[~mech.peak_bins(E)]):
            break
    good = [g for g in (one_over_f(96, rng) for _ in range(20)) if np.isfinite(mech.overlap(g, E)["X"])][:2]
    r = mech.overlap(np.vstack(good + [wch]), E)
    assert r["n_ch_fit"] == 2 and r["X_flat0"] == pytest.approx(np.sum([mech.overlap(g, E)["X"] for g in good]) / 3)
    r1 = mech.overlap(np.vstack([good[0], wch]), E)
    assert np.isnan(r1["X"]) and r1["X_flat0"] == pytest.approx(mech.overlap(good[0], E)["X"] / 2)
    r2 = mech.overlap(np.vstack([good[0], np.full(96, 1.0)]), E)
    assert np.isnan(r2["X_flat0"])


def test_channel_use_by_design_and_stratum():
    d = pd.DataFrame({"design_type": ["block", "block", "randomised"], "stratum": ["sleep", "sleep", "sleep"],
                      "immunity_type": ["exposed", "exposed", "immune"], "n_ch": [19, 19, 11],
                      "n_ch_fit": [19, 10, 11]})
    u = mech.channel_use(d)
    assert u["by_design"]["block"] == pytest.approx((1 + 10 / 19) / 2) and u["by_stratum"]["sleep"] > 0.8


def test_edf_prefilters_read_the_signal_header_only(tmp_path, monkeypatch):
    ns = 3
    labels, pre = ["C3", "C4", "EMG"], ["HP:0.5Hz LP:35Hz N:50Hz", "HP:1.5Hz LP:30Hz", ""]
    hdr = b"0       " + b"X" * 176 + f"{256 * (ns + 1):<8}".encode() + b" " * 44 + b"10      " + b"1       " \
        + f"{ns:<4}".encode()
    sig = b"".join(f"{l:<16}".encode() for l in labels) + b" " * (80 + 8 * 5) * ns \
        + b"".join(f"{p:<80}".encode() for p in pre) + b"128     " * ns + b" " * 32 * ns
    fn = tmp_path / "t.edf"
    fn.write_bytes(hdr + sig)
    reads = []
    real_open = open

    class Track:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

        def seek(self, n, *a):
            return self.fh.seek(n, *a)

        def read(self, n):
            pos = self.fh.tell()
            reads.append((pos, pos + n))
            return self.fh.read(n)
    monkeypatch.setattr(mech, "_open_binary", lambda p: Track(real_open(p, "rb")))
    out = mech.edf_prefilters(str(fn))
    assert out == {"C3": {"hp": 0.5, "lp": 35.0}, "C4": {"hp": 1.5, "lp": 30.0}, "EMG": {"hp": None, "lp": None}}
    assert all(b <= 8 or a >= 184 for a, b in reads)


def test_marker_range_clears_the_filter_corners():
    assert mech.marker_range([{"hp": 0.5, "lp": 35.0}]) == ((2.0, 25.0), None)
    (lo, hi), why = mech.marker_range([{"hp": 1.5, "lp": 35.0}, {"hp": 0.5, "lp": 30.0}])
    assert lo == pytest.approx(3.0) and hi == pytest.approx(24.0) and "narrowed" in why
    assert mech.marker_range([{"hp": None, "lp": None}]) == ((2.0, 25.0), None)


def test_fixed_mode_exponent_ignores_an_alpha_peak():
    f = np.arange(2.0, 25.25, 0.25)
    P = f ** -1.6 * (1 + 4 * np.exp(-(f - 10) ** 2 / 2))
    assert mech.fixed_exponent(f, P) == pytest.approx(1.6, abs=0.02)
    assert mech.fixed_exponent(f, f ** -1.6) == pytest.approx(1.6, abs=1e-6)


def test_random_intercept_rows_are_the_default_variant_once_per_patient_and_pipeline():
    d = pd.DataFrame({"patient": ["P1", "P1", "P1", "P1", "P2", "P2"], "pipe": ["a", "a", "a", "b", "a", "b"],
                      "uid": ["U1", "U1", "U2", "U1", "U3", "U3"], "variant": ["x", "d", "d", "e", "d", "e"],
                      "primary_variant": ["d", "d", "d", "e", "d", "e"], "X": [0.2, 0.1, 0.5, 0.3, 0.4, None]})
    r = mech.ri_rows(d)
    want = [("P1", "a", "d", "U1"), ("P1", "b", "e", "U1"), ("P2", "a", "d", "U3")]
    assert list(zip(r["patient"], r["pipe"], r["variant"], r["uid"])) == want


def test_immunity_typing_is_mechanistic():
    def draw_1in8(g):
        return mech.contrast_weights(g.permutation(np.r_[np.ones(60, int), np.zeros(420, int)]), [((1,), (0,))])
    E = mech.expected_spectrum(draw_1in8, n_draws=300).at(480)
    assert mech.immunity_type(mech.x_reference(E)) == "immune"
    E12 = mech.expected_spectrum(b12_draw).at(48)
    assert mech.immunity_type(mech.x_reference(E12)) == "exposed"
    assert mech.immunity_type(np.nan) is None
    assert [mech.immunity_type(v) for v in (-0.5, -0.05, 0.05, 0.1, 0.11)] == ["exposed", "immune", "immune", "immune",
                                                                               "exposed"]
    mbvt = rn.load_pipe("mbvt")
    ds = mech.design_spectrum(mbvt, rn.variant_specs(mbvt, "bdsp")["vt3_mur20_ds7_full_gt23"], "published", n_draws=60)
    x = mech.x_reference(ds.at(ds.n))
    assert x < -0.1 and mech.immunity_type(x) == "exposed"


def test_three_way_typing_from_x_ref():
    mbvt = rn.load_pipe("mbvt")
    ds = mech.design_spectrum(mbvt, rn.variant_specs(mbvt, "bdsp")["vt3_mur20_ds7_full_gt23"], "published", n_draws=60)
    assert mech.x_ref_class(mech.x_reference(ds.at(ds.n))) == "deflated"

    def draw_1in8(g):
        return mech.contrast_weights(g.permutation(np.r_[np.ones(60, int), np.zeros(420, int)]), [((1,), (0,))])
    assert mech.x_ref_class(mech.x_reference(mech.expected_spectrum(draw_1in8, n_draws=300).at(480))) == "immune"
    assert mech.x_ref_class(mech.x_reference(mech.expected_spectrum(b12_draw).at(48))) == "inflated"
    assert [mech.x_ref_class(v) for v in (-0.11, -0.1, 0.0, 0.1, 0.11)] == ["deflated", "immune", "immune", "immune",
                                                                          "inflated"]
    assert mech.x_ref_class(np.nan) is None
    tab = mech.design_table(["mbmi", "dubois"], n_draws=60)
    pub = tab[tab["kind"] == "published"]
    assert (pub[pub["pipe"] == "mbmi"]["x_ref_class"] == "immune").all()
    assert (pub[pub["variant"].str.startswith("assessment")]["x_ref_class"] == "inflated").all()


def test_deflated_report_uses_deflated_rows_only():
    rng = np.random.default_rng(39)
    rows = []
    for i in range(120):
        for pipe, cls, v, p_call in (("a", "deflated", "va", 0.04), ("a", "deflated", "vd", 0.02),
                                     ("b", "immune", "vb", 0.5), ("c", "inflated", "vc", 0.9)):
            rows.append({"null": "nstim0_bdsp", "pipe": pipe, "variant": v, "fe": f"{pipe}|{v}", "patient": f"P{i}",
                         "uid": f"P{i}", "x_ref_class": cls, "chi": rng.normal(1.5, 0.3), "w": 0.5,
                         "call": bool(rng.random() < p_call)})
    d = pd.DataFrame(rows)
    choices = {("nstim0_bdsp", "a"): {"validity_primary": "va", "default": "vd"},
               ("nstim0_bdsp", "b"): {"validity_primary": "vb", "default": "vb"},
               ("nstim0_bdsp", "c"): {"validity_primary": "vc", "default": "vc"}}
    out = mech.deflated_report(d, choices, B=40)
    assert out["pipelines"] == ["a"] and set(out["rates"]) == {"a"} and out["nominal"] == 0.05
    for role, v in (("validity_primary", "va"), ("default", "vd")):
        r = out["rates"]["a"][role]
        want = d[(d["pipe"] == "a") & (d["variant"] == v)]["call"].mean()
        assert r["variant"] == v and r["n"] == 120 and r["rate"] == pytest.approx(want) and r["ci95"] is not None
    assert out["chi_slope"]["n_rows"] == 240 and "z_chi" in out["chi_slope"]["coef"]
    d2 = d.assign(x_ref_class=np.where(d["pipe"] == "a", "immune", d["x_ref_class"]))
    assert mech.deflated_report(d2, choices, B=40)["pipelines"] == []
    tab = mech.design_table(["mbmi", "dubois"], n_draws=60)
    pub = tab[tab["kind"] == "published"]
    assert (pub[pub["pipe"] == "mbmi"]["immunity_type"] == "immune").all()
    assert (pub[pub["variant"].str.startswith("assessment")]["immunity_type"] == "exposed").all()


def _a3_inputs(tmp_path, rng, n=30):
    cases = [str(i + 1) for i in range(n)]
    mp = pd.DataFrame({"case": cases + ["99", "98"],
                       "uid": [f"Patient_{i + 1}_{'VS' if i % 2 else 'MCS+'}" for i in range(n)] + [None, None],
                       "status": ["matched"] * n + ["ambiguous", "unmatched"]})
    mp.to_csv(tmp_path / "map.csv", index=False)
    oc = pd.DataFrame({"case": cases + ["99", "98"], "state_of_consciousness": "VS", "age": 50,
                       mech.A3_AROUSAL: rng.integers(0, 4, n + 2), "crs_r_arousal_last": 3,
                       mech.A3_OUTCOME: [["Consc", "PDOC", "D"][i % 3] for i in range(n + 2)]})
    oc.to_csv(tmp_path / "outcomes.csv", index=False)
    return str(tmp_path / "map.csv"), str(tmp_path / "outcomes.csv"), oc


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_a3_input_constants_and_the_committed_mapping():
    assert mech.A3_MAP_SHA256 == "c959cce7c9a856f7ccf3039b83a0cbca0aa07e005a32590c9d82e6430d04f212"
    assert mech.A3_OUTCOMES_SHA256 == "7f601f6e035d07b6eb0358a6610508facc2c842c6b0050c567ce4dfa17c6c7fe"
    assert mech.A3_MAP.endswith(os.path.join("docs", "pdoc-case-map.csv"))
    assert mech.A3_OUTCOMES.endswith(os.path.join("output", "raw", "pdoc_clinical", "pdoc_outcomes.csv"))
    assert mech.is_committed(mech.A3_MAP) and _sha(mech.A3_MAP) == mech.A3_MAP_SHA256


def test_is_committed(tmp_path):
    assert mech.is_committed(mech.K_TABLE)
    f = tmp_path / "x.csv"
    f.write_text("case,uid,status\n")
    assert not mech.is_committed(str(f))


def test_a3_reads_outcomes_only_after_every_check(tmp_path):
    m, o, oc = _a3_inputs(tmp_path, np.random.default_rng(36))
    ok = dict(map_sha=_sha(m), outcome_sha=_sha(o), committed=lambda p: True)
    for bad, match in ((dict(ok, committed=lambda p: False), "committed"), (dict(ok, map_sha="0" * 64), "sha256"),
                       (dict(ok, outcome_sha="0" * 64), "sha256")):
        with pytest.raises(RuntimeError, match=match):
            mech.a3_table(m, o, **bad)
    t, counts = mech.a3_table(m, o, **ok)
    assert counts == {"matched": 30, "ambiguous": 1, "unmatched": 1} and len(t) == 30
    first = dict(zip(oc["case"], oc[mech.A3_AROUSAL]))
    assert all(t.set_index("case")["arousal"][c] == first[c] for c in t["case"])
    assert set(t["patient"]) == {f"Patient_{i + 1}" for i in range(30)}
    assert not {"state_of_consciousness", "age", "crs_r_arousal_last"} & set(t.columns)
    with open(o, "a") as fh:
        fh.write("31,VS,50,2,3,Consc\n")
    with pytest.raises(RuntimeError, match="sha256"):
        mech.a3_table(m, o, **ok)
    import inspect
    sig = inspect.signature(mech.a3_table).parameters
    assert sig["map_sha"].default == mech.A3_MAP_SHA256 and sig["outcome_sha"].default == mech.A3_OUTCOMES_SHA256
    assert sig["committed"].default is mech.is_committed


def test_a3_is_not_run_without_the_release_flag(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(mech, "a3_table", lambda *a, **k: opened.append(a))
    assert "not run" in mech.a3(pd.DataFrame(), release=False)["note"] and not opened


def test_design_draws_are_the_design_spectrums_draws():
    for pipe, v in (("mbmi", "gu17_cspfold_peak"), ("cornell", "theoretical_bh_keep40_r2")):
        mod = rn.load_pipe(pipe)
        spec = rn.variant_specs(mod, "bdsp")[v]
        ds = mech.design_spectrum(mod, spec, "published", n_draws=40)
        dr = mech.design_draws(mod, spec, "published", n_draws=40)
        assert list(dr) == [ds.n] and len(dr[ds.n]) == 40
        cons = mech.contrasts(mod, mech.module_cfg(mod, spec))
        assert np.allclose(mech.masked_label_spectrum(dr[ds.n], cons, np.ones(ds.n, bool)), ds.at(ds.n))


def test_resting_markers_use_the_given_band():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(2, 128 * 480))
    _, ra = mech.resting_markers(x, 128.0, (2.0, 25.0))
    f = np.arange(0, 64.25, 0.25)
    want = np.sum((f >= 8) & (f <= 12)) / np.sum((f >= 2) & (f <= 25))
    assert ra == pytest.approx(want, rel=0.1)
    _, ra2 = mech.resting_markers(x, 128.0, (3.0, 24.0))
    assert ra2 > ra and all(np.isnan(v) for v in mech.resting_markers(x, 128.0, None))


def test_immunity_check_uses_the_mechanistic_typing():
    rng = np.random.default_rng(38)
    rows = []
    for i in range(80):
        for pipe, typ, dsg in (("a", "immune", "randomised"), ("b", "exposed", "randomised"),
                               ("c", "exposed", "block")):
            rows.append({"patient": f"P{i}", "uid": f"P{i}", "pipe": pipe, "variant": "v", "fe": pipe + "|v",
                         "immunity_type": typ, "design_type": dsg, "chi": rng.normal(1.5, 0.3),
                         "call": float(rng.random() < 0.2), "w": 1.0})
    out = mech.immunity(pd.DataFrame(rows), pd.DataFrame(), B=30)
    assert out["immune_designs"]["n_rows"] == 80 and out["pipelines_immune"] == ["a"]
    assert out["slope_contrast"]["n_rows"] == 240 and "z_chi_x_exposed" in out["slope_contrast"]["coef"]


def test_mask_cache_keeps_variant_settings_apart():
    rec = _synthetic_recording(seconds=700)
    cor = rn.load_pipe("cornell")
    specs = rn.variant_specs(cor, "bdsp")
    ov = rn._sched(cor, cor.schedule, specs["theoretical_bh_keep40_r2"], 30.0, rn._rng("cornell", "SYN", 0, 1))
    rx = mech.RecordingX(rec, mech.Spectra(None, 30))
    k40 = rx.mask(cor, mech.module_cfg(cor, specs["theoretical_bh_keep40_r2"]), ov)
    thr = rx.mask(cor, mech.module_cfg(cor, specs["theoretical_bh_thresh_r2"]), ov)
    assert k40.sum() == 4 * int(round(cor.KEEP_FRAC * 24)) and thr.sum() > k40.sum()
