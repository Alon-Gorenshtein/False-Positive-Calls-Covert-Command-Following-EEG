"""Tests for 128_nonstationarity.py (part C2): synthetic recordings and synthetic shard rows only."""
import importlib
import json
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

ns = importlib.import_module("128_nonstationarity")
mech = ns.mech
rn = ns.rn


def pink(n, fs, rng, chi=1.0):
    """Unit-SD Gaussian noise with a 1/f^chi power spectrum (DC removed)."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-chi / 2.0)
    x = np.fft.irfft(amp * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))), n=n)
    return x / x.std()


def envelope(n, fs, log_sd, tau_s, rng):
    """Log-normal amplitude envelope exp(log_sd g - log_sd^2 / 2), g a unit-variance Ornstein-Uhlenbeck process with
    correlation time tau_s; mean 1."""
    if log_sd == 0:
        return np.ones(n)
    a = math.exp(-1.0 / (tau_s * fs))
    g = lfilter([math.sqrt(1.0 - a * a)], [1.0, -a], rng.normal(size=n), zi=[a * rng.normal()])[0]
    return np.exp(log_sd * g - log_sd ** 2 / 2.0)


def eeg_like_noise(seconds, fs=128.0, n_ch=19, log_sd=0.0, tau_s=10.0, rng=None, n_src=6):
    """Mixed pink sources, an alpha source and sensor noise, times one log-normal amplitude envelope."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = int(round(seconds * fs))
    src = np.array([pink(n, fs, rng) for _ in range(n_src)])
    mix = rng.normal(size=(n_ch, n_src)) / math.sqrt(n_src)
    alpha = ns.mc.bandpass(rng.normal(size=n), fs, 8.0, 12.0)
    alpha /= alpha.std()
    x = mix @ src + (0.5 * rng.uniform(0, 1, n_ch))[:, None] * alpha[None] + 0.2 * rng.normal(size=(n_ch, n))
    return 20.0 * x * envelope(n, fs, log_sd, tau_s, rng)[None]


def _rec(x, uid="SYN", fs=128.0):
    return rn.en.Recording(source="bdsp", uid=uid, patient="P" + uid, x=np.asarray(x, np.float32), fs=fs,
                           names=list(rn.en.NAMES19))


def _noise(seconds, log_sd=0.0, tau=30.0, seed=0):
    return eeg_like_noise(seconds, log_sd=log_sd, tau_s=tau, rng=np.random.default_rng(seed))


def _overlay(pipe, variant, start=30.0, uid="SYN"):
    mod = rn.load_pipe(pipe)
    spec = rn.variant_specs(mod, "bdsp")[variant]
    return mod, mech.module_cfg(mod, spec), rn._sched(mod, mod.schedule, spec, start, rn._rng(pipe, uid, 0, 1))



def test_envelope_has_the_stated_log_sd_and_correlation_time():
    fs, n = 128.0, 128 * 3600
    e = envelope(n, fs, 0.5, 10.0, np.random.default_rng(1))
    g = np.log(e)
    assert g.std() == pytest.approx(0.5, rel=0.1) and np.mean(e) == pytest.approx(1.0, abs=0.05)
    lag = int(10 * fs)
    r = np.corrcoef(g[:-lag], g[lag:])[0, 1]
    assert r == pytest.approx(np.exp(-1), abs=0.1)
    assert np.all(envelope(n, fs, 0.0, 10.0, np.random.default_rng(1)) == 1.0)


def test_eeg_like_noise_shape_and_seed():
    a = _noise(60, 0.5, seed=3)
    assert a.shape == (19, 60 * 128) and np.all(np.isfinite(a))
    assert np.array_equal(a, _noise(60, 0.5, seed=3)) and not np.array_equal(a, _noise(60, 0.5, seed=4))



@pytest.mark.parametrize("pipe,variant", [("cornell", "theoretical_bh_keep40_r2"), ("cruse12p", "random_perclass"),
                                          ("columbia", "as_coded"), ("edlow17", "permsecond_cvsecond_alt")])
def test_motor_log_feature_is_116s(pipe, variant):
    rec = _rec(_noise(2000, seed=5))
    mod, cfg, ov = _overlay(pipe, variant)
    seg, i0 = mech.mc.span(np.asarray(rec.x), rec.fs, *mod.extent(ov))
    F = ns.epoch_log_power(rec, mod, cfg, ov, np.asarray(seg, float))
    assert np.allclose(F, mech.feature_series(rec, mod, cfg, ov))


def test_erp_log_feature_is_log_power_on_the_116_grid():
    rec = _rec(_noise(1200, seed=6))
    mod, cfg, ov = _overlay("gibson16", "p3b_two025")
    seg, i0 = mech.mc.span(np.asarray(rec.x), rec.fs, *mod.extent(ov))
    F = ns.epoch_log_power(rec, mod, cfg, ov, np.asarray(seg, float))
    g = mech.grid_windows(mod, cfg, ov, rec.fs)
    xf = mech.mc.bandpass(mech.spatial(mod, rec, np.asarray(seg, float))[0], rec.fs, *mech.ERP_BAND)
    a, b = int(round(g["win"][7, 0] * rec.fs)) - i0, int(round(g["win"][7, 1] * rec.fs)) - i0
    assert F.shape[1] == len(g["win"]) and np.allclose(F[:, 7], np.log(np.mean(xf[:, a:b] ** 2, 1) + mech.LOG_FLOOR))


def test_surrogates_are_117s_phase_randomisation_seeded_by_uid_and_span():
    assert ns.phase_randomise is ns.m117.phase_randomise
    seg = _noise(20, seed=7)
    s1 = ns.surrogates(seg, "U1", 100, 2660, n=3)
    s2 = ns.surrogates(seg, "U1", 100, 2660, n=3)
    s3 = ns.surrogates(seg, "U2", 100, 2660, n=3)
    assert len(s1) == 3 and all(np.array_equal(a, b) for a, b in zip(s1, s2))
    assert not np.allclose(s1[0], s3[0]) and not np.allclose(s1[0], s1[1])
    P = lambda z: np.abs(np.fft.rfft(z, axis=1)) ** 2
    X = np.fft.rfft(seg, axis=1)
    S = np.fft.rfft(s1[0], axis=1)
    assert np.allclose(P(s1[0]), P(seg)) and np.allclose(S[0] * np.conj(S[1]), X[0] * np.conj(X[1]))


def _mean_logver(pipe, variant, log_sd, seeds, seconds=2000, tau=30.0):
    mod, cfg, ov = _overlay(pipe, variant)
    out = []
    for s in seeds:
        rec = _rec(_noise(seconds, log_sd, tau, seed=s), uid=f"S{s}")
        out.append(ns.window_ver(rec, mod, cfg, ov)["log_ver"])
    return np.array(out)


@pytest.mark.parametrize("pipe,variant", [("cornell", "theoretical_bh_keep40_r2"), ("gibson16", "p3b_two025")])
def test_ver_is_about_one_when_stationary_and_above_one_under_an_envelope(pipe, variant):
    st = _mean_logver(pipe, variant, 0.0, range(6))
    mo = _mean_logver(pipe, variant, 0.5, range(6))
    assert abs(st.mean()) < 0.15 and np.all(np.abs(st) < 0.4)
    assert mo.mean() > 0.5 and np.all(mo > st.max())


def test_ver_rises_with_the_envelope_log_sd():
    m = [np.mean(_mean_logver("cruse12p", "random_perclass", sd, range(3))) for sd in (0.0, 0.25, 0.5, 0.75)]
    assert all(b > a for a, b in zip(m, m[1:]))


def test_ver_is_seed_deterministic_and_order_free():
    mod, cfg, ov = _overlay("cornell", "theoretical_bh_keep40_r2")
    rec = _rec(_noise(2000, 0.5, seed=11), uid="DET")
    a, b = ns.window_ver(rec, mod, cfg, ov), ns.window_ver(rec, mod, cfg, ov)
    assert a["log_ver"] == b["log_ver"] and a["n_surr"] == ns.N_SURR == 10
    rec2 = _rec(np.asarray(rec.x), uid="OTHER")
    assert ns.window_ver(rec2, mod, cfg, ov)["log_ver"] != a["log_ver"]


def test_constant_channel_is_left_out_and_all_constant_is_na():
    mod, cfg, ov = _overlay("cornell", "theoretical_bh_keep40_r2")
    x = _noise(2000, seed=12)
    x[:] = 0.0
    r = ns.window_ver(_rec(x), mod, cfg, ov)
    assert r["log_ver"] is None and r["reason"] == "no_variance"



def _row(null, pipe, variant, uid, j, **kw):
    r = {"key": f"{uid}|{null}|{pipe}|{variant}|{j}", "null": null, "pipe": pipe, "variant": variant, "uid": uid,
         "patient": "P" + uid, "j": j, "status": "ok", "resp": "none", "montage": "native", "seq": True,
         "overlap": False,
         "split": "analysis", "dev": False, "source": "bdsp", "call": bool(j % 2), "p": 0.5, "stat": 1.0,
         "start_s": 30.0, "extent_t0": 30.0, "extent_t1": 600.0, "n_onsets": 10}
    r.update(kw)
    return r


def _shards(tmp_path):
    rows = {"ncmd_bdsp__cornell__native__none": [_row("ncmd_bdsp", "cornell", "v1", u, j) for u in "AB" for j in (0, 1)]
            + [_row("ncmd_bdsp", "cornell", "v1", "C", 0, status="na: no_usable_trials")]
            + [_row("ncmd_bdsp", "cornell", "v1", "A", 0)],
            "nstim0_bdsp__mbvt__native__none": [_row("nstim0_bdsp", "mbvt", "w", u, 0) for u in "AB"]
            + [_row("nstim0_bdsp", "mbvt", "w", "A", 0, key="A|x", seq=False)],
            "ncmd_bdsp__cornell__native__grid": [_row("ncmd_bdsp", "cornell", "v1", "A", 0, key="g", resp="grid")],
            "npassive_kojimaB__mbvt__native__none": [_row("npassive_kojimaB", "mbvt", "w", "K", 0, source="kojimaB")],
            "nstim0_pdoc__mbvt__native__none": [_row("nstim0_pdoc", "mbvt", "w", "Patient_1_VS", 0, source="pdoc")]}
    for d, rr in rows.items():
        os.makedirs(tmp_path / d)
        with open(tmp_path / d / "rows_0.jsonl", "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rr) + "\n")
    return str(tmp_path)


def test_selective_loader_gives_116s_first_overlay_rows(tmp_path):
    root = _shards(tmp_path)
    raw = ns.load_bdsp_rows("analysis", root=root)
    assert (raw["j"] == 0).all() and set(raw["null"]) == {"ncmd_bdsp", "nstim0_bdsp"}
    got = ns.first_overlay_rows(raw)
    want = mech.m1_rows(mech.mg.load_rows("analysis", root=root))
    assert sorted(got["key"]) == sorted(want["key"]) == sorted(
        ["A|ncmd_bdsp|cornell|v1|0", "B|ncmd_bdsp|cornell|v1|0", "A|nstim0_bdsp|mbvt|w|0", "B|nstim0_bdsp|mbvt|w|0"])


def test_atlas_rows_carry_no_outcome(tmp_path):
    d = ns.load_bdsp_rows("analysis", root=_shards(tmp_path), outcomes=False)
    assert not set(ns.OUTCOME_FIELDS) & set(d.columns) and len(d)


def test_loader_refuses_dev_rows_in_analysis(tmp_path):
    root = _shards(tmp_path)
    fn = os.path.join(root, "ncmd_bdsp__cornell__native__none", "rows_0.jsonl")
    with open(fn, "a") as fh:
        fh.write(json.dumps(_row("ncmd_bdsp", "cornell", "v1", "D", 0, dev=True)) + "\n")
    with pytest.raises(RuntimeError, match="development"):
        ns.load_bdsp_rows("analysis", root=root)


def test_non_immune_rows_only():
    d = pd.DataFrame({"pipe": ["mbmi", "dubois", "dubois", "cornell", "sokoliuk21"],
                      "variant": ["a", "training_x", "assessment_x", "c", "s"]})
    typ = {("mbmi", "a"): "immune", ("dubois", "training_x"): "immune", ("dubois", "assessment_x"): "exposed",
           ("cornell", "c"): "exposed", ("sokoliuk21", "s"): None}
    r = ns.non_immune(d, typ)
    assert list(r["pipe"]) == ["dubois", "cornell", "sokoliuk21"] and list(r["variant"]) == ["assessment_x", "c", "s"]


def test_typing_comes_from_the_116_design_table():
    tab = mech.design_table(["mbmi", "cornell"], n_draws=40)
    typ = ns.typing_from_table(tab)
    assert set(v for (p, _), v in typ.items() if p == "mbmi") == {"immune"}
    assert set(v for (p, _), v in typ.items() if p == "cornell") == {"exposed"}



def _model_rows(rng, beta, n_pat=300):
    rows = []
    for i in range(n_pat):
        lv = rng.normal(0.3, 0.4)
        for pipe, nv, base in (("cornell", 2, -2.0), ("mbvt", 3, -2.5)):
            for v in range(nv):
                z = (lv - 0.3) / 0.4
                rows.append({"patient": f"P{i}", "uid": f"P{i}", "pipe": pipe, "variant": f"{pipe}{v}",
                             "log_ver": lv, "call": bool(rng.random() < 1 / (1 + np.exp(-(base + beta * z))))})
    return pd.DataFrame(rows)


def test_model_recovers_a_planted_ver_effect_and_stays_null_without_one():
    rng = np.random.default_rng(13)
    d = _model_rows(rng, np.log(1.6))
    out = ns.ver_model(d, {"cornell": 2, "mbvt": 3}, B=200, name="t")
    c = out["coef"]["z_log_ver"]
    assert c["or"] > 1.3 and c["or_ci95"][0] > 1 and c["p_greater"] < 0.01 and out["n_boot_failed"] == 0
    c0 = ns.ver_model(_model_rows(rng, 0.0), {"cornell": 2, "mbvt": 3}, B=200, name="t")["coef"]["z_log_ver"]
    assert c0["or_ci95"][0] < 1 < c0["or_ci95"][1]


def test_variant_weights_count_each_pipeline_equally():
    d = _model_rows(np.random.default_rng(14), 0.0, n_pat=10)
    w = ns.prep(d, {"cornell": 2, "mbvt": 3})
    assert np.allclose(w.groupby("pipe")["w"].sum().to_numpy(), 10.0)
    assert ns.variants_present(d) == {"cornell": 2, "mbvt": 3}


def test_fresh_bar():
    m = lambda o, lo: {"coef": {"z_log_ver": {"or": o, "or_ci95": [lo, o + 0.2], "p_greater": 0.01}}}
    assert ns.fresh_bar(m(1.25, 1.05))["confirmed"] is True
    assert ns.fresh_bar(m(1.19, 1.05))["confirmed"] is False
    assert ns.fresh_bar(m(1.30, 0.99))["confirmed"] is False
    assert ns.fresh_bar({"note": "not estimated"})["confirmed"] is False





def test_results_are_whole_files_of_one_code_version(tmp_path):
    d = str(tmp_path / "ver")
    ns.write_result(d, "U1", [{"key": "k", "log_ver": 0.1}])
    assert ns.done_uids(d) == {"U1"} and ns.read_results(d)["log_ver"].tolist() == [0.1]
    fn = os.path.join(d, "U1.json")
    o = json.load(open(fn))
    o["sha256_128"] = "0" * 64
    json.dump(o, open(fn, "w"))
    with pytest.raises(RuntimeError, match="another version"):
        ns.done_uids(d)


def test_recording_ver_shares_windows_across_variants():
    rec = _rec(_noise(2600, 0.25, seed=15), uid="REC")
    mod = rn.load_pipe("cornell")
    specs = rn.variant_specs(mod, "bdsp")
    rows = []
    for v in ("theoretical_bh_keep40_r2", "jackknife_bh_keep40_r2", "theoretical_bh_keep40_r4"):
        ov = rn._sched(mod, mod.schedule, specs[v], 30.0, rn._rng("cornell", "REC", 0, 1))
        t0, t1 = mod.extent(ov)
        rows.append({"key": f"REC|{v}", "uid": "REC", "patient": "PREC", "pipe": "cornell", "variant": v, "j": 0,
                     "start_s": 30.0, "extent_t0": t0, "extent_t1": t1, "n_onsets": len(ov.onsets_s)})
    out = ns.recording_ver(rec, pd.DataFrame(rows))
    by = {r["variant"]: r for r in out}
    assert by["theoretical_bh_keep40_r2"]["log_ver"] == by["jackknife_bh_keep40_r2"]["log_ver"]
    assert by["theoretical_bh_keep40_r4"]["n_epochs"] != by["theoretical_bh_keep40_r2"]["n_epochs"]
    assert all("call" not in r for r in out)


def test_window_ver_uses_ten_surrogates_of_the_raw_window(monkeypatch):
    mod, cfg, ov = _overlay("cornell", "theoretical_bh_keep40_r2")
    rec = _rec(_noise(2000, seed=16), uid="RAW")
    seen = {}
    real = ns.surrogates

    def spy(seg, uid, i0, i1, n=ns.N_SURR):
        seen.update(shape=seg.shape, n=n, uid=uid, span=(i0, i1))
        return real(seg, uid, i0, i1, n)
    monkeypatch.setattr(ns, "surrogates", spy)
    r = ns.window_ver(rec, mod, cfg, ov)
    t0, t1 = mod.extent(ov)
    assert seen["n"] == 10 and seen["uid"] == "RAW" and seen["shape"][0] == 19
    assert seen["span"] == (r["span_i0"], r["span_i1"]) and seen["span"][0] == int(np.floor(t0 * rec.fs))


def test_a_variant_without_a_typing_is_refused():
    with pytest.raises(RuntimeError, match="typing"):
        ns.non_immune(pd.DataFrame({"pipe": ["cornell"], "variant": ["new"]}), {("cornell", "old"): "exposed"})


def test_ver_model_is_116s_weighted_fe_model():
    d = _model_rows(np.random.default_rng(17), np.log(1.3), n_pat=120)
    d.loc[d["patient"] == "P0", "log_ver"] = np.nan
    out = ns.ver_model(d, {"cornell": 2, "mbvt": 3}, B=50, name="t")
    e = mech._prep(d[d["log_ver"].notna()], {"cornell": 2, "mbvt": 3})
    z, _, _ = mech.zscore(e["log_ver"].astype(float), e["w"])
    ref = mech.fe_model(e, {"z_log_ver": z}, 50, "t", e["w"])
    assert out["coef"]["z_log_ver"]["beta"] == pytest.approx(ref["coef"]["z_log_ver"]["beta"])
    assert out["coef"]["z_log_ver"]["or_ci95"] == pytest.approx(ref["coef"]["z_log_ver"]["or_ci95"])
    assert out["n_rows"] == len(d) - 5


def test_analysis_rows_swap_columbias_first_overlays_for_113s(monkeypatch):
    rows = pd.DataFrame([
        {"key": "c0", "null": "ncmd_bdsp", "pipe": "columbia", "resp": "none", "j": 0, "uid": "U1"},
        {"key": "k0", "null": "ncmd_bdsp", "pipe": "cornell", "resp": "none", "j": 0, "uid": "U1"}])
    seen = {}
    monkeypatch.setattr(ns.mg, "drop_touched", lambda df: (df[df["uid"] != "T"], {}))
    monkeypatch.setattr(ns.rn, "columbia113", lambda: "C")
    monkeypatch.setattr(ns.mg, "dev_uids", lambda: set())
    monkeypatch.setattr(ns.mg, "touched_uids", lambda: set())
    monkeypatch.setattr(ns.mg, "check_113", lambda c, df, dev, t: seen.setdefault("checked", list(df["key"])))
    monkeypatch.setattr(ns.mg, "rows_from_113", lambda c: pd.DataFrame([{"key": "113|U1", "null": "ncmd_bdsp",
                                                                        "pipe": "columbia", "j": 0, "uid": "U1",
                                                                        "call": True, "p": 0.01}]))
    out = ns.analysis_rows(pd.concat([rows, pd.DataFrame([{"key": "t0", "null": "ncmd_bdsp", "pipe": "cornell",
                                                            "resp": "none", "j": 0, "uid": "T"}])]))
    assert sorted(out["key"]) == ["113|U1", "k0"] and seen["checked"] == ["k0"] and "p" not in out
    assert "call" in out and "call" not in ns.analysis_rows(rows, outcomes=False)


def test_ver_stage_then_stats_end_to_end(tmp_path, monkeypatch):
    mod = rn.load_pipe("cornell")
    specs = rn.variant_specs(mod, "bdsp")
    recs, rows = {}, []
    for i in range(4):
        uid = f"E{i}"
        recs[uid] = _rec(_noise(1200, 0.25 * i, seed=20 + i), uid=uid)
        for v in ("theoretical_bh_keep40_r2", "jackknife_bh_keep40_r2"):
            ov = rn._sched(mod, mod.schedule, specs[v], 30.0, rn._rng("cornell", uid, 0, 1))
            t0, t1 = mod.extent(ov)
            rows.append({"key": f"{uid}|{v}", "uid": uid, "patient": "P" + uid, "pipe": "cornell", "variant": v, "j": 0,
                         "start_s": 30.0, "extent_t0": t0, "extent_t1": t1, "n_onsets": len(ov.onsets_s),
                         "call": i >= 2})
    rows = pd.DataFrame(rows)
    monkeypatch.setattr(ns.en, "_load", lambda it: recs[it["uid"]])
    d = str(tmp_path / "ver")
    ns.run_ver(rows.drop(columns="call"), {u: {"uid": u} for u in recs}, d)
    assert ns.done_uids(d) == set(recs)
    ns.run_ver(rows.drop(columns="call"), {u: {"uid": u} for u in recs}, d)
    out, merged = ns.stats(rows, ns.read_results(d), {"cornell": 2}, 30, "t", "test")
    assert out["rows"] == {"n_rows": 8, "n_na": 0, "by_reason": {}} and "coef" in out["model"]
    assert merged["log_ver"].notna().all() and len(merged) == 8
    with pytest.raises(RuntimeError, match="not in the listing"):
        ns.run_ver(rows.drop(columns="call"), {"E0": {"uid": "E0"}}, str(tmp_path / "ver2"))



def _fresh_shards(tmp_path):
    rows = {"ncmd_bdsp_fresh__cornell__native__none": [_row("ncmd_bdsp_fresh", "cornell", v, u, j, split="c2_fresh")
                                                       for u in "AB" for v in ("vp", "df", "other") for j in (0, 1)],
            "nstim0_bdsp_fresh__mbvt__native__none": [_row("nstim0_bdsp_fresh", "mbvt", "w", u, 0, split="c2_fresh")
                                                      for u in "AB"],
            "ncmd_bdsp__cornell__native__none": [_row("ncmd_bdsp", "cornell", "vp", "Z", 0)]}
    for d, rr in rows.items():
        os.makedirs(tmp_path / d)
        with open(tmp_path / d / "rows00.jsonl", "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rr) + "\n")
    return str(tmp_path)


def test_fresh_rows_are_the_fresh_nulls_first_overlays(tmp_path):
    root = _fresh_shards(tmp_path)
    d = ns.first_overlay_rows(ns.load_bdsp_rows("c2_fresh", root=root, nulls=ns.FRESH_NULLS), nulls=ns.FRESH_NULLS)
    assert len(d) == 8 and set(d["null"]) == set(ns.FRESH_NULLS) and (d["j"] == 0).all()
    with pytest.raises(RuntimeError, match="split"):
        ns.load_bdsp_rows("analysis", root=root, nulls=ns.FRESH_NULLS)


def test_fresh_model_keeps_validity_primaries_and_defaults():
    d = pd.DataFrame({"null": ["ncmd_bdsp_fresh"] * 3 + ["nstim0_bdsp_fresh"],
                      "pipe": ["cornell"] * 3 + ["mbvt"], "variant": ["vp", "df", "other", "w"]})
    roles = {("cornell", "vp"): ["validity_primary"], ("cornell", "df"): ["default"],
             ("mbvt", "w"): ["validity_primary", "default"]}
    r = ns.c2_variants(d, roles)
    assert list(r["variant"]) == ["vp", "df", "w"]
    assert ns.variants_present(r) == {"cornell": 2, "mbvt": 1}


def test_a_shortened_twin_counts_as_its_role_variant():
    d = pd.DataFrame({"pipe": ["cruse12p"] * 3, "variant": ["pooled_b4", "perclass_b4", "other_b4"],
                      "twin_of": [None, "perclass", "other"]})
    roles = {("cruse12p", "pooled_b4"): ["validity_primary"], ("cruse12p", "perclass"): ["default"]}
    r = ns.c2_variants(d, roles)
    assert list(r["variant"]) == ["pooled_b4", "perclass_b4"] and ns.variants_present(r) == {"cruse12p": 2}
    assert list(ns.c2_variants(d.drop(columns="twin_of"), roles)["variant"]) == ["pooled_b4"]


def test_fresh_items_from_the_pinned_list(tmp_path, monkeypatch):
    for sub in ("sig", "morgoth"):
        os.makedirs(tmp_path / "c2" / sub)
    for uid in ("Bonobo1_1", "Bonobo2_3"):
        (tmp_path / "c2" / "sig" / f"{uid}.npz").write_bytes(b"")
        (tmp_path / "c2" / "morgoth" / f"{uid}.parquet").write_bytes(b"")
    man = tmp_path / "c2_fresh_400.csv"
    pd.DataFrame({"rank": [2, 1], "patient_id": ["Bonobo2", "Bonobo1"], "uid": ["Bonobo2_3", "Bonobo1_1"],
                  "window_start_s": [10.0, 0.0]}).to_csv(man, index=False)
    with pytest.raises(RuntimeError, match="sha256"):
        ns.fresh_items(str(tmp_path / "c2"), str(man))
    monkeypatch.setattr(ns, "FRESH_LIST_SHA256", ns.sha256_file(str(man)))
    it = ns.fresh_items(str(tmp_path / "c2"), str(man))
    assert list(it) == ["Bonobo1_1", "Bonobo2_3"] and it["Bonobo2_3"]["patient"] == "Bonobo2"
    assert it["Bonobo1_1"]["path"].endswith("c2/sig/Bonobo1_1.npz") and it["Bonobo1_1"]["source"] == "bdsp"
    assert it["Bonobo1_1"]["morgoth"].endswith("c2/morgoth/Bonobo1_1.parquet")
    os.remove(tmp_path / "c2" / "morgoth" / "Bonobo2_3.parquet")
    with pytest.raises(FileNotFoundError):
        ns.fresh_items(str(tmp_path / "c2"), str(man))


def test_the_pinned_fresh_list_hash():
    assert ns.FRESH_LIST_SHA256 == "216c6a2977264d2c3f5e354df6185daa83782a69a7e176de8fbb1ede92c45d39"


def test_fresh_rows_combine_loader_typing_and_c2_variants(tmp_path, monkeypatch):
    root = _fresh_shards(tmp_path)
    monkeypatch.setattr(ns, "pinned_typing", lambda: {("cornell", "vp"): "exposed", ("cornell", "df"): "immune",
                                                          ("cornell", "other"): "exposed", ("mbvt", "w"): None})
    monkeypatch.setattr(ns, "variant_roles", lambda: {("cornell", "vp"): ["validity_primary"],
                                                      ("cornell", "df"): ["default"], ("mbvt", "w"): ["default"]})
    d = ns.fresh_rows(root, outcomes=False)
    assert sorted(set(d["variant"])) == ["vp", "w"] and len(d) == 4 and "call" not in d
    assert "call" in ns.fresh_rows(root, outcomes=True)



A = ns.atlas_module()
VP = {"mbvt": "vt3_mur20_ds7_full_gt23", "cornell": "theoretical_bh_keep40_r2"}


def test_atlas_recording_is_the_atlas_own_unit(monkeypatch):
    seen = {}

    def spy(mod, fn, *args):
        rec, ov = args[-3], args[-2]
        seen.setdefault("x", np.array(rec.x)), seen.setdefault("start", float(ov.onsets_s[0]))
        seen.setdefault("extent", mod.extent(ov))
        return None, "skipped", 0.0
    monkeypatch.setattr(A.rn, "_call_run", spy)
    R = A.roles(rn.load_pipe("mbvt"), VP["mbvt"])
    A.run_unit("mbvt", "env_sd0.5_tau10", 3, R, set(), lambda row: None)
    rec, start = ns.atlas_recording(A, "mbvt", "env_sd0.5_tau10", 3, R)
    assert np.array_equal(rec.x, seen["x"])
    mod = rn.load_pipe("mbvt")
    v, e = next((v, e) for v, e in R.items() if e["kind"] == "published")
    ov = rn._sched(mod, mod.schedule, e["spec"], start, rn._rng("mbvt", rec.uid, 0, 1))
    assert mod.extent(ov) == seen["extent"]


def test_atlas_units_are_the_envelope_cells():
    us = ns.atlas_units(A, ["mbvt", "columbia"], n=2)
    cells = {c for _, c, _ in us}
    assert all(A.CELLS[c]["family"] == "env" for c in cells)
    assert {c for p, c, _ in us if p == "mbvt"} == {c for c in A.CELLS if c.startswith("env_")}
    assert {c for p, c, _ in us if p == "columbia"} == {"env_sd0.5_tau10"}


def test_atlas_ver_rows(tmp_path, monkeypatch):
    d = str(tmp_path / "atlas")
    typ = {("mbvt", VP["mbvt"]): "exposed", ("mbvt", A.rn.primary_variant(rn.load_pipe("mbvt"), "bdsp")): "exposed"}
    keep = {("mbvt", "env_sd0_tau1", 0), ("mbvt", "env_sd0.75_tau30", 0)}
    real_units = ns.atlas_units
    monkeypatch.setattr(ns, "atlas_units", lambda A_, pipes=None, n=None: [u for u in real_units(A_, pipes, n)
                                                                            if u in keep])
    ns.run_atlas_ver(0, 1, n=1, out_dir=d, typing=typ, vp=VP, pipes=["mbvt"])
    n0 = len(ns.read_atlas_rows(d))
    ns.run_atlas_ver(0, 1, n=1, out_dir=d, typing=typ, vp=VP, pipes=["mbvt"])
    assert sum(1 for _ in open(os.path.join(d, "rows00.jsonl"))) == n0
    rows = ns.read_atlas_rows(d)
    assert len(rows) == n0 > 0 and "call" not in rows and set(rows["cell"]) == {"env_sd0_tau1", "env_sd0.75_tau30"}
    assert rows["key"].str.match(r"^mbvt\|env_sd[0-9.]+_tau[0-9]+\|0\|").all()
    m = rows.groupby("cell")["log_ver"].mean()
    assert m["env_sd0.75_tau30"] > m["env_sd0_tau1"] + 0.5 and abs(m["env_sd0_tau1"]) < 0.3
    assert set(rows["sha256_atlas"]) == {ns.sha256_file(A.__file__)}


def test_atlas_skips_immune_variants(tmp_path):
    R = {"mbvt": A.roles(rn.load_pipe("mbvt"), VP["mbvt"])}
    rows = []
    ns.atlas_ver(A, [("mbvt", "env_sd0_tau1", 0)], R, {k: "immune" for k in [("mbvt", v) for v in R["mbvt"]]}, set(),
                 rows.append)
    assert rows == []


def test_atlas_rows_of_another_version_are_refused(tmp_path):
    d = tmp_path / "atlas"
    d.mkdir()
    (d / "rows00.jsonl").write_text(json.dumps({"key": "k", "sha256_128": "0" * 64, "sha256_atlas": "a"}) + "\n")
    with pytest.raises(RuntimeError, match="another version"):
        ns.read_atlas_rows(str(d))
    (d / "rows00.jsonl").write_text("".join(json.dumps({"key": k, "sha256_128": ns.own_sha256(), "sha256_atlas": h})
                                            + "\n" for k, h in (("k1", "a"), ("k2", "b"))))
    with pytest.raises(RuntimeError, match="another version"):
        ns.read_atlas_rows(str(d))


def test_atlas_map_uses_both_distributions():
    syn = pd.DataFrame({"pipe": "cornell", "variant": "v", "cell": ["env_sd0_tau1"] * 2 + ["env_sd0.5_tau10"] * 2,
                        "log_sd": [0, 0, 0.5, 0.5], "tau_s": [1, 1, 10, 10], "log_ver": [0.0, 0.1, 1.0, 1.2]})
    d = pd.DataFrame({"pipe": "cornell", "variant": "v", "log_ver": [0.2, 0.4, np.nan], "stratum": ["a", "a", "b"]})
    out = ns.atlas_map(syn, d, {("cornell", "v"): ["default"]})["cornell|v"]
    assert out["real"]["n"] == 2 and out["synthetic"]["env_sd0.5_tau10"]["mean"] == pytest.approx(1.1)
    assert out["real_by_stratum"]["b"]["n"] == 0 and out["synthetic"]["env_sd0.5_tau10"]["log_sd"] == 0.5



def test_ver_shards_partition_the_recordings(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ns.en, "_load", lambda it: it["uid"])
    monkeypatch.setattr(ns, "recording_ver", lambda rec, rows, n_surr: seen.append(rec) or [{"log_ver": 0.0}])
    rows = pd.DataFrame({"uid": list("abcde"), "key": list("abcde")})
    items = {u: {"uid": u} for u in "abcde"}
    for s in range(2):
        ns.run_ver(rows, items, str(tmp_path / "v"), shard=s, nshards=2)
    assert seen == ["a", "c", "e", "b", "d"] and ns.done_uids(str(tmp_path / "v")) == set("abcde")


def test_stats_refuse_while_a_recording_lacks_its_result(tmp_path):
    d = str(tmp_path / "v")
    ns.write_result(d, "a", [{"key": "a", "log_ver": 0.0}])
    rows = pd.DataFrame({"uid": ["a", "b"]})
    with pytest.raises(RuntimeError, match="without a VER result"):
        ns.require_all(rows, d)
    ns.write_result(d, "b", [{"key": "b", "log_ver": 0.0}])
    ns.require_all(rows, d)


def test_phase1_variant_counts_are_the_non_immune_frozen_variants():
    typ = {("dubois", v): ("immune" if v.startswith("training") else "exposed")
           for v in rn.variant_specs(rn.load_pipe("dubois"), "bdsp")}
    typ.update({("cornell", v): "exposed" for v in rn.variant_specs(rn.load_pipe("cornell"), "bdsp")})
    nv = ns.non_immune_variant_counts(["dubois", "cornell"], typ)
    assert nv == {"dubois": sum(1 for (p, v), k in typ.items() if p == "dubois" and k != "immune"),
                  "cornell": mech.n_frozen_variants("cornell")}
    assert nv["dubois"] < mech.n_frozen_variants("dubois")
