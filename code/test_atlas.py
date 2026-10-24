"""Tests for 129_atlas.py (part C, C1). Synthetic data only.
cd code && /usr/bin/python3 -m pytest test_atlas.py -q"""
import importlib
import json
import os

import numpy as np
import pytest
from scipy.signal import welch

from bench import engine as en
from bench import locked_spec

at = importlib.import_module("129_atlas")
fr = importlib.import_module("130_fidelity_readings")
rn = importlib.import_module("114_benchmark")

FS = 128.0



def test_grid_is_exactly_c1():
    names = at.cell_names("cornell")
    assert names[:4] == ["white", "pink1", "pink2", "alpha"] and len(names) == 24
    env = [at.CELLS[c] for c in names[4:]]
    assert sorted({(e["log_sd"], e["tau_s"]) for e in env}) == sorted(
        (s, t) for s in (0.0, 0.25, 0.5, 0.75) for t in (1, 5, 10, 30, 60))
    assert at.CELLS["pink1"]["beta"] == 1.0 and at.CELLS["pink2"]["beta"] == 2.0
    for p in ("columbia", "dubois"):
        assert at.cell_names(p) == ["white", "pink1", "env_sd0.5_tau10"] and at.n_per_cell(p) == 30
    assert at.n_per_cell("mbvt") == 60 and at.PIPES == locked_spec.PIPES
    u = at.units(["mbvt", "dubois"])
    assert len(u) == 24 * 60 + 3 * 30 and len(set(u)) == len(u)



def _slope(x, lo=2.0, hi=40.0):
    f, p = welch(x, fs=FS, nperseg=512, axis=-1)
    m = (f >= lo) & (f <= hi)
    return np.polyfit(np.log(f[m]), np.log(p[..., m].mean(0)), 1)[0]


def test_noise_shapes_dtype_and_seeding():
    for c in ("white", "pink1", "alpha", "env_sd0.5_tau10"):
        a = at.noise(at.CELLS[c], 100.0, np.random.default_rng(1))
        b = at.noise(at.CELLS[c], 100.0, np.random.default_rng(1))
        d = at.noise(at.CELLS[c], 100.0, np.random.default_rng(2))
        assert a.shape == (19, 12800) and a.dtype == np.float32 and np.array_equal(a, b) and not np.array_equal(a, d)
        assert np.all(np.isfinite(a)) and 5.0 < float(a.std()) < 40.0


def test_noise_spectra():
    g = lambda c: at.noise(at.CELLS[c], 400.0, np.random.default_rng(3))
    assert abs(_slope(g("white"))) < 0.15
    assert _slope(g("pink1")) == pytest.approx(-1.0, abs=0.15)
    assert _slope(g("pink2"), 2.0, 30.0) == pytest.approx(-2.0, abs=0.25)
    f, p = welch(g("alpha"), fs=FS, nperseg=1024, axis=-1)
    pm = p.mean(0)
    assert 9.0 <= f[np.argmax(pm)] <= 11.0 and pm.max() > 20 * np.median(pm)


def test_envelope_log_sd_correlation_time_and_power_normalisation():
    n = int(6000 * FS)
    assert np.array_equal(at.envelope(n, FS, 0.0, 10.0, np.random.default_rng(0)), np.ones(n))
    for sd, tau in ((0.5, 10.0), (0.25, 5.0), (0.75, 30.0)):
        e = at.envelope(n, FS, sd, tau, np.random.default_rng(4))
        le = np.log(e)
        assert le.std() == pytest.approx(sd, rel=0.12)
        if (sd, tau) == (0.5, 10.0):
            assert float(np.mean(e ** 2)) == pytest.approx(1.0, abs=0.15)
        k = int(tau * FS)
        z = (le - le.mean()) / le.std()
        ac = float(np.mean(z[:-k] * z[k:]))
        assert ac == pytest.approx(np.exp(-1.0), abs=0.12)
        assert float(np.mean(z[:-3 * k] * z[3 * k:])) < 0.2


def test_env_family_is_spatially_mixed_and_slowly_fluctuating():
    rng = lambda: np.random.default_rng(5)
    x0 = at.noise(at.CELLS["env_sd0_tau10"], 1200.0, rng())
    x5 = at.noise(at.CELLS["env_sd0.75_tau10"], 1200.0, rng())
    xh = at.noise(at.CELLS["env_sd0.5_tau10"], 1200.0, rng())
    i = {c: k for k, c in enumerate(en.NAMES19)}
    r = np.corrcoef(x0)
    assert r[i["C3"], i["P3"]] > r[i["C3"], i["O2"]] + 0.2 and r[i["C3"], i["P3"]] > 0.3
    w = lambda x: x.reshape(19, -1, int(10 * FS)).var(-1).mean(0)
    cv0, cv5 = w(x0).std() / w(x0).mean(), w(x5).std() / w(x5).mean()
    assert cv5 > 3 * cv0
    assert float((xh ** 2).mean()) == pytest.approx(float((x0 ** 2).mean()), rel=0.35)



def _vp_files(tmp_path, cmd, erp):
    fs = {}
    for fam, m in (("cmd", cmd), ("erp", erp)):
        f = tmp_path / f"vp_{fam}.json"
        f.write_text(json.dumps({"published": {p: {"validity_primary": v, "variants": {"x": {"rate": 0.5}}}
                                               for p, v in m.items()}, "headline": {"rate": 0.9}}))
        fs[fam] = str(f)
    return fs


def test_validity_primaries_are_read_by_name_only(tmp_path):
    fs = _vp_files(tmp_path, {"cornell": "jackknife_by_keep40_r4"}, {"mbvt": "vt2_ann18_ds7_median_binom_n10"})
    assert at.validity_primaries(fs) == {"cornell": "jackknife_by_keep40_r4", "mbvt": "vt2_ann18_ds7_median_binom_n10"}
    with pytest.raises(RuntimeError, match="missing"):
        at.validity_primaries({"cmd": str(tmp_path / "absent.json")})


def test_roles_default_validity_primary_randomised_and_c3():
    mod = rn.load_pipe("cornell")
    R = at.roles(mod, "jackknife_by_keep40_r4")
    pub = {n: e for n, e in R.items() if e["kind"] == "published"}
    assert pub["theoretical_bh_keep40_r4"]["roles"] == ["default"]
    assert pub["jackknife_by_keep40_r4"]["roles"] == ["validity_primary"]
    specs = rn.variant_specs(mod, "bdsp")
    groups = rn.timing_groups(mod, specs, list(specs), rn.primary_variant(mod, "bdsp"))
    reps = {g for g, vs in groups.items() if "theoretical_bh_keep40_r4" in vs or "jackknife_by_keep40_r4" in vs}
    rnd = {e["group"] for e in R.values() if e["kind"] == "randomized"}
    assert rnd == reps and len(rnd) == 1
    assert [e["roles"] for e in R.values() if e["kind"] == "randomized"] == [["randomised_default",
                                                                               "randomised_validity_primary"]]
    assert [e["reading"] for e in R.values() if e["kind"] == "c3"] == fr.readings("cornell")
    same = at.roles(mod, "theoretical_bh_keep40_r4")
    assert same["theoretical_bh_keep40_r4"]["roles"] == ["default", "validity_primary"]
    mb = at.roles(rn.load_pipe("mbvt"), "vt2_ann18_ds7_median_binom_n10")
    assert not any(e["kind"] == "c3" for e in mb.values())
    arms = [e for e in mb.values() if e["kind"] == "randomized"]
    assert [e["roles"] for e in arms] == [["randomised_default"], ["randomised_validity_primary"]]
    assert len({e["group"] for e in arms}) == 2





def test_cochran_armitage_matches_the_linear_by_linear_test():
    from statsmodels.stats.contingency_tables import Table
    x, r, N = [0.0, 0.25, 0.5, 0.75], [2, 5, 9, 14], [60, 60, 60, 60]
    z, p = at.cochran_armitage(x, r, N)
    tab = np.array([[ri, ni - ri] for ri, ni in zip(r, N)])
    ll = Table(tab).test_ordinal_association(row_scores=np.array(x), col_scores=np.array([1.0, 0.0]))
    assert z == pytest.approx(abs(ll.zscore) * np.sqrt(sum(N) / (sum(N) - 1)), rel=1e-9) and p < 0.001
    assert at.cochran_armitage(x, r[::-1], N)[1] > 0.99
    assert at.cochran_armitage(x, [0, 0, 0, 0], N) == (0.0, 1.0)
    assert at.cochran_armitage(x, N, N) == (0.0, 1.0)
    assert at.cochran_armitage(x, [3, 3, 3, 3], N)[1] == pytest.approx(0.5)


def test_holm_and_binomial_range():
    assert at.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert at.holm([]) == []
    assert at.binom_range(60) == (0, 7) and at.binom_range(30) == (0, 4)



def _rows(pipe, variant, kind, roles, counts, n=60):
    """counts: {cell: calls}; n ok rows per cell."""
    out = []
    for cell, k in counts.items():
        c = at.CELLS[cell]
        for r in range(n):
            out.append({"key": f"{pipe}|{cell}|{r}|{variant}", "pipe": pipe, "cell": cell, "family": c["family"],
                        "log_sd": c.get("log_sd"), "tau_s": c.get("tau_s"), "r": r, "variant": variant, "kind": kind,
                        "roles": roles, "status": "ok", "call": r < k})
    return out


def _env(levels):
    """{cell: calls} over the 20 envelope cells, calls per cell set by log-SD."""
    return {f"env_sd{s:g}_tau{t:g}": levels[s] for s in (0.0, 0.25, 0.5, 0.75) for t in (1, 5, 10, 30, 60)}


IMM = {("nc", "vp", "published"): {"immunity_type": None, "x_ref_class": None},
       ("ex", "vp", "published"): {"immunity_type": "exposed", "x_ref_class": "inflated"},
       ("im", "vp", "published"): {"immunity_type": "immune", "x_ref_class": "immune"},
       ("mbvt", "vp", "published"): {"immunity_type": "exposed", "x_ref_class": "deflated"}}


def test_bars_on_constructed_rows():
    rows = (_rows("ex", "vp", "published", ["validity_primary"], {"white": 2, **_env({0.0: 3, 0.25: 6, 0.5: 12,
                                                                                         0.75: 20})})
            + _rows("ex", "rand:g", "randomized", ["randomised_validity_primary"], {"white": 3, **_env(
                {0.0: 3, 0.25: 3, 0.5: 3, 0.75: 3})})
            + _rows("im", "vp", "published", ["validity_primary"], {"white": 3, **_env({0.0: 3, 0.25: 3, 0.5: 3,
                                                                                         0.75: 3})})
            + _rows("mbvt", "vp", "published", ["validity_primary"], {"white": 11})
            + _rows("nc", "vp", "published", ["validity_primary"], {"white": 3, **_env({0.0: 3, 0.25: 30, 0.5: 30,
                                                                                         0.75: 40})}))
    s = at.summarise(rows, {"ex": "vp", "im": "vp", "mbvt": "vp", "nc": "vp"}, IMM)
    b = s["bars"]
    assert b["implementation"]["per_pipeline"]["ex"]["in_range"] and b["implementation"]["per_pipeline"]["im"]["in_range"]
    mb = b["implementation"]["per_pipeline"]["mbvt"]
    assert not mb["in_range"] and mb["known_exception"] and b["implementation"]["pass"]
    tr = b["mechanism"]["per_pipeline"]
    assert set(tr) == {"ex"} and tr["ex"]["holm_p"] < 0.05 and tr["ex"]["rise"]
    assert b["mechanism"]["not_tested"] == {"im": "immune by design", "mbvt": "no envelope grid",
                                            "nc": "no label contrast (116 design typing)"}
    assert "nc" not in s["nogo"]["per_pipeline"]
    assert b["construction"]["pass"] and set(b["construction"]["per_arm"]) == {"ex|rand:g"}
    k = s["nogo"]
    assert not k["nogo"] and k["per_pipeline"]["ex"]["rate"] == pytest.approx((6 + 12) * 5 / 600)
    flat = (_rows("ex", "vp", "published", ["validity_primary"], {"white": 2, **_env({0.0: 3, 0.25: 4, 0.5: 5,
                                                                                         0.75: 20})}))
    assert at.summarise(flat, {"ex": "vp"}, IMM)["nogo"]["nogo"]


def test_a_randomised_trend_fails_the_construction_check():
    rows = _rows("ex", "rand:g", "randomized", ["randomised_default"], _env({0.0: 1, 0.25: 4, 0.5: 9, 0.75: 16}))
    c = at.summarise(rows, {"ex": "vp"}, IMM)["bars"]["construction"]
    assert not c["pass"] and c["per_arm"]["ex|rand:g"]["holm_p"] < 0.05


def test_white_noise_outside_the_range_fails_the_implementation_check():
    rows = _rows("ex", "vp", "published", ["validity_primary"], {"white": 9})
    i = at.summarise(rows, {"ex": "vp"}, IMM)["bars"]["implementation"]
    assert not i["per_pipeline"]["ex"]["in_range"] and not i["pass"]


def test_error_and_na_rows_are_counted_and_left_out():
    rows = _rows("ex", "vp", "published", ["validity_primary"], {"white": 3}, n=58)
    rows += [dict(rows[0], key="e1", r=98, status="error: ValueError: x", call=None),
             dict(rows[0], key="e2", r=99, status="na: no_usable_trials", call=None)]
    t = at.summarise(rows, {"ex": "vp"}, IMM)["pipelines"]["ex"]["variants"]["vp"]["cells"]["white"]
    assert t == {"n": 58, "calls": 3, "rate": pytest.approx(3 / 58), "na": 1, "error": 1}







def test_a_unit_is_the_same_synthetic_recording_for_every_variant(tmp_path, monkeypatch):
    seen = []
    mod = rn.load_pipe("mbvt")
    real = mod.run

    def spy(rec, ov, variant, rng):
        seen.append((rec.uid, float(np.asarray(rec.x, float).sum()), float(ov.onsets_s[0])))
        return real(rec, ov, variant, rng)

    monkeypatch.setattr(mod, "run", spy)
    vp = {"mbvt": "vt2_ann18_ds7_median_binom_n10"}
    monkeypatch.setattr(at, "validity_primaries", lambda files=None: vp)
    at.run_shard(0, 1, pipes=["mbvt"], out_root=str(tmp_path / "sh"), guard=False, cells={"mbvt": ["pink1"]}, n=1)
    assert len(seen) == 2 and len({s[:2] for s in seen}) == 1
    uid = "atlas|pink1|0"
    first = [mod.schedule(at.PAD_S + rn.LEAD_S, rn._rng("mbvt", uid, 0, 1), v).onsets_s[0]
             for v in (rn.primary_variant(mod, "bdsp"), vp["mbvt"])]
    assert [s[2] for s in seen] == first and all(s[0] == uid for s in seen)


def test_dubois_c3_reading_in_the_atlas_has_the_default_runs_folds(tmp_path, monkeypatch):
    """Reduced search, permutations and B (speed); the C3 random-CV reading must see the default run's folds."""
    pdb, wr = importlib.import_module("bench.pipe_dubois"), importlib.import_module("bench.wrappers")
    monkeypatch.setattr(pdb, "CONFIGS", [('motor', 'mu-lowgamma', 1.0, 4), ('motor', 'mu-lowgamma', 2.0, 6)])
    monkeypatch.setattr(pdb, "N_PERM_LABELS", 5)
    monkeypatch.setitem(wr.RAND_B, "dubois", 10)
    vp = {"dubois": "assessment_blocked_perm"}
    monkeypatch.setattr(at, "validity_primaries", lambda files=None: vp)
    out = str(tmp_path / "sh")
    at.run_shard(0, 1, pipes=["dubois"], out_root=out, guard=False, cells={"dubois": ["white"]}, n=1)
    rows = {r["variant"]: r for r in map(json.loads, open(at.shard_file(0, out)))}
    assert all(r["status"] == "ok" for r in rows.values()), {k: r["status"] for k, r in rows.items()}
    default = rows["assessment_random_phase1"]
    c3r = rows["c3:c3_b_assessment_random_phase1"]
    assert c3r["x_p_t_A"] == default["p"] and c3r["x_peak_da"] == default["x_peak_da"]
    assert rows["c3:c3_b_assessment_blocked_phase1"]["kind"] == "c3"
