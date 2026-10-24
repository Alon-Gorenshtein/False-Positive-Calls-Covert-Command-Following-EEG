"""bench/wrappers.py: stat_only() returns run()'s observed statistic for the same rng (checked
against run() itself, and against a deliberately broken wrapper that draws its CV partition from another stream), the
fallback to run() for modules without a wrapper, and the shortened BDSP variants of cruse11 / cruse12p (timing fits a
30-min segment, every native variant has one short twin, module globals restored). Synthetic noise only.
cd code && /usr/bin/python3 -m pytest bench/tests/test_wrappers.py -q"""
import math

import numpy as np
import pytest

from bench import motor_common as mc
from bench import pipe_columbia as pcol
from bench import pipe_cruse11 as pc11
from bench import pipe_cruse12p as pc12
from bench import pipe_dubois as pdb
from bench import pipe_edlow17 as ped
from bench import wrappers as wr
from bench.tests import _synth as sy

FIT_30MIN = 1800 - 20


def _assert_same_stat(stat_fn, mod, rec, ov, variant, seed=7):
    full = mod.run(rec, ov, variant, np.random.default_rng(seed))
    fast = stat_fn(mod, rec, ov, variant, np.random.default_rng(seed))
    mc._CACHE.clear()
    if not np.isclose(mod.stat_for_size_correction(full), mod.stat_for_size_correction(fast), rtol=0, atol=1e-12):
        raise AssertionError(f"{mod.NAME}: stat_only {fast['stat']} != run() {full['stat']}")
    assert fast["extra"]["inference"] is False


def test_columbia_stat_only_is_the_observed_loto_auc(monkeypatch):
    monkeypatch.setattr(pcol, "N_PERM", 2)
    rec = sy.rec(1600, seed=11)
    _assert_same_stat(wr.stat_only, pcol, rec, pcol.schedule(10.0), "as_coded")


def test_edlow_stat_only_is_the_observed_cv_accuracy():
    rec = sy.rec(620, seed=12)
    v = dict(ped.VARIANTS["permsecond_cvsecond_alt"], n_perm=2)
    _assert_same_stat(wr.stat_only, ped, rec, ped.schedule(10.0, np.random.default_rng(0), v), v)


def test_a_wrapper_with_its_own_cv_partition_stream_is_caught():
    rec = sy.rec(620, seed=12)
    v = dict(ped.VARIANTS["permsecond_cvsecond_alt"], n_perm=2)

    def broken(mod, rec, ov, variant, rng):
        return wr.stat_only(mod, rec, ov, variant, np.random.default_rng(12345))
    with pytest.raises(AssertionError):
        _assert_same_stat(broken, ped, rec, ped.schedule(10.0, np.random.default_rng(0), v), v)


def test_dubois_perm_variant_keeps_the_peak_without_label_permutations():
    rec = sy.rec(1100, seed=13)
    v = dict(pdb.VARIANTS["assessment_random_perm"], configs=pdb.CONFIGS[:2], n_perm_labels=2)
    _assert_same_stat(wr.stat_only, pdb, rec, pdb.schedule(10.0, np.random.default_rng(0), v), v)


def test_cruse11_gf13_keeps_the_observed_accuracy_without_relabellings():
    rec = sy.rec(2200, seed=14)
    v = "gf13_b4_hilbert"
    _assert_same_stat(wr.stat_only, pc11, rec, pc11.schedule(10.0, np.random.default_rng(0), v), v)
    assert pc11.GF13_MAX_LABELINGS == 1000


def test_unwrapped_modules_and_variants_run_in_full():
    calls = []

    class M:
        NAME = "other"

        @staticmethod
        def run(rec, ov, variant, rng):
            calls.append(variant)
            return mc.result(1.0, 0.01, True)
    r = wr.stat_only(M, None, None, "v", None)
    assert calls == ["v"] and r["extra"]["inference"] is True and r["call"] is True
    assert wr.wrapped("dubois", pdb.VARIANTS["assessment_random_t_only"]) is False
    assert wr.wrapped("dubois", pdb.VARIANTS["assessment_random_perm"]) is True
    assert wr.wrapped("cruse11", pc11.VARIANTS["cr11_b4_hilbert_ci99"]) is False



def test_cruse11_short_variants_fit_a_30_min_segment_and_twin_every_native():
    sv = wr.short_variants(pc11, "bdsp")
    assert len(sv) == 8 and all(s["arg"]["n_blocks"] == wr.SHORT["cruse11"]["n_blocks"] for s in sv.values())
    twins = sorted(v for s in sv.values() for v in s["twin_of"])
    assert twins == sorted(pc11.VARIANTS)
    for name, s in sv.items():
        assert wr.run_len(pc11, s) <= FIT_30MIN
        ov = pc11.schedule(10.0, np.random.default_rng(1), s["arg"])
        assert len(np.unique(ov.meta["block"])) == 2 * s["arg"]["n_blocks"]
    assert wr.short_variants(pc11, "pdoc") == {}


def test_gf13_with_three_block_pairs_cannot_reach_p_05():
    """Documented property of the shortened GF13 variant: 20 balanced relabellings, the observed one and its mirror
    both reach |acc - 50%|, so p >= 2 / 20."""
    labs, exhaustive = pc11._labelings(6, 3, 0)
    assert exhaustive and len(labs) == math.comb(6, 3) == 20
    assert all(tuple(sorted(set(range(6)) - set(l))) in labs for l in labs) and 2 / 20 > 0.05


def test_cruse12p_short_variants_override_the_block_count_only_inside_the_call():
    assert np.isclose(wr.cruse12p_run_s(pc12, pc12.N_BLOCKS), pc12.RUN_S)
    sv = wr.short_variants(pc12, "bdsp")
    assert sorted(s["twin_of"][0] for s in sv.values()) == sorted(pc12.VARIANTS)
    for s in sv.values():
        assert wr.run_len(pc12, s) <= FIT_30MIN
        with wr.module_globals(pc12, s["globals"]):
            ov = pc12.schedule(10.0, np.random.default_rng(2), s["arg"])
        assert len(ov.labels) == 3 * pc12.PER_CLASS * s["globals"]["N_BLOCKS"]
    assert pc12.N_BLOCKS == 5
    with pytest.raises(RuntimeError):
        with wr.module_globals(pc12, {"N_BLOCKS": 2}):
            raise RuntimeError("boom")
    assert pc12.N_BLOCKS == 5


def test_native_lengths_do_not_fit_a_30_min_segment():
    assert pc11.RUN_S > FIT_30MIN and pc12.RUN_S > FIT_30MIN


def test_short_variants_run_end_to_end_on_synthetic_noise():
    rec = sy.rec(1800, seed=15)
    sv = wr.short_variants(pc11, "bdsp")["cr12n_b3_hilbert"]
    ov = pc11.schedule(10.0, np.random.default_rng(3), sv["arg"])
    r = pc11.run(rec, ov, sv["arg"], np.random.default_rng(4))
    assert 0 <= r["stat"] <= 1 and r["extra"]["n_kept"] <= len(ov.labels)
    s12 = wr.short_variants(pc12, "bdsp")["random_perclass_b4"]
    with wr.module_globals(pc12, s12["globals"]):
        ov = pc12.schedule(10.0, np.random.default_rng(5), s12["arg"])
        r = pc12.run_randomized(rec, ov, dict(s12["arg"], n_perm_rand=3), np.random.default_rng(6))
    assert r["p"] is not None and pc12.N_BLOCKS == 5



PIPES = ("columbia", "cornell", "cruse11", "cruse12p", "dubois", "edlow17", "mbmi", "mbvt", "chennu13", "gibson16",
         "sokoliuk21")


def test_one_b_per_module_wrapper():
    assert set(wr.RAND_B) == set(PIPES) and set(wr.RAND_B.values()) == {200}
    assert mc.stop_count(200, 0.05) == 10


def _exact(test, n=200, seed=0):
    """200 synthetic null draws (observed and permutation statistics exchangeable): the early-stopped decision must
    equal the full-B decision every time, and the call count must lie in the 99.9% binomial range of the nominal
    rate floor(0.05 (B + 1)) / (B + 1)."""
    from scipy.stats import binom
    g = np.random.default_rng(seed)
    calls = 0
    for _ in range(n):
        vals = g.standard_normal(wr.RAND_B["mbvt"] + 1)
        stopped, full = test(vals, True), test(vals, False)
        if stopped != full:
            raise AssertionError("early stopping changed a decision")
        calls += stopped
    rate = np.floor(0.05 * 201 - 1e-9) / 201
    lo, hi = binom.ppf(0.0005, n, rate), binom.isf(0.0005, n, rate)
    if not lo <= calls <= hi:
        raise AssertionError(f"{calls} calls in {n} null draws outside [{lo}, {hi}]")


def _perm_test_call(vals, early, stop=None):
    it = iter(vals[1:])
    return mc.perm_test(vals[0], lambda: next(it), wr.RAND_B["mbvt"], 0.05, early_stop=early)["p"] < 0.05


def _columbia_call(vals, early, stop=None):
    ce = pcol.ce
    it = iter(vals[1:])
    B = wr.RAND_B["columbia"]
    perm, stopped = ce._run_perms(np.array([vals[0]]), lambda: next(it), B,
                                  (wr.columbia_stop_at(B) if stop is None else stop) if early else None)
    return ce._result(np.array([vals[0]]), perm, B, stopped)["call"]


def test_perm_test_is_exact_at_b_200():
    _exact(_perm_test_call)


def test_columbia_trialflip_machinery_is_exact_at_b_200():
    _exact(_columbia_call)


def test_a_stopping_threshold_one_too_low_is_caught():
    with pytest.raises(AssertionError):
        _exact(lambda v, e: _columbia_call(v, e, stop=wr.columbia_stop_at(200) - 1))


def test_the_wrappers_pass_b_to_the_modules(monkeypatch):
    seen = []
    real = mc.perm_test
    monkeypatch.setattr(mc, "perm_test", lambda obs, draw, n_perm, alpha, **k: (seen.append(n_perm),
                                                                                 real(obs, draw, n_perm, alpha, **k))[1])
    rec = sy.rec(620, seed=16)
    v = "permsecond_cvsecond_alt"
    wr.run_randomized(ped, rec, ped.schedule_randomized(10.0, np.random.default_rng(0), v), v, np.random.default_rng(1))
    assert seen == [200]
    calls = {}
    monkeypatch.setattr(pcol, "_xy", lambda rec, ov: (np.zeros((96, 2)), np.tile([True, False], 48),
                                                      np.repeat(np.arange(48), 2)))
    monkeypatch.setattr(pcol.ce, "trialflip_test", lambda X, y, g, rs, n_perm, stop_at: (
        calls.update(n_perm=n_perm, stop_at=stop_at),
        {"auc": 0.5, "p": 0.5, "call": False, "call_paper": False, "se": 0.0, "stopped_early": True,
         "n_perm_run": 11})[1])
    r = wr.run_randomized(pcol, None, pcol.schedule(10.0), "as_coded", np.random.default_rng(2))
    assert calls == {"n_perm": 200, "stop_at": 10} and r["extra"]["n_perm_run"] == 11


def test_mbvt_randomized_method_holds_its_level_at_b_200_on_noise():
    """200 null draws on synthetic noise (20 recordings x 10 randomized schedules; exact conditional on the data)."""
    from scipy.stats import binom
    from bench import pipe_mbvt as pm
    v = "vt2_ann18_ds7_full_gt23"
    calls = 0
    for k in range(20):
        rec = sy.rec(pm.run_s(v) + 20, seed=100 + k)
        for d in range(10):
            ov = pm.schedule_randomized(10.0, np.random.default_rng(1000 * k + d), v)
            calls += wr.run_randomized(pm, rec, ov, dict(pm.VARIANTS[v], reps=2), np.random.default_rng(d))["call"]
        mc._CACHE.clear()
    assert binom.ppf(0.0005, 200, 0.05) <= calls <= binom.isf(0.0005, 200, 0.05)



def test_min_attainable_p():
    sv = wr.short_variants(pc11, "bdsp")
    assert np.isclose(wr.min_attainable_p(pc11, sv["gf13_b3_hilbert"]["arg"]), 2 / 20)
    assert wr.cannot_call(pc11, sv["gf13_b3_hilbert"]["arg"])
    assert np.isclose(wr.min_attainable_p(pc11, "gf13_b4_hilbert"), 2 / 70) and not wr.cannot_call(pc11, "gf13_b4_hilbert")
    assert np.isclose(wr.min_attainable_p(pc11, "gf13_b6_hilbert"), 2 / 924)
    assert wr.min_attainable_p(pc11, sv["cr11_b3_hilbert_ci99"]["arg"]) == 0.0
    assert np.isclose(wr.min_attainable_p(pcol, "as_coded"), 1 / 501)
    assert np.isclose(wr.min_attainable_p(pdb, "assessment_random_perm"), 1 / 101)
    assert wr.min_attainable_p(pdb, "assessment_random_t_only") == 0.0
    assert np.isclose(wr.min_attainable_p(pc12, "random_perclass"), 1 / 1001)



PDOC6 = ["F3", "C3", "O1", "F4", "C4", "O2"]


def _pdoc_rec(dur, seed):
    r = sy.rec(dur, seed=seed, names=PDOC6)
    return r


def test_columbia_is_not_applicable_on_the_pdoc_montage():
    assert wr.pdoc_overrides(pcol) is None and "19" in wr.PDOC_NA_MODULES["columbia"]


def test_pdoc_overrides_restore_the_modules():
    from bench import pipe_gibson16 as pg
    with wr.module_globals(pdb, wr.pdoc_overrides(pdb)):
        assert set(pdb.CHANSETS["full"]) == set(PDOC6) and set(pdb.CHANSETS["motor"]) == {"F3", "C3", "F4", "C4"}
    assert len(pdb.CHANSETS["full"]) == 17
    assert wr.pdoc_variant_na(pg, "ssep_avg_nonharm") and not wr.pdoc_variant_na(pg, "p3b_two025")


def _run6(mod, variant, dur, seed=30, **cfg):
    rec = _pdoc_rec(dur, seed)
    v = dict(mod.VARIANTS[variant], **cfg) if cfg else variant
    with wr.module_globals(mod, wr.pdoc_overrides(mod)):
        ov = mod.schedule(10.0, np.random.default_rng(0), v)
        r = mod.run(rec, ov, v, np.random.default_rng(1))
    mc._CACHE.clear()
    return r


@pytest.mark.parametrize("name,variant,dur,cfg", [
    ("cornell", "theoretical_bh_keep40_r4", 1100, {}),
    ("cruse11", "cr12n_b4_hilbert", 1850, {}),
    ("cruse12p", "random_perclass", 1900, {"n_perm": 3}),
    ("dubois", "assessment_random_t_only", 1100, {}),
    ("edlow17", "permsecond_cvsecond_alt", 620, {"n_perm": 3}),
    ("mbmi", "gu17_cspfold_peak", 1150, {}),
    ("mbvt", "vt3_mur20_seg8_full_gt23", 200, {}),
    ("chennu13", "p1_samples10_none_b20", 2560, {}),
    ("gibson16", "p3b_two025", 1150, {}),
    ("sokoliuk21", "roc116_as_coded_as_coded_none", 1400, {}),
])
def test_every_pdoc_module_computes_on_the_six_channels(name, variant, dur, cfg):
    import importlib
    mod = importlib.import_module(f"bench.pipe_{name}")
    if name == "dubois":
        cfg = {"configs": [c for c in mod.CONFIGS if c[3] == 4][:2]}
    native = list(getattr(mod, "NATIVE_CH"))
    r = _run6(mod, variant, dur, **cfg)
    assert np.isfinite(r["stat"]) and list(mod.NATIVE_CH) == native



def test_p_censored_reads_the_stopping_flag_and_cruse12p_comparisons():
    r = lambda p, **ex: {"stat": 0.0, "p": p, "call": False, "extra": ex}
    assert wr.p_censored(r(26 / 1001, stopped_early=True)) and not wr.p_censored(r(0.01, stopped_early=False))
    both = {"right": {"p": 26 / 1001, "stopped_early": True}, "left": {"p": 26 / 1001, "stopped_early": True}}
    one = {"right": {"p": 26 / 1001, "stopped_early": True}, "left": {"p": 0.004, "stopped_early": False}}
    assert wr.p_censored(r(26 / 1001, comparisons=both)) and not wr.p_censored(r(0.004, comparisons=one))
    assert not wr.p_censored(None) and not wr.p_censored(r(None))
