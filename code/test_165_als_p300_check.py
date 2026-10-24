"""Tests for code/165_als_p300_check.py: the 0.1-30 Hz zero-phase filter, flash-locked baseline-corrected window means,
the within-sequence target-flag permutation test, the per-patient statistic and the decision rule. Synthetic arrays only.
cd code && /usr/bin/python3 -m pytest test_165_als_p300_check.py -q"""
import importlib

import numpy as np
import pytest

kc = importlib.import_module("165_als_p300_check")
FS = 256.0


def test_bandpass_removes_dc_and_60hz_keeps_10hz_and_is_zero_phase():
    t = np.arange(int(60 * FS)) / FS
    x = 50.0 + np.sin(2 * np.pi * 10 * t) + 2 * np.sin(2 * np.pi * 60 * t)
    y = kc.bandpass(x, FS)
    mid = slice(int(20 * FS), int(40 * FS))
    assert abs(y[mid].mean()) < 0.05
    assert np.corrcoef(y[mid], np.sin(2 * np.pi * 10 * t)[mid])[0, 1] > 0.99
    assert np.std(y[mid]) == pytest.approx(1 / np.sqrt(2), rel=0.1)


def test_epoch_means_are_baseline_corrected_window_means_and_edge_flashes_are_dropped():
    n = int(10 * FS)
    x = np.full(n, 7.0)
    on = np.array([int(2 * FS), int(5 * FS)])
    x[on[0] + 64: on[0] + 154] += 3.0
    a, ok = kc.epoch_means(x, FS, np.r_[on, 3, n - 10])
    assert list(ok) == [True, True, False, False]
    assert a[0] == pytest.approx(3.0, abs=0.02) and a[1] == pytest.approx(0.0, abs=1e-9)


def test_epoch_window_is_250_to_600_ms_with_a_minus_100_to_0_baseline():
    x = np.zeros(int(6 * FS))
    on = np.array([int(2 * FS)])
    x[on[0] - 20: on[0]] = 4.0
    a, _ = kc.epoch_means(x, FS, on)
    assert a[0] == pytest.approx(-4.0 * 20 / 26, abs=1e-9)
    y = np.zeros(int(6 * FS))
    y[on[0] + 154: on[0] + 200] = 9.0
    assert kc.epoch_means(y, FS, on)[0][0] == pytest.approx(0.0)


def test_epoch_means_accepts_another_window_and_keeps_the_baseline():
    x = np.zeros(int(6 * FS))
    on = np.array([int(2 * FS)])
    x[on[0] + 51: on[0] + 102] = 5.0
    a, _ = kc.epoch_means(x, FS, on, win_ms=(200.0, 400.0))
    assert a[0] == pytest.approx(5.0)
    assert kc.epoch_means(x, FS, on)[0][0] < 5.0



def test_shuffle_keeps_the_number_of_target_flags_in_every_sequence():
    rng = np.random.default_rng(0)
    seq = np.repeat(np.arange(5), [10, 12, 8, 20, 9])
    flags = np.zeros(len(seq), bool)
    for s in range(5):
        i = np.flatnonzero(seq == s)
        flags[rng.choice(i, 3, replace=False)] = True
    sh = kc.shuffle_flags(flags, seq, np.random.default_rng(1))
    for s in range(5):
        assert sh[seq == s].sum() == 3
    assert (sh != flags).any()


def test_a_planted_p300_is_significant_and_noise_is_not_and_the_test_is_one_sided():
    rng = np.random.default_rng(2)
    seq = np.repeat(np.arange(30), 20)
    flags = np.zeros(len(seq), bool)
    for s in range(30):
        flags[s * 20 + rng.choice(20, 4, replace=False)] = True
    a = rng.standard_normal(len(seq)) * 5
    p_null = kc.perm_test_all(a, flags, seq, B=500, rng=np.random.default_rng(3))
    assert p_null["p"] > 0.01
    p_eff = kc.perm_test_all(a + 2.0 * flags, flags, seq, B=500, rng=np.random.default_rng(3))
    assert p_eff["p"] < 0.01 and p_eff["diff_uv"] == pytest.approx(2.0, abs=0.7)
    p_rev = kc.perm_test_all(a - 2.0 * flags, flags, seq, B=500, rng=np.random.default_rng(3))
    assert p_rev["p"] > 0.95 and p_rev["diff_uv"] < 0
    assert p_eff["n_target"] == 120 and p_eff["n_nontarget"] == 480


def test_perm_test_is_reproducible_and_p_is_never_zero():
    rng = np.random.default_rng(4)
    seq = np.repeat(np.arange(10), 10)
    flags = np.tile([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 10).astype(bool)
    a = rng.standard_normal(100) + 50 * flags
    r1 = kc.perm_test_all(a, flags, seq, B=200, rng=np.random.default_rng(9))
    r2 = kc.perm_test_all(a, flags, seq, B=200, rng=np.random.default_rng(9))
    assert r1 == r2 and r1["p"] == pytest.approx(1 / 201)


def test_sequences_with_no_target_or_only_targets_are_ignored_by_the_shuffle_not_crashed_on():
    seq = np.r_[np.zeros(5, int), np.ones(5, int)]
    flags = np.r_[np.zeros(5, bool), np.ones(5, bool)]
    sh = kc.shuffle_flags(flags, seq, np.random.default_rng(0))
    assert (sh == flags).all()



def synth_runs(effect, n_runs=3, n_seq=4, per_seq=24, seed=0):
    """Runs shaped like bigp3_als_io.flashes(): Pz is channel 5; a `effect` µV bump 250-600 ms after target flashes."""
    rng = np.random.default_rng(seed)
    runs = []
    for r in range(n_runs):
        soa = 64
        n = 300 + n_seq * (per_seq * soa + 700)
        x = rng.standard_normal((7, n)).astype(np.float32) * 5
        on, tg, sq = [], [], []
        t = 300
        for s in range(n_seq):
            flags = np.zeros(per_seq, bool)
            flags[rng.choice(per_seq, 4, replace=False)] = True
            for k in range(per_seq):
                on.append(t + k * soa)
                tg.append(flags[k])
                sq.append(s)
            t += per_seq * soa + 700
        on = np.array(on)
        for o, f in zip(on, tg):
            if f:
                x[5, o + 64: o + 154] += effect
        runs.append({"x": x, "fs": FS, "onset": on, "is_target": np.array(tg), "seq": np.array(sq),
                     "meta": {"phase": "Train" if r == 0 else "Test"}})
    return runs


def test_patient_stat_pools_runs_uses_pz_and_finds_the_effect():
    runs = synth_runs(6.0)
    r = kc.patient_stat(runs, B=300, seed=(1, 2, 3))
    assert r["n_target"] + r["n_nontarget"] + r["n_excluded_near"] == 3 * 4 * 24 - r["n_dropped"]
    assert r["p"] < 0.01 and r["diff_uv"] > 3 and r["n_excluded_near"] > 0
    assert r["n_runs"] == 3
    assert set(r["all_nontargets"]) == {"diff_uv", "p", "n_target", "n_nontarget", "B"}
    null = kc.patient_stat(synth_runs(0.0, seed=5), B=300, seed=(1, 2, 3))
    assert null["p"] > 0.01 and abs(null["diff_uv"]) < 3


def test_sequence_ids_stay_distinct_across_runs():
    runs = synth_runs(0.0, n_runs=2, n_seq=2, per_seq=10)
    seq, _ = kc.global_sequence(runs)
    assert len(np.unique(seq)) == 4



def test_eligible_nontargets_are_at_least_one_second_from_every_target_within_the_sequence():
    on = (np.arange(20) * 64).astype(int)
    flags = np.zeros(20, bool)
    flags[10] = True
    el = kc.eligible(on, flags, np.zeros(20, int), FS)
    assert el.sum() == 13 and not el[7:14].any() and el[:7].all() and el[14:].all() and not el[10]
    two = np.zeros(20, bool)
    two[[3, 15]] = True
    assert kc.eligible(on, two, np.zeros(20, int), FS).sum() == 6


def test_a_flash_exactly_four_flashes_from_a_target_is_eligible_even_when_the_edf_rate_is_256_00008():
    """B and L headers carry 256.00005-256.00013 Hz: 4 flashes = 256 samples must still count as >= 1 s."""
    on = (np.arange(20) * 64).astype(int)
    flags = np.zeros(20, bool)
    flags[10] = True
    for fs in (256.0, 256.00006, 256.00013):
        el = kc.eligible(on, flags, np.zeros(20, int), fs)
        assert el.sum() == 13 and el[6] and el[14] and not el[7] and not el[13], fs
    seq = np.zeros(20, int)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(20)
    r1 = kc.perm_test(a, flags, seq, on, 256.0, B=50, rng=np.random.default_rng(1))
    r2 = kc.perm_test(a, flags, seq, on, 256.00008, B=50, rng=np.random.default_rng(1))
    assert r1 == r2


def test_eligibility_is_per_sequence_so_a_target_never_excludes_flashes_of_another_sequence():
    on = np.r_[np.arange(10) * 64, 10 ** 5 + np.arange(10) * 64]
    seq = np.r_[np.zeros(10, int), np.ones(10, int)]
    flags = np.zeros(20, bool)
    flags[5] = True
    el = kc.eligible(on, flags, seq, FS)
    assert el[10:].all() and el[:10].sum() == 3


def test_far_perm_test_uses_only_far_nontargets_and_ignores_a_contaminated_neighbour():
    rng = np.random.default_rng(0)
    n_seq, per = 40, 24
    seq = np.repeat(np.arange(n_seq), per)
    on = np.tile(np.arange(per) * 64, n_seq) + seq * 10 ** 4
    flags = np.zeros(len(seq), bool)
    for s in range(n_seq):
        flags[s * per + 8] = True
    a = rng.standard_normal(len(seq))
    near = (np.abs(on - np.repeat(on[flags], per)) < 256) & ~flags
    a[near] += 5.0
    r = kc.perm_test(a, flags, seq, on, FS, B=300, rng=np.random.default_rng(1))
    assert r["n_target"] == 40 and r["n_nontarget"] < 40 * 23 and r["n_excluded_near"] > 0
    assert abs(r["diff_uv"]) < 1.0 and r["p"] > 0.05
    a2 = rng.standard_normal(len(seq)) + 1.5 * flags
    assert kc.perm_test(a2, flags, seq, on, FS, B=300, rng=np.random.default_rng(1))["p"] < 0.01


def test_far_perm_test_is_one_sided_reproducible_and_recomputes_eligibility_each_draw():
    rng = np.random.default_rng(3)
    n_seq, per = 30, 24
    seq = np.repeat(np.arange(n_seq), per)
    on = np.tile(np.arange(per) * 64, n_seq) + seq * 10 ** 4
    flags = np.zeros(len(seq), bool)
    for s in range(n_seq):
        flags[s * per + rng.choice(per, 2, replace=False)] = True
    a = rng.standard_normal(len(seq))
    r1 = kc.perm_test(a - 2.0 * flags, flags, seq, on, FS, B=200, rng=np.random.default_rng(5))
    r2 = kc.perm_test(a - 2.0 * flags, flags, seq, on, FS, B=200, rng=np.random.default_rng(5))
    assert r1 == r2 and r1["p"] > 0.9 and r1["diff_uv"] < 0
    el = kc.eligible(on, flags, seq, FS)
    assert r1["diff_uv"] == pytest.approx((a - 2.0 * flags)[flags].mean() - (a - 2.0 * flags)[el].mean())


def test_a_sequence_with_no_target_contributes_all_its_flashes_as_eligible():
    on = np.arange(10) * 64
    assert kc.eligible(on, np.zeros(10, bool), np.zeros(10, int), FS).all()



def test_target_gaps_are_measured_within_sequences_in_flashes_and_seconds():
    on = np.arange(30) * 64
    flags = np.zeros(30, bool)
    flags[[2, 7, 13]] = True
    g = kc.target_gaps(on, flags, np.zeros(30, int), FS)
    assert g["min_gap_flashes"] == 5 and g["min_gap_s"] == pytest.approx(1.25)
    assert kc.target_gaps(on, np.zeros(30, bool), np.zeros(30, int), FS)["min_gap_flashes"] is None



def test_go_needs_at_least_34_of_48_significant_and_a_positive_grand_average():
    mk = lambda k, d=1.0: [{"p": 0.01 if i < k else 0.5, "diff_uv": d} for i in range(48)]
    r = kc.go_rule(mk(34))
    assert r["go"] is True and r["n_significant"] == 34 and r["condition_1_met"] and r["condition_2_met"]
    assert kc.go_rule(mk(33))["go"] is False and kc.go_rule(mk(33))["condition_2_met"] is True
    neg = kc.go_rule(mk(40, d=-0.1))
    assert neg["go"] is False and neg["condition_1_met"] is True and neg["condition_2_met"] is False
    mixed = [{"p": 0.01, "diff_uv": 5.0}] * 34 + [{"p": 0.5, "diff_uv": -20.0}] * 14
    assert kc.go_rule(mixed)["go"] is False
    assert kc.go_rule([{"p": 0.05, "diff_uv": 1.0}] * 48)["n_significant"] == 0


def test_design_facts_pick_the_first_train_file_in_time_order_not_by_result():
    def run(sess, file, phase, tgt_gap):
        on = np.arange(30) * 64
        fl = np.zeros(30, bool)
        fl[[2, 2 + tgt_gap]] = True
        return {"x": None, "fs": FS, "onset": on, "is_target": fl, "seq": np.zeros(30, int),
                "meta": {"session": sess, "file": file, "phase": phase, "soa_s": 0.25, "flash_ms": 31.25}}
    runs = [run("SE002", "x_Train01.edf", "Train", 5), run("SE001", "x_Train02.edf", "Train", 6),
            run("SE001", "x_Train01.edf", "Train", 7), run("SE001", "x_Test01.edf", "Test", 4)]
    f = kc.design_facts(runs)
    assert f["first_train_file"]["file"] == "x_Train01.edf" and f["first_train_file"]["session"] == "SE001"
    assert f["first_train_file"]["min_gap_flashes"] == 7 and f["first_train_file"]["n_flashes"] == 30
    assert f["first_train_file"]["target_probability"] == pytest.approx(2 / 30)
    assert f["all_runs"]["min_gap_flashes"] == 4 and f["all_runs"]["n_runs_with_gap_below_4"] == 0
    assert kc.design_facts(runs[:1] + [run("SE001", "y_Test01.edf", "Test", 3)])["all_runs"]["n_runs_with_gap_below_4"] == 1
