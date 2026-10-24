"""Synthetic arrays only; every guard is also run against a deliberately broken input. cd code && /usr/bin/python3 -m
pytest test_162_stroke_check.py -q"""
import importlib

import numpy as np
import pytest

kc = importlib.import_module("162_stroke_check")


def planted(n, n_ch, rng, effect=3.0, fs=250, T=1000):
    """n trials per class; class 1 has `effect` x the variance in channel 0, class 0 in channel 1."""
    X = rng.standard_normal((2 * n, n_ch, T))
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    X[y == 1, 0] *= effect
    X[y == 0, 1] *= effect
    return X, y


def null(n, n_ch, rng, T=1000):
    return rng.standard_normal((2 * n, n_ch, T)), np.r_[np.zeros(n, int), np.ones(n, int)]



def test_binomial_bound_reproduces_63_19_at_60_trials_and_gives_the_40_trial_bound():
    assert round(100 * kc.chance_bound(60), 2) == 63.19
    assert round(100 * kc.chance_bound(40), 2) == 66.20
    assert kc.chance_bound(40) > kc.chance_bound(60) > 0.5



def test_csp_lda_finds_a_planted_variance_difference():
    X, y = planted(20, 6, np.random.default_rng(0))
    p = kc.fit_predict(X[::2], y[::2], X[1::2], k=2)
    assert (p == y[1::2]).mean() > 0.95


def test_csp_lda_is_at_chance_without_an_effect_and_a_leaky_fit_is_not():
    rng = np.random.default_rng(1)
    clean, leaky = [], []
    for _ in range(12):
        X, y = null(20, 8, rng, T=250)
        clean.append(kc.cv_accuracy(X, y, folds=10, reps=2, seed=3, k=2))
        leaky.append(kc.cv_accuracy(X, y, folds=10, reps=2, seed=3, k=2, _leak=True))
    assert 0.44 < np.mean(clean) < 0.56
    assert np.mean(leaky) > np.mean(clean) + 0.05


def test_cv_accuracy_is_seeded_and_runs_stratified_folds():
    X, y = planted(20, 6, np.random.default_rng(2))
    a = kc.cv_accuracy(X, y, folds=10, reps=3, seed=5, k=2)
    assert a == kc.cv_accuracy(X, y, folds=10, reps=3, seed=5, k=2) and 0.9 < a <= 1.0
    with pytest.raises(ValueError, match="folds"):
        kc.cv_accuracy(X[:6], y[:6], folds=10, reps=1, seed=0, k=2)


def test_csp_uses_fewer_filters_when_there_are_few_channels():
    X, y = planted(15, 3, np.random.default_rng(3))
    W = kc.csp(X, y, k=2)
    assert W.shape == (3, 2)


def test_leave_one_run_out_trains_on_other_runs_only():
    rng = np.random.default_rng(4)
    X, y = planted(20, 6, rng)
    run = np.tile([0, 1, 2, 3], 10)
    acc, per = kc.loro_accuracy(X, y, run, k=2)
    assert acc > 0.9 and len(per) == 4
    y2 = y.copy()
    y2[run == 3] = 1 - y2[run == 3]
    acc2, per2 = kc.loro_accuracy(X, y2, run, k=2)
    assert per2[3] < 0.2



def test_label_permutation_within_run_detects_a_planted_desynchronisation_and_not_noise():
    rng = np.random.default_rng(5)
    run = np.repeat(np.arange(4), 20)
    lab = np.tile(np.r_[np.ones(10, int), np.zeros(10, int)], 4)
    base = rng.standard_normal(80) * 0.3
    erd = base - 0.8 * lab
    obs, p = kc.perm_diff(erd, lab, run, B=1000, rng=np.random.default_rng(0))
    assert obs < -0.5 and p < 0.01
    obs0, p0 = kc.perm_diff(base, lab, run, B=1000, rng=np.random.default_rng(0))
    assert p0 > 0.05
    _, pr = kc.perm_diff(base + 0.8 * lab, lab, run, B=1000, rng=np.random.default_rng(0))
    assert pr > 0.95


def test_permutations_stay_inside_runs():
    rng = np.random.default_rng(6)
    run = np.repeat(np.arange(3), 8)
    lab = np.tile([1, 1, 1, 1, 0, 0, 0, 0], 3)
    for _ in range(20):
        q = kc.permute_within(lab, run, rng)
        for r in range(3):
            assert q[run == r].sum() == 4


def test_signflip_detects_a_negative_mean_and_is_one_sided():
    rng = np.random.default_rng(7)
    d = rng.standard_normal(40) * 0.3 - 0.5
    assert kc.signflip_p(d, B=1000, rng=np.random.default_rng(0)) < 0.01
    assert kc.signflip_p(-d, B=1000, rng=np.random.default_rng(0)) > 0.95


def test_log_power_is_the_log_mean_square_of_the_window():
    x = np.ones((2, 3, 100)) * 2.0
    assert kc.log_power(x, 0, 100) == pytest.approx(np.log(4.0) * np.ones((2, 3)))
    assert kc.log_power(x, 10, 20).shape == (2, 3)



def _g(acc, erd_p, idle_p, idle_like):
    return [{"acc": a, "erd_p": e, "idle_like_imagery": i} for a, e, i in zip(acc, erd_p, idle_like)]


def test_gait_nogo_fires_on_each_of_its_three_conditions_and_only_then():
    ok = kc.gait_nogo(_g([0.7, 0.65, 0.62], [0.01, 0.2, 0.03], None, [False, False, False]))
    assert ok["nogo"] is False
    low = kc.gait_nogo(_g([0.55, 0.58, 0.8], [0.01, 0.01, 0.01], None, [False] * 3))
    assert low["nogo"] and low["low_accuracy"] and not low["no_erd"] and not low["idle_like_imagery"]
    noerd = kc.gait_nogo(_g([0.7, 0.7, 0.7], [0.2, 0.06, 0.5], None, [False] * 3))
    assert noerd["nogo"] and noerd["no_erd"]
    idle = kc.gait_nogo(_g([0.7, 0.7, 0.7], [0.01, 0.01, 0.01], None, [True, True, False]))
    assert idle["nogo"] and idle["idle_like_imagery"]
    assert not kc.gait_nogo(_g([0.7, 0.7, 0.7], [0.01, 0.01, 0.01], None, [True, False, False]))["idle_like_imagery"]


def test_acute_nogo_fires_when_occipital_is_at_least_as_good_as_sensorimotor():
    assert kc.acute_nogo(0.60, 0.55)["nogo"] is False
    assert kc.acute_nogo(0.55, 0.55)["nogo"] is True
    assert kc.acute_nogo(0.54, 0.58)["nogo"] is True


def test_write_section_merges_without_dropping_the_other_section(tmp_path):
    p = str(tmp_path / "k.json")
    kc.write_section("gait", {"a": 1}, p)
    kc.write_section("acute", {"b": 2}, p)
    kc.write_section("gait", {"a": 3}, p)
    import json
    assert json.load(open(p)) == {"gait": {"a": 3}, "acute": {"b": 2}}


def test_acute_patient_scores_the_two_channel_groups_separately():
    rng = np.random.default_rng(8)
    X, y = planted(20, 6, rng)
    names = ["C3", "C4", "Cz", "O1", "O2", "Oz"]
    lab = np.where(y == 1, "right", "left")
    r = kc.acute_patient(X, list(lab), names, reps=2)
    assert r["sensorimotor_acc"] > 0.9 and r["occipital_acc"] < 0.75
    assert r["n_sensorimotor"] == 3 and r["n_occipital"] == 3 and r["occipital_channels"] == ["O1", "O2", "Oz"]


def test_duplicate_recordings_are_found_even_when_they_differ_below_the_rounding_step():
    rng = np.random.default_rng(9)
    A, B = rng.standard_normal((5, 4, 50)) * 10, rng.standard_normal((5, 4, 50)) * 10
    assert kc.duplicates({4: A, 5: A + 1e-4, 6: B}) == {5: 4}
    assert kc.duplicates({1: A, 2: B}) == {}



sg = importlib.import_module("stroke_gait_io")


def synth_person(rng, mi_gain=0.5, idle_gain=1.0, n_runs=4, per_class=10):
    """A gait-like person at 250 Hz: runs of 20 trials (10 MI + 10 idle, shuffled), 12 s each; white noise on every
    channel; C3 / Cz / C4 scaled by mi_gain in the MI task window (onset + 0.5..4.5 s) and idle_gain in the idle one."""
    names = sg.channels()
    fs = 250
    hot = [names.index(c) for c in ("C3", "Cz", "C4")]
    runs, srcs, events = [], [], {}
    for r in range(n_runs):
        x = rng.standard_normal((len(names), 12 * fs * per_class * 2 + 2 * fs))
        order = rng.permutation([1] * per_class + [0] * per_class)
        ev = []
        for k, lab in enumerate(order):
            t0 = int((1.0 + 12 * k) * fs)
            b, c, o = (1, 2, 3) if lab else (7, 8, 9)
            ev += [(t0, b), (t0 + 3 * fs, c), (t0 + 4 * fs, o)]
            g = mi_gain if lab else idle_gain
            a, z = t0 + 4 * fs + int(0.5 * fs), t0 + 4 * fs + int(4.5 * fs)
            x[np.ix_(hot, np.arange(a, z))] *= g
        runs.append(x.astype(np.float32))
        s = sg.sources(np.array(ev), fs, x.shape[1], r, "pre")
        srcs += s
        events.update({(r, q.start): q.cls for q in s})
    return sg.pc().Person(pid="Liu_S99", fs=float(fs), names=names, runs=runs, sources=srcs,
                          meta={"events": events, "sessions": ["pre"]})


def test_trial_table_pairs_every_task_excerpt_with_its_own_pre_cue_fixation():
    p = synth_person(np.random.default_rng(10), n_runs=1, per_class=3)
    tt = kc.trial_table(p.sources)
    assert len(tt) == 6 and {t["cls"] for t in tt} == {"mi", "idle"}
    for t in tt:
        assert t["fix"].run == t["task"].run and t["fix"].cls == "fixation"
        assert t["task"].start - t["fix"].start == 4 * 250
        assert t["fix"].tag.split(":")[1] == "fix-" + t["cls"] + "@" + t["fix"].tag.split("@")[1]


def test_trial_table_allows_a_missing_fixation_only_for_a_trial_that_began_before_the_run():
    P = sg.pc().Source
    early = P(run=0, start=375, n=1250, cls="mi", tag="pre:mi@375")
    tt = kc.trial_table([early])
    assert len(tt) == 1 and tt[0]["fix"] is None
    mid = P(run=0, start=5000, n=1250, cls="mi", tag="pre:mi@5000")
    with pytest.raises(ValueError, match="fixation"):
        kc.trial_table([mid])


def test_gait_subject_sees_a_planted_desynchronisation_and_not_idle_like_imagery():
    p = synth_person(np.random.default_rng(11))
    r = kc.gait_subject(p, B=200)
    assert r["n_mi"] == r["n_idle"] == 40 and r["n_runs"] == 4
    assert r["acc"] > 0.9
    assert r["erd_diff_log"] < -1.0 and r["erd_p"] < 0.01
    assert r["mi_vs_fixation_log"] < -1.0 and r["mi_vs_fixation_p"] < 0.01
    assert abs(r["idle_vs_fixation_log"]) < 0.3 and r["idle_like_imagery"] is False
    assert set(r["channel_erd"]) == {"C3", "Cz", "C4"}
    reg = r["acc_by_region"]
    assert reg["sensorimotor"] > 0.9 and reg["frontal"] < 0.7 and reg["posterior_temporal"] < 0.7
    assert reg["n_channels"] == {"sensorimotor": 13, "frontal": 7, "posterior_temporal": 9}


def test_gait_subject_flags_idle_that_desynchronizes_like_imagery_and_finds_no_imagery_effect_then():
    p = synth_person(np.random.default_rng(12), mi_gain=0.5, idle_gain=0.5)
    r = kc.gait_subject(p, B=200)
    assert r["idle_like_imagery"] is True and r["erd_p"] > 0.05
    assert r["acc"] < 0.7


def test_gait_subject_at_null_has_no_effect_anywhere():
    r = kc.gait_subject(synth_person(np.random.default_rng(13), mi_gain=1.0), B=200)
    assert r["erd_p"] > 0.01 and r["mi_vs_fixation_p"] > 0.01 and r["acc"] < 0.7


def test_dataset_facts_carry_the_video_question_with_its_quotes_and_the_known_anomalies():
    f = kc.dataset_facts()
    assert "video of gripping motion" in f["acute"]["video_quote_paper"]
    assert f["acute"]["hand_specific_video"] is True and "left-hand-motor-imagery" in f["acute"]["video_files"][0]
    assert f["gait"]["event_codes"]["3"].startswith("MI onset") and f["gait"]["licence"] == "CC BY 4.0"
    assert f["acute"]["licence"] == "CC BY 4.0" and f["acute"]["reference_channel"] == "CPz"
