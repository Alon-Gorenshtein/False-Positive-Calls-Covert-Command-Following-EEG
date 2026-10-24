import numpy as np
import doc_null as dn


def test_block_schedule_is_blocked_and_balanced():
    y, blocks = dn.schedule("block", np.random.default_rng(0))
    assert len(y) == 48 and y.sum() == 24
    assert len(np.unique(blocks)) == 12
    for b in np.unique(blocks):
        assert len(set(y[blocks == b].tolist())) == 1 and (blocks == b).sum() == 4
    assert all(y[blocks == b][0] != y[blocks == b + 1][0] for b in range(11))


def test_random_schedule_balanced():
    y, _ = dn.schedule("random", np.random.default_rng(1))
    assert len(y) == 48 and y.sum() == 24


def test_b3_enumerates_all_block_assignments():
    assert len(dn.block_assignments(12)) == 924


def _slow_features(rng, slow, n=48, d=4, trial_s=10):
    """White trial noise plus, if slow, six sinusoidal fluctuations with 1-10 min periods (arousal-like drift)."""
    t = np.arange(n) * trial_s
    F = rng.normal(0, 0.5 if slow else 1.0, (n, d))
    if slow:
        for _ in range(6):
            per, ph = rng.uniform(60, 600), rng.uniform(0, 2 * np.pi, d)
            F += rng.normal(0, 1, d) * np.sin(2 * np.pi * t[:, None] / per + ph)
    return F


def _rates(slow, runs, seed, n_perm=200):
    rng = np.random.default_rng(seed)
    calls = {k: 0 for k in ("B1", "B2", "B1long", "B2long", "B3", "B4", "R")}
    for _ in range(runs):
        F = _slow_features(rng, slow)
        yb, blocks = dn.schedule("block", rng)
        yl, _ = dn.schedule("block", rng, block=8)
        yr, _ = dn.schedule("random", rng)
        calls["B1"] += dn.b1(F, yb, rng)["call"]
        calls["B2"] += dn.b2(F, yb, rng, n_perm=n_perm)["call"]
        calls["B1long"] += dn.b1(F, yl, rng)["call"]
        calls["B2long"] += dn.b2(F, yl, rng, n_perm=n_perm)["call"]
        calls["B3"] += dn.b3(F, yb, blocks, rng)["call"]
        calls["B4"] += dn.b2(F, yr, rng, n_perm=n_perm)["call"]
        calls["R"] += dn.r_test(F, yr)["call"]
    return {k: v / runs for k, v in calls.items()}


def test_independent_trials_all_designs_nominal():
    r = _rates(False, 100, 2)
    assert all(v <= 0.12 for v in r.values()), r


def test_slow_fluctuations_inflate_long_block_trial_label_tests_only():
    r = _rates(True, 100, 3)
    assert r["B1long"] >= 0.15 and r["B2long"] >= 0.15, r
    assert r["B3"] <= 0.12 and r["B4"] <= 0.12 and r["R"] <= 0.12, r


def test_trial_features_shape_and_log_power():
    rng = np.random.default_rng(4)
    fs = 100
    x = rng.normal(0, 1, (fs * 600, 3))
    starts = np.arange(48) * 10 * fs + 5 * fs
    F = dn.trial_features(x, fs, starts)
    assert F.shape == (48, 3) and np.all(np.isfinite(F))


def test_published_designs_nominal_on_independent_noise():
    rng = np.random.default_rng(11)
    calls = {"col_a": 0, "col_b": 0, "cruse": 0, "cornell": 0}
    runs = 100
    for _ in range(runs):
        cmd = dn.alternating(48, rng)
        Fc = rng.normal(0, 1, (48 * 5, 16))
        calls["col_a"] += dn.columbia(Fc, cmd, rng, n_perm=200, grouped=False)["call"]
        calls["col_b"] += dn.columbia(Fc, cmd, rng, n_perm=200, grouped=True)["call"]
        y, blocks = dn.schedule("block", rng, block=6)
        calls["cruse"] += dn.cruse(rng.normal(0, 1, (48, 4)), y, blocks)["call"]
        P = rng.normal(0, 1, (16, 4, 30))
        calls["cornell"] += dn.cornell(P[::2], P[1::2])["call"]
    r = {k: v / runs for k, v in calls.items()}
    assert all(v <= 0.12 for v in r.values()), r


def test_cruse_folds_are_adjacent_block_pairs():
    y, blocks = dn.schedule("block", np.random.default_rng(0), block=6)
    folds = dn.adjacent_pair_folds(blocks)
    assert len(set(folds.tolist())) == 4
    for f in range(4):
        b = sorted(set(blocks[folds == f].tolist()))
        assert b == [2 * f, 2 * f + 1] and set(y[folds == f].tolist()) == {0, 1}


def test_alternating_commands():
    c = dn.alternating(48, np.random.default_rng(1))
    assert c.sum() == 24 and np.all(c[1:] != c[:-1])
