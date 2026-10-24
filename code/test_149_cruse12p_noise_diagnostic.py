import importlib

import numpy as np

dg = importlib.import_module("149_cruse12p_noise_diagnostic")
at, rn = dg.at, dg.rn
R = (dg.R0 + 3, dg.R0 + 7)


def _inputs(cell, r, variant="random_perclass"):
    mod = rn.load_pipe(dg.PIPE)
    spec = rn._spec(mod, variant, "bdsp")
    ov, err, inside = dg.overlay(mod, spec, r)
    assert err is None and inside
    return mod, spec, dg.draw_rec(cell, r), ov




def test_mirror_with_defaults_is_run_exactly_on_white_and_pink():
    for cell in ("white", "pink"):
        for variant in dg.VARIANTS:
            for r in R:
                mod, spec, rec, ov = _inputs(cell, r, variant)
                a = mod.run(rec, ov, spec["arg"], dg.run_rng(spec, r))
                rn._clear_cache()
                b = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, r))
                rn._clear_cache()
                assert a == b


def test_regroup_changes_only_the_null_and_full_run_keeps_the_first_decision():
    mod, spec, rec, ov = _inputs("white", R[0])
    a = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, R[0]))
    b = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, R[0]), regroup=True)
    assert a["extra"]["comparisons"]["right"]["acc"] == b["extra"]["comparisons"]["right"]["acc"]
    f = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, R[0]), early_stop=False,
                      imbalance_rng=np.random.default_rng(0))
    ra, rf = a["extra"]["comparisons"]["right"], f["extra"]["comparisons"]["right"]
    assert rf["n_perm_run"] == mod.N_PERM and rf["acc"] == ra["acc"]
    assert (rf["p"] < mod.ALPHA) == (ra["p"] < mod.ALPHA)
    assert rf["imbalance_obs"] == 0.0 < rf["imbalance_perm"]
    rn._clear_cache()


def test_drop_keeps_exactly_the_target_and_never_adds():
    keep = np.ones(20, bool)
    keep[[2, 5]] = False
    k = dg.drop_keep(keep, 11, np.random.default_rng(1))
    assert k.sum() == 11 and not k[2] and not k[5]
    assert dg.drop_keep(keep, 30, np.random.default_rng(1)).sum() == 18
    mod, spec, rec, ov = _inputs("white", R[1])
    res = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, R[1]), n_keep=120,
                        drop_rng=np.random.default_rng(2))
    assert res["extra"]["n_kept"] == 120
    rn._clear_cache()


def test_summary_statistics():
    rows = [{"r": i, "status": "ok", "call": i < 3} for i in range(10)] + [{"r": 99, "status": "na: x"}]
    s = dg.rate(rows)
    assert (s["calls"], s["n"], s["n_na"]) == (3, 10, 1) and s["ci95"][0] < 0.3 < s["ci95"][1]
    assert abs(s["p_binom_greater_5pct"] - dg.bm.binom_greater(3, 10, 0.05)) < 1e-15
    a = [{"r": i, "status": "ok", "call": i < 5} for i in range(10)]
    b = [{"r": i, "status": "ok", "call": i < 1} for i in range(10)]
    m = dg.mcnemar(a, b)
    assert (m["n_paired"], m["only_first"], m["only_second"]) == (10, 4, 0) and abs(m["p_exact"] - 0.125) < 1e-12
    assert dg.fold_imbalance(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])) == 0.0
    assert dg.fold_imbalance(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 0.5


def test_one_row_of_each_arm_runs():
    kept = np.array([60, 120, 150])
    for arm in dg.ARMS:
        if arm.endswith("full"):
            continue
        row = dg.run_arm(arm, R[0], kept)
        assert row["status"] == "ok" and row["min_ioi_s"] >= rn.load_pipe(dg.PIPE).EPOCH_S
        if "drop" in arm:
            assert row["n_kept"] == row["n_keep_target"] in kept
        else:
            assert row["n_kept"] == 180


def test_within_fold_permutation_keeps_every_fold_count_and_changes_only_the_null():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    grp = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    yp = dg.permute_within(y, grp, np.random.default_rng(3))
    assert all(yp[grp == f].sum() == y[grp == f].sum() for f in range(3)) and sorted(yp) == sorted(y)
    mod, spec, rec, ov = _inputs("white", R[0], "contiguous_perclass")
    a = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, R[0]), early_stop=False)
    b = dg.run_mirror(mod, rec, ov, spec["arg"], dg.run_rng(spec, R[0]), early_stop=False, withinfold=True)
    ra, rb = a["extra"]["comparisons"]["right"], b["extra"]["comparisons"]["right"]
    assert ra["acc"] == rb["acc"] and ra["null_mean"] != rb["null_mean"]
    rn._clear_cache()


def test_design_imbalance_reads_labels_only():
    d = dg.design_imbalance(2, n_perm=5)
    assert d["random"]["n"] == d["contiguous"]["n"] == 4 and d["random"]["observed_mean"] == 0.0
    assert d["random"]["permuted_mean"] > 0


def test_a_shared_draw_gives_the_rows_of_separate_draws():
    kept = np.array([60, 120, 150])
    rec = dg.draw_rec("white", R[1])
    rn._clear_cache()
    for arm in ("white|random_perclass", "white|random_perclass|drop", "white|contiguous_perclass|withinfold"):
        a = dg.run_arm(arm, R[1], kept, rec)
        b = dg.run_arm(arm, R[1], kept)
        a.pop("seconds"), b.pop("seconds")
        assert a == b
    rn._clear_cache()
