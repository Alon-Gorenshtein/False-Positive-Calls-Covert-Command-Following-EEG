import importlib
import numpy as np

cn = importlib.import_module("113_columbia_null")


def test_recording_order_is_fixed_by_seed_over_the_whole_sample():
    a, b = cn.sample_order(500), cn.sample_order(500)
    assert np.array_equal(a, b) and sorted(a) == list(range(500))


def test_randomized_order_flips_whole_trials_only():
    y = np.tile(np.r_[np.ones(5, bool), np.zeros(5, bool)], 48)
    g = np.repeat(np.arange(48), 10)
    yr = cn.randomized_order(y, g, seed=3)
    for t in range(48):
        seg = yr[g == t]
        assert seg[:5].all() != seg[5:].all() and len(set(seg[:5])) == 1 and len(set(seg[5:])) == 1
    assert 10 < np.sum(yr[::10] != y[::10]) < 38


def test_trial_series_drift_is_scale_free():
    rng = np.random.default_rng(1)
    X = np.exp(rng.normal(size=(480, 76)))
    g = np.repeat(np.arange(48), 10)
    assert np.isclose(cn.trial_drift(X, g), cn.trial_drift(5 * X, g))


def test_dev_split_is_the_last_20_and_never_analysed():
    elig = [(r, r, f"s{r}") for r in range(200)]
    analysis, dev = cn.split_dev(elig)
    assert [d[0] for d in dev] == list(range(180, 200))
    assert len(analysis) == 180 and not set(a[0] for a in analysis) & set(d[0] for d in dev)


def test_n2_features_use_valid_sleep_rows_and_argmax_stage():
    import pandas as pd
    t = np.arange(100.0)
    p = {"sleep_awake": 0.1, "sleep_n1": 0.1, "sleep_n2": 0.1, "sleep_n3": 0.1, "sleep_rem": 0.1}
    m = pd.DataFrame({"t_s": t, "valid_sleep": True, **{k: np.full(100, v) for k, v in p.items()}})
    m.loc[:59, "sleep_awake"] = 0.6
    m.loc[60:, "sleep_n2"] = 0.6
    m.loc[90:, "valid_sleep"] = False
    m.loc[95:, "sleep_n2"] = np.nan
    f = cn.n2_features(m, start_s=50, span_s=30)
    assert f["n2_share_rec"] == 30 / 90 and f["n2_any"]
    assert f["n2_share_overlay"] == 21 / 31
    g = cn.n2_features(m.assign(sleep_n2=0.1), start_s=0, span_s=50)
    assert g["n2_share_rec"] == 0 and not g["n2_any"]


def test_summary_reports_n2_contrast_with_two_sided_ci():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    n2 = rng.random(n) < 0.4
    call = np.where(n2, rng.random(n) < 0.30, rng.random(n) < 0.05)
    df = pd.DataFrame({"rank": range(n), "primary": True, "n2_any": n2, "n2_share_overlay": n2 * 0.5,
                       "drift": rng.normal(size=n), "stratum": "sleep", "transition": False,
                       "i_auc": 0.5, **{f"{k}_call": call if k == "i" else rng.random(n) < 0.05 for k in ("i", "ii", "iii")},
                       **{f"{k}_call_paper": False for k in ("i", "ii", "iii")}})
    h3 = cn.summarize(df)["primary"]["H3_n2"]
    assert h3["n_n2"] == int(n2.sum())
    lo, hi = h3["i_diff_ci95"]
    assert lo > 0 and h3["i_rate_n2"] > h3["i_rate_no_n2"]
    assert "i_or_adj_drift" in h3




def test_dev_split_must_match_the_frozen_file(tmp_path):
    import json, pytest
    elig = [(r, r, f"s{r}") for r in range(40)]
    f = tmp_path / "dev_split.json"
    json.dump({"bdsp": [[r, r, f"s{r}"] for r in range(20, 40)]}, open(f, "w"))
    assert cn.checked_split(elig, frozen=str(f))[1][0][2] == "s20"
    json.dump({"bdsp": [[r, r, f"s{r}"] for r in range(19, 39)]}, open(f, "w"))
    with pytest.raises(RuntimeError):
        cn.checked_split(elig, frozen=str(f))




def test_extras_order_is_seeded_and_separate_from_the_design_order():
    a, b = cn.sample_order(500, seed=cn.SEED_EXTRA), cn.sample_order(500, seed=cn.SEED_EXTRA)
    assert np.array_equal(a, b) and not np.array_equal(a, cn.sample_order(500))


def test_summary_adds_an_all_with_extras_block_only_when_extras_exist():
    import pandas as pd
    rng = np.random.default_rng(1)
    n = 60
    base = {"n2_any": rng.random(n) < 0.4, "n2_share_overlay": 0.1, "drift": rng.normal(size=n), "stratum": "sleep",
            "transition": False, "i_auc": 0.5, **{f"{k}_call": rng.random(n) < 0.1 for k in ("i", "ii", "iii")},
            **{f"{k}_call_paper": False for k in ("i", "ii", "iii")}}
    d = pd.DataFrame({"rank": range(n), "primary": np.arange(n) < 30, "source": "design", **base})
    assert "all_with_extras" not in cn.summarize(d)
    e = d.assign(source=np.where(np.arange(n) < 40, "design", "extra"), primary=np.arange(n) < 30)
    out = cn.summarize(e)
    assert out["all"]["n"] == 40 and out["all_with_extras"]["n"] == 60 and out["primary"]["n"] == 30
