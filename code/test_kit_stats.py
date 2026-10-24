import numpy as np
import pytest
import kit_stats as ks


def test_prop_ci_exact():
    lo, hi = ks.prop_ci(5, 10)
    assert lo == pytest.approx(0.187, abs=0.002) and hi == pytest.approx(0.813, abs=0.002)


def test_mcnemar_exact():
    a = np.array([True] * 10 + [False] * 10)
    assert ks.mcnemar_exact(a, a) == 1.0
    b = np.array([False] * 10 + [False] * 10)
    assert ks.mcnemar_exact(a, b) == pytest.approx(2 * 0.5 ** 10)


def test_one_sided_vs_and_median_ci():
    assert ks.one_sided_vs(5, 100, 0.05) > 0.4
    assert ks.median_ci(np.full(30, 0.7)) == [0.7, 0.7]


def test_cluster_rate_pools_people_and_resamples_them():
    import kit_stats as ks
    v = np.r_[np.ones(10), np.zeros(10), np.zeros(20)]
    g = np.r_[np.zeros(10), np.ones(10), np.full(20, 2)]
    rate, ci, draws = ks.cluster_rate(v, g, B=500)
    assert rate == 0.25 and ci[0] <= 0.25 <= ci[1] and len(draws) == 500


def test_not_above_flags_a_rate_well_above_the_bound():
    import kit_stats as ks
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(50), 20)
    _, p_hi = ks.not_above(rng.random(1000) < 0.20, g, 0.05)
    _, p_lo = ks.not_above(rng.random(1000) < 0.02, g, 0.05)
    assert p_hi < 0.05 and p_lo >= 0.05
