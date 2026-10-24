"""Proportion CIs, exact McNemar, one-sided binomial and a person-level bootstrap CI for medians."""
import numpy as np
from scipy.stats import binomtest


def prop_ci(k, n):
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def mcnemar_exact(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def one_sided_vs(k, n, p0):
    return float(binomtest(int(k), int(n), p0, alternative="greater").pvalue)


def median_ci(values, B=2000, seed=20261012):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, v.size, replace=True)) for _ in range(B)]
    return [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]


def cluster_rate(values, groups, B=2000, seed=20261012):
    """Pooled rate with a person-cluster bootstrap (people resampled with replacement)."""
    v, g = np.asarray(values, float), np.asarray(groups)
    ids = np.unique(g)
    sums = np.array([v[g == i].sum() for i in ids])
    cnts = np.array([(g == i).sum() for i in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), (B, len(ids)))
    draws = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(sums.sum() / cnts.sum()), [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))], draws


def not_above(values, groups, x, B=2000, seed=20261012):
    """One-sided cluster-bootstrap test of rate > x; p = share of bootstrap rates <= x."""
    rate, _, draws = cluster_rate(values, groups, B, seed)
    return rate, float(np.mean(draws <= x))
