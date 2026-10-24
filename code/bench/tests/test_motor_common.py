"""motor_common.admissible_stat: the re-draw step of the conditional randomization tests (team-decision: reject and
re-draw labellings below a module's minimum, never zero-fill)."""
import numpy as np
import pytest

from bench import motor_common as mc


def test_admissible_stat_redraws_until_the_minimum_is_met():
    seq = iter([0, 1, 2, 7, 3])
    calls = []

    def ok(v):
        calls.append(v)
        return v >= 2
    assert mc.admissible_stat(lambda: next(seq), ok, lambda v: 10.0 * v) == 20.0
    assert calls == [0, 1, 2]


def test_admissible_stat_raises_when_the_cap_is_hit():
    with pytest.raises(mc.NoUsableTrials, match='randomization space too small'):
        mc.admissible_stat(lambda: 0, lambda v: False, lambda v: 1.0, cap=50)
    assert mc.MAX_REDRAWS == 10000



def _broken_perm_test(obs, draw, n_perm, alpha, plus_one=True, early_stop=True, tol=1e-12):
    """The old rule, kept only to show the property check catches it: NaN >= obs is False, so a NaN null draw counts
    as BELOW the observed statistic and lowers p."""
    stop = mc.stop_count(n_perm, alpha, plus_one) if early_stop else None
    null, ge = [], 0
    for _ in range(n_perm):
        v = float(draw())
        null.append(v)
        ge += v >= obs - tol
        if stop is not None and ge >= stop and len(null) < n_perm:
            break
    p = (ge + 1.0) / (n_perm + 1) if plus_one else ge / float(n_perm)
    return {'p': float(p)}


def _p(test, obs, values, **kw):
    it = iter(values)
    return test(obs, lambda: next(it), len(values), 0.05, **kw)['p']


def _nan_never_lowers_p(test, n_cases=300, seed=0):
    """Property: making any subset of the null draws non-finite never lowers p (plus_one on/off, early stop on/off)."""
    g = np.random.default_rng(seed)
    for _ in range(n_cases):
        B = int(g.choice([19, 40, 100]))
        vals = g.normal(size=B)
        obs = float(g.normal() + g.choice([0.0, 2.0]))
        bad = vals.copy()
        bad[g.random(B) < g.choice([0.05, 0.3, 1.0])] = g.choice([np.nan, np.inf, -np.inf])
        for kw in ({}, {'plus_one': False}, {'early_stop': False}, {'plus_one': False, 'early_stop': False}):
            if _p(test, obs, list(bad), **kw) < _p(test, obs, list(vals), **kw) - 1e-15:
                return False
    return True


def test_a_nonfinite_null_draw_never_lowers_p():
    assert _nan_never_lowers_p(mc.perm_test)


def test_the_property_check_catches_the_broken_rule():
    assert not _nan_never_lowers_p(_broken_perm_test)


def test_nonfinite_null_draws_count_as_exceedances_and_are_reported():
    vals = [0.1, np.nan, 0.2, np.inf, -np.inf, 0.3, 0.0, 0.4, 0.5, 0.05]
    t = mc.perm_test(1.0, iter(vals).__next__, len(vals), 0.05, early_stop=False)
    assert t['n_nonfinite_null'] == 3 and np.isclose(t['p'], (3 + 1) / (10 + 1))
    finite = mc.perm_test(1.0, iter([0.1] * 10).__next__, 10, 0.05, early_stop=False)
    assert finite['n_nonfinite_null'] == 0 and np.isclose(finite['p'], 1 / 11)


def test_nonfinite_draws_also_trigger_the_decision_exact_stop():
    t = mc.perm_test(1.0, lambda: np.nan, 500, 0.05)
    assert t['stopped'] and t['n_run'] == mc.stop_count(500, 0.05) and t['p'] >= 0.05
    assert t['n_nonfinite_null'] == t['n_run']


def test_a_nonfinite_observed_statistic_raises_no_usable_trials():
    for obs in (np.nan, np.inf, -np.inf):
        with pytest.raises(mc.NoUsableTrials, match='observed'):
            mc.perm_test(obs, lambda: 0.0, 20, 0.05)
