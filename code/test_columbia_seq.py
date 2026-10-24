import importlib
import numpy as np

ce = importlib.import_module("112_columbia_exact")


def _toy(seed, effect):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(12), 10)
    y = np.tile(np.r_[np.ones(5, bool), np.zeros(5, bool)], 12)
    X = rng.normal(size=(120, 8)) + effect * y[:, None]
    return X, y, groups


def test_early_stop_prefix_and_decision_match_full_run():
    for fn in (ce.exact_test, ce.trialflip_test):
        for seed, effect in ((0, 0.0), (1, 0.0), (2, 1.5)):
            X, y, g = _toy(seed, effect)
            full = fn(X, y, g, rng=np.random.RandomState(seed), n_perm=40)
            seq = fn(X, y, g, rng=np.random.RandomState(seed), n_perm=40, stop_at=3)
            k = len(seq["perm"])
            assert np.allclose(seq["perm"], full["perm"][:k])
            assert seq["call"] == full["call"] and seq["call_paper"] == full["call_paper"]
            if seq["stopped_early"]:
                assert not full["call"] and np.sum(full["perm"][:k] >= full["auc"]) == 3


def test_stop_at_default_is_decision_exact_for_500_permutations():
    assert ce.STOP_AT == 25 and (ce.STOP_AT + 1) / 501 > ce.ALPHA
