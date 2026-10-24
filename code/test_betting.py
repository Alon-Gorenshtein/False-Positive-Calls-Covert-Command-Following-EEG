import numpy as np
from pyriemann.estimation import Covariances
from sklearn.metrics import roc_auc_score
import betting


def _covs(y, effect, seed, n_ch=3, n_t=100):
    """Synthetic epochs: class 1 has its channel-0 variance multiplied by (1 + effect)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(y), n_ch, n_t))
    X[np.asarray(y) == 1, 0] *= np.sqrt(1 + effect)
    return Covariances(estimator="oas").fit_transform(X)


def _prior(effect=1.0, people=6, n=60):
    feats, labels = [], []
    for p in range(people):
        y = np.tile([0, 1], n // 2)
        feats.append(betting.recentered_tangent(_covs(y, effect, seed=1000 + p)))
        labels.append(y)
    return betting.fit_prior(feats, labels)


def test_uninformative_predictions_leave_wealth_at_one():
    y = np.tile([0, 1], 20)
    assert np.allclose(betting.wealth_path(np.full(40, 0.5), y), 1.0)


def test_confident_correct_predictions_cross_the_threshold():
    y = np.tile([0, 1], 20)
    w = betting.wealth_path(np.where(y == 1, 0.9, 0.1), y)
    n = betting.first_crossing(w)
    assert n is not None and n < 15 and w[n - 1] >= betting.THRESHOLD


def test_prediction_for_a_trial_never_uses_that_trials_label():
    y = np.tile([0, 1], 20)
    covs = _covs(y, 1.0, seed=1)
    prior = _prior()
    base = betting.sequential_test(covs, y, prior=prior)["q"]
    for t in (12, 25, 39):
        y2 = y.copy()
        y2[t] = 1 - y2[t]
        q2 = betting.sequential_test(covs, y2, prior=prior)["q"]
        assert np.allclose(base[: t + 1], q2[: t + 1])


def test_running_mean_restarts_at_each_session():
    y = np.tile([0, 1], 20)
    covs = _covs(y, 0.0, seed=2)
    session = np.repeat([0, 1], 20)
    z = betting.step_features(covs, session, t=20)
    assert np.allclose(z[20], 0.0)
    assert not np.allclose(betting.step_features(covs, np.zeros(40, int), t=20)[20], 0.0)


def test_false_calls_stay_near_alpha_when_labels_are_coin_flips():
    prior = _prior()
    calls = 0
    sims = 150
    for s in range(sims):
        rng = np.random.default_rng(50_000 + s)
        y = rng.integers(0, 2, 60)
        covs = _covs(np.zeros(60, int), 0.0, seed=60_000 + s)
        calls += betting.sequential_test(covs, y, prior=prior)["called"]
    assert calls / sims <= 0.10


def test_a_real_class_difference_is_called_within_the_session():
    y = np.tile([0, 1], 40)
    res = betting.sequential_test(_covs(y, 1.5, seed=3), y, prior=_prior())
    assert res["called"] and res["n_to_call"] <= 60


def test_population_prior_transfers_without_calibration():
    y = np.tile([0, 1], 30)
    z = betting.recentered_tangent(_covs(y, 1.0, seed=4))
    assert roc_auc_score(y, _prior().predict_proba(z)[:, 1]) > 0.7


def test_sessions_must_be_contiguous_and_in_order():
    y = np.tile([0, 1], 10)
    covs = _covs(y, 0.0, seed=5)
    import pytest
    with pytest.raises(ValueError):
        betting.sequential_test(covs, y, session=np.array([1] * 5 + [2] * 5 + [1] * 10))


def test_reset_wealth_restarts_each_session():
    y = np.tile([0, 1], 20)
    session = np.repeat([1, 2], 20)
    q = np.where(np.arange(40) < 20, 0.5, np.where(y == 1, 0.9, 0.1))
    per = betting.session_max_wealth(q, y, session)
    assert per[1] == 1.0 and per[2] == betting.wealth_path(q[20:], y[20:]).max()


def test_secondary_predictor_uses_the_prior_alone_until_the_person_model_is_ready():
    y = np.tile([0, 1], 20)
    covs = _covs(y, 1.0, seed=6)
    prior = _prior()
    q = betting.sequential_test(covs, y, prior=prior, prior_alone_until_ready=True, C=0.1)["q"]
    z = betting.step_features(covs, np.zeros(40, int), t=5)
    assert np.isclose(q[5], prior.predict_proba(z[5:6])[0, 1])
