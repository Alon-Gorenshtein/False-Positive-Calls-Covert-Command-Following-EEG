"""E6: an anytime-valid randomization test by betting, with a Riemannian population prior.
Before trial t's label is used, a predictor that has seen labels 1..t-1 and EEG up to t gives q_t = P(imagery).
The bet p_t = 0.5 + lambda (q_t - 0.5) multiplies wealth by p_t / pi (imagery) or (1 - p_t) / (1 - pi) (rest).
If the EEG does not depend on the instruction, wealth is a test martingale, so P(wealth ever >= 1 / alpha) <= alpha
(Ville's inequality). Wealth is averaged over lambda in LAMBDAS. Engine: Podkopaev & Ramdas, NeurIPS 2023."""
import numpy as np
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAMBDAS = (0.25, 0.5, 1.0)
ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA
MIN_PER_CLASS = 5


def wealth_path(q, y, pi=0.5, lambdas=LAMBDAS):
    q, y = np.asarray(q, float), np.asarray(y, int)
    paths = []
    for lam in lambdas:
        p = 0.5 + lam * (q - 0.5)
        paths.append(np.cumprod(np.where(y == 1, p / pi, (1 - p) / (1 - pi))))
    return np.mean(paths, axis=0)


def first_crossing(w, threshold=THRESHOLD):
    """Number of trials until wealth first reaches the threshold, or None."""
    hit = np.flatnonzero(np.asarray(w) >= threshold)
    return int(hit[0]) + 1 if hit.size else None


def _model(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def recentered_tangent(covs):
    """Training people: tangent vectors at their own full-session Riemannian mean (unlabeled recentering)."""
    return tangent_space(covs, mean_riemann(covs))


def fit_prior(features, labels):
    """Population prior B: tangent-space logistic regression pooled over other people."""
    return _model().fit(np.concatenate(features), np.concatenate(labels))


def _running_means(covs, session):
    """Online Riemannian mean of each session's trials so far (geodesic update; uses no labels)."""
    means = np.empty_like(covs)
    k = 0
    for t in range(len(covs)):
        k = 1 if t == 0 or session[t] != session[t - 1] else k + 1
        means[t] = covs[t] if k == 1 else geodesic_riemann(means[t - 1], covs[t], 1.0 / k)
    return means


def _features(covs, session, means, t):
    """Tangent vectors of trials 0..t as seen at step t: the current session at its running mean, earlier
    sessions at their own final means."""
    out = []
    for s in dict.fromkeys(session[: t + 1]):
        idx = np.flatnonzero(session[: t + 1] == s)
        out.append(tangent_space(covs[idx], means[idx[-1]]))
    return np.concatenate(out)


def step_features(covs, session, t):
    session = np.asarray(session)
    return _features(covs, session, _running_means(covs, session), t)


def _check_sessions(session):
    starts = session[np.r_[True, session[1:] != session[:-1]]]
    if len(set(starts)) != len(starts) or np.any(np.diff(starts) < 0):
        raise ValueError("sessions must be contiguous and in increasing order")


def session_max_wealth(q, y, session, pi=0.5, lambdas=LAMBDAS):
    """Wealth restarted at each session (same predictions): {session: max wealth within it}. With S sessions,
    calling when any session reaches S / alpha is valid by a union bound."""
    q, y, session = np.asarray(q), np.asarray(y), np.asarray(session)
    return {s: float(wealth_path(q[session == s], y[session == s], pi, lambdas).max())
            for s in dict.fromkeys(session.tolist())}


def sequential_test(covs, y, prior=None, session=None, pi=0.5, lambdas=LAMBDAS, min_per_class=MIN_PER_CLASS,
                    C=1.0, prior_alone_until_ready=False):
    """Primary (fixed in advance): C = 1 and q = mean(person, prior), the person model at 0.5 until ready.
    Secondary (added later, after the expert review): prior alone until the person model is ready, C chosen on OpenBCI."""
    y = np.asarray(y, int)
    session = np.zeros(len(y), int) if session is None else np.asarray(session)
    _check_sessions(session)
    means = _running_means(covs, session)
    q = np.full(len(y), 0.5)
    for t in range(len(y)):
        z = _features(covs, session, means, t)
        ready = t > 0 and min(np.sum(y[:t] == 0), np.sum(y[:t] == 1)) >= min_per_class
        q_person = _model(C).fit(z[:t], y[:t]).predict_proba(z[t:t + 1])[0, 1] if ready else 0.5
        if prior is None:
            q[t] = q_person
        else:
            q_prior = prior.predict_proba(z[t:t + 1])[0, 1]
            q[t] = q_prior if (prior_alone_until_ready and not ready) else 0.5 * (q_person + q_prior)
    w = wealth_path(q, y, pi, lambdas)
    n = first_crossing(w)
    return {"q": q, "wealth": w, "called": n is not None, "n_to_call": n, "max_wealth": float(w.max())}
