"""Real-DoC-EEG null: command-following designs overlaid on recordings in
which no command was given, so every call is false.

Designs (48 trials of 10 s, 24 per class):
- B1: 12 blocks of 4 same-class trials, alternating; 5-fold CV linear SVM accuracy, one-sided binomial test vs 50%.
- B2: the B1 schedule; CV AUROC (ridge classifier), trial-label permutation test.
- B3: the B1 schedule and statistic, permuting whole blocks over all C(12, 6) = 924 balanced assignments (exact).
- B4: per-trial randomized labels, analysed with B2's statistic.
- B2-alt: 48 alternating 10-s instruction periods, five 2-s epochs each; epoch-level CV AUROC with (a) epoch-label and
  (b) period-level permutation.
- R: per-trial randomized labels, E6 betting (betting.wealth_path) with an online person model and an optional prior;
  a call when wealth reaches 20.
All calls at alpha = 0.05."""
from itertools import combinations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import betting

ALPHA = 0.05
N_TRIALS, BLOCK = 48, 4
RIDGE = 1.0
FOLDS = 5


def schedule(design, rng, n=N_TRIALS, block=BLOCK):
    nb = n // block
    blocks = np.repeat(np.arange(nb), block)
    if design == "block":
        first = int(rng.integers(0, 2))
        y = np.array([(first + b) % 2 for b in range(nb)])[blocks]
    else:
        y = rng.permutation(np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)])
    return y.astype(int), blocks


def block_assignments(nb=N_TRIALS // BLOCK):
    out = []
    for ones in combinations(range(nb), nb // 2):
        a = np.zeros(nb, int)
        a[list(ones)] = 1
        out.append(a)
    return np.array(out)


def _folds(y, rng, k=FOLDS):
    f = np.zeros(len(y), int)
    for c in (0, 1):
        I = rng.permutation(np.flatnonzero(y == c))
        f[I] = np.arange(len(I)) % k
    return f


def _oof_decisions(F, Y, folds):
    """Out-of-fold ridge-classifier decisions for many label vectors at once (rows of Y). The ridge solution is linear
    in the labels, so each fold's projection is computed once."""
    F, Y = np.asarray(F, float), np.atleast_2d(np.asarray(Y, float))
    D = np.zeros(Y.shape)
    for k in np.unique(folds):
        te, tr = folds == k, folds != k
        mu = F[tr].mean(0)
        Xc = F[tr] - mu
        A = np.linalg.solve(Xc.T @ Xc + RIDGE * np.eye(F.shape[1]), Xc.T)
        Yc = Y[:, tr] - Y[:, tr].mean(1, keepdims=True)
        D[:, te] = ((F[te] - mu) @ (A @ Yc.T)).T
    return D


def _auroc_rows(D, Y):
    """AUROC of decisions D against labels Y, row by row (Mann-Whitney with average ranks)."""
    D, Y = np.atleast_2d(D), np.atleast_2d(Y).astype(bool)
    order = np.argsort(D, axis=1, kind="mergesort")
    ranks = np.empty_like(D)
    r = np.arange(1, D.shape[1] + 1, dtype=float)
    np.put_along_axis(ranks, order, np.broadcast_to(r, D.shape), axis=1)
    n1 = Y.sum(1)
    n0 = Y.shape[1] - n1
    return ((ranks * Y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def b1(F, y, rng):
    folds = _folds(y, rng)
    correct = 0
    for k in range(FOLDS):
        te, tr = folds == k, folds != k
        m = LinearSVC(C=1.0, max_iter=5000).fit(F[tr], y[tr])
        correct += int((m.predict(F[te]) == y[te]).sum())
    p = float(binom.sf(correct - 1, len(y), 0.5))
    return {"acc": correct / len(y), "p": p, "call": bool(p < ALPHA)}


def _perm_test(F, y, rng, perms):
    folds = _folds(y, rng)
    Y = np.vstack([y, perms])
    auc = _auroc_rows(_oof_decisions(F, Y, folds), Y)
    p = float((1 + np.sum(auc[1:] >= auc[0])) / (1 + len(perms)))
    return {"auc": float(auc[0]), "p": p, "call": bool(p < ALPHA)}


def b2(F, y, rng, n_perm=1000):
    perms = np.array([rng.permutation(y) for _ in range(n_perm)])
    return _perm_test(F, y, rng, perms)


def b3(F, y, blocks, rng):
    """Exact block permutation: every balanced assignment of classes to blocks (the observed one included)."""
    folds = _folds(y, rng)
    A = block_assignments(len(np.unique(blocks)))
    Y = A[:, blocks]
    auc = _auroc_rows(_oof_decisions(F, Y, folds), Y)
    obs_row = int(np.flatnonzero((Y == y).all(1))[0])
    p = float(np.mean(auc >= auc[obs_row]))
    return {"auc": float(auc[obs_row]), "p": p, "call": bool(p < ALPHA)}


def b2alt(F_epochs, period_labels, rng, n_perm=1000, per=5):
    """F_epochs: (n_periods * per, d) in time order; period_labels: (n_periods)."""
    y = np.repeat(period_labels, per)
    folds = _folds(y, rng)
    pe = np.array([rng.permutation(y) for _ in range(n_perm)])
    pp = np.array([np.repeat(rng.permutation(period_labels), per) for _ in range(n_perm)])
    out = {}
    for name, perms in (("a_epoch", pe), ("b_period", pp)):
        Y = np.vstack([y, perms])
        auc = _auroc_rows(_oof_decisions(F_epochs, Y, folds), Y)
        p = float((1 + np.sum(auc[1:] >= auc[0])) / (1 + n_perm))
        out[name] = {"auc": float(auc[0]), "p": p, "call": bool(p < ALPHA)}
    return out


def _online_z(F):
    F = np.asarray(F, float)
    Z = np.zeros_like(F)
    for t in range(len(F)):
        m, s = F[: t + 1].mean(0), F[: t + 1].std(0)
        Z[t] = (F[t] - m) / np.where(s > 1e-9, s, 1.0)
    return Z


def r_test(F, y, prior=None, min_per_class=betting.MIN_PER_CLASS):
    """E6 betting on per-trial randomized labels. q_t uses only trials before t (labels) and EEG up to t."""
    Z = _online_z(F)
    y = np.asarray(y, int)
    q = np.full(len(y), 0.5)
    for t in range(len(y)):
        qs = []
        if prior is not None:
            qs.append(float(prior.predict_proba(Z[t:t + 1])[0, 1]))
        if min(np.sum(y[:t] == 0), np.sum(y[:t] == 1)) >= min_per_class:
            m = LogisticRegression(C=1.0, max_iter=1000).fit(Z[:t], y[:t])
            qs.append(float(m.predict_proba(Z[t:t + 1])[0, 1]))
        if qs:
            q[t] = float(np.mean(qs))
    w = betting.wealth_path(q, y)
    return {"max_wealth": float(w.max()), "call": bool(w.max() >= betting.THRESHOLD), "path": w}


def trial_features(x, fs, starts, win=(0.5, 4.5), band=(8.0, 30.0)):
    """Log band power per channel in [start + win[0], start + win[1]) for each trial start (samples)."""
    sos = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band", output="sos")
    xf = sosfiltfilt(sos, np.asarray(x, float), axis=0)
    a, b = int(win[0] * fs), int(win[1] * fs)
    return np.log(np.array([np.mean(xf[s + a: s + b] ** 2, axis=0) for s in np.asarray(starts, int)]) + 1e-12)



def alternating(n, rng):
    """n commands alternating keep (1) / stop (0) with a random first command."""
    first = int(rng.integers(0, 2))
    return np.array([(first + i) % 2 for i in range(n)], int)


def columbia(F_clips, cmd, rng, n_perm=500, grouped=False, per=5):
    """Columbia-style analysis: five clips per command, CV AUC of a linear classifier, one-tailed permutation test.
    grouped=False: clip-level folds and clip-level shuffles ("as described"); grouped=True: folds and shuffles by
    command."""
    y = np.repeat(np.asarray(cmd, int), per)
    groups = np.repeat(np.arange(len(cmd)), per)
    if grouped:
        f_cmd = _folds(np.asarray(cmd, int), rng)
        folds = f_cmd[groups]
        perms = np.array([np.repeat(rng.permutation(cmd), per) for _ in range(n_perm)])
    else:
        folds = _folds(y, rng)
        perms = np.array([rng.permutation(y) for _ in range(n_perm)])
    Y = np.vstack([y, perms])
    auc = _auroc_rows(_oof_decisions(F_clips, Y, folds), Y)
    p = float((1 + np.sum(auc[1:] >= auc[0])) / (1 + n_perm))
    return {"auc": float(auc[0]), "p": p, "call": bool(p < ALPHA)}


def adjacent_pair_folds(blocks):
    """Cruse's CV as reanalysed by Goldfine 2013: always test on a pair of neighbouring blocks."""
    return np.asarray(blocks, int) // 2


def cruse(F, y, blocks):
    folds = adjacent_pair_folds(blocks)
    correct = 0
    for k in np.unique(folds):
        te, tr = folds == k, folds != k
        m = LinearSVC(C=1.0, max_iter=5000).fit(F[tr], y[tr])
        correct += int((m.predict(F[te]) == y[te]).sum())
    p = float(binom.sf(correct - 1, len(y), 0.5))
    return {"acc": correct / len(y), "p": p, "call": bool(p < ALPHA)}


def cornell(P_move, P_stop, q=ALPHA):
    """Cornell PSD branch: per channel x frequency Welch t-test on log PSD (segments x channels x freqs), Benjamini-
    Hochberg across all tests; a call if any test survives."""
    from scipy.stats import ttest_ind
    t, p = ttest_ind(np.asarray(P_move), np.asarray(P_stop), axis=0, equal_var=False)
    p = np.sort(np.ravel(p))
    m = len(p)
    ok = p <= q * np.arange(1, m + 1) / m
    return {"min_p": float(p[0]), "n_sig": int(np.flatnonzero(ok).max() + 1) if ok.any() else 0,
            "call": bool(ok.any())}
