import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
import evoked


def test_erp_model_separates_a_synthetic_p300():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    E = rng.standard_normal((300, 4, 55))
    E[y == 1, :2] += np.exp(-((np.arange(55) - 25) / 6.0) ** 2)
    p = cross_val_predict(evoked.erp_model(), E, y, cv=5, method="predict_proba")[:, 1]
    assert roc_auc_score(y, p) > 0.8


def test_cca_picks_the_stimulated_frequency():
    fs = 250
    t = np.arange(int(3.5 * fs)) / fs
    E = np.random.default_rng(1).standard_normal((10, 3, len(t))) + 0.8 * np.sin(2 * np.pi * 8.57 * t)
    assert (evoked.cca_scores(E, fs, [12, 8.57, 6.67, 5.45]).argmax(1) == 1).all()


def test_cut_drops_out_of_range_epochs_and_resamples():
    ep, keep = evoked.cut(np.zeros((2000, 2)), 1000, [-300, 500, 1990], 0.25, 0.8, 100)
    assert keep.tolist() == [False, True, False] and ep.shape == (1, 2, 55)
