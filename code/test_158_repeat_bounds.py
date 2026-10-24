import importlib
import json
import os

import numpy as np
import pytest

rb = importlib.import_module("158_repeat_bounds")
sr = importlib.import_module("145_stated_rule")
ROOT = rb.ROOT


def test_bh_any_matches_the_two_recording_rule():
    rng = np.random.default_rng(1)
    p = rng.random((5000, 2)) * 0.1
    a = rng.random((5000, 2))
    want = sr.bh2_paper(p[:, 0], a[:, 0], p[:, 1], a[:, 1])
    assert np.array_equal(rb.bh_any(p, a), want)


def test_bh_any_one_and_three_recordings():
    assert rb.bh_any(np.array([[0.04]]), np.array([[0.6]]))[0]
    assert not rb.bh_any(np.array([[0.04]]), np.array([[0.4]]))[0]
    assert rb.bh_any(np.array([[0.04, 0.03, 0.02]]), np.array([[0.4, 0.4, 0.6]]))[0]
    assert not rb.bh_any(np.array([[0.02, 0.5, 0.9]]), np.array([[0.6, 0.6, 0.6]]))[0]








def test_censored_p_values_are_not_tested_for_uniformity():
    p = np.r_[np.full(97, 0.05), 0.01, 0.02, 0.03]
    cens = np.r_[np.ones(97, bool), np.zeros(3, bool)]
    c = rb.calibration(p, cens)
    assert c["share_censored"] == pytest.approx(0.97) and c["ks_uniform_p"] is None and "censored" in c["note"]
    c2 = rb.calibration(np.linspace(0.005, 0.995, 100), np.zeros(100, bool))
    assert c2["ks_uniform_p"] > 0.5 and c2["share_p_below_05"] == pytest.approx(0.05)
