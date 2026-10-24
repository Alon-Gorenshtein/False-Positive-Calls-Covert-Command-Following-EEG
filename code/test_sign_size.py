"""Tests for 141_sign_size.py: synthetic tables only."""
import importlib

import numpy as np
import pandas as pd
import pytest

sz = importlib.import_module("141_sign_size")


def _rows(rng, n_rec=400, b=0.8):
    out = []
    for i in range(n_rec):
        for pipe in ("cornell", "edlow17"):
            for v in range(2):
                a, p = abs(rng.normal()), abs(rng.normal())
                ra = abs(rng.normal())
                out.append({"uid": f"U{i}", "patient": f"U{i}", "pipe": pipe, "variant": f"{pipe}{v}",
                            "abs_t_aligned": a, "abs_t_placebo": p, "rand_abs_t_aligned": ra,
                            "rand_variant": f"{pipe}r", "rand_call": bool(rng.random() < 0.12),
                            "call": bool(rng.random() < 1 / (1 + np.exp(2.0 - b * (a - 1))))})
    return pd.DataFrame(out)


def test_counterfactual_calls_follow_the_fitted_model():
    alpha, beta = np.array([-2.0, -1.0]), np.array([0.5, 0.0])
    Z = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    code = np.array([0, 1, 1])
    p = sz.predicted(alpha, beta, Z, code)
    assert p == pytest.approx(1 / (1 + np.exp(-np.array([-1.5, 0.0, -1.0]))))
    cf = sz.predicted(alpha, beta, sz.set_column(Z, 0, -1.0), code)
    assert cf == pytest.approx(1 / (1 + np.exp(-np.array([-2.5, -1.5, -1.5]))))


def test_attributable_calls_are_positive_with_an_effect_and_about_zero_without():
    nv = {"cornell": 2, "edlow17": 2}
    choices = {"cornell": {"validity_primary": "cornell0", "default": "cornell1"},
               "edlow17": {"validity_primary": "edlow170", "default": "edlow170"}}
    eff = sz.attributable(_rows(np.random.default_rng(1)), nv, choices, B=100, seed=[1, 1])
    assert eff["pooled"]["extra_per_100_model"] > 1 and eff["pooled"]["extra_per_100_model_ci95"][0] > 0
    assert eff["by_variant"]["cornell|cornell0"]["role"] == ["validity_primary"]
    assert eff["by_variant"]["edlow17|edlow170"]["role"] == ["validity_primary", "default"]
    null = sz.attributable(_rows(np.random.default_rng(2), b=0.0), nv, choices, B=100, seed=[1, 1])
    lo, hi = null["pooled"]["extra_per_100_model_ci95"]
    assert lo < 0.5 and abs(null["pooled"]["extra_per_100_model"]) < 1.5


def test_dose_response_rises_by_quintile_within_variant():
    d = _rows(np.random.default_rng(3), n_rec=1000, b=1.5)
    r = sz.dose_response(d, "abs_t_aligned", "call", "variant")
    rates = [r["pooled"][q]["rate"] for q in range(5)]
    assert rates[4] > rates[0] + 0.1 and r["pooled"][0]["n"] == pytest.approx(len(d) / 5, rel=0.02)
    flat = sz.dose_response(d, "rand_abs_t_aligned", "rand_call", "rand_variant")
    assert abs(flat["pooled"][4]["rate"] - flat["pooled"][0]["rate"]) < 0.05
    assert set(r["by_pipeline"]) == {"cornell", "edlow17"}
