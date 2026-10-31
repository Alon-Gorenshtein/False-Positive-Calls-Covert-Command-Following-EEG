"""Tests of 176_clustered.py on synthetic rows."""
import importlib

import pytest

cl = importlib.import_module("176_clustered")


def test_cluster_bootstrap_resamples_patients_not_sessions():
    clusters = ["big"] * 50 + [f"p{i}" for i in range(9)]
    values = [1.0] * 50 + [0.0] * 9
    est, lo, hi = cl.cluster_mean_ci(clusters, values, b=4000)
    assert est == pytest.approx(50 / 59)
    assert lo < 0.5 and hi > 0.95


def test_paired_difference_uses_same_sessions():
    rows = []
    for i, (a, c) in enumerate([(1, 1), (1, 0), (1, 1), (0, 0)]):
        for reading, call in (("cornell_c3", a), ("cornell_cmd", c)):
            rows.append({"version": "true", "session_id": f"S{i}", "subject": i, "timepoint": "pre",
                         "sha256": f"h{i}", "reading": reading, "call": bool(call)})
    rows.append({"version": "true", "session_id": "S9", "subject": 9, "timepoint": "pre", "sha256": "h9",
                 "reading": "cornell_c3", "call": True})
    out = cl.stroke(rows)["paired_c3_minus_cmd"]
    assert out["n_sessions"] == 4 and out["difference"] == pytest.approx(0.25)
    assert (out["c3_only"], out["cmd_only"]) == (1, 0)


def test_sign_test_counts_ties_out():
    r = cl.sign_test([2, 1, 0, 0, -1, 1])
    assert (r["positive"], r["negative"], r["ties"]) == (3, 1, 2)
    assert r["p_exact"] == pytest.approx(min(1.0, 2 * (1 + 4) / 16))
