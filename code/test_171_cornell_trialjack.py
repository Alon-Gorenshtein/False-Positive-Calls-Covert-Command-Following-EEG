"""Tests of 171_cornell_trialjack.py on synthetic spectra and synthetic recordings only."""
import importlib

import numpy as np
import pytest

from bench import pipe_cornell as pc
from bench.tests import _synth as S

tj = importlib.import_module("171_cornell_trialjack")
fr = importlib.import_module("130_fidelity_readings")


def test_one_epoch_groups_reproduce_the_module_jackknife():
    g = np.random.default_rng(0)
    S1, S2 = g.exponential(1.0, (12, 1, 2, 4)), g.exponential(1.0, (9, 1, 2, 4))
    dz, vdz = pc.tgt(S1, S2)
    dz2, vdz2, df = tj.tgt_grouped(S1, S2, np.arange(12), np.arange(9))
    np.testing.assert_allclose(dz2, dz, rtol=1e-12)
    np.testing.assert_allclose(vdz2, vdz, rtol=1e-10)
    assert df == 12 + 9 - 2


def _z_var(dependent, unit, n_rep=300, seed=1):
    g = np.random.default_rng(seed)
    zs = []
    for _ in range(n_rep):
        n_cmd, n_ep, K = 16, 3, 5
        S1, S2 = g.exponential(1.0, (n_cmd * n_ep, K, 1, 2)), g.exponential(1.0, (n_cmd * n_ep, K, 1, 2))
        grp = np.repeat(np.arange(n_cmd), n_ep)
        if dependent:
            S1 *= np.repeat(np.exp(0.6 * g.standard_normal(n_cmd)), n_ep)[:, None, None, None]
            S2 *= np.repeat(np.exp(0.6 * g.standard_normal(n_cmd)), n_ep)[:, None, None, None]
        if unit == "sample":
            dz, vdz = pc.tgt(S1, S2)
        else:
            dz, vdz, _ = tj.tgt_grouped(S1, S2, grp, grp)
        zs.append((dz / np.sqrt(vdz)).ravel())
    return np.concatenate(zs).var()


def test_command_jackknife_is_calibrated_when_samples_are_independent():
    assert 0.75 < _z_var(False, "command") < 1.35


def test_command_jackknife_absorbs_within_command_dependence():
    v_sample, v_cmd = _z_var(True, "sample"), _z_var(True, "command")
    assert v_sample > 1.5
    assert 0.75 < v_cmd < 1.35


@pytest.fixture(scope="module")
def window():
    ov = pc.schedule(20.0, None, "jackknife_bh_keep40_r4")
    r = S.rec(pc.extent(ov)[1] + 10.0, seed=5, drift=0.4)
    return r, ov


def test_sample_unit_reproduces_the_c3_reading(window):
    r, ov = window
    ref = fr.cornell(r, ov)
    got = tj.cornell_unit(r, ov, unit="sample", ref="normal")
    assert got["call"] == ref["call"]
    assert got["extra"]["om1"] == ref["extra"]["om1"]
    assert got["extra"]["om2_contig"] == ref["extra"]["om2_contig"]


def test_planted_response_is_still_called(window):
    r, ov = window
    task_on = ov.onsets_s[ov.labels == pc.TASK]
    rp = S.plant(r, task_on, ["C3", "C4", "Cz", "F3", "F4", "P3", "P4"], pc.CMD_S + 1.0, pc.CMD_S + 10.0, depth=0.8)
    for ref in ("normal", "t"):
        assert tj.cornell_unit(rp, ov, unit="command", ref=ref)["call"]


def test_t_reference_is_never_more_permissive(window):
    r, ov = window
    a = tj.cornell_unit(r, ov, unit="command", ref="normal")
    b = tj.cornell_unit(r, ov, unit="command", ref="t")
    assert b["extra"]["n_sig_allruns"] <= a["extra"]["n_sig_allruns"]
    assert (not b["call"]) or a["call"]


def test_decimated_grid_keeps_one_bin_per_resolution_element(window):
    r, ov = window
    F = dict(pc._feat(r, ov))
    D = tj.decimate(F)
    assert np.allclose(np.diff(D["f"]), 2.0)
    assert D["S"].shape[-1] == len(D["f"]) == 11
    res = tj.cornell_unit(r, ov, unit="command", ref="t", grid="dec")
    assert res["extra"]["grid"] == "dec" and res["extra"]["vif"] > 0


def test_vif_is_near_one_for_independent_samples_and_large_with_dependence():
    g = np.random.default_rng(3)
    grp = np.repeat(np.arange(16), 3)
    v_ind, v_dep = [], []
    for _ in range(100):
        S1, S2 = g.exponential(1.0, (48, 5, 1, 6)), g.exponential(1.0, (48, 5, 1, 6))
        v_ind.append(tj.tgt_grouped(S1, S2, grp, grp)[1].mean())
        S1 *= np.repeat(np.exp(0.6 * g.standard_normal(16)), 3)[:, None, None, None]
        S2 *= np.repeat(np.exp(0.6 * g.standard_normal(16)), 3)[:, None, None, None]
        v_dep.append(tj.tgt_grouped(S1, S2, grp, grp)[1].mean())
    assert 0.7 < np.median(v_ind) < 1.4
    assert np.median(v_dep) > 2.0


def test_summary_reports_variance_inflation_by_sample_level_call():
    rows = []
    for i, (vif, call) in enumerate([(1.0, False), (2.0, False), (3.0, True), (5.0, True)]):
        rows.append({"unit": f"u{i}", "reading": "sample_normal", "call": call, "vif": vif, "matches_stored": True})
        rows.append({"unit": f"u{i}", "reading": "command_normal", "call": False, "vif": vif})
    v = tj.summarize(rows)["vif"]
    assert v["n"] == 4 and v["median"] == pytest.approx(2.5) and v["share_ge_2"] == pytest.approx(0.75)
    assert v["median_sample_positive"] == pytest.approx(4.0) and v["median_sample_negative"] == pytest.approx(1.5)
    assert tj.summarize(rows)["stored_calls_reproduced"] == {"k": 4, "n": 4}
