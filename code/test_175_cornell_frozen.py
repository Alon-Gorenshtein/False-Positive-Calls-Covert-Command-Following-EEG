"""Tests of 175_cornell_frozen.py on synthetic rows (no EEG)."""
import importlib
import json
import os

import pytest
from scipy.stats import binom

cf = importlib.import_module("175_cornell_frozen")


def _unit(u, sample, primary, reproduced=True, cohort="rep"):
    rows = [{"unit": u, "reading": name, "call": call} for name, call in
            (("sample_normal", sample), ("command_normal", primary), ("command_t", primary), ("command_t_dec", primary))]
    for r in rows:
        r.update(cohort=cohort, stored_call=sample, reproduced=reproduced)
    return rows


def test_freeze_record_matches_constants():
    assert os.path.exists(cf.FREEZE), "the freeze record must be committed before any run"
    frozen = json.load(open(cf.FREEZE))
    assert {k: frozen[k] for k in cf.FROZEN} == cf.FROZEN


def test_not_reproduced_leaves_denominator_and_is_counted():
    rows = _unit("a", True, False) + _unit("b", True, True, reproduced=False) + [{"unit": "c", "cohort": "rep",
                                                                                   "status": "no_published_overlay"}]
    out, units = cf.cohort_summary(rows)
    assert out["n_units"] == 1 and list(units) == ["a"]
    assert out["left_out"] == {"not_reproduced": 1, "no_overlay_or_error": 1, "no_usable_trials": 0}
    rows2 = [{"unit": "d", "reading": "sample_normal", "call": None, "cohort": "rep", "reproduced": False},
             {"unit": "d", "reading": "command_normal", "call": None, "cohort": "rep", "reproduced": False}]
    assert cf.cohort_summary(rows2)[0]["left_out"]["no_usable_trials"] == 1


def test_below_bar_test_is_one_sided_exact():
    assert cf.p_below(2, 342) == pytest.approx(binom.cdf(2, 342, 0.05))


def test_holm_two_cohorts():
    assert cf.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert cf.holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])


def test_mark_compares_sample_call_with_stored():
    rows = [{"unit": "a", "reading": "sample_normal", "call": True}, {"unit": "a", "reading": "command_normal",
                                                                       "call": False}]
    assert all(r["reproduced"] for r in cf.mark([dict(r) for r in rows], True, "rep"))
    assert not any(r["reproduced"] for r in cf.mark([dict(r) for r in rows], False, "rep"))
    assert not any(r["reproduced"] for r in cf.mark([dict(r) for r in rows], None, "rep"))


def test_summary_calibrated_flag_and_strata():
    rows = []
    for i in range(400):
        rows += _unit(f"u{i}", i < 90, i < 2)
    res = cf.summarize({"rep": rows}, strata={"rep": {f"u{i}": ("x" if i < 200 else "y") for i in range(400)}})
    c = res["cohorts"]["rep"]
    assert c["readings"]["command_normal"]["k"] == 2 and c["calibrated"]
    assert res["strata"]["rep"]["x"]["sample_normal"]["k"] == 90 and res["strata"]["rep"]["y"]["command_normal"]["k"] == 0
