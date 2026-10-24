import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ca", Path(__file__).with_name("150_classifier_agreement.py"))
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)


def test_kappa_perfect_and_chance():
    assert ca.kappa([("A", "A"), ("B", "B")]) == (1.0, 1.0)
    po, k = ca.kappa([("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")])
    assert po == 0.5 and abs(k) < 1e-12


def test_rows_one_per_choice_and_first_class():
    f = {"docs/stated-rule-motor.json": {"p": {
        "v1": {"axes": {"ax": {"value": "x", "class": "STATED", "second_classifier": {"class": "NOT_STATED"}}}},
        "v2": {"axes": {"ax": {"value": "x", "class": "STATED", "second_classifier": {"class": "NOT_STATED"}}}}}},
         "docs/stated-rule-erp.json": {}}
    first = {"docs/stated-rule-motor.json": {"p": {"v1": {"axes": {"ax": {"class": "STATED"}}},
                                                   "v2": {"axes": {"ax": {"class": "CONTRADICTS"}}}}},
             "docs/stated-rule-erp.json": {}}
    pairs = ca.rows(first=first, second=f)
    assert sorted(pairs) == [("CONTRADICTS", "NOT_STATED"), ("STATED", "NOT_STATED")]


def test_rows_skip_axes_without_second_class():
    f = {"docs/stated-rule-motor.json": {"p": {"v1": {"axes": {"ax": {"value": "x", "class": "STATED"}}}}},
         "docs/stated-rule-erp.json": {}}
    first = {"docs/stated-rule-motor.json": {"p": {"v1": {"axes": {"ax": {"class": "STATED"}}}}},
             "docs/stated-rule-erp.json": {}}
    assert ca.rows(first=first, second=f) == []
