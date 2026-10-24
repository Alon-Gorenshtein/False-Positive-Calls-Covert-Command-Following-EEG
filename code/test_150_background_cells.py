import importlib

import pandas as pd

m = importlib.import_module("150_icare_stated_rule")


def test_background_cells_count_calls_by_grade_and_keep_ungraded_patients_in_all():
    calls = pd.DataFrame({"patient": ["a", "b", "c", "d", "e"], "call": [True, False, True, True, False]})
    grades = {"a": "suppressed", "b": "suppressed", "c": "continuous", "d": None, "e": "continuous"}
    c = m.background_cells(calls, grades)
    assert (c["suppressed"]["k"], c["suppressed"]["n"]) == (1, 2)
    assert (c["continuous"]["k"], c["continuous"]["n"]) == (1, 2)
    assert (c["all"]["k"], c["all"]["n"]) == (3, 5) and "None" not in c and "nan" not in c


def test_cli_offers_the_all_windows_background_split():
    src = open(m.__file__).read()
    assert '"--by-background-all"' in src and "def by_background_all_windows" in src
