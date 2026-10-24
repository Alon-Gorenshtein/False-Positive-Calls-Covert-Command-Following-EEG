import importlib
import numpy as np
import pandas as pd

nul = importlib.import_module("91_doc_null")


def test_repeated_testing_probability():
    rows = []
    for rec, p in (("a", 0.0), ("b", 1.0)):
        for k in range(20):
            rows.append({"recording": rec, "offset": k, "call": bool(p)})
    df = pd.DataFrame(rows)
    assert abs(nul.any_call_in_sessions(df, 5) - 0.5) < 1e-9


def test_per_patient_summary():
    df = pd.DataFrame({"recording": ["a"] * 4 + ["b"] * 4, "offset": list(range(4)) * 2,
                       "call": [1, 0, 0, 0, 0, 0, 0, 0]})
    s = nul.per_patient(df)
    assert s["share_patients_any_call"] == 0.5 and s["max_patient_rate"] == 0.25
