import importlib
import json
import os

import pytest

fz = importlib.import_module("152_replication_freeze")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sr():
    return json.load(open(os.path.join(ROOT, "output", "tables", "stated_rule.json")))






def test_prediction_interval():
    lo, hi = fz.prediction_interval(164, 703, 700)
    assert lo < 164 / 703 < hi and abs((hi - lo) / 2 - 1.96 * (0.2333 * 0.7667 * (1 / 703 + 1 / 700)) ** .5) < 1e-3






