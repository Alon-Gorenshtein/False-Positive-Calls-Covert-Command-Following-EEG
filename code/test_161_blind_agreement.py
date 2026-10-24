import importlib

import numpy as np
import pytest

ba = importlib.import_module("161_blind_agreement")


class Ov:
    def __init__(self, onsets, labels, meta):
        self.onsets_s, self.labels, self.meta, self.kind = np.asarray(onsets), np.asarray(labels), meta, "x"


def test_dubois_events_keep_class_and_block():
    ov = Ov([10.0, 18.0, 26.0], [1, 2, 1], {"block": np.array([0, 0, 1])})
    assert ba.dubois_events(ov, start_s=0.0) == [(10.0, "left", 0), (18.0, "right", 0), (26.0, "left", 1)]


def test_p300_events_relabel_target_as_left():
    ov = Ov([1.0, 1.2, 1.4], [1, 2, 3], {})
    assert ba.p300_events(ov, start_s=0.5) == [(0.5, "left"), (0.7, "right"), (0.9, "distractor")]


def test_kappa():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    assert ba.kappa(a, a) == pytest.approx(1.0)
    assert ba.kappa(a, ~a) < 0
