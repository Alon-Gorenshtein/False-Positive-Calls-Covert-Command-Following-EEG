"""Tests of 177_cornell_lr.py on synthetic recordings (no patient data)."""
import importlib
from types import SimpleNamespace

import numpy as np

lr = importlib.import_module("177_cornell_lr")
pc = lr.pc


def _synthetic(erd, seed=0, n_trials=40):
    """19 channels at 250 Hz; a 12-Hz rhythm everywhere; with erd, left imagery damps it at C4 and right at C3."""
    g = np.random.default_rng(seed)
    fs, period = lr.FS, 8.0
    n = int((n_trials * period + 10) * fs)
    names = list(pc.NATIVE_CH)
    t = np.arange(n) / fs
    x = g.standard_normal((len(names), n)) * 5.0
    amp = np.ones((len(names), n))
    hands = ["left" if i % 2 == 0 else "right" for i in range(n_trials)]
    onsets = 5.0 + period * np.arange(n_trials)
    for on, h in zip(onsets, hands):
        if erd:
            ch = names.index("C4" if h == "left" else "C3")
            amp[ch, int(on * fs):int((on + 4.0) * fs)] = 0.2
    x += 20.0 * amp * np.sin(2 * np.pi * 12.0 * t + g.uniform(0, 2 * np.pi, (len(names), 1)))
    return SimpleNamespace(x=x, fs=fs, names=names, uid="syn"), onsets, hands


def test_pair_swap_keeps_classes_balanced_and_swaps_half_the_pairs():
    lab = np.array([pc.TASK, pc.REST] * 20)
    out = lr.pair_swap(lab, np.random.default_rng(1))
    assert (out == pc.TASK).sum() == 20
    swapped = sum(out[2 * i] != lab[2 * i] for i in range(20))
    assert swapped == 10


def test_planted_contralateral_erd_is_called_with_the_contralateral_sign():
    rec, on, hands = _synthetic(True)
    ov = lr.overlay(on, hands)
    F = lr.features(rec, ov)
    res, contra, post_only = lr.bins(F, np.asarray(ov.labels)[F["cmd"]], "command")
    assert res["call"] and contra and not post_only


def test_no_effect_is_not_called():
    rec, on, hands = _synthetic(False, seed=3)
    ov = lr.overlay(on, hands)
    F = lr.features(rec, ov)
    res, contra, _ = lr.bins(F, np.asarray(ov.labels)[F["cmd"]], "command")
    assert not res["call"]


def test_summary_reading_rule():
    rows = []
    for s in range(4):
        rows.append({"subject": s, "draw": 0, "reading": "sample", "call": s < 2, "contralateral": s == 0,
                     "posterior_only": False})
        rows += [{"subject": s, "draw": d, "reading": "sample", "call": False, "contralateral": False,
                  "posterior_only": False} for d in range(1, 21)]
    out = lr.summarize(rows)["sample"]
    assert out["detection"]["k"] == 2 and out["true_calls_contralateral"] == 1 and out["sensorimotor_specific"]
    assert out["null_rate"]["estimate"] == 0.0


def test_pad_reflects_and_keeps_the_signal():
    x = np.arange(20.0).reshape(1, 20).repeat(2, 0) * np.ones((2, 1))
    old = lr.PAD_S
    try:
        lr.PAD_S = 2 / lr.FS
        y = lr.pad(x)
    finally:
        lr.PAD_S = old
    assert y.shape == (2, 24) and np.array_equal(y[:, 2:-2], x) and y[0, 0] == x[0, 2]
