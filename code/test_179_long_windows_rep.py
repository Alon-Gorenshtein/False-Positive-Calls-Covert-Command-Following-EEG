"""Tests of 179_long_windows_rep.py (no patient data, no Spark)."""
import importlib

import numpy as np

lw = importlib.import_module("179_long_windows_rep")


def test_long_starts_follow_the_base_runs():
    s = lw.long_starts([(0, 600)], 2186)
    assert s[0] == 0 and s[-1] == 600 + 1800 - 2186 and len(s) == 215
    assert len(lw.long_starts([(0, 50)], 2186)) == 0


def test_choose_takes_the_longest_width_that_fits_and_is_seeded():
    w, s, n = lw.choose([(0, 800)], rank=7)
    assert w == 2550 and 0 <= s <= 800 + 1800 - 2550 and n == 800 + 1800 - 2550 + 1
    assert lw.choose([(0, 800)], rank=7) == (w, s, n)
    assert lw.choose([(0, 100)], rank=7)[0] == 1891
    assert lw.choose([(0, 50)], rank=7) == (0, -1, 0)


def test_every_window_inside_its_stretch():
    g = np.random.default_rng(0)
    for rank in range(50):
        a = int(g.integers(0, 500))
        b = a + int(g.integers(0, 1200))
        w, s, _ = lw.choose([(a, b)], rank)
        if w:
            assert a <= s and s + w <= b + 1800


def test_extractor_carries_the_workers_rule_verbatim():
    src = lw.EXTRACT_SRC
    assert "def padding_seconds" in src and "def eligible_starts" in src and 'h5py.File(path, "r")' in src
    assert "MARGIN_S = 30" in src and "is not %d-s eligible" in src
