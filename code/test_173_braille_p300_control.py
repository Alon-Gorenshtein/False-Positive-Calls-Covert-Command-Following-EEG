"""Tests of 173_braille_p300_control.py on synthetic recordings only (no Braille data)."""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench import pipe_mbvt as mv

bp = importlib.import_module("173_braille_p300_control")
NAMES = ["Fz", "C3", "Cz", "C4", "CP1", "CPz", "CP2", "Pz", "P3", "P4", "O1", "O2"]
TARGETS = [0, 1, 2, 3, 4, 5, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7]


def fake_events(seed=0, n_cycles=16, soa=0.3, gap=14.8, t0=5.0):
    g = np.random.default_rng(seed)
    rows, t = [], t0
    for c in range(n_cycles):
        order = g.permutation(np.repeat(np.arange(8), 10))
        for f in order:
            rows.append({"onset_s": t, "finger": int(f), "is_target": int(f) == TARGETS[c], "cycle": c,
                         "target": TARGETS[c]})
            t += soa
        t += gap
    return pd.DataFrame(rows)


def fake_run(seed=0, p300_uv=0.0, fs=250.0):
    ev = fake_events(seed)
    n = int((ev["onset_s"].max() + 10) * fs)
    g = np.random.default_rng(seed + 100)
    x = (g.standard_normal((len(NAMES), n)) * 5.0).astype(np.float32)
    if p300_uv:
        tt = np.arange(int(0.8 * fs)) / fs
        wave = p300_uv * np.exp(-(tt - 0.4) ** 2 / (2 * 0.07 ** 2))
        topo = np.array([0.5, 0.6, 1.0, 0.6, 0.8, 1.0, 0.8, 1.0, 0.7, 0.7, 0.3, 0.3])
        for t in ev.loc[ev["is_target"], "onset_s"]:
            i = int(round(t * fs))
            x[:, i:i + len(wave)] += (topo[:, None] * wave[None, :]).astype(np.float32)
    return SimpleNamespace(subject=99, session="large", x=x, fs=fs, names=list(NAMES), events=ev, sha256="x")


def test_overlay_groups_hold_one_stimulus_per_finger_and_one_target():
    ev = fake_events(1)
    ov = bp.overlay(ev, (0, 1, 2), {c: TARGETS[c] for c in range(16)})
    grp, pos = ov.meta["group"], ov.meta["pos"]
    assert grp.max() + 1 == 30
    for gi in range(30):
        m = grp == gi
        assert sorted(pos[m]) == list(range(8))
        assert (ov.labels[m] == mv.TARGET).sum() == 1
        assert pos[m][ov.labels[m] == mv.TARGET][0] == TARGETS[gi // 10]


def test_overlay_refuses_a_cycle_with_missing_touches():
    ev = fake_events(2)
    ev = ev.drop(ev.index[3])
    with pytest.raises(ValueError):
        bp.overlay(ev, (0,), {0: 0})


def test_sham_is_a_derangement_that_keeps_the_runs_distinct_fingers():
    true_t = {c: TARGETS[c] for c in range(16)}
    for s in range(50):
        sh = bp.sham_targets(true_t, s)
        assert set(sh) == set(true_t)
        assert all(sh[c] != true_t[c] for c in true_t)
        for cycles in bp.RUNS.values():
            assert len({sh[c] for c in cycles}) == len({true_t[c] for c in cycles})
        assert all(sh[a] == sh[b] for a in true_t for b in true_t if true_t[a] == true_t[b])


def test_finger_specific_responses_without_attention_are_not_called_more_under_sham():
    """Every finger has its own evoked response and nothing depends on attention: true and sham labels must be
    called alike (both near the rule's own chance behaviour), and the randomization test must not call either."""
    calls = {"true": 0, "sham": 0, "rand_true": 0, "rand_sham": 0}
    n = 6
    for seed in range(n):
        r = fake_run(seed + 20)
        g = np.random.default_rng(seed)
        topo = g.standard_normal((8, len(NAMES)))
        fs = r.fs
        tt = np.arange(int(0.6 * fs)) / fs
        wave = 6.0 * np.exp(-(tt - 0.15) ** 2 / (2 * 0.04 ** 2))
        for _, e in r.events.iterrows():
            i = int(round(e["onset_s"] * fs))
            r.x[:, i:i + len(wave)] += (topo[int(e["finger"])][:, None] * wave[None, :]).astype(np.float32)
        rows = bp.run_session(r)
        for v in ("true", "sham"):
            calls[v] += bp.session_calls(rows, "mbvt_primary", v)[(99, "large")]
            calls["rand_" + v] += bp.session_calls(rows, "randomization", v)[(99, "large")]
    assert abs(calls["true"] - calls["sham"]) <= 2, calls
    assert calls["rand_true"] <= 1 and calls["rand_sham"] <= 1, calls


@pytest.fixture(scope="module")
def planted_rows():
    return bp.run_session(fake_run(3, p300_uv=8.0))


def test_planted_p300_is_called_and_sham_is_not(planted_rows):
    s_true = bp.session_calls(planted_rows, "mbvt_primary", "true")
    s_sham = bp.session_calls(planted_rows, "mbvt_primary", "sham")
    assert s_true[(99, "large")] is True
    assert s_sham[(99, "large")] is False
    r = [x for x in planted_rows if x["reading"] == "randomization" and x["version"] == "true" and x["run"] == "A"][0]
    assert r["call"] is True


def test_rows_cover_every_run_version_reading(planted_rows):
    keys = {(r["version"], r["run"], r["reading"]) for r in planted_rows}
    for v in bp.VERSIONS:
        for rn in bp.RUNS:
            for rd in list(bp.READINGS) + ["blind", "randomization"]:
                assert (v, rn, rd) in keys


def test_summary_pairs_sessions_and_counts():
    rows = []
    for s in range(10):
        for v, call in (("true", s < 8), ("sham", s < 1)):
            for rn in bp.RUNS:
                rows.append({"subject": s, "session": "large", "run": rn, "version": v, "reading": "mbvt_primary",
                             "call": call and rn == "A"})
    out = bp.summarize(rows)["readings"]["mbvt_primary"]
    assert out["n_sessions"] == 10 and out["true"]["k"] == 8 and out["sham"]["k"] == 1
    assert out["mcnemar"]["b"] == 7 and out["mcnemar"]["c"] == 0


def test_same_hand_share_recomputes_the_drawn_derangement():
    ev = fake_events(4)
    share = bp.sham_same_hand(99, "large", ev)
    true_t = {c: TARGETS[c] for c in range(16)}
    sh = bp.sham_targets(true_t, np.random.default_rng([20261029, 173, 99, True]))
    for r, cyc in bp.RUNS.items():
        assert share[r] == pytest.approx(np.mean([(sh[c] < 4) == (true_t[c] < 4) for c in cyc]))
