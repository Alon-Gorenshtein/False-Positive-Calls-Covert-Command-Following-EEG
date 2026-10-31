"""Tests of 172_stroke_native.py on synthetic runs only (no stroke data)."""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench import pipe_cornell as pc
from bench import pipe_dubois as pdb
from bench.tests import _synth as S

sn = importlib.import_module("172_stroke_native")
NAMES = list(dict.fromkeys(list(pc.NATIVE_CH) + list(pdb.CHANSETS["full"])))


def fake_session(seed=0, n_runs=4, n_trials=20, erd=0.0, class_erd=0.0, fs=250.0):
    g = np.random.default_rng(seed)
    runs, trials = [], {}
    for k in range(n_runs):
        rest = 8.0 + np.arange(n_trials) * 15.0
        mi = rest + 8.6
        typ = g.permutation(np.r_[np.full(n_trials // 2, "affected_hand"),
                                  np.full(n_trials - n_trials // 2, "sixth_finger")])
        r = S.rec(rest[-1] + 25.0, fs=fs, seed=seed * 10 + k, names=NAMES)
        if erd:
            r = S.plant(r, mi, ["C3", "C4", "Cz", "F3", "F4", "P3", "P4"], 0.3, 6.5, depth=erd)
        if class_erd:
            r = S.plant(r, mi[typ == "affected_hand"], ["C3", "F3", "P3"], 0.3, 6.5, depth=class_erd)
        run = SimpleNamespace(subject=99, timepoint="pre", run=k + 1, x=r.x, fs=fs, names=list(NAMES),
                              sha256=f"h{k}")
        trials[k + 1] = pd.DataFrame({"rest_s": rest, "mi_s": mi, "mi_type": typ})
        runs.append(run)
    return runs, (lambda run: trials[run.run])


def test_cornell_overlay_interleaves_rest_then_imagery():
    tr = pd.DataFrame({"rest_s": [10.0, 25.0], "mi_s": [18.6, 33.6], "mi_type": [2, 3]})
    ov = sn.cornell_overlay(tr, 2)
    assert list(ov.onsets_s) == [10.0, 18.6, 25.0, 33.6]
    assert list(ov.labels) == [pc.REST, pc.TASK, pc.REST, pc.TASK]
    assert set(ov.meta["run"]) == {2}


@pytest.fixture(scope="module")
def planted():
    return fake_session(1, erd=0.8)


def test_features_merge_runs_and_label_epochs(planted):
    runs, tr = planted
    F, lab = sn.cornell_features(runs, tr)
    assert F["S"].shape[0] == len(lab) == 4 * 20 * 2 * 2
    assert set(np.unique(F["run"])) == {0, 1, 2, 3}
    assert (lab == pc.TASK).sum() == (lab == pc.REST).sum()
    assert len(np.unique(F["cmd"])) == 4 * 40


def test_rest_sham_splits_each_rest_period_and_uses_no_imagery(planted):
    runs, tr = planted
    F, lab = sn.cornell_features(runs, tr)
    Fs, ls = sn.cornell_sham(F, lab, 0)
    assert len(ls) == int((lab == pc.REST).sum())
    assert set(np.unique(Fs["cmd"] // 2)) == set(np.unique(F["cmd"][lab == pc.REST]))
    for c in np.unique(Fs["cmd"] // 2):
        m = Fs["cmd"] // 2 == c
        assert sorted(ls[m]) == sorted([pc.TASK, pc.REST])
    assert (ls == pc.TASK).sum() == (ls == pc.REST).sum()
    pos = (Fs["cmd"] % 2)
    for r in np.unique(Fs["run"]):
        m = Fs["run"] == r
        first_task = len(np.unique(Fs["cmd"][m & (pos == 0) & (ls == pc.TASK)] // 2))
        assert first_task == len(np.unique(Fs["cmd"][m] // 2)) // 2


def test_mismatch_swaps_whole_trials_and_keeps_epochs(planted):
    runs, tr = planted
    F, lab = sn.cornell_features(runs, tr)
    lm = sn.cornell_mismatch(F, lab, 1)
    trial = F["cmd"] // 2
    for t in np.unique(trial):
        m = trial == t
        same, swapped = np.array_equal(lm[m], lab[m]), np.array_equal(lm[m], np.where(lab[m] == pc.TASK, pc.REST, pc.TASK))
        assert same or swapped
    for r in np.unique(F["run"]):
        ts = np.unique(trial[F["run"] == r])
        n_sw = sum(not np.array_equal(lm[trial == t], lab[trial == t]) for t in ts)
        assert n_sw == len(ts) // 2


def test_null_design_is_not_called_under_drift_by_the_true_or_the_matched_nulls(monkeypatch):
    """No response anywhere, strong slow drift: the recorded design and both matched nulls stay quiet under the
    command-level reading (the c3 reading is reported, not asserted)."""
    monkeypatch.setattr(sn, "SHAM_DRAWS", 3)
    calls = {"true": 0, "sham": 0, "mismatch": 0}
    for seed in range(3):
        runs, tr = fake_session(40 + seed)
        for r in runs:
            r.x = S.noise(r.x.shape[1] / r.fs, r.fs, seed=100 + seed * 10 + r.run, drift=1.0, names=NAMES)
        rows = sn.cornell_rows(runs, tr, {"subject": 99, "timepoint": "pre", "session_id": "S99_pre"}, seed)
        for r in rows:
            if r["reading"] == "cornell_cmd_t" and r["call"]:
                calls[r["version"]] += 1
    assert calls["true"] == 0 and calls["sham"] <= 1 and calls["mismatch"] <= 1, calls


def test_trials_without_margin_are_dropped():
    run = SimpleNamespace(x=np.zeros((2, int(100 * 250))), fs=250.0)
    tr = pd.DataFrame({"rest_s": [2.0, 10.0, 80.0], "mi_s": [10.6, 18.6, 88.6], "mi_type": ["affected_hand"] * 3})
    assert list(sn.usable(run, tr)["rest_s"]) == [10.0]


def test_planted_imagery_is_called_and_nulls_rarely(planted, monkeypatch):
    runs, tr = planted
    monkeypatch.setattr(sn, "SHAM_DRAWS", 5)
    rows = sn.cornell_rows(runs, tr, {"subject": 99, "timepoint": "pre", "session_id": "S99_pre"}, 0)
    t = {r["reading"]: r["call"] for r in rows if r["version"] == "true"}
    assert t["cornell_c3"] and t["cornell_cmd"]
    for v in ("sham", "mismatch"):
        sh = [r["call"] for r in rows if r["version"] == v and r["reading"] == "cornell_cmd_t"]
        assert len(sh) == 5 and sum(sh) <= 1, (v, sh)


def test_dubois_rows_true_and_sham(monkeypatch):
    runs, tr = fake_session(2, n_runs=4, class_erd=0.9)
    monkeypatch.setattr(sn, "SHAM_DRAWS", 2)
    rows = sn.dubois_rows(runs, tr, {"subject": 99, "timepoint": "pre", "session_id": "S99_pre"}, 0)
    got = {(r["reading"], r["version"], r["draw"]) for r in rows}
    assert got == {(rd, v, d) for rd in ("dubois_literal", "dubois_reading_b") for v, d in
                   (("true", 0), ("sham", 1), ("sham", 2))}
    assert all(r["call"] is not None for r in rows), [r.get("status") for r in rows]
    t = {r["reading"]: r for r in rows if r["version"] == "true"}
    assert t["dubois_literal"]["peak_da"] > 0.7


def test_summary_detection_sham_and_difference():
    rows = []
    for p in range(6):
        for tp in ("pre", "post"):
            sid = f"S{p}_{tp}"
            base = {"session_id": sid, "sha256": sid, "subject": p, "timepoint": tp}
            rows.append({**base, "reading": "cornell_cmd", "version": "true", "call": p < 4})
            for d in range(1, 11):
                rows.append({**base, "reading": "cornell_cmd", "version": "sham", "call": d == 1})
    out = sn.summarize(rows)["readings"]["cornell_cmd"]
    assert out["n_sessions"] == 12 and out["n_patients"] == 6
    assert out["detection"]["rate"] == pytest.approx(8 / 12)
    assert out["nulls"]["sham"]["rate"]["rate"] == pytest.approx(0.1)
    assert out["nulls"]["sham"]["separates"] is True


def test_planted_imagery_does_not_leak_into_the_mismatch(monkeypatch):
    runs, tr = fake_session(7, erd=0.8)
    monkeypatch.setattr(sn, "SHAM_DRAWS", 10)
    rows = sn.cornell_rows(runs, tr, {"subject": 99, "timepoint": "pre", "session_id": "S99_pre"}, 3)
    mm = [r["call"] for r in rows if r["version"] == "mismatch" and r["reading"] == "cornell_c3"]
    assert sum(mm) <= 1, mm


def test_calibrated_bar_only_for_the_rest_sham():
    rows = []
    for p in range(4):
        sid = f"S{p}_pre"
        base = {"session_id": sid, "sha256": sid, "subject": p, "timepoint": "pre"}
        rows.append({**base, "reading": "cornell_cmd", "version": "true", "call": True})
        for d in range(1, 6):
            rows.append({**base, "reading": "cornell_cmd", "version": "sham", "call": False})
            rows.append({**base, "reading": "cornell_cmd", "version": "mismatch", "call": d == 1})
    out = sn.summarize(rows)["readings"]["cornell_cmd"]["nulls"]
    assert "calibrated_upper_le_10pct" in out["sham"] and "calibrated_upper_le_10pct" not in out["mismatch"]


def test_duplicate_sessions_rule():
    rows = [{"session_id": "S2_post", "sha256": "a", "subject": 2, "timepoint": "post"},
            {"session_id": "S2_follow", "sha256": "a", "subject": 2, "timepoint": "follow"},
            {"session_id": "S21_pre", "sha256": "b", "subject": 21, "timepoint": "pre"},
            {"session_id": "S22_pre", "sha256": "b", "subject": 22, "timepoint": "pre"},
            {"session_id": "S3_pre", "sha256": "c", "subject": 3, "timepoint": "pre"}]
    d = sn.duplicate_sessions(rows)
    assert set(d) == {"S2_follow", "S21_pre", "S22_pre"}
