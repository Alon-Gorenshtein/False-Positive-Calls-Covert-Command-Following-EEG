import importlib
import numpy as np
import pandas as pd
import pytest
from bench import engine as en

mc = importlib.import_module("120_m1_confirm")
mk = mc.mk
bn = mc.bn


def _rec(minutes=60, pad=((5, 25),), fs=128.0, seed=0, uid="synth"):
    """19-channel synthetic segment (uV) with a slow 8-30 Hz-amplitude drift, exact-zero padding over the given
    minute ranges, every Morgoth window normal_awake."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fs)
    drift = np.exp(0.5 * np.sin(2 * np.pi * np.arange(n) / (fs * 600)))
    x = (rng.normal(0, 20, (19, n)) * drift).astype(np.float32)
    for a, b in pad:
        x[:, int(a * 60 * fs): int(b * 60 * fs)] = 0
    t = np.arange(0, minutes * 60, 10.0)
    return en.Recording(source="bdsp", uid=uid, patient="p0", x=x, fs=fs, names=list(en.NAMES19),
                        meta={"morgoth_t": t, "morgoth_lab": np.array(["normal_awake"] * len(t), object)})


def test_clean_slots_and_run_eligibility_match_108():
    rec = _rec()
    x4 = mc.overlay_channels(rec)
    assert x4.shape == (rec.x.shape[1], 4)
    assert np.array_equal(x4[:, 1], rec.x[rec.names.index("F4")].astype(float))
    clean = mc.clean_slots(x4, rec.fs)
    assert len(clean) == mk.n_grid(len(x4), rec.fs)
    assert clean[26] and not clean[27:153].any() and clean[153]
    el = set(bn.eligible_starts(x4.T, rec.fs, run_s=480).tolist())
    for g in range(len(clean) - 48):
        assert bool(clean[g:g + 48].all()) == (10 * g in el)


def test_spans_keep_maximal_clean_stretches_long_enough_for_a_run():
    c = np.r_[np.ones(50, bool), np.zeros(3, bool), np.ones(47, bool), np.zeros(1, bool), np.ones(48, bool)]
    assert mc.spans(c) == [(0, 50), (101, 149)]
    assert mc.spans(np.zeros(10, bool)) == []


def test_run_offsets_are_seeded_distinct_and_inside_spans():
    sp = [(0, 300), (400, 470)]
    a, b, c = mc.run_offsets(sp, 3), mc.run_offsets(sp, 3), mc.run_offsets(sp, 4)
    assert a == b and a != c and len(a) == mc.N_RUNS == 20 and len(set(a)) == 20
    assert all(any(s <= g and g + 48 <= e for s, e in sp) for g in a)
    few = mc.run_offsets([(10, 70)], 3)
    assert sorted(few) == list(range(10, 23))


def test_padding_never_enters_a_series_and_runs_stay_in_clean_spans(monkeypatch):
    rec = _rec()
    seen = []
    real = mk.series_set

    def spy(F, si, n_surr):
        seen.append(np.array(F))
        return real(F, si, n_surr)

    monkeypatch.setattr(mk, "series_set", spy)
    rows, diag = mc.segment_rows(5, rec, n_surr=2, n_perm=30, n_runs=6)
    sp = mc.spans(mc.clean_slots(mc.overlay_channels(rec), rec.fs))
    assert [len(F) for F in seen] == [b - a for a, b in sp if any(a <= g < b for g in diag["offsets"])]
    assert all(F.min() > -10 for F in seen)
    df = pd.DataFrame(rows)
    for g, s in df.groupby("grid_index").span.first().items():
        a, b = sp[s]
        assert a <= g and g + 48 <= b
    assert len(df) == 6 * 4 * (1 + 2 + 1 + 2) and df.offset.nunique() == 6


def test_segment_rows_pair_every_series_on_one_schedule_and_one_test(monkeypatch):
    rec = _rec(pad=())
    real = mk.series_set

    def same(F, si, n_surr):
        S, d = real(F, si, n_surr)
        return {k: [F] * len(v) for k, v in S.items()}, d

    monkeypatch.setattr(mk, "series_set", same)
    df = pd.DataFrame(mc.segment_rows(7, rec, n_surr=2, n_perm=30, n_runs=4)[0])
    assert (df.groupby(["offset", "block"]).p.nunique() == 1).all()
    assert (df.groupby(["offset", "block"]).first_label.nunique() == 1).all()
    assert set(df.stratum) == {"normal_awake"} and set(df.seed_index) == {7} and set(df.recording) == {"synth"}
    assert sorted(df.block.unique()) == [2, 4, 8, 12]


def test_segment_without_an_eligible_run_is_excluded():
    rows, diag = mc.segment_rows(1, _rec(minutes=20, pad=((3, 12),)), n_surr=1, n_perm=10)
    assert rows == [] and diag["excluded"] == "no eligible run"


def test_items_are_the_extras_with_their_csv_row_or_the_dev_split(monkeypatch):
    fake = [dict(uid="d1", meta={"extra": False, "rank": 5}), dict(uid="e1", meta={"extra": True, "extra_row": 9}),
            dict(uid="e2", meta={"extra": True, "extra_row": 2})]
    monkeypatch.setattr(en, "_items", lambda source, split: [it for it in fake if split == "analysis"
                                                             or not it["meta"]["extra"]])
    assert [(j, it["uid"]) for j, it in mc.items("analysis")] == [(9, "e1"), (2, "e2")]
    assert [(j, it["uid"]) for j, it in mc.items("dev")] == [(5, "d1")]


def test_summary_is_117s_nogo_rule_labelled_confirmatory():
    rng = np.random.default_rng(0)
    parts = []
    for series, rate in (("observed", 0.25), ("phase", 0.15), ("prewhitened", 0.05), ("iaaft", 0.15)):
        for blk in (2, 4, 8, 12):
            for i in range(10):
                parts.append(pd.DataFrame({"recording": f"r{i}", "series": series, "block": blk,
                                           "call": rng.random(100) < rate}))
    df = pd.concat(parts, ignore_index=True)
    s = mc.summarize(df, [], {"n_listed": 10, "n_present": 10, "complete": True})
    ref = mk.summarize(df, [])
    assert s["nogo_rule"] == ref["nogo_rule"] and s["rates"] == ref["rates"]
    assert s["label"].startswith("confirmatory") and s["listing"]["complete"] is True
    assert "share of the block-design excess" in s["framing"]


