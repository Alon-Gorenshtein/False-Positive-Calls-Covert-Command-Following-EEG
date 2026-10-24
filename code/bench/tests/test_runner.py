"""Tests for code/114_benchmark.py. Synthetic recordings and stub pipeline
modules only; nothing here reads real EEG. The dev-only guard and the dev/analysis separation are each also run
against a deliberately broken version, which the same checks must catch.
cd code && /usr/bin/python3 -m pytest bench/tests/test_runner.py -q"""
import importlib
import json
import os
import sys
import types
import zlib

import numpy as np
import pytest

from bench import engine as en
from bench import motor_common as mc

rn = importlib.import_module("114_benchmark")
NAMES19 = list(en.NAMES19)
SECRETS = ("SECRET-HIS-77", "1950-01-01", "2001-02-03", "DOE^JOHN")



def _rec(uid="u1", patient="p1", dur=400.0, fs=128.0, seed=0, source="pdoc", dev=True, names=NAMES19, meta=None):
    g = np.random.default_rng(seed)
    x = (10 * g.standard_normal((len(names), int(dur * fs)))).astype(np.float32)
    m = {"dev": dev, "subject_info": {"his_id": SECRETS[0], "birthday": SECRETS[1]}, "meas_date": SECRETS[2],
         "patient_name": SECRETS[3]}
    m.update(meta or {})
    return en.Recording(source=source, uid=uid, patient=patient, x=x, fs=fs, names=list(names), meta=m)


def _stub(name="stub", family="cmd", run_s=60.0):
    """A fast pipeline module with the interface of bench/pipe_*.py. Variants a and b share a schedule, c does not
    (and sits between them, as cornell's r2/r4 variants alternate)."""
    m = types.ModuleType(f"bench.pipe_{name}")
    m.NAME, m.GRADE, m.FAMILY = name, "AS DESCRIBED", family
    m.NATIVE_CH = list(NAMES19)
    m.RUN_S = run_s
    m.VARIANTS = {"a": {"as_coded": True, "gap": 5.0}, "c": {"gap": 6.0}, "b": {"as_coded": False, "gap": 5.0}}
    m.DEFAULT = "a"
    m.TARGET = 1
    m.RESP_LABELS = (1,)
    m.LOG = []
    kind = "cmd" if family == "cmd" else "stim"

    def _ov(start_s, g, variant, randomized):
        on = float(start_s) + np.arange(8) * mc.cfg(m.VARIANTS, variant)["gap"]
        lab = g.permutation(np.repeat([1, 2], 4))
        return en.Overlay(onsets_s=on, labels=lab, kind=kind,
                          meta={"resp_hand": {1: "left", 2: "right"}, "resp_dur_s": 2.0, "randomized": randomized,
                                "start_s": float(start_s)})

    def schedule(start_s, rng, variant="a"):
        m.LOG.append(("schedule", variant, float(start_s)))
        return _ov(start_s, mc.gen(rng), variant, False)

    def schedule_randomized(start_s, rng, variant="a"):
        m.LOG.append(("schedule_randomized", variant, float(start_s)))
        return _ov(start_s, mc.gen(rng), variant, True)

    def extent(ov):
        return float(ov.onsets_s.min()) - 1.0, float(ov.onsets_s.max()) + 4.0

    def _stat(rec, ov, rng):
        i = np.round(np.asarray(ov.onsets_s) * rec.fs).astype(int)
        return float(np.mean(np.asarray(rec.x, float)[:, i] ** 2) / 100.0 + 0.01 * mc.gen(rng).normal())

    def run(rec, overlay, variant="a", rng=None):
        s = _stat(rec, overlay, rng)
        m.LOG.append(("run", variant, overlay.meta["start_s"], list(rec.names), list(m.NATIVE_CH), overlay))
        return mc.result(s, 0.5, s > 1.0, n_kept=len(overlay.onsets_s), curve=np.arange(100), nested={"a": 1})

    def run_randomized(rec, overlay, variant="a", rng=None):
        s = _stat(rec, overlay, rng)
        m.LOG.append(("rand", variant, overlay.meta["start_s"], list(rec.names), list(m.NATIVE_CH), overlay))
        return mc.result(s, 0.2, False, n_perm_run=7)

    m.schedule, m.schedule_randomized, m.extent = schedule, schedule_randomized, extent
    m.run, m.run_randomized = run, run_randomized
    m.stat_for_size_correction = lambda r: r["stat"]
    m.patient_rule = lambda results: any(r["call"] for r in results)
    return m


def _ctx(**kw):
    c = dict(null="ncmd_pdoc", split="dev", montage="native", resp="none", n_recs=3, max_overlays=None,
             locator={"p1"}, uncorrected=set(), col113=None)
    c.update(kw)
    return c


def _rows(fn):
    with open(fn) as fh:
        return [json.loads(l) for l in fh]


def _collect(rec, mod, **kw):
    out = []
    rn.run_recording(rec, mod, rn.parse_null(kw.pop("null", "ncmd_pdoc")), _ctx(**kw), set(), out.append)
    return out


@pytest.fixture
def roots(tmp_path, monkeypatch):
    sh = {"dev": str(tmp_path / "sh_dev"), "analysis": str(tmp_path / "sh_analysis")}
    monkeypatch.setattr(rn, "SH", sh)
    empty = tmp_path / "touched_none.json"
    empty.write_text(json.dumps({"excluded": []}))
    monkeypatch.setattr(rn, "TOUCHED_FILE", str(empty))
    frozen = [tmp_path / "dev_split.json", tmp_path / "passive_dev_split.json"]
    for f in frozen:
        f.write_text("{}")
    monkeypatch.setattr(rn, "FROZEN_FILES", tuple(str(f) for f in frozen))
    return sh


def _spec_path(tmp_path, locked):
    """An empty stand-in for the private notes file the working copy's guard reads (not read in this copy)."""
    p = tmp_path / ("notes_s2.txt" if locked else "notes.txt")
    p.write_text("")
    return str(p)


def _fake_engine(monkeypatch, recs_by_split, calls=None):
    """Engine listing/loading replaced by in-memory recordings (the runner uses en._items and en._load)."""
    def items(source, split):
        if calls is not None:
            calls.append((source, split))
        return [{"source": source, "uid": r.uid, "patient": r.patient, "rec": r} for r in recs_by_split.get(split, [])]
    monkeypatch.setattr(en, "_items", items)
    monkeypatch.setattr(en, "_load", lambda it: it["rec"])



def test_m_overlays_reach_200_draws_with_at_most_10():
    assert [rn.m_overlays(n) for n in (3, 20, 38, 200, 480)] == [10, 10, 6, 1, 1]


def test_overlays_are_deterministic_and_the_first_is_the_seeded_clean_choice():
    mod, rec = _stub(), _rec(dur=2000)
    a = rn.plan_overlays(rec, mod, "pdoc", 3, ["a", "b", "c"])
    b = rn.plan_overlays(rec, mod, "pdoc", 3, ["a", "b", "c"])
    assert [o["start_s"] for o in a] == [o["start_s"] for o in b] and len(a) == 10
    need = rn.need_seconds(mod, ["a", "b", "c"])
    spans = en.clean_spans(rec, need)
    seed = rn.SEED + zlib.crc32(("stub" + "u1").encode())
    assert a[0]["start_s"] == np.random.default_rng(seed).permutation(spans)[0] + rn.LEAD_S
    s = np.array([o["start_s"] for o in a])
    d = np.abs(s[:, None] - s[None])[np.triu_indices(len(s), 1)]
    assert (d >= need).all() and not any(o["overlap"] for o in a)
    c = rn.plan_overlays(_rec(uid="u2", dur=2000), mod, "pdoc", 3, ["a", "b", "c"])
    assert c[0]["start_s"] != a[0]["start_s"]
    other = _stub(name="stub2")
    assert rn.plan_overlays(rec, other, "pdoc", 3, ["a"])[0]["start_s"] != a[0]["start_s"]


def test_overlapping_draws_fill_m_only_when_the_recording_is_too_short():
    mod, rec = _stub(), _rec(dur=300)
    p = rn.plan_overlays(rec, mod, "pdoc", 3, ["a"])
    assert len(p) == 10 and len({o["start_s"] for o in p}) == 10
    assert p[0]["seq"] and not p[0]["overlap"] and any(o["overlap"] for o in p)
    assert all(not o["seq"] for o in p if o["overlap"])


def test_sequence_windows_for_composite_rules():
    rec = _rec(dur=3000)
    col = rn.plan_overlays(rec, _stub(name="columbia"), "pdoc", 480, ["a"])
    assert sum(o["seq"] for o in col) == 8 and len(col) == 8
    dub = rn.plan_overlays(rec, _stub(name="dubois"), "pdoc", 480, ["a"])
    assert sum(o["seq"] for o in dub) == 12
    one = rn.plan_overlays(rec, _stub(name="cornell"), "pdoc", 480, ["a"])
    assert len(one) == 1 and one[0]["seq"]
    mb = rn.plan_overlays(rec, _stub(name="mbvt"), "pdoc", 480, ["a"])
    need = rn.need_seconds(_stub(name="mbvt"), ["a"])
    tiles = en.patient_windows([rec], need)
    assert len(mb) == len(tiles) and {o["start_s"] - rn.LEAD_S for o in mb} == {s for _, s in tiles}
    mb_bdsp = rn.plan_overlays(_rec(dur=3000, source="bdsp"), _stub(name="mbvt"), "bdsp", 480, ["a"])
    assert len(mb_bdsp) == 10 and all(o["seq"] for o in mb_bdsp)


def test_max_overlays_truncates_without_changing_starts():
    mod, rec = _stub(), _rec(dur=2000)
    full = rn.plan_overlays(rec, mod, "pdoc", 3, ["a"])
    cut = rn.plan_overlays(rec, mod, "pdoc", 3, ["a"], max_overlays=2)
    assert [o["start_s"] for o in cut] == [o["start_s"] for o in full[:2]]


def test_too_short_recording_gives_one_no_fit_row():
    rows = _collect(_rec(dur=60), _stub())
    assert len(rows) == 1 and rows[0]["status"] == "no_fit" and rows[0]["variant"] is None



def test_every_variant_runs_and_randomized_uses_each_groups_representative_at_the_same_start():
    mod = _stub()
    rows = _collect(_rec(dur=2000), mod, n_recs=480)
    runs = [e for e in mod.LOG if e[0] == "run"]
    rands = [e for e in mod.LOG if e[0] == "rand"]
    assert sorted(e[1] for e in runs) == ["a", "b", "c"]
    assert [e[1] for e in rands] == [dict(mod.VARIANTS[g], n_perm_rand=200) for g in ("a", "c")]
    assert len({e[2] for e in runs + rands}) == 1
    assert rands[0][5].meta["randomized"] and not runs[0][5].meta["randomized"]
    assert len(rows) == 3 and {r["variant"] for r in rows} == {"a", "b", "c"}
    by = {r["variant"]: r for r in rows}
    assert (by["a"]["rand_stat"], by["a"]["rand_p"]) == (by["b"]["rand_stat"], by["b"]["rand_p"])
    assert [by[v]["rand_variant"] for v in "abc"] == ["a", "a", "c"] and all(r["status"] == "ok" for r in rows)


def test_randomized_only_overlays_carry_no_published_variant():
    mod = _stub()
    rows = _collect(_rec(dur=2000), mod, n_recs=3)
    seq = [r for r in rows if r["seq"]]
    rest = [r for r in rows if not r["seq"]]
    assert len(seq) == 3 and len(rest) == 9 * 2 and all(r["variant"] is None and r["stat"] is None for r in rest)
    assert sorted(r["group"] for r in rest) == ["a"] * 9 + ["c"] * 9
    assert all(r["rand_stat"] is not None for r in rows)


def test_rows_are_reproducible():
    a = _collect(_rec(dur=2000), _stub())
    b = _collect(_rec(dur=2000), _stub())
    strip = lambda rs: [{k: v for k, v in r.items() if "seconds" not in k} for r in rs]
    assert strip(a) == strip(b)


def test_rows_carry_the_covariates():
    r = _collect(_rec(dur=2000), _stub(), n_recs=480)[0]
    for k in ("uid", "patient", "start_s", "stratum", "transitioning", "artifact_share", "artifact_label", "montage",
              "extent_t0", "extent_t1", "n_usable", "stat", "p", "call", "size_stat", "rand_stat", "rand_p",
              "rand_call", "seconds", "rand_seconds", "split", "dev", "key"):
        assert k in r
    assert r["stratum"] == "na" and r["n_usable"] == 8 and r["montage"] == "native"


def test_no_header_fields_or_signal_in_rows():
    rows = _collect(_rec(dur=2000), _stub())
    text = json.dumps(rows)
    for s in SECRETS + ("subject_info", "meas_date", "patient_name", "birthday", "his_id"):
        assert s not in text
    for r in rows:
        assert all(v is None or isinstance(v, (str, int, float, bool)) for v in r.values())


def test_family_mismatch_is_refused():
    with pytest.raises(ValueError):
        rn.run_recording(_rec(), _stub(family="erp"), rn.parse_null("ncmd_pdoc"), _ctx(), set(), lambda r: None)



def test_grid_locates_on_statistics_only_seeded_per_uid_and_effect(monkeypatch):
    calls = []
    real = en.inject

    def spy(rec, overlay, resp_mask, model, effect, rng, **kw):
        calls.append((model, effect, np.asarray(resp_mask).copy(), bool(overlay.meta["randomized"])))
        return real(rec, overlay, resp_mask, model, effect, rng, **kw)
    monkeypatch.setattr(en, "inject", spy)
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erd": [0.5], "ers": [0.5], "erp": [3.0]}, "test"))
    mod = _stub()
    null = _collect(_rec(dur=2000), mod, n_recs=480)
    grid = _collect(_rec(dur=2000), mod, n_recs=480, resp="grid")
    assert len(grid) == 3 * 2 * 2
    assert {r["start_s"] for r in grid} == {r["start_s"] for r in null}
    assert len(calls) == 2 * 2 * 2 and not any(c[3] for c in calls)
    assert all(r["mode"] == "stat_only" and r["in_locator"] and "rand_stat" not in r for r in grid)
    grid2 = _collect(_rec(dur=2000), _stub(), n_recs=480, resp="grid")
    assert [r["stat"] for r in grid] == [r["stat"] for r in grid2]
    half = [c for c in calls if c[0] == "erd" and len(c[2]) == 8]
    assert any(not c[2].all() for c in half) and any(c[2].all() for c in half)


def test_grid_runs_only_locator_patients():
    monkeypatch_grid = {"erd": [0.5], "ers": [0.5]}
    rn_grid = rn.load_grid
    try:
        rn.load_grid = lambda split: (monkeypatch_grid, "test")
        assert _collect(_rec(dur=2000, patient="p9"), _stub(), n_recs=480, resp="grid") == []
    finally:
        rn.load_grid = rn_grid


def test_responders_run_on_the_first_overlay_only(monkeypatch):
    """Sensitivity pairs the proposed method with the comparator on one overlay per recording (j = 0, the null run's
    seeded overlay); randomized-only overlays have no comparator, so responders are not injected there."""
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erd": [0.5], "ers": [0.5]}, "test"))
    null = _collect(_rec(dur=2000), _stub(), n_recs=3)
    grid = _collect(_rec(dur=2000), _stub(), n_recs=3, resp="grid")
    assert {r["j"] for r in grid} == {0} and len(grid) == 3 * 2 * 2
    assert {r["start_s"] for r in grid} == {r["start_s"] for r in null if r["j"] == 0}


STUB_GROUP = {"a": "a", "b": "a", "c": "c", "d": "a"}


def _eval_file(tmp_path, monkeypatch, points):
    """Frozen evaluation points; a {model|prob: {effect, comparator}} entry is one group (the comparator's)."""
    def one(e):
        if "groups" in e:
            return e
        g = STUB_GROUP[e["comparator"]]
        return {"groups": {g: dict(e, reached=True)}, "pipeline_group": g}
    points = {n: {p: {k: one(e) for k, e in d.items()} for p, d in v.items()} for n, v in points.items()}
    fn = tmp_path / "eval_points.json"
    fn.write_text(json.dumps(points))
    monkeypatch.setattr(rn, "EVAL_FILE", {"dev": str(fn), "analysis": str(fn)})


def test_eval_runs_the_comparator_statistic_the_proposed_method_and_the_subsample_permutation(monkeypatch, tmp_path):
    _eval_file(tmp_path, monkeypatch, {"ncmd_pdoc": {"stub": {"erd|1": {"effect": 0.5, "comparator": "b"},
                                                               "ers|0.5": {"effect": 0.2, "comparator": "c"},
                                                               "erd|0.5": {"effect": None, "comparator": "a"}}}})
    mod = _stub()
    rows = _collect(_rec(dur=2000), mod, n_recs=480, resp="eval", locator=set(), uncorrected={"p1"})
    assert sorted((r["model"], r["prob"], r["effect"], r["variant"]) for r in rows) == [
        ("erd", 1.0, 0.5, "b"), ("ers", 0.5, 0.2, "c")]
    for r in rows:
        assert r["rand_stat"] is not None and r["rand_variant"] == STUB_GROUP[r["variant"]]
        assert r["in_uncorrected"] and not r["in_locator"]
        assert r["perm_stat"] == r["stat"] and r["perm_p"] is not None
    assert len([e for e in mod.LOG if e[0] == "rand"]) == 2
    rows2 = _collect(_rec(dur=2000), _stub(), n_recs=480, resp="eval", locator=set(), uncorrected=set())
    assert all(r["perm_stat"] is None for r in rows2)
    with pytest.raises(RuntimeError):
        monkeypatch.setattr(rn, "EVAL_FILE", {"dev": str(tmp_path / "none.json")})
        _collect(_rec(dur=2000), _stub(), n_recs=480, resp="eval")


def test_erp_responders_only_at_target_like_onsets(monkeypatch):
    masks = []
    real = en.inject
    monkeypatch.setattr(en, "inject", lambda rec, ov, m, *a, **k: (masks.append((np.asarray(ov.labels).copy(),
                                                                                np.asarray(m).copy())),
                                                                  real(rec, ov, m, *a, **k))[1])
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erp": [3.0]}, "test"))
    rows = _collect(_rec(dur=2000), _stub(family="erp"), null="nstim0_pdoc", n_recs=480, resp="grid")
    assert {r["model"] for r in rows} == {"erp"}
    assert all(not m[lab != 1].any() for lab, m in masks) and any(m.any() for _, m in masks)


def test_subsamples_are_seeded_disjoint_and_order_free():
    pats = [f"P{i:03d}" for i in range(300)]
    loc, unc = rn.subsamples(pats, "bdsp")
    assert len(loc) == 60 and len(unc) == 100 and not loc & unc
    assert rn.subsamples(pats[::-1] + pats[:5], "bdsp") == (loc, unc)
    assert rn.subsamples(pats, "pdoc")[0] != loc
    small = rn.subsamples(pats[:38], "pdoc")
    assert small[0] == set(pats[:38]) and small[1] == set()


def test_grid_mode_loads_only_locator_patients(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erd": [0.5], "ers": [0.5]}, "test"))
    t = np.arange(0, 300, 10.0)
    recs = [_rec(uid=f"r{k}", patient=f"P{k:03d}", dur=300, seed=k, source="bdsp",
                 meta={"morgoth_t": t, "morgoth_lab": np.array(["normal_awake"] * len(t))}) for k in range(80)]
    _fake_engine(monkeypatch, {"dev": recs})
    loaded = []
    real = en._load
    monkeypatch.setattr(en, "_load", lambda it: (loaded.append(it["patient"]), real(it))[1])
    rn.run_shard("ncmd_bdsp", "stub", "dev", resp="grid", spec_path=_spec_path(tmp_path, False))
    loc, _ = rn.subsamples([r.patient for r in recs], "bdsp")
    assert set(loaded) == loc and len(loaded) == 60



def _with_short(monkeypatch, mod):
    short = {"a_short": {"arg": dict(mod.VARIANTS["a"], gap=2.0), "globals": {"RUN_S": 30.0}, "twin_of": ["a"],
                         "short": True}}
    real = rn.wr.short_variants
    monkeypatch.setattr(rn.wr, "short_variants", lambda m, source: short if m is mod else real(m, source))


def test_native_and_short_twins_share_a_start_where_native_fits(monkeypatch):
    mod = _stub()
    _with_short(monkeypatch, mod)
    long = _collect(_rec(dur=2000), mod, n_recs=480)
    assert {r["variant"] for r in long} == {"a", "b", "c", "a_short"} and len({r["start_s"] for r in long}) == 1
    tw = [r for r in long if r["variant"] == "a_short"][0]
    assert tw["short"] and tw["twin_of"] == "a" and tw["status"] == "ok"
    short_only = _collect(_rec(dur=58, uid="u9"), mod, n_recs=480)
    assert [r["variant"] for r in short_only] == ["a_short"]
    assert short_only[0]["rand_variant"] == "a_short" and long[0]["rand_variant"] == "a"
    assert [e[1] for e in mod.LOG if e[0] == "rand"][-1] == dict(mod.VARIANTS["a"], gap=2.0, n_perm_rand=200)
    assert _collect(_rec(dur=40, uid="u8"), mod, n_recs=480)[0]["status"] == "no_fit"


def test_icare_is_dropped_from_the_command_pipelines():
    with pytest.raises(ValueError, match="8-min"):
        rn.parse_null("ncmd_icare")



@pytest.fixture(autouse=True)
def _stub_columbia_randomized(monkeypatch):
    """A stub named "columbia" has no 112 features: its own run_randomized stands in for the B wrapper's 112 call."""
    real = rn.wr.run_randomized

    def rand(mod, rec, ov, v, rng):
        if mod.NAME == "columbia" and not hasattr(mod, "ce"):
            return mod.run_randomized(rec, ov, v, rng)
        return real(mod, rec, ov, v, rng)
    monkeypatch.setattr(rn.wr, "run_randomized", rand)


def _bdsp_rec(uid, dur=4000, seed=0):
    t = np.arange(0, dur, 10.0)
    return _rec(uid=uid, patient=f"pt_{uid}", dur=dur, seed=seed, source="bdsp", dev=False,
                meta={"morgoth_t": t, "morgoth_lab": np.array(["normal_awake"] * len(t))})


def test_columbia_null_adds_the_second_window_beside_113s(monkeypatch):
    mod = _stub(name="columbia", run_s=1505.0)
    rec = _bdsp_rec("c1")
    rows = _collect(rec, mod, null="ncmd_bdsp", n_recs=480, col113={"c1": 500.0})
    assert len(rows) == 3 and {r["j"] for r in rows} == {1} and all("rand_stat" not in r for r in rows)
    s = rows[0]["start_s"]
    assert s - rn.LEAD_S >= 500.0 + 1505 or s - rn.LEAD_S + rn.need_seconds(mod, ["a"]) <= 499.0
    assert _collect(_bdsp_rec("c2", dur=2000), mod, null="ncmd_bdsp", n_recs=480,
                    col113={"c2": 100.0})[0]["status"] == "no_second_window"
    assert _collect(_bdsp_rec("c3"), mod, null="ncmd_bdsp", n_recs=480, col113={})[0]["status"] == "not_in_113"


def test_columbia_responders_use_113s_start(monkeypatch):
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erd": [0.5], "ers": [0.5]}, "test"))
    monkeypatch.setattr(rn.wr, "wrapped", lambda name, cfg: False)
    mod = _stub(name="columbia", run_s=1505.0)
    rows = _collect(_bdsp_rec("c1"), mod, null="ncmd_bdsp", n_recs=480, col113={"c1": 500.0}, resp="grid",
                    locator={"pt_c1"})
    assert {r["start_s"] for r in rows} == {500.0} and {r["j"] for r in rows} == {0}
    assert {r["status"] for r in rows} == {"ok"}


def test_columbia_analysis_reads_113_rows_and_refuses_without_them(monkeypatch, roots, tmp_path):
    import pandas as pd
    csv = tmp_path / "sample.csv"
    pd.DataFrame({"patient_id": ["pt_c1", "pt_c2"], "file": ["c1.mat", "c2.mat"]}).to_csv(csv, index=False)
    pq = tmp_path / "col.parquet"
    pd.DataFrame({"sample_idx": [0], "start_s": [500.0], "i_auc": [0.5], "i_p": [0.4], "i_call": [False],
                  "iii_auc": [0.5], "iii_p": [0.6], "iii_call": [False], "stratum": ["normal_awake"],
                  "transition": [False], "rank": [3], "primary": [True]}).to_parquet(pq)
    monkeypatch.setattr(rn, "SAMPLE_CSV", str(csv))
    monkeypatch.setattr(rn, "COLUMBIA_ROWS", str(tmp_path / "missing.parquet"))
    monkeypatch.setitem(sys.modules, "bench.pipe_columbia", _stub(name="columbia", run_s=1505.0))
    _fake_engine(monkeypatch, {"analysis": [_bdsp_rec("c1"), _bdsp_rec("c2", seed=1)]})
    with pytest.raises(RuntimeError):
        rn.run_shard("ncmd_bdsp", "columbia", "analysis", spec_path=_spec_path(tmp_path, True))
    monkeypatch.setattr(rn, "COLUMBIA_ROWS", str(pq))
    c = rn.columbia113()
    assert list(c["uid"]) == ["c1"] and list(c["patient"]) == ["pt_c1"]
    rn.run_shard("ncmd_bdsp", "columbia", "analysis", spec_path=_spec_path(tmp_path, True))
    rows = _rows(rn.shard_file("ncmd_bdsp", "columbia", "native", "none", "analysis", 0))
    assert {r["uid"]: r["status"] for r in rows if r["uid"] == "c2"} == {"c2": "not_in_113"}
    assert {r["j"] for r in rows if r["uid"] == "c1"} == {1}



def test_kit8_montage_restricts_channels_and_restores_the_module():
    mod = _stub()
    _collect(_rec(dur=2000), mod, n_recs=480, montage="kit8")
    run = [e for e in mod.LOG if e[0] == "run"][0]
    montage = importlib.import_module("montage")
    want = [NAMES19[i] for i in montage.kit8_indices(NAMES19)]
    assert run[3] == want and run[4] == want and mod.NATIVE_CH == NAMES19


def test_chancount_order_is_nested_and_starts_with_10_20():
    import mne
    pos = mne.channels.make_standard_montage("standard_1005").ch_names
    names = [n for n in pos if n in ("Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2 AF3 AF4 F5 F1 F2 F6 "
                                     "FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 C5 C1 C2 C6 TP7 CP5 CP3 CP1 CPz CP2 CP4 "
                                     "CP6 TP8 P5 P1 P2 P6 PO7 PO3 POz PO4 PO8 Oz AF7 AF8 F9 F10 TP9 TP10 Iz").split()]
    order = rn.nested_channel_order(names[::-1])
    assert order == rn.nested_channel_order(names)
    ten20 = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
             "O1", "O2"}
    assert set(order[:19]) == ten20 and sorted(order) == sorted(names)
    n = len(names)
    assert rn.chancount_sizes(n) == [k for k in (19, 32, 64) if k < n] + [n]
    assert rn.chancount_sizes(30) == [19, 30] and rn.chancount_sizes(64) == [19, 32, 64]



def _passive_stubs(monkeypatch, recs, na=None, contrasts=("oddball",)):
    """passive.load_passive / passive_overlay and adapters.{contrasts_for, adapt, usable_variants, run,
    run_randomized, recording_for, NotApplicable, DESIGNS} as bench-passive specified them. Variant c is skipped."""
    ps = types.ModuleType("bench.passive")
    ps.load_passive = lambda dataset, split: iter(recs)
    ps.passive_overlay = lambda rec, contrast: en.Overlay(
        onsets_s=10.0 + np.arange(40) * 1.0, labels=np.tile([1, 0, 0, 0], 10), kind="stim",
        meta={"contrast": contrast, "start_s": 10.0, "randomized": False})
    ad = types.ModuleType("bench.adapters")

    class NotApplicable(ValueError):
        pass

    def adapt(pipe, rec, ov, rng=None, design="published"):
        if na:
            raise NotApplicable(na)
        info = {"pipe": pipe.NAME, "contrast": ov.meta["contrast"], "design": design, "changed": ["grouped", "x"],
                "usable": {"n_groups": 7}, "dropped": {"edge": 1}, "skip_variants": ["c"], "periodic": None,
                "run": design == "published", "randomized": design == "randomized", "randomized_reason": None,
                "rand_variant": "a" if design == "randomized" else None}
        return en.Overlay(onsets_s=ov.onsets_s, labels=ov.labels, kind="stim",
                          meta=dict(ov.meta, adapter=info, randomized=design == "randomized"))

    def run(pipe, rec, a, v, rng=None):
        assert a.meta["adapter"]["run"] and v not in a.meta["adapter"]["skip_variants"]
        return pipe.run(rec, a, v, rng)

    def run_randomized(pipe, rec, a, rng=None, variant=None):
        assert a.meta["adapter"]["design"] == "randomized"
        return pipe.run_randomized(rec, a, variant or a.meta["adapter"]["rand_variant"], rng)
    ad.NotApplicable, ad.DESIGNS, ad.adapt, ad.run, ad.run_randomized = NotApplicable, ("published", "randomized"), \
        adapt, run, run_randomized
    ad.contrasts_for = lambda pipe, dataset: contrasts
    ad.usable_variants = lambda pipe, a: [v for v in pipe.VARIANTS if v not in a.meta["adapter"]["skip_variants"]]
    ad.recording_for = lambda pipe, rec: rec
    ad.LOG = []

    def instruction_overlay(pipe, rec, ov, rng=None):
        assert ov.meta["contrast"] == "oddball"
        if rec.meta.get("too_short"):
            raise NotApplicable("too few blocks")
        return en.Overlay(onsets_s=ov.onsets_s, labels=ov.labels, kind="stim",
                          meta=dict(ov.meta, adapter={"pipe": pipe.NAME, "design": "instruction",
                                                      "block_rule": "module", "changed": ["blocks"],
                                                      "usable": {"n_blocks": 8, "n_count": 4, "n_ignore": 4}}))

    def run_instruction(pipe, rec, a, rng=None, n_perm=500):
        assert isinstance(rng, np.random.Generator) and n_perm == 200
        ad.LOG.append(("run", rec.uid))
        return mc.result(float(rng.normal()), 0.3, False, n_blocks=8, n_perm_run=30, stopped_early=True)

    def pool_instruction_features(pipe, runs):
        uids = [rec.uid for rec, a in runs]
        assert uids
        return {"kind": pipe.NAME, "uids": uids, "nb": 8 * len(uids)}

    def run_instruction_pooled(pipe, F, rng=None, n_perm=500):
        assert n_perm == 200
        ad.LOG.append(("pooled", tuple(F["uids"])))
        return mc.result(0.1, 0.02, True, n_blocks=F["nb"], n_runs=len(F["uids"]), instruction=np.arange(3),
                         n_perm_run=500, stopped_early=False)
    ad.instruction_overlay, ad.run_instruction = instruction_overlay, run_instruction
    ad.pool_instruction_features, ad.run_instruction_pooled = pool_instruction_features, run_instruction_pooled
    monkeypatch.setitem(sys.modules, "bench.passive", ps)
    monkeypatch.setitem(sys.modules, "bench.adapters", ad)


def test_npassive_published_calls_are_the_sensory_family_without_a_randomized_arm(monkeypatch, roots, tmp_path):
    mod = _stub(name="stuberp", family="erp")
    monkeypatch.setitem(sys.modules, "bench.pipe_stuberp", mod)
    _passive_stubs(monkeypatch, [_rec(uid="s1", patient="s1"), _rec(uid="s2", patient="s2", seed=1)])
    rn.run_shard("npassive_erpcore_mmn", "stuberp", "dev", spec_path=_spec_path(tmp_path, False))
    rows = _rows(rn.shard_file("npassive_erpcore_mmn", "stuberp", "native", "none", "dev", 0))
    pub = [r for r in rows if r["design"] == "published"]
    assert len(pub) == 6 and sorted(r["status"] for r in pub) == ["ok"] * 4 + ["skipped_by_adapter"] * 2
    ok = [r for r in pub if r["status"] == "ok"]
    assert {r["variant"] for r in ok} == {"a", "b"} and {r["contrast"] for r in pub} == {"oddball"}
    assert {r["adapt"] for r in ok} == {"grouped; x"} and {r["adapt_usable_n_groups"] for r in ok} == {7}
    assert {r["family_tag"] for r in pub} == {"sensory_read_as_volition"}
    assert all("rand_stat" not in r for r in pub) and not [e for e in mod.LOG if e[0] == "rand"]


def test_npassive_instruction_test_per_subject_is_primary_and_per_run_secondary(monkeypatch, roots, tmp_path):
    mod = _stub(name="stuberp", family="erp")
    monkeypatch.setitem(sys.modules, "bench.pipe_stuberp", mod)
    recs = [_rec(uid="s1_r1", patient="s1"), _rec(uid="s1_r2", patient="s1", seed=1),
            _rec(uid="s1_r3", patient="s1", seed=2, meta={"too_short": True}), _rec(uid="s2_r1", patient="s2", seed=3)]
    _passive_stubs(monkeypatch, recs)
    pre = _spec_path(tmp_path, False)
    rn.run_shard("npassive_kojimaB", "stuberp", "dev", spec_path=pre)
    fn = rn.shard_file("npassive_kojimaB", "stuberp", "native", "none", "dev", 0)
    rows = _rows(fn)
    ins = [r for r in rows if r["design"] == "instruction"]
    subj = {r["patient"]: r for r in ins if r["unit"] == "subject"}
    run = [r for r in ins if r["unit"] == "run"]
    assert subj["s1"]["x_n_runs"] == 2 and subj["s1"]["n_skipped_runs"] == 1 and subj["s1"]["uid"] == "s1"
    assert subj["s1"]["rand_call"] is True and subj["s1"]["rand_p"] == 0.02 and subj["s2"]["x_n_runs"] == 1
    assert sorted(r["uid"] for r in run) == ["s1_r1", "s1_r2", "s1_r3", "s2_r1"]
    assert {r["uid"]: r["status"] for r in run} == {"s1_r1": "ok", "s1_r2": "ok", "s1_r3": "na: too few blocks",
                                                    "s2_r1": "ok"}
    assert all(r["rand_stat"] is not None for r in run if r["status"] == "ok")
    ad = sys.modules["bench.adapters"]
    assert [e for e in ad.LOG if e[0] == "pooled"] == [("pooled", ("s1_r1", "s1_r2")), ("pooled", ("s2_r1",))]
    first = open(fn).read()
    rn.run_shard("npassive_kojimaB", "stuberp", "dev", spec_path=pre)
    assert open(fn).read() == first and len([e for e in ad.LOG if e[0] == "pooled"]) == 2


def test_npassive_shards_by_subject(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stuberp", _stub(name="stuberp", family="erp"))
    recs = [_rec(uid=f"s{k}_r{r}", patient=f"s{k}", seed=10 * k + r) for k in range(5) for r in range(2)]
    pre = _spec_path(tmp_path, False)
    for sh in range(2):
        _passive_stubs(monkeypatch, recs)
        rn.run_shard("npassive_erpcore_mmn", "stuberp", "dev", shard=sh, nshards=2, spec_path=pre)
    got = [{r["patient"] for r in _rows(rn.shard_file("npassive_erpcore_mmn", "stuberp", "native", "none", "dev",
                                                       sh))} for sh in (0, 1)]
    assert got[0] | got[1] == {f"s{k}" for k in range(5)} and not got[0] & got[1]


def test_npassive_not_applicable_is_a_row(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stuberp", _stub(name="stuberp", family="erp"))
    _passive_stubs(monkeypatch, [_rec(uid="s1", patient="s1")], na="not applicable here")
    rn.run_shard("npassive_kojimaB", "stuberp", "dev", spec_path=_spec_path(tmp_path, False))
    rows = [r for r in _rows(rn.shard_file("npassive_kojimaB", "stuberp", "native", "none", "dev", 0))
            if r["design"] == "published"]
    assert len(rows) == 1 and rows[0]["status"] == "na: not applicable here"
    _passive_stubs(monkeypatch, [_rec(uid="s2", patient="s2")], contrasts=())
    rn.run_shard("npassive_kojimaB", "stuberp", "dev", spec_path=_spec_path(tmp_path, False))
    rows = [r for r in _rows(rn.shard_file("npassive_kojimaB", "stuberp", "native", "none", "dev", 0))
            if r["design"] == "published"]
    assert rows[-1]["uid"] == "s2" and rows[-1]["status"].startswith("na: no matching contrast")


def test_chancount_runs_chennu_only_on_nested_subsets(monkeypatch, roots, tmp_path):
    import mne
    names = mne.channels.make_standard_montage("standard_1020").ch_names[:40]
    mod = _stub(name="chennu13", family="erp")
    monkeypatch.setitem(sys.modules, "bench.pipe_chennu13", mod)
    _passive_stubs(monkeypatch, [_rec(uid="s1", patient="s1", names=names)])
    rn.run_shard("chancount_ds003061", "chennu13", "dev", spec_path=_spec_path(tmp_path, False))
    rows = _rows(rn.shard_file("chancount_ds003061", "chennu13", "native", "none", "dev", 0))
    ok = [r for r in rows if r["status"] == "ok"]
    assert sorted({r["n_channels"] for r in rows}) == [19, 32, 40] and len(ok) == 3 * 2
    seen = sorted({len(e[3]) for e in mod.LOG if e[0] == "run"})
    assert seen == [19, 32, 40] and mod.NATIVE_CH == NAMES19
    with pytest.raises(ValueError):
        rn.run_shard("chancount_ds003061", "stuberp", "dev", spec_path=_spec_path(tmp_path, False))


def test_npassive_with_the_real_adapters_and_mbvt_on_a_synthetic_two_stream_recording(monkeypatch, roots, tmp_path):
    """Integration (synthetic noise; no statistic or call is checked): kojimaB gives mbvt two contrasts, each variant
    runs under exactly one (VT2 on oddball, VT3 on equiprobable), skipped variants are rows, events never reach rows."""
    ta = importlib.import_module("bench.tests.test_adapters")
    kinds, on, seq, train = ta.stream(3000, np.random.default_rng(4), rare_p=0.2, two=True, soa=0.3)
    src = {c: ta.overlay(kinds, on, seq, train, contrast=c, dataset="kojimaB") for c in ("oddball", "equiprobable")}
    r0 = ta.rec_for(src["oddball"], seed=4)
    rec = en.Recording(source="kojimaB", uid="kojimaB_sub-01_run-1", patient="kojimaB_sub-01", x=r0.x, fs=r0.fs,
                       names=r0.names, meta={"dataset": "kojimaB", "dev": True, "events": pd_marker(),
                                             "subject": "SECRET-HIS-77"})
    ps = types.ModuleType("bench.passive")
    ps.load_passive = lambda dataset, split: iter([rec])
    ps.passive_overlay = lambda r, contrast: src[contrast]
    monkeypatch.setitem(sys.modules, "bench.passive", ps)
    rn.run_shard("npassive_kojimaB", "mbvt", "dev", spec_path=_spec_path(tmp_path, False))
    rows = _rows(rn.shard_file("npassive_kojimaB", "mbvt", "native", "none", "dev", 0))
    pm = rn.load_pipe("mbvt")
    ok = [r for r in rows if r["status"] == "ok" and r["design"] == "published"]
    assert ok and {r["contrast"] for r in rows if r["design"] == "published"} == {"oddball", "equiprobable"}
    for r in ok:
        assert r["contrast"] == ("oddball" if pm.VARIANTS[r["variant"]]["paradigm"] == "vt2" else "equiprobable")
    assert len({r["variant"] for r in ok}) == len(ok)
    pub = [r for r in rows if r["design"] == "published"]
    assert len(pub) == 2 * len(pm.VARIANTS) and {r["uid"] for r in pub} == {rec.uid}
    text = json.dumps(rows)
    assert "SECRET-HIS-77" not in text and "EVENTS-TABLE" not in text and "#mbvt" not in text


def pd_marker():
    import pandas as pd
    return pd.DataFrame({"code": ["EVENTS-TABLE"]})



def test_resumable_and_never_overwrites(monkeypatch, roots, tmp_path):
    mod = _stub()
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", mod)
    _fake_engine(monkeypatch, {"dev": [_rec(uid="r1", patient="p1", dur=2000), _rec(uid="r2", patient="p2",
                                                                                     dur=2000, seed=1)]})
    pre = _spec_path(tmp_path, False)
    rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=pre)
    fn = rn.shard_file("ncmd_pdoc", "stub", "native", "none", "dev", 0)
    first = open(fn).read()
    n_calls = len(mod.LOG)
    rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=pre)
    assert open(fn).read() == first and len(mod.LOG) == n_calls
    lines = first.splitlines(keepends=True)
    with open(fn, "w") as fh:
        fh.writelines(lines[:-2])
    rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=pre)
    after = open(fn).read()
    assert after.startswith("".join(lines[:-2])) and len(after.splitlines()) == len(lines)
    strip = lambda t: sorted(json.dumps({k: v for k, v in json.loads(l).items() if "seconds" not in k})
                             for l in t.splitlines())
    assert strip(after) == strip(first)


def test_shards_partition_the_recordings(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    recs = [_rec(uid=f"r{k}", patient=f"p{k}", dur=500, seed=k) for k in range(5)]
    _fake_engine(monkeypatch, {"dev": recs})
    pre = _spec_path(tmp_path, False)
    for s in range(2):
        rn.run_shard("ncmd_pdoc", "stub", "dev", shard=s, nshards=2, spec_path=pre)
    u = [{r["uid"] for r in _rows(rn.shard_file("ncmd_pdoc", "stub", "native", "none", "dev", s))} for s in (0, 1)]
    assert u[0] | u[1] == {f"r{k}" for k in range(5)} and not u[0] & u[1]













def _leak_check(roots):
    """No dev recording in analysis outputs, no analysis recording in dev outputs, split stamped on every row."""
    for split, root in roots.items():
        for dp, _, fs in os.walk(root):
            for f in fs:
                for r in _rows(os.path.join(dp, f)):
                    assert r["split"] == split, "row written under the wrong split"
                    assert r["dev"] is (split == "dev"), f"dev={r['dev']} recording in the {split} outputs"


def test_dev_and_analysis_outputs_are_separate(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    _fake_engine(monkeypatch, {"dev": [_rec(uid="d1", dur=500)], "analysis": [_rec(uid="a1", dur=500, dev=False)]})
    rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=_spec_path(tmp_path, False))
    rn.run_shard("ncmd_pdoc", "stub", "analysis", spec_path=_spec_path(tmp_path, True))
    _leak_check(roots)
    assert os.path.exists(rn.shard_file("ncmd_pdoc", "stub", "native", "none", "analysis", 0))


def test_a_dev_recording_in_analysis_is_refused(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    _fake_engine(monkeypatch, {"analysis": [_rec(uid="leak", dur=500, dev=True)], "dev": [_rec(uid="x", dev=False)]})
    with pytest.raises(RuntimeError):
        rn.run_shard("ncmd_pdoc", "stub", "analysis", spec_path=_spec_path(tmp_path, True))
    with pytest.raises(RuntimeError):
        rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=_spec_path(tmp_path, False))


def test_a_runner_without_the_membership_check_leaks_and_is_caught(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    _fake_engine(monkeypatch, {"analysis": [_rec(uid="leak", dur=500, dev=True)]})
    monkeypatch.setattr(rn, "check_member", lambda rec, split: None)
    rn.run_shard("ncmd_pdoc", "stub", "analysis", spec_path=_spec_path(tmp_path, True))
    with pytest.raises(AssertionError):
        _leak_check(roots)


def _provisional_bdsp(monkeypatch, pending):
    recs = [_rec(uid=f"b{r}", patient=f"pb{r}", dur=500, seed=r, source="bdsp",
                 meta={"rank": r, "split_final": False}) for r in (480, 485, 490, 495, 499)]
    items = [{"source": "bdsp", "uid": r.uid, "patient": r.patient, "meta": dict(r.meta), "rec": r} for r in recs]
    monkeypatch.setattr(en, "_items", lambda source, split: list(items) if split == "dev" else [])
    monkeypatch.setattr(en, "_load", lambda it: it["rec"])
    monkeypatch.setattr(en, "bdsp_split", lambda *a, **k: {"pending": pending, "final": False})


def _assert_provisional_safe(iterate):
    """While the BDSP dev split is provisional, the `pending` lowest-ranked dev recordings may still move to analysis
    (each newly usable recording ranked after the first dev one displaces the lowest-ranked one): never run them."""
    got = [rec.uid for rec, _ in iterate()]
    if {"b480", "b485"} & set(got):
        raise AssertionError(f"ran provisional dev recordings that can still move to analysis: {got}")
    assert got == ["b490", "b495", "b499"]


def test_provisional_dev_recordings_that_can_still_move_are_skipped(monkeypatch):
    _provisional_bdsp(monkeypatch, pending=2)
    _assert_provisional_safe(lambda: rn.iter_recordings(rn.parse_null("ncmd_bdsp"), "dev", None, 0, 1))
    _assert_provisional_safe(lambda: rn.iter_recordings(rn.parse_null("ncmd_bdsp"), "dev", 3, 0, 1))


def test_a_runner_that_takes_the_first_provisional_dev_recordings_is_caught(monkeypatch):
    _provisional_bdsp(monkeypatch, pending=2)
    monkeypatch.setattr(rn, "safe_dev", lambda items, pending: items)
    with pytest.raises(AssertionError):
        _assert_provisional_safe(lambda: rn.iter_recordings(rn.parse_null("ncmd_bdsp"), "dev", None, 0, 1))


def test_rows_of_another_split_in_a_shard_file_are_refused(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    _fake_engine(monkeypatch, {"dev": [_rec(uid="d1", dur=500)]})
    fn = rn.shard_file("ncmd_pdoc", "stub", "native", "none", "dev", 0)
    os.makedirs(os.path.dirname(fn))
    with open(fn, "w") as fh:
        fh.write(json.dumps({"key": "zz", "split": "analysis"}) + "\n")
    with pytest.raises(RuntimeError):
        rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=_spec_path(tmp_path, False))



@pytest.fixture
def touched(tmp_path, monkeypatch):
    fn = tmp_path / "touched.json"
    fn.write_text(json.dumps({"excluded": ["a2"], "loaded_only_not_excluded": ["a3"]}))
    monkeypatch.setattr(rn, "TOUCHED_FILE", str(fn))
    return fn


def _assert_no_touched(roots, excluded):
    for dp, _, fs in os.walk(roots["analysis"]):
        for f in fs:
            hit = {r["uid"] for r in _rows(os.path.join(dp, f))} & set(excluded)
            if hit:
                raise AssertionError(f"recordings touched before the freeze in analysis outputs: {sorted(hit)}")


def _analysis_engine(monkeypatch):
    loaded = []
    recs = [_rec(uid=u, patient=f"p{u}", dur=500, dev=False, seed=k) for k, u in enumerate(("a1", "a2", "a3"))]
    _fake_engine(monkeypatch, {"analysis": recs})
    real = en._load
    monkeypatch.setattr(en, "_load", lambda it: (loaded.append(it["uid"]), real(it))[1])
    return loaded


def test_touched_recordings_are_skipped_in_analysis_before_loading(monkeypatch, roots, tmp_path, touched):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    loaded = _analysis_engine(monkeypatch)
    rn.run_shard("ncmd_pdoc", "stub", "analysis", spec_path=_spec_path(tmp_path, True))
    rows = _rows(rn.shard_file("ncmd_pdoc", "stub", "native", "none", "analysis", 0))
    assert {r["uid"] for r in rows} == {"a1", "a3"} and loaded == ["a1", "a3"]
    _assert_no_touched(roots, ["a2"])


def test_a_runner_that_ignores_the_touched_list_is_caught(monkeypatch, roots, tmp_path, touched):
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    _analysis_engine(monkeypatch)
    monkeypatch.setattr(rn, "touched_uids", lambda: set())
    rn.run_shard("ncmd_pdoc", "stub", "analysis", spec_path=_spec_path(tmp_path, True))
    with pytest.raises(AssertionError):
        _assert_no_touched(roots, ["a2"])


def test_analysis_refuses_without_the_touched_list(monkeypatch, roots, tmp_path):
    monkeypatch.setattr(rn, "TOUCHED_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    calls = []
    _fake_engine(monkeypatch, {"analysis": [_rec(uid="a1", dev=False)]}, calls)
    with pytest.raises(RuntimeError):
        rn.run_shard("ncmd_pdoc", "stub", "analysis", spec_path=_spec_path(tmp_path, True))
    assert calls == []


def test_dev_runs_do_not_need_the_touched_list(monkeypatch, roots, tmp_path):
    monkeypatch.setattr(rn, "TOUCHED_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setitem(sys.modules, "bench.pipe_stub", _stub())
    _fake_engine(monkeypatch, {"dev": [_rec(uid="d1", dur=500)]})
    rn.run_shard("ncmd_pdoc", "stub", "dev", spec_path=_spec_path(tmp_path, False))



def test_chennu_block_count_follows_the_source_decision():
    ch = rn.load_pipe("chennu13")
    assert {ch.VARIANTS[v]["n_blocks"] for v in rn.variants_for(ch, "bdsp")} == {12}
    assert {ch.VARIANTS[v]["n_blocks"] for v in rn.variants_for(ch, "pdoc")} == {20}
    assert rn.primary_variant(ch, "bdsp") == "p1_samples10_none_b12"
    assert rn.primary_variant(ch, "pdoc") == ch.DEFAULT
    assert rn.primary_variant(rn.load_pipe("cruse12p"), "bdsp") == "random_perclass"
    assert rn.need_seconds(ch, rn.variants_for(ch, "bdsp")) == rn.LEAD_S + 1505 + rn.TAIL_S


def test_instruction_test_with_the_real_adapters_and_mbvt_pools_a_subjects_runs(monkeypatch, roots, tmp_path):
    """Integration (synthetic noise; no statistic or call is checked): two runs of one subject pooled by the real
    adapters.pool_instruction_features / run_instruction_pooled, one row per run and one per subject."""
    ta = importlib.import_module("bench.tests.test_adapters")
    recs = []
    for r in range(2):
        kinds, on, seq, train = ta.stream(1100, np.random.default_rng(20 + r), rare_p=0.15, soa=0.3)
        ov = ta.overlay(kinds, on, seq, train, contrast="oddball", dataset="erpcore_mmn")
        r0 = ta.rec_for(ov, seed=20 + r)
        recs.append((en.Recording(source="erpcore_mmn", uid=f"erpcore_mmn_sub-01_run-{r}", patient="erpcore_mmn_sub-01",
                                  x=r0.x, fs=r0.fs, names=r0.names, meta={"dataset": "erpcore_mmn", "dev": True}), ov))
    ps = types.ModuleType("bench.passive")
    ps.load_passive = lambda dataset, split: iter([r for r, _ in recs])
    ps.passive_overlay = lambda rec, contrast: dict((r.uid, o) for r, o in recs)[rec.uid]
    monkeypatch.setitem(sys.modules, "bench.passive", ps)
    rn.run_shard("npassive_erpcore_mmn", "mbvt", "dev", spec_path=_spec_path(tmp_path, False))
    rows = _rows(rn.shard_file("npassive_erpcore_mmn", "mbvt", "native", "none", "dev", 0))
    ins = [r for r in rows if r["design"] == "instruction"]
    assert sorted(r["unit"] for r in ins) == ["run", "run", "subject"], [r["status"] for r in ins]
    subj = [r for r in ins if r["unit"] == "subject"][0]
    assert subj["status"] == "ok" and subj["x_n_runs"] == 2 and subj["rand_p"] is not None
    assert all(r["status"] == "ok" for r in ins)


def test_a_pool_that_stops_early_still_leaves_every_run_its_rows(monkeypatch, roots, tmp_path):
    monkeypatch.setitem(sys.modules, "bench.pipe_stuberp", _stub(name="stuberp", family="erp"))
    recs = [_rec(uid=f"s1_r{r}", patient="s1", seed=r) for r in range(3)]
    _passive_stubs(monkeypatch, recs)
    ad = sys.modules["bench.adapters"]

    def refuse(pipe, runs):
        next(iter(runs))
        raise ValueError("mismatched runs")
    monkeypatch.setattr(ad, "pool_instruction_features", refuse)
    rn.run_shard("npassive_erpcore_mmn", "stuberp", "dev", spec_path=_spec_path(tmp_path, False))
    rows = _rows(rn.shard_file("npassive_erpcore_mmn", "stuberp", "native", "none", "dev", 0))
    assert sorted(r["uid"] for r in rows if r.get("unit") == "run") == ["s1_r0", "s1_r1", "s1_r2"]
    assert len([r for r in rows if r["design"] == "published"]) == 3 * 3
    assert [r["status"] for r in rows if r.get("unit") == "subject"] == ["na: mismatched runs"]




def test_the_randomized_method_goes_through_the_b_wrapper(monkeypatch):
    seen = []
    real = rn.wr.run_randomized
    monkeypatch.setattr(rn.wr, "run_randomized", lambda mod, rec, ov, v, rng: (seen.append((mod.NAME, v)),
                                                                              real(mod, rec, ov, v, rng))[1])
    rows = _collect(_rec(dur=2000), _stub(), n_recs=3)
    assert set(seen) == {("stub", "a"), ("stub", "c")} and len(seen) == 2 * len({r["j"] for r in rows})


def test_columbia_eval_runs_ers_on_the_uncorrected_subsample_only(monkeypatch, tmp_path):
    monkeypatch.setattr(rn.wr, "wrapped", lambda name, cfg: False)
    _eval_file(tmp_path, monkeypatch, {"ncmd_bdsp": {"columbia": {"erd|1": {"effect": 0.5, "comparator": "a"},
                                                                  "erd|0.5": {"effect": 0.5, "comparator": "a"},
                                                                  "ers|1": {"effect": 0.5, "comparator": "a"}}}})
    mod = _stub(name="columbia", run_s=1505.0)
    out = _collect(_bdsp_rec("c1"), mod, null="ncmd_bdsp", n_recs=480, col113={"c1": 500.0}, resp="eval",
                   locator=set(), uncorrected=set())
    assert sorted((r["model"], r["prob"]) for r in out) == [("erd", 0.5), ("erd", 1.0)]
    assert {r["eval_set"] for r in out} == {"all"}
    sub = _collect(_bdsp_rec("c1"), _stub(name="columbia", run_s=1505.0), null="ncmd_bdsp", n_recs=480,
                   col113={"c1": 500.0}, resp="eval", locator=set(), uncorrected={"pt_c1"})
    assert {(r["model"], r["eval_set"]) for r in sub} == {("erd", "all"), ("ers", "uncorrected_subsample")}


def test_other_pipelines_run_every_point_on_all_patients(monkeypatch, tmp_path):
    _eval_file(tmp_path, monkeypatch, {"ncmd_pdoc": {"stub": {"erd|1": {"effect": 0.5, "comparator": "a"},
                                                               "ers|1": {"effect": 0.5, "comparator": "a"}}}})
    rows = _collect(_rec(dur=2000), _stub(), n_recs=480, resp="eval", locator=set(), uncorrected=set())
    assert sorted(r["model"] for r in rows) == ["erd", "ers"] and {r["eval_set"] for r in rows} == {"all"}


def test_rows_flag_variants_that_cannot_call(monkeypatch):
    monkeypatch.setattr(rn.wr, "min_attainable_p", lambda mod, v: 0.1 if v == "c" else 1 / 201)
    rows = {r["variant"]: r for r in _collect(_rec(dur=2000), _stub(), n_recs=480)}
    assert rows["c"]["cannot_call"] is True and rows["c"]["min_attainable_p"] == 0.1
    assert rows["a"]["cannot_call"] is False and np.isclose(rows["a"]["min_attainable_p"], 1 / 201)



def test_columbia_uncorrected_arm_runs_at_erd_probability_1_only(monkeypatch, tmp_path):
    monkeypatch.setattr(rn.wr, "wrapped", lambda name, cfg: name == "columbia")
    monkeypatch.setattr(rn.wr, "stat_only", lambda mod, rec, ov, v, rng: dict(
        mod.run(rec, ov, v, rng), p=None, call=False, extra={"inference": False}))
    pts = {"erd|1": {"effect": 0.5, "comparator": "a"}, "erd|0.5": {"effect": 0.5, "comparator": "a"},
           "ers|1": {"effect": 0.5, "comparator": "a"}, "ers|0.5": {"effect": 0.5, "comparator": "a"}}
    _eval_file(tmp_path, monkeypatch, {"ncmd_bdsp": {"columbia": pts}, "ncmd_pdoc": {"stub": pts}})
    mod = _stub(name="columbia", run_s=1505.0)
    rows = _collect(_bdsp_rec("c1"), mod, null="ncmd_bdsp", n_recs=480, col113={"c1": 500.0}, resp="eval",
                    locator=set(), uncorrected={"pt_c1"})
    assert len(rows) == 4
    perm = {(r["model"], r["prob"]): r["perm_stat"] is not None for r in rows}
    assert perm == {("erd", 1.0): True, ("erd", 0.5): False, ("ers", 1.0): False, ("ers", 0.5): False}
    assert len([e for e in mod.LOG if e[0] == "run"]) == 4 + 1
    other = _collect(_rec(dur=2000), _stub(), n_recs=480, resp="eval", locator=set(), uncorrected={"p1"})
    assert len(other) == 4 and all(r["perm_stat"] is not None for r in other)


def _pdoc_stub(monkeypatch, chans=("F3", "C3", "F4", "C4")):
    mod = _stub()
    monkeypatch.setitem(rn.wr.PDOC_CH, "stub", {"NATIVE_CH": list(chans)})
    return mod


def test_pdoc_rows_use_the_documented_channel_set_and_restore_the_module(monkeypatch):
    mod = _pdoc_stub(monkeypatch)
    rows = _collect(_rec(dur=2000), mod, n_recs=3)
    used = {tuple(e[4]) for e in mod.LOG if e[0] in ("run", "rand")}
    assert used == {("F3", "C3", "F4", "C4")} and mod.NATIVE_CH == NAMES19
    assert {r["channel_set"] for r in rows} == {"F3,C3,F4,C4"}
    mod2 = _pdoc_stub(monkeypatch)
    _collect(_bdsp_rec("b1", dur=2000), mod2, null="ncmd_bdsp", n_recs=3)
    assert {tuple(e[4]) for e in mod2.LOG if e[0] == "run"} == {tuple(NAMES19)}


def test_pdoc_drops_variants_that_read_an_absent_site(monkeypatch):
    monkeypatch.setattr(rn.wr, "pdoc_variant_na", lambda mod, v: v == "c")
    assert set(rn.variant_specs(_stub(), "pdoc")) == {"a", "b"}
    assert set(rn.variant_specs(_stub(), "bdsp")) == {"a", "b", "c"}


def test_pdoc_arm_is_nulls_only_and_columbia_is_not_applicable(roots, tmp_path, monkeypatch):
    pre = _spec_path(tmp_path, False)
    for resp in ("grid", "eval"):
        with pytest.raises(ValueError, match="nulls only"):
            rn.run_shard("ncmd_pdoc", "cornell", "dev", resp=resp, spec_path=pre)
    with pytest.raises(ValueError, match="kit8"):
        rn.run_shard("nstim0_pdoc", "chennu13", "dev", montage="kit8", spec_path=pre)
    with pytest.raises(ValueError, match="N/A on pDoC"):
        rn.run_shard("ncmd_pdoc", "columbia", "dev", spec_path=pre)
    with pytest.raises(ValueError, match="N/A on pDoC"):
        _collect(_rec(dur=2000), _stub(name="columbia", run_s=60.0), n_recs=3)



def _erp_ov(labels, start=20.0):
    on = start + 5.0 * np.arange(len(labels))
    return en.Overlay(onsets_s=on, labels=np.asarray(labels), kind="stim", meta={"start_s": start})


def _assert_injected_copies_never_alias(keyfn, n=40):
    """Copies injected on the same onsets with different labels (made one after the other, the first freed, as the
    runner does) must never share a cache key."""
    rec = _rec(dur=120)
    mod = _stub(family="erp")
    lab1 = np.array([1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2])
    for i in range(n):
        lab2 = np.random.default_rng(i).permutation(lab1)
        if (lab2 == lab1).all():
            continue
        a = rn._injected(rec, mod, _erp_ov(lab1), ("erp", 5.0, 1.0), rec.uid, i, 0)
        on = np.asarray(_erp_ov(lab1).onsets_s, float)
        k1 = keyfn(a, on)
        del a
        b = rn._injected(rec, mod, _erp_ov(lab2), ("erp", 5.0, 1.0), rec.uid, i, 1)
        if keyfn(b, on) == k1:
            raise AssertionError(f"draw {i}: two different injected signals share a cache key")
        del b


def test_injected_copies_never_share_a_cache_key(monkeypatch):
    monkeypatch.setattr(mc, "id", lambda o: 0, raising=False)
    _assert_injected_copies_never_alias(mc.rec_key)


def test_a_key_without_the_recording_token_is_caught():
    broken = lambda r, on: (r.uid, np.asarray(r.x).shape, float(r.fs), float(np.asarray(r.x)[:, ::4093].sum()),
                            on.tobytes())
    with pytest.raises(AssertionError, match="share a cache key"):
        _assert_injected_copies_never_alias(broken)


def test_every_recording_gets_its_own_token_and_injection_copies_meta():
    import dataclasses
    rec = _rec(dur=120)
    assert rec.token != _rec(dur=120).token and dataclasses.replace(rec, uid="v").token != rec.token
    cp = rn._injected(rec, _stub(family="erp"), _erp_ov([1, 2, 2, 2]), ("erp", 5.0, 1.0), rec.uid, 0, 0)
    assert cp.token != rec.token and cp.meta is not rec.meta and cp.meta == rec.meta


def _caching_stub(family="cmd"):
    """A stub whose features are memoised on (recording, onsets) and checked against the signal it is given."""
    m = _stub(family=family)
    m.VARIANTS["d"] = {"gap": 5.0, "flip": True}
    m.CHECK = []
    real_sched, real_rand = m.schedule, m.schedule_randomized

    def flip(ov, variant):
        if mc.cfg(m.VARIANTS, variant).get("flip"):
            ov.labels = 3 - np.asarray(ov.labels)
        return ov

    m.schedule = lambda s, rng, variant="a": flip(real_sched(s, rng, variant), variant)
    m.schedule_randomized = lambda s, rng, variant="a": flip(real_rand(s, rng, variant), variant)
    digest = lambda rec: float(np.asarray(rec.x, float).sum())

    def check(rec, ov):
        got = mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: digest(rec))
        m.CHECK.append(got == digest(rec))

    real_run, real_rr = m.run, m.run_randomized
    m.run = lambda rec, ov, variant="a", rng=None: (check(rec, ov), real_run(rec, ov, variant, rng))[1]
    m.run_randomized = lambda rec, ov, variant="a", rng=None: (check(rec, ov), real_rr(rec, ov, variant, rng))[1]
    return m


def test_runner_features_always_come_from_the_signal_scored(monkeypatch, tmp_path):
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erd": [0.4, 0.6], "ers": [0.4]}, "test"))
    m = _caching_stub()
    _collect(_rec(dur=2000), m, n_recs=480, resp="grid")
    pts = {"erd|1": {"groups": {"a": {"effect": 0.4, "comparator": "d", "reached": True}}, "pipeline_group": "a"}}
    _eval_file(tmp_path, monkeypatch, {"ncmd_pdoc": {"stub": pts}})
    _collect(_rec(dur=2000), m, n_recs=480, resp="eval", locator=set(), uncorrected={"p1"})
    assert len(m.CHECK) > 20 and all(m.CHECK)



def test_infinite_size_statistic_is_capped_not_dropped():
    m = _stub()
    vals = iter([np.inf, -np.inf, np.nan] * 100)
    m.stat_for_size_correction = lambda r: next(vals)
    rows = _collect(_rec(dur=2000), m, n_recs=3)
    got = [r[k] for r in rows for k in ("size_stat", "rand_size_stat") if k in r]
    cap = rn.wr.SIZE_CAP
    assert cap >= 1e9 and set(got) == {cap, -cap, None} and got.count(cap) >= 2



def test_analysis_refuses_without_either_frozen_dev_file(roots, tmp_path, monkeypatch):
    pre = _spec_path(tmp_path, True)
    rn.guard_split("analysis", pre)
    for missing in rn.FROZEN_FILES:
        os.rename(missing, missing + ".gone")
        with pytest.raises(RuntimeError, match="frozen"):
            rn.guard_split("analysis", pre)
        with pytest.raises(RuntimeError, match="frozen"):
            rn.run_shard("ncmd_pdoc", "cornell", "analysis", spec_path=pre)
        os.rename(missing + ".gone", missing)
    rn.guard_split("dev", pre)


def test_frozen_files_are_the_engine_and_passive_dev_files():
    passive = importlib.import_module("bench.passive")
    assert set(rn.FROZEN_FILES) == {en.DEV_FILE, passive.DEV_FILE}



def test_kit8_is_not_applicable_to_columbia_and_dubois(roots, tmp_path):
    pre = _spec_path(tmp_path, False)
    for pipe in ("columbia", "dubois"):
        with pytest.raises(ValueError, match="N/A on Kit-8"):
            rn.run_shard("ncmd_bdsp", pipe, "dev", montage="kit8", spec_path=pre)
    with pytest.raises(ValueError, match="N/A on Kit-8"):
        _collect(_bdsp_rec("k1", dur=2000), _stub(name="dubois", run_s=60.0), null="ncmd_bdsp", n_recs=3,
                 montage="kit8")
    assert set(rn.KIT8_NA) == {"columbia", "dubois"}





def test_rows_flag_censored_p_values(monkeypatch):
    m = _stub()
    m.run = lambda rec, ov, variant="a", rng=None: mc.result(0.3, 26 / 1001, False, stopped_early=variant == "a")
    rows = {r["variant"]: r for r in _collect(_rec(dur=2000), m, n_recs=3) if r.get("variant") and r["j"] == 0}
    assert rows["a"]["p_censored"] is True and rows["b"]["p_censored"] is False



def _gap(ov):
    return float(np.round(np.diff(np.asarray(ov.onsets_s, float))[0], 6))


def test_timing_groups_by_trial_count_timing_and_response_model():
    m = _stub()
    specs = rn.variant_specs(m, "pdoc")
    assert rn.timing_groups(m, specs, list(specs), "a") == {"a": ["a", "b"], "c": ["c"]}
    m.VARIANTS["e"] = {"gap": 5.0}
    real = m.schedule
    m.schedule = lambda s, rng, variant="a": (lambda ov: (ov.meta.update(resp_dur_s=4.0) if variant == "e" else None,
                                                           ov)[1])(real(s, rng, variant))
    specs = rn.variant_specs(m, "pdoc")
    assert rn.timing_groups(m, specs, list(specs), "a") == {"a": ["a", "b"], "c": ["c"], "e": ["e"]}


def test_null_rows_carry_their_own_groups_randomized_run(monkeypatch):
    m = _stub()
    rows = _collect(_rec(dur=2000), m, n_recs=3)
    seq = {r["variant"]: r for r in rows if r["j"] == 0}
    assert {v: (r["group"], r["rand_variant"]) for v, r in seq.items()} == {"a": ("a", "a"), "b": ("a", "a"),
                                                                            "c": ("c", "c")}
    extra = [r for r in rows if r["j"] > 0]
    assert extra and {(r["variant"], r["group"]) for r in extra} == {(None, "a"), (None, "c")}
    rand = [e for e in m.LOG if e[0] == "rand"]
    assert {(mc.cfg(m.VARIANTS, e[1])["gap"], _gap(e[5])) for e in rand} == {(5.0, 5.0), (6.0, 6.0)}
    assert len(rand) == 2 * len({r["j"] for r in rows})


def test_eval_pairs_each_groups_comparator_with_its_groups_randomized_run(monkeypatch, tmp_path):
    pts = {"erd|1": {"groups": {"a": {"effect": 0.4, "comparator": "b", "reached": True},
                                "c": {"effect": 0.6, "comparator": "c", "reached": False}}, "pipeline_group": "a"}}
    _eval_file(tmp_path, monkeypatch, {"ncmd_pdoc": {"stub": pts}})
    m = _stub()
    rows = _collect(_rec(dur=2000), m, n_recs=480, resp="eval", locator=set(), uncorrected=set())
    got = {(r["variant"], r["effect"], r["group"], r["rand_variant"], r["pipeline_group"], r["reached"]) for r in rows}
    assert got == {("b", 0.4, "a", "a", True, True), ("c", 0.6, "c", "c", False, False)}
    rand = {(mc.cfg(m.VARIANTS, e[1])["gap"], _gap(e[5])) for e in m.LOG if e[0] == "rand"}
    assert rand == {(5.0, 5.0), (6.0, 6.0)}



def _mb_stub():
    m = _stub(name="mbmi")
    m.VARIANTS = {"s": {"gap": 5.0, "n": 8, "as_coded": True}, "l": {"gap": 5.0, "n": 16}}
    m.DEFAULT = "s"
    m.run_s = lambda variant="s": {"s": 60.0, "l": 120.0}[variant if isinstance(variant, str) else
                                                          ("l" if variant["n"] == 16 else "s")]

    def _ov(start_s, g, variant, randomized):
        c = mc.cfg(m.VARIANTS, variant)
        on = float(start_s) + np.arange(c["n"]) * c["gap"]
        return en.Overlay(onsets_s=on, labels=g.permutation(np.repeat([1, 2], c["n"] // 2)), kind="cmd",
                          meta={"resp_hand": {1: "left", 2: "right"}, "resp_dur_s": 2.0, "randomized": randomized,
                                "start_s": float(start_s)})
    m.schedule = lambda s, rng, variant="s": _ov(s, mc.gen(rng), variant, False)
    m.schedule_randomized = lambda s, rng, variant="s": _ov(s, mc.gen(rng), variant, True)
    return m


def test_mindbeagle_tiles_each_trial_count_group_on_its_own_length():
    rows = _collect(_bdsp_rec("t1", dur=2000), _mb_stub(), null="ncmd_bdsp", n_recs=3)
    for v, tile, need in (("s", "n8", 80), ("l", "n16", 140)):
        r = sorted((x for x in rows if x["variant"] == v), key=lambda x: x["j"])
        assert len(r) == rn.MB_K_BDSP and {x["tile"] for x in r} == {tile}
        st = sorted(x["start_s"] for x in r)
        assert min(np.diff(st)) >= need
    s_starts = {x["start_s"] for x in rows if x["variant"] == "s"}
    assert len(s_starts) == rn.MB_K_BDSP



def test_resume_gives_identical_responder_rows(monkeypatch):
    monkeypatch.setattr(rn, "load_grid", lambda split, spec_path=None: ({"erd": [0.6], "ers": [0.6]}, "test"))

    def stub():
        m = _stub()
        m.run = lambda rec, ov, variant="a", rng=None: mc.result(float(np.asarray(rec.x, float)[
            :, int(ov.onsets_s[0] * rec.fs):int((ov.onsets_s[-1] + 6) * rec.fs)].sum()), 0.5, False)
        return m
    full = _collect(_rec(dur=2000), stub(), n_recs=3, resp="grid")
    done = {r["key"] for r in full if r["variant"] == "a"}
    out = []
    rn.run_recording(_rec(dur=2000), stub(), rn.parse_null("ncmd_pdoc"), _ctx(resp="grid"), done, out.append)
    want = {r["key"]: r["stat"] for r in full if r["variant"] != "a"}
    assert out and {r["key"]: r["stat"] for r in out} == want


def test_randomized_runs_never_read_another_overlays_cached_layout():
    """Two timing groups on the same onsets (different response duration) with layout-dependent features (as mbvt's
    site layout): each randomized run must see its own overlay's labels, never the other group's cached ones."""
    m = _stub()
    m.VARIANTS["e"] = {"gap": 5.0}
    real = m.schedule, m.schedule_randomized
    tag = lambda ov, v: (ov.meta.update(resp_dur_s=4.0) if v == "e" else None, ov)[1]
    m.schedule = lambda s, rng, variant="a": tag(real[0](s, rng, variant), variant)
    m.schedule_randomized = lambda s, rng, variant="a": tag(real[1](s, rng, variant), variant)
    seen = []
    rr = m.run_randomized

    def run_randomized(rec, ov, variant="a", rng=None):
        got = mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: np.asarray(ov.labels).copy())
        seen.append(np.array_equal(got, ov.labels))
        return rr(rec, ov, variant, rng)
    m.run_randomized = run_randomized
    _collect(_rec(dur=2000), m, n_recs=3)
    assert len(seen) >= 6 and all(seen)


def test_a_groups_representative_is_the_variant_closest_to_the_primary():
    m = _stub()
    m.VARIANTS = {"p": {"gap": 5.0, "x": 1, "y": 1}, "q1": {"gap": 6.0, "x": 2, "y": 2},
                  "q2": {"gap": 6.0, "x": 1, "y": 1}, "q3": {"gap": 6.0, "x": 1, "y": 2, "as_coded": True}}
    m.DEFAULT = "p"
    specs = rn.variant_specs(m, "pdoc")
    assert rn.timing_groups(m, specs, list(specs), "p") == {"p": ["p"], "q2": ["q1", "q2", "q3"]}



def _113_lists(tmp_path, monkeypatch):
    import pandas as pd
    design, extra = tmp_path / "sample.csv", tmp_path / "extra.csv"
    pd.DataFrame({"patient_id": ["pt_d0", "pt_d1"], "file": ["d0.mat", "d1.mat"]}).to_csv(design, index=False)
    pd.DataFrame({"patient_id": ["pt_x0", "pt_x1", "pt_x2"], "file": ["x0.mat", "x1.mat", "x2.mat"]}).to_csv(
        extra, index=False)
    monkeypatch.setattr(rn, "SAMPLE_CSV", str(design))
    monkeypatch.setattr(rn, "EXTRA_CSV", str(extra))


def _113_parquet(tmp_path, rows):
    import pandas as pd
    base = {"start_s": 500.0, "i_auc": 0.5, "i_p": 0.4, "i_call": False, "iii_auc": 0.5, "iii_p": 0.6,
            "iii_call": False, "stratum": "normal_awake", "transition": False, "rank": 0, "primary": False}
    pq = tmp_path / "col.parquet"
    pd.DataFrame([dict(base, **r) for r in rows]).to_parquet(pq)
    return str(pq)


def test_columbia113_maps_design_rows_by_index_and_extras_by_stem(tmp_path, monkeypatch):
    _113_lists(tmp_path, monkeypatch)
    pq = _113_parquet(tmp_path, [{"sample_idx": 1, "source": "design", "stem": None},
                                 {"sample_idx": 2, "source": "extra", "stem": "x2"},
                                 {"sample_idx": 0, "source": "extra", "stem": "x0"}])
    c = rn.columbia113(pq)
    assert list(c["uid"]) == ["d1", "x2", "x0"] and list(c["patient"]) == ["pt_d1", "pt_x2", "pt_x0"]
    assert list(c["source"]) == ["design", "extra", "extra"]
    legacy = _113_parquet(tmp_path, [{"sample_idx": 0}])
    assert list(rn.columbia113(legacy)["uid"]) == ["d0"]


def test_columbia113_refuses_an_extra_whose_stem_disagrees_with_its_list(tmp_path, monkeypatch):
    _113_lists(tmp_path, monkeypatch)
    pq = _113_parquet(tmp_path, [{"sample_idx": 1, "source": "extra", "stem": "x2"}])
    with pytest.raises(RuntimeError, match="stem"):
        rn.columbia113(pq)
    pq = _113_parquet(tmp_path, [{"sample_idx": 0, "source": "other", "stem": "x0"}])
    with pytest.raises(RuntimeError, match="source"):
        rn.columbia113(pq)


def test_extra_csv_is_the_engines_extra_list():
    assert os.path.basename(rn.EXTRA_CSV) == "bench_extra_500.csv" and os.path.dirname(rn.EXTRA_CSV) == en.META



def test_timing_groups_separate_paradigms_on_the_same_timing():
    m = _stub()
    m.VARIANTS = {"vt3_a": {"gap": 5.0, "paradigm": "vt3"}, "vt2_a": {"gap": 5.0, "paradigm": "vt2"},
                  "vt3_b": {"gap": 5.0, "paradigm": "vt3", "x": 1}, "vt2_b": {"gap": 5.0, "paradigm": "vt2", "x": 1}}
    m.DEFAULT = "vt3_a"
    specs = rn.variant_specs(m, "pdoc")
    assert rn.timing_groups(m, specs, list(specs), "vt3_a") == {"vt3_a": ["vt3_a", "vt3_b"], "vt2_a": ["vt2_a", "vt2_b"]}


def test_mbvt_groups_split_vt2_and_vt3():
    mod = rn.load_pipe("mbvt")
    specs = rn.variant_specs(mod, "bdsp")
    g = rn.timing_groups(mod, specs, list(specs), rn.primary_variant(mod, "bdsp"))
    assert len(g) == 4 and all(len({mod.VARIANTS[v]["paradigm"] for v in vs}) == 1 for vs in g.values())


def test_dubois_non_pipeline_group_is_evaluated_on_the_subsample_only(monkeypatch, tmp_path):
    pts = {"erd|1": {"groups": {"a": {"effect": 0.4, "comparator": "b", "reached": True},
                                "c": {"effect": 0.6, "comparator": "c", "reached": True}}, "pipeline_group": "a"}}
    _eval_file(tmp_path, monkeypatch, {"ncmd_bdsp": {"dubois": pts}})
    kw = dict(null="ncmd_bdsp", n_recs=480, resp="eval", locator=set())
    out = _collect(_bdsp_rec("d1", dur=2000), _stub(name="dubois"), uncorrected=set(), **kw)
    assert {(r["group"], r["eval_set"]) for r in out} == {("a", "all")}
    sub = _collect(_bdsp_rec("d1", dur=2000), _stub(name="dubois"), uncorrected={"pt_d1"}, **kw)
    assert {(r["group"], r["eval_set"]) for r in sub} == {("a", "all"), ("c", "uncorrected_subsample")}
    _eval_file(tmp_path, monkeypatch, {"ncmd_pdoc": {"stub": pts}})
    other = _collect(_rec(dur=2000), _stub(), n_recs=480, resp="eval", locator=set(), uncorrected=set())
    assert {(r["group"], r["eval_set"]) for r in other} == {("a", "all"), ("c", "all")}


@pytest.fixture
def no_usable(monkeypatch):
    """bench.motor_common.NoUsableTrials (the module owners add it; a stand-in until then)."""
    if not hasattr(mc, "NoUsableTrials"):
        monkeypatch.setattr(mc, "NoUsableTrials", type("NoUsableTrials", (ValueError,), {}), raising=False)
    return mc.NoUsableTrials


def _raising_stub(name, exc, where="run"):
    m = _stub(name=name)
    real = getattr(m, where)

    def boom(rec, ov, variant="a", rng=None):
        if variant in ("a", "b") or isinstance(variant, dict):
            raise exc
        return real(rec, ov, variant, rng)
    setattr(m, where, boom)
    return m


def test_no_usable_trials_is_an_na_row_not_an_error(no_usable):
    rows = _collect(_rec(dur=2000), _raising_stub("stub", no_usable("all 60 trials rejected")), n_recs=3)
    st = {r["variant"]: r["status"] for r in rows if r["j"] == 0}
    assert st == {"a": "na: no_usable_trials", "b": "na: no_usable_trials", "c": "ok"}
    na = [r for r in rows if r["status"] == "na: no_usable_trials"]
    assert all(r["stat"] is None and r["call"] is None and "rejected" in r["na_detail"] for r in na)
    assert all(r.get("stratum") is not None for r in na)


def test_only_no_usable_trials_is_na_and_every_other_error_stays_an_error(no_usable, monkeypatch):
    """A module's own name never turns an exception into N/A: a cruse11 ValueError is an error (a real bug stays
    visible); its NoUsableTrials is na."""
    monkeypatch.setattr(rn.wr, "short_variants", lambda mod, source: {})
    for name, fam in (("cruse11", "cmd"), ("cruse12p", "cmd"), ("cornell", "cmd"), ("gibson16", "erp"), ("stub", "cmd")):
        null = "nstim0_bdsp" if fam == "erp" else "ncmd_bdsp"
        for exc, want in ((ValueError("boom"), "error: ValueError: boom"),
                          (no_usable("all trials rejected"), "na: no_usable_trials")):
            m = _raising_stub(name, exc)
            m.FAMILY = fam
            rows = _collect(_bdsp_rec("m1", dur=2000), m, null=null, n_recs=3)
            assert {r["status"] for r in rows if r["variant"] == "a"} == {want}, (name, want)
    assert not hasattr(rn, "NA_MODULES")


def test_a_randomized_run_without_usable_trials_is_na(no_usable):
    rows = _collect(_rec(dur=2000), _raising_stub("stub", no_usable("none kept"), "run_randomized"), n_recs=3)
    rest = [r for r in rows if r["variant"] is None and r["group"] == "a"]
    assert rest and {r["status"] for r in rest} == {"na: no_usable_trials"}
    pub = [r for r in rows if r["variant"] == "a"]
    assert pub[0]["status"] == "ok" and pub[0]["rand_call"] is None and pub[0]["rand_error"] == "na: no_usable_trials"


def test_npassive_published_no_usable_trials_is_na_and_other_errors_stay_errors(monkeypatch, roots, tmp_path,
                                                                                no_usable):
    """The N-passive published path (run_events) maps a module's NoUsableTrials to na: no_usable_trials like the
    engine nulls; any other exception stays an error."""
    for exc, want in ((no_usable("2 of 15 groups survive"), "na: no_usable_trials"),
                      (ValueError("boom"), "error: ValueError: boom")):
        mod = _raising_stub("stuberp", exc)
        mod.FAMILY = "erp"
        monkeypatch.setitem(sys.modules, "bench.pipe_stuberp", mod)
        _passive_stubs(monkeypatch, [_rec(uid="s1", patient="s1")])
        pre = _spec_path(tmp_path, False)
        rn.run_shard("npassive_erpcore_mmn", "stuberp", "dev", spec_path=pre)
        rows = [r for r in _rows(rn.shard_file("npassive_erpcore_mmn", "stuberp", "native", "none", "dev", 0))
                if r["design"] == "published"]
        st = {r["variant"]: r["status"] for r in rows}
        assert st == {"a": want, "b": want, "c": "skipped_by_adapter"}, st
        if want.startswith("na"):
            assert all("2 of 15" in r["na_detail"] and r["call"] is None for r in rows if r["variant"] in "ab")
        os.remove(rn.shard_file("npassive_erpcore_mmn", "stuberp", "native", "none", "dev", 0))
