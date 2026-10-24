"""Tests for 130_fidelity_readings.py (C3).
Synthetic data only. cd code && /usr/bin/python3 -m pytest test_fidelity_readings.py -q"""
import functools
import importlib
import json
import os

import numpy as np
import pytest
from scipy.stats import ttest_rel

from bench import engine as en
from bench import motor_common as mc
from bench import pipe_cornell as pc
from bench import pipe_dubois as pdb
from bench import wrappers as wr
from bench.tests import _synth as sy

fr = importlib.import_module("130_fidelity_readings")
rn = importlib.import_module("114_benchmark")

LOCKED = "415921a4bde6ca83b47691cb7ab1b7713c25691029b604541312a932c598cf45"
SMALL = [('motor', 'mu-lowgamma', 1.0, 4), ('motor', 'mu-lowgamma', 2.0, 6)]



def test_readings_and_their_bases():
    assert fr.readings("cornell") == ["c3_jackknife_bh_keep40_r4_om2contig"]
    assert fr.readings("dubois") == ["c3_b_assessment_random_phase1", "c3_b_assessment_blocked_phase1"]
    assert fr.readings("mbvt") == []
    assert fr.base("cornell", "c3_jackknife_bh_keep40_r4_om2contig") == "jackknife_bh_keep40_r4"
    assert fr.base("dubois", "c3_b_assessment_blocked_phase1") == "assessment_blocked_phase1"
    assert all(fr.base(p, r) in m.VARIANTS for p, m in (("cornell", pc), ("dubois", pdb)) for r in fr.readings(p))



_F = np.fft.rfftfreq(int(round(pc.EPOCH_S * 128.0)), 1 / 128.0)
F_HZ = _F[(_F >= pc.FPASS[0] - 1e-9) & (_F <= pc.FPASS[1] + 1e-9)]


def _rej(ch_bins, n_ch=19):
    rej = np.zeros((n_ch, len(F_HZ)), bool)
    for ch, b in ch_bins:
        rej[ch, b] = True
    return rej


def test_om2_contiguity_needs_a_range_above_2_hz():
    z = np.ones((19, len(F_HZ)))
    assert not fr.om2_contiguous(z, _rej([(3, 10)]), F_HZ)
    assert F_HZ[17] - F_HZ[11] == 2.0
    assert not fr.om2_contiguous(z, _rej([(3, b) for b in range(11, 18)]), F_HZ)
    assert fr.om2_contiguous(z, _rej([(3, b) for b in range(10, 18)]), F_HZ)
    zs = z.copy()
    zs[3, 14:] = -1.0
    assert not fr.om2_contiguous(zs, _rej([(3, b) for b in range(10, 18)]), F_HZ)
    assert not fr.om2_contiguous(z, _rej([(3, b) for b in range(10, 14)] + [(4, b) for b in range(14, 18)]), F_HZ)
    assert not fr.om2_contiguous(z, _rej([(3, b) for b in list(range(10, 14)) + list(range(15, 19))]), F_HZ)


def _cornell_rec(seed, planted):
    ov = pc.schedule(10.0, None, "jackknife_bh_keep40_r4")
    r = sy.rec(pc.extent(ov)[1] + 5.0, seed=seed)
    if planted:
        r = sy.plant(r, ov.onsets_s[ov.labels == pc.TASK] + pc.CMD_S, ["C3", "C4", "Cz", "P3", "P4"], 1.0, 10.0,
                     depth=0.7)
    return r, ov


@pytest.mark.parametrize("seed,planted", [(1, True), (2, True), (3, False), (4, False)])
def test_cornell_reading_is_the_frozen_jackknife_run_with_a_stricter_om2(seed, planted):
    r, ov = _cornell_rec(seed, planted)
    fz = pc.run(r, ov, "jackknife_bh_keep40_r4")
    res = fr.run("cornell", r, ov, "c3_jackknife_bh_keep40_r4_om2contig")
    e = res["extra"]
    assert e["om1"] == fz["extra"]["om1"] and e["om2_anybin"] == fz["extra"]["om2"]
    assert e["call_anybin"] == fz["call"] and res["stat"] == fz["stat"] and res["p"] == fz["p"]
    assert (not res["call"]) or fz["call"]
    assert res["call"] == (e["om1"] and e["om2_contig"])
    if planted:
        assert res["call"]


def test_cornell_reading_schedule_is_its_base_schedule():
    a = fr.schedule("cornell", 40.0, np.random.default_rng(1), "c3_jackknife_bh_keep40_r4_om2contig")
    b = pc.schedule(40.0, np.random.default_rng(1), "jackknife_bh_keep40_r4")
    assert np.array_equal(a.onsets_s, b.onsets_s) and np.array_equal(a.labels, b.labels)



def test_dubois_reading_b_takes_peak_times_from_the_fold_mean_curve():
    rng = np.random.default_rng(5)
    fold = rng.uniform(0.4, 0.6, (6, len(pdb.TIMES)))
    task, base = pdb._masks()
    fold[:, np.nonzero(task)[0][20]] += np.array([0.3, 0.0, 0.25, 0.0, 0.2, 0.0])
    mean = fold.mean(0)
    tv, p, ti, bi = fr.reading_b(mean, fold)
    assert ti == np.nonzero(task)[0][np.argmax(mean[task])] and bi == np.nonzero(base)[0][np.argmax(mean[base])]
    ref = ttest_rel(fold[:, ti], fold[:, bi], alternative="greater")
    assert tv == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    tA, pA = pdb.paired_t(fold)
    assert pA <= p


def test_dubois_reading_b_zero_variance_convention_is_the_frozen_one():
    fold = np.full((6, len(pdb.TIMES)), 0.5)
    task, _ = pdb._masks()
    fold[:, np.nonzero(task)[0][3]] = 0.6
    assert fr.reading_b(fold.mean(0), fold)[:2] == (np.inf, 0.0)
    assert fr.reading_b(np.full(len(pdb.TIMES), 0.5), np.full((6, len(pdb.TIMES)), 0.5))[:2] == (0.0, 1.0)


@functools.lru_cache(maxsize=None)
def _dubois_rec(seed):
    ov = pdb.schedule(10.0, np.random.default_rng(seed), "assessment_random_phase1")
    return sy.rec(pdb.extent(ov)[1] + 5.0, seed=seed), ov


@pytest.mark.parametrize("cv", ["random", "blocked"])
def test_dubois_reading_b_shares_the_frozen_search_and_folds(cv):
    r, ov = _dubois_rec(7)
    base = dict(pdb.VARIANTS[f"assessment_{cv}_phase1"], configs=SMALL)
    fz = pdb.run(r, ov, base, np.random.default_rng(11))
    res = fr.dubois(r, ov, f"c3_b_assessment_{cv}_phase1", np.random.default_rng(11), base=base)
    e = res["extra"]
    assert e["p_t_A"] == fz["p"] and e["peak_da"] == fz["extra"]["peak_da"] and e["config"] == fz["extra"]["config"]
    assert res["call"] == (e["peak_da"] > pdb.DA_70 or res["p"] < pdb.ALPHA)
    assert e["calls"]["t_only"] == (res["p"] < pdb.ALPHA) and e["calls"]["phase1"] == res["call"]









def test_outputs_are_refused_inside_the_bench_shards(tmp_path):
    for p in (rn.SH["analysis"], rn.SH["dev"], os.path.join(rn.SH["analysis"], "x", "rows00.jsonl")):
        with pytest.raises(RuntimeError, match="_bench_shards"):
            fr.outside_bench_shards(p)
    fr.outside_bench_shards(str(tmp_path / "rows00.jsonl"))
    assert "_bench_shards" not in fr.shard_file("ncmd_bdsp", "cornell", 0)
    assert os.path.dirname(os.path.dirname(fr.shard_file("ncmd_bdsp", "cornell", 0))) == fr.SH
    assert fr.OUT.endswith(os.path.join("output", "tables", "c3_fidelity.json"))


def _fake_listing(monkeypatch, recs):
    """Real recordings are never reachable from a test: the listing, loading and reading are all fakes."""
    monkeypatch.setattr(rn, "touched_uids", lambda: set())
    monkeypatch.setattr(en, "_items", lambda source, split: [{"uid": r.uid, "patient": r.patient, "rec": r,
                                                              "meta": {}} for r in recs])
    monkeypatch.setattr(en, "_load", lambda it: it["rec"])





def _bdsp_like(dur_s, uid="Bfake_0", seed=3):
    r = sy.rec(dur_s, seed=seed)
    t = np.arange(int(dur_s))
    return en.Recording(source="bdsp", uid=uid, patient="Bfake", x=np.asarray(r.x, np.float32), fs=128.0,
                        names=list(en.NAMES19), meta={"dev": False, "morgoth_t": t,
                                                      "morgoth_lab": np.array(["normal_awake"] * len(t), object)})


@pytest.mark.parametrize("pipe", ["cornell", "dubois"])
def test_driver_gives_each_reading_the_phase1_overlay_and_run_seed_of_its_base(pipe, monkeypatch):
    """Spy on the frozen run() while 114 runs the Phase-1 (resp none) recording, then compare with the driver's jobs."""
    mod = rn.load_pipe(pipe)
    rec = _bdsp_like(1300.0 if pipe == "cornell" else 1500.0)
    seen = {}

    def spy(r, ov, variant, rng):
        seen[(variant, len(seen))] = (np.asarray(ov.onsets_s).copy(), np.asarray(ov.labels).copy(),
                                      rng.bit_generator.state["state"])
        return mc.result(0.0, 1.0, False)

    monkeypatch.setattr(mod, "run", spy)
    monkeypatch.setattr(mod, "run_randomized", lambda r, ov, v, rng: mc.result(0.0, 1.0, False))
    spec = rn.parse_null("ncmd_bdsp")
    rows = []
    n_recs = 40
    ctx = fr.ctx_for(spec, n_recs) | {"spec_path": None}
    rn.run_recording(rec, mod, spec, ctx, set(), rows.append)
    jobs, why = fr.phase1_jobs(rec, mod, spec, n_recs)
    assert why is None and jobs
    j0 = [r for r in rows if r.get("variant") and r.get("j") == 0]
    for jb in jobs:
        hits = [v for (var, _), v in seen.items() if var == jb["base"]]
        rowj = [r for r in rows if r.get("variant") == jb["base"] and r.get("j") == jb["j"]]
        assert rowj and rowj[0]["start_s"] == jb["start_s"]
        on, lab, st = hits[[r["j"] for r in rows if r.get("variant") == jb["base"]].index(jb["j"])]
        assert np.array_equal(on, jb["ov"].onsets_s) and np.array_equal(lab, jb["ov"].labels)
        assert st == jb["rng"].bit_generator.state["state"]
    assert {jb["j"] for jb in jobs} == {r["j"] for r in rows if r.get("variant") and r.get("seq")}
    assert j0 and any(r.get("variant") is None and r.get("j", 0) > 0 for r in rows)
    first = fr.phase1_jobs(rec, mod, spec, n_recs, first_only=True)[0]
    assert {jb["j"] for jb in first} == {0}


