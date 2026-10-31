"""Tests for code/115_benchmark_merge.py on synthetic rows only: Holm, the
bootstrap (patient-level, relabelled draws), the published-pipeline bar, the proposed-method validity families, the
cross-fitted size correction (and a deliberately broken same-half version, which the check must catch), the
evaluation-point rule, the non-inferiority CI, composite curves (BH within patient) and the dev/analysis separation.
cd code && /usr/bin/python3 -m pytest bench/tests/test_merge.py -q"""
import importlib
import json
import os
import sys
import types
import zlib

import numpy as np
import pandas as pd
import pytest

mg = importlib.import_module("115_benchmark_merge")


@pytest.fixture(autouse=True)
def _frozen_dev_files(tmp_path, monkeypatch):
    """The analysis guard needs the frozen dev files (tests never read the real ones)."""
    rn = importlib.import_module("114_benchmark")
    frozen = [tmp_path / "dev_split_t.json", tmp_path / "passive_dev_split_t.json"]
    for f in frozen:
        f.write_text("{}")
    monkeypatch.setattr(rn, "FROZEN_FILES", tuple(str(f) for f in frozen))


def _half(p):
    return zlib.crc32(p.encode()) & 1


def _null_rows(pipe, variant, calls, null="ncmd_bdsp", stat=None, j=0, stratum="normal_awake", **kw):
    """One row per patient (calls: 0/1 per patient)."""
    out = []
    for i, c in enumerate(calls):
        out.append({"null": null, "pipe": pipe, "variant": variant, "patient": f"P{i:04d}", "uid": f"U{i:04d}",
                    "j": j, "seq": True, "overlap": False, "resp": "none", "montage": "native", "status": "ok",
                    "call": bool(c), "p": 0.01 if c else 0.5, "stat": float(c) if stat is None else float(stat[i]),
                    "size_stat": float(c) if stat is None else float(stat[i]), "as_coded": False,
                    "stratum": stratum, "transitioning": False, "artifact_label": "<10%", "split": "dev",
                    "dev": True, "model": "none", "effect": 0.0, "prob": 0.0, **kw})
    return out



def test_holm():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(mg.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert mg.holm([]).size == 0 and np.allclose(mg.holm([0.2]), [0.2])
    assert np.allclose(mg.holm([0.5, 0.6, 0.01]), [1.0, 1.0, 0.03])


def test_one_sided_binomial():
    from scipy.stats import binom
    assert np.isclose(mg.binom_greater(3, 10), binom.sf(2, 10, 0.05))
    assert mg.binom_greater(0, 10) == 1.0


def test_bootstrap_draws_are_relabelled_patient_copies():
    """A patient drawn twice counts as two patients: the fast per-patient draw equals building the resampled rows with
    new ids and applying the composite rule per new id; keeping the original ids (merging the copies) does not."""
    rule = lambda ps: bool(np.any(mg.bh_reject(ps)))
    rng = np.random.default_rng(0)
    pats = {f"P{i}": list(rng.uniform(0, 0.2, rng.integers(1, 4))) for i in range(30)}
    names = sorted(pats)
    outcome = np.array([rule(pats[p]) for p in names], float)
    idx = mg.boot_idx(len(names), 200, seed=1)
    fast = mg.mean_draws(outcome, idx)
    relabelled, merged = [], []
    for row in idx:
        rows = [(f"{names[i]}#{k}", p) for k, i in enumerate(row) for p in pats[names[i]]]
        d = pd.DataFrame(rows, columns=["pid", "p"])
        relabelled.append(d.groupby("pid").p.apply(lambda s: rule(list(s))).mean())
        d["orig"] = d.pid.str.split("#").str[0]
        merged.append(d.groupby("orig").p.apply(lambda s: rule(list(s))).mean())
    assert np.allclose(fast, relabelled)
    assert not np.allclose(fast, merged)


def test_ratio_draws_resample_patients_not_rows():
    s, n = np.array([2.0, 0.0, 1.0]), np.array([2.0, 1.0, 4.0])
    idx = np.array([[0, 0, 1], [2, 2, 2]])
    assert np.allclose(mg.ratio_draws(s, n, idx), [4 / 5, 3 / 12])



def test_published_bar_validity_primary_and_holm():
    rng = np.random.default_rng(3)
    rows = (_null_rows("A", "a_loose", rng.random(400) < 0.20) + _null_rows("A", "a_tight", rng.random(400) < 0.12)
            + _null_rows("B", "b1", rng.random(400) < 0.05) + _null_rows("C", "c1", rng.random(60) < 0.10))
    out = mg.published(pd.DataFrame(rows), B=500)
    assert out["A"]["validity_primary"] == "a_tight"
    assert out["A"]["variants"]["a_tight"]["rate"] < out["A"]["variants"]["a_loose"]["rate"]
    assert out["A"]["inflated"] and not out["B"]["inflated"]
    adj = mg.holm([out[k]["p_boot"] for k in ("A", "B", "C")])
    assert [out[k]["inflated"] for k in ("A", "B", "C")] == list(adj <= 0.05)
    for k in "ABC":
        v = out[k]["variants"][out[k]["validity_primary"]]
        assert out[k]["inflated_unadjusted"] == (v["lower95"] > 0.05)
        assert v["n_patients"] == (60 if k == "C" else 400)
    assert out["A"]["variant_range"] == [out["A"]["variants"]["a_tight"]["rate"], out["A"]["variants"]["a_loose"]
                                         ["rate"]]


def test_published_bar_ties_prefer_as_coded():
    calls = [1] * 5 + [0] * 95
    rows = _null_rows("A", "x", calls) + [dict(r, as_coded=True) for r in _null_rows("A", "y", calls)]
    assert mg.published(pd.DataFrame(rows), B=100)["A"]["validity_primary"] == "y"


def test_published_uses_the_first_overlay_only():
    rows = _null_rows("A", "a", [0] * 50) + _null_rows("A", "a", [1] * 50, j=1)
    assert mg.published(pd.DataFrame(rows), B=100)["A"]["variants"]["a"]["rate"] == 0.0



def _rand_rows(pipe, null, per_rec, n_recs, stratum="normal_awake", hot=None):
    rows = []
    rng = np.random.default_rng(zlib.crc32(null.encode()))
    for r in range(n_recs):
        for j in range(per_rec):
            c = (hot == r) or (rng.random() < 0.02)
            rows.append({"null": null, "pipe": pipe, "uid": f"{null}{r}", "patient": f"{null}{r}", "j": j,
                         "variant": None if j else "a", "resp": "none", "montage": "native", "status": "ok",
                         "rand_call": bool(c), "rand_stratum": stratum, "rand_transitioning": False,
                         "rand_artifact_label": "<10%", "split": "dev", "dev": True})
    return rows


def test_proposed_validity_passes_at_the_nominal_rate():
    rows = _rand_rows("A", "ncmd_bdsp", 1, 400) + _rand_rows("A", "ncmd_pdoc", 6, 38)
    v = mg.proposed_validity(pd.DataFrame(rows))["A"]
    assert v["point"] <= 0.05 and v["pass"]
    assert set(v["families"]) == {"null_set", "stratum", "recording"}
    assert {t["unit"] for t in v["families"]["null_set"]} == {"ncmd_bdsp", "ncmd_pdoc"}


def test_one_bad_recording_fails_validity_even_at_a_low_pooled_rate():
    rows = _rand_rows("A", "ncmd_pdoc", 10, 38, hot=5)
    v = mg.proposed_validity(pd.DataFrame(rows))["A"]
    rec = {t["unit"]: t for t in v["families"]["recording"]}
    assert rec["ncmd_pdoc|ncmd_pdoc5"]["p_holm"] <= 0.05 and not v["pass"]


def test_validity_point_above_5_percent_fails():
    rows = _rand_rows("A", "ncmd_bdsp", 1, 100)
    for r in rows[:7]:
        r["rand_call"] = True
    for r in rows[7:]:
        r["rand_call"] = False
    assert not mg.proposed_validity(pd.DataFrame(rows))["A"]["pass"]



def _crossfit_data():
    """Null stats: low in half 0, high in half 1; responders in between, so a threshold from the other half and one from
    the same half give opposite calls."""
    null, resp = [], []
    pats = [f"Q{i:03d}" for i in range(200)]
    for i, p in enumerate(pats):
        h = _half(p)
        for s in ("normal_awake", "sleep"):
            null.append({"patient": p, "stratum": s, "transitioning": False, "size_stat": (1.0 if h == 0 else 10.0)
                         + 0.01 * i + (0.5 if s == "sleep" else 0.0)})
            resp.append({"patient": p, "stratum": s, "transitioning": False, "size_stat": 5.0, "effect": 1.0})
    tie = next(f"T{i:03d}" for i in range(100) if _half(f"T{i:03d}") == 0)
    other = [r["size_stat"] for r in null if _half(r["patient"]) == 1 and r["stratum"] == "normal_awake"]
    resp.append({"patient": tie, "stratum": "normal_awake", "transitioning": False,
                 "size_stat": float(np.quantile(other, 0.95, method="higher")), "effect": 1.0})
    return pd.DataFrame(null), pd.DataFrame(resp)


def _assert_crossfit(fn):
    """The bar: every responder row is judged by the 95th percentile ('higher') of the null values of the OTHER half in
    its stratum; it calls when its statistic is >= that threshold."""
    null, resp = _crossfit_data()
    out = fn(null, resp)
    for _, r in out.iterrows():
        h = _half(r["patient"])
        assert r["thr_half"] == 1 - h, "threshold applied to the half it was estimated on"
        other = null[(null.patient.map(_half) == 1 - h) & (null.stratum == r["stratum"])].size_stat
        assert np.isclose(r["thr"], np.quantile(other, 0.95, method="higher"))
        assert r["corr_call"] == (r["size_stat"] >= r["thr"])


def test_crossfit_threshold_comes_from_the_other_half():
    _assert_crossfit(mg.crossfit_calls)
    null, resp = _crossfit_data()
    out = mg.crossfit_calls(null, resp)
    q = out.patient.str.startswith("Q")
    assert out[q & (out.patient.map(_half) == 1)].corr_call.all()
    assert not out[q & (out.patient.map(_half) == 0)].corr_call.any() and out[~q].corr_call.all()


def test_a_same_half_calibration_is_caught():
    def same_half(null, resp):
        out = resp.copy()
        h = out.patient.map(_half)
        thr = [np.quantile(null[(null.patient.map(_half) == hh) & (null.stratum == s)].size_stat, 0.95,
                           method="higher") for hh, s in zip(h, out.stratum)]
        out["thr"], out["thr_half"] = thr, h
        out["corr_call"] = out.size_stat > out.thr
        return out
    with pytest.raises(AssertionError):
        _assert_crossfit(same_half)


def test_small_strata_fall_back_to_the_pooled_half_threshold():
    null, resp = _crossfit_data()
    null = pd.concat([null, pd.DataFrame([{"patient": "Q000", "stratum": "rare", "transitioning": False,
                                           "size_stat": 99.0}])])
    resp = pd.DataFrame([{"patient": p, "stratum": "rare", "transitioning": False, "size_stat": 5.0, "effect": 1.0}
                         for p in ("Q001", "Q002", "Q003", "Q004")])
    out = mg.crossfit_calls(null, resp)
    assert out.thr_pooled.all()
    for _, r in out.iterrows():
        other = null[null.patient.map(_half) == 1 - _half(r["patient"])].size_stat
        assert np.isclose(r["thr"], np.quantile(other, 0.95, method="higher"))


def test_sensitivity_is_the_average_of_the_two_halves():
    p0 = next(f"Q{i:03d}" for i in range(100) if _half(f"Q{i:03d}") == 0)
    p1 = next(f"Q{i:03d}" for i in range(100) if _half(f"Q{i:03d}") == 1)
    df = pd.DataFrame({"patient": [p0] * 3 + [p1], "corr_call": [True, True, False, False]})
    h = df.patient.map(_half)
    assert h.nunique() == 2
    want = np.mean([df[h == k].corr_call.mean() for k in (0, 1)])
    assert np.isclose(mg.halves_mean(df, "corr_call"), want) and np.isclose(want, 1 / 3)



def test_evaluation_point_is_the_smallest_effect_reaching_50_percent():
    assert mg.evaluation_point({0.4: 0.8, 0.1: 0.2, 0.2: 0.55, 0.3: 0.45}) == 0.2
    assert mg.evaluation_point({0.1: 0.2, 0.2: 0.49}) is None
    assert mg.evaluation_point({0.1: 0.5}) == 0.1


def _pairs(n, p_prop, p_comp, seed=0):
    g = np.random.default_rng(seed)
    pats = [f"N{i:04d}" for i in range(n)]
    return pd.DataFrame({"patient": pats, "prop": g.random(n) < p_prop, "comp": g.random(n) < p_comp})


def test_ni_ci_is_a_patient_paired_bootstrap_stratified_by_half():
    d = _pairs(600, 0.7, 0.7)
    r = mg.ni_test(d, "prop", "comp", B=400, seed=5)
    h = d.patient.map(_half)
    est = np.mean([d[h == k].prop.mean() - d[h == k].comp.mean() for k in (0, 1)])
    assert np.isclose(r["diff"], est) and r["ci95"][0] < est < r["ci95"][1]
    g = np.random.default_rng(5)
    draws = []
    parts = [d[h == k].reset_index(drop=True) for k in (0, 1)]
    idx = [g.integers(0, len(p), (400, len(p))) for p in parts]
    for b in range(400):
        draws.append(np.mean([p.prop.to_numpy(float)[i[b]].mean() - p.comp.to_numpy(float)[i[b]].mean()
                              for p, i in zip(parts, idx)]))
    assert np.allclose(r["ci95"], np.quantile(draws, [0.025, 0.975]))
    assert r["noninferior"] == (r["ci95"][0] > -0.05)


def test_ni_decisions():
    assert mg.ni_test(_pairs(2000, 0.80, 0.80, 1), "prop", "comp", B=300)["noninferior"]
    assert not mg.ni_test(_pairs(2000, 0.55, 0.80, 2), "prop", "comp", B=300)["noninferior"]



def _seq_rows(pipe, variant, p_by_patient, null="ncmd_bdsp"):
    rows = []
    for pat, ps in p_by_patient.items():
        for j, p in enumerate(ps):
            rows.append({"null": null, "pipe": pipe, "variant": variant, "patient": pat, "uid": pat, "j": j,
                         "seq": True, "overlap": False, "resp": "none", "montage": "native", "status": "ok",
                         "p": p, "call": p < 0.05, "stat": 0.6, "stratum": "normal_awake", "transitioning": False,
                         "split": "dev", "dev": True})
    return rows


def test_columbia_composite_is_bh_within_patient_over_the_first_k():
    col = importlib.import_module("bench.pipe_columbia")
    pats = {"A": [0.04, 0.5], "B": [0.5, 0.01], "C": [0.3], "D": [0.6, 0.7, 0.001]}
    curve = mg.composite_curve(pd.DataFrame(_seq_rows("columbia", "as_coded", pats)), col.patient_rule, (1, 2, 3),
                               B=200)
    by_k = {c["k"]: c for c in curve}
    assert by_k[1]["n"] == 4 and np.isclose(by_k[1]["rate"], 1 / 4)
    assert by_k[2]["n"] == 3 and np.isclose(by_k[2]["rate"], 1 / 3)
    assert by_k[3]["n"] == 1 and by_k[3]["rate"] == 1.0
    assert by_k[1]["ci95"][0] <= by_k[1]["rate"] <= by_k[1]["ci95"][1]


def test_any_of_k_curve_rises_and_extra_fields_reach_the_rule():
    pats = {f"P{i}": list(np.random.default_rng(i).uniform(0, 1, 5)) for i in range(50)}
    seen = []

    def rule(results):
        seen.append(results[0]["extra"])
        return any(r["call"] for r in results)
    rows = pd.DataFrame(_seq_rows("x", "v", pats))
    rows["x_acc"] = 0.3
    curve = mg.composite_curve(rows, rule, (1, 3, 5), B=100)
    rates = [c["rate"] for c in curve]
    assert rates == sorted(rates) and seen[0] == {"acc": 0.3}


def test_composite_ignores_overlapping_and_randomized_only_overlays():
    rows = _seq_rows("x", "v", {"A": [0.01, 0.01]})
    rows[1]["overlap"] = True
    curve = mg.composite_curve(pd.DataFrame(rows), lambda r: True, (1, 2), B=50)
    assert [c["n"] for c in curve] == [1, 0]


def test_any_approach_is_an_upper_bound_over_pipelines_on_the_same_patients():
    rows = pd.DataFrame(_null_rows("A", "a", [1, 0, 0, 0]) + _null_rows("B", "b", [0, 1, 0, 0])
                        + _null_rows("C", "c", [0, 0], null="nstim0_bdsp"))
    out = mg.any_approach(rows, {("ncmd_bdsp", "A"): "a", ("ncmd_bdsp", "B"): "b", ("nstim0_bdsp", "C"): "c"}, B=100)
    assert out["n_patients"] == 2 and out["rate"] == 1.0 and sorted(out["pipelines"]) == ["A", "B", "C"]



def _write(root, sub, rows):
    os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, sub, "rows00.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_merge_refuses_dev_rows_in_analysis_and_wrong_split(tmp_path):
    root = str(tmp_path / "sh")
    rows = [dict(r, key=f"k{i}") for i, r in enumerate(_null_rows("A", "a", [0, 1]))]
    _write(root, "ncmd_bdsp__A__native__none", [dict(r, split="analysis", dev=False) for r in rows])
    assert len(mg.load_rows("analysis", root)) == 2
    _write(root, "ncmd_bdsp__B__native__none", [dict(rows[0], split="analysis", dev=True)])
    with pytest.raises(RuntimeError):
        mg.load_rows("analysis", root)
    _write(root, "ncmd_bdsp__B__native__none", [dict(rows[0], split="dev", dev=True)])
    with pytest.raises(RuntimeError):
        mg.load_rows("analysis", root)


def test_duplicate_keys_are_counted_once(tmp_path):
    root = str(tmp_path / "sh")
    r = dict(_null_rows("A", "a", [1])[0], key="k0")
    _write(root, "ncmd_bdsp__A__native__none", [r, r])
    assert len(mg.load_rows("dev", root)) == 1


def test_coverage_counts_no_fit_errors_and_windows():
    rows = _seq_rows("x", "v", {"A": [0.5, 0.5], "B": [0.5]})
    rows.append(dict(rows[0], patient="C", uid="C", status="no_fit", variant=None))
    rows.append(dict(rows[0], patient="D", uid="D", status="error: ValueError: x"))
    cov = mg.coverage(pd.DataFrame(rows))["ncmd_bdsp"]["x"]
    assert cov["recordings"] == 4 and cov["no_fit"] == 1 and cov["error_rows"] == 1
    assert cov["seq_windows_max"] == 2 and cov["recordings_with_k"] == {"1": 2, "2": 1}


def test_each_contrast_is_its_own_randomized_draw():
    """mbvt on kojimaB runs run_randomized once per contrast on the same recording: both count."""
    rows = _rand_rows("mbvt", "npassive_kojimaB", 1, 30)
    two = [dict(r, contrast=c) for r in rows for c in ("oddball", "equiprobable")]
    v = mg.proposed_validity(pd.DataFrame(two))["mbvt"]
    assert v["n"] == 60
    assert mg.proposed_validity(pd.DataFrame(rows))["mbvt"]["n"] == 30



def _touched_file(tmp_path, monkeypatch):
    fn = tmp_path / "touched.json"
    fn.write_text(json.dumps({"excluded": ["U0001"], "loaded_only_not_excluded": ["U0002"]}))
    monkeypatch.setattr(mg, "TOUCHED_FILE", str(fn))


def _analysis_rows(tmp_path):
    root = str(tmp_path / "sh")
    rows = [dict(r, key=f"k{i}", split="analysis", dev=False, rand_call=False, rand_stratum="normal_awake",
                 rand_transitioning=False, rand_artifact_label="<10%")
            for i, r in enumerate(_null_rows("A", "a", [0, 1, 1, 0]))]
    _write(root, "ncmd_bdsp__A__native__none", rows)
    return root


def _assert_merge_drops_touched(out):
    res = json.load(open(os.path.join(out, "benchmark_ncmd_bdsp.json")))
    v = res["published"]["A"]["variants"]["a"]
    if v["n"] != 3 or v["calls"] != 1:
        raise AssertionError(f"touched recording counted: n={v['n']} calls={v['calls']}")
    head = json.load(open(os.path.join(out, "benchmark_headline.json")))
    assert head["excluded_touched"] == {"recordings": 1, "rows": 1}


def test_merge_drops_recordings_touched_before_the_freeze(tmp_path, monkeypatch):
    _touched_file(tmp_path, monkeypatch)
    out = str(tmp_path / "t")
    mg.merge("analysis", root=_analysis_rows(tmp_path), out=out, B=50, spec_path=_locked(tmp_path))
    _assert_merge_drops_touched(out)


def test_a_merge_that_ignores_the_touched_list_is_caught(tmp_path, monkeypatch):
    _touched_file(tmp_path, monkeypatch)
    monkeypatch.setattr(mg, "touched_uids", lambda: set())
    out = str(tmp_path / "t")
    mg.merge("analysis", root=_analysis_rows(tmp_path), out=out, B=50, spec_path=_locked(tmp_path))
    with pytest.raises(AssertionError):
        _assert_merge_drops_touched(out)


def test_analysis_merge_refuses_without_the_touched_list(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "TOUCHED_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError):
        mg.merge("analysis", root=_analysis_rows(tmp_path), out=str(tmp_path / "t"), B=50, spec_path=_locked(tmp_path))


def _locked(tmp_path):
    """An empty stand-in for a local file the guard reads (not read here)."""
    p = tmp_path / "notes_s2.txt"
    p.write_text("")
    return str(p)



def _sens_rows(n=200, seed=0):
    """Null rows of variants v1 / v2 of pipe A; locator (60 patients) grid rows; evaluation rows of all patients at
    one effect for the comparator; the uncorrected subsample (100 others) carries the permutation p."""
    g = np.random.default_rng(seed)
    pats = [f"S{i:04d}" for i in range(n)]
    loc, unc = set(pats[:60]), set(pats[60:160])
    null, grid, ev = [], [], []
    for v, k in (("v1", 1.0), ("v2", 2.0)):
        for i, p in enumerate(pats):
            null.append({"null": "ncmd_bdsp", "pipe": "A", "variant": v, "patient": p, "uid": p, "j": 0, "seq": True,
                         "overlap": False, "resp": "none", "montage": "native", "status": "ok",
                         "size_stat": g.normal(), "p": g.uniform(), "call": False, "stratum": "normal_awake",
                         "transitioning": False, "as_coded": v == "v1", "split": "dev", "dev": True,
                         "model": "none", "effect": 0.0, "prob": 0.0})
            for e in (0.25, 0.5, 1.0, 2.0):
                if p in loc:
                    grid.append({**null[-1], "resp": "grid", "model": "erd", "effect": e, "prob": 1.0,
                                 "size_stat": g.normal(e * k), "in_locator": True, "p": None})
    for p in pats:
        z = g.normal(1.0 * 2.0)
        ev.append({**null[0], "patient": p, "uid": p, "variant": "v2", "resp": "eval", "model": "erd",
                   "effect": 1.0, "prob": 1.0, "size_stat": z, "p": None, "in_locator": p in loc,
                   "in_uncorrected": p in unc, "rand_call": bool(g.random() < 0.85),
                   "perm_p": float(g.uniform(0, 0.02)) if p in unc else None,
                   "perm_call": bool(g.random() < 0.9) if p in unc else None})
    return pd.DataFrame(null + grid + ev), loc, unc


def test_locator_picks_the_comparator_and_the_smallest_effect_reaching_50_percent():
    df, loc, _ = _sens_rows()
    out = mg.locate(df)["ncmd_bdsp"]["A"]["erd|1"]
    assert out["comparator"] == "v2" and out["n_patients"] == 60
    curve = {float(k): v for k, v in out["curves"]["v2"].items()}
    assert out["evaluation_effect"] == mg.evaluation_point(curve)
    assert out["corrected_size"] <= 0.10


def test_eval_points_are_written_once_and_a_changed_location_is_flagged(tmp_path):
    fn = str(tmp_path / "ep.json")
    pts = {"ncmd_bdsp": {"A": {"erd|1": {"effect": 1.0, "comparator": "v2"}}}}
    assert mg.write_eval_points(pts, fn) == []
    assert mg.write_eval_points(pts, fn) == []
    moved = {"ncmd_bdsp": {"A": {"erd|1": {"effect": 0.5, "comparator": "v2"}}}}
    assert mg.write_eval_points(moved, fn) == ["ncmd_bdsp|A|erd|1"]
    assert json.load(open(fn)) == pts


def test_evaluation_reports_ni_outside_the_locator_and_flags_the_overlap():
    df, loc, unc = _sens_rows()
    r = mg.evaluate(df, B=200)["ncmd_bdsp"]["A"]["erd|1"]
    assert r["comparator"] == "v2" and r["effect"] == 1.0
    assert r["outside_locator"]["n_patients"] == 140 and r["overlap_with_locator"] is True
    assert r["all"]["n_patients"] == 200 and {"diff", "ci95", "noninferior"} <= set(r["outside_locator"]["ni"])
    assert r["uncorrected"]["label"] == "inflated" and r["uncorrected"]["n_patients"] == 100


def test_p_based_correction_more_sensitive_by_over_5_points_is_flagged():
    df, _, _ = _sens_rows()
    r = mg.evaluate(df, B=100)["ncmd_bdsp"]["A"]["erd|1"]["p_based"]
    assert r["sensitivity_p"] - r["sensitivity_stat"] > 0.05 and r["flag_p_more_sensitive"] is True
    df.loc[df["resp"] == "eval", "perm_p"] = np.where(df.loc[df["resp"] == "eval", "in_uncorrected"], 0.99, None)
    r = mg.evaluate(df, B=100)["ncmd_bdsp"]["A"]["erd|1"]["p_based"]
    assert r["flag_p_more_sensitive"] is False



def _c113(n=6):
    return pd.DataFrame({"uid": [f"U{i:04d}" for i in range(n)], "patient": [f"P{i:04d}" for i in range(n)],
                         "start_s": 500.0, "i_auc": 0.5, "i_p": [0.01, 0.5, 0.5, 0.5, 0.2, 0.7],
                         "i_call": [True, False, False, False, False, False], "i_nperm": 25, "i_stopped": True,
                         "i_call_paper": False, "iii_auc": 0.5, "iii_p": 0.6, "iii_call": False, "iii_nperm": 25,
                         "stratum": "normal_awake", "transition": False, "rank": range(n), "sample_idx": range(n),
                         "primary": True})


def test_113_rows_map_test_i_to_published_and_iii_to_the_randomized_method():
    rows = mg.rows_from_113(_c113())
    assert set(rows["pipe"]) == {"columbia"} and set(rows["null"]) == {"ncmd_bdsp"} and (rows["j"] == 0).all()
    assert list(rows["call"]) == list(_c113()["i_call"]) and list(rows["size_stat"]) == list(_c113()["i_auc"])
    assert (rows["rand_call"] == _c113()["iii_call"]).all() and (rows["rand_p"] == 0.6).all()
    assert (rows["variant"] == "as_coded").all() and (rows["split"] == "analysis").all() and not rows["dev"].any()


def test_113_exclusions_must_match_the_benchmark():
    c = _c113()
    bench = pd.DataFrame({"null": "nstim0_bdsp", "pipe": "mbvt", "uid": list(c["uid"]) + ["U9999"],
                          "status": "ok"})
    ok = mg.check_113(c, bench, dev=set(), touched=set())
    assert ok["contained"] and ok["missing"] == {}
    with pytest.raises(RuntimeError):
        mg.check_113(c, bench, dev={"U0002"}, touched=set())
    with pytest.raises(RuntimeError):
        mg.check_113(c, bench, dev=set(), touched={"U0003"})
    short = bench[bench["uid"] != "U0001"]
    r = mg.check_113(c, short, dev=set(), touched=set())
    assert not r["contained"] and r["missing"] == {"nstim0_bdsp|mbvt": 1}



def test_short_twins_paired_difference_and_coverage_by_stratum():
    nat = _null_rows("cruse11", "cr11_b4", [1, 0, 1, 0, 0, 0])
    sh = [dict(r, variant="cr11_b3", short=True, twin_of="cr11_b4,cr11_b6") for r in
          _null_rows("cruse11", "cr11_b3", [0, 0, 1, 0, 0, 0, 0, 1])]
    for r in sh[6:]:
        r["stratum"] = "sleep"
    out = mg.short_twins(pd.DataFrame(nat + sh), B=200)["ncmd_bdsp"]["cruse11"]["cr11_b3"]
    pair = out["pairs"]["cr11_b4"]
    assert pair["n_patients"] == 6 and np.isclose(pair["diff_native_minus_short"], (2 - 1) / 6)
    assert pair["ci95"][0] <= pair["diff_native_minus_short"] <= pair["ci95"][1]
    assert out["coverage"] == {"normal_awake": {"short": 6, "both": 6}, "sleep": {"short": 2, "both": 0}}
    assert "cr11_b6" not in out["pairs"]



def _brute_independent(p, idx=None):
    col = importlib.import_module("bench.pipe_columbia")
    u = np.arange(len(p)) if idx is None else np.asarray(idx)
    hits = [col.patient_rule([{"p": p[u[a]]}, {"p": p[u[b]]}]) for a in range(len(u)) for b in range(a + 1, len(u))
            if u[a] != u[b]]
    return float(np.mean(hits))


def _assert_independent_days(fn):
    """fn(p, idx) -> per-draw rate: the point estimate (no copies) and bootstrap draws (copies of one patient are the
    same recording, never a pair) must equal the brute force over pairs of different patients."""
    g = np.random.default_rng(3)
    p = np.r_[np.full(6, 0.01), g.uniform(0.06, 1, 34)]
    if not np.isclose(fn(p, np.arange(len(p))[None])[0], _brute_independent(p)):
        raise AssertionError("point estimate differs from the brute force over pairs of different patients")
    idx = g.integers(0, len(p), (4, len(p)))
    for k in range(4):
        if not np.isclose(fn(p, idx[k:k + 1])[0], _brute_independent(p, idx[k])):
            raise AssertionError("a bootstrap draw pairs a patient's recording with its own copy")


def test_independent_days_bound_is_bh_over_pairs_of_different_patients():
    _assert_independent_days(mg.bh2_pairs_rate)


def test_an_independent_days_bound_that_pairs_a_recording_with_its_copy_is_caught():
    def broken(p, idx):
        bh2 = lambda x, y: min(x, y) <= 0.025 or max(x, y) <= 0.05
        return np.array([np.mean([bh2(p[r[a]], p[r[b]]) for a in range(len(r)) for b in range(a + 1, len(r))])
                         for r in np.atleast_2d(idx)])
    with pytest.raises(AssertionError):
        _assert_independent_days(broken)


def test_columbia_k2_bracket_reports_both_bounds():
    rows = []
    for i in range(40):
        for j, pv in enumerate((0.04, 0.03) if i < 5 else (0.5, 0.6)):
            if j == 1 and i % 2:
                continue
            rows.append({"null": "ncmd_bdsp", "pipe": "columbia", "variant": "as_coded", "patient": f"P{i}",
                         "uid": f"U{i}", "j": j, "seq": True, "overlap": False, "resp": "none",
                         "montage": "native", "status": "ok", "p": pv, "call": pv < 0.05})
    out = mg.columbia_k2_bracket(pd.DataFrame(rows), B=100)["ncmd_bdsp"]
    ind, same = out["independent_days_bound"], out["same_session_bound"]
    assert ind["label"].startswith("independent-days bound") and same["label"] == "same-session bound, subsample n = 20"
    assert ind["n_patients"] == 40 and same["n_patients"] == 20
    assert np.isclose(same["rate"], 3 / 20)
    p0 = np.array([0.04] * 5 + [0.5] * 35)
    assert np.isclose(ind["rate"], _brute_independent(p0))



def test_headline_notes_icare_is_dropped(tmp_path):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    _write(root, "ncmd_bdsp__x__native__none", [dict(r, key=f"k{i}") for i, r in
                                                 enumerate(_seq_rows("x", "v", {"A": [0.5]}))])
    head = mg.merge("dev", root=root, out=out, B=50)
    assert "8-min" in head["coverage_notes"]["icare"]


def test_analysis_merge_takes_columbia_from_113_and_checks_its_exclusions(tmp_path, monkeypatch):
    rn = importlib.import_module("114_benchmark")
    c = _c113()
    pd.DataFrame({"patient_id": c["patient"], "file": c["uid"] + ".mat"}).to_csv(tmp_path / "s.csv", index=False)
    c.drop(columns=["uid", "patient"]).to_parquet(tmp_path / "c.parquet")
    (tmp_path / "dev.json").write_text(json.dumps({"bdsp": [[1, 99, "UDEV"]], "pdoc": []}))
    monkeypatch.setattr(rn, "SAMPLE_CSV", str(tmp_path / "s.csv"))
    monkeypatch.setattr(rn, "COLUMBIA_ROWS", str(tmp_path / "c.parquet"))
    monkeypatch.setattr(rn.en, "DEV_FILE", str(tmp_path / "dev.json"))
    _touched_file(tmp_path, monkeypatch)
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    other = [dict(r, key=f"o{i}", split="analysis", dev=False, uid=u, patient=p) for i, (r, u, p) in
             enumerate(zip(_null_rows("mbvt", "v", [0] * 6, null="nstim0_bdsp"), c["uid"], c["patient"]))]
    _write(root, "nstim0_bdsp__mbvt__native__none", [r for r in other if r["uid"] != "U0001"])
    stale = [dict(r, key=f"c{i}", split="analysis", dev=False, uid=u, patient=p) for i, (r, u, p) in
             enumerate(zip(_null_rows("columbia", "as_coded", [1] * 6), c["uid"], c["patient"])) if u != "U0001"]
    _write(root, "ncmd_bdsp__columbia__native__none", stale)
    with pytest.raises(RuntimeError):
        mg.merge("analysis", root=root, out=out, B=50, spec_path=_locked(tmp_path))
    c[c["uid"] != "U0001"].drop(columns=["uid", "patient"]).to_parquet(tmp_path / "c.parquet")
    head = mg.merge("analysis", root=root, out=out, B=50, spec_path=_locked(tmp_path))
    res = json.load(open(os.path.join(out, "benchmark_ncmd_bdsp.json")))
    v = res["published"]["columbia"]["variants"]["as_coded"]
    assert v["n"] == 5 and v["calls"] == 1 and head["columbia_113"]["contained"]
    assert res["proposed_validity"]["columbia"]["n"] == 5



def _passive_rows():
    rows = []
    for pipe, calls in (("mbvt", [1, 1, 0, 1, 0, 1]), ("gibson16", [0, 0, 0, 1, 0, 0])):
        rows += [dict(r, design="published", family_tag="sensory_read_as_volition", contrast="oddball")
                 for r in _null_rows(pipe, "v", calls, null="npassive_erpcore_mmn")]
    for i in range(6):
        rows.append({"null": "npassive_erpcore_mmn", "pipe": "mbvt", "variant": None, "patient": f"P{i:04d}",
                     "uid": f"P{i:04d}", "j": 0, "seq": True, "overlap": False, "resp": "none", "montage": "native",
                     "status": "ok", "design": "instruction", "unit": "subject", "contrast": "oddball",
                     "rand_call": False, "split": "dev", "dev": True})
        for r in range(2):
            rows.append({**rows[-1], "uid": f"P{i:04d}_run-{r}", "unit": "run", "rand_call": r == 0 and i < 3})
    return pd.DataFrame(rows)


def test_npassive_published_calls_are_the_sensory_family(tmp_path):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    df = _passive_rows()
    _write(root, "npassive_erpcore_mmn__mbvt__native__none",
           [dict(r, key=f"k{i}") for i, r in enumerate(df.to_dict("records"))])
    mg.merge("dev", root=root, out=out, B=50)
    res = json.load(open(os.path.join(out, "benchmark_npassive_erpcore_mmn.json")))
    fam = res["sensory_read_as_volition"]
    assert set(fam) == {"mbvt", "gibson16"} and res["published"] == {}
    assert [fam[p]["inflated"] for p in sorted(fam)] == list(mg.holm([fam[p]["p_boot"] for p in sorted(fam)])
                                                             <= 0.05)
    head = json.load(open(os.path.join(out, "benchmark_headline.json")))
    assert set(head["null_sets"]["npassive_erpcore_mmn"]["sensory_read_as_volition"]) == {"mbvt", "gibson16"}


def test_npassive_instruction_test_is_a_construction_check_outside_validity():
    """"""
    df = _passive_rows()
    engine = pd.DataFrame(_rand_rows("mbvt", "nstim0_bdsp", 2, 30))
    v = mg.proposed_validity(pd.concat([df, engine], ignore_index=True))["mbvt"]
    assert v["n"] == 60 and set(v["point_by_null"]) == {"nstim0_bdsp"}
    assert "mbvt" not in mg.proposed_validity(df)
    cc = mg.construction_check(df)["npassive_erpcore_mmn"]["mbvt"]
    assert "construction" in cc["label"] and cc["per_subject"]["n"] == 6 and cc["per_subject"]["calls"] == 0
    assert cc["per_run"]["n"] == 12 and np.isclose(cc["per_run"]["rate"], 3 / 12)



def _cannot_rows():
    rng = np.random.default_rng(5)
    rows = ([dict(r, min_attainable_p=0.1, cannot_call=True) for r in _null_rows("A", "gf13_b3", [0] * 200)]
            + [dict(r, min_attainable_p=1 / 501, cannot_call=False)
               for r in _null_rows("A", "ok", rng.random(200) < 0.08)]
            + [dict(r, min_attainable_p=0.1, cannot_call=True) for r in _null_rows("B", "only", [0] * 200)])
    return pd.DataFrame(rows)


def _assert_validity_primary_can_call(fn):
    out = fn(_cannot_rows(), B=100)
    if out["A"]["validity_primary"] != "ok":
        raise AssertionError(f"validity primary {out['A']['validity_primary']!r} cannot call by construction")
    assert out["A"]["variants"]["gf13_b3"]["cannot_call"] and not out["A"]["variants"]["ok"]["cannot_call"]
    assert out["B"]["validity_primary"] is None and out["B"]["inflated"] is False and "p_holm" not in out["B"]


def test_a_variant_that_cannot_call_is_never_the_validity_primary():
    _assert_validity_primary_can_call(mg.published)


def test_a_merge_that_lets_a_cannot_call_variant_win_is_caught(monkeypatch):
    monkeypatch.setattr(mg, "cannot_call_of", lambda d, pipe, variant: False)
    with pytest.raises(AssertionError):
        _assert_validity_primary_can_call(mg.published)


def _assert_comparator_can_call(locate):
    df, _, _ = _sens_rows()
    df["cannot_call"] = df["variant"] == "v2"
    out = locate(df)["ncmd_bdsp"]["A"]["erd|1"]
    if out["comparator"] != "v1":
        raise AssertionError(f"comparator {out['comparator']!r} cannot call by construction")
    assert out["cannot_call"] == ["v2"]


def test_a_variant_that_cannot_call_is_never_the_comparator():
    _assert_comparator_can_call(mg.locate)


def test_a_locator_that_lets_a_cannot_call_variant_win_is_caught(monkeypatch):
    monkeypatch.setattr(mg, "cannot_call_of", lambda d, pipe, variant: False)
    with pytest.raises(AssertionError):
        _assert_comparator_can_call(mg.locate)


def test_cannot_call_is_derived_when_rows_do_not_carry_it():
    sv = importlib.import_module("bench.wrappers").short_variants(importlib.import_module("bench.pipe_cruse11"), "bdsp")
    d = pd.DataFrame(_null_rows("cruse11", "gf13_b3_hilbert", [0, 0]) + _null_rows("cruse11", "gf13_b4_hilbert", [0, 0]))
    assert "gf13_b3_hilbert" in sv
    assert mg.cannot_call_of(d, "cruse11", "gf13_b3_hilbert") and not mg.cannot_call_of(d, "cruse11", "gf13_b4_hilbert")
    assert np.isclose(mg.rows_from_113(_c113())["min_attainable_p"].iloc[0], 1 / 501)


def test_evaluation_labels_the_columbia_ers_subsample():
    df, _, unc = _sens_rows()
    ev = df["resp"] == "eval"
    df.loc[ev, "eval_set"] = "uncorrected_subsample"
    df = df[~ev | df["patient"].isin(unc)]
    r = mg.evaluate(df, B=50)["ncmd_bdsp"]["A"]["erd|1"]
    assert r["eval_set"] == "uncorrected_subsample" and r["all"]["n_patients"] == 100



def test_columbia_uncorrected_arm_off_its_point_is_explicitly_empty():
    df, _, _ = _sens_rows()
    df["pipe"] = "columbia"
    ev = df["resp"] == "eval"
    half = df[ev].assign(prob=0.5, perm_p=None, perm_call=None)
    r = mg.evaluate(pd.concat([df, half], ignore_index=True), B=50)["ncmd_bdsp"]["columbia"]
    assert r["erd|1"]["uncorrected"]["n_patients"] == 100 and "note" not in r["erd|1"]["uncorrected"]
    off = r["erd|0.5"]["uncorrected"]
    assert off["n_patients"] == 0 and "ERD p_resp = 1 only" in off["note"]
    assert "ERD p_resp = 1 only" in r["erd|0.5"]["p_based"]["note"]


def test_headline_notes_the_pdoc_arm(tmp_path):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    _write(root, "ncmd_bdsp__x__native__none", [dict(r, key=f"k{i}") for i, r in
                                                 enumerate(_seq_rows("x", "v", {"A": [0.5]}))])
    head = mg.merge("dev", root=root, out=out, B=50)
    note = head["coverage_notes"]["pdoc"]
    assert "N/A on pDoC" in note["columbia"] and "nulls only" in note["arm"]
    assert note["channel_sets"]["mbmi"] == "F3,F4,C3,C4" and note["n_a_variants"]["gibson16"]
    assert all(v.startswith("ssep") for v in note["n_a_variants"]["gibson16"])



def _section_value(res, s):
    v = res
    for k in s.split("."):
        v = (v or {}).get(k)
    return v


def _assert_sections_explained(res, sections):
    why = res["empty_sections"]
    for s in sections:
        if mg.is_empty(_section_value(res, s)):
            assert isinstance(why.get(s), str) and why[s] and not why[s].startswith("UNEXPECTED"), (s, why.get(s))
        else:
            assert s not in why, s


def test_every_empty_section_says_why(tmp_path):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    _write(root, "ncmd_bdsp__cornell__native__none",
           [dict(r, key=f"k{i}") for i, r in enumerate(_null_rows("cornell", "v", [0, 1, 0, 0],
                                                                       rand_call=False, rand_variant="v"))])
    _write(root, "npassive_erpcore_mmn__mbvt__native__none",
           [dict(r, key=f"p{i}") for i, r in enumerate(_passive_rows().to_dict("records"))])
    mg.merge("dev", root=root, out=out, B=50)
    for n in ("ncmd_bdsp", "npassive_erpcore_mmn"):
        res = json.load(open(os.path.join(out, f"benchmark_{n}.json")))
        _assert_sections_explained(res, mg.SECTIONS)
    head = json.load(open(os.path.join(out, "benchmark_headline.json")))
    _assert_sections_explained(head, mg.HEAD_SECTIONS)
    assert "112/113" in json.load(open(os.path.join(out, "benchmark_ncmd_bdsp.json")))["empty_sections"]["columbia_k2"]


def test_a_section_empty_for_no_known_reason_is_flagged():
    d = pd.DataFrame(_null_rows("cornell", "v", [0, 1]))
    res = {s: {} for s in mg.SECTIONS if "." not in s}
    res["sensitivity"] = {"locator": {}, "evaluation": {}}
    why = mg.empty_reasons("ncmd_bdsp", res, d, "dev")
    assert why["published"].startswith("UNEXPECTED") and why["coverage"].startswith("UNEXPECTED")
    assert not why["chancount"].startswith("UNEXPECTED")


def test_headline_notes_kit8_not_applicable(tmp_path):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    _write(root, "ncmd_bdsp__x__native__none", [dict(r, key=f"k{i}") for i, r in
                                                 enumerate(_seq_rows("x", "v", {"A": [0.5]}))])
    head = mg.merge("dev", root=root, out=out, B=50)
    assert set(head["coverage_notes"]["kit8"]) == {"columbia", "dubois"}



def _p_null(n, exact, pile):
    """n null rows over 200 patients: `exact` (value, count) uncensored, the rest stopped at the pile."""
    rows, i = [], 0
    for v, k in exact:
        rows += [{"p": v, "p_censored": False}] * k
    rows += [{"p": pile, "p_censored": True}] * (n - len(rows))
    return pd.DataFrame([dict(r, patient=f"P{i:04d}", stratum="normal_awake", transitioning=False)
                         for i, r in enumerate(rows)])


def _p_resp(ps):
    return pd.DataFrame([{"patient": f"R{i:04d}", "perm_p": p, "perm_p_censored": c, "stratum": "normal_awake",
                          "transitioning": False} for i, (p, c) in enumerate(ps)])


def test_p_based_threshold_on_the_cruse12p_pile_is_not_estimable():
    pile = 26 / 1001
    null = _p_null(400, [(0.001, 8)], pile)
    resp = _p_resp([(pile, True)] * 50 + [(0.0005, False)] * 50)
    out = mg.crossfit_calls(null, resp, stat="p", resp_stat="perm_p", q=mg.ALPHA, upper=False)
    assert out["corr_call"].isna().all() and out["thr_not_estimable"].all()


def test_p_based_threshold_below_the_edlow17_pile_calls_strictly_below_it():
    pile = 25 / 500
    null = _p_null(400, [(0.02, 40)], pile)
    resp = _p_resp([(pile, True), (0.02, False), (0.01, False)] * 20)
    out = mg.crossfit_calls(null, resp, stat="p", resp_stat="perm_p", q=mg.ALPHA, upper=False)
    calls = dict(zip(out["perm_p"], out["corr_call"]))
    assert not out["thr_not_estimable"].any() and np.isclose(out["thr"].astype(float), 0.02).all()
    assert calls[pile] is False and calls[0.02] is False and calls[0.01] is True


def test_evaluation_reports_a_not_estimable_p_based_correction():
    df, _, _ = _sens_rows()
    nul = (df["resp"] == "none")
    df.loc[nul, "p"] = 26 / 1001
    df.loc[nul, "p_censored"] = True
    r = mg.evaluate(df, B=50)["ncmd_bdsp"]["A"]["erd|1"]["p_based"]
    assert "not estimable" in r["note"] and "flag_p_more_sensitive" not in r


def test_113_rows_carry_censoring_of_stopped_p_values():
    c = _c113()
    rows = mg.rows_from_113(c)
    assert (rows["p_censored"].to_numpy() == c["i_stopped"].astype(bool).to_numpy()).all()



def test_validity_primary_is_selected_on_the_common_recordings():
    short = _null_rows("cruse11", "short", [1, 1] + [0] * 18)
    native = _null_rows("cruse11", "native", [1, 0, 0, 0, 0])
    r = mg.published(pd.DataFrame(short + native), B=50)["cruse11"]
    assert r["validity_primary"] == "native"
    sel = r["selection"]
    assert sel["n_common"] == 5 and np.isclose(sel["rates_common"]["short"], 0.4)
    assert r["variants"]["short"]["n"] == 20 and r["variants"]["native"]["n"] == 5



def _headline(tmp_path, rows, sub="ncmd_bdsp__x__native__none"):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    _write(root, sub, [dict(r, key=f"k{i}") for i, r in enumerate(rows)])
    return mg.merge("dev", root=root, out=out, B=50)


def test_headline_columbia_reports_the_k2_bracket_and_labels_the_same_session_value(tmp_path):
    g = np.random.default_rng(3)
    pats = {f"C{i:03d}": [float(v) for v in g.uniform(0, 1, 2)] for i in range(30)}
    pats.update({f"S{i:03d}": [float(g.uniform())] for i in range(10)})
    head = _headline(tmp_path, _seq_rows("columbia", "as_coded", pats))
    col = head["null_sets"]["ncmd_bdsp"]["own_rule"]["columbia"]
    assert col["k"] == 2 and col["label"] == "same-session bound, subsample n = 30"
    br = col["k2_bracket"]
    assert br["independent_days_bound"]["label"].startswith("independent-days bound")
    assert br["same_session_bound"]["label"] == "same-session bound, subsample n = 30"
    assert br["independent_days_bound"]["n_patients"] == 40


def test_headline_says_when_k_falls_back(tmp_path):
    head = _headline(tmp_path, _seq_rows("dubois", "assessment_random_phase1", {f"D{i}": [0.5] for i in range(8)}))
    d = head["null_sets"]["ncmd_bdsp"]["own_rule"]["dubois"]
    assert d["k"] == 1 and "k = 12 not available" in d["note"]



def _group_rows(seed=0):
    """_sens_rows plus variant v3 in its own timing group g2 (weaker, never reaching 50% corrected sensitivity), with
    v1 / v2 in g1; evaluation rows for both groups (g1 the pipeline group)."""
    df, loc, unc = _sens_rows(seed=seed)
    df["group"] = "g1"
    g = np.random.default_rng(seed + 1)
    v3 = df[(df["variant"] == "v1") & (df["resp"].isin(["none", "grid"]))].copy()
    v3["variant"], v3["group"] = "v3", "g2"
    v3.loc[v3["resp"] == "none", "size_stat"] = g.normal(size=int((v3["resp"] == "none").sum()))
    gr = v3["resp"] == "grid"
    v3.loc[gr, "size_stat"] = g.normal(0.05, 1.0, int(gr.sum()))
    ev = df[df["resp"] == "eval"].copy()
    df.loc[df["resp"] == "eval", "pipeline_group"] = True
    df.loc[df["resp"] == "eval", "reached"] = True
    e3 = ev.assign(variant="v3", group="g2", effect=2.0, pipeline_group=False, reached=False,
                   size_stat=g.normal(0.1, 1.0, len(ev)))
    return pd.concat([df, v3, e3], ignore_index=True), loc, unc


def test_locator_picks_a_comparator_per_timing_group_and_the_pipeline_group():
    df, _, _ = _group_rows()
    r = mg.locate(df)["ncmd_bdsp"]["A"]["erd|1"]
    assert set(r["groups"]) == {"g1", "g2"} and r["pipeline_group"] == "g1"
    assert r["groups"]["g1"]["comparator"] == "v2" and r["groups"]["g2"]["comparator"] == "v3"
    assert r["comparator"] == "v2" and r["evaluation_effect"] == r["groups"]["g1"]["evaluation_effect"]


def test_a_group_that_never_reaches_50_percent_is_evaluated_at_the_largest_grid_effect():
    df, _, _ = _group_rows()
    g2 = mg.locate(df)["ncmd_bdsp"]["A"]["erd|1"]["groups"]["g2"]
    assert g2["reached"] is False and g2["evaluation_effect"] == 2.0 and g2["label"] == "50% not reached"
    pts = mg.eval_points_of(mg.locate(df))["ncmd_bdsp"]["A"]["erd|1"]
    assert pts["pipeline_group"] == "g1" and pts["groups"]["g2"] == {"effect": 2.0, "comparator": "v3",
                                                                     "reached": False}


def test_evaluation_reports_every_group_and_the_pipeline_groups_ni():
    df, _, _ = _group_rows()
    r = mg.evaluate(df, B=50)["ncmd_bdsp"]["A"]["erd|1"]
    assert set(r["groups"]) == {"g1", "g2"} and r["pipeline_group"] == "g1"
    assert r["comparator"] == "v2" and r["groups"]["g2"]["comparator"] == "v3"
    assert r["groups"]["g2"]["label"] == "50% not reached" and "label" not in r["groups"]["g1"]
    assert r["outside_locator"] == r["groups"]["g1"]["outside_locator"]


def test_validity_families_split_by_timing_group():
    rows = _rand_rows("A", "ncmd_bdsp", 2, 40)
    rows += [dict(r, rand_variant="g2", rand_call=False) for r in _rand_rows("A", "ncmd_bdsp", 2, 40)]
    for r in rows[:80]:
        r["rand_variant"] = "g1"
    v = mg.proposed_validity(pd.DataFrame(rows))["A"]
    assert v["n"] == 160 and set(v["point_by_group"]) == {"g1", "g2"}
    assert {t["unit"] for t in v["families"]["null_set"]} == {"ncmd_bdsp|g1", "ncmd_bdsp|g2"}



def _freeze_rows(drop_one=False):
    df, loc, _ = _group_rows()
    df["family"], df["stat"] = "cmd", df["size_stat"]
    gr = df["resp"] == "grid"
    if drop_one:
        first = sorted(loc)[0]
        df = df[~(gr & (df["patient"] == first) & (df["effect"] == 2.0) & (df["variant"] == "v2"))]
    return df, loc


def test_eval_points_are_written_only_by_an_explicit_freeze_on_a_complete_locator(tmp_path, monkeypatch):
    out = str(tmp_path / "t")
    monkeypatch.setattr(mg, "grid_for", lambda split, spec_path=None: {"erd": [0.25, 0.5, 1.0, 2.0], "ers": []})
    monkeypatch.setattr(mg, "LOCATOR_PROBS", (1.0,))
    fn = os.path.join(out, "_eval_points.json")
    roots = {}
    for name, drop in (("partial", True), ("complete", False)):
        df, loc = _freeze_rows(drop_one=drop)
        roots[name] = str(tmp_path / name)
        _write(roots[name], "ncmd_bdsp__A__native__grid",
               [dict(r, key=f"k{i}") for i, r in enumerate(df.replace({np.nan: None}).to_dict("records"))])
    mg.merge("dev", root=roots["complete"], out=out, B=20)
    assert not os.path.exists(fn)
    with pytest.raises(RuntimeError, match="locator incomplete"):
        mg.merge("dev", root=roots["partial"], out=out, B=20, freeze=True, locator_sets={"ncmd_bdsp": loc})
    assert not os.path.exists(fn)
    head = mg.merge("dev", root=roots["complete"], out=out, B=20, freeze=True, locator_sets={"ncmd_bdsp": loc})
    assert os.path.exists(fn) and head["eval_points"]["frozen"] is True
    assert json.load(open(fn))["ncmd_bdsp"]["A"]["erd|1"]["pipeline_group"] == "g1"


def test_locator_completeness_counts_missing_and_partial_patients():
    df, loc = _freeze_rows(drop_one=True)
    grid = {"erd": [0.25, 0.5, 1.0, 2.0], "ers": []}
    c = mg.locator_completeness(df, {"ncmd_bdsp": loc | {"NEW"}}, grid, probs=(1.0,))["ncmd_bdsp|A"]
    assert c["complete"] is False and c["missing"] == ["NEW"] and c["incomplete"] == [sorted(loc)[0]]
    assert c["n_expected"] == 61 and c["n_complete"] == 59


def test_headline_ni_names_the_pipeline_group_and_every_groups_decision(tmp_path):
    df, _, _ = _group_rows()
    df["stat"] = df["size_stat"]
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    _write(root, "ncmd_bdsp__A__native__eval", [dict(r, key=f"k{i}") for i, r in
                                                 enumerate(df.replace({np.nan: None}).to_dict("records"))])
    head = mg.merge("dev", root=root, out=out, B=20)
    ni = head["null_sets"]["ncmd_bdsp"]["ni"]["A"]["erd|1"]
    assert ni["pipeline_group"] == "g1" and ni["comparator"] == "v2" and set(ni["groups_noninferior"]) == {"g1", "g2"}



def test_113_check_covers_both_sources():
    c = _c113().assign(source=["design"] * 4 + ["extra"] * 2)
    bench = pd.DataFrame({"null": "nstim0_bdsp", "pipe": "mbvt", "uid": list(c["uid"][:5]), "status": "ok"})
    r = mg.check_113(c, bench, dev=set(), touched=set())
    assert r["n_113_by_source"] == {"design": 4, "extra": 2}
    assert not r["contained"] and r["missing_by_source"] == {"nstim0_bdsp|mbvt": {"extra": 1}}
    with pytest.raises(RuntimeError):
        mg.check_113(c, bench, dev=set(), touched={"U0005"})
    rows = mg.rows_from_113(c)
    assert list(rows["source_113"]) == list(c["source"]) and (rows["source"] == "bdsp").all()



def test_no_usable_trial_rows_leave_the_denominators_and_are_counted_by_stratum():
    ok = _null_rows("cornell", "v", [1, 0, 1, 0])
    na = [dict(r, patient=f"Q{i}", uid=f"Q{i}", status="na: no_usable_trials", call=None, stat=None, size_stat=None,
               stratum="burst_suppression" if i < 3 else "normal_awake")
          for i, r in enumerate(_null_rows("cornell", "v", [0] * 5))]
    df = pd.DataFrame(ok + na)
    pub = mg.published(df, B=50)["cornell"]
    r = pub["variants"]["v"]
    assert r["n"] == 4 and r["rate"] == 0.5 and pub["n_na_no_usable_trials"] == 5
    cov = mg.coverage(df)["ncmd_bdsp"]["cornell"]
    assert cov["na_no_usable_trials"] == {"recordings": 5, "by_stratum": {"burst_suppression": 3, "normal_awake": 2}}
    rr = _rand_rows("cornell", "ncmd_bdsp", 1, 4)
    rr += [dict(x, uid=f"N{i}", patient=f"N{i}", status="na: no_usable_trials", rand_call=None)
           for i, x in enumerate(_rand_rows("cornell", "ncmd_bdsp", 1, 3))]
    assert mg.proposed_validity(pd.DataFrame(rr))["cornell"]["n"] == 4
    assert mg.coverage(pd.DataFrame(rr))["ncmd_bdsp"]["cornell"]["na_no_usable_trials"]["recordings"] == 3



def test_headline_any_approach_excludes_covert_processing_pipelines(tmp_path):
    root, out = str(tmp_path / "sh"), str(tmp_path / "t")
    ch = _null_rows("chennu13", "c", [1, 0, 0, 0], null="nstim0_bdsp")
    so = _null_rows("sokoliuk21", "s", [0, 1, 1, 0], null="nstim0_bdsp")
    _write(root, "nstim0_bdsp__chennu13__native__none", [dict(r, key=f"c{i}") for i, r in enumerate(ch)])
    _write(root, "nstim0_bdsp__sokoliuk21__native__none", [dict(r, key=f"s{i}") for i, r in enumerate(so)])
    head = mg.merge("dev", root=root, out=out, B=50)
    cmd = head["any_approach"]["bdsp"]
    assert cmd["pipelines"] == ["chennu13"] and cmd["rate"] == 0.25
    ccp = head["any_approach_incl_covert_processing"]["bdsp"]
    assert ccp["pipelines"] == ["chennu13", "sokoliuk21"] and ccp["rate"] == 0.75
    assert "not command-following" in ccp["label"]



def test_stratum_threshold_needs_100_null_draws_in_the_calibrating_half():
    g = np.random.default_rng(7)
    big = _null_rows("x", "v", [0] * 300, stat=g.normal(0, 1, 300), stratum="normal_awake")
    small = [dict(r, patient=f"S{i:04d}", uid=f"US{i:04d}") for i, r in
             enumerate(_null_rows("x", "v", [0] * 120, stat=g.normal(0, 1, 120), stratum="sleep"))]
    null = pd.DataFrame(big + small)
    resp = null.assign(model="erd", prob=1.0, effect=0.4, resp="grid")
    out = mg.crossfit_calls(null, resp)
    halves = null.assign(h=null.patient.map(_half)).groupby(["stratum", "h"]).size()
    assert (halves["normal_awake"] >= 100).all() and (halves["sleep"] < 100).all()
    assert not out[out.stratum == "normal_awake"].thr_pooled.any()
    assert out[out.stratum == "sleep"].thr_pooled.all()


def _cov_rows(pipe, variant, calls, na=0, null="nstim0_bdsp"):
    rows = _null_rows(pipe, variant, calls, null=null)
    rows += [dict(r, patient=f"N{i:04d}", uid=f"UN{i:04d}", status=mg.NA_NO_USABLE, call=None)
             for i, r in enumerate(_null_rows(pipe, variant, [0] * na, null=null))]
    return rows


def test_low_coverage_variant_is_never_validity_primary_but_is_reported():
    rows = pd.DataFrame(_cov_rows("chennu13", "a", [0] * 10, na=15) + _cov_rows("chennu13", "b", [1] + [0] * 24))
    out = mg.published(rows, B=50)["chennu13"]
    assert out["validity_primary"] == "b"
    assert out["variants"]["a"]["low_coverage"] and abs(out["variants"]["a"]["coverage"] - 0.4) < 1e-9
    assert not out["variants"]["b"]["low_coverage"] and out["low_coverage"] == ["a"]


def test_coverage_counts_unusable_and_error_rows_but_not_no_fit():
    rows = _cov_rows("x", "v", [0] * 7, na=2)
    rows += [dict(rows[0], patient="E1", uid="UE1", status="error: boom"),
             dict(rows[0], patient="F1", uid="UF1", status="no_fit")]
    cov, n = mg.coverage_of(pd.DataFrame(rows), "nstim0_bdsp", "x", "v")
    assert n == 10 and abs(cov - 0.7) < 1e-9


def test_a_low_coverage_variant_is_never_the_comparator():
    df, _, _ = _sens_rows()
    nul_v2 = (df["resp"] == "none") & (df["variant"] == "v2")
    idx = df[nul_v2].index[::2]
    df.loc[idx, "status"] = mg.NA_NO_USABLE
    out = mg.locate(df)["ncmd_bdsp"]["A"]["erd|1"]
    assert out["comparator"] == "v1" and out["low_coverage"] == ["v2"]


def test_a_channel_count_null_set_explains_its_empty_published_and_composites():
    """chancount_* rows carry montage n<k> (never native): they fill the chancount section, so published and
    composites are empty by design. An empty chancount section on such a set is still flagged."""
    d = pd.DataFrame(_null_rows("chennu13", "v", [0, 1])).assign(null="chancount_kojimaB", montage="n19")
    res = {s: {} for s in mg.SECTIONS if "." not in s}
    res["sensitivity"] = {"locator": {}, "evaluation": {}}
    why = mg.empty_reasons("chancount_kojimaB", res, d, "dev")
    for s in ("published", "composites"):
        assert not why[s].startswith("UNEXPECTED") and "chancount" in why[s], (s, why[s])
    assert why["chancount"].startswith("UNEXPECTED")
    assert mg.empty_reasons("ncmd_bdsp", res, d.assign(null="ncmd_bdsp"), "dev")["published"].startswith("UNEXPECTED")
