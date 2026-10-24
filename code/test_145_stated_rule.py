import importlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

sr = importlib.import_module("145_stated_rule")
mg = sr.mg



def _ax(*classes):
    return {f"a{i}": {"value": "x", "class": c, "quote": "q", "source": "s"} for i, c in enumerate(classes)}


def _entry(eligible, *classes, short=False, axes_only=None):
    e = {"axes": _ax(*classes), "stated_rule_eligible": eligible, "cannot_call": False, "low_coverage": None,
         "note": "", "shortened_to_fit": short}
    if axes_only is not None:
        e["stated_rule_eligible_variant_axes_only"] = axes_only
    return e


def _classification():
    return {
        "edlow17": {"v_ok": _entry(True, "STATED", "NOT_STATED"), "v_ok2": _entry(True, "STATED"),
                    "v_bad": _entry(False, "STATED", "CONTRADICTS")},
        "chennu13": {"b20": _entry(True, "STATED"), "b12": _entry(False, "STATED", "SHORTENED_STRUCTURAL", short=True)},
        "mbvt": {"median": _entry(True, "STATED"), "full": _entry(False, "STATED", "CONTRADICTS"),
                 "max": _entry(False, "STATED", "CONTRADICTS")},
        "dubois": {"c3_b_x": _entry(False, "STATED", "NOT_STATED", axes_only=True)},
        "sensitivity_eligible": {"sens": {"pipeline": "mbvt", "variants": ["full", "max", "median"],
                                          "reclassified_axes": ["summary"]}},
        "shortened_to_fit": {"class": "SHORTENED_STRUCTURAL", "variants": {"chennu13": ["b12"]}}}


def _with_summary_axis(cl):
    for v in ("full", "max"):
        cl["mbvt"][v]["axes"] = {"summary": {"class": "CONTRADICTS"}, "thr": {"class": "STATED"}}
    cl["mbvt"]["median"]["axes"] = {"summary": {"class": "STATED"}, "thr": {"class": "STATED"}}
    return cl


def test_classification_sets_split_headline_shortened_and_sensitivity_and_ignore_axes_only_flags():
    s = sr.classification_sets(_with_summary_axis(_classification()))
    assert s["headline"] == {"edlow17": ["v_ok", "v_ok2"], "chennu13": ["b20"], "mbvt": ["median"]}
    assert s["shortened"] == {"chennu13": ["b12"]}
    assert s["sensitivity"] == {"sens": {"pipe": "mbvt", "variants": ["full", "max", "median"],
                                         "reclassified_axes": ["summary"], "status": "separate"}}
    assert "axes_only" not in s and s["shortened_flag_but_contradicting"] == {}
    assert all("c3_b_x" not in v for v in s["headline"].values())


def test_a_shortened_flag_on_a_contradicting_variant_gives_no_row_but_an_unlisted_clean_one_is_refused():
    cl = _with_summary_axis(_classification())
    cl["chennu13"]["b12x"] = _entry(False, "SHORTENED_STRUCTURAL", "CONTRADICTS", short=True)
    s = sr.classification_sets(cl)
    assert s["shortened"] == {"chennu13": ["b12"]} and s["shortened_flag_but_contradicting"] == {"chennu13": ["b12x"]}
    cl["chennu13"]["b12y"] = _entry(False, "SHORTENED_STRUCTURAL", "STATED", short=True)
    with pytest.raises(RuntimeError, match="shortened_to_fit list"):
        sr.classification_sets(cl)


def test_a_contradicting_variant_in_the_eligible_set_is_refused():
    cl = _with_summary_axis(_classification())
    cl["edlow17"]["v_bad"]["stated_rule_eligible"] = True
    with pytest.raises(RuntimeError, match="CONTRADICTS"):
        sr.classification_sets(cl)


def test_a_shortened_variant_in_the_headline_is_refused():
    cl = _with_summary_axis(_classification())
    cl["chennu13"]["b12"]["stated_rule_eligible"] = True
    with pytest.raises(RuntimeError, match="shortened"):
        sr.classification_sets(cl)
    cl = _with_summary_axis(_classification())
    cl["shortened_to_fit"]["variants"]["chennu13"] = ["b12", "b20"]
    with pytest.raises(RuntimeError):
        sr.classification_sets(cl)


def test_sensitivity_set_may_differ_from_the_primary_only_on_its_reclassified_axis():
    cl = _with_summary_axis(_classification())
    cl["mbvt"]["full"]["axes"]["thr"] = {"class": "CONTRADICTS"}
    with pytest.raises(RuntimeError, match="sensitivity"):
        sr.classification_sets(cl)


def test_an_unknown_axis_class_is_refused():
    cl = _with_summary_axis(_classification())
    cl["edlow17"]["v_ok"]["axes"]["a0"]["class"] = "PROBABLY_STATED"
    with pytest.raises(RuntimeError, match="unknown"):
        sr.classification_sets(cl)


def test_the_real_classification_files_pass_the_guards_if_present():
    if not all(os.path.exists(p) for p in sr.FILES):
        pytest.skip("classification files not on disk")
    s = sr.classification_sets(sr.load_classification())
    for pipe, vs in s["headline"].items():
        assert not set(vs) & set(s["shortened"].get(pipe, []))



def _r(patient, variant, call, uid=None, j=0, status="ok", null="ncmd_bdsp", pipe="edlow17", **kw):
    r = {"key": f"{uid or patient + '_0'}|{j}|{variant}", "null": null, "pipe": pipe, "uid": uid or f"{patient}_0",
         "patient": patient, "variant": variant, "j": j, "status": status, "call": call, "resp": "none",
         "montage": "native", "seq": True, "overlap": False, "as_coded": False, "min_attainable_p": 0.002,
         "p": 0.01 if call else 0.5, "stat": 0.0}
    r.update(kw)
    return r


def test_one_row_per_patient_keeps_the_first_recording_for_every_variant_of_a_pipeline():
    d = pd.DataFrame([_r("a", "v1", True, uid="a_1"), _r("a", "v2", False, uid="a_1"), _r("a", "v1", False, uid="a_0",
                                                                                         status="no_fit"),
                      _r("a", "v2", True, uid="a_0"), _r("b", "v1", True)])
    out, dropped = sr.one_per_patient(d)
    assert sorted(zip(out["patient"], out["variant"], out["uid"])) == [("a", "v1", "a_0"), ("a", "v2", "a_0"),
                                                                       ("b", "v1", "b_0")]
    assert dropped == 2


def test_usable_drops_cannot_call_and_low_coverage_exactly_as_115():
    rows = [_r(f"p{i}", "ok", False) for i in range(10)]
    rows += [_r(f"p{i}", "cc", False, min_attainable_p=0.1) for i in range(10)]
    rows += [_r(f"p{i}", "lowcov", False, status="ok" if i < 6 else "na: no_usable_trials") for i in range(10)]
    df = pd.DataFrame(rows)
    ok, rec = sr.usable(df, "ncmd_bdsp", "edlow17", ["ok", "cc", "lowcov"])
    assert ok == ["ok"]
    assert rec["cc"]["cannot_call"] and not rec["lowcov"]["cannot_call"]
    assert rec["lowcov"]["coverage"] == pytest.approx(0.6) and rec["lowcov"]["low_coverage"]
    assert rec["lowcov"]["coverage"] == mg.coverage_of(df, "ncmd_bdsp", "edlow17", "lowcov")[0]


def test_stated_rule_reading_is_115s_selection_on_the_eligible_set_only():
    rows = [_r(f"p{i}", "liberal", i < 5) for i in range(10)] + [_r(f"p{i}", "strict", i < 1) for i in range(10)]
    rows += [_r(f"p{i}", "contradicting", False) for i in range(10)]
    d = pd.DataFrame(rows)
    v, sel = sr.select(d, ["liberal", "strict"], {})
    assert v == "strict" and sel["n_common"] == 10
    assert sr.select(d, ["liberal", "strict", "contradicting"], {})[0] == "contradicting"
    assert sr.select(d, [], {})[0] is None


def test_rate_is_exact_on_one_row_per_patient():
    d = pd.DataFrame([_r(f"p{i}", "v", i < 3) for i in range(40)])
    e = sr.exact_rate(d)
    assert (e["k"], e["n"], e["n_patients"]) == (3, 40, 40)
    assert e["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(3, 40))
    with pytest.raises(RuntimeError):
        sr.exact_rate(pd.DataFrame([_r("p", "v", True), _r("p", "v", False, uid="p_1")]))
    with pytest.raises(RuntimeError, match="missing"):
        sr.exact_rate(pd.DataFrame([_r("p", "v", True), _r("q", "v", None)]))


def test_call_paper_rows_refuse_an_ok_row_without_call_paper():
    d = pd.DataFrame([_r("c0", "as_coded", True, pipe="columbia", x_call_paper=False),
                      _r("c1", "as_coded", True, pipe="columbia", x_call_paper=None)])
    with pytest.raises(RuntimeError, match="missing"):
        sr.call_paper_rows(d)
    ok = sr.call_paper_rows(d.iloc[:1])
    assert list(ok["call"]) == [False] and list(ok["variant"]) == [sr.CALL_PAPER]


def test_holm_within_a_bdsp_null_set_and_no_inflated_label_on_pdoc():
    e = {"a": {"null": "ncmd_bdsp", "stated_rule": {"p_greater_nominal": 0.01}},
         "b": {"null": "ncmd_bdsp", "stated_rule": {"p_greater_nominal": 0.04}},
         "c": {"null": "ncmd_pdoc", "stated_rule": {"p_greater_nominal": 0.001}}}
    sr.add_holm(e)
    assert e["a"]["stated_rule"]["p_holm"] == pytest.approx(0.02) and e["a"]["stated_rule"]["inflated"]
    assert e["b"]["stated_rule"]["p_holm"] == pytest.approx(0.04) and e["b"]["stated_rule"]["inflated"]
    assert "p_holm" not in e["c"]["stated_rule"] and "inflated" not in e["c"]["stated_rule"]



def _halves_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pats = [f"q{i}" for i in range(n)]
    h = mg._half(pats)
    rows = []
    for p, hh in zip(pats, h):
        rows.append(_r(p, "x", bool(rng.random() < (0.05 if hh == 0 else 0.4))))
        rows.append(_r(p, "y", bool(rng.random() < (0.4 if hh == 0 else 0.05))))
    return pd.DataFrame(rows), pats, h


def test_crossfit_selects_on_one_half_and_estimates_on_the_other():
    d, pats, h = _halves_frame()
    cf = sr.crossfit(d, ["x", "y"], {}, B=50, seed=[1])
    for part in cf["halves"]:
        sel = d[d["patient"].isin([p for p, hh in zip(pats, h) if hh == part["select_half"]])]
        est = d[d["patient"].isin([p for p, hh in zip(pats, h) if hh != part["select_half"]])]
        own = {v: {"rate": float(sel[sel["variant"] == v]["call"].mean())} for v in ("x", "y")}
        assert part["variant"] == mg._select_primary(sel, own, {})[0]
        e = est[est["variant"] == part["variant"]]
        assert (part["k"], part["n"]) == (int(e["call"].sum()), len(e))
    assert cf["average"] == pytest.approx(np.mean([x["rate"] for x in cf["halves"]]))
    assert cf["halves"][0]["variant"] == "x" and cf["halves"][1]["variant"] == "y"
    assert cf["ci95"][0] <= cf["average"] <= cf["ci95"][1]


def test_crossfit_leak_across_halves_is_refused(monkeypatch):
    d, _, _ = _halves_frame()
    monkeypatch.setattr(sr, "_estimation_mask", lambda half, h: np.ones(len(half), bool))
    with pytest.raises(RuntimeError, match="leak"):
        sr.crossfit(d, ["x", "y"], {}, B=5, seed=[1])


def test_crossfit_refuses_a_patient_with_rows_in_both_halves(monkeypatch):
    d, _, _ = _halves_frame()
    monkeypatch.setattr(sr, "_halves_of", lambda patients: np.arange(len(patients)) % 2)
    with pytest.raises(RuntimeError, match="both halves"):
        sr.crossfit(d, ["x", "y"], {}, B=5, seed=[1])


def test_weighted_selection_with_unit_weights_equals_115s_selection():
    rng = np.random.default_rng(3)
    for t in range(60):
        n, names = int(rng.integers(3, 12)), ["a", "b", "c"]
        rows = []
        for i in range(n):
            for v in names:
                if rng.random() < 0.85:
                    rows.append(_r(f"p{i}", v, bool(rng.random() < 0.3), as_coded=(v == "c")))
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        coded = {v: bool(d[d["variant"] == v]["as_coded"].any()) for v in names}
        ok = [v for v in names if (d["variant"] == v).any()]
        own = {v: {"rate": float(d[d["variant"] == v]["call"].astype(float).mean())} for v in ok}
        want = mg._select_primary(d, own, coded)[0]
        C, pats = sr._matrix(d, ok)
        got = sr._select_weighted(C, np.ones(len(pats)), ok, coded)
        assert got == want, (t, got, want)
        keep = rng.random(len(pats)) < 0.5
        sub = d[d["patient"].isin([p for p, k in zip(pats, keep) if k])]
        if sub.empty:
            continue
        ok_s = [v for v in ok if (sub["variant"] == v).any()]
        own = {v: {"rate": float(sub[sub["variant"] == v]["call"].astype(float).mean())} for v in ok_s}
        assert sr._select_weighted(C, keep.astype(float), ok, coded) == mg._select_primary(sub, own, coded)[0]



def test_paper_pairs_rate_equals_115s_bh2_when_every_auc_is_above_half():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 0.12, 30)
    idx = np.vstack([np.arange(30), rng.integers(0, 30, 30), rng.integers(0, 30, 30)])
    assert sr.paper_pairs_rate(p, np.full(30, 0.7), idx) == pytest.approx(mg.bh2_pairs_rate(p, idx))


def test_paper_rule_needs_auc_above_half_on_a_rejected_recording():
    o = sr.bh2_paper(np.array([0.01, 0.01, 0.04, 0.04, 0.2]), np.array([0.4, 0.6, 0.6, 0.4, 0.9]),
                     np.array([0.3, 0.3, 0.04, 0.04, 0.2]), np.array([0.9, 0.4, 0.4, 0.4, 0.9]))
    assert list(o) == [False, True, True, False, False]
    p = np.array([0.01, 0.2])
    assert sr.paper_pairs_rate(p, np.array([0.4, 0.9]), np.arange(2)[None])[0] == 0.0
    assert sr.paper_pairs_rate(p, np.array([0.6, 0.4]), np.arange(2)[None])[0] == 1.0


def test_columbia_bracket_on_call_paper_reports_both_bounds_with_exact_ci_on_the_same_session_one():
    rows = []
    for i in range(12):
        rows.append(_r(f"c{i}", sr.CALL_PAPER, False, pipe="columbia", j=0, p=0.01 if i < 2 else 0.5, stat=0.6))
        if i < 6:
            rows.append(_r(f"c{i}", sr.CALL_PAPER, False, pipe="columbia", j=1, p=0.04 if i == 0 else 0.5, stat=0.6))
    b = sr.columbia_bracket(pd.DataFrame(rows), paper=True, B=50, seed=[2])
    ind, same = b["independent_days_bound"], b["same_session_bound"]
    assert ind["n_patients"] == 12 and ind["rate"] == pytest.approx(mg.bh2_pairs_rate(
        np.array([0.01] * 2 + [0.5] * 10), np.arange(12)[None])[0])
    assert same["n_patients"] == 6 and (same["k"], same["n"]) == (2, 6)
    assert same["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(2, 6))



def test_composite_at_k_uses_115s_patient_outcomes_and_gives_an_exact_ci():
    rows = []
    for i in range(10):
        rows.append(_r(f"d{i}", "rb", i == 0, pipe="dubois", j=0))
        if i < 8:
            rows.append(_r(f"d{i}", "rb", i == 1, pipe="dubois", j=1))
    c = sr.composite(pd.DataFrame(rows), "dubois", 2)
    assert (c["k_runs"], c["n"], c["k"]) == (2, 8, 2)
    assert c["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(2, 8))
    assert c["n_patients_with_fewer_windows"] == 2


def test_composite_without_second_windows_is_not_available():
    rows = [_r(f"d{i}", "rb", False, pipe="dubois", j=0) for i in range(5)]
    assert sr.composite(pd.DataFrame(rows), "dubois", 2) == {"k_runs": 2, "status": "NOT AVAILABLE",
                                                             "note": "no patient has 2 windows of this reading"}



def test_strata_cell_adds_exact_intervals_to_143s_cell():
    rows = pd.DataFrame([dict(_r(f"s{i}", "v", i < 2), stratum="normal_awake", transitioning=False, a9_flag=False,
                              extent_t0=0.0, extent_t1=1.0, rand_call=i < 1, rand_stratum="normal_awake",
                              rand_transitioning=False, rand_extent_t0=0.0, rand_extent_t1=1.0, rand_a9_flag=False)
                         for i in range(10)])
    c = sr.stratum_cell(rows, "awake", B=20, seed_parts=("t",))
    assert (c["reading"]["k"], c["reading"]["n"]) == (2, 10)
    assert c["reading"]["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(2, 10))
    assert c["randomised"]["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(1, 10))






def test_output_guard_is_143s():
    assert sr.assert_no_identifiers is sr.st.assert_no_identifiers


def test_files_merge_meta_keys_by_pipeline_and_refuse_a_pipeline_twice(tmp_path):
    a = {"edlow17": {"v": _entry(True, "STATED")}, "shortened_to_fit": {"variants": {"cruse11": ["x"]}, "rule": "r1"},
         "errata": [1], "constant_departures": {"edlow17": []}}
    b = {"chennu13": {"w": _entry(True, "STATED")}, "shortened_to_fit": {"variants": {"chennu13": ["y"]}, "rule": "r2"},
         "errata": [2], "constant_departures": {"chennu13": []}, "sensitivity_eligible": {"s": {}}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    m = sr.load_classification((str(pa), str(pb)))
    assert m["shortened_to_fit"]["variants"] == {"cruse11": ["x"], "chennu13": ["y"]}
    assert m["errata"] == [1, 2] and set(m["constant_departures"]) == {"edlow17", "chennu13"}
    pb.write_text(json.dumps(dict(b, edlow17={})))
    with pytest.raises(RuntimeError):
        sr.load_classification((str(pa), str(pb)))


def test_weighted_selection_drops_a_variant_without_rows_in_the_half_as_the_point_estimate_does():
    rows = [_r("p0", "a", False), _r("p0", "b", True), _r("p1", "a", False), _r("p1", "b", False)]
    rows += [_r(f"pb{i}", "b", False) for i in range(3)] + [_r(f"pa{i}", "a", True) for i in range(2)]
    rows += [_r("pc", "c", False)]
    d = pd.DataFrame(rows)
    C, pats = sr._matrix(d, ["a", "b", "c"])
    w = np.array([0.0 if p == "pc" else 1.0 for p in pats])
    sub = d[d["patient"] != "pc"]
    own = {v: {"rate": float(sub[sub["variant"] == v]["call"].astype(float).mean())} for v in ("a", "b")}
    assert mg._select_primary(sub, own, {})[0] == "a"
    assert sr._select_weighted(C, w, ["a", "b", "c"], {}) == "a"


def test_meta_keys_of_one_pipeline_in_both_files_are_refused(tmp_path):
    a = {"edlow17": {}, "shortened_to_fit": {"variants": {"cruse11": ["x"]}}}
    b = {"chennu13": {}, "shortened_to_fit": {"variants": {"cruse11": ["y"]}}}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    with pytest.raises(RuntimeError, match="more than one"):
        sr.load_classification((str(pa), str(pb)))



def test_a_sensitivity_set_without_its_reclassified_axes_is_refused_unless_identical_to_main():
    cl = _with_summary_axis(_classification())
    del cl["sensitivity_eligible"]["sens"]["reclassified_axes"]
    with pytest.raises(RuntimeError, match="reclassified_axes"):
        sr.classification_sets(cl)
    cl = _with_summary_axis(_classification())
    for v in ("full", "max"):
        cl["mbvt"][v] = _entry(True, "STATED")
    del cl["sensitivity_eligible"]["sens"]["reclassified_axes"]
    s = sr.classification_sets(cl)
    assert s["headline"]["mbvt"] == ["median", "full", "max"]
    assert s["sensitivity"]["sens"]["status"] == "retired: identical to the main eligible set"


def test_published_k_is_read_from_the_k_table():
    with open(sr.K_TABLE) as fh:
        kt = json.load(fh)
    assert sr.published_k() == {p: int(e["k"]) for p, e in kt.items()}


def test_columbia_bracket_on_a_non_paper_variant_is_115s_p_only_rule():
    rows = []
    for i in range(10):
        rows.append(_r(f"c{i}", "as_coded", False, pipe="columbia", j=0, p=0.01 if i < 3 else 0.5, stat=0.4))
    d = pd.DataFrame(rows)
    paper = sr.columbia_bracket(d, paper=True, B=10, seed=[1])["independent_days_bound"]["rate"]
    coded = sr.columbia_bracket(d, paper=False, B=10, seed=[1])["independent_days_bound"]["rate"]
    assert paper == 0.0
    assert coded == pytest.approx(mg.bh2_pairs_rate(np.array([0.01] * 3 + [0.5] * 7), np.arange(10)[None])[0])


def test_composite_rows_follow_the_selected_variant_whatever_its_source():
    c3 = pd.DataFrame([dict(_r(f"d{i}", None, i == 0, pipe="dubois", j=j), reading="c3_b_assessment_random_phase1",
                            base="assessment_random_phase1") for i in range(4) for j in (0, 1)])
    frozen = pd.DataFrame([_r(f"d{i}", "assessment_blocked_perm", i == 1, pipe="dubois", j=j)
                           for i in range(4) for j in (0, 1, 2)])
    g = sr.composite_rows("ncmd_bdsp", "dubois", "c3_b_assessment_random_phase1", c3.assign(null="ncmd_bdsp"),
                          frozen, None, {f"d{i}_0" for i in range(4)})
    assert set(g["variant"]) == {"c3_b_assessment_random_phase1"} and len(g) == 8
    g = sr.composite_rows("ncmd_bdsp", "dubois", "assessment_blocked_perm", c3.assign(null="ncmd_bdsp"), frozen,
                          None, {f"d{i}_0" for i in range(3)})
    assert set(g["variant"]) == {"assessment_blocked_perm"} and sorted(g["patient"].unique()) == ["d0", "d1", "d2"]


def test_mbvt_session_note_comes_from_the_eligible_paradigms():
    cl = {"mbvt": {"a": {"axes": {"paradigm": {"value": "vt3", "class": "STATED"}}, "stated_rule_eligible": True},
                   "b": {"axes": {"paradigm": {"value": "vt2", "class": "CONTRADICTS"}}, "stated_rule_eligible": False}}}
    assert sr.eligible_paradigms(cl, "mbvt", ["a"]) == ["vt3"]



def test_covert_processing_entries_stay_outside_holm():
    e = {"a": {"null": "nstim0_bdsp", "stated_rule": {"p_greater_nominal": 0.01}},
         "s": {"null": "nstim0_bdsp", "covert_processing": True, "stated_rule": {"p_greater_nominal": 0.001}}}
    sr.add_holm(e)
    assert e["a"]["stated_rule"]["p_holm"] == pytest.approx(0.01)
    assert "p_holm" not in e["s"]["stated_rule"] and "inflated" not in e["s"]["stated_rule"]
    assert sr.is_covert("sokoliuk21") and not sr.is_covert("cornell")


def test_columbia_pairs_bootstrap_counts_each_drawn_copy_as_its_own_unit():
    """Relabelled draws: a patient drawn twice is two units, each paired with every unit of another patient (pairs
    of one patient's copies are not pairs of different patients, as in 115). Merging copies (original ids kept)
    would give a different rate."""
    def by_units(p, auc, draw):
        pos = tot = 0
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                if draw[i] != draw[j]:
                    tot += 1
                    pos += bool(sr.bh2_paper(p[draw[i]], auc[draw[i]], p[draw[j]], auc[draw[j]]))
        return pos / tot
    p, auc = np.array([0.01, 0.5, 0.5, 0.04]), np.full(4, 0.7)
    draw = np.array([0, 0, 1, 3])
    assert sr.paper_pairs_rate(p, auc, draw[None])[0] == pytest.approx(by_units(p, auc, draw)) == pytest.approx(4 / 5)
    assert by_units(p, auc, np.unique(draw)) == pytest.approx(2 / 3)
    rng = np.random.default_rng(9)
    q, a = rng.uniform(0, 0.1, 12), rng.uniform(0.3, 0.8, 12)
    for _ in range(20):
        d = rng.integers(0, 12, 12)
        if len(set(d)) > 1:
            assert sr.paper_pairs_rate(q, a, d[None])[0] == pytest.approx(by_units(q, a, d))


def test_c3_join_takes_base_rows_from_the_combined_frame_with_its_stale_base_column():
    base = pd.DataFrame([dict(_r(f"b{i}", "jk_base", False, pipe="cornell"), base=None, reading=None, start_s=10.0,
                              stratum="normal_awake", transitioning=False, a9_flag=False, extent_t0=0.0,
                              extent_t1=1.0, rand_call=False, rand_variant="g") for i in range(3)])
    c3 = pd.DataFrame([{"null": "ncmd_bdsp", "pipe": "cornell", "uid": f"b{i}_0", "patient": f"b{i}", "j": 0,
                        "start_s": 10.0, "status": "ok", "call": i == 0, "reading": "c3_x", "base": "jk_base",
                        "key": f"b{i}_0|0|c3_x"} for i in range(3)])
    g, counts = sr.c3_on_base_rows(c3, base)
    assert len(g) == 3 and list(g["call"]) == [True, False, False] and counts["c3_rows_without_base_row"] == 0



def _k1_frames():
    g = pd.DataFrame([_r("A", "v", True, j=1), _r("A", "v", False, j=2), _r("B", "v", False, j=0),
                      _r("B", "v", True, j=1), _r("C", "v", True, j=0)])
    pub = pd.DataFrame([_r("B", "v", False, j=0), _r("C", "v", True, j=0)])
    return g, pub


def test_k1_check_matches_the_first_usable_window_and_counts_na_first_overlays():
    g, pub = _k1_frames()
    c = sr.k1_check(g, pub, "edlow17")
    assert (c["composite_k1"]["k"], c["composite_k1"]["n"]) == (2, 3)
    assert (c["first_usable"]["k"], c["first_usable"]["n"]) == (2, 3)
    assert (c["first_overlay"]["k"], c["first_overlay"]["n"]) == (1, 2)
    assert c["n_first_overlay_na_later_usable"] == 1


def test_k1_check_stops_when_the_composite_does_not_take_the_first_usable_window(monkeypatch):
    g, pub = _k1_frames()
    real = mg.patient_outcomes

    def last_window(rows, rule, k):
        r = rows.sort_values(["uid", "j"], ascending=[True, False])
        return real(r.assign(j=-r["j"]), rule, k)
    monkeypatch.setattr(sr.mg, "patient_outcomes", last_window)
    with pytest.raises(RuntimeError, match="first usable"):
        sr.k1_check(g, pub, "edlow17")


def test_k1_check_stops_when_a_usable_first_overlay_is_not_the_first_usable_window():
    g, pub = _k1_frames()
    with pytest.raises(RuntimeError, match="j = 0"):
        sr.k1_check(g[~((g["patient"] == "B") & (g["j"] == 0))], pub, "edlow17")
    with pytest.raises(RuntimeError, match="j = 0"):
        sr.k1_check(g.assign(call=np.where((g["patient"] == "C") & (g["j"] == 0), False, g["call"])), pub,
                    "edlow17")



def test_held_out_test_pools_each_halfs_calls_of_the_variant_picked_on_the_other_half():
    d, pats, h = _halves_frame()
    cf = sr.crossfit(d, ["x", "y"], {}, B=20, seed=[1])
    t = sr.held_out_test(cf)
    k = n = 0
    for part in cf["halves"]:
        est = d[d["patient"].isin([p for p, hh in zip(pats, h) if hh != part["select_half"]])]
        e = est[est["variant"] == part["variant"]]
        k, n = k + int(e["call"].sum()), n + len(e)
    assert (t["k"], t["n"]) == (k, n) and n == len(pats)
    assert t["label"] == "held-out (valid) test"
    assert t["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(k, n))
    assert t["p_greater_nominal"] == pytest.approx(mg.binom_greater(k, n, mg.NOMINAL))
    assert t["variants_by_select_half"] == {"0": "x", "1": "y"}


def test_held_out_test_is_not_available_without_a_selection_and_refuses_a_missing_half():
    no_sel = {"select_half": 0, "variant": None, "k": 0, "n": 0, "rate": None}
    other = {"select_half": 1, "variant": "y", "k": 1, "n": 10, "rate": 0.1}
    t = sr.held_out_test({"halves": [no_sel, other]})
    assert t["status"] == "NOT AVAILABLE" and "p_greater_nominal" not in t
    t = sr.held_out_test({"halves": [dict(no_sel, variant="x"), other]})
    assert t["status"] == "NOT AVAILABLE" and "p_greater_nominal" not in t
    for halves in ([other], [other, other]):
        with pytest.raises(RuntimeError, match="both halves"):
            sr.held_out_test({"halves": halves})


def test_held_out_holm_within_a_bdsp_null_set_leaves_pdoc_covert_and_the_full_data_p_alone():
    e = {"a": {"null": "ncmd_bdsp", "stated_rule": {"p_greater_nominal": 0.5}, "held_out_test": {"p_greater_nominal": 0.01}},
         "b": {"null": "ncmd_bdsp", "stated_rule": {"p_greater_nominal": 0.5}, "held_out_test": {"p_greater_nominal": 0.04}},
         "na": {"null": "ncmd_bdsp", "stated_rule": {"p_greater_nominal": 0.5},
                "held_out_test": {"status": "NOT AVAILABLE"}},
         "c": {"null": "ncmd_pdoc", "held_out_test": {"p_greater_nominal": 0.001}},
         "s": {"null": "nstim0_bdsp", "covert_processing": True, "held_out_test": {"p_greater_nominal": 0.001}}}
    sr.add_holm(e, "held_out_test")
    assert e["a"]["held_out_test"]["p_holm"] == pytest.approx(0.02) and e["a"]["held_out_test"]["inflated"]
    assert e["b"]["held_out_test"]["p_holm"] == pytest.approx(0.04) and e["b"]["held_out_test"]["inflated"]
    for k in ("na", "c", "s"):
        assert "p_holm" not in e[k]["held_out_test"] and "inflated" not in e[k]["held_out_test"]
    assert all("p_holm" not in e[k]["stated_rule"] for k in ("a", "b", "na"))



def test_range_endpoints_name_their_variants_with_exact_intervals_and_label_the_max_as_selected():
    cells = {"a": (2, 40), "b": (9, 40), "c": (2, 50), "d": (4, 80)}
    ep = sr.range_endpoints({v: sr.st.exact_cell(k, n) for v, (k, n) in cells.items()})
    assert ep["max"]["label"] == "max over eligible readings (selected, not an estimate)"
    assert ep["min"]["label"] == "min over eligible readings (selected, not an estimate)"
    assert [r["variant"] for r in ep["max"]["readings"]] == ["b"]
    assert [r["variant"] for r in ep["min"]["readings"]] == ["c"]
    for side, v in (("max", "b"), ("min", "c")):
        r = ep[side]["readings"][0]
        assert (r["k"], r["n"]) == cells[v] and r["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(*cells[v]))
        assert ep[side]["rate"] == pytest.approx(cells[v][0] / cells[v][1])
    tie = sr.range_endpoints({v: sr.st.exact_cell(*cells[v]) for v in ("d", "a")})
    assert [r["variant"] for r in tie["min"]["readings"]] == ["a", "d"] == [r["variant"] for r in
                                                                            tie["max"]["readings"]]


def test_assess_puts_the_held_out_test_beside_the_full_data_p_and_the_endpoints_in_the_range(monkeypatch):
    d, _, _ = _halves_frame()
    monkeypatch.setattr(sr, "usable", lambda df, nul, pipe, cands: (list(cands), {}))
    out, _ = sr._assess(d, "ncmd_bdsp", "edlow17", ["x", "y"], {}, 20, "stated rule")
    assert out["held_out_test"] == sr.held_out_test(out["crossfit"])
    assert out["stated_rule"]["p_greater_nominal"] == sr.exact_rate(d[d["variant"] == out["variant"]])[
        "p_greater_nominal"]
    assert out["eligible_range"]["endpoints"] == sr.range_endpoints(out["eligible_range"]["rates"])



def _vt_rows(n_pat=8, n_win=6):
    return pd.DataFrame([_r(f"m{i}", "vt", (i + 2 * j) % 7 == 0, pipe="mbvt", null="nstim0_bdsp", j=j)
                         for i in range(n_pat) for j in range(n_win)])


def test_best_of_k_adds_the_published_k_max_and_marks_larger_k_beyond_published_k():
    g = _vt_rows()
    cur = sr.best_of_k_curve(g, "mbvt", (1, 3, 5), 2, B=20, seed=[1])
    assert [c["k"] for c in cur] == [1, 2, 3, 5]
    assert [c["reading"] for c in cur] == ["within published k", "within published k", "beyond published k",
                                           "beyond published k"]
    ref = {c["k"]: c for c in mg.composite_curve(g, mg._rule(mg._load_pipe("mbvt"), rule="best_run"), (1, 2, 3, 5),
                                                 20, [1])}
    for c in cur:
        assert (c["n"], c["rate"]) == (ref[c["k"]]["n"], ref[c["k"]]["rate"])
        assert c["ci95_cp"] == pytest.approx(sr.st.clopper_pearson(int(round(c["rate"] * c["n"])), c["n"]))
    assert [c["k"] for c in sr.best_of_k_curve(g, "mbvt", (1, 3, 5), 3, B=20, seed=[1])] == [1, 3, 5]


def test_published_k_max_is_required_for_the_run_and_never_below_the_k_tables_k():
    with pytest.raises(SystemExit):
        sr.main(["--motor-sha", "0" * 64, "--erp-sha", "0" * 64])
    kmap = sr.published_k()
    assert sr.check_k_max(kmap["mbvt"], kmap) == kmap["mbvt"]
    with pytest.raises(RuntimeError, match="below"):
        sr.check_k_max(kmap["mbvt"] - 1, kmap)



def test_deflation_reports_median_iqr_and_share_below_chance_on_one_row_per_patient():
    vals = [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.60, None]
    d = pd.DataFrame([_r(f"e{i}", "v", False, stat=v) for i, v in enumerate(vals)])
    out = sr.deflation(d, "edlow17")
    x = np.array([v for v in vals if v is not None])
    assert (out["n"], out["n_missing"], out["chance"]) == (7, 1, 0.5)
    assert out["median"] == pytest.approx(np.median(x))
    assert out["iqr"] == pytest.approx(list(np.quantile(x, [0.25, 0.75])))
    assert (out["n_below_chance"], out["share_below_chance"]) == (5, pytest.approx(5 / 7))
    with pytest.raises(RuntimeError, match="one row per patient"):
        sr.deflation(pd.concat([d, d.iloc[:1]]), "edlow17")


def test_deflation_reads_each_pipelines_own_statistic_and_chance_or_says_there_is_none():
    d = pd.DataFrame([_r(f"b{i}", "v", False, pipe="dubois", stat=3.0, x_peak_da=0.4) for i in range(4)])
    out = sr.deflation(d, "dubois")
    assert out["median"] == pytest.approx(0.4) and out["share_below_chance"] == 1.0
    m = pd.DataFrame([_r(f"m{i}", "v", False, pipe="mbvt", stat=0.1 * i) for i in range(4)])
    assert sr.deflation(m, "mbvt")["chance"] == pytest.approx(1 / 8)
    assert sr.deflation(m, "mbvt")["n_below_chance"] == 2
    for pipe in ("cornell", "chennu13", "gibson16", "sokoliuk21"):
        assert sr.deflation(d.assign(pipe=pipe), pipe) == {"status": "no accuracy-type statistic"}


def test_assess_reports_the_deflation_diagnostic_for_the_selected_variant_and_each_range_endpoint(monkeypatch):
    d, _, _ = _halves_frame()
    rng = np.random.default_rng(3)
    d = d.assign(stat=rng.uniform(0.2, 0.8, len(d)))
    monkeypatch.setattr(sr, "usable", lambda df, nul, pipe, cands: (list(cands), {}))
    out, _ = sr._assess(d, "ncmd_bdsp", "edlow17", ["x", "y"], {}, 20, "stated rule")
    df = out["deflation"]
    assert df["selected"] == dict(variant=out["variant"], **sr.deflation(d[d["variant"] == out["variant"]], "edlow17"))
    for side in ("min", "max"):
        want = [dict(variant=r["variant"], **sr.deflation(d[d["variant"] == r["variant"]], "edlow17"))
                for r in out["eligible_range"]["endpoints"][side]["readings"]]
        assert df["endpoints"][side] == want






def _union(cl, strict_of):
    """Give every variant a strict flag: strict_of(pipe, variant, entry) -> bool."""
    for pipe, vs in cl.items():
        if pipe in sr.META:
            continue
        for v, e in vs.items():
            e["stated_rule_eligible_strict"] = bool(strict_of(pipe, v, e))
    return cl


def test_without_strict_flags_the_classification_is_read_as_before_and_strict_is_absent():
    s = sr.classification_sets(_with_summary_axis(_classification()))
    assert s["strict"] is None
    with pytest.raises(RuntimeError, match="strict"):
        sr.require_strict(s)


def test_union_flags_admit_a_variant_the_first_classifier_contradicted_but_strict_must_be_clean_and_inside():
    cl = _with_summary_axis(_classification())
    cl["edlow17"]["v_bad"]["stated_rule_eligible"] = True
    cl = _union(cl, lambda p, v, e: e["stated_rule_eligible"] and v != "v_bad" and v != "v_ok2")
    s = sr.classification_sets(cl)
    assert s["headline"]["edlow17"] == ["v_ok", "v_ok2", "v_bad"]
    assert s["strict"]["edlow17"] == ["v_ok"] and sr.require_strict(s) is s
    cl["edlow17"]["v_bad"]["stated_rule_eligible_strict"] = True
    with pytest.raises(RuntimeError, match="CONTRADICTS"):
        sr.classification_sets(cl)
    cl["edlow17"]["v_bad"]["stated_rule_eligible_strict"] = False
    cl["edlow17"]["v_bad"]["stated_rule_eligible"] = False
    cl["edlow17"]["v_ok"]["stated_rule_eligible"] = False
    with pytest.raises(RuntimeError, match="strict"):
        sr.classification_sets(cl)


def test_a_strict_flag_missing_on_some_variants_is_refused():
    cl = _union(_with_summary_axis(_classification()), lambda p, v, e: e["stated_rule_eligible"])
    del cl["mbvt"]["median"]["stated_rule_eligible_strict"]
    with pytest.raises(RuntimeError, match="strict"):
        sr.classification_sets(cl)


def test_files_keep_each_files_second_classifier_block(tmp_path):
    a = {"edlow17": {"v": _entry(True, "STATED")}, "second_classifier": {"rule": "a"}}
    b = {"chennu13": {"w": _entry(True, "STATED")}, "second_classifier": {"rule": "b"}}
    pa, pb = tmp_path / "motor.json", tmp_path / "erp.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    m = sr.load_classification((str(pa), str(pb)))
    assert m["second_classifier"] == {"motor.json": {"rule": "a"}, "erp.json": {"rule": "b"}}
    assert set(sr.classification_sets(m)["headline"]) == {"edlow17", "chennu13"}




def test_the_call_test_reads_the_reference_classes_not_the_current_ones():
    axes = {"mbmi": {"axes": ["threshold"]}}
    ax = lambda c: {"axes": {"threshold": {"value": "x", "class": c}}}
    ref = {"mbmi": {"a": ax("STATED"), "b": ax("NOT_STATED")}}
    cur = {"mbmi": {"a": ax("CONTRADICTS"), "b": ax("STATED")}}
    sets = {"headline": {"mbmi": ["a", "b"]}, "strict": {"mbmi": ["a"]}}
    assert sr.reading_candidates(ref, sets, "mbmi", "primary", axes)[0] == ["a"]
    assert sr.reading_candidates(cur, sets, "mbmi", "primary", axes)[0] == ["b"]



def _call_cl():
    ax = lambda **k: {a: {"value": "x", "class": c} for a, c in k.items()}
    return {"cruse11": {"a": {"axes": ax(method="STATED", base="STATED", ci="NOT_STATED")},
                        "b": {"axes": ax(method="STATED", base="NOT_STATED")},
                        "c": {"axes": ax(method="STATED", base="STATED", ci="STATED")}},
            "edlow17": {"x": {"axes": ax(perm="NOT_STATED")}, "y": {"axes": ax(perm="NOT_STATED")}},
            "cornell": {"q": {"axes": ax(om2="NOT_STATED", fdr="NOT_STATED", var="STATED")}}}


def test_a_call_rule_is_stated_only_when_every_call_axis_present_is_stated():
    axes = {"cruse11": {"axes": ["method", "base", "ci"]}, "edlow17": {"axes": []},
            "cornell": {"axes": ["om2", "fdr"]}}
    cl = _call_cl()
    assert sr.call_stated(cl, "cruse11", ["a", "b", "c"], axes) == (["c"], "stated")
    assert sr.call_stated(cl, "cruse11", ["a", "b"], axes) == ([], "call rule not fully stated")
    assert sr.call_stated(cl, "cornell", ["q"], axes) == ([], "call rule not stated")
    assert sr.call_stated(cl, "edlow17", ["x", "y"], axes) == (["x", "y"], "stated (common to every variant)")
    with pytest.raises(RuntimeError, match="decision"):
        sr.call_stated(cl, "mbvt", [], axes)
    with pytest.raises(RuntimeError, match="no variant"):
        sr.call_stated(cl, "cornell", ["q"], {"cornell": {"axes": ["om2", "fbr"]}})
    cl["cornell"]["r"] = {"axes": {"var": {"class": "STATED"}}}
    with pytest.raises(RuntimeError, match="none of the call axes"):
        sr.call_stated(cl, "cornell", ["q", "r"], axes)







def test_reading_candidates_primary_generous_and_strict():
    cl = _call_cl()
    axes = {"cruse11": {"axes": ["method", "base", "ci"]}, "edlow17": {"axes": []}}
    sets = {"headline": {"cruse11": ["a", "b", "c"], "edlow17": ["x", "y"]},
            "strict": {"cruse11": ["a", "b"], "edlow17": ["x"]}}
    c, label, rule = sr.reading_candidates(cl, sets, "cruse11", "primary", axes)
    assert c == ["c"] and label == "stated rule" and rule["status"] == "stated" and rule["stated_call_variants"] == ["c"]
    c, label, rule = sr.reading_candidates(cl, sets, "cruse11", "sensitivity_strict", axes)
    assert c == ["a", "b"] and label == f"{sr.STRICT_LABEL} (call rule not fully stated: most favourable reading)"
    c, label, rule = sr.reading_candidates(cl, sets, "cruse11", "sensitivity_generous", axes)
    assert c == ["a", "b", "c"] and label == sr.GENEROUS_LABEL and rule is None
    c, label, _ = sr.reading_candidates(cl, sets, "edlow17", "sensitivity_strict", axes)
    assert c == ["x"] and label == sr.STRICT_LABEL
    assert sr.GENEROUS_LABEL == "sensitivity: most favourable reading incl. unstated call rules"
    assert sr.STRICT_LABEL == "sensitivity: strict (both blind classifiers agree)"



def _e2e(monkeypatch, tmp_path, strict_flags=True):
    ax = lambda **k: {a: {"value": "x", "class": c} for a, c in k.items()}
    cl = {"edlow17": {"x": {"axes": ax(perm="NOT_STATED"), "stated_rule_eligible": True},
                      "y": {"axes": ax(perm="NOT_STATED"), "stated_rule_eligible": True}},
          "mbmi": {"t_ok": {"axes": ax(threshold="STATED", summary="NOT_STATED"), "stated_rule_eligible": True},
                   "t_ns": {"axes": ax(threshold="NOT_STATED", summary="NOT_STATED"), "stated_rule_eligible": True}},
          "shortened_to_fit": {"variants": {}}}
    if strict_flags:
        for p, v, f in (("edlow17", "x", True), ("edlow17", "y", False), ("mbmi", "t_ok", False), ("mbmi", "t_ns", True)):
            cl[p][v]["stated_rule_eligible_strict"] = f
    cl_ref = json.loads(json.dumps(cl))
    if strict_flags:
        cl["mbmi"]["t_ok"]["axes"]["threshold"]["class"] = "CONTRADICTS"
    d, _, _ = _halves_frame()
    rng = np.random.default_rng(5)
    m = pd.DataFrame([_r(f"q{i}", v, bool(rng.random() < (0.03 if v == "t_ok" else 0.2)), pipe="mbmi", stat=0.5)
                      for i in range(40) for v in ("t_ok", "t_ns")])
    rows = pd.concat([d.assign(stat=rng.uniform(0.3, 0.7, len(d))), m], ignore_index=True)
    c3 = pd.DataFrame(columns=["null", "pipe", "reading", "status", "j", "uid", "call"])
    monkeypatch.setattr(sr, "check_classification", lambda files, shas: None)
    monkeypatch.setattr(sr, "load_classification_at", lambda ref: cl_ref)
    monkeypatch.setattr(sr, "load_classification", lambda: cl)
    monkeypatch.setattr(sr.st, "choices_from_headline",
                        lambda: {("ncmd_bdsp", "edlow17"): ("x", "x"), ("ncmd_bdsp", "mbmi"): ("t_ok", "t_ok")})
    reg = rows.assign(call=np.where(rows["variant"] == "t_ok", True, rows["call"]))
    monkeypatch.setattr(sr, "prepare", lambda b4=True: {"df": rows, "c3": c3, "rows": rows, "rows_pinned": reg,
                                                "n_dropped": 0, "res": None, "res_rows": None, "tables": None,
                                                "b4": {"stub": True}})
    monkeypatch.setattr(sr, "check_a1", lambda df, res, tables: {"stub": True})
    monkeypatch.setattr(sr, "label_rows", lambda b, res, tables: b)
    monkeypatch.setattr(sr, "stratum_cell", lambda g, s, B, parts=(): {"n": int(len(g))})
    monkeypatch.setattr(sr, "load_all_variant_rows", lambda want, b4set: rows[[(n, p) in want and v in want[(n, p)]
                                                                               for n, p, v in zip(rows["null"],
                                                                                                  rows["pipe"],
                                                                                                  rows["variant"])]])
    monkeypatch.setattr(sr, "OUT", str(tmp_path / "stated_rule.json"))
    return sr.main(["--motor-sha", "0" * 64, "--erp-sha", "0" * 64, "--mbvt-published-k-max", "7", "--B", "20"])


def test_main_builds_the_primary_and_both_sensitivities_with_the_full_set(monkeypatch, tmp_path):
    out = _e2e(monkeypatch, tmp_path)
    assert out == json.loads((tmp_path / "stated_rule.json").read_text())
    assert out["call_principle"] == sr.CALL_PRINCIPLE and out["first_k"] == sr.FIRST_K
    assert out["class_reference"]["commit"] == sr.CLASS_REF and set(out["call_axes"]) == set(sr.CALL_AXES)
    assert out["class_reference"]["class_changes_since"] == [
        {"pipe": "mbmi", "axis": "threshold", "from": "STATED", "to": "CONTRADICTS", "n_variants": 1}]
    prim, gen, strict = out["entries"], out["sensitivity_generous"]["entries"], out["sensitivity_strict"]["entries"]
    assert prim["ncmd_bdsp|mbmi"]["eligible_after_rows"] == ["t_ok"] and prim["ncmd_bdsp|mbmi"]["label"] == "stated rule"
    assert prim["ncmd_bdsp|mbmi"]["call_rule"]["status"] == "stated"
    assert prim["ncmd_bdsp|edlow17"]["eligible_after_rows"] == ["x", "y"]
    assert gen["ncmd_bdsp|mbmi"]["eligible_after_rows"] == ["t_ok", "t_ns"]
    assert gen["ncmd_bdsp|mbmi"]["label"] == sr.GENEROUS_LABEL
    assert strict["ncmd_bdsp|edlow17"]["eligible_after_rows"] == ["x"]
    assert strict["ncmd_bdsp|mbmi"]["eligible_after_rows"] == ["t_ns"]
    assert strict["ncmd_bdsp|mbmi"]["label"] == f"{sr.STRICT_LABEL} (call rule not stated: most favourable reading)"
    for block in (out, out["sensitivity_generous"], out["sensitivity_strict"]):
        for key, e in block["entries"].items():
            assert "p_holm" in e["stated_rule"] and "p_holm" in e["held_out_test"]
            assert e["eligible_range"]["endpoints"]["max"]["label"] == sr.MAX_LABEL
            assert e["deflation"]["selected"]["variant"] == e["variant"]
            c = block["per_patient_at_published_k"][key]
            assert c["variant"] == e["variant"] and c["first_k"] == sr.FIRST_K and "composite" in c
            assert key in block["strata"]
    assert out["sensitivity_strict"]["label"] == sr.STRICT_LABEL
    b = prim["ncmd_bdsp|mbmi"]["beside"]["validity_primary"]
    assert b["variant"] == "t_ok" and b["k"] == b["n"] == 40
    assert prim["ncmd_bdsp|mbmi"]["stated_rule"]["k"] < 40
    assert out["cruse11_b4"]["stub"] is True and "replaced" in out["cruse11_b4"]["note"]


def test_main_refuses_without_strict_flags(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="strict"):
        _e2e(monkeypatch, tmp_path, strict_flags=False)



B4 = importlib.import_module("148_cruse11_b4")


def _b4_names():
    return B4.b4_natives(sr.rn.load_pipe("cruse11"))


def test_148_rows_replace_only_cruse11_bdsp_b4_rows_and_report_the_replacement():
    v4 = _b4_names()
    old = [_r(f"c{i}", v, False, pipe="cruse11", uid=f"c{i}_0") for i in range(3) for v in v4[:2]]
    old += [_r(f"c{i}", "cr11_b6_hilbert_ci99", False, pipe="cruse11", uid=f"c{i}_0") for i in range(3)]
    old += [_r(f"c{i}", v4[0], False, pipe="cruse11", null="ncmd_pdoc", uid=f"c{i}_0") for i in range(3)]
    old += [_r("e0", "v", False)]
    df = pd.DataFrame(old)
    new = pd.DataFrame([_r(f"n{i}", v, True, pipe="cruse11", uid=f"n{i}_0", stat=1.0) for i in range(5) for v in v4]
                       + [_r("n9", None, None, pipe="cruse11", uid="n9_0", status="no_fit")])
    out, keys, info = sr.swap_b4(df, new)
    b = out[(out["null"] == "ncmd_bdsp") & (out["pipe"] == "cruse11") & out["variant"].isin(v4)]
    assert set(b["uid"]) == {f"n{i}_0" for i in range(5)} and (b["stat"] == 1.0).all()
    assert len(out[out["variant"] == "cr11_b6_hilbert_ci99"]) == 3
    assert len(out[out["null"] == "ncmd_pdoc"]) == 3 and len(out[out["pipe"] == "edlow17"]) == 1
    assert keys == set(df[(df["null"] == "ncmd_bdsp") & df["variant"].isin(v4)]["key"])
    assert (info["n_recordings_old"], info["n_recordings_new"], info["n_rows_no_fit_new"]) == (3, 5, 1)
    with pytest.raises(RuntimeError, match="148"):
        sr.swap_b4(df, new.iloc[:0])
    with pytest.raises(RuntimeError, match="b4"):
        sr.swap_b4(df, new[new["variant"] != v4[0]])
    with pytest.raises(RuntimeError, match="null set"):
        sr.swap_b4(df, new.assign(null="ncmd_pdoc"))


def _shard(root, nul, pipe, rows):
    d = os.path.join(root, f"{nul}__{pipe}__native__none")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rows00.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r, split="analysis", dev=False)) + "\n")


def test_composite_rows_of_the_b4_variants_come_from_148s_folder(tmp_path, monkeypatch):
    v4 = _b4_names()
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _shard(a, "ncmd_bdsp", "cruse11", [_r("c0", v4[0], False, pipe="cruse11", src="old"),
                                       _r("c0", "cr11_b6_hilbert_ci99", False, pipe="cruse11", src="old")])
    _shard(b, "ncmd_bdsp", "cruse11", [_r("c0", v4[0], True, pipe="cruse11", src="new")])
    _shard(a, "ncmd_bdsp", "edlow17", [_r("e0", "x", False, src="old")])
    monkeypatch.setattr(sr.st, "SHARDS", a)
    monkeypatch.setattr(sr.mg, "drop_touched", lambda df: (df, {}))
    want = {("ncmd_bdsp", "cruse11"): {v4[0], "cr11_b6_hilbert_ci99"}, ("ncmd_bdsp", "edlow17"): {"x"}}
    g = sr.load_all_variant_rows(want, set(v4), shards=a, b4_root=b)
    src = dict(zip(g["variant"], g["src"]))
    assert src == {v4[0]: "new", "cr11_b6_hilbert_ci99": "old", "x": "old"}


def test_the_strict_sokoliuk21_row_carries_the_convention_flag():
    assert sr.reading_flag("sensitivity_strict", "sokoliuk21") == sr.STRICT_NOTES["sokoliuk21"]
    assert "top-down" in sr.STRICT_NOTES["sokoliuk21"] and "surr_fdr" in sr.STRICT_NOTES["sokoliuk21"]
    assert sr.reading_flag("primary", "sokoliuk21") is None and sr.reading_flag("sensitivity_strict", "mbvt") is None


def test_interim_rows_keep_the_pinned_cruse11_b4_rows_and_say_so():
    df = pd.DataFrame([_r("c0", _b4_names()[0], False, pipe="cruse11")])
    out, keys, info = sr.swap_b4(df, None)
    assert out.equals(df) and keys == set() and info["status"] == sr.B4_INTERIM
    assert "interim" in sr.B4_INTERIM and "6-block" in sr.B4_INTERIM


def test_main_interim_flag_reaches_prepare(monkeypatch):
    got = []

    def stop(b4=True):
        got.append(b4)
        raise RuntimeError("stop after prepare")
    monkeypatch.setattr(sr, "prepare", stop)
    monkeypatch.setattr(sr, "check_classification", lambda files, shas: None)
    monkeypatch.setattr(sr, "load_classification", lambda: {"shortened_to_fit": {"variants": {}}})
    monkeypatch.setattr(sr, "load_classification_at", lambda ref: {"shortened_to_fit": {"variants": {}}})
    monkeypatch.setattr(sr, "require_strict", lambda s: s)
    monkeypatch.setattr(sr.st, "choices_from_headline", lambda: {})
    for argv in ([], ["--interim-without-b4"]):
        with pytest.raises(RuntimeError, match="stop after prepare"):
            sr.main(["--motor-sha", "0" * 64, "--erp-sha", "0" * 64, "--mbvt-published-k-max", "7"] + argv)
    assert got == [True, False]
