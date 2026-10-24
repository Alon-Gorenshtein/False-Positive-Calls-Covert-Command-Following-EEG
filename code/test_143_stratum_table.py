import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest

st = importlib.import_module("143_stratum_table")
mg = st.mg


def _row(uid, patient, call, stratum="normal_awake", transitioning=False, a9=False, variant="v", j=0, **kw):
    r = {"uid": uid, "patient": patient, "call": call, "stratum": stratum, "transitioning": transitioning,
         "a9_flag": a9, "variant": variant, "j": j, "extent_t0": 0.0, "extent_t1": 100.0}
    r.update(kw)
    return r



def test_rate_is_a_per_patient_ratio_with_na_left_out_and_counted():
    d = pd.DataFrame([_row("a_0", "a", True), _row("a_1", "a", False), _row("b_0", "b", False),
                      _row("c_0", "c", None)])
    r = st.rate(d, "call", B=200, seed=[1])
    assert (r["k"], r["n"], r["n_patients"], r["n_na"]) == (1, 3, 2, 1)
    assert r["rate"] == pytest.approx(1 / 3)
    assert r["ci95"][0] <= r["rate"] <= r["ci95"][1]


def test_rate_counts_a_patient_drawn_twice_twice():
    """Relabelled clusters: a draw holding patient a twice weighs a twice (115.boot_idx, ratio_draws)."""
    d = pd.DataFrame([_row("a_0", "a", True), _row("b_0", "b", False), _row("b_1", "b", False),
                      _row("c_0", "c", False)])
    r = st.rate(d, "call", idx=np.array([[0, 0, 1], [1, 1, 2], [0, 1, 2]]))
    assert r["draws"] == pytest.approx([2 / 4, 0.0, 1 / 4])


def test_rate_of_nothing_is_empty_not_zero():
    r = st.rate(pd.DataFrame(columns=["patient", "call"]), "call", B=10, seed=[1])
    assert r["n"] == 0 and r["rate"] is None and r["ci95"] is None



def test_bdsp_strata_are_116s_awake_and_disorganised_on_clean_non_transitioning_windows():
    d = pd.DataFrame([
        _row("1", "1", True, "normal_awake"),
        _row("2", "2", True, "normal_awake", transitioning=True),
        _row("3", "3", True, "normal_awake", a9=True),
        _row("4", "4", True, "normal_awake", a9=None),
        _row("5", "5", True, "burst_suppression"), _row("6", "6", True, "iiic_periodic"),
        _row("7", "7", True, "diffuse_slowing"), _row("8", "8", True, "sleep"), _row("9", "9", True, "other_awake"),
        _row("10", "10", True, "mixed"), _row("11", "11", True, "diffuse_slowing", transitioning=None)])
    assert st.STRATUM_RULES["awake"]["source"] == "bdsp"
    assert list(d.loc[st.stratum_mask(d, "awake"), "uid"]) == ["1"]
    assert list(d.loc[st.stratum_mask(d, "icu_typical"), "uid"]) == ["5", "6", "7", "11"]
    assert set(st.DISORGANISED) == set(st.mech.DISORGANISED)


def test_pdoc_wake_like_is_the_lower_delta_flag_on_a9_clean_windows():
    d = pd.DataFrame([_row("1", "1", True, "na", lower_delta=True), _row("2", "2", True, "na", lower_delta=False),
                      _row("3", "3", True, "na", lower_delta=None), _row("4", "4", True, "na", lower_delta=True,
                                                                        a9=True)])
    assert st.STRATUM_RULES["pdoc_wake_like"]["source"] == "pdoc"
    assert list(d.loc[st.stratum_mask(d, "pdoc_wake_like"), "uid"]) == ["1"]



def test_readings_follow_the_c0_grades_and_never_use_a_liberal_default():
    vp = {"cornell": "jk", "dubois": "perm", "edlow17": "block", "mbvt": "binom", "mbmi": "mean57"}
    dflt = {"cornell": "theoretical", "dubois": "assessment_random_phase1", "edlow17": "permsecond_cvsecond_alt",
            "mbvt": "vt3_full", "mbmi": "gu17_peak"}
    c = st.readings_for("cornell", vp["cornell"], dflt["cornell"])
    assert [r["kind"] for r in c] == ["c3"] and c[0]["variant"] == "jackknife_bh_keep40_r4"
    assert c[0]["reading"] == "c3_jackknife_bh_keep40_r4_om2contig"
    d = st.readings_for("dubois", vp["dubois"], dflt["dubois"])
    assert [r["reading"] for r in d] == ["c3_b_assessment_random_phase1", "c3_b_assessment_blocked_phase1"]
    for r in c + d:
        assert r["variant"] != dflt[r["pipe"]] or r["kind"] == "c3"
        assert r["grade"] == "C3 FIDELITY-CORRECTED (default: LIBERAL DEPARTURE)"
    e = st.readings_for("edlow17", vp["edlow17"], dflt["edlow17"])
    assert [(r["kind"], r["variant"], r["grade"]) for r in e] == [
        ("default", "permsecond_cvsecond_alt", "FAITHFUL LITERAL READING")]
    m = st.readings_for("mbmi", vp["mbmi"], dflt["mbmi"])
    assert [(r["kind"], r["variant"]) for r in m] == [("validity_primary", "mean57")]
    assert "not graded" in m[0]["grade"]
    with pytest.raises(ValueError):
        st.readings_for("sokoliuk21", "x", "y")


def test_every_command_pipeline_has_a_reading_rule_and_sokoliuk_is_left_out():
    assert set(st.CMD_PIPES) == set(st.mech.PIPES) - {"sokoliuk21"}
    assert "sokoliuk21" in st.LEFT_OUT



def _base(uid, start=10.0, **kw):
    return dict(_row(uid, uid[0], False, variant="jackknife_bh_keep40_r4", null="ncmd_bdsp", pipe="cornell",
                     start_s=start, status="ok", rand_variant="g", rand_call=False), **kw)


def _c3(uid, call, start=10.0, status="ok", j=0):
    return {"uid": uid, "patient": uid[0], "null": "ncmd_bdsp", "pipe": "cornell", "j": j, "start_s": start,
            "status": status, "call": call, "reading": "c3_jackknife_bh_keep40_r4_om2contig",
            "base": "jackknife_bh_keep40_r4", "key": f"{uid}|{j}|c3"}


def test_c3_rows_take_the_strata_and_window_of_their_base_variant_row():
    base = pd.DataFrame([_base("a_0", stratum="burst_suppression", rand_call=True), _base("b_0"),
                         dict(_base("c_0"), variant="other")])
    c3 = pd.DataFrame([_c3("a_0", True), _c3("b_0", False), _c3("c_0", True), _c3("d_0", True, status="na: x"),
                       _c3("a_0", True, j=1)])
    out, counts = st.c3_on_base(c3, base)
    assert list(out["uid"]) == ["a_0", "b_0"]
    assert list(out["stratum"]) == ["burst_suppression", "normal_awake"]
    assert list(out["call"]) == [True, False]
    assert list(out["rand_call"]) == [True, False]
    assert counts == {"c3_rows_ok": 3, "c3_rows_not_ok": 1, "c3_rows_without_base_row": 1}


def test_c3_row_on_another_window_than_its_base_is_refused():
    with pytest.raises(RuntimeError):
        st.c3_on_base(pd.DataFrame([_c3("a_0", True, start=11.0)]), pd.DataFrame([_base("a_0")]))



def test_randomised_frame_uses_the_randomised_arms_own_window_and_labels():
    d = pd.DataFrame([dict(_row("a_0", "a", True), rand_call=False, rand_stratum="sleep", rand_transitioning=True,
                           rand_extent_t0=5.0, rand_extent_t1=95.0, rand_variant="g"),
                      dict(_row("b_0", "b", True), rand_call=None, rand_variant="g")])
    r, n_missing = st.randomised(d)
    assert n_missing == 1 and len(r) == 1
    assert r.iloc[0][["call", "stratum", "transitioning", "extent_t0", "extent_t1"]].tolist() == \
        [False, "sleep", True, 5.0, 95.0]
    assert "a9_flag" not in r



PC = {"cells": {"openbmi|cornell|published|jackknife_bh_keep40_r4|true": {"k": 51, "n": 51, "rate": 1.0,
                                                                          "ci": [0.93, 1.0]},
                "openbmi|cornell|randomized|theoretical_bh_keep40_r4|true": {"k": 49, "n": 51, "rate": 0.96,
                                                                             "ci": [0.86, 0.99]},
                "erp|mbvt|published|vt3|true": {"k": 10, "n": 13, "rate": 0.77, "ci": [0.46, 0.95]}},
      "headline": {"openbmi|cornell": {"role": "primary"}, "erp|mbvt": {"role": "primary"}}}


def test_positive_control_takes_the_primary_arm_cell_of_the_variant_and_says_when_it_is_not_the_reading():
    a = st.positive_control(PC, "cornell", "jackknife_bh_keep40_r4", "theoretical_bh_keep40_r4", exact=False)
    assert a["arm"] == "openbmi" and (a["published"]["k"], a["published"]["n"]) == (51, 51)
    assert (a["randomised"]["k"], a["randomised"]["n"]) == (49, 51)
    assert a["exact_reading"] is False and "nearest" in a["note"]
    b = st.positive_control(PC, "mbvt", "vt3", "missing_group", exact=True)
    assert b["published"]["k"] == 10 and b["randomised"] is None and b["exact_reading"] is True
    assert b["note"] == "randomised group not run in A5"
    assert "variant not run in A5" in st.positive_control(PC, "mbvt", "vt2", "g", exact=True)["note"]
    c = st.positive_control(PC, "cruse11", "x", "x", exact=True)
    assert c["arm"] is None and c["published"] is None and "no reuse-free" in c["note"]



def _ours(k, n):
    return {"k": k, "n": n, "rate": k / n, "ci95": [0.0, 0.0]}


def test_consistency_uses_exact_tests_not_a_degenerate_bootstrap_interval():
    inside = st.consistency(_ours(2, 26), {"k": 2, "n": 26, "kind": "null"})
    assert inside["verdict"] == "inside" and inside["p_fisher"] == pytest.approx(1.0)
    assert inside["published_ci95"][0] < 2 / 26 < inside["published_ci95"][1]
    zero = st.consistency(_ours(0, 19), {"k": 2, "n": 26, "kind": "null"})
    assert zero["verdict"] == "inside" and zero["p_fisher"] == pytest.approx(0.5010101, rel=1e-6)
    above = st.consistency(_ours(268, 703), {"k": 2, "n": 26, "kind": "null"})
    assert above["verdict"] == "above" and above["p_fisher"] == pytest.approx(0.0013555, rel=1e-4)
    assert above["p_at_most_k_at_our_rate"] == pytest.approx(
        sum(__import__("math").comb(26, i) * (268 / 703) ** i * (1 - 268 / 703) ** (26 - i) for i in range(3)))
    assert st.consistency(_ours(0, 300), {"k": 9, "n": 26, "kind": "null"})["verdict"] == "below"
    ub = st.consistency(_ours(6, 20), {"k": 138, "n": 293, "kind": "upper_bound"})
    assert ub["verdict"] == "not above the published positive fraction"
    low = st.consistency(_ours(0, 40), {"k": 138, "n": 293, "kind": "upper_bound"})
    assert low["verdict"] == "not above the published positive fraction" and low["p_fisher_greater"] == 1.0
    ub2 = st.consistency(_ours(17, 20), {"k": 138, "n": 293, "kind": "upper_bound"})
    assert ub2["verdict"] == "above the published positive fraction" and ub2["p_fisher_greater"] < 0.001
    assert st.consistency({"rate": None, "ci95": None}, {"k": 0, "n": 3, "kind": "null"})["verdict"] == "no data"
    design = {"kind": "design", "rate": 0.05}
    assert st.consistency(_ours(6, 121), design)["verdict"] == "inside"
    assert st.consistency(_ours(268, 703), design)["verdict"] == "above"
    assert st.consistency(_ours(1, 223), design)["verdict"] == "below"


def test_every_published_benchmark_names_its_source_doi_and_location():
    tags = {t for e in st.PUBLISHED.values() for t in (b["source"] for b in e["benchmarks"])}
    assert tags <= set(st.SOURCES)
    for pipe in st.CMD_PIPES:
        assert pipe in st.PUBLISHED
        for bm in st.PUBLISHED[pipe]["benchmarks"]:
            assert set(bm) >= {"source", "what", "where", "kind", "closest"}
            assert bm["kind"] in ("null", "upper_bound", "design")
            assert set(bm["closest"]) <= set(st.STRATUM_RULES) and bm["closest"]
    for s in st.SOURCES.values():
        assert s["doi"].startswith("10.")



def test_output_with_a_recording_or_patient_identifier_is_refused():
    st.assert_no_identifiers({"a": [1, "ok", {"b": "jackknife_bh_keep40_r4"}]})
    for bad in ({"x": "Bonobo99999_9"}, {"x": ["Patient_999_XX"]}, {"Bonobo99999": 1}, {"x": "sub-99999"}):
        with pytest.raises(RuntimeError):
            st.assert_no_identifiers(bad)



def _shard(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True)
    with open(d / "rows00.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _lrow(key, j, split="analysis", dev=False, **kw):
    r = {"key": key, "j": j, "split": split, "dev": dev, "null": "ncmd_bdsp", "pipe": "cornell", "status": "ok",
         "resp": "none", "montage": "native", "variant": "v", "seq": True, "overlap": False, "uid": key,
         "patient": key}
    r.update(kw)
    return r


def test_loader_equals_115_load_rows_restricted_to_first_overlays(tmp_path):
    rows = [_lrow("a", 0, call=True), _lrow("a", 0, call=False), _lrow("b", 1), _lrow("c", 0, status="na: x"),
            {"key": "d", "null": "ncmd_bdsp", "pipe": "cornell", "status": "no_fit", "split": "analysis",
             "dev": False, "uid": "d", "patient": "d"}]
    _shard(tmp_path, "ncmd_bdsp__cornell__native__none", rows)
    _shard(tmp_path, "ncmd_bdsp__cornell__native__grid", [_lrow("e", 0, resp="grid")])
    got = st.load_first_overlays(str(tmp_path))
    ref = mg.load_rows("analysis", str(tmp_path))
    ref = ref[(ref["j"] == 0) & ref["dir"].str.endswith("__none")]
    assert list(got["key"]) == list(ref["key"]) == ["a", "c"]
    assert list(got["call"].iloc[:1]) == [True]


def test_loader_refuses_rows_of_another_split_or_a_dev_recording(tmp_path):
    _shard(tmp_path / "s1", "ncmd_bdsp__cornell__native__none", [_lrow("a", 1, split="dev")])
    with pytest.raises(RuntimeError):
        st.load_first_overlays(str(tmp_path / "s1"))
    _shard(tmp_path / "s2", "ncmd_bdsp__cornell__native__none", [_lrow("a", 0, dev=True)])
    with pytest.raises(RuntimeError):
        st.load_first_overlays(str(tmp_path / "s2"))



def test_cell_gives_the_reading_and_its_randomised_counterpart_on_the_same_stratum():
    rows = pd.DataFrame([
        dict(_row("a_0", "a", True, "normal_awake"), rand_call=False, rand_stratum="normal_awake",
             rand_transitioning=False, rand_extent_t0=0.0, rand_extent_t1=100.0, rand_a9_flag=False),
        dict(_row("b_0", "b", False, "normal_awake"), rand_call=True, rand_stratum="burst_suppression",
             rand_transitioning=False, rand_extent_t0=0.0, rand_extent_t1=100.0, rand_a9_flag=False),
        dict(_row("c_0", "c", True, "burst_suppression"), rand_call=False, rand_stratum="burst_suppression",
             rand_transitioning=False, rand_extent_t0=0.0, rand_extent_t1=100.0, rand_a9_flag=False)])
    c = st.cell(rows, "awake", B=50, seed_parts=("t",))
    assert (c["reading"]["k"], c["reading"]["n"]) == (1, 2)
    assert (c["randomised"]["k"], c["randomised"]["n"]) == (0, 1)
    i = st.cell(rows, "icu_typical", B=50, seed_parts=("t",))
    assert (i["reading"]["k"], i["reading"]["n"]) == (1, 1)
    assert (i["randomised"]["k"], i["randomised"]["n"]) == (1, 2)


def test_build_checks_every_stratum_against_each_benchmark_and_marks_the_closest_population():
    def frame(strata, calls, ld=None):
        return pd.DataFrame([dict(_row(f"p{i}_0", f"p{i}", c, s), lower_delta=None if ld is None else ld[i],
                                  rand_call=False, rand_stratum=s, rand_transitioning=False, rand_extent_t0=0.0,
                                  rand_extent_t1=100.0, rand_a9_flag=False, rand_lower_delta=None if ld is None
                                  else ld[i], rand_variant="g")
                             for i, (s, c) in enumerate(zip(strata, calls))])
    bd = frame(["normal_awake", "normal_awake", "burst_suppression"], [True, False, False])
    pdc = frame(["na", "na"], [False, True], ld=[True, True])
    rows = {("ncmd_bdsp", "edlow17", "dflt"): bd, ("ncmd_pdoc", "edlow17", "dflt"): pdc}
    choices = {("ncmd_bdsp", "edlow17"): ("vp", "dflt"), ("ncmd_pdoc", "edlow17"): ("vp", "dflt")}
    t = st.build(rows, {}, choices, {"cells": {}, "headline": {}}, B=20)
    e = [x for x in t if x["pipe"] == "edlow17"]
    assert len(e) == 1 and e[0]["kind"] == "default" and e[0]["variants"] == {"bdsp": "dflt", "pdoc": "dflt"}
    assert (e[0]["cells"]["awake"]["reading"]["k"], e[0]["cells"]["awake"]["reading"]["n"]) == (1, 2)
    assert (e[0]["cells"]["pdoc_wake_like"]["reading"]["k"], e[0]["cells"]["pdoc_wake_like"]["reading"]["n"]) == (1, 2)
    ch = e[0]["published"][0]["checks"]
    assert set(ch) == {"icu_typical", "awake", "pdoc_wake_like"}
    assert [s for s, v in ch.items() if v["closest_population"]] == st.PUBLISHED["edlow17"]["benchmarks"][0]["closest"]
    assert e[0]["published"][0]["doi"] == st.SOURCES["ED17"]["doi"]


def test_a_reading_with_different_variants_per_source_names_both():
    def frame(s):
        return pd.DataFrame([dict(_row("p_0", "p", False, s), lower_delta=True, rand_call=False, rand_stratum=s,
                                  rand_extent_t0=0.0, rand_extent_t1=100.0, rand_a9_flag=False,
                                  rand_lower_delta=True, rand_variant="g")])
    rows = {("ncmd_bdsp", "mbmi", "vp_b"): frame("normal_awake"), ("ncmd_pdoc", "mbmi", "vp_p"): frame("na")}
    choices = {("ncmd_bdsp", "mbmi"): ("vp_b", "d"), ("ncmd_pdoc", "mbmi"): ("vp_p", "d")}
    e = [x for x in st.build(rows, {}, choices, {"cells": {}, "headline": {}}, B=10) if x["pipe"] == "mbmi"][0]
    assert e["reading"] == "bdsp: vp_b; pdoc: vp_p"
    assert e["cells"]["awake"]["variant"] == "vp_b" and e["cells"]["pdoc_wake_like"]["variant"] == "vp_p"



def test_clopper_pearson_matches_the_closed_forms_and_known_values():
    assert st.clopper_pearson(0, 10) == pytest.approx([0.0, 1 - 0.025 ** (1 / 10)])
    assert st.clopper_pearson(10, 10) == pytest.approx([0.025 ** (1 / 10), 1.0])
    assert st.clopper_pearson(2, 26) == pytest.approx([0.0094554, 0.2513029], abs=1e-6)
    assert st.clopper_pearson(5, 20) == pytest.approx([0.0865715, 0.4910459], abs=1e-6)
    for k, n in ((2, 26), (5, 20), (1, 7)):
        lo, hi = st.clopper_pearson(k, n)
        assert __import__("scipy.stats").stats.binom.sf(k - 1, n, lo) == pytest.approx(0.025)
        assert __import__("scipy.stats").stats.binom.cdf(k, n, hi) == pytest.approx(0.025)
    lo, hi = st.clopper_pearson(1, 30, level=0.90)
    assert lo == pytest.approx(1 - 0.95 ** (1 / 30), rel=1e-2) and hi < st.clopper_pearson(1, 30)[1]
    with pytest.raises(ValueError):
        st.clopper_pearson(3, 2)
    assert st.clopper_pearson(0, 0) is None


def test_exact_one_sided_binomial_against_nominal():
    assert st.binom_vs_nominal(0, 10) == 1.0
    assert st.binom_vs_nominal(1, 10) == pytest.approx(1 - 0.95 ** 10)
    assert st.binom_vs_nominal(3, 10) == pytest.approx(
        1 - sum(__import__("math").comb(10, i) * 0.05 ** i * 0.95 ** (10 - i) for i in range(3)))
    assert st.binom_vs_nominal(2, 10, p0=0.5) == pytest.approx(1 - 11 / 1024)
    assert st.binom_vs_nominal(0, 0) == 1.0


def test_holm_is_the_step_down_adjustment_in_input_order():
    assert list(st.holm([0.01, 0.04, 0.03, 0.005])) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert list(st.holm([0.5, 0.9])) == pytest.approx([1.0, 1.0])
    assert len(st.holm([])) == 0


def test_exact_cell_gives_the_count_rate_cp_interval_and_one_sided_p():
    e = st.exact_cell(3, 40)
    assert (e["k"], e["n"]) == (3, 40) and e["rate"] == pytest.approx(0.075)
    assert e["ci95_cp"] == pytest.approx(st.clopper_pearson(3, 40))
    assert e["p_greater_nominal"] == pytest.approx(st.binom_vs_nominal(3, 40))
    assert st.exact_cell(0, 0) == {"k": 0, "n": 0, "rate": None, "ci95_cp": None, "p_greater_nominal": 1.0}
