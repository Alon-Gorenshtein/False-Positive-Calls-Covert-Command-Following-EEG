import glob
import importlib
import json
import os

import numpy as np
import pytest

wn = importlib.import_module("147_white_noise_eligible")
at, rn = wn.at, wn.rn


def _e(eligible, strict=None):
    e = {"axes": {"a": {"value": "x", "class": "STATED"}}, "stated_rule_eligible": eligible}
    if strict is not None:
        e["stated_rule_eligible_strict"] = strict
    return e



def test_eligible_variants_are_every_union_eligible_variant_and_strict_must_be_inside_union():
    cl = {"mbvt": {"a": _e(True, True), "b": _e(True, False), "c": _e(False, False)},
          "cornell": {"d": _e(False)}, "errata": [], "shortened_to_fit": {"variants": {}}}
    assert wn.eligible_variants(cl) == {"mbvt": ["a", "b"]}
    cl["mbvt"]["c"] = _e(False, True)
    with pytest.raises(RuntimeError, match="strict"):
        wn.eligible_variants(cl)


def test_the_variant_set_is_every_union_eligible_variant_with_its_reference_eligibility_beside():
    cl = {"mbvt": {"a": _e(True, True), "b": _e(True, False), "c": _e(False, False)}}
    ref = {"mbvt": {"a": _e(True), "b": _e(False), "c": _e(True)}, "gibson16": {"g": _e(True)}}
    vs = wn.variant_set(cl, ref)
    assert vs == {"mbvt": {"a": {"eligible_now": True, "eligible_at_ref": True, "strict_now": True},
                           "b": {"eligible_now": True, "eligible_at_ref": False, "strict_now": False}}}


def test_run_spec_maps_call_paper_c3_readings_and_frozen_variants():
    s = wn.run_spec("columbia", wn.CALL_PAPER)
    assert (s["kind"], s["arg"], s["call_field"]) == ("published", "as_coded", "call_paper")
    s = wn.run_spec("dubois", "c3_b_assessment_random_phase1")
    assert (s["kind"], s["arg"], s["reading"]) == ("c3", "assessment_random_phase1", "c3_b_assessment_random_phase1")
    s = wn.run_spec("mbmi", "or17_120_cspfold_mean57")
    assert (s["kind"], s["arg"], s["call_field"]) == ("published", "or17_120_cspfold_mean57", None)
    with pytest.raises(KeyError):
        wn.run_spec("mbmi", "no_such_variant")




def _atlas_row(pipe, name, r):
    for fn in sorted(glob.glob(os.path.join(at.SH, "rows*.jsonl"))):
        with open(fn) as fh:
            for line in fh:
                if f'"{pipe}|white|{r}|{name}"' in line:
                    return json.loads(line)
    return None


@pytest.mark.parametrize("pipe,variant,atlas_name", [("mbmi", "or17_120_cspfold_mean57", "or17_120_cspfold_mean57"),
                                                     ("cornell", "c3_jackknife_bh_keep40_r4_om2contig",
                                                      "c3:c3_jackknife_bh_keep40_r4_om2contig"),
                                                     ("dubois", "c3_b_assessment_random_phase1",
                                                      "c3:c3_b_assessment_random_phase1")])
def test_a_paired_unit_reproduces_the_atlas_row_of_the_same_variant(pipe, variant, atlas_name):
    ref = _atlas_row(pipe, atlas_name, 0)
    if ref is None:
        pytest.skip("atlas shard rows not on this disk")
    p = wn.plan(pipe, [variant])
    rows = []
    wn.run_unit(pipe, 0, wn.R_for(pipe, p["paired"]), p["atlas_need_s"], "paired", set(), rows.append)
    (row,) = rows
    assert row["status"] == ref["status"] == "ok" and row["call"] == ref["call"]
    assert row["p"] == pytest.approx(ref["p"], rel=1e-9, abs=1e-12)
    assert row["stat"] == pytest.approx(ref["stat"], rel=1e-9, abs=1e-12)
    assert row["draw"] == "paired" and row["need_s"] == p["atlas_need_s"]
    rows2 = []
    wn.run_unit(pipe, 0, wn.R_for(pipe, p["paired"]), p["atlas_need_s"], "paired", {row["key"]}, rows2.append)
    assert rows2 == []


def test_call_paper_rows_take_the_papers_call_and_refuse_a_missing_field():
    assert wn.call_of({"call": True, "extra": {"call_paper": False}}, "call_paper") is False
    assert wn.call_of({"call": True, "extra": {}}, None) is True
    with pytest.raises(RuntimeError, match="call_paper"):
        wn.call_of({"call": True, "extra": {}}, "call_paper")



def _row(pipe, v, r, call, status="ok", draw="paired"):
    return {"key": f"{pipe}|white|{r}|{draw}|{v}", "pipe": pipe, "variant": v, "r": r, "draw": draw, "need_s": 100,
            "status": status, "call": call if status == "ok" else None}


def test_merge_gives_exact_intervals_without_na_rows_labels_draws_and_counts_atlas_agreement():
    rows = [_row("mbmi", "v", r, r < 3) for r in range(58)] + [_row("mbmi", "v", 58, None, "na: no_usable_trials"),
                                                                  _row("mbmi", "v", 59, None, "error: X")]
    rows += [_row("chennu13", "b20", r, False, draw="unpaired") for r in range(60)]
    atlas = {f"mbmi|white|{r}|v": {"status": "ok", "call": r < 3 if r != 7 else True} for r in range(30)}
    t = wn.summarise(rows, atlas, {("mbmi", "v"): "v", ("chennu13", "b20"): None})
    v = t["mbmi"]["v"]
    assert (v["k"], v["n"], v["n_na"], v["n_error"]) == (3, 58, 1, 1)
    assert v["ci95_cp"] == pytest.approx(wn.st.clopper_pearson(3, 58))
    assert v["draw"] == "paired with the atlas draw"
    assert v["atlas_agreement"] == {"n_compared": 30, "n_same_call": 29}
    b = t["chennu13"]["b20"]
    assert b["draw"].startswith("unpaired") and b["atlas_agreement"] is None and (b["k"], b["n"]) == (0, 60)



def test_swap_is_read_from_sysctl_and_the_run_stops_above_2_gb():
    s = "vm.swapusage: total = 12288.00M  used = 11797.19M  free = 490.81M  (encrypted)"
    assert wn.swap_used_mb(s) == pytest.approx(11797.19)
    assert wn.swap_used_mb("vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M") == 0.0
    with pytest.raises(SystemExit):
        wn.check_swap(s)
    wn.check_swap("vm.swapusage: total = 4096.00M  used = 2047.00M  free = 2049.00M")


def test_rows_never_go_to_the_bench_or_atlas_shards():
    assert os.path.abspath(wn.SH) not in (os.path.abspath(at.SH),)
    assert wn.shard_file(0).startswith(os.path.abspath(wn.SH) + os.sep)
    with pytest.raises(RuntimeError):
        wn.shard_file(0, root=at.SH)
    with pytest.raises(RuntimeError):
        wn.shard_file(0, root=wn.fr.BENCH_SHARDS[0])


def test_on_linux_the_guard_is_available_memory_not_swap():
    mi = "MemTotal: 126000000 kB\nMemFree: 1000 kB\nMemAvailable: 9437184 kB\nSwapTotal: 16000000 kB\nSwapFree: 1 kB\n"
    assert wn.mem_available_gb(mi) == pytest.approx(9.0)
    assert wn.check_memory(platform="linux", meminfo=mi) == pytest.approx(9.0)
    with pytest.raises(SystemExit):
        wn.check_memory(platform="linux", meminfo=mi.replace("9437184", "7340032"))
    with pytest.raises(SystemExit):
        wn.check_memory(platform="darwin", swap="vm.swapusage: total = 4096.00M  used = 3000.00M  free = 1.00M")
