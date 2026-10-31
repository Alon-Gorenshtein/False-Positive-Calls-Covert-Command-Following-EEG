"""Tests of 178_long_schedules.py (no patient data)."""
import importlib

ls = importlib.import_module("178_long_schedules")
b = ls.b


def test_needs_are_the_primary_readings_own():
    assert b.need_seconds(b.load_pipe("cruse11"), ["cr11_b4_hilbert_ci99"], "bdsp") == ls.NEED["cruse11"] == 2186
    assert b.need_seconds(b.load_pipe("cruse12p"), ["random_perclass"], "bdsp") == ls.NEED["cruse12p"] == 1891


def test_native_chennu_runs_twenty_blocks_and_restores():
    mod = b.load_pipe("chennu13")
    before = (dict(b.SOURCE_VARIANTS), dict(b.PRIMARY_OVERRIDE), b._jobs, b._rand_fields)
    with ls.native_chennu(set(ls.CHENNU_ELIGIBLE)):
        native = [v for v, s in b.variant_specs(mod, "bdsp").items() if not s["short"]]
        assert set(ls.CHENNU_ELIGIBLE) <= set(native) and all(v.endswith("_b20") for v in native)
        assert b.need_seconds(mod, list(ls.CHENNU_ELIGIBLE), "bdsp") == ls.NEED["chennu13"]
    assert (dict(b.SOURCE_VARIANTS), dict(b.PRIMARY_OVERRIDE), b._jobs, b._rand_fields) == before


def test_frozen_only_restricts_to_one_reading_and_restores():
    mod = b.load_pipe("cruse11")
    vs0 = b.variant_specs
    with ls.frozen_only("cruse11", "cr11_b4_hilbert_ci99"):
        assert list(b.variant_specs(mod, "bdsp")) == ["cr11_b4_hilbert_ci99"]
        assert len(b.variant_specs(b.load_pipe("cruse12p"), "bdsp")) > 1
    assert b.variant_specs is vs0
    with ls.frozen_only("chennu13", "p1_samples10_zvar_b20"):
        assert list(b.variant_specs(b.load_pipe("chennu13"), "bdsp")) == ["p1_samples10_zvar_b20"]
    assert b.SOURCE_VARIANTS[("chennu13", "bdsp")] == {"n_blocks": 12}


def test_cell_counts_first_overlay_of_the_frozen_reading():
    rows = [{"uid": "a", "status": "ok", "variant": "v", "j": 0, "call": True},
            {"uid": "b", "status": "ok", "variant": "v", "j": 0, "call": False},
            {"uid": "c", "status": "no_fit", "variant": None},
            {"uid": "d", "status": "ok", "variant": "w", "j": 0, "call": True}]
    c = ls.cell(rows, "v")
    assert (c["k"], c["n"]) == (1, 2) and c["windows_run"] == 4 and c["status"]["no_fit"] == 1


def test_cell_refuses_a_missing_call():
    import pytest
    with pytest.raises(RuntimeError):
        ls.cell([{"uid": "a", "status": "ok", "variant": "v", "j": 0, "call": None}], "v")
