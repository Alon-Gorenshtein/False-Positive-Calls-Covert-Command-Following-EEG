import montage


def test_exact_names_map_to_themselves_case_insensitively():
    ch = ["FP1", "FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8", "O1"]
    assert [ch[i] for i in montage.kit8_indices(ch)] == ["FZ", "C3", "CZ", "C4", "PZ", "PO7", "OZ", "PO8"]


def test_missing_positions_take_the_nearest_unused_electrode():
    ch = ["Fz", "C3", "Cz", "C4", "Pz", "PO3", "PO4", "PO9", "PO10", "Oz", "O1", "O2"]
    names = [ch[i] for i in montage.kit8_indices(ch)]
    assert len(set(names)) == 8 and names[5] in ("PO3", "PO9", "O1") and names[7] in ("PO4", "PO10", "O2")


def test_small_montage_is_used_whole_and_old_names_alias():
    assert montage.kit8_indices(["Cz", "Oz", "P4", "P3", "PO8", "Pz", "PO7"]) == list(range(7))
    assert montage.canonical("T5") == "P7"


def test_excluded_channels_are_never_picked():
    ch = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "PO5", "Oz", "O1", "PO8", "P8"]
    assert "PO7" not in [ch[i] for i in montage.kit8_indices(ch, exclude=["PO7", "Oz"])]
