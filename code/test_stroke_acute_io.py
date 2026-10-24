"""Tests for code/stroke_acute_io.py (Wei/Liu 2023 acute-stroke hand MI, figshare 21679035, CC BY 4.0).
Synthetic fixtures first; one smoke test reads one real subject if it is on disk.
cd code && /usr/bin/python3 -m pytest test_stroke_acute_io.py -q"""
import importlib
import os

import numpy as np
import pytest

sa = importlib.import_module("stroke_acute_io")

EVENTS = ("onset\tduration\ttrial_type\tresponse_time\tvalue\tstim_file\n"
          "1\t2000\t1\tN/A\t1\tstim1.png\n2001\t4000\t1\tN/A\t2\tstim2.mp4\n6001\t2000\t1\tN/A\t3\tstim3.png\n"
          "8001\t2000\t2\tN/A\t1\tstim1.png\n10001\t4000\t2\tN/A\t2\tstim2.mp4\n14001\t2000\t2\tN/A\t3\tstim3.png\n")


def test_labels_are_the_trial_types_of_the_motor_imagery_rows():
    assert sa.labels_from_events(EVENTS) == ["left", "right"]


def test_labels_refuse_an_unknown_trial_type():
    with pytest.raises(ValueError, match="trial_type"):
        sa.labels_from_events(EVENTS.replace("\t2\tN/A\t2\t", "\t7\tN/A\t2\t"))


def test_mi_onsets_are_the_rising_edges_of_code_2_only():
    mk = np.zeros(100)
    mk[[0, 20, 50, 60, 90]] = [1, 2, 3, 1, 2]
    assert list(sa.mi_onsets(mk)) == [20, 90]


def test_mi_onsets_tolerate_a_pulse_wider_than_one_sample():
    mk = np.zeros(50)
    mk[10:13] = 2
    assert list(sa.mi_onsets(mk)) == [10]


def test_grid_onsets_pick_the_code_2_edge_that_is_followed_by_code_3_one_MI_duration_later():
    mk = np.zeros(400)
    lead = [25, 40, 1, 10]
    for k in range(4):
        mk[100 * k] = 1
        mk[100 * k + lead[k]] = 2
        mk[100 * k + lead[k] + 50] = 3
    mk[100 * 0 + 60] = 2
    mk[100 * 2 + 70] = 3
    mk[100 * 3 + 80] = 2
    assert list(sa.mi_onsets_grid(mk, period=100, dur=50, tol=3, n=4)) == [25, 140, 201, 310]


def test_grid_onsets_refuse_a_missing_or_doubled_marker_and_a_short_recording():
    mk = np.zeros(300)
    mk[[25, 75, 125, 175]] = [2, 3, 2, 3]
    with pytest.raises(ValueError, match="trial 2"):
        sa.mi_onsets_grid(mk, period=100, dur=50, tol=3, n=3)
    mk[[210, 260, 220, 270]] = [2, 3, 2, 3]
    with pytest.raises(ValueError, match="trial 2"):
        sa.mi_onsets_grid(mk, period=100, dur=50, tol=3, n=3)
    with pytest.raises(ValueError, match="shorter"):
        sa.mi_onsets_grid(np.zeros(250), period=100, dur=50, tol=3, n=3)


def test_epochs_cut_the_four_seconds_after_each_onset_and_drop_none_silently():
    x = np.tile(np.arange(40.0), (2, 1))
    E = sa.cut(x, [3, 10], 5)
    assert E.shape == (2, 2, 5) and list(E[0, 0]) == [3, 4, 5, 6, 7] and list(E[1, 1]) == [10, 11, 12, 13, 14]
    with pytest.raises(ValueError, match="past the end"):
        sa.cut(x, [3, 38], 5)


def test_reference_channel_and_eog_are_not_eeg():
    names = ["FP1", "C3", "CPz", "HEOL", "HEOR", ""]
    assert sa.eeg_rows(names, np.array([5.0, 4.0, 0.0, 9.0, 9.0, 0.06])) == [0, 1]


def test_eeg_rows_refuse_a_flat_channel_that_is_not_the_reference():
    with pytest.raises(ValueError, match="flat"):
        sa.eeg_rows(["FP1", "C3", "CPz"], np.array([5.0, 0.0, 0.0]))


def test_channel_groups_are_fc_c_cp_and_occipital_without_the_reference():
    names = ["FP1", "FCz", "FC3", "FT7", "C3", "Cz", "CP3", "TP7", "Oz", "O1", "O2", "T3"]
    assert [names[i] for i in sa.group_idx(names, "sensorimotor")] == ["FCz", "FC3", "C3", "Cz", "CP3"]
    assert [names[i] for i in sa.group_idx(names, "occipital")] == ["Oz", "O1", "O2"]


def test_the_unpacked_edf_path_is_inside_the_destination(tmp_path):
    with pytest.raises(ValueError):
        sa.safe_member("edffile/../../x.edf")
    assert sa.safe_member("edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf")


EDF = os.path.join(sa.RAW, "edffile", "sub-01", "eeg", "sub-01_task-motor-imagery_eeg.edf")


@pytest.mark.skipif(not os.path.exists(EDF), reason="acute stroke set not downloaded")
def test_real_subject_gives_40_balanced_4s_epochs_at_250_hz():
    X, y, ch = sa.trials(1)
    assert X.shape == (40, 29, 1000) and X.dtype == np.float32 and "CPz" not in ch
    assert y.count("left") == y.count("right") == 20
    assert 1.0 < X.std() < 500.0


def test_licence_must_be_cc_by_4():
    sa.check_licence({"license": {"name": "CC BY 4.0"}})
    with pytest.raises(ValueError, match="licence"):
        sa.check_licence({"license": {"name": "CC BY-NC 4.0"}})


def test_wanted_skips_only_the_duplicate_raw_trial_archive():
    files = [{"name": n} for n in ("edffile.zip", "sourcedata.zip", "stimuli.zip", "README.txt")]
    assert [f["name"] for f in sa.wanted(files)] == ["edffile.zip", "stimuli.zip", "README.txt"]
    assert [f["name"] for f in sa.wanted(files, skip=())] == ["edffile.zip", "sourcedata.zip", "stimuli.zip", "README.txt"]
