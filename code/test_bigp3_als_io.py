"""Tests for code/bigp3_als_io.py (BigP3BCI, PhysioNet 10.13026/0byy-ry86, CC BY 4.0; ALS studies B, F, L, N).
Synthetic EDF-like arrays for the parser; one real-patient smoke test (reads the other study's cache, writes to tmp).
cd code && /usr/bin/python3 -m pytest test_bigp3_als_io.py -q"""
import importlib
import os

import numpy as np
import pytest

io = importlib.import_module("bigp3_als_io")
FS = 256.0


def synth_run(n_seq=2, per_seq=12, soa=64, dur=8, n_chars=4, gap=900, lead=200, seed=0, targets_per_seq=3):
    """EDF-like channel arrays: StimulusBegin pulses of `dur` samples every `soa` samples inside PhaseInSequence == 2
    segments, StimulusType 1 on target flashes, one 0/1 channel per character, CurrentTarget = target character index."""
    rng = np.random.default_rng(seed)
    n = lead + n_seq * (per_seq * soa + gap) + 500
    sb, st, ph, sc, ct = (np.zeros(n) for _ in range(5))
    chars = np.zeros((n_chars, n))
    t = lead
    for s in range(n_seq):
        ph[t - 50:t + per_seq * soa + 50] = 2
        tgt_char = int(rng.integers(1, n_chars + 1))
        flags = np.zeros(per_seq, int)
        flags[rng.choice(per_seq, targets_per_seq, replace=False)] = 1
        for k in range(per_seq):
            a = t + k * soa
            sb[a:a + dur] = 1
            sc[a:a + dur] = 100 + s * per_seq + k
            ct[a:a + dur] = tgt_char
            st[a:a + dur] = flags[k]
            on = {tgt_char - 1} if flags[k] else set()
            on |= {int(x) for x in rng.choice([c for c in range(n_chars) if c != tgt_char - 1], 1)}
            for c in on:
                chars[c, a:a + dur] = 1
        t += per_seq * soa + gap
    eeg = rng.standard_normal((7, n))
    return {"eeg": eeg, "sb": sb, "st": st, "ph": ph, "sc": sc, "ct": ct, "chars": chars}



def test_flash_onsets_are_the_rising_edges_even_when_the_first_sample_is_high():
    sb = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], float)
    assert list(io.flash_onsets(sb)) == [0, 4, 8]
    assert list(io.flash_lengths(sb)) == [2, 3, 1]


def test_sequences_are_the_contiguous_phase_2_stretches_and_every_flash_must_sit_in_one():
    ph = np.array([0, 2, 2, 2, 3, 0, 2, 2, 0], float)
    a, b = io.sequences(ph)
    assert list(a) == [1, 6] and list(b) == [3, 7]
    assert list(io.assign_sequence(np.array([1, 3, 6]), a, b)) == [0, 0, 1]
    with pytest.raises(ValueError, match="outside"):
        io.assign_sequence(np.array([1, 4]), a, b)


def test_resolve_channels_reports_every_missing_site_then_raises():
    names = ["EEG_Fz", "EEG_C3", "EEG_Cz", "EEG_C4", "EEG_P3", "EEG_Pz", "EEG_P4", "EEG_Oz"]
    assert io.resolve_channels(names) == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError) as e:
        io.resolve_channels([n for n in names if n not in ("EEG_C3", "EEG_P4")])
    assert "C3" in str(e.value) and "P4" in str(e.value)



def test_build_run_gives_onsets_target_flags_sequences_and_identity():
    r = synth_run()
    run = io.build_run(**r, fs=FS)
    assert len(run["onset"]) == 24 and run["is_target"].sum() == 6 and run["is_target"].dtype == bool
    assert list(run["seq"]) == [0] * 12 + [1] * 12
    assert run["onset"][1] - run["onset"][0] == 64
    assert run["chars"].shape == (24, 4) and run["code"][0] == 100 and run["code"][12] == 112
    assert run["x"].shape[0] == 7 and run["x"].dtype == np.float32


def test_build_run_meta_has_soa_flash_duration_and_per_sequence_counts():
    m = io.run_meta(io.build_run(**synth_run(per_seq=10, targets_per_seq=2), fs=FS), fs=FS)
    assert m["soa_s"] == pytest.approx(64 / FS) and m["flash_ms"] == pytest.approx(8 / FS * 1000)
    assert m["flashes_per_sequence"] == [10, 10] and m["targets_per_sequence"] == [2, 2] and m["n_sequences"] == 2


def test_build_run_refuses_a_target_flag_that_no_single_character_explains():
    r = synth_run()
    first = np.flatnonzero(r["st"] > 0)[0]
    r["st"][first: first + 8] = 0
    with pytest.raises(ValueError, match="StimulusType"):
        io.build_run(**r, fs=FS)


def test_a_current_target_that_disagrees_with_the_flags_is_counted_not_trusted():
    """B_01 SE003 Test02 / Test05: CurrentTarget says 21 for one character trial but the flagged flashes light character 25.
    StimulusType (the flash's own label) is the target flag; the disagreement is recorded."""
    r = synth_run(n_chars=6)
    on = np.flatnonzero(r["sb"] > 0.5)
    s1 = on[on > on[11]]
    tc = int(r["ct"][s1[0]])
    r["ct"][r["ph"] == 2] = np.where(r["ct"][r["ph"] == 2] == tc, 6 if tc != 6 else 5, r["ct"][r["ph"] == 2])
    run = io.build_run(**r, fs=FS)
    assert run["n_current_target_mismatch_seq"] >= 1
    assert run["is_target"].sum() == 6
    assert io.run_meta(run, FS)["n_current_target_mismatch_seq"] == run["n_current_target_mismatch_seq"]


def test_clean_run_has_no_current_target_mismatch():
    assert io.build_run(**synth_run(), fs=FS)["n_current_target_mismatch_seq"] == 0


def test_build_run_refuses_a_flash_outside_a_sequence_and_unequal_flash_lengths():
    r = synth_run()
    r["ph"][:] = 0
    with pytest.raises(ValueError, match="outside"):
        io.build_run(**r, fs=FS)
    r = synth_run()
    a = np.flatnonzero(r["sb"] > 0)[0]
    r["sb"][a + 8] = 1
    r["chars"][:, a + 8] = r["chars"][:, a]
    r["st"][a + 8] = r["st"][a]
    with pytest.raises(ValueError, match="length"):
        io.build_run(**r, fs=FS)



def test_path_info_parses_study_subject_session_phase_and_condition():
    p = "StudyF/F_03/SE002/Test/DynBigram/F_03_SE002_DynBigram_Test04.edf"
    assert io.path_info(p) == {"study": "F", "subject": "F_03", "session": "SE002", "phase": "Test", "paradigm": "DynBigram",
                               "file": "F_03_SE002_DynBigram_Test04.edf"}
    with pytest.raises(ValueError):
        io.path_info("StudyF/F_03/SE002/Valid/Dyn/x.edf")


def test_list_edfs_skips_apple_double_files_and_other_subjects(tmp_path):
    d = tmp_path / "StudyB" / "B_01" / "SE001" / "Train" / "CB"
    d.mkdir(parents=True)
    for n in ("B_01_SE001_CB_Train02.edf", "B_01_SE001_CB_Train01.edf", "._B_01_SE001_CB_Train01.edf"):
        (d / n).write_bytes(b"x")
    o = tmp_path / "StudyB" / "B_02" / "SE001" / "Train" / "CB"
    o.mkdir(parents=True)
    (o / "B_02_SE001_CB_Train01.edf").write_bytes(b"x")
    got = io.list_edfs("B", "B_01", src=str(tmp_path))
    assert [os.path.basename(g) for g in got] == ["B_01_SE001_CB_Train01.edf", "B_01_SE001_CB_Train02.edf"]
    assert io.list_edfs("B", "B_09", src=str(tmp_path)) == []


def test_output_path_and_the_source_tree_is_never_a_write_target(tmp_path):
    assert io.npz_path("B", "B_01", out="/o") == "/o/B_B_01.npz"
    with pytest.raises(ValueError, match="source"):
        io.check_out_not_in_source(os.path.join(io.SRC, "x"))
    assert io.check_out_not_in_source(str(tmp_path)) is None



REAL = os.path.join(io.SRC, "StudyN")


@pytest.mark.skipif(not os.path.isdir(REAL), reason="BigP3BCI cache not on disk")
def test_real_patient_extract_then_flashes_round_trip(tmp_path):
    subj = sorted(s for s in os.listdir(REAL) if not s.startswith("._"))[0]
    path = io.extract("N", subj, out=str(tmp_path))
    runs = io.flashes("N", subj, out=str(tmp_path))
    assert os.path.exists(path) and len(runs) == len(io.list_edfs("N", subj))
    r = runs[0]
    assert r["x"].shape[0] == 7 and r["x"].dtype == np.float32 and 255.9 < r["fs"] < 256.1
    assert r["is_target"].sum() > 0 and r["is_target"].mean() < 0.2
    assert r["meta"]["soa_s"] == pytest.approx(0.25) and r["meta"]["flash_ms"] == pytest.approx(31.25)
    assert r["meta"]["phase"] in ("Train", "Test") and r["meta"]["paradigm"] in ("Dry", "Wet")
    assert 1.0 < np.std(r["x"]) < 5000.0
    assert r["onset"].min() >= 0 and r["onset"].max() < r["x"].shape[1]
