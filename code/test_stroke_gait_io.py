"""Tests for code/stroke_gait_io.py (Liu 2025 gait MI after stroke, Zenodo 18987384 re-host of figshare 27130299).
Synthetic fixtures first; one smoke test reads one real subject if its npz is on disk.
cd code && /usr/bin/python3 -m pytest test_stroke_gait_io.py -q"""
import hashlib
import importlib
import os

import numpy as np
import pytest

sg = importlib.import_module("stroke_gait_io")
pc = importlib.import_module("118_positive_control")



def _fake(blob, short=0):
    calls = {"n": 0}

    def fetch(url, a, b):
        calls["n"] += 1
        out = blob[a:b + 1]
        return out[:-1] if short and calls["n"] <= short else out
    return fetch, calls


def test_download_reassembles_the_file_and_checks_md5(tmp_path):
    blob = np.random.default_rng(0).bytes(10_000)
    fetch, _ = _fake(blob)
    p = sg.download("u", str(tmp_path / "f.bin"), len(blob), hashlib.md5(blob).hexdigest(), conns=3, chunk=1500,
                    fetch=fetch, min_free=0)
    assert open(p, "rb").read() == blob and not os.path.exists(p + ".part")


def test_download_refuses_a_wrong_md5_and_leaves_no_final_file(tmp_path):
    blob = np.random.default_rng(1).bytes(4_000)
    fetch, _ = _fake(blob)
    with pytest.raises(IOError, match="md5"):
        sg.download("u", str(tmp_path / "f.bin"), len(blob), "0" * 32, conns=2, chunk=1000, fetch=fetch, min_free=0)
    assert not os.path.exists(tmp_path / "f.bin")


def test_download_retries_a_short_read_then_succeeds_but_gives_up_if_it_persists(tmp_path):
    blob = np.random.default_rng(2).bytes(3_000)
    fetch, calls = _fake(blob, short=1)
    p = sg.download("u", str(tmp_path / "a.bin"), len(blob), hashlib.md5(blob).hexdigest(), conns=1, chunk=1000,
                    fetch=fetch, min_free=0, tries=3)
    assert open(p, "rb").read() == blob and calls["n"] == 4
    always_short, _ = _fake(blob, short=10 ** 6)
    with pytest.raises(IOError, match="short"):
        sg.download("u", str(tmp_path / "b.bin"), len(blob), None, conns=1, chunk=1000, fetch=always_short,
                    min_free=0, tries=2)


def test_download_stops_when_the_disk_floor_would_be_breached(tmp_path):
    fetch, _ = _fake(b"x" * 100)
    with pytest.raises(RuntimeError, match="free"):
        sg.download("u", str(tmp_path / "f.bin"), 100, None, fetch=fetch, min_free=10 ** 18)


def test_download_reuses_a_complete_verified_file(tmp_path):
    blob = b"abc" * 100
    f = tmp_path / "f.bin"
    f.write_bytes(blob)

    def boom(*a):
        raise AssertionError("must not download")
    assert sg.download("u", str(f), len(blob), hashlib.md5(blob).hexdigest(), fetch=boom, min_free=0) == str(f)



def test_resolve_maps_the_ten_ten_names_to_the_old_ones():
    file_names = [{"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(c, c) for c in pc.MOTOR_CHANNELS]
    file_names += ["ECG", "HEOL"]
    idx = sg.resolve(file_names)
    assert [file_names[i] for i in idx][pc.MOTOR_CHANNELS.index("T7")] == "T3"
    assert [file_names[i] for i in idx][pc.MOTOR_CHANNELS.index("P8")] == "T6"
    assert len(idx) == len(sg.channels()) == len(pc.MOTOR_CHANNELS) - 1


def test_resolve_tolerates_only_the_known_absent_cpz():
    assert "CPz" not in sg.channels() and len(sg.channels()) == len(pc.MOTOR_CHANNELS) - 1
    have = [c for c in pc.MOTOR_CHANNELS if c != "CPz"]
    assert len(sg.resolve(have)) == len(have)
    with pytest.raises(ValueError, match="Cz"):
        sg.resolve([c for c in have if c != "Cz"])


def test_resolve_is_case_insensitive_and_reports_every_missing_channel():
    names = [c.upper() for c in sg.channels() if c not in ("CP3", "C5")]
    with pytest.raises(ValueError) as e:
        sg.resolve(names)
    assert "CP3" in str(e.value) and "C5" in str(e.value)



FS = 250.0


def _trial(t0, kind):
    """One trial's markers (sample at 250 Hz, code) from baseline time t0 s: kind 'mi' or 'idle'."""
    b, c, o = (1, 2, 3) if kind == "mi" else (7, 8, 9)
    return [(int(t0 * FS), b), (int((t0 + 3) * FS), c), (int((t0 + 4) * FS), o)]


def _run_events(kinds, gap=12.0, t0=0.5):
    ev = []
    for k, kind in enumerate(kinds):
        ev += _trial(t0 + gap * k, kind)
    return np.array(ev, int)


def test_sources_are_imagery_idle_and_fixation_only_with_the_right_windows():
    ev = _run_events(["mi", "idle", "idle", "mi"])
    n = int((0.5 + 12 * 4 + 1) * FS)
    src = sg.sources(ev, FS, n, run=2, tag="pre")
    by = {c: [s for s in src if s.cls == c] for c in ("mi", "idle", "fixation")}
    assert [len(by[c]) for c in ("mi", "idle", "fixation")] == [2, 2, 4]
    for s in by["mi"] + by["idle"]:
        assert s.n == int(5 * FS) and s.run == 2 and s.tag.startswith("pre:")
    for s in by["fixation"]:
        assert s.n == int(3 * FS)
    assert by["mi"][0].start == ev[2, 0] and by["idle"][0].start == ev[5, 0]


def test_no_source_reaches_into_the_post_trial_relaxation():
    """Relaxation of a trial = [onset + 5 s, next baseline marker). No source may overlap it."""
    ev = _run_events(["mi", "idle", "mi"])
    src = sg.sources(ev, FS, int(60 * FS), run=0, tag="pre")
    onsets = [int(e[0]) for e in ev if e[1] in (3, 9)]
    bases = [int(e[0]) for e in ev if e[1] in (1, 7)] + [10 ** 9]
    for o, nxt in zip(onsets, bases[1:]):
        lo, hi = o + int(5 * FS), nxt
        for s in src:
            assert s.start + s.n <= lo or s.start >= hi, (s, lo, hi)
    assert {s.cls for s in src} == {"mi", "idle", "fixation"}


def test_a_trial_cut_off_by_the_end_of_the_run_is_dropped_not_padded():
    ev = _run_events(["mi", "idle"])
    n = int(ev[-1, 0] + 2 * FS)
    src = sg.sources(ev, FS, n, run=0, tag="pre")
    assert [s.cls for s in src if s.cls != "fixation"] == ["mi"]


def test_phase_markers_inside_the_task_period_are_ignored_not_refused():
    """Every session, the pure-MI ones included, carries markers 4-6 / 10-12 at +1.0 / +1.6 / +3.4 s into the task
    (verified on sub-01 ses-pre: S 10, S 11, S 12 follow every idle onset); they are timestamps, not stimulation."""
    ev = _run_events(["mi", "idle"])
    extra = []
    for o in ev[ev[:, 1] == 3, 0]:
        extra += [(o + int(t * FS), c) for t, c in ((1.0, 4), (1.6, 5), (3.4, 6))]
    for o in ev[ev[:, 1] == 9, 0]:
        extra += [(o + int(t * FS), c) for t, c in ((1.0, 10), (1.6, 11), (3.4, 12))]
    both = np.array(sorted(map(tuple, np.vstack([ev, extra]).tolist())))
    a = sg.sources(ev, FS, int(40 * FS), run=0, tag="pre")
    b = sg.sources(both, FS, int(40 * FS), run=0, tag="pre")
    assert [(s.start, s.n, s.cls) for s in a] == [(s.start, s.n, s.cls) for s in b]


def test_a_trial_that_begins_before_the_recording_keeps_its_task_excerpt_but_has_no_fixation():
    """sub-03 pre run 3 starts at the cue (baseline marker is before sample 0): imagery is intact, fixation is not."""
    late = _run_events(["idle"], t0=0.0)[1:]
    late = late - np.array([[int(2.5 * FS), 0]] * len(late))
    src = sg.sources(late, FS, int(30 * FS), run=0, tag="pre")
    assert [s.cls for s in src] == ["idle"]
    mid = _run_events(["mi", "idle"])
    mid = mid[mid[:, 1] != 7]
    with pytest.raises(ValueError, match="baseline"):
        sg.sources(mid, FS, int(40 * FS), run=0, tag="pre")


def test_sources_refuse_a_missing_or_misplaced_cue():
    bad = _run_events(["mi"])[[0, 2]]
    with pytest.raises(ValueError, match="cue"):
        sg.sources(bad, FS, int(30 * FS), run=0, tag="pre")
    shifted = _run_events(["mi"])
    shifted[2, 0] += int(0.6 * FS)
    with pytest.raises(ValueError, match="cue"):
        sg.sources(shifted, FS, int(30 * FS), run=0, tag="pre")


def test_only_pure_mi_sessions_are_accepted():
    for s in ("pre", "post", "follow"):
        assert sg.pure_session(s) == s
    for s in ("mises", "miies", "MI-SES", "MI-IES"):
        with pytest.raises(ValueError, match="stimulation"):
            sg.pure_session(s)


def test_split_runs_cuts_at_the_boundaries_and_shifts_events():
    x = np.arange(30.0).reshape(1, 30).repeat(2, 0)
    bounds = np.array([0, 10, 30])
    ev = np.array([[3, 1], [12, 2], [25, 3]])
    runs, evs = sg.split_runs(x, bounds, ev)
    assert [r.shape[1] for r in runs] == [10, 20]
    assert evs[0].tolist() == [[3, 1]] and evs[1].tolist() == [[2, 2], [15, 3]]



def test_marker_codes_come_from_the_brainvision_descriptions():
    assert sg.code_of("Stimulus/S  7") == 7 and sg.code_of("Stimulus/S 12") == 12 and sg.code_of("Stimulus/S  1") == 1
    assert sg.code_of("New Segment/") is None and sg.code_of("Comment/x") is None


def test_pack_concatenates_runs_and_offsets_the_events():
    r1 = (np.zeros((2, 10), np.float32), np.array([[3, 1], [5, 2]]))
    r2 = (np.ones((2, 6), np.float32), np.array([[1, 7]]))
    x, bounds, ev = sg.pack([r1, r2])
    assert x.shape == (2, 16) and list(bounds) == [0, 10, 16] and ev.tolist() == [[3, 1], [5, 2], [11, 7]]
    with pytest.raises(ValueError, match="channel"):
        sg.pack([r1, (np.zeros((3, 4), np.float32), np.zeros((0, 2), int))])


def _fake_npz(root, subj, session, n_runs=2, fs=FS):
    """Two 60-s runs, each with 2 MI and 2 idle trials, channels = MOTOR_CHANNELS, noise + a run-specific offset."""
    rng = np.random.default_rng(subj)
    runs = []
    for r in range(n_runs):
        n = int(60 * fs)
        ev = _run_events(["mi", "idle", "idle", "mi"], gap=12.0, t0=1.0)
        runs.append((rng.standard_normal((len(sg.channels()), n)).astype(np.float32) + r, ev))
    x, bounds, ev = sg.pack(runs)
    os.makedirs(root, exist_ok=True)
    np.savez(os.path.join(root, f"sub-{subj:02d}_{session}.npz"), x=x, names=np.array(sg.channels()), fs=fs,
             bounds=bounds, events=ev)


def test_person_is_a_118_person_with_run_split_sources_and_session_provenance(tmp_path):
    root = str(tmp_path)
    _fake_npz(root, 3, "pre")
    _fake_npz(root, 3, "post")
    p = sg.person(3, ["pre", "post"], root=root)
    assert isinstance(p, pc.Person) and p.pid == "Liu_S03" and p.fs == FS and p.names == sg.channels()
    assert len(p.runs) == 4 and all(r.shape[0] == len(sg.channels()) for r in p.runs)
    assert sorted({s.cls for s in p.sources}) == ["fixation", "idle", "mi"]
    assert sum(s.cls == "mi" for s in p.sources) == 8 and sum(s.cls == "idle" for s in p.sources) == 8
    assert {s.tag.split(":")[0] for s in p.sources} == {"pre", "post"}
    assert {s.run for s in p.sources if s.tag.startswith("post")} == {2, 3}
    for s in p.sources:
        assert 0 <= s.start and s.start + s.n <= p.runs[s.run].shape[1]
    assert p.meta["sessions"] == ["pre", "post"] and p.meta["source"] == "stroke_gait"
    pc.check_provenance(p)


def test_person_refuses_a_stimulation_session_and_a_missing_npz(tmp_path):
    with pytest.raises(ValueError, match="stimulation"):
        sg.person(1, ["pre", "mises"], root=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        sg.person(1, ["pre"], root=str(tmp_path))


def test_person_refuses_an_npz_with_the_wrong_channels(tmp_path):
    _fake_npz(str(tmp_path), 4, "pre")
    z = dict(np.load(str(tmp_path / "sub-04_pre.npz")))
    z["names"] = z["names"][::-1]
    np.savez(str(tmp_path / "sub-04_pre.npz"), **z)
    with pytest.raises(ValueError, match="channels"):
        sg.person(4, ["pre"], root=str(tmp_path))



def _write_bv(folder, names, data, fs, markers, base="run"):
    """Minimal BrainVision triple: float32 multiplexed, resolution 1 µV per unit, `markers` = [(sample0, 'S  7')]."""
    n = data.shape[1]
    head = ["Brain Vision Data Exchange Header File Version 1.0", "", "[Common Infos]", "Codepage=UTF-8",
            f"DataFile={base}.eeg", f"MarkerFile={base}.vmrk", "DataFormat=BINARY", "DataOrientation=MULTIPLEXED",
            f"NumberOfChannels={len(names)}", f"SamplingInterval={1e6 / fs}", "", "[Binary Infos]",
            "BinaryFormat=IEEE_FLOAT_32", "", "[Channel Infos]"]
    head += [f"Ch{i + 1}={c},,1,µV" for i, c in enumerate(names)]
    open(os.path.join(folder, base + ".vhdr"), "w", encoding="utf-8").write("\n".join(head) + "\n")
    vm = ["Brain Vision Data Exchange Marker File, Version 1.0", "", "[Common Infos]", "Codepage=UTF-8",
          f"DataFile={base}.eeg", "", "[Marker Infos]"]
    vm += [f"Mk{i + 1}=Stimulus,{d},{s + 1},1,0" for i, (s, d) in enumerate(markers)]
    open(os.path.join(folder, base + ".vmrk"), "w", encoding="utf-8").write("\n".join(vm) + "\n")
    data.T.astype("<f4").tofile(os.path.join(folder, base + ".eeg"))
    return os.path.join(folder, base + ".vhdr")


def _bv_names(rename=True):
    names = [c for c in sg.channels()]
    if not rename:
        names = [{"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(c, c) for c in names]
    return names + ["ECG", "HEOR"]


def test_read_run_gives_microvolts_at_250_hz_for_the_motor_channels_and_events_at_250_hz(tmp_path):
    fs0, n = 1000.0, 12000
    t = np.arange(n) / fs0
    names = _bv_names(rename=False)
    data = np.zeros((len(names), n))
    c3 = names.index("C3")
    data[c3] = 100.0 * np.sin(2 * np.pi * 10 * t) + 5000.0
    data[names.index("ECG")] = 3e5
    mk = [(656, "S  7"), (3652, "S  8"), (4652, "S  9"), (5652, "S 10")]
    f = _write_bv(str(tmp_path), names, data, fs0, mk)
    x, ev = sg.read_run(f)
    assert x.shape == (len(sg.channels()), n // 4) and x.dtype == np.float32
    row = x[sg.channels().index("C3")]
    assert np.std(row[100:-100]) == pytest.approx(100 / np.sqrt(2), rel=0.02)
    assert np.abs(row).max() < 150
    assert ev.tolist() == [[164, 7], [913, 8], [1163, 9], [1413, 10]]


def test_read_run_reports_every_missing_channel_and_refuses_another_sampling_rate(tmp_path):
    names = [c for c in _bv_names() if c not in ("CP3", "C5")]
    f = _write_bv(str(tmp_path), names, np.zeros((len(names), 400)), 1000.0, [(10, "S  1")])
    with pytest.raises(ValueError, match="CP3"):
        sg.read_run(f)
    names = _bv_names()
    f2 = _write_bv(str(tmp_path), names, np.zeros((len(names), 400)), 500.0, [(10, "S  1")], base="r2")
    with pytest.raises(ValueError, match="1000"):
        sg.read_run(f2)



def test_a_missing_optional_session_is_reported_but_a_missing_required_one_raises(capsys):
    assert sg.check_sessions({"pre", "post"}, ("pre", "post", "follow"), required=("pre",)) == ["follow"]
    assert "follow" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="pre"):
        sg.check_sessions({"post"}, ("pre", "post"), required=("pre",))



def test_only_pure_session_eeg_members_of_the_subject_are_selected():
    names = ["sub-01/ses-pre/eeg/sub-01_ses-pre_task-MotorImagery_run-01_eeg.vhdr",
             "sub-01/ses-pre/eeg/sub-01_ses-pre_task-MotorImagery_run-01_eeg.eeg",
             "sub-01/ses-pre/eeg/sub-01_ses-pre_task-MotorImagery_run-01_events.tsv",
             "sub-01/ses-mises/eeg/sub-01_ses-mises_task-MotorImagery_run-01_eeg.vhdr",
             "sub-01/ses-miies/eeg/sub-01_ses-miies_task-MotorImagery_run-01_eeg.eeg",
             "sub-01/ses-post/eeg/sub-01_ses-post_task-MotorImagery_run-02_eeg.vmrk",
             "sub-02/ses-pre/eeg/sub-02_ses-pre_task-MotorImagery_run-01_eeg.vhdr",
             "../evil/sub-01_ses-pre_task-MotorImagery_run-01_eeg.vhdr"]
    sel = sg.members(names, 1, ("pre", "post", "follow"))
    assert sorted(sel) == sorted([names[0], names[1], names[5]])
    assert sg.members(names, 1, ("pre",)) == [names[0], names[1]]


def test_run_stems_group_the_triples_by_session_and_run_number():
    sel = ["sub-01/ses-pre/eeg/sub-01_ses-pre_task-MotorImagery_run-02_eeg.vhdr",
           "sub-01/ses-pre/eeg/sub-01_ses-pre_task-MotorImagery_run-01_eeg.vhdr",
           "sub-01/ses-post/eeg/sub-01_ses-post_task-MotorImagery_run-01_eeg.vhdr",
           "sub-01/ses-post/eeg/sub-01_ses-post_task-MotorImagery_run-01_eeg.eeg"]
    g = sg.run_headers(sel)
    assert list(g) == ["pre", "post"] and [r for r, _ in g["pre"]] == [1, 2] and len(g["post"]) == 1



@pytest.mark.skipif(not os.path.exists(sg.npz_path(1, "pre")), reason="sub-01 not fetched")
def test_real_subject_pre_session_has_four_runs_of_ten_plus_ten_trials_with_the_documented_timing():
    p = sg.person(1, ["pre"])
    assert p.fs == 250.0 and len(p.runs) == 4 and p.names == sg.channels() and "CPz" not in p.names
    n = {c: sum(s.cls == c for s in p.sources) for c in ("mi", "idle", "fixation")}
    assert n == {"mi": 40, "idle": 40, "fixation": 80}
    assert all(s.n == 1250 for s in p.sources if s.cls in ("mi", "idle")) and all(s.n == 750 for s in p.sources if s.cls == "fixation")
    pc.check_provenance(p)
    z = np.load(sg.npz_path(1, "pre"))
    ev = z["events"]
    for o in ev[ev[:, 1] == 3, 0][:5]:
        assert np.any(np.abs(ev[ev[:, 1] == 2, 0] - (o - 250)) <= 5) and np.any(np.abs(ev[ev[:, 1] == 1, 0] - (o - 1000)) <= 5)
    x = np.concatenate(p.runs, axis=1)
    assert 1.0 < np.std(x) < 5000.0 and np.isfinite(x).all()
