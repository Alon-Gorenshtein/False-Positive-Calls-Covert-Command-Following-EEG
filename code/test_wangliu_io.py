"""Synthetic inputs only, except test_smoke_dev_subject_pre, which runs when WANGLIU_NETWORK=1 (fetches DEV_SUBJECT 'pre',
~90 MB by range requests). cd code && python -m pytest -q -p no:cacheprovider test_wangliu_io.py"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

import wangliu_io as wl


def names(subj, tp, group, runs):
    return [f"sub-{subj:02d}/{tp}/sub-{subj:02d}_{group}_{tp}_{r:02d}_original.{e}" for r in runs for e in ("fdt", "set")]


FAKE = (["sub-01/", "sub-01/pre/", "sub-01/pre/._sub-01_B_pre_01_original.set",
         "__MACOSX/sub-01/pre/._sub-01_B_pre_02_original.fdt", "sub-01/pre/._sub-01_B_pre_03_original.fdt"]
        + names(1, "pre", "B", [3, 1, 2]) + names(1, "post", "B", [1]) + names(11, "pre", "B", [1])
        + names(7, "pre", "T", [2, 1]) + names(7, "post", "T", [1]) + names(7, "follow", "T", [4]))


def test_session_members_sorts_by_run_and_ignores_sidecars_directories_and_other_sessions():
    got = wl.session_members(1, "pre", FAKE)
    assert got == [f"sub-01/pre/sub-01_B_pre_{r:02d}_original.{e}" for r in (1, 2, 3) for e in ("fdt", "set")]
    assert not any(os.path.basename(n).startswith("._") for n in got)


def test_session_members_reads_the_t_group_letter_and_takes_zip_member_dicts():
    ms = [{"name": n, "offset": 0, "csize": 1, "usize": 1, "method": 8} for n in FAKE]
    assert wl.session_members(7, "pre", ms) == names(7, "pre", "T", [1, 2])
    assert wl.session_members(7, "follow", ms) == names(7, "follow", "T", [4])
    assert wl.session_members(2, "pre", ms) == []


def test_session_members_refuses_an_unknown_timepoint_and_a_run_without_its_fdt():
    with pytest.raises(ValueError, match="timepoint"):
        wl.session_members(1, "followup", FAKE)
    with pytest.raises(ValueError, match="fdt"):
        wl.session_members(1, "pre", [n for n in FAKE if n != "sub-01/pre/sub-01_B_pre_02_original.fdt"])


def test_sessions_lists_the_timepoints_present_in_order():
    assert wl.sessions(1, FAKE) == ["pre", "post"]
    assert wl.sessions(7, FAKE) == ["pre", "post", "follow"]
    assert wl.sessions(2, FAKE) == []


EEG = [f"E{i}" for i in range(59)]


def run_of(events, x=None, fs=250.0, names=EEG):
    """A Run with the given (seconds, code) events; x defaults to a tiny zero array (trials() never reads it)."""
    ev = pd.DataFrame(events, columns=["latency_s", "code"]).astype({"latency_s": float, "code": int})
    x = np.zeros((len(names), 10), np.float32) if x is None else x
    return wl.Run(subject=1, timepoint="pre", run=1, group="B", x=x, fs=fs, names=list(names), events=ev, sha256="",
                  meta={})


def standard(n=20, gap=8.6, mi_s=7.0, start=2.0, seed=0):
    """n designed trials: rest marker, MI marker `gap` s later (types shuffled, half each), next rest `mi_s` s after it."""
    types = np.random.default_rng(seed).permutation(np.r_[np.full(n // 2, 2), np.full(n - n // 2, 3)])
    ev, t = [], start
    for k in range(n):
        ev += [(t, 1), (t + gap, int(types[k]))]
        t += gap + mi_s
    return ev


ZERO = {"rest": 0, "mi": 0, "rest_without_next_mi": 0, "gap_outside_window": 0, "rest_superseded": 0}


def test_trials_pairs_every_rest_with_its_mi_in_a_standard_run():
    ev = standard()
    r = run_of(ev)
    tr = wl.trials(r)
    assert list(tr.columns) == ["rest_s", "mi_s", "mi_type"] and len(tr) == 20
    assert np.allclose(tr.mi_s - tr.rest_s, 8.6)
    assert np.allclose(tr.rest_s, [t for t, c in ev if c == 1])
    assert tr.mi_type.value_counts().to_dict() == {"affected_hand": 10, "sixth_finger": 10}
    assert list(tr.mi_type) == [{2: "affected_hand", 3: "sixth_finger"}[c] for _, c in ev if c != 1]
    assert r.meta["unpaired"] == ZERO


def test_trials_of_the_aborted_run_is_empty_and_counts_the_rest_markers():
    r = run_of([(2.0 + 15.6 * k, 1) for k in range(5)])
    tr = wl.trials(r)
    assert len(tr) == 0 and list(tr.columns) == ["rest_s", "mi_s", "mi_type"]
    assert r.meta["unpaired"] == {**ZERO, "rest": 5, "rest_without_next_mi": 5}


@pytest.mark.parametrize("drop", [0, 7, 19])
def test_trials_with_19_rest_markers_for_20_mi_drops_only_the_orphan_mi(drop):
    ev = standard()
    rests = [i for i, (_, c) in enumerate(ev) if c == 1]
    del ev[rests[drop]]
    r = run_of(ev)
    tr = wl.trials(r)
    assert len(tr) == 19 and np.allclose(tr.mi_s - tr.rest_s, 8.6)
    assert r.meta["unpaired"] == {**ZERO, "mi": 1}


def test_trials_keeps_all_trials_of_a_run_with_23():
    r = run_of(standard(23))
    assert len(wl.trials(r)) == 23 and r.meta["unpaired"] == ZERO


def test_trials_drops_both_markers_when_the_gap_is_outside_the_window():
    ev = standard()
    ev[5 * 2 + 1] = (ev[5 * 2][0] + 9.5, ev[5 * 2 + 1][1])
    ev[9 * 2 + 1] = (ev[9 * 2][0] + 7.9, ev[9 * 2 + 1][1])
    r = run_of(ev)
    tr = wl.trials(r)
    assert len(tr) == 18 and np.allclose(tr.mi_s - tr.rest_s, 8.6)
    assert r.meta["unpaired"] == {**ZERO, "rest": 2, "mi": 2, "gap_outside_window": 2}


def test_trials_window_edges_are_inclusive():
    lo, hi = wl.REST_MI_GAP_S
    r = run_of([(1.0, 1), (1.0 + lo, 2), (30.0, 1), (30.0 + hi, 3)])
    assert len(wl.trials(r)) == 2


def test_trials_rest_followed_by_another_rest_pairs_only_the_later_one():
    ev = standard()
    del ev[4 * 2 + 1]
    ev.insert(1, (ev[0][0] + 0.2, 1))
    r = run_of(ev)
    tr = wl.trials(r)
    assert len(tr) == 19 and tr.mi_s.is_unique and np.allclose(tr.mi_s - tr.rest_s, [8.4] + [8.6] * 18)
    assert r.meta["unpaired"] == {**ZERO, "rest": 2, "gap_outside_window": 1, "rest_superseded": 1}


def test_trials_ignores_event_order_in_the_frame():
    ev = standard()
    r = run_of(ev[::-1])
    assert len(wl.trials(r)) == 20 and r.meta["unpaired"] == ZERO


def eeg_like(sd_uv=20.0, seconds=60, seed=1):
    """59 channels of white noise (sd_uv) riding on DC offsets of up to +-30 mV and a slow drift, as in a DC-coupled
    unfiltered recording."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 250)
    t = np.arange(n) / 250
    off = rng.uniform(-3e4, 3e4, (59, 1)) + 500 * np.sin(2 * np.pi * 0.02 * t)
    return (off + sd_uv * rng.standard_normal((59, n))).astype(np.float32)


def test_check_passes_a_dc_coupled_run_in_microvolts_and_counts_events():
    r = run_of(standard(), x=eeg_like())
    out = wl.check(r)
    assert out["rest"] == 20 and out["affected_hand"] == 10 and out["sixth_finger"] == 10
    assert out["channels"] == 59 and out["samples"] == 15000
    assert 18 < out["median_sd_uv"] < 22


def test_check_refuses_unsorted_events():
    ev = standard()
    ev[3], ev[4] = ev[4], ev[3]
    with pytest.raises(ValueError, match="sorted"):
        wl.check(run_of(ev, x=eeg_like()))


@pytest.mark.parametrize("bad", [0, 4, 255])
def test_check_refuses_codes_outside_1_2_3(bad):
    ev = standard()
    ev.insert(5, (ev[4][0] + 0.5, bad))
    with pytest.raises(ValueError, match="code"):
        wl.check(run_of(ev, x=eeg_like()))


@pytest.mark.parametrize("scale", [1e-6, 1e3])
def test_check_refuses_a_signal_not_in_microvolts(scale):
    with pytest.raises(ValueError, match="units"):
        wl.check(run_of(standard(), x=eeg_like() * np.float32(scale)))


def test_check_unit_bounds_are_1_and_200_microvolts():
    wl.check(run_of(standard(), x=eeg_like(sd_uv=1.5)))
    wl.check(run_of(standard(), x=eeg_like(sd_uv=150.0)))
    with pytest.raises(ValueError, match="units"):
        wl.check(run_of(standard(), x=eeg_like(sd_uv=0.5)))
    with pytest.raises(ValueError, match="units"):
        wl.check(run_of(standard(), x=eeg_like(sd_uv=250.0)))


def test_check_refuses_rows_that_do_not_match_the_59_names_or_another_rate():
    with pytest.raises(ValueError, match="channels"):
        wl.check(run_of(standard(), x=eeg_like()[:58]))
    with pytest.raises(ValueError, match="channels"):
        wl.check(run_of(standard(), x=eeg_like()[:58], names=EEG[:58]))
    with pytest.raises(ValueError, match="rate"):
        wl.check(run_of(standard(), x=eeg_like(), fs=500.0))


SET_NAME, FDT_NAME = "sub-01/pre/sub-01_B_pre_01_original.set", "sub-01/pre/sub-01_B_pre_01_original.fdt"
SET_BYTES = b"EEGLAB header " * 20000
FDT_BYTES = np.random.default_rng(0).standard_normal(50000).astype(np.float32).tobytes()


def archive(monkeypatch=None):
    """A zip with a directory entry, a deflated .set and a stored .fdt; with monkeypatch, ZIP64_LIMIT is lowered so that
    every size / offset field and the end record are zip64."""
    if monkeypatch is not None:
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 64)
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        z.writestr("sub-01/", b"")
        z.writestr(SET_NAME, SET_BYTES, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr(FDT_NAME, FDT_BYTES, compress_type=zipfile.ZIP_STORED)
        z.writestr("sub-01/pre/._sub-01_B_pre_01_original.set", b"\x00\x05\x16\x07 sidecar")
    return bio.getvalue()


def server(buf, calls=None, chunk=4096, fail_after=None):
    """Range function over buf (inclusive a..b, in chunks), recording calls; fail_after = bytes served before the first
    call breaks off with a ConnectionError (later calls succeed)."""
    state = {"failed": False}

    def rng(a, b):
        assert 0 <= a <= b < len(buf)
        if calls is not None:
            calls.append((a, b))
        data = buf[a:b + 1]
        for i in range(0, len(data), chunk):
            if fail_after is not None and not state["failed"] and i >= fail_after:
                state["failed"] = True
                raise ConnectionError("connection reset")
            yield data[i:i + chunk]
    return rng


@pytest.mark.parametrize("zip64", [False, True])
def test_central_directory_matches_zipfile(zip64, monkeypatch):
    buf = archive(monkeypatch if zip64 else None)
    if zip64:
        assert b"PK\x06\x06" in buf and b"PK\x06\x07" in buf
    got = {m["name"]: m for m in wl._central(len(buf), server(buf))}
    ref = {i.filename: i for i in zipfile.ZipFile(io.BytesIO(buf)).infolist()}
    assert set(got) == set(ref)
    for n, i in ref.items():
        assert (got[n]["offset"], got[n]["csize"], got[n]["usize"], got[n]["method"], got[n]["crc"]) == \
            (i.header_offset, i.compress_size, i.file_size, i.compress_type, i.CRC)
    assert got[SET_NAME]["method"] == 8 and got[FDT_NAME]["method"] == 0


@pytest.mark.parametrize("zip64", [False, True])
def test_extract_inflates_deflated_and_copies_stored_members(zip64, tmp_path, monkeypatch):
    buf = archive(monkeypatch if zip64 else None)
    ms = {m["name"]: m for m in wl._central(len(buf), server(buf))}
    for n, want in ((SET_NAME, SET_BYTES), (FDT_NAME, FDT_BYTES)):
        out = wl._extract(ms[n], str(tmp_path / os.path.basename(n)), server(buf))
        assert open(out, "rb").read() == want
    assert sorted(os.listdir(tmp_path)) == sorted(os.path.basename(n) for n in (SET_NAME, FDT_NAME))


def test_extract_skips_a_complete_file_without_a_request(tmp_path):
    buf = archive()
    m = {x["name"]: x for x in wl._central(len(buf), server(buf))}[FDT_NAME]
    path = tmp_path / "x.fdt"
    path.write_bytes(FDT_BYTES)
    calls = []
    wl._extract(m, str(path), server(buf, calls))
    assert calls == []


def test_extract_resumes_from_a_partial_download(tmp_path):
    buf = archive()
    m = {x["name"]: x for x in wl._central(len(buf), server(buf))}[SET_NAME]
    path = str(tmp_path / "x.set")
    full = []
    wl._extract(m, path, server(buf, full))
    data_start, data_end = full[-1]
    os.remove(path)
    k = m["csize"] // 3
    open(path + ".part", "wb").write(buf[data_start:data_start + k])
    calls = []
    wl._extract(m, path, server(buf, calls))
    assert open(path, "rb").read() == SET_BYTES and not os.path.exists(path + ".part")
    assert calls[-1] == (data_start + k, data_end)


def test_extract_survives_a_dropped_connection_within_one_call(tmp_path):
    buf = archive()
    m = {x["name"]: x for x in wl._central(len(buf), server(buf))}[FDT_NAME]
    path = str(tmp_path / "x.fdt")
    wl._extract(m, path, server(buf, fail_after=50000))
    assert open(path, "rb").read() == FDT_BYTES


def test_extract_refuses_corrupt_bytes_and_a_size_mismatch_and_leaves_no_file(tmp_path):
    buf = archive()
    ms = {x["name"]: x for x in wl._central(len(buf), server(buf))}
    i = buf.index(FDT_BYTES[:64]) + 1000
    bad = buf[:i] + bytes([buf[i] ^ 0xFF]) + buf[i + 1:]
    path = str(tmp_path / "x.fdt")
    with pytest.raises(IOError, match="crc"):
        wl._extract(ms[FDT_NAME], path, server(bad))
    assert os.listdir(tmp_path) == []
    with pytest.raises(IOError, match="size"):
        wl._extract({**ms[SET_NAME], "usize": ms[SET_NAME]["usize"] + 1}, str(tmp_path / "x.set"), server(buf))
    assert os.listdir(tmp_path) == []


def offline(monkeypatch, buf, licence="CC BY 4.0", size=None, calls=None):
    """Route zip_members / fetch to the in-memory archive buf: the article listing, the archive size and the ranges."""
    art = {"license": {"name": licence}, "files": [{"id": wl.SOURCE_ZIP_ID, "name": "sourcedata.zip", "size": len(buf),
                                                     "computed_md5": "0" * 32}]}
    monkeypatch.setattr(wl, "_article", lambda: art)
    monkeypatch.setattr(wl, "_size", lambda: len(buf) if size is None else size)
    monkeypatch.setattr(wl, "_range", server(buf, calls))


def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    for f in ("_article", "_size", "_range"):
        monkeypatch.setattr(wl, f, boom)


def test_zip_members_lists_the_archive_then_reads_the_cache_without_network(tmp_path, monkeypatch):
    buf = archive()
    offline(monkeypatch, buf)
    ms = wl.zip_members(str(tmp_path))
    assert {m["name"] for m in ms} == {i.filename for i in zipfile.ZipFile(io.BytesIO(buf)).infolist()}
    assert all(set(m) >= {"name", "offset", "csize", "usize", "method"} for m in ms)
    no_network(monkeypatch)
    assert wl.zip_members(str(tmp_path)) == ms
    with pytest.raises(AssertionError, match="network"):
        wl.zip_members(str(tmp_path), refresh=True)


def test_zip_members_refuses_another_licence_and_a_size_mismatch(tmp_path, monkeypatch):
    buf = archive()
    offline(monkeypatch, buf, licence="CC BY-NC 4.0")
    with pytest.raises(ValueError, match="licence"):
        wl.zip_members(str(tmp_path))
    offline(monkeypatch, buf, size=len(buf) + 1)
    with pytest.raises(IOError, match="size"):
        wl.zip_members(str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_fetch_extracts_the_session_returns_set_paths_and_skips_complete_files(tmp_path, monkeypatch):
    buf = archive()
    calls = []
    offline(monkeypatch, buf, calls=calls)
    got = wl.fetch(1, "pre", root=str(tmp_path))
    d = tmp_path / "sourcedata" / "sub-01" / "pre"
    assert got == [str(d / "sub-01_B_pre_01_original.set")]
    assert (d / "sub-01_B_pre_01_original.set").read_bytes() == SET_BYTES
    assert (d / "sub-01_B_pre_01_original.fdt").read_bytes() == FDT_BYTES
    assert sorted(os.listdir(d)) == ["sub-01_B_pre_01_original.fdt", "sub-01_B_pre_01_original.set"]
    calls.clear()
    assert wl.fetch(1, "pre", root=str(tmp_path)) == got and calls == []
    with pytest.raises(FileNotFoundError):
        wl.fetch(1, "follow", root=str(tmp_path))


FILE_NAMES = ([f"E{i}" for i in range(30)] + ["ECG"] + [f"E{i}" for i in range(30, 59)] + ["HEOR", "HEOL", "VEOU", "VEOL"])


def eeglab_pair(d, base="sub-07_T_post_03_original", names=FILE_NAMES, n=500, events=(("2", 2151.0), ("1", 1.0)),
                fdt_extra=0):
    """A flat-format EEGLAB .set / .fdt pair as the archive holds it: data float32 written channel-fastest (MATLAB order
    of nbchan x pnts), events as type strings with 1-based sample latencies, and a malformed 'comments' variable (its
    dimensions claim more than it stores), so scipy.io.loadmat of the whole file fails as on the real files. Row c of
    the data is 1000 * c + (sample index mod 7). -> (set path, data [nbchan, n])."""
    import scipy.io as sio
    import struct
    data = (1000.0 * np.arange(len(names))[:, None] + np.arange(n)[None, :] % 7).astype(np.float32)
    ch = np.zeros((1, len(names)), dtype=[("labels", "O")])
    ch["labels"][0] = names
    ev = np.zeros((len(events), 1), dtype=[("type", "O"), ("latency", "O")])
    for k, (t, lat) in enumerate(events):
        ev[k, 0] = (t, lat)
    path = os.path.join(d, base + ".set")
    sio.savemat(path, {"nbchan": float(len(names)), "trials": 1.0, "pnts": float(n), "srate": 250.0, "xmin": 0.0,
                       "chanlocs": ch, "event": ev, "datfile": base + ".fdt", "data": base + ".fdt", "ref": "common",
                       "comments": "abcd", "filepath": ""}, do_compression=False)
    b = bytearray(open(path, "rb").read())
    i = b.index(b"comments")
    b[i - 12:i - 8] = struct.pack("<i", 400)
    open(path, "wb").write(b)
    np.r_[data.T.ravel(), np.zeros(fdt_extra, np.float32)].astype("<f4").tofile(os.path.join(d, base + ".fdt"))
    return path, data


def test_load_run_reads_a_set_whose_whole_file_loadmat_fails(tmp_path):
    import hashlib
    import scipy.io as sio
    path, data = eeglab_pair(str(tmp_path))
    with pytest.raises(TypeError, match="buffer is too small"):
        sio.loadmat(path)
    r = wl.load_run(path)
    keep = [i for i, c in enumerate(FILE_NAMES) if c not in ("ECG", "HEOR", "HEOL", "VEOU", "VEOL")]
    assert (r.subject, r.timepoint, r.run, r.group, r.fs) == (7, "post", 3, "T", 250.0)
    assert r.names == [f"E{i}" for i in range(59)]
    assert r.x.dtype == np.float32 and r.x.shape == (59, 500) and np.array_equal(r.x, data[keep])
    assert r.sha256 == hashlib.sha256(open(path[:-4] + ".fdt", "rb").read()).hexdigest()
    assert r.events.to_dict("list") == {"latency_s": [0.0, 8.6], "code": [1, 2]}
    assert r.events["code"].dtype.kind == "i" and r.events["latency_s"].dtype.kind == "f"
    assert sorted(r.meta["dropped"]) == ["ECG", "HEOL", "HEOR", "VEOL", "VEOU"]
    assert r.meta["events_sorted_in_file"] is False


def test_load_run_refuses_a_wrong_fdt_size_and_an_unexpected_channel_set(tmp_path):
    path, _ = eeglab_pair(str(tmp_path), fdt_extra=3)
    with pytest.raises(ValueError, match="fdt"):
        wl.load_run(path)
    path, _ = eeglab_pair(str(tmp_path), base="sub-01_B_pre_01_original", names=FILE_NAMES[:-1])
    with pytest.raises(ValueError, match="channels"):
        wl.load_run(path)
    path, _ = eeglab_pair(str(tmp_path), base="sub-01_B_pre_02_original", names=FILE_NAMES[:-1] + ["E59"])
    with pytest.raises(ValueError, match="channels"):
        wl.load_run(path)


def test_load_run_refuses_an_event_type_that_is_not_an_integer(tmp_path):
    path, _ = eeglab_pair(str(tmp_path), events=(("1", 1.0), ("boundary", 30.0)))
    with pytest.raises(ValueError, match="boundary"):
        wl.load_run(path)


@pytest.mark.skipif(os.environ.get("WANGLIU_NETWORK") != "1", reason="network: set WANGLIU_NETWORK=1")
def test_smoke_dev_subject_pre():
    """DEV_SUBJECT 'pre' only (plumbing; never analysed): fetch, load, pair, check every run."""
    import hashlib
    paths = wl.fetch(wl.DEV_SUBJECT, "pre")
    assert len(paths) == 4
    for p in paths:
        r = wl.load_run(p)
        assert (r.subject, r.timepoint, r.group) == (wl.DEV_SUBJECT, "pre", "B")
        assert r.x.dtype == np.float32 and r.x.shape[0] == 59 and len(r.names) == 59 and r.fs == 250.0
        assert {"C3", "C4", "Cz"} <= set(r.names) and not {"ECG", "HEOR", "HEOL", "VEOU", "VEOL", "CPz"} & set(r.names)
        assert r.sha256 == hashlib.sha256(open(p[:-4] + ".fdt", "rb").read()).hexdigest()
        tr = wl.trials(r)
        out = wl.check(r)
        print(f"\nrun {r.run:02d} x {r.x.shape} trials {len(tr)} {tr.mi_type.value_counts().to_dict()} "
              f"unpaired {r.meta['unpaired']} median_sd_uv {out['median_sd_uv']:.2f} events {out}")
        assert 18 <= len(tr) <= 23
