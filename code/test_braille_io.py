"""Tests for code/braille_io.py (Braille BCI tactile P300, G-Node GIN 10.12751/g-node.dyby2g, CC0).
Unit tests use synthetic events / runs / a fake HTTP server / a hand-written BDF and never touch the network. One smoke test
downloads and loads the dev subject's large session; it runs only with BRAILLE_NETWORK=1.
cd code && ../../.venv/bin/python -m pytest -q -p no:cacheprovider test_braille_io.py"""
import gzip
import hashlib
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import braille_io as bio

ORDER = (0, 1, 2, 3, 4, 5, 7, 6, 0, 1, 2, 3, 4, 5, 6, 7)


def fake_bids(n_cycles=2, soa=0.3, gap=10.0, t0=5.0, seed=0, subj=6, session="large"):
    """events.tsv-like rows: per cycle 80 stimuli (each cell 0-7 ten times, shuffled), target cell from ORDER."""
    rng = np.random.default_rng(seed)
    rows, t = [], t0
    for c in range(n_cycles):
        tgt = ORDER[c % 16]
        for v in rng.permutation(np.repeat(np.arange(8), 10)):
            rows.append({"onset": round(t, 3), "duration": 0, "trial_type": int(v == tgt), "value": int(v),
                         "sample": int(round(t * 500)), "target": tgt, "finger": int(v), "epoch_id": 0, "session_id": c,
                         "user": subj, "reg": session, "fingers": "{0:0, 1:1, 2:2, 3:3,4:4,5:5,6:6,7:7}"})
            t += soa
        t += gap
    return pd.DataFrame(rows)


def test_constants_match_the_dataset_layout():
    assert bio.SUBJECTS == [6] + list(range(8, 27)) and len(bio.SUBJECTS) == 20 and 7 not in bio.SUBJECTS
    assert bio.SESSIONS == ("large", "small") and bio.DEV_SUBJECT == 6 and bio.FS == 500.0
    assert len(bio.CHANNELS) == 48 and len(bio.SCALP) == 45
    assert bio.SCALP == [c for c in bio.CHANNELS if c not in ("ecg", "A1", "A2")]
    assert bio.RAW.endswith("output/raw/braille_p300") and bio.RAW.startswith(bio.ROOT)


def test_paths_are_the_bids_paths_listed_on_gin_without_zero_padding():
    assert bio.paths(6, "large") == {"bdf": "sub-6/ses-large/eeg/sub-6_ses-large_task-bci_eeg.bdf",
                                     "events": "sub-6/ses-large/eeg/sub-6_ses-large_task-bci_events.tsv"}
    assert bio.paths(26, "small")["bdf"] == "sub-26/ses-small/eeg/sub-26_ses-small_task-bci_eeg.bdf"


@pytest.mark.parametrize("subj,ses", [(7, "large"), (5, "large"), (27, "small"), (6, "medium")])
def test_paths_refuses_a_subject_or_session_that_does_not_exist(subj, ses):
    with pytest.raises(ValueError):
        bio.paths(subj, ses)


def test_events_table_has_the_interface_columns_sorted_by_onset():
    raw = fake_bids()
    ev = bio.events_table(raw.sample(frac=1.0, random_state=1))
    assert list(ev.columns) == ["onset_s", "finger", "is_target", "cycle", "target"]
    assert ev.onset_s.is_monotonic_increasing and len(ev) == 160
    assert ev.onset_s.dtype == float and ev.is_target.dtype == bool
    assert all(ev[c].dtype.kind == "i" for c in ("finger", "cycle", "target"))
    assert ev.is_target.sum() == 20 and (ev.is_target == (ev.finger == ev.target)).all()
    assert list(ev.groupby("cycle").target.first()) == [0, 1]


def test_events_table_refuses_a_sample_column_more_than_2_ms_off_the_onset():
    raw = fake_bids()
    raw.loc[5, "sample"] += 1
    bio.events_table(raw)
    raw.loc[5, "sample"] += 1
    with pytest.raises(ValueError, match="sample"):
        bio.events_table(raw)


def test_events_table_refuses_a_finger_column_that_differs_from_the_stimulated_cell():
    raw = fake_bids()
    raw.loc[3, "finger"] = (raw.loc[3, "value"] + 1) % 8
    with pytest.raises(ValueError, match="finger"):
        bio.events_table(raw)


def test_events_table_refuses_a_trial_type_other_than_0_or_1():
    raw = fake_bids()
    raw.loc[0, "trial_type"] = 2
    with pytest.raises(ValueError, match="trial_type"):
        bio.events_table(raw)


def fake_run(ev=None, x=None, seed=0):
    """A Run of 45 rows of 20-µV noise (plus a DC offset, as in the raw files) with 2 well-formed cycles."""
    rng = np.random.default_rng(seed)
    if x is None:
        x = (rng.normal(0.0, 20.0, (45, 5000)) + rng.uniform(-3e4, 3e4, (45, 1))).astype(np.float32)
    ev = bio.events_table(fake_bids()) if ev is None else ev
    return bio.Run(subject=6, session="large", x=x, fs=500.0, names=list(bio.SCALP), events=ev, sha256="0" * 64, meta={})


def test_check_passes_a_well_formed_run_and_returns_the_counts():
    c = bio.check(fake_run())
    assert c["n_stim"] == 160 and c["n_target"] == 20 and c["n_cycles"] == 2
    assert c["per_cycle"] == {"stim": [80, 80], "target": [10, 10], "per_finger": [10, 10]}
    assert c["target_order"] == [0, 1]
    assert 6.0 < c["median_sd_uv"] < 10.0


def test_check_refuses_unsorted_events():
    ev = fake_run().events
    ev.loc[[3, 4], "onset_s"] = ev.loc[[4, 3], "onset_s"].to_numpy()
    with pytest.raises(ValueError, match="sorted"):
        bio.check(fake_run(ev))


def test_check_refuses_a_cycle_missing_a_stimulus():
    ev = fake_run().events.drop(index=7).reset_index(drop=True)
    with pytest.raises(ValueError, match="cycle.*10 stimuli"):
        bio.check(fake_run(ev))


def test_check_refuses_a_cycle_with_11_of_one_finger_and_9_of_another():
    ev = fake_run().events
    i = ev.index[(ev.cycle == 1) & (ev.finger == 3) & ~ev.is_target][0]
    ev.loc[i, "finger"] = 4
    with pytest.raises(ValueError, match="cycle.*10 stimuli"):
        bio.check(fake_run(ev))


def test_check_refuses_a_finger_outside_0_to_7():
    ev = fake_run().events
    ev.loc[ev.index[ev.finger == 5][0], "finger"] = 8
    with pytest.raises(ValueError, match="finger"):
        bio.check(fake_run(ev))


def test_check_refuses_a_target_that_changes_within_a_cycle():
    ev = fake_run().events
    i = ev.index[(ev.cycle == 0) & (ev.finger == 6)][0]
    ev.loc[i, "target"] = 7
    with pytest.raises(ValueError, match="target.*constant"):
        bio.check(fake_run(ev))


def test_check_refuses_a_target_flag_that_disagrees_with_finger_equals_target():
    ev = fake_run().events
    ev.loc[ev.index[ev.is_target][0], "is_target"] = False
    with pytest.raises(ValueError, match="is_target"):
        bio.check(fake_run(ev))


@pytest.mark.parametrize("scale", [1e-6, 1e3])
def test_check_refuses_a_signal_not_in_microvolts(scale):
    run = fake_run()
    run.x = run.x * np.float32(scale)
    with pytest.raises(ValueError, match="µV"):
        bio.check(run)


class FakeGin:
    """GIN's raw endpoint as probed in Oct 2026. 'annex' files (the BDFs): HEAD and identity GET carry Content-Length, Range
    is ignored (unless ranges=True), gzip comes chunked with no length. 'git' files (events.tsv): no length on HEAD or on an
    identity GET (chunked); a gzip GET carries the length of the gzip stream and honours Range on it. `short` drops bytes."""

    def __init__(self, files, ranges=False, short=0):
        self.files, self.ranges, self.short, self.calls = files, ranges, short, []

    def __call__(self, url, headers, method="GET"):
        self.calls.append((method, url, dict(headers)))
        if url not in self.files:
            raise IOError(f"{url}: HTTP 404")
        data, kind = self.files[url]
        enc = headers.get("Accept-Encoding", "identity")
        body = gzip.compress(data, mtime=0) if enc == "gzip" else data
        h = {"content-encoding": "gzip"} if enc == "gzip" else {}
        sized = (kind == "annex") == (enc == "identity")
        if method == "HEAD":
            return 200, ({"content-length": str(len(data))} if kind == "annex" else {}), iter(())
        if not sized:
            return 200, dict(h, **{"transfer-encoding": "chunked"}), iter([body])
        rng = headers.get("Range")
        if rng and (kind == "git" or self.ranges):
            a = int(rng[len("bytes="):-1])
            status, part = 206, body[a:]
            h["content-range"] = f"bytes {a}-{len(body) - 1}/{len(body)}"
        else:
            status, part = 200, body
        h["content-length"] = str(len(part))
        part = part[:len(part) - self.short]
        return status, h, iter([part[i:i + 7] for i in range(0, len(part), 7)])


DATA = bytes(range(256)) * 3
URL = "https://example.invalid/f"


def gets(srv):
    return [c for c in srv.calls if c[0] == "GET"]


def test_download_fetches_an_annexed_file_uncompressed_and_leaves_no_part(tmp_path):
    srv, path = FakeGin({URL: (DATA, "annex")}), str(tmp_path / "a" / "f.bdf")
    assert bio.download(URL, path, http=srv) == path
    assert open(path, "rb").read() == DATA and os.listdir(tmp_path / "a") == ["f.bdf"]
    assert gets(srv)[0][2]["Accept-Encoding"] == "identity"


def test_download_skips_a_complete_file(tmp_path):
    srv, path = FakeGin({URL: (DATA, "annex")}), str(tmp_path / "f.bdf")
    open(path, "wb").write(DATA)
    bio.download(URL, path, http=srv)
    assert gets(srv) == []


def test_download_replaces_a_file_whose_size_differs_from_content_length(tmp_path):
    srv, path = FakeGin({URL: (DATA, "annex")}), str(tmp_path / "f.bdf")
    open(path, "wb").write(DATA[:-1])
    bio.download(URL, path, http=srv)
    assert open(path, "rb").read() == DATA


def test_download_restarts_a_partial_when_the_server_ignores_range(tmp_path):
    srv, path = FakeGin({URL: (DATA, "annex")}), str(tmp_path / "f.bdf")
    open(path + ".identity.part", "wb").write(b"X" * 10)
    bio.download(URL, path, http=srv)
    assert gets(srv)[0][2]["Range"] == "bytes=10-" and open(path, "rb").read() == DATA


def test_download_resumes_a_partial_when_the_server_honours_range(tmp_path):
    srv, path = FakeGin({URL: (DATA, "annex")}, ranges=True), str(tmp_path / "f.bdf")
    open(path + ".identity.part", "wb").write(DATA[:10])
    bio.download(URL, path, http=srv)
    assert open(path, "rb").read() == DATA and not os.path.exists(path + ".identity.part")


def test_download_fetches_a_plain_git_file_gzip_encoded_then_decompresses(tmp_path):
    srv, path = FakeGin({URL: (DATA, "git")}), str(tmp_path / "f.tsv")
    bio.download(URL, path, http=srv)
    assert open(path, "rb").read() == DATA and os.listdir(tmp_path) == ["f.tsv"]
    assert gets(srv)[0][2]["Accept-Encoding"] == "gzip"


def test_download_resumes_a_plain_git_file_on_the_gzip_stream(tmp_path):
    srv, path = FakeGin({URL: (DATA, "git")}), str(tmp_path / "f.tsv")
    open(path + ".gzip.part", "wb").write(gzip.compress(DATA, mtime=0)[:20])
    bio.download(URL, path, http=srv)
    assert gets(srv)[0][2]["Range"] == "bytes=20-" and open(path, "rb").read() == DATA


@pytest.mark.parametrize("kind,enc", [("annex", "identity"), ("git", "gzip")])
def test_download_short_body_raises_keeps_the_part_and_writes_no_file(tmp_path, kind, enc):
    srv, path = FakeGin({URL: (DATA, kind)}, short=5), str(tmp_path / "f")
    with pytest.raises(IOError, match="bytes"):
        bio.download(URL, path, http=srv)
    assert not os.path.exists(path) and os.path.exists(f"{path}.{enc}.part")
    bio.download(URL, path, http=FakeGin({URL: (DATA, kind)}))
    assert open(path, "rb").read() == DATA


def test_download_refuses_a_plain_body_with_no_length_to_check(tmp_path):
    def chunked(url, headers, method="GET"):
        return 200, {"transfer-encoding": "chunked"}, iter([DATA])
    with pytest.raises(IOError, match="identity"):
        bio.download(URL, str(tmp_path / "f"), http=chunked)
    assert not os.path.exists(tmp_path / "f")


def test_download_accepts_a_gzip_body_with_no_length_and_checks_its_crc(tmp_path):
    z = gzip.compress(DATA, mtime=0)

    def chunked_gzip(body):
        def http(url, headers, method="GET"):
            return 200, {"transfer-encoding": "chunked", "content-encoding": "gzip"}, iter([body])
        return http
    path = str(tmp_path / "f")
    assert open(bio.download(URL, path, http=chunked_gzip(z)), "rb").read() == DATA
    bad = z[:-8] + bytes(8)
    with pytest.raises(Exception):
        bio.download(URL, str(tmp_path / "g"), http=chunked_gzip(bad))
    assert not os.path.exists(tmp_path / "g")


HTML = """<a href="/doi/BrailleBCI/src/master/sub-6/ses-large">..</a>
<a href="/doi/BrailleBCI/src/master/sub-6/ses-large/eeg/sub-6_ses-large_task-bci_eeg.bdf">x</a>
<a href="/doi/BrailleBCI/src/master/sub-6/ses-large/eeg/._sub-6_ses-large_task-bci_eeg.bdf">x</a>
<a href="/doi/BrailleBCI/src/master/sub-6/ses-large/eeg/sub-6_ses-large_task-bci_events.tsv">x</a>
<a href="/doi/BrailleBCI/src/master/sub-6/ses-large/eeg/sub-6_ses-large_task-bci_events.tsv">again</a>
<a href="/doi/BrailleBCI/src/master/sub-6/ses-large/eeg?lang=de-DE">de</a>"""


def test_names_in_lists_the_folder_entries_ignoring_appledouble_and_language_links():
    assert bio.names_in(HTML, "sub-6/ses-large/eeg") == ["sub-6_ses-large_task-bci_eeg.bdf",
                                                        "sub-6_ses-large_task-bci_events.tsv"]


def fake_gin_for(subj, session, bdf=DATA, tsv=b"onset\n"):
    p = bio.paths(subj, session)
    return FakeGin({f"{bio.RAW_URL}/{p['bdf']}": (bdf, "annex"), f"{bio.RAW_URL}/{p['events']}": (tsv, "git")})


def test_fetch_downloads_the_bdf_and_events_to_their_bids_paths_under_root(tmp_path, monkeypatch):
    p = bio.paths(6, "large")
    monkeypatch.setattr(bio, "listing", lambda s, ses: [os.path.basename(v) for v in p.values()])
    monkeypatch.setattr(bio, "_http", fake_gin_for(6, "large"))
    got = bio.fetch(6, "large", root=str(tmp_path))
    assert got == {k: os.path.join(str(tmp_path), v) for k, v in p.items()}
    assert open(got["bdf"], "rb").read() == DATA and open(got["events"], "rb").read() == b"onset\n"


def test_fetch_refuses_when_the_server_listing_lacks_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bio, "listing", lambda s, ses: ["sub-6_ses-large_task-bci_events.tsv"])
    monkeypatch.setattr(bio, "_http", fake_gin_for(6, "large"))
    with pytest.raises(FileNotFoundError, match="eeg.bdf"):
        bio.fetch(6, "large", root=str(tmp_path))


def write_bdf(path, x_uv, names, fs=500, pmin=-200000, pmax=200000):
    """Minimal 24-bit BDF like the dataset's: 1-s records, unit uV, one physical range for every row, no Status row."""
    ns, n = x_uv.shape
    spr, nrec = int(fs), n // int(fs)
    dmin, dmax = -8388608, 8388607
    f = lambda v, w: str(v).ljust(w)[:w].encode("latin1")
    each = lambda v, w: f(v, w) * ns
    head = (b"\xffBIOSEMI" + f("X", 80) + f("X", 80) + f("11.06.19", 8) + f("10.37.00", 8) + f(256 * (ns + 1), 8)
            + f("24BIT", 44) + f(nrec, 8) + f(1, 8) + f(ns, 4) + b"".join(f(c, 16) for c in names) + each("", 80)
            + each("uV", 8) + each(pmin, 8) + each(pmax, 8) + each(dmin, 8) + each(dmax, 8) + each("", 80)
            + each(spr, 8) + each("", 32))
    d = np.round((x_uv[:, :nrec * spr] - pmin) / ((pmax - pmin) / (dmax - dmin)) + dmin).astype("<i4")
    d = d.reshape(ns, nrec, spr).transpose(1, 0, 2).reshape(-1)
    open(path, "wb").write(head + d.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())


def synth_session(root, subj=6, session="large", secs=52, names=None, events=None, seed=0):
    """BDF (48 rows; A2 = -A1, a large ecg, DC offsets) and events.tsv at their BIDS paths under root -> x_uv written."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 20.0, (48, secs * 500)) + rng.uniform(-3e4, 3e4, (48, 1))
    x[bio.CHANNELS.index("A2")] = -x[bio.CHANNELS.index("A1")]
    x[bio.CHANNELS.index("ecg")] *= 50
    p = {k: os.path.join(root, v) for k, v in bio.paths(subj, session).items()}
    os.makedirs(os.path.dirname(p["bdf"]), exist_ok=True)
    write_bdf(p["bdf"], x, names or bio.CHANNELS)
    ev = fake_bids(t0=1.0, gap=1.0, subj=subj, session=session) if events is None else events
    ev.to_csv(p["events"], sep="\t", index=False)
    return x


def no_fetch(*a, **k):
    raise AssertionError("fetch called although both files are on disk")


def test_load_reads_a_local_session_in_microvolts_without_the_network(tmp_path, monkeypatch):
    monkeypatch.setattr(bio, "fetch", no_fetch)
    x = synth_session(str(tmp_path))
    run = bio.load(6, "large", root=str(tmp_path))
    keep = [bio.CHANNELS.index(c) for c in bio.SCALP]
    assert run.x.dtype == np.float32 and run.x.shape == (45, 52 * 500) and run.names == bio.SCALP
    assert np.abs(run.x - x[keep]).max() < 0.05
    assert run.fs == 500.0 and run.subject == 6 and run.session == "large"
    pd.testing.assert_frame_equal(run.events, bio.events_table(fake_bids(t0=1.0, gap=1.0)))
    bdf = os.path.join(str(tmp_path), bio.paths(6, "large")["bdf"])
    assert run.sha256 == hashlib.sha256(open(bdf, "rb").read()).hexdigest()
    assert run.meta["a1_plus_a2_max_uv"] < 0.05 and run.meta["n_samples"] == 52 * 500
    assert run.meta["fingers"] == "{0:0, 1:1, 2:2, 3:3,4:4,5:5,6:6,7:7}"
    assert bio.check(run)["n_stim"] == 160


def test_load_fetches_when_a_file_is_missing(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(subj, session, root):
        calls.append((subj, session, root))
        synth_session(root)
        return {k: os.path.join(root, v) for k, v in bio.paths(subj, session).items()}
    monkeypatch.setattr(bio, "fetch", fake_fetch)
    run = bio.load(6, "large", root=str(tmp_path))
    assert calls == [(6, "large", str(tmp_path))] and run.x.shape[0] == 45


def test_load_refuses_a_file_whose_channels_differ_from_the_documented_48(tmp_path, monkeypatch):
    monkeypatch.setattr(bio, "fetch", no_fetch)
    names = list(bio.CHANNELS)
    names[3], names[4] = names[4], names[3]
    synth_session(str(tmp_path), names=names)
    with pytest.raises(ValueError, match="channels"):
        bio.load(6, "large", root=str(tmp_path))


def test_load_refuses_an_events_file_of_another_subject_or_session(tmp_path, monkeypatch):
    monkeypatch.setattr(bio, "fetch", no_fetch)
    synth_session(str(tmp_path), events=fake_bids(t0=1.0, gap=1.0, session="small"))
    with pytest.raises(ValueError, match="small"):
        bio.load(6, "large", root=str(tmp_path))


def test_load_refuses_events_past_the_end_of_the_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(bio, "fetch", no_fetch)
    synth_session(str(tmp_path), secs=40)
    with pytest.raises(ValueError, match="past the end"):
        bio.load(6, "large", root=str(tmp_path))


warnings.filterwarnings("ignore", r"Unknown pytest\.mark\.network", pytest.PytestUnknownMarkWarning)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("BRAILLE_NETWORK") != "1", reason="network: set BRAILLE_NETWORK=1 (downloads ~46 MB)")
def test_network_smoke_dev_subject_large_session():
    run = bio.load(bio.DEV_SUBJECT, "large")
    c = bio.check(run)
    print(f"\nsub-{run.subject} {run.session}: x {run.x.shape} {run.x.dtype}, fs {run.fs}, sha256 {run.sha256}\n"
          f"check {c}\nmeta {run.meta}")
    assert run.x.shape[0] == 45 and run.x.dtype == np.float32 and run.fs == 500.0 and run.names == bio.SCALP
    assert c["n_cycles"] == 16 and c["n_stim"] == 1280 and c["n_target"] == 160
    assert c["per_cycle"] == {"stim": [80, 80], "target": [10, 10], "per_finger": [10, 10]}


def test_events_only_fetches_the_events_file_and_never_the_bdf(tmp_path, monkeypatch):
    got = []

    def fake_download(url, path, http=None):
        got.append(url)
        ev = fake_run().events
        tsv = pd.DataFrame({"onset": ev["onset_s"], "sample": (ev["onset_s"] * bio.FS).round().astype(int),
                            "value": ev["finger"], "trial_type": ev["is_target"].astype(int), "target": ev["target"],
                            "session_id": ev["cycle"]})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tsv.to_csv(path, sep="\t", index=False)
        return path
    monkeypatch.setattr(bio, "download", fake_download)
    ev = bio.events_only(6, "large", root=str(tmp_path))
    assert len(got) == 1 and got[0].endswith("_events.tsv") and len(ev) == 160
