"""Tests for code/122_icare_arm.py (: the post-arrest coma arm on
the public I-CARE patients). Synthetic fixtures, a local stub HTTP server and the three dev patients only; no analysis
patient is ever fetched here, and no outcome value is printed. Every guard is also run against a deliberately broken
input or a broken version of the code, and the test checks that the guard catches it.
cd code && /usr/bin/python3 -m pytest test_icare_arm.py -q"""
import http.server
import importlib
import io
import json
import os
import re
import threading
import urllib.error

import numpy as np
import pandas as pd
import pytest

ic = importlib.import_module("122_icare_arm")
bn = importlib.import_module("108_bdsp_null")
rn = importlib.import_module("114_benchmark")
mx = importlib.import_module("116_mechanism")

HEA_0284 = """0284_001_004_EEG 19 500 1578500
0284_001_004_EEG.mat 16+24 17.980017665549088(23877)/nu 16 0 24177 -9802 0 Fp1
0284_001_004_EEG.mat 16+24 17.747962055418498(23144)/nu 16 0 25579 -1248 0 Fp2
0284_001_004_EEG.mat 16+24 15.71809151860282(16747)/nu 16 0 17350 -9939 0 F3
0284_001_004_EEG.mat 16+24 269.9389283253154(24279)/nu 16 0 -2188 19141 0 F4
0284_001_004_EEG.mat 16+24 698.3019922526984(-3375)/nu 16 0 -16809 2874 0 C3
0284_001_004_EEG.mat 16+24 697.5761065469161(2589)/nu 16 0 15532 32655 0 C4
0284_001_004_EEG.mat 16+24 701.9541996236107(-891)/nu 16 0 -7609 -6860 0 P3
0284_001_004_EEG.mat 16+24 49.64623711994421(30600)/nu 16 0 28835 -18301 0 P4
0284_001_004_EEG.mat 16+24 276.2734382807173(16282)/nu 16 0 12613 19441 0 O1
0284_001_004_EEG.mat 16+24 224.5126053998151(13034)/nu 16 0 8320 -5442 0 O2
0284_001_004_EEG.mat 16+24 11.380209409349538(3088)/nu 16 0 2956 -10295 0 F7
0284_001_004_EEG.mat 16+24 265.2443402064087(15309)/nu 16 0 7901 8940 0 F8
0284_001_004_EEG.mat 16+24 324.8151913727913(7391)/nu 16 0 1999 -6089 0 T3
0284_001_004_EEG.mat 16+24 11.470662849369385(3372)/nu 16 0 3322 4092 0 T4
0284_001_004_EEG.mat 16+24 17.942040512491676(23752)/nu 16 0 23517 24041 0 T5
0284_001_004_EEG.mat 16+24 125.71528465052826(27697)/nu 16 0 25156 7456 0 T6
0284_001_004_EEG.mat 16+24 632.4865310207448(7628)/nu 16 0 -8370 -30165 0 Fz
0284_001_004_EEG.mat 16+24 800.7974016256527(5318)/nu 16 0 -5865 -15701 0 Cz
0284_001_004_EEG.mat 16+24 638.5044813350738(4770)/nu 16 0 6703 -8518 0 Pz
#Utility frequency: 50
#Start time: 4:07:23
#End time: 4:59:59
"""

LISTING = """<html>
<head><title>Index of /protected/published-projects/i-care/2.1/training/0284/</title></head>
<body>
<h1>Index of /protected/published-projects/i-care/2.1/training/0284/</h1><hr><pre><a href="../">../</a>
<a href="0284.txt">0284.txt</a>                                           05-Jun-2023 02:03                 117
<a href="0284_001_004_ECG.hea">0284_001_004_ECG.hea</a>                               02-Dec-2023 04:21                 174
<a href="0284_001_004_EEG.hea">0284_001_004_EEG.hea</a>                               02-Dec-2023 04:21                1553
<a href="0284_001_004_EEG.mat">0284_001_004_EEG.mat</a>                               05-Jun-2023 01:49            59983024
<a href="0284_005_008_EEG.mat">0284_005_008_EEG.mat</a>                               05-Jun-2023 01:49             8417024
<a href="0284_006_008_EEG.mat">0284_006_008_EEG.mat</a>                               05-Jun-2023 01:50            59907024
<a href="0284_010_012_OTHER.mat">0284_010_012_OTHER.mat</a>                             05-Jun-2023 01:50             3600024
</pre><hr></body>
</html>
"""



class Stub(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.files, self.fail, self.truncate, self.log = {}, {}, {}, []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}/"


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        srv, path = self.server, self.path
        srv.log.append((path, self.headers.get("Range")))
        if srv.fail.get(path, 0) > 0:
            srv.fail[path] -= 1
            self.send_response(srv.fail_code if hasattr(srv, "fail_code") else 502)
            self.end_headers()
            return
        if path not in srv.files:
            self.send_response(404)
            self.end_headers()
            return
        data = srv.files[path]
        rng = self.headers.get("Range")
        if rng:
            a, b = re.match(r"bytes=(\d+)-(\d*)", rng).groups()
            a, b = int(a), (int(b) if b else len(data) - 1)
            b = min(b, len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if srv.truncate.get(path, 0) > 0:
            srv.truncate[path] -= 1
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture()
def stub():
    s = Stub()
    yield s
    s.shutdown()
    s.server_close()


def no_sleep(_s):
    pass



def test_listing_parse_and_segments():
    ent = ic.parse_listing(LISTING)
    assert ("0284.txt", 117) in ent and ("0284_001_004_EEG.mat", 59983024) in ent
    segs = ic.eeg_segments(ent)
    assert segs == [{"seg": 1, "hour": 4, "bytes": 59983024}, {"seg": 5, "hour": 8, "bytes": 8417024},
                    {"seg": 6, "hour": 8, "bytes": 59907024}]


def segs_at(hours, size=68400024):
    return [{"seg": i + 1, "hour": h, "bytes": size} for i, h in enumerate(hours)]


def test_candidates_nearest_24_within_12_36():
    c = ic.candidates(segs_at([3, 11, 12, 20, 23, 24, 25, 30, 36, 37, 50]))
    assert [s["hour"] for s in c] == [24, 23, 25, 20, 30, 12, 36]
    assert ic.candidates(segs_at([0, 5, 11, 37, 100])) == []


def test_candidates_same_hour_ordered_by_segment_index():
    c = ic.candidates([{"seg": 7, "hour": 24, "bytes": 1}, {"seg": 3, "hour": 24, "bytes": 1},
                       {"seg": 9, "hour": 24, "bytes": 1}])
    assert [s["seg"] for s in c] == [3, 7, 9]


def test_candidates_order_is_not_the_input_order():
    segs = segs_at([30, 24, 12])
    assert ic.candidates(segs) == ic.candidates(list(reversed(segs)))


def test_parse_header_gain_baseline_and_names():
    h = ic.parse_header(HEA_0284)
    assert h["n_sig"] == 19 and h["fs"] == 500.0 and h["n_samp"] == 1578500
    assert h["names"][0] == "Fp1" and h["names"][-1] == "Pz" and len(h["names"]) == 19
    assert h["gains"][0] == pytest.approx(17.980017665549088) and h["baselines"][0] == 23877.0
    assert h["baselines"][4] == -3375.0
    assert h["offset"] == 24



def make_f16(n_ch, n_samp, rng, offset=24):
    dig = rng.integers(-3000, 3000, (n_ch, n_samp)).astype("<i2")
    return dig, bytes(rng.integers(0, 255, offset, dtype=np.uint8)) + dig.T.tobytes()


def test_byte_range_arithmetic_and_decode_match_the_array():
    rng = np.random.default_rng(1)
    n_ch, n = 5, 5000
    dig, blob = make_f16(n_ch, n, rng)
    for i0, i1 in [(0, 10), (7, 1234), (4990, 5000), (0, 5000), (2500, 2501)]:
        a, b = ic.byte_range(n_ch, i0, i1)
        assert a == 24 + i0 * n_ch * 2 and b == 24 + i1 * n_ch * 2 - 1
        assert len(blob[a:b + 1]) == (i1 - i0) * n_ch * 2
        np.testing.assert_array_equal(ic.decode_digital(blob[a:b + 1], n_ch), dig[:, i0:i1])


def test_byte_range_refuses_empty_or_negative_spans():
    for bad in [(5, 5), (6, 5), (-1, 4)]:
        with pytest.raises(ValueError):
            ic.byte_range(4, *bad)


def test_decode_physical_units_and_invalid_sentinel():
    dig = np.array([[100, -32768, 300], [50, 60, 70]], dtype="<i2")
    hdr = {"gains": [2.0, 10.0], "baselines": [100.0, 0.0]}
    uv = ic.to_microvolts(dig, hdr)
    np.testing.assert_allclose(uv[0], [0.0, 0.0, 100.0])
    np.testing.assert_allclose(uv[1], [5.0, 6.0, 7.0])
    assert uv.dtype == np.float32


def test_sentinel_is_padding_by_the_108_rule():
    n = 500
    rng = np.random.default_rng(0)
    dig = rng.integers(-500, 500, (3, 10 * n)).astype("<i2")
    dig[1, 3 * n:4 * n] = ic.INVALID_DIGITAL
    uv = ic.to_microvolts(dig, {"gains": [1.0] * 3, "baselines": [0.0] * 3})
    pad = bn.padding_seconds(uv, n)
    assert pad[3] and pad.sum() == 1



def url_of(stub, path="/f.mat"):
    return stub.base.rstrip("/") + path


def test_fetch_range_retries_after_two_502s(stub):
    data = bytes(range(256)) * 40
    stub.files["/f.mat"] = data
    stub.fail["/f.mat"] = 2
    got = ic.fetch_range(url_of(stub), 100, 4099, sleep=no_sleep)
    assert got == data[100:4100]
    assert len(stub.log) == 3


def test_fetch_range_persistent_502_raises_and_never_returns_short(stub):
    stub.files["/f.mat"] = b"x" * 1000
    stub.fail["/f.mat"] = 10 ** 6
    with pytest.raises(ic.DownloadError):
        ic.fetch_range(url_of(stub), 0, 99, tries=5, sleep=no_sleep)
    assert len(stub.log) == 5


def test_fetch_range_does_not_retry_a_404(stub):
    with pytest.raises(ic.DownloadError):
        ic.fetch_range(url_of(stub, "/missing"), 0, 9, sleep=no_sleep)
    assert len(stub.log) == 1


def test_fetch_range_resumes_after_a_dropped_connection(stub):
    data = bytes(rng_bytes(1 << 17))
    stub.files["/f.mat"] = data
    stub.truncate["/f.mat"] = 1
    got = ic.fetch_range(url_of(stub), 1000, 1000 + 60000 - 1, sleep=no_sleep)
    assert got == data[1000:61000]
    starts = [int(re.match(r"bytes=(\d+)-", r).group(1)) for _, r in stub.log]
    assert starts[0] == 1000 and starts[1] > 1000


def test_fetch_range_refuses_a_server_that_ignores_range(stub):
    stub.files["/f.mat"] = b"abcdefgh" * 100

    class Ignore(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "800")
            self.end_headers()
            self.wfile.write(b"abcdefgh" * 100)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ignore)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ic.DownloadError):
            ic.fetch_range(f"http://127.0.0.1:{srv.server_address[1]}/f", 8, 15, tries=2, sleep=no_sleep)
    finally:
        srv.shutdown()
        srv.server_close()


def rng_bytes(n, seed=3):
    return np.random.default_rng(seed).integers(0, 255, n, dtype=np.uint8).tobytes()



FS = 500


def noise(n_s, rng, n_ch=3, sd=20.0):
    return rng.normal(0, sd, (n_ch, n_s * FS)).astype(np.float32)


class ArrayReader:
    """A segment held in memory; counts the seconds it was asked for."""

    fs = FS

    def __init__(self, x):
        self.x, self.asked = x, 0
        self.n_s = x.shape[1] // FS

    def read(self, t0, t1):
        self.asked += t1 - t0
        return self.x[:, t0 * FS:t1 * FS]


def reference_start(x):
    """108.eligible_starts itself, on the recording as read, without the channels flat over the whole segment (the
    recording reference): an independent statement of the pre-reference rule."""
    live = np.asarray(x, np.float64).std(1) >= bn.ZERO_UV
    s = bn.eligible_starts(x[live], FS, run_s=ic.WINDOW_S) if live.any() else np.array([], int)
    return int(s[0]) if len(s) else None


def test_find_window_is_the_first_eligible_start_of_108_on_the_whole_segment():
    rng = np.random.default_rng(10)
    for case in range(4):
        x = noise(2400, rng)
        if case == 1:
            x[:, 0:400 * FS] = 0.0
        if case == 2:
            x[:, 100 * FS:101 * FS] = 0.0
            x[:, 900 * FS:1000 * FS] *= 0.01
        if case == 3:
            x[:, :] = 0.0
        rd = ArrayReader(x)
        got = ic.find_window(rd.read, rd.n_s)
        assert got == reference_start(x), case
    assert reference_start(x) is None and got is None


def test_find_window_skips_padding_and_keeps_the_30_s_margin():
    rng = np.random.default_rng(11)
    x = noise(3000, rng)
    x[:, 200 * FS:210 * FS] = 0.0
    rd = ArrayReader(x)
    s = ic.find_window(rd.read, rd.n_s)
    assert s == 240
    assert s == reference_start(x)


def test_find_window_needs_the_lookahead_a_floor_stretch_straddling_the_prefix_edge():
    """A noise-floor stretch (>= 60 s) that starts just inside the first 1,830 s is only recognised once 60 s more are
    read; a prefix-only decision would accept start 0. Broken version: decide on the 1,830-s prefix."""
    rng = np.random.default_rng(12)
    x = noise(4000, rng)
    x[:, 1820 * FS:1900 * FS] *= 0.01
    rd = ArrayReader(x)
    s = ic.find_window(rd.read, rd.n_s)
    assert s == reference_start(x) and s > 0
    prefix = x[:, :1830 * FS]
    assert reference_start(prefix) == 0


def test_find_window_reads_little_when_the_start_is_clean():
    rng = np.random.default_rng(13)
    x = noise(3000, rng)
    rd = ArrayReader(x)
    assert ic.find_window(rd.read, rd.n_s) == 0
    assert rd.asked <= ic.WINDOW_S + ic.MARGIN_S + ic.FLOOR_S + ic.BLOCK_S


def test_find_window_none_when_the_segment_is_too_short():
    rng = np.random.default_rng(14)
    x = noise(1700, rng)
    rd = ArrayReader(x)
    assert ic.find_window(rd.read, rd.n_s) is None


def test_find_window_at_file_end_needs_no_lookahead():
    rng = np.random.default_rng(15)
    x = noise(1800, rng)
    rd = ArrayReader(x)
    assert ic.find_window(rd.read, rd.n_s) == 0



class FakeSource:
    """selection source: segments by (seg) -> an ArrayReader; counts the segments opened."""

    def __init__(self, arrays, sizes=None):
        self.arrays, self.opened, self.readers = arrays, [], {}

    def open(self, pid, segment):
        self.opened.append(segment["seg"])
        x = self.arrays[segment["seg"]]
        self.readers[segment["seg"]] = ArrayReader(x)
        return self.readers[segment["seg"]], x.shape[1] // FS


def seg(i, hour, n_s):
    return {"seg": i, "hour": hour, "bytes": ic.HEADER_BYTES + n_s * FS * 19 * 2}


def test_select_window_takes_the_segment_nearest_24_h():
    rng = np.random.default_rng(20)
    arrays = {1: noise(2000, rng), 2: noise(2000, rng), 3: noise(2000, rng)}
    src = FakeSource(arrays)
    out = ic.select_window("0001", [seg(1, 20, 2000), seg(2, 24, 2000), seg(3, 30, 2000)], src)
    assert out["status"] == "ok" and out["segment"]["seg"] == 2 and out["start_s"] == 0
    assert src.opened == [2]


def test_select_window_fallback_order_and_reasons():
    rng = np.random.default_rng(22)
    padded = np.zeros((3, 2000 * FS), np.float32)
    arrays = {1: noise(2000, rng), 2: padded, 3: noise(2000, rng)}
    src = FakeSource(arrays)
    out = ic.select_window("0001", [seg(1, 22, 2000), seg(2, 24, 2000), seg(3, 25, 2000)], src)
    assert out["status"] == "ok" and out["segment"]["seg"] == 3 and out["segment"]["hour"] == 25
    assert src.opened == [2, 3]
    assert out["tried"] == [{"seg": 2, "hour": 24, "result": "no_clean_window"}]


def test_select_window_too_short_segments_are_skipped_without_reading():
    rng = np.random.default_rng(23)
    arrays = {1: noise(1000, rng), 2: noise(2000, rng)}
    src = FakeSource(arrays)
    out = ic.select_window("0001", [seg(1, 24, 1000), seg(2, 26, 2000)], src)
    assert out["segment"]["seg"] == 2 and src.opened == [1, 2]
    assert src.readers[1].asked == 0
    assert out["tried"] == [{"seg": 1, "hour": 24, "result": "too_short"}]


def test_select_window_start_inside_the_segment_when_the_head_is_padded():
    rng = np.random.default_rng(24)
    x = noise(3000, rng)
    x[:, :500 * FS] = 0.0
    out = ic.select_window("0001", [seg(1, 24, 3000)], FakeSource({1: x}))
    assert out["status"] == "ok" and out["start_s"] == 530


def test_select_window_excludes_with_a_reason():
    out = ic.select_window("0001", segs_at([3, 5, 40]), FakeSource({}))
    assert out["status"] == "excluded" and out["reason"] == "no_eeg_segment_12_36h"
    out = ic.select_window("0001", [seg(1, 24, 1000)], FakeSource({1: np.zeros((3, 1000 * FS), np.float32)}))
    assert out["status"] == "excluded" and out["reason"] == "no_segment_long_enough"
    out = ic.select_window("0001", [seg(1, 24, 2000), seg(2, 26, 2000)],
                           FakeSource({1: np.zeros((3, 2000 * FS), np.float32),
                                       2: np.zeros((3, 2000 * FS), np.float32)}))
    assert out["status"] == "excluded" and out["reason"] == "no_clean_30min_window"
    assert [t["seg"] for t in out["tried"]] == [1, 2]


def test_rosc_hours_of_the_window_is_the_segment_hour_plus_the_window_midpoint():
    assert ic.rosc_hours(24, 0) == pytest.approx(24.25)
    assert ic.rosc_hours(23, 1800) == pytest.approx(23 + (1800 + 900) / 3600)



def test_resample_window_length_alignment_and_passband():
    fs_in, fs_out = 500, 128
    t = np.arange(0, 2000 * fs_in) / fs_in
    x = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 10 * t + 1.0)]).astype(np.float32)
    y = ic.resample_window(x, fs_in, 100, ic.WINDOW_S, fs_out)
    assert y.shape == (2, ic.WINDOW_S * fs_out) and y.dtype == np.float32
    ts = 100 + np.arange(ic.WINDOW_S * fs_out) / fs_out
    np.testing.assert_allclose(y[0], np.sin(2 * np.pi * 10 * ts), atol=2e-3)



IDS = [f"{i:04d}" for i in (284, 286, 296, 299, 303, 306, 307, 309, 314, 316, 318, 320)]


def test_processing_order_is_seeded_reproducible_and_input_order_free():
    a = ic.processing_order(IDS)
    assert a == ic.processing_order(IDS) == ic.processing_order(list(reversed(IDS)))
    want = np.random.default_rng(20261014).permutation(sorted(IDS))
    assert a == [str(v) for v in want]
    assert a != sorted(IDS)
    assert sorted(a) == sorted(IDS)


def test_processing_order_changes_with_the_seed():
    assert ic.processing_order(IDS, seed=1) != ic.processing_order(IDS)


def test_dev_patients_are_three_seeded_and_never_among_the_d3_n2_40():
    n2 = {"0303", "0306"}
    d = ic.pick_dev(IDS, n2)
    assert len(d) == 3 and set(d) <= set(IDS) - n2
    assert d == ic.pick_dev(list(reversed(IDS)), n2)
    pool = sorted(set(IDS) - n2)
    want = np.random.default_rng(20261014).choice(pool, 3, replace=False)
    assert d == [str(v) for v in want]





TXT = "Patient: 0284\nHospital: A\nAge: 53\nSex: Male\nROSC: nan\nOHCA: True\nShockable Rhythm: True\nTTM: 33\n" \
      "Outcome: Good\nCPC: 1\n"
TXT2 = "Patient: 0286\nHospital: B\nAge: 70\nSex: Female\nROSC: 12.0\nOHCA: False\nShockable Rhythm: nan\nTTM: nan\n" \
       "Outcome: Poor\nCPC: 5\n"


def test_parse_metadata_splits_covariates_from_outcome():
    cov, out = ic.parse_metadata(TXT)
    assert cov == {"patient": "0284", "hospital": "A", "age": 53.0, "sex": "Male", "rosc_min": None, "ohca": True,
                   "shockable": True, "ttm": 33.0}
    assert out == {"patient": "0284", "outcome": "Good", "cpc": 1}
    cov2, out2 = ic.parse_metadata(TXT2)
    assert cov2["shockable"] is None and cov2["ttm"] is None and cov2["ohca"] is False
    assert out2 == {"patient": "0286", "outcome": "Poor", "cpc": 5}
    assert not ({"outcome", "cpc"} & set(cov))


def stub_corpus(stub):
    stub.files["/RECORDS"] = b"training/0284/\ntraining/0286/\n"
    stub.files["/training/0284/"] = LISTING.encode()
    stub.files["/training/0286/"] = LISTING.replace("0284", "0286").encode()
    stub.files["/training/0284/0284.txt"] = TXT.encode()
    stub.files["/training/0286/0286.txt"] = TXT2.encode()


def test_index_writes_outcomes_to_a_separate_file_and_prints_none(stub, tmp_path, capsys, monkeypatch):
    stub_corpus(stub)
    monkeypatch.setattr(ic, "BASE_URL", stub.base)
    monkeypatch.setattr(ic, "OUT", str(tmp_path))
    ic.build_index(sleep=no_sleep, workers=2)
    cap = capsys.readouterr()
    shown = (cap.out + cap.err).lower()
    assert "good" not in shown and "poor" not in shown and "cpc" not in shown and "outcome" not in shown
    cov = pd.read_csv(tmp_path / "_icare_index" / "covariates.csv", dtype={"patient": str})
    assert not ({"outcome", "cpc"} & set(c.lower() for c in cov.columns)) and len(cov) == 2
    outc = pd.read_csv(tmp_path / "_icare_outcome" / "outcomes.csv", dtype={"patient": str})
    assert set(outc.columns) == {"patient", "outcome", "cpc"} and set(outc.outcome) == {"Good", "Poor"}
    idx = json.load(open(tmp_path / "_icare_index" / "index.json"))
    assert "Good" not in json.dumps(idx) and "Poor" not in json.dumps(idx)


def test_the_output_check_catches_a_leak():
    """Broken version: a leak detector must trip on an outcome word."""
    with pytest.raises(AssertionError):
        ic.assert_no_outcome_text("patient 0284 Good")
    ic.assert_no_outcome_text("patient 0284 indexed 5 segments")
    with pytest.raises(AssertionError):
        ic.assert_no_outcome_text("CPC 3")



def test_memory_free_percentage_parse():
    txt = "The system has 17179869184 (1048576 pages with a page size of 16384).\n...\nSystem-wide memory free " \
          "percentage: 40%\n"
    assert ic.parse_free_pct(txt) == 40
    with pytest.raises(ValueError):
        ic.parse_free_pct("nothing here")






en = rn.en
DEV_SHARDS = os.path.join(rn.SH["dev"], "ncmd_bdsp__cornell__native__none")


def pink(n_ch, n, rng, sd=20.0):
    f = np.fft.rfftfreq(n)
    amp = np.zeros(len(f))
    amp[1:] = f[1:] ** -0.5
    spec = amp * (rng.normal(size=(n_ch, len(f))) + 1j * rng.normal(size=(n_ch, len(f))))
    x = np.fft.irfft(spec, n=n, axis=1)
    return (x / x.std(1, keepdims=True) * sd).astype(np.float32)


def morgoth_stub(x, names, pid):
    """A Morgoth-shaped frame, one row per 10-s window starting every second: all 'normal awake'."""
    n = x.shape[1] // 128 - 9
    return pd.DataFrame({"t_s": np.arange(n, dtype=np.int32), "valid": True, "valid_sleep": True,
                         "normal_p": 0.9, "bs_p": 0.01, "iiic_other": 0.99, "slowing_generalized": 0.05,
                         "sleep_awake": 0.9, "sleep_n1": 0.05, "sleep_n2": 0.02, "sleep_n3": 0.01,
                         "sleep_rem": 0.02})


@pytest.fixture(scope="module")
def window_x():
    return pink(19, ic.WINDOW_S * ic.FS_OUT, np.random.default_rng(5))


WIN_META = {"segment": {"seg": 3, "hour": 24, "bytes": 1}, "start_s": 0, "rosc_hours": 24.25, "tried": [],
            "names": list(en.NAMES19)}
_X0 = pink(19, ic.WINDOW_S * ic.FS_OUT, np.random.default_rng(5))
WIN_META.update(ic.window_padding(_X0, ic.median_reference(_X0), list(en.NAMES19)))


@pytest.fixture(scope="module")
def processed(window_x, tmp_path_factory):
    out = tmp_path_factory.mktemp("icare_out")
    sp = mx.Spectra(None)
    res = ic.process_patient("0900", window_x, WIN_META, "dev", morgoth_stub, sp, n_recs=607, out=str(out),
                             pipes=("cornell",))
    return out, res


def jl(path):
    lines = [ln for ln in open(path) if ln.strip()] if os.path.exists(path) else ic.read_lines(path)
    return [json.loads(ln) for ln in lines]


def xjson(out, pid, split="dev"):
    return ic.read_data_json(ic.paths(split, str(out))["x"] + f"/{pid}.json")




def test_rows_are_json_with_no_nan(processed):
    out, _ = processed
    for ln in ic.read_lines(ic.paths("dev", str(out))["rows"] + "/0900.jsonl"):
        json.loads(ln, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_x_per_first_overlay_row_and_x_ref(processed):
    out, res = processed
    x = xjson(out, "0900")
    assert x["sha256_122"] == ic.own_sha256() and x["sha256_116"] == mx.own_sha256()
    rows = [r for r in jl(ic.paths("dev", str(out))["rows"] + "/0900.jsonl") if r["status"] == "ok" and r["j"] == 0]
    assert sorted(r["key"] for r in rows) == sorted(r["key"] for r in x["x_rows"])
    assert all(r["source"] == "icare" for r in x["x_rows"])
    assert any(r["X"] is not None for r in x["x_rows"])
    w = x["xref_windows"]
    assert len(w) == 3
    ok = [v["X"] for v in w if v.get("skipped") is None and v.get("X") is not None]
    assert x["xref"]["n_windows_used"] == len(ok) and len(ok) >= 1
    assert x["xref"]["X_ref"] == pytest.approx(float(np.median(ok)))


def test_x_ref_is_116s_s4_recording_on_the_window(processed, window_x):
    out, _ = processed
    rec = en.Recording(source="bdsp", uid="0900", patient="0900", x=ic.median_reference(window_x), fs=128.0,
                       names=list(en.NAMES19))
    want = [r for r in mx.s4_recording(rec, mx.s4_spectrum()) if r.get("skipped") is None]
    x = xjson(out, "0900")
    got = [r for r in x["xref_windows"] if r.get("skipped") is None]
    nan = lambda v: np.nan if v is None else v
    assert [nan(r["X"]) for r in got] == pytest.approx([nan(r["X"]) for r in want], nan_ok=True)


def test_morgoth_summary_and_labels_are_stored(processed):
    out, _ = processed
    x = xjson(out, "0900")
    assert x["morgoth"]["stratum"] == "normal_awake" and x["morgoth"]["transitioning"] is False
    assert os.path.exists(ic.paths("dev", str(out))["morgoth"] + "/0900.parquet")
    assert x["window"]["rosc_hours"] == 24.25


def test_spindle_features_are_stored_for_the_first_overlay_windows(processed):
    out, _ = processed
    x = xjson(out, "0900")
    assert x["sha256_123"] and len(x["spindle"]) >= 1
    assert {"t0", "t1", "a7_density", "sigma_slow_rel"} <= set(x["spindle"][0])


def test_window_quality_is_stored(processed):
    out, _ = processed
    x = xjson(out, "0900")
    assert set(x["channel_sd_uv"]) == set(en.NAMES19) and all(v > 1 for v in x["channel_sd_uv"].values())
    assert 0.0 <= x["artifact_share"] <= 1.0 and x["artifact_label"] in ("<10%", "10-30%", ">30%")


def test_covariates_and_outcome_are_not_in_the_patient_files(processed):
    out, _ = processed
    blob = json.dumps(xjson(out, "0900")) + "\n".join(ic.read_lines(ic.paths("dev", str(out))["rows"] + "/0900.jsonl"))
    assert not OUTCOME_RE.search(blob)


OUTCOME_RE = re.compile(r'"(outcome|cpc)"|\bGood\b|\bPoor\b', re.I)


def test_a_dev_run_prints_no_outcome(window_x, tmp_path, capsys):
    ic.process_patient("0901", window_x, WIN_META, "dev", morgoth_stub, mx.Spectra(None), n_recs=607,
                       out=str(tmp_path), pipes=("cornell",))
    cap = capsys.readouterr()
    assert cap.out.strip() and not OUTCOME_RE.search(cap.out + cap.err)


def test_say_refuses_an_outcome_word(capsys):
    with pytest.raises(AssertionError):
        ic.say("0284 Poor")
    ic.say("0284 done")
    assert "done" in capsys.readouterr().out


def test_uid_and_patient_are_the_id_and_the_split_flag_follows(window_x, tmp_path):
    ic.process_patient("0902", window_x, WIN_META, "analysis", morgoth_stub, mx.Spectra(None), n_recs=607,
                       out=str(tmp_path), pipes=("cornell",))
    rows = jl(ic.paths("analysis", str(tmp_path))["rows"] + "/0902.jsonl")
    assert {r["split"] for r in rows} == {"analysis"} and {r["dev"] for r in rows} == {False}


def test_one_overlay_per_recording_at_607_patients():
    assert rn.m_overlays(607) == 1



class RawSource:
    """A 19-channel 500-Hz segment per segment index; channel c carries the constant c * 5 plus noise, in a shuffled
    file order, so the reorder to the engine's channel order can be checked."""

    def __init__(self, n_s=1900, shuffle=(3, 0, 1, 2)):
        rng = np.random.default_rng(31)
        self.order = list(rng.permutation(19))
        self.names = [en.NAMES19[i] for i in self.order]
        self.n_s, self.opened, self.hdr = n_s, [], None
        self.x = np.stack([5.0 * i + rng.normal(0, 20, n_s * FS) for i in self.order]).astype(np.float32)

    def open(self, pid, segment):
        self.opened.append((pid, segment["seg"]))
        self.hdr = {"names": self.names}
        return ArrayReader(self.x), self.n_s


def test_fetch_stage_reorders_channels_and_writes_only_the_window(tmp_path):
    src = RawSource()
    segs = [{"seg": 4, "hour": 24, "bytes": ic.HEADER_BYTES + src.n_s * FS * 19 * 2}]
    res = ic.fetch_window("0903", segs, src, "dev", out=str(tmp_path))
    assert res["status"] == "ok"
    st = np.load(res["stage"])
    x = st["x"]
    assert x.shape == (19, ic.WINDOW_S * ic.FS_OUT) and x.dtype == np.float32
    assert list(st["names"]) == list(en.NAMES19)
    means = x.mean(1)
    assert means == pytest.approx([5.0 * i - 45.0 for i in range(19)], abs=3.0)
    assert np.abs(np.median(x, axis=0)).max() < 0.05
    assert json.loads(str(st["window"]))["start_s"] == 0
    files = [f for r, _, fs in os.walk(tmp_path) for f in fs]
    assert files == [os.path.basename(res["stage"])]


def test_fetch_stage_records_an_exclusion_with_its_reason(tmp_path):
    res = ic.fetch_window("0904", [{"seg": 1, "hour": 3, "bytes": 1}], RawSource(), "dev", out=str(tmp_path))
    assert res["status"] == "excluded" and res["reason"] == "no_eeg_segment_12_36h"
    ex = [json.loads(ln) for ln in open(ic.paths("dev", str(tmp_path))["window"] + "/0904.json")]
    assert ex[0]["status"] == "excluded" and ex[0]["reason"] == "no_eeg_segment_12_36h"


def test_fetch_failure_is_reported_not_excluded(tmp_path):
    class Boom:
        def open(self, pid, seg):
            raise ic.DownloadError("persistent")
    segs = [{"seg": 1, "hour": 24, "bytes": ic.HEADER_BYTES + 1900 * FS * 19 * 2}]
    res = ic.fetch_window("0905", segs, Boom(), "dev", out=str(tmp_path))
    assert res["status"] == "download_failed"
    assert not os.path.exists(ic.paths("dev", str(tmp_path))["window"] + "/0905.json")



def small_index(n):
    return {f"{9000 + i:04d}": [{"seg": 1, "hour": 24, "bytes": ic.HEADER_BYTES + 1900 * FS * 19 * 2}]
            for i in range(n)}


class SpyProcess:
    def __init__(self):
        self.seen = []

    def __call__(self, pid, x, win, split, label_fn, spectra, n_recs, out, pipes):
        self.seen.append(pid)
        _stub_done(pid, split, out)


def _stub_done(pid, split, out):
    p = ic.paths(split, out)
    os.makedirs(p["x"], exist_ok=True)
    ic._atomic_json(os.path.join(p["x"], f"{pid}.json"), {"patient": pid, "sha256_122": ic.own_sha256()})


def test_run_arm_processes_in_the_given_seeded_order_and_deletes_every_staged_window(tmp_path, monkeypatch):
    spy = SpyProcess()
    monkeypatch.setattr(ic, "process_patient", spy)
    idx = small_index(4)
    order = ic.processing_order(list(idx))
    src = RawSource()
    counts = ic.run_arm(order, idx, "analysis", str(tmp_path), make_source=lambda: src, workers=1, prefetch=1,
                        pause=lambda: None, sleep=no_sleep)
    assert spy.seen == order and counts["done"] == 4
    stage = ic.paths("analysis", str(tmp_path))["stage"]
    assert os.path.isdir(stage) and [f for f in os.listdir(stage) if not f.startswith("._")] == []


def test_run_arm_resumes_and_skips_finished_patients(tmp_path, monkeypatch):
    spy = SpyProcess()
    monkeypatch.setattr(ic, "process_patient", spy)
    idx = small_index(3)
    ids = sorted(idx)
    _stub_done(ids[0], "analysis", str(tmp_path))
    ic.run_arm(ids, idx, "analysis", str(tmp_path), make_source=lambda: RawSource(), workers=1, prefetch=1,
               pause=lambda: None, sleep=no_sleep)
    assert spy.seen == ids[1:]


def test_run_arm_refuses_a_result_from_another_version(tmp_path):
    p = ic.paths("analysis", str(tmp_path))
    os.makedirs(p["x"])
    ic._atomic_json(os.path.join(p["x"], "9000.json"), {"patient": "9000", "sha256_122": "0" * 64})
    with pytest.raises(RuntimeError, match="another version"):
        ic.is_finished("9000", "analysis", str(tmp_path))


def test_run_arm_retries_a_failed_download_and_never_excludes_it(tmp_path, monkeypatch):
    spy = SpyProcess()
    monkeypatch.setattr(ic, "process_patient", spy)
    idx = small_index(2)
    calls = {"n": 0}

    class Flaky(RawSource):
        def open(self, pid, segment):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ic.DownloadError("502 twice too many")
            return super().open(pid, segment)
    src = Flaky()
    counts = ic.run_arm(sorted(idx), idx, "analysis", str(tmp_path), make_source=lambda: src, workers=1, prefetch=1,
                        pause=lambda: None, sleep=no_sleep)
    assert counts["done"] == 2 and counts["download_failed"] == 0 and counts["excluded"] == 0
    assert sorted(spy.seen) == sorted(idx)
    assert "download" in open(ic.paths("analysis", str(tmp_path))["failures"]).read()


def test_run_arm_counts_exclusions_and_a_processing_error_keeps_the_window(tmp_path, monkeypatch):
    idx = small_index(2)
    idx["9100"] = [{"seg": 1, "hour": 5, "bytes": 1}]

    def boom(pid, x, win, split, label_fn, spectra, n_recs, out, pipes):
        raise ValueError("bug")
    monkeypatch.setattr(ic, "process_patient", boom)
    counts = ic.run_arm(["9000", "9100", "9001"], idx, "analysis", str(tmp_path), make_source=lambda: RawSource(),
                        workers=1, prefetch=1, pause=lambda: None, sleep=no_sleep)
    assert counts["excluded"] == 1 and counts["process_error"] == 2 and counts["done"] == 0
    assert sorted(os.listdir(ic.paths("analysis", str(tmp_path))["stage"])) == ["9000.npz", "9001.npz"]


def test_dev_patients_are_drawn_once_and_stored(tmp_path):
    idx = small_index(12)
    d1 = ic.dev_patients(idx, str(tmp_path), d3=["9001"])
    assert len(d1) == 3 and "9001" not in d1
    assert ic.dev_patients({"9999": idx["9000"]}, str(tmp_path), d3=[]) == d1



def make_hospital_file(stub, pid, seg, hour, fs, names, n_s, rng):
    """A format-16 file of the I-CARE layout (24-byte offset, sample-interleaved) and its .hea on the stub server."""
    n_ch, n = len(names), n_s * fs
    gains = rng.uniform(5, 50, n_ch)
    bases = rng.integers(-5000, 5000, n_ch)
    dig = rng.integers(-2000, 2000, (n_ch, n)).astype("<i2")
    stem = f"{pid}_{seg:03d}_{hour:03d}_EEG"
    hea = f"{stem} {n_ch} {fs} {n}\n" + "".join(
        f"{stem}.mat 16+24 {gains[i]}({bases[i]})/nu 16 0 0 0 0 {names[i]}\n" for i in range(n_ch))
    blob = bytes(24) + dig.T.tobytes()
    stub.files[f"/training/{pid}/{stem}.hea"] = hea.encode()
    stub.files[f"/training/{pid}/{stem}.mat"] = blob
    uv = ((dig.astype(float) - bases[:, None]) / gains[:, None]).astype(np.float32)
    return len(blob), uv


@pytest.mark.parametrize("fs", [256, 200, 512])
def test_http_source_picks_the_19_sites_in_engine_order_at_the_files_rate(stub, fs):
    rng = np.random.default_rng(7)
    sites = list(en.NAMES19)

    names = list(rng.permutation(sites + ["Fpz", "F9"]))
    size, uv = make_hospital_file(stub, "0700", 3, 24, fs, names, 30, rng)
    src = ic.HttpSource(base=stub.base, sleep=no_sleep)
    reader, total_s = src.open("0700", {"seg": 3, "hour": 24, "bytes": size})
    assert total_s == 30 and reader.fs == fs and src.hdr["names"] == list(en.NAMES19)
    got = reader.read(4, 9)
    want = uv[[names.index(c) for c in en.NAMES19]][:, 4 * fs:9 * fs]
    assert got.shape == (19, 5 * fs)
    np.testing.assert_allclose(got, want, atol=1e-3)
    assert stub.log[-1][1] == f"bytes={24 + 4 * fs * 21 * 2}-{24 + 9 * fs * 21 * 2 - 1}"


def test_http_source_missing_site_is_a_segment_error_and_the_next_segment_is_tried(stub):
    rng = np.random.default_rng(8)
    sites = list(en.NAMES19)
    size1, _ = make_hospital_file(stub, "0701", 1, 24, 256, [c for c in sites if c != "Cz"], 30, rng)
    size2, _ = make_hospital_file(stub, "0701", 2, 25, 256, sites, 2000, rng)
    src = ic.HttpSource(base=stub.base, sleep=no_sleep)
    with pytest.raises(ic.SegmentError):
        src.open("0701", {"seg": 1, "hour": 24, "bytes": size1})
    out = ic.select_window("0701", [{"seg": 1, "hour": 24, "bytes": size1}, {"seg": 2, "hour": 25, "bytes": size2}],
                           src)
    assert out["status"] == "ok" and out["segment"]["seg"] == 2 and out["fs"] == 256
    assert out["tried"][0]["result"].startswith("unreadable")
    only = ic.select_window("0701", [{"seg": 1, "hour": 24, "bytes": size1}], src)
    assert only["status"] == "excluded" and only["reason"] == "no_readable_segment"


def test_http_source_refuses_a_size_mismatch(stub):
    rng = np.random.default_rng(9)
    size, _ = make_hospital_file(stub, "0702", 1, 24, 256, list(en.NAMES19), 10, rng)
    with pytest.raises(ic.SegmentError):
        ic.HttpSource(base=stub.base, sleep=no_sleep).open("0702", {"seg": 1, "hour": 24, "bytes": size + 2})


@pytest.mark.parametrize("fs_in", [200, 250, 256, 512, 1024])
def test_resample_window_any_rate(fs_in):
    t = np.arange(0, 60 * fs_in) / fs_in
    x = np.sin(2 * np.pi * 7 * t)[None].astype(np.float32)
    y = ic.resample_window(x, fs_in, 20, 30, 128)
    assert y.shape == (1, 30 * 128)
    ts = 20 + np.arange(30 * 128) / 128
    np.testing.assert_allclose(y[0], np.sin(2 * np.pi * 7 * ts), atol=3e-3)


def test_http_source_end_to_end_window_from_a_stub_hospital(stub, tmp_path):
    """Header, ranges, 108's rule, resampling, reordering and staging against a 256-Hz, 21-channel file."""
    rng = np.random.default_rng(10)
    names = list(rng.permutation(list(en.NAMES19) + ["Fpz", "F9"]))
    size, uv = make_hospital_file(stub, "0703", 5, 24, 256, names, 1950, rng)
    res = ic.fetch_window("0703", [{"seg": 5, "hour": 24, "bytes": size}], ic.HttpSource(base=stub.base,
                                                                                         sleep=no_sleep),
                          "dev", out=str(tmp_path))
    assert res["status"] == "ok" and res["window"]["fs_in"] == 256 and res["window"]["n_channels_in_file"] == 21
    x, win = ic.load_stage(res["stage"])
    assert x.shape == (19, ic.WINDOW_S * 128) and win["names"] == list(en.NAMES19)
    ranges = [r for _, r in stub.log if r]
    assert win["start_s"] == 0 and len(ranges) == 7 and ranges[0].startswith("bytes=24-")
    assert sum(1 for p, r in stub.log if p.endswith(".hea")) == 1



HOSPITALS = ["A", "B", "D", "E", "F"]
STRATA = ["normal_awake", "sleep", "diffuse_slowing", "iiic_periodic", "burst_suppression", "other_awake", "mixed"]
DEV_IDS = ["0990", "0991", "0992"]


def synth_arm(root, n=360, effect=0.0, seed=0, n_excluded=12, bg_effect=0.6):
    """A finished arm on disk (index, covariates, the outcome in its own file, per-patient files). Good outcome depends
    on z(X_ref) with log-odds `effect` per SD, on the background and on the hospital. Returns the patient table."""
    rng = np.random.default_rng(seed)
    ids = [f"{1000 + i:04d}" for i in range(n)]
    p = ic.paths("analysis", str(root))
    for k in ("index", "outcome", "x", "window", "rows"):
        os.makedirs(p[k], exist_ok=True)
    allids = ids + DEV_IDS
    ic._atomic_json(os.path.join(p["index"], "index.json"), {i: [{"seg": 1, "hour": 24, "bytes": 1}] for i in allids})
    ic._atomic_json(os.path.join(p["index"], "dev_patients.json"), DEV_IDS)
    hosp = rng.choice(HOSPITALS, len(allids))
    age = rng.normal(62, 12, len(allids)).round()
    ohca = rng.random(len(allids)) < 0.9
    shock = rng.random(len(allids)) < 0.5
    ttm = rng.choice([33.0, 36.0, np.nan], len(allids))
    xref = rng.normal(0.2, 0.5, len(allids))
    strat = rng.choice(STRATA, len(allids), p=[0.1, 0.05, 0.3, 0.2, 0.2, 0.1, 0.05])
    rosc = rng.uniform(20, 28, len(allids))
    grade = rng.choice(["continuous", "discontinuous", "burst_suppression", "suppressed"], len(allids),
                       p=[0.3, 0.3, 0.3, 0.1])
    volt = np.where(grade == "suppressed", np.nan, rng.normal(3.5, 0.5, len(allids)))
    nflat = (rng.random(len(allids)) < 0.15).astype(int)
    org = grade == "continuous"
    z = (xref - xref.mean()) / xref.std()
    eta = -0.2 + effect * z + bg_effect * org - 0.02 * (age - 62) + 0.4 * shock + 0.3 * (np.array(hosp) == "A")
    good = rng.random(len(allids)) < 1 / (1 + np.exp(-eta))
    pd.DataFrame({"patient": allids, "hospital": hosp, "age": age, "sex": "Male", "rosc_min": np.nan, "ohca": ohca,
                  "shockable": shock, "ttm": ttm}).to_csv(os.path.join(p["index"], "covariates.csv"), index=False)
    pd.DataFrame({"patient": allids, "outcome": np.where(good, "Good", "Poor"),
                  "cpc": np.where(good, 1, 4)}).to_csv(os.path.join(p["outcome"], "outcomes.csv"), index=False)
    for i, pid in enumerate(allids):
        if pid in ids[len(ids) - n_excluded:]:
            ic._atomic_json(os.path.join(p["window"], f"{pid}.json"), {"patient": pid, "status": "excluded",
                                                                       "reason": "no_clean_30min_window",
                                                                       "padding_rule": ic.PADDING_RULE})
            continue
        ic._atomic_json(os.path.join(p["x"], f"{pid}.json"), {
            "patient": pid, "sha256_122": ic.own_sha256(),
            "window": {"rosc_hours": float(rosc[i]), "n_flat": int(nflat[i])},
            "acns": {"S": 0.0, "grade": str(grade[i]), "voltage_log_uv": None if np.isnan(volt[i]) else float(volt[i]),
                     "voltage_na": bool(np.isnan(volt[i])), "periodic": bool(rng.random() < 0.1)},
            "xref": {"X_ref": float(xref[i]) if pid != ids[3] else None, "n_windows": 3, "n_windows_used": 3},
            "morgoth": {"stratum": str(strat[i]), "transitioning": False}, "x_rows": []})
    return ids


def test_analysis_order_excludes_dev_and_is_seeded(tmp_path):
    ids = synth_arm(tmp_path, n=30)
    idx = ic.load_index(str(tmp_path))
    dev = ic.dev_patients(idx, str(tmp_path))
    assert dev == DEV_IDS
    order = ic.analysis_order(idx, dev)
    assert set(order) == set(ids) and order == [i for i in ic.processing_order(list(idx)) if i not in DEV_IDS]
    assert ic.analysis_order(idx, dev, 10) == order[:10]








def test_background_grade_mapping():
    g = ic.background_grade
    assert [g(s) for s in ["normal_awake", "sleep", "diffuse_slowing", "iiic_periodic", "burst_suppression",
                           "other_awake", "mixed", "unlabelled", None]] == \
        ["organised", "organised", "diffuse_slowing", "iiic_periodic", "burst_suppression", "other", "other", "other",
         "other"]


def test_design_matrix_encoding(tmp_path):
    d = pd.DataFrame({"patient": ["a", "b", "c", "d"], "hospital": list("ABAB"), "outcome_good": [1.0, 0, 1, 0],
                      "xref": [0.0, 1.0, 2.0, 3.0], "stratum": ["sleep", "burst_suppression", "mixed", "diffuse_slowing"],
                      "acns_grade": ["continuous", "discontinuous", "burst_suppression", "suppressed"],
                      "acns_periodic": [False, True, False, False], "acns_voltage": [3.0, 2.0, 1.0, None],
                      "rosc_hours": [24.0, 25, 26, 27], "age": [50.0, 60, 70, 80], "ohca": [True, False, True, True],
                      "shockable": [True, True, False, False], "ttm": [33.0, 36.0, np.nan, 33.0]})
    frame, covs, (m, sd) = ic.design_matrix(d)
    assert list(covs) == ["z_xref", "acns_discontinuous", "acns_burst_suppression", "acns_suppressed", "periodic",
                          "voltage", "voltage_na", "rosc_hours", "age", "ohca", "shockable", "ttm33", "ttm36"]
    np.testing.assert_allclose(covs["z_xref"], (np.array([0, 1, 2, 3.0]) - 1.5) / np.std([0, 1, 2, 3.0]))
    assert covs["acns_discontinuous"].tolist() == [0, 1, 0, 0] and covs["acns_suppressed"].tolist() == [0, 0, 0, 1]
    assert covs["acns_burst_suppression"].tolist() == [0, 0, 1, 0]
    assert covs["periodic"].tolist() == [0, 1, 0, 0]
    assert covs["voltage"].tolist() == [3.0, 2.0, 1.0, 1.0] and covs["voltage_na"].tolist() == [0, 0, 0, 1]
    assert covs["ttm33"].tolist() == [1, 0, 0, 1] and covs["ttm36"].tolist() == [0, 1, 0, 0]
    assert frame["call"].tolist() == [1, 0, 1, 0] and frame["fe"].tolist() == list("ABAB")
    _, mc_, _ = ic.design_matrix(d, background="morgoth")
    assert list(mc_)[1:5] == ["bg_diffuse_slowing", "bg_iiic_periodic", "bg_burst_suppression", "bg_other"]
    assert mc_["bg_burst_suppression"].tolist() == [0, 1, 0, 0] and mc_["bg_other"].tolist() == [0, 0, 1, 0]


def kept_of(tmp_path, **kw):
    ids = synth_arm(tmp_path, **kw)
    idx = ic.load_index(str(tmp_path))
    t = ic.patient_table(ic.analysis_order(idx, DEV_IDS), idx, str(tmp_path))
    return ic.model_set(t)[0], t








def test_primary_constants_are_the_pinned_ones():
    assert ic.PRIMARY_B == 2000 and ic.ALPHA == 0.05 and ic.SEED == 20261014
    assert mx.PENALTY_C == 100.0 and ic.N_CONN == 6





















@pytest.fixture(scope="module")
def first_rows(processed):
    out, _ = processed
    rows = ic.load_rows(["0900"], "dev", str(out))
    return ic.first_overlay_with_x(rows, ic.load_x(["0900"], "dev", str(out)))


def test_first_overlay_rows_carry_x(first_rows):
    assert len(first_rows) == 16 and first_rows["X"].notna().any()
    assert (first_rows["j"] == 0).all() and first_rows["null"].eq("ncmd_icare").all()


def test_false_call_rates_structure(first_rows):
    classes = ic.design_classes(None, pipes=["cornell"])
    r = ic.false_call_rates(first_rows.assign(patient="p"), classes, B=20)
    assert set(r["per_pipeline"]) == {"cornell"} and set(r["by_design_type"]) == {"inflated", "immune", "deflated"}
    e = r["per_pipeline"]["cornell"]
    assert e["default"]["variant"] == "theoretical_bh_keep40_r4" and e["default"]["n"] >= 1
    assert r["rows_without_class"] == 0







def test_123_spindles_is_importable_and_its_features_are_what_the_arm_stores():
    """Fails if 123_spindles is missing or loses spindle_features / recording_features: the arm deletes the window
    after processing, so the B5 features must be computed from it at build time."""
    sp = importlib.import_module("123_spindles")
    assert callable(sp.spindle_features) and callable(sp.recording_features)
    src = open(ic.__file__).read()
    assert "123_spindles" in src and "recording_features" in src and "def spindle_features" not in src
    x = np.random.default_rng(0).normal(0, 20, (2, 128 * 60))
    f = sp.spindle_features(x[0], x[1], 128.0)
    assert {"a7_density", "sigma_slow_rel", "reason"} <= set(f)


def test_analyse_has_no_spindle_analysis():
    src = open(ic.__file__).read()
    a = src[src.index("def analyse("):]
    assert "spindle" not in a.lower() and "a7_density" not in src[src.index("def analyse("):]



def true_signals(n_s=20, fs=128, seed=0):
    return np.random.default_rng(seed).normal(0, 20, (19, n_s * fs))


def test_car_of_a_c3_referenced_recording_equals_car_of_the_same_data_referenced_elsewhere():
    s = true_signals()
    names = list(en.NAMES19)
    for ref in ("C3", "Fz", "T4"):
        x = s - s[names.index(ref)]
        np.testing.assert_allclose(ic.median_reference(x), ic.median_reference(s), atol=1e-4)
    assert np.abs(np.median(ic.median_reference(s), axis=0)).max() < 1e-3


def test_car_without_the_fix_would_differ(  ):
    """Broken version: no re-reference. The C3-referenced data differ from the Fz-referenced data."""
    s = true_signals()
    names = list(en.NAMES19)
    a, b = s - s[names.index("C3")], s - s[names.index("Fz")]
    assert not np.allclose(a, b, atol=1e-3)


def test_c3_reference_channel_reads_flat_before_and_is_not_flat_after():
    s = true_signals()
    names = list(en.NAMES19)
    x = s - s[names.index("C3")]
    assert x[names.index("C3")].std() == 0.0
    assert ic.median_reference(x)[names.index("C3")].std() > 5


@pytest.mark.parametrize("pipe", ["cornell", "edlow17", "cruse11", "cruse12p", "columbia", "chennu13", "sokoliuk21"])
def test_the_spatial_transforms_of_the_reference_free_modules_are_unchanged_by_the_re_reference(pipe):
    """Laplacian (Hjorth), CSD, bipolar and average-reference outputs are invariant to a common offset channel:
    the recording reference and the common average give the same module input."""
    s = true_signals(seed=3)
    names = list(en.NAMES19)
    ref = s - s[names.index("C3")]
    rec = en.Recording(source="bdsp", uid="t", patient="t", x=ref.astype(np.float32), fs=128.0, names=names)
    mod = rn.load_pipe(pipe)
    a, na = mx.spatial(mod, rec, ref)
    b, nb = mx.spatial(mod, rec, ic.median_reference(ref))
    assert na == nb
    np.testing.assert_allclose(a, b, atol=5e-3 * np.abs(a).max() + 1e-4)


def test_a_referential_module_is_changed_by_the_re_reference():
    """Control: the guard above can fail. A module reading raw channels (no spatial transform) sees different data."""
    s = true_signals(seed=3)
    names = list(en.NAMES19)
    ref = s - s[names.index("C3")]
    rec = en.Recording(source="bdsp", uid="t", patient="t", x=ref.astype(np.float32), fs=128.0, names=names)
    mod = rn.load_pipe("mbvt")
    a, _ = mx.spatial(mod, rec, ref)
    b, _ = mx.spatial(mod, rec, ic.median_reference(ref))
    assert not np.allclose(a, b, atol=1e-3)


class RefSource(RawSource):
    """A RawSource whose data are the same brain recorded against `ref` (a channel name), with a flat-zero reference."""

    def __init__(self, ref, n_s=1900, seed=41):
        super().__init__(n_s)
        rng = np.random.default_rng(seed)
        true = rng.normal(0, 30, (19, n_s * FS)).astype(np.float32)
        ix = {c: i for i, c in enumerate(en.NAMES19)}
        eng = true - true[ix[ref]]
        self.x = np.stack([eng[ix[c]] for c in self.names]).astype(np.float32)


def test_staged_window_is_the_same_whatever_channel_the_recording_was_referenced_to(tmp_path):
    segs = [{"seg": 4, "hour": 24, "bytes": 1}]
    outs = {}
    for ref in ("C3", "Fz"):
        res = ic.fetch_window("0910", segs, RefSource(ref), "dev", out=str(tmp_path / ref))
        assert res["status"] == "ok"
        outs[ref] = ic.load_stage(res["stage"])[0]
    np.testing.assert_allclose(outs["C3"], outs["Fz"], atol=2e-3)


def test_a_flat_zero_reference_channel_is_not_padding_after_the_re_reference_and_is_flagged(tmp_path):
    segs = [{"seg": 4, "hour": 24, "bytes": 1}]
    res = ic.fetch_window("0911", segs, RefSource("C3"), "dev", out=str(tmp_path))
    assert res["status"] == "ok" and res["window"]["start_s"] == 0
    w = res["window"]
    assert w["n_flat"] == 1 and w["flat_channels"] == ["C3"] and w["rereferenced"] == "median_19"
    assert w["channel_sd_before_car_uv"]["C3"] < 1e-3 and w["channel_sd_before_car_uv"]["Fz"] > 10


def test_no_flat_channel_when_the_reference_is_not_among_the_sites(tmp_path):
    res = ic.fetch_window("0912", [{"seg": 4, "hour": 24, "bytes": 1}], RawSource(), "dev", out=str(tmp_path))
    assert res["window"]["n_flat"] == 0 and res["window"]["flat_channels"] == []


def test_the_flat_reference_channel_is_not_padding_in_the_selection_scan():
    """The channel flat over the whole segment is the recording reference. Broken version: 108's rule applied naively to the
    recording as read finds the zero C3 channel padding in every second."""
    src = RefSource("C3")
    rd = ArrayReader(src.x)
    assert bn.eligible_starts(src.x, FS, run_s=ic.WINDOW_S).size == 0
    assert ic.find_window(rd.read, rd.n_s) == 0


def test_process_patient_rereferences_idempotently(window_x, tmp_path):
    a = ic.median_reference(window_x)
    np.testing.assert_allclose(ic.median_reference(a), a, atol=1e-4)



FSA = 128


def test_acns_grade_boundaries():
    g = ic.acns_grade
    assert [g(v) for v in (0.0, 0.0999, 0.10, 0.4999, 0.50, 0.9899, 0.99, 0.9901, 1.0)] == \
        ["continuous", "continuous", "discontinuous", "discontinuous", "burst_suppression", "burst_suppression",
         "burst_suppression", "suppressed", "suppressed"]


def test_double_banana_is_18_derivations_of_the_19_sites():
    assert len(ic.DOUBLE_BANANA) == 18
    sites = {c for pair in ic.DOUBLE_BANANA for c in pair}
    assert sites <= set(en.NAMES19) and "Fpz" not in sites
    assert all(a != b for a, b in ic.DOUBLE_BANANA) and len(set(ic.DOUBLE_BANANA)) == 18
    x = np.arange(19.0)[:, None] * np.ones((19, 4))
    b = ic.double_banana(x, list(en.NAMES19))
    i = list(en.NAMES19).index
    assert b.shape == (18, 4) and b[0, 0] == i("Fp1") - i("F7")


def test_epoch_amplitude_is_half_the_range_and_the_median_over_derivations():
    b = np.zeros((18, 2 * FSA))
    for k in range(18):
        b[k, :FSA] = np.where(np.arange(FSA) % 2 == 0, 1.0, -1.0) * (k + 1)
        b[k, FSA:] = np.where(np.arange(FSA) % 2 == 0, 1.0, -1.0) * 4
    amp = ic.epoch_amplitude(b, FSA)
    assert amp.tolist() == [9.5, 4.0]


def test_suppression_threshold_is_strictly_under_10_uv():
    assert ic.suppressed([9.99, 10.0, 10.01, 0.0]).tolist() == [True, False, False, True]
    amp = ic.epoch_amplitude(ic.acns_bandpass(ic.double_banana(loud_quiet(0.3), list(en.NAMES19)), FSA), FSA)
    assert ic.acns_continuity(loud_quiet(0.3), list(en.NAMES19), FSA)["S"] == pytest.approx(ic.suppressed(amp).mean())


def loud_quiet(q, n_s=600, cycle=100, seed=0, loud=30.0, quiet=1.0):
    """19 independent white-noise channels; in each cycle the first q share of the seconds is quiet."""
    rng = np.random.default_rng(seed)
    sec = np.arange(n_s) % cycle
    g = np.where(sec < round(q * cycle), quiet, loud)
    return rng.normal(0, 1, (19, n_s * FSA)) * np.repeat(g, FSA)[None]


@pytest.mark.parametrize("q,grade", [(0.0, "continuous"), (0.3, "discontinuous"), (0.7, "burst_suppression"),
                                     (1.0, "suppressed")])
def test_acns_grade_on_synthetic_signals(q, grade):
    r = ic.acns_continuity(loud_quiet(q), list(en.NAMES19), FSA)
    assert r["grade"] == grade and r["n_epochs"] == 600
    assert abs(r["S"] - q) < 0.08


def test_acns_voltage_is_log_median_of_the_non_suppressed_epochs_and_na_when_none():
    loud = ic.acns_continuity(loud_quiet(0.0, loud=30.0), list(en.NAMES19), FSA)
    louder = ic.acns_continuity(loud_quiet(0.0, loud=60.0), list(en.NAMES19), FSA)
    assert louder["voltage_log_uv"] - loud["voltage_log_uv"] == pytest.approx(np.log(2.0), abs=0.1)
    assert loud["voltage_na"] is False
    none = ic.acns_continuity(loud_quiet(1.0), list(en.NAMES19), FSA)
    assert none["voltage_log_uv"] is None and none["voltage_na"] is True and none["grade"] == "suppressed"


def test_acns_is_unchanged_by_the_recording_reference():
    s = loud_quiet(0.3, seed=2)
    names = list(en.NAMES19)
    a = ic.acns_continuity(s - s[names.index("C3")], names, FSA)
    b = ic.acns_continuity(s - s[names.index("Fz")], names, FSA)
    assert a["S"] == b["S"] and a["voltage_log_uv"] == pytest.approx(b["voltage_log_uv"], abs=1e-6)


def test_acns_ignores_drift_and_mains_outside_the_band():
    """Channel-specific 0.1 Hz drift (500 uV) and 50 Hz (200 uV) on a quiet background do not make the epochs
    non-suppressed. Broken version: without the band-pass the same signal is far above the threshold."""
    n = 600
    t = np.arange(n * FSA) / FSA
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 1, (19, n * FSA))
    x = quiet + rng.uniform(100, 500, (19, 1)) * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6, (19, 1))) \
        + rng.uniform(50, 200, (19, 1)) * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 6, (19, 1)))
    assert ic.acns_continuity(x, list(en.NAMES19), FSA)["S"] > 0.97
    unfiltered = ic.epoch_amplitude(ic.double_banana(x, list(en.NAMES19)), FSA)
    assert (unfiltered >= ic.SUPPRESSED_UV).mean() > 0.9


def test_periodic_is_the_morgoth_iiic_share_at_10_percent():
    s = loud_quiet(0.0)
    names = list(en.NAMES19)
    assert ic.acns_continuity(s, names, FSA, 0.0999)["periodic"] is False
    assert ic.acns_continuity(s, names, FSA, 0.10)["periodic"] is True


def test_acns_and_flat_channels_are_stored_by_process_patient(processed):
    out, _ = processed
    x = xjson(out, "0900")
    assert x["acns"]["grade"] in ("continuous", "discontinuous", "burst_suppression", "suppressed")
    assert 0.0 <= x["acns"]["S"] <= 1.0 and x["acns"]["n_epochs"] == 1800 and "periodic" in x["acns"]







def test_shards_are_disjoint_slices_of_the_seeded_order_and_cover_it():
    order = ic.processing_order(IDS)
    parts = [ic.take_shard(order, i, 3) for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(order) and len(set(sum(parts, []))) == len(order)
    assert parts[1] == order[1::3]
    assert ic.take_shard(order, 0, 1, limit=4) == order[:4]


def test_median_reference_leaks_a_large_artefact_channel_far_less_than_the_mean():
    """One 250-uV frontal channel (eye / EMG) among 19 channels of 20 uV: its leak into C3 under the mean reference
    (250 / 19 uV) against the median reference. Leak = SD of (C3 with the artefact - C3 without it)."""
    names = list(en.NAMES19)
    rng = np.random.default_rng(5)
    clean = rng.normal(0, 20, (19, 128 * 600))
    dirty = clean.copy()
    dirty[names.index("Fp1")] = rng.normal(0, 250, 128 * 600)
    c3 = names.index("C3")
    mean_ref = lambda x: x - x.mean(0, keepdims=True)
    leak_mean = (mean_ref(dirty)[c3] - mean_ref(clean)[c3]).std()
    leak_med = (ic.median_reference(dirty)[c3] - ic.median_reference(clean)[c3]).std()
    assert leak_mean == pytest.approx(250 / 19, rel=0.1)
    assert leak_med < 0.4 * leak_mean
    assert leak_med < 8.0
    dirty2 = clean.copy()
    dirty2[names.index("Fp1")] = rng.normal(0, 1000, 128 * 600)
    leak_mean2 = (mean_ref(dirty2)[c3] - mean_ref(clean)[c3]).std()
    leak_med2 = (ic.median_reference(dirty2)[c3] - ic.median_reference(clean)[c3]).std()
    assert leak_mean2 > 3 * leak_mean and leak_med2 < 1.5 * leak_med


def test_median_reference_is_idempotent_and_handles_a_flat_channel():
    s = true_signals(seed=8)
    s[4] = 0.0
    a = ic.median_reference(s)
    np.testing.assert_allclose(ic.median_reference(a), a, atol=1e-5)
    assert a.dtype == np.float32 and a.shape == s.shape


def test_x_ref_reports_the_unusable_windows_and_why(processed):
    out, _ = processed
    x = xjson(out, "0900")
    r = x["xref"]
    assert r["n_fit_degenerate"] == r["reasons_not_used"].get("fit_degenerate", 0)
    assert r["n_windows"] - r["n_windows_used"] == sum(r["reasons_not_used"].values())



def test_a_partly_done_patient_is_redone_from_scratch_without_duplicate_rows(window_x, tmp_path, monkeypatch):
    out = str(tmp_path)
    p = ic.paths("analysis", out)
    sp = mx.Spectra(None)
    ic.process_patient("0920", window_x, WIN_META, "analysis", morgoth_stub, sp, 607, out, ("cornell",))
    rows1 = ic.read_lines(p["rows"] + "/0920.jsonl")
    os.remove(ic.existing(p["x"] + "/0920.json"))
    assert not ic.is_finished("0920", "analysis", out)
    ic.process_patient("0920", window_x, WIN_META, "analysis", morgoth_stub, sp, 607, out, ("cornell",))
    rows2 = ic.read_lines(p["rows"] + "/0920.jsonl")
    assert len(rows2) == len(rows1) and len({json.loads(r)["key"] for r in rows2}) == len(rows2)
    assert ic.is_finished("0920", "analysis", out)
    assert not [f for f in os.listdir(p["rows"]) if ".tmp" in f]



F128 = 128


def rec_like(n_s, rng, sd=20.0, n_ch=19):
    return rng.normal(0, sd, (n_ch, n_s * F128)).astype(np.float32)


def old_rule(pre):
    """The first version: 108's rule on the median-referenced signal."""
    return bn.padding_seconds(ic.median_reference(pre), F128)


def mutant_zero_after_reference(x, fs):
    """Broken version 1: the exact-zero test back after the reference (the noise-floor part stays on the recording)."""
    post = ic.median_reference(x)
    zero = np.asarray(post[:, :(x.shape[1] // fs) * fs], float).reshape(x.shape[0], -1, fs).std(2).min(0) < bn.ZERO_UV
    return zero | ic.padding_parts(x, fs)["floor"]


def mutant_floor_after_reference(x, fs):
    """Broken version 2: the noise-floor test back after the reference (the exact-zero part stays on the recording)."""
    return ic.padding_parts(x, fs)["zero"] | ic.padding_parts(ic.median_reference(x), fs)["floor"]


def body_median_channel_is_not_padding(mask_fn):
    """19 channels (odd), a large common component, fixed small offsets: one channel is the median in every sample."""
    rng = np.random.default_rng(1)
    common = rng.normal(0, 100.0, (1, 120 * F128))
    pre = (common + np.linspace(-45, 45, 19)[:, None]).astype(np.float32)
    post = ic.median_reference(pre)
    assert (post == 0).all(axis=1).sum() == 1
    assert not mask_fn(pre, F128).any()


def body_all_zero_for_20_s_is_padding(mask_fn):
    pre = rec_like(200, np.random.default_rng(2))
    pre[:, 50 * F128:70 * F128] = 0.0
    m = mask_fn(pre, F128)
    assert m[50:70].all() and m.sum() == 20


def body_one_channel_at_zero_is_padding(mask_fn):
    pre = rec_like(200, np.random.default_rng(3))
    pre[5, 50 * F128:70 * F128] = 0.0
    m = mask_fn(pre, F128)
    assert m[50:70].all() and m.sum() == 20


def body_flat_reference_is_not_padding(mask_fn):
    pre = rec_like(200, np.random.default_rng(4))
    pre[2, :] = 0.0
    assert not mask_fn(pre, F128).any()


def body_suppressed_background_is_not_padding(mask_fn):
    """Channel differences of 1.5 uV over a 100 uV common component: every channel is far above 2 uV as recorded, but the
    median-referenced channels are all under 2 uV (a suppressed post-arrest background)."""
    rng = np.random.default_rng(5)
    common = rng.normal(0, 100.0, (1, 200 * F128))
    pre = (common + rng.normal(0, 1.5, (19, 200 * F128))).astype(np.float32)
    assert ic.median_reference(pre).std(1).max() < bn.FLOOR_UV
    assert not mask_fn(pre, F128).any()


def body_real_flat_line_is_padding(mask_fn):
    """All channels under 2 uV as recorded for 60 s: padding (and 59 s is not)."""
    rng = np.random.default_rng(6)
    for dur, expect in ((60, True), (59, False)):
        pre = rec_like(200, rng)
        pre[:, 80 * F128:(80 + dur) * F128] = rng.normal(0, 0.5, (19, dur * F128))
        m = mask_fn(pre, F128)
        assert (m[80:80 + dur].all() and m.sum() == dur) if expect else not m.any()


def test_1_a_median_channel_that_reads_exactly_zero_is_not_padding_and_the_old_rule_flagged_it():
    body_median_channel_is_not_padding(ic.padding_mask)
    pre = (np.random.default_rng(1).normal(0, 100.0, (1, 120 * F128)) + np.linspace(-45, 45, 19)[:, None]).astype(np.float32)
    assert old_rule(pre).all()


def test_2_all_channels_zero_for_20_s_is_padding():
    body_all_zero_for_20_s_is_padding(ic.padding_mask)


def test_3_one_non_reference_channel_at_exact_zero_is_padding_and_the_old_rule_missed_it():
    body_one_channel_at_zero_is_padding(ic.padding_mask)
    pre = rec_like(200, np.random.default_rng(3))
    pre[5, 50 * F128:70 * F128] = 0.0
    assert not old_rule(pre).any()


def test_4_a_channel_flat_over_the_whole_window_is_the_reference_not_padding():
    body_flat_reference_is_not_padding(ic.padding_mask)
    pre = rec_like(200, np.random.default_rng(4))
    pre[2, :] = 0.0
    assert bn.padding_seconds(pre, F128).all()


def test_5_the_zero_rule_moved_back_after_the_reference_fails_tests_1_and_3():
    with pytest.raises(AssertionError):
        body_median_channel_is_not_padding(mutant_zero_after_reference)
    with pytest.raises(AssertionError):
        body_one_channel_at_zero_is_padding(mutant_zero_after_reference)
    body_all_zero_for_20_s_is_padding(mutant_zero_after_reference)


def test_6_a_suppressed_background_is_not_padding_and_the_old_rule_threw_it_out():
    body_suppressed_background_is_not_padding(ic.padding_mask)
    rng = np.random.default_rng(5)
    pre = (rng.normal(0, 100.0, (1, 200 * F128)) + rng.normal(0, 1.5, (19, 200 * F128))).astype(np.float32)
    assert old_rule(pre).all()
    pre3 = (np.random.default_rng(7).normal(0, 100.0, (1, 200 * F128)) + np.random.default_rng(8).normal(0, 3.0, (19, 200 * F128))).astype(np.float32)
    assert not old_rule(pre3).any() and not ic.padding_mask(pre3, F128).any()


def test_7_a_real_flat_line_as_recorded_is_padding_for_60_s():
    body_real_flat_line_is_padding(ic.padding_mask)


def test_7b_the_noise_floor_moved_back_after_the_reference_fails_test_6():
    with pytest.raises(AssertionError):
        body_suppressed_background_is_not_padding(mutant_floor_after_reference)
    body_real_flat_line_is_padding(mutant_floor_after_reference)


def test_the_selection_scan_skips_a_single_channel_dropout_the_old_rule_missed():
    rng = np.random.default_rng(21)
    x = noise(3000, rng)
    x[1, 200 * FS:220 * FS] = 0.0
    rd = ArrayReader(x)
    assert ic.find_window(rd.read, rd.n_s) == 250
    assert reference_start(x) == 250
    rd2 = ArrayReader(x)
    assert bn.eligible_starts(ic.median_reference(x), FS, run_s=ic.WINDOW_S)[0] == 0


def test_the_selection_scan_accepts_a_segment_whose_median_channel_is_exactly_zero():
    rng = np.random.default_rng(22)
    common = rng.normal(0, 100.0, (1, 2400 * FS))
    x = (common + np.linspace(-45, 45, 19)[:, None]).astype(np.float32)
    rd = ArrayReader(x)
    assert ic.find_window(rd.read, rd.n_s) == 0
    assert bn.eligible_starts(ic.median_reference(x), FS, run_s=ic.WINDOW_S).size == 0


def test_the_staged_window_stores_the_pre_reference_mask_and_the_old_rules_counts(tmp_path):
    class Src(RawSource):
        def __init__(self):
            super().__init__(n_s=1900)
            common = np.random.default_rng(23).normal(0, 100.0, (1, self.n_s * FS))
            self.x = (common + np.linspace(-45, 45, 19)[:, None]).astype(np.float32)
    res = ic.fetch_window("0930", [{"seg": 4, "hour": 24, "bytes": 1}], Src(), "dev", out=str(tmp_path))
    w = res["window"]
    assert res["status"] == "ok" and w["padding_rule"] == ic.PADDING_RULE and w["pad_s"] == []
    assert w["padding_counts"]["old_zero_post_reference"] == 1800 and w["padding_counts"]["new_pad"] == 0
    assert ic.load_stage(res["stage"])[1]["padding_rule"] == ic.PADDING_RULE


def _recording_for_engine(pre):
    post = ic.median_reference(pre)
    return en.Recording(source="bdsp", uid="0940", patient="0940", x=post, fs=128.0, names=list(en.NAMES19), meta={}), post


def test_the_engines_padding_calls_get_the_pre_reference_mask_for_the_pinned_array_only():
    rng = np.random.default_rng(24)
    pre = rec_like(300, rng)
    pre[4, 100 * F128:105 * F128] = 0.0
    rec, post = _recording_for_engine(pre)
    mask = ic.padding_mask(pre, F128)
    assert mask.sum() == 5 and not bn.padding_seconds(post, F128).any()
    ic.set_padding_context(rec.x, mask)
    try:
        assert (bn.padding_seconds(rec.x, F128) == mask).all()
        starts = np.asarray(en.clean_spans(rec, 60))
        assert not any(70 <= s <= 105 for s in starts) and 0 in starts
        assert not bn.padding_seconds(rec.x.copy(), F128).any()
        assert ic._PAD_CTX["miss_full"] == 1 and ic._PAD_CTX["hit"] >= 2
    finally:
        ic.clear_padding_context()
    assert not bn.padding_seconds(rec.x, F128).any()


def test_artifact_seconds_equal_the_engines_when_nothing_is_exactly_zero_and_differ_when_the_median_channel_is():
    rng = np.random.default_rng(25)
    pre = rec_like(300, rng)
    rec, post = _recording_for_engine(pre)
    parts = ic.window_padding(pre, post, list(en.NAMES19))
    flat = np.zeros(300, bool); flat[parts["flat_s"]] = True
    pad = np.zeros(300, bool); pad[parts["pad_s"]] = True
    mine = ic.artifact_seconds_icare(rec, flat, pad)
    assert (mine == en._artifact_seconds(en.Recording(source="bdsp", uid="0941", patient="0941", x=post, fs=128.0,
                                                      names=list(en.NAMES19), meta={}))).all()
    common = np.random.default_rng(26).normal(0, 100.0, (1, 300 * F128))
    pre2 = (common + np.linspace(-45, 45, 19)[:, None] + np.random.default_rng(27).normal(0, 0.5, (19, 300 * F128))).astype(np.float32)
    rec2, post2 = _recording_for_engine(pre2)
    parts2 = ic.window_padding(pre2, post2, list(en.NAMES19))
    assert parts2["pad_s"] == [] and parts2["flat_s"] == []
    flat2, pad2 = np.zeros(300, bool), np.zeros(300, bool)
    assert not ic.artifact_seconds_icare(rec2, flat2, pad2).any()
    assert en._artifact_seconds(en.Recording(source="bdsp", uid="0942", patient="0942", x=post2, fs=128.0,
                                             names=list(en.NAMES19), meta={})).all()




def test_files_of_the_prior_version_count_as_finished_but_block_the_analysis_until_rechecked(tmp_path):
    out = str(tmp_path)
    p = ic.paths("analysis", out)
    for d in (p["x"], p["window"], p["index"]):
        os.makedirs(d, exist_ok=True)
    prior = ic.PRIOR_SHA256[0]
    ic._atomic_json(os.path.join(p["x"], "0950.json"), {"patient": "0950", "sha256_122": prior})
    ic._atomic_json(os.path.join(p["window"], "0951.json"), {"patient": "0951", "status": "excluded", "reason": "no_clean_30min_window"})
    ic._atomic_json(os.path.join(p["window"], "0952.json"), {"patient": "0952", "status": "excluded", "reason": "no_clean_30min_window",
                                                              "padding_rule": ic.PADDING_RULE})
    ic._atomic_json(os.path.join(p["x"], "0953.json"), {"patient": "0953", "sha256_122": "0" * 64})
    assert ic.is_finished("0950", "analysis", out) and ic.is_finished("0951", "analysis", out)
    with pytest.raises(RuntimeError, match="another version"):
        ic.is_finished("0953", "analysis", out)
    assert ic.prior_version_pending(["0950", "0951", "0952"], "analysis", out) == ["0950", "0951"]
    ic._atomic_json(os.path.join(p["index"], "rechecked_unchanged.json"), ["0950", "0951"])
    assert ic.prior_version_pending(["0950", "0951", "0952"], "analysis", out) == []



class DCSource(RawSource):
    """A RawSource whose channel `ch` is constant at `dc` uV over the whole segment as recorded (a dead electrode held at
    an offset), as in I-CARE 0548 (P4), 0555 (F7) and 0824 (F7)."""

    def __init__(self, ch="P4", dc=37.0, n_s=1900, head_s=0):
        super().__init__(n_s + head_s)
        self.x[self.names.index(ch)] = dc
        if head_s:
            self.x[self.names.index("Fz"), :head_s * FS] = 0.0


def test_a_channel_flat_at_an_offset_as_recorded_is_not_padding_when_the_window_starts_the_segment(tmp_path):
    """Broken version (b44a6da1): the window at second 0 is resampled with zero padding on its left, the offset channel
    rings in second 0, fails the flat-over-the-whole test at 128 Hz, and its constant seconds are all exact-zero padding."""
    res = ic.fetch_window("0940", [{"seg": 4, "hour": 24, "bytes": 1}], DCSource(), "dev", out=str(tmp_path))
    assert res["status"] == "ok" and res["window"]["start_s"] == 0
    w = res["window"]
    assert w["padding_counts"]["new_zero"] == 0 and w["pad_s"] == [] and w["flat_s"] == []
    assert "P4" in w["ref_flat_channels"] and "P4" in w["flat_channels"]


def test_a_channel_flat_at_an_offset_inside_the_segment_was_and_is_not_padding(tmp_path):
    """Control: with real context on both sides the resampler leaves the channel constant; both versions agree."""
    res = ic.fetch_window("0941", [{"seg": 4, "hour": 24, "bytes": 1}], DCSource(head_s=100), "dev", out=str(tmp_path))
    assert res["status"] == "ok" and res["window"]["start_s"] == 130
    w = res["window"]
    assert w["padding_counts"]["new_zero"] == 0 and w["pad_s"] == [] and "P4" in w["ref_flat_channels"]


def test_v1_outputs_stand_only_without_an_exact_zero_second_and_v1_exclusions_stand(tmp_path):
    """A v1 x file blocks the analysis when a live channel had an exact-zero second (new_zero > 0: the v1 flat test may
    have mislabelled the reference) or carries no padding record; with new_zero 0 it is the v2 output. A v1 exclusion
    stands (the scan is unchanged). Broken version: treating every v1 file as current lets 0548-like windows through."""
    out = str(tmp_path)
    p = ic.paths("analysis", out)
    for d in (p["x"], p["window"], p["index"]):
        os.makedirs(d, exist_ok=True)
    v1 = ic.V1_SHA256
    ic._atomic_json(os.path.join(p["x"], "0960.json"), {"patient": "0960", "sha256_122": v1,
                                                         "window": {"padding_counts": {"new_zero": 0}}})
    ic._atomic_json(os.path.join(p["x"], "0961.json"), {"patient": "0961", "sha256_122": v1,
                                                         "window": {"padding_counts": {"new_zero": 1799}}})
    ic._atomic_json(os.path.join(p["x"], "0962.json"), {"patient": "0962", "sha256_122": v1, "window": {}})
    ic._atomic_json(os.path.join(p["window"], "0963.json"), {"patient": "0963", "status": "excluded",
                                                              "reason": "no_clean_30min_window", "padding_rule": ic.V1_RULE})
    ic._atomic_json(os.path.join(p["x"], "0964.json"), {"patient": "0964", "sha256_122": ic.own_sha256()})
    ids = ["0960", "0961", "0962", "0963", "0964"]
    assert all(ic.is_finished(i, "analysis", out) for i in ids)
    assert ic.prior_version_pending(ids, "analysis", out) == ["0961", "0962"]
    assert ic.PADDING_RULE != ic.V1_RULE
