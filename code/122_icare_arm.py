"""Post-arrest coma arm of the null benchmark (I-CARE), outside the frozen-code hash.
Built test-first on synthetic fixtures, a local stub server and the three dev patients (code/test_icare_arm.py).

Data: I-CARE v2.1, the public 607 patients (PhysioNet, open, no login), read by HTTP Range. WFDB format 16 with a
24-byte offset, sample-interleaved (a time span is one contiguous byte range); gain and baseline come from the .hea
((digital - baseline) / gain = uV; the WFDB invalid sentinel -32768 is read as exact zero, i.e. padding by 108's
rule). The files differ by hospital: 200, 250, 256, 500, 512, 1,024 or 2,048 Hz and 19-21 channels (extra Fpz / F9),
the 10-20 sites in different orders. The 19 sites of the frozen engine are picked by name, in the engine's order, and
every analysis is read at the file's own rate until the final resampling to 128 Hz.

B1 window (label-free):
- Per patient, the EEG segments whose start hour (the third number of the file name) lies in 12-36 h, nearest 24 h
  first (candidates: ties go to the earlier hour, then the segment index). A segment shorter than 1,800 s cannot hold
  the window and is skipped unread.
- Inside a segment, the window is the first 1,800-s stretch that passes 108's padding / noise-floor rule exactly as the
  frozen engine applies it (108.eligible_starts on all 19 channels in uV: no exact-zero second, no >= 60-s stretch with
  every channel under 2 uV, and 30 s clear of any such stretch). The segment is read forward in 300-s blocks; a start
  is accepted once the bytes through start + 1,800 + 30 + 60 s are in hand (the 60 s are the noise-floor stretch that
  could begin just inside the margin), or the segment has ended. So the decision equals eligible_starts on the whole
  segment.
- No window in the segment: the next-nearest segment; none left: the patient is excluded and the reason is counted
  (no_eeg_segment_12_36h, no_segment_long_enough, no_readable_segment, no_clean_30min_window). A download that keeps failing is not an
  exclusion: it is retried (backoff, resumed ranges) and reported as unfinished.
- Hours since ROSC of the window = the segment hour + (window start in the segment + 900 s) / 3600 (the hour index is
  an integer bin: +/- 1 h).
- The 1,800-s window (10 s of context each side for the resampler) is downsampled 500 -> 128 Hz (resample_poly, the
  BDSP x19 rate), channels in the engine's order (bench.engine.NAMES19), and staged as float32; the raw bytes are never
  written to disk. The staged window is deleted once the rows and features are written.

Padding rule (deviation of 2026-10-19, made before any I-CARE outcome was computed): 108's two-part rule (an exact-zero
second on any channel; a stretch of >= 60 s with every channel under 2 uV; 30 s clear of either) is evaluated on the
recording AS RECORDED, before any re-reference, as it was for BDSP; a channel flat over the whole segment (the recording
reference) is not tested for exact zeros. The first version evaluated it after the median reference, which flagged the
median channel as padding whenever one channel stayed the median for a whole second (a large common component) and missed
a real single-channel dropout; its noise-floor part would also have discarded a suppressed background. The median reference
itself is unchanged for everything downstream. The mask is stored with the window (pad_s, flat_s, pad_pair_s,
padding_counts with the old rule's counts for the report) and handed to the frozen engine for the one array being
processed (_install_padding_patch); the engine's artifact seconds and 123's pad_frac use it too.

Re-reference (label-free): I-CARE is recorded against C3 (0737 and 0868 read C3 at 0.3-0.4 uV, the amplitudes rise
with distance from it). Every window is re-referenced to the per-sample MEDIAN of all 19 channels (a flat reference
channel included; the mean would carry one large frontal EMG / eye channel into every other channel) BEFORE Morgoth, the
pipelines, X, X_ref and the spindle features (the padding rule above is evaluated before it). The per-channel
SD as recorded (before) is stored with the window, with n_flat and the names of the channels under 1 uV.

B2 pipelines and features, per patient (process_patient):
- Morgoth background labels (the five local heads, bdsp_morgoth, one persistent worker process in the Morgoth venv, MPS)
  on the 128-Hz re-referenced window, as on BDSP: stratum and transitioning of the whole window by 108.run_stratum.
- ACNS continuity (Westhall 2016), quantitative, on the reference-free longitudinal bipolar double banana (18
  derivations, 0.5-30 Hz zero-phase Butterworth): per 1-s epoch the amplitude (max - min) / 2 per derivation, the epoch
  is suppressed if the median over derivations is under 10 uV; S = share of suppressed epochs; grade: continuous
  S < 0.10, discontinuous 0.10-0.49, burst-suppression 0.50-0.99, suppressed S > 0.99; periodic = Morgoth iiic_periodic
  share >= 0.10; voltage = log median epoch amplitude over the non-suppressed epochs (none: NA).
- A frozen-engine Recording of source "bdsp" (the engine's BDSP semantics: BDSP variants, shortened twins, Morgoth
  strata from the labels above; the rows say source "icare"). All 11 frozen pipelines, native montage, no responders,
  through 114.run_recording (not edited): null sets ncmd_icare (command pipelines) and nstim0_icare (ERP pipelines),
  the same overlay and call rules as the BDSP null. One row set per patient: _icare_rows/<id>.jsonl.gz (COMPRESS; plain .jsonl when False).
- X per first-overlay row by 116's own functions (recording_results: overlay rebuilt with the runner's seeds, design
  spectra, X and the sensitivity X); X_ref = 116.s4_recording (the S4 b12 schedule, log 8-30 Hz at C3 / C4, consecutive
  non-overlapping 480-s windows, padding windows skipped) and the patient's median over its windows.
- The spindle features of part B5 (123_spindles.recording_features on the first-overlay windows of the rows) are
  computed here too, because the window is deleted afterwards.
- Covariates (age, OHCA, shockable rhythm, TTM, hospital, ROSC hours of the window) are stored apart from the outcome
  and CPC (_icare_outcome/outcomes.csv, read by --analyse only). Nothing the build prints contains an outcome value or
  count (say() refuses it).

B3 --analyse (once, at completion, while any patient of the seeded order is
neither finished nor excluded, and when the summary exists) -> output/tables/icare_arm.json. The dev patients (drawn
once, seed 20261014, from the indexed patients not among the D3-N2 forty; stored in _icare_index/dev_patients.json)
never enter it. Patients of the model: processed, with X_ref, hospital, age, OHCA, shockable rhythm, hours since ROSC
and the outcome (complete case; every other exclusion is counted with its reason and compared with the included on
outcome).
- (1) false-call rates: per pipeline, the validity-primary and the a-priori default variant (116.variant_choices), and by
  116's three-way design type (inflated / immune / deflated, pooled with 116's row weights and per pipeline), patient-
  cluster bootstrap CIs.
- (2) transport: 116.transport with the BDSP analysis M1 fit predicting the I-CARE first-overlay rows (calibration
  intercept and slope, as A2).
- (3) the ONE primary test: good outcome ~ z(X_ref) + ACNS continuity grade (dummies against continuous) + periodic +
  voltage (NA: the minimum of the others and an indicator) + hours since ROSC + age + OHCA + shockable + TTM (33, 36;
  unknown = both 0) + hospital fixed effects; weak L2 (116: C = 100 on standardised covariates; hospital effects about
  their mean); OR per SD of X_ref, two-sided alpha 0.05; patient-cluster bootstrap of 2,000 relabelled draws
  (116.fe_model, seed [20261014, crc32("icare_primary")]). X_ref is the patient's median over the 480-s windows of the
  window (116.s4_recording). The Morgoth-stratum dummies (organised = normal awake or sleep, the reference; diffuse
  slowing, IIIC / periodic, burst suppression, other) are a fixed sensitivity set.
- Secondaries (estimates, never ranked): the change in AUROC of background + z(X_ref) over the background-only model
  (leave-one-hospital-out, pooled; bootstrap on the fixed predictions); per pipeline, the a-priori default variant's
  false call against good outcome with the primary model's adjustment but no X_ref (OR and CI); the primary model with
  each hospital left out; false-call rates per hospital.
- Sensitivities: the primary model with the Morgoth background set, without patients with any flat channel (SD under
  1 uV as recorded), and without the 40 patients of the D3 N2 null.

python 122_icare_arm.py --index
python 122_icare_arm.py --run --split dev|analysis [--limit N] [--workers W] [--shard i --nshards n]
python 122_icare_arm.py --analyse [--fallback-n 300]
python 122_icare_arm.py --probe-range
(.venv_morgoth/bin/python 122_icare_arm.py --morgoth-worker  is started by --run itself.)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import concurrent.futures as cf
import contextlib
import gzip
import hashlib
import http.client
import importlib
import io
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

def _bn():
    """108 (imports doc_null and pyriemann, absent from the Morgoth venv): loaded on first use."""
    return importlib.import_module("108_bdsp_null")


def __getattr__(name):
    if name in ("MARGIN_S", "FLOOR_S", "bn"):
        return _bn() if name == "bn" else getattr(_bn(), name)
    raise AttributeError(name)


BASE_URL = "https://physionet.org/files/i-care/2.1/"
UA = {"User-Agent": "icare-research/1.0"}
OUT = os.path.join(ROOT, "output", "intermediate")
SPEC_PATH = None
MORGOTH_PY = os.path.join(ROOT, ".venv_morgoth", "bin", "python")
SEED = 20261014
HEADER_BYTES, BYTES_PER_SAMPLE = 24, 2
FS_IN, FS_OUT = 500, 128
INVALID_DIGITAL = -32768
WINDOW_S = 1800
HOUR_TARGET, HOUR_LO, HOUR_HI = 24, 12, 36
BLOCK_S = 300
RESAMPLE_PAD_S = 10
N_CONN = 6
PREFETCH = 8
MIN_FREE_MEM_PCT, MIN_FREE_DISK_GB = 20, 20
FLAT_SD_UV = 1.0
PADDING_RULE = "pre_reference_v2"
PRIOR_SHA256 = ("c624aadfa815192f395b1f6fdc610e2a7b5e7c6965dafd3b25d9baaeca263a1e",
                "b44a6da191b742ca862178c1342bc6649b3b755a1bdcdf11c51fa4b67a235af5")
V1_SHA256 = PRIOR_SHA256[1]
V1_RULE = "pre_reference_v1"
N_DEV = 3
ICARE_NULLS = ("ncmd_icare", "nstim0_icare")
COMPRESS = True
PRIMARY_B = 2000
EXCLUSION_REASONS = ("no_eeg_segment_12_36h", "no_segment_long_enough", "no_readable_segment", "no_clean_30min_window")
OUTCOME_WORDS = re.compile(r"\b(good|poor|cpc|outcome)\b", re.I)



def assert_no_outcome_text(s):
    assert not OUTCOME_WORDS.search(str(s)), "refused to print text carrying an outcome word"


def say(*parts):
    s = " ".join(str(p) for p in parts)
    assert_no_outcome_text(s)
    print(s, flush=True)



def paths(split="analysis", out=None):
    tables = os.path.join(ROOT, "output", "tables") if out is None else os.path.join(out, "tables")
    out = out or OUT
    sfx = "_dev" if split == "dev" else ""
    j = lambda *p: os.path.join(out, *p)
    return {"index": j("_icare_index"), "outcome": j("_icare_outcome"), "stage": j("_icare_stage"),
            "rows": j("_icare_rows" + sfx), "x": j("_icare_x" + sfx), "morgoth": j("_icare_morgoth" + sfx),
            "window": j("_icare_window" + sfx), "spectra": j("_icare_spectra", "design_spectra.pkl"),
            "failures": j("_icare_index", f"failures{sfx}.jsonl"),
            "summary": os.path.join(tables, "icare_arm" + sfx + ".json")}



def _file_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


_OWN_SHA256 = _file_sha256()


def own_sha256():
    return _OWN_SHA256


def guard_pinned(spec_path=None):
    """Not applied in this copy: the check reads files of the private working copy."""
    return None


def parse_free_pct(text):
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", text)
    if not m:
        raise ValueError("no free-memory percentage in the memory_pressure output")
    return int(m.group(1))


def wait_for_resources(out=None, sleep=time.sleep):
    """Pause while free memory is under MIN_FREE_MEM_PCT or free disk under MIN_FREE_DISK_GB (other jobs run)."""
    while True:
        try:
            mem = parse_free_pct(subprocess.run(["memory_pressure"], capture_output=True, text=True,
                                                timeout=30).stdout)
        except Exception:
            mem = 100
        disk = shutil.disk_usage(out or OUT).free / 2**30
        if mem >= MIN_FREE_MEM_PCT and disk >= MIN_FREE_DISK_GB:
            return
        say(f"paused: free memory {mem}%, free disk {disk:.1f} GiB")
        sleep(60)



class DownloadError(RuntimeError):
    pass


TRANSIENT = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException, OSError)


def fetch_range(url, start, end, tries=12, timeout=60, sleep=time.sleep, backoff=2.0, max_backoff=120.0):
    """Bytes [start, end] (inclusive) of url by HTTP Range. Transient failures (5xx, 429, resets, timeouts, short
    bodies) are retried with exponential backoff and the transfer resumes from the bytes already in hand; a run of
    `tries` failures without progress raises DownloadError (the caller reports it; nothing is skipped silently). Other
    4xx and a server that ignores Range raise at once."""
    need = end - start + 1
    if need <= 0:
        raise ValueError("empty range")
    buf, failures = bytearray(), 0
    while len(buf) < need:
        a, before = start + len(buf), len(buf)
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status = getattr(r, "status", None) or r.getcode()
                if status != 206:
                    raise DownloadError(f"range not honored (HTTP {status}) for {url}")
                m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", r.headers.get("Content-Range", ""))
                if not m or int(m.group(1)) != a:
                    raise DownloadError(f"bad Content-Range {r.headers.get('Content-Range')!r} (asked from {a})")
                while len(buf) < need:
                    chunk = r.read(min(1 << 16, need - len(buf)))
                    if not chunk:
                        break
                    buf += chunk
        except DownloadError:
            raise
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise DownloadError(f"HTTP {e.code} for {url}") from e
        except TRANSIENT:
            pass
        if len(buf) >= need:
            break
        failures = 0 if len(buf) > before else failures + 1
        if failures >= tries:
            raise DownloadError(f"{url}: {failures} consecutive failures at byte {start + len(buf)} of {end}")
        sleep(min(backoff * 2 ** max(failures - 1, 0), max_backoff))
    return bytes(buf)


def fetch_text(url, tries=8, timeout=60, sleep=time.sleep):
    last = None
    for k in range(tries):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise DownloadError(f"HTTP {e.code} for {url}") from e
            last = e
        except TRANSIENT as e:
            last = e
        sleep(min(2.0 * 2 ** k, 120.0))
    raise DownloadError(f"{url}: {tries} failures ({type(last).__name__})")



def parse_listing(html):
    """[(name, bytes)] of an Apache-style directory listing."""
    return [(m.group(1), int(m.group(2))) for m in
            re.finditer(r'<a href="([^"?/][^"]*)">[^<]*</a>\s+\S+\s+\S+\s+(\d+)\s*$', html, re.M)]


def eeg_segments(entries):
    """[{seg, hour, bytes}] of the EEG .mat files (not ECG / OTHER / .hea), by segment index."""
    out = []
    for name, size in entries:
        m = re.fullmatch(r"\d+_(\d+)_(\d+)_EEG\.mat", name)
        if m:
            out.append({"seg": int(m.group(1)), "hour": int(m.group(2)), "bytes": int(size)})
    return sorted(out, key=lambda s: s["seg"])


def candidates(segments, target=HOUR_TARGET, lo=HOUR_LO, hi=HOUR_HI):
    """B1: segments starting in lo-hi h, nearest `target` first (ties: the earlier hour, then the segment index)."""
    c = [s for s in segments if lo <= s["hour"] <= hi]
    return sorted(c, key=lambda s: (abs(s["hour"] - target), s["hour"], s["seg"]))


def parse_header(text):
    """WFDB .hea -> n_sig, fs, n_samp, offset (bytes before the samples: the 24 of '16+24'), names, gains, baselines."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    f0 = lines[0].split()
    n_sig, fs, n_samp = int(f0[1]), float(f0[2]), int(f0[3])
    gains, baselines, names, offset = [], [], [], 0
    for ln in lines[1:1 + n_sig]:
        p = ln.split()
        m = re.match(r"([0-9.eE+-]+)(?:\(([0-9.eE+-]+)\))?", p[2])
        gain = float(m.group(1)) if m else 1.0
        adc_zero = float(p[4]) if len(p) > 4 else 0.0
        gains.append(gain if gain != 0 else 1.0)
        baselines.append(float(m.group(2)) if m and m.group(2) is not None else adc_zero)
        names.append(p[-1])
        fmt = re.fullmatch(r"(\d+)(?:\+(\d+))?", p[1])
        offset = int(fmt.group(2) or 0) if fmt else 0
    return {"n_sig": n_sig, "fs": fs, "n_samp": n_samp, "offset": offset, "names": names, "gains": gains,
            "baselines": baselines}


def _num(v, cast=float):
    return None if v is None or str(v).strip().lower() in ("nan", "", "none") else cast(v)


def _bool(v):
    s = str(v).strip().lower()
    return None if s in ("nan", "", "none") else s == "true"


def parse_metadata(text):
    """<id>.txt -> (covariates, outcome). The outcome and CPC go to their own record and file."""
    kv = {}
    for ln in text.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            kv[k.strip()] = v.strip()
    cov = {"patient": kv["Patient"], "hospital": _num(kv.get("Hospital"), str), "age": _num(kv.get("Age")),
           "sex": _num(kv.get("Sex"), str), "rosc_min": _num(kv.get("ROSC")), "ohca": _bool(kv.get("OHCA")),
           "shockable": _bool(kv.get("Shockable Rhythm")), "ttm": _num(kv.get("TTM"))}
    out = {"patient": kv["Patient"], "outcome": _num(kv.get("Outcome"), str),
           "cpc": _num(kv.get("CPC"), lambda v: int(float(v)))}
    return cov, out



def byte_range(n_ch, i0, i1, offset=HEADER_BYTES):
    """Inclusive (first, last) byte of samples [i0, i1) of a sample-interleaved format-16 file."""
    if i0 < 0 or i1 <= i0:
        raise ValueError(f"bad sample span [{i0}, {i1})")
    return offset + i0 * n_ch * BYTES_PER_SAMPLE, offset + i1 * n_ch * BYTES_PER_SAMPLE - 1


def decode_digital(raw, n_ch):
    """(n_ch, n) int16 from a byte range of whole frames."""
    if len(raw) % (n_ch * BYTES_PER_SAMPLE):
        raise ValueError("byte range is not a whole number of frames")
    return np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch).T


def to_microvolts(dig, hdr):
    """(digital - baseline) / gain in uV, float32; the invalid sentinel is exact zero (padding)."""
    g = np.asarray(hdr["gains"], float)[:, None]
    b = np.asarray(hdr["baselines"], float)[:, None]
    uv = (dig.astype(np.float64) - b) / g
    uv[dig == INVALID_DIGITAL] = 0.0
    return uv.astype(np.float32)



def median_reference(x):
    """Re-reference to the per-sample MEDIAN across all the channels (a flat or reference channel is one valid value).
    Invariant to the recording reference (median(V - r) = median(V) - r), idempotent, and, unlike the mean, not pulled
    by one large artefact channel (frontal EMG or eye movement) into every other channel. float32, same shape."""
    x = np.asarray(x, np.float32)
    return x - np.median(x, axis=0, keepdims=True)



def _floor_stretches(low, floor_s):
    """Seconds inside runs of >= floor_s consecutive True seconds."""
    out = np.zeros(len(low), bool)
    d = np.diff(np.r_[0, np.asarray(low, int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= floor_s:
            out[a:b] = True
    return out


def padding_parts(x, fs, flat_as_read=None):
    """x: channels x samples as recorded. -> {sd per channel and second, flat_ref per channel (SD < 108's ZERO_UV over the
    whole of x, or flat over the same span at the file's own rate: flat_as_read), zero (an exact-zero second on a channel
    that is not flat_ref), floor (a >= FLOOR_S stretch with every channel under FLOOR_UV), pad = zero | floor} over whole
    seconds."""
    bn = _bn()
    fs = int(fs)
    nc, n = x.shape[0], x.shape[1] // fs
    if n == 0:
        return {"sd": np.zeros((nc, 0)), "flat_ref": np.zeros(nc, bool), "zero": np.zeros(0, bool),
                "floor": np.zeros(0, bool), "pad": np.zeros(0, bool)}
    a = np.asarray(x[:, :n * fs], np.float64)
    sd = a.reshape(nc, n, fs).std(2)
    flat_ref = a.std(1) < bn.ZERO_UV
    if flat_as_read is not None:
        flat_ref |= np.asarray(flat_as_read, bool)
    live = ~flat_ref
    zero = (sd[live] < bn.ZERO_UV).any(0) if live.any() else np.zeros(n, bool)
    floor = _floor_stretches(sd.max(0) < bn.FLOOR_UV, bn.FLOOR_S)
    return {"sd": sd, "flat_ref": flat_ref, "zero": zero, "floor": floor, "pad": zero | floor}


def padding_mask(x, fs):
    """Per whole second: padding by 108's rule on x as recorded (see above)."""
    return padding_parts(x, fs)["pad"]


def eligible_starts_pre(x, fs, run_s=WINDOW_S):
    """108.eligible_starts with the padding mask above: run starts with no padding within the run or MARGIN_S around it."""
    bn = _bn()
    pad = padding_mask(x, fs)
    n = len(pad)
    bad = pad.copy()
    for k in range(1, bn.MARGIN_S + 1):
        bad[k:] |= pad[:-k]
        bad[:-k] |= pad[k:]
    c = np.r_[0, np.cumsum(bad.astype(int))]
    s = np.arange(0, n - run_s + 1)
    return s[(c[s + run_s] - c[s]) == 0]


def _pair_pad(parts, idx):
    """The same rule restricted to the channels idx (123's pad_frac reads C3 and C4)."""
    bn = _bn()
    sd, fr = parts["sd"][idx], parts["flat_ref"][idx]
    zero = (sd[~fr] < bn.ZERO_UV).any(0) if (~fr).any() else np.zeros(sd.shape[1], bool)
    return zero | _floor_stretches(sd.max(0) < bn.FLOOR_UV, bn.FLOOR_S)


def window_padding(pre, post, names, fs=FS_OUT, flat_as_read=None):
    """The padding record of a staged window: the new mask from the pre-reference data (pad_s, flat_s for the engine's
    artifact seconds, pad_pair_s for 123's C3 / C4 pad_frac), and the old rule's counts on the re-referenced data
    (report only). flat_as_read: the channels flat over the window at the file's own rate (the reference, as in the
    selection scan)."""
    bn, en = _bn(), _rn().en
    parts = padding_parts(pre, fs, flat_as_read)
    live = ~parts["flat_ref"]
    flat = (parts["sd"][live] < en.FLAT_UV).any(0) if live.any() else np.zeros(len(parts["pad"]), bool)
    pair = [names.index("C3"), names.index("C4")]
    old = padding_parts(post, fs)
    old_zero = old["sd"].min(0) < bn.ZERO_UV if old["sd"].shape[1] else np.zeros(0, bool)
    return {"padding_rule": PADDING_RULE,
            "pad_s": np.flatnonzero(parts["pad"]).tolist(), "flat_s": np.flatnonzero(flat).tolist(),
            "pad_pair_s": np.flatnonzero(_pair_pad(parts, pair)).tolist(),
            "ref_flat_channels": [c for c, f in zip(names, parts["flat_ref"]) if f],
            "padding_counts": {"new_zero": int(parts["zero"].sum()), "new_floor": int(parts["floor"].sum()),
                               "new_pad": int(parts["pad"].sum()),
                               "old_zero_post_reference": int(old_zero.sum()),
                               "old_floor_post_reference": int(old["floor"].sum()),
                               "old_pad_post_reference": int((old_zero | old["floor"]).sum())}}


_PAD_CTX = {"x": None, "mask": None, "hit": 0, "miss_full": 0, "miss_other": 0}
_ORIG_PADDING = []


def _install_padding_patch():
    """The frozen engine and 116 call 108.padding_seconds(rec.x, fs) on the post-reference array. For the one array the
    arm registers (the recording being processed) the call returns the pre-reference mask; any other array goes to 108
    unchanged. Nothing in 108, 114, 116 or the engine is edited."""
    bn = _bn()
    if getattr(bn.padding_seconds, "_icare_patch", False):
        return
    orig = bn.padding_seconds
    _ORIG_PADDING.append(orig)

    def padding_seconds(x, fs):
        c = _PAD_CTX
        if c["x"] is not None and x is c["x"]:
            c["hit"] += 1
            return c["mask"][:x.shape[1] // int(fs)].copy()
        if c["x"] is not None and getattr(x, "shape", None) == c["x"].shape:
            c["miss_full"] += 1
        elif c["x"] is not None:
            c["miss_other"] += 1
        return orig(x, fs)
    padding_seconds._icare_patch = True
    bn.padding_seconds = padding_seconds


def set_padding_context(x, mask):
    _install_padding_patch()
    _PAD_CTX.update(x=x, mask=np.asarray(mask, bool), hit=0, miss_full=0, miss_other=0)


def clear_padding_context():
    audit = {k: _PAD_CTX[k] for k in ("hit", "miss_full", "miss_other")}
    _PAD_CTX.update(x=None, mask=None)
    return audit


def _mask_of(win, key, n):
    if win.get("padding_rule") != PADDING_RULE or key not in win:
        raise RuntimeError("the staged window carries no pre-reference padding record: move it away to stage it again")
    m = np.zeros(n, bool)
    m[np.asarray(win[key], int)] = True
    return m


def artifact_seconds_icare(rec, flat_s, pad):
    """The engine's _artifact_seconds with the exact-zero 'flat' term taken from the pre-reference data (win flat_s) and
    its padding mask from the pre-reference rule; everything else (saturation, 30-45 Hz power against the median over the
    kept seconds) as the engine has it, on the re-referenced signal."""
    from scipy.signal import periodogram
    en = _rn().en
    fs = int(round(rec.fs))
    nc, n = rec.x.shape[0], rec.x.shape[1] // fs
    hf = np.empty((nc, n))
    bad = np.zeros(n, bool)
    for a in range(0, n, en.ART_CHUNK_S):
        b = min(n, a + en.ART_CHUNK_S)
        seg = np.asarray(rec.x[:, a * fs: b * fs], float).reshape(nc, b - a, fs)
        bad[a:b] = (np.abs(seg) > en.SAT_UV).any(2).any(0)
        f, p = periodogram(seg, fs=fs, window="hann", axis=-1)
        hf[:, a:b] = p[..., (f >= en.HF_BAND[0]) & (f < en.HF_BAND[1])].sum(-1) * (f[1] - f[0])
    bad |= np.asarray(flat_s, bool)[:n]
    keep = ~np.asarray(pad, bool)[:n]
    if keep.any():
        med = np.median(hf[:, keep], axis=1)
        bad |= (hf > en.HF_RATIO * med[:, None]).any(0)
    return bad


def _scan(read, total_s, fs=FS_IN, window_s=WINDOW_S, block_s=BLOCK_S):
    """(first eligible start, the prefix read) of a segment of total_s whole seconds; read(t0, t1) -> uV array. The
    108 rule is applied to the prefix as recorded (eligible_starts_pre: the channels flat over the whole prefix are the
    recording reference); the prefix returned is as read."""
    bn = _bn()
    look = window_s + bn.MARGIN_S + bn.FLOOR_S
    chunks, t = [], 0
    while t < total_s:
        t1 = min(total_s, t + block_s)
        chunks.append(read(t, t1))
        t = t1
        if t < min(look, total_s):
            continue
        x = np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
        chunks = [x]
        st = eligible_starts_pre(x, fs, run_s=window_s)
        if len(st) and (st[0] + look <= t or t >= total_s):
            return int(st[0]), x
    x = np.concatenate(chunks, axis=1) if len(chunks) > 1 else (chunks[0] if chunks else np.zeros((0, 0)))
    return None, x


def find_window(read, total_s, **kw):
    return _scan(read, total_s, **kw)[0]


def rosc_hours(hour, start_s):
    """Hours since ROSC at the window midpoint (the hour index is an integer bin)."""
    return hour + (start_s + WINDOW_S / 2) / 3600.0


class SegmentError(RuntimeError):
    """A segment that cannot be used as is (unexpected header, a required site missing, size mismatch): not a download
    failure, so it is never retried; the next-nearest segment is tried and the reason recorded."""


def select_window(pid, segments, source):
    """B1: the window of one patient. source.open(pid, segment) -> (reader with read(t0, t1) and fs, whole seconds)."""
    cands = candidates(segments)
    if not cands:
        return {"status": "excluded", "reason": "no_eeg_segment_12_36h", "tried": []}
    tried = []
    for s in cands:
        try:
            reader, total_s = source.open(pid, s)
        except SegmentError as e:
            tried.append({"seg": s["seg"], "hour": s["hour"], "result": "unreadable: " + str(e)[:120]})
            continue
        if total_s < WINDOW_S:
            tried.append({"seg": s["seg"], "hour": s["hour"], "result": "too_short"})
            continue
        fs = int(getattr(reader, "fs", FS_IN))
        start, prefix = _scan(reader.read, total_s, fs=fs)
        if start is None:
            tried.append({"seg": s["seg"], "hour": s["hour"], "result": "no_clean_window"})
            continue
        return {"status": "ok", "segment": s, "start_s": start, "tried": tried, "prefix": prefix, "fs": fs,
                "rosc_hours": rosc_hours(s["hour"], start)}
    if all(t["result"] == "too_short" for t in tried):
        reason = "no_segment_long_enough"
    elif all(t["result"].startswith("unreadable") or t["result"] == "too_short" for t in tried):
        reason = "no_readable_segment"
    else:
        reason = "no_clean_30min_window"
    return {"status": "excluded", "reason": reason, "tried": tried}


def resample_window(x, fs_in, start_s, window_s, fs_out, pad_s=RESAMPLE_PAD_S):
    """The window [start_s, start_s + window_s) of x (channels, samples at fs_in) at fs_out, float32: resampled with
    pad_s of real context each side where the segment has it, then cropped to the window."""
    n_s = x.shape[1] // fs_in
    a, b = max(0, start_s - pad_s), min(n_s, start_s + window_s + pad_s)
    g = math.gcd(int(fs_out), int(fs_in))
    y = resample_poly(np.asarray(x[:, a * fs_in:b * fs_in], np.float64), int(fs_out) // g, int(fs_in) // g, axis=1)
    lead = (start_s - a) * fs_out
    return np.ascontiguousarray(y[:, lead:lead + window_s * fs_out]).astype(np.float32)


class HttpSource:
    """Selection source over PhysioNet: opens a segment's header and reads its samples by byte range. The files differ
    by hospital (200-2048 Hz, 19-21 channels, 10-20 sites in different orders); the reader returns the 19 sites of
    the engine, in the engine's order, in uV, at the file's own rate."""

    def __init__(self, base=None, sleep=time.sleep):
        self.base, self.sleep, self.hdr = base or BASE_URL, sleep, {}

    def url(self, pid, segment, ext):
        return f"{self.base}training/{pid}/{pid}_{segment['seg']:03d}_{segment['hour']:03d}_EEG.{ext}"

    def open(self, pid, segment):
        hdr = parse_header(fetch_text(self.url(pid, segment, "hea"), sleep=self.sleep))
        eng = list(importlib.import_module("bench.engine").NAMES19)
        if hdr["offset"] != HEADER_BYTES or hdr["fs"] != int(hdr["fs"]):
            raise SegmentError(f"unexpected header (fs {hdr['fs']}, offset {hdr['offset']})")
        if any(c not in hdr["names"] for c in eng):
            raise SegmentError("a 10-20 site is missing: " + ",".join(c for c in eng if c not in hdr["names"]))
        if HEADER_BYTES + hdr["n_samp"] * hdr["n_sig"] * BYTES_PER_SAMPLE != segment["bytes"]:
            raise SegmentError(f"header says {hdr['n_samp']} samples, the file has {segment['bytes']} bytes")
        pick = [hdr["names"].index(c) for c in eng]
        file_names, n_sig = hdr["names"], hdr["n_sig"]
        hdr = {**hdr, "file_names": file_names, "names": eng, "fs": int(hdr["fs"]),
               "gains": [hdr["gains"][i] for i in pick], "baselines": [hdr["baselines"][i] for i in pick]}
        self.hdr = hdr
        return _Reader(self.url(pid, segment, "mat"), n_sig, pick, hdr, self.sleep), hdr["n_samp"] // hdr["fs"]


class _Reader:
    """read(t0, t1): seconds [t0, t1) of the 19 engine sites in uV, from one byte range of the file."""

    def __init__(self, url, n_sig, pick, hdr, sleep):
        self.url, self.n_sig, self.pick, self.hdr, self.sleep, self.fs = url, n_sig, pick, hdr, sleep, hdr["fs"]

    def read(self, t0, t1):
        a, b = byte_range(self.n_sig, t0 * self.fs, t1 * self.fs)
        raw = fetch_range(self.url, a, b, sleep=self.sleep)
        return to_microvolts(decode_digital(raw, self.n_sig)[self.pick], self.hdr)



def processing_order(ids, seed=SEED):
    """The seeded processing order: a permutation of the sorted ids (a partial run is a random sample)."""
    return [str(v) for v in np.random.default_rng(seed).permutation(sorted(ids))]


def pick_dev(ids, d3_n2, seed=SEED, n=N_DEV):
    """The dev patients: n seeded draws from the patients NOT among the D3-N2 forty. Excluded from the analysis."""
    pool = sorted(set(ids) - set(d3_n2))
    return [str(v) for v in np.random.default_rng(seed).choice(pool, n, replace=False)]


def d3_n2_patients(root=None):
    """The 40 patients of the D3 N2 null (already seen): the basenames of the existing icare_null cache."""
    if root is None:
        root = importlib.import_module("bench.engine").ICARE
    return sorted({f.split("_")[0] for f in os.listdir(root) if f.endswith(".npz") and not f.startswith("._")})



def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh)
    os.replace(path + ".tmp", path)


def _data_path(base):
    return base + ".gz" if COMPRESS else base


def _open(path, mode="r"):
    return gzip.open(path, mode + "t") if path.endswith(".gz") else open(path, mode)


def existing(base):
    """The file of a per-patient record, compressed or not (None if absent)."""
    for c in (base + ".gz", base):
        if os.path.exists(c):
            return c
    return None


def _tmp(path):
    return path[:-3] + ".tmp.gz" if path.endswith(".gz") else path + ".tmp"


def write_data_json(base, obj):
    path = _data_path(base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _open(_tmp(path), "w") as fh:
        json.dump(obj, fh)
    os.replace(_tmp(path), path)


def read_data_json(base):
    with _open(existing(base)) as fh:
        return json.load(fh)


def read_lines(base):
    with _open(existing(base)) as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def build_index(sleep=time.sleep, workers=N_CONN):
    """RECORDS, then per patient the directory listing (segments, sizes) and <id>.txt. Covariates, the outcome file and
    the segment index are written apart; counts only are printed."""
    p = paths()
    ids = sorted(ln.strip().split("/")[1] for ln in fetch_text(BASE_URL + "RECORDS", sleep=sleep).splitlines()
                 if ln.strip().startswith("training/"))

    def one(pid):
        ent = parse_listing(fetch_text(f"{BASE_URL}training/{pid}/", sleep=sleep))
        cov, out = parse_metadata(fetch_text(f"{BASE_URL}training/{pid}/{pid}.txt", sleep=sleep))
        return pid, eeg_segments(ent), cov, out
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, ids))
    os.makedirs(p["index"], exist_ok=True)
    os.makedirs(p["outcome"], exist_ok=True)
    _atomic_json(os.path.join(p["index"], "index.json"), {pid: segs for pid, segs, _, _ in res})
    cov = pd.DataFrame([c for _, _, c, _ in res])
    cov.to_csv(os.path.join(p["index"], "covariates.csv"), index=False)
    pd.DataFrame([o for _, _, _, o in res]).to_csv(os.path.join(p["outcome"], "outcomes.csv"), index=False)
    n_cand = sum(bool(candidates(segs)) for _, segs, _, _ in res)
    say(f"index: {len(res)} patients, {n_cand} with an EEG segment starting 12-36 h after ROSC")
    return res




def _rn():
    return importlib.import_module("114_benchmark")


def _mx():
    return importlib.import_module("116_mechanism")


def _mg():
    return importlib.import_module("115_benchmark_merge")


def _clean(o):
    """JSON-ready: numpy scalars and arrays as Python values, non-finite floats as null (missing is never zero)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def stage_path(pid, out=None):
    return os.path.join(paths("analysis", out)["stage"], f"{pid}.npz")


def fetch_window(pid, segments, source, split, out=None):
    """B1 for one patient: select the window (reading only the bytes it needs), resample, reorder the channels to the
    engine's, and stage the 128-Hz window. -> {status: ok (stage path, window) | excluded (reason, written to
    _icare_window/<id>.json) | download_failed (nothing recorded: the caller retries)}."""
    p = paths(split, out)
    try:
        sel = select_window(pid, segments, source)
    except DownloadError as e:
        return {"status": "download_failed", "error": str(e)[:300]}
    if sel["status"] != "ok":
        _atomic_json(os.path.join(p["window"], f"{pid}.json"), {"patient": pid, "status": "excluded",
                                                                  "reason": sel["reason"], "tried": sel["tried"],
                                                                  "padding_rule": PADDING_RULE})
        return sel
    names = list(source.hdr["names"])
    eng = list(_rn().en.NAMES19)
    if sorted(names) != sorted(eng):
        raise RuntimeError(f"{pid}: channels {names} are not the 19 sites of the 10-20 montage")
    raw = sel["prefix"][[names.index(c) for c in eng]]
    x = resample_window(raw, sel["fs"], sel["start_s"], WINDOW_S, FS_OUT)
    sd = np.asarray(x, np.float64).std(1)
    a0, fsi = sel["start_s"] * int(sel["fs"]), int(sel["fs"])
    flat_read = np.array([np.std(raw[i, a0:a0 + WINDOW_S * fsi], dtype=np.float64) < _bn().ZERO_UV
                          for i in range(len(eng))])
    flat = [c for c, v, f in zip(eng, sd, flat_read) if v < FLAT_SD_UV or f]
    x_pre = x
    x = median_reference(x)
    padding = window_padding(x_pre, x, eng, flat_as_read=flat_read)
    win = {"segment": sel["segment"], "start_s": sel["start_s"], "rosc_hours": sel["rosc_hours"],
           "tried": sel["tried"], "fs_in": sel["fs"], "n_channels_in_file": len(source.hdr.get("file_names", names)),
           "channel_sd_before_car_uv": dict(zip(eng, sd)), "n_flat": len(flat), "flat_channels": flat,
           "rereferenced": "median_19", **padding}
    path = stage_path(pid, out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", x=x, names=np.array(eng), window=json.dumps(_clean(win)))
    os.replace(path + ".tmp.npz", path)
    return {"status": "ok", "stage": path, "window": win}


def load_stage(path):
    with np.load(path) as z:
        win = json.loads(str(z["window"]))
        win["names"] = [str(n) for n in z["names"]]
        return np.asarray(z["x"], np.float32), win



MARK = "@@122 "


def morgoth_worker():
    """Runs under .venv_morgoth: loads the five heads once, then labels each 128-Hz window named on stdin."""
    import bdsp_morgoth as bm
    import bdsp_stream as bs
    dev = bm.device()
    models = bm.load_models(dev)
    print(MARK + "READY", flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        mat = os.path.join(req["out_dir"], req["stem"] + ".mat")
        try:
            os.makedirs(req["out_dir"], exist_ok=True)
            bs.write_mat(mat, np.load(req["npy"]), float(FS_OUT), req["names"])
            df, secs = bm.run(mat, models, dev, out_dir=req["out_dir"])
            print(MARK + "OK " + json.dumps({k: round(v, 1) for k, v in secs.items()}), flush=True)
        except Exception as e:
            print(MARK + "ERR " + repr(e)[:300], flush=True)
        finally:
            if os.path.exists(mat):
                os.remove(mat)


class MorgothClient:
    """label(x, names, pid) -> the Morgoth frame of the 128-Hz window (bdsp_morgoth.run's parquet, t_s = window
    start in seconds). The worker is started on first use and kept for the run."""

    def __init__(self, split, out=None):
        self.split, self.out, self.proc = split, out, None

    def _start(self):
        log = open(os.path.join(paths(self.split, self.out)["morgoth"] + "_worker.log"), "ab") \
            if os.path.isdir(paths(self.split, self.out)["morgoth"]) else subprocess.DEVNULL
        self.proc = subprocess.Popen([MORGOTH_PY, "-u", os.path.abspath(__file__), "--morgoth-worker"], cwd=CODE,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        self._reply("READY")

    def _reply(self, want):
        for line in self.proc.stdout:
            if line.startswith(MARK):
                msg = line[len(MARK):].strip()
                if msg.startswith("ERR"):
                    raise RuntimeError("Morgoth worker: " + msg)
                if msg.startswith(want):
                    return msg
        raise RuntimeError("Morgoth worker exited")

    def __call__(self, x, names, pid):
        d = paths(self.split, self.out)["morgoth"]
        os.makedirs(d, exist_ok=True)
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        npy = os.path.join(paths("analysis", self.out)["stage"], f"{pid}_morgoth_in.npy")
        os.makedirs(os.path.dirname(npy), exist_ok=True)
        np.save(npy, x)
        try:
            self.proc.stdin.write(json.dumps({"npy": npy, "names": list(names), "out_dir": d, "stem": pid}) + "\n")
            self.proc.stdin.flush()
            self._reply("OK")
        finally:
            if os.path.exists(npy):
                os.remove(npy)
        return pd.read_parquet(os.path.join(d, f"{pid}.parquet"))

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)



DOUBLE_BANANA = (("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
                 ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
                 ("Fz", "Cz"), ("Cz", "Pz"),
                 ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
                 ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"))
ACNS_BAND = (0.5, 30.0)
SUPPRESSED_UV = 10.0
PERIODIC_SHARE = 0.10


def double_banana(x, names):
    """(18, samples): the longitudinal bipolar derivations (DOUBLE_BANANA) of x (rows = names)."""
    i = {c: k for k, c in enumerate(names)}
    return np.stack([np.asarray(x[i[a]], np.float64) - np.asarray(x[i[b]], np.float64) for a, b in DOUBLE_BANANA])


def acns_bandpass(b, fs):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [ACNS_BAND[0] / (fs / 2), ACNS_BAND[1] / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, b, axis=1)


def epoch_amplitude(b, fs):
    """Per 1-s epoch: (max - min) / 2 of each derivation, then the median across derivations -> (n_epochs)."""
    fs = int(round(fs))
    n = b.shape[1] // fs
    e = b[:, :n * fs].reshape(b.shape[0], n, fs)
    return np.median((e.max(2) - e.min(2)) / 2.0, axis=0)


def suppressed(amp):
    """The epochs whose median amplitude over derivations is under SUPPRESSED_UV (strictly)."""
    return np.asarray(amp) < SUPPRESSED_UV


def acns_grade(S):
    """continuous S < 0.10; discontinuous 0.10 <= S < 0.50; burst-suppression 0.50 <= S <= 0.99; suppressed S > 0.99."""
    if S < 0.10:
        return "continuous"
    if S < 0.50:
        return "discontinuous"
    if S <= 0.99:
        return "burst_suppression"
    return "suppressed"


def acns_continuity(x, names, fs, periodic_share=0.0):
    """B3's background covariates of the primary model: S (share of suppressed 1-s epochs), the grade, the voltage (log
    median epoch amplitude over the non-suppressed epochs; None when there are none), and periodic (Morgoth
    iiic_periodic share >= 0.10)."""
    amp = epoch_amplitude(acns_bandpass(double_banana(x, names), fs), fs)
    supp = suppressed(amp)
    S = float(supp.mean())
    volt = float(np.log(np.median(amp[~supp]))) if (~supp).any() else None
    return {"S": S, "grade": acns_grade(S), "n_epochs": int(len(amp)), "n_suppressed": int(supp.sum()),
            "voltage_log_uv": volt, "voltage_na": volt is None, "periodic_share": float(periodic_share),
            "periodic": bool(periodic_share >= PERIODIC_SHARE)}



_E_S4 = []


def e_s4():
    if not _E_S4:
        _E_S4.append(_mx().s4_spectrum())
    return _E_S4[0]


def morgoth_summary(df):
    """Stratum and transitioning of the whole window (108.run_stratum over every Morgoth window) and the share of
    each label among the valid windows."""
    lab = _bn().row_labels(df)
    stratum, trans = _bn().run_stratum(lab)
    ok = lab != "invalid"
    frac = {k: float(np.mean(lab[ok] == k)) for k in sorted(set(lab[ok]))} if ok.any() else {}
    return {"stratum": str(stratum), "transitioning": bool(trans), "valid_frac": float(ok.mean()),
            "label_frac": frac, "n_windows": int(len(df))}


def make_recording(pid, x, df, dev, win=None):
    """The frozen engine's Recording of the window: source "bdsp" (so the engine's BDSP semantics apply: BDSP variants,
    shortened twins, Morgoth strata), the Morgoth labels in its meta."""
    en = _rn().en
    meta = {"dev": bool(dev), "morgoth_t": df["t_s"].to_numpy(), "morgoth_lab": _bn().row_labels(df),
            "icare_window": win}
    return en.Recording(source="bdsp", uid=pid, patient=pid, x=np.asarray(x, np.float32), fs=float(FS_OUT),
                        names=list(en.NAMES19), meta=meta)


def run_pipelines(rec, split, n_recs, pipes=None):
    """All frozen pipelines on the window through 114.run_recording (not edited): native montage, no responders, the
    BDSP overlay and call rules. -> the rows as JSON lines (114's own serialisation: no NaN), source "icare"."""
    rn = _rn()
    pipes = list(pipes or importlib.import_module("bench.locked_spec").PIPES)
    ctx = {"split": split, "montage": "native", "resp": "none", "max_overlays": None, "locator": None,
           "uncorrected": None, "col113": None, "spec_path": None, "n_recs": n_recs}
    lines, done = [], set()

    def write(row):
        lines.append(json.dumps({**row, "source": "icare"}, allow_nan=False))
        done.add(row["key"])
    for pipe in pipes:
        mod = rn.load_pipe(pipe)
        kind = "ncmd" if mod.FAMILY == "cmd" else "nstim0"
        spec = {"name": f"{kind}_icare", "kind": kind, "family": mod.FAMILY, "source": "bdsp"}
        with contextlib.redirect_stdout(io.StringIO()):
            rn.run_recording(rec, mod, spec, ctx, done, write)
    return lines


def first_overlay_rows(rows):
    """The first-overlay published rows of the arm (116's M1 rows: ok, no responder, native, sequence overlay)."""
    d = _mg()._null_seq(pd.DataFrame(rows))
    return d[d["null"].isin(ICARE_NULLS) & (d["j"] == 0)]


def x_ref(rec):
    """X_ref: 116.s4_recording (S4 b12 schedule, log 8-30 Hz power at C3 / C4, consecutive 480-s windows from the start,
    padding windows skipped); the patient's value is the median over the windows that gave an X."""
    mx = _mx()
    w = mx.s4_recording(rec, e_s4(), mx.MARKER_HZ)
    used = [v["X"] for v in w if v.get("skipped") is None and v.get("X") is not None and np.isfinite(v["X"])]
    why = {}
    for v in w:
        r = v.get("skipped") or (v.get("reason") if v.get("X") is None or not np.isfinite(v["X"]) else None)
        if r:
            why[str(r)] = why.get(str(r), 0) + 1
    return w, {"X_ref": float(np.median(used)) if used else None, "n_windows": len(w), "n_windows_used": len(used),
               "n_fit_degenerate": why.get("fit_degenerate", 0), "reasons_not_used": why}


def spindle_rows(rec, d, win=None):
    """part B5: 123_spindles' features on the first-overlay windows of the rows (the window is deleted afterwards).
    123 reads pad_frac from 108.padding_seconds on a two-channel (C3, C4) cut of the re-referenced signal, which the
    fixed mask cannot reach; pad_frac is therefore taken from the pre-reference C3 / C4 mask of the window wherever
    a padding second lies in the window or 123 found one."""
    sp = importlib.import_module("123_spindles")
    w = pd.DataFrame({"t0": d["extent_t0"].astype(float).map(sp._r), "t1": d["extent_t1"].astype(float).map(sp._r)})
    w = w.drop_duplicates().sort_values(["t0", "t1"], kind="stable").reset_index(drop=True)
    rows = sp.recording_features(rec, w)
    if win is not None:
        pair = _mask_of(win, "pad_pair_s", rec.x.shape[1] // FS_OUT)
        for r in rows:
            t0, t1, old = r.get("t0"), r.get("t1"), r.get("pad_frac")
            if t0 is None or t1 is None or old is None or (isinstance(old, float) and np.isnan(old)):
                continue
            k0, k1 = int(round(t0)), int(round(t1))
            if k1 > k0 and (old > 0 or pair[k0:k1].any()):
                r["pad_frac"] = float(pair[k0:k1].mean())
    return rows, sp.own_sha256()


def write_lines(base, lines):
    path = _data_path(base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _open(_tmp(path), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(_tmp(path), path)


def process_patient(pid, x, win, split, label_fn, spectra, n_recs, out=None, pipes=None):
    """Morgoth labels, the 11 pipelines, X per first-overlay row, X_ref and the spindle features of one staged window.
    Writes _icare_morgoth/<id>.parquet, _icare_rows/<id>.jsonl and, last, _icare_x/<id>.json (the patient is done when
    that file exists). The window is not touched on disk here. The pre-reference padding mask of the window is
    fixed for the frozen engine for the duration (see _install_padding_patch)."""
    try:
        return _process_patient(pid, x, win, split, label_fn, spectra, n_recs, out, pipes)
    finally:
        clear_padding_context()


def _process_patient(pid, x, win, split, label_fn, spectra, n_recs, out=None, pipes=None):
    p, mx, t = paths(split, out), _mx(), {}
    x = median_reference(x)
    tic = time.perf_counter()
    df = label_fn(x, win["names"], pid)
    os.makedirs(p["morgoth"], exist_ok=True)
    df.to_parquet(os.path.join(p["morgoth"], f"{pid}.parquet"), index=False, compression="zstd")
    t["morgoth"] = time.perf_counter() - tic
    rec = make_recording(pid, x, df, split == "dev", win)
    n_sec = rec.x.shape[1] // FS_OUT
    pad, flat_s = _mask_of(win, "pad_s", n_sec), _mask_of(win, "flat_s", n_sec)
    rec.meta["_artifact_s"] = artifact_seconds_icare(rec, flat_s, pad)
    set_padding_context(rec.x, pad)
    tic = time.perf_counter()
    lines = run_pipelines(rec, split, n_recs, pipes)
    write_lines(os.path.join(p["rows"], f"{pid}.jsonl"), lines)
    t["pipelines"] = time.perf_counter() - tic
    rows = [json.loads(ln) for ln in lines]
    tic = time.perf_counter()
    d = first_overlay_rows(rows).assign(part="m1")
    xres = mx.recording_results(rec, d, spectra, split) if len(d) else []
    xres = [{**r, "source": "icare"} for r in xres]
    t["x"] = time.perf_counter() - tic
    tic = time.perf_counter()
    xw, xr = x_ref(rec)
    t["x_ref"] = time.perf_counter() - tic
    tic = time.perf_counter()
    spind, sha123 = spindle_rows(rec, d, win) if len(d) else ([], None)
    t["spindle"] = time.perf_counter() - tic
    share, label = _rn().en.artifact_index(rec, 0.0, float(WINDOW_S))
    msum = morgoth_summary(df)
    acns = acns_continuity(rec.x, rec.names, FS_OUT, msum["label_frac"].get("iiic_periodic", 0.0))
    rec_out = {"patient": pid, "split": split, "sha256_122": own_sha256(), "sha256_116": mx.own_sha256(),
               "sha256_123": sha123, "window": win, "morgoth": msum, "acns": acns, "x_rows": xres,
               "channel_sd_uv": dict(zip(rec.names, np.asarray(rec.x, np.float64).std(1))),
               "artifact_share": share, "artifact_label": label,
               "xref_windows": xw, "xref": xr, "spindle": spind, "seconds": t,
               "n_rows": len(rows), "n_ok_rows": int(sum(r["status"] == "ok" for r in rows)),
               "padding_audit": clear_padding_context()}
    if rec_out["padding_audit"]["miss_full"]:
        raise RuntimeError("padding_seconds saw a full-size array other than the pinned recording: "
                           f"{rec_out['padding_audit']}")
    write_data_json(os.path.join(p["x"], f"{pid}.json"), _clean(rec_out))
    if getattr(spectra, "path", None):
        spectra.save()
    say(f"{pid}: {len(rows)} rows ({rec_out['n_ok_rows']} ok), {len(xres)} X rows, X_ref windows {xr['n_windows_used']}"
        f"/{xr['n_windows']}, " + ", ".join(f"{k} {v:.0f} s" for k, v in t.items()))
    return rec_out




def load_index(out=None):
    with open(os.path.join(paths("analysis", out)["index"], "index.json")) as fh:
        return json.load(fh)


def dev_patients(index, out=None, d3=None):
    """The three dev patients, drawn once (seed 20261014) from the indexed patients that are not among the D3-N2 forty;
    the draw is stored and never redrawn."""
    f = os.path.join(paths("analysis", out)["index"], "dev_patients.json")
    if os.path.exists(f):
        with open(f) as fh:
            return json.load(fh)
    d3 = d3_n2_patients() if d3 is None else d3
    dev = pick_dev(list(index), d3)
    _atomic_json(f, dev)
    return dev


def is_finished(pid, split, out=None):
    p = paths(split, out)
    fx = existing(os.path.join(p["x"], f"{pid}.json"))
    if fx:
        with _open(fx) as fh:
            h = json.load(fh).get("sha256_122")
        if h != own_sha256() and h not in PRIOR_SHA256:
            raise RuntimeError(f"{fx} comes from another version of 122_icare_arm.py: move it away before running "
                               "this version")
        return True
    return os.path.exists(os.path.join(p["window"], f"{pid}.json"))


def log_failure(split, out, pid, kind, error):
    p = paths(split, out)["failures"]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps({"patient": pid, "kind": kind, "error": str(error)[:300], "time": time.strftime("%F %T")})
                 + "\n")


def run_arm(pids, index, split, out=None, make_source=HttpSource, label_fn=None, spectra=None, workers=N_CONN,
            prefetch=PREFETCH, rounds=3, n_recs=None, pipes=None, pause=lambda: wait_for_resources(),
            sleep=time.sleep):
    """Fetch (threads) and process (this thread, one patient at a time, in the given order) every unfinished patient.
    A download that keeps failing is retried in the next round (never an exclusion); a processing error keeps the
    staged window and is logged. -> counts."""
    n_recs = n_recs or sum(bool(candidates(s)) for s in index.values())
    todo = [pid for pid in pids if not is_finished(pid, split, out)]
    say(f"{split}: {len(pids) - len(todo)} of {len(pids)} patients finished, {len(todo)} to do")
    counts = {"done": 0, "excluded": 0, "download_failed": 0, "process_error": 0}
    consecutive = 0
    for rnd in range(rounds):
        failed = []
        with cf.ThreadPoolExecutor(workers) as ex:
            futs, nxt = {}, 0

            def fetch(pid):
                pause()
                path = stage_path(pid, out)
                if os.path.exists(path):
                    try:
                        fresh = load_stage(path)[1].get("padding_rule") == PADDING_RULE
                    except Exception:
                        fresh = False
                    if fresh:
                        return {"status": "ok", "stage": path}
                    os.remove(path)
                return fetch_window(pid, index[pid], make_source(), split, out)

            for pid in todo:
                while nxt < len(todo) and len(futs) < workers + prefetch:
                    futs[todo[nxt]] = ex.submit(fetch, todo[nxt])
                    nxt += 1
                r = futs.pop(pid).result()
                if r["status"] == "download_failed":
                    log_failure(split, out, pid, "download", r["error"])
                    failed.append(pid)
                    counts["download_failed"] += 1
                elif r["status"] == "excluded":
                    counts["excluded"] += 1
                    say(f"{pid}: excluded, {r['reason']}")
                else:
                    try:
                        pause()
                        x, win = load_stage(r["stage"])
                        process_patient(pid, x, win, split, label_fn, spectra, n_recs, out, pipes)
                        os.remove(r["stage"])
                        counts["done"] += 1
                        consecutive = 0
                    except Exception as e:
                        log_failure(split, out, pid, "process", f"{type(e).__name__}: {e}")
                        counts["process_error"] += 1
                        consecutive += 1
                        say(f"{pid}: processing error {type(e).__name__}: "
                            f"{OUTCOME_WORDS.sub('[x]', str(e)[:200])}")
                        if consecutive >= 3:
                            raise RuntimeError("three processing errors in a row: stopping (see the failures log)")
        todo = failed
        if not todo:
            break
        say(f"round {rnd + 1}: {len(todo)} download(s) failed, retrying after a pause")
        sleep(60)
    counts["download_failed"] = len(todo)
    return counts


def take_shard(pids, shard=0, nshards=1, limit=None):
    """This analysis process's slice of the seeded order (every nshards-th patient from `shard`), then the first limit."""
    return list(pids)[shard::nshards][:limit]


def status(split="analysis", out=None):
    """Counts only: finished, excluded by reason, unfinished."""
    index = load_index(out)
    p = paths(split, out)
    ex, done, todo = {}, 0, 0
    for pid in index:
        fw = os.path.join(p["window"], f"{pid}.json")
        if existing(os.path.join(p["x"], f"{pid}.json")):
            done += 1
        elif os.path.exists(fw):
            r = json.load(open(fw))["reason"]
            ex[r] = ex.get(r, 0) + 1
        else:
            todo += 1
    return {"patients": len(index), "finished": done, "excluded": ex, "unfinished": todo}



BG_GRADE = {"normal_awake": "organised", "sleep": "organised", "diffuse_slowing": "diffuse_slowing",
            "iiic_periodic": "iiic_periodic", "burst_suppression": "burst_suppression"}
BG_DUMMIES = ("diffuse_slowing", "iiic_periodic", "burst_suppression", "other")
COVARIATES = ("rosc_hours", "age", "ohca", "shockable", "ttm33", "ttm36")
ALPHA = 0.05


def background_grade(stratum):
    """The background grade of the primary model, from the 108 Morgoth stratum of the whole window: organised
    (normal awake or sleep, the reference), diffuse slowing, IIIC / periodic, burst suppression; anything else (other
    awake, mixed, unlabelled) is "other"."""
    return BG_GRADE.get(str(stratum), "other")


def analysis_order(index, dev, fallback_n=None):
    """The seeded order of the analysis patients: the dev patients never enter; the fallback is the first N of it."""
    order = [p for p in processing_order(list(index)) if p not in set(dev)]
    return order[:fallback_n]


def patient_table(order, index=None, out=None, split="analysis"):
    """One row per patient of the order: status (included = processed; excluded with its reason; unfinished), the
    covariates, the window (hours since ROSC), X_ref, the background stratum, and the outcome from its own file."""
    p = paths(split, out)
    cov = pd.read_csv(os.path.join(p["index"], "covariates.csv"), dtype={"patient": str}).set_index("patient")
    outc = pd.read_csv(os.path.join(paths("analysis", out)["outcome"], "outcomes.csv"),
                       dtype={"patient": str}).set_index("patient")
    d3 = set(d3_n2_patients()) if index is None or "d3" not in index else set(index["d3"])
    rows = []
    for pid in order:
        r = {"patient": pid, "status": "unfinished", "reason": None, "in_d3n2": pid in d3}
        c = cov.loc[pid]
        r.update(hospital=c["hospital"] if pd.notna(c["hospital"]) else None,
                 age=float(c["age"]) if pd.notna(c["age"]) else None,
                 ohca=None if pd.isna(c["ohca"]) else bool(c["ohca"]),
                 shockable=None if pd.isna(c["shockable"]) else bool(c["shockable"]),
                 ttm=float(c["ttm"]) if pd.notna(c["ttm"]) else None)
        o = outc.loc[pid]
        r["outcome_good"] = None if pd.isna(o["outcome"]) else float(o["outcome"] == "Good")
        fx, fw = os.path.join(p["x"], f"{pid}.json"), os.path.join(p["window"], f"{pid}.json")
        if existing(fx):
            if not is_finished(pid, split, out):
                raise RuntimeError(f"{pid}: unfinished")
            x = read_data_json(fx)
            a = x["acns"]
            r.update(status="included", rosc_hours=x["window"]["rosc_hours"], xref=x["xref"]["X_ref"],
                     stratum=x["morgoth"]["stratum"], transitioning=x["morgoth"]["transitioning"],
                     acns_grade=a["grade"], acns_S=a["S"], acns_voltage=a["voltage_log_uv"],
                     acns_periodic=a["periodic"], n_flat=x["window"].get("n_flat"))
        elif os.path.exists(fw):
            with open(fw) as fh:
                r.update(status="excluded", reason=json.load(fh)["reason"])
        rows.append(r)
    return pd.DataFrame(rows)


def model_set(t):
    """The patients of the primary model (complete case) and the exclusions at this stage with their reasons."""
    inc = t[t["status"] == "included"].copy()
    why = pd.Series("", index=inc.index)
    for col, name in (("xref", "x_ref_na"), ("acns_grade", "acns_missing"), ("hospital", "hospital_missing"),
                      ("age", "age_missing"),
                      ("ohca", "ohca_missing"), ("shockable", "shockable_missing"),
                      ("rosc_hours", "rosc_hours_missing"), ("outcome_good", "outcome_missing")):
        miss = inc[col].isna() & (why == "")
        why[miss] = name
    keep = inc[why == ""].copy()
    keep["bg"] = keep["stratum"].map(background_grade)
    return keep, why[why != ""].value_counts().to_dict()


ACNS_DUMMIES = ("discontinuous", "burst_suppression", "suppressed")


def background_covariates(d, background="acns"):
    """The background adjustment. "acns" (primary): the ACNS continuity grade as dummies against continuous, periodic
    (Morgoth iiic_periodic share >= 0.10), voltage (log median amplitude of the non-suppressed epochs; where there is
    none: the minimum of the others and voltage_na = 1). "morgoth" (fixed sensitivity): the Morgoth stratum dummies
    against organised."""
    if background == "morgoth":
        bg = d["bg"] if "bg" in d else d["stratum"].map(background_grade)
        return {f"bg_{k}": (bg == k).to_numpy(float) for k in BG_DUMMIES}
    v = pd.to_numeric(d["acns_voltage"], errors="coerce").to_numpy(float)
    na = np.isnan(v)
    v = np.where(na, np.nanmin(v) if (~na).any() else 0.0, v)
    covs = {f"acns_{k}": (d["acns_grade"] == k).to_numpy(float) for k in ACNS_DUMMIES}
    covs.update(periodic=d["acns_periodic"].astype(float).to_numpy(), voltage=v, voltage_na=na.astype(float))
    return covs


def design_matrix(d, mean_sd=None, background="acns"):
    """Standardised-in-the-fit covariates of the primary model: z(X_ref), the background covariates, hours since ROSC,
    age, OHCA, shockable, TTM 33 / 36 (unknown: both 0). X_ref is z-scored here (mean and SD of the model's patients),
    so its odds ratio is per SD. -> (frame with call / fe / patient / uid, ordered covariate dict, (mean, sd))."""
    x = d["xref"].astype(float).to_numpy()
    m, sd = mean_sd if mean_sd else (float(x.mean()), float(x.std()))
    covs = {"z_xref": (x - m) / sd}
    covs.update(background_covariates(d, background))
    covs.update(rosc_hours=d["rosc_hours"].astype(float).to_numpy(), age=d["age"].astype(float).to_numpy(),
                ohca=d["ohca"].astype(float).to_numpy(), shockable=d["shockable"].astype(float).to_numpy(),
                ttm33=(d["ttm"] == 33).to_numpy(float), ttm36=(d["ttm"] == 36).to_numpy(float))
    frame = pd.DataFrame({"call": d["outcome_good"].astype(float).to_numpy(), "fe": d["hospital"].to_numpy(),
                          "patient": d["patient"].to_numpy(), "uid": d["patient"].to_numpy()})
    return frame, covs, (m, sd)


def primary(d, B=PRIMARY_B, name="icare_primary", background="acns"):
    """THE primary test (B3 (3)): good outcome ~ z(X_ref) + ACNS continuity grade + periodic + voltage + hours since
    ROSC + age + OHCA + shockable + TTM (33 / 36 / unknown) + hospital fixed effects; weak L2 (116: C = 100 on
    standardised covariates); OR per SD of X_ref, two-sided alpha 0.05, patient-cluster bootstrap (116.fe_model:
    relabelled draws, seed [20261014, crc32(name)]). background="morgoth": the fixed sensitivity set."""
    frame, covs, (m, sd) = design_matrix(d, background=background)
    res = _mx().fe_model(frame, covs, B, name, None, "fe")
    c = res.get("coef", {}).get("z_xref")
    out = {"n_patients": int(len(d)), "n_hospitals": int(d["hospital"].nunique()), "x_ref_mean": m, "x_ref_sd": sd,
           "background": background, "model": res}
    if c:
        out.update(or_per_sd=c["or"], or_ci95=c["or_ci95"], p_two_sided=c["p_two_sided"],
                   significant=bool(c["p_two_sided"] < ALPHA))
    else:
        out["note"] = "the full-sample fit did not converge"
    return out


def _fit_predict(Ztr, ytr, Zte):
    mx = _mx()
    sd = Ztr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    mu = Ztr.mean(0)
    f = mx.fe_logit(ytr, (Ztr - mu) / sd, np.zeros(len(ytr), int), np.ones(len(ytr)), lam=mx.LAM)
    return mx.expit(f["alpha"][0] + ((Zte - mu) / sd) @ f["beta"])


def loho_predictions(d):
    """Leave-one-hospital-out predictions of the background-only model (the ACNS covariates) and of background +
    z(X_ref): every hospital's
    patients are predicted by models fitted on the other hospitals only (weak-L2 logistic, no hospital term)."""
    y = d["outcome_good"].astype(float).to_numpy()
    bg = np.column_stack(list(background_covariates(d).values()))
    x = d["xref"].astype(float).to_numpy()[:, None]
    hosp = d["hospital"].to_numpy()
    pa, pb = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for h in sorted(set(hosp)):
        te, tr = hosp == h, hosp != h
        pa[te] = _fit_predict(bg[tr], y[tr], bg[te])
        pb[te] = _fit_predict(np.hstack([bg[tr], x[tr]]), y[tr], np.hstack([bg[te], x[te]]))
    return pa, pb


def delta_auroc(d, B=PRIMARY_B, seed="icare_delta_auroc"):
    """Secondary: AUROC of background + z(X_ref) over the background-only model, leave-one-hospital-out (loho_
    predictions pooled); the CI resamples patients with their fixed predictions."""
    mx = _mx()
    y = d["outcome_good"].astype(float).to_numpy()
    hosp = d["hospital"].to_numpy()
    pa, pb = loho_predictions(d)
    auc_a, auc_b = mx._auc(pa, y), mx._auc(pb, y)
    ci = mx.cluster_index(d["patient"].to_numpy())
    draws = _mg().boot_idx(len(ci["ids"]), B, mx._seed(seed))
    delta = np.full(B, np.nan)
    for i in range(B):
        idx, _ = mx.boot_sample(ci, draws[i])
        delta[i] = mx._auc(pb[idx], y[idx]) - mx._auc(pa[idx], y[idx])
    t = mx._ptests(auc_b - auc_a, delta)
    per_site = {str(h): {"n": int((hosp == h).sum()), "auc_background": mx._auc(pa[hosp == h], y[hosp == h]),
                         "auc_with_x_ref": mx._auc(pb[hosp == h], y[hosp == h])} for h in sorted(set(hosp))}
    return {"auc_background_only": auc_a, "auc_with_x_ref": auc_b, "delta_auc": float(auc_b - auc_a),
            "delta_auc_ci95": t["ci95"], "B": B, "per_hospital": per_site,
            "note": "leave-one-hospital-out predictions pooled; patient bootstrap on the fixed predictions"}


def leave_one_hospital_out(d, B=PRIMARY_B):
    """Secondary: the primary model re-estimated with each hospital left out (estimates only)."""
    out = {}
    for h in sorted(d["hospital"].unique()):
        e = d[d["hospital"] != h]
        r = primary(e, B, f"icare_loho_{h}")
        out[str(h)] = {k: r.get(k) for k in ("n_patients", "or_per_sd", "or_ci95", "p_two_sided")}
    return out


def without_d3_n2(d, B=PRIMARY_B):
    """Sensitivity: the primary model without the 40 patients of the D3 N2 null (already seen)."""
    e = d[~d["in_d3n2"]]
    r = primary(e, B, "icare_primary_without_d3n2")
    return {"n_removed": int(d["in_d3n2"].sum()), **{k: r.get(k) for k in ("n_patients", "or_per_sd", "or_ci95",
                                                                           "p_two_sided", "significant")}}


def with_morgoth_background(d, B=PRIMARY_B):
    """Fixed sensitivity: the primary model with the Morgoth-stratum dummies in place of the ACNS covariates."""
    r = primary(d, B, "icare_primary_morgoth", background="morgoth")
    return {k: r.get(k) for k in ("n_patients", "background", "or_per_sd", "or_ci95", "p_two_sided", "significant")}


def without_flat_channels(d, B=PRIMARY_B):
    """Sensitivity: patients with any flat channel (SD under 1 uV as recorded: the recording reference) left out."""
    e = d[d["n_flat"].fillna(0) == 0]
    r = primary(e, B, "icare_primary_without_flat")
    return {"n_removed": int(len(d) - len(e)), **{k: r.get(k) for k in ("n_patients", "or_per_sd", "or_ci95",
                                                                        "p_two_sided", "significant")}}



def load_rows(patients, split="analysis", out=None):
    """The arm's rows of the given patients (all pipelines, all overlays), as one frame."""
    p = paths(split, out)["rows"]
    rows = []
    for pid in patients:
        rows += [json.loads(ln) for ln in read_lines(os.path.join(p, f"{pid}.jsonl"))]
    return pd.DataFrame(rows)


def load_x(patients, split="analysis", out=None):
    """The X rows of the first overlays (116's recording_results) of the given patients."""
    p = paths(split, out)["x"]
    rows = []
    for pid in patients:
        rows += read_data_json(os.path.join(p, f"{pid}.json"))["x_rows"]
    return pd.DataFrame(rows)


def first_overlay_with_x(rows, xres):
    mx = _mx()
    d = first_overlay_rows(rows.to_dict("records"))
    pub = xres[xres["kind"] == "published"].drop(columns=["uid", "patient", "variant", "j", "kind", "split", "source"],
                                                 errors="ignore")
    return d.merge(pub, on=["key", "pipe"], how="left", validate="one_to_one")


def design_classes(spectra_path=None, pipes=None):
    """(pipe, variant) -> 116's three-way class from X on the reference 1/f series, from the design table."""
    mx = _mx()
    tab = mx.design_table(pipes=pipes, spectra=mx.Spectra(spectra_path))
    tp = tab[tab["kind"] == "published"].drop_duplicates(["pipe", "variant"])
    return {(a, b): mx.x_ref_class(v) for a, b, v in zip(tp["pipe"], tp["variant"], tp["x_ref_1f"])}


def _wrate_ci(d, B, seed):
    """Weighted false-call rate (row weight w) with a patient-cluster bootstrap CI."""
    mg = _mg()
    if d.empty:
        return {"n": 0, "n_patients": 0, "rate": None, "ci95": None}
    g = d.assign(wc=d["w"] * d["call"].astype(float)).groupby("patient", sort=True)
    s, n = g["wc"].sum().to_numpy(float), g["w"].sum().to_numpy(float)
    draws = mg.ratio_draws(s, n, mg.boot_idx(len(s), B, seed))
    return {"n": int(len(d)), "n_patients": int(len(s)), "rate": float(s.sum() / n.sum()), "ci95": mg._ci(draws)}


def false_call_rates(first, classes, B=PRIMARY_B):
    """Estimand (1): per pipeline, the false-call rate of the validity-primary and the a-priori default variant (115 /
    114 choices) with a patient-bootstrap CI; and by three-way design type (inflated / immune / deflated), pooled with
    116's row weights and per pipeline."""
    mx = _mx()
    rn = _rn()
    first = first.assign(source="bdsp")
    choices = mx.variant_choices(first, B)
    n_var = {p: mx.n_frozen_variants(p) for p in sorted(set(first["pipe"]))}
    out = {"per_pipeline": {}, "by_design_type": {}, "by_pipeline_and_design_type": {}}
    for (nul, pipe), ch in sorted(choices.items()):
        g = first[(first["pipe"] == pipe) & (first["null"] == nul)]
        e = {"null": nul, "default_as_coded": ch["default_as_coded"]}
        for role in ("validity_primary", "default"):
            v = ch.get(role)
            e[role] = dict(variant=v, **mx._rate_ci(g[g["variant"] == v], B, mx._seed(f"icare_rate|{pipe}|{role}")))
        out["per_pipeline"][pipe] = e
    f = mx._prep(first.assign(call=first["call"].astype(bool)), n_var)
    f["design_class"] = [classes.get((a, b)) for a, b in zip(f["pipe"], f["variant"])]
    out["rows_without_class"] = int(f["design_class"].isna().sum())
    for c in ("inflated", "immune", "deflated"):
        g = f[f["design_class"] == c]
        out["by_design_type"][c] = _wrate_ci(g, B, mx._seed(f"icare_rate|class|{c}"))
        for pipe, gp in g.groupby("pipe"):
            out["by_pipeline_and_design_type"].setdefault(pipe, {})[c] = mx._rate_ci(
                gp, B, mx._seed(f"icare_rate|{pipe}|{c}"))
    return out


def call_status_vs_outcome(first, d, B=PRIMARY_B):
    """Secondary: per pipeline, the a-priori default variant's first-overlay call (1 = false call) against good outcome
    with the primary model's adjustment (background grade, hours since ROSC, age, OHCA, shockable, TTM, hospital fixed
    effects; no X_ref): OR with a patient-bootstrap CI. Never ranked."""
    mx, rn = _mx(), _rn()
    out = {}
    base_frame, base_covs, _ = design_matrix(d)
    base_covs = {k: v for k, v in base_covs.items() if k != "z_xref"}
    for pipe in sorted(set(first["pipe"])):
        dflt = rn.primary_variant(rn.load_pipe(pipe), "bdsp")
        g = first[(first["pipe"] == pipe) & (first["variant"] == dflt)].drop_duplicates("patient")
        call = d["patient"].map(dict(zip(g["patient"], g["call"].astype(float))))
        e = d[call.notna().to_numpy()]
        if e.empty:
            out[pipe] = {"variant": dflt, "note": "no rows"}
            continue
        frame, covs, _ = design_matrix(e)
        covs = {"call_status": call[call.notna()].to_numpy(float), **{k: v for k, v in covs.items() if k != "z_xref"}}
        r = mx.fe_model(frame, covs, B, f"icare_call_{pipe}", None, "fe")
        c = r.get("coef", {}).get("call_status")
        n1 = int(covs["call_status"].sum())
        out[pipe] = {"variant": dflt, "n_patients": int(len(e)), "n_false_calls": n1,
                     "sparse": bool(n1 < 10 or len(e) - n1 < 10),
                     **({"or": c["or"], "or_ci95": c["or_ci95"]} if c else {"note": "fit did not converge"})}
    return out


def site_rates(first, d, B=PRIMARY_B):
    """Secondary: per hospital, the false-call rate of each pipeline's a-priori default variant (patient bootstrap)."""
    mx, rn = _mx(), _rn()
    site = dict(zip(d["patient"], d["hospital"]))
    f = first.assign(hospital=first["patient"].map(site))
    f = f[f["hospital"].notna()]
    out = {}
    for pipe in sorted(set(f["pipe"])):
        dflt = rn.primary_variant(rn.load_pipe(pipe), "bdsp")
        g = f[(f["pipe"] == pipe) & (f["variant"] == dflt)]
        out[pipe] = {str(h): mx._rate_ci(gh, B, mx._seed(f"icare_site|{pipe}|{h}")) for h, gh in g.groupby("hospital")}
    return out


def bdsp_m1_frame(split="analysis"):
    """The BDSP M1 rows with X (116.run_stats' frame, before the A9 flags) and the frozen-variant counts."""
    mx = _mx()
    p = mx.paths(split)
    df = mx.load_rows(split)
    res = mx.read_results(p["results"])
    if res.empty:
        raise RuntimeError(f"no BDSP X rows in {p['results']}: the BDSP mechanism stage has not run")
    nv = {pp: mx.n_frozen_variants(pp) for pp in sorted(set(mx.m1_rows(df)["pipe"]))}
    pub = res[res["kind"] == "published"].drop(columns=["uid", "patient", "variant", "j", "kind", "split", "source"],
                                               errors="ignore")
    m1 = mx._prep(mx.m1_rows(df).merge(pub, on=["key", "pipe"], how="left", validate="one_to_one"), nv)
    return m1, nv


def transport_to_icare(bdsp, nv_bdsp, first, B=PRIMARY_B):
    """Estimand (2): the BDSP-fitted M1 model (116.transport, as A2 with the I-CARE first overlays in the place of the
    pDoC rows) -> calibration intercept and slope."""
    mx = _mx()
    nv = {p: mx.n_frozen_variants(p) for p in sorted(set(first["pipe"]))}
    return mx.transport(bdsp, first, nv_bdsp, nv, B, name="icare_transport")


def excluded_vs_included(t, kept):
    """B4: every exclusion with its reason, and the excluded vs the included on outcome (counts, Fisher exact)."""
    from scipy.stats import fisher_exact
    inc = t[t["patient"].isin(kept["patient"])]
    exc = t[~t["patient"].isin(kept["patient"])]
    why = exc["reason"].fillna(exc["status"]).value_counts().to_dict()
    ok = exc[exc["outcome_good"].notna()]
    ni, ki = int(len(inc)), int(inc["outcome_good"].sum())
    ne, ke = int(len(ok)), int(ok["outcome_good"].sum())
    p = float(fisher_exact([[ki, ni - ki], [ke, ne - ke]])[1]) if ni and ne else None
    return {"excluded_by_reason": why, "n_included": ni, "n_excluded": int(len(exc)),
            "good_included": ki, "good_excluded": ke, "n_excluded_with_outcome": ne, "fisher_p": p}


def prior_version_pending(order, split="analysis", out=None):
    """Patients whose x file was written by a PRIOR_SHA256 version of this file (the post-reference padding rule) and that
    are not listed in _icare_index/rechecked_unchanged.json (patients whose window and padding mask were rechecked under the
    current rule and found unchanged), and the excluded patients whose exclusion was decided before the new rule. The
    analysis refuses to run while any exist."""
    p = paths(split, out)
    f = os.path.join(p["index"], "rechecked_unchanged.json")
    ok = set(json.load(open(f))) if os.path.exists(f) else set()
    pending = []
    for pid in order:
        if pid in ok:
            continue
        fx = existing(os.path.join(p["x"], f"{pid}.json"))
        fw = os.path.join(p["window"], f"{pid}.json")
        if fx:
            with _open(fx) as fh:
                xj = json.load(fh)
            h = xj.get("sha256_122")
            if h == V1_SHA256 and ((xj.get("window") or {}).get("padding_counts") or {}).get("new_zero", 1) == 0:
                continue
            if h in PRIOR_SHA256:
                pending.append(pid)
        elif os.path.exists(fw):
            with open(fw) as fh:
                if json.load(fh).get("padding_rule") not in (PADDING_RULE, V1_RULE):
                    pending.append(pid)
    return pending


def analyse(fallback_n=None, out=None, split="analysis", B=PRIMARY_B, bdsp_loader=bdsp_m1_frame,
            parts=("primary", "secondary", "rates", "transport"), spec_path=None, check=True):
    """Everything of B3, once. Prints counts only."""
    p = paths(split, out)
    if check:
        guard_pinned(spec_path)
    if os.path.exists(p["summary"]):
        raise RuntimeError(f"{p['summary']} exists: the analysis runs once (move it away to rerun)")
    index = load_index(out)
    dev = dev_patients(index, out)
    order = analysis_order(index, dev, fallback_n)
    t = patient_table(order, index, out, split)
    pending = prior_version_pending(order, split, out)
    if pending:
        raise RuntimeError(f"{len(pending)} patient(s) were processed under the post-reference padding rule and not rechecked "
                           f"(e.g. {pending[:5]}): rerun or recheck them first")
    unfinished = t[t["status"] == "unfinished"]
    if len(unfinished):
        raise RuntimeError(f"{len(unfinished)} patient(s) of the order are neither finished nor excluded "
                           f"(e.g. {unfinished['patient'].tolist()[:5]}): the analysis runs at completion only")
    kept, dropped = model_set(t)
    res = {"split": split, "sha256_122": own_sha256(), "B": B, "fallback_n": fallback_n,
           "dev_patients_excluded": sorted(dev), "n_order": len(order),
           "status": t["status"].value_counts().to_dict(), "n_model": int(len(kept)),
           "dropped_at_model": dropped, "exclusions": excluded_vs_included(t, kept)}
    if "primary" in parts:
        res["primary"] = primary(kept, B)
        res["sensitivity_without_d3n2"] = without_d3_n2(kept, B)
        res["sensitivity_morgoth_background"] = with_morgoth_background(kept, B)
        res["sensitivity_without_flat_channels"] = without_flat_channels(kept, B)
    if "secondary" in parts:
        res["delta_auroc"] = delta_auroc(kept, B)
        res["leave_one_hospital_out"] = leave_one_hospital_out(kept, B)
    if "rates" in parts or "transport" in parts or "secondary" in parts:
        rows = load_rows(kept["patient"], split, out)
        first = first_overlay_with_x(rows, load_x(kept["patient"], split, out))
        if "rates" in parts:
            res["false_call_rates"] = false_call_rates(first, design_classes(paths(split, out)["spectra"]), B)
        if "secondary" in parts:
            res["call_status_vs_outcome"] = call_status_vs_outcome(first, kept, B)
            res["site_rates"] = site_rates(first, kept, B)
        if "transport" in parts:
            bdsp, nv = bdsp_loader()
            res["transport"] = transport_to_icare(bdsp, nv, first, B)
    os.makedirs(os.path.dirname(p["summary"]), exist_ok=True)
    with open(p["summary"], "w") as fh:
        json.dump(_mg()._jsonable(res), fh, indent=1, default=float)
    say(f"analysis {split}: {len(order)} patients in the order, {len(kept)} in the primary model -> {p['summary']}")
    return res

def probe_range(pid="0284", segment=None):
    """Range support on one file: the status and Content-Range of a 64-byte request from byte 100."""
    segment = segment or {"seg": 1, "hour": 4}
    url = HttpSource().url(pid, segment, "mat")
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=100-163"})
    with urllib.request.urlopen(req, timeout=60) as r:
        n = len(r.read())
        return {"status": r.status, "content_range": r.headers.get("Content-Range"), "bytes": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--probe-range", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--split", choices=("dev", "analysis"), default="analysis")
    ap.add_argument("--limit", type=int, default=None, help="the first N patients of the run's order (dev smoke)")
    ap.add_argument("--workers", type=int, default=N_CONN, help="download connections of this process")
    ap.add_argument("--shard", type=int, default=0, help="analysis process i of --nshards (a slice of the seeded order)")
    ap.add_argument("--nshards", type=int, default=1, help="analysis processes; workers x nshards <= 6 connections")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--fallback-n", type=int, default=None, help="analyse the first N of the seeded order only")
    ap.add_argument("--morgoth-worker", action="store_true")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.morgoth_worker:
        return morgoth_worker()
    if a.probe_range:
        say("range probe:", probe_range())
    if a.index:
        build_index()
    if a.status:
        say("status:", json.dumps(status(a.split)))
    if a.run:
        if a.workers * a.nshards > N_CONN or not 0 <= a.shard < a.nshards:
            raise SystemExit(f"at most {N_CONN} connections in all (workers x nshards), and 0 <= shard < nshards")
        index = load_index()
        dev = dev_patients(index)
        if a.split == "analysis":
            guard_pinned()
            pids = [p for p in processing_order(list(index)) if p not in dev]
        else:
            pids = list(dev)
        pids = take_shard(pids, a.shard, a.nshards, a.limit)
        mx = _mx()
        cli = MorgothClient(a.split)
        sp = mx.Spectra(paths(a.split)["spectra"])
        try:
            counts = run_arm(pids, index, a.split, label_fn=cli, spectra=sp, workers=a.workers)
        finally:
            cli.close()
        say("run:", json.dumps(counts))
    if a.analyse:
        analyse(fallback_n=a.fallback_n)


if __name__ == "__main__":
    main()
