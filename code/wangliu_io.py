"""Wang Z, Liu Y et al. 2026, "sixth-finger" and affected-hand motor imagery after stroke (Sci Data, doi
10.1038/s41597-026-07787-y; figshare article 30509165, doi 10.6084/m9.figshare.30509165, CC BY 4.0). 24 patients, each in
group B or T (the letter in the file names; the release's metadata files do not explain it), timepoints pre / post / follow
(sub-05, 17, 18, 20, 21 have no follow), 4 runs per timepoint: 268 EEGLAB runs sub-XX/<tp>/sub-XX_{B|T}_<tp>_YY_original
.set + .fdt in sourcedata.zip (file 64877880, 5.18 GB, zip64, every file deflated). Nothing downloads the whole archive (the
README warns against figshare 'Download All'): zip_members() reads its central directory and fetch() one session's members
by HTTP range requests, streamed to disk, resumable, size- and CRC-checked, into RAW/sourcedata/<member path>.
A run: 250 Hz, 64 rows = 59 EEG (no CPz, the online reference; the file's ref field says 'common') + ECG (constant in every
DEV_SUBJECT run) + HEOR HEOL VEOU VEOL, which load_run drops; µV, DC-coupled (offsets up to ~33 mV), unfiltered (no
high-pass, no notch). The .fdt is float32 written channel-fastest. scipy.io.loadmat fails on a whole .set ('buffer is too
small': the de-identified filepath / comments are malformed), so only SET_VARS are read. The top-level electrodes.tsv
(40 sites) and task-MotorImagery_eeg.json (60 EEG, average reference) do not describe these raw files.
Events (EEGLAB 1-based sample latencies): 1 rest onset (green cross, 7 s), 2 affected-hand MI cue, 3 sixth-finger MI cue;
designed trial 7-s rest, 1-s grey circle, 6-s MI. Rest to MI marker 8.55-8.59 s in DEV_SUBJECT pre (8.548-8.716 s reported
over all runs), MI to the next rest 6.6-12.0 s. A standard run is 20 rest + 10 + 10 in random order, rest always first;
reported exceptions (confirmation data, not read here): sub-15 follow 02 aborted (5 rest, no MI), sub-17 pre 04 (19 rest
for 20 MI), 21-23 trials in sub-06 follow 03, sub-10 post 02, sub-13 pre 01. trials() pairs rest and MI markers and counts
what it drops; check() guards event order, codes and units. DEV_SUBJECT is for plumbing only and is never analysed."""
import hashlib
import json
import os
import re
import shutil
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
RAW = os.path.join(ROOT, "output", "raw", "wangliu_stroke")
ARTICLE, SOURCE_ZIP_ID = 30509165, 64877880
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
URL = f"https://ndownloader.figshare.com/files/{SOURCE_ZIP_ID}"
LICENCE = "CC BY 4.0"
MIN_FREE, CHUNK = 20e9, 1 << 20
SUBJECTS = list(range(1, 25))
TIMEPOINTS = ("pre", "post", "follow")
DEV_SUBJECT = 1
FS = 250.0
N_EEG = 59
NOT_EEG = ("ECG", "HEOR", "HEOL", "VEOU", "VEOL")
SET_VARS = ["nbchan", "trials", "pnts", "srate", "chanlocs", "event", "datfile", "ref"]
REST, MI_TYPE = 1, {2: "affected_hand", 3: "sixth_finger"}
REST_MI_GAP_S = (8.4, 8.9)
HP_HZ, SD_UV = 1.0, (1.0, 200.0)


def _range(a, b, url=URL):
    """Bytes a..b (inclusive) of the archive as a stream of chunks: one ranged GET (figshare redirects to S3), which must
    answer 206 with exactly that span."""
    import requests
    with requests.get(url, headers={"Range": f"bytes={a}-{b}"}, stream=True, timeout=120) as r:
        if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {a}-{b}/"):
            raise IOError(f"{url}: HTTP {r.status_code}, Content-Range {r.headers.get('Content-Range')} for bytes {a}-{b}")
        yield from r.iter_content(CHUNK)


def _read(a, b, rng):
    d = b"".join(rng(a, b))
    if len(d) != b - a + 1:
        raise IOError(f"short read: {len(d)} of {b - a + 1} bytes at {a}")
    return d


def _central(size, rng=_range):
    """Entries of a zip of `size` bytes, read through rng(a, b) from its end record (zip64 aware) and central directory:
    name, offset (of the local header), csize, usize, method (0 stored, 8 deflated), crc (CRC-32)."""
    tail = _read(max(0, size - 65558), size - 1, rng)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise IOError("no end-of-central-directory record")
    n, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    j = tail.rfind(b"PK\x06\x07", 0, i)
    if j >= 0:
        e64 = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        rec = _read(e64, e64 + 55, rng)
        if rec[:4] != b"PK\x06\x06":
            raise IOError("no zip64 end record where the locator points")
        n, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
    cd = _read(cd_off, cd_off + cd_size - 1, rng)
    out, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        h = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        flags, method, crc, csize, usize, nlen, elen, clen, off = h[3], h[4], h[7], h[8], h[9], h[10], h[11], h[12], h[16]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        extra, q = cd[p + 46 + nlen:p + 46 + nlen + elen], 0
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                v = iter(struct.unpack(f"<{hl // 8}Q", extra[q + 4:q + 4 + hl - hl % 8]))
                usize = next(v) if usize == 0xFFFFFFFF else usize
                csize = next(v) if csize == 0xFFFFFFFF else csize
                off = next(v) if off == 0xFFFFFFFF else off
            q += 4 + hl
        out.append({"name": name, "offset": off, "csize": csize, "usize": usize, "method": method, "crc": crc})
        p += 46 + nlen + elen + clen
    if len(out) != n:
        raise IOError(f"central directory holds {len(out)} entries, end record says {n}")
    return out


def _extract(m, path, rng=_range, tries=4, min_free=MIN_FREE):
    """One member to `path`. Its compressed bytes are streamed (appended, so a rerun or a dropped connection resumes) into
    path.part, then inflated (method 8) or copied (0) into path.tmp, which must have usize bytes and the member's CRC-32
    before it becomes path. A file already of usize bytes is skipped. A failed check leaves no file."""
    if os.path.exists(path) and os.path.getsize(path) == m["usize"]:
        return path
    if m["method"] not in (0, 8):
        raise ValueError(f"{m['name']}: compression method {m['method']} (only stored / deflated)")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if shutil.disk_usage(d).free - m["csize"] - m["usize"] < min_free:
        raise RuntimeError(f"{d}: under {min_free / 1e9:.0f} GB would stay free")
    head = _read(m["offset"], m["offset"] + 29, rng)
    if head[:4] != b"PK\x03\x04":
        raise IOError(f"{m['name']}: no local header at {m['offset']}")
    start = m["offset"] + 30 + sum(struct.unpack("<HH", head[26:30]))
    part, tmp = path + ".part", path + ".tmp"
    size = lambda f: os.path.getsize(f) if os.path.exists(f) else 0
    for k in range(tries):
        if size(part) >= m["csize"]:
            break
        try:
            with open(part, "ab") as fh:
                for b in rng(start + size(part), start + m["csize"] - 1):
                    fh.write(b)
        except OSError:
            if k == tries - 1:
                raise
    if size(part) != m["csize"]:
        if size(part) > m["csize"]:
            os.remove(part)
        raise IOError(f"{m['name']}: {size(part)} of {m['csize']} compressed bytes after {tries} tries")
    try:
        z = zlib.decompressobj(-15) if m["method"] == 8 else None
        crc = n = 0
        with open(part, "rb") as src, open(tmp, "wb") as dst:
            for b in iter(lambda: src.read(CHUNK), b""):
                b = z.decompress(b) if z else b
                crc, n = zlib.crc32(b, crc), n + len(b)
                dst.write(b)
            if z and not z.eof:
                raise IOError(f"{m['name']}: deflate stream truncated")
        if n != m["usize"]:
            raise IOError(f"{m['name']}: {n} bytes, the central directory says {m['usize']}: size mismatch")
        if m.get("crc") is not None and crc != m["crc"]:
            raise IOError(f"{m['name']}: crc mismatch")
    except (OSError, zlib.error) as e:
        for f in (tmp, part):
            if os.path.exists(f):
                os.remove(f)
        raise IOError(str(e)) from e
    os.replace(tmp, path)
    os.remove(part)
    return path


def _article():
    import requests
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    return r.json()


def _size(url=URL):
    """Archive size from the Content-Range of a one-byte ranged GET."""
    import requests
    with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        if r.status_code != 206 or "/" not in cr:
            raise IOError(f"{url}: HTTP {r.status_code}, Content-Range {cr!r}")
        return int(cr.rsplit("/", 1)[1])


def zip_members(root=RAW, refresh=False):
    """Central directory of sourcedata.zip (name, offset, csize, usize, method, crc per entry) read by range requests
    (~0.1 MB), after the article's licence (CC BY 4.0) and the archive's size against the API listing are checked; cached
    in root/sourcedata_members.json, which is read instead unless refresh (needed after a new figshare version)."""
    path = os.path.join(root, "sourcedata_members.json")
    if os.path.exists(path) and not refresh:
        with open(path) as fh:
            return json.load(fh)["members"]
    art = _article()
    if art.get("license", {}).get("name") != LICENCE:
        raise ValueError(f"unexpected licence {art.get('license')}: {LICENCE} required")
    f = [x for x in art["files"] if x["id"] == SOURCE_ZIP_ID and x["name"] == "sourcedata.zip"]
    if len(f) != 1:
        raise ValueError(f"file {SOURCE_ZIP_ID} (sourcedata.zip) not in article {ARTICLE}")
    size = _size()
    if size != f[0]["size"]:
        raise IOError(f"archive size {size}, the article lists {f[0]['size']}")
    ms = _central(size, _range)
    os.makedirs(root, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump({"source": URL, "article": ARTICLE, "licence": LICENCE, "size": size, "md5": f[0]["computed_md5"],
                   "members": ms}, fh, indent=0)
    os.replace(path + ".tmp", path)
    return ms


_MEMBER = re.compile(r"^sub-(\d\d)/(pre|post|follow)/sub-\1_([BT])_\2_(\d\d)_original\.(set|fdt)$")


def _runs(members):
    """{(subject, timepoint): {run: {ext: member name}}}; directories, '._*' sidecars, '..' paths and anything else that
    is not sub-XX/<tp>/sub-XX_{B|T}_<tp>_YY_original.{set|fdt} are ignored. members: zip_members() dicts or names."""
    out = {}
    for n in (m["name"] if isinstance(m, dict) else str(m) for m in members):
        m = _MEMBER.match(n)
        if m and ".." not in n.split("/"):
            out.setdefault((int(m.group(1)), m.group(2)), {}).setdefault(int(m.group(4)), {})[m.group(5)] = n
    return out


def session_members(subj, timepoint, members=None):
    """The .fdt / .set member names of one session's runs, sorted by run number. A run lacking either file is an error."""
    if timepoint not in TIMEPOINTS:
        raise ValueError(f"unknown timepoint {timepoint!r}; one of {TIMEPOINTS}")
    runs = _runs(zip_members() if members is None else members).get((subj, timepoint), {})
    out = []
    for r in sorted(runs):
        if set(runs[r]) != {"set", "fdt"}:
            raise ValueError(f"sub-{subj:02d} {timepoint} run {r:02d}: has {sorted(runs[r])}, needs .set and .fdt")
        out += [runs[r]["fdt"], runs[r]["set"]]
    return out


def sessions(subj, members=None):
    """Timepoints with at least one run of this subject in the archive listing, in TIMEPOINTS order."""
    runs = _runs(zip_members() if members is None else members)
    return [tp for tp in TIMEPOINTS if runs.get((subj, tp))]


def local(name, root=RAW):
    return os.path.join(root, "sourcedata", *name.split("/"))


def fetch(subj, timepoint, root=RAW):
    """Extract one session's runs (.set + .fdt; never '._*' sidecars) from the remote archive by range requests into
    root/sourcedata/<member path>; complete files are skipped, partial ones resumed. -> the .set paths, by run."""
    ms = {m["name"]: m for m in zip_members(root)}
    names = session_members(subj, timepoint, list(ms.values()))
    if not names:
        raise FileNotFoundError(f"sub-{subj:02d} {timepoint}: no runs in the archive")
    for n in names:
        _extract(ms[n], local(n, root), _range)
    return [local(n, root) for n in names if n.endswith(".set")]


@dataclass
class Run:
    subject: int
    timepoint: str
    run: int
    group: str
    x: np.ndarray
    fs: float
    names: list
    events: pd.DataFrame
    sha256: str
    meta: dict = field(default_factory=dict)


_FILE = re.compile(r"^sub-(\d\d)_([BT])_(pre|post|follow)_(\d\d)_original\.set$")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def load_run(set_path):
    """One run -> Run: x = the 59 EEG rows (file order; NOT_EEG dropped by name) of the .fdt, µV, float32; events sorted
    by latency (latency_s = (EEGLAB 1-based sample - 1) / fs; integer codes). The .fdt is float32 written channel-fastest
    (MATLAB order of nbchan x pnts) and must hold exactly nbchan * pnts values. ValueError on an unexpected file name,
    channel set, rate, epoched file, .fdt size or a non-integer event type."""
    from scipy.io import loadmat
    m = _FILE.match(os.path.basename(set_path))
    if not m:
        raise ValueError(f"{set_path}: not a sub-XX_{{B|T}}_<timepoint>_YY_original.set file")
    h = loadmat(set_path, variable_names=SET_VARS, squeeze_me=True, struct_as_record=False)
    C, N, fs = int(h["nbchan"]), int(h["pnts"]), float(h["srate"])
    fdt = set_path[:-4] + ".fdt"
    if int(h["trials"]) != 1 or abs(fs - FS) > 1e-9 or str(h["datfile"]) != os.path.basename(fdt):
        raise ValueError(f"{set_path}: trials {h['trials']}, srate {fs}, datfile {h['datfile']!r}")
    names = [str(c.labels) for c in np.atleast_1d(h["chanlocs"])]
    keep = [i for i, c in enumerate(names) if c not in NOT_EEG]
    if len(names) != C or len(set(names)) != C or len(keep) != N_EEG or sorted(set(names) & set(NOT_EEG)) != sorted(NOT_EEG):
        raise ValueError(f"{set_path}: channels {names}; expected {N_EEG} EEG + {NOT_EEG}")
    if os.path.getsize(fdt) != 4 * C * N:
        raise ValueError(f"{fdt}: {os.path.getsize(fdt)} bytes, {C} x {N} float32 is {4 * C * N}")
    raw = np.memmap(fdt, "<f4", "r", shape=(N, C))
    x = np.ascontiguousarray(raw[:, keep].T, dtype=np.float32)
    del raw
    ev = list(np.atleast_1d(h["event"])) if np.size(h["event"]) else []
    code = []
    for e in ev:
        try:
            code.append(int(str(e.type).strip()))
        except ValueError:
            raise ValueError(f"{set_path}: event type {e.type!r} is not an integer code") from None
    lat = np.array([(float(e.latency) - 1) / fs for e in ev], float)
    events = pd.DataFrame({"latency_s": lat, "code": np.array(code, int)}).sort_values("latency_s", kind="stable")
    return Run(subject=int(m.group(1)), timepoint=m.group(3), run=int(m.group(4)), group=m.group(2), x=x, fs=fs,
               names=[names[i] for i in keep], events=events.reset_index(drop=True), sha256=_sha256(fdt),
               meta={"set": set_path, "fdt": fdt, "nbchan_file": C, "pnts": N, "duration_s": N / fs,
                     "dropped": [c for c in names if c in NOT_EEG], "ref_field": str(h["ref"]),
                     "events_sorted_in_file": bool(np.all(np.diff(lat) >= 0))})


def trials(run):
    """Trials of a run: each rest marker (1) paired with the next MI marker (2 / 3) when the gap lies in REST_MI_GAP_S
    (inclusive). An MI marker claimed by two rest markers goes to the later (nearer) one. Everything unpaired is dropped
    and counted in run.meta['unpaired']: rest / mi markers left, and why each rest was left (no MI after it, gap outside
    the window, MI taken by a later rest). -> DataFrame rest_s, mi_s, mi_type ('affected_hand' / 'sixth_finger')."""
    ev = run.events.sort_values("latency_s", kind="stable")
    t, c = ev["latency_s"].to_numpy(float), ev["code"].to_numpy(int)
    mi = np.flatnonzero(np.isin(c, list(MI_TYPE)))
    lo, hi = REST_MI_GAP_S
    n = {"rest_without_next_mi": 0, "gap_outside_window": 0, "rest_superseded": 0}
    claim = {}
    for i in np.flatnonzero(c == REST):
        nxt = mi[mi > i]
        if not len(nxt):
            n["rest_without_next_mi"] += 1
        elif not lo - 1e-9 <= t[nxt[0]] - t[i] <= hi + 1e-9:
            n["gap_outside_window"] += 1
        else:
            n["rest_superseded"] += nxt[0] in claim
            claim[nxt[0]] = i
    run.meta["unpaired"] = {"rest": int((c == REST).sum()) - len(claim), "mi": len(mi) - len(claim), **n}
    j = sorted(claim)
    return pd.DataFrame({"rest_s": t[[claim[k] for k in j]], "mi_s": t[j], "mi_type": [MI_TYPE[c[k]] for k in j]},
                        columns=["rest_s", "mi_s", "mi_type"])


def hp_sd(x, fs, hz=HP_HZ, block=8):
    """SD of each row after a zero-phase 4th-order Butterworth high-pass at `hz` (sosfiltfilt's odd padding absorbs the DC
    offsets of these DC-coupled recordings); `block` rows at a time to keep the float64 copies small."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, hz, btype="highpass", fs=fs, output="sos")
    out = []
    for a in range(0, x.shape[0], block):
        out.append(sosfiltfilt(sos, np.asarray(x[a:a + block], np.float64), axis=1).std(axis=1))
    return np.concatenate(out)


def check(run):
    """Sanity of a Run; ValueError when the events are not sorted by latency, carry a code outside {1, 2, 3}, x has not one
    row per name (N_EEG) or another rate than FS, or the median over channels of the 1-Hz high-passed SD lies outside
    SD_UV (units not µV). -> counts: rest / affected_hand / sixth_finger markers, channels, samples, median_sd_uv."""
    lat, code = run.events["latency_s"].to_numpy(float), run.events["code"].to_numpy(int)
    tag = f"sub-{run.subject:02d} {run.timepoint} run {run.run:02d}"
    if np.any(np.diff(lat) < 0):
        raise ValueError(f"{tag}: events not sorted by latency")
    odd = sorted(set(code.tolist()) - {REST, *MI_TYPE})
    if odd:
        raise ValueError(f"{tag}: event code(s) {odd} outside {{1, 2, 3}}")
    if run.x.ndim != 2 or run.x.shape[0] != len(run.names) or len(run.names) != N_EEG:
        raise ValueError(f"{tag}: x {run.x.shape} for {len(run.names)} names; {N_EEG} EEG channels expected")
    if abs(run.fs - FS) > 1e-9:
        raise ValueError(f"{tag}: sampling rate {run.fs}, expected {FS}")
    sd = float(np.median(hp_sd(run.x, run.fs)))
    if not SD_UV[0] <= sd <= SD_UV[1]:
        raise ValueError(f"{tag}: median {HP_HZ:g}-Hz high-passed SD {sd:.3g}, outside {SD_UV} µV: units?")
    return {"rest": int((code == REST).sum()), **{v: int((code == k).sum()) for k, v in MI_TYPE.items()},
            "channels": int(run.x.shape[0]), "samples": int(run.x.shape[1]), "median_sd_uv": sd}
