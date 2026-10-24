"""Liu 2025 lower-limb (gait) motor imagery after stroke (Sci Data 12:314, doi 10.1038/s41597-025-04618-4; figshare
10.6084/m9.figshare.27130299, CC BY 4.0; BIDS re-host Zenodo 10.5281/zenodo.18987384, CC BY 4.0, one zip per subject).
27 patients; sessions pre / post / follow are pure MI (no stimulation), mises / miies are MI + electrical stimulation
and are never read. Trial: baseline marker (1 MI, 7 idle) at 0 s, 3-s fixation, text cue (2 / 8) at 3 s, imagery or
idle onset (3 / 9) at 4 s for 5 s, then relaxation. fetch() keeps MOTOR_CHANNELS (less CPz, which the cap lacks) at 250 Hz
in µV, one npz per session; person() builds a 118 Person (imagery / cued idle / pre-cue fixation sources, never the relaxation)."""
import concurrent.futures as cf
import hashlib
import os
import shutil
import time

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
RAW = os.path.join(ROOT, "output", "raw", "stroke_gait")
ZENODO = "https://zenodo.org/api/records/18987384"
MIN_FREE = 20e9


def _get_range(url, a, b):
    import requests
    r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=300)
    if r.status_code == 429:
        time.sleep(float(r.headers.get("retry-after", 30)))
        raise IOError("429 rate limited")
    if r.status_code != 206:
        raise IOError(f"{url}: HTTP {r.status_code} for a ranged request (206 expected)")
    return r.content


def download(url, path, size, md5=None, conns=12, chunk=16 << 20, fetch=_get_range, min_free=MIN_FREE, tries=4):
    """Fetch `url` (exactly `size` bytes) into `path` with `conns` parallel ranged requests (the host throttles per
    connection). Every chunk must have exactly its length (short reads are retried, then refused); the md5 must match.
    An existing complete file with the right md5 is reused. Never leaves a final file that failed a check."""
    if os.path.exists(path) and os.path.getsize(path) == size and (md5 is None or _md5(path) == md5):
        return path
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    if free - size < min_free:
        raise RuntimeError(f"{free / 1e9:.1f} GB free; {size / 1e9:.2f} GB needed and {min_free / 1e9:.0f} GB must stay free")
    part = path + ".part"
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.ftruncate(fd, size)
        spans = [(a, min(a + chunk, size) - 1) for a in range(0, size, chunk)]

        def one(span):
            a, b = span
            for k in range(tries):
                try:
                    data = fetch(url, a, b)
                except IOError:
                    if k == tries - 1:
                        raise
                    continue
                if len(data) == b - a + 1:
                    os.pwrite(fd, data, a)
                    return
            raise IOError(f"{url}: short read for bytes {a}-{b} after {tries} tries")
        with cf.ThreadPoolExecutor(conns) as ex:
            for f in [ex.submit(one, s) for s in spans]:
                f.result()
    except BaseException:
        os.close(fd)
        os.remove(part)
        raise
    os.close(fd)
    if md5 is not None and _md5(part) != md5:
        os.remove(part)
        raise IOError(f"{url}: md5 mismatch after download")
    os.replace(part, path)
    return path


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


FS_OUT = 250.0
PURE = ("pre", "post", "follow")
STIM = ("mises", "miies", "mi-ses", "mi-ies", "ses", "ies")
MI, IDLE = {"base": 1, "cue": 2, "onset": 3}, {"base": 7, "cue": 8, "onset": 9}
FIX_S, CUE_AT_S, ONSET_AT_S, TASK_S = 3.0, 3.0, 4.0, 5.0
TOL_S = 0.15
RENAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
ABSENT_OK = ("CPz",)

_P = []


def pc():
    """The 118 module (Source, Person, MOTOR_CHANNELS), imported once."""
    if not _P:
        import importlib
        import sys
        if CODE not in sys.path:
            sys.path.insert(0, CODE)
        _P.append(importlib.import_module("118_positive_control"))
    return _P[0]


def pure_session(name):
    if str(name).lower() in STIM:
        raise ValueError(f"session {name!r} carries electrical stimulation; only {PURE} are pure MI")
    if name not in PURE:
        raise ValueError(f"unknown session {name!r}")
    return name


def channels():
    """118 MOTOR_CHANNELS less the sites this cap does not have (ABSENT_OK); any other missing site is an error."""
    return [c for c in pc().MOTOR_CHANNELS if c not in ABSENT_OK]


def resolve(file_names, wanted=None):
    """Row index in the file of each wanted channel (default channels()), case-insensitive, 10-10 names mapped to the
    dataset's T3 / T4 / T5 / T6 when the file uses them (this Zenodo re-host already uses T7 / T8 / P7 / P8). Every
    missing channel is reported in the error."""
    wanted = list(channels() if wanted is None else wanted)
    low = {str(c).lower(): i for i, c in enumerate(file_names)}
    idx, missing = [], []
    for c in wanted:
        j = low.get(RENAME.get(c, c).lower(), low.get(c.lower()))
        if j is None:
            missing.append(c)
        idx.append(j)
    if missing:
        raise ValueError(f"channels missing from the recording: {missing}")
    return idx


def sources(ev, fs, n, run, tag):
    """Sources of one run from its markers ev (k x 2: sample, code at `fs`): for each trial the imagery / idle excerpt
    [onset, onset + 5 s) (class 'mi' / 'idle'; onset = marker 3 / 9 = text cue + 1 s) and its 3-s pre-cue fixation
    [baseline, + 3 s) (class 'fixation'). The cue (2 / 8) must precede the onset by 1 s and the baseline (1 / 7) the cue by
    3 s (+- 0.15 s); a baseline that would precede sample 0 (a run that starts mid-trial) only costs that trial its fixation excerpt. The
    post-trial relaxation is never a source; a trial cut off by the end of the run is dropped."""
    ev = np.asarray(ev, int).reshape(-1, 2)
    out = []
    tol = TOL_S * fs
    for kind, cls in ((MI, "mi"), (IDLE, "idle")):
        for s in ev[ev[:, 1] == kind["onset"], 0]:
            cue = ev[(ev[:, 1] == kind["cue"]) & (abs(ev[:, 0] - (s - (ONSET_AT_S - CUE_AT_S) * fs)) <= tol), 0]
            base = ev[(ev[:, 1] == kind["base"]) & (abs(ev[:, 0] - (s - ONSET_AT_S * fs)) <= tol), 0]
            if len(cue) != 1:
                raise ValueError(f"{tag}: onset marker at {s} has no unique cue 1 s before")
            if len(base) > 1 or (len(base) == 0 and s - ONSET_AT_S * fs + tol >= 0):
                raise ValueError(f"{tag}: onset marker at {s} has no unique baseline 4 s before")
            if s + int(TASK_S * fs) <= n:
                out.append(pc().Source(run=run, start=int(s), n=int(TASK_S * fs), cls=cls, tag=f"{tag}:{cls}@{s}"))
            if len(base) == 1:
                out.append(pc().Source(run=run, start=int(base[0]), n=int(FIX_S * fs), cls="fixation",
                                       tag=f"{tag}:fix-{cls}@{base[0]}"))
    return sorted(out, key=lambda q: q.start)


def split_runs(x, bounds, ev):
    """Cut x (ch x samples) at `bounds` (n_runs + 1 cumulative starts) and shift ev (sample, code) into each run."""
    ev = np.asarray(ev, int).reshape(-1, 2)
    runs, evs = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        runs.append(x[:, a:b])
        m = (ev[:, 0] >= a) & (ev[:, 0] < b)
        evs.append(ev[m] - np.array([a, 0]))
    return runs, evs


def code_of(description):
    """Integer stimulus code of a BrainVision marker description ('Stimulus/S 12' -> 12), else None."""
    import re
    m = re.fullmatch(r"Stimulus/S\s*(\d+)", str(description).strip())
    return int(m.group(1)) if m else None


def pack(runs):
    """[(x ch x n, events k x 2 (sample in the run, code))] -> (x concatenated, bounds = cumulative run starts, events in
    concatenated samples)."""
    if len({r[0].shape[0] for r in runs}) != 1:
        raise ValueError("runs differ in channel count")
    sizes = [r[0].shape[1] for r in runs]
    bounds = np.r_[0, np.cumsum(sizes)].astype(int)
    ev = [np.asarray(e, int).reshape(-1, 2) + np.array([b, 0]) for (_, e), b in zip(runs, bounds[:-1])]
    return np.concatenate([r[0] for r in runs], axis=1).astype(np.float32), bounds, np.vstack(ev)


def npz_path(subj, session, root=RAW):
    return os.path.join(root, f"sub-{subj:02d}_{session}.npz")


def person(subj, sessions, root=RAW):
    """One 118 Person from the saved pure-MI sessions: each saved run is a run; sources by `sources` (imagery, cued idle,
    pre-cue fixation; never the relaxation) tagged '<session>:<class>@<start>'."""
    runs, srcs, events, nums = [], [], {}, []
    for ses in sessions:
        pure_session(ses)
    for ses in sessions:
        z = np.load(npz_path(subj, ses, root))
        if list(z["names"]) != channels():
            raise ValueError(f"sub-{subj:02d} {ses}: unexpected channels")
        fs = float(z["fs"])
        parts, evs = split_runs(np.asarray(z["x"], np.float32), z["bounds"], z["events"])
        for k, (x, ev) in enumerate(zip(parts, evs)):
            run = len(runs)
            src = sources(ev, fs, x.shape[1], run, ses)
            srcs += src
            events.update({(run, q.start): q.cls for q in src})
            runs.append(x)
            nums.append(f"{ses}-run{k + 1}")
    return pc().Person(pid=f"Liu_S{subj:02d}", fs=fs, names=channels(), runs=runs, sources=srcs,
                       meta={"events": events, "run_numbers": nums, "source": "stroke_gait", "sessions": list(sessions)})


FS_IN = 1000.0


def read_run(vhdr):
    """One BrainVision run -> (x [channels(), n / 4] float32 µV at 250 Hz, events [k, 2] (sample at 250 Hz, code)).
    Only the wanted channels are read. The per-channel median is removed first: the raw recordings carry DC offsets of
    up to +-0.4 V, which the polyphase resampler's zero padding would turn into an edge transient (the offset carries no
    information at 8-30 Hz). Channels missing -> ValueError naming all of them; any rate but 1000 Hz -> ValueError."""
    import mne
    from bench import motor_common as mc
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    if abs(raw.info["sfreq"] - FS_IN) > 1e-9:
        raise ValueError(f"{vhdr}: sampling rate {raw.info['sfreq']}, expected {FS_IN:.0f}")
    idx = resolve(raw.ch_names)
    x = raw.get_data(picks=idx) * 1e6
    x -= np.median(x, axis=1, keepdims=True)
    x = mc.resample(x, FS_IN, FS_OUT).astype(np.float32)
    ev = [(int(round(o * FS_OUT)), c) for o, d in zip(raw.annotations.onset, raw.annotations.description)
          if (c := code_of(d)) is not None]
    return x, np.array(ev, int).reshape(-1, 2)


def members(names, subj, sessions=PURE):
    """Archive members to unpack: the .vhdr / .vmrk / .eeg of the given pure sessions of this subject. Nothing else
    (stimulation sessions, other subjects, anything with '..') is ever extracted."""
    import re
    for s in sessions:
        pure_session(s)
    pat = re.compile(rf"^sub-{subj:02d}/ses-({'|'.join(sessions)})/eeg/sub-{subj:02d}_ses-\1_task-MotorImagery_run-\d\d_eeg\.(vhdr|vmrk|eeg)$")
    return [n for n in names if pat.match(n) and ".." not in n.split("/")]


def run_headers(sel):
    """{session: [(run number, .vhdr member)]} in PURE order, runs ascending."""
    import re
    out = {}
    for n in sel:
        m = re.search(r"/ses-(\w+)/eeg/.*_run-(\d\d)_eeg\.vhdr$", n)
        if m:
            out.setdefault(m.group(1), []).append((int(m.group(2)), n))
    return {s: sorted(out[s]) for s in PURE if s in out}


def check_sessions(found, sessions, required=("pre",), who=""):
    """Sessions of `sessions` absent from the archive: reported (printed, returned); a missing required one raises."""
    gone = [s for s in sessions if s not in found]
    miss = [s for s in gone if s in required]
    if miss:
        raise FileNotFoundError(f"{who}: required session(s) {miss} not in the archive")
    if gone:
        print(f"{who}: session(s) {gone} not in the archive (this patient did not complete them)")
    return gone


def zenodo_file(subj):
    """(url, size, md5) of sub-XX.zip from the record's API listing; refuses a record that is not CC BY 4.0."""
    import requests
    rec = requests.get(ZENODO, timeout=60).json()
    if rec["metadata"].get("license", {}).get("id") != "cc-by-4.0":
        raise ValueError(f"unexpected licence {rec['metadata'].get('license')}")
    f = [x for x in rec["files"] if x["key"] == f"sub-{subj:02d}.zip"]
    if len(f) != 1:
        raise ValueError(f"sub-{subj:02d}.zip not in record")
    return f[0]["links"]["self"], int(f[0]["size"]), f[0]["checksum"].split(":", 1)[1]


def fetch(subj, sessions=PURE, root=RAW, conns=24, required=("pre",)):
    """Download ONE subject (zip, ~1 GB) into root/_download, unpack only the pure-MI sessions, write
    root/sub-XX_<session>.npz per session (x float32 µV 250 Hz MOTOR_CHANNELS, names, fs, bounds = run starts, events =
    (sample, code) in the concatenated samples, run_numbers, source) and delete the download. Returns the npz paths."""
    import zipfile
    outs = {s: npz_path(subj, s, root) for s in sessions}
    if all(os.path.exists(p) for p in outs.values()) or (os.path.exists(os.path.join(root, f"sub-{subj:02d}.done"))
                                                         and all(os.path.exists(outs[s]) for s in required)):
        return [p for p in outs.values() if os.path.exists(p)]
    dl = os.path.join(root, "_download")
    os.makedirs(dl, exist_ok=True)
    url, size, md5 = zenodo_file(subj)
    zpath = download(url, os.path.join(dl, f"sub-{subj:02d}.zip"), size, md5, conns=conns)
    work = os.path.join(dl, f"sub-{subj:02d}_x")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        with zipfile.ZipFile(zpath) as z:
            sel = members(z.namelist(), subj, sessions)
            for n in sel:
                z.extract(n, work)
        for ses, hdrs in run_headers(sel).items():
            runs = [read_run(os.path.join(work, h)) for _, h in hdrs]
            x, bounds, ev = pack(runs)
            tmp = outs[ses] + ".tmp.npz"
            np.savez(tmp, x=x, names=np.array(channels()), fs=FS_OUT, bounds=bounds, events=ev,
                     run_numbers=np.array([r for r, _ in hdrs]), source=f"{ZENODO} sub-{subj:02d}.zip md5 {md5}")
            os.replace(tmp, outs[ses])
        open(os.path.join(root, f"sub-{subj:02d}.done"), "w").write(f"converted {sorted(run_headers(sel))}\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if os.path.exists(zpath):
            os.remove(zpath)
    check_sessions({s for s, p in outs.items() if os.path.exists(p)}, sessions, required, f"sub-{subj:02d}")
    return [p for p in outs.values() if os.path.exists(p)]
