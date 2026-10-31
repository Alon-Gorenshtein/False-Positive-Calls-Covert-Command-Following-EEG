"""Braille BCI tactile P300 (Grigoryan R, Goranskaya D, Demchinsky A, Ryabova K, Kuleshov D, Kaplan A; G-Node GIN
doi 10.12751/g-node.dyby2g, https://gin.g-node.org/doi/BrailleBCI, CC0 1.0 (LICENSE and datacite.yml), BIDS). 10 blind
and 10 sighted participants, SUBJECTS sub-6 and sub-8 .. sub-26 (no sub-7), each with ses-large (8-dot) and ses-small
(1-dot) stimuli. One finger rests on each of 8 Braille cells (right hand 0 pinky 1 ring 2 middle 3 index, left hand
4 index 5 middle 6 ring 7 pinky); a selection cycle raises every cell 10 times in random order (80 stimuli, SOA median
0.300 s, min 0.252 s; target rate 1/8) and the participant counts the target cell's activations; 16 cycles per session,
~13 s apart, targets 0 1 2 3 4 5 7 6 0 1 2 3 4 5 6 7 (sub-6 large).
Layout: sub-<n>/ses-<s>/eeg/sub-<n>_ses-<s>_task-bci_{eeg.bdf, events.tsv, channels.tsv, electrodes.tsv, eeg.json}; subject
numbers are not zero-padded (all 40 sessions listed on GIN). Quirks (sub-6 large): the BDF has the 48 rows of CHANNELS,
500 Hz, unit uV, no Status row and no annotations; it is raw (DC offsets -123 .. +26 mV, strong 50 Hz line) although
channels.tsv says 0.1-35 Hz. Reference A1A2 (linked earlobes): the A1 and A2 rows are exact negatives (|A1 + A2| < 0.3
µV); both are dropped with ecg (ecg_r_peak_direction 0 / 2 = no valid ECG; 0 for sub-6), leaving SCALP (45). Events come
only from events.tsv: value = stimulated cell (= its `finger` column, `fingers` map identity), trial_type 1 target,
target = current target cell, session_id = cycle 0-15, sample = onset * 500 exactly; epoch_id is always 0. GIN serving
quirks: see download(). load(subj, session) -> Run (x float32 µV, 45 x n); check(run) validates design and units.
DEV_SUBJECT is for plumbing only, never analysed."""
import gzip
import hashlib
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
RAW = os.path.join(ROOT, "output", "raw", "braille_p300")
REPO = "https://gin.g-node.org/doi/BrailleBCI"
RAW_URL, TREE_URL = f"{REPO}/raw/master", f"{REPO}/src/master"
DOI, LICENCE = "10.12751/g-node.dyby2g", "CC0 1.0"
SUBJECTS = [6] + list(range(8, 27))
SESSIONS = ("large", "small")
DEV_SUBJECT = 6
FS = 500.0
CHANNELS = ["Fp1", "ecg", "Fp2", "F3", "Fz", "F4", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "T7", "C5", "C3", "C1",
            "Cz", "C2", "C4", "C6", "T8", "CP5", "A1", "CP3", "CP1", "CPz", "CP2", "CP4", "A2", "CP6", "P7", "P5", "P3", "P1",
            "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
DROP = ("ecg", "A1", "A2")
SCALP = ["Fp1", "Fp2", "F3", "Fz", "F4", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "T7", "C5", "C3", "C1", "Cz", "C2",
         "C4", "C6", "T8", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
         "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
TOL_S = 0.002
N_PER_FINGER = 10
SD_UV = (1.0, 2000.0)


@dataclass
class Run:
    subject: int
    session: str
    x: np.ndarray
    fs: float
    names: list
    events: pd.DataFrame
    sha256: str
    meta: dict


def paths(subj, session):
    """BIDS paths in the repository (as listed on GIN; subject numbers are not zero-padded)."""
    if subj not in SUBJECTS:
        raise ValueError(f"no subject {subj!r}: the dataset has {SUBJECTS}")
    if session not in SESSIONS:
        raise ValueError(f"no session {session!r}: the dataset has {SESSIONS}")
    stem = f"sub-{subj}/ses-{session}/eeg/sub-{subj}_ses-{session}_task-bci"
    return {"bdf": f"{stem}_eeg.bdf", "events": f"{stem}_events.tsv"}


def events_table(ev, fs=FS, tol_s=TOL_S):
    """events.tsv rows -> DataFrame(onset_s, finger, is_target, cycle, target) sorted by onset. finger = value (stimulated
    cell 0-7, the code space of `target`); the file's own `finger` column must equal it (identity `fingers` map), or the
    README's finger coding would not apply. is_target = trial_type (0 / 1 only); cycle = session_id. sample / fs must equal
    onset within tol_s."""
    bad = np.abs(ev["sample"].to_numpy(float) / fs - ev["onset"].to_numpy(float))
    if bad.max() > tol_s + 1e-9:
        raise ValueError(f"sample / {fs:g} differs from onset by up to {bad.max() * 1e3:.1f} ms (> {tol_s * 1e3:g} ms)")
    if "finger" in ev and (ev["finger"].to_numpy() != ev["value"].to_numpy()).any():
        fmap = ev["fingers"].iloc[0] if "fingers" in ev else None
        raise ValueError(f"finger column differs from value (stimulated cell); fingers map {fmap!r}")
    if not ev["trial_type"].isin([0, 1]).all():
        raise ValueError(f"trial_type values {sorted(set(ev['trial_type']))}: only 0 / 1 expected")
    out = pd.DataFrame({"onset_s": ev["onset"].astype(float), "finger": ev["value"].astype(int),
                        "is_target": ev["trial_type"].astype(int) == 1, "cycle": ev["session_id"].astype(int),
                        "target": ev["target"].astype(int)})
    return out.sort_values("onset_s", kind="stable").reset_index(drop=True)


def hp_sd(x, fs, hz=1.0, hi=40.0):
    """SD of each row after a 4th-order zero-phase `hz`-`hi` Hz band-pass (row by row: float64 copies of one row only);
    the 40-Hz edge keeps the 50-Hz line out of a check that exists to catch V / nV scaling."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [hz, hi], btype="bandpass", fs=fs, output="sos")
    return np.array([sosfiltfilt(sos, r.astype(np.float64) - r.mean(dtype=np.float64)).std() for r in x])


def check(run, sd_uv=SD_UV):
    """Design and units of one run; ValueError naming the first problem. Events sorted by onset; every cycle holds exactly
    N_PER_FINGER stimuli of each finger 0-7; one target per cycle; is_target == (finger == target); median over rows of the
    1-40 Hz band-passed SD within sd_uv µV (catches V / nV). -> counts, per-cycle [min, max] summaries, target order and
    the median SD."""
    ev = run.events
    if not ev.onset_s.is_monotonic_increasing:
        raise ValueError("events are not sorted by onset")
    if not ev.finger.isin(range(8)).all():
        raise ValueError(f"finger values {sorted(set(ev.finger) - set(range(8)))} outside 0-7")
    n = ev.groupby(["cycle", "finger"]).size().unstack(fill_value=0).reindex(columns=range(8), fill_value=0)
    bad = n.index[(n != N_PER_FINGER).any(axis=1)].tolist()
    if bad:
        raise ValueError(f"cycle(s) {bad} do not hold exactly {N_PER_FINGER} stimuli of every finger 0-7")
    k = ev.groupby("cycle").target.nunique()
    if (k != 1).any():
        raise ValueError(f"target is not constant within cycle(s) {k.index[k != 1].tolist()}")
    wrong = ev.is_target.astype(bool) != (ev.finger == ev.target)
    if wrong.any():
        raise ValueError(f"is_target != (finger == target) at {int(wrong.sum())} event(s), first at onset "
                         f"{ev.onset_s[wrong].iloc[0]:.3f} s")
    sd = float(np.median(hp_sd(run.x, run.fs)))
    if not sd_uv[0] <= sd <= sd_uv[1]:
        raise ValueError(f"median 1-40 Hz band-passed SD {sd:.3g} outside {sd_uv[0]:g}-{sd_uv[1]:g} µV: x is not in µV?")
    per = ev.groupby("cycle")
    span = lambda s: [int(s.min()), int(s.max())]
    return {"n_stim": len(ev), "n_target": int(ev.is_target.sum()), "n_cycles": int(ev.cycle.nunique()),
            "per_cycle": {"stim": span(per.size()), "target": span(per.is_target.sum()), "per_finger": span(n.values)},
            "target_order": per.target.first().astype(int).tolist(), "median_sd_uv": sd}


def _http(url, headers, method="GET"):
    """(status, lower-cased headers, iterator over the body bytes as sent: not gzip-decoded)."""
    import requests
    r = requests.request(method, url, headers=headers, stream=True, timeout=300)
    h = {k.lower(): v for k, v in r.headers.items()}
    if r.status_code not in (200, 206):
        r.close()
        raise IOError(f"{url}: HTTP {r.status_code}")
    if method == "HEAD":
        r.close()
        return r.status_code, h, iter(())

    def body():
        with r:
            yield from r.raw.stream(1 << 20, decode_content=False)
    return r.status_code, h, body()


def download(url, path, http=None):
    """Fetch url into path, resumably; the bytes on the wire are size-checked against the server's Content-Length (or the
    Content-Range total) before path appears. A file already at path with the HEAD Content-Length (or, when HEAD gives none,
    any file there: it only ever appears after a checked transfer) is not fetched again. GIN (Oct 2026): annexed files
    (the BDFs) carry a length only uncompressed and ignore Range (a partial then restarts from 0); plain-git files
    (events.tsv) carry one only gzip-encoded, and honour Range on that stream. So identity is asked for when HEAD gives a
    length, gzip otherwise; the partial is <path>.<encoding>.part and survives a failed or short transfer."""
    http = http or _http
    h = http(url, {"Accept-Encoding": "identity"}, "HEAD")[1]
    size = int(h["content-length"]) if "content-length" in h else None
    if os.path.exists(path) and (size is None or os.path.getsize(path) == size):
        return path
    enc = "identity" if size is not None else "gzip"
    part = f"{path}.{enc}.part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    status, h, body = http(url, dict({"Accept-Encoding": enc}, **({"Range": f"bytes={have}-"} if have else {})))
    if status == 206:
        m = re.fullmatch(r"bytes (\d+)-\d+/(\d+)", h.get("content-range", ""))
        if not m or int(m.group(1)) != have:
            raise IOError(f"{url}: Content-Range {h.get('content-range')!r} does not resume at byte {have}")
        total = int(m.group(2))
    elif "content-length" in h:
        have, total = 0, int(h["content-length"])
    elif enc == "gzip":
        have, total = 0, None
    else:
        raise IOError(f"{url}: no Content-Length for {enc}; refusing a transfer whose size cannot be checked")
    if h.get("content-encoding", "identity") != enc:
        raise IOError(f"{url}: asked for {enc}, got {h.get('content-encoding', 'identity')}")
    if size is not None and total != size:
        raise IOError(f"{url}: GET says {total} bytes, HEAD said {size}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(part, "ab" if have else "wb") as fh:
        for b in body:
            fh.write(b)
    got = os.path.getsize(part)
    if total is not None and got != total:
        if got > total:
            os.remove(part)
        raise IOError(f"{url}: {got} of {total} bytes{'; partial kept for resume' if got < total else ''}")
    if enc == "gzip":
        with gzip.open(part) as src, open(path + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(path + ".tmp", path)
        os.remove(part)
    else:
        os.replace(part, path)
    return path


def names_in(html, folder):
    """File / folder names directly in `folder` from a GIN tree page, in order, once each; '._*' (AppleDouble) ignored."""
    out = []
    for n in re.findall(rf'href="/doi/BrailleBCI/src/master/{re.escape(folder)}/([^"/?#]+)"', html):
        if not n.startswith("._") and n not in out:
            out.append(n)
    return out


def listing(subj, session):
    """Names in the session's eeg folder on GIN (HTML tree; the API has no contents endpoint)."""
    import requests
    folder = os.path.dirname(paths(subj, session)["bdf"])
    r = requests.get(f"{TREE_URL}/{folder}", timeout=60)
    r.raise_for_status()
    return names_in(r.text, folder)


def fetch(subj, session, root=RAW):
    """Download the BDF and events.tsv of one session to root/<BIDS path>, after checking the server lists both. Resumable,
    complete files skipped, every transfer size-checked (download()). -> {bdf, events: local path}."""
    p = paths(subj, session)
    have = listing(subj, session)
    miss = [os.path.basename(v) for v in p.values() if os.path.basename(v) not in have]
    if miss:
        raise FileNotFoundError(f"sub-{subj} ses-{session}: {miss} not in the GIN listing {have}")
    return {k: download(f"{RAW_URL}/{v}", os.path.join(root, v)) for k, v in p.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def events_only(subj, session, root=RAW):
    """events_table of one session from its events.tsv alone (downloaded if missing; the BDF is never fetched)."""
    rel = paths(subj, session)["events"]
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        download(f"{RAW_URL}/{rel}", path)
    return events_table(pd.read_csv(path, sep="\t"))


def load(subj, session, root=RAW):
    """One session -> Run: the 45 SCALP rows in µV (float32; DC offsets kept), events from events.tsv. Fetches the files
    if either is missing. Refuses a rate other than FS, channels other than CHANNELS in file order, an events file of another
    user / reg, and events past the end of the recording. meta: source, files, dropped rows, the `fingers` map, n_samples and
    max |A1 + A2| (the earlobe rows are mirror images: ~0). Peak memory ~5x the float32 x (all 48 rows are read as float64)."""
    import mne
    p = {k: os.path.join(root, v) for k, v in paths(subj, session).items()}
    if not all(os.path.exists(v) for v in p.values()):
        p = fetch(subj, session, root)
    raw = mne.io.read_raw_bdf(p["bdf"], preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - FS) > 1e-9:
        raise ValueError(f"{p['bdf']}: sampling rate {raw.info['sfreq']}, expected {FS:g}")
    if raw.ch_names != CHANNELS:
        raise ValueError(f"{p['bdf']}: channels {raw.ch_names} differ from the documented 48 {CHANNELS}")
    n = raw.n_times
    a = raw.get_data(picks=["A1", "A2"]) * 1e6
    x = (raw.get_data(picks=SCALP) * 1e6).astype(np.float32)
    del raw
    tsv = pd.read_csv(p["events"], sep="\t")
    who = sorted({(int(u), str(r)) for u, r in zip(tsv["user"], tsv["reg"])})
    if who != [(subj, session)]:
        raise ValueError(f"{p['events']}: rows of user / reg {who}, expected [({subj}, {session!r})]")
    if not 0 <= tsv["sample"].min() <= tsv["sample"].max() < n:
        raise ValueError(f"{p['events']}: event samples {tsv['sample'].min()}-{tsv['sample'].max()} past the end of the "
                         f"recording (0-{n - 1})")
    meta = {"doi": DOI, "licence": LICENCE, "files": paths(subj, session), "dropped": list(DROP),
            "fingers": str(tsv["fingers"].iloc[0]) if "fingers" in tsv else None, "n_samples": int(n),
            "a1_plus_a2_max_uv": float(np.abs(a.sum(axis=0)).max())}
    return Run(subject=subj, session=session, x=x, fs=FS, names=list(SCALP), events=events_table(tsv),
               sha256=sha256(p["bdf"]), meta=meta)
