"""Acute-stroke hand motor imagery (50 patients, 1-30 days post-stroke; Liu H et al., Sci Data 2023, doi 10.1038/s41597-023-02787-8;
figshare 10.6084/m9.figshare.21679035, CC BY 4.0, BIDS). 40 trials per patient, one run, strictly alternating left /
right hand: ~2-s instruction, 4-s MI while a hand-specific gripping video plays (stimuli/left-hand-motor-imagery vs
right-hand-motor-imagery stim2.mp4), 2-s break; 29 EEG channels + 2 EOG at 500 Hz, reference CPz (CPz is therefore a
flat row in the EDF and is dropped). The EDF is the 40 epochs of 8 s end to end; its last, unnamed channel carries the
markers 1 / 2 / 3 = instruction / MI / break (the instruction phase varies, and 18 files have stray edges: MI onset = the
code-2 edge followed by a code-3 edge 4 s later); labels come from the BIDS events file (alternating 1 = left, 2 = right; checked equal to the per-subject
sourcedata labels of all 50 patients). fetch_all() downloads everything except the 1.9 GB raw-trial duplicate;
trials(subj) returns the 4-s MI epochs in µV at 250 Hz."""
import io
import json
import os
import re
import zipfile

import numpy as np

import stroke_gait_io as sg

RAW = os.path.join(sg.ROOT, "output", "raw", "stroke_acute")
ARTICLE = "https://api.figshare.com/v2/articles/21679035"
SKIP = ("sourcedata.zip", "code.zip")
FS_IN, FS_OUT, MI_S = 500.0, 250.0, 4.0
TRIAL_S = 8.0
LABEL = {1: "left", 2: "right"}
REF, NOT_EEG = "CPz", ("HEOL", "HEOR", "VEOR", "")
_MEMBER = re.compile(r"^(edffile/sub-\d\d/eeg/sub-\d\d_task-motor-imagery_eeg\.edf|stimuli/[\w./-]+)$")


def check_licence(article):
    if "CC BY 4.0" not in str(article.get("license", {}).get("name", "")):
        raise ValueError(f"unexpected licence {article.get('license')}: CC BY 4.0 required")


def wanted(files, skip=SKIP):
    return [f for f in files if f["name"] not in skip]


def safe_member(name):
    """An archive member we unpack: only the EDFs and the stimuli, no absolute path or '..'."""
    if not _MEMBER.match(name) or ".." in name.split("/"):
        raise ValueError(f"refusing archive member {name!r}")
    return name


def fetch_all(dest=RAW, skip=SKIP, conns=8):
    """Download the figshare article (licence checked first) into dest, unpack edffile.zip and stimuli.zip and delete the
    zips. Every file is size- and md5-checked against the API listing."""
    import requests
    art = requests.get(ARTICLE, timeout=60).json()
    check_licence(art)
    os.makedirs(dest, exist_ok=True)
    for f in wanted(art["files"], skip):
        path = sg.download(f["download_url"], os.path.join(dest, f["name"]), f["size"], f["computed_md5"], conns=conns)
        if f["name"].endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for m in z.namelist():
                    if m.endswith("/"):
                        continue
                    z.extract(safe_member(m), dest)
            os.remove(path)
    with open(os.path.join(dest, "article.json"), "w") as fh:
        json.dump({k: art[k] for k in ("title", "doi", "license", "files")}, fh, indent=1)
    return dest



def labels_from_events(text):
    """Class of each trial: trial_type of the rows with value 2 (MI beginning); 1 left hand, 2 right hand."""
    rows = [ln.split("\t") for ln in text.strip().splitlines()]
    head, body = rows[0], rows[1:]
    i_t, i_v = head.index("trial_type"), head.index("value")
    out = []
    for r in body:
        if r[i_v] == "2":
            if int(r[i_t]) not in LABEL:
                raise ValueError(f"trial_type {r[i_t]} is neither 1 (left) nor 2 (right)")
            out.append(LABEL[int(r[i_t])])
    return out


def mi_onsets(marker):
    """Sample of each rising edge of marker code 2 (MI beginning)."""
    m = np.round(np.asarray(marker, float)).astype(int)
    return np.flatnonzero((m == 2) & (np.r_[0, m[:-1]] != 2))


def mi_onsets_grid(marker, period, dur, tol, n):
    """MI-onset sample of each of n trials. Trial k is the window [k * period, (k + 1) * period) (the recording is the 40
    8-s epochs end to end). Its MI marker is the code-2 edge in the window that has a code-3 edge dur +- tol samples
    later (MI lasts 4 s, then the break marker). The instruction phase varies, so the onset is not on a fixed grid, and the
    marker channel carries stray 1 / 2 / 3 edges in 18 of the 50 recordings. No such pair, or two, in a window is an error."""
    if len(marker) < n * period:
        raise ValueError(f"recording shorter than {n} trials of {period} samples")
    e2, e3 = mi_onsets(marker), edges(marker, 3)
    out = []
    for k in range(n):
        c = [int(a) for a in e2[(e2 >= k * period) & (e2 < (k + 1) * period)] if np.any(np.abs(e3 - a - dur) <= tol)]
        if len(c) != 1:
            raise ValueError(f"trial {k}: {len(c)} code-2 markers followed by a code-3 marker {dur} +- {tol} samples later")
        out.append(c[0])
    return np.array(out)


def edges(marker, code):
    m = np.round(np.asarray(marker, float)).astype(int)
    return np.flatnonzero((m == code) & (np.r_[0, m[:-1]] != code))


def cut(x, onsets, n):
    """Epochs x[:, o: o + n] -> (n_epochs, n_ch, n); an epoch past the end of the recording is an error."""
    if max(onsets) + n > x.shape[1]:
        raise ValueError(f"epoch past the end of the recording ({max(onsets) + n} > {x.shape[1]})")
    return np.stack([x[:, o:o + n] for o in onsets])


def eeg_rows(names, std):
    """Rows that are EEG: not EOG / the unnamed marker row; the reference channel (CPz) is exactly flat and is dropped; any
    other flat channel is an error."""
    keep = []
    for i, (c, s) in enumerate(zip(names, std)):
        if c in NOT_EEG or c == REF:
            continue
        if s < 1e-9:
            raise ValueError(f"flat channel {c}")
        keep.append(i)
    return keep


def group_idx(names, group):
    """Indices of the sensorimotor (FC*, C*, CP* sites) or occipital (O*, PO*) channels; FT / TP / T sites are neither."""
    pat = {"sensorimotor": r"^(FC|CP|C)(z|\d+)$", "occipital": r"^(PO|O)(z|\d+)$"}[group]
    return [i for i, c in enumerate(names) if re.match(pat, c) and c != REF]


def edf_path(subj, root=RAW):
    return os.path.join(root, "edffile", f"sub-{subj:02d}", "eeg", f"sub-{subj:02d}_task-motor-imagery_eeg.edf")


def trials(subj, root=RAW, band=None):
    """-> (X [40, n_ch, 1000] float32 µV at 250 Hz, y list of 'left' / 'right', ch_names). band = (lo, hi) Hz zero-phase
    band-pass applied to the continuous recording before epoching (default none)."""
    import mne
    from bench import motor_common as mc
    raw = mne.io.read_raw_edf(edf_path(subj, root), preload=True, verbose="ERROR")
    if abs(raw.info["sfreq"] - FS_IN) > 1e-9:
        raise ValueError(f"sub-{subj:02d}: sampling rate {raw.info['sfreq']}")
    x = raw.get_data() * 1e6
    marker = x[-1]
    rows = eeg_rows(raw.ch_names, x.std(axis=1))
    names = [raw.ch_names[i] for i in rows]
    y = labels_from_events(open(os.path.join(root, "task-motor-imagery_events.tsv")).read())
    on = mi_onsets_grid(marker, int(TRIAL_S * FS_IN), int(MI_S * FS_IN), int(0.05 * FS_IN), len(y))
    sig = x[rows]
    if band is not None:
        from scipy.signal import butter, sosfiltfilt
        sig = sosfiltfilt(butter(4, band, btype="bandpass", fs=FS_IN, output="sos"), sig, axis=-1)
    sig = mc.resample(sig, FS_IN, FS_OUT)
    E = cut(sig, np.round(on * FS_OUT / FS_IN).astype(int), int(MI_S * FS_OUT))
    return E.astype(np.float32), y, names
