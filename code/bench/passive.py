"""N-passive loaders of the common null benchmark (N-passive). Real stimuli, no task intent:
each recording comes back as an engine.Recording carrying its real event stream, and passive_overlay() turns that
stream into the stimulus Overlay the ERP pipelines see (through bench/adapters.py).

Datasets ("kojimaB" = Kojima & Kanoh 2024B, task 2stream):
- erpcore_mmn: EEGLAB .set + .fdt, 1024 Hz, Biosemi (reference-free). Stream = codes 70, 80, 180 (STATUS rows 1 / 4
  are not stimuli). Rare = 70 (deviant), standard = 80, 180 (first 15 standards) excluded.
- ds003061: EEGLAB .set, 256 Hz, Biosemi. Stream = every stimulus slot ("response" is not one). Rare = "noise"
  (uncounted distractor), standard = "standard"; oddball targets (with or without a response), false alarms
  ("*_with_reponse"), "ignore" and "condition 5" excluded.
- kojimaB: BrainVision INT_16, 1000 Hz, right-mastoid reference. Per trial (start codes 201-204), the stream is the
  one that does NOT hold the target. Rare = its deviants 101-104 (same_stream_as_target False), standard = its
  standards 1 / 2; the attended stream (targets 111-114, same-stream non-targets, its standards) is not part of the
  stream. rare_type 1 = low deviant (D1 / D3), 2 = high deviant (D2 / D4).

Contrasts (passive_overlay): labels 1 = the rare no-intent stimulus ("target-like"), 0 = a standard of the same
stream. "oddball" = rare vs standard (every dataset); "equiprobable" (kojimaB only) = the same stimuli, with the
two unattended deviants (meta rare_type 1 vs 2) to be contrasted against each other by VT3-type designs; the
adapters build the pipeline's own classes from rare_type.
Stream bookkeeping per recording: seq = position in the stream INCLUDING excluded stimuli (two contrast stimuli are
consecutive iff their seq differ by 1 and they share a train); train = pause-free train, a new one after an onset gap
> PAUSE_X x the recording's median gap, or at each Kojima trial start.

Signals: units µV, float32, channels x samples, scalp channels only (names matched case-insensitively to MNE's
standard_1005 and spelled as there: "FP1" -> "Fp1"; EOG, EXG, GSR, ECG and the like dropped). Reference: the two
Biosemi sets are reference-free (CMS) and are re-referenced to the common average of their scalp channels; Kojima
keeps its recorded right-mastoid reference. Only signals, sfreq and the prebuilt event indexes
(output/raw/passive/<dataset>/events_index.parquet, code/passive_index.py) are read: never subject_info, meas_date
or participant fields. Files are opened with mne / pandas as data only.

Splits: dev = 2 subjects per dataset, np.random.default_rng(20261024).choice(n, 2, replace=False) over the sorted
subject list; analysis = the rest. freeze_passive_dev_split() writes them to output/bdsp/meta/passive_dev_split.json
(the sibling of engine.DEV_FILE) with engine's refusal rules: never overwrite, and refuse while any dataset is
incomplete (subject or recording count below EXPECTED, or a file missing). Once that file exists, the split is
read from it."""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import engine as en

PASSIVE = os.path.join(en.ROOT, "output", "raw", "passive")
DEV_FILE = os.path.join(en.META, "passive_dev_split.json")

DATASETS = ("erpcore_mmn", "ds003061", "kojimaB")
FOLDER = {"erpcore_mmn": "erpcore_mmn", "ds003061": "ds003061", "kojimaB": "kojima2024b_2stream"}
EXPECTED = {"erpcore_mmn": (40, 40), "ds003061": (13, 39), "kojimaB": (15, 90)}
CONTRASTS = ("oddball", "equiprobable")
SPLITS = ("dev", "analysis", "all")
DEV_SEED, DEV_N = 20261024, 2

ERPCORE_STIM = {70, 80, 180}
KOJIMA_TONES = {1, 2, 101, 102, 103, 104, 111, 112, 113, 114}
TARGET_STREAM = {201: 1, 202: 1, 203: 2, 204: 2}
RARE = {"erpcore_mmn": {70}, "ds003061": {"noise"}, "kojimaB": {101, 102, 103, 104}}
STANDARD = {"erpcore_mmn": {80}, "ds003061": {"standard"}, "kojimaB": {1, 2}}
KOJIMA_RARE_TYPE = {101: 1, 103: 1, 102: 2, 104: 2}
PAUSE_X = 2.0
REFERENCE = {"erpcore_mmn": "average", "ds003061": "average", "kojimaB": "right mastoid (as recorded)"}


def _check(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, not {dataset!r}")



def events(dataset):
    """The dataset's event index (one row per event)."""
    _check(dataset)
    ev = pd.read_parquet(os.path.join(PASSIVE, FOLDER[dataset], "events_index.parquet"))
    ev["subject"] = ev["subject"].astype(str)
    return ev


def _unattended(ev):
    """Kojima: True for tones of the stream that does not hold the current trial's target."""
    s = ev["stream"].to_numpy(dtype=float, na_value=np.nan)
    t = ev["target_code"].map(TARGET_STREAM).to_numpy(dtype=float, na_value=np.nan)
    return ~np.isnan(s) & ~np.isnan(t) & (s != t)


def stream_table(dataset, ev):
    """The stimulus stream per recording, sorted by (subject, run, sample): label 1 = rare no-intent stimulus,
    0 = standard, -1 = excluded; rare_type, seq, train (module docstring)."""
    _check(dataset)
    if dataset == "erpcore_mmn":
        st = ev[ev["code"].isin(ERPCORE_STIM)]
    elif dataset == "ds003061":
        st = ev[ev["role"] != "not_stimulus"]
    else:
        st = ev[ev["code"].isin(KOJIMA_TONES).to_numpy() & _unattended(ev)]
    st = st.sort_values(["subject", "run", "sample"]).reset_index(drop=True)
    rare, std = st["code"].isin(RARE[dataset]).to_numpy(), st["code"].isin(STANDARD[dataset]).to_numpy()
    st["label"] = np.where(rare, 1, np.where(std, 0, -1))
    if dataset == "kojimaB":
        st["rare_type"] = np.where(rare, st["code"].map(KOJIMA_RARE_TYPE).fillna(0).astype(int), 0)
    else:
        st["rare_type"] = rare.astype(int)
    st["onset_s"] = st["sample"] / st["sfreq"]
    st["seq"] = st.groupby(["subject", "run"]).cumcount()
    train = np.zeros(len(st), int)
    for _, idx in st.groupby(["subject", "run"]).indices.items():
        on = st["onset_s"].to_numpy()[idx]
        gap = np.diff(on)
        new = gap > PAUSE_X * np.median(gap) if len(gap) else np.zeros(0, bool)
        if dataset == "kojimaB":
            new |= np.diff(st["trial"].to_numpy()[idx]) != 0
        train[idx] = np.r_[0, np.cumsum(new)]
    st["train"] = train
    keep = ["subject", "run", "sample", "onset_s", "sfreq", "code", "role", "label", "rare_type", "seq", "train"]
    return st[keep + (["trial"] if dataset == "kojimaB" else [])]


def passive_overlay(rec, contrast="oddball"):
    """Overlay(kind="stim") of the recording's contrast stimuli: labels 1 = rare no-intent stimulus, 0 = standard
    (module docstring). onsets_s in s from the first sample of rec.x."""
    if contrast not in CONTRASTS:
        raise ValueError(f"contrast must be one of {CONTRASTS}, not {contrast!r}")
    st = rec.meta["events"]
    con = st[st["label"] >= 0]
    lab = con["label"].to_numpy(int)
    rt = con["rare_type"].to_numpy(int)
    if contrast == "equiprobable" and len(np.unique(rt[lab == 1])) < 2:
        raise ValueError(f"{rec.meta['dataset']}: no two equiprobable rare stimuli (equiprobable is kojimaB only)")
    on, tr = st["onset_s"].to_numpy(float), st["train"].to_numpy()
    same = tr[1:] == tr[:-1]
    soa = float(np.median(np.diff(on)[same])) if same.any() else float("nan")
    excl = st[st["label"] < 0]["role"].value_counts()
    train = con["train"].to_numpy(int)
    n = len(con)
    return en.Overlay(onsets_s=con["onset_s"].to_numpy(float), labels=lab, kind="stim",
                      meta={"dataset": rec.meta["dataset"], "contrast": contrast, "code": con["code"].to_numpy(),
                            "rare_type": rt, "seq": con["seq"].to_numpy(int), "train": train,
                            "trial": np.arange(n), "block": train.copy(), "run": np.zeros(n, int), "soa_s": soa,
                            "excluded": {str(k): int(v) for k, v in excl.items()}})



def subjects(dataset):
    _check(dataset)
    ev = pd.read_parquet(os.path.join(PASSIVE, FOLDER[dataset], "events_index.parquet"), columns=["subject"])
    return sorted(ev["subject"].astype(str).unique())


def _runs(dataset):
    """{subject: [run, ...]} from the event index."""
    ev = pd.read_parquet(os.path.join(PASSIVE, FOLDER[dataset], "events_index.parquet"), columns=["subject", "run"])
    ev["subject"] = ev["subject"].astype(str)
    return {s: sorted(int(r) for r in g["run"].unique()) for s, g in ev.groupby("subject")}


def recording_files(dataset, subject, run):
    """The signal file of one recording and every companion file it needs."""
    d = os.path.join(PASSIVE, FOLDER[dataset], f"sub-{subject}", "eeg")
    if dataset == "erpcore_mmn":
        b = os.path.join(d, f"sub-{subject}_task-MMN_eeg")
        return [b + ".set", b + ".fdt"]
    if dataset == "ds003061":
        return [os.path.join(d, f"sub-{subject}_task-P300_run-{run}_eeg.set")]
    b = os.path.join(d, f"sub-{subject}_task-2stream_run-{run}_eeg")
    return [b + ".vhdr", b + ".vmrk", b + ".eeg"]


def final(dataset):
    """True when the subject list and every recording are complete (EXPECTED counts, all files present)."""
    runs = _runs(dataset)
    n_sub, n_rec = EXPECTED[dataset]
    if len(runs) != n_sub or sum(len(r) for r in runs.values()) != n_rec:
        return False
    return all(os.path.exists(f) for s, rr in runs.items() for r in rr for f in recording_files(dataset, s, r))


def passive_split(dataset, dev_file=None):
    """{"dev", "analysis": subject lists, "frozen"}: the frozen file's members when dev_file (default DEV_FILE)
    exists, else the seeded draw over the sorted subject list."""
    _check(dataset)
    dev_file = DEV_FILE if dev_file is None else dev_file
    subs = subjects(dataset)
    fz = en._frozen(dev_file)
    if fz is None:
        idx = np.random.default_rng(DEV_SEED).choice(len(subs), DEV_N, replace=False)
        dev = sorted(subs[i] for i in idx)
    else:
        dev = [str(s) for s in fz["datasets"][dataset]]
    return {"dev": dev, "analysis": [s for s in subs if s not in dev], "frozen": fz is not None}


def freeze_passive_dev_split(path=None):
    """Write the passive dev members to path (default DEV_FILE). Refuses if the file exists or while any dataset is
    not final. Returns what was written."""
    path = path or DEV_FILE
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: the passive dev split is already frozen")
    bad = [d for d in DATASETS if not final(d)]
    if bad:
        raise RuntimeError(f"passive dev split not final: {bad} incomplete (subjects / recordings below EXPECTED or "
                           "files missing)")
    out = {"frozen_at": datetime.now().isoformat(timespec="seconds"),
           "rule": "per dataset: np.random.default_rng(20261024).choice(n, 2, replace=False) over the sorted subject "
                   "list",
           "datasets": {d: passive_split(d, "")["dev"] for d in DATASETS},
           "n_subjects": {d: len(subjects(d)) for d in DATASETS}}
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    return out



_STD = None


def normalise_names(ch_names):
    """[(index, canonical name)] of the scalp channels: names found case-insensitively in MNE's standard_1005 montage,
    spelled as there. Everything else (EOG, EXG, GSR, ECG, ...) is dropped."""
    global _STD
    if _STD is None:
        import mne
        _STD = {c.lower(): c for c in mne.channels.make_standard_montage("standard_1005").ch_names}
    return [(i, _STD[c.strip().lower()]) for i, c in enumerate(ch_names) if c.strip().lower() in _STD]


def _patch_eeglab_chaninfo():
    """The ds003522_io / passive_index guard: some EEGLAB files store chaninfo without 'nodatchans', which mne
    indexes (it only carries fiducial positions)."""
    import mne.io.eeglab.eeglab as el
    if getattr(el._get_montage_information, "_nodatchans_patch", False):
        return
    orig = el._get_montage_information

    def safe(eeg, *a, **k):
        ci = eeg.get("chaninfo", None) if hasattr(eeg, "get") else None
        if ci is not None and (not isinstance(ci, dict) or "nodatchans" not in ci):
            eeg["chaninfo"] = {**(ci if isinstance(ci, dict) else {}), "nodatchans": {}}
        return orig(eeg, *a, **k)
    safe._nodatchans_patch = True
    el._get_montage_information = safe


def _read_signal(dataset, path):
    """(x float32 µV scalp channels x samples, fs, names). Reads only the signal, sfreq and channel names."""
    import mne
    if path.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    else:
        _patch_eeglab_chaninfo()
        raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    keep = normalise_names(raw.ch_names)
    x = raw.get_data(picks=[i for i, _ in keep])
    del raw
    x *= 1e6
    x = x.astype(np.float32)
    if REFERENCE[dataset] == "average":
        x -= x.mean(0, dtype=np.float64).astype(np.float32)[None]
    return x, fs, [n for _, n in keep]


def _items(dataset, split):
    _check(dataset)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    sp = passive_split(dataset)
    subs = sp["dev"] if split == "dev" else sp["analysis"] if split == "analysis" else sp["dev"] + sp["analysis"]
    runs = _runs(dataset)
    one = dataset == "erpcore_mmn"
    out = []
    for s in sorted(subs):
        for r in runs[s]:
            out.append({"dataset": dataset, "subject": s, "run": r, "path": recording_files(dataset, s, r)[0],
                        "uid": f"{dataset}_sub-{s}" + ("" if one else f"_run-{r}"), "patient": f"{dataset}_sub-{s}",
                        "dev": s in sp["dev"], "split_frozen": sp["frozen"]})
    return out


def _load(it, st):
    d = it["dataset"]
    rows = st[(st["subject"] == it["subject"]) & (st["run"] == it["run"])].reset_index(drop=True)
    x, fs, names = _read_signal(d, it["path"])
    if not np.allclose(rows["sfreq"], fs) or int(rows["sample"].max()) >= x.shape[1]:
        raise ValueError(f"{it['uid']}: event index does not fit the signal (sfreq or sample range)")
    meta = {"dataset": d, "subject": it["subject"], "run": it["run"], "file": os.path.relpath(it["path"], en.ROOT),
            "reference": REFERENCE[d], "events": rows, "dev": it["dev"], "split_frozen": it["split_frozen"]}
    return en.Recording(source=d, uid=it["uid"], patient=it["patient"], x=x, fs=fs, names=names, meta=meta)


def load_passive(dataset, split):
    """Iterator of the split's Recordings (one per recording, loaded one at a time) with the real event stream in
    meta["events"]; build the overlay with passive_overlay(rec, contrast)."""
    items = _items(dataset, split)
    st = stream_table(dataset, events(dataset))
    return (_load(it, st) for it in items)
