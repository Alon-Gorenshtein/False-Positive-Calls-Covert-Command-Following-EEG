"""Null-overlay engine of the common null benchmark (Engine section). Loads no-command EEG one recording at a time, gives clean spans, Morgoth strata and the artifact
index, and injects simulated responders. Pipelines (bench/pipe_*.py) build Overlays; the caller applies response
probability through resp_mask.

Data (units uV, float32, channels x samples):
- bdsp: x19/names19 of output/bdsp/sig/<stem>.npz (128 Hz), Morgoth labels from output/bdsp/morgoth (108.row_labels).
  One recording per patient (sample_500.csv). Plus the extra BDSP listing (bench_extra_500.csv; output/bdsp_extra/
  {sig,morgoth}, same format), analysis only. Patients must be disjoint across the design sample, the extras and
  train_600.csv (and unique within each): listing the extras refuses otherwise (check_disjoint).
- pdoc: the 41 Mendeley PSGs (42 files minus one byte-identical duplicate, moved to doc_psg_excluded/) (output/raw/doc_psg/*.edf). EEG = the bare 10-20 scalp channels present (F3 F4 C3 C4 O1 O2
  in every file), read with mne include= so they stay at their native 128 Hz (reading all channels would upsample them
  to the 256 Hz file rate). Excluded: referenced duplicates ("F3:A2"), A1/A2, EOG, EMG, ECG. V -> uV.
- icare: output/raw/icare_null/<pid>_<hour>.npz, 8-min windows (19 channels, 1-45 Hz, 100 Hz), reordered to names19.

Splits:
- dev: the last DEV_N (20) eligible BDSP recordings in the seeded Columbia order (113.split_dev of 113.eligible_list,
  computed by scanning that order from the end), plus 3 pDoC PSGs: np.random.default_rng(20261023).choice(n, 3,
  replace=False) over the sorted .edf names ("._" files excluded). I-CARE has no dev recordings.
- analysis: every other usable recording (BDSP: x19 + Morgoth present, whatever its clean length): the design-sample
  analysis recordings in rank order, then the extras in csv order. Extras are listed once both their npz (with x19)
  and Morgoth file exist; bdsp_extras()["complete"] is True when all are present, or when the last pull run logged
  BDSP_EXTRA_PULL_DONE and every missing one has a FAIL line in it (the analysis listing warns until then). The
  extras have no dev members.
- The BDSP dev split is "final" only when every recording ranked after the first dev recording is usable; until then
  a later arrival can change it (load_recordings warns).
- freeze_dev_split() writes the BDSP and pDoC dev members to output/bdsp/meta/dev_split.json once the BDSP split is
  final (it refuses while pending, and never overwrites). Once that file exists, load_recordings takes dev membership
  from it, not from a fresh scan, so later data cannot shift it; analysis = every other usable recording.

Artifact index (fixed): over the 1-s segments of the recording's whole-second grid that lie inside the window,
the share with, on ANY channel, 30-45 Hz power (Hann periodogram) > 4 x that channel's recording median (median over
non-padding seconds), SD < 0.1 uV (flat) or |x| > 500 uV (saturated). Labels "<10%" | "10-30%" | ">30%" (0.10 and
0.30 fall in "10-30%").

Responders (inject):
- erd / ers, source level. resp_hand names the hand that moves (or is imagined moving); its template is
  CONTRALATERAL: a Gaussian topography over the MNE standard_1020 positions centred on C3 for the right hand and on
  C4 for the left hand, a_c = exp(-d_c^2 / (2 sigma^2)), sigma = SIGMA_M = mean distance from C3 to its 4 nearest
  10-20 neighbours (P3 F3 Cz T3; 7.25 cm). a = 1 at the centre; channels without a standard position get 0.
  Lead field A = [a] (or [a_left, a_right] for "both"), W = pinv(A) (w = a / ||a||^2 for one hand), s_band = W x_band,
  and x -/+ effect * r(t) * A s_band. x_band: 8-30 Hz, zero-phase FFT mask with raised-cosine edges inside the band
  (gain 1 on 9-29 Hz, 0 outside 8-30 Hz), computed on the response span padded by 3 s (outer 1 s tapered). r(t):
  raised-cosine ramps of 0.5 s at both ends. Response span: [onset + delay + U(0, 1) s, onset + delay + dur] (the end
  is anchored to the commanded period, so jitter shortens it). effect = delta, the removed (ERD) or added (ERS) share
  of the source's band component (ERD: 0 <= delta <= 1).
  overlay.meta: "resp_dur_s" (required; scalar or per onset), "resp_hand" (required; the moving hand, "left" |
  "right" | "both" | "none", a per-onset sequence, or a {label code: hand} dict), "resp_delay_s" (optional, default
  0).
- erp: effect uV x exp(-(t - 0.45 - j)^2 / (2 * 0.075^2)), j ~ U(-0.1, 0.1) s, over 0-1 s after onset (the peak falls
  at 350-550 ms, inside the 300-600 ms P3 range). Topography: the same Gaussian centred midway between Cz and Pz,
  scaled to 1 at Cz and Pz.
- No response where stratum(rec, response period) is burst_suppression (BDSP only; other sources are "na").
- Jitters are drawn for every onset before the mask is applied, so one rng gives the same jitters for any resp_mask.
- onsets_s are seconds from the start of the recording. x is never modified in place."""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import warnings
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import periodogram

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import importlib

bn = importlib.import_module("108_bdsp_null")
cn = importlib.import_module("113_columbia_null")

SIG = os.path.join(ROOT, "output", "bdsp", "sig")
MOR = os.path.join(ROOT, "output", "bdsp", "morgoth")
META = os.path.join(ROOT, "output", "bdsp", "meta")
PDOC_RAW = os.path.join(ROOT, "output", "raw", "doc_psg")
ICARE = os.path.join(ROOT, "output", "raw", "icare_null")
DEV_FILE = os.path.join(META, "dev_split.json")
EXTRA_SIG = os.path.join(ROOT, "output", "bdsp_extra", "sig")
EXTRA_MOR = os.path.join(ROOT, "output", "bdsp_extra", "morgoth")
EXTRA_LOG = os.path.join(ROOT, "output", "intermediate", "_bdsp_extra_pull.log")
EXTRA_TAG = "BDSP_EXTRA_PULL"
DESIGN_LIST, EXTRA_LIST, TRAIN_LIST = "sample_500.csv", "bench_extra_500.csv", "train_600.csv"

SOURCES = ("bdsp", "pdoc", "icare")
SPLITS = ("dev", "analysis", "all")
NAMES19 = ("Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6",
           "O2")
DEV_N = cn.DEV_N
DEV_PDOC_SEED, DEV_PDOC_N = 20261023, 3
MORGOTH_WIN_S = 10

HF_BAND, HF_RATIO, FLAT_UV, SAT_UV = (30.0, 45.0), 4.0, 0.1, 500.0
ART_CHUNK_S = 1800

BAND, BAND_EDGE = (8.0, 30.0), 1.0
RAMP_S, ERD_JITTER_S, PAD_S, TAPER_S = 0.5, 1.0, 3.0, 1.0
ERP_LEN_S, ERP_PEAK_S, ERP_SD_S, ERP_JITTER_S = 1.0, 0.45, 0.075, 0.1
HANDS = ("left", "right", "both", "none")


_TOKENS = itertools.count(1)


@dataclass(eq=False)
class Recording:
    source: str
    uid: str
    patient: str
    x: np.ndarray
    fs: float
    names: list
    meta: dict = field(default_factory=dict)
    token: int = field(default_factory=lambda: next(_TOKENS), init=False, compare=False, repr=False)


@dataclass(eq=False)
class Overlay:
    onsets_s: np.ndarray
    labels: np.ndarray
    kind: str
    meta: dict = field(default_factory=dict)



def _usable(stem, sig, mor):
    """True if the design npz carries x19 and its Morgoth file exists (a file still being written counts as absent)."""
    f = os.path.join(sig, f"{stem}.npz")
    if not (os.path.exists(f) and os.path.exists(os.path.join(mor, f"{stem}.parquet"))):
        return False
    try:
        return "x19" in np.load(f).files
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _eligible(stem, sig):
    z = np.load(os.path.join(sig, f"{stem}.npz"))
    return len(bn.eligible_starts(np.asarray(z["x19"], np.float32), float(z["fs"]), run_s=cn.SPAN_S)) > 0


def _frozen(dev_file):
    if dev_file and os.path.exists(dev_file):
        with open(dev_file) as fh:
            return json.load(fh)
    return None


def bdsp_split(sig=None, mor=None, meta=None, dev_file=None):
    """{"dev", "analysis": [(rank, sample index, stem)] in rank order, "patient": {stem: id}, "final", "pending",
    "frozen"}. dev = 113.split_dev(113.eligible_list())[1], found by scanning the seeded order from the end, or the
    members frozen in dev_file when that file exists."""
    sig, mor, meta = sig or SIG, mor or MOR, meta or META
    s = pd.read_csv(os.path.join(meta, "sample_500.csv"))
    order = cn.sample_order(len(s))
    fz = _frozen(dev_file)
    fixed = None if fz is None else [(int(r), int(i), str(t)) for r, i, t in fz["bdsp"]]
    skip = set() if fixed is None else {t for _, _, t in fixed}
    dev, rest, missing = [], [], []
    for rank in range(len(order) - 1, -1, -1):
        i = int(order[rank])
        stem = s.file[i].replace(".mat", "")
        if stem in skip:
            continue
        if not _usable(stem, sig, mor):
            missing.append(rank)
        elif fixed is None and len(dev) < DEV_N and _eligible(stem, sig):
            dev.append((rank, i, stem))
        else:
            rest.append((rank, i, stem))
    dev, rest = (dev[::-1] if fixed is None else fixed), rest[::-1]
    first = dev[0][0] if dev else -1
    pending = 0 if fixed is not None else sum(r > first for r in missing)
    return {"dev": dev, "analysis": rest, "patient": dict(zip(s.file.str.replace(".mat", "", regex=False),
                                                              s.patient_id.astype(str))),
            "final": bool(fixed is not None or (len(dev) == DEV_N and pending == 0)), "pending": int(pending),
            "frozen": fixed is not None}


def check_disjoint(lists):
    """Refuse (RuntimeError) if a patient is listed twice in one BDSP list or appears in two lists. lists: {name: csv}."""
    owner, clash = {}, []
    for name, path in lists.items():
        ids = pd.read_csv(path).patient_id.astype(str)
        twice = sorted(set(ids[ids.duplicated()]))
        if twice:
            raise RuntimeError(f"BDSP {name} list: patient(s) listed twice: {twice[:5]}")
        for p in ids:
            if p in owner:
                clash.append(f"{p} ({owner[p]} + {name})")
            owner[p] = name
    if clash:
        raise RuntimeError(f"BDSP lists share {len(clash)} patient(s): {clash[:5]}")


def _pull_log(log, tag):
    """(finished, failed stems) of the last run in a bdsp_pull log: finished if '<tag>_DONE' follows its start line."""
    if not log or not os.path.exists(log):
        return False, set()
    with open(log) as fh:
        lines = fh.read().splitlines()
    last = lines[max((k for k, l in enumerate(lines) if " sampled, " in l), default=0):]
    failed = {m.group(1).replace(".mat", "") for l in last for m in [re.search(r"\] FAIL (\S+):", l)] if m}
    return any(f"{tag}_DONE" in l for l in last), failed


def bdsp_extras(meta=None, sig=None, mor=None, log=None):
    """The extra BDSP listing (analysis only): {"analysis": [(csv row, stem)] of the usable ones in csv order,
    "patient": {stem: id}, "n_listed", "n_present", "missing", "failed", "complete"}. Checks patient disjointness
    against the design sample and train_600.csv first."""
    meta, sig, mor, log = meta or META, sig or EXTRA_SIG, mor or EXTRA_MOR, log or EXTRA_LOG
    path = os.path.join(meta, EXTRA_LIST)
    check_disjoint({"design": os.path.join(meta, DESIGN_LIST), "extra": path, "train": os.path.join(meta, TRAIN_LIST)})
    s = pd.read_csv(path)
    stems = s.file.astype(str).str.replace(".mat", "", regex=False).tolist()
    present, missing = [], []
    for j, stem in enumerate(stems):
        if _usable(stem, sig, mor):
            present.append((j, stem))
        else:
            missing.append(stem)
    finished, failed = _pull_log(log, EXTRA_TAG)
    return {"analysis": present, "patient": dict(zip(stems, s.patient_id.astype(str))), "n_listed": len(stems),
            "n_present": len(present), "missing": missing, "failed": sorted(failed & set(missing)),
            "complete": bool(not missing or (finished and set(missing) <= failed))}


def _listing(root, ext):
    return sorted(f for f in os.listdir(root) if f.endswith(ext) and not f.startswith("._"))


def pdoc_split(raw=None, dev_file=None):
    files = _listing(raw or PDOC_RAW, ".edf")
    fz = _frozen(dev_file)
    if fz is None:
        idx = np.random.default_rng(DEV_PDOC_SEED).choice(len(files), DEV_PDOC_N, replace=False)
        dev = sorted(files[i] for i in idx)
    else:
        dev = [str(f) for f in fz["pdoc"]]
    return {"dev": dev, "analysis": [f for f in files if f not in dev], "frozen": fz is not None}


def freeze_dev_split(path=None, sig=None, mor=None, meta=None, raw=None):
    """Write the BDSP and pDoC dev members to path (default DEV_FILE). Refuses while the BDSP dev split is not final
    (pull or x19 backfill pending) and if the file already exists. Returns what was written."""
    path = path or DEV_FILE
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: the dev split is already frozen")
    sp = bdsp_split(sig, mor, meta)
    if not sp["final"]:
        raise RuntimeError(f"BDSP dev split is not final ({len(sp['dev'])} dev, {sp['pending']} recording(s) ranked "
                           "after the first dev recording not usable yet): freeze after the pull and x19 backfill")
    pdoc = pdoc_split(raw)
    out = {"frozen_at": datetime.now().isoformat(timespec="seconds"),
           "rule": "BDSP: last 20 eligible recordings in the seeded Columbia order (113.split_dev); pDoC: "
                   "np.random.default_rng(20261023).choice(n, 3, replace=False) over the sorted .edf names",
           "bdsp": [[int(r), int(i), str(t)] for r, i, t in sp["dev"]],
           "pdoc": pdoc["dev"],
           "bdsp_usable": len(sp["dev"]) + len(sp["analysis"]), "pdoc_files": len(pdoc["dev"]) + len(pdoc["analysis"])}
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    return out


def icare_split(root=ICARE):
    return {"dev": [], "analysis": _listing(root, ".npz")}


def _check(source, split):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")


def _pick(sp, split):
    return sp["dev"] if split == "dev" else sp["analysis"] if split == "analysis" else sp["dev"] + sp["analysis"]


def _items(source, split):
    """Descriptors (source, path, uid, patient, meta) of one split, without loading any signal."""
    _check(source, split)
    out = []
    if source == "bdsp":
        sp = bdsp_split(SIG, MOR, META, DEV_FILE)
        if not sp["final"]:
            warnings.warn(f"BDSP dev split is provisional: {sp['pending']} recording(s) ranked after the first dev "
                          "recording are not usable yet (pull or x19 backfill pending)")
        dev = {t for _, _, t in sp["dev"]}
        for rank, i, stem in sorted(_pick(sp, split)):
            out.append(dict(source=source, path=os.path.join(SIG, f"{stem}.npz"),
                            morgoth=os.path.join(MOR, f"{stem}.parquet"), uid=stem, patient=sp["patient"][stem],
                            meta={"rank": rank, "sample_idx": i, "dev": stem in dev, "extra": False,
                                  "split_final": sp["final"], "split_frozen": sp["frozen"]}))
        if split != "dev":
            ex = bdsp_extras()
            if not ex["complete"]:
                warnings.warn(f"BDSP extra listing incomplete: {ex['n_present']}/{ex['n_listed']} present (pull "
                              "running or unfinished)")
            for j, stem in ex["analysis"]:
                out.append(dict(source=source, path=os.path.join(EXTRA_SIG, f"{stem}.npz"),
                                morgoth=os.path.join(EXTRA_MOR, f"{stem}.parquet"), uid=stem,
                                patient=ex["patient"][stem],
                                meta={"extra_row": j, "dev": False, "extra": True, "extras_complete": ex["complete"],
                                      "split_final": sp["final"], "split_frozen": sp["frozen"]}))
        uids = [it["uid"] for it in out]
        if len(set(uids)) != len(uids):
            raise RuntimeError("a BDSP uid is listed twice across the design and extra listings")
    elif source == "pdoc":
        sp = pdoc_split(PDOC_RAW, DEV_FILE)
        for f in _pick(sp, split):
            m = re.match(r"Patient[ _](\d+)_(.+)\.edf$", f)
            out.append(dict(source=source, path=os.path.join(PDOC_RAW, f), uid=f[:-4],
                            patient=f"Patient_{m.group(1)}" if m else f[:-4],
                            meta={"file": f, "diagnosis": m.group(2) if m else "", "dev": f in sp["dev"],
                                  "split_frozen": sp["frozen"]}))
    else:
        for f in _pick(icare_split(ICARE), split):
            out.append(dict(source=source, path=os.path.join(ICARE, f), uid=f[:-4], patient=f.split("_")[0],
                            meta={"file": f, "dev": False}))
    return out


def _load(it):
    src, path = it["source"], it["path"]
    meta = dict(it["meta"])
    if src == "bdsp":
        z = np.load(path)
        x, fs, names = np.asarray(z["x19"], np.float32), float(z["fs"]), [str(n) for n in z["names19"]]
        m = pd.read_parquet(it.get("morgoth") or os.path.join(MOR, f"{it['uid']}.parquet"))
        meta |= {"morgoth_t": m.t_s.to_numpy(), "morgoth_lab": bn.row_labels(m)}
    elif src == "pdoc":
        import mne
        hdr = mne.io.read_raw_edf(path, preload=False, verbose="ERROR").ch_names
        names = [c for c in NAMES19 if c in hdr]
        raw = mne.io.read_raw_edf(path, include=names, preload=True, verbose="ERROR")
        x, fs = (raw.get_data(picks=names) * 1e6).astype(np.float32), float(raw.info["sfreq"])
        del raw
    else:
        z = np.load(path)
        have = [str(n) for n in z["names"]]
        names = [c for c in NAMES19 if c in have]
        x = np.asarray(z["x"], np.float32)[[have.index(c) for c in names]]
        fs = float(z["sf"])
        meta |= {"pid": str(z["pid"]), "hour": int(z["hour"]), "seg": int(z["seg"]), "util": float(z["util"])}
    return Recording(source=src, uid=it["uid"], patient=it["patient"], x=x, fs=fs, names=names, meta=meta)


def load_recordings(source, split):
    """Iterator of Recordings of one source and split, loaded one at a time."""
    items = _items(source, split)
    return (_load(it) for it in items)



def clean_spans(rec, min_s):
    """Whole-second starts s whose [s, s + min_s) holds no padding and stays 30 s from any (108.eligible_starts)."""
    return bn.eligible_starts(rec.x, rec.fs, run_s=int(np.ceil(min_s)))


def stratum(rec, start_s, dur_s):
    """(Morgoth stratum, transitioning) by 108.run_stratum over the 10-s Morgoth windows starting in
    [floor(start), start + dur - 10] (at least the window at floor(start)). Non-BDSP: ("na", False)."""
    if rec.source != "bdsp":
        return "na", False
    t, lab = rec.meta["morgoth_t"], rec.meta["morgoth_lab"]
    lo = np.floor(start_s)
    sel = (t >= lo) & (t <= max(lo, start_s + dur_s - MORGOTH_WIN_S))
    if not sel.any():
        return "unlabelled", False
    st, tr = bn.run_stratum(lab[sel])
    return str(st), bool(tr)


def artifact_label(share):
    return "<10%" if share < 0.10 else "10-30%" if share <= 0.30 else ">30%"


def _artifact_seconds(rec):
    """Per whole second: True if any channel is flat, saturated or has 30-45 Hz power > 4 x its recording median.
    Cached in rec.meta["_artifact_s"]."""
    if "_artifact_s" in rec.meta:
        return rec.meta["_artifact_s"]
    fs = int(round(rec.fs))
    nc, n = rec.x.shape[0], rec.x.shape[1] // fs
    hf = np.empty((nc, n))
    bad = np.zeros(n, bool)
    for a in range(0, n, ART_CHUNK_S):
        b = min(n, a + ART_CHUNK_S)
        seg = np.asarray(rec.x[:, a * fs: b * fs], float).reshape(nc, b - a, fs)
        bad[a:b] = (seg.std(2) < FLAT_UV).any(0) | (np.abs(seg) > SAT_UV).any(2).any(0)
        f, p = periodogram(seg, fs=fs, window="hann", axis=-1)
        hf[:, a:b] = p[..., (f >= HF_BAND[0]) & (f < HF_BAND[1])].sum(-1) * (f[1] - f[0])
    keep = ~bn.padding_seconds(rec.x, fs)
    if keep.any():
        med = np.median(hf[:, keep], axis=1)
        bad |= (hf > HF_RATIO * med[:, None]).any(0)
    rec.meta["_artifact_s"] = bad
    return bad


def artifact_index(rec, start_s, dur_s):
    """(share of artifact seconds in the window, stratum label)."""
    bad = _artifact_seconds(rec)
    k0, k1 = max(0, int(np.ceil(start_s - 1e-9))), min(len(bad), int(np.floor(start_s + dur_s + 1e-9)))
    if k1 <= k0:
        return float("nan"), "na"
    share = float(bad[k0:k1].mean())
    return share, artifact_label(share)



_POS = None


def _positions(names):
    """Standard 10-20 positions (m) of the channels; NaN where the name has none."""
    global _POS
    if _POS is None:
        import mne
        _POS = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    return np.array([_POS[c] if c in _POS else (np.nan,) * 3 for c in names], float)


def _sigma():
    P = _positions(NAMES19)
    d = np.linalg.norm(P - P[NAMES19.index("C3")], axis=1)
    return float(np.sort(d)[1:5].mean())


SIGMA_M = _sigma()


def _gauss(names, centre):
    d = np.linalg.norm(_positions(names) - centre, axis=1)
    return np.nan_to_num(np.exp(-d ** 2 / (2 * SIGMA_M ** 2)), nan=0.0)


def hand_topography(names, hand):
    """Gaussian topography over the hand area contralateral to the moving hand: C3 for "right", C4 for "left"; 1 at
    the centre."""
    if hand not in ("left", "right"):
        raise ValueError(f"hand must be left or right, not {hand!r}")
    return _gauss(names, _positions(["C3" if hand == "right" else "C4"])[0])


def erp_topography(names):
    """Gaussian topography centred midway between Cz and Pz; 1 at Cz and Pz."""
    mid = _positions(["Cz", "Pz"]).mean(0)
    return _gauss(names, mid) / _gauss(["Cz"], mid)[0]



def _per_onset(v, n, what):
    v = np.broadcast_to(np.asarray(v, float), (n,)) if np.ndim(v) == 0 else np.asarray(v, float)
    if v.shape != (n,):
        raise ValueError(f"{what}: need a scalar or one value per onset")
    return v


def _hands(spec, labels, n):
    if isinstance(spec, str):
        h = [spec] * n
    elif isinstance(spec, dict):
        h = [spec[int(l)] for l in labels]
    else:
        h = list(spec)
    if len(h) != n or any(v not in HANDS for v in h):
        raise ValueError(f"resp_hand: one of {HANDS} per onset")
    return h


def _band(x, fs):
    """Zero-phase 8-30 Hz FFT mask with raised-cosine edges inside the band (rows = channels)."""
    n = x.shape[1]
    f = np.fft.rfftfreq(n, 1 / fs)
    lo, hi = BAND
    g = ((f >= lo + BAND_EDGE) & (f <= hi - BAND_EDGE)).astype(float)
    up, dn = (f > lo) & (f < lo + BAND_EDGE), (f > hi - BAND_EDGE) & (f < hi)
    g[up] = 0.5 * (1 - np.cos(np.pi * (f[up] - lo) / BAND_EDGE))
    g[dn] = 0.5 * (1 + np.cos(np.pi * (f[dn] - hi + BAND_EDGE) / BAND_EDGE))
    return np.fft.irfft(np.fft.rfft(x, axis=1) * g, n=n, axis=1)


def _ramp(k):
    return 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))


def _envelope(n, fs, ramp_s):
    r = np.ones(n)
    k = min(int(round(ramp_s * fs)), n // 2)
    if k > 0:
        r[:k] = _ramp(k)
        r[n - k:] = _ramp(k)[::-1]
    return r


def _suppressed(rec, start_s, dur_s):
    return stratum(rec, start_s, dur_s)[0] == "burst_suppression"


def inject(rec, overlay, resp_mask, model, effect, rng, *, ramp_s=RAMP_S):
    """A copy of rec.x (float32) with simulated responses at the masked-in onsets (module docstring)."""
    if model not in ("erd", "ers", "erp"):
        raise ValueError(f"model must be erd, ers or erp, not {model!r}")
    on = np.asarray(overlay.onsets_s, float)
    mask = np.asarray(resp_mask, bool)
    if mask.shape != on.shape:
        raise ValueError("resp_mask needs one entry per onset")
    x = rec.x
    out = np.array(x, dtype=np.float32, copy=True)
    fs, N = float(rec.fs), x.shape[1]
    if model == "erp":
        jit = rng.uniform(-ERP_JITTER_S, ERP_JITTER_S, len(on))
        a = erp_topography(rec.names)
        t = np.arange(int(round(ERP_LEN_S * fs))) / fs
        for o, m, j in zip(on, mask, jit):
            if not m or _suppressed(rec, o, ERP_LEN_S):
                continue
            i = int(round(o * fs))
            k = min(len(t), N - i)
            if k <= 0:
                continue
            w = effect * np.exp(-(t[:k] - ERP_PEAK_S - j) ** 2 / (2 * ERP_SD_S ** 2))
            out[:, i: i + k] += (a[:, None] * w[None]).astype(np.float32)
        return out
    if model == "erd" and not 0 <= effect <= 1:
        raise ValueError("ERD effect is a share of the source band component: 0 <= effect <= 1")
    dur = _per_onset(overlay.meta["resp_dur_s"], len(on), "resp_dur_s")
    hands = _hands(overlay.meta["resp_hand"], overlay.labels, len(on))
    delay = _per_onset(overlay.meta.get("resp_delay_s", 0.0), len(on), "resp_delay_s")
    jit = rng.uniform(0.0, ERD_JITTER_S, len(on))
    sign = -1.0 if model == "erd" else 1.0
    lead = {}
    for h in set(hands) - {"none"}:
        A = np.column_stack([hand_topography(rec.names, s) for s in (("left", "right") if h == "both" else (h,))])
        lead[h] = (A, np.linalg.pinv(A))
    pad, taper = int(round(PAD_S * fs)), int(round(TAPER_S * fs))
    for o, m, h, d, dl, j in zip(on, mask, hands, dur, delay, jit):
        if not m or h == "none" or _suppressed(rec, o + dl, d):
            continue
        i0, i1 = max(0, int(round((o + dl + j) * fs))), min(N, int(round((o + dl + d) * fs)))
        if i1 - i0 < 2:
            continue
        a0, b0 = max(0, i0 - pad), min(N, i1 + pad)
        seg = np.asarray(x[:, a0:b0], float)
        seg = seg - seg.mean(1, keepdims=True)
        tl, tr = min(taper, i0 - a0), min(taper, b0 - i1)
        if tl:
            seg[:, :tl] *= _ramp(tl)
        if tr:
            seg[:, seg.shape[1] - tr:] *= _ramp(tr)[::-1]
        A, W = lead[h]
        s = np.einsum("kc,ct->kt", W, _band(seg, fs)[:, i0 - a0: i1 - a0])
        out[:, i0:i1] += (sign * effect * _envelope(i1 - i0, fs, ramp_s)[None]
                          * np.einsum("ck,kt->ct", A, s)).astype(np.float32)
    return out



def patient_windows(recs, run_s, k=None, rng=None):
    """Non-overlapping (rec, start_s) windows of run_s clean seconds, tiled from the start of each clean stretch of
    each recording in turn. With k: the first k, or k drawn without replacement by rng (time order kept)."""
    w = []
    for rec in recs:
        last = -np.inf
        for s in clean_spans(rec, run_s):
            if s - last >= run_s:
                w.append((rec, int(s)))
                last = s
    if k is not None and k < len(w):
        w = w[:k] if rng is None else [w[i] for i in np.sort(rng.choice(len(w), k, replace=False))]
    return w


def within_patient_sequences(source, split, run_s, k=None, rng=None):
    """Iterator of (patient, [(rec, start_s), ...]) per patient (patient_windows over all of the patient's
    recordings, in order). Patients with no window give an empty list."""
    groups = OrderedDict()
    for it in _items(source, split):
        groups.setdefault(it["patient"], []).append(it)
    for patient, its in groups.items():
        yield patient, patient_windows([_load(it) for it in its], run_s, k, rng)
