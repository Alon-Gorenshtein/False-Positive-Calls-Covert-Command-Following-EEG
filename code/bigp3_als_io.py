"""BigP3BCI (PhysioNet, doi 10.13026/0byy-ry86, CC BY 4.0), ALS studies B, F, L, N (19 + 10 + 11 + 8 patients). Loader for a
flash-locked P300 positive control. Source EDFs live in ANOTHER study's cache and are only ever read.
Event coding (PhysioNet page, Table 3; verified on the files): StimulusBegin = 1 while a flash is on (its rising edge is the
flash onset); StimulusType at that sample = 1 for a target flash, 0 non-target; PhaseInSequence 2 = during a character trial
(one 'sequence'); the 36 channels <char>_<row>_<col> are 1 while that character is lit; CurrentTarget = index of the target
character (channel index - 1 is lit exactly on target flashes); StimulusCode = running flash code (row / column code in RC).
extract(study, subject) keeps Fz C3 Cz C4 P3 Pz P4 in µV at the native rate, per run the flash onsets, target flags,
sequence ids, lit characters and codes, and writes output/raw/bigp3_als/<study>_<subject>.npz; flashes() reads it back."""
import json
import os
import re

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
SRC = os.environ.get("BIGP3BCI_DATA", os.path.join(ROOT, "data", "bigP3BCI-data"))
OUT = os.path.join(ROOT, "output", "raw", "bigp3_als")
CHANNELS = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4"]
STUDIES = ("B", "F", "L", "N")
LICENCE = {"name": "Creative Commons Attribution 4.0 International Public License", "short": "CC BY 4.0",
           "doi_v1.0.0": "10.13026/0byy-ry86", "source": "https://physionet.org/content/bigp3bci/1.0.0/ (Access Policy: "
                                                          "anyone can access the files, as long as they conform to the license)"}


def flash_onsets(sb):
    """Sample of each rising edge of StimulusBegin (a high first sample counts)."""
    b = (np.asarray(sb) > 0.5).astype(int)
    return np.flatnonzero(np.diff(np.r_[0, b]) == 1)


def flash_lengths(sb):
    b = (np.asarray(sb) > 0.5).astype(int)
    d = np.diff(np.r_[0, b, 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def sequences(ph):
    """(first, last) sample of each contiguous PhaseInSequence == 2 stretch."""
    p = (np.asarray(ph) == 2).astype(int)
    d = np.diff(np.r_[0, p, 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def assign_sequence(onsets, starts, ends):
    seq = np.full(len(onsets), -1)
    for k, (a, b) in enumerate(zip(starts, ends)):
        seq[(onsets >= a) & (onsets <= b)] = k
    if (seq < 0).any():
        raise ValueError(f"{int((seq < 0).sum())} flash onset(s) outside every PhaseInSequence == 2 stretch")
    return seq


def resolve_channels(names, wanted=CHANNELS):
    """Row of EEG_<site> for each wanted site; every missing site is reported, then ValueError."""
    idx = {n: i for i, n in enumerate(names)}
    missing = [c for c in wanted if f"EEG_{c}" not in idx]
    if missing:
        raise ValueError(f"EEG channel(s) missing from the recording: {missing}")
    return [idx[f"EEG_{c}"] for c in wanted]


def build_run(eeg, sb, st, ph, sc, ct, chars, fs):
    """One run -> dict(x float32 µV, onset, is_target, code, seq, chars (n_flash x n_char) uint8, target_char). Every flash
    must have the same length and lie in a sequence. The target flag is StimulusType; in each sequence it must be explained by
    exactly one lit character. CurrentTarget (the logged target) is only compared: sequences where it disagrees with that
    character are counted in n_current_target_mismatch_seq (2 runs of B_01 have one)."""
    on = flash_onsets(sb)
    if len(on) == 0:
        raise ValueError("no flashes in the run")
    if len(set(flash_lengths(sb).tolist())) != 1:
        raise ValueError(f"unequal flash length(s) {sorted(set(flash_lengths(sb).tolist()))}")
    seq = assign_sequence(on, *sequences(ph))
    lit = (np.asarray(chars)[:, on].T > 0.5)
    tgt = np.asarray(st)[on] > 0.5
    ti = np.asarray(ct)[on].astype(int)
    mismatch = 0
    for s in np.unique(seq):
        i = seq == s
        if not tgt[i].any():
            continue
        c = np.flatnonzero(lit[i][tgt[i]].all(0) & ~lit[i][~tgt[i]].any(0)) if (~tgt[i]).any() else np.flatnonzero(lit[i].all(0))
        if len(c) != 1:
            raise ValueError(f"sequence {int(s)}: StimulusType flags are not explained by one lit character ({len(c)} candidates)")
        mismatch += int(not (ti[i] == c[0] + 1).all())
    return {"x": np.asarray(eeg, np.float32), "onset": on.astype(np.int64), "is_target": tgt, "code": np.asarray(sc)[on].astype(np.int32),
            "seq": seq.astype(np.int32), "chars": lit.astype(np.uint8), "target_char": ti.astype(np.int16),
            "flash_len": int(flash_lengths(sb)[0]), "n_current_target_mismatch_seq": mismatch}


def run_meta(run, fs):
    """SOA (median flash-to-flash interval inside sequences), flash duration, flashes and targets per sequence."""
    on, seq, tgt = run["onset"], run["seq"], run["is_target"]
    same = seq[1:] == seq[:-1]
    soa = float(np.median(np.diff(on)[same]) / fs) if same.any() else float("nan")
    k = np.unique(seq)
    return {"soa_s": soa, "n_flashes": int(len(on)), "n_targets": int(tgt.sum()), "n_sequences": int(len(k)),
            "flashes_per_sequence": [int((seq == s).sum()) for s in k], "targets_per_sequence": [int(tgt[seq == s].sum()) for s in k],
            "flash_ms": float(run["flash_len"] / fs * 1000), "n_current_target_mismatch_seq": int(run["n_current_target_mismatch_seq"])}


def path_info(rel):
    parts = rel.split("/")
    if len(parts) < 6 or not parts[0].startswith("Study") or parts[3] not in ("Train", "Test"):
        raise ValueError(f"unexpected BigP3 path: {rel!r}")
    return {"study": parts[0][5:], "subject": parts[1], "session": parts[2], "phase": parts[3], "paradigm": parts[4], "file": parts[-1]}


def list_edfs(study, subject, src=SRC):
    root = os.path.join(src, f"Study{study}", subject)
    out = []
    for d, _, fn in os.walk(root):
        out += [os.path.join(d, f) for f in fn if f.endswith(".edf") and not f.startswith("._")]
    return sorted(out)


def npz_path(study, subject, out=OUT):
    return os.path.join(out, f"{study}_{subject}.npz")


def check_out_not_in_source(out):
    if os.path.abspath(out).startswith(os.path.abspath(SRC)):
        raise ValueError("refusing to write under the BigP3BCI source cache (another study's read-only data)")


def read_edf(path):
    """One EDF -> the arrays build_run takes (+ fs). Reads the signals; the patient-identification header is never touched."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    names = raw.ch_names
    rows = resolve_channels(names)
    g = lambda n: raw.get_data(picks=[names.index(n)])[0]
    chars = [n for n in names if re.fullmatch(r".+_\d+_\d+", n)]
    return {"eeg": raw.get_data(picks=rows) * 1e6, "sb": g("StimulusBegin"), "st": g("StimulusType"), "ph": g("PhaseInSequence"),
            "sc": g("StimulusCode"), "ct": g("CurrentTarget"), "chars": raw.get_data(picks=[names.index(c) for c in chars])}, \
        float(raw.info["sfreq"]), len(chars)


def extract(study, subject, src=SRC, out=OUT):
    """Read the patient's EDFs, write <out>/<study>_<subject>.npz (see the module docstring). Returns the path."""
    check_out_not_in_source(out)
    files = list_edfs(study, subject, src)
    if not files:
        raise FileNotFoundError(f"no EDFs for study {study} subject {subject} under {src}")
    os.makedirs(out, exist_ok=True)
    arrays, metas = {}, []
    for k, f in enumerate(files):
        info = path_info(os.path.relpath(f, src))
        d, fs, n_char = read_edf(f)
        run = build_run(**d, fs=fs)
        m = run_meta(run, fs)
        m.update({"run": k, "fs": fs, "n_chars": n_char, "session": info["session"], "phase": info["phase"],
                  "paradigm": info["paradigm"], "file": info["file"]})
        metas.append(m)
        for key, v in run.items():
            if key in ("flash_len", "n_current_target_mismatch_seq"):
                continue
            arrays[f"r{k:03d}_{key}"] = v
    tmp = npz_path(study, subject, out) + ".tmp.npz"
    np.savez(tmp, channels=np.array(CHANNELS), meta=np.array(json.dumps(metas)), **arrays)
    os.replace(tmp, npz_path(study, subject, out))
    return npz_path(study, subject, out)


def flashes(study, subject, out=OUT):
    """Per-run dicts from the npz: x (7 x n float32 µV), fs, onset, is_target, code, seq, chars, target_char, meta."""
    z = np.load(npz_path(study, subject, out))
    if list(z["channels"]) != CHANNELS:
        raise ValueError("unexpected channels in the npz")
    metas = json.loads(str(z["meta"]))
    return [dict({key: z[f"r{m['run']:03d}_{key}"] for key in ("x", "onset", "is_target", "code", "seq", "chars", "target_char")},
                 fs=m["fs"], meta=m) for m in metas]
