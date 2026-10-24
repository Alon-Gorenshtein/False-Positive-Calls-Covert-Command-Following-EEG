"""C2 fresh draw, the DGX-Spark-side worker (part C, C2; added later). Driven
by code/126_c2_fresh.py; one file, four uses:
- scan: piped over ssh stdin to the host SpikeNet2 python, run from /tmp with no bytecode and no HDF5 file locking;
  opens each listed .mat read-only and PRINTS fs, samples and the runs of 30-min eligible starts. Writes nothing.
- extract: in ~/work/c2_fresh inside the morgoth:spark container (no network), the BDSP folders mounted READ-ONLY and
  the output dir ~/work/c2_fresh/c2 as <base>: the fixed 30-min window of each listed segment -> <base>/sig/<uid>.npz
  in the design-npz format (x/names: the 11 overlay and Kit-8
  channels; x19/names19; float32 uV; 128 Hz; plus the window's start and the unit/reference check).
- morgoth: bdsp_morgoth.run on the original .mat (the whole segment, as for the Phase-1 rows; the 2-h memory guard
  is raised to the segment length for the rare longer segment) on the GPU if idle (fp32, TF32 off) ->
  morgoth_full/<uid>.parquet, and the rows of the 10-s windows inside the 30-min window with t_s rebased to the window
  start -> morgoth/<uid>.parquet.
- a9: 121_activation.detector_table on the windows -> activation/_activation_c2_fresh(.csv, _recordings.csv).
The padding rule is a verbatim copy of 108.padding_seconds / eligible_starts with 108's constants, applied on x19 as the
benchmark engine applies it (test_c2_fresh checks the source text and the constants)."""
import os
import sys

import numpy as np

ZERO_UV = 1e-3
FLOOR_UV, FLOOR_S = 2.0, 60
MARGIN_S = 30
RUN_S = 480
FS, WINDOW_S = 128, 1800
MORGOTH_WIN_S = 10
EEG19 = ("Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
         "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2")
SIG_CHANNELS = ("F3", "F4", "C3", "C4", "Fz", "Cz", "Pz", "T5", "O1", "O2", "T6")
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def padding_seconds(x, fs):
    """Per whole second: True if it is padding (part 1): exact zeros on any overlay channel, or part of a
    stretch of >= FLOOR_S s with every overlay channel below FLOOR_UV. x: channels x samples."""
    fs = int(fs)
    n = x.shape[1] // fs
    sd = np.asarray(x[:, : n * fs], float).reshape(x.shape[0], n, fs).std(2)
    pad = sd.min(0) < ZERO_UV
    low = sd.max(0) < FLOOR_UV
    d = np.diff(np.r_[0, low.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= FLOOR_S:
            pad[a:b] = True
    return pad


def eligible_starts(x, fs, run_s=RUN_S):
    """Run starts (whole seconds) with no padding within the run or MARGIN_S around it. Large artifacts are kept on
    purpose (fixed in advance: no artifact rejection)."""
    pad = padding_seconds(x, fs)
    n = len(pad)
    bad = pad.copy()
    for k in range(1, MARGIN_S + 1):
        bad[k:] |= pad[:-k]
        bad[:-k] |= pad[k:]
    c = np.r_[0, np.cumsum(bad.astype(int))]
    s = np.arange(0, n - run_s + 1)
    return s[(c[s + run_s] - c[s]) == 0]


def read_mat(path):
    """(data float64 [channels x samples], fs, names) of a MATLAB v7.3 BDSP file, opened read-only."""
    import h5py
    with h5py.File(path, "r") as f:
        data = f["data"][()].T
        fs = float(np.asarray(f["Fs"][()]).squeeze())
        names = ["".join(map(chr, f[r][()].flatten())).strip() for r in f["channels"][()].flatten()]
    if data.shape[0] != len(names):
        raise ValueError(f"{path}: {data.shape[0]} rows vs {len(names)} channel names")
    return data, fs, names


def pick(data, names, wanted):
    idx = {ALIASES.get(n.upper(), n.upper()): i for i, n in enumerate(names)}
    missing = [w for w in wanted if w.upper() not in idx]
    if missing:
        raise ValueError(f"missing channels {missing}")
    return data[[idx[w.upper()] for w in wanted]]


def runs(starts):
    """Sorted whole-second starts -> [(first, last)] of consecutive runs."""
    s = np.asarray(starts, int)
    if not len(s):
        return []
    cut = np.flatnonzero(np.diff(s) != 1)
    return [(int(a), int(b)) for a, b in zip(np.r_[s[0], s[cut + 1]], np.r_[s[cut], s[-1]])]


def scan_one(path):
    """(fs, samples, status, runs of 30-min eligible starts on x19 as float32)."""
    data, fs, names = read_mat(path)
    if fs != FS:
        return fs, data.shape[1], "fs", []
    try:
        x19 = pick(data, names, EEG19).astype(np.float32)
    except ValueError:
        return fs, data.shape[1], "channels", []
    r = runs(eligible_starts(x19, fs, run_s=WINDOW_S))
    return fs, data.shape[1], ("ok" if r else "no_window"), r


def window_npz(path, start_s, uid, pid):
    """The design-npz fields of one 30-min window starting at start_s (refused unless 108-eligible on x19)."""
    data, fs, names = read_mat(path)
    if fs != FS:
        raise ValueError(f"{uid}: fs {fs}")
    x19_full = pick(data, names, EEG19).astype(np.float32)
    if start_s not in set(eligible_starts(x19_full, fs, run_s=WINDOW_S).tolist()):
        raise ValueError(f"{uid}: start {start_s} s is not 30-min eligible")
    a, b = int(start_s * FS), int((start_s + WINDOW_S) * FS)
    win = data[:, a:b]
    eeg = [i for i, n in enumerate(names) if n.upper() != "EKG"]
    return dict(x=pick(win, names, SIG_CHANNELS).astype(np.float32), names=np.array(SIG_CHANNELS),
                x19=x19_full[:, a:b], names19=np.array(EEG19), fs=fs, dur_s=WINDOW_S * 1.0,
                full_dur_s=data.shape[1] / fs, window_start_s=int(start_s), pid=pid, uid=uid,
                all_names=np.array(names), med_abs=np.median(np.abs(win), axis=1),
                car_med_abs=np.median(np.abs(win[eeg].mean(axis=0))))


def _save_npz(path, fields):
    with open(path + ".part", "wb") as fh:
        np.savez(fh, **fields)
    os.replace(path + ".part", path)


def window_rows(full, start_s, win_s=WINDOW_S, morgoth_win_s=MORGOTH_WIN_S):
    """Morgoth rows of the 10-s windows that lie inside [start, start + win), t_s rebased to the window start."""
    keep = (full.t_s >= start_s) & (full.t_s <= start_s + win_s - morgoth_win_s)
    out = full[keep].copy()
    out["t_s"] = (out["t_s"] - start_s).astype(full.t_s.dtype)
    return out.reset_index(drop=True)


def morgoth_hours(full_dur_s, guard_h=2):
    """Morgoth's max_length_hour for one segment: its 2-h memory guard (bdsp_morgoth.MAX_HOURS, the Phase-1 value),
    raised to the whole segment when longer, so a window past 2 h is labelled from the same whole-segment run."""
    return max(guard_h, int(np.ceil(full_dur_s / 3600)))


def _listing(path):
    import pandas as pd
    return pd.read_csv(path)


def main(argv):
    cmd = argv[0]
    if cmd == "scan":
        for i, p in enumerate(argv[1:]):
            try:
                fs, n, status, r = scan_one(p)
            except Exception as e:
                fs, n, status, r = 0, 0, "error:" + type(e).__name__, []
            print("SCAN\t%d\t%g\t%d\t%s\t%s" % (i, fs, n, status, ";".join(f"{a}-{b}" for a, b in r)), flush=True)
        return
    base, bdsp, listing = argv[1], argv[2], argv[3]
    rows = _listing(listing)
    if len(argv) > 4:
        if cmd == "a9":
            raise SystemExit("a9 runs on all rows at once")
        k, n = map(int, argv[4].split("/"))
        rows = rows.iloc[k::n]
    code = os.path.dirname(os.path.abspath(__file__))
    if cmd == "extract":
        os.makedirs(f"{base}/sig", exist_ok=True)
        for r in rows.itertuples():
            out = f"{base}/sig/{r.uid}.npz"
            if not os.path.exists(out):
                _save_npz(out, window_npz(f"{bdsp}/{r.dir}/{r.file}", int(r.window_start_s), r.uid, r.patient_id))
            print("EXTRACTED", r.rank, flush=True)
    elif cmd == "morgoth":
        import torch
        sys.path.insert(0, code)
        import bdsp_morgoth as bm
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = bm.load_models(dev)
        os.makedirs(f"{base}/morgoth", exist_ok=True)
        for r in rows.itertuples():
            out = f"{base}/morgoth/{r.uid}.parquet"
            if os.path.exists(out):
                continue
            hours = morgoth_hours(r.n_samples / r.fs, bm.MAX_HOURS)
            orig = bm.windows
            bm.windows = lambda path, key, _o=orig, _h=hours: _o(path, key, max_hours=_h)
            try:
                full, secs = bm.run(f"{bdsp}/{r.dir}/{r.file}", models, dev, out_dir=f"{base}/morgoth_full")
            finally:
                bm.windows = orig
            w = window_rows(full, int(r.window_start_s))
            w.to_parquet(out + ".part", index=False)
            os.replace(out + ".part", out)
            print("MORGOTH", r.rank, dev.type, round(sum(secs.values()), 1), len(w), flush=True)
    elif cmd == "a9":
        import importlib
        from types import SimpleNamespace
        sys.path.insert(0, code)
        act = importlib.import_module("121_activation")
        recs = []
        for r in rows.itertuples():
            with np.load(f"{base}/sig/{r.uid}.npz") as z:
                recs.append(SimpleNamespace(uid=r.uid, x=np.asarray(z["x19"], np.float32),
                                            names=[str(n) for n in z["names19"]], fs=float(z["fs"])))
        iv, rec_tab = act.detector_table(recs)
        os.makedirs(f"{base}/activation", exist_ok=True)
        iv.to_csv(f"{base}/activation/_activation_c2_fresh.csv", index=False)
        rec_tab.to_csv(f"{base}/activation/_activation_c2_fresh_recordings.csv", index=False)
        print("A9", len(rec_tab), int((rec_tab.n_intervals > 0).sum()), flush=True)
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
