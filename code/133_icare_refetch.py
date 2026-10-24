"""Additional fetches for the I-CARE arm, outside the frozen-code hash; built test-first on a stub server
(code/test_icare_refetch.py). Reads no outcome and no covariate: it takes the fixed segment, start and processing of
each patient from the arm's own per-patient files (122_icare_arm.py, imported, never edited). It runs only after the
main arm has finished (it would compete for PhysioNet bandwidth); --estimate reads headers and directory listings
only.

Two fetches, by HTTP Range, resumable (a finished output is skipped), the same retry and resume as 122:
1. --windows: the EXACT fixed 30-min window of every analysis patient whose window was not kept by the sidecar
   (not in keep_windows/). Same segment, same start_s, same bytes (the fixed slice [start - 10 s, start + 1810 s)
   of the segment, clipped to the segment: the context 122 gave the resampler), same gains and baselines, same channel
   order, same 500 -> 128 Hz resample_poly and median reference as 122.fetch_window. Written in the sidecar's format
   (keep_windows/<id>.npz: x float32 (19, 230400) at 128 Hz median-referenced, names, window = the arm's own window
   record) so one reader serves both. Verified against what 122 stored for that patient: the 19 channel SDs before and
   after the reference must reproduce (max abs difference recorded; a mismatch is flagged, never silently accepted).
   Manifest: keep_windows/refetch_manifest.jsonl (id, sha256 of the file, bytes read, SD differences, seconds).
2. --ecg: for EVERY analysis patient with a window, the ECG file of the same segment (<id>_<seg>_<hour>_ECG.{hea,mat}) over
   the same 30-min span, at the ECG file's own rate, all channels, by Range. keep_ecg/<id>.npz (ecg float32
   (channels, samples) physical units, invalid sentinel = NaN, fs, names, window) and keep_ecg/<id>.json (status, header
   facts, bytes). Status other than ok is recorded, never filled: no_ecg_file (the hospital recorded none for that
   segment), ecg_length_mismatch (its duration differs from the EEG segment's by more than 2 s), ecg_window_outside,
   unreadable_header. Missing is not zero.
--estimate: for the planned patients, directory listing + the ECG header only: the expected MB and hours of both fetches.

python 133_icare_refetch.py --estimate [--rate-kbs 153] [--limit N]
python 133_icare_refetch.py --windows|--ecg [--workers W] [--shard i --nshards n] [--limit N]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import concurrent.futures as cf
import hashlib
import importlib
import json
import re
import sys
import time
import warnings

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
ic = importlib.import_module("122_icare_arm")

SPEC_PATH = ic.SPEC_PATH
KEEP_WINDOWS = "keep_windows"
KEEP_ECG = "keep_ecg"
ECG_LENGTH_TOL_S = 2.0
SD_TOL = 1e-6



def _file_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


_OWN = _file_sha256()


def own_sha256():
    return _OWN


def guard_pinned(spec_path=None):
    """Not applied in this copy: the check reads files of the private working copy."""
    return None



def keep_dirs(root=None):
    root = root or ROOT
    return os.path.join(root, KEEP_WINDOWS), os.path.join(root, KEEP_ECG)


def planned_patients(out=None, split="analysis"):
    """Analysis patients (dev excluded) whose arm record is finished with a window: [(id, x-record)]. Reads the
    per-patient files only (no covariate or outcome file)."""
    index = ic.load_index(out)
    dev = set(ic.dev_patients(index, out))
    p = ic.paths(split, out)
    rows = []
    for pid in ic.processing_order(list(index)):
        if pid in dev:
            continue
        if ic.existing(os.path.join(p["x"], f"{pid}.json")):
            rows.append((pid, ic.read_data_json(os.path.join(p["x"], f"{pid}.json"))))
    return rows


def window_plan(rows, keep_windows):
    """Patients whose window the sidecar did not keep."""
    return [(pid, x) for pid, x in rows if not os.path.exists(os.path.join(keep_windows, f"{pid}.npz"))]



def read_slice(reader, a, b, block_s=ic.BLOCK_S):
    """Seconds [a, b) of the 19 engine sites in uV at the file's rate, in blocks (the same bytes as one read)."""
    return np.concatenate([reader.read(t, min(b, t + block_s)) for t in range(a, b, block_s)], axis=1)


def refetch_window(pid, x_rec, source, keep_windows):
    """The fixed window of one patient. -> manifest row. The window is written whole (tmp + rename)."""
    win = x_rec["window"]
    seg, start, fs = win["segment"], int(win["start_s"]), int(win["fs_in"])
    t0 = time.perf_counter()
    reader, total_s = source.open(pid, seg)
    if int(reader.fs) != fs:
        raise ic.SegmentError(f"{pid}: file rate {reader.fs} differs from the pinned {fs}")
    a, b = max(0, start - ic.RESAMPLE_PAD_S), min(total_s, start + ic.WINDOW_S + ic.RESAMPLE_PAD_S)
    raw = read_slice(reader, a, b)
    y = ic.resample_window(raw, fs, start - a, ic.WINDOW_S, ic.FS_OUT)
    eng = list(source.hdr["names"])
    sd = np.asarray(y, np.float64).std(1)
    y = ic.median_reference(y)
    sd_after = np.asarray(y, np.float64).std(1)
    d_before = max(abs(float(s) - float(win["channel_sd_before_car_uv"][c])) / max(abs(float(win["channel_sd_before_car_uv"][c])), 1e-12)
                   for c, s in zip(eng, sd))
    d_after = max(abs(float(s) - float(x_rec["channel_sd_uv"][c])) / max(abs(float(x_rec["channel_sd_uv"][c])), 1e-12)
                  for c, s in zip(eng, sd_after))
    meta = dict(win, refetched=True)
    path = os.path.join(keep_windows, f"{pid}.npz")
    os.makedirs(keep_windows, exist_ok=True)
    np.savez(path + ".tmp.npz", x=y, names=np.array(eng), window=json.dumps(ic._clean(meta)))
    os.replace(path + ".tmp.npz", path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    row = {"patient": pid, "file": f"{pid}.npz", "sha256": h.hexdigest(), "bytes_read": int(source.hdr["n_sig"]) * fs * (b - a) * ic.BYTES_PER_SAMPLE,
           "sd_before_rel_diff": d_before,
           "sd_after_rel_diff": d_after, "verified": bool(d_before <= SD_TOL and d_after <= SD_TOL),
           "seconds": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(keep_windows, "refetch_manifest.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row



def ecg_to_physical(dig, hdr):
    """(digital - baseline) / gain, float32; the WFDB invalid sentinel is NaN (missing is not zero)."""
    g = np.asarray(hdr["gains"], float)[:, None]
    b = np.asarray(hdr["baselines"], float)[:, None]
    out = (dig.astype(np.float64) - b) / g
    out[dig == ic.INVALID_DIGITAL] = np.nan
    return out.astype(np.float32)


def _status(path_json, rec):
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json + ".tmp", "w") as fh:
        json.dump(ic._clean(rec), fh)
    os.replace(path_json + ".tmp", path_json)


def eeg_total_s(win):
    """Seconds of the EEG segment, from the fixed window record: (bytes - 24) / (channels in file x 2 x rate)."""
    return (win["segment"]["bytes"] - ic.HEADER_BYTES) // (win["n_channels_in_file"] * ic.BYTES_PER_SAMPLE * int(win["fs_in"]))


def ecg_names(pid, seg):
    stem = f"{pid}_{seg['seg']:03d}_{seg['hour']:03d}_ECG"
    return stem + ".hea", stem + ".mat"


def fetch_ecg(pid, x_rec, listing, keep_ecg, base=None, sleep=None, estimate=False):
    """The ECG of the fixed segment over the fixed 30-min span. listing: {file name: bytes} of the patient's
    directory. -> status record (written to keep_ecg/<id>.json; the samples to keep_ecg/<id>.npz when ok)."""
    base = base or ic.BASE_URL
    kw = {"sleep": sleep} if sleep else {}
    win = x_rec["window"]
    seg, start = win["segment"], int(win["start_s"])
    hea, mat = ecg_names(pid, seg)
    rec = {"patient": pid, "segment": seg["seg"], "hour": seg["hour"], "start_s": start, "status": None}
    if mat not in listing or hea not in listing:
        rec["status"] = "no_ecg_file"
        return rec if estimate else (_status(os.path.join(keep_ecg, f"{pid}.json"), rec) or rec)
    try:
        h = ic.parse_header(ic.fetch_text(f"{base}training/{pid}/{hea}", **kw))
        fs = h["fs"]
        if h["offset"] != ic.HEADER_BYTES or fs != int(fs) or ic.HEADER_BYTES + h["n_samp"] * h["n_sig"] * 2 != listing[mat]:
            raise ic.SegmentError("unexpected ECG header or size")
    except ic.SegmentError as e:
        rec.update(status="unreadable_header", detail=str(e)[:120])
        return rec if estimate else (_status(os.path.join(keep_ecg, f"{pid}.json"), rec) or rec)
    fs = int(fs)
    rec.update(fs=fs, n_channels=h["n_sig"], names=h["names"], n_samples=h["n_samp"],
               window_bytes=h["n_sig"] * fs * ic.WINDOW_S * ic.BYTES_PER_SAMPLE)
    if abs(h["n_samp"] / fs - eeg_total_s(win)) > ECG_LENGTH_TOL_S:
        rec.update(status="ecg_length_mismatch", ecg_s=h["n_samp"] / fs, eeg_s=eeg_total_s(win))
        return rec if estimate else (_status(os.path.join(keep_ecg, f"{pid}.json"), rec) or rec)
    i0, i1 = start * fs, (start + ic.WINDOW_S) * fs
    if i1 > h["n_samp"]:
        rec.update(status="ecg_window_outside")
        return rec if estimate else (_status(os.path.join(keep_ecg, f"{pid}.json"), rec) or rec)
    if estimate:
        rec["status"] = "ok"
        return rec
    url = f"{base}training/{pid}/{mat}"
    blocks = []
    for t in range(start, start + ic.WINDOW_S, ic.BLOCK_S):
        a, b = ic.byte_range(h["n_sig"], t * fs, min(start + ic.WINDOW_S, t + ic.BLOCK_S) * fs)
        blocks.append(ic.decode_digital(ic.fetch_range(url, a, b, **kw), h["n_sig"]))
    ecg = ecg_to_physical(np.concatenate(blocks, axis=1), h)
    os.makedirs(keep_ecg, exist_ok=True)
    path = os.path.join(keep_ecg, f"{pid}.npz")
    np.savez(path + ".tmp.npz", ecg=ecg, fs=np.array(fs), names=np.array(h["names"]),
             window=json.dumps(ic._clean({"segment": seg, "start_s": start, "rosc_hours": win.get("rosc_hours")})))
    os.replace(path + ".tmp.npz", path)
    rec.update(status="ok", n_nan=int(np.isnan(ecg).sum()), file=f"{pid}.npz")
    _status(os.path.join(keep_ecg, f"{pid}.json"), rec)
    return rec



def estimate(rows, keep_windows, rate_kbs=153.0, workers=2, base=None, limit=None, sleep=None):
    """Directory listings + ECG headers only. Expected bytes and hours of both fetches at rate_kbs kB/s in total."""
    kw = {"sleep": sleep} if sleep else {}
    base = base or ic.BASE_URL
    rows = rows[:limit]
    todo = {pid for pid, _ in window_plan(rows, keep_windows)}
    w_bytes = sum(int(x["window"]["n_channels_in_file"]) * int(x["window"]["fs_in"]) * ic.BYTES_PER_SAMPLE
                  * (ic.WINDOW_S + 2 * ic.RESAMPLE_PAD_S) for pid, x in rows if pid in todo)

    def one(item):
        pid, x = item
        listing = dict(ic.parse_listing(ic.fetch_text(f"{base}training/{pid}/", **kw)))
        return fetch_ecg(pid, x, listing, None, base=base, sleep=sleep, estimate=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(one, rows))
    ok = [r for r in recs if r["status"] == "ok"]
    e_bytes = sum(r["window_bytes"] for r in ok)
    by = {}
    for r in recs:
        by[r["status"]] = by.get(r["status"], 0) + 1
    shapes = {}
    for r in ok:
        shapes[(r["n_channels"], r["fs"])] = shapes.get((r["n_channels"], r["fs"]), 0) + 1
    hours = lambda b: b / (rate_kbs * 1e3) / 3600.0
    return {"patients_with_window": len(rows), "windows_to_refetch": len(todo), "window_MB": round(w_bytes / 1e6),
            "window_hours": round(hours(w_bytes), 1), "ecg_status": by, "ecg_MB": round(e_bytes / 1e6),
            "ecg_hours": round(hours(e_bytes), 1), "ecg_channels_fs": {f"{k[0]}ch@{k[1]}Hz": v for k, v in sorted(shapes.items())},
            "total_MB": round((w_bytes + e_bytes) / 1e6), "total_hours": round(hours(w_bytes + e_bytes), 1),
            "rate_kbs": rate_kbs}



def run_windows(rows, keep_windows, workers=1, shard=0, nshards=1, limit=None, make_source=ic.HttpSource):
    todo = ic.take_shard([pid for pid, _ in window_plan(rows, keep_windows)], shard, nshards, limit)
    recs = dict(rows)
    ic.say(f"windows: {len(todo)} to fetch")

    def one(pid):
        try:
            return refetch_window(pid, recs[pid], make_source(), keep_windows)
        except (ic.DownloadError, ic.SegmentError) as e:
            ic.log_failure("analysis", None, pid, "refetch", e)
            return {"patient": pid, "verified": None, "error": str(e)[:120]}
    done = bad = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            done += r.get("verified") is True
            bad += r.get("verified") is not True
            ic.say(f"{r['patient']}: " + ("verified" if r.get("verified") else f"NOT verified ({r.get('error') or 'SD mismatch'})"))
    ic.say(f"windows done: {done} verified, {bad} not verified")
    return done, bad


def run_ecg(rows, keep_ecg, workers=1, shard=0, nshards=1, limit=None, base=None):
    todo = [pid for pid in ic.take_shard([pid for pid, _ in rows], shard, nshards, limit)
            if not os.path.exists(os.path.join(keep_ecg, f"{pid}.json"))]
    recs = dict(rows)
    counts = {}

    def one(pid):
        try:
            listing = dict(ic.parse_listing(ic.fetch_text(f"{base or ic.BASE_URL}training/{pid}/")))
            return fetch_ecg(pid, recs[pid], listing, keep_ecg, base=base)
        except ic.DownloadError as e:
            ic.log_failure("analysis", None, pid, "ecg", e)
            return {"patient": pid, "status": "download_failed"}
    with cf.ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
    ic.say("ecg:", json.dumps(counts))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--windows", action="store_true")
    ap.add_argument("--ecg", action="store_true")
    ap.add_argument("--rate-kbs", type=float, default=153.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    kw, ke = keep_dirs()
    rows = planned_patients()
    if a.estimate:
        ic.say("estimate:", json.dumps(estimate(rows, kw, a.rate_kbs, limit=a.limit)))
    if a.windows or a.ecg:
        guard_pinned()
    if a.windows:
        run_windows(rows, kw, a.workers, a.shard, a.nshards, a.limit)
    if a.ecg:
        run_ecg(rows, ke, a.workers, a.shard, a.nshards, a.limit)


if __name__ == "__main__":
    main()
