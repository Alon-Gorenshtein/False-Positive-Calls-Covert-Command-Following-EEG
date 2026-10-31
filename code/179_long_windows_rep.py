"""Long windows for the replication patients (the 730 of 126 extend, c2_fresh_ext.csv), for the analyses whose
schedules do not fit their 30-minute windows (178.NEED). The BDSP folders on the DGX Spark are READ-ONLY: the
extractor goes over ssh stdin, runs from /tmp with no bytecode and no HDF5 locking, opens the .mat with h5py in mode
"r" and writes the window to stdout only; every byte lands on the SSD.

list: per listed patient, in the segment its 30-minute window came from, the clean stretches of c2_scan.csv (runs of
  30-min eligible starts, c2_worker's 108 rule: a window of W s starting at s is eligible iff s lies in [a, b + 1800 - W]
  for a run a-b); W = the longest of the needs that fits (patients with none are listed with W = 0 and left out of every
  long-window denominator); start drawn uniformly from the W-eligible starts by np.random.default_rng([SEED, 179,
  rank]). Fixed before any long window is read. -> output/bdsp/meta/long_rep_windows.csv and its sha256.
extract: each listed window -> output/bdsp/long/sig/<uid>.npz (x19 float32 uV in c2_worker's EEG19 order, 128 Hz;
  the start re-checked on the Spark against c2_worker.eligible_starts of the whole file at run_s = W).
run --pipe P: the frozen reading (178.frozen) on each extracted window with W >= its need, first overlay, through
  178.rows_for (114 unchanged); n_recs = the 730 listed, as 153. Sokoliuk 2021 (1,367 s) runs on the 730 replication
  30-minute windows (153.items). Windows not yet extracted are skipped; a later run completes them.
  -> output/intermediate/_long/rep__<pipe>/rows.jsonl
python 179_long_windows_rep.py list | extract [--workers W] [--limit N] | run --pipe P"""
import argparse
import hashlib
import importlib
import inspect
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
cw = importlib.import_module("c2_worker")
ls = importlib.import_module("178_long_schedules")

META = os.path.join(ROOT, "output", "bdsp", "meta")
SCAN, EXT = os.path.join(META, "c2_scan.csv"), os.path.join(META, "c2_fresh_ext.csv")
LISTING = os.path.join(META, "long_rep_windows.csv")
SIG = os.path.join(ROOT, "output", "bdsp", "long", "sig")
SEED, STREAM = 20261030, 179
HOST, BDSP = "dgx-spark", "Desktop/spikenet2data"
REMOTE_PY = "~/miniforge3/envs/SpikeNet2/bin/python"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", HOST]
WIDTHS = tuple(sorted(set(ls.NEED.values()), reverse=True))


def parse_runs(runs):
    """'a-b;c-d' -> [(a, b)] (whole-second runs of 30-min eligible starts)."""
    out = []
    for r in str(runs).split(";"):
        if "-" in r:
            a, b = r.split("-")
            out.append((int(a), int(b)))
    return out


def long_starts(runs, w, base=cw.WINDOW_S):
    """Whole-second starts of w-second windows inside the clean stretches implied by the base-length runs."""
    out = [np.arange(a, b + base - w + 1) for a, b in runs if b + base - w >= a]
    return np.concatenate(out) if out else np.zeros(0, int)


def choose(runs, rank, widths=WIDTHS):
    """(W, start, n eligible starts): the longest width that fits, its start drawn with the patient's seed."""
    for w in widths:
        s = long_starts(runs, w)
        if len(s):
            return w, int(np.random.default_rng([SEED, STREAM, int(rank)]).choice(s)), len(s)
    return 0, -1, 0


def build_listing(ext=EXT, scan=SCAN):
    e = pd.read_csv(ext, dtype={"patient_id": str, "uid": str}).sort_values("rank", kind="stable")
    s = pd.read_csv(scan, dtype={"patient_id": str})
    m = e.merge(s[["patient_id", "seg_rank", "file", "runs"]], on=["patient_id", "seg_rank", "file"], how="left")
    if len(m) != len(e) or m["runs"].isna().any():
        raise RuntimeError("a listed window has no scan row: refused")
    picks = [choose(parse_runs(r), k) for r, k in zip(m["runs"], m["rank"])]
    m["W"], m["start_s"], m["n_eligible_long"] = zip(*picks)
    return m[["rank", "patient_id", "uid", "dir", "file", "seg_rank", "window_start_s", "W", "start_s",
              "n_eligible_long"]]


EXTRACT_SRC = "\n".join([
    "import io, sys", "import numpy as np",
    *[f"{k} = {getattr(cw, k)!r}" for k in ("ZERO_UV", "FLOOR_UV", "FLOOR_S", "MARGIN_S", "RUN_S", "EEG19", "ALIASES")],
    inspect.getsource(cw.padding_seconds), inspect.getsource(cw.eligible_starts), inspect.getsource(cw.read_mat),
    inspect.getsource(cw.pick),
    r'''
def payload(path, start, w):
    data, fs, names = read_mat(path)
    if fs != 128:
        raise ValueError("fs %g" % fs)
    x19 = pick(data, names, EEG19).astype(np.float32)
    if start not in set(eligible_starts(x19, fs, run_s=w).tolist()):
        raise ValueError("start %d is not %d-s eligible" % (start, w))
    a, b = int(start * fs), int((start + w) * fs)
    win = data[:, a:b]
    eeg = [i for i, n in enumerate(names) if n.upper() != "EKG"]
    return dict(x19=x19[:, a:b], names19=np.array(EEG19), fs=fs, dur_s=float(w), full_dur_s=data.shape[1] / fs,
                window_start_s=int(start), med_abs=np.median(np.abs(win), axis=1),
                car_med_abs=np.median(np.abs(win[eeg].mean(axis=0))))


if __name__ == "__main__":
    buf = io.BytesIO()
    np.savez(buf, **payload(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
    sys.stdout.buffer.write(buf.getvalue())
'''])


def _ok(path, w):
    try:
        with np.load(path) as z:
            return z["x19"].shape == (19, int(w) * 128)
    except Exception:
        return False


def extract_one(r, out=SIG, tries=3):
    dest = os.path.join(out, f"{r.uid}.npz")
    if os.path.exists(dest):
        return "exists"
    if r.dir not in ("EEGs", "negative_EEGs") or not re.fullmatch(r"Bonobo\d+_\d+\.mat", r.file):
        raise ValueError(f"refusing unexpected path {r.dir}/{r.file}")
    cmd = (f"cd /tmp && PYTHONDONTWRITEBYTECODE=1 HDF5_USE_FILE_LOCKING=FALSE {REMOTE_PY} -B - "
           f"\"$HOME/{BDSP}/{r.dir}/{r.file}\" {int(r.start_s)} {int(r.W)}")
    err = ""
    for _ in range(tries):
        with open(dest + ".part", "wb") as fh:
            res = subprocess.run(SSH + [cmd], input=EXTRACT_SRC.encode(), stdout=fh, stderr=subprocess.PIPE)
        if res.returncode == 0 and _ok(dest + ".part", r.W):
            with np.load(dest + ".part") as z:
                fields = dict(z)
            fields.update(uid=r.uid, pid=r.patient_id)
            with open(dest + ".part2", "wb") as fh:
                np.savez(fh, **fields)
            os.replace(dest + ".part2", dest)
            os.remove(dest + ".part")
            return "ok"
        err = res.stderr.decode()[-300:]
        time.sleep(5)
    if os.path.exists(dest + ".part"):
        os.remove(dest + ".part")
    return "failed: " + err


def recording(path, uid, pid, rank):
    """The long window as the engine loads a clinical EEG window without Morgoth labels (153.load_without_morgoth)."""
    en = ls.b.en
    with np.load(path) as z:
        x, fs, names = np.asarray(z["x19"], np.float32), float(z["fs"]), [str(n) for n in z["names19"]]
    meta = {"dev": False, "fresh": True, "fresh_rank": int(rank), "long_window": True,
            "morgoth_t": np.array([], float), "morgoth_lab": np.array([], object)}
    return en.Recording(source="bdsp", uid=uid, patient=pid, x=x, fs=fs, names=names, meta=meta)


def run(pipe, limit=None):
    import json
    fz = ls.frozen()
    kind = fz["readings"][pipe]["null_kind"]
    fn = os.path.join(ls.LONG, f"rep__{pipe}", "rows.jsonl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = {json.loads(l)["uid"] for l in open(fn) if l.strip()} if os.path.exists(fn) else set()
    if pipe == "sokoliuk21":
        rp = importlib.import_module("153_replication_rows")
        units = [(it["uid"], lambda it=it: rp.load_without_morgoth(it)) for it in rp.items()]
        null, split = f"{kind}_bdsp_rep", "c2_ext"
    else:
        m = pd.read_csv(LISTING, dtype={"patient_id": str, "uid": str})
        m = m[m["W"] >= ls.NEED[pipe]]
        units = [(r.uid, lambda r=r: recording(os.path.join(SIG, f"{r.uid}.npz"), r.uid, r.patient_id, r.rank))
                 for r in m.itertuples() if os.path.exists(os.path.join(SIG, f"{r.uid}.npz"))]
        null, split = f"{kind}_bdsp_rep_long", "c2_ext_long"
    todo = [(u, f) for u, f in units if u not in done][:limit]
    print(f"{pipe}: {len(units)} windows available, {len(todo)} to run", flush=True)
    with open(fn, "a") as fh:
        for uid, load in todo:
            t = time.perf_counter()
            for r in ls.rows_for(load(), pipe, null, split, 730, fz):
                fh.write(json.dumps(r, allow_nan=False) + "\n")
            fh.flush()
            print(f"{pipe} {uid} {time.perf_counter() - t:.1f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "extract", "run"])
    ap.add_argument("--pipe", choices=["cruse11", "cruse12p", "chennu13", "sokoliuk21"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        if os.path.exists(LISTING):
            raise SystemExit(f"{LISTING} exists: never rewritten")
        m = build_listing()
        m.to_csv(LISTING, index=False)
        print(m["W"].value_counts().to_dict(), hashlib.sha256(open(LISTING, "rb").read()).hexdigest())
        return
    if a.cmd == "run":
        run(a.pipe, a.limit)
        return
    m = pd.read_csv(LISTING, dtype={"patient_id": str, "uid": str})
    m = m[m["W"] > 0].head(a.limit) if a.limit else m[m["W"] > 0]
    os.makedirs(SIG, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (r, res) in enumerate(zip(m.itertuples(), ex.map(extract_one, m.itertuples()))):
            print(f"{i + 1}/{len(m)} {r.uid} W={r.W} {res} {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
