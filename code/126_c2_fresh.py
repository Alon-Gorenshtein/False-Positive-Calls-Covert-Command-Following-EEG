"""C2 fresh draw on the DGX Spark (part C, C2; added later): the local driver
of code/c2_worker.py. The BDSP folders on the Spark are READ-ONLY. Built test-first (code/test_c2_fresh.py).

scan (now; metadata only, nothing written on the Spark): the candidates of c2_fresh_listing.csv (125_c2_listing.py)
are scanned in rank order by c2_worker scan (ssh stdin, from /tmp, no bytecode, no HDF5 locking), which prints each
segment's fs, samples and runs of 30-min eligible starts (108 rule on x19). select() then applies 125's rule: walking
the ranks, a patient is taken with the first of its segments in seg_rank order that is "ok", at a window start drawn
uniformly from its eligible starts by np.random.default_rng([SEED, STREAM, rank]); stop at C2_N.
-> output/bdsp/meta/c2_scan.csv (every scanned segment) and c2_fresh_400.csv (the fixed list) with its sha256.

extend (the fresh-draw extension; after its fixing): the same rule with n = 400 + N_EXT. The scan resumes in
steps of EXT_STEP ranks until the selection is complete; the first 400 selected must reproduce the fixed
c2_fresh_400.csv byte for byte (sha256 PINNED_400), else it is refused; the extension is the rows after them,
refused if any of its patients is touched (125.touched_patients) or repeated. c2_fresh_400.csv is never rewritten.
-> output/bdsp/meta/c2_fresh_ext.csv (same columns) and its sha256.
python3 126_c2_fresh.py scan | extend [--n-ext N]"""
import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
META = os.path.join(ROOT, "output", "bdsp", "meta")
lst = importlib.import_module("125_c2_listing")
SEED, STREAM, C2_N = lst.SEED, lst.STREAM, lst.C2_N
HOST = "dgx-spark"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST]
BDSP = "$HOME/Desktop/spikenet2data"
HOST_PY = "~/miniforge3/envs/SpikeNet2/bin/python"
SCAN_HORIZON, SCAN_BATCH, SCAN_PROCS = 1500, 50, 3
SCAN_COLS = ["rank", "patient_id", "seg_rank", "dir", "file", "fs", "n_samples", "status", "runs"]
_NAME = re.compile(r"^Bonobo\d+_\d+\.mat$")
PINNED_400 = "216c6a2977264d2c3f5e354df6185daa83782a69a7e176de8fbb1ede92c45d39"
N_EXT = 730
EXT_STEP = 200
EXT_CSV = "c2_fresh_ext.csv"


def remote_cmd(paths, root=BDSP, py=HOST_PY):
    """The scan command: from /tmp, no bytecode, no HDF5 locking; the worker comes on stdin."""
    for d, f in paths:
        if d not in ("EEGs", "negative_EEGs") or not _NAME.match(f):
            raise ValueError(f"refusing unexpected path {d}/{f}")
    args = " ".join(f'"{root}/{d}/{f}"' for d, f in paths)
    return f"cd /tmp && PYTHONDONTWRITEBYTECODE=1 HDF5_USE_FILE_LOCKING=FALSE {py} -B - scan {args}"


def parse_scan(lines):
    """SCAN lines -> [(index, fs, n_samples, status, runs string)]."""
    out = []
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) == 6 and p[0] == "SCAN":
            out.append((int(p[1]), float(p[2]), int(p[3]), p[4], p[5]))
    return out


def expand(runs):
    """'a-b;c-d' -> the whole-second starts a..b, c..d."""
    if not isinstance(runs, str) or not runs:
        return np.array([], int)
    return np.concatenate([np.arange(int(a), int(b) + 1) for a, b in (r.split("-") for r in runs.split(";"))])


def select(listing, scan, n=C2_N):
    """125's rule on scanned segments: the fixed list (rank, patient_id, uid, dir, file, seg_rank, fs,
    n_samples, window_start_s, n_eligible_starts). Refuses to pass a patient whose segments were not all scanned."""
    sc = scan.set_index("file")
    out = []
    for rank, g in listing.groupby("rank", sort=True):
        if len(out) == n:
            break
        for r in g.sort_values("seg_rank").itertuples():
            if r.file not in sc.index:
                raise RuntimeError(f"rank {rank}: segment not scanned; scan further before selecting")
            s = sc.loc[r.file]
            if s.status != "ok":
                continue
            starts = expand(s.runs)
            start = int(np.random.default_rng([SEED, STREAM, int(rank)]).choice(starts))
            out.append(dict(rank=int(rank), patient_id=r.patient_id, uid=r.file[:-4], dir=r.dir, file=r.file,
                            seg_rank=int(r.seg_rank), fs=float(s.fs), n_samples=int(s.n_samples),
                            window_start_s=start, n_eligible_starts=len(starts)))
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} patients selected within the listing")
    return pd.DataFrame(out)


def scan(listing, horizon=SCAN_HORIZON, out_csv=None, worker=None, ssh=SSH, root=BDSP, py=HOST_PY):
    """Scan every segment of the first `horizon` ranks not yet in out_csv (resumable); returns the scan table."""
    out_csv = out_csv or os.path.join(META, "c2_scan.csv")
    worker = worker or open(os.path.join(CODE, "c2_worker.py")).read()
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=SCAN_COLS)
    todo = listing[(listing["rank"] < horizon) & ~listing.file.isin(set(done.file))]
    batches = [todo.iloc[i:i + SCAN_BATCH] for i in range(0, len(todo), SCAN_BATCH)]
    lock, rows = threading.Lock(), []

    def one(b):
        res = subprocess.run(ssh + [remote_cmd(list(zip(b.dir, b.file)), root, py)], input=worker, capture_output=True,
                             text=True)
        got = parse_scan(res.stdout.splitlines())
        if res.returncode != 0 or len(got) != len(b):
            raise RuntimeError(f"scan batch failed (rc {res.returncode}, {len(got)}/{len(b)}): {res.stderr[-300:]}")
        new = [dict(rank=r.rank, patient_id=r.patient_id, seg_rank=r.seg_rank, dir=r.dir, file=r.file, fs=fs,
                    n_samples=n, status=st, runs=ru) for r, (i, fs, n, st, ru) in zip(b.itertuples(), got)]
        with lock:
            rows.extend(new)
            pd.DataFrame(new, columns=SCAN_COLS).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv),
                                                       index=False)
            print(f"scanned {len(rows)}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(SCAN_PROCS) as ex:
        list(ex.map(one, batches))
    return pd.read_csv(out_csv)


def extend(listing, pinned, n_ext, out_csv, scan_csv=None, start=SCAN_HORIZON, step=EXT_STEP, touched=None,
           expect=PINNED_400, **scan_kw):
    """(extension frame, its sha256): select(n = len(fixed) + n_ext) after scanning far enough; refused unless
    the first rows reproduce the fixed list byte for byte and the extension is untouched and unrepeated."""
    reg = open(pinned, "rb").read()
    if hashlib.sha256(reg).hexdigest() != expect:
        raise RuntimeError(f"{pinned} is not the pinned list (sha256)")
    n0 = len(pd.read_csv(pinned))
    horizon = start
    while True:
        sc = scan(listing, horizon, out_csv=scan_csv, **scan_kw)
        try:
            sel = select(listing, sc, n=n0 + n_ext)
            break
        except RuntimeError as e:
            if "not scanned" not in str(e) or horizon > listing["rank"].max():
                raise
            horizon += step
    if sel.iloc[:n0].to_csv(index=False).encode() != reg:
        raise RuntimeError("the first rows do not reproduce the pinned list")
    ext = sel.iloc[n0:].reset_index(drop=True)
    gone = lst.touched_patients() if touched is None else set(touched)
    if ext.patient_id.isin(gone).any() or ext.patient_id.isin(sel.iloc[:n0].patient_id).any() \
            or ext.patient_id.duplicated().any() or ext.uid.duplicated().any():
        raise RuntimeError("an extension patient is touched, already drawn or repeated")
    if os.path.exists(out_csv):
        raise RuntimeError(f"{out_csv} exists; never rewritten")
    ext.to_csv(out_csv + ".part", index=False)
    os.replace(out_csv + ".part", out_csv)
    return ext, hashlib.sha256(open(out_csv, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "extend"])
    ap.add_argument("--horizon", type=int, default=SCAN_HORIZON)
    ap.add_argument("--n-ext", type=int, default=N_EXT)
    a = ap.parse_args(argv)
    listing = pd.read_csv(os.path.join(META, lst.LISTING))
    if a.cmd == "extend":
        ext, sha = extend(listing, os.path.join(META, "c2_fresh_400.csv"), a.n_ext, os.path.join(META, EXT_CSV),
                          scan_csv=os.path.join(META, "c2_scan.csv"))
        print(json.dumps(dict(extension=len(ext), first_rank=int(ext["rank"].min()), last_rank=int(ext["rank"].max()),
                              dirs=ext.dir.value_counts().to_dict(), listing=EXT_CSV, sha256=sha)), flush=True)
        return
    sc = scan(listing, a.horizon)
    sel = select(listing, sc)
    path = os.path.join(META, "c2_fresh_400.csv")
    sel.to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    seen = sc[sc["rank"] <= sel["rank"].max()]
    print(json.dumps(dict(selected=len(sel), last_rank=int(sel["rank"].max()), scanned_segments=len(sc),
                          scanned_up_to_last=len(seen), status=seen.status.value_counts().to_dict(),
                          listing=os.path.relpath(path, ROOT),
                          sha256=hashlib.sha256(open(path, "rb").read()).hexdigest()), indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
