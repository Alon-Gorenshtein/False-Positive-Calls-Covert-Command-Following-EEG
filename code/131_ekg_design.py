"""EKG for the arousal check (design data only).
EKG for the arousal check (d8f1033 and its 2026-10-16 clarification; part D
design phase, added later; design data only). Built test-first (code/test_ekg_design.py).

select (local npz metadata only): pool = the 500 sample_500 recordings (dev included); order =
np.random.default_rng([SEED, STREAM]).permutation of the sorted uids; walking it, a recording is taken if its source
lists exactly one EKG channel (all_names of the cached npz, read from the source file at the design pull) and its cached
x19 holds an 1800-s window by 108.eligible_starts (19 channels, as the engine applies it); start =
np.random.default_rng([SEED, STREAM, position]).choice(eligible starts); failures counted by reason; stop at N.
-> output/bdsp/ekg_design/manifest.csv (position, uid, dir, file, start_s, n_eligible_starts, channel, fs) and its sha256.

pull: per manifest row, the Spark sends only that channel over [start, start + 1800 s) at its native rate as float32
(REMOTE_SRC over ssh stdin, run from /tmp with no bytecode and no HDF5 locking; h5py mode "r"; stdout only; nothing is
written on the Spark). The source's channel name and rate must match the manifest. -> output/bdsp/ekg_design/<uid>.npz
(ekg float32 [samples], fs, start_s, channel, uid). Resumable; atomic writes.

check: the same remote read of one manifest recording's C3 over its window must equal the cached x19 C3 slice bit for
bit (time alignment of the stream with the cached x19).
python3 131_ekg_design.py select | pull | check"""
import argparse
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
META = os.path.join(ROOT, "output", "bdsp", "meta")
SIG = os.path.join(ROOT, "output", "bdsp", "sig")
OUT = os.path.join(ROOT, "output", "bdsp", "ekg_design")
SEED, STREAM, N = 20261016, 3, 100
WINDOW_S = 1800
EKG = re.compile(r"EKG|ECG", re.I)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "dgx-spark"]
BDSP = "$HOME/Desktop/spikenet2data"
HOST_PY = "~/miniforge3/envs/SpikeNet2/bin/python"
_NAME = re.compile(r"^Bonobo\d+_\d+\.mat$")
MANIFEST_COLS = ["position", "uid", "dir", "file", "start_s", "n_eligible_starts", "channel", "fs"]

REMOTE_SRC = r'''
import io
import sys

import h5py
import numpy as np


def window(path, chans, start, n):
    """float32 [len(chans) x n] of the named channels over samples [start, start + n) of a v7.3 file, read-only."""
    with h5py.File(path, "r") as f:
        fs = float(np.asarray(f["Fs"][()]).squeeze())
        names = ["".join(map(chr, f[r][()].flatten())).strip() for r in f["channels"][()].flatten()]
        n_full = f["data"].shape[0]
        if start < 0 or start + n > n_full:
            raise ValueError("window outside the recording")
        idx = [names.index(c) for c in chans]
        x = np.stack([f["data"][start:start + n, i] for i in idx]).astype(np.float32)
    return dict(x=x, names=np.array(chans), fs=fs, n_full=n_full, all_names=np.array(names))


if __name__ == "__main__":
    buf = io.BytesIO()
    np.savez(buf, **window(sys.argv[1], sys.argv[2].split("|"), int(sys.argv[3]), int(sys.argv[4])))
    sys.stdout.buffer.write(buf.getvalue())
'''


def ekg_channels(names):
    return [str(n) for n in names if EKG.search(str(n))]


def select(uids, load, n=N):
    """(manifest rows, failure counts). load(uid) -> dict with all_names, x19, fs (a cached design npz)."""
    bn = importlib.import_module("108_bdsp_null")
    order = np.random.default_rng([SEED, STREAM]).permutation(len(uids))
    rows, fail = [], Counter()
    for pos, i in enumerate(order):
        if len(rows) == n:
            break
        z = load(uids[i])
        ekg = ekg_channels(z["all_names"])
        if len(ekg) != 1:
            fail["no_ekg" if not ekg else "several_ekg"] += 1
            continue
        el = bn.eligible_starts(np.asarray(z["x19"], np.float32), float(z["fs"]), run_s=WINDOW_S)
        if not len(el):
            fail["no_30min_window"] += 1
            continue
        start = int(np.random.default_rng([SEED, STREAM, pos]).choice(el))
        rows.append(dict(position=pos, uid=uids[i], start_s=start, n_eligible_starts=len(el), channel=ekg[0],
                         fs=float(z["fs"])))
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} usable recordings in the pool")
    return rows, dict(fail)


def remote_cmd(d, f, chans, start, n, root=BDSP, py=HOST_PY):
    if d not in ("EEGs", "negative_EEGs") or not _NAME.match(f):
        raise ValueError(f"refusing unexpected path {d}/{f}")
    if not all(re.fullmatch(r"[A-Za-z0-9 +\-]+", c) for c in chans):
        raise ValueError(f"refusing unexpected channel names {chans!r}")
    q = "|".join(chans)
    return (f"cd /tmp && PYTHONDONTWRITEBYTECODE=1 HDF5_USE_FILE_LOCKING=FALSE {py} -B - "
            f"\"{root}/{d}/{f}\" '{q}' {int(start)} {int(n)}")


def fetch(d, f, chans, start_s, fs, ssh=SSH, root=BDSP, py=HOST_PY):
    """The remote window as a loaded payload dict."""
    a, n = int(round(start_s * fs)), int(round(WINDOW_S * fs))
    res = subprocess.run(ssh + [remote_cmd(d, f, chans, a, n, root, py)], input=REMOTE_SRC.encode(),
                         capture_output=True)
    if res.returncode != 0:
        raise IOError(f"remote read failed (rc {res.returncode}): {res.stderr.decode(errors='replace')[-300:]}")
    with np.load(io.BytesIO(res.stdout)) as z:
        return {k: z[k] for k in z.files}


def pull_one(r, out=OUT, **kw):
    dest = os.path.join(out, f"{r.uid}.npz")
    if os.path.exists(dest):
        return dest, False
    p = fetch(r.dir, r.file, [r.channel], r.start_s, r.fs, **kw)
    if float(p["fs"]) != float(r.fs) or str(p["names"][0]) != r.channel:
        raise ValueError(f"{r.uid}: source channel or rate differs from the manifest")
    x = p["x"][0]
    if x.shape != (int(round(WINDOW_S * r.fs)),) or x.dtype != np.float32:
        raise ValueError(f"{r.uid}: unexpected EKG shape {x.shape} {x.dtype}")
    os.makedirs(out, exist_ok=True)
    with open(dest + ".part", "wb") as fh:
        np.savez(fh, ekg=x, fs=float(p["fs"]), start_s=int(r.start_s), channel=r.channel, uid=r.uid)
    os.replace(dest + ".part", dest)
    return dest, True


def _load_npz(uid, sig=SIG):
    with np.load(os.path.join(sig, f"{uid}.npz")) as z:
        return {"all_names": z["all_names"], "x19": z["x19"], "fs": z["fs"]}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "pull", "check"])
    a = ap.parse_args(argv)
    man = os.path.join(OUT, "manifest.csv")
    if a.cmd == "select":
        smp = pd.read_csv(os.path.join(META, "sample_500.csv"))
        loc = {f[:-4]: (d, f) for d, f in zip(smp.dir, smp.file)}
        rows, fail = select(sorted(loc), _load_npz)
        df = pd.DataFrame(rows)
        df["dir"], df["file"] = [loc[u][0] for u in df.uid], [loc[u][1] for u in df.uid]
        os.makedirs(OUT, exist_ok=True)
        df[MANIFEST_COLS].to_csv(man + ".part", index=False)
        os.replace(man + ".part", man)
        print(json.dumps(dict(selected=len(df), last_position=int(df.position.max()), failures=fail,
                              channels=df.channel.value_counts().to_dict(), fs=df.fs.value_counts().to_dict(),
                              manifest_sha256=sha256(man))), flush=True)
    elif a.cmd == "pull":
        df = pd.read_csv(man)
        new = sum(pull_one(r)[1] for r in df.itertuples())
        files = [os.path.join(OUT, f"{u}.npz") for u in df.uid]
        print(json.dumps(dict(listed=len(df), present=sum(map(os.path.exists, files)), new=new,
                              gb=round(sum(os.path.getsize(f) for f in files) / 1e9, 4))), flush=True)
    else:
        r = next(pd.read_csv(man).itertuples())
        p = fetch(r.dir, r.file, ["C3"], r.start_s, r.fs)
        with np.load(os.path.join(SIG, f"{r.uid}.npz")) as z:
            c3 = z["x19"][list(z["names19"]).index("C3")]
        a0 = int(round(r.start_s * r.fs))
        same = p["x"][0].tobytes() == c3[a0:a0 + int(round(WINDOW_S * r.fs))].tobytes()
        print(json.dumps(dict(c3_bit_identical=bool(same))), flush=True)
        if not same:
            sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
