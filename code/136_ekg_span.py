"""Whole-span EKG of the 974 Phase-1 analysis recordings, for a within-recording exploratory analysis (added later;
discovery data already read, run without fixing on the lead's instruction). Built test-first
(code/test_ekg_span.py).

Pool and order: 134_ekg_phase1.pool() (engine analysis split minus 114.touched_uids(), 974, sorted by uid; position =
index). Per recording, from its cached npz: n = the length of its x19 (the span the benchmark sees: the source file from
sample 0, cut at the 2-h guard), and its source's EKG channel (exactly one, 131's rule; else no_ekg / several_ekg,
counted, nothing streamed). The EKG over samples [0, n) is read on the read-only Spark by 131's remote reader unchanged
(131_ekg_design.REMOTE_SRC and remote_cmd: ssh stdin, from /tmp, no bytecode, no HDF5 locking, h5py mode "r", stdout only;
nothing is written on the Spark), at most STREAMS at a time. Sample k of the EKG is sample k of the cached x19.
-> output/bdsp/ekg_full/<uid>.npz (ekg float32 [n], fs, start_s = 0, n_samples, channel, uid) and manifest.csv
(position, uid, patient, source, dir, file, status, n_samples, channel, fs, ekg_sha256 of the array bytes), whose
sha256 is printed. Resumable; atomic writes.
check: C3 over [0, n) of the first design recording through the same path equals its cached x19 C3, bit for bit.
python3 136_ekg_span.py run | check"""
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
ROOT = os.path.dirname(CODE)
OUT = os.path.join(ROOT, "output", "bdsp", "ekg_full")
ek = importlib.import_module("131_ekg_design")
ep = importlib.import_module("134_ekg_phase1")
STREAMS = 3
COLS = ["position", "uid", "patient", "source", "dir", "file", "status", "n_samples", "channel", "fs", "ekg_sha256"]


def x19_len(path):
    """Samples of the cached x19, from the .npy header inside the npz (the array is not loaded)."""
    with zipfile.ZipFile(path) as zf, zf.open("x19.npy") as fh:
        v = np.lib.format.read_magic(fh)
        shape = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)[0]
    if len(shape) != 2 or shape[0] != 19:
        raise ValueError(f"{path}: x19 shape {shape}")
    return int(shape[1])


def meta(path):
    """(n samples of x19, fs, EKG channel names) of one cached npz, reading only small members."""
    with np.load(path) as z:
        fs, names = float(z["fs"]), ek.ekg_channels(z["all_names"])
    return x19_len(path), fs, names


def fetch_span(d, f, chans, n, ssh=ek.SSH, root=ek.BDSP, py=ek.HOST_PY):
    """131's remote reader over samples [0, n) of the named channels: the loaded payload."""
    res = subprocess.run(ssh + [ek.remote_cmd(d, f, chans, 0, n, root, py)], input=ek.REMOTE_SRC.encode(),
                         capture_output=True)
    if res.returncode != 0:
        raise IOError(f"remote read failed (rc {res.returncode}): {res.stderr.decode(errors='replace')[-300:]}")
    with np.load(io.BytesIO(res.stdout)) as z:
        return {k: z[k] for k in z.files}


def plan(entries, src, meta_fn=meta):
    rows = []
    for pos, (uid, patient, source, path) in enumerate(entries):
        n, fs, ch = meta_fn(path)
        status = "no_ekg" if not ch else "several_ekg" if len(ch) > 1 else "ok"
        d, f = src[uid]
        rows.append(dict(position=pos, uid=uid, patient=patient, source=source, dir=d, file=f, status=status,
                         n_samples=n, channel="|".join(ch), fs=fs, ekg_sha256=""))
    return rows


def pull_one(r, out=OUT, **kw):
    dest = os.path.join(out, f"{r['uid']}.npz")
    if not os.path.exists(dest):
        p = fetch_span(r["dir"], r["file"], [r["channel"]], r["n_samples"], **kw)
        x = p["x"][0]
        if float(p["fs"]) != r["fs"] or str(p["names"][0]) != r["channel"] or x.shape != (r["n_samples"],) \
                or x.dtype != np.float32:
            raise ValueError(f"position {r['position']}: source differs from the cached recording")
        with open(dest + ".part", "wb") as fh:
            np.savez(fh, ekg=x, fs=r["fs"], start_s=0, n_samples=r["n_samples"], channel=r["channel"], uid=r["uid"])
        os.replace(dest + ".part", dest)
    with np.load(dest) as z:
        if int(z["n_samples"]) != r["n_samples"] or str(z["channel"]) != r["channel"] or len(z["ekg"]) != r["n_samples"]:
            raise ValueError(f"position {r['position']}: stored EKG does not match the plan")
        return hashlib.sha256(z["ekg"].tobytes()).hexdigest()


def run(entries, src, meta_fn=meta, out=OUT, streams=STREAMS, **kw):
    rows = plan(entries, src, meta_fn)
    os.makedirs(out, exist_ok=True)
    ok = [r for r in rows if r["status"] == "ok"]
    with ThreadPoolExecutor(streams) as ex:
        for r, sha in zip(ok, ex.map(lambda r: pull_one(r, out, **kw), ok)):
            r["ekg_sha256"] = sha
    man = os.path.join(out, "manifest.csv")
    pd.DataFrame(rows, columns=COLS).to_csv(man + ".part", index=False)
    os.replace(man + ".part", man)
    return pd.DataFrame(rows, columns=COLS), hashlib.sha256(open(man, "rb").read()).hexdigest()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    entries, src = ep.pool(), ep.sources()
    if argv == ["run"]:
        df, sha = run(entries, src)
        files = [os.path.join(OUT, f"{u}.npz") for u in df.uid[df.status == "ok"]]
        print(json.dumps(dict(recordings=len(df), status=Counter(df.status), fs=Counter(df.fs[df.status == "ok"]),
                              gb=round(sum(map(os.path.getsize, files)) / 1e9, 4), manifest_sha256=sha)), flush=True)
    elif argv == ["check"]:
        uid, _, source, path = next(e for e in entries if e[2] == "design")
        d, f = src[uid]
        n = x19_len(path)
        p = fetch_span(d, f, ["C3"], n)
        with np.load(path) as z:
            c3 = z["x19"][list(z["names19"]).index("C3")]
        same = p["x"][0].tobytes() == c3.tobytes()
        print(json.dumps(dict(c3_bit_identical_from_sample_0=bool(same), n_samples=n)), flush=True)
        if not same:
            sys.exit(1)
    else:
        raise SystemExit("usage: 136_ekg_span.py run | check")


if __name__ == "__main__":
    sys.exit(main())
