"""BDSP SpikeNet2 clinical EEG (MGH, credentialed), streamed read-only from the DGX Spark.
BDSP SpikeNet2 clinical EEG (MGH, credentialed), streamed read-only from the DGX Spark for the check in
. The Spark copy is READ-ONLY: only `find`, `cat` and a read-only h5py extraction that writes
to stdout (fetch_extract) run there; every byte lands on the SSD and raw files are deleted after extraction.
python bdsp_stream.py   -> writes ../output/bdsp/meta/spark_listing.txt and sample_500.csv (deterministic)"""
import os
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

HOST = "dgx-spark"
ROOT = "Desktop/spikenet2data"
DIRS = ("EEGs", "negative_EEGs")
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST]
META = "../output/bdsp/meta"
CHANNELS = ("Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
            "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG")
MIN_BYTES = 10_000_000
_NAME = re.compile(r"^Bonobo\d+_\d+\.mat$")


def patient_id(fname):
    """Bonobo00001 from Bonobo00001_0.mat: the stem before the last underscore."""
    return os.path.basename(fname).rsplit(".", 1)[0].rsplit("_", 1)[0]


def parse_listing(lines):
    """[(dir, file, bytes)] for .mat lines of `find -printf '%h %f %s'` relative to ROOT."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] in DIRS and _NAME.match(parts[1]):
            out.append((parts[0], parts[1], int(parts[2])))
    return out


def list_files():
    cmd = " ".join(f"{ROOT}/{d}" for d in DIRS)
    res = subprocess.run(SSH + [f"find {cmd} -maxdepth 1 -name '*.mat' -printf '%h %f %s\\n'"],
                         capture_output=True, text=True, check=True)
    return [ln.replace(f"{ROOT}/", "", 1) for ln in res.stdout.splitlines()]


def choose_sample(n=500, seed=20261018, listing=None, min_bytes=MIN_BYTES):
    """One segment per patient from the union of EEGs/ and negative_EEGs/. Patients are drawn without replacement
    from the sorted eligible IDs, then one of each patient's eligible segments, both from one seeded generator."""
    rows = parse_listing(listing if listing is not None else list_files())
    by_pid = {}
    for d, f, size in rows:
        if size >= min_bytes:
            by_pid.setdefault(patient_id(f), []).append((d, f, size))
    pids = sorted(by_pid)
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(pids), size=min(n, len(pids)), replace=False):
        segs = sorted(by_pid[pids[i]])
        d, f, size = segs[rng.integers(len(segs))]
        out.append({"patient_id": pids[i], "dir": d, "file": f, "bytes": size, "n_eligible_segments": len(segs)})
    return pd.DataFrame(out)


def fetch(file, dest_dir, size=None, retries=3, ssh=SSH):
    """Stream one Spark file (e.g. 'EEGs/Bonobo00001_0.mat') with `ssh cat` into dest_dir; check its byte size.
    Returns (local path, seconds). Writes to .part and renames, so a partial file is never mistaken for a good one."""
    d, f = file.split("/")
    if d not in DIRS or not _NAME.match(f):
        raise ValueError(f"refusing unexpected path {file!r}")
    os.makedirs(dest_dir, exist_ok=True)
    dest, part = os.path.join(dest_dir, f), os.path.join(dest_dir, f + ".part")
    if size is None:
        res = subprocess.run(ssh + [f"find {ROOT}/{file} -printf '%s'"], capture_output=True, text=True, check=True)
        size = int(res.stdout.strip())
    err = None
    for attempt in range(retries):
        t0 = time.time()
        with open(part, "wb") as fh:
            res = subprocess.run(ssh + [f"cat {ROOT}/{file}"], stdout=fh, stderr=subprocess.PIPE)
        got = os.path.getsize(part)
        if res.returncode == 0 and got == size:
            os.replace(part, dest)
            return dest, time.time() - t0
        err = f"rc={res.returncode} bytes {got}/{size} {res.stderr.decode(errors='replace')[-200:]}"
        time.sleep(5 * (attempt + 1))
    os.remove(part)
    raise IOError(f"fetch {file} failed after {retries} tries: {err}")


def read(path):
    """(data float64 uV [channels x samples], fs, channel names) from a MATLAB v7.3 file. h5py sees MATLAB's
    column-major data as [samples x channels]; channel names are a cell array of char refs."""
    with h5py.File(path, "r") as f:
        data = f["data"][()].T
        fs = float(np.asarray(f["Fs"][()]).squeeze())
        names = ["".join(map(chr, f[r][()].flatten())).strip() for r in f["channels"][()].flatten()]
    if data.shape[0] != len(names):
        raise ValueError(f"{path}: {data.shape[0]} rows vs {len(names)} channel names")
    return data, fs, names


REMOTE_PY = "~/miniforge3/envs/SpikeNet2/bin/python"
SSH_STDIN = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST]
EXTRACT_SRC = r'''
import io
import sys

import h5py
import numpy as np

ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def payload(path, want, max_s):
    """x float32 [channels x samples] for the wanted channels (every channel, in file order, if want is empty) over
    the first max_s seconds, plus bdsp_features' unit/reference check computed on the float64 source."""
    with h5py.File(path, "r") as f:
        fs = float(np.asarray(f["Fs"][()]).squeeze())
        names = ["".join(map(chr, f[r][()].flatten())).strip() for r in f["channels"][()].flatten()]
        n_full = f["data"].shape[0]
        data = f["data"][:int(round(max_s * fs))].T
    idx = {ALIASES.get(n.upper(), n.upper()): i for i, n in enumerate(names)}
    want = list(want) or names
    missing = [w for w in want if ALIASES.get(w.upper(), w.upper()) not in idx]
    if missing:
        raise ValueError("missing channels %s" % missing)
    eeg = [i for i, n in enumerate(names) if n.upper() != "EKG"]
    return dict(x=data[[idx[ALIASES.get(w.upper(), w.upper())] for w in want]].astype(np.float32),
                names=np.array(want), fs=fs, dur_s=data.shape[1] / fs, full_dur_s=n_full / fs,
                all_names=np.array(names), med_abs=np.median(np.abs(data), axis=1),
                car_med_abs=np.median(np.abs(data[eeg].mean(axis=0))))


if __name__ == "__main__":
    buf = io.BytesIO()
    np.savez(buf, **payload(sys.argv[1], [w for w in sys.argv[2].split(",") if w], float(sys.argv[3])))
    sys.stdout.buffer.write(buf.getvalue())
'''


def payload_ok(path):
    try:
        with np.load(path) as z:
            return z["x"].ndim == 2 and z["x"].shape[1] > 0 and z["x"].shape[0] == len(z["names"])
    except Exception:
        return False


def fetch_extract(file, dest_dir, want=(), max_s=7200, retries=3, ssh=SSH_STDIN, remote_py=REMOTE_PY):
    """Server-side extraction of one Spark file into dest_dir/<stem>.npz (see EXTRACT_SRC); want= sends every
    channel. Returns (local path, seconds). Writes to .part and renames once the payload loads."""
    d, f = file.split("/")
    if d not in DIRS or not _NAME.match(f):
        raise ValueError(f"refusing unexpected path {file!r}")
    if not all(re.fullmatch(r"[A-Za-z0-9]+", w) for w in want):
        raise ValueError(f"refusing unexpected channel names {want!r}")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f[:-4] + ".npz")
    part = dest + ".part"
    cmd = (f"cd /tmp && PYTHONDONTWRITEBYTECODE=1 HDF5_USE_FILE_LOCKING=FALSE {remote_py} -B - "
           f"\"$HOME/{ROOT}/{file}\" '{','.join(want)}' {float(max_s)}")
    err = None
    for attempt in range(retries):
        t0 = time.time()
        with open(part, "wb") as fh:
            res = subprocess.run(ssh + [cmd], input=EXTRACT_SRC.encode(), stdout=fh, stderr=subprocess.PIPE)
        if res.returncode == 0 and payload_ok(part):
            os.replace(part, dest)
            return dest, time.time() - t0
        err = f"rc={res.returncode} {res.stderr.decode(errors='replace')[-300:]}"
        time.sleep(5 * (attempt + 1))
    os.remove(part)
    raise IOError(f"extract {file} failed after {retries} tries: {err}")


def write_mat(path, data, fs, names):
    """A local file in the Spark's MATLAB v7.3 layout (data [samples x channels] float64, Fs, channels as a cell of
    char refs, MATLAB_class attributes), so Morgoth's own loader reads a streamed payload unchanged."""
    with h5py.File(path + ".part", "w") as f:
        refs = []
        for i, n in enumerate(names):
            ds = f.create_dataset(f"#refs#/c{i}", data=np.array([[ord(c)] for c in n], dtype=np.uint16))
            ds.attrs["MATLAB_class"] = np.bytes_("char")
            ds.attrs["MATLAB_int_decode"] = np.int32(2)
            refs.append(ds.ref)
        f.create_dataset("channels", data=np.array([refs], dtype=h5py.ref_dtype)).attrs["MATLAB_class"] = \
            np.bytes_("cell")
        f.create_dataset("Fs", data=np.array([[float(fs)]])).attrs["MATLAB_class"] = np.bytes_("double")
        f.create_dataset("data", data=np.ascontiguousarray(np.asarray(data, dtype=np.float64).T)).attrs[
            "MATLAB_class"] = np.bytes_("double")
    os.replace(path + ".part", path)
    return path


def main():
    os.makedirs(META, exist_ok=True)
    listing = list_files()
    with open(f"{META}/spark_listing.txt", "w") as fh:
        fh.write("\n".join(listing) + "\n")
    s = choose_sample(listing=listing)
    s.to_csv(f"{META}/sample_500.csv", index=False)
    print(f"{len(parse_listing(listing))} files, {s.patient_id.nunique()} sampled patients, "
          f"{s.bytes.sum() / 1e9:.1f} GB to stream", flush=True)


if __name__ == "__main__":
    sys.exit(main())
