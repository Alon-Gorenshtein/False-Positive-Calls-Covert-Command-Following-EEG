"""EKG of the fresh 400 for the arousal analysis. Runs on the DGX Spark, where the C2 windows live and where 133 runs;
only result tables leave the Spark. Built test-first (code/test_c2_ekg.py).

run: in the morgoth:spark container (no network; BDSP folders mounted READ-ONLY; ~/work/c2_fresh writable), for each
row of a fixed list (c2_fresh_400.csv, sha256 PINNED_LIST, or its extension c2_fresh_ext.csv, sha256 EXT_LIST
once fixed; any other list is refused) the source file's
single EKG channel over [window_start_s, window_start_s + 1800 s) at its native rate as float32, read by the same
window() as the check extractor (131_ekg_design.REMOTE_SRC; h5py mode "r"). The start must equal the window_start_s of
the stored C2 window (c2/sig/<uid>.npz), so sample k of the EKG is sample k of that window's x19.
-> <base>/ekg/<uid>.npz (ekg float32 [samples], fs, start_s, channel, uid) and <base>/ekg/manifest.csv (rank,
patient_id, uid, start_s, status, channel, fs, n_samples, ekg_sha256). status: ok | no_ekg | several_ekg (no file;
D1 counts these NA; the fresh 400 are never replaced). Resumable; atomic writes.
design-check <path> <start_s>: prints the sha256 of one file's EKG window and writes nothing (bit-identity against
the check copy of a DESIGN recording).
python -B c2_ekg.py run <base> <bdsp root> <list> | design-check <path> <start_s>"""
import hashlib
import importlib
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
ek = importlib.import_module("131_ekg_design")
PINNED_LIST = "216c6a2977264d2c3f5e354df6185daa83782a69a7e176de8fbb1ede92c45d39"
EXT_LIST = None
WINDOW_S = ek.WINDOW_S
MANIFEST_COLS = ["rank", "patient_id", "uid", "start_s", "status", "channel", "fs", "n_samples", "ekg_sha256"]
_ns = {"__name__": "c2_ekg_window"}
exec(compile(ek.REMOTE_SRC, "131_ekg_design.REMOTE_SRC", "exec"), _ns)
window = _ns["window"]


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pinned_lists():
    return {PINNED_LIST} | ({EXT_LIST} if EXT_LIST else set())


def pinned_list(path, expect=None):
    """The fixed list (the fresh 400 or, once fixed, its extension) as a frame; any other file is
    refused. expect: a sha256 or a set of them (default: pinned_lists())."""
    import pandas as pd
    allowed = pinned_lists() if expect is None else {expect} if isinstance(expect, str) else set(expect)
    if sha256_file(path) not in allowed:
        raise RuntimeError(f"{path} is not the pinned C2 list (sha256)")
    return pd.read_csv(path)


def source_names(path):
    import h5py
    with h5py.File(path, "r") as f:
        return ["".join(map(chr, f[r][()].flatten())).strip() for r in f["channels"][()].flatten()]


def ekg_window(path, start_s):
    """(status, channel, ekg float32 or None, fs) of one source file's EKG over [start, start + WINDOW_S)."""
    ch = ek.ekg_channels(source_names(path))
    if len(ch) != 1:
        return ("no_ekg" if not ch else "several_ekg"), "|".join(ch), None, float("nan")
    import h5py
    with h5py.File(path, "r") as f:
        fs = float(np.asarray(f["Fs"][()]).squeeze())
    p = window(path, ch, int(round(start_s * fs)), int(round(WINDOW_S * fs)))
    if float(p["fs"]) != fs:
        raise ValueError(f"{path}: rate changed between reads")
    return "ok", ch[0], p["x"][0], fs


def run(base, bdsp, listing, expect=None):
    import pandas as pd
    rows = pinned_list(listing, expect)
    out = os.path.join(base, "ekg")
    os.makedirs(out, exist_ok=True)
    man = []
    for r in rows.itertuples():
        with np.load(os.path.join(base, "sig", f"{r.uid}.npz")) as z:
            if int(z["window_start_s"]) != int(r.window_start_s) or str(z["uid"]) != r.uid:
                raise ValueError(f"rank {r.rank}: stored C2 window does not match the pinned list")
        dest = os.path.join(out, f"{r.uid}.npz")
        if os.path.exists(dest):
            with np.load(dest) as z:
                status, channel, x, fs = "ok", str(z["channel"]), z["ekg"], float(z["fs"])
        else:
            status, channel, x, fs = ekg_window(os.path.join(bdsp, r.dir, r.file), int(r.window_start_s))
            if status == "ok":
                with open(dest + ".part", "wb") as fh:
                    np.savez(fh, ekg=x, fs=fs, start_s=int(r.window_start_s), channel=channel, uid=r.uid)
                os.replace(dest + ".part", dest)
        man.append(dict(rank=r.rank, patient_id=r.patient_id, uid=r.uid, start_s=int(r.window_start_s), status=status,
                        channel=channel, fs=fs, n_samples=0 if x is None else len(x),
                        ekg_sha256="" if x is None else hashlib.sha256(x.tobytes()).hexdigest()))
        print("EKG", r.rank, status, flush=True)
    path = os.path.join(out, "manifest.csv")
    pd.DataFrame(man, columns=MANIFEST_COLS).to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    print("MANIFEST", len(man), sum(m["status"] == "ok" for m in man), sha256_file(path), flush=True)


def main(argv):
    if argv[0] == "run":
        run(*argv[1:4])
    elif argv[0] == "design-check":
        status, channel, x, fs = ekg_window(argv[1], int(argv[2]))
        print("CHECK", status, fs, None if x is None else (x.dtype, x.shape, hashlib.sha256(x.tobytes()).hexdigest()),
              flush=True)
    else:
        raise SystemExit(f"unknown command {argv[0]!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
