"""Phase-1 EKG for the hypothesis-generating part of the arousal analysis. Built test-first (code/test_ekg_phase1.py).

Pool: the BDSP analysis recordings of the benchmark (engine._items("bdsp", "analysis"), design-sample and extra
recordings) minus the recordings touched before the dev freeze (114.touched_uids()): 974, sorted by uid; any other
count is refused. position = index in that order. Per recording, from its cached npz (x19 as float32, all_names), in
this order (as 131.select; a recording failing both gets the EKG status):
- no_ekg / several_ekg: the source lists no or several EKG channels (counted, no EKG);
- no_window: no 1800-s window by 108.eligible_starts (19 channels, 30-s margin) in the cached x19 (counted, no EKG);
- ok: start = np.random.default_rng([SEED, STREAM, position]).choice(eligible starts); the EKG over
  [start, start + 1800 s) at its native rate as float32, streamed from the read-only Spark by 131's reader unchanged
  (131_ekg_design.pull_one: REMOTE_SRC over ssh stdin, h5py mode "r", stdout only; nothing is written on the Spark).
-> output/bdsp/ekg_phase1/<uid>.npz (131's format: ekg, fs, start_s, channel, uid) and manifest.csv (position, uid,
patient, source, dir, file, status, start_s, n_eligible_starts, channel, fs, ekg_sha256), whose sha256 is printed.
Resumable; atomic writes.
python3 134_ekg_phase1.py run"""
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
ROOT = os.path.dirname(CODE)
META = os.path.join(ROOT, "output", "bdsp", "meta")
OUT = os.path.join(ROOT, "output", "bdsp", "ekg_phase1")
ek = importlib.import_module("131_ekg_design")
SEED, STREAM, POOL_N = 20261016, 4, 974
WINDOW_S = ek.WINDOW_S
LISTS = ("sample_500.csv", "bench_extra_500.csv")
COLS = ["position", "uid", "patient", "source", "dir", "file", "status", "start_s", "n_eligible_starts", "channel",
        "fs", "ekg_sha256"]


def pool(n=POOL_N):
    """[(uid, patient, source, cached npz path)] sorted by uid: the analysis split minus the touched recordings."""
    from bench import engine as en
    touched = importlib.import_module("114_benchmark").touched_uids()
    items = [it for it in en._items("bdsp", "analysis") if it["uid"] not in touched]
    out = sorted((it["uid"], it["patient"], "extra" if it["meta"]["extra"] else "design", it["path"]) for it in items)
    if len(out) != n or len({u for u, *_ in out}) != n:
        raise RuntimeError(f"{len(out)} analysis recordings, not the pinned {n}")
    return out


def sources(meta=META):
    """uid -> (dir, file) of the source .mat, from the design and extra listings."""
    m = {}
    for name in LISTS:
        for d, f in pd.read_csv(os.path.join(meta, name), usecols=["dir", "file"]).itertuples(index=False):
            m[f[:-4]] = (d, f)
    return m


def plan(entries, src, load):
    """Manifest rows (ekg_sha256 empty) for [(uid, patient, source, path)] in position order. load(path) -> dict with
    all_names, x19, fs."""
    bn = importlib.import_module("108_bdsp_null")
    rows = []
    for pos, (uid, patient, source, path) in enumerate(entries):
        z = load(path)
        fs = float(z["fs"])
        el = bn.eligible_starts(np.asarray(z["x19"], np.float32), fs, run_s=WINDOW_S)
        ch = ek.ekg_channels(z["all_names"])
        status = "no_ekg" if not ch else "several_ekg" if len(ch) > 1 else "no_window" if not len(el) else "ok"
        start = int(np.random.default_rng([SEED, STREAM, pos]).choice(el)) if status == "ok" else None
        d, f = src[uid]
        rows.append(dict(position=pos, uid=uid, patient=patient, source=source, dir=d, file=f, status=status,
                         start_s=start, n_eligible_starts=len(el), channel="|".join(ch), fs=fs, ekg_sha256=""))
    return rows


def _load(path):
    with np.load(path) as z:
        return {"all_names": z["all_names"], "x19": z["x19"], "fs": z["fs"]}


def run(entries, src, load=_load, out=OUT, **fetch_kw):
    rows = plan(entries, src, load)
    for r in rows:
        if r["status"] != "ok":
            continue
        dest, _ = ek.pull_one(SimpleNamespace(uid=r["uid"], dir=r["dir"], file=r["file"], start_s=r["start_s"],
                                              fs=r["fs"], channel=r["channel"]), out=out, **fetch_kw)
        with np.load(dest) as z:
            if int(z["start_s"]) != r["start_s"] or str(z["channel"]) != r["channel"]:
                raise ValueError(f"position {r['position']}: stored EKG does not match the plan")
            r["ekg_sha256"] = hashlib.sha256(z["ekg"].tobytes()).hexdigest()
    os.makedirs(out, exist_ok=True)
    man = os.path.join(out, "manifest.csv")
    df = pd.DataFrame(rows, columns=COLS)
    df["start_s"] = df["start_s"].astype("Int64")
    df.to_csv(man + ".part", index=False)
    os.replace(man + ".part", man)
    return df, hashlib.sha256(open(man, "rb").read()).hexdigest()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["run"]:
        raise SystemExit("usage: 134_ekg_phase1.py run")
    df, sha = run(pool(), sources())
    files = [os.path.join(OUT, f"{u}.npz") for u in df.uid[df.status == "ok"]]
    print(json.dumps(dict(recordings=len(df), status=Counter(df.status), channels=Counter(df.channel[df.status == "ok"]),
                          fs=Counter(df.fs[df.status == "ok"]), gb=round(sum(map(os.path.getsize, files)) / 1e9, 4),
                          manifest_sha256=sha)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
