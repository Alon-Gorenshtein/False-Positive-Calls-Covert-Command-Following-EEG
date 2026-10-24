"""C2 confirmation null rows (part C, C2; added later): the rows 114
--resp none gives, on the fresh seeded BDSP windows of 125's draw. Built test-first (code/test_127_fresh_rows.py).

The frozen runner is imported, never edited: every row comes from 114.run_recording on a recording loaded by
engine._load, so the frozen-code hash (bench.locked_spec.code_hash) stays FROZEN.
- Null sets: ncmd_bdsp_fresh (cornell, cruse11, cruse12p, edlow17) and nstim0_bdsp_fresh (mbvt, chennu13, gibson16);
  Columbia and du Bois are omitted for compute (C2), mbmi and sokoliuk21 are not in C2. The spec's source is "bdsp",
  so the BDSP rules apply unchanged: the 19-channel montage, Morgoth strata, the shortened BDSP twins, chennu13's
  12-block primary and mindBEAGLE's 10 tiles per recording. Rows say null *_bdsp_fresh and split c2_fresh.
- Every variant and the randomized method per timing group on every sequence overlay, plus the extra overlays up to
  m = min(10, ceil(200 / windows listed)) (114.m_overlays; n_recs = the listed windows).
- Windows: <root>/sig/<uid>.npz (x19, names19, fs; the fixed 30-min window is the whole array) and
  <root>/morgoth/<uid>.parquet (rebased to the window start), in the format of output/bdsp/{sig,morgoth}; listed
  by the fixed manifest c2_fresh_400.csv (rank, patient_id, uid, ...), taken in rank order. The manifest's
  sha256 must equal the fixed one. Refused: a uid or patient listed twice, a missing file, a patient in
  sample_500.csv, bench_extra_500.csv, train_600.csv or touched_before_freeze.json (a missing list is an error).
- Rows go to output/intermediate/_fresh_shards/<null>__<pipe>__native__none/rows<shard>.jsonl, appended to and
  resumable (114._done: a shard file holding rows of another split is refused); never under _bench_shards.
python3 127_fresh_rows.py --pipe NAME --root DIR --manifest CSV --sha256 HEX [--shard i --nshards n] [--limit N]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib
import json
import re
import sys
import time
import warnings

import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
b = importlib.import_module("114_benchmark")

FROZEN = "415921a4bde6ca83b47691cb7ab1b7713c25691029b604541312a932c598cf45"
SPLIT = "c2_fresh"
SHARDS = os.path.join(ROOT, "output", "intermediate", "_fresh_shards")
FRESH_PIPES = {"cornell": "ncmd", "cruse11": "ncmd", "cruse12p": "ncmd", "edlow17": "ncmd",
               "mbvt": "nstim0", "chennu13": "nstim0", "gibson16": "nstim0"}
EXCLUDE_CSVS = ("sample_500.csv", "bench_extra_500.csv", "train_600.csv")
_PATIENT = re.compile(r"^(Bonobo\d+)_\d+")


def fresh_spec(pipe):
    """114's null spec for the pipeline's fresh null set: source "bdsp", so the BDSP rules apply as written."""
    if pipe not in FRESH_PIPES:
        raise ValueError(f"{pipe} is not a C2 pipeline ({', '.join(FRESH_PIPES)})")
    kind = FRESH_PIPES[pipe]
    return {"name": f"{kind}_bdsp_fresh", "kind": kind, "family": "cmd" if kind == "ncmd" else "erp",
            "source": "bdsp"}


def shard_file(pipe, shard, root=SHARDS):
    root = os.path.abspath(root)
    for sh in b.SH.values():
        sh = os.path.abspath(sh)
        if root == sh or root.startswith(sh + os.sep):
            raise ValueError(f"{root}: fresh rows never go under {sh}")
    return os.path.join(root, f"{fresh_spec(pipe)['name']}__{pipe}__native__none", f"rows{shard:02d}.jsonl")


def check_sha256(path, want):
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        raise RuntimeError(f"{path}: sha256 {got} is not the pinned {want}")
    return got


def excluded_patients(meta=None):
    """Patients of the design sample, the extras, the training pool and the recordings touched before the freeze."""
    meta = meta or b.en.META
    out = set()
    for name in EXCLUDE_CSVS:
        path = os.path.join(meta, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"exclusion list {name} is missing")
        out |= set(pd.read_csv(path, usecols=["patient_id"]).patient_id.astype(str))
    path = os.path.join(meta, "touched_before_freeze.json")
    if not os.path.exists(path):
        raise FileNotFoundError("exclusion list touched_before_freeze.json is missing")
    with open(path) as fh:
        t = json.load(fh)
    for stem in list(t.get("excluded", [])) + list(t.get("loaded_only_not_excluded", [])):
        m = _PATIENT.match(str(stem))
        if m:
            out.add(m.group(1))
    return out


def fresh_items(root, manifest, excluded):
    """Engine descriptors of the listed windows in rank order (no signal loaded)."""
    df = pd.read_csv(manifest, dtype={"patient_id": str, "uid": str}).sort_values("rank", kind="stable")
    for col in ("uid", "patient_id"):
        if df[col].duplicated().any():
            raise RuntimeError(f"{manifest}: a {col} is listed twice")
    bad = sorted(set(df.patient_id) & set(excluded))
    if bad:
        raise RuntimeError(f"{len(bad)} listed patient(s) were touched by this study before")
    out = []
    for r in df.itertuples(index=False):
        it = dict(source="bdsp", path=os.path.join(str(root), "sig", f"{r.uid}.npz"),
                  morgoth=os.path.join(str(root), "morgoth", f"{r.uid}.parquet"), uid=r.uid, patient=r.patient_id,
                  meta={"dev": False, "fresh": True, "fresh_rank": int(r.rank)})
        for k in ("path", "morgoth"):
            if not os.path.exists(it[k]):
                raise FileNotFoundError(it[k])
        out.append(it)
    return out


def run(pipe, items, n_recs, shard=0, nshards=1, limit=None, root=SHARDS):
    """114 --resp none rows of this shard's windows (every nshards-th, from shard) -> the shard file's path."""
    spec = fresh_spec(pipe)
    mod = b.load_pipe(pipe)
    fn = shard_file(pipe, shard, root)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    for f in os.listdir(os.path.dirname(fn)):
        if f.endswith(".jsonl"):
            b._done(os.path.join(os.path.dirname(fn), f), SPLIT)
    done = b._done(fn, SPLIT)
    ctx = {"null": spec["name"], "split": SPLIT, "montage": "native", "resp": "none", "max_overlays": None,
           "locator": None, "uncorrected": None, "col113": None, "spec_path": None, "n_recs": n_recs}
    with open(fn, "a") as fh:
        def write(row):
            fh.write(json.dumps(row, allow_nan=False) + "\n")
            fh.flush()
            done.add(row["key"])
        for i, it in enumerate(items[:limit]):
            if i % nshards != shard:
                continue
            t = time.perf_counter()
            rec = b.en._load(it)
            if rec.meta.get("dev") is True:
                raise RuntimeError(f"{rec.uid}: development recording refused")
            b.run_recording(rec, mod, spec, ctx, done, write)
            print(f"{spec['name']} {pipe} {rec.uid} done in {time.perf_counter() - t:.1f} s", flush=True)
            del rec
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", required=True, choices=sorted(FRESH_PIPES))
    ap.add_argument("--root", required=True, help="folder holding sig/ and morgoth/")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sha256", required=True, help="the manifest's pinned sha256")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.chdir(CODE)
    check_sha256(a.manifest, a.sha256)
    items = fresh_items(a.root, a.manifest, excluded_patients())
    run(a.pipe, items, len(items), a.shard, a.nshards, a.limit)


if __name__ == "__main__":
    main()
