"""C2 fresh-patient draw (part C, C2; added later): the seeded, ordered
candidate listing of BDSP patients disjoint from every patient this study has touched. The confirmation takes the first
C2_N (400) of them, in rank order, that hold a 30-min eligible window. Built test-first (code/test_c2_listing.py);
reads metadata only, never a signal.

- Pool: the read-only Spark listing output/bdsp/meta/spark_listing.txt ("dir file bytes"; the EEGs/ and
  negative_EEGs/ .mat files; re-listed with `find` on 2026-10-16 and identical).
- Excluded patients: every patient in sample_500.csv (the design sample, with its dev split), bench_extra_500.csv (the
  extras: the analysis extras, the A4 confirmation set and the forecaster's confirmation list, byte-identical to
  confirm_500.csv), train_600.csv (the forecaster training pool) and touched_before_freeze.json, plus every patient
  with any file named Bonobo<id>_<n>* anywhere under output/ (e.g. the Morgoth reference recording). A missing
  exclusion file is an error.
- Candidate segments: files of >= MIN_BYTES_30MIN bytes. A 30-min segment is 1800 s x 128 Hz x 20 channels x 8 bytes
  = 36.9 MB raw, and none of the 1,600 files pulled so far compressed below 0.515 of its raw size, so a file under
  half of that cannot hold 30 min. A patient with no such file is not a candidate.
- Order: np.random.default_rng([SEED, 2]) permutes the sorted candidate patient IDs (rank 0, 1, ...); then, walking the
  ranks, the same generator permutes each patient's candidate segments (seg_rank; sorted by dir, file first).
- Rule applied at extraction (fixed here): walking the ranks, a patient is taken with the first of its segments, in
  seg_rank order, that holds a 1800-s window admitted by 108.eligible_starts applied as the benchmark engine applies it
  (on all 19 EEG channels of x19: no padding within the window or 30 s around it). The window start is drawn uniformly
  from that segment's eligible starts by np.random.default_rng([SEED, 2, rank]). Stop at C2_N patients. On the design
  and extra recordings 31.6% of segments hold such a window (19 or 4 overlay channels alike).
-> output/bdsp/meta/c2_fresh_listing.csv (rank, patient_id, seg_rank, dir, file, bytes) and its sha256
python3 125_c2_listing.py"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
META = os.path.join(ROOT, "output", "bdsp", "meta")
OUTPUT = os.path.join(ROOT, "output")
LISTING = "c2_fresh_listing.csv"
SEED, STREAM = 20261016, 2
C2_N = 400
WINDOW_S = 1800
MIN_BYTES_30MIN = int(0.5 * WINDOW_S * 128 * 20 * 8)
EXCLUDE_CSVS = ("sample_500.csv", "bench_extra_500.csv", "train_600.csv")
_FILE = re.compile(r"^(Bonobo\d+)_\d+\.mat$")
_TOUCHED = re.compile(r"^(Bonobo\d+)_\d+")


def touched_patients(meta=META, output=OUTPUT):
    """Patient IDs in the exclusion lists, touched_before_freeze.json and any Bonobo-named file under output/."""
    out = set()
    for name in EXCLUDE_CSVS:
        path = os.path.join(meta, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"exclusion list {name} is missing")
        out |= set(pd.read_csv(path, usecols=["patient_id"]).patient_id.astype(str))
    path = os.path.join(meta, "touched_before_freeze.json")
    if not os.path.exists(path):
        raise FileNotFoundError("exclusion list touched_before_freeze.json is missing")
    t = json.load(open(path))
    for stem in list(t.get("excluded", [])) + list(t.get("loaded_only_not_excluded", [])):
        m = _TOUCHED.match(str(stem))
        if m:
            out.add(m.group(1))
    for _, _, files in os.walk(output):
        for f in files:
            m = _TOUCHED.match(f)
            if m and not f.startswith("._"):
                out.add(m.group(1))
    return out


def listing(meta=META, output=OUTPUT):
    """(ordered candidate listing, counts)."""
    rows = []
    for line in open(os.path.join(meta, "spark_listing.txt")):
        parts = line.split()
        if len(parts) >= 3 and parts[0] in ("EEGs", "negative_EEGs") and _FILE.match(parts[1]):
            rows.append((parts[0], parts[1], int(parts[2]), _FILE.match(parts[1]).group(1)))
    every = {pid for *_, pid in rows}
    gone = touched_patients(meta, output)
    segs = {}
    for d, f, size, pid in rows:
        if pid not in gone and size >= MIN_BYTES_30MIN:
            segs.setdefault(pid, []).append((d, f, size))
    pids = sorted(segs)
    rng = np.random.default_rng([SEED, STREAM])
    out = []
    for rank, i in enumerate(rng.permutation(len(pids))):
        mine = sorted(segs[pids[i]])
        for k, j in enumerate(rng.permutation(len(mine))):
            d, f, size = mine[j]
            out.append(dict(rank=rank, patient_id=pids[i], seg_rank=k, dir=d, file=f, bytes=size))
    info = dict(listed_files=len(rows), listed_patients=len(every), excluded_patients=len(every & gone),
                pool_patients=len(every - gone), patients_without_a_30min_file=len(every - gone) - len(pids),
                candidate_patients=len(pids), candidate_segments=len(out))
    return pd.DataFrame(out, columns=["rank", "patient_id", "seg_rank", "dir", "file", "bytes"]), info


def main():
    df, info = listing()
    path = os.path.join(META, LISTING)
    df.to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(json.dumps(dict(info, listing=os.path.relpath(path, ROOT), sha256=sha), indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
