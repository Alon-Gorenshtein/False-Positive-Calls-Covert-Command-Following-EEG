"""Claassen 2019 on the six sites of the prolonged-DoC recordings (F3 F4 C3 C4 O1 O2), the reduced-montage rule every
whole-head analysis follows there (bench.wrappers.pdoc_overrides: all six sites; Laplacians over the six). The released
pipeline (112, frozen, not edited) runs with two module globals set for the call: USE_CH = the six sites in 112's
order and the spherical-spline CSD matrices rebuilt for them (112.csd_matrices); 6 sites x 4 bands = 24 features,
the same LinearSVC, leave-one-trial-out CV, permutation test and call.

clinical: the bridge. On the 481 clinical EEG recordings of 113, the six-site pipeline at 113's own window, schedule
  and seeds (test i: exact_test, RandomState(seed + i); test iii: trialflip_test on the randomized order,
  seeds + 20011 / + 30011), paired with 113's stored 19-site results. -> _long/claassen6_clin/rows<shard>.jsonl
doc: the prolonged-DoC null set through 114.run_recording (unchanged: first overlay, its randomization-based row), with
  Columbia admitted to the pDoC arm under the six-site set. -> _long/claassen6_doc/rows.jsonl
merge: clinical six-site and 19-site call rates (call_paper, the stated rule), exact McNemar of the pairs; DoC rates.
  -> output/tables/claassen_six.json
python 181_claassen_six.py clinical [--shard i --nshards n [--part j --nparts m]] [--limit N] | doc | merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import glob
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binom

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
os.chdir(CODE)
ls = importlib.import_module("178_long_schedules")
b = ls.b
wr = importlib.import_module("bench.wrappers")
ce = importlib.import_module("bench.motor_common").load_112()
cn = importlib.import_module("113_columbia_null")

SIX = [c for c in ce.USE_CH if c in ("F3", "F4", "C3", "C4", "O1", "O2")]
ROWS113 = os.path.join(ROOT, "output", "intermediate", "columbia_null_rows.parquet")
OUT = os.path.join(ROOT, "output", "tables", "claassen_six.json")


@contextlib.contextmanager
def six_sites():
    """112 on the six sites for the duration of the block (its globals restored after)."""
    use, csd = list(ce.USE_CH), ce._CSD
    ce.USE_CH[:] = SIX
    ce._CSD = ce.csd_matrices(names=SIX)
    try:
        yield
    finally:
        ce.USE_CH[:] = use
        ce._CSD = csd


@contextlib.contextmanager
def pdoc_admits_columbia():
    na, ch = dict(wr.PDOC_NA_MODULES), dict(wr.PDOC_CH)
    wr.PDOC_NA_MODULES.pop("columbia", None)
    wr.PDOC_CH["columbia"] = {"NATIVE_CH": list(SIX)}
    try:
        yield
    finally:
        wr.PDOC_NA_MODULES.clear()
        wr.PDOC_NA_MODULES.update(na)
        wr.PDOC_CH.clear()
        wr.PDOC_CH.update(ch)


def stem_of(source, i):
    lst = cn.SOURCES[source][2]
    return pd.read_csv(lst).file[int(i)].replace(".mat", "")


def clinical_one(row):
    """113.run_one's tests i and iii at its stored start and seeds, on the six sites."""
    sig, _, _, seed = cn.SOURCES[row["source"]]
    i = int(row["sample_idx"])
    z = np.load(f"{sig}/{stem_of(row['source'], i)}.npz")
    x, fs, names = np.asarray(z["x19"], np.float32), float(z["fs"]), [str(n) for n in z["names19"]]
    on, lab = ce.schedule(int(row["start_s"]))
    X, y, g = ce.features(x, fs, on, lab, names)
    r1 = ce.exact_test(X, y, g, np.random.RandomState(seed + i), stop_at=ce.STOP_AT)
    yr = cn.randomized_order(y, g, seed + 20011 + i)
    r3 = ce.trialflip_test(X, yr, g, np.random.RandomState(seed + 30011 + i), stop_at=ce.STOP_AT)
    out = {"source": row["source"], "sample_idx": i, "start_s": int(row["start_s"]), "n_features": int(X.shape[1])}
    for k, r in (("i", r1), ("iii", r3)):
        out |= {f"{k}_auc": r["auc"], f"{k}_p": r["p"], f"{k}_call": bool(r["call"]),
                f"{k}_call_paper": bool(r["call_paper"]), f"{k}_nperm": int(r["n_perm_run"])}
    return out


def clinical(shard=0, nshards=1, limit=None, part=0, nparts=1):
    """Rows k with k % nshards == shard; a shard is split again into nparts by (k // nshards) % nparts, a split that
    does not move when a part restarts, each part writing its own file (merge reads them all)."""
    d = pd.read_parquet(ROWS113).sort_values(["source", "sample_idx"]).reset_index(drop=True)
    fn = os.path.join(ls.LONG, "claassen6_clin", f"rows{shard:02d}" + (f"_p{part}" if nparts > 1 else "") + ".jsonl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = {(json.loads(l)["source"], json.loads(l)["sample_idx"])
            for f in glob.glob(os.path.join(os.path.dirname(fn), f"rows{shard:02d}*.jsonl"))
            if not os.path.basename(f).startswith("._") for l in open(f) if l.strip()}
    todo = [r for k, r in d.iterrows() if k % nshards == shard and (k // nshards) % nparts == part
            and (r["source"], int(r["sample_idx"])) not in done]
    with open(fn, "a") as fh, six_sites():
        for r in todo[:limit]:
            t = time.perf_counter()
            fh.write(json.dumps(clinical_one(r)) + "\n")
            fh.flush()
            print(f"{r['source']} {int(r['sample_idx'])} {time.perf_counter() - t:.1f} s", flush=True)


def doc(limit=None):
    spec = b.parse_null("ncmd_pdoc")
    mod = b.load_pipe("columbia")
    fn = os.path.join(ls.LONG, "claassen6_doc", "rows.jsonl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = {json.loads(l)["key"] for l in open(fn) if l.strip()} if os.path.exists(fn) else set()
    ctx = {"null": "ncmd_pdoc", "split": "analysis", "montage": "native", "resp": "none", "max_overlays": 1,
           "locator": None, "uncorrected": None, "col113": None, "spec_path": None}
    with open(fn, "a") as fh, six_sites(), pdoc_admits_columbia():
        def write(row):
            fh.write(json.dumps(row, allow_nan=False) + "\n")
            fh.flush()
            done.add(row["key"])
        for k, (rec, n) in enumerate(b.iter_recordings(spec, "analysis", limit, 0, 1, b.touched_uids())):
            t = time.perf_counter()
            b.run_recording(rec, mod, spec, {**ctx, "n_recs": n}, done, write)
            print(f"columbia pdoc {rec.uid} {time.perf_counter() - t:.1f} s", flush=True)


def _cell(k, n):
    st = importlib.import_module("143_stratum_table")
    return st.exact_cell(int(k), int(n))


def merge(out=OUT):
    six = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(os.path.join(ls.LONG, "claassen6_clin", "rows*.jsonl")))
                        if not os.path.basename(f).startswith("._") for l in open(f) if l.strip()])
    old = pd.read_parquet(ROWS113)
    m = old.merge(six, on=["source", "sample_idx", "start_s"], suffixes=("_19", "_6"))
    res = {"what": __doc__.split("\n")[0], "sites": SIX, "clinical": {"n_paired": int(len(m)), "n_listed": int(len(old))}}
    for t in ("i", "iii"):
        a, c = m[f"{t}_call_paper_19"].astype(bool), m[f"{t}_call_paper_6"].astype(bool)
        only19, only6 = int((a & ~c).sum()), int((~a & c).sum())
        res["clinical"][t] = {"sites19": _cell(a.sum(), len(m)), "sites6": _cell(c.sum(), len(m)),
                              "both": int((a & c).sum()), "only19": only19, "only6": only6,
                              "mcnemar_exact_p": float(min(1.0, 2 * binom.cdf(min(only19, only6), only19 + only6, 0.5)))
                              if only19 + only6 else 1.0}
    fn = os.path.join(ls.LONG, "claassen6_doc", "rows.jsonl")
    if os.path.exists(fn):
        d = pd.DataFrame([json.loads(l) for l in open(fn) if l.strip()])
        ok = d[(d["status"] == "ok") & (d["j"] == 0) & (d["variant"] == "as_coded")]
        res["doc"] = {"published": _cell(ok["x_call_paper"].astype(bool).sum(), len(ok)),
                      "randomized": _cell(ok["rand_call"].astype(bool).sum(), ok["rand_call"].notna().sum())
                      if "rand_call" in ok else None,
                      "status": d.groupby("uid")["status"].first().value_counts().to_dict()}
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clinical", "doc", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "clinical":
        clinical(a.shard, a.nshards, a.limit, a.part, a.nparts)
    elif a.cmd == "doc":
        doc(a.limit)
    else:
        merge()


if __name__ == "__main__":
    main()
