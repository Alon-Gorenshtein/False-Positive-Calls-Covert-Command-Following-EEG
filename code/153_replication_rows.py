"""Replication rows: the frozen primary reading of each analysis (replication_frozen.json) on the extension windows
drawn after the readings were fixed (126 extend: c2_fresh_ext.csv). 114 plans every window as in the development run
(every variant is listed, so overlay starts do not change); only the frozen variants' jobs run, without
randomization-based rows. The fidelity-corrected readings (Cornell, du Bois) run through 130.run on 130.phase1_jobs,
first overlay. Merge: the tests fixed in replication_frozen.json.
python3 153_replication_rows.py --pipe P [--shard i --nshards n] [--limit N] | --merge"""
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
b = importlib.import_module("114_benchmark")
fid = importlib.import_module("130_fidelity_readings")
fr = importlib.import_module("127_fresh_rows")
st = importlib.import_module("143_stratum_table")
sr = importlib.import_module("145_stated_rule")
fz = importlib.import_module("152_replication_freeze")
MEM = importlib.import_module("155_nominal_rates")

FROZEN_ALL = json.load(open(os.path.join(ROOT, "output", "tables", "replication_frozen.json")))
FROZEN = FROZEN_ALL["pipes"]
SPLIT = "c2_ext"
EXT_ROOT = os.path.join(ROOT, "output", "bdsp", "ext")
MANIFEST = os.path.join(ROOT, "output", "bdsp", "meta", "c2_fresh_ext.csv")
SHARDS = os.path.join(ROOT, "output", "intermediate", "_rep_shards")
OUT = os.path.join(ROOT, "output", "tables", "replication.json")
DEV_SIG = os.path.join(ROOT, "output", "bdsp")


def rep_spec(pipe):
    kind = FROZEN[pipe]["null"].split("_")[0]
    return {"name": f"{kind}_bdsp_rep", "kind": kind, "family": "cmd" if kind == "ncmd" else "erp", "source": "bdsp"}


def keep_114(pipe):
    v = FROZEN[pipe]["variant"]
    base = [] if v.startswith("c3_") else [v.split(":")[0]]
    return set(base) | set(FROZEN[pipe]["extra_variants"])


@contextlib.contextmanager
def only_variants(keep):
    jobs0, rand0 = b._jobs, b._rand_fields
    b._jobs = lambda *a: [x for x in jobs0(*a) if x[1] in keep]
    b._rand_fields = lambda *a, **k: {}
    try:
        yield
    finally:
        b._jobs, b._rand_fields = jobs0, rand0


def load_without_morgoth(it):
    """A clinical EEG window as the engine loads it, without Morgoth labels. No call reads them; only the stratum
    covariate does, and it is then 'unlabelled' (strata can be added later from the stored extents)."""
    z = np.load(it["path"])
    meta = dict(it["meta"], morgoth_t=np.array([], float), morgoth_lab=np.array([], object))
    return b.en.Recording(source="bdsp", uid=it["uid"], patient=it["patient"], x=np.asarray(z["x19"], np.float32),
                          fs=float(z["fs"]), names=[str(n) for n in z["names19"]], meta=meta)


def rows_for_item(item, pipe, n_recs, split=SPLIT, max_overlays=None):
    rec = b.en._load(item) if item.get("morgoth") else load_without_morgoth(item)
    out = []
    keep = keep_114(pipe)
    mod = b.load_pipe(pipe)
    spec = rep_spec(pipe)
    if keep:
        ctx = {"null": spec["name"], "split": split, "montage": "native", "resp": "none",
               "max_overlays": max_overlays or FROZEN[pipe]["max_overlays"], "locator": None, "uncorrected": None, "col113": None,
               "spec_path": None, "n_recs": n_recs}
        with only_variants(keep):
            b.run_recording(rec, mod, spec, ctx, set(), out.append)
    v = FROZEN[pipe]["variant"]
    if v.startswith("c3_"):
        jobs, why = fid.phase1_jobs(rec, mod, spec, n_recs, first_only=True)
        base_row = {"null": spec["name"], "pipe": pipe, "uid": rec.uid, "patient": rec.patient, "split": split}
        mine = [j for j in jobs if j["reading"] == v]
        if not mine:
            out.append({**base_row, "key": f"{rec.uid}|nofit|{v}", "reading": v,
                        "status": why or "no_published_overlay"})
        for jb in mine:
            res, err, sec = ((None, jb["err"], 0.0) if jb["ov"] is None else
                             b._call_run(mod, fid.run, pipe, rec, jb["ov"], jb["reading"], jb["rng"]))
            out.append({**base_row, "key": f"{rec.uid}|{jb['j']}|{v}", "j": jb["j"], "start_s": jb["start_s"],
                        "reading": v, "status": b._status(err), "error": err,
                        "call": None if res is None else bool(res["call"]), "p": None if res is None else res["p"],
                        "stat": None if res is None else res["stat"], "seconds": sec,
                        **b.scalars(res and res.get("extra"))})
    b._clear_cache()
    return out


def items_without_morgoth(root, manifest, excluded, allow_missing=False):
    """127.fresh_items without Morgoth files: the same rank order and the same refusal of a repeated or touched
    patient; each window's signal file must exist (allow_missing: windows not yet copied are left out)."""
    df = pd.read_csv(manifest, dtype={"patient_id": str, "uid": str}).sort_values("rank", kind="stable")
    for col in ("uid", "patient_id"):
        if df[col].duplicated().any():
            raise RuntimeError(f"{manifest}: a {col} is listed twice")
    bad = sorted(set(df.patient_id) & set(excluded))
    if bad:
        raise RuntimeError(f"{len(bad)} listed patient(s) were touched by this study before")
    out, seen = [], []
    for r in df.itertuples(index=False):
        it = dict(source="bdsp", path=os.path.join(root, "sig", f"{r.uid}.npz"), morgoth=None, uid=r.uid,
                  patient=r.patient_id, meta={"dev": False, "fresh": True, "fresh_rank": int(r.rank),
                                              "fresh_rank_index": len(seen)})
        seen.append(r.uid)
        if not os.path.exists(it["path"]):
            if allow_missing:
                continue
            raise FileNotFoundError(it["path"])
        out.append(it)
    return out


def items(allow_missing=False):
    return items_without_morgoth(EXT_ROOT, MANIFEST, fr.excluded_patients(), allow_missing)


def shard_path(pipe, shard):
    return os.path.join(SHARDS, f"{rep_spec(pipe)['name']}__{pipe}", f"rows{shard:02d}.jsonl")


def run(pipe, shard, nshards, limit=None, available=False):
    """Rows of every nshards-th listed window from shard (by rank among all 730, so shards are stable while files are
    still arriving); available: skip windows not yet copied (a later full run completes them)."""
    full = items_without_morgoth(EXT_ROOT, MANIFEST, fr.excluded_patients(), allow_missing=True)
    n_listed = len(pd.read_csv(MANIFEST))
    its = full if available else items()
    fn = shard_path(pipe, shard)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done_uids = set()
    if os.path.exists(fn):
        done_uids = {json.loads(l)["uid"] for l in open(fn) if l.strip()}
    with open(fn, "a") as fh:
        for it in its[:limit]:
            i = it["meta"]["fresh_rank_index"]
            if i % nshards != shard or it["uid"] in done_uids:
                continue
            if not MEM.memory_ok():
                raise SystemExit("free memory below the guard: stopped (rerun resumes)")
            t = time.perf_counter()
            for r in rows_for_item(it, pipe, n_listed):
                r["split"] = SPLIT
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            print(f"{pipe} {i} {time.perf_counter() - t:.1f} s", flush=True)


def _call(r, pipe):
    return r.get("x_call_paper") if pipe == "columbia" else r.get("call")


def _j(r):
    try:
        return int(float(r.get("j")))
    except (TypeError, ValueError):
        return -1


def first_overlay_cell(rows, pipe):
    v = FROZEN[pipe]["variant"]
    sel = [r for r in rows if r.get("pipe") == pipe and _j(r) == 0 and r.get("status") == "ok"
           and (r.get("reading") == v or r.get("variant") == v.split(":")[0])]
    calls = [_call(r, pipe) for r in sel]
    if any(c is None for c in calls):
        raise RuntimeError(f"{pipe}: an ok row without a call (missing is not a negative)")
    return st.exact_cell(sum(map(bool, calls)), len(calls))


def tests(cells):
    t = FROZEN_ALL["test"]
    out = {"confirmatory": {}, "specificity_controls": {}, "holm_over_7": {}}
    conf = t["confirmatory"]["pipes"]
    ps = [float(binom.sf(cells[p]["k"] - 1, cells[p]["n"], 0.05)) for p in conf]
    for p, raw, h in zip(conf, ps, st.holm(ps)):
        d = FROZEN[p]["development_crossfit"]
        lo, hi = fz.prediction_interval(d["k"], d["n"], cells[p]["n"])
        rate = cells[p]["k"] / cells[p]["n"]
        out["confirmatory"][p] = {"p": raw, "holm_p": float(h), "prediction_interval": [lo, hi],
                                  "inside_prediction_interval": bool(lo <= rate <= hi)}
    ctl = t["specificity_controls"]["pipes"]
    bound = t["specificity_controls"]["bound"]
    ps = [float(binom.cdf(cells[p]["k"], cells[p]["n"], bound)) for p in ctl]
    for p, raw, h in zip(ctl, ps, st.holm(ps)):
        out["specificity_controls"][p] = {"p_below_bound": raw, "holm_p": float(h)}
    allp = list(FROZEN)
    ps = [float(binom.sf(cells[p]["k"] - 1, cells[p]["n"], 0.05)) for p in allp]
    out["holm_over_7"] = {p: float(h) for p, h in zip(allp, st.holm(ps))}
    return out


def load_rows(pipe):
    rows = []
    for fn in sorted(glob.glob(os.path.join(SHARDS, f"*__{pipe}", "rows*.jsonl"))):
        if not os.path.basename(fn).startswith("._"):
            rows += [json.loads(l) for l in open(fn) if l.strip()]
    return rows


def merge(out=OUT):
    res = {"n_listed": len(pd.read_csv(MANIFEST)), "pipes": {}}
    cells = {}
    for pipe in FROZEN:
        rows = load_rows(pipe)
        cells[pipe] = first_overlay_cell(rows, pipe)
        res["pipes"][pipe] = {"variant": FROZEN[pipe]["variant"], "cell": cells[pipe],
                              "development": FROZEN[pipe]["development"],
                              "n_windows": len({r["uid"] for r in rows}),
                              "n_not_ok_first": sum(1 for r in rows if _j(r) == 0 and r.get("status") != "ok"
                                                    and (r.get("reading") == FROZEN[pipe]["variant"] or
                                                         r.get("variant") == FROZEN[pipe]["variant"].split(":")[0]))}
        if pipe == "mbvt":
            by = {}
            for r in rows:
                if r.get("variant") == FROZEN[pipe]["variant"] and r.get("status") == "ok" and 0 <= _j(r) < 2:
                    by.setdefault(r["patient"], []).append(bool(r["call"]))
            two = [any(v) for v in by.values() if len(v) == 2]
            res["mbvt_two_run"] = st.exact_cell(sum(two), len(two))
    res["tests"] = tests(cells)
    for pipe, h in res["tests"]["holm_over_7"].items():
        res["pipes"][pipe]["holm_p"] = h
    sr.assert_no_identifiers(res)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({p: (v["cell"]["k"], v["cell"]["n"], round(v["cell"]["rate"] or 0, 3))
                      for p, v in res["pipes"].items()}, indent=1))


def dev_item(uid):
    """A development window as an engine item, built as 127.fresh_items builds one (tests only)."""
    it = dict(source="bdsp", path=os.path.join(DEV_SIG, "sig", f"{uid}.npz"),
              morgoth=os.path.join(DEV_SIG, "morgoth", f"{uid}.parquet"), uid=uid, patient=uid.split("_")[0],
              meta={"dev": False})
    for k in ("path", "morgoth"):
        if not os.path.exists(it[k]):
            raise FileNotFoundError(it[k])
    return it


def stored_c3(null, pipe, uid, reading):
    c = st.c3_rows()
    m = c[(c["null"] == null) & (c["pipe"] == pipe) & (c["uid"] == uid) & (c["reading"] == reading)
          & (c["j"].fillna(-1).astype(int) == 0)]
    return None if m.empty else m.iloc[0].to_dict()


def dev_uid_with_rows(pipe, j=0):
    v, null = FROZEN[pipe]["variant"], FROZEN[pipe]["null"]
    local = {f[:-4] for f in os.listdir(os.path.join(DEV_SIG, "sig")) if f.endswith(".npz") and not f.startswith("._")}
    if v.startswith("c3_"):
        c = st.c3_rows()
        c = c[(c["null"] == null) & (c["pipe"] == pipe) & (c["reading"] == v) & (c["status"] == "ok")]
        return next(u for u in c["uid"] if u in local)
    want = sorted(keep_114(pipe))[0]
    pat = os.path.join(ROOT, "output", "intermediate", "_bench_shards", f"{null}__{pipe}__native__none", "rows*.jsonl")
    for fn in sorted(glob.glob(pat)):
        if os.path.basename(fn).startswith("._"):
            continue
        for line in open(fn):
            r = json.loads(line)
            if r.get("variant") == want and str(r.get("j")) == str(j) and r.get("status") == "ok" and r["uid"] in local \
                    and r.get("resp") == "none":
                return r["uid"]
    raise LookupError(pipe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", choices=sorted(FROZEN))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--available", action="store_true", help="skip windows not yet copied")
    a = ap.parse_args()
    merge() if a.merge else run(a.pipe, a.shard, a.nshards, a.limit, a.available)


if __name__ == "__main__":
    main()
