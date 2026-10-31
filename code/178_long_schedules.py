"""Analyses whose published schedules do not fit a 30-minute window, run at full length where a long enough clean
window exists. Needs (114.need_seconds of the primary reading, lead and tail included): Chennu 2013 2,550 s, Cruse 2011 2,186 s, Cruse 2012
1,891 s. Nothing frozen is edited: 114 runs unchanged under run-time overrides, and every row goes to its own folder
(output/intermediate/_long), never to the benchmark's shard files.

chennu-dev: Chennu 2013 on clinical EEG, development (analysis split, the 974 patients). The benchmark ran Chennu on
  this source only as its 12-block shortened twin (114.SOURCE_VARIANTS), so it has no stated-rule
  reading here. Here its native 20-block eligible readings (CHENNU_ELIGIBLE, the 145 candidates of the DoC arm) run on
  the first overlay of every analysis-split recording of at least NEED["chennu13"] s, through 114.iter_recordings
  over the full listing (overlay seeding as in the benchmark) and 114.run_recording; no randomization rows.
chennu-select: 145's rule on those rows (usable: coverage and cannot-call as 115 computes them; select: the fewest
  false calls on the recordings where every candidate ran; the cross-fit by crc32 parity halves and its held-out
  test) -> output/tables/long_schedule_dev.json.
freeze: the readings carried to the new windows (179, 180), written before any of them is analysed.
merge: per cohort and analysis, the frozen reading's first-overlay rows -> k/n, exact 95% CI, one-sided exact binomial
  P against 5% (descriptive; outside the benchmark's and the replication's Holm families), with the windows run and
  their status counts -> output/tables/long_schedule.json.
python 178_long_schedules.py chennu-dev [--shard i --nshards n] [--limit N] | chennu-select | freeze | merge"""
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

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
b = importlib.import_module("114_benchmark")

NEED = {"cruse11": 2186, "cruse12p": 1891, "chennu13": 2550}
CHENNU_NULL = "nstim0_bdsp"
CHENNU_ELIGIBLE = ("p1_samples10_zvar_b20", "p1_ms40_zvar_b20")
LONG = os.path.join(ROOT, "output", "intermediate", "_long")
DEV_OUT = os.path.join(ROOT, "output", "tables", "long_schedule_dev.json")
FREEZE = os.path.join(ROOT, "output", "tables", "long_schedule_frozen.json")
STATED = os.path.join(ROOT, "output", "tables", "stated_rule.json")
REP_LISTING = os.path.join(ROOT, "output", "bdsp", "meta", "long_rep_windows.csv")
DEV_READINGS = {"cruse11": "ncmd_bdsp|cruse11", "cruse12p": "ncmd_bdsp|cruse12p", "sokoliuk21": "nstim0_bdsp|sokoliuk21"}


@contextlib.contextmanager
def native_chennu(keep):
    """114 with Chennu 2013 at its native 20 blocks on clinical EEG, only the readings in keep, no randomization."""
    sv, po, jobs0, rand0 = dict(b.SOURCE_VARIANTS), dict(b.PRIMARY_OVERRIDE), b._jobs, b._rand_fields
    b.SOURCE_VARIANTS[("chennu13", "bdsp")] = {"n_blocks": 20}
    b.PRIMARY_OVERRIDE.pop(("chennu13", "bdsp"), None)
    b._jobs = lambda *a: [x for x in jobs0(*a) if x[1] in keep]
    b._rand_fields = lambda *a, **k: {}
    try:
        yield
    finally:
        b.SOURCE_VARIANTS.clear()
        b.SOURCE_VARIANTS.update(sv)
        b.PRIMARY_OVERRIDE.clear()
        b.PRIMARY_OVERRIDE.update(po)
        b._jobs, b._rand_fields = jobs0, rand0


def long_patients(items, need):
    """Patients whose recording holds at least need seconds (the engine then looks for a clean overlay inside)."""
    out = set()
    for it in items:
        with np.load(it["path"]) as z:
            if float(z["dur_s"]) >= need:
                out.add(it["patient"])
    return out


def _resume(fn):
    if not os.path.exists(fn):
        return set()
    return {json.loads(l)["key"] for l in open(fn) if l.strip()}


def chennu_dev(shard=0, nshards=1, limit=None):
    spec = b.parse_null(CHENNU_NULL)
    mod = b.load_pipe("chennu13")
    skip = b.touched_uids()
    items = b.list_items(spec, "analysis", skip)
    only = long_patients(items, NEED["chennu13"])
    fn = os.path.join(LONG, "chennu_dev", f"rows{shard:02d}.jsonl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = _resume(fn)
    ctx = {"null": CHENNU_NULL, "split": "analysis", "montage": "native", "resp": "none", "max_overlays": 1,
           "locator": None, "uncorrected": None, "col113": None, "spec_path": None}
    print(f"{len(only)} of {len(items)} recordings hold {NEED['chennu13']} s", flush=True)
    with open(fn, "a") as fh, native_chennu(set(CHENNU_ELIGIBLE)):
        def write(row):
            fh.write(json.dumps(row, allow_nan=False) + "\n")
            fh.flush()
            done.add(row["key"])
        for i, (rec, n) in enumerate(b.iter_recordings(spec, "analysis", None, shard, nshards, skip, only, items)):
            if limit is not None and i >= limit:
                break
            t = time.perf_counter()
            b.run_recording(rec, mod, spec, {**ctx, "n_recs": n}, done, write)
            print(f"chennu13 {rec.uid} {time.perf_counter() - t:.1f} s", flush=True)
            del rec


@contextlib.contextmanager
def frozen_only(pipe, variant):
    """114 restricted to one reading: variant_specs (so the window need is the reading's own, as 148 does for Cruse
    2011's 4-block readings), its jobs only, no randomization; Chennu 2013 at its native 20 blocks."""
    vs0, jobs0, rand0 = b.variant_specs, b._jobs, b._rand_fields
    sv, po = dict(b.SOURCE_VARIANTS), dict(b.PRIMARY_OVERRIDE)
    if pipe == "chennu13":
        b.SOURCE_VARIANTS[("chennu13", "bdsp")] = {"n_blocks": 20}
        b.PRIMARY_OVERRIDE.pop(("chennu13", "bdsp"), None)

    def restricted(m, source):
        sp = vs0(m, source)
        if m.NAME != pipe or source != "bdsp":
            return sp
        if variant not in sp:
            raise RuntimeError(f"{pipe}: {variant} is not among its clinical-EEG variants")
        return {variant: sp[variant]}
    b.variant_specs = restricted
    b._jobs = lambda *a: [x for x in jobs0(*a) if x[1] == variant]
    b._rand_fields = lambda *a, **k: {}
    try:
        yield
    finally:
        b.variant_specs, b._jobs, b._rand_fields = vs0, jobs0, rand0
        b.SOURCE_VARIANTS.clear()
        b.SOURCE_VARIANTS.update(sv)
        b.PRIMARY_OVERRIDE.clear()
        b.PRIMARY_OVERRIDE.update(po)


def frozen():
    return json.load(open(FREEZE))


def rows_for(rec, pipe, null, split, n_recs, fz=None):
    """114's rows of the frozen reading of pipe on one recording (first overlay)."""
    r = (fz or frozen())["readings"][pipe]
    mod = b.load_pipe(pipe)
    spec = {"name": null, "kind": null.split("_")[0], "family": mod.FAMILY, "source": "bdsp"}
    ctx = {"null": null, "split": split, "montage": "native", "resp": "none", "max_overlays": 1, "locator": None,
           "uncorrected": None, "col113": None, "spec_path": None, "n_recs": n_recs}
    out = []
    with frozen_only(pipe, r["variant"]):
        b.run_recording(rec, mod, spec, ctx, set(), out.append)
    b._clear_cache()
    return out


def freeze(out=FREEZE):
    """The readings and window rules fixed before any long window is analysed (committed before the runs)."""
    import hashlib
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: never rewritten")
    st = json.load(open(STATED))["entries"]
    dev = json.load(open(DEV_OUT))["chennu13"]
    readings = {p: {"variant": st[k]["variant"], "null_kind": k.split("_")[0], "from": f"stated_rule.json {k}"}
                for p, k in DEV_READINGS.items()}
    readings["chennu13"] = {"variant": dev["variant"], "null_kind": "nstim0",
                            "from": "long_schedule_dev.json (145's rule on the native 20-block rows)"}
    for p, r in readings.items():
        r["need_s"] = NEED.get(p, 1367)
    res = {"what": "Readings carried unchanged to windows they were not chosen on: long replication windows (179), "
                   "long post-arrest windows (180) and, for Sokoliuk 2021, the 730 replication 30-minute windows (153)",
           "readings": readings, "first_overlay_only": True,
           "windows": {"replication_long": {"listing": os.path.relpath(REP_LISTING, ROOT),
                                            "sha256": hashlib.sha256(open(REP_LISTING, "rb").read()).hexdigest(),
                                            "rule": "179 docstring"},
                       "replication_30min": "153.items() (Sokoliuk 2021 only; its 1,367 s fit)",
                       "post_arrest_long": "180 docstring (122's selection at W = the longest need that fits)"},
           "report": "per analysis and cohort: k/n of the first overlay, exact Clopper-Pearson 95% CI, one-sided "
                     "exact binomial P against 5%; descriptive, outside the Holm families of the benchmark and of the "
                     "replication",
           "n_recs": {"replication": 730, "post_arrest": "150.model_patients()[1]"}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(readings, indent=1))
    return res


MERGE_OUT = os.path.join(ROOT, "output", "tables", "long_schedule.json")
COHORTS = (("replication", "rep"), ("post_arrest", "icare"))


def cell(rows, variant):
    """k/n of the frozen reading's first overlay over windows, one row per window; status counts per window."""
    import collections
    st = importlib.import_module("143_stratum_table")
    first, status = {}, {}
    for r in rows:
        status.setdefault(r["uid"], r["status"])
        if r.get("variant") == variant and r.get("j") == 0:
            if r["uid"] in first:
                raise RuntimeError(f"{r['uid']}: two first-overlay rows of {variant}")
            first[r["uid"]] = r
            status[r["uid"]] = r["status"]
    ok = [r for r in first.values() if r["status"] == "ok"]
    if any(r.get("call") is None for r in ok):
        raise RuntimeError("an ok row without a call (missing is not a negative)")
    k = sum(bool(r["call"]) for r in ok)
    out = dict(st.exact_cell(k, len(ok))) if ok else {"k": 0, "n": 0}
    out.update(windows_run=len(status), status=dict(collections.Counter(status.values())))
    return out


def merge(out=MERGE_OUT):
    fz = frozen()
    res = {"what": "Long-schedule analyses at full length: first-overlay false-call rates of the frozen readings",
           "frozen": fz["readings"], "development": json.load(open(DEV_OUT))["chennu13"].get("stated_rule"),
           "cells": {}}
    for cohort, pre in COHORTS:
        for pipe, r in fz["readings"].items():
            rows = load_rows(f"{pre}__{pipe}")
            if rows:
                res["cells"][f"{cohort}|{pipe}"] = cell(rows, r["variant"])
    json.dump(res, open(out, "w"), indent=1, default=float)
    for k, v in res["cells"].items():
        print(k, v.get("k"), v.get("n"), v.get("status"))
    return res


def load_rows(sub):
    rows = []
    for fn in sorted(glob.glob(os.path.join(LONG, sub, "rows*.jsonl"))):
        if not os.path.basename(fn).startswith("._"):
            rows += [json.loads(l) for l in open(fn) if l.strip()]
    return rows


def chennu_select(out=DEV_OUT):
    """145._assess on the native rows: the clinical primary reading of Chennu 2013, its exact rate, the eligible
    range, the cross-fit and its held-out test. Outside the Holm family (added after the benchmark's tests)."""
    import pandas as pd
    s = importlib.import_module("145_stated_rule")
    df = pd.DataFrame(load_rows("chennu_dev")).drop_duplicates("key", keep="first")
    dp, dropped = s.one_per_patient(df)
    coded = {v: bool(s.mg._flag(g, "as_coded").any()) for v, g in dp.groupby("variant")}
    entry, _ = s._assess(dp, CHENNU_NULL, "chennu13", list(CHENNU_ELIGIBLE), coded, s.B_BOOT,
                         "primary, native 20 blocks on recordings that hold them")
    status = df.groupby("uid")["status"].agg(lambda x: "ok" if (x == "ok").any() else x.iloc[0])
    res = {"what": __doc__.split("\n")[0], "need_s": NEED["chennu13"], "null": CHENNU_NULL,
           "recordings": {"listed": int(len(b.list_items(b.parse_null(CHENNU_NULL), "analysis", b.touched_uids()))),
                          "long_enough": int(df["uid"].nunique()), "clean_fit": int((status == "ok").sum()),
                          "no_fit": int((status == "no_fit").sum()), "later_recordings_dropped": dropped},
           "chennu13": entry}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: entry.get(k) for k in ("variant", "stated_rule", "held_out_test")}, default=float, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["chennu-dev", "chennu-select", "freeze", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "chennu-dev":
        chennu_dev(a.shard, a.nshards, a.limit)
    elif a.cmd == "chennu-select":
        chennu_select()
    elif a.cmd == "freeze":
        freeze()
    else:
        merge()


if __name__ == "__main__":
    main()
