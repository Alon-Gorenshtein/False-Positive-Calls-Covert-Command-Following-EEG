"""Columbia development windows through the common planner. The Columbia development rate (33 of 481) came from the
Columbia runner (113), which placed each first overlay by its own seeded draw; the replication placed every analysis,
Columbia included, with the common planner of 114, and Columbia called 25 of 730. Here the same 481 development
windows are run through the replication driver (153.rows_for_item, development settings: n_recs 974, first overlay),
so the two development rates differ only in where the overlay sits.
Comparisons fixed here, before the run:
- paired, same 481 patients: the call (as_coded:call_paper) at the planner's first overlay against the stored call at
  the runner's first overlay; exact McNemar test (two-sided binomial on the discordant pairs);
- the planner's development rate against the replication rate (25 of 730); two-sided Fisher exact test.
Placement explains the difference if the planner's rate is lower than the stored one (McNemar P < .05) and does not
differ from the replication rate (Fisher P >= .05). Reported whatever it shows. A window with an error or no fitting
overlay leaves the paired denominator (missing is not a negative).
python3 166_columbia_dev_placement.py --shard i --nshards n | --merge -> output/tables/columbia_dev_placement.json
(no identifiers)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import importlib
import json
import sys
import time

from scipy.stats import binomtest, fisher_exact

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
rp = importlib.import_module("153_replication_rows")
st = rp.st
SIG_DIRS = (os.path.join(ROOT, "output", "bdsp", "sig"), os.path.join(ROOT, "output", "bdsp_extra", "sig"))
SHARDS = os.path.join(ROOT, "output", "intermediate", "_col_dev_placement")
OUT = os.path.join(ROOT, "output", "tables", "columbia_dev_placement.json")
N_RECS = 974
PIPE = "columbia"


def development_windows():
    """The 481 Columbia development first windows and their stored calls, as in the frozen development cell."""
    rb = importlib.import_module("158_repeat_bounds")
    d = rb._first("ncmd_bdsp", PIPE, "as_coded")
    calls = {str(u): bool(str(c) == "True" or c is True) for u, c in zip(d["uid"], d["x_call_paper"])}
    pats = {str(u): str(p) for u, p in zip(d["uid"], d["patient"])}
    return sorted(calls), calls, pats


def item(uid, patient):
    for root in SIG_DIRS:
        path = os.path.join(root, f"{uid}.npz")
        if os.path.exists(path):
            return dict(source="bdsp", path=path, morgoth=None, uid=uid, patient=patient, meta={"dev": False})
    raise FileNotFoundError(uid)


def shard_path(shard):
    return os.path.join(SHARDS, f"rows{shard:02d}.jsonl")


def run(shard, nshards, limit=None):
    uids, _, pats = development_windows()
    os.makedirs(SHARDS, exist_ok=True)
    fn = shard_path(shard)
    done = {json.loads(l)["uid"] for l in open(fn) if l.strip()} if os.path.exists(fn) else set()
    with open(fn, "a") as fh:
        for i, u in enumerate(uids[:limit]):
            if i % nshards != shard or u in done:
                continue
            if not rp.MEM.memory_ok():
                raise SystemExit("free memory below the guard: stopped (rerun resumes)")
            t = time.perf_counter()
            rows = rp.rows_for_item(item(u, pats[u]), PIPE, n_recs=N_RECS, split="test", max_overlays=1)
            if not rows:
                rows = [{"pipe": PIPE, "uid": u, "patient": pats[u], "status": "no_rows"}]
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            print(f"{PIPE} {i} {time.perf_counter() - t:.1f} s", flush=True)


def planner_calls(rows):
    """{uid: call} for ok first-overlay rows of the frozen reading; other rows leave the denominator."""
    v = rp.FROZEN[PIPE]["variant"].split(":")[0]
    out = {}
    for r in rows:
        if r.get("pipe") == PIPE and rp._j(r) == 0 and r.get("status") == "ok" and r.get("variant") == v:
            c = r.get("x_call_paper")
            if c is None:
                raise RuntimeError("an ok row without a call (missing is not a negative)")
            out[r["uid"]] = bool(c)
    return out


def compare(stored, planned, replication):
    """stored, planned: {uid: call}; replication: {k, n}. Paired on uids present in both."""
    both = sorted(set(stored) & set(planned))
    a = sum(stored[u] and not planned[u] for u in both)
    b = sum(planned[u] and not stored[u] for u in both)
    kp, n = sum(planned[u] for u in both), len(both)
    ks = sum(stored[u] for u in both)
    mc = float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0
    fe = float(fisher_exact([[kp, n - kp], [replication["k"], replication["n"] - replication["k"]]])[1])
    return {"n_paired": n, "n_stored": len(stored), "n_planned_ok": len(planned),
            "stored": st.exact_cell(ks, n), "planned": st.exact_cell(kp, n), "replication": replication,
            "discordant": {"stored_only": int(a), "planned_only": int(b)}, "mcnemar_exact_p": mc,
            "fisher_planned_vs_replication_p": fe,
            "placement_explains": bool(kp < ks and mc < 0.05 and fe >= 0.05)}


def load_rows():
    rows = []
    for fn in sorted(glob.glob(os.path.join(SHARDS, "rows*.jsonl"))):
        if not os.path.basename(fn).startswith("._"):
            rows += [json.loads(l) for l in open(fn) if l.strip()]
    return rows


def merge(out=OUT):
    uids, stored, _ = development_windows()
    rows = load_rows()
    rep = json.load(open(os.path.join(ROOT, "output", "tables", "replication.json")))["pipes"][PIPE]["cell"]
    res = {"what": __doc__.split("\n")[0], "n_development_windows": len(uids),
           "n_windows_run": len({r["uid"] for r in rows}),
           "not_ok": {s: sum(1 for r in rows if rp._j(r) in (0, -1) and r.get("status") != "ok" and r.get("status") == s)
                      for s in sorted({r.get("status") for r in rows if r.get("status") != "ok"})},
           **compare(stored, planner_calls(rows), {"k": rep["k"], "n": rep["n"]})}
    rp.sr.assert_no_identifiers(res)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("n_paired", "stored", "planned", "discordant", "mcnemar_exact_p",
                                          "fisher_planned_vs_replication_p", "placement_explains")}, indent=1,
                     default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a.shard, a.nshards, a.limit)


if __name__ == "__main__":
    main()
