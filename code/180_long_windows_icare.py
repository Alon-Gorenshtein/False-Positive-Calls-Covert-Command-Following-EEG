"""Long post-arrest windows (I-CARE 2.1, open access on PhysioNet, read by byte range; nothing raw is kept) for the
analyses whose schedules do not fit the arm's 30-minute window (178.NEED).

fetch: per patient with a processed window (150.window_patients, 544), in the segment its 30-minute window came from
  (the arm's X record), 122's own selection and staging, unchanged: 122.fetch_window (122._scan: the first start whose
  window is clean under the 108 rule applied to the data as recorded; resampled to 128 Hz, median reference, padding
  seconds) with the window length set to W inside this process (122.WINDOW_S and _scan's default), W = the longest
  need that fits (2,550 s, then 2,186, then 1,891). Reads are cached per patient, so a segment is downloaded once.
  One patient per process (the overrides are process-wide). -> output/intermediate/_long/icare_W<W>/ (122's layout:
  _icare_stage/<pid>.npz, or _icare_window/<pid>.json for a width that did not fit) and _long/icare_fetch.jsonl.
run --pipe P: the frozen reading (178.frozen) on each staged window with W >= its need, first overlay, through
  178.rows_for (114 unchanged): the Recording as 122.make_recording builds it (no Morgoth labels: only the stratum
  column reads them) with the arm's padding context (150.recording's rule: pad_s / flat_s of the staged window);
  n_recs as 150.model_patients. -> output/intermediate/_long/icare__<pipe>/rows.jsonl
reach: the patients reached by the fetch (a long window staged, or none in the segment) and those not reached, by
  hospital (122.patient_table; no outcome is read). -> output/tables/icare_long_reach.json
python 180_long_windows_icare.py fetch [--workers W] [--limit N] | run --pipe P [--limit N] | reach"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
ls = importlib.import_module("178_long_schedules")

WIDTHS = tuple(sorted(set(ls.NEED.values()), reverse=True))
LOG = os.path.join(ls.LONG, "icare_fetch.jsonl")


def out_dir(w):
    return os.path.join(ls.LONG, f"icare_W{w}")


def _arm():
    return importlib.import_module("122_icare_arm")


def set_width(arm, w):
    """122's window length inside this process: the module constant (selection, resampling, ROSC hours) and _scan's
    default argument (bound when 122 was imported)."""
    arm.WINDOW_S = int(w)
    d = list(arm._scan.__defaults__)
    d[1] = int(w)
    arm._scan.__defaults__ = tuple(d)


def cached_source(arm):
    class CachedSource(arm.HttpSource):
        """122.HttpSource whose reads are kept for the patient, so a second width re-reads nothing."""
        def __init__(self):
            super().__init__()
            self.cache = {}

        def open(self, pid, segment):
            reader, total = super().open(pid, segment)
            orig, seg = reader.read, segment["seg"]

            def read(t0, t1):
                k = (seg, int(t0), int(t1))
                if k not in self.cache:
                    self.cache[k] = orig(t0, t1)
                return self.cache[k]
            reader.read = read
            return reader, total
    return CachedSource()


def pinned_segment(arm, pid):
    x = arm.read_data_json(os.path.join(arm.paths("analysis")["x"], f"{pid}.json"))
    return x["window"]["segment"]


def fetch_patient(pid, widths=WIDTHS):
    arm = _arm()
    seg = pinned_segment(arm, pid)
    src = cached_source(arm)
    tried = []
    for w in widths:
        set_width(arm, w)
        res = arm.fetch_window(pid, [seg], src, "analysis", out=out_dir(w))
        if res["status"] == "download_failed":
            return {"patient": pid, "status": "download_failed", "error": res.get("error"), "tried": tried}
        if res["status"] == "ok":
            return {"patient": pid, "status": "ok", "W": w, "segment": seg, "start_s": res["window"]["start_s"],
                    "tried": tried}
        tried.append({"W": w, "reason": res.get("reason")})
    return {"patient": pid, "status": "no_long_window", "segment": seg, "tried": tried}


def done_patients():
    if not os.path.exists(LOG):
        return set()
    return {json.loads(l)["patient"] for l in open(LOG) if l.strip() and json.loads(l)["status"] != "download_failed"}


def fetch(workers=2, limit=None):
    ic = importlib.import_module("150_icare_stated_rule")
    pats = [p for p in ic.window_patients() if p not in done_patients()][:limit]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex, open(LOG, "a") as fh:
        for i, r in enumerate(ex.map(fetch_patient, pats)):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(f"{i + 1}/{len(pats)} {r['patient']} {r['status']} W={r.get('W')} {time.time() - t0:.0f} s",
                  flush=True)


def staged():
    """{patient: (W, stage path)} of the long windows staged so far."""
    out = {}
    if os.path.exists(LOG):
        for l in open(LOG):
            r = json.loads(l) if l.strip() else None
            if r and r["status"] == "ok":
                p = os.path.join(out_dir(r["W"]), "_icare_stage", f"{r['patient']}.npz")
                if os.path.exists(p):
                    out[r["patient"]] = (int(r["W"]), p)
    return out


def recording(arm, pid, path):
    import numpy as np
    import pandas as pd
    import glob
    x, win = arm.load_stage(path)
    cols = pd.read_parquet(sorted(glob.glob(os.path.join(arm.paths("analysis")["morgoth"], "*.parquet")))[0]).columns
    rec = arm.make_recording(pid, x, pd.DataFrame(columns=cols), False, win)
    n = rec.x.shape[1] // arm.FS_OUT
    if win.get("padding_rule") not in (arm.PADDING_RULE, arm.V1_RULE):
        raise RuntimeError(f"{pid}: padding rule {win.get('padding_rule')!r}")
    pad, flat = np.zeros(n, bool), np.zeros(n, bool)
    pad[np.asarray(win["pad_s"], int)] = True
    flat[np.asarray(win["flat_s"], int)] = True
    rec.meta["_artifact_s"] = arm.artifact_seconds_icare(rec, flat, pad)
    return rec, pad


def run(pipe, limit=None):
    arm = _arm()
    ic = importlib.import_module("150_icare_stated_rule")
    fz = ls.frozen()
    kind = fz["readings"][pipe]["null_kind"]
    n_recs = ic.model_patients()[1]
    fn = os.path.join(ls.LONG, f"icare__{pipe}", "rows.jsonl")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = {json.loads(l)["uid"] for l in open(fn) if l.strip()} if os.path.exists(fn) else set()
    todo = [(p, path) for p, (w, path) in sorted(staged().items()) if w >= ls.NEED[pipe] and p not in done][:limit]
    print(f"{pipe}: {len(todo)} staged windows to run", flush=True)
    with open(fn, "a") as fh:
        for pid, path in todo:
            t = time.perf_counter()
            rec, pad = recording(arm, pid, path)
            arm.set_padding_context(rec.x, pad)
            try:
                rows = ls.rows_for(rec, pipe, f"{kind}_icare", "analysis", n_recs, fz)
            finally:
                arm.clear_padding_context()
            for r in rows:
                fh.write(json.dumps({**r, "source": "icare"}, allow_nan=False) + "\n")
            fh.flush()
            print(f"{pipe} {pid} {time.perf_counter() - t:.1f} s", flush=True)


def reach(out=os.path.join(ROOT, "output", "tables", "icare_long_reach.json")):
    arm = _arm()
    index = arm.load_index()
    t = arm.patient_table(arm.analysis_order(index, arm.dev_patients(index)), index)
    t = t.loc[t["status"] == "included", ["patient", "hospital"]].astype(str)
    st = staged()
    tried = {json.loads(l)["patient"] for l in open(LOG) if l.strip() and json.loads(l)["status"] != "download_failed"}
    t["group"] = [("staged" if p in st else "no_long_window") if p in tried else "not_reached" for p in t["patient"]]
    res = {"what": "Post-arrest patients reached by the long-window fetch, by hospital (no outcome read)",
           "n": {g: int(n) for g, n in t["group"].value_counts().items()},
           "W": {str(w): sum(1 for v in st.values() if v[0] == w) for w in WIDTHS},
           "reached_ids": [min(t.loc[t["group"] != "not_reached", "patient"]),
                           max(t.loc[t["group"] != "not_reached", "patient"])],
           "hospital": {h: {g: int(((t["hospital"] == h) & (t["group"] == g)).sum())
                            for g in ("staged", "no_long_window", "not_reached")}
                        for h in sorted(t["hospital"].unique())}}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "reach"])
    ap.add_argument("--pipe", choices=["cruse11", "cruse12p", "chennu13"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.workers, a.limit)
    elif a.cmd == "reach":
        reach()
    else:
        run(a.pipe, a.limit)


if __name__ == "__main__":
    main()
