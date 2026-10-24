"""Every stated-rule-eligible variant of du Bois, edlow17 and cornell on the one-source-per-slot pseudo-sessions of
code/144_onesource_control.py, true and shuffle control (added later, labelled; the fixed A5 and the 144 read stay
as reported). Question: the stated-rule selection (code/145_stated_rule.py) is not fixed yet, so detection and the
shuffle-control call rate are needed for every candidate.

Scope: docs/stated-rule-motor.json (committed 9a94bcc; its sha256 ELIG_SHA256 is checked before anything runs),
the variants with stated_rule_eligible true: du Bois 14 (12 module variants + 130's reading B
c3_b_assessment_{random,blocked}_phase1), edlow17 4, cornell 9 (8 jackknife variants + 130's
c3_jackknife_bh_keep40_r4_om2contig). A name that is neither a module variant nor a 130 reading is refused. Sessions
as A5 / 144: du Bois and edlow17 on OpenBMI session 1, cornell on sessions 1+2; all 51 analysis people.

Imported unchanged: 118 (schedules, seeds, splice grid, crossfade, shuffle control, rows, rates, tests), 144
(motor_session_one with allocate_one and the guard check_one_source on every session), 130 (the readings).
- A module variant runs on its own published schedule with 118's seeds (schedule pc._rng(pid, pipe, 1), session
  pc._rng(pid, pipe, version, "session", sig), run pc._rng(pid, pipe, 2, "published", variant, version)), so its row
  is the row 144 would write for it (du Bois' default reproduces 144's published default row).
- A 130 reading runs on its base variant's session with the base variant's run() seed, so du Bois reading B sees the
  folds of the base run (as 130 specifies). Variants sharing a schedule share one session per version, and the
  modules' label-independent features (mc.memo in each module's _feat, which 130's readings also call) are computed
  once per session.
- Only the published readings: the randomization-based method is not part of this run.
Sessions that cannot be built: status "nc: ..." (144), never a call, outside the rate.
python 146_stated_rule_control.py --run [--workers 10] [--people S01,S02] [--dev]   (streams OpenBMI per person)
python 146_stated_rule_control.py --platform-check OUT.json   (synthetic person, du Bois + cornell; no study rows)
python 146_stated_rule_control.py --summarize   -> output/tables/positive_control_stated.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import collections
import glob
import hashlib
import importlib
import json
import sys
import time
import warnings

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
oc = importlib.import_module("144_onesource_control")
fr = importlib.import_module("130_fidelity_readings")
pc = oc.pc
mc, wr, rn = pc.mc, pc.wr, pc.rn
ROOT = pc.ROOT

ELIG = os.path.join(ROOT, "docs", "stated-rule-motor-first.json")
ELIG_SHA256 = "872ce6e6739e7047f5b054fe50a5db7ba0c8118a1a20a3659f685861cba36572"
ELIG_COMMIT = "9a94bcc"
PIPES = ("dubois", "edlow17", "cornell")
SESSIONS = {"dubois": (1,), "edlow17": (1,), "cornell": (1, 2)}
ARM = "openbmi_onesource_stated"
OUT = os.path.join(ROOT, "output", "intermediate", "_a5_stated")
SUMMARY = os.path.join(ROOT, "output", "tables", "positive_control_stated.json")
LABEL = ("added later: every stated-rule-eligible variant on the one-source-per-slot pseudo-sessions (144); the "
         "pinned A5 (positive_control.json) and the 144 read stay as reported")


def eligibility_bytes(path=ELIG, sha256=ELIG_SHA256):
    """The eligibility file (the first classification's motor file): its bytes if its sha256 matches; anything
    else is refused."""
    raw = open(path, "rb").read()
    if sha256 is None or hashlib.sha256(raw).hexdigest() == sha256:
        return raw
    raise ValueError(f"{path}: sha256 {hashlib.sha256(raw).hexdigest()} != {sha256}")


def jobs(pipe, path=ELIG, sha256=ELIG_SHA256):
    """[{variant, kind (module | reading), base}] of the eligible variants, in the file's order."""
    d = json.loads(eligibility_bytes(path, sha256))
    mod = pc.load_pipe(pipe)
    out = []
    for v, e in d[pipe].items():
        if not (isinstance(e, dict) and e.get("stated_rule_eligible")):
            continue
        if v in mod.VARIANTS:
            out.append({"variant": v, "kind": "module", "base": v})
        elif v in fr.READINGS.get(pipe, {}):
            out.append({"variant": v, "kind": "reading", "base": fr.base(pipe, v)})
        else:
            raise ValueError(f"{pipe}: {v} is neither a module variant nor a 130 reading")
    return out


def _session(person, mod, ov, sig, version):
    try:
        s = oc.motor_session_one(person, mod, ov, pc._rng(person.pid, mod.NAME, version, "session", pc._sig_id(sig)),
                                 version == "control")
        return s, None
    except oc.NotConstructible as e:
        return None, f"{oc.NC} {e}"
    except Exception as e:
        return None, f"error: session: {type(e).__name__}: {str(e)[:200]}"


def run_person_stated(person, mod, dev, done, write, versions=pc.VERSIONS):
    """Every eligible variant of one pipeline on one person, per version; variants grouped by schedule so each session
    is built (and its features computed) once."""
    pid, name = person.pid, mod.NAME
    js = jobs(name)
    for version in versions:
        todo = [j for j in js if f"{ARM}|{pid}|{name}|published|{j['variant']}|{version}" not in done]
        groups = collections.OrderedDict()
        for j in todo:
            ov = mod.schedule(pc.START_S, pc._rng(pid, name, 1), j["base"])
            sig = ("published", pc._sig(mod, ov))
            groups.setdefault(sig, (ov, []))[1].append(j)
        for sig, (ov, members) in groups.items():
            mc._CACHE.clear()
            s, err = _session(person, mod, ov, sig, version)
            for j in sorted(members, key=lambda j: j["kind"] != "module"):
                v, base = j["variant"], j["base"]
                row = {"key": f"{ARM}|{pid}|{name}|published|{v}|{version}", "arm": ARM, "person": pid, "dev": dev,
                       "pipe": name, "design": "published", "variant": v, "version": version, "kind": j["kind"],
                       "base_variant": base, "primary_variant": pc.primary_variant(mod),
                       "allocation": "one_source_per_slot", "added_later": True,
                       "sessions": ",".join(str(x) for x in person.meta.get("sessions", ())),
                       "eligibility_sha256": ELIG_SHA256}
                if s is None:
                    row.update(status=err, call=None)
                else:
                    row.update({f"session_{k}": v_ for k, v_ in s.info.items() if not isinstance(v_, dict)})
                    for k in ("demand_s", "avail_s"):
                        row[f"session_{k}"] = json.dumps(s.info[k])
                    g = pc._rng(pid, name, 2, "published", base, version)
                    if j["kind"] == "module":
                        res, e, sec = rn._call_run(mod, mod.run, s.rec, s.overlay, v, g)
                    else:
                        res, e, sec = rn._call_run(mod, fr.run, name, s.rec, s.overlay, v, g)
                    row.update(pc._result(res, e, sec, mod))
                    mp, _, _ = rn._call(wr.min_attainable_p, mod, base)
                    row.update(min_attainable_p=rn._scalar(mp), cannot_call=bool(mp is not None and mp > 0.05))
                write(row)
            mc._CACHE.clear()



def _rows_path(pipe, pid, out_dir):
    return os.path.join(out_dir, f"{pipe}__{pid}.jsonl")


def _keys(pid, pipe):
    return {f"{ARM}|{pid}|{pipe}|published|{j['variant']}|{ver}" for ver in pc.VERSIONS for j in jobs(pipe)}


def person_job(args):
    """All three pipelines of one OpenBMI subject: fetch sessions 1 and 2, run, one rows file per pipeline (complete
    files are skipped on resume), then delete the subject's files."""
    subj, dev, out_dir = args
    pid = f"OpenBMI_S{subj:02d}"
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    todo = [p for p in PIPES if not _keys(pid, p) <= pc._done(_rows_path(p, pid, out_dir))]
    if not todo:
        return pid, "done"
    t0 = time.perf_counter()
    log = []
    try:
        for sess in sorted({s for p in todo for s in SESSIONS[p]}):
            pc.openbmi_fetch(subj, sess)
        for p in todo:
            mod = pc.load_pipe(p)
            person = oc.load(subj, SESSIONS[p])
            done = pc._done(_rows_path(p, pid, out_dir))
            fh, write = pc._writer(_rows_path(p, pid, out_dir), done)
            try:
                t1 = time.perf_counter()
                run_person_stated(person, mod, dev, done, write)
                log.append(f"{p} {time.perf_counter() - t1:.0f}s")
            finally:
                fh.close()
            del person
    finally:
        for sess in (1, 2):
            f = pc.openbmi_npz(subj, sess)
            if os.path.exists(f):
                os.remove(f)
    return pid, f"{' '.join(log)} | {time.perf_counter() - t0:.0f}s"


def run(workers=10, given=None, dev=False, out_dir=OUT):
    import multiprocessing as mp
    for p in PIPES:
        jobs(p)
    os.makedirs(out_dir, exist_ok=True)
    allp = pc.people("openbmi", dev)
    if given:
        bad = [g for g in given if g not in allp]
        if bad:
            raise ValueError(f"{bad}: not {'development' if dev else 'analysis'} people")
        allp = [q for q in allp if q in given]
    todo = [(int(q.lstrip("S")), dev, out_dir) for q in allp]
    with mp.get_context("spawn").Pool(min(workers, len(todo)), maxtasksperchild=1) as pool:
        for pid, msg in pool.imap_unordered(person_job, todo):
            print(f"{time.strftime('%m-%d %H:%M:%S')} {pid} {msg}", flush=True)


def platform_check(path):
    rows = []
    person = oc.synthetic_person(n_trials=260)
    for p in ("dubois", "cornell"):
        run_person_stated(person, pc.load_pipe(p), True, set(), rows.append)
    keep = ("key", "status", "call", "stat", "p", "session_n_slots", "session_n_sources")
    with open(path, "w") as fh:
        json.dump([{k: rn._scalar(r.get(k)) if not isinstance(r.get(k), (str, type(None))) else r.get(k)
                    for k in keep} for r in rows], fh, indent=1)



def write_summary(path=SUMMARY, rows_dir=OUT):
    """Per pipeline and eligible variant: detection (true) and shuffle-control call rate with exact CP CIs and the
    one-sided exact binomial test against 5% (118.summarize), not-constructible count, slots per source, and the
    fixed A5 true cell beside it (module variants; A5 has no 130 readings)."""
    rows = oc._read(sorted(glob.glob(os.path.join(rows_dir, "*.jsonl"))))
    if any(r.get("dev") for r in rows):
        raise RuntimeError("development rows in the stated-rule output")
    nc = [r for r in rows if (r.get("status") or "").startswith(oc.NC)]
    cells = pc.summarize([r for r in rows if r not in nc])["cells"]
    a5 = pc.summarize(oc._read([pc.out_file("openbmi", p, False) for p in PIPES]))["cells"]
    out = {"label": LABEL, "eligibility_file": os.path.relpath(ELIG, ROOT), "eligibility_sha256": ELIG_SHA256,
           "sessions": SESSIONS, "variants": {}, "bars": {p: pc.BARS[p] for p in PIPES}}
    for pipe in PIPES:
        mine = [r for r in rows if r["pipe"] == pipe]
        if not mine:
            continue
        out["variants"][pipe] = {}
        for v in dict.fromkeys(r["variant"] for r in mine):
            rr = [r for r in mine if r["variant"] == v]
            sps = {ver: [r["session_slots_per_source"] for r in rr if r["version"] == ver and r["status"] == "ok"]
                   for ver in pc.VERSIONS}
            out["variants"][pipe][v] = {
                "kind": rr[0].get("kind"), "base": rr[0].get("base_variant"),
                "true": cells.get(f"{ARM}|{pipe}|published|{v}|true"),
                "control": cells.get(f"{ARM}|{pipe}|published|{v}|control"),
                "n_not_constructible": sum(r in nc for r in rr),
                "slots_per_source_median": {k: float(np.median(x)) for k, x in sps.items() if x},
                "reuse_max_any": max((r["session_reuse_max"] for r in rr if r.get("session_reuse_max")), default=None),
                "a5_true": a5.get(f"openbmi|{pipe}|published|{v}|true")}
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--people", help="comma-separated OpenBMI ids, e.g. S01,S02")
    ap.add_argument("--dev", action="store_true", help="the 3 OpenBMI development people (rows never reported)")
    ap.add_argument("--out")
    ap.add_argument("--platform-check")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.run:
        if a.workers > 10:
            raise SystemExit("at most 10 workers")
        run(a.workers, a.people.split(",") if a.people else None, a.dev, a.out or (OUT + "_dev" if a.dev else OUT))
    elif a.platform_check:
        platform_check(a.platform_check)
    elif a.summarize:
        write_summary()
        print(SUMMARY)


if __name__ == "__main__":
    main()
