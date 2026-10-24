"""A5 positive control rebuilt with ONE source trial per slot (the original A5 stays as reported). 118's reuse-free
packing lets several disjoint excerpts of one source trial fill slots of the same (pseudo-)class (118.allocate: "a
long source hosts several excerpts"; 118.pools_for: each source sits in one pseudo-class pool), which clusters
trials within a class and may inflate the non-randomized shuffle controls. Here every slot of a session takes its
own source trial, from the start, and no source fills a second slot (check_one_source asserts it on every session,
true and control, before any module runs).

Everything else is 118's, imported unchanged: the module's own schedule and randomized schedule (same seeds, so the
schedules equal A5's), the class-independent splice grid (118._motor_layout), the equal-power crossfade, the
source-level shuffle control (labels permuted over source units), the row schema, rates, Clopper-Pearson CIs, the
one-sided exact binomial test of each control against 5% (118.summarize) and A5's verdict logic.

Scope (team-decision, 2026-10-18): the primary arm only, true and shuffle-control versions, the published default
variant plus the randomization-based method of its timing group (118.jobs_for "primary"):
- OpenBMI: cornell, mbmi (sessions 1+2), dubois, edlow17 (session 1), all 51 analysis people; columbia on the A5
  seeded sample of 20 (SAMPLE_SEED), on sessions 1+2 INSTEAD OF A5's session 1 (341 rest slots need more than the
  ~200 rest stretches of one session).
- chennu13, gibson16 (Kojima B): carried over from the fixed A5 rows, not rerun (their units are already one whole
  trial each: n_units = n_sources in every A5 row).
- cruse12p and mbvt: not constructible under one source per slot (NOT_CONSTRUCTIBLE); cruse12p is checked per person
  by allocation alone (no module run).
Sessions that cannot be built are rows with status "nc: not constructible ..." (never a call, outside the rate).
python 144_onesource_control.py --run [--workers 10] [--people S01,S02] [--dev]   (streams OpenBMI per person)
python 144_onesource_control.py --platform-check OUT.json   (synthetic person; compare machines)
python 144_onesource_control.py --summarize   -> output/tables/positive_control_onesource.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import collections
import glob
import importlib
import json
import sys
import time
import warnings

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
pc = importlib.import_module("118_positive_control")
mc, en, wr, rn = pc.mc, pc.en, pc.wr, pc.rn
ROOT = pc.ROOT

ARM = "openbmi_onesource"
PIPES = ("cornell", "mbmi", "dubois", "edlow17", "columbia")
SESSIONS = {"cornell": (1, 2), "mbmi": (1, 2), "dubois": (1,), "edlow17": (1,), "columbia": (1, 2)}
SAMPLE = {"columbia": 20}
CARRIED = ("chennu13", "gibson16")
NOT_CONSTRUCTIBLE = {
    "cruse12p": "OpenBMI sessions 1+2: 443-453 rest slots per session against ~400 rest stretches (no third session)",
    "mbvt": "Kojima B: 60 groups per session against 12 trials per counted identity (VT2: 60 from 12; VT3: 30 + 30 "
            "from 12 + 12)"}
NC = "nc: not constructible (one source per slot):"
OUT = os.path.join(ROOT, "output", "intermediate", "_a5_onesource")
SUMMARY = os.path.join(ROOT, "output", "tables", "positive_control_onesource.json")
LABEL = ("added later rebuild with one source trial per slot; the pinned A5 "
         "(positive_control.json) stays as reported")


class NotConstructible(ValueError):
    pass


def allocate_one(slots, sources, pools, rng, c):
    """One source per slot: every slot of class k takes an UNUSED source of pool k long enough for need = (b - a) + c
    samples, from its start, drawn at random among those; none left -> NotConstructible. Sets slot src / off; stats
    as 118.allocate (reuse_max 1)."""
    used, need_tot = set(), collections.Counter()
    for s in slots:
        k, need = s["cls"], s["b"] - s["a"] + c
        room = [i for i in pools[k] if i not in used and sources[i].n >= need]
        if not room:
            n_k = sum(1 for t in slots if t["cls"] == k)
            raise NotConstructible(f"{n_k} {k} slots, {len(pools[k])} {k} source trials")
        i = room[int(rng.integers(len(room)))]
        used.add(i)
        need_tot[k] += need
        s.update(src=int(i), off=0)
    avail = {k: sum(sources[i].n for i in p) for k, p in pools.items()}
    ratio = {k: need_tot[k] / avail[k] for k in need_tot}
    n_src = len({s["src"] for s in slots})
    return {"reuse_max": 1, "demand_ratio": ratio, "demand_ratio_max": float(max(ratio.values())),
            "n_sources": n_src, "slots_per_source": len(slots) / n_src}


def check_one_source(slots):
    """The guard: no source trial fills more than one slot."""
    n = collections.Counter(s["src"] for s in slots)
    dup = {i: k for i, k in n.items() if k > 1}
    assert not dup, f"{len(dup)} source(s) fill more than one slot (e.g. source {next(iter(dup))}: {max(dup.values())})"


def motor_session_one(person, mod, ov, rng, control=False, xfade_s=pc.XFADE_S):
    """118.motor_session with allocate_one and the guard (the construction is otherwise line for line 118's)."""
    pc.check_provenance(person)
    fs = person.fs
    dataset = person.meta.get("source")
    slots, pools, c = pc._motor_layout(mod, ov, person.sources, fs, rng, control, xfade_s, dataset)
    st = allocate_one(slots, person.sources, pools, rng, c)
    check_one_source(slots)
    rows, names = pc.module_channels(mod, person.names)
    T = slots[-1]["b"]
    x = np.zeros((len(rows), T), np.float32)
    up, dn = pc._ramps(c, "equal_power") if c else (None, None)
    for k, s in enumerate(slots):
        src = person.sources[s["src"]]
        piece = np.asarray(person.runs[src.run][rows, src.start + s["off"]: src.start + s["off"] + s["b"] - s["a"] + c],
                           float)
        pc._place(x, piece, s["a"], c, k == 0, k == len(slots) - 1, up, dn)
    demand = collections.Counter()
    for s in slots:
        demand[s["cls"]] += (s["b"] - s["a"]) / fs
    avail = {cl: sum(person.sources[i].n for i in p) / fs for cl, p in pools.items()}
    info = {"control": bool(control), "n_slots": len(slots), "n_sources": st["n_sources"], "reuse_max": st["reuse_max"],
            "slots_per_source": st["slots_per_source"], "demand_ratio_max": float(st["demand_ratio_max"]),
            "demand_s": {k: round(v, 1) for k, v in demand.items()}, "avail_s": {k: round(v, 1) for k, v in avail.items()},
            "xfade_s": xfade_s, "seconds": T / fs}
    rec = en.Recording(source=dataset or "openbmi", uid=f"{person.pid}#{mod.NAME}", patient=person.pid, x=x, fs=fs,
                       names=names, meta={"pseudo_session": True, "control": bool(control), "one_source": True})
    return pc.Session(rec, ov, slots, info)


def run_person_one(person, mod, dev, done, write, versions=pc.VERSIONS):
    """118.run_person for the primary scope (default variant + its randomized group), with motor_session_one; the
    same seeds, keys (arm ARM) and row fields, plus allocation / added_later / session_slots_per_source."""
    pid, name = person.pid, mod.NAME
    groups = pc.rand_groups(mod)
    prim = pc.primary_variant(mod)
    for version in versions:
        control = version == "control"
        sessions = {}
        for design, v in pc.jobs_for(mod, version, "primary", "primary"):
            key = f"{ARM}|{pid}|{name}|{design}|{v}|{version}"
            if key in done:
                continue
            if design == "published":
                ov = mod.schedule(pc.START_S, pc._rng(pid, name, 1), v)
            else:
                ov = mod.schedule_randomized(pc.START_S, pc._rng(pid, name, 3, v), v)
            sig = (design, pc._sig(mod, ov))
            if sig not in sessions:
                mc._CACHE.clear()
                sessions.clear()
                try:
                    sessions[sig] = (motor_session_one(person, mod, ov,
                                                       pc._rng(pid, name, version, "session", pc._sig_id(sig)),
                                                       control), None)
                except NotConstructible as e:
                    sessions[sig] = (None, f"{NC} {e}")
                except Exception as e:
                    sessions[sig] = (None, f"error: session: {type(e).__name__}: {str(e)[:200]}")
            s, err = sessions[sig]
            row = {"key": key, "arm": ARM, "person": pid, "dev": dev, "pipe": name, "design": design, "variant": v,
                   "version": version, "primary_variant": prim, "allocation": "one_source_per_slot", "added_later": True,
                   "sessions": ",".join(str(x) for x in person.meta.get("sessions", ())),
                   "as_coded": bool(mod.VARIANTS.get(v, {}).get("as_coded", False)),
                   "group_members": ",".join(groups[v]) if design == "randomized" else None}
            if s is None:
                row.update(status=err, call=None)
            else:
                row.update({f"session_{k}": v_ for k, v_ in s.info.items() if not isinstance(v_, dict)})
                for k in ("demand_s", "avail_s"):
                    row[f"session_{k}"] = json.dumps(s.info[k])
                g = pc._rng(pid, name, 2, design, v, version)
                if design == "published":
                    res, e, sec = rn._call_run(mod, mod.run, s.rec, s.overlay, v, g)
                else:
                    res, e, sec = rn._call_run(mod, wr.run_randomized, mod, s.rec, s.overlay, v, g)
                row.update(pc._result(res, e, sec, mod))
                mp, _, _ = rn._call(wr.min_attainable_p, mod, v)
                row.update(min_attainable_p=rn._scalar(mp), cannot_call=bool(mp is not None and mp > 0.05))
            write(row)
        sessions.clear()
        mc._CACHE.clear()


def constructible(person, mod):
    """Allocation alone (no module run): '' if every session of the scope can be built, else the reason."""
    for version in pc.VERSIONS:
        for design, v in pc.jobs_for(mod, version, "primary", "primary"):
            pid, name = person.pid, mod.NAME
            ov = (mod.schedule(pc.START_S, pc._rng(pid, name, 1), v) if design == "published"
                  else mod.schedule_randomized(pc.START_S, pc._rng(pid, name, 3, v), v))
            sig = (design, pc._sig(mod, ov))
            rng = pc._rng(pid, name, version, "session", pc._sig_id(sig))
            slots, pools, c = pc._motor_layout(mod, ov, person.sources, person.fs, rng, version == "control",
                                               pc.XFADE_S, "openbmi")
            try:
                allocate_one(slots, person.sources, pools, rng, c)
            except NotConstructible as e:
                return f"{design} {version}: {e}"
    return ""



def people(pipe, dev=False):
    return pc.people("openbmi", dev, sample=None if dev else SAMPLE.get(pipe))


def load(subj, sessions):
    p = pc.openbmi_person(subj, sessions)
    p.meta["sessions"] = tuple(sessions)
    return p


def _rows_path(pipe, pid, out_dir):
    return os.path.join(out_dir, f"{pipe}__{pid}.jsonl")


def _job_keys(pid, mod):
    return [f"{ARM}|{pid}|{mod.NAME}|{d}|{v}|{ver}" for ver in pc.VERSIONS
            for d, v in pc.jobs_for(mod, ver, "primary", "primary")]


def person_job(args):
    """Every pending pipeline of one OpenBMI subject: fetch its sessions, run, write one rows file per pipeline
    (complete files are skipped on resume), check cruse12p by allocation, delete the subject's files."""
    subj, dev, out_dir = args
    pid = f"OpenBMI_S{subj:02d}"
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mods = {p: pc.load_pipe(p) for p in PIPES if f"S{subj:02d}" in people(p, dev)}
    todo = {p: m for p, m in mods.items()
            if not set(_job_keys(pid, m)) <= pc._done(_rows_path(p, pid, out_dir))}
    nc_path = os.path.join(out_dir, f"cruse12p_constructible__{pid}.json")
    if not todo and os.path.exists(nc_path):
        return pid, "done"
    t0 = time.perf_counter()
    for sess in sorted({s for p in todo for s in SESSIONS[p]} | {1, 2}):
        pc.openbmi_fetch(subj, sess)
    log = []
    try:
        for p, m in todo.items():
            person = load(subj, SESSIONS[p])
            done = pc._done(_rows_path(p, pid, out_dir))
            fh, write = pc._writer(_rows_path(p, pid, out_dir), done)
            try:
                t1 = time.perf_counter()
                run_person_one(person, m, dev, done, write)
                log.append(f"{p} {time.perf_counter() - t1:.0f}s")
            finally:
                fh.close()
            del person
        if not os.path.exists(nc_path):
            person = load(subj, (1, 2))
            why = constructible(person, pc.load_pipe("cruse12p"))
            with open(nc_path + ".tmp", "w") as fh:
                json.dump({"person": pid, "constructible": not why, "reason": why}, fh)
            os.replace(nc_path + ".tmp", nc_path)
            del person
    finally:
        for sess in (1, 2):
            f = pc.openbmi_npz(subj, sess)
            if os.path.exists(f):
                os.remove(f)
    return pid, f"{' '.join(log)} | {time.perf_counter() - t0:.0f}s"


def run(workers=10, given=None, dev=False, out_dir=OUT):
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    allp = sorted({q for p in PIPES for q in people(p, dev)})
    if given:
        bad = [g for g in given if g not in allp]
        if bad:
            raise ValueError(f"{bad}: not {'development' if dev else 'analysis'} people of the scope")
        allp = [q for q in allp if q in given]
    heavy = set(people("columbia", dev))
    order = [q for q in allp if q in heavy] + [q for q in allp if q not in heavy]
    jobs = [(int(q.lstrip("S")), dev, out_dir) for q in order]
    with mp.get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
        for pid, msg in pool.imap_unordered(person_job, jobs):
            print(f"{time.strftime('%m-%d %H:%M:%S')} {pid} {msg}", flush=True)



def synthetic_person(seed=20261018, n_trials=200):
    rng = np.random.default_rng(seed)
    fs = pc.OPENBMI_FS
    t, pos = [], int(3 * fs)
    for _ in range(n_trials):
        t.append(pos)
        pos += int(4 * fs) + int(rng.uniform(4.5, 7.5) * fs) + int(3 * fs)
    n = pos + int(3 * fs)
    x = mc.bandpass(rng.normal(size=(len(pc.MOTOR_CHANNELS), n)), fs, 1.0, 40.0) * 10
    y = np.tile([1, 2], n_trials // 2)
    src = pc.openbmi_sources(t, y, n, fs, 0, "syn")
    return pc.Person(pid="OpenBMI_S99", fs=fs, names=list(pc.MOTOR_CHANNELS), runs=[x.astype(np.float32)],
                     sources=src, meta={"source": "openbmi", "events": {(s.run, s.start): s.cls for s in src},
                                        "sessions": ()})


def platform_check(path, pipes=("cornell", "mbmi", "edlow17", "dubois")):
    rows = []
    person = synthetic_person()
    for p in pipes:
        run_person_one(person, pc.load_pipe(p), True, set(), rows.append)
    keep = ("key", "status", "call", "stat", "p", "size_stat", "session_n_slots", "session_n_sources")
    with open(path, "w") as fh:
        json.dump([{k: rn._scalar(r.get(k)) if not isinstance(r.get(k), (str, type(None))) else r.get(k)
                    for k in keep} for r in rows], fh, indent=1)



def _read(paths):
    rows = []
    for f in paths:
        if not os.path.basename(f).startswith("._"):
            with open(f) as fh:
                rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def _sps(rows):
    """Median slots (motor) or units (ERP) per distinct source, per design | version, over ok rows."""
    out = collections.defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("session_slots_per_source") is not None:
            out[f"{r['design']}|{r['version']}"].append(r["session_slots_per_source"])
        elif r.get("session_n_units"):
            out[f"{r['design']}|{r['version']}"].append(r["session_n_units"] / r["session_n_sources"])
    return {k: float(np.median(v)) for k, v in out.items()}


def _verdict(dp, reuse):
    if reuse and max(reuse) > 1:
        return "reuse-confounded"
    if dp is None or not dp.get("n"):
        return "no usable people"
    return "fidelity concern" if dp.get("fidelity_concern") else "consistent with the bar"


def write_summary(path=SUMMARY, rows_dir=OUT):
    rows = _read(sorted(glob.glob(os.path.join(rows_dir, "*.jsonl"))))
    if any(r.get("dev") for r in rows):
        raise RuntimeError("development rows in the one-source output")
    nc = [r for r in rows if (r.get("status") or "").startswith(NC)]
    cells = pc.summarize([r for r in rows if r not in nc])["cells"]
    head = {}
    for pipe in PIPES:
        mine = [r for r in rows if r["pipe"] == pipe]
        if not mine:
            continue
        mod = pc.load_pipe(pipe)
        prim = pc.primary_variant(mod)
        rep = next(g for g, vs in pc.rand_groups(mod).items() if prim in vs)
        get = lambda d, v, ver: cells.get(f"{ARM}|{pipe}|{d}|{v}|{ver}")
        reuse = [r["session_reuse_max"] for r in mine if r.get("session_reuse_max") is not None]
        dp = get("published", prim, "true")
        head[pipe] = {"source": "rebuilt, one source trial per slot", "sessions": list(SESSIONS[pipe]),
                      "sessions_note": "A5 used session 1" if SESSIONS[pipe] != pc.OPENBMI_SESSIONS[pipe] else None,
                      "verdict": _verdict(dp, reuse), "bar": pc.BARS[pipe], "default_variant": prim,
                      "randomized_group": rep, "default_published": dp,
                      "default_randomized": get("randomized", rep, "true"),
                      "control_published": get("published", prim, "control"),
                      "control_randomized": get("randomized", rep, "control"),
                      "n_not_constructible": sum(r in nc for r in mine),
                      "people_not_constructible": sorted({r["person"] for r in mine if r in nc}),
                      "reuse_max_any": max(reuse) if reuse else None, "slots_per_source_median": _sps(mine)}
    a5 = [r for p in CARRIED for r in _read([pc.out_file("erp", p, False)])]
    carried = pc.summarize(a5)["cells"]
    for pipe in CARRIED:
        mod = pc.load_pipe(pipe)
        prim = pc.primary_variant(mod)
        rep = next(g for g, vs in pc.rand_groups(mod).items() if prim in vs)
        get = lambda d, v, ver: carried.get(f"erp|{pipe}|{d}|{v}|{ver}")
        mine = [r for r in a5 if r["pipe"] == pipe and ((r["design"], r["variant"]) in
                                                        (("published", prim), ("randomized", rep)))]
        reuse = [r["session_reuse_max"] for r in mine if r.get("session_reuse_max") is not None]
        dp = get("published", prim, "true")
        head[pipe] = {"source": "carried over from the pinned A5 rows (one trial per unit)",
                      "verdict": _verdict(dp, reuse), "bar": pc.BARS[pipe], "default_variant": prim,
                      "randomized_group": rep, "default_published": dp,
                      "default_randomized": get("randomized", rep, "true"),
                      "control_published": get("published", prim, "control"),
                      "control_randomized": get("randomized", rep, "control"),
                      "reuse_max_any": max(reuse) if reuse else None, "slots_per_source_median": _sps(mine)}
    checks = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(rows_dir, "cruse12p_constructible__*.json")))
              if not os.path.basename(f).startswith("._")]
    out = {"label": LABEL, "cells": cells, "headline": head, "not_constructible": NOT_CONSTRUCTIBLE,
           "cruse12p_check": {"n_people": len(checks), "n_constructible": sum(c["constructible"] for c in checks),
                              "reasons": dict(collections.Counter(c["reason"].split(":")[0] for c in checks))},
           "bars": pc.BARS}
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
        out = a.out or (OUT + "_dev" if a.dev else OUT)
        run(a.workers, a.people.split(",") if a.people else None, a.dev, out)
    elif a.platform_check:
        platform_check(a.platform_check)
    elif a.summarize:
        write_summary()
        print(SUMMARY)


if __name__ == "__main__":
    main()
