"""White-noise false-call rate of every stated-rule eligible variant (added later; synthetic noise, no patient data).
Built test-first (code/test_147_white_noise_eligible.py). 129_atlas.py is left as fixed; this script reuses its
noise, seeds and harness conventions with its own shard folder.
- Variants: every variant marked stated_rule_eligible (the union; stated_rule_eligible_strict must lie inside it) in
  docs/stated-rule-motor.json and docs/stated-rule-erp.json, sokoliuk21's included; Columbia's as_coded:call_paper
  (as_coded run, the papers' call: AUC > 0.5 and p < 0.05) and the C3 readings (130_fidelity_readings.run on their
  base schedule) as in 145. Each runs on BDSP specs, 19 channels (129.NAMES), 128 Hz.
- Cell: 129's white noise, replicates r 0-59 for every pipeline, uid "atlas|white|<r>", noise seeded
  [129.SEED, crc(pipe), crc("white"), r]; schedule _rng(pipe, uid, 0, 1), run _rng(pipe, uid, 0, 2, crc(the variant
  run: base for a C3 reading, as_coded for call_paper)), start 129.PAD_S + 114.LEAD_S: 129.run_unit's seeds.
- Pairing: a variant whose own clean-window need fits the atlas unit's (114.need_seconds over the variants 129 ran
  for the pipeline) runs on the atlas draw (paired: the same noise and schedules as the atlas rows; where the atlas
  ran the same variant its calls are compared). A longer variant runs on its pipeline's own longer draw (same seed,
  the longest need of those variants), labelled unpaired.
- Rows -> output/intermediate/_white_shards (never _atlas_shards or _bench_shards); --merge ->
  output/tables/white_noise_eligible.json: per variant k / n (N/A and error rows counted, outside the denominator)
  and the exact Clopper-Pearson 95% CI. Refused unless the classification files have the given sha256; the run stops (resumable) whenever swap use exceeds 2 GB (macOS) or
  available memory falls below 8 GB (Linux).
python3 147_white_noise_eligible.py --motor-sha HEX --erp-sha HEX --shard i --nshards n [--pipes a,b] [--limit N]
  | --merge --motor-sha HEX --erp-sha HEX"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import hashlib
import importlib
import json
import re
import subprocess
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
at = importlib.import_module("129_atlas")
rn, fr, en = at.rn, at.fr, at.en
sr = importlib.import_module("145_stated_rule")
st = sr.st

ROOT = en.ROOT
SH = os.path.join(ROOT, "output", "intermediate", "_white_shards")
OUT = os.path.join(ROOT, "output", "tables", "white_noise_eligible.json")
CELL = "white"
N = 60
CALL_PAPER = sr.CALL_PAPER
SWAP_MAX_MB = 2048.0
MEM_MIN_GB = 8.0
META = sr.META



def eligible_variants(cl):
    """{pipe: [variants]} marked stated_rule_eligible; a strict-eligible variant outside it is refused."""
    out = {}
    for pipe, vs in cl.items():
        if pipe in META:
            continue
        for v, e in vs.items():
            if e.get("stated_rule_eligible_strict") and not e.get("stated_rule_eligible"):
                raise RuntimeError(f"{pipe}|{v}: strict-eligible but not union-eligible: refused")
            if e.get("stated_rule_eligible"):
                out.setdefault(pipe, []).append(v)
    return {p: sorted(v) for p, v in out.items()}


def variant_set(cl, cl_ref):
    """{pipe: {variant: {eligible_now, eligible_at_ref, strict_now}}} over every union-eligible variant of the current
    files (the strict-eligible ones lie inside); eligibility at 145.CLASS_REF is recorded beside."""
    now, ref = eligible_variants(cl), eligible_variants(cl_ref)
    out = {}
    for p in sorted(now):
        for v in now[p]:
            out.setdefault(p, {})[v] = {"eligible_now": v in now.get(p, []), "eligible_at_ref": v in ref.get(p, []),
                                        "strict_now": bool(cl.get(p, {}).get(v, {}).get(sr.STRICT_FLAG))}
    return out


def run_spec(pipe, v):
    """How a classification variant runs: kind (published | c3), the frozen variant run (arg), its spec, and the
    result field holding the call (call_paper for Columbia's papers' rule)."""
    mod = rn.load_pipe(pipe)
    if v == CALL_PAPER:
        return {"kind": "published", "arg": "as_coded", "spec": rn._spec(mod, "as_coded", at.SOURCE),
                "call_field": "call_paper"}
    if v in fr.READINGS.get(pipe, {}):
        b = fr.base(pipe, v)
        return {"kind": "c3", "arg": b, "spec": rn._spec(mod, b, at.SOURCE), "reading": v, "call_field": None}
    return {"kind": "published", "arg": v, "spec": rn._spec(mod, v, at.SOURCE), "call_field": None}


def R_for(pipe, variants):
    return {v: run_spec(pipe, v) for v in variants}


def atlas_need(pipe):
    mod = rn.load_pipe(pipe)
    return rn.need_seconds(mod, at._arg_names(at.roles(mod, at.validity_primaries().get(pipe))), at.SOURCE)


def plan(pipe, variants):
    """The paired variants (fit the atlas unit's window: its draw) and the unpaired ones (their own longer draw)."""
    mod = rn.load_pipe(pipe)
    need0 = atlas_need(pipe)
    own = {v: rn.need_seconds(mod, [run_spec(pipe, v)["arg"]], at.SOURCE) for v in variants}
    paired = sorted(v for v in variants if own[v] <= need0)
    unpaired = sorted(v for v in variants if own[v] > need0)
    return {"atlas_need_s": need0, "paired": paired, "unpaired": unpaired,
            "unpaired_need_s": max((own[v] for v in unpaired), default=None), "own_need_s": own}



def call_of(res, field):
    if field is None:
        return bool(res["call"])
    if field not in (res.get("extra") or {}) or res["extra"][field] is None:
        raise RuntimeError(f"result without {field}: refused (missing is not a negative)")
    return bool(res["extra"][field])


def run_unit(pipe, r, R, need, draw, done, write):
    """129.run_unit's recording, overlay and seeds at a given clean-window need, for the variants in R."""
    mod = rn.load_pipe(pipe)
    uid = f"atlas|{CELL}|{r}"
    keys = {n: f"{pipe}|{CELL}|{r}|{draw}|{n}" for n in R}
    todo = [n for n in R if keys[n] not in done]
    if not todo:
        return
    rec = en.Recording(source="atlas", uid=uid, patient=uid,
                       x=at.noise(at.CELLS[CELL], at.PAD_S + need + at.PAD_S,
                                  np.random.default_rng([at.SEED, rn._crc(pipe), rn._crc(CELL), r])),
                       fs=at.FS, names=list(at.NAMES), meta={"dev": False, "cell": CELL})
    o = {"j": 0, "start_s": at.PAD_S + rn.LEAD_S, "window": (at.PAD_S, at.PAD_S + need)}
    rn._clear_cache()
    for n in todo:
        e = R[n]
        ov, err, _ = rn._call(rn._sched, mod, mod.schedule, e["spec"], o["start_s"], rn._rng(pipe, uid, 0, 1))
        g_run = rn._rng(pipe, uid, 0, 2, rn._crc(e["arg"]))
        if ov is not None and not rn._inside(mod, ov, o):
            ov, err = None, "extent_outside_window"
        if ov is None:
            res, sec = None, 0.0
        elif e["kind"] == "c3":
            res, err, sec = rn._call_run(mod, fr.run, pipe, rec, ov, e["reading"], g_run)
        else:
            res, err, sec = rn._call_run(mod, rn._run, mod, mod.run, e["spec"], rec, ov, g_run)
        call = None
        if res is not None:
            try:
                call = call_of(res, e["call_field"])
            except RuntimeError as x:
                res, err = None, f"error: {x}"
        write({"key": keys[n], "pipe": pipe, "cell": CELL, "r": r, "draw": draw, "need_s": need, "variant": n,
               "kind": e["kind"], "run_as": e["arg"], "status": rn._status(err), "error": err, "call": call,
               "p": None if res is None else res["p"], "stat": None if res is None else res["stat"],
               "seconds": sec, **rn.scalars(res and res.get("extra"))})
    rn._clear_cache()


def swap_used_mb(text=None):
    text = text if text is not None else subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                                                        text=True, check=True).stdout
    m = re.search(r"used = ([0-9.]+)M", text)
    if not m:
        raise RuntimeError(f"cannot read swap use from {text!r}")
    return float(m.group(1))


def check_swap(text=None):
    used = swap_used_mb(text)
    if used > SWAP_MAX_MB:
        raise SystemExit(f"swap use {used:.0f} MB is above {SWAP_MAX_MB:.0f} MB: stopped (rerun resumes)")
    return used


def mem_available_gb(meminfo=None):
    if meminfo is None:
        with open("/proc/meminfo") as fh:
            meminfo = fh.read()
    m = re.search(r"MemAvailable:\s+(\d+) kB", meminfo)
    if not m:
        raise RuntimeError("cannot read MemAvailable")
    return int(m.group(1)) / 1048576


def check_memory(platform=None, meminfo=None, swap=None):
    """macOS: swap in use at most 2 GB; Linux: at least 8 GB available. Otherwise stop (a rerun resumes)."""
    platform = platform or sys.platform
    if platform == "darwin":
        return check_swap(swap)
    gb = mem_available_gb(meminfo)
    if gb < MEM_MIN_GB:
        raise SystemExit(f"{gb:.1f} GB available, below {MEM_MIN_GB:.0f} GB: stopped (rerun resumes)")
    return gb


def shard_file(shard, root=None):
    root = os.path.abspath(root or SH)
    if root == os.path.abspath(at.SH) or root.startswith(os.path.abspath(at.SH) + os.sep):
        raise RuntimeError(f"{root}: never the atlas shards")
    return fr.outside_bench_shards(os.path.join(root, f"rows{shard:02d}.jsonl"))


def _done(fn):
    if not os.path.exists(fn):
        return set()
    with open(fn) as fh:
        return {json.loads(l)["key"] for l in fh if l.strip()}


def units(pipes):
    """[(pipe, r)], replicate-major so every shard gets a mix of pipelines."""
    return sorted(((p, r) for p in pipes for r in range(N)), key=lambda u: (u[1], pipes.index(u[0])))


def guard(motor_sha, erp_sha):
    """The classification files' sha256 (the code-hash check is not applied here)."""
    sr.check_classification(sr.FILES, (motor_sha, erp_sha))


def run_shard(shard, nshards, motor_sha, erp_sha, pipes=None, limit=None, root=None):
    guard(motor_sha, erp_sha)
    check_memory()
    elig = {p: sorted(vs) for p, vs in variant_set(sr.load_classification(), sr.load_classification_at()).items()}
    pipes = [p for p in sorted(elig) if not pipes or p in pipes]
    plans = {p: plan(p, elig[p]) for p in pipes}
    Rs = {p: (R_for(p, plans[p]["paired"]), R_for(p, plans[p]["unpaired"])) for p in pipes}
    us = [u for i, u in enumerate(units(pipes)) if i % nshards == shard][:limit]
    fn = shard_file(shard, root)
    done = _done(fn)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "a") as fh:
        def write(row):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
        for pipe, r in us:
            check_memory()
            Rp, Ru = Rs[pipe]
            if Rp:
                run_unit(pipe, r, Rp, plans[pipe]["atlas_need_s"], "paired", done, write)
            if Ru:
                run_unit(pipe, r, Ru, plans[pipe]["unpaired_need_s"], "unpaired", done, write)



def atlas_name(pipe, v):
    """The atlas row name of the same variant, where the atlas ran it."""
    return "as_coded" if v == CALL_PAPER else ("c3:" + v if v in fr.READINGS.get(pipe, {}) else v)


def atlas_calls(root=None):
    """{atlas key: {status, call}} of the atlas white rows (Columbia's as_coded row carries call_paper too)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root or at.SH, "rows*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if '"cell": "white"' not in line:
                    continue
                r = json.loads(line)
                out[r["key"]] = {"status": r["status"], "call": r["call"], "x_call_paper": r.get("x_call_paper")}
    return out


def summarise(rows, atlas, names):
    """{pipe: {variant: {k, n, rate, ci95_cp, n_na, n_error, draw, need_s, atlas_agreement}}}; names maps (pipe,
    variant) to the atlas name of the same variant (None: not in the atlas)."""
    by = {}
    for r in rows:
        by.setdefault((r["pipe"], r["variant"]), []).append(r)
    out = {}
    for (pipe, v), rs in sorted(by.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        if any(r["call"] is None for r in ok):
            raise RuntimeError(f"{pipe}|{v}: an ok row without a call: refused")
        k, n = sum(bool(r["call"]) for r in ok), len(ok)
        draws = {r["draw"] for r in rs}
        if len(draws) != 1:
            raise RuntimeError(f"{pipe}|{v}: rows on more than one draw")
        draw = draws.pop()
        agree = None
        an = names.get((pipe, v))
        if draw == "paired" and an is not None:
            field = "x_call_paper" if v == CALL_PAPER else "call"
            pairs = [(r, atlas[f"{pipe}|{CELL}|{r['r']}|{an}"]) for r in rs if f"{pipe}|{CELL}|{r['r']}|{an}" in atlas]
            agree = {"n_compared": len(pairs), "n_same_call": sum(r["status"] == a["status"] and
                                                                  (r["call"] == a.get(field, a["call"]))
                                                                  for r, a in pairs)}
        out.setdefault(pipe, {})[v] = {
            "k": k, "n": n, "rate": k / n if n else None, "ci95_cp": st.clopper_pearson(k, n) if n else None,
            "n_na": sum(r["status"].startswith(rn.NA_NO_USABLE) or r["status"] == "extent_outside_window"
                        for r in rs if r["status"] != "ok"),
            "n_error": sum(r["status"].startswith("error") for r in rs),
            "draw": "paired with the atlas draw" if draw == "paired" else
                    f"unpaired: its own longer draw ({rs[0]['need_s']} s clean window)",
            "need_s": rs[0]["need_s"], "atlas_agreement": agree}
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def merge(motor_sha, erp_sha, root=None, out=None):
    sr.check_classification(sr.FILES, (motor_sha, erp_sha))
    rows = []
    for f in sorted(glob.glob(os.path.join(root or SH, "rows*.jsonl"))):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    rows = list({r["key"]: r for r in rows}.values())
    vset = variant_set(sr.load_classification(), sr.load_classification_at())
    elig = {p: sorted(vs) for p, vs in vset.items()}
    have = {(r["pipe"], r["variant"]) for r in rows}
    missing = sorted(f"{p}|{v}" for p, vs in elig.items() for v in vs if (p, v) not in have)
    atlas = atlas_calls()
    names = {(p, v): (atlas_name(p, v) if any(k.startswith(f"{p}|{CELL}|") and k.endswith("|" + atlas_name(p, v))
                                              for k in atlas) else None) for p, v in have}
    s = {"what": __doc__.split("\n")[0], "added_later": True, "cell": CELL, "n_replicates": N, "seed": at.SEED,
         "montage": "19-channel BDSP (129.NAMES), BDSP specs",
         "files": {os.path.basename(f): _sha256(f) for f in (os.path.abspath(__file__), os.path.join(CODE, "129_atlas.py"),
                                                             os.path.join(CODE, "130_fidelity_readings.py"))},
         "locked_code_hash": None,
         "classification": {os.path.relpath(p, ROOT): _sha256(p) for p in sr.FILES},
         "class_reference": sr.CLASS_REF, "n_rows": len(rows), "eligible_without_rows": missing,
         "variants": {p: {v: dict(x, **vset.get(p, {}).get(v, {})) for v, x in t.items()}
                      for p, t in summarise(rows, atlas, names).items()}}
    out = out or OUT
    with open(out + ".tmp", "w") as fh:
        json.dump(s, fh, indent=1, default=float)
    os.replace(out + ".tmp", out)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--motor-sha", required=True)
    ap.add_argument("--erp-sha", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pipes", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    if a.merge:
        merge(a.motor_sha, a.erp_sha)
    else:
        run_shard(a.shard, a.nshards, a.motor_sha, a.erp_sha, a.pipes.split(",") if a.pipes else None, a.limit)


if __name__ == "__main__":
    main()
