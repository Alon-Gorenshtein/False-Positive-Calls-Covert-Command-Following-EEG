"""Fidelity-corrected readings (part C, C3; added later, descriptive). New reading functions BESIDE the frozen modules: they call
bench.pipe_cornell and bench.pipe_dubois for features, pruning, statistics and the configuration search unchanged,
and edit nothing that the frozen-code hash covers (415921a4..., bench/locked_spec.code_files). Their rows never go
to _bench_shards and never enter the fixed min-rule selection or the headline.
- cornell, c3_jackknife_bh_keep40_r4_om2contig. Base jackknife_bh_keep40_r4: the default's FDR, pruning and run
  count with the papers' jackknife variance ("p <= 0.05 by jackknife method", GF11 §2.8, CU18 p. 1408). OM1 as
  frozen. OM2 on the all-runs TGT after FDR over all channels x frequencies, but a difference counts only as a run of
  FDR-surviving, same-sign bins in one channel spanning > 2 Hz (last minus first bin frequency; GF11's contiguity
  rule for "a difference identified by the TGT", the rule OM1 already applies). Call = OM1 and OM2 (contiguous). The
  frozen any-bin OM2 and its call are kept in extra.
- dubois, reading B, c3_b_assessment_{random,blocked}_phase1. Base assessment_{random,blocked}_phase1 (the default's
  run type and call). Features, folds, configuration search and smoothing as frozen (the folds are drawn from the
  same rng in the same order, so a seed shared with the frozen run() gives the same folds). The peak TIMES come from
  the selected configuration's fold-mean smoothed DA curve, in the task (0.4-7 s) and baseline (-1-0 s) intervals;
  each fold's smoothed DA at those two times enters the one-tailed paired t-test (the frozen zero-variance rule).
  Call = the stated Phase-I rule, peak DA > 70% or the t-test p < 0.05; t_only and the frozen reading A's p are kept
  in extra. The +-150-ms refinement on the unsmoothed curve is not implemented (as in the frozen module). Driver:
  the readings on the Phase-1 windows.
- Null sets ncmd_bdsp and ncmd_pdoc, pipelines cornell and dubois. Recordings: 114's analysis listing (the same
  touched-before-freeze skip, every recording checked against the split). Windows: 114._plan with the null run's
  context (resp none, Columbia's 113 reuse not involved), so each published (seq) overlay is the Phase-1 window of the
  published rows; --first-only keeps j = 0 (115's first-overlay rule).
- Each reading runs on its base variant's schedule drawn with 114's schedule seed (_rng(pipe, uid, j, 1)) and with
  the seed 114 gives the base variant's run() (_rng(pipe, uid, j, 2, crc(base))): du Bois random CV then has the
  folds of the frozen default run on that window. pDoC runs inside wr.pdoc_overrides, as 114 does.
- Rows -> output/intermediate/_c3_shards/<null>__<pipe>/rows<shard>.jsonl (resumable, appended; never _bench_shards).
  --merge -> output/tables/c3_fidelity.json: per reading, the first-overlay rate and the rate over every published
  overlay, patient-bootstrap 95% CI (B = 2000); N/A and error rows counted, left out of denominators (as 115).
python 130_fidelity_readings.py --null ncmd_bdsp --pipe cornell --shard i --nshards n [--limit N] [--first-only] | --merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import glob
import hashlib
import importlib
import json

import numpy as np
from scipy.stats import norm, t as tdist

from bench import engine as en
from bench import motor_common as mc
from bench import pipe_cornell as pc
from bench import pipe_dubois as pdb
from bench import wrappers as wr

rn = importlib.import_module("114_benchmark")

SPEC_PATH = None
LOCKED_HASH = "415921a4bde6ca83b47691cb7ab1b7713c25691029b604541312a932c598cf45"
BENCH_SHARDS = (os.path.join(en.ROOT, "output", "intermediate", "_bench_shards"),
                os.path.join(en.ROOT, "output", "intermediate", "_bench_shards_dev"))

READINGS = {
    "cornell": {"c3_jackknife_bh_keep40_r4_om2contig": {"base": "jackknife_bh_keep40_r4"}},
    "dubois": {"c3_b_assessment_random_phase1": {"base": "assessment_random_phase1"},
               "c3_b_assessment_blocked_phase1": {"base": "assessment_blocked_phase1"}},
}
MODULES = {"cornell": pc, "dubois": pdb}


def readings(pipe):
    return list(READINGS.get(pipe, {}))


def base(pipe, reading):
    return READINGS[pipe][reading]["base"]


def schedule(pipe, start_s, rng, reading):
    """The reading's design: its base variant's schedule."""
    return MODULES[pipe].schedule(start_s, rng, base(pipe, reading))


def run(pipe, rec, ov, reading, rng=None):
    return {"cornell": cornell, "dubois": dubois}[pipe](rec, ov, reading, rng)



def om2_contiguous(z, rej, f):
    """True if the FDR-surviving bins rej (ch, f) form a same-sign run of z in one channel spanning > 2 Hz."""
    return any(r[4] > pc.MIN_RANGE_HZ for r in pc.regions(rej, np.sign(z), f))


def cornell(rec, ov, reading="c3_jackknife_bh_keep40_r4_om2contig", rng=None, base_variant=None):
    c = mc.cfg(pc.VARIANTS, base_variant or base("cornell", reading))
    F = dict(pc._feat(rec, ov))
    lab_ep = np.asarray(ov.labels)[F["cmd"]]
    F["lab"] = lab_ep
    keep = pc.kept(F, lab_ep, c["prune"])
    pc._require(F, lab_ep, keep)
    runs = sorted(np.unique(F["run"]).tolist())
    o = pc.outcome(F, lab_ep, keep, runs, c)
    inr = np.isin(F["run"], runs)
    m1, m2 = keep & inr & (lab_ep == pc.TASK), keep & inr & (lab_ep == pc.REST)
    contig = False
    if m1.sum() >= 2 and m2.sum() >= 2:
        z = pc.zscore(*pc.tgt(F["S"][m1], F["S"][m2]), c["var"])
        rej = mc.fdr_reject(2 * norm.sf(np.abs(z)).ravel(), pc.Q_FDR, c["fdr"]).reshape(z.shape)
        contig = om2_contiguous(z, rej, F["f"])
    return mc.result(o["zmax"], o["qmin"], o["om1"] and contig, om1=o["om1"], om2_contig=bool(contig),
                     om2_anybin=o["om2"], call_anybin=bool(o["om1"] and o["om2"]), n_kept=int(keep.sum()),
                     n_epochs=int(len(keep)))



def _paired_greater(d):
    """One-tailed paired t of fold differences d with the frozen zero-variance rule (pipe_dubois.paired_t)."""
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return (np.inf if m > 0 else 0.0), (0.0 if m > 0 else 1.0)
    tv = m / (s / np.sqrt(len(d)))
    return float(tv), float(tdist.sf(tv, len(d) - 1))


def reading_b(mean_best, fold_best):
    """(t, p, task time index, baseline time index): peak times of the fold-mean curve, per-fold values at them."""
    task, bm = pdb._masks()
    ti = int(np.nonzero(task)[0][np.argmax(mean_best[task])])
    bi = int(np.nonzero(bm)[0][np.argmax(mean_best[bm])])
    tv, p = _paired_greater(fold_best[:, ti] - fold_best[:, bi])
    return tv, p, ti, bi


def dubois(rec, ov, reading="c3_b_assessment_random_phase1", rng=None, base=None):
    """Reading B. base: the base variant (a VARIANTS key, or a dict as the frozen tests pass); default the reading's."""
    c = mc.cfg(pdb.VARIANTS, base if base is not None else READINGS["dubois"][reading]["base"])
    g = mc.gen(rng)
    P = pdb._feat(rec, ov)
    configs = c.get("configs", pdb.CONFIGS)
    y = np.asarray(ov.labels)
    folds = pdb.make_folds(y, np.asarray(ov.meta["block"]), c["cv"], g)
    best, mean, sm = pdb.search(P, y, folds, configs)
    task, _ = pdb._masks()
    peak = float(mean[best, task].max())
    tv, p, ti, bi = reading_b(mean[best], sm[best])
    _, p_a = pdb.paired_t(sm[best])
    calls = {"phase1": bool(peak > pdb.DA_70 or p < pdb.ALPHA), "t_only": bool(p < pdb.ALPHA)}
    return mc.result(tv, p, calls[c["call"]], peak_da=peak, t_peak_s=float(pdb.TIMES[ti]),
                     t_base_s=float(pdb.TIMES[bi]), p_t_A=p_a, config=configs[best], calls=calls)



def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def in_dir(path, fn):
    """fn() with path as the working directory (locked_spec lists the hashed files relative to code/)."""
    with _cwd(path):
        return fn()


def locked_hash():
    """Not available in this copy: the frozen-code hash is checked in the private working copy."""
    raise RuntimeError("the frozen-code hash is checked in the private working copy only")


def pinned(files, spec_path=None):
    """Not applied in this copy: the check reads files of the private working copy."""
    return True


def guard(files, spec_path=None):
    """Not applied in this copy: the check reads files of the private working copy."""
    return None


def outside_bench_shards(path):
    """Refuse any output path inside the benchmark's shard directories."""
    p = os.path.abspath(path)
    for d in BENCH_SHARDS:
        d = os.path.abspath(d)
        if p == d or p.startswith(d + os.sep):
            raise RuntimeError(f"{path}: these outputs never go to _bench_shards")
    return path


SH = os.path.join(en.ROOT, "output", "intermediate", "_c3_shards")
OUT = os.path.join(en.ROOT, "output", "tables", "c3_fidelity.json")
NULLS = ("ncmd_bdsp", "ncmd_pdoc")
PIPES = tuple(READINGS)
FILES = [os.path.abspath(__file__)]
DRAWS, SEED = 2000, 20261016


def ctx_for(spec, n_recs):
    """114's context of the Phase-1 null run (resp none, native montage, analysis split)."""
    return {"null": spec["name"], "split": "analysis", "montage": "native", "resp": "none", "n_recs": n_recs,
            "max_overlays": None, "locator": None, "uncorrected": None, "col113": None}


def shard_file(null, pipe, shard, root=None):
    return os.path.join(root or SH, f"{null}__{pipe}", f"rows{shard:02d}.jsonl")


def phase1_jobs(rec, mod, spec, n_recs, first_only=False, specs=None):
    """([{j, start_s, reading, base, ov, err, rng}], why): one job per reading on every published overlay."""
    specs = specs if specs is not None else rn.variant_specs(mod, spec["source"])
    plan, _, _, why = rn._plan(rec, mod, spec, ctx_for(spec, n_recs), specs)
    jobs = []
    for o in plan:
        if not o["seq"] or (first_only and o["j"] != 0):
            continue
        for reading in readings(mod.NAME):
            b = base(mod.NAME, reading)
            sp = specs[b] if b in specs else rn._spec(mod, b, spec["source"])
            ov, err, _ = rn._call(rn._sched, mod, mod.schedule, sp, o["start_s"], rn._rng(mod.NAME, rec.uid, o["j"], 1))
            if ov is not None and not rn._inside(mod, ov, o):
                ov, err = None, "extent_outside_window"
            jobs.append({"j": o["j"], "start_s": o["start_s"], "reading": reading, "base": b, "ov": ov, "err": err,
                         "rng": rn._rng(mod.NAME, rec.uid, o["j"], 2, rn._crc(b))})
    return jobs, (None if plan else why)


def _done(fn):
    if not os.path.exists(fn):
        return set()
    with open(fn) as fh:
        return {json.loads(l)["key"] for l in fh if l.strip()}


def run_shard(null, pipe, shard, nshards, limit=None, first_only=False, out_root=None):
    guard(FILES)
    spec = rn.parse_null(null)
    if null not in NULLS or pipe not in PIPES:
        raise ValueError(f"C3 readings exist for {PIPES} on {NULLS}, not {pipe} on {null}")
    rn.guard_split("analysis")
    mod = rn.load_pipe(pipe)
    fn = outside_bench_shards(shard_file(null, pipe, shard, out_root))
    skip = rn.touched_uids()
    items = rn.list_items(spec, "analysis", skip)
    specs = rn.variant_specs(mod, spec["source"])
    done = _done(fn)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    pdoc_g = wr.pdoc_overrides(mod) if spec["source"] == "pdoc" else {}
    with open(fn, "a") as fh, wr.module_globals(mod, pdoc_g):
        def write(row):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
        for rec, _ in rn.iter_recordings(spec, "analysis", limit, shard, nshards, skip, items=items):
            jobs, why = phase1_jobs(rec, mod, spec, len(items), first_only, specs)
            base_row = {"null": null, "pipe": pipe, "uid": rec.uid, "patient": rec.patient}
            if not jobs:
                if f"{rec.uid}|nofit" not in done:
                    write({**base_row, "key": f"{rec.uid}|nofit", "status": why or "no_published_overlay"})
                continue
            for jb in jobs:
                key = f"{rec.uid}|{jb['j']}|{jb['reading']}"
                if key in done:
                    continue
                if jb["ov"] is None:
                    res, err, sec = None, jb["err"], 0.0
                else:
                    res, err, sec = rn._call_run(mod, run, pipe, rec, jb["ov"], jb["reading"], jb["rng"])
                write({**base_row, "key": key, "j": jb["j"], "start_s": jb["start_s"], "reading": jb["reading"],
                       "base": jb["base"], "status": rn._status(err), "error": err,
                       "call": None if res is None else bool(res["call"]),
                       "p": None if res is None else res["p"], "stat": None if res is None else res["stat"],
                       "seconds": sec, **rn.scalars(res and res.get("extra"))})
            rn._clear_cache()


def _rate(rows, rng):
    """{n, calls, rate, ci95} over the ok rows, patient-bootstrap percentile CI (each patient one cluster)."""
    pats = sorted({r["patient"] for r in rows})
    if not pats:
        return {"n": 0, "calls": 0, "rate": None, "ci95": None}
    idx = {p: i for i, p in enumerate(pats)}
    c, n = np.zeros(len(pats)), np.zeros(len(pats))
    for r in rows:
        c[idx[r["patient"]]] += bool(r["call"])
        n[idx[r["patient"]]] += 1
    draw = rng.integers(0, len(pats), size=(DRAWS, len(pats)))
    rates = c[draw].sum(1) / n[draw].sum(1)
    return {"n": int(n.sum()), "calls": int(c.sum()), "patients": len(pats), "rate": float(c.sum() / n.sum()),
            "ci95": [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]}


def merge(root=None, out=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(root or SH, "*", "rows*.jsonl"))):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    res = {"part": "C3", "added_later": True,
           "note": "fidelity-corrected readings, descriptive: they never enter the pinned min-rule selection or "
                   "the headline, and never go to _bench_shards",
           "locked_code_hash": None, "files": {os.path.basename(f): sha256(f) for f in FILES}}
    rng = np.random.default_rng(SEED)
    for null in sorted({r["null"] for r in rows}):
        for pipe in sorted({r["pipe"] for r in rows if r["null"] == null}):
            sub = [r for r in rows if r["null"] == null and r["pipe"] == pipe]
            out_p = {"recordings": len({r["uid"] for r in sub}),
                     "no_window": sum(1 for r in sub if r["key"].endswith("|nofit"))}
            for reading in readings(pipe):
                rr = [r for r in sub if r.get("reading") == reading]
                ok = [r for r in rr if r["status"] == "ok"]
                stat = {}
                for r in rr:
                    stat[r["status"]] = stat.get(r["status"], 0) + 1
                out_p[reading] = {"first_overlay": _rate([r for r in ok if r["j"] == 0], rng),
                                  "all_published_overlays": _rate(ok, rng), "statuses": stat}
            res.setdefault(null, {})[pipe] = out_p
    out = out or OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(out + ".tmp", out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", choices=NULLS)
    ap.add_argument("--pipe", choices=PIPES)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--first-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        run_shard(a.null, a.pipe, a.shard, a.nshards, a.limit, a.first_only)


if __name__ == "__main__":
    main()
