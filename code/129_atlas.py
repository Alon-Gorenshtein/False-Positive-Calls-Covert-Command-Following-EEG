"""C1 synthetic calibration atlas (part C, C1; added later; no patient data).
Harness: each module's frozen run() on synthetic noise.
Noise (19 channels, engine.NAMES19, 128 Hz, float32 uV; seeded per pipeline x cell x replicate):
- white: iid Gaussian, SD AMP_UV;
- pink1, pink2: independent per channel, amplitude spectrum f^(-beta/2) (beta 1, 2), flat below F0_HZ, SD AMP_UV;
- alpha (stationary): independent per channel, a Gaussian band at ALPHA_HZ (SD ALPHA_BW_HZ), SD AMP_UV;
- env_sd<s>_tau<t> (EEG-like): 19 pink (beta 1) sources at the 10-20 positions mixed by a Gaussian volume-conduction
  kernel (MIX_SIGMA_M, standard_1020 distances; each channel rescaled to PINK_UV), plus one alpha source (ALPHA_UV)
  over a posterior topography (OCC sites 1, others OCC_FLOOR); the sum times ONE global log-normal amplitude
  envelope exp(s * e(t) - s^2), e an Ornstein-Uhlenbeck process (unit variance, correlation time t s; generated at
  ENV_FS and interpolated), so the power envelope has mean 1 at every log-SD: s in {0, 0.25, 0.5, 0.75} x t in
  {1, 5, 10, 30, 60}. s = 0 is the stationary EEG-like control (5 seeded cells, one per t);
- every family but white adds FLOOR_UV iid sensor noise (unmodulated).
Pipelines (bench/locked_spec.PIPES): each module's FROZEN run() through 114's own helpers, on BDSP specs:
- default = 114.primary_variant(mod, "bdsp"); validity primary = the "validity_primary" name of the 115 BDSP null-set
  outputs (benchmark_ncmd_bdsp.json / benchmark_nstim0_bdsp.json; only that key is read);
- randomised = 114's randomization-based method (wrappers.run_randomized, B = RAND_B, schedule_randomized) of the
  timing group (114.timing_groups representative) of the default and of the validity primary;
- C3 readings (130_fidelity_readings.py) where they exist (cornell, dubois), on their base schedule. One unit = one
  synthetic recording per pipeline x cell x replicate; every variant of the unit runs on it with 114's seeds (uid
  "atlas|<cell>|<r>", overlay j = 0: schedule _rng(pipe, uid, 0, 1), run _rng(..., 0, 2, crc(variant)), randomized
  schedule/run _rng(..., 0, 3/4, crc(group))). The overlay starts at PAD_S + LEAD_S inside a clean window of
  114.need_seconds. n = 60 per cell; columbia and dubois: white, pink1 and env_sd0.5_tau10 only, n = 30. Bars (C1):
  implementation (white noise: validity primaries inside the binomial 95% range of 5%; mbvt and dubois known
  exceptions, reported); mechanism (non-immune validity primaries, 116 immunity type "exposed" from
  mechanism_design_table.csv: one-sided Cochran-Armitage trend of the false-call rate over envelope log-SD, the 5
  correlation times pooled per level, Holm across pipelines; not tested: immune designs, designs 116 types as having
  no label contrast (sokoliuk21's passive ITPC), and pipelines without the full envelope grid (columbia, dubois));
  construction (randomised arms: the same trend test, Holm across arms; pass if none rises); NOGO (if no non-immune
  validity primary exceeds 10% over the log-SD 0.25 and 0.5 cells, the non-stationarity mechanism is dropped). N/A
  and error rows are counted and left out of every denominator. Rows -> output/intermediate/_atlas_shards (never
  _bench_shards); --merge -> output/tables/atlas.json.
python 129_atlas.py --shard i --nshards n [--pipes a,b] [--limit N] | --merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import csv
import glob
import importlib
import json
from collections import OrderedDict

import numpy as np
from scipy.signal import lfilter
from scipy.stats import binom, norm

from bench import engine as en
from bench import locked_spec
from bench import wrappers as wr

rn = importlib.import_module("114_benchmark")
fr = importlib.import_module("130_fidelity_readings")

TABLES = os.path.join(en.ROOT, "output", "tables")
SH = os.path.join(en.ROOT, "output", "intermediate", "_atlas_shards")
OUT = os.path.join(TABLES, "atlas.json")
VP_FILES = {"cmd": os.path.join(TABLES, "benchmark_ncmd_bdsp.json"),
            "erp": os.path.join(TABLES, "benchmark_nstim0_bdsp.json")}
DESIGN_TABLE = os.path.join(TABLES, "mechanism_design_table.csv")
FILES = [os.path.abspath(__file__), os.path.join(en.CODE, "130_fidelity_readings.py")]
SOURCE = "bdsp"
SEED = 20261016
FS = 128.0
NAMES = list(en.NAMES19)
PIPES = list(locked_spec.PIPES)
AMP_UV, FLOOR_UV, F0_HZ = 20.0, 1.0, 0.5
ALPHA_HZ, ALPHA_BW_HZ = 10.0, 1.0
PINK_UV, ALPHA_UV, MIX_SIGMA_M = 15.0, 10.0, 0.06
OCC, OCC_FLOOR = ("O1", "O2", "P3", "P4", "Pz", "T5", "T6"), 0.3
ENV_FS = 32.0
LOG_SD, TAU_S = (0.0, 0.25, 0.5, 0.75), (1, 5, 10, 30, 60)
N_CELL, N_SMALL = 60, 30
SMALL_PIPES, SMALL_CELLS = ("columbia", "dubois"), ("white", "pink1", "env_sd0.5_tau10")
KNOWN_WHITE_EXCEPTIONS = ("mbvt", "dubois")
NOGO_RATE, NOGO_SD = 0.10, (0.25, 0.5)
ALPHA = 0.05
PAD_S = 30.0

CELLS = OrderedDict([("white", {"family": "white"}), ("pink1", {"family": "pink", "beta": 1.0}),
                     ("pink2", {"family": "pink", "beta": 2.0}), ("alpha", {"family": "alpha"})])
for _s in LOG_SD:
    for _t in TAU_S:
        CELLS[f"env_sd{_s:g}_tau{_t:g}"] = {"family": "env", "log_sd": _s, "tau_s": _t}


def cell_names(pipe):
    return list(SMALL_CELLS) if pipe in SMALL_PIPES else list(CELLS)


def n_per_cell(pipe):
    return N_SMALL if pipe in SMALL_PIPES else N_CELL


def units(pipes=None, cells=None, n=None):
    """[(pipe, cell, r)], replicate-major so every shard gets a mix of pipelines and cells."""
    pipes = pipes or PIPES
    out = [(p, c, r) for p in pipes for c in ((cells or {}).get(p) or cell_names(p))
           for r in range(n if n is not None else n_per_cell(p))]
    return sorted(out, key=lambda u: (u[2], list(CELLS).index(u[1]), pipes.index(u[0])))



def _shaped(n, rng, gain):
    f = np.fft.rfftfreq(n, 1 / FS)
    s = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * gain(f), n)
    return s / s.std()


def _pink(n, rng, beta):
    return _shaped(n, rng, lambda f: np.where(f > 0, np.maximum(f, F0_HZ) ** (-beta / 2), 0.0))


def _alpha(n, rng):
    return _shaped(n, rng, lambda f: np.exp(-(f - ALPHA_HZ) ** 2 / (2 * ALPHA_BW_HZ ** 2)))


def envelope(n, fs, sd, tau, rng):
    """exp(sd * e - sd^2), e an Ornstein-Uhlenbeck process with unit variance and correlation time tau (s)."""
    if sd == 0:
        return np.ones(n)
    m = int(np.ceil(n / fs * ENV_FS)) + 2
    phi = np.exp(-1.0 / (ENV_FS * tau))
    e = lfilter([np.sqrt(1 - phi ** 2)], [1.0, -phi], rng.standard_normal(m), zi=[phi * rng.standard_normal()])[0]
    return np.exp(sd * np.interp(np.arange(n) / fs, np.arange(m) / ENV_FS, e) - sd ** 2)


def _mixing():
    P = en._positions(NAMES)
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    return np.exp(-d ** 2 / (2 * MIX_SIGMA_M ** 2))


def noise(cell, dur_s, rng):
    """(19, n) float32 uV for one cell (module docstring)."""
    n = int(round(dur_s * FS))
    fam = cell["family"]
    if fam == "white":
        return (AMP_UV * rng.standard_normal((19, n))).astype(np.float32)
    if fam == "pink":
        x = AMP_UV * np.array([_pink(n, rng, cell["beta"]) for _ in NAMES])
    elif fam == "alpha":
        x = AMP_UV * np.array([_alpha(n, rng) for _ in NAMES])
    else:
        S = np.array([_pink(n, rng, 1.0) for _ in NAMES])
        x = np.einsum("ij,jt->it", _mixing(), S)
        x = PINK_UV * x / x.std(1, keepdims=True)
        occ = np.array([1.0 if c in OCC else OCC_FLOOR for c in NAMES])
        x = (x + ALPHA_UV * occ[:, None] * _alpha(n, rng)[None]) * envelope(n, FS, cell["log_sd"], cell["tau_s"],
                                                                            rng)[None]
    return (x + FLOOR_UV * rng.standard_normal((19, n))).astype(np.float32)



def validity_primaries(files=None):
    """{pipe: validity-primary variant}: ONLY the 'validity_primary' key of each 115 BDSP null-set output is read."""
    out = {}
    for fam, f in (files or VP_FILES).items():
        if not os.path.exists(f):
            raise RuntimeError(f"{f} is missing: the atlas runs each pipeline's validity primary (115 merge output)")
        with open(f) as fh:
            pub = json.load(fh)["published"]
        out.update({p: v["validity_primary"] for p, v in pub.items() if isinstance(v, dict) and
                    v.get("validity_primary")})
    return out


def immunity(path=None):
    """{(pipe, variant, kind): {immunity_type, x_ref_class}} from 116's design table (BDSP rows)."""
    out = {}
    with open(path or DESIGN_TABLE) as fh:
        for row in csv.DictReader(fh):
            if SOURCE in row["sources"].split("+"):
                out[(row["pipe"], row["variant"], row["kind"])] = {"immunity_type": row["immunity_type"] or None,
                                                                   "x_ref_class": row["x_ref_class"] or None}
    return out


def roles(mod, vp):
    """OrderedDict name -> {kind, roles, spec[, group][, reading, base]} of the variants one unit runs."""
    specs = rn.variant_specs(mod, SOURCE)
    prim = rn.primary_variant(mod, SOURCE)
    out = OrderedDict()

    def add(name, kind, role, spec, **kw):
        out.setdefault(name, {"kind": kind, "roles": [], "spec": spec, **kw})["roles"].append(role)

    pairs = [("default", prim)] + ([("validity_primary", vp)] if vp else [])
    for role, v in pairs:
        add(v, "published", role, specs[v] if v in specs else rn._spec(mod, v, SOURCE))
    tiers = rn.tiling_tiers(mod, specs)
    for role, v in pairs:
        tier = next(t for _, t in tiers if v in t)
        groups = rn.timing_groups(mod, {u: specs[u] for u in tier}, tier, prim)
        g = next(rep for rep, vs in groups.items() if v in vs)
        add("rand:" + g, "randomized", "randomised_" + role, specs[g], group=g)
    for reading in fr.readings(mod.NAME):
        b = fr.base(mod.NAME, reading)
        add("c3:" + reading, "c3", "c3", specs[b] if b in specs else rn._spec(mod, b, SOURCE), reading=reading, base=b)
    return out


def _arg_names(R):
    return sorted({e["group"] if e["kind"] == "randomized" else e.get("base", n) for n, e in R.items()})



def shard_file(shard, root=None):
    return os.path.join(root or SH, f"rows{shard:02d}.jsonl")


def _done(fn):
    if not os.path.exists(fn):
        return set()
    with open(fn) as fh:
        return {json.loads(l)["key"] for l in fh if l.strip()}


def run_unit(pipe, cell, r, R, done, write):
    mod = rn.load_pipe(pipe)
    uid = f"atlas|{cell}|{r}"
    keys = {n: f"{pipe}|{cell}|{r}|{n}" for n in R}
    todo = [n for n in R if keys[n] not in done]
    if not todo:
        return
    need = rn.need_seconds(mod, _arg_names(R), SOURCE)
    c = CELLS[cell]
    rec = en.Recording(source="atlas", uid=uid, patient=uid,
                       x=noise(c, PAD_S + need + PAD_S, np.random.default_rng([SEED, rn._crc(pipe), rn._crc(cell), r])),
                       fs=FS, names=list(NAMES), meta={"dev": False, "cell": cell})
    o = {"j": 0, "start_s": PAD_S + rn.LEAD_S, "window": (PAD_S, PAD_S + need)}
    rn._clear_cache()
    for n in todo:
        e = R[n]
        if e["kind"] == "randomized":
            rn._clear_cache()
            ov, err, _ = rn._call(rn._sched, mod, mod.schedule_randomized, e["spec"], o["start_s"],
                                  rn._rng(pipe, uid, 0, 3, rn._crc(e["group"])))
            fn, args = rn._run, (mod, lambda r_, o_, a, gg: wr.run_randomized(mod, r_, o_, a, gg), e["spec"], rec)
            g_run = rn._rng(pipe, uid, 0, 4, rn._crc(e["group"]))
        else:
            ov, err, _ = rn._call(rn._sched, mod, mod.schedule, e["spec"], o["start_s"], rn._rng(pipe, uid, 0, 1))
            if e["kind"] == "c3":
                fn, args = fr.run, (pipe, rec)
                g_run = rn._rng(pipe, uid, 0, 2, rn._crc(e["base"]))
            else:
                fn, args = rn._run, (mod, mod.run, e["spec"], rec)
                g_run = rn._rng(pipe, uid, 0, 2, rn._crc(n))
        if ov is not None and not rn._inside(mod, ov, o):
            ov, err = None, "extent_outside_window"
        if ov is None:
            res, sec = None, 0.0
        elif e["kind"] == "c3":
            res, err, sec = rn._call_run(mod, fn, *args, ov, e["reading"], g_run)
        else:
            res, err, sec = rn._call_run(mod, fn, *args, ov, g_run)
        if e["kind"] == "randomized":
            rn._clear_cache()
        write({"key": keys[n], "pipe": pipe, "cell": cell, "family": c["family"], "log_sd": c.get("log_sd"),
               "tau_s": c.get("tau_s"), "r": r, "variant": n, "kind": e["kind"], "roles": e["roles"],
               "group": e.get("group"), "status": rn._status(err), "error": err,
               "call": None if res is None else bool(res["call"]), "p": None if res is None else res["p"],
               "stat": None if res is None else res["stat"], "seconds": sec, **rn.scalars(res and res.get("extra"))})
    rn._clear_cache()


def run_shard(shard, nshards, pipes=None, limit=None, out_root=None, guard=True, cells=None, n=None):
    if guard:
        fr.guard(FILES)
    fn = fr.outside_bench_shards(shard_file(shard, out_root))
    vp = validity_primaries()
    us = units(pipes, cells, n)
    us = [u for i, u in enumerate(us) if i % nshards == shard][:limit]
    done = _done(fn)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    R = {p: roles(rn.load_pipe(p), vp.get(p)) for p in sorted({u[0] for u in us})}
    with open(fn, "a") as fh:
        def write(row):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
        for pipe, cell, r in us:
            run_unit(pipe, cell, r, R[pipe], done, write)



def cochran_armitage(x, r, N):
    """(z, one-sided p for a RISE): Cochran-Armitage trend test of r/N over scores x (variance with N, not N - 1).
    No call at all, or every run a call: (0, 1)."""
    x, r, N = (np.asarray(v, float) for v in (x, r, N))
    p = r.sum() / N.sum()
    var = p * (1 - p) * (np.sum(N * x ** 2) - np.sum(N * x) ** 2 / N.sum())
    if var <= 0:
        return 0.0, 1.0
    z = float(np.sum(x * (r - N * p)) / np.sqrt(var))
    return z, float(norm.sf(z))


def holm(ps):
    """Holm step-down adjusted p values, in input order."""
    m = len(ps)
    order = np.argsort(ps, kind="stable")
    adj, run = [0.0] * m, 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def binom_range(n, p0=ALPHA):
    """Central binomial 95% range of the call count at the nominal rate."""
    return int(binom.ppf(0.025, n, p0)), int(binom.ppf(0.975, n, p0))



def _cells_table(rows):
    t = {}
    for r in rows:
        d = t.setdefault(r["cell"], {"n": 0, "calls": 0, "rate": None, "na": 0, "error": 0})
        if r["status"] == "ok":
            d["n"] += 1
            d["calls"] += bool(r["call"])
        elif r["status"].startswith("na"):
            d["na"] += 1
        else:
            d["error"] += 1
    for d in t.values():
        d["rate"] = d["calls"] / d["n"] if d["n"] else None
    return t


def _by_level(cells):
    """(scores, calls, n) per envelope log-SD level, correlation times pooled; None without the full grid."""
    lv = {s: [0, 0] for s in LOG_SD}
    for name, d in cells.items():
        c = CELLS[name]
        if c["family"] == "env":
            lv[c["log_sd"]][0] += d["calls"]
            lv[c["log_sd"]][1] += d["n"]
    if any(v[1] == 0 for v in lv.values()):
        return None
    return list(LOG_SD), [lv[s][0] for s in LOG_SD], [lv[s][1] for s in LOG_SD]


def summarise(rows, vp, imm):
    pipes = OrderedDict()
    for r in rows:
        ent = pipes.setdefault(r["pipe"], {}).setdefault(r["variant"], {"kind": r["kind"], "roles": r["roles"],
                                                                           "rows": []})
        ent["rows"].append(r)
    out = {"pipelines": {}, "bars": {}, "nogo": {}}
    for p, vs in pipes.items():
        v_imm = imm.get((p, vp.get(p), "published"), {})
        out["pipelines"][p] = {"validity_primary": vp.get(p), "immunity_type": v_imm.get("immunity_type"),
                               "x_ref_class": v_imm.get("x_ref_class"),
                               "variants": {n: {"kind": e["kind"], "roles": e["roles"],
                                                "cells": _cells_table(e["rows"])} for n, e in vs.items()}}
    impl, trend, not_tested, arms, nogo = {}, {}, {}, {}, {}
    for p, d in out["pipelines"].items():
        vpn = d["validity_primary"]
        cells = d["variants"].get(vpn, {}).get("cells", {})
        w = cells.get("white")
        if w and w["n"]:
            lo, hi = binom_range(w["n"])
            impl[p] = {"n": w["n"], "calls": w["calls"], "range": [lo, hi], "in_range": lo <= w["calls"] <= hi,
                       "known_exception": p in KNOWN_WHITE_EXCEPTIONS}
        exposed = d["immunity_type"] == "exposed"
        lv = _by_level(cells)
        if vpn in d["variants"]:
            if not exposed:
                not_tested[p] = ("immune by design" if d["immunity_type"] == "immune" else
                                 "no label contrast (116 design typing)")
            elif lv is None:
                not_tested[p] = "no envelope grid"
            else:
                z, pv = cochran_armitage(*lv)
                trend[p] = {"z": z, "p": pv, "calls": lv[1], "n": lv[2]}
            if exposed:
                k = [(dd["calls"], dd["n"]) for nm, dd in cells.items()
                     if CELLS[nm]["family"] == "env" and CELLS[nm]["log_sd"] in NOGO_SD]
                c_, n_ = sum(a for a, _ in k), sum(b for _, b in k)
                nogo[p] = {"calls": c_, "n": n_, "rate": c_ / n_ if n_ else None,
                           "exceeds": bool(n_ and c_ / n_ > NOGO_RATE)}
        for n, e in d["variants"].items():
            if e["kind"] == "randomized":
                lr = _by_level(e["cells"])
                if lr is not None:
                    z, pv = cochran_armitage(*lr)
                    arms[f"{p}|{n}"] = {"z": z, "p": pv, "calls": lr[1], "n": lr[2]}
    for tab in (trend, arms):
        for k, a in zip(tab, holm([t["p"] for t in tab.values()])):
            tab[k]["holm_p"] = a
            tab[k]["rise"] = bool(a < ALPHA)
    out["bars"]["implementation"] = {"per_pipeline": impl, "known_exceptions": list(KNOWN_WHITE_EXCEPTIONS),
                                     "pass": all(v["in_range"] or v["known_exception"] for v in impl.values())}
    out["bars"]["mechanism"] = {"per_pipeline": trend, "not_tested": not_tested,
                                "n_rise": sum(t["rise"] for t in trend.values())}
    out["bars"]["construction"] = {"per_arm": arms, "pass": not any(t["rise"] for t in arms.values())}
    out["nogo"] = {"per_pipeline": nogo, "threshold": NOGO_RATE, "log_sd": list(NOGO_SD),
                   "nogo": not any(v["exceeds"] for v in nogo.values()),
                   "rule": "if no non-immune validity primary exceeds 10% at log-SD 0.25-0.5, the non-stationarity "
                           "mechanism is dropped"}
    return out


def merge(root=None, out=None, vp=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(root or SH, "rows*.jsonl"))):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    rows = list({r["key"]: r for r in rows}.values())
    s = summarise(rows, vp if vp is not None else validity_primaries(), immunity())
    s.update({"part": "C1", "added_later": True, "locked_code_hash": None,
              "files": {os.path.basename(f): fr.sha256(f) for f in FILES}, "seed": SEED, "n_rows": len(rows),
              "grid": {"cells": list(CELLS), "n_per_cell": N_CELL, "small_pipes": list(SMALL_PIPES),
                       "small_cells": list(SMALL_CELLS), "n_small": N_SMALL}})
    out = out or OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        json.dump(s, fh, indent=1, default=float)
    os.replace(out + ".tmp", out)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--pipes", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        run_shard(a.shard, a.nshards, a.pipes.split(",") if a.pipes else None, a.limit)


if __name__ == "__main__":
    main()
