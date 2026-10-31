"""Runner of the common null benchmark. One shard of one null set x pipeline x montage x responder setting per call; rows go to
a resumable jsonl that is appended to, never overwritten.
python 114_benchmark.py --null NAME --pipe NAME --split dev|analysis --shard i --nshards n
                        [--resp none|grid|eval] [--montage native|kit8] [--limit N] [--max-overlays N]

Guards:
- split "analysis" is refused while either frozen dev file (engine dev_split.json, passive_dev_split.json) is missing; "all" is
  always refused (it would put the development recordings into analysis outputs). The check runs
  before any recording is listed.
- Every recording is checked against the split: meta["dev"] must be True in "dev" and must not be True otherwise.
- While the BDSP dev split is provisional (engine warns), the `pending` lowest-ranked dev recordings are skipped: each
  recording still to become usable can push one of them into analysis.
- Recordings touched before the dev freeze (key "excluded" of the
  git-ignored output/bdsp/meta/touched_before_freeze.json) are never loaded in the analysis split; the analysis split is
  refused when that file is missing.
- Dev rows go to output/intermediate/_bench_shards_dev, analysis rows to output/intermediate/_bench_shards; every row
  carries its split, and a shard file holding rows of another split is refused.
- Rows hold whitelisted scalars only: no header field (subject_info, meas_date, ...) and no signal.

Null sets (FAMILY of the pipeline must match):
- ncmd_bdsp | ncmd_pdoc: "cmd" pipelines, their own schedule(). I-CARE is dropped (second stage): its 8-min windows fit no
  command pipeline;
- nstim0_bdsp | nstim0_pdoc: "erp" pipelines, their own schedule() on EEG with no stimuli;
- npassive_<dataset>: "erp" pipelines on real events (bench.passive + bench.adapters, imported lazily), sharded by
  subject: the published calls per matching contrast (adapters.contrasts_for; mbvt on kojimaB has two) through the
  adapter wrappers = the "sensory read as volition" family (no randomized arm: a label permutation on physically
  different stimuli is not a volition null); the proposed method = the instruction-randomization test, PRIMARY per
  subject (the subject's runs pooled), SECONDARY per run. N/A, skipped variants and runs left out of a pool are rows;
- chancount_<dataset>: chennu13 only, on real events, channels subsampled by nested_channel_order to 19 / 32 / 64 /
  native.

Overlays (engine sources):
- The clean window an overlay needs = LEAD_S + the longest run_s of the variants run on that source + TAIL_S (every
  module's extent() lies inside it; checked per overlay). start_s = window start + LEAD_S.
- Starts: rng = default_rng(SEED + crc32(pipe + uid)) permutes engine.clean_spans(rec, window); the first is the seeded
  choice (j = 0); later ones are accepted greedily when they do not overlap any accepted window.
- Published variants run on the "sequence" overlays: 1 per recording (fixed-schedule), 8 for Columbia (BH within
  patient, k = 1..8), 12 for du Bois (12 sessions). mindBEAGLE-type (mbvt, mbmi): the non-overlapping tiling of the
  clean stretches (engine.patient_windows) in seeded random order, 10 per BDSP recording, all of a pDoC night; each
  trial-count group is tiled on its own length (tiling_tiers: mbmi's 60-trial variants on short tiles, or17_120 on its
  own; rows say tile), non-overlapping within the group.
- Timing groups (timing_groups): the variants whose published schedules have the same trial count, onsets and
  resp_dur_s and the same number of responder-carrying onsets; the representative is the primary variant, else its
  short twin, else the group's variant closest to the primary's settings (then as coded, then tier order). The randomization-based method (run_randomized,
  schedule_randomized of the representative at the same start) runs ONCE PER GROUP, on that group's timing, on every
  sequence overlay and on extra overlays up to m = min(10, ceil(200 / recordings in the split)); when the recording
  cannot hold m non-overlapping windows the rest are distinct overlapping starts (overlap = True, never used for
  composites). Published rows carry their group and its randomized run (rand_variant = the representative);
  randomized-only overlays give one row per group.
- Primary variant: the module's DEFAULT (else run()'s default), except Chennu on BDSP: its 12-block twin (native 20 on the pDoC nights).
- Shortened BDSP variants (second stage; bench.wrappers): cruse11 3 + 3 blocks, cruse12p 4 x 36 trials. Where the native
  length fits, native and short twins run on one overlay (paired); otherwise the short twins alone (the short twins
  form their own timing groups).
- Columbia on BDSP analysis reuses 113 (second stage): its null published (test i) and randomized (test iii) rows come from
  output/intermediate/columbia_null_rows.parquet (115 reads them), design-sample and extra recordings alike
  (columbia113: part 4's source / stem). Here the null run adds only a second seeded clean
  window that does not overlap 113's overlay (the same-session k = 2 bound; published variant only), and the
  responders use 113's start. Recordings 113 did not run give status "not_in_113".
- Per-overlay seeds: SeedSequence([SEED, crc32(pipe + uid), j, stream]); responders add crc32 of (model, effect, prob).
  Injection streams are numbered over the schedule groups of ALL the tier's variants (tier order), so a resumed shard
  draws the same responders. Every injected copy is a new engine.Recording with its own token and a copy of meta; the
  feature caches key on the token, and are cleared after each randomized run and whenever the injected copy changes.

Responders: engine.inject on the first overlay (j = 0, the null run's seeded overlay; the randomized-only overlays
have no comparator to pair with). Size correction is statistic-based (second stage), so the comparator needs one fit per
overlay: bench.wrappers.stat_only (no permutation inference where the module allows it).
- --resp grid (evaluation-point locator): the LOCATOR_N (60) seeded locator patients only (subsamples: a seeded
  permutation of the source's patients), every variant at every grid point, statistic only, no randomized method.
- --resp eval: every patient, per model x probability and timing group, the group's comparator at the group's
  evaluation effect (frozen by 115 --freeze-eval-points in EVAL_FILE; statistic only) and the group's randomized
  method (permutation inference) on the same timing; rows say group, pipeline_group and reached; on the
  next UNCORRECTED_N (100) seeded patients also run() in full (perm_*: the uncorrected arm and the p-based correction).
  Columbia's ERS points run on that subsample only (SUBSAMPLE_ONLY; rows say eval_set), and so does every du Bois
  timing group but the pipeline group (SUBSAMPLE_GROUPS); every other point, group and pipeline on all patients. Columbia's uncorrected arm runs at ERD p_resp = 1 only (UNCORRECTED_POINTS; its other
  points have perm_* None).
- pDoC (ncmd_pdoc, nstim0_pdoc) is the reduced-montage arm, nulls only (second stage): no responders, no kit8; each module
  reads its documented subset of F3 F4 C3 C4 O1 O2 (bench.wrappers.PDOC_CH; rows say channel_set); Columbia is N/A
  (refused); variants reading an absent site are not run (gibson16 SSEP: Pz). The mindBEAGLE-type tilings of each
  night and the composite sequences run as on BDSP.
- The randomization-based method runs at B = 200 (bench.wrappers.run_randomized / RAND_B; the N-passive instruction
  test too), exact at any B with decision-exact early stopping.
- Published-variant rows carry min_attainable_p and cannot_call (bench.wrappers): variants that cannot call by
  construction are run and reported; 115 never selects them.
Motor ("cmd"): erd and ers; resp_hand comes from the
overlay meta. ERP: erp at the module's target-like labels (ERP_RESP_LABELS). Response probability 1 and 0.5 (Bernoulli
per onset). The effect grid is read from GRID_FILE (second stage); on the dev split DEV_GRID is used while that file does
not exist and rows say grid = "dev-provisional". Sokoliuk 2021 has no responder model here (speech tracking is not an
evoked response at the trial onset): its grid rows say status "no_responder_model".

A module that cannot analyse a recording (it raises bench.motor_common.NoUsableTrials: no trial survived its artefact
rule) gives status "na: no_usable_trials" (the exception in na_detail; rand_error says the same for the randomized
arm); every other exception stays "error: ...": 115 leaves such rows out of every false-call
and sensitivity denominator and counts them in coverage by stratum.
size_stat = stat_for_size_correction with +-inf capped at +-1e12 (bench.wrappers.size_stat: du Bois's zero-variance
t); p_censored (also rand_ / perm_) marks an early-stopped p, a lower bound at its pile (bench.wrappers.p_censored).
One row per (overlay, variant, model, effect, prob[, channel count]): mode (full | stat_only), inference, in_locator,
in_uncorrected, short, twin_of, covariates (Morgoth stratum, transitioning,
artifact index and label over the overlay's extent, montage, extent, onsets, usable trials), stat / p / call /
size_stat of run(), rand_* of run_randomized(), seconds. Module calls that raise give status "error: ..." rows (counted
by the merge), so one bad recording does not stop a shard."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import importlib
import inspect
import itertools
import json
import math
import sys
import time
import warnings
import weakref
import zlib

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from bench import engine as en
from bench import wrappers as wr

SPEC_PATH = None
SH = {"dev": os.path.join(ROOT, "output", "intermediate", "_bench_shards_dev"),
      "analysis": os.path.join(ROOT, "output", "intermediate", "_bench_shards")}
GRID_FILE = os.path.join(ROOT, "docs", "benchmark-effect-grid.json")
TOUCHED_FILE = os.path.join(ROOT, "output", "bdsp", "meta", "touched_before_freeze.json")
FROZEN_FILES = (en.DEV_FILE, os.path.join(en.META, "passive_dev_split.json"))
EVAL_FILE = {"dev": os.path.join(ROOT, "output", "intermediate", "_bench_eval_points_dev.json"),
             "analysis": os.path.join(ROOT, "output", "intermediate", "_bench_eval_points.json")}
COLUMBIA_ROWS = os.path.join(ROOT, "output", "intermediate", "columbia_null_rows.parquet")
SAMPLE_CSV = os.path.join(en.META, "sample_500.csv")
EXTRA_CSV = os.path.join(en.META, "bench_extra_500.csv")
COLUMBIA_SPAN_S = 1505
LOCATOR_N, UNCORRECTED_N = 60, 100
SUBSAMPLE_ONLY = {("columbia", "ers")}
UNCORRECTED_POINTS = {"columbia": {("erd", 1.0)}}
SUBSAMPLE_GROUPS = {"dubois"}
NA_NO_USABLE = "na: no_usable_trials"
KIT8_NA = {"columbia": "N/A on Kit-8: 112's features and CSD are defined on the 19-channel montage (USE_CH)",
           "dubois": "N/A on Kit-8: its channel sets (CHANSETS full / motor) are fixed 10-20 lists, not NATIVE_CH"}
SEED = 20261025
LEAD_S, TAIL_S = 10, 10
RAND_TARGET, M_MAX = 200, 10
SEQ_K = {"columbia": 8, "dubois": 12}
MB_PIPES, MB_K_BDSP = ("mbvt", "mbmi"), 10
PASSIVE = ("erpcore_mmn", "ds003061", "kojimaB")
ADAPT_USABLE = {"mbvt": lambda u: u.get("n_groups"),
                "chennu13": lambda u: u.get("n_explicit", 0) + u.get("n_implicit", 0),
                "gibson16": lambda u: u.get("n_target", 0) + u.get("n_nontarget", 0) + u.get("n_back", 0)}
CHANCOUNT = (19, 32, 64)
SOURCE_VARIANTS = {("chennu13", "bdsp"): {"n_blocks": 12}, ("chennu13", "pdoc"): {"n_blocks": 20}}
PRIMARY_OVERRIDE = {("chennu13", "bdsp"): "p1_samples10_none_b12"}
ERP_RESP_LABELS = {"mbvt": "TARGET", "chennu13": "EXPL", "gibson16": "TARGET"}
MODELS = {"cmd": ("erd", "ers"), "erp": ("erp",)}
PROBS = (1.0, 0.5)
DEV_GRID = {"erd": [0.2, 0.4, 0.6], "ers": [0.2, 0.4, 0.6], "erp": [1.0, 2.5, 5.0]}
USABLE_KEYS = ("n_kept", "n_trials", "n_groups", "n_cols", "n_explicit", "n_a")
NULL_ROW = ("none", 0.0, 0.0)



def locked_committed(spec_path=None):
    """Not applied here: the check reads files that are not distributed with this code."""
    return True


def guard_split(split, spec_path=None):
    if split == "dev":
        return
    if split == "all":
        raise RuntimeError("split 'all' mixes the development recordings into analysis outputs: never run")
    if split != "analysis":
        raise ValueError(f"split must be dev or analysis, not {split!r}")
    if not locked_committed(spec_path):
        raise RuntimeError("analysis split refused")
    missing = [f for f in FROZEN_FILES if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"analysis split refused: the frozen dev split file(s) {missing} are missing (the loaders "
                           "would fall back to a fresh scan or draw)")


def touched_uids():
    """Recordings on which an outcome was computed while they were provisional dev: key "excluded" of the git-ignored TOUCHED_FILE. Skipped in the analysis split, which is refused
    without the file."""
    if not os.path.exists(TOUCHED_FILE):
        raise RuntimeError(f"{TOUCHED_FILE} is missing: the analysis split needs the list of recordings touched before "
                           "the dev freeze")
    with open(TOUCHED_FILE) as fh:
        return set(json.load(fh)["excluded"])


def check_member(rec, split):
    dev = rec.meta.get("dev") is True
    if split == "dev" and not dev:
        raise RuntimeError(f"{rec.uid}: not a development recording, refused in the dev split")
    if split != "dev" and dev:
        raise RuntimeError(f"{rec.uid}: development recording refused in the {split} split")



def parse_null(null):
    kind, _, src = null.partition("_")
    if src == "icare":
        raise ValueError("I-CARE is dropped from the command pipelines (second stage): its 8-min windows fit none "
                         "(shortest need: edlow17, 606 s)")
    if kind in ("ncmd", "nstim0") and src in en.SOURCES:
        return {"name": null, "kind": kind, "family": "cmd" if kind == "ncmd" else "erp", "source": src}
    if kind in ("npassive", "chancount") and src in PASSIVE:
        return {"name": null, "kind": kind, "family": "erp", "source": src}
    raise ValueError(f"unknown null set {null!r}")


def load_pipe(name):
    return importlib.import_module(f"bench.pipe_{name}")


def variants_for(mod, source):
    rule = SOURCE_VARIANTS.get((mod.NAME, source), {})
    return [v for v, c in mod.VARIANTS.items() if all(c.get(k) == val for k, val in rule.items())]


def primary_variant(mod, source):
    if (mod.NAME, source) in PRIMARY_OVERRIDE:
        return PRIMARY_OVERRIDE[(mod.NAME, source)]
    if hasattr(mod, "DEFAULT"):
        return mod.DEFAULT
    return inspect.signature(mod.run).parameters["variant"].default


def variant_specs(mod, source):
    """{name: spec} of the variants run on the source: the native ones (SOURCE_VARIANTS filter; on pDoC without the
    variants that read an absent site, bench.wrappers.pdoc_variant_na) and the shortened BDSP twins
    (bench.wrappers.short_variants). spec: arg (what the module is called with), globals (module overrides for the
    call), twin_of, short."""
    out = {v: {"arg": v, "globals": {}, "twin_of": None, "short": False} for v in variants_for(mod, source)
           if not (source == "pdoc" and wr.pdoc_variant_na(mod, v))}
    out.update(wr.short_variants(mod, source))
    return out


def _spec(mod, name, source=None):
    if name in mod.VARIANTS:
        return {"arg": name, "globals": {}, "twin_of": None, "short": False}
    for src in ([source] if source else wr.SHORT_SOURCES):
        sv = wr.short_variants(mod, src)
        if name in sv:
            return sv[name]
    raise KeyError(f"{mod.NAME}: no variant {name!r}")


def need_seconds(mod, variants, source=None):
    """Clean seconds one overlay needs: LEAD_S + the longest run_s of the variants + TAIL_S."""
    run = max(wr.run_len(mod, _spec(mod, v, source)) for v in variants)
    return int(math.ceil(LEAD_S + run + TAIL_S))


def m_overlays(n_recs):
    return min(M_MAX, int(math.ceil(RAND_TARGET / max(n_recs, 1))))


def seq_k(name, source):
    if name in MB_PIPES:
        return MB_K_BDSP if source == "bdsp" else None
    return SEQ_K.get(name, 1)


def rec_seed(pipe, uid):
    return SEED + zlib.crc32((pipe + uid).encode())


def _rng(pipe, uid, *parts):
    return np.random.default_rng([SEED, zlib.crc32((pipe + uid).encode())] + [int(p) for p in parts])


def _crc(s):
    return zlib.crc32(str(s).encode())


def subsamples(patients, source):
    """(locator, uncorrected): a seeded permutation of the source's sorted patients; the first LOCATOR_N locate the
    evaluation point on the effect grid, the next UNCORRECTED_N carry the published permutation inference at it."""
    pats = sorted(set(patients))
    order = np.random.default_rng([SEED, _crc(f"locator|{source}")]).permutation(len(pats))
    return ({pats[i] for i in order[:LOCATOR_N]}, {pats[i] for i in order[LOCATOR_N:LOCATOR_N + UNCORRECTED_N]})


def columbia113(path=None):
    """113's merged rows (columbia_null_rows.parquet, after 113 --merge) with uid, patient and source. source
    "design" (or no source column): sample_idx indexes sample_500.csv; "extra" (part 4): sample_idx indexes
    bench_extra_500.csv. A row's stem, where present, must be the file at its index (refused otherwise)."""
    import pandas as pd
    path = path or COLUMBIA_ROWS
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is missing: Columbia on BDSP reuses 113's rows (run 113 --merge first)")
    df = pd.read_parquet(path)
    src = df["source"].fillna("design").astype(str) if "source" in df else pd.Series("design", index=df.index)
    bad = sorted(set(src) - {"design", "extra"})
    if bad:
        raise RuntimeError(f"113 rows of unknown source {bad}")
    uid, pat = pd.Series(None, index=df.index, dtype=object), pd.Series(None, index=df.index, dtype=object)
    for name, fn in (("design", SAMPLE_CSV), ("extra", EXTRA_CSV)):
        m = (src == name).to_numpy()
        if not m.any():
            continue
        s = pd.read_csv(fn)
        i = df.loc[m, "sample_idx"].to_numpy(int)
        stems = s["file"].str.replace(".mat", "", regex=False).to_numpy()[i]
        if "stem" in df:
            given = df.loc[m, "stem"]
            clash = given.notna().to_numpy() & (given.astype(str).to_numpy() != stems)
            if clash.any():
                raise RuntimeError(f"113 {name} rows whose stem is not the file at their sample_idx: "
                                   f"{list(given[clash])[:5]}")
        uid[m], pat[m] = stems, s["patient_id"].astype(str).to_numpy()[i]
    df["uid"], df["patient"], df["source"] = uid.to_numpy(), pat.to_numpy(), src.to_numpy()
    return df


def _second_window(rec, mod, need, start113):
    """A seeded clean window that does not overlap 113's overlay of the recording (the same-session k = 2 bound)."""
    lo, hi = start113 - 1, start113 + COLUMBIA_SPAN_S
    spans = np.asarray(en.clean_spans(rec, need))
    ok = spans[(spans >= hi) | (spans + need <= lo)]
    if not len(ok):
        return []
    s = int(np.random.default_rng(rec_seed(mod.NAME, rec.uid)).permutation(ok)[0])
    return [{"j": 1, "start_s": float(s + LEAD_S), "window": (s, s + need), "seq": True, "overlap": False,
             "rand": False}]


def plan_overlays(rec, mod, source, n_recs, variants, max_overlays=None):
    """[{j, start_s, window, seq, overlap, rand}] (module docstring)."""
    need = need_seconds(mod, variants, source)
    spans = np.asarray(en.clean_spans(rec, need))
    if len(spans) == 0:
        return []
    rng = np.random.default_rng(rec_seed(mod.NAME, rec.uid))
    k, m = seq_k(mod.NAME, source), m_overlays(n_recs)
    if mod.NAME in MB_PIPES:
        tiles = [s for _, s in en.patient_windows([rec], need)]
        acc = [tiles[i] for i in rng.permutation(len(tiles))]
        seq = acc if k is None else acc[:k]
        order = rng.permutation(spans)
    else:
        order = rng.permutation(spans)
        want = max(k, m)
        acc = []
        for s in order:
            if all(abs(int(s) - a) >= need for a in acc):
                acc.append(int(s))
                if len(acc) == want:
                    break
        seq = acc[:k]
    out = [{"start": int(s), "seq": True, "overlap": False} for s in seq]
    out += [{"start": int(s), "seq": False, "overlap": False} for s in acc[len(seq):m]]
    if len(out) < m:
        used = {o["start"] for o in out}
        for s in order:
            if len(out) >= m:
                break
            if int(s) not in used:
                used.add(int(s))
                out.append({"start": int(s), "seq": False, "overlap": True})
    out = out[:max_overlays] if max_overlays else out
    return [{"j": j, "start_s": float(o["start"] + LEAD_S), "window": (o["start"], o["start"] + need),
             "seq": o["seq"], "overlap": o["overlap"], "rand": True} for j, o in enumerate(out)]



def nested_channel_order(names):
    """Fixed nested order for the channel-count arm: the 10-20 sites present (engine.NAMES19 order, T3/T4/T5/T6 =
    T7/T8/P7/P8), then the remaining channels by farthest-point sampling over the MNE standard_1005 positions (each
    next channel maximises its smallest distance to those already chosen; ties by name), then channels without a
    standard position, by name."""
    mt = importlib.import_module("montage")
    pos = mt._positions()
    canon = {n: mt.canonical(n) for n in names}
    order = []
    for c in en.NAMES19:
        for n in sorted(names):
            if n not in order and canon[n].lower() == mt.canonical(c).lower():
                order.append(n)
    rest = sorted(n for n in names if n not in order and canon[n].lower() in pos)
    while rest:
        chosen = np.array([pos[canon[n].lower()] for n in order if canon[n].lower() in pos]).reshape(-1, 3)
        d = [np.min(np.linalg.norm(chosen - pos[canon[n].lower()], axis=1)) if len(chosen) else 0.0 for n in rest]
        best = max(range(len(rest)), key=lambda i: (round(float(d[i]), 9), -i))
        order.append(rest.pop(best))
    return order + sorted(n for n in names if n not in order)


def chancount_sizes(n):
    return [k for k in CHANCOUNT if k < n] + [n]


def subset(rec, names):
    """rec restricted to the named channels; the artifact-index cache is per channel set, so it is not carried over."""
    idx = [rec.names.index(n) for n in names]
    return en.Recording(source=rec.source, uid=rec.uid, patient=rec.patient, x=np.asarray(rec.x)[idx], fs=rec.fs,
                        names=list(names), meta={k: v for k, v in rec.meta.items() if k != "_artifact_s"})


def kit8(rec):
    mt = importlib.import_module("montage")
    return subset(rec, [rec.names[i] for i in mt.kit8_indices(rec.names)])


@contextlib.contextmanager
def channels(mod, names):
    """The module reads the given channels (NATIVE_CH and its 10-20 substitute set) for the duration."""
    if names is None:
        yield
        return
    saved = {a: getattr(mod, a) for a in ("NATIVE_CH", "SUB_1020") if hasattr(mod, a)}
    try:
        for a in saved:
            setattr(mod, a, list(names))
        yield
    finally:
        for a, v in saved.items():
            setattr(mod, a, v)



def _scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, str):
        return v
    return None


def scalars(extra, prefix="x_"):
    """Scalar entries of a module's extra dict (arrays, lists and dicts dropped)."""
    out = {}
    for k, v in (extra or {}).items():
        if v is None or isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating)):
            out[prefix + str(k)] = _scalar(v)
    return out


def _usable(res, ov):
    ex = (res or {}).get("extra", {})
    for k in USABLE_KEYS:
        if k in ex and _scalar(ex[k]) is not None:
            return int(ex[k])
    return None


def _call(fn, *a):
    t = time.perf_counter()
    try:
        return fn(*a), None, time.perf_counter() - t
    except Exception as e:
        return None, f"error: {type(e).__name__}: {str(e)[:200]}", time.perf_counter() - t


def _no_usable(e):
    nu = getattr(sys.modules.get("bench.motor_common"), "NoUsableTrials", None)
    return isinstance(nu, type) and isinstance(e, nu)


def _call_run(mod, fn, *a):
    """_call for a module's analysis (run, the statistic, the randomized method): bench.motor_common.NoUsableTrials
    (raised by a module when no trial survives its artefact rule) gives "na: no_usable_trials [Type: message]", and
    such rows leave every denominator (115); every other exception stays "error: ..." (a bug is never hidden as N/A)."""
    t = time.perf_counter()
    try:
        return fn(*a), None, time.perf_counter() - t
    except Exception as e:
        tag = NA_NO_USABLE if _no_usable(e) else "error"
        return None, f"{tag}: {type(e).__name__}: {str(e)[:200]}" if tag == "error" else \
            f"{NA_NO_USABLE} [{type(e).__name__}: {str(e)[:200]}]", time.perf_counter() - t


def _status(err):
    """Row status of a module call's error: ok, na: no_usable_trials (the detail goes to na_detail), or the error."""
    if err is None:
        return "ok"
    return NA_NO_USABLE if err.startswith(NA_NO_USABLE) else err


def _covariates(rec, ov, mod, cache, prefix=""):
    t0, t1 = (float(v) for v in mod.extent(ov))
    if (t0, t1) not in cache:
        st, tr = en.stratum(rec, t0, t1 - t0)
        share, lab = en.artifact_index(rec, t0, t1 - t0)
        cache[(t0, t1)] = {"stratum": st, "transitioning": tr, "artifact_share": _scalar(share),
                           "artifact_label": lab, "extent_t0": t0, "extent_t1": t1}
    return {prefix + k: v for k, v in cache[(t0, t1)].items()}


def _result_fields(mod, res, err, sec, prefix=""):
    if res is None:
        out = {prefix + "stat": None, prefix + "p": None, prefix + "call": None, prefix + "size_stat": None,
               prefix + "p_censored": None, prefix + "seconds": sec, prefix + "error": err}
        if err and err.startswith(NA_NO_USABLE):
            out[prefix + "error"], out[prefix + "na_detail"] = NA_NO_USABLE, err[len(NA_NO_USABLE):].strip(" []")
        return out
    size, e2, _ = _call(wr.size_stat, mod, res)
    return {prefix + "stat": _scalar(res["stat"]), prefix + "p": _scalar(res["p"]), prefix + "call": bool(res["call"]),
            prefix + "size_stat": _scalar(size), prefix + "p_censored": wr.p_censored(res), prefix + "seconds": sec,
            prefix + "error": e2}


def _key(uid, j, variant, model, effect, prob, montage):
    return "|".join([uid, str(j), variant or "-", model, f"{effect:g}", f"{prob:g}", montage])


def load_grid(split, spec_path=None):
    """(effect grid by model, tag): the grid file, else DEV_GRID on the dev split only."""
    if os.path.exists(GRID_FILE):
        with open(GRID_FILE) as fh:
            return json.load(fh), "locked" if locked_committed(spec_path) else "dev-provisional"
    if split != "dev":
        raise RuntimeError(f"no second-stage effect grid ({GRID_FILE}): responders are run on the dev split only")
    return DEV_GRID, "dev-provisional"


def eval_points(null, pipe, split):
    """[((model, effect, prob), comparator, timing group, row fields)]: per model x probability and timing group, the
    group's comparator at its evaluation effect, frozen by 115 --freeze-eval-points (EVAL_FILE[split])."""
    fn = EVAL_FILE[split]
    if not os.path.exists(fn):
        raise RuntimeError(f"{fn} is missing: run --resp grid, then 115 --split {split} --freeze-eval-points on a "
                           "complete locator")
    with open(fn) as fh:
        d = json.load(fh).get(null, {}).get(pipe, {})
    out = []
    for k, e in sorted(d.items()):
        model, prob = k.split("|")
        for g, x in sorted(e.get("groups", {}).items()):
            if x.get("effect") is not None and x.get("comparator") is not None:
                out.append(((model, float(x["effect"]), float(prob)), x["comparator"], g,
                            {"pipeline_group": g == e.get("pipeline_group"), "reached": bool(x.get("reached", True))}))
    return out


def grid_points(family, resp, split, null=None, pipe=None, spec_path=None):
    """[(grid point, variants or None for all, timing group or None, row fields)], tag. none: the null; grid: the
    effect grid (locator); eval: per model x probability and timing group, the group's comparator at its effect."""
    if resp == "none":
        return [(NULL_ROW, None, None, {})], None
    if resp == "eval":
        return [(gp, [v], g, ex) for gp, v, g, ex in eval_points(null, pipe, split)], "eval"
    grid, tag = load_grid(split, spec_path)
    return [((m, float(e), p), None, None, {}) for m in MODELS[family] for e in grid.get(m, []) for p in PROBS], tag


def uncorrected_point(name, model, prob):
    """True if the uncorrected arm (run() in full on the subsample) runs at this evaluation point."""
    pts = UNCORRECTED_POINTS.get(name)
    return pts is None or (model, float(prob)) in pts


def resp_mask(mod, ov, prob, rng):
    lab = np.asarray(ov.labels)
    if mod.FAMILY == "erp":
        want = getattr(mod, "RESP_LABELS", None)
        if want is None:
            want = (getattr(mod, ERP_RESP_LABELS[mod.NAME]),)
        m = np.isin(lab, want)
    else:
        m = np.ones(len(lab), bool)
    return m & (rng.random(len(lab)) < prob) if prob < 1 else m


def has_responder(mod):
    return mod.FAMILY == "cmd" or hasattr(mod, "RESP_LABELS") or mod.NAME in ERP_RESP_LABELS


def _injected(rec, mod, ov, gp, uid, j, stream):
    """Copy of rec with the responder of grid point gp on overlay ov; seeded per (uid, effect, model, prob, overlay)."""
    model, effect, prob = gp
    if model == "none":
        return rec
    ss = np.random.SeedSequence([SEED, _crc(uid), _crc(f"{model}|{effect:g}|{prob:g}"), int(j), int(stream)])
    g_mask, g_inj = (np.random.default_rng(s) for s in ss.spawn(2))
    x = en.inject(rec, ov, resp_mask(mod, ov, prob, g_mask), model, effect, g_inj)
    return en.Recording(source=rec.source, uid=rec.uid, patient=rec.patient, x=x, fs=rec.fs, names=list(rec.names),
                        meta=dict(rec.meta))


def _base_row(rec, mod, spec, ctx, extra):
    return {"null": spec["name"], "pipe": mod.NAME, "family": mod.FAMILY, "grade": getattr(mod, "GRADE", None),
            "split": ctx["split"], "dev": rec.meta.get("dev") is True,
            "split_final": _scalar(rec.meta.get("split_final")) if "split_final" in rec.meta else None,
            "source": rec.source, "uid": rec.uid, "patient": rec.patient, "resp": ctx["resp"], **extra}


def _clear_cache():
    mc = sys.modules.get("bench.motor_common")
    if mc is not None:
        mc._CACHE.clear()


def _plan(rec, mod, spec, ctx, specs):
    """(plan, variants of the tier that fits, need): native + short twins at the native need where it fits, else the
    short twins alone; Columbia on BDSP analysis takes 113's overlay (responders) or a window beside it (null)."""
    names = list(specs)
    short = [v for v in names if specs[v]["short"]]
    tiers = [names] + ([short] if short and len(short) < len(names) else [])
    col = ctx.get("col113")
    if col is not None:
        need = need_seconds(mod, names, spec["source"])
        if rec.uid not in col:
            return [], names, need, "not_in_113"
        if ctx["resp"] == "none":
            p = _second_window(rec, mod, need, col[rec.uid])
            return p, names, need, None if p else "no_second_window"
        st = float(col[rec.uid])
        return [{"j": 0, "start_s": st, "window": (st - 1, st + COLUMBIA_SPAN_S), "seq": True, "overlap": False,
                 "rand": True}], names, need, None
    for t in tiers:
        p = plan_overlays(rec, mod, spec["source"], ctx["n_recs"], t, ctx.get("max_overlays"))
        if p:
            return p, t, need_seconds(mod, t, spec["source"]), None
    return [], names, need_seconds(mod, tiers[-1], spec["source"]), "no_fit"


GROUP_START_S = 100.0
_GROUPS = weakref.WeakKeyDictionary()


def _representative(vs, specs, prim, mod):
    """The variant whose schedule_randomized and features a timing group's randomized method uses: the primary, else
    its short twin, else the group's variant whose settings agree with the primary's on the most keys (ties: as coded,
    then tier order), so groups differ from the primary in timing only where they can."""
    if prim in vs:
        return prim
    tw = [v for v in vs if prim in (specs[v]["twin_of"] or [])]
    if tw:
        return tw[0]
    pc = {k: x for k, x in wr._cfg(mod, prim).items() if k not in ("cite", "as_coded")} if prim in mod.VARIANTS else {}

    def score(v):
        c = wr._cfg(mod, specs[v]["arg"])
        return (-sum(c.get(k) == x for k, x in pc.items()),
                not (mod.VARIANTS.get(v, {}).get("as_coded") and not specs[v]["short"]), vs.index(v))
    return min(vs, key=score)


def _memo_groups(mod, key, fn):
    cache = _GROUPS.setdefault(mod, {})
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def timing_groups(mod, specs, tier, prim):
    """{representative: [variants]} over the tier: variants of the same PARADIGM (the variant's 'paradigm', e.g. mbvt
    VT2 / VT3: different designs and re-draws) whose published schedules (drawn at GROUP_START_S with one fixed seed)
    have the same onsets (trial count and timing), the same resp_dur_s and the same number of responder-carrying
    onsets (the response model). Schedules that raise form their own group per variant."""
    return _memo_groups(mod, ("timing", tuple(tier), prim), lambda: _timing_groups(mod, specs, tier, prim))


def _timing_groups(mod, specs, tier, prim):
    sigs = {}
    for v in tier:
        try:
            ov = _sched(mod, mod.schedule, specs[v], GROUP_START_S, np.random.default_rng([SEED, _crc(mod.NAME)]))
            sig = (np.round(np.asarray(ov.onsets_s, float), 6).tobytes(), _scalar(ov.meta.get("resp_dur_s")),
                   int(resp_mask(mod, ov, 1.0, None).sum()) if has_responder(mod) else 0,
                   wr._cfg(mod, specs[v]["arg"]).get("paradigm"))
        except Exception:
            sig = ("error", v)
        sigs.setdefault(sig, []).append(v)
    return {_representative(vs, specs, prim, mod): vs for vs in sigs.values()}


def tiling_tiers(mod, specs):
    """[(tile label or None, variants)]: mindBEAGLE-type pipelines tile each trial-count group on its own length (the
    60-trial MI variants on short tiles, or17_120 on its own); every other pipeline runs all variants together."""
    if mod.NAME not in MB_PIPES:
        return [(None, list(specs))]
    return _memo_groups(mod, ("tiles", tuple(specs)), lambda: _tiling_tiers(mod, specs))


def _tiling_tiers(mod, specs):
    by = {}
    for v in specs:
        try:
            n = len(_sched(mod, mod.schedule, specs[v], GROUP_START_S, np.random.default_rng(SEED)).onsets_s)
        except Exception:
            n = -1
        by.setdefault(n, []).append(v)
    if len(by) == 1:
        return [(None, list(specs))]
    return [(f"n{n}", vs) for n, vs in sorted(by.items())]


def _sched(mod, fn, spec, start_s, rng):
    with wr.module_globals(mod, spec["globals"]):
        return fn(start_s, rng, spec["arg"])


def _run(mod, fn, spec, rec, ov, rng):
    with wr.module_globals(mod, spec["globals"]):
        return fn(rec, ov, spec["arg"], rng)


def run_recording(rec, mod, spec, ctx, done, write):
    """"""
    if mod.FAMILY != spec["family"]:
        raise ValueError(f"{mod.NAME} is a {mod.FAMILY!r} pipeline; {spec['name']} needs {spec['family']!r}")
    resp, montage = ctx["resp"], ctx["montage"]
    in_loc = rec.patient in (ctx.get("locator") or ())
    in_unc = rec.patient in (ctx.get("uncorrected") or ())
    if resp == "grid" and not in_loc:
        return
    names = None
    if montage == "kit8":
        if mod.NAME in KIT8_NA:
            raise ValueError(KIT8_NA[mod.NAME])
        rec = kit8(rec)
        names = rec.names
    pdoc_g = {}
    if spec["source"] == "pdoc":
        pdoc_g = wr.pdoc_overrides(mod)
        if pdoc_g is None:
            raise ValueError(wr.PDOC_NA_MODULES[mod.NAME])
    specs = variant_specs(mod, spec["source"])
    prim = primary_variant(mod, spec["source"])
    tasks, grid_tag = grid_points(mod.FAMILY, resp, ctx["split"], spec["name"], mod.NAME, ctx.get("spec_path"))
    base = {"montage": montage, "n_channels": len(rec.names), "grid": grid_tag, "primary_variant": prim}
    if spec["source"] == "pdoc":
        base["channel_set"] = wr.pdoc_channel_set(mod)
    if resp != "none":
        base.update(in_locator=in_loc, in_uncorrected=in_unc)
    with channels(mod, names), wr.module_globals(mod, pdoc_g):
        for tile, tvars in tiling_tiers(mod, specs):
            _run_tile(rec, mod, spec, ctx, done, write, {v: specs[v] for v in tvars}, prim, tasks,
                      dict(base, tile=tile), tile, in_unc)
    _clear_cache()


def _run_tile(rec, mod, spec, ctx, done, write, specs, prim, tasks, base, tile, in_unc):
    resp, montage = ctx["resp"], ctx["montage"]
    sfx = f"|{tile}" if tile else ""
    plan, tier, need, why = _plan(rec, mod, spec, ctx, specs)
    if not plan:
        k = f"{rec.uid}|nofit{sfx}|{montage}"
        if k not in done:
            write(_base_row(rec, mod, spec, ctx, {**base, "key": k, "status": why, "variant": None, "need_s": need}))
        return
    if resp != "none":
        plan = plan[:1]
        if not has_responder(mod):
            for o in plan:
                k = f"{rec.uid}|{o['j']}|noresp{sfx}|{montage}"
                if k not in done:
                    write(_base_row(rec, mod, spec, ctx, {**base, "key": k, "status": "no_responder_model",
                                                          "variant": None, "j": o["j"], "start_s": o["start_s"]}))
            return
    groups = timing_groups(mod, specs, tier, prim)
    gof = {v: g for g, vs in groups.items() for v in vs}
    cov_cache = {}
    for o in plan:
        _run_overlay(rec, mod, spec, ctx, done, write, specs, tier, groups, gof, o, tasks, base, in_unc, cov_cache)


def _jobs(mod, resp, tier, groups, gof, o, tasks, in_unc):
    """[(grid point, variant or None (a randomized-only row), timing group, extra fields)] of one overlay."""
    jobs = []
    for gp, vs, g, extra in tasks:
        if resp == "eval":
            sub = (mod.NAME, gp[0]) in SUBSAMPLE_ONLY or (mod.NAME in SUBSAMPLE_GROUPS
                                                          and not extra.get("pipeline_group", True))
            if sub and not in_unc or g not in groups:
                continue
            ex = dict(extra, eval_set="uncorrected_subsample" if sub else "all")
            jobs += [(gp, v, g, ex) for v in vs if v in tier]
        elif o["seq"]:
            jobs += [(gp, v, gof[v], extra) for v in (vs if vs is not None else tier) if v in tier]
        elif resp == "none" and o["rand"]:
            jobs += [(gp, None, g_, extra) for g_ in groups]
    return jobs


def _rand_fields(rec, mod, specs, groups, o, gp, g, ovr, cov_cache):
    """rand_* fields of the randomized method of timing group g (its representative's schedule_randomized and
    features) at grid point gp; the feature cache is cleared before and after (the randomized overlay's site layout
    and injected signal stay with it), and the injected copy is freed."""
    j = o["j"]
    ovr, err_r, _ = ovr
    f = {"rand_variant": g}
    if ovr is None or not _inside(mod, ovr, o):
        f.update(_result_fields(mod, None, err_r or "extent_outside_window", 0.0, "rand_"))
        return f
    _clear_cache()
    rec_r, e_inj, _ = _call(_injected, rec, mod, ovr, gp, rec.uid, j, 1000 + list(groups).index(g))
    res, err, sec = (None, e_inj, 0.0) if rec_r is None else _call_run(
        mod, _run, mod, lambda r, o_, a, gg: wr.run_randomized(mod, r, o_, a, gg), specs[g], rec_r, ovr,
        _rng(mod.NAME, rec.uid, j, 4, _crc(g)))
    del rec_r
    _clear_cache()
    f.update(_result_fields(mod, res, err, sec, "rand_"))
    f.update(_covariates(rec, ovr, mod, cov_cache, "rand_"))
    f.update(scalars(res and res.get("extra"), "rand_x_"))
    return f


def _run_overlay(rec, mod, spec, ctx, done, write, specs, tier, groups, gof, o, tasks, base, in_unc, cov_cache):
    resp, montage, j = ctx["resp"], ctx["montage"], o["j"]
    key = lambda gp, v, g: _key(rec.uid, j, v if v is not None else "~" + g, *gp, montage)
    jobs = [x for x in _jobs(mod, resp, tier, groups, gof, o, tasks, in_unc) if key(*x[:3]) not in done]
    if not jobs:
        return
    t_ov = time.perf_counter()
    sched, sig_idx = {}, {}
    if any(v is not None for _, v, _, _ in jobs):
        for v in tier:
            ov, err, _ = _call(_sched, mod, mod.schedule, specs[v], o["start_s"], _rng(mod.NAME, rec.uid, j, 1))
            sig = None if ov is None else (np.asarray(ov.onsets_s, float).tobytes(), np.asarray(ov.labels).tobytes())
            sched[v] = (ov, err, sig)
            if sig is not None:
                sig_idx.setdefault(sig, len(sig_idx))
    do_rand = resp == "eval" or (resp == "none" and o["rand"])
    ovr = {}
    if do_rand:
        for g in dict.fromkeys(g for _, _, g, _ in jobs):
            ovr[g] = _call(_sched, mod, mod.schedule_randomized, specs[g], o["start_s"],
                           _rng(mod.NAME, rec.uid, j, 3, _crc(g)))
    for gp in dict.fromkeys(gp for gp, _, _, _ in jobs):
        here = [x for x in jobs if x[0] == gp]
        rr = {g: _rand_fields(rec, mod, specs, groups, o, gp, g, ovr[g], cov_cache) for g in ovr
              if any(x[2] == g for x in here)}
        ro = {"j": j, "start_s": o["start_s"], "seq": o["seq"], "overlap": o["overlap"], "model": gp[0],
              "effect": gp[1], "prob": gp[2]}
        pub = sorted((x for x in here if x[1] is not None),
                     key=lambda x: (-1 if sched[x[1]][2] is None else sig_idx[sched[x[1]][2]], tier.index(x[1])))
        inj = {}
        for _, v, g, extra in pub:
            row = _base_row(rec, mod, spec, ctx, {**base, **ro, **extra, "key": key(gp, v, g), "variant": v,
                                                  "group": g})
            _published_row(row, rec, mod, spec, ctx, specs[v], v, sched[v], sig_idx, gp, o, inj, in_unc, cov_cache)
            row.update(rr.get(g, {}))
            write(row)
        inj.clear()
        if gp[0] != "none":
            _clear_cache()
        for _, v, g, extra in (x for x in here if x[1] is None):
            r = rr.get(g, {})
            row = _base_row(rec, mod, spec, ctx, {**base, **ro, **extra, "key": key(gp, None, g), "variant": None,
                                                  "group": g, "status": _status(r.get("rand_error")), "stat": None,
                                                  "p": None, "call": None, "size_stat": None})
            row.update(r)
            write(row)
    print(f"{spec['name']} {mod.NAME} {rec.uid} j={j} seq={o['seq']} {len(jobs)} rows "
          f"{time.perf_counter() - t_ov:.1f} s", flush=True)


def _published_row(row, rec, mod, spec, ctx, sp, v, sched_v, sig_idx, gp, o, inj, in_unc, cov_cache):
    """Fill one published-variant row: run() (null) or the statistic only (responders) on the injected copy of its
    schedule group (one copy per schedule; the cache is cleared whenever the copy is replaced)."""
    resp, j = ctx["resp"], o["j"]
    ov, err, sig = sched_v
    cfg = mod.VARIANTS.get(v, sp["arg"] if isinstance(sp["arg"], dict) else {})
    with wr.module_globals(mod, sp["globals"]):
        mp, _, _ = _call(wr.min_attainable_p, mod, sp["arg"])
    row.update({"min_attainable_p": _scalar(mp), "cannot_call": bool(mp is not None and mp > 0.05),
                "as_coded": bool(cfg.get("as_coded", False)) and not sp["short"],
                "short": sp["short"], "twin_of": ",".join(sp["twin_of"]) if sp["twin_of"] else None,
                "mode": "full" if resp == "none" else "stat_only",
                "n_onsets": None if ov is None else int(len(ov.onsets_s))})
    if ov is not None and not _inside(mod, ov, o):
        err, ov = "extent_outside_window", None
    if ov is None:
        row.update(_result_fields(mod, None, err, 0.0))
        row["status"] = err
        return
    if sig not in inj:
        if gp[0] != "none":
            inj.clear()
            _clear_cache()
        inj[sig] = _call(_injected, rec, mod, ov, gp, rec.uid, j, sig_idx[sig])
    rec_v, e_inj, _ = inj[sig]
    g_run = _rng(mod.NAME, rec.uid, j, 2, _crc(v))
    if rec_v is None:
        res, err, sec = None, e_inj, 0.0
    elif resp == "none":
        res, err, sec = _call_run(mod, _run, mod, mod.run, sp, rec_v, ov, g_run)
    else:
        res, err, sec = _call_run(mod, _run, mod, lambda r, o_, a, g: wr.stat_only(mod, r, o_, a, g), sp, rec_v, ov,
                                  g_run)
    row.update(_covariates(rec, ov, mod, cov_cache))
    row.update(_result_fields(mod, res, err, sec))
    row.update(scalars(res and res.get("extra")))
    row["n_usable"] = _usable(res, ov)
    row["status"] = _status(err)
    if res is not None and resp != "none":
        row["inference"] = bool(res["extra"].get("inference", True))
        if not row["inference"]:
            row["p"], row["call"] = None, None
    if resp == "eval":
        full = None
        if in_unc and res is not None and uncorrected_point(mod.NAME, gp[0], gp[2]):
            full = (res, None, sec) if row["inference"] else _call_run(
                mod, _run, mod, mod.run, sp, rec_v, ov, _rng(mod.NAME, rec.uid, j, 2, _crc(v)))
        row.update(_result_fields(mod, full[0], full[1], full[2], "perm_") if full else
                   {"perm_stat": None, "perm_p": None, "perm_call": None, "perm_size_stat": None,
                    "perm_p_censored": None})


def _inside(mod, ov, o):
    t0, t1 = mod.extent(ov)
    return o["window"][0] - 1e-6 <= t0 and t1 <= o["window"][1] + 1e-6


def _adapter_fields(info):
    """Scalars of an adapted overlay's meta["adapter"] (counts, what changed, the contrast)."""
    usable = info.get("usable") or {}
    out = {"contrast": info.get("contrast"), "adapt": "; ".join(info.get("changed") or []) or None,
           "adapt_n_usable": ADAPT_USABLE[info["pipe"]](usable) if info.get("pipe") in ADAPT_USABLE else None,
           "adapt_n_skipped_variants": len(info.get("skip_variants") or []),
           "adapt_periodic": _scalar(info.get("periodic")),
           "channel_substitution": ", ".join(f"{s} as {t}" for t, s in info["channel_substitution"].items())
           if info.get("channel_substitution") else None}
    out.update({f"adapt_usable_{k}": _scalar(v) for k, v in usable.items()})
    out.update({f"adapt_dropped_{k}": _scalar(v) for k, v in (info.get("dropped") or {}).items()})
    return out


def run_events(rec, mod, spec, ctx, done, write, passive, adapters):
    """npassive_* / chancount_*: the published pipelines on real events. Per matching contrast
    (adapters.contrasts_for): the published-design overlay for run() of every usable variant, through the adapter
    wrappers; N/A (adapters.NotApplicable) and skipped variants are rows. On N-passive these calls form the separate
    "sensory read as volition" family (a label permutation on physically different stimuli is not a volition null);
    the proposed method there is the instruction-randomization test (run_instruction_subject)."""
    if mod.FAMILY != "erp":
        raise ValueError(f"{spec['name']} needs an 'erp' pipeline, not {mod.NAME}")
    if spec["kind"] == "chancount":
        order = nested_channel_order(rec.names)
        sets = [(f"n{k}", order[:k]) for k in chancount_sizes(len(order))]
    else:
        sets = [(ctx["montage"], kit8(rec).names if ctx["montage"] == "kit8" else None)]
    base = {"j": 0, "start_s": None, "seq": True, "overlap": False, "model": "none", "effect": 0.0, "prob": 0.0,
            "primary_variant": None, "design": "published",
            "family_tag": "channel_count" if spec["kind"] == "chancount" else "sensory_read_as_volition"}

    def na(tag, contrast, why):
        k = f"{rec.uid}|na|{tag}|{contrast}"
        if k not in done:
            write(_base_row(rec, mod, spec, ctx, {**base, "key": k, "montage": tag, "contrast": contrast,
                                                  "variant": None, "status": f"na: {why}"}))
    contrasts = tuple(adapters.contrasts_for(mod, spec["source"]))
    if not contrasts:
        for tag, _ in sets:
            na(tag, "-", f"no matching contrast for {mod.NAME} on {spec['source']}")
        return
    for contrast in contrasts:
        try:
            src = passive.passive_overlay(rec, contrast)
        except ValueError as e:
            for tag, _ in sets:
                na(tag, contrast, e)
            continue
        for tag, names in sets:
            r = rec if names is None else subset(rec, names)
            seen = None if names is None else adapters.recording_for(mod, r).names
            cov = {}
            with channels(mod, seen):
                key = {v: _key(rec.uid, 0, v, *NULL_ROW, tag) + f"|{contrast}" for v in mod.VARIANTS}
                todo = [v for v in mod.VARIANTS if key[v] not in done]
                if not todo:
                    continue
                try:
                    pub = adapters.adapt(mod, r, src, None, design="published")
                except adapters.NotApplicable as e:
                    na(tag, contrast, e)
                    continue
                usable = list(adapters.usable_variants(mod, pub))
                info = _adapter_fields(pub.meta["adapter"])
                for v in todo:
                    row = _base_row(r, mod, spec, ctx, {**base, **info, "key": key[v], "montage": tag,
                                                        "n_channels": len(r.names), "variant": v,
                                                        "as_coded": bool(mod.VARIANTS[v].get("as_coded", False)),
                                                        "n_onsets": int(len(pub.onsets_s))})
                    if v not in usable:
                        row["status"] = "skipped_by_adapter"
                        write(row)
                        continue
                    res, err, sec = _call_run(mod, adapters.run, mod, r, pub, v,
                                              _rng(mod.NAME, rec.uid, 0, 2, _crc(v), _crc(contrast)))
                    row.update(_covariates(r, pub, mod, cov))
                    row.update(_result_fields(mod, res, err, sec))
                    row.update(scalars(res and res.get("extra")))
                    row["n_usable"] = _usable(res, pub)
                    row["status"] = _status(err)
                    write(row)
            _clear_cache()


def _instruction_row(rec, mod, spec, ctx, key, unit, res, err, sec, extra):
    row = _base_row(rec, mod, spec, ctx, {"key": key, "j": 0, "start_s": None, "seq": True, "overlap": False,
                                          "model": "none", "effect": 0.0, "prob": 0.0, "montage": "native",
                                          "contrast": "oddball", "design": "instruction", "unit": unit,
                                          "variant": None, "rand_variant": None, **extra})
    row.update(_result_fields(mod, res, err, sec, "rand_"))
    row.update(scalars(res and res.get("extra")))
    row["status"] = "ok" if err is None else err
    return row


def run_instruction_subject(patient, recs, mod, spec, ctx, done, write, passive, adapters):
    """One subject of an N-passive null: the published family per run (run_events), the instruction-randomization
    test per run (SECONDARY, unit "run") and pooled over the subject's runs (PRIMARY, unit "subject":
    adapters.pool_instruction_features / run_instruction_pooled). Each recording is loaded once: the per-run work is
    done inside the generator the pool consumes. Runs that cannot form the blocks are rows and left out of the pool."""
    skipped = []
    native = ctx["montage"] == "native"

    def runs():
        for rec in recs:
            run_events(rec, mod, spec, ctx, done, write, passive, adapters)
            if not native:
                continue
            k = f"{rec.uid}|instruction|run|native"
            try:
                a = adapters.instruction_overlay(mod, rec, passive.passive_overlay(rec, "oddball"))
            except adapters.NotApplicable as e:
                skipped.append(rec.uid)
                if k not in done:
                    write(_instruction_row(rec, mod, spec, ctx, k, "run", None, f"na: {e}", 0.0, {}))
                continue
            except Exception as e:
                skipped.append(rec.uid)
                if k not in done:
                    write(_instruction_row(rec, mod, spec, ctx, k, "run", None,
                                           f"error: {type(e).__name__}: {str(e)[:200]}", 0.0, {}))
                continue
            if k not in done:
                res, err, sec = _call(lambda: adapters.run_instruction(mod, rec, a, _rng(mod.NAME, rec.uid, 0, 6),
                                                                       n_perm=wr.rand_b(mod.NAME)))
                write(_instruction_row(rec, mod, spec, ctx, k, "run", res, err, sec,
                                       _adapter_fields(a.meta["adapter"])))
            _clear_cache()
            yield rec, a

    gen = runs()
    ks = f"{patient}|instruction|subject|native"
    if not native or ks in done:
        for _ in gen:
            pass
        return
    F, err = None, None
    try:
        F = adapters.pool_instruction_features(mod, gen)
    except Exception as e:
        err = f"na: {e}" if isinstance(e, ValueError) else f"error: {type(e).__name__}: {str(e)[:200]}"
    for _ in gen:
        pass
    res, sec = None, 0.0
    if F is not None:
        res, err, sec = _call(lambda: adapters.run_instruction_pooled(mod, F, n_perm=wr.rand_b(mod.NAME)))
    subj = en.Recording(source=spec["source"], uid=patient, patient=patient, x=np.zeros((0, 0)), fs=0.0, names=[],
                        meta={"dev": ctx["split"] == "dev"})
    write(_instruction_row(subj, mod, spec, ctx, ks, "subject", res, err, sec,
                           {"n_skipped_runs": len(skipped), "skipped_runs": ",".join(skipped) or None}))
    _clear_cache()



def shard_dir(null, pipe, montage, resp, split):
    return os.path.join(SH[split], f"{null}__{pipe}__{montage}__{resp}")


def shard_file(null, pipe, montage, resp, split, shard):
    return os.path.join(shard_dir(null, pipe, montage, resp, split), f"rows{shard:02d}.jsonl")


def list_items(spec, split, skip=frozenset()):
    """Engine descriptors of the null set's split (no signal loaded), without the skipped uids."""
    return [it for it in en._items(spec["source"], split) if it["uid"] not in skip]


def safe_dev(items, pending):
    """While the BDSP dev split is provisional, each of the `pending` recordings (ranked after the first dev recording,
    not usable yet) can displace the lowest-ranked dev recording into analysis: those `pending` are skipped."""
    return sorted(items, key=lambda it: it.get("meta", {}).get("rank", 0))[pending:]


def iter_recordings(spec, split, limit, shard, nshards, skip=frozenset(), patients=None, items=None):
    """(recording, recordings in the split) for this shard; engine sources are listed first and only this shard's
    recordings are loaded. Every recording is checked against the split. Provisional BDSP dev split: safe_dev. skip:
    uids never loaded (the recordings touched before the dev freeze, in the analysis split). patients: load only
    these (the locator of --resp grid). items: a listing already made by list_items."""
    if spec["kind"] in ("ncmd", "nstim0"):
        items = list_items(spec, split, skip) if items is None else items
        n = len(items)
        if spec["source"] == "bdsp" and split == "dev" and any(not it.get("meta", {}).get("split_final", True)
                                                               for it in items):
            pending = en.bdsp_split(en.SIG, en.MOR, en.META, en.DEV_FILE)["pending"]
            keep = safe_dev(items, pending)
            print(f"provisional BDSP dev split: {pending} pending, skipping "
                  f"{sorted(it['uid'] for it in items if it not in keep)}", flush=True)
            items = keep
        for i, it in enumerate(items[:limit]):
            if i % nshards == shard and (patients is None or it.get("patient") in patients):
                rec = en._load(it)
                check_member(rec, split)
                yield rec, n
    else:
        passive = importlib.import_module("bench.passive")
        for i, rec in enumerate(passive.load_passive(spec["source"], split)):
            if limit is not None and i >= limit:
                break
            if i % nshards == shard and rec.uid not in skip:
                check_member(rec, split)
                yield rec, None


def iter_subjects(spec, split, limit, shard, nshards, skip=frozenset()):
    """(patient, generator of that patient's recordings) for this shard, shards by subject (all of a subject's runs
    in one shard; bench.passive yields recordings sorted by subject). Every recording is checked against the split."""
    passive = importlib.import_module("bench.passive")
    seen = set()
    for i, (pat, grp) in enumerate(itertools.groupby(passive.load_passive(spec["source"], split),
                                                     key=lambda r: r.patient)):
        if pat in seen:
            raise RuntimeError(f"{spec['source']}: recordings of {pat} are not consecutive")
        seen.add(pat)
        if limit is not None and i >= limit:
            break
        if i % nshards != shard:
            continue

        def recs(grp=grp):
            for rec in grp:
                if rec.uid in skip:
                    continue
                check_member(rec, split)
                yield rec
        yield pat, recs()


def _done(fn, split):
    done = set()
    if os.path.exists(fn):
        with open(fn) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") != split:
                    raise RuntimeError(f"{fn} holds rows of split {r.get('split')!r}: refused for split {split!r}")
                done.add(r["key"])
    return done


def run_shard(null, pipe, split, shard=0, nshards=1, resp="none", montage="native", limit=None, max_overlays=None,
              spec_path=None):
    guard_split(split, spec_path)
    skip = touched_uids() if split != "dev" else frozenset()
    spec = parse_null(null)
    if spec["kind"] == "chancount" and pipe != "chennu13":
        raise ValueError("the channel-count arm runs chennu13 only")
    if resp not in ("none", "grid", "eval") or montage not in ("native", "kit8"):
        raise ValueError("resp must be none|grid|eval and montage native|kit8")
    if spec["kind"] in ("npassive", "chancount") and resp != "none":
        raise ValueError("responders are injected on the no-command / no-stimulus nulls only")
    if spec["source"] == "pdoc":
        if resp != "none":
            raise ValueError("the pDoC arm is nulls only (reduced montage, second stage): no responders")
        if montage != "native":
            raise ValueError("the pDoC arm is the reduced-montage arm itself: no kit8")
        if pipe in wr.PDOC_NA_MODULES:
            raise ValueError(wr.PDOC_NA_MODULES[pipe])
    if montage == "kit8" and pipe in KIT8_NA:
        raise ValueError(KIT8_NA[pipe])
    mod = load_pipe(pipe)
    if mod.FAMILY != spec["family"]:
        raise ValueError(f"{pipe} is a {mod.FAMILY!r} pipeline; {null} needs {spec['family']!r}")
    tag = montage if spec["kind"] != "chancount" else "native"
    fn = shard_file(null, pipe, tag, resp, split, shard)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = _done(fn, split)
    ctx = {"null": null, "split": split, "montage": montage, "resp": resp, "max_overlays": max_overlays,
           "locator": None, "uncorrected": None, "col113": None, "spec_path": spec_path}
    if pipe == "columbia" and spec["source"] == "bdsp" and split == "analysis":
        c = columbia113()
        ctx["col113"] = dict(zip(c["uid"], c["start_s"].astype(float)))
    items = only = None
    if spec["kind"] in ("ncmd", "nstim0") and resp != "none":
        items = list_items(spec, split, skip)
        ctx["locator"], ctx["uncorrected"] = subsamples([it["patient"] for it in items], spec["source"])
        only = ctx["locator"] if resp == "grid" else None
    passive = adapters = None
    if spec["kind"] in ("npassive", "chancount"):
        passive = importlib.import_module("bench.passive")
        adapters = importlib.import_module("bench.adapters")
    with open(fn, "a") as fh:
        def write(row):
            fh.write(json.dumps(row, allow_nan=False) + "\n")
            fh.flush()
            done.add(row["key"])
        if skip:
            print(f"{len(skip)} recording(s) touched before the dev freeze are skipped", flush=True)
        if spec["kind"] == "npassive":
            for pat, recs in iter_subjects(spec, split, limit, shard, nshards, skip):
                t = time.perf_counter()
                run_instruction_subject(pat, recs, mod, spec, ctx, done, write, passive, adapters)
                print(f"{null} {pipe} {pat} done in {time.perf_counter() - t:.1f} s", flush=True)
            return
        for rec, n in iter_recordings(spec, split, limit, shard, nshards, skip, only, items):
            t = time.perf_counter()
            if passive is None:
                run_recording(rec, mod, spec, {**ctx, "n_recs": n}, done, write)
            else:
                run_events(rec, mod, spec, ctx, done, write, passive, adapters)
            print(f"{null} {pipe} {rec.uid} done in {time.perf_counter() - t:.1f} s", flush=True)
            del rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", required=True)
    ap.add_argument("--pipe", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "analysis"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--resp", default="none", choices=("none", "grid", "eval"))
    ap.add_argument("--montage", default="native", choices=("native", "kit8"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-overlays", type=int, default=None, help="timing/debug: first N overlays per recording")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    run_shard(a.null, a.pipe, a.split, a.shard, a.nshards, a.resp, a.montage, a.limit, a.max_overlays)


if __name__ == "__main__":
    main()
