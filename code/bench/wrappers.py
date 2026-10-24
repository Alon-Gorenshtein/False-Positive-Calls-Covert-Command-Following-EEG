"""Thin per-module wrappers for the null benchmark's analyses, so that no module's internals are edited.

stat_only(mod, rec, ov, variant, rng): the observed statistic of run() (the scalar stat_for_size_correction reads)
without permutation inference, for the statistic-based size correction of the responder runs. It equals run()'s
observed statistic for the same rng where the module draws that statistic before any permutation:
- columbia: 112.cv_scores on the overlay's features, the mean LOTO fold AUC (112.exact_test computes it first);
- edlow17: the mean CV accuracy (the first cv_partitions draw of run());
- dubois 'perm' variants: run() with n_perm_labels = 0 (the selected configuration's peak DA);
- cruse11 gf13 variants: run() with GF13_MAX_LABELINGS = 0 (no block relabelling).
Everything else calls run() itself: no permutation inference in run() (cornell, cruse11 cr11 / cr12n, dubois
phase1 / t_only, mbmi, mbvt, gibson16 SSEP, sokoliuk21 roc116), a randomisation p as the statistic (chennu13, gibson16
P3a / P3b, sokoliuk21 surrogate rules), or cruse12p (its second comparison draws its CV partition after the first
comparison's permutations; < 1 s per overlay). extra['inference'] says whether the module's own p / call were computed.

Shortened BDSP variants (BDSP segments are 30 min; the native lengths need 3,264 s / 1,891 s):
- cruse11: 3 hand + 3 toe blocks (fewer block pairs), from the module's own _variant(); one short variant per
  (method, band power, baseline CI), twin of the native b4 and b6 variants. GF13 with 3 + 3 blocks has 20 balanced
  relabellings and the mirror of the observed one ties it, so its p is >= 2 / 20: it cannot call.
- cruse12p: 4 blocks of 36 trials (144), the module's N_BLOCKS (and RUN_S) overridden for the duration of each call.
short_variants(mod, source) -> {name: {"arg": variant key or dict, "globals": overrides, "twin_of": [native names],
"short": True}}; module_globals applies "globals"; run_len gives the assessment length.

run_randomized(mod, rec, ov, variant, rng): the randomization-based method with B = RAND_B[module] Monte Carlo
re-draws, p = (ge + 1) / (B + 1), exact at any B, and decision-exact early stopping at the
B-specific count (motor_common.stop_count; 10 at B = 200). Every module but Columbia reads B from the variant's
'n_perm_rand'; Columbia's run_randomized fixes 112's B = 500 / STOP_AT = 25, so its wrapper calls 112.trialflip_test
with B and stop_at = columbia_stop_at(B) itself (113's test iii, which the benchmark reuses for Columbia's null rows,
ran at B = 500).

min_attainable_p(mod, variant): the smallest p the published variant can return by construction; cannot_call when it
exceeds .05 (e.g. GF13 with 3 + 3 blocks: 2 / 20). Such variants are run and reported but never selected as the
validity primary or the sensitivity comparator (115).

size_stat(mod, res): stat_for_size_correction with +-inf capped at +-SIZE_CAP (du Bois's zero-variance paired t), so
the strongest responders stay in the size-corrected comparison. p_censored(res): True when res's p is an
early-stopped lower bound (its pile), which the p-based correction treats as censored (115).

pdoc_overrides(mod): the pDoC reduced-montage arm (no responders). Every Mendeley PSG
holds the bare 10-20 sites F3 F4 C3 C4 O1 O2 only, so each module reads a documented subset (PDOC_CH), set through its
channel globals for the call (module_globals):
- whole-head montages (cornell, edlow17, chennu13, sokoliuk21: the 19 10-20 sites; dubois 'full') and the ERP
  pipelines whose sites are central / parietal / occipital (mbvt, gibson16): all six; no parietal site exists;
- motor-strip montages (cruse11's 3 x 3 grid around C3-C4, mbmi's 16 central sites, dubois 'motor'): F3 F4 C3 C4;
- cruse12p's bipolar pairs straddling C3 / C4 (FC3-CP3, FC4-CP4; BDSP F3-P3, F4-P4): F3-C3 and F4-C4.
Hjorth Laplacians (cornell, edlow17, cruse11) are taken over the six sites, so their neighbours are far apart.
Columbia is N/A (PDOC_NA_MODULES); gibson16's SSEP variants read Pz and are N/A (pdoc_variant_na)."""
import contextlib
import math

import numpy as np

from . import motor_common as mc

SHORT = {"cruse11": {"n_blocks": 3}, "cruse12p": {"N_BLOCKS": 4}}
RAND_B = {"columbia": 200, "cornell": 200, "cruse11": 200, "cruse12p": 200, "dubois": 200, "edlow17": 200, "mbmi": 200,
          "mbvt": 200, "chennu13": 200, "gibson16": 200, "sokoliuk21": 200}
SHORT_SOURCES = ("bdsp",)


@contextlib.contextmanager
def module_globals(mod, overrides):
    """Module-level names set to the given values for the duration (restored on exit, also after an exception)."""
    saved = {k: getattr(mod, k) for k in overrides}
    try:
        for k, v in overrides.items():
            setattr(mod, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)


def cruse12p_run_s(mod, n_blocks):
    """pipe_cruse12p.RUN_S for n_blocks blocks (the module's own expression)."""
    return (n_blocks * 3 * mod.PER_CLASS * (mod.INSTR_S + mod.SILENCE_S[1]) + (n_blocks - 1) * mod.BREAK_S
            + mod.EPOCH_S + 5.0)


def short_variants(mod, source):
    if source not in SHORT_SOURCES or mod.NAME not in SHORT:
        return {}
    out = {}
    if mod.NAME == "cruse11":
        nb = SHORT["cruse11"]["n_blocks"]
        for v, c in mod.VARIANTS.items():
            name = f"{c['method']}_b{nb}_{c['bp']}" + (f"_ci{str(c['ci_base'])[2:]}" if c["ci_base"] else "")
            spec = out.setdefault(name, {"arg": mod._variant(c["method"], nb, c["bp"], c["ci_base"]), "globals": {},
                                         "twin_of": [], "short": True})
            spec["twin_of"].append(v)
    elif mod.NAME == "cruse12p":
        nb = SHORT["cruse12p"]["N_BLOCKS"]
        for v, c in mod.VARIANTS.items():
            out[f"{v}_b{nb}"] = {"arg": dict(c), "globals": {"N_BLOCKS": nb, "RUN_S": cruse12p_run_s(mod, nb)},
                                 "twin_of": [v], "short": True}
    return out


def run_len(mod, spec):
    """Assessment length of a variant spec (its globals applied)."""
    with module_globals(mod, spec.get("globals", {})):
        return float(mod.run_s(spec["arg"]) if hasattr(mod, "run_s") else mod.RUN_S)


def wrapped(name, cfg):
    """True if stat_only skips permutation inference for this module and variant settings."""
    return (name in ("columbia", "edlow17") or (name == "dubois" and cfg.get("call") == "perm")
            or (name == "cruse11" and cfg.get("method") == "gf13"))


def _mark(res, inference):
    res["extra"]["inference"] = inference
    if not inference:
        res["p"], res["call"] = None, False
    return res


def stat_only(mod, rec, ov, variant, rng):
    name = mod.NAME
    cfg = mc.cfg(mod.VARIANTS, variant) if hasattr(mod, "VARIANTS") and (
        isinstance(variant, dict) or variant in getattr(mod, "VARIANTS", {})) else {}
    if not wrapped(name, cfg):
        return _mark(mod.run(rec, ov, variant, rng), True)
    if name == "columbia":
        X, y, g = mod._xy(rec, ov)
        scores = mod.ce.cv_scores(X, y, g, mod.ce._as_rs(rng))
        return _mark(mc.result(scores.mean(0), None, False, n_folds=int(len(scores))), False)
    if name == "edlow17":
        gen = mc.gen(rng)
        X, blocks, keep = mod._feat(rec, ov)
        y = np.asarray(ov.meta["block_class"])[blocks]
        obs = mod.cv_accuracy(X, y, mod.cv_partitions(len(y), blocks, cfg["cv_unit"], gen))
        return _mark(mc.result(obs, None, False, n_cols=int(len(y))), False)
    if name == "dubois":
        return _mark(mod.run(rec, ov, dict(cfg, n_perm_labels=0), rng), False)
    with module_globals(mod, {"GF13_MAX_LABELINGS": 0}):
        return _mark(mod.run(rec, ov, variant, rng), False)



def rand_b(name):
    return RAND_B.get(name, 200)


def columbia_stop_at(B, alpha=0.05):
    """112._run_perms stops once this many re-draws reach the observed AUC: then p = (ge + 1) / (B + 1) >= alpha."""
    return mc.stop_count(B, alpha)


def _cfg(mod, variant):
    if isinstance(variant, dict) or variant in getattr(mod, "VARIANTS", {}):
        return mc.cfg(mod.VARIANTS, variant)
    return {}


def run_randomized(mod, rec, ov, variant, rng):
    B = rand_b(mod.NAME)
    if mod.NAME == "columbia":
        X, y, g = mod._xy(rec, ov)
        return mod._pack(mod.ce.trialflip_test(X, y, g, mod.ce._as_rs(rng), n_perm=B, stop_at=columbia_stop_at(B)))
    return mod.run_randomized(rec, ov, dict(_cfg(mod, variant), n_perm_rand=B), rng)



def min_attainable_p(mod, variant):
    """The smallest p the published variant's call rule can return (0.0: analytic test or threshold rule)."""
    name, c = mod.NAME, _cfg(mod, variant)
    if name == "cruse11" and c.get("method") == "gf13":
        n = math.comb(2 * c["n_blocks"], c["n_blocks"])
        return 2.0 / n if n <= mod.GF13_MAX_LABELINGS else 1.0 / (mod.GF13_MAX_LABELINGS + 1)
    if name == "columbia":
        return 1.0 / (mod.N_PERM + 1)
    if name == "cruse12p":
        return 1.0 / (c.get("n_perm", mod.N_PERM) + 1)
    if name == "dubois" and c.get("call") == "perm":
        return 1.0 / (c.get("n_perm_labels", mod.N_PERM_LABELS) + 1)
    if name in ("chennu13",) or (name == "gibson16" and c.get("call_on") in ("p3a", "p3b")):
        return 1.0 / (mod.NUMRAND + 1)
    if name == "sokoliuk21" and c.get("rule", "roc116") != "roc116":
        return 0.5 / mod.REPS
    return 0.0


def cannot_call(mod, variant, alpha=0.05):
    return min_attainable_p(mod, variant) > alpha



SIZE_CAP = 1e12


def size_stat(mod, res):
    """mod.stat_for_size_correction(res) with +-inf capped at +-SIZE_CAP (du Bois's zero-variance paired t is +-inf for
    the strongest responders; dropping it would remove them from the locator and the comparison). NaN stays NaN."""
    v = float(mod.stat_for_size_correction(res))
    return math.copysign(SIZE_CAP, v) if math.isinf(v) else v



def p_censored(res):
    """True if res's p is only a lower bound: the run stopped early at its pile (decision-exact early stopping:
    (stop_count + 1) / (B + 1), or stop_count / B without the plus one). cruse12p's p is the smaller of two
    comparisons' p: censored when every comparison giving that minimum stopped."""
    if not res or res.get("p") is None:
        return False
    ex = res.get("extra") or {}
    if ex.get("stopped_early"):
        return True
    comps = ex.get("comparisons")
    if isinstance(comps, dict):
        hit = [c for c in comps.values() if isinstance(c, dict) and c.get("p") is not None
               and c["p"] <= res["p"] + 1e-12]
        return bool(hit) and all(bool(c.get("stopped_early")) for c in hit)
    return False



PDOC6 = ["F3", "F4", "C3", "C4", "O1", "O2"]
_MOTOR4 = ["F3", "F4", "C3", "C4"]
PDOC_CH = {"cornell": {"NATIVE_CH": PDOC6},
           "cruse11": {"SUB_1020": _MOTOR4, "NATIVE_CH": _MOTOR4},
           "cruse12p": {"NATIVE_CH": ["F3", "C3", "F4", "C4"], "SUB_1020": ["F3", "C3", "F4", "C4"]},
           "dubois": {"CHANSETS": {"full": PDOC6, "motor": _MOTOR4}, "NATIVE_CH": PDOC6},
           "edlow17": {"NATIVE_CH": PDOC6},
           "mbmi": {"NATIVE_CH": _MOTOR4, "SUB_1020": _MOTOR4},
           "mbvt": {"NATIVE_CH": PDOC6, "SUB_1020": PDOC6},
           "chennu13": {"NATIVE_CH": PDOC6},
           "gibson16": {"NATIVE_CH": PDOC6, "SUB_1020": PDOC6},
           "sokoliuk21": {"NATIVE_CH": PDOC6}}
PDOC_NA_MODULES = {"columbia": "N/A on pDoC: the released pipeline's CSD and features are defined on its 19-channel "
                               "montage (112), which a 6-channel PSG cannot supply"}


def pdoc_overrides(mod):
    """Module globals for the pDoC channel set; None when the module is N/A there; {} for a module without a set."""
    if mod.NAME in PDOC_NA_MODULES:
        return None
    return {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in PDOC_CH.get(mod.NAME, {}).items()}


def pdoc_channel_set(mod):
    """The channels the module reads on pDoC, as a comma-joined string (row field), or None."""
    ov = PDOC_CH.get(mod.NAME, {})
    sets = [",".join(v) if isinstance(v, list) else ";".join(f"{k}:{','.join(c)}" for k, c in v.items())
            for k, v in sorted(ov.items()) if k in ("NATIVE_CH", "CHANSETS")]
    return sets[0] if len(sets) == 1 else (" | ".join(sets) or None)


def pdoc_variant_na(mod, variant):
    """True for a variant whose analysis reads a site absent from every pDoC PSG (gibson16 SSEP: Pz)."""
    return mod.NAME == "gibson16" and _cfg(mod, variant).get("call_on") == "ssep"
