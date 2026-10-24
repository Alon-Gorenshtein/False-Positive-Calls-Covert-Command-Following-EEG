"""Pipeline adapters for real-stimulus overlays. The ERP modules were
written for their own schedules; adapt(pipe, rec, overlay) rebuilds a passive overlay (bench/passive.py
passive_overlay: labels 1 rare no-intent stimulus / 0 standard; meta rare_type, seq, train, contrast) into the classes,
groups and blocks the module's run() and run_randomized() read. Timing is ALWAYS as recorded.

Matching null per design (contrasts_for): oddball designs get the "oddball" contrast (rare vs
standard) on every dataset; the equiprobable counterbalanced designs (VT3 wrists, Gibson's two wrists, Chennu's two
counted / uncounted words) get kojimaB's "equiprobable" contrast (its two unattended deviants, rare_type 1 = "A" vs
2 = "B", against each other, with that stream's standards as the frequent stimuli). mbvt runs both on kojimaB: VT2
variants on "oddball", VT3 variants on "equiprobable"; every (pipeline, dataset, variant) thus has one contrast.

Designs: adapt(..., design="published") builds the published design (its assignment rule as the module's schedule()
draws it) for run(); design="randomized" builds the randomized design (as schedule_randomized() draws it) for
run_randomized(), so that the observed assignment comes from the same distribution as the module's re-draw
(exchangeability). Call the modules through run() / run_randomized() here: they refuse the wrong design (and, for
VT3, a per-instruction target-wrist assignment on the randomized path), refuse skipped variants, and pass the module
recording_for(pipe, rec).

Common to all adapters:
- stimuli whose module extent (pipe.extent) would leave the recording are dropped first ("edge");
- two stimuli are consecutive iff their stream positions differ by 1 (no excluded stimulus between them) and they
  share a pause-free train;
- adapter draws use rng, default np.random.default_rng(SEED + crc32(pipe + uid)) for the published design (the
  runner's seeding rule) and SEED + crc32(pipe + uid + ":randomized") for the randomized design;
- recording_for(pipe, rec): the same signal array under a module-specific uid (motor_common.memo keys its feature
  cache on uid + onsets, not on the module, so two modules on identical onsets would share features), with the
  SUBSTITUTE renames where the module's own site is missing;
- the result's meta["adapter"]: pipe, contrast, design, changed (what was changed, in words), usable (trial counts
  before the module's own artefact rejection), dropped (counts), skip_variants (run() must not be called with these),
  periodic (gibson16), run (run() applies: published design), randomized (run_randomized applies: randomized design,
  when the structure allows), randomized_reason, rand_variant (the module's primary variant for run_randomized, moved
  to this overlay's paradigm when needed) and, when used, channel_substitution.
- NotApplicable (raised): sokoliuk21 (no isochronous speech in N-passive), a montage without the module's native
  or 10-20 channel set, a contrast that is not the pipeline's matching null on the dataset, or too
  few trials for the module's structure.

mbvt (cat. §4): non-overlapping windows of 8 consecutive stimuli with exactly 1 rare + 7 standards (VT2; rare =
target on the right wrist, standards = non-targets on the left) or exactly one A + one B + 6 standards (VT3; A = left
wrist, B = right wrist, standards = distractors; the target wrist per instruction of 15 groups drawn 2 left / 2 right
as pipe_mbvt.schedule draws it; randomized design: per group, G // 2 left in random order, as schedule_randomized
draws its 30 / 30). Windows are taken earliest-first, which gives the largest number of non-overlapping windows
(equal-length interval scheduling); at most N_GROUPS (60, one run) are kept, the rest counted. The timing variants
(mur20 / ann18) are moot. VT2 has one design: the rare's position in its group is fixed by the recorded sequence.
chennu13 (cat. §2): blocks of chennu_block_len() consecutive contrast stimuli (the median words per block of the
module's own schedule, 97); a trailing partial block is dropped. The explicit word per block is drawn by the
module's rule (balanced, no 5 in a row). oddball: explicit = rare (word = the block's explicit word), implicit = the
standards (the other word), no distractors. equiprobable: A = YES, B = NO, explicit = the block's explicit word,
the standards = distractors (not epoched by the module). The n_blocks variants (20 / 12) are moot. Both designs draw
the explicit order by the same rule (run_randomized re-draws with it). Montages without T7 / T8 but with C5 / C6 (ERP
CORE) use C5 as T7 and C6 as T8, the nearest available sites (3.9 cm); the 19-channel layout is kept.
gibson16 (cat. §6): blocks = the pause-free trains. oddball: rare = the target wrist (right), standards = back;
there is no equiprobable non-target wrist, so the P3b variants and run_randomized do not apply. equiprobable: A =
left wrist, B = right wrist, standards = back, target wrist alternating per block starting right (the module's
published order; randomized design: nb // 2 right in random order, as schedule_randomized draws its 7 / 7). The
SSEP variants are skipped unless the recorded SOA is periodic: the SD of the SOAs between consecutive adapted
stimuli <= PERIODIC_CV x their mean (Kojima's unattended stream ~0.1%; ds003061 ~2.6% and ERP CORE ~5.1%: jittered).
sokoliuk21 (cat. §8): N/A.

Instruction randomization (instruction_overlay / run_instruction): the exact null of the
randomization-based method on N-passive data. Label permutation on physically different stimuli is not a volition
null, so the published pipelines' passive calls form a separate "sensory read as volition" family; here physical
stimulation is identical across conditions and only a sham instruction is randomized.
- Blocks: the stream after edge trimming, cut into consecutive blocks of the module's block length
  (instruction_block_len: mbvt 120 = one instruction of 15 groups of 8, chennu13 97 words, gibson16 the median
  stimuli per block of its schedule) inside each pause-free train when that gives >= MIN_BLOCKS blocks; else
  MIN_BLOCKS (8) blocks shared across the trains in proportion to their length, each train cut into equal parts.
  Fewer than 8 balanced blocks cannot reach p < .05 (6 blocks: 20 assignments, smallest p 1 / 20). Every block
  needs >= 2 rare and >= 2 standard stimuli.
- Sham instruction per block: "count" (COUNT) or "ignore" (IGNORE), nb // 2 count, in random order, seeded (default
  SEED + crc32(pipe + uid + ":instruction")). Kojima B: the sham instruction is an attended stream per block; count
  blocks attend the analysed stream (the really unattended stream of that trial: the only one analysed), ignore
  blocks the other stream. A count block therefore always conflicts with the real attention of the trial; such
  blocks are kept, because every analysed stimulus is really unattended in every block, so the sham label cannot
  carry real attention and the null stays exact (meta sham_stream, analysed_stream; usable n_conflict).
- Statistic, the module's own deviant-minus-standard feature difference, count minus ignore (one-sided):
  mbvt: per-epoch segment-mean features as pipe_mbvt.features (no grouping), Fisher LDA direction rare vs standard
  over all kept epochs (label-free with respect to the instructions); per block the mean rare score minus the mean
  standard score; mean over count blocks minus mean over ignore blocks.
  chennu13: GFP (pipe_chennu13.gfp_rows) of the pooled rare mean minus that of the pooled standard mean, per
  condition; the 400-700 ms maximum of count minus ignore.
  gibson16: per block the rare-minus-standard mean epoch over 50-750 ms; block-level independent-samples t (count vs
  ignore, pipe_gibson16.indep_t), threshold one-tailed .05 (df nb - 2), largest positive cluster mass over the
  module's spatio-temporal neighbours.
- Null: the block instructions re-drawn (balanced, the same draw), B = INSTR_B (500), decision-exact early stopping
  (motor_common.perm_test).
- Admissibility (the artefact rules fix which epochs are kept; the labels do not): a labelling is admissible when
  each condition keeps >= MIN_PER_BLOCK rare and standard epochs, holds >= INSTR_MIN_BLOCKS usable blocks (>= 1 rare
  and >= 1 standard kept epoch; mbvt 1, gibson16 2 for its t) and gives a finite statistic. Blocks without a usable
  pair are left out of the block-level statistics. An inadmissible re-draw is rejected and re-drawn
  (motor_common.admissible_stat: a conditional randomization test, exact over the admissible labellings; cap 10,000,
  then NoUsableTrials); it is never counted below or above the observed value. An inadmissible observed labelling,
  or too few kept epochs to fit the statistic, raises NoUsableTrials: an unusable assessment is never a negative.
- Unit: PRIMARY = the person. pool_instruction_features() takes one subject's runs (blocks stay within runs; runs are
  read one at a time) and run_instruction_pooled() draws the instructions balanced over all the subject's blocks
  (pooled_instruction; default seed SEED + crc32(pipe + patient + ":pooled")) and re-draws them the same way; the
  mbvt LDA direction is fitted on the subject's pooled epochs. SECONDARY = per run: run_instruction() on one
  instruction_overlay with its own draw."""
import dataclasses
import weakref
import zlib

import numpy as np
from scipy.stats import t as t_dist

from . import erp_common as ec
from . import motor_common as mc
from . import pipe_chennu13 as pc
from . import pipe_gibson16 as pg
from . import pipe_mbvt as pm
from . import pipe_sokoliuk21 as ps

SEED = 20261024
PIPES = {"mbvt": pm, "chennu13": pc, "gibson16": pg, "sokoliuk21": ps}
SUB_1020 = {"mbvt": pm.SUB_1020, "chennu13": pc.NATIVE_CH, "gibson16": pg.SUB_1020}
EQUIPROBABLE_DATASETS = ("kojimaB",)
PERIODIC_CV, PERIODIC_MIN = 0.01, 10
CHENNU_DRAWS = 10
DESIGNS = ("published", "randomized")
MIN_BLOCKS, MIN_PER_BLOCK, INSTR_B = 8, 2, 500
COUNT, IGNORE = 1, 0
INSTR_MIN_BLOCKS = {"mbvt": 1, "gibson16": 2}
KOJIMA_STREAM_OF = {1: 1, 101: 1, 102: 1, 2: 2, 103: 2, 104: 2}
SUBSTITUTE = {"chennu13": {"T7": "C5", "T8": "C6"}}


class NotApplicable(ValueError):
    """The pipeline cannot run on this passive recording / overlay; the message says why (reported as N/A)."""


def _module(pipe):
    return PIPES[pipe] if isinstance(pipe, str) else pipe


def contrasts_for(pipe, dataset):
    """The contrast(s) that are the pipeline's matching null on the dataset (module docstring)."""
    name = _module(pipe).NAME
    if name == "sokoliuk21":
        return ()
    eq = dataset in EQUIPROBABLE_DATASETS
    if name == "mbvt":
        return ("oddball", "equiprobable") if eq else ("oddball",)
    return ("equiprobable",) if eq else ("oddball",)


def usable_variants(pipe, adapted):
    skip = set(adapted.meta["adapter"]["skip_variants"])
    return [v for v in _module(pipe).VARIANTS if v not in skip]


_BLOCK_LEN = None


def chennu_block_len():
    """Median words per block of pipe_chennu13.schedule over CHENNU_DRAWS seeded draws (seeds 0..9)."""
    global _BLOCK_LEN
    if _BLOCK_LEN is None:
        n = [np.bincount(pc.schedule(0.0, np.random.default_rng(s), pc.DEFAULT).meta["block"])
             for s in range(CHENNU_DRAWS)]
        _BLOCK_LEN = int(np.median(np.concatenate(n)))
    return _BLOCK_LEN


def _links(seq, train):
    """links[k]: stimuli k and k + 1 are consecutive (stream positions differ by 1, same train)."""
    return (np.diff(seq) == 1) & (np.diff(train) == 0)


def _windows(link, valid, L):
    """Earliest-first non-overlapping windows [i, i + L) of consecutive stimuli with valid(i, i + L)."""
    out, i, n = [], 0, len(link) + 1
    while i + L <= n:
        if link[i:i + L - 1].all() and valid(i, i + L):
            out.append(i)
            i += L
        else:
            i += 1
    return out


def _edge_keep(m, rec, on):
    lo, hi = m.extent(mc.Overlay(onsets_s=np.array([0.0]), labels=np.array([1]), kind="stim"))
    return (on + lo >= 0) & (on + hi <= (rec.x.shape[1] - 1) / float(rec.fs))


def _substitution(name, names):
    """{module site: recorded site} for the module's sites missing from names whose substitute is present."""
    have = set(mc.canon(names))
    return {t: s for t, s in SUBSTITUTE.get(name, {}).items() if t not in have and s in have}


_VIEWS = weakref.WeakKeyDictionary()


def recording_for(pipe, rec):
    """The recording the module sees (module docstring): no data copied. One view per (recording object, module),
    held only as long as the recording, so the module's feature cache (keyed on the view's token) hits across
    variants and calls."""
    name = _module(pipe).NAME
    try:
        views = _VIEWS.setdefault(rec, {})
    except TypeError:
        views = {}
    if name not in views:
        inv = {s: t for t, s in _substitution(name, rec.names).items()}
        views[name] = dataclasses.replace(rec, uid=f"{rec.uid}#{name}", names=[inv.get(c, c) for c in rec.names],
                                          meta=dict(rec.meta))
    return views[name]


def _channels_ok(m, rec):
    have = set(mc.canon(recording_for(m, rec).names))
    return set(mc.canon(m.NATIVE_CH)) <= have or set(mc.canon(SUB_1020[m.NAME])) <= have


def _overlay(on, labels, meta, info):
    return mc.Overlay(onsets_s=np.asarray(on, float), labels=np.asarray(labels, int), kind="stim",
                      meta=dict(meta, adapter=info))


def adapt(pipe, rec, overlay, rng=None, design="published"):
    """The overlay the module's run() (design "published") or run_randomized() (design "randomized") accepts
    (module docstring). Raises NotApplicable."""
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, not {design!r}")
    m = _module(pipe)
    name = m.NAME
    dataset = overlay.meta.get("dataset", rec.meta.get("dataset"))
    contrast = overlay.meta["contrast"]
    if name == "sokoliuk21":
        raise NotApplicable("sokoliuk21: not applicable to N-passive (no isochronous speech)")
    if contrast not in contrasts_for(m, dataset):
        raise NotApplicable(f"{name}: '{contrast}' is not its matching null on {dataset}")
    if not _channels_ok(m, rec):
        raise NotApplicable(f"{name}: {rec.uid} lacks the module's native and 10-20 channel sets")
    key = f"{name}{rec.uid}" + ("" if design == "published" else ":randomized")
    g = mc.gen(np.random.default_rng(SEED + zlib.crc32(key.encode())) if rng is None else rng)
    on = np.asarray(overlay.onsets_s, float)
    keep = _edge_keep(m, rec, on)
    idx = np.nonzero(keep)[0]
    info = {"pipe": name, "contrast": contrast, "design": design, "changed": [], "usable": {},
            "dropped": {"edge": int((~keep).sum())}, "skip_variants": [], "periodic": None,
            "run": design == "published", "randomized": True, "randomized_reason": None, "rand_variant": None}
    if info["dropped"]["edge"]:
        info["changed"].append(f"{info['dropped']['edge']} stimuli within the module's extent margin of the "
                               "recording edges dropped")
    sub = _substitution(name, rec.names)
    if sub:
        info["channel_substitution"] = sub
        info["changed"].append(", ".join(f"{s} used as {t}" for t, s in sub.items())
                               + ": the nearest available sites (3.9 cm); 19-channel layout kept")
    build = {"mbvt": _mbvt, "chennu13": _chennu, "gibson16": _gibson}[name]
    out = build(overlay, idx, contrast, design, g, info)
    if design == "published":
        info["randomized"] = False
        info["randomized_reason"] = "published design: run_randomized takes the design='randomized' overlay"
    elif info["randomized"]:
        info["rand_variant"] = _rand_variant(m, info["skip_variants"])
    return out


def _rand_variant(m, skip):
    """The module's DEFAULT, or the usable variant whose settings differ from DEFAULT's in the fewest keys."""
    if m.DEFAULT not in skip:
        return m.DEFAULT
    ref = {k: v for k, v in m.VARIANTS[m.DEFAULT].items() if k != "cite"}
    ok = [v for v in m.VARIANTS if v not in skip]
    return min(ok, key=lambda v: sum(m.VARIANTS[v].get(k) != x for k, x in ref.items()))


def run(pipe, rec, adapted, variant, rng=None):
    """The module's run() on a published-design overlay (refuses the randomized design and skipped variants)."""
    m = _module(pipe)
    info = adapted.meta["adapter"]
    if not info["run"]:
        raise ValueError(f"{m.NAME}: run() takes the design='published' overlay, not {info['design']!r}")
    if isinstance(variant, str) and variant in info["skip_variants"]:
        raise ValueError(f"{m.NAME}: variant {variant!r} is skipped on this overlay")
    return m.run(recording_for(m, rec), adapted, variant, rng)


def _check_randomized(m, adapted):
    info = adapted.meta["adapter"]
    if info.get("design") != "randomized" or not info["randomized"]:
        why = info.get("randomized_reason") or "not a randomized-design overlay"
        raise ValueError(f"{m.NAME}: run_randomized refused ({why})")
    if m.NAME == "mbvt" and adapted.meta.get("paradigm") == "vt3":
        tw = np.asarray(adapted.meta["target_wrist"])
        per = pm.N_GROUPS // pm.N_INSTR
        blocks = [tw[i:i + per] for i in range(0, len(tw) - per + 1, per)]
        if blocks and all(len(set(b)) == 1 for b in blocks):
            raise ValueError("mbvt: target wrist constant within every instruction block (a per-instruction "
                             "assignment), not exchangeable with run_randomized's per-group re-draw")


def run_randomized(pipe, rec, adapted, rng=None, variant=None):
    """The module's run_randomized() on a randomized-design overlay (refuses anything else), with rand_variant unless
    a variant is given."""
    m = _module(pipe)
    _check_randomized(m, adapted)
    info = adapted.meta["adapter"]
    variant = info["rand_variant"] if variant is None else variant
    if isinstance(variant, str) and variant in info["skip_variants"]:
        raise ValueError(f"{m.NAME}: variant {variant!r} is skipped on this overlay")
    return m.run_randomized(recording_for(m, rec), adapted, variant, rng)



def _mbvt(ov, idx, contrast, design, g, info):
    vt = "vt2" if contrast == "oddball" else "vt3"
    lab, rt = np.asarray(ov.labels)[idx], np.asarray(ov.meta["rare_type"])[idx]
    link = _links(np.asarray(ov.meta["seq"])[idx], np.asarray(ov.meta["train"])[idx])
    if vt == "vt2":
        def valid(a, b):
            return lab[a:b].sum() == 1
    else:
        def valid(a, b):
            return (rt[a:b] == 1).sum() == 1 and (rt[a:b] == 2).sum() == 1 and lab[a:b].sum() == 2
    starts = _windows(link, valid, pm.GROUP)
    G_all = len(starts)
    starts = starts[:pm.N_GROUPS]
    G = len(starts)
    if G < 4:
        raise NotApplicable(f"mbvt: {G} {vt.upper()} groups of 8 consecutive stimuli (the module needs >= 4)")
    sel = np.concatenate([np.arange(s, s + pm.GROUP) for s in starts])
    grp, pos = np.repeat(np.arange(G), pm.GROUP), np.tile(np.arange(pm.GROUP), G)
    l, r = lab[sel], rt[sel]
    per = pm.N_GROUPS // pm.N_INSTR
    if vt == "vt2":
        site = np.where(l == 1, pm.RIGHT, pm.LEFT)
        labels = np.where(l == 1, pm.TARGET, pm.NONTARGET)
        tw = np.full(G, pm.RIGHT)
        info["usable"] = {"n_groups": G, "n_rare": int((l == 1).sum()), "n_standard": int((l == 0).sum())}
        info["changed"].append("VT2: rare = target (right wrist), standards = non-targets (left wrist)")
    else:
        site = np.where(r == 1, pm.LEFT, np.where(r == 2, pm.RIGHT, pm.OTHER))
        if design == "published":
            per_instr = mc.balanced_labels((pm.LEFT, pm.RIGHT), (pm.N_INSTR // 2, pm.N_INSTR // 2), g)
            tw = per_instr[np.arange(G) // per]
            how = "per instruction (15 groups) 2 left / 2 right as schedule() draws it"
        else:
            tw = mc.balanced_labels((pm.LEFT, pm.RIGHT), (G // 2, G - G // 2), g)
            how = f"per group, {G // 2} left / {G - G // 2} right in random order as schedule_randomized() draws it"
        labels = np.where(site == pm.OTHER, pm.DISTRACTOR, np.where(site == tw[grp], pm.TARGET, pm.NONTARGET))
        info["usable"] = {"n_groups": G, "n_target": int((labels == pm.TARGET).sum()),
                          "n_nontarget": int((labels == pm.NONTARGET).sum()),
                          "n_distractor": int((labels == pm.DISTRACTOR).sum())}
        info["changed"].append(f"VT3: rare A = left wrist, rare B = right wrist, standards = distractors; target "
                               f"wrist {how}")
    info["dropped"] |= {"not_in_group": int(len(idx) - pm.GROUP * G_all), "beyond_run": int(pm.GROUP * (G_all - G))}
    info["changed"].append(f"{G} groups of 8 consecutive stimuli (earliest-first, at most {pm.N_GROUPS}); SOA as "
                           "recorded, timing variants mur20 / ann18 moot")
    info["skip_variants"] = sorted(v for v, c in pm.VARIANTS.items() if c["paradigm"] != vt)
    meta = {"group": grp, "pos": pos, "site": site, "block": grp // per, "trial": grp, "run": np.zeros(len(sel), int),
            "target_wrist": tw, "paradigm": vt}
    return _overlay(np.asarray(ov.onsets_s, float)[idx][sel], labels, meta, info)



def _chennu(ov, idx, contrast, design, g, info):
    L = chennu_block_len()
    nb = len(idx) // L
    if nb < 1:
        raise NotApplicable(f"chennu13: {len(idx)} stimuli, fewer than one block of {L}")
    sel = idx[:nb * L]
    blk = np.repeat(np.arange(nb), L)
    lab, rt = np.asarray(ov.labels)[sel], np.asarray(ov.meta["rare_type"])[sel]
    expl = np.asarray(pc._explicit_order(nb, g))
    other = (pc.YES + pc.NO) - expl[blk]
    if contrast == "oddball":
        word = np.where(lab == 1, expl[blk], other)
        labels = np.where(lab == 1, pc.EXPL, pc.IMPL)
        info["changed"].append("explicit = rare (its word = the block's explicit word), implicit = standards, no "
                               "distractors")
    else:
        word = np.where(rt == 1, pc.YES, np.where(rt == 2, pc.NO, 0))
        labels = np.where(word == 0, pc.DIST, np.where(word == expl[blk], pc.EXPL, pc.IMPL))
        info["changed"].append("rare A = YES, rare B = NO, explicit = the block's explicit word, standards = "
                               "distractors")
    trial = np.tile(np.arange(L), nb)
    snum = np.zeros(len(sel), int)
    for i in range(len(sel)):
        snum[i] = 0 if word[i] else (snum[i - 1] if trial[i] else 0) + 1
    info["usable"] = {"n_blocks": nb, "n_explicit": int((labels == pc.EXPL).sum()),
                      "n_implicit": int((labels == pc.IMPL).sum()), "n_distractor": int((labels == pc.DIST).sum())}
    info["dropped"]["partial_block"] = int(len(idx) - nb * L)
    info["changed"].append(f"{nb} blocks of {L} consecutive stimuli (the module's median block length); explicit "
                           "word per block by the module's rule; n_blocks variants moot")
    if nb < 2:
        info["randomized"] = False
        info["randomized_reason"] = "one block: the explicit-word re-draw has no other order"
    meta = {"block": blk, "trial": trial, "run": np.zeros(len(sel), int), "word": word, "snum": snum,
            "explicit_word": expl}
    return _overlay(np.asarray(ov.onsets_s, float)[sel], labels, meta, info)



def _gibson(ov, idx, contrast, design, g, info):
    lab, rt = np.asarray(ov.labels)[idx], np.asarray(ov.meta["rare_type"])[idx]
    on = np.asarray(ov.onsets_s, float)[idx]
    _, blk = np.unique(np.asarray(ov.meta["train"])[idx], return_inverse=True)
    nb = int(blk.max()) + 1 if len(blk) else 0
    if contrast == "oddball":
        site = np.where(lab == 1, pg.RIGHT, pg.UPPER_BACK)
        tw = np.full(nb, pg.RIGHT)
        labels = np.where(lab == 1, pg.TARGET, pg.BACK)
        info["skip_variants"] = [v for v, c in pg.VARIANTS.items() if c["call_on"] == "p3b"]
        info["randomized"] = False
        info["randomized_reason"] = "oddball: no second (non-target) wrist to re-draw"
        info["changed"].append("rare = target wrist (right), standards = back; no non-target wrist: P3b variants "
                               "and run_randomized not applicable")
    else:
        site = np.where(rt == 1, pg.LEFT, np.where(rt == 2, pg.RIGHT, pg.UPPER_BACK))
        if design == "published":
            tw = np.array([pg.RIGHT if b % 2 == 0 else pg.LEFT for b in range(nb)])
            how = "alternating per block (pause-free train) starting right"
        else:
            tw = mc.balanced_labels((pg.RIGHT, pg.LEFT), (nb // 2, nb - nb // 2), g)
            how = f"per block, {nb // 2} right / {nb - nb // 2} left in random order as schedule_randomized() draws it"
        labels = np.where(site == pg.UPPER_BACK, pg.BACK, np.where(site == tw[blk], pg.TARGET, pg.NONTARGET))
        info["changed"].append(f"rare A = left wrist, rare B = right wrist, standards = back; target wrist {how}")
        if nb < 2:
            info["randomized"] = False
            info["randomized_reason"] = "one block: the target-wrist re-draw has no other order"
    link = _links(np.asarray(ov.meta["seq"])[idx], np.asarray(ov.meta["train"])[idx])
    soa = np.diff(on)[link]
    cv = float(soa.std() / soa.mean()) if len(soa) >= PERIODIC_MIN else float("nan")
    info["periodic"] = bool(cv <= PERIODIC_CV)
    if not info["periodic"]:
        info["skip_variants"] += [v for v, c in pg.VARIANTS.items() if c["call_on"] == "ssep"]
        info["changed"].append(f"SSEP variants skipped: recorded SOA not periodic (CV {cv:.3f} > {PERIODIC_CV})")
    info["skip_variants"] = sorted(info["skip_variants"])
    info["usable"] = {"n_target": int((labels == pg.TARGET).sum()), "n_nontarget": int((labels == pg.NONTARGET).sum()),
                      "n_back": int((labels == pg.BACK).sum()), "n_blocks": nb, "soa_cv": cv}
    info["changed"].append("blocks = pause-free trains; SOA as recorded")
    meta = {"block": blk, "site": site, "target_wrist": tw, "trial": np.arange(len(idx)),
            "run": np.zeros(len(idx), int)}
    return _overlay(on, labels, meta, info)



_INSTR_LEN = {}


def instruction_block_len(pipe):
    """Stimuli per block of the module's own design (module docstring)."""
    name = _module(pipe).NAME
    if name not in _INSTR_LEN:
        if name == "mbvt":
            _INSTR_LEN[name] = pm.GROUP * (pm.N_GROUPS // pm.N_INSTR)
        elif name == "chennu13":
            _INSTR_LEN[name] = chennu_block_len()
        else:
            n = [np.bincount(pg.schedule(0.0, np.random.default_rng(s), pg.DEFAULT).meta["block"])
                 for s in range(CHENNU_DRAWS)]
            _INSTR_LEN[name] = int(np.median(np.concatenate(n)))
    return _INSTR_LEN[name]


def _instruction_blocks(train, L):
    """(block id per stimulus, -1 = unused; "module" | "fallback") for stimuli in time order (module docstring)."""
    train = np.asarray(train)
    _, start, size = np.unique(train, return_index=True, return_counts=True)
    out = np.full(len(train), -1)
    b = 0
    if (size // L).sum() >= MIN_BLOCKS:
        for s, n in zip(start, size):
            for j in range(n // L):
                out[s + j * L: s + (j + 1) * L] = b
                b += 1
        return out, "module"
    share = MIN_BLOCKS * size / size.sum()
    k = np.floor(share).astype(int)
    k[np.argsort(-(share - k), kind="stable")[:MIN_BLOCKS - k.sum()]] += 1
    for s, n, kk in zip(start, size, k):
        for part in (np.array_split(np.arange(s, s + n), kk) if kk else []):
            out[part] = b
            b += 1
    return out, "fallback"


def instruction_overlay(pipe, rec, overlay, rng=None):
    """The passive stream in sham-instruction blocks (module docstring): labels 1 rare / 0 standard, meta block,
    instruction (per block), Kojima B sham_stream / analysed_stream. Raises NotApplicable.
    Kojima B: only the really unattended stream of each trial is analysed (the passive overlay holds nothing else);
    a sham count block "attends" that stream and so conflicts with the trial's real attention. Such blocks are kept:
    every analysed stimulus is really unattended whatever the sham label, so the label cannot carry real attention."""
    m = _module(pipe)
    name = m.NAME
    if name == "sokoliuk21":
        raise NotApplicable("sokoliuk21: not applicable to N-passive (no isochronous speech)")
    if not _channels_ok(m, rec):
        raise NotApplicable(f"{name}: {rec.uid} lacks the module's native and 10-20 channel sets")
    dataset = overlay.meta.get("dataset", rec.meta.get("dataset"))
    g = mc.gen(np.random.default_rng(SEED + zlib.crc32(f"{name}{rec.uid}:instruction".encode())) if rng is None
               else rng)
    on = np.asarray(overlay.onsets_s, float)
    keep = _edge_keep(m, rec, on)
    idx = np.nonzero(keep)[0]
    L = instruction_block_len(m)
    block, rule = _instruction_blocks(np.asarray(overlay.meta["train"])[idx], L)
    use, blk = idx[block >= 0], block[block >= 0]
    nb = int(blk.max()) + 1 if len(blk) else 0
    lab = np.asarray(overlay.labels)[use]
    n_rare = np.bincount(blk, weights=lab == 1, minlength=nb).astype(int)
    n_std = np.bincount(blk, weights=lab == 0, minlength=nb).astype(int)
    if nb < MIN_BLOCKS or n_rare.min() < MIN_PER_BLOCK or n_std.min() < MIN_PER_BLOCK:
        raise NotApplicable(f"{name}: {len(idx)} stimuli give {nb} blocks; >= {MIN_BLOCKS} blocks with >= "
                            f"{MIN_PER_BLOCK} rare and standard stimuli each are needed")
    inst = mc.balanced_labels((COUNT, IGNORE), (nb // 2, nb - nb // 2), g)
    info = {"pipe": name, "contrast": overlay.meta.get("contrast"), "design": "instruction", "block_rule": rule,
            "block_len": L if rule == "module" else None, "changed": [], "skip_variants": [], "run": False,
            "randomized": False, "randomized_reason": "instruction overlay: use run_instruction",
            "dropped": {"edge": int((~keep).sum()), "not_in_block": int(len(idx) - len(use))},
            "usable": {"n_blocks": nb, "n_count": int(nb // 2), "n_ignore": int(nb - nb // 2),
                       "rare_per_block": [int(n_rare.min()), int(n_rare.max())],
                       "standard_per_block": [int(n_std.min()), int(n_std.max())]}}
    info["changed"].append(f"{nb} consecutive blocks ("
                           + (f"the module's {L} stimuli" if rule == "module" else f"{MIN_BLOCKS} equal parts")
                           + " inside each train); sham instruction count / ignore per block, balanced, seeded")
    sub = _substitution(name, rec.names)
    if sub:
        info["channel_substitution"] = sub
        info["changed"].append(", ".join(f"{s} used as {t}" for t, s in sub.items())
                               + ": the nearest available sites (3.9 cm); 19-channel layout kept")
    meta = {"dataset": dataset, "block": blk, "instruction": inst, "trial": np.arange(len(use)),
            "run": np.zeros(len(use), int), "train": np.asarray(overlay.meta["train"])[use],
            "rare_type": np.asarray(overlay.meta["rare_type"])[use]}
    if dataset in EQUIPROBABLE_DATASETS:
        code = np.asarray(overlay.meta["code"])[use]
        st = np.array([KOJIMA_STREAM_OF[int(c)] for c in code])
        an = np.array([st[blk == b][0] for b in range(nb)])
        if any(len(set(st[blk == b])) > 1 for b in range(nb)):
            raise ValueError(f"{rec.uid}: a block holds two analysed streams")
        meta |= {"analysed_stream": an, "sham_stream": np.where(inst == COUNT, an, 3 - an)}
        info["usable"]["n_conflict"] = int(np.sum(meta["sham_stream"] == an))
        info["changed"].append("Kojima B: the sham instruction is an attended stream per block (count = the analysed, "
                               "really unattended stream; ignore = the other); conflicting count blocks kept")
    return _overlay(on[use], lab, meta, info)


def _mbvt_epoch_features(rec, ov, feat):
    """pipe_mbvt.features without the grouping: (per-epoch segment-mean features, kept = within +-100 µV)."""
    f = pm.FEATS[feat]
    x, _, _ = ec.pick(rec, pm.NATIVE_CH, pm.SUB_1020)
    fs = float(rec.fs)
    seg, i0 = mc.span(x, fs, *pm.extent(ov))
    seg = mc.bandpass(seg, fs, 0.1, 30.0)
    n_pre, n_post = int(round(0.1 * fs)), int(round(0.6 * fs))
    ep = ec.sample_epochs(seg, fs, np.asarray(ov.onsets_s, float) - i0 / fs, n_pre, n_post)
    ok = np.abs(ep).max(axis=(1, 2)) <= 100.0
    if f["base"]:
        ep = ep - ep[..., :n_pre].mean(-1, keepdims=True)
    lo = n_pre + int(round(f["win"][0] * fs))
    parts = np.array_split(np.arange(lo, ep.shape[-1]), f["n"])
    return np.stack([ep[..., p].mean(-1) for p in parts], -1).reshape(len(ep), -1), ok


def _instr_sums(F, rare):
    """F with per-block sums and counts of the rare (S1, N1) and standard (S0, N0) epochs' values."""
    b, V, nb = F["block"], F["val"], F["nb"]
    S1, S0 = np.zeros((nb, V.shape[1])), np.zeros((nb, V.shape[1]))
    for k in range(nb):
        S1[k] = V[(b == k) & rare].sum(0)
        S0[k] = V[(b == k) & ~rare].sum(0)
    N1, N0 = np.bincount(b[rare], minlength=nb).astype(float), np.bincount(b[~rare], minlength=nb).astype(float)
    return dict(F, rare=rare, S1=S1, S0=S0, N1=N1, N0=N0, usable=(N1 > 0) & (N0 > 0))


def _instr_raw(m, rec, ov, uid, keep_val=True):
    """One run's instruction-free features: per kept epoch the module's values with block and rare; chennu13 /
    gibson16 summed per block at once (keep_val False drops the epochs), mbvt kept as feature vectors (the LDA
    direction is fitted after pooling)."""
    rare, block = np.asarray(ov.labels) == 1, np.asarray(ov.meta["block"])
    F = {"kind": m.NAME, "nb": len(ov.meta["instruction"]), "uid": uid, "patient": rec.patient}
    if m.NAME == "mbvt":
        X, kept = _mbvt_epoch_features(rec, ov, pm.VARIANTS[pm.DEFAULT]["feat"])
        F |= {"X": X[kept], "block": block[kept], "rare": rare[kept], "sig": (tuple(rec.names), float(rec.fs))}
        return F
    if m.NAME == "chennu13":
        Fc = pc.features(rec, ov)
        kept = pc._keep(Fc["ep"], pc.VARIANTS[pc.DEFAULT]["reject"])
        ep = Fc["ep"][kept]
        val = ep.reshape(len(ep), int(np.prod(ep.shape[1:])))
        F |= {"shape": ep.shape[1:], "n0": Fc["n0"], "W": pc.corrwin(Fc["times_ms"], pc.WINDOWS["p3b"])}
    else:
        Fg = pg.features(rec, ov)
        kept = Fg["keep"]
        W = pg._win(Fg)
        X = Fg["ep"][..., W]
        val = X.reshape(len(X), int(np.prod(X.shape[1:])))
        nbr = mc.cfg(pg.VARIANTS, pg.DEFAULT).get("neighbours", "delaunay")
        F["A"] = pg._graph(X.shape[1], X.shape[2], ec.triangulation_neighbours(Fg["names"], nbr))
    F |= {"val": np.asarray(val, float), "block": block[kept], "sig": (tuple(rec.names), float(rec.fs), val.shape[1:])}
    F = _instr_sums(F, rare[kept])
    if not keep_val:
        del F["val"]
    return F


def _instr_pool(m, raws):
    """One subject's (or one run's) features: blocks renumbered run after run; mbvt scores on the LDA direction
    fitted on all pooled epochs."""
    if len({r["sig"] for r in raws}) > 1:
        raise ValueError(f"{m.NAME}: runs differ in channels, sampling rate or epoch shape")
    off = np.cumsum([0] + [r["nb"] for r in raws])
    block = np.concatenate([r["block"] + o for r, o in zip(raws, off)])
    rare = np.concatenate([r["rare"] for r in raws])
    F = {"kind": m.NAME, "nb": int(off[-1]), "block": block, "rare": rare,
         "run_blocks": [(r["uid"], r["nb"]) for r in raws], "patient": raws[0]["patient"], "n_runs": len(raws)}
    if m.NAME == "mbvt":
        X = np.concatenate([r["X"] for r in raws])
        if min(rare.sum(), (~rare).sum()) < MIN_PER_BLOCK:
            raise mc.NoUsableTrials(f"mbvt: {int(rare.sum())} rare and {int((~rare).sum())} standard epochs kept, "
                                    f">= {MIN_PER_BLOCK} of each needed to fit the LDA direction")
        F["val"] = (X @ pm.lda_w(X, rare))[:, None]
        return _instr_sums(F, rare)
    for k in ("S1", "S0"):
        F[k] = np.concatenate([r[k] for r in raws])
    for k in ("N1", "N0"):
        F[k] = np.concatenate([r[k] for r in raws])
    F["usable"] = (F["N1"] > 0) & (F["N0"] > 0)
    F |= {k: raws[0][k] for k in ("shape", "n0", "W", "A") if k in raws[0]}
    if all("val" in r for r in raws):
        F["val"] = np.concatenate([r["val"] for r in raws])
    return F


def _instr_ok(F, c):
    """Admissible labelling (c: True = count block): the minimum kept epochs and usable blocks per condition."""
    for sel in (c, ~c):
        if F["N1"][sel].sum() < MIN_PER_BLOCK or F["N0"][sel].sum() < MIN_PER_BLOCK:
            return False
        if F["usable"][sel].sum() < INSTR_MIN_BLOCKS.get(F["kind"], 0):
            return False
    return True


def _instr_stat(m, F, instruction):
    """Count-minus-ignore contrast of the deviant-minus-standard feature difference (module docstring), over the
    usable blocks; NaN where undefined (check _instr_ok first)."""
    c = np.asarray(instruction) == COUNT
    if F["kind"] == "chennu13":
        def delta(sel):
            mr = (F["S1"][sel].sum(0) / F["N1"][sel].sum()).reshape(F["shape"])
            ms = (F["S0"][sel].sum(0) / F["N0"][sel].sum()).reshape(F["shape"])
            return pc.gfp_rows(mr, F["n0"]) - pc.gfp_rows(ms, F["n0"])
        with np.errstate(invalid="ignore", divide="ignore"):
            return float((delta(c) - delta(~c))[F["W"]].max())
    u = F["usable"]
    d = F["S1"][u] / F["N1"][u, None] - F["S0"][u] / F["N0"][u, None]
    c = c[u]
    if c.sum() < 1 or (~c).sum() < 1:
        return float("nan")
    if F["kind"] == "mbvt":
        return float(d[c, 0].mean() - d[~c, 0].mean())
    a, b = d[c], d[~c]
    if min(len(a), len(b)) < 2:
        return float("nan")
    t = pg.indep_t(a.sum(0), (a * a).sum(0), len(a), b.sum(0), (b * b).sum(0), len(b))
    return pg._max_cluster(t, t > t_dist.ppf(1 - pg.CLUSTER_ALPHA, len(d) - 2), F["A"], 1.0)


def _redraw_instruction(meta, g):
    inst = np.asarray(meta["instruction"])
    n_count = int(np.sum(inst == COUNT))
    return mc.balanced_labels((COUNT, IGNORE), (n_count, len(inst) - n_count), g)


def _admissible_value(m, F, instruction):
    """The statistic when the labelling is admissible and the statistic finite, else None."""
    if not _instr_ok(F, np.asarray(instruction) == COUNT):
        return None
    v = _instr_stat(m, F, instruction)
    return v if np.isfinite(v) else None


def _null_draw(m, F, meta, g, tally=None):
    """One null value: block instructions re-drawn until admissible (motor_common.admissible_stat); the rejected
    re-draws are counted in tally["inadmissible"]."""
    box = {}

    def ok(lab):
        box["v"] = _admissible_value(m, F, lab)
        if box["v"] is None and tally is not None:
            tally["inadmissible"] = tally.get("inadmissible", 0) + 1
        return box["v"] is not None
    return mc.admissible_stat(lambda: _redraw_instruction(meta, g), ok, lambda lab: box["v"])


def _observed(m, F, instruction, what):
    obs = _admissible_value(m, F, instruction)
    if obs is None:
        c = np.asarray(instruction) == COUNT
        raise mc.NoUsableTrials(
            f"{m.NAME}: {what}: the observed instructions leave count {int(F['N1'][c].sum())} rare / "
            f"{int(F['N0'][c].sum())} standard epochs in {int(F['usable'][c].sum())} usable blocks and ignore "
            f"{int(F['N1'][~c].sum())} / {int(F['N0'][~c].sum())} in {int(F['usable'][~c].sum())} (>= "
            f"{MIN_PER_BLOCK} epochs and {INSTR_MIN_BLOCKS.get(m.NAME, 0)} usable blocks each, finite statistic)")
    return obs


def run_instruction(pipe, rec, ov, rng=None, n_perm=INSTR_B, alpha=0.05):
    """The instruction-randomization test on an instruction_overlay (module docstring)."""
    m = _module(pipe)
    if ov.meta.get("adapter", {}).get("design") != "instruction":
        raise ValueError(f"{m.NAME}: run_instruction takes an instruction_overlay")
    g = mc.gen(rng)
    view = recording_for(m, rec)
    F = mc.memo(mc.rec_key(view, np.asarray(ov.onsets_s, float), np.asarray(ov.labels), np.asarray(ov.meta["block"]),
                           "instruction"), lambda: _instr_pool(m, [_instr_raw(m, view, ov, rec.uid)]))
    obs = _observed(m, F, ov.meta["instruction"], rec.uid)
    tally = {}
    t = mc.perm_test(obs, lambda: _null_draw(m, F, ov.meta, g, tally), n_perm, alpha)
    return mc.result(obs, t["p"], t["p"] < alpha, n_perm_run=t["n_run"], stopped_early=t["stopped"],
                     n_blocks=F["nb"], n_usable_blocks=int(F["usable"].sum()), n_rare=int(F["N1"].sum()),
                     n_standard=int(F["N0"].sum()), n_inadmissible_draws=tally.get("inadmissible", 0), size_stat=obs)


def pool_instruction_features(pipe, runs):
    """One subject's pooled features from (rec, instruction_overlay) pairs, consumed one at a time (pass a
    generator so recordings load one at a time). Raises ValueError for another pipeline's or design's overlay, mixed
    patients, mismatched runs or no runs."""
    m = _module(pipe)
    raws = []
    for rec, ov in runs:
        info = ov.meta.get("adapter", {})
        if info.get("design") != "instruction" or info.get("pipe") != m.NAME:
            raise ValueError(f"{m.NAME}: {rec.uid} is not a {m.NAME} instruction_overlay")
        raws.append(_instr_raw(m, recording_for(m, rec), ov, rec.uid, keep_val=False))
    if not raws:
        raise ValueError(f"{m.NAME}: no runs to pool")
    if len({r["patient"] for r in raws}) > 1:
        raise ValueError(f"{m.NAME}: runs of more than one patient")
    return _instr_pool(m, raws)


def pooled_instruction(n_blocks, rng):
    """Balanced count / ignore over all of a subject's blocks: n_blocks // 2 count, in random order."""
    return mc.balanced_labels((COUNT, IGNORE), (n_blocks // 2, n_blocks - n_blocks // 2), mc.gen(rng))


def run_instruction_pooled(pipe, F, rng=None, n_perm=INSTR_B, alpha=0.05):
    """The per-subject (primary) instruction-randomization test on pool_instruction_features output: the observed
    instructions are pooled_instruction(F["nb"], rng), the null re-draws them the same way."""
    m = _module(pipe)
    if F["kind"] != m.NAME:
        raise ValueError(f"{m.NAME}: features of {F['kind']}")
    key = f"{m.NAME}{F['patient']}:pooled"
    g = mc.gen(np.random.default_rng(SEED + zlib.crc32(key.encode())) if rng is None else rng)
    inst = pooled_instruction(F["nb"], g)
    by_run, k = {}, 0
    for uid, nb in F["run_blocks"]:
        by_run[uid] = inst[k:k + nb]
        k += nb
    extra = {"n_blocks": F["nb"], "n_runs": F["n_runs"], "instruction": inst, "instruction_by_run": by_run}
    obs = _observed(m, F, inst, F["patient"])
    tally = {}
    t = mc.perm_test(obs, lambda: _null_draw(m, F, {"instruction": inst}, g, tally), n_perm, alpha)
    return mc.result(obs, t["p"], t["p"] < alpha, n_perm_run=t["n_run"], stopped_early=t["stopped"],
                     n_usable_blocks=int(F["usable"].sum()), n_rare=int(F["N1"].sum()), n_standard=int(F["N0"].sum()),
                     n_inadmissible_draws=tally.get("inadmissible", 0), size_stat=obs, **extra)
