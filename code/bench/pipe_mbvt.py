"""mindBEAGLE-type vibrotactile P300, VT2 and VT3 (g.tec; Guger 2017/2018, Annen 2018, Murovec 2020), AS DESCRIBED.
The system is commercial and no author code exists (cat. §4f). Spec: the erp catalogue §4 (cat. §4).
- Design (cat. §4a, §4b): sequences ("groups") of 8 stimuli in random order with one target; 480 stimuli per run =
  60 groups = 4 instructions x 15 groups (GUG17 "15 targets, 7*15 non-targets" per instruction; MUR20 "480 stimuli
  per run, with 60 groups of 8"). VT2: 1 rare wrist (target, 1/8) vs the frequent wrist (7/8), an oddball (ANN18:
  "VT2 7/8 left, 1/8 right", target always the right wrist). VT3: per group one left-wrist and one right-wrist
  stimulus and 6 distractors (75 / 12.5 / 12.5); the instructed wrist is the target, the other wrist and the
  distractors are the 7 non-targets; the target wrist is pseudo-random per instruction, 2 left and 2 right (GUG18).
- Timing (cat. §4a, the papers disagree): 'mur20' = 100-ms pulse + 100-ms pause (SOA 200 ms), the run ~2.5 min, so
  4 instruction pauses of (150 - 96) / 4 = 13.5 s [derived]; 'ann18' = 30-ms pulse + 270-ms ISI (SOA 300 ms),
  "480 stimuli ... 2.4 min" = 144 s, so no pause [derived]. Jitter NOT STATED (none).
- Signal (cat. §4a): 0.1-30 Hz (ANN18: 4th-order Butterworth; zero-phase here, NOT STATED); 8 channels Fz C3 Cz C4
  CP1 CPz CP2 Pz (GUG17/18; MUR20 FCz for Fz): 10-20 data have Fz C3 Cz C4 P3 Pz P4 (CP1 -> P3, CP2 -> P4, CPz ->
  Pz and FCz -> Fz, duplicates dropped: 7 channels). Artefacts: +-100 µV (GUG17/18, MUR20), then "re-trim to 1:7"
  (MUR20): a group is dropped when any of its 8 epochs is rejected (ours, keeps 1:7 exactly).
- Features (cat. §4a; variants): 'seg8' = -100 to 600 ms "sub-divided into 8 data segments of equal duration"
  (GUG18), segment means, no baseline (NOT STATED); 'ds7' = 0-600 ms "down sampled to 7 samples" (ANN18; 7
  equal-segment means), baseline -100 to 0 ms. Fisher LDA "without any further optimization" (MUR20).
- Accuracy (MUR20, cat. §4b): groups randomly split into two equal pools; LDA trained on one pool's single trials;
  tested on 1, 2, ... k averaged groups of the other pool (the target's average vs the 7 non-target averages, the
  j-th non-target of each group averaged with the j-th of the others, ours); 1 if the target's LDA score is the
  largest; 10 random splits averaged -> a curve of multiples of 10%. ANN18's "10-fold cross-validation" is read as
  these 10 splits (the same software). The reported point is NOT STATED (variants: the full test pool, the curve's
  maximum, its median).
- Threshold (cat. §4c): "> 23%" (GUG18; the binomial n behind it NOT STATED) or, as variants, the one-sided 5%
  binomial critical accuracy at p0 = 1/8 for n = 10 / 20 / 30 / 60 (40 / 30 / 26.7 / 21.7%) [derived].
- Patient composite (cat. §4d; patient_rule's rule): '>= 1 run' of VT2 or VT3 above threshold (GUG18), 'best run'
  (the best run above threshold; identical for a threshold rule), 'best60' (MUR20: "at least 60% accuracy at least
  once"), ANN18's VT2 > 70% gate before VT3 (VT3 judged by > 70% or by the variant's threshold). No multiplicity
  correction is stated anywhere (cat. §4d).
- The per-sample Kruskal-Wallis ERP display (GUG18) has no stated per-patient rule: not a call here.
The randomization-based counterpart (run_randomized): the module's features, split-halves and LDA, with the LDA's
continuous target-minus-non-target score margin as statistic (the accuracy saturates and ties); the null re-draws
the randomized element per group: VT2 the target's position within the group (the published within-group order is
random), VT3 which wrist is the target in each group (schedule_randomized draws it per group, balanced 30 / 30);
B = 500 with decision-exact early stopping."""
import numpy as np
from scipy.stats import binom

from . import erp_common as ec
from . import motor_common as mc

NAME = 'mbvt'
GRADE = 'AS DESCRIBED'
FAMILY = 'erp'
NATIVE_CH = ['Fz', 'C3', 'Cz', 'C4', 'CP1', 'CPz', 'CP2', 'Pz']
SUB_1020 = ['Fz', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
TARGET, NONTARGET, DISTRACTOR = 1, 2, 3
LEFT, RIGHT, OTHER = 1, 2, 0
GROUP, N_GROUPS, N_INSTR = 8, 60, 4
TIMINGS = {'mur20': {'soa': 0.2, 'instr': 13.5, 'cite': 'MUR20 100 ms on + 100 ms off, ~2.5 min [derived pause]'},
           'ann18': {'soa': 0.3, 'instr': 0.0, 'cite': 'ANN18 30 ms + 270 ms ISI, 480 stimuli in 2.4 min'}}
FEATS = {'seg8': {'win': (-0.1, 0.6), 'n': 8, 'base': False, 'cite': 'GUG18 -100-600 ms, 8 equal segments'},
         'ds7': {'win': (0.0, 0.6), 'n': 7, 'base': True, 'cite': 'ANN18 0-600 ms down-sampled to 7, baseline -100'}}
SUMMARIES = ('full', 'max', 'median')
P0 = 1.0 / GROUP
MIN_GROUPS = 4


def binom_crit(n, alpha=0.05, p0=P0):
    """Smallest accuracy k / n with P(X >= k) <= alpha, X ~ Bin(n, p0) [derived, cat. §4c]."""
    k = int(np.nonzero(binom.sf(np.arange(n + 1) - 1, n, p0) <= alpha)[0][0])
    return k / n


THRESHOLDS = {'gt23': ('gt', 0.23)}
for _n in (10, 20, 30, 60):
    THRESHOLDS[f'binom_n{_n}'] = ('ge', binom_crit(_n))
VARIANTS = {}
for _p in ('vt2', 'vt3'):
    for _tm, _tc in TIMINGS.items():
        for _ft, _fc in FEATS.items():
            for _sm in SUMMARIES:
                for _th, (_op, _val) in THRESHOLDS.items():
                    VARIANTS[f'{_p}_{_tm}_{_ft}_{_sm}_{_th}'] = {
                        'paradigm': _p, 'timing': _tm, 'feat': _ft, 'summary': _sm, 'thr': _th, 'op': _op,
                        'thr_value': _val, 'reps': 10,
                        'cite': f"cat. §4; {_tc['cite']}; {_fc['cite']}; reported curve point NOT STATED; "
                                f"n behind '> 23%' NOT STATED ({_th})"}
DEFAULT = 'vt3_mur20_seg8_full_gt23'
LEAD_S, TAIL_S = 3.0, 4.0
RUN_S = max(LEAD_S + N_INSTR * t['instr'] + N_GROUPS * GROUP * t['soa'] for t in TIMINGS.values()) + TAIL_S
PATIENT_RULES = {
    'ge1_run': 'GUG18: VT2 or VT3 above threshold in >= 1 run',
    'best_run': 'best run above threshold (MUR20 "In the best run"); = ge1_run for a threshold rule',
    'best60': 'MUR20: at least 60% accuracy at least once',
    'ann18_gate70': 'ANN18: some VT2 run > 70%, then some VT3 run > 70%',
    'ann18_gate_thr': 'ANN18: some VT2 run > 70%, then some VT3 run above the variant threshold',
}


def run_s(variant=DEFAULT):
    t = TIMINGS[mc.cfg(VARIANTS, variant)['timing']]
    return LEAD_S + N_INSTR * t['instr'] + N_GROUPS * GROUP * t['soa'] + TAIL_S



def _build(start_s, g, c, target_wrist_per_group):
    t = TIMINGS[c['timing']]
    on, lab, site, grp, pos = [], [], [], [], []
    tm = float(start_s) + LEAD_S
    for gi in range(N_GROUPS):
        if gi % (N_GROUPS // N_INSTR) == 0:
            tm += t['instr']
        if c['paradigm'] == 'vt2':
            order = g.permutation(np.r_[RIGHT, np.full(GROUP - 1, LEFT)])
        else:
            order = g.permutation(np.r_[LEFT, RIGHT, np.full(GROUP - 2, OTHER)])
        for k, s in enumerate(order):
            on.append(tm)
            site.append(int(s))
            if c['paradigm'] == 'vt2':
                lab.append(TARGET if s == RIGHT else NONTARGET)
            else:
                lab.append(TARGET if s == target_wrist_per_group[gi] else DISTRACTOR if s == OTHER else NONTARGET)
            grp.append(gi)
            pos.append(k)
            tm += t['soa']
    n = len(on)
    return ec.Overlay(onsets_s=np.array(on), labels=np.array(lab, int), kind='stim',
                      meta={'group': np.array(grp), 'pos': np.array(pos), 'site': np.array(site),
                            'block': np.array(grp) // (N_GROUPS // N_INSTR), 'trial': np.array(grp),
                            'run': np.zeros(n, int), 'target_wrist': np.asarray(target_wrist_per_group),
                            'paradigm': c['paradigm']})


def schedule(start_s, rng, variant=DEFAULT):
    """VT3: the target wrist per instruction, 2 left and 2 right in random order (GUG18 "pseudo-randomly ... equal
    number of left and right targets"). VT2: the rare (right) wrist is always the target (ANN18)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    per_instr = mc.balanced_labels((LEFT, RIGHT), (N_INSTR // 2, N_INSTR // 2), g)
    tw = np.repeat(per_instr, N_GROUPS // N_INSTR) if c['paradigm'] == 'vt3' else np.full(N_GROUPS, RIGHT)
    return _build(start_s, g, c, tw)


def schedule_randomized(start_s, rng, variant=DEFAULT):
    """Same timing. VT2: as published (the within-group order is already random). VT3: the target wrist drawn per
    group, balanced 30 / 30 (complete randomization)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    if c['paradigm'] == 'vt2':
        return schedule(start_s, g, variant)
    tw = mc.balanced_labels((LEFT, RIGHT), (N_GROUPS // 2, N_GROUPS // 2), g)
    return _build(start_s, g, c, tw)


def extent(ov):
    return float(ov.onsets_s.min()) - 2.5, float(ov.onsets_s.max()) + 0.6 + 2.5



def features(rec, ov, feat):
    """-> dict(E (G, 8, d) per-group features in presentation order, groups kept, site (G, 8), which)."""
    f = FEATS[feat]
    x, used, which = ec.pick(rec, NATIVE_CH, SUB_1020)
    fs = float(rec.fs)
    seg, i0 = mc.span(x, fs, *extent(ov))
    seg = mc.bandpass(seg, fs, 0.1, 30.0)
    n_pre, n_post = int(round(0.1 * fs)), int(round(0.6 * fs))
    ep = ec.sample_epochs(seg, fs, np.asarray(ov.onsets_s, float) - i0 / fs, n_pre, n_post)
    bad = np.abs(ep).max(axis=(1, 2)) > 100.0
    if f['base']:
        ep = ep - ep[..., :n_pre].mean(-1, keepdims=True)
    lo = n_pre + int(round(f['win'][0] * fs))
    parts = np.array_split(np.arange(lo, ep.shape[-1]), f['n'])
    X = np.stack([ep[..., p].mean(-1) for p in parts], -1).reshape(len(ep), -1)
    grp, pos = np.asarray(ov.meta['group']), np.asarray(ov.meta['pos'])
    G = grp.max() + 1
    order = np.lexsort((pos, grp))
    E = X[order].reshape(G, GROUP, -1)
    keep = ~bad[order].reshape(G, GROUP).any(1)
    site = np.asarray(ov.meta['site'])[order].reshape(G, GROUP)
    return {'E': E[keep], 'keep': keep, 'site': site[keep], 'which': which}


def _feat(rec, ov, feat):
    """Cached features; the key holds everything features() reads from the overlay: onsets, group, pos, site
."""
    key = mc.rec_key(rec, np.asarray(ov.onsets_s, float), feat,
                     *(np.asarray(ov.meta[k]) for k in ('group', 'pos', 'site')))
    return mc.memo(key, lambda: features(rec, ov, feat))


def _tpos(ov, keep):
    """Target position within each kept group."""
    lab = np.asarray(ov.labels)
    grp, pos = np.asarray(ov.meta['group']), np.asarray(ov.meta['pos'])
    t = np.full(grp.max() + 1, -1)
    t[grp[lab == TARGET]] = pos[lab == TARGET]
    return t[keep]


def lda_w(X, y):
    """Fisher LDA direction (pooled within-class covariance, pseudo-inverse)."""
    m1, m0 = X[y].mean(0), X[~y].mean(0)
    R = np.r_[X[y] - m1, X[~y] - m0]
    return np.linalg.pinv(R.T @ R / (len(X) - 2)) @ (m1 - m0)


def split_eval(E, tpos, g, reps=10):
    """MUR20 (cat. §4b) over `reps` random group split-halves (LDA trained on one pool's single trials, scored on the
    other's groups). -> (curve, margin): curve (n_test) = the mean 8-way correctness after averaging 1..n_test test
    groups; margin = the mean over splits of the test groups' mean target score minus the mean of their 7 non-target
    scores (continuous: no ceiling, no ties). Both come from the same splits (one permutation per split)."""
    G, P, d = E.shape
    h = G // 2
    slots = np.arange(P)
    curve, margin = np.zeros(h), 0.0
    for _ in range(reps):
        perm = g.permutation(G)
        tr, te = perm[:h], perm[h:2 * h]
        y = (slots[None, :] == tpos[tr][:, None]).ravel()
        w = lda_w(E[tr].reshape(-1, d), y)
        s = E[te] @ w
        is_t = slots[None, :] == tpos[te][:, None]
        st = s[is_t]
        sn = s[~is_t].reshape(h, P - 1)
        k = np.arange(1, h + 1)
        ct = np.cumsum(st) / k
        cn = np.cumsum(sn, 0) / k[:, None]
        curve += ct > cn.max(1)
        margin += float(np.mean(st - sn.mean(1)))
    return curve / reps, margin / reps


def accuracy_curve(E, tpos, g, reps=10):
    return split_eval(E, tpos, g, reps)[0]


def score_margin(E, tpos, g, reps=10):
    return split_eval(E, tpos, g, reps)[1]


def summarize(curve, how):
    return float(curve[-1] if how == 'full' else curve.max() if how == 'max' else np.median(curve))


def passes(acc, c):
    return acc > c['thr_value'] if c['op'] == 'gt' else acc >= c['thr_value'] - 1e-12


def _require(F):
    n = int(F['keep'].sum())
    if n < MIN_GROUPS:
        raise mc.NoUsableTrials(f'mbvt: {n} of {len(F["keep"])} groups survive rejection (the split-halves need >= '
                                f'{MIN_GROUPS})')


def run(rec, overlay, variant=DEFAULT, rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay, c['feat'])
    _require(F)
    tpos = _tpos(overlay, F['keep'])
    curve, margin = split_eval(F['E'], tpos, g, c['reps'])
    acc = summarize(curve, c['summary'])
    return mc.result(acc, None, passes(acc, c), paradigm=c['paradigm'], acc=acc, thr=c['thr_value'], op=c['op'],
                     curve=curve, n_groups=len(tpos), channels=F['which'], size_stat=acc, size_stat_cont=margin)


def _redraw(F, c, g, ov):
    """Target positions under the design's randomization, over the overlay's own groups (a real overlay may carry
    fewer than N_GROUPS): VT2 uniform within the group; VT3 the left or the right wrist of each group, a complete
    randomization keeping the overlay's number of left-target groups (schedule_randomized draws it 30 / 30)."""
    G = len(F['E'])
    if c['paradigm'] == 'vt2':
        return g.integers(0, GROUP, G)
    tw = np.asarray(ov.meta['target_wrist'])
    n_groups = int(np.asarray(ov.meta['group']).max()) + 1
    if len(tw) != n_groups:
        raise ValueError(f'target_wrist has {len(tw)} entries for {n_groups} groups')
    n_left = int(np.sum(tw == LEFT))
    left = mc.balanced_labels((LEFT, RIGHT), (n_left, n_groups - n_left), g)[F['keep']]
    return np.argmax(F['site'] == left[:, None], axis=1)


def run_randomized(rec, overlay, variant=DEFAULT, rng=None):
    """The module's features, split-halves and LDA; statistic = score_margin (the 8-way accuracy is a multiple of
    10% that saturates at 100% and ties under the re-draws, which leaves the test without power for clear
    responders); the null re-draws the per-group randomized element (_redraw), B = 500 with decision-exact early
    stopping. Every kept group holds exactly one target under any re-draw, so no re-draw falls below MIN_GROUPS."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay, c['feat'])
    _require(F)
    tpos = _tpos(overlay, F['keep'])
    obs = score_margin(F['E'], tpos, g, c['reps'])
    t = mc.perm_test(obs, lambda: score_margin(F['E'], _redraw(F, c, g, overlay), g, c['reps']),
                     c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     paradigm=c['paradigm'], size_stat=obs, size_stat_cont=obs)


def stat_for_size_correction(result):
    """size_stat_cont: the LDA score margin of run()'s own split-halves (split_eval; the accuracy is a
    multiple of 1 / reps and ties), or run_randomized()'s statistic (the same margin). Falls back to the earlier
    size_stat (the accuracy)."""
    e = result['extra']
    return float(e.get('size_stat_cont', e.get('size_stat', result['stat'])))


def patient_rule(results, rule='ge1_run'):
    """results: run() outputs of one patient's runs (VT2 and/or VT3), in time order. Rules: PATIENT_RULES."""
    if rule in ('ge1_run', 'best_run'):
        return any(r['call'] for r in results)
    acc = [(r['extra']['paradigm'], r['extra']['acc'], r['call']) for r in results]
    if rule == 'best60':
        return any(a >= 0.6 - 1e-12 for _, a, _ in acc)
    gate = any(p == 'vt2' and a > 0.7 for p, a, _ in acc)
    if rule == 'ann18_gate70':
        return gate and any(p == 'vt3' and a > 0.7 for p, a, _ in acc)
    if rule == 'ann18_gate_thr':
        return gate and any(p == 'vt3' and call for p, _, call in acc)
    raise ValueError(f'unknown rule {rule!r}')
