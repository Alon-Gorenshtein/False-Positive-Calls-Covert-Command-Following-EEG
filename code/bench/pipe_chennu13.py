"""Chennu 2013 (NeuroImage Clin 3:450) auditory word counting, P3a / P3b by GFP randomisation. CORE-EXACT: the
statistics re-implemented line by line from the authors' MATLAB code, srivaschennu/P300Words @ 1d716d48 (no licence;
read, not copied or imported): compgfp.m (GFP max-statistic randomisation),
corrclust.m (temporal clusters), calctct.m + corrtct.m (topographic consistency test). Spec: cat. §2.
- Design (cat. §2; paradigm/p300words.m): 20 blocks; per block YES and NO each 13 + round(3 U) times (l. 390,
  paper 13-16), each word's positions advancing by trepint + round(2 jitter U) - jitter (l. 418-433), the sequence
  padded by trepint and filled with distractors (l. 442-457); one word every 900-1100 ms (l. 494, paper); words start
  13 s into the block (prefixwait 6 + fixcrosstime 5 + postfixwait 2, l. 63-65, 476-486); after the last word a 0.5-s
  beep, 1 s and a 10-s break (l. 317-359). The explicit target is YES in half of the blocks, in a random order with
  no 5 identical in a row (l. 174-198). trepint, jitter, groupsize, ontime and offtime live in param_patient.mat, which
  is not in the repository: trepint 6 / jitter 3 (2-8 intervening words, paper) and groupsize 1 are ours.
- Preprocessing (dataimport.m l. 67-72, epochdata.m l. 33-37, compgfp.m l. 76): eegfilt fir1 low-pass 20 Hz then
  high-pass 0.5 Hz (two-pass), epochs -300 to 800 ms around every word, baseline -300 to 0 ms, average reference.
  The variance-based rejection, visual confirmation, ICA and interpolation are manual (cat. §2): 'reject' variant.
- compgfp (trial mode, l. 193-199, 248-269): condition vs 'base' = the same epochs with the window's samples replaced
  by the first len(window) samples (l. 183-185), or condition vs condition; epochs pooled and split by a cumulative
  randperm 1000 times; GFP of each split's mean, GFP baseline-subtracted over samples up to t = 0 (l. 301-307).
- corrclust (l. 13-90, defaults alpha .05, ttesttail 1, clustsize 10 samples): over ALL rows (observed first),
  per-sample t-like value (row - mean) / (std / sqrt(R)); per-sample p = share of rows whose window maximum is >= the
  value; p < .05 samples grouped into contiguous runs; the largest positive sum over runs of >= 10 samples; cluster p
  = share of rows with a sum >= the observed (observed included); significant if p <= .05 (l. 88).
- TCT (calctct l. 202-211, corrtct l. 27-58): channel order permuted per epoch (cumulative) 1000 times; per row the
  number of window samples whose GFP is in the top 5% of rows; p = share of rows with at least the observed count;
  the code applies no threshold to it (plottct.m prints p): significant if p <= .05 here, as corrclust.
- Windows 100-400 ms (P3a) and 400-700 ms (P3b) (batchrun.m l. 23-27; cat. §2).
- Call (cat. §2 "Call rule and multiplicity": no formal rule): P1 met three statistical criteria (P3b cluster vs
  baseline, TCT, explicit > implicit cluster) plus a topographic correlation with the controls' P3b (r = 0.57, no
  threshold, controls' data not available): dropped. Holm across 21 patients is a group step, not per patient.
NATIVE_CH: the 91 HydroCel channels of their 128-channel EGI net are in no benchmark source; the 19 10-20 channels are
used (both the GFP and the TCT depend on the channel count).
The randomization-based counterpart (run_randomized) keeps CH13's max-statistic on the explicit vs implicit GFP
difference (400-700 ms) and draws the null by re-drawing which word is the explicit target in each block (the unit
the design randomizes), balanced with no 5 in a row as in the published draw."""
import numpy as np

from . import erp_common as ec
from . import motor_common as mc

NAME = 'chennu13'
GRADE = 'CORE-EXACT'
FAMILY = 'erp'
NATIVE_CH = list(ec.EEG19)
EXPL, IMPL, DIST = 1, 2, 3
YES, NO = 1, 2
RUNCOUNT_BASE, RUNCOUNT_RANGE = 13, 3
TREPINT, TREPJIT = 6, 3
ON_S, ON_JIT_S, OFF_S = 1.0, 0.1, 0.0
PREFIX_S, FIXCROSS_S, POSTFIX_S = 6.0, 5.0, 2.0
END_S = 0.5 + 1.0 + 10.0
EPOCH = (-0.3, 0.8)
LP_HZ, HP_HZ = 20.0, 0.5
WINDOWS = {'p3a': (100.0, 400.0), 'p3b': (400.0, 700.0)}
NUMRAND, ALPHA, CLUSTSIZE, CLUST_MS = 1000, 0.05, 10, 40.0
ZVAR = 3.0
MIN_EPOCHS = 2
MAX_SAME = 4
RULES = {
    'p1': 'P3b cluster vs baseline AND TCT AND explicit > implicit cluster (P1, cat. §2)',
    'p3b_base': 'P3b cluster vs own baseline',
    'p3b_base_tct': 'P3b cluster vs own baseline AND TCT',
    'p3_any': 'cluster vs own baseline in the P3a or the P3b window ("significant ERP responses", cat. §2)',
}
VARIANTS = {}
for _r, _rc in RULES.items():
    for _cs in ('samples10', 'ms40'):
        for _rej in ('none', 'zvar'):
            for _nb in (20, 12):
                VARIANTS[f'{_r}_{_cs}_{_rej}_b{_nb}'] = {
                    'rule': _r, 'clust': _cs, 'reject': _rej, 'n_blocks': _nb,
                    'as_coded': _r == 'p1' and _cs == 'samples10' and _rej == 'none' and _nb == 20,
                    'cite': f'cat. §2; {_rc}; clustsize 10 samples (corrclust.m l. 6, = 40 ms at 250 Hz) vs 40 ms; '
                            'artefact handling manual (cat. §2); 20 blocks published (~40 min), 12 fit the 1505-s '
                            'window of a 30-min BDSP segment'}
DEFAULT = 'p1_samples10_none_b20'


RUN_S_BY_BLOCKS = {20: 2530.0, 12: 1505.0}


def run_s(variant=DEFAULT):
    return RUN_S_BY_BLOCKS[mc.cfg(VARIANTS, variant)['n_blocks']]


RUN_S = max(RUN_S_BY_BLOCKS.values())



def _mround(v):
    """MATLAB round (half away from zero) for v >= 0."""
    return int(np.floor(v + 0.5))


def _positions(g, count, avoid=()):
    pos, prev = [], 0
    for _ in range(count):
        while True:
            p = prev + TREPINT + _mround(g.random() * TREPJIT * 2) - TREPJIT
            if p not in avoid:
                break
        pos.append(p)
        prev = p
    return pos


def _block_words(g, explicit):
    """One block's word codes: YES / NO / 0 (distractor) in presentation order (setupblock, l. 390-457)."""
    count = RUNCOUNT_BASE + _mround(g.random() * RUNCOUNT_RANGE)
    first = _positions(g, count)
    second = _positions(g, count, set(first))
    seq = np.zeros(max(max(first), max(second)) + TREPINT, int)
    seq[np.array(first) - 1] = explicit
    seq[np.array(second) - 1] = YES + NO - explicit
    return seq


def _explicit_order(n_blocks, g):
    return mc.constrained_order((YES, NO), (n_blocks // 2, n_blocks - n_blocks // 2), MAX_SAME, g)


def schedule(start_s, rng, variant=DEFAULT):
    """Word onsets (labels 1 explicit, 2 implicit, 3 distractor). Draws are repeated until the last epoch ends 9 s
    before start_s + run_s(variant) (rare: the run_s values sit ~3 SD above the mean span)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    while True:
        ov = _draw(start_s, g, c['n_blocks'])
        if ov.onsets_s.max() + EPOCH[1] + 9.0 <= float(start_s) + RUN_S_BY_BLOCKS[c['n_blocks']]:
            return ov


def _draw(start_s, g, nb):
    expl = _explicit_order(nb, g)
    on, lab, blk, word, snum, trial = [], [], [], [], [], []
    t = float(start_s) + 1.0
    for b in range(nb):
        seq = _block_words(g, int(expl[b]))
        tw = t + PREFIX_S + FIXCROSS_S + POSTFIX_S
        k = 0
        for i, w in enumerate(seq):
            on.append(tw)
            word.append(w)
            lab.append(EXPL if w == expl[b] else IMPL if w else DIST)
            k = 0 if w else k + 1
            snum.append(k)
            blk.append(b)
            trial.append(i)
            tw += ON_S + g.random() * ON_JIT_S * 2 - ON_JIT_S + OFF_S
        t = tw + END_S
    n = len(on)
    return ec.Overlay(onsets_s=np.array(on), labels=np.array(lab, int), kind='stim',
                      meta={'block': np.array(blk), 'trial': np.array(trial), 'run': np.zeros(n, int),
                            'word': np.array(word), 'snum': np.array(snum), 'explicit_word': np.array(expl)})


def schedule_randomized(start_s, rng, variant=DEFAULT):
    """The published design already randomizes the explicit word per block (cat. §2; p300words.m l. 174-198)."""
    return schedule(start_s, rng, variant)


def extent(ov):
    return float(ov.onsets_s.min()) + EPOCH[0] - 8.0, float(ov.onsets_s.max()) + EPOCH[1] + 8.0



def features(rec, ov):
    """Label-free: target-word epochs (both YES and NO), baseline-corrected, average-referenced, float64.
    -> dict(ep (n, C, T), sel (indices into the overlay), times_ms, n0 (index of t = 0), which)."""
    x, used, which = ec.pick(rec, NATIVE_CH, NATIVE_CH)
    fs = float(rec.fs)
    seg, i0 = mc.span(x, fs, *extent(ov))
    seg = ec.eegfilt_fir1(seg, fs, 0, LP_HZ)
    seg = ec.eegfilt_fir1(seg, fs, HP_HZ, 0)
    sel = np.nonzero(np.asarray(ov.labels) != DIST)[0]
    n_pre, n_post = int(round(-EPOCH[0] * fs)), int(round(EPOCH[1] * fs))
    ep = ec.sample_epochs(seg, fs, np.asarray(ov.onsets_s, float)[sel] - i0 / fs, n_pre, n_post)
    ep = ep - ep[..., :n_pre + 1].mean(-1, keepdims=True)
    ep = ep - ep.mean(1, keepdims=True)
    times = np.arange(-n_pre, n_post) / fs * 1000.0
    return {'ep': ep, 'sel': sel, 'times_ms': times, 'n0': n_pre, 'which': which}


def _feat(rec, ov):
    """Cached features; the key holds the onsets and which of them are epoched (labels != DIST)."""
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float), np.asarray(ov.labels) != DIST),
                   lambda: features(rec, ov))


def _keep(ep, reject):
    if reject == 'none':
        return np.ones(len(ep), bool)
    v = ep.var(-1).max(1)
    z = (v - v.mean()) / v.std()
    return z <= ZVAR


def corrwin(times_ms, win):
    return np.nonzero((times_ms >= win[0]) & (times_ms <= win[1]))[0]


def gfp_rows(means, n0):
    """calcgfp: eeg_gfp of each (C, T) mean, then rmbase over samples 1:find(times == 0) (t = 0 included)."""
    g = ec.gfp(means)
    return g - g[..., :n0 + 1].mean(-1, keepdims=True)



def compgfp(c1, c2, n0, numrand, g):
    """gfpdiff (numrand + 1, T): row 0 observed, then cumulative randperm splits of the pooled epochs (l. 256-269)."""
    merged = np.concatenate([c1, c2])
    N, n1 = len(merged), len(c1)
    shape = merged.shape[1:]
    Xf = merged.reshape(N, -1)
    tot = Xf.sum(0)
    state = np.arange(N)
    M = np.zeros((numrand + 1, N))
    M[0, :n1] = 1.0
    for n in range(1, numrand + 1):
        state = state[g.permutation(N)]
        M[n, state[:n1]] = 1.0
    S1 = M @ Xf
    m1 = (S1 / n1).reshape((-1,) + shape)
    m2 = ((tot[None] - S1) / (N - n1)).reshape((-1,) + shape)
    return gfp_rows(m1, n0) - gfp_rows(m2, n0)


def base_condition(c1, win_idx):
    """compgfp.m l. 183-185: the window's samples replaced by the epoch's first len(window) samples."""
    c2 = c1.copy()
    c2[..., win_idx] = c1[..., :len(win_idx)]
    return c2


def _runs_best(mask, valu, minlen):
    """corrclust.m l. 56-80 per row: the largest positive sum of valu over runs of mask (scan p = 2..T, a run ending
    at the last sample excludes it) with length >= minlen; 0 when none. mask, valu: (R, T)."""
    R, T = mask.shape
    m = mask.copy()
    m[:, -1] = False
    m[:, 0] = False
    start = m & ~np.c_[np.zeros((R, 1), bool), m[:, :-1]]
    rid = np.cumsum(start, 1) * m
    nrun = int(rid.max()) if rid.size else 0
    if nrun == 0:
        return np.zeros(R)
    key = (np.arange(R)[:, None] * (nrun + 1) + rid)[m]
    sums = np.bincount(key, weights=valu[m], minlength=R * (nrun + 1)).reshape(R, nrun + 1)
    lens = np.bincount(key, minlength=R * (nrun + 1)).reshape(R, nrun + 1)
    ok = (lens >= minlen) & (sums > 0)
    ok[:, 0] = False
    return np.where(ok, sums, 0.0).max(1)


def corrclust(gfpdiff, win_idx, alpha=ALPHA, clustsize=CLUSTSIZE):
    """-> dict(tstat (observed largest cluster sum), prob, best (per row), win (sample indices) or None)."""
    R, T = gfpdiff.shape
    W = np.asarray(win_idx)
    pdist = gfpdiff[:, W].max(1)
    G = gfpdiff[:, W]
    valu = np.zeros((R, T))
    valu[:, W] = (G - G.mean(0)) / (G.std(0, ddof=1) / np.sqrt(R))
    sp = np.sort(pdist)
    pprob = np.ones((R, T))
    pprob[:, W] = (R - np.searchsorted(sp, G, side='left')) / R
    best = _runs_best(pprob < alpha, valu, clustsize)
    prob = float(np.sum(best >= best[0]) / R)
    return {'tstat': float(best[0]), 'prob': prob, 'best': best, 'sig': bool(best[0] > 0 and prob <= alpha),
            'obs_max': float(G[0].max())}



def tct(c1, win_idx, n0, numrand, g, alpha=ALPHA):
    """calctct (trial mode) + corrtct: -> dict(prob, hits (per row), sig)."""
    n, C, T = c1.shape
    Xr = c1.reshape(n * C, T)
    idx = np.tile(np.arange(C), (n, 1))
    base = (np.arange(n) * C)[:, None]
    rows = np.empty((numrand + 1, T))
    rows[0] = gfp_rows(c1.mean(0), n0)
    for b in range(1, numrand + 1):
        perm = np.argsort(g.random((n, C)), axis=1)
        idx = np.take_along_axis(idx, perm, axis=1)
        rows[b] = gfp_rows(Xr[(base + idx).ravel()].reshape(n, C, T).mean(0), n0)
    W = np.asarray(win_idx)
    G = rows[:, W]
    R = numrand + 1
    srt = np.sort(G, axis=0)
    ge = R - np.stack([np.searchsorted(srt[:, j], G[:, j], side='left') for j in range(G.shape[1])], 1)
    hits = np.sum(ge / R < alpha, axis=1)
    prob = float(np.sum(hits[0] <= hits) / R)
    return {'prob': prob, 'hits': hits, 'sig': bool(prob <= alpha)}



def _clustsize(c, fs):
    return CLUSTSIZE if c['clust'] == 'samples10' else max(1, int(round(CLUST_MS / 1000.0 * fs)))


def _conditions(F, ov, keep):
    lab = np.asarray(ov.labels)[F['sel']]
    return F['ep'][keep & (lab == EXPL)], F['ep'][keep & (lab == IMPL)]


def run(rec, overlay, variant=DEFAULT, rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay)
    keep = _keep(F['ep'], c['reject'])
    ex, im = _conditions(F, overlay, keep)
    fs = float(rec.fs)
    cs, n0, tm = _clustsize(c, fs), F['n0'], F['times_ms']
    for what, e in (('explicit', ex),) + ((('implicit', im),) if c['rule'] == 'p1' else ()):
        if len(e) < MIN_EPOCHS:
            raise mc.NoUsableTrials(f'chennu13: {len(e)} {what}-word epochs survive rejection (each condition the '
                                    f'rule reads needs >= {MIN_EPOCHS})')
    out = {}
    wins = ('p3a', 'p3b') if c['rule'] == 'p3_any' else ('p3b',)
    for w in wins:
        W = corrwin(tm, WINDOWS[w])
        out[f'base_{w}'] = corrclust(compgfp(ex, base_condition(ex, W), n0, NUMRAND, g), W, clustsize=cs)
    Wb = corrwin(tm, WINDOWS['p3b'])
    if c['rule'] in ('p1', 'p3b_base_tct'):
        out['tct_p3b'] = tct(ex, Wb, n0, NUMRAND, g)
    if c['rule'] == 'p1':
        out['ei_p3b'] = corrclust(compgfp(ex, im, n0, NUMRAND, g), Wb, clustsize=cs)
    parts = [k for k in out]
    if c['rule'] == 'p3_any':
        call = out['base_p3a']['sig'] or out['base_p3b']['sig']
        pk = min(out['base_p3a']['prob'], out['base_p3b']['prob'])
        cont = max(out['base_p3a']['obs_max'], out['base_p3b']['obs_max'])
    else:
        call = all(out[k]['sig'] for k in parts)
        pk = max(out[k]['prob'] for k in parts)
        cont = out['base_p3b']['obs_max']
        if c['rule'] == 'p1':
            cont = min(cont, out['ei_p3b']['obs_max'])
    extra = {k: {'prob': v['prob'], 'sig': v['sig'], 'tstat': v.get('tstat'), 'obs_max': v.get('obs_max')}
             for k, v in out.items()}
    return mc.result(out['base_p3b']['tstat'], pk, call, size_stat=1.0 - pk, size_stat_cont=cont, n_explicit=len(ex),
                     n_implicit=len(im), clustsize=cs, channels=F['which'], parts=extra)


def run_randomized(rec, overlay, variant=DEFAULT, rng=None):
    """CH13's own max-statistic (cat. §2: each sample's GFP difference "compared to the maximum GFP differences
    obtained within that time window"): T = the 400-700 ms maximum of GFP(explicit mean) - GFP(implicit mean). The
    null re-draws which word is explicit in each block (balanced, no 5 in a row: the published draw), B = 500 with
    decision-exact early stopping. corrclust's cluster p is not used here: most rows have no cluster (sum 0), and the
    ties make it conservative (0 / 200 false calls on noise in the tests). After rejection a re-drawn order can leave
    fewer than MIN_EPOCHS explicit or implicit epochs: it is re-drawn (mc.admissible_stat, a conditional
    randomization test), never scored."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay)
    keep = _keep(F['ep'], c['reject'])
    meta = overlay.meta
    blk = np.asarray(meta['block'])[F['sel']][keep]
    word = np.asarray(meta['word'])[F['sel']][keep]
    expl = np.asarray(meta['explicit_word'])
    ep = F['ep'][keep]
    nb = len(expl)
    shape = ep.shape[1:]
    S = {w: np.stack([ep[(blk == b) & (word == w)].sum(0).ravel() for b in range(nb)]) for w in (YES, NO)}
    N = {w: np.array([np.sum((blk == b) & (word == w)) for b in range(nb)], float) for w in (YES, NO)}
    W = corrwin(F['times_ms'], WINDOWS['p3b'])

    def counts(a):
        a = np.asarray(a, float)
        return a @ N[YES] + (1 - a) @ N[NO], a @ N[NO] + (1 - a) @ N[YES]

    def ok(a):
        return min(counts(a)) >= MIN_EPOCHS

    def stat(a):
        ne, ni = counts(a)
        a = np.asarray(a, float)
        me = ((a @ S[YES] + (1 - a) @ S[NO]) / ne).reshape(shape)
        mi = ((a @ S[NO] + (1 - a) @ S[YES]) / ni).reshape(shape)
        return float((gfp_rows(me, F['n0']) - gfp_rows(mi, F['n0']))[W].max())

    a_obs = expl == YES
    if not ok(a_obs):
        ne, ni = counts(a_obs)
        raise mc.NoUsableTrials(f'chennu13: {int(ne)} explicit and {int(ni)} implicit epochs survive rejection (each '
                                f'needs >= {MIN_EPOCHS})')
    obs = stat(a_obs)
    t = mc.perm_test(obs, lambda: mc.admissible_stat(lambda: _explicit_order(nb, g) == YES, ok, stat),
                     c.get('n_perm_rand', 500), ALPHA)
    return mc.result(obs, t['p'], t['p'] < ALPHA, n_perm_run=t['n_run'], stopped_early=t['stopped'], size_stat=obs,
                     size_stat_cont=obs)


def stat_for_size_correction(result):
    """size_stat_cont: a continuous, data-only GFP max-statistic in µV (no ties at a p floor). run():
    the 400-700 ms maximum of the observed explicit-vs-own-baseline GFP difference (p3b_base, p3b_base_tct; the TCT
    has no µV scale and is not in it), the larger of the P3a and P3b windows' maxima (p3_any), the smaller of the
    baseline and the explicit-vs-implicit maxima (p1, both criteria must hold). run_randomized(): its statistic.
    Falls back to the earlier size_stat (1 - p) for results that predate it."""
    e = result['extra']
    return float(e.get('size_stat_cont', e.get('size_stat', result['stat'])))


def patient_rule(results):
    """CH13 reports one session per patient (cat. §2); for k > 1 sessions 'any' (NOT STATED). Holm across patients
    (P1 only, cat. §2) is a cohort-level step and is not applied per patient."""
    return any(r['call'] for r in results)
