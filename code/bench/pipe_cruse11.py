"""Cruse et al. 2011 Lancet hand-vs-toe imagery (and its Cruse 2012 Neurology variant), AS DESCRIBED, plus the
Goldfine 2013 block-aware reanalysis. Spec: the motor catalogue §1-2.
- Design (§1.1): blocks of 15 tones, one imagery class per block (hand or toe); n blocks of each class (4-8 in
  patients, 6 in controls), pseudorandom block order with no more than two equal blocks in a row; each block opens
  with a spoken instruction and the tones start 5 s later; ISI 4.5-9.5 s (CR12N §1.7: 3-6.5 s, uniform); a relax
  instruction closes the block; 1-2 min break between blocks.
- Signal (§1.2-1.3): 1-40 Hz; Laplacian = each channel minus the average of its four geodesically nearest neighbours;
  epochs -1.5 to 4 s around each tone; trials with large artefacts excluded; motor-strip electrodes; down-sampled to
  100 Hz; log power at every time point in 7-13, 13-19, 19-25, 25-30 Hz; all values in 0.5-3.5 s after the tone
  (CR12N: 0.5-2.5 s) concatenated into one vector per trial.
- Classifier (§1.3): linear SVM, z-scored with the training-set mean and SD in each fold; CV leaves out one hand and
  one toe block at a time, the pairs being neighbouring blocks (hand block i with toe block i, GF13 §2.1).
- Statistic (§1.4-1.5): all single-trial CV outcomes pooled as one binomial; Clopper-Pearson 99% CI (binofit) must
  exclude 50% (positives above chance only); CR11 also requires the action accuracy to lie above the binomial CI of the
  same analysis on a 500-ms pre-tone baseline window. CR12N: the baseline check is NOT STATED, so it is not applied.
- GF13 (§2.1): accuracy over ALL hand-block x toe-block test pairs; significance by a permutation that re-labels whole
  blocks (exhaustive balanced relabelling; |acc - 50%| two-sided, which reproduces the stated minimum p = 1/35 with
  4 + 4 blocks [derived in §2.1]); call p <= 0.05. Its FDR across subjects is a cohort-level step, not a per-patient
  rule, and is left to the analysis (p is returned).
Multiverse (VARIANTS): method {cr11, cr12n, gf13} x blocks per class {4, 6} (§1.1: patients 4-8, median 114 trials ~
4 per class [derived]; controls 6) x band-power estimator {hilbert, sq250} (§1.3 NOT STATED) and, for cr11, the
baseline-check CI {99%, 99.9%} (§1.4 NOT STATED). Other NOT STATED choices are fixed in FIXED."""
import itertools
import math

import numpy as np
from scipy.signal import hilbert
from scipy.stats import binomtest
from sklearn.svm import SVC

from . import motor_common as mc

NAME = 'cruse11'
GRADE = 'AS DESCRIBED'
FAMILY = 'cmd'
SUB_1020 = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
NATIVE_CH = list(SUB_1020)
HAND, TOE = 1, 2
TONES = 15
INSTR_S, RELAX_S, BREAK_S = 5.0, 2.0, (60.0, 120.0)
EPOCH = (-1.5, 4.0)
FS = 100.0
BANDS = ((7, 13), (13, 19), (19, 25), (25, 30))
BASE_WIN = (-0.5, 0.0)
GF13_MAX_LABELINGS = 1000
MIN_PER_CLASS, MIN_FOLDS, MIN_GF13_BLOCKS = 2, 2, 2

FIXED = {
    'channels': '9 sites F3 Fz F4 C3 Cz C4 P3 Pz P4 on 10-20 data (25 EGI names NOT STATED, §1.2)',
    'laplacian': 'equal-weight average of the 4 nearest sites by great-circle arc over the 19 channels (§1.2)',
    'filter': '1-40 Hz zero-phase 4th-order Butterworth on the continuous signal (type and order NOT STATED, §1.2)',
    'baseline_correction': 'a constant per epoch and channel; it cannot change 7-30 Hz band power, so it is omitted',
    'artefacts': "'large movement artifacts' (§1.2, method NOT STATED): reject a trial if any of the 9 channels "
                 'exceeds |100| µV in the -1.5 to 4 s epoch after 1-40 Hz filtering (the OR17 rule, §6.2)',
    'isi': 'uniform on 4.5-9.5 s (CR11, distribution NOT STATED; CR12N states uniform for 3-6.5 s)',
    'block_timing': f'instruction 5 s before the first tone, relax instruction {RELAX_S:.0f} s one ISI after the '
                    'last tone, break uniform 60-120 s',
    'block_order': 'uniform over orders with <= 2 equal blocks in a row and n of each class',
    'svm': 'C = 1 (libsvm default; NOT STATED, §1.3), linear kernel',
    'windows': 'action samples t in [0.5, 3.5) s at 100 Hz = 300 per channel and band (edges NOT STATED, §1.3)',
    'gf13_perm': f'all balanced block relabellings when <= {GF13_MAX_LABELINGS}, else {GF13_MAX_LABELINGS} random '
                 'ones (CR13: "at least 1000 permutations"); permutation unit and count NOT STATED in GF13 main text',
    'minimum': f'>= {MIN_PER_CLASS} surviving trials of each class and survivors in >= {MIN_FOLDS} CV folds (so every '
               f'test fold has training trials); GF13: survivors in >= {MIN_GF13_BLOCKS} blocks of each class (so '
               'every hand-toe test pair trains on both classes); fewer -> mc.NoUsableTrials; run_randomized '
               're-draws (never zero-fills) labellings that fall below it',
}


def _variant(method, nb, bp, ci=None):
    isi, act = ((3.0, 6.5), (0.5, 2.5)) if method == 'cr12n' else ((4.5, 9.5), (0.5, 3.5))
    return {'method': method, 'n_blocks': nb, 'bp': bp, 'isi': isi, 'action': act,
            'baseline_check': method == 'cr11', 'ci_base': ci,
            'cite': {'cr11': '§1.1-1.5', 'cr12n': '§1.7', 'gf13': '§2.1'}[method]
            + '; blocks §1.1; band power §1.3 NOT STATED' + ('; baseline CI §1.4 NOT STATED' if ci else '')}


VARIANTS = {}
for _nb in (4, 6):
    for _bp in ('hilbert', 'sq250'):
        for _ci in (0.99, 0.999):
            VARIANTS[f'cr11_b{_nb}_{_bp}_ci{str(_ci)[2:]}'] = _variant('cr11', _nb, _bp, _ci)
        VARIANTS[f'cr12n_b{_nb}_{_bp}'] = _variant('cr12n', _nb, _bp)
        VARIANTS[f'gf13_b{_nb}_{_bp}'] = _variant('gf13', _nb, _bp)
RUN_S = 2 * 6 * (INSTR_S + TONES * 9.5 + RELAX_S + BREAK_S[1]) + 10.0


def run_s(variant):
    """Longest possible assessment for one variant (RUN_S is the maximum over variants)."""
    c = mc.cfg(VARIANTS, variant)
    return 2 * c['n_blocks'] * (INSTR_S + TONES * c['isi'][1] + RELAX_S + BREAK_S[1]) + 10.0



def _resp(c):
    """engine.inject responders: right-hand imagery after each tone for the action window; toe imagery has no hand
    template ('none')."""
    return {'resp_hand': {HAND: 'right', TOE: 'none'}, 'resp_delay_s': 0.0, 'resp_dur_s': c['action'][1]}


def _timeline(start_s, block_cls, isi, g):
    """Tone onsets, labels, block ids for blocks of the given classes in order."""
    t, on, lab, blk = float(start_s), [], [], []
    for b, c in enumerate(block_cls):
        t += INSTR_S
        for k in range(TONES):
            on.append(t)
            lab.append(c)
            blk.append(b)
            t += g.uniform(*isi)
        t += RELAX_S + g.uniform(*BREAK_S)
    return np.array(on), np.array(lab, int), np.array(blk, int)


def _pair_folds(block_cls_of_trial, blk):
    """Fold = rank of the block within its class: hand block i and toe block i are left out together (GF13 §2.1)."""
    order = {}
    for b in np.unique(blk):
        c = block_cls_of_trial[blk == b][0]
        order[b] = sum(1 for bb in np.unique(blk) if bb < b and block_cls_of_trial[blk == bb][0] == c)
    return np.array([order[b] for b in blk])


def schedule(start_s, rng, variant='cr11_b4_hilbert_ci99'):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    cls = mc.constrained_order((HAND, TOE), c['n_blocks'], 2, g)
    on, lab, blk = _timeline(start_s, cls, c['isi'], g)
    return mc.Overlay(onsets_s=on, labels=lab, kind='cmd',
                      meta={'trial': np.arange(len(on)), 'block': blk, 'run': np.zeros(len(on), int),
                            'fold': _pair_folds(lab, blk), 'block_class': cls, **_resp(c)})


def schedule_randomized(start_s, rng, variant='cr11_b4_hilbert_ci99'):
    """Same block timing; the class of every tone is randomized (n x 15 of each, uniform order). Folds = pairs of
    time-adjacent blocks (blocks 2i and 2i + 1)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    nb = 2 * c['n_blocks']
    on, _, blk = _timeline(start_s, np.zeros(nb, int), c['isi'], g)
    lab = mc.balanced_labels((HAND, TOE), (len(on) // 2, len(on) - len(on) // 2), g)
    return mc.Overlay(onsets_s=on, labels=lab, kind='cmd',
                      meta={'trial': np.arange(len(on)), 'block': blk, 'run': np.zeros(len(on), int),
                            'fold': blk // 2, 'block_class': None, **_resp(c)})



def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized read; must be clean."""
    return float(ov.onsets_s.min()) + EPOCH[0] - 5.0, float(ov.onsets_s.max()) + EPOCH[1] + 5.0



def _band_power(x, bp):
    out = []
    for lo, hi in BANDS:
        b = mc.bandpass(x, FS, lo, hi)
        if bp == 'hilbert':
            pw = np.abs(hilbert(b, axis=-1)) ** 2
        else:
            k = np.ones(25) / 25.0
            pw = np.apply_along_axis(lambda v: np.convolve(v, k, 'same'), -1, b ** 2)
        out.append(np.log(np.maximum(pw, 1e-12)))
    return np.stack(out, 1)


def features(rec, ov, bp, action):
    """-> (Xa (n, 9 * 4 * n_action), Xb (n, 9 * 4 * 50) baseline window, keep mask, channel set)."""
    x = np.asarray(rec.x, float)
    seg, i0 = mc.span(x, rec.fs, *extent(ov))
    seg = mc.bandpass(seg, rec.fs, 1.0, 40.0)
    nm = mc.canon(rec.names)
    rows = [nm.index(c) for c in SUB_1020]
    onl = ov.onsets_s - i0 / rec.fs
    keep = np.abs(mc.epochs(seg[rows], rec.fs, onl, *EPOCH)).max(axis=(1, 2)) <= 100.0
    lap = mc.hjorth_matrix(rec.names, k_interior=4, k_edge=4, weights='equal') @ seg
    sel = mc.resample(lap[rows], rec.fs, FS)
    P = _band_power(sel, bp)
    ep = mc.epochs(P.reshape(-1, P.shape[-1]), FS, onl, *EPOCH)
    t = EPOCH[0] + np.arange(ep.shape[-1]) / FS
    a = (t >= action[0] - 1e-9) & (t < action[1] - 1e-9)
    b = (t >= BASE_WIN[0] - 1e-9) & (t < BASE_WIN[1] - 1e-9)
    return ep[:, :, a].reshape(len(ep), -1), ep[:, :, b].reshape(len(ep), -1), keep, '1020'


def _feat(rec, ov, c):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float), c['bp'], c['action']),
                   lambda: features(rec, ov, c['bp'], c['action']))



def fold_kernels(X, test_sets):
    """For each test set: z-score with the training rows' mean and SD, linear kernels (train x train, test x train)."""
    out = []
    n = len(X)
    for te in test_sets:
        tr = np.setdiff1d(np.arange(n), te)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        out.append((tr, te, Z[tr] @ Z[tr].T, Z[te] @ Z[tr].T))
    return out


def cv_outcomes(kernels, y):
    """Correct (1/0) for every tested trial, folds concatenated. A fold whose training rows hold one class only
    predicts that class."""
    res = []
    for tr, te, Ktr, Kte in kernels:
        if len(np.unique(y[tr])) < 2:
            pred = np.full(len(te), y[tr][0])
        else:
            pred = SVC(kernel='precomputed', C=1.0).fit(Ktr, y[tr]).predict(Kte)
        res.append(pred == y[te])
    return np.concatenate(res)


def _require(y, fold):
    for name, cls in (('hand', HAND), ('toe', TOE)):
        n = int(np.sum(y == cls))
        if n < MIN_PER_CLASS:
            raise mc.NoUsableTrials(f'cruse11: {n} {name} trial(s) survive the artefact rule, '
                                    f'>= {MIN_PER_CLASS} needed')
    nf = len(np.unique(fold))
    if nf < MIN_FOLDS:
        raise mc.NoUsableTrials(f'cruse11: the survivors fall in {nf} CV fold(s), >= {MIN_FOLDS} needed')


def _require_gf13(y, blk):
    for name, cls in (('hand', HAND), ('toe', TOE)):
        nb = len(np.unique(blk[y == cls]))
        if nb < MIN_GF13_BLOCKS:
            raise mc.NoUsableTrials(f'cruse11 gf13: survivors in {nb} {name} block(s), >= {MIN_GF13_BLOCKS} needed')


def _binofit_call(k, n, ci_level=0.99):
    lo, hi = mc.cp_interval(k, n, ci_level)
    return k / n > 0.5 and lo > 0.5


def run(rec, overlay, variant='cr11_b4_hilbert_ci99', rng=None):
    c = mc.cfg(VARIANTS, variant)
    Xa, Xb, keep, which = _feat(rec, overlay, c)
    y = np.asarray(overlay.labels)[keep]
    blk = np.asarray(overlay.meta['block'])[keep]
    if c['method'] == 'gf13':
        _require_gf13(y, blk)
        return _run_gf13(Xa[keep], y, blk, c, rng, which)
    fold = np.asarray(overlay.meta['fold'])[keep]
    _require(y, fold)
    sets = [np.nonzero(fold == f)[0] for f in np.unique(fold)]
    out = cv_outcomes(fold_kernels(Xa[keep], sets), y)
    k, n = int(out.sum()), len(out)
    lo, hi = mc.cp_interval(k, n, 0.99)
    p = binomtest(k, n, 0.5).pvalue
    call = _binofit_call(k, n)
    extra = {'acc': k / n, 'n_tested': n, 'ci99': (lo, hi), 'ci999': mc.cp_interval(k, n, 0.999),
             'n_kept': int(keep.sum()), 'channels': which}
    if c['baseline_check']:
        ob = cv_outcomes(fold_kernels(Xb[keep], sets), y)
        blo, bhi = mc.cp_interval(int(ob.sum()), len(ob), c['ci_base'])
        extra.update(baseline_acc=float(ob.mean()), baseline_ci=(blo, bhi), above_baseline=bool(k / n > bhi))
        call = call and k / n > bhi
    return mc.result(k / n, p, call, **extra)


def _labelings(nblk, n_hand, rng):
    """Balanced block relabellings (hand = 1): all of them when few enough, else GF13_MAX_LABELINGS random draws."""
    g = mc.gen(rng)
    if math.comb(nblk, n_hand) <= GF13_MAX_LABELINGS:
        combos = list(itertools.combinations(range(nblk), n_hand))
        return [combos[i] for i in g.permutation(len(combos))], True
    return [tuple(np.sort(g.choice(nblk, n_hand, replace=False))) for _ in range(GF13_MAX_LABELINGS)], False


def _run_gf13(X, y, blk, c, rng, which):
    blocks = np.unique(blk)
    bcls = np.array([y[blk == b][0] for b in blocks])
    pairs = {}
    for a, b in itertools.combinations(range(len(blocks)), 2):
        pairs[(a, b)] = fold_kernels(X, [np.nonzero((blk == blocks[a]) | (blk == blocks[b]))[0]])[0]

    def acc(cls):
        yy = cls[np.searchsorted(blocks, blk)]
        ks = [pairs[(a, b)] for (a, b) in pairs if cls[a] != cls[b]]
        return float(cv_outcomes(ks, yy).mean())

    obs = acc(bcls)
    labs, exhaustive = _labelings(len(blocks), int(np.sum(bcls == HAND)), rng)
    dev, ge, n_run, N = abs(obs - 0.5), 0, 0, len(labs)
    stop = int(np.floor(0.05 * N)) + 1 if exhaustive else int(np.floor(0.05 * (N + 1) - 1)) + 1
    for hand in labs:
        cls = np.full(len(blocks), TOE)
        cls[list(hand)] = HAND
        ge += abs(acc(cls) - 0.5) >= dev - 1e-12
        n_run += 1
        if ge >= stop and n_run < N:
            break
    p = ge / N if exhaustive else (ge + 1) / (N + 1)
    return mc.result(obs, p, p <= 0.05, acc=obs, n_labelings=len(labs), exhaustive=exhaustive, n_run=n_run,
                     stopped_early=n_run < len(labs), n_kept=len(y), channels=which,
                     note='FDR across subjects (GF13) is applied at cohort level')


def run_randomized(rec, overlay, variant='cr11_b4_hilbert_ci99', rng=None):
    """The variant's features and linear SVM, CV leaving out pairs of time-adjacent blocks; statistic = pooled CV
    accuracy; the null re-draws the per-tone randomization (balanced, uniform order), B = 500, early stopping. A draw
    leaving a class below the minimum among the surviving trials is re-drawn (conditional randomization test)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    Xa, _, keep, which = _feat(rec, overlay, c)
    lab_all = np.asarray(overlay.labels)
    fold = np.asarray(overlay.meta['fold'])[keep]
    _require(lab_all[keep], fold)
    kern = fold_kernels(Xa[keep], [np.nonzero(fold == f)[0] for f in np.unique(fold)])
    n_h = int(np.sum(lab_all == HAND))
    stat = lambda lab: float(cv_outcomes(kern, lab[keep]).mean())
    ok = lambda lab: min(np.sum(lab[keep] == HAND), np.sum(lab[keep] == TOE)) >= MIN_PER_CLASS
    draw = lambda: mc.balanced_labels((HAND, TOE), (n_h, len(lab_all) - n_h), g)
    obs = stat(lab_all)
    t = mc.perm_test(obs, lambda: mc.admissible_stat(draw, ok, stat), c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'], n_kept=int(keep.sum()), channels=which)


def stat_for_size_correction(result):
    return result['stat']


def patient_rule(results):
    """One assessment per patient (§1.1); across assessments any positive (GF13's cross-subject FDR: cohort level)."""
    return any(r['call'] for r in results)
