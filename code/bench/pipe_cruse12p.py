"""Cruse et al. 2012 PLoS One (attempted movement vs rest), AS DESCRIBED, the CORRECTED-METHOD control: its
max-over-time label permutation is expected to hold its level. Spec: the motor catalogue §3.1-3.4.
- Design (§3.1): instructions 'right hand' / 'left hand' / 'relax', 2 s each, then 4-7 s silence (uniform); blocks of
  36 trials (12 per instruction), pseudorandom with at most 3 equal instructions in a row; 5 blocks = 180 trials.
  Classes are interleaved trial by trial.
- Signal (§3.2): FC3, FC4, CP3, CP4 -> bipolar C3' = FC3-CP3, C4' = FC4-CP4; down-sampled to 100 Hz; two-way
  least-squares FIR (EEGLAB eegfilt), high-pass 1 Hz then low-pass 40 Hz; 6-s epochs from instruction onset.
- Features (§3.3): spectrogram with a 1-s window moving in 50-ms steps -> 100 time points from 1 s before to 3.95 s
  after the instruction's offset; log band power in 7-13, 13-19, 19-25, 25-30 Hz; 2 channels x 4 bands = 8 features.
- Classifier (§3.3): naive Bayes per time point, 10-fold CV, accuracy averaged over folds, smoothed with a 500-ms
  sliding window; statistic = the best smoothed accuracy over time.
- Inference (§3.3): 1000 permutations of the trial class labels, CV repeated, max over time recorded; one-tailed;
  alpha .05/2 = .025 for each of left-vs-rest and right-vs-rest. Call: either comparison significant (§3.4, implied).
  With 1000 permutations, count/1000 < .025 and (count + 1)/1001 < .025 are the same decision (count <= 24), so the
  unstated p formula does not matter; p = (count + 1)/(B + 1) is reported. Decision-exact early stopping.
Chosen where NOT STATED (each listed in FIXED): substitutes for 10-10 sites on 10-20 data, the artefact rule that
replaces visual rejection, spectrogram window shape and band edges, smoothing alignment, the break between blocks.
Multiverse (VARIANTS): CV group assignment (random vs contiguous, §3.3 NOT STATED) x naive-Bayes variance (per class vs
pooled, §3.6 'naive-Bayes variance handling' unknown)."""
import numpy as np
from scipy.signal import firls, filtfilt, get_window

from . import motor_common as mc

NAME = 'cruse12p'
GRADE = 'AS DESCRIBED'
FAMILY = 'cmd'
NATIVE_CH = ['FC3', 'CP3', 'FC4', 'CP4']
SUB_1020 = ['F3', 'P3', 'F4', 'P4']
RIGHT, LEFT, RELAX = 1, 2, 0
N_BLOCKS, PER_CLASS = 5, 12
INSTR_S, SILENCE_S = 2.0, (4.0, 7.0)
BREAK_S = 60.0
FS = 100.0
EPOCH_S = 6.0
WIN, STEP = 100, 5
N_T = 100
BANDS = ((7, 13), (13, 19), (19, 25), (25, 30))
SMOOTH = 10
N_PERM, ALPHA = 1000, 0.025
MIN_PER_CLASS = 3
RUN_S = N_BLOCKS * 3 * PER_CLASS * (INSTR_S + SILENCE_S[1]) + (N_BLOCKS - 1) * BREAK_S + EPOCH_S + 5.0

FIXED = {
    'channels': '10-20 data lack FC3/FC4/CP3/CP4: bipolar F3-P3 and F4-P4 (both straddle C3/C4 as the originals do)',
    'artefacts': "§3.2 'identified by visual inspection and removed': automated, reject a trial if any of the 4 "
                 'channels exceeds |100| µV after filtering (the OR17 mindBEAGLE rule, §6.2)',
    'filter': "§3.2 eegfilt two-way least-squares FIR: order 3*fix(fs/cutoff), 15% transition band (EEGLAB defaults)",
    'spectrogram': "§3.3 MATLAB 'spectrogram' 1-s window: Hamming (its default for an integer window), nfft 100 "
                   '(1-Hz bins); band power = sum of |S|^2 over bins lo <= f < hi (last band includes 30 Hz)',
    'time_points': 'window k spans samples [5k, 5k + 100) of the 0-6 s epoch, labelled by its end (1.00-5.95 s after '
                   'onset = -1.00 to 3.95 s from the instruction offset, §3.3)',
    'smoothing': "500-ms sliding window = 10-point boxcar over complete windows ('valid', 91 values)",
    'naive_bayes': 'Gaussian, class priors = training frequencies, unbiased variances',
    'blocks': f'pseudorandom order per block (max 3 in a row, §3.1); {BREAK_S:.0f}-s break between blocks (NOT '
              'STATED: "short breaks")',
    'perm_folds': 'permutations reuse the observed CV group assignment (random groups are exchangeable)',
    'minimum': f'>= {MIN_PER_CLASS} surviving trials of each instruction, and every CV training set holding >= 2 '
               'trials of each class of a comparison (per-class unbiased variances); fewer -> mc.NoUsableTrials. '
               f'Stratified random groups always meet the second rule at {MIN_PER_CLASS} per class; contiguous groups '
               'may not. run_randomized re-draws (never zero-fills) orders that fall below it',
}
VARIANTS = {f'{cv}_{var}': {'cv': cv, 'nb_var': var,
                            'cite': '§3.3 CV assignment NOT STATED; §3.6 naive-Bayes variance unknown'}
            for cv in ('random', 'contiguous') for var in ('perclass', 'pooled')}



def _onsets(start_s, n, rng):
    g = mc.gen(rng)
    gaps = INSTR_S + g.uniform(*SILENCE_S, size=n)
    on = float(start_s) + np.r_[0.0, np.cumsum(gaps[:-1])]
    blk = np.arange(n) // (3 * PER_CLASS)
    return on + blk * BREAK_S, blk


def schedule(start_s, rng, variant=None):
    g = mc.gen(rng)
    lab = np.concatenate([mc.constrained_order((RIGHT, LEFT, RELAX), PER_CLASS, 3, g) for _ in range(N_BLOCKS)])
    on, blk = _onsets(start_s, len(lab), g)
    return mc.Overlay(onsets_s=on, labels=lab, kind='cmd',
                      meta={'trial': np.arange(len(lab)), 'block': blk, 'run': np.zeros(len(lab), int),
                            'resp_hand': {RIGHT: 'right', LEFT: 'left', RELAX: 'none'}, 'resp_delay_s': INSTR_S,
                            'resp_dur_s': SILENCE_S[0]})


def schedule_randomized(start_s, rng, variant=None):
    """The published order is already randomized per trial (pseudorandom within blocks, §3.1); same generator."""
    return schedule(start_s, rng, variant)



def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized read; must be clean."""
    return float(ov.onsets_s.min()) - 5.0, float(ov.onsets_s.max()) + EPOCH_S + 5.0



def _eegfilt(x, fs, lo=None, hi=None):
    """EEGLAB eegfilt-style least-squares FIR, applied forward and backward."""
    nyq, trans = fs / 2.0, 0.15
    order = max(15, 3 * int(fs / (lo if lo else hi)))
    order += order % 2
    if lo:
        f = [0, lo * (1 - trans) / nyq, lo / nyq, 1]
        a = [0, 0, 1, 1]
    else:
        f = [0, hi / nyq, hi * (1 + trans) / nyq, 1]
        a = [1, 1, 0, 0]
    b = firls(order + 1, f, a)
    return filtfilt(b, [1.0], x, axis=-1, padlen=min(3 * len(b), x.shape[-1] - 1))


def features(rec, ov):
    """-> (F (n_trials, 100, 8) log band power, keep (bool: passed the artefact rule), channel set)."""
    x, used, which = mc.pick(rec, NATIVE_CH, SUB_1020)
    seg, i0 = mc.span(x, rec.fs, *extent(ov))
    seg = mc.resample(seg, rec.fs, FS)
    seg = _eegfilt(_eegfilt(seg, FS, lo=1.0), FS, hi=40.0)
    ep = mc.epochs(seg, FS, ov.onsets_s - i0 / rec.fs, 0.0, EPOCH_S)
    keep = np.abs(ep).max(axis=(1, 2)) <= 100.0
    bip = np.stack([ep[:, 0] - ep[:, 1], ep[:, 2] - ep[:, 3]], 1)
    win = get_window('hamming', WIN)
    idx = np.arange(N_T)[:, None] * STEP + np.arange(WIN)[None]
    S = np.abs(np.fft.rfft(bip[:, :, idx] * win, n=WIN, axis=-1)) ** 2
    f = np.fft.rfftfreq(WIN, 1 / FS)
    bp = []
    for k, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & ((f < hi) if k < len(BANDS) - 1 else (f <= hi))
        bp.append(np.log(S[..., m].sum(-1)))
    F = np.stack(bp, -1).transpose(0, 2, 1, 3).reshape(len(ep), N_T, 2 * len(BANDS))
    return F, keep, which



def cv_groups(n, kind, rng, y=None):
    """10 approximately equal groups: stratified random, or contiguous in trial order."""
    if kind == 'contiguous':
        return np.arange(n) * 10 // n
    g = mc.gen(rng)
    out = np.empty(n, int)
    for c in np.unique(y):
        i = np.nonzero(y == c)[0]
        out[g.permutation(i)] = (np.arange(len(i)) + g.integers(10)) % 10
    return out


def nb_cv_accuracy(F, y, groups, nb_var):
    """Gaussian naive Bayes at every time point, all folds at once. -> mean-over-folds accuracy (n_t). Defined for
    any labels (permuted or re-drawn ones included): a class absent from a training set is never predicted there."""
    folds = np.unique(groups)
    Y = (y[:, None] == np.array([0, 1])[None]).astype(float)
    TR = (groups[:, None] != folds[None]).astype(float)
    W = TR[:, :, None] * Y[:, None, :]
    cnt = W.sum(0)
    mu = np.einsum('nkc,ntd->kctd', W, F) / np.maximum(cnt, 1)[:, :, None, None]
    sq = np.einsum('nkc,ntd->kctd', W, F ** 2)
    if nb_var == 'pooled':
        ss = (sq - cnt[:, :, None, None] * mu ** 2).sum(1, keepdims=True)
        var = np.repeat(ss / np.maximum(cnt.sum(1) - 2, 1)[:, None, None, None], 2, axis=1)
    else:
        var = (sq - cnt[:, :, None, None] * mu ** 2) / np.maximum(cnt - 1, 1)[:, :, None, None]
    var = np.maximum(var, 1e-12)
    with np.errstate(divide='ignore'):
        prior = np.log(cnt / cnt.sum(1, keepdims=True))
    fi = np.searchsorted(folds, groups)
    m, v = mu[fi], var[fi]
    ll = -0.5 * (np.log(v) + (F[:, None] - m) ** 2 / v).sum(-1) + prior[fi][:, :, None]
    correct = (np.argmax(ll, 1) == y[:, None]).astype(float)
    acc = np.stack([correct[groups == f].mean(0) for f in folds])
    return acc.mean(0)


def smoothed_max(acc):
    return float(np.convolve(acc, np.ones(SMOOTH) / SMOOTH, 'valid').max())


def _comparison(F, lab, cls):
    m = (lab == cls) | (lab == RELAX)
    return F[m], (lab[m] == cls).astype(int)


def _require_trials(lab):
    for name, cls in (('right', RIGHT), ('left', LEFT), ('relax', RELAX)):
        n = int(np.sum(lab == cls))
        if n < MIN_PER_CLASS:
            raise mc.NoUsableTrials(f'cruse12p: {n} {name} trial(s) survive the artefact rule, '
                                    f'>= {MIN_PER_CLASS} needed')


def _folds_ok(y, grp):
    return all(min(np.sum(y[grp != f] == 0), np.sum(y[grp != f] == 1)) >= 2 for f in np.unique(grp))


def _require_folds(name, y, grp):
    if not _folds_ok(y, grp):
        raise mc.NoUsableTrials(f'cruse12p {name}-vs-relax: a CV training set holds < 2 trials of a class')


def _feat(rec, ov):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: features(rec, ov))


def run(rec, overlay, variant='random_perclass', rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F, keep, which = _feat(rec, overlay)
    lab = np.asarray(overlay.labels)[keep]
    F = F[keep]
    _require_trials(lab)
    out = {}
    for name, cls in (('right', RIGHT), ('left', LEFT)):
        Fc, y = _comparison(F, lab, cls)
        grp = cv_groups(len(y), c['cv'], g, y)
        _require_folds(name, y, grp)
        obs = smoothed_max(nb_cv_accuracy(Fc, y, grp, c['nb_var']))
        t = mc.perm_test(obs, lambda: smoothed_max(nb_cv_accuracy(Fc, g.permutation(y), grp, c['nb_var'])),
                         c.get('n_perm', N_PERM), ALPHA)
        out[name] = {'acc': obs, 'p': t['p'], 'n_perm_run': t['n_run'], 'stopped_early': t['stopped'],
                     'n_nonfinite_null': t['n_nonfinite_null'], 'n': len(y)}
    p = min(out['right']['p'], out['left']['p'])
    stat = max(out['right']['acc'], out['left']['acc'])
    return mc.result(stat, p, p < ALPHA, comparisons=out, n_kept=int(keep.sum()), channels=which,
                     n_nonfinite_null=sum(o['n_nonfinite_null'] for o in out.values()))


def run_randomized(rec, overlay, variant='random_perclass', rng=None):
    """Same features and classifier; statistic = max over both comparisons of the best smoothed accuracy (one family,
    alpha .05); the null re-draws the whole pseudorandom order (§3.1 generator) for the kept trials, B = 500. An
    order below the minimum among the surviving trials is re-drawn (conditional randomization test)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F, keep, which = _feat(rec, overlay)
    F = F[keep]
    grp_seed = mc.seed_of(g)
    lab_obs = np.asarray(overlay.labels)[keep]
    _require_trials(lab_obs)

    def groups(lab, cls):
        Fc, y = _comparison(F, lab, cls)
        return Fc, y, cv_groups(len(y), c['cv'], np.random.default_rng(grp_seed + cls), y)

    def ok(lab):
        """The module's minimum (FIXED['minimum']) for a re-drawn order among the surviving trials."""
        if any(np.sum(lab == cls) < MIN_PER_CLASS for cls in (RIGHT, LEFT, RELAX)):
            return False
        return all(_folds_ok(*groups(lab, cls)[1:]) for cls in (RIGHT, LEFT))

    def stat(lab):
        return max(smoothed_max(nb_cv_accuracy(Fc, y, grp, c['nb_var']))
                   for Fc, y, grp in (groups(lab, RIGHT), groups(lab, LEFT)))

    def draw_labels():
        lab = np.concatenate([mc.constrained_order((RIGHT, LEFT, RELAX), PER_CLASS, 3, g) for _ in range(N_BLOCKS)])
        return lab[keep]

    for name, cls in (('right', RIGHT), ('left', LEFT)):
        _require_folds(name, *groups(lab_obs, cls)[1:])
    obs = stat(lab_obs)
    t = mc.perm_test(obs, lambda: mc.admissible_stat(draw_labels, ok, stat), c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'], n_kept=int(keep.sum()), channels=which)


def stat_for_size_correction(result):
    return result['stat']


def patient_rule(results):
    """One assessment per patient in §3.4; across assessments: any positive."""
    return any(r['call'] for r in results)
