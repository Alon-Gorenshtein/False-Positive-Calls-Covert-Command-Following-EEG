"""Du Bois et al. 2026 Communications Medicine (Ulster/Bath MI-BCI, after Coyle 2015), AS DESCRIBED.
Spec: the motor catalogue §9.2 (and §9.1 for the lineage).
- Design (§9.2): Assessment run = six blocks of 15 trials alternating between two imagined movements (three blocks
  each), a 0.5-s beep cueing imagery every 8 s, a recorded instruction at each block start, ~17 min in all; Training
  run = 60 randomly ordered trials (30 per class), command then 5 s imagery and a 3-s relax cue.
- Configuration search (§9.2, [derived] 84): 2 channel sets x 3 band sets x classification windows of 1 and 2 s x
  4-10 MI-selected features. FBCSP-MuI (filter-bank CSP, mutual-information feature selection) + RLDA, six-fold CV,
  decoding accuracy (DA) time series smoothed with a 200-ms moving average; the configuration with the highest peak
  DA in the task interval (0.4-7 s after the cue) is kept.
- Test (§9.2): task DA vs baseline DA (-1 to 0 s) by a one-tailed paired t-test across CV folds, alpha 0.05;
  per fold the peak in each interval ('peak vs baseline peak'). 100 permutations of the trial labels with the full
  six-fold CV give a per-time-point p = (#{perm >= obs} + 1)/(n + 1).
- Call (§9.2): Phase-I progression = DA > 70% OR the t-test significant; 'significant runs' = the t-test alone.
Multiverse (VARIANTS): run type {assessment (block design), training (randomized)} (§9.2 'which Phase-I runs count'
NOT STATED) x six-fold assignment {random stratified, blocked: leave-one-block-out / contiguous sixths} (§9.2 NOT
STATED) x call {phase1, t_only, perm} (§9.2: how the permutation p and the t-test combine is NOT STATED; 'perm' =
permutation p at the peak time < 0.05 with the selected configuration fixed, since repeating the search inside the
permutations is NOT STATED either).
Fixed choices for the other NOT STATED items are in FIXED."""
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import uniform_filter1d
from scipy.stats import norm, t as tdist
from sklearn.covariance import ledoit_wolf

from . import motor_common as mc

NAME = 'dubois'
GRADE = 'AS DESCRIBED'
FAMILY = 'cmd'
CHANSETS = {'full': ['F3', 'Fz', 'F4', 'F7', 'F8', 'C3', 'Cz', 'C4', 'T7', 'T8', 'P3', 'Pz', 'P4', 'P7', 'P8',
                     'O1', 'O2'],
            'motor': ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']}
NATIVE_CH = list(CHANSETS['full'])
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 13), 'lowbeta': (13, 20), 'highbeta': (20, 30),
         'lowgamma': (30, 40)}
BANDSETS = {'delta-lowbeta': ['delta', 'theta', 'mu', 'lowbeta'],
            'theta-highbeta': ['theta', 'mu', 'lowbeta', 'highbeta'],
            'mu-lowgamma': ['mu', 'lowbeta', 'highbeta', 'lowgamma']}
WINDOWS = (1.0, 2.0)
N_FEATS = tuple(range(4, 11))
CLASS_A, CLASS_B = 1, 2
N_FOLD = 6
DT = 0.1
TIMES = np.round(np.arange(-1.0, 7.0 + 1e-9, DT), 6)
TASK, BASE = (0.4, 7.0), (-1.0, 0.0)
SMOOTH = 3
TRAIN_STEP = 0.5
DA_70, ALPHA, N_PERM_LABELS = 0.70, 0.05, 100
INSTR_S, BEEP_EVERY_S, REST_BETWEEN_S, TRIALS_PER_BLOCK = 5.0, 8.0, 60.0, 15
TRAIN_TRIAL_S = 8.0
RUN_S = 6 * (INSTR_S + TRIALS_PER_BLOCK * BEEP_EVERY_S) + 5 * REST_BETWEEN_S + 15.0

FIXED = {
    'channels': '10-20 substitutes: full = 17 sites (all but Fp1/Fp2), motor = F3 Fz F4 C3 Cz C4 P3 Pz P4 '
                '(the 16/9 g.Nautilus sites are not in the catalogue)',
    'bands': 'band sets bracket named bands: delta 1-4, theta 4-8, mu 8-13, low beta 13-20, high beta 20-30, low '
             'gamma 30-40 Hz (sub-band edges NOT in the catalogue); zero-phase 4th-order Butterworth',
    'csp': '2 filters from each end per band (4 per band, 16 per band set), log of normalized variance',
    'mi': 'mutual information of each feature with the class by the plug-in Gaussian class-conditional estimator '
          '(FBCSP Parzen estimator NOT STATED); top-k by MI',
    'rlda': 'LDA with Ledoit-Wolf shrinkage of the pooled within-class covariance',
    'training': 'one CSP + MI + RLDA per fold, trained on trailing windows at 0.5-s steps lying inside 0.4-7 s, '
                'applied to the trailing window at every 0.1-s step from -1 to 7 s (time-resolved retraining NOT '
                'STATED; not run, see the report)',
    'smoothing': '200-ms moving average = centred 3-point boxcar at 0.1-s steps, edges by nearest value',
    'assessment_timing': f'{INSTR_S:.0f}-s instruction then the first beep; {REST_BETWEEN_S:.0f} s rest between '
                         'blocks (makes the stated ~17 min [derived]); blocks A B A B A B',
    'training_timing': f'cue every {TRAIN_TRIAL_S:.0f} s (command + 5 s imagery + 3 s relax; command length NOT '
                       'STATED)',
    'sampling': 'native rate (DB26 down-sampled 250 to 125 Hz; BDSP is 128 Hz)',
    't_test': 'per fold max of the smoothed DA in 0.4-7 s vs max in -1-0 s; paired t over 6 folds, one-tailed; '
              'zero-variance differences: p = 0 if the mean is positive, else 1',
}
VARIANTS = {f'{rt}_{cv}_{call}': {'run': rt, 'cv': cv, 'call': call,
                                  'cite': '§9.2 run type, CV assignment and call combination NOT STATED'}
            for rt in ('assessment', 'training') for cv in ('random', 'blocked') for call in ('phase1', 't_only',
                                                                                               'perm')}
CONFIGS = [(cs, bs, w, k) for cs in CHANSETS for bs in BANDSETS for w in WINDOWS for k in N_FEATS]



def schedule(start_s, rng, variant='assessment_random_phase1'):
    c = mc.cfg(VARIANTS, variant)
    if c['run'] == 'assessment':
        on, lab, blk = [], [], []
        t = float(start_s)
        for b in range(6):
            t += INSTR_S
            for k in range(TRIALS_PER_BLOCK):
                on.append(t + k * BEEP_EVERY_S)
                lab.append(CLASS_A if b % 2 == 0 else CLASS_B)
                blk.append(b)
            t += TRIALS_PER_BLOCK * BEEP_EVERY_S + REST_BETWEEN_S
        on, lab, blk = np.array(on), np.array(lab), np.array(blk)
    else:
        lab = mc.balanced_labels((CLASS_A, CLASS_B), (30, 30), rng)
        on = float(start_s) + np.arange(60) * TRAIN_TRIAL_S
        blk = np.arange(60) * N_FOLD // 60
    return mc.Overlay(onsets_s=on, labels=lab, kind='cmd',
                      meta={'trial': np.arange(len(on)), 'block': blk, 'run': np.zeros(len(on), int),
                            'resp_hand': {CLASS_A: 'left', CLASS_B: 'right'}, 'resp_delay_s': 0.0,
                            'resp_dur_s': TASK[1] if c['run'] == 'assessment' else 5.0})


def schedule_randomized(start_s, rng, variant='assessment_random_phase1'):
    """Same timing; the class of every trial randomized (balanced, uniform order). The training run already is."""
    ov = schedule(start_s, rng, variant)
    n = len(ov.labels)
    ov.labels = mc.balanced_labels((CLASS_A, CLASS_B), (n // 2, n - n // 2), rng)
    return ov



def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized read; must be clean."""
    lead = max(WINDOWS) - TIMES[0]
    return float(ov.onsets_s.min()) - lead - 5.0, float(ov.onsets_s.max()) + TIMES[-1] + 5.0



def features(rec, ov):
    """-> {(chanset, window): array (n_bands, n_trials, n_times, ch * ch) float32}."""
    x = np.asarray(rec.x, float)
    nm = mc.canon(rec.names)
    x = x[[nm.index(c) for c in CHANSETS['full']]]
    lead = max(WINDOWS) - TIMES[0]
    seg, i0 = mc.span(x, rec.fs, *extent(ov))
    onl = ov.onsets_s - i0 / rec.fs
    sub = [CHANSETS['full'].index(c) for c in CHANSETS['motor']]
    ends = np.round((TIMES + lead) * rec.fs).astype(int)
    out = {(cs, w): np.empty((len(BANDS), len(onl), len(TIMES), len(CHANSETS[cs]) ** 2), np.float32)
           for cs in CHANSETS for w in WINDOWS}
    for b, (lo, hi) in enumerate(BANDS.values()):
        ep = mc.epochs(mc.bandpass(seg, rec.fs, lo, hi), rec.fs, onl, -lead, TIMES[-1] + 1.0 / rec.fs)
        for i in range(len(ep)):
            e = ep[i]
            P = np.concatenate([np.zeros((1, e.shape[0], e.shape[0])),
                                np.cumsum(e.T[:, :, None] * e.T[:, None, :], 0)])
            for w in WINDOWS:
                nw = int(round(w * rec.fs))
                C = (P[ends + 1] - P[ends + 1 - nw]) / nw
                out[('full', w)][b, i] = C.reshape(len(TIMES), -1)
                out[('motor', w)][b, i] = C[:, sub][:, :, sub].reshape(len(TIMES), -1)
    return out



def _train_idx(w):
    tt = np.round(np.arange(TASK[0] + w, TASK[1] + 1e-9, TRAIN_STEP), 6)
    return np.searchsorted(TIMES, tt - 1e-9)


def _csp_logvar(Cf, tr_mask, y, ti, ch):
    """Cf (n, T, ch*ch) of one band -> features (n, T, 4) from CSP fitted on training trials at times ti."""
    diag = np.arange(ch) * (ch + 1)
    Ct = Cf[tr_mask][:, ti].astype(float)
    Ct = Ct / Ct[..., diag].sum(-1, keepdims=True)
    ytr = y[tr_mask]
    Ca = Ct[ytr == CLASS_A].mean((0, 1)).reshape(ch, ch)
    Cb = Ct[ytr == CLASS_B].mean((0, 1)).reshape(ch, ch)
    _, V = eigh(Ca, Ca + Cb)
    W = V[:, [0, 1, -2, -1]]
    M = np.einsum('ck,dk->cdk', W, W).reshape(ch * ch, 4)
    v = np.maximum(Cf.reshape(-1, ch * ch) @ M.astype(np.float32), 1e-20).reshape(Cf.shape[0], Cf.shape[1], 4)
    return np.log(v / v.sum(-1, keepdims=True))


def mi_scores(F, y):
    """Plug-in MI(feature; class) with Gaussian class conditionals, per column."""
    p1 = np.mean(y == CLASS_B)
    a, b = F[y == CLASS_A], F[y == CLASS_B]
    l0 = norm.logpdf(F, a.mean(0), a.std(0) + 1e-9) + np.log(1 - p1)
    l1 = norm.logpdf(F, b.mean(0), b.std(0) + 1e-9) + np.log(p1)
    q = 1.0 / (1.0 + np.exp(np.clip(l0 - l1, -50, 50)))
    hc = -np.mean(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12), 0)
    h = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    return h - hc


def rlda(F, y):
    a, b = F[y == CLASS_A], F[y == CLASS_B]
    ma, mb = a.mean(0), b.mean(0)
    S = ledoit_wolf(np.r_[a - ma, b - mb], assume_centered=True)[0]
    w = np.linalg.solve(S, mb - ma)
    return w, -w @ (ma + mb) / 2 + np.log(len(b) / len(a))


def cv_da(P, y, folds, configs):
    """DA (n_configs, n_folds, n_times): every configuration, every fold, every time step."""
    fold_ids = np.unique(folds)
    out = np.zeros((len(configs), len(fold_ids), len(TIMES)))
    band_names = list(BANDS)
    for (cs, w) in {(c[0], c[2]) for c in configs}:
        Cw, ch, ti = P[(cs, w)], len(CHANSETS[cs]), _train_idx(w)
        need = sorted({band_names.index(b) for c in configs if c[0] == cs and c[2] == w for b in BANDSETS[c[1]]})
        for fi, f in enumerate(fold_ids):
            tr = folds != f
            feats = {b: _csp_logvar(Cw[b], tr, y, ti, ch) for b in need}
            ytr = np.repeat(y[tr], len(ti))
            for j, (cs2, bs, w2, k) in enumerate(configs):
                if (cs2, w2) != (cs, w):
                    continue
                Fb = np.concatenate([feats[band_names.index(b)] for b in BANDSETS[bs]], -1)
                Ftr = Fb[tr][:, ti].reshape(-1, Fb.shape[-1])
                sel = np.argsort(-mi_scores(Ftr, ytr), kind='stable')[:k]
                wv, b0 = rlda(Ftr[:, sel], ytr)
                d = Fb[~tr][:, :, sel] @ wv + b0
                out[j, fi] = ((d > 0) == (y[~tr] == CLASS_B)[:, None]).mean(0)
    return out


def smooth(da):
    return uniform_filter1d(da, SMOOTH, axis=-1, mode='nearest')


def _masks():
    return ((TIMES >= TASK[0] - 1e-9) & (TIMES <= TASK[1] + 1e-9),
            (TIMES >= BASE[0] - 1e-9) & (TIMES <= BASE[1] + 1e-9))


def make_folds(y, blocks, kind, rng):
    if kind == 'blocked':
        return np.asarray(blocks)
    g = mc.gen(rng)
    out = np.empty(len(y), int)
    for cls in (CLASS_A, CLASS_B):
        i = np.nonzero(y == cls)[0]
        out[g.permutation(i)] = (np.arange(len(i)) + g.integers(N_FOLD)) % N_FOLD
    return out


def search(P, y, folds, configs=CONFIGS):
    """-> (best config index, smoothed mean DA of every config (n_cfg, T), smoothed per-fold DA (n_cfg, k, T))."""
    sm = smooth(cv_da(P, y, folds, configs))
    mean = sm.mean(1)
    task, _ = _masks()
    best = int(np.argmax(mean[:, task].max(1)))
    return best, mean, sm


def paired_t(fold_da):
    task, base = _masks()
    d = fold_da[:, task].max(1) - fold_da[:, base].max(1)
    m, s = d.mean(), d.std(ddof=1)
    if s == 0:
        return (np.inf if m > 0 else 0.0), (0.0 if m > 0 else 1.0)
    tv = m / (s / np.sqrt(len(d)))
    return float(tv), float(tdist.sf(tv, len(d) - 1))


def _feat(rec, ov):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: features(rec, ov), size=2)


def run(rec, overlay, variant='assessment_random_phase1', rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    P = _feat(rec, overlay)
    configs = c.get('configs', CONFIGS)
    y = np.asarray(overlay.labels)
    blocks = np.asarray(overlay.meta['block'])
    folds = make_folds(y, blocks, c['cv'], g)
    best, mean, sm = search(P, y, folds, configs)
    task, _ = _masks()
    peak = float(mean[best, task].max())
    t_peak = float(TIMES[task][np.argmax(mean[best, task])])
    tv, p_t = paired_t(sm[best])
    calls = {'phase1': bool(peak > DA_70 or p_t < ALPHA), 't_only': bool(p_t < ALPHA)}
    extra = {'peak_da': peak, 't_peak_s': t_peak, 't': tv, 'p_t': p_t, 'config': configs[best], 'calls': calls}
    if c['call'] == 'perm':
        ip = int(np.nonzero(np.isclose(TIMES, t_peak))[0][0])
        ge = 0
        for _ in range(c.get('n_perm_labels', N_PERM_LABELS)):
            yp = g.permutation(y)
            da = smooth(cv_da(P, yp, make_folds(yp, blocks, c['cv'], g), [configs[best]])).mean(1)[0]
            ge += da[ip] >= mean[best, ip] - 1e-12
        n = c.get('n_perm_labels', N_PERM_LABELS)
        p_perm = (ge + 1.0) / (n + 1)
        calls['perm'] = bool(p_perm < ALPHA)
        extra['p_perm'] = p_perm
        return mc.result(peak, p_perm, calls['perm'], **extra)
    return mc.result(tv, p_t, calls[c['call']], **extra)


def run_randomized(rec, overlay, variant='assessment_random_phase1', rng=None):
    """Same features, configuration search and classifier; statistic = the selected configuration's peak smoothed
    DA in the task interval (the search repeated in every draw); the null re-draws the balanced per-trial order,
    B = 500, early stopping. Depends on the run type and CV keys only (not on the call key)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    P = _feat(rec, overlay)
    configs = c.get('configs', CONFIGS)
    y = np.asarray(overlay.labels)
    blocks = np.asarray(overlay.meta['block'])
    task, _ = _masks()

    def stat(lab):
        best, mean, _ = search(P, lab, make_folds(lab, blocks, c['cv'], g), configs)
        return float(mean[best, task].max())

    n_a = int(np.sum(y == CLASS_A))
    obs = stat(y)
    t = mc.perm_test(obs, lambda: stat(mc.balanced_labels((CLASS_A, CLASS_B), (n_a, len(y) - n_a), g)),
                     c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'])


def stat_for_size_correction(result):
    """phase1 / t_only: the paired t (the stated significance test; the 70% arm is not a scalar threshold);
    perm: the peak DA."""
    return result['stat']


def patient_rule(results):
    """'a significant session among 12': any of the patient's runs."""
    return any(r['call'] for r in results)
