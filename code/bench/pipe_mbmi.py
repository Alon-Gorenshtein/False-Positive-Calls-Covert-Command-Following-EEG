"""mindBEAGLE-type motor imagery (g.tec; Guger 2017, Ortner 2017 JoVE, Chatelle 2018), AS DESCRIBED.
Spec: the motor catalogue §6.
- Design (§6.1): left or right hand imagery cued in random order, 60 trials (30 per hand; one OR17 passage says 120).
  A trial starts with an attention beep, 2 s of silence, the 'LEFT'/'RIGHT' command, then imagery for 4 s (GU17),
  6 s (CH18) or to the end of an 8-s trial (OR17), then a relax command and a pause (GU17 0.5-2 s random; CH18 1 s;
  OR17 1-s breaks, and in its protocol a ~1-min pause after ~4 min).
- Signal and classifier (§6.2): 8-30 Hz; CSP trained on the window 3-5 s after the instruction; variance of a 1.5-s
  window, log-transformed and normalized -> 4 features; LDA; trials with |amplitude| > 100 µV rejected; accuracy
  computed in 0.5-s steps from 1.5 s after the beep to the end of the trial; 10 x 10-fold CV (GU17, OR17) or 10-fold
  (CH18), averaged over repetitions.
- Call (§6.3): accuracy above the binomial 95% limit, 63.19% for 60 trials (GU17; = the Clopper-Pearson upper bound
  at 30/60 [derived, 0.63194]) or the Clopper-Pearson line for the trials used (OR17); no permutation; no correction
  for the time-step search; 'If the accuracy level is below the significance level, repeat the run' (OR17) -> the
  patient composite is any of k runs ('best run'; for a threshold rule they coincide).
Multiverse (VARIANTS): timing {gu17, ch18, or17_60, or17_120} (§6.1 discrepancy; each carries its source's CV and
threshold) x CSP fitted inside each fold or once on all trials (§6.2 NOT STATED) x the reported number = peak of the
time course or mean over seconds 5-7 (§6.2 NOT STATED; GU17 quotes 'the mean accuracy from second 5 to 7').
Other NOT STATED choices are fixed in FIXED."""
import numpy as np

from . import motor_common as mc

NAME = 'mbmi'
GRADE = 'AS DESCRIBED'
FAMILY = 'cmd'
NATIVE_CH = ['Fz', 'C3', 'Cz', 'C4', 'CP1', 'CPz', 'CP2', 'Pz', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4']
SUB_1020 = ['Fz', 'F3', 'F4', 'C3', 'Cz', 'C4', 'T7', 'T8', 'P3', 'Pz', 'P4']
LEFT, RIGHT = 1, 2
CUE_S = 2.0
CSP_WIN = (CUE_S + 3.0, CUE_S + 5.0)
LDA_WIN = (CUE_S + 3.5, CUE_S + 5.0)
VAR_S, T_FIRST, T_STEP, T_END = 1.5, 1.5, 0.5, 8.0
TIMES = np.arange(T_FIRST, T_END + 1e-9, T_STEP)
MEAN_WIN = (5.0, 7.0)
THR_GU17 = 0.6319
N_FOLD = 10
TIMING = {
    'gu17': {'n': 60, 'imag': 4.0, 'gap': (0.5, 2.0), 'pause': None, 'reps': 10, 'thr': 'fixed',
             'cite': '§6.1 GU17 (4-s imagery, 0.5-2 s random interval, 30 per hand); §6.2-6.3 10 x 10 CV, 63.19%'},
    'ch18': {'n': 60, 'imag': 6.0, 'gap': (1.0, 1.0), 'pause': None, 'reps': 1, 'thr': 'cp',
             'cite': '§6.1 CH18 (beep, 2 s, command, 6 s, relax, 1 s); §6.2 10-fold CV; §6.3 binomial p < .05'},
    'or17_60': {'n': 60, 'imag': 6.0, 'gap': (1.0, 1.0), 'pause': (30, 60.0), 'reps': 10, 'thr': 'cp',
                'cite': '§6.1 OR17 protocol (60 commands, pause ~1 min after ~4 min; 8-s trials + 1-s breaks)'},
    'or17_120': {'n': 120, 'imag': 6.0, 'gap': (1.0, 1.0), 'pause': None, 'reps': 10, 'thr': 'cp',
                 'cite': '§6.1 OR17 introduction (120 trials of 8 s, 1-s breaks)'},
}
FIXED = {
    'channels': '10-20 data lack 11 of the 16 sites: the nearest 10-20 site of each, duplicates dropped -> 11 '
                '(FC3->F3, FC4->F4, C5->T7, C6->T8, C1->C3, C2->C4, CP1/CP3->P3, CP2/CP4->P4, CPz->Pz)',
    'cue': 'command 2 s after the beep for every timing (CH18; GU17/OR17 NOT STATED); OR17 imagery = to the end of the '
           '8-s trial; relax-command length 0',
    'filters': 'zero-phase 4th-order Butterworth: 0.1-30 Hz for the artefact rule (GU17 amplifier band), 8-30 Hz for '
               'CSP (type and order NOT STATED)',
    'artefacts': 'OR17: reject a trial if |x| > 100 µV anywhere in 0-8 s after its beep (0.1-30 Hz data)',
    'csp': 'trace-normalized class covariances, generalized eigenvectors, 2 filters from each end (OR17: four '
           'features); features log(var_i / sum var) (CH18 "log-transformed and normalized")',
    'lda': 'trained on the 1.5-s window at the end of the CSP window (3.5-5 s after the instruction); pooled '
           'covariance, class priors from training frequencies; applied to the trailing 1.5-s window at every step',
    'time_course': f'{len(TIMES)} steps, t = 1.5, 2.0, ... 8.0 s after the beep, window [t - 1.5, t]',
    'cv': 'stratified random 10-fold partitions (fold assignment NOT STATED, §6.2)',
    'minimum': f'>= {N_FOLD} surviving trials of each hand (every stratified 10-fold test fold holds both); fewer -> '
               'mc.NoUsableTrials; run_randomized re-draws (never zero-fills) orders that fall below it',
}
VARIANTS = {}
for _tm, _t in TIMING.items():
    for _csp in ('fold', 'global'):
        for _sm in ('peak', 'mean57'):
            VARIANTS[f'{_tm}_csp{_csp}_{_sm}'] = dict(_t, timing=_tm, csp=_csp, summary=_sm,
                                                       cite=_t['cite'] + '; CSP refit NOT STATED; summary NOT STATED')
RUN_S = 120 * (CUE_S + 6.0 + 1.0) + 70.0


def run_s(variant):
    """Longest possible assessment for one variant (RUN_S is the maximum over variants)."""
    c = mc.cfg(VARIANTS, variant)
    return c['n'] * (CUE_S + c['imag'] + c['gap'][1]) + (c['pause'][1] if c['pause'] else 0.0) + T_END + 10.0



def schedule(start_s, rng, variant='gu17_cspfold_peak'):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    n = c['n']
    lab = mc.balanced_labels((LEFT, RIGHT), (n // 2, n // 2), g)
    gaps = CUE_S + c['imag'] + g.uniform(*c['gap'], size=n)
    on = float(start_s) + np.r_[0.0, np.cumsum(gaps[:-1])]
    if c['pause']:
        on[c['pause'][0]:] += c['pause'][1]
    return mc.Overlay(onsets_s=on, labels=lab, kind='cmd',
                      meta={'trial': np.arange(n), 'block': np.zeros(n, int), 'run': np.zeros(n, int),
                            'cue_s': np.full(n, CUE_S), 'resp_hand': {LEFT: 'left', RIGHT: 'right'},
                            'resp_delay_s': CUE_S, 'resp_dur_s': c['imag']})


def schedule_randomized(start_s, rng, variant='gu17_cspfold_peak'):
    """The published order is already randomized per trial (§6.1); same generator."""
    return schedule(start_s, rng, variant)



def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized read; must be clean."""
    return float(ov.onsets_s.min()) - 5.0, float(ov.onsets_s.max()) + T_END + 5.0



def _covs(ep, fs, t0, t1):
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
    s = ep[..., i0:i1]
    s = s - s.mean(-1, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / s.shape[-1]


def features(rec, ov):
    """Label-free part: covariances of 8-30 Hz epochs. -> dict(csp (n, c, c), lda (n, c, c), t (n, T, c, c)), keep."""
    x, used, which = mc.pick(rec, NATIVE_CH, SUB_1020)
    seg, i0 = mc.span(x, rec.fs, *extent(ov))
    onl = ov.onsets_s - i0 / rec.fs
    wide = mc.epochs(mc.bandpass(seg, rec.fs, 0.1, 30.0), rec.fs, onl, 0.0, T_END)
    keep = np.abs(wide).max(axis=(1, 2)) <= 100.0
    ep = mc.epochs(mc.bandpass(seg, rec.fs, 8.0, 30.0), rec.fs, onl, 0.0, T_END)
    C = {'csp': _covs(ep, rec.fs, *CSP_WIN), 'lda': _covs(ep, rec.fs, *LDA_WIN),
         't': np.stack([_covs(ep, rec.fs, t - VAR_S, t) for t in TIMES], 1)}
    return C, keep, which


def csp_filters(Ccsp, y):
    """Two filters from each end of the generalized eigen-decomposition of the trace-normalized class covariances."""
    Cn = Ccsp / np.trace(Ccsp, axis1=1, axis2=2)[:, None, None]
    Ca, Cb = Cn[y == LEFT].mean(0), Cn[y == RIGHT].mean(0)
    from scipy.linalg import eigh
    _, V = eigh(Ca, Ca + Cb)
    return V[:, [0, 1, -2, -1]]


def logvar(C, W):
    v = np.einsum('...cd,ck,dk->...k', C, W, W)
    return np.log(v / v.sum(-1, keepdims=True))


def lda_fit(F, y):
    m1, m2 = F[y == LEFT].mean(0), F[y == RIGHT].mean(0)
    R = np.r_[F[y == LEFT] - m1, F[y == RIGHT] - m2]
    S = R.T @ R / (len(F) - 2)
    w = np.linalg.solve(S + 1e-10 * np.eye(len(S)), m2 - m1)
    b = -w @ (m1 + m2) / 2 + np.log(np.mean(y == RIGHT) / np.mean(y == LEFT))
    return w, b


def time_course(C, y, c, g):
    """Mean over repetitions of the per-step CV accuracy (every trial tested once per repetition)."""
    n = len(y)
    W_all = csp_filters(C['csp'], y) if c['csp'] == 'global' else None
    acc = np.zeros(len(TIMES))
    for _ in range(c['reps']):
        fold = np.empty(n, int)
        for cls in (LEFT, RIGHT):
            i = np.nonzero(y == cls)[0]
            fold[g.permutation(i)] = (np.arange(len(i)) + g.integers(N_FOLD)) % N_FOLD
        correct = np.zeros((n, len(TIMES)))
        for f in range(N_FOLD):
            te, tr = fold == f, fold != f
            W = W_all if W_all is not None else csp_filters(C['csp'][tr], y[tr])
            w, b = lda_fit(logvar(C['lda'][tr], W), y[tr])
            d = logvar(C['t'][te], W) @ w + b
            correct[te] = (d > 0) == (y[te] == RIGHT)[:, None]
        acc += correct.mean(0)
    return acc / c['reps']


def summarize(acc, how):
    if how == 'peak':
        return float(acc.max())
    m = (TIMES >= MEAN_WIN[0] - 1e-9) & (TIMES <= MEAN_WIN[1] + 1e-9)
    return float(acc[m].mean())


def threshold(c, n):
    return THR_GU17 if c['thr'] == 'fixed' else mc.cp_interval(n / 2.0, n, 0.95)[1]


def _feat(rec, ov):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: features(rec, ov))


def _sub(C, keep):
    return {k: v[keep] for k, v in C.items()}


def require(y):
    for name, cls in (('left', LEFT), ('right', RIGHT)):
        n = int(np.sum(y == cls))
        if n < N_FOLD:
            raise mc.NoUsableTrials(f'mbmi: {n} {name}-hand trial(s) survive the 100-µV rule, >= {N_FOLD} needed')


def run(rec, overlay, variant='gu17_cspfold_peak', rng=None):
    c = mc.cfg(VARIANTS, variant)
    C, keep, which = _feat(rec, overlay)
    y = np.asarray(overlay.labels)[keep]
    require(y)
    acc = time_course(_sub(C, keep), y, c, mc.gen(rng))
    s, thr = summarize(acc, c['summary']), threshold(c, len(y))
    return mc.result(s, None, s > thr, threshold=thr, time_course=acc, n_kept=int(keep.sum()), channels=which)


def run_randomized(rec, overlay, variant='gu17_cspfold_peak', rng=None):
    """Same features, CSP + LDA and summary; the null re-draws the balanced L/R order (§6.1), B = 500; a re-drawn
    order leaving a hand below the minimum among the surviving trials is re-drawn (conditional randomization test)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    C, keep, which = _feat(rec, overlay)
    Ck = _sub(C, keep)
    lab = np.asarray(overlay.labels)
    n_l = int(np.sum(lab == LEFT))
    y = lab[keep]
    require(y)
    ok = lambda yy: min(np.sum(yy == LEFT), np.sum(yy == RIGHT)) >= N_FOLD
    stat = lambda yy: summarize(time_course(Ck, yy, c, g), c['summary'])
    draw_labels = lambda: mc.balanced_labels((LEFT, RIGHT), (n_l, len(lab) - n_l), g)[keep]
    obs = stat(y)
    t = mc.perm_test(obs, lambda: mc.admissible_stat(draw_labels, ok, stat), c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'], n_kept=int(keep.sum()), channels=which)


def stat_for_size_correction(result):
    return result['stat']


def patient_rule(results):
    """'Repeat the run if below the significance level' (OR17, §6.3): positive if any of the k runs is."""
    return any(r['call'] for r in results)
