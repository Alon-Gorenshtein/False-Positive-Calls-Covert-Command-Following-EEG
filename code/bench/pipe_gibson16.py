"""Gibson 2016 (Ann Neurol 80:412) vibrotactile selective attention: SSEP F-test, P3a and P3b cluster-mass tests.
AS DESCRIBED (no code found, cat. §6). Spec: the erp catalogue §6 (cat. §6), quotes from the AAM pages.
- Design (AAM p. 5): stimulators on both wrists (10% each) and the upper back (80%); one 50-ms vibration every
  200 ms (no jitter); 14 blocks of ~1 min; per block 28-32 vibrations per wrist (uniform integers); 3-21 back
  vibrations between wrist vibrations; count the target wrist; right wrist first, then alternating.
  NOT STATED, ours: the distribution of the back-run lengths (3 + Geometric(1/2) truncated at 21: mean 4, which
  gives the stated 80 / 10 / 10 split), back runs also before the first and after the last wrist vibration, and the
  between-block pause (instruction + touching the wrist) of GAP_S.
- EEG (AAM p. 7): FC1 Fz FC2 C3 Cz C4 CP1 CP2 Pz Oz PO7 PO8 (10-20 data: FC1 -> F3, FC2 -> F4, CP1 -> P3, CP2 -> P4,
  Oz -> O1 and O2, PO7 -> T5, PO8 -> T6, the nearest 10-20 sites); 0.5-30 Hz (type NOT STATED: zero-phase 4th-order
  Butterworth here); 1-s epochs with 200 ms before the stimulus; linear detrend and baseline correction; trials
  beyond +-100 µV rejected; trials with kurtosis beyond 2.5 SD rejected (ours: each trial's largest channel kurtosis,
  z-scored over trials). Right-earlobe reference: not available (data used as given).
- SSEP (AAM p. 8): Pz, the average of all trials, DFT over the whole 1-s epoch (1-Hz bins); F = power at 5 and 10 Hz
  over the mean power of "the ten adjacent ~1 Hz frequency bins", present if F(2, 20) >= 3.49. Which ten bins, and
  whether 5 and 10 Hz are averaged or tested alone: NOT STATED (variants). Documented property (cat. §6 [mine]):
  averaging epochs locked to every stimulus of a strictly periodic 200-ms train is a comb filter at 5, 10, 15 Hz,
  so the test is positive on stimulus-free EEG (tests).
- P3a / P3b (AAM p. 8): wrists vs an equal-sized random subset of back stimuli (P3a), target vs non-target wrist with
  trial numbers matched (P3b); 50-750 ms; FieldTrip cluster mass: trial-level independent-samples t, clusteralpha .05,
  spatio-temporal neighbours (NOT STATED: see below), maxsum, 1000 random permutations of the
  trial labels, p = (#{null > obs} + 1) / (1000 + 1) (FieldTrip clusterstat). Tail and the cluster-level alpha: NOT
  STATED (variants: two-tailed at .025 per tail as in the FieldTrip tutorials, two-tailed at FieldTrip's default .05
  per tail, one-tailed positive at .05). Neighbours (FieldTrip version NOT STATED; variant axis 'neighbours'):
  'delaunay' (the projected layout's Delaunay edges; the unsuffixed variant names) or 'fieldtrip2018' (FieldTrip
  ft_prepare_neighbours @ 7f5a1c60, 2018-01-19: x- and y-halved triangulations merged, edges beyond mean + 3 SD
  dropped; suffix _fieldtrip2018).
- Call (cat. §6): the command-following evidence is a P3b effect (top-down attention); P3a and the SSEP are reported
  as variants because the benchmark isolates the SSEP's periodic-train property. One assessment per patient.
The randomization-based counterpart (run_randomized): the target wrist is randomized per block (balanced 7 / 7; the
published order alternates) and the null re-draws it; statistic = the largest positive P3b cluster mass (one-tailed,
all kept trials, no subsampling), B = 500 with decision-exact early stopping."""
import numpy as np
from scipy.signal import detrend
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import f as f_dist, kurtosis, t as t_dist

from . import erp_common as ec
from . import motor_common as mc

NoUsableTrials = mc.NoUsableTrials

NAME = 'gibson16'
GRADE = 'AS DESCRIBED'
FAMILY = 'erp'
NATIVE_CH = ['FC1', 'Fz', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'Pz', 'Oz', 'PO7', 'PO8']
SUB_1020 = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4', 'Pz', 'O1', 'O2', 'T5', 'T6']
TARGET, NONTARGET, BACK = 1, 2, 3
LEFT, RIGHT, UPPER_BACK = 1, 2, 0
N_BLOCKS, SOA_S = 14, 0.2
WRIST_N = (28, 32)
BACK_MIN, BACK_MAX = 3, 21
GAP_S, LEAD_S = 15.0, 3.0
EPOCH_S, PRE_S = 1.0, 0.2
BAND = (0.5, 30.0)
REJECT_UV, KURT_SD = 100.0, 2.5
WIN_MS = (50.0, 750.0)
NUMRAND, CLUSTER_ALPHA = 1000, 0.05
F_CRIT = 3.49
MIN_SSEP, MIN_PER_CLASS = 1, 3
SSEP_HZ = (5, 10)
CALLS = {
    'ssep_avg_nonharm': {'call_on': 'ssep', 'ssep': 'avg', 'bins': 'nonharm'},
    'ssep_avg_plain': {'call_on': 'ssep', 'ssep': 'avg', 'bins': 'plain'},
    'ssep_f5_nonharm': {'call_on': 'ssep', 'ssep': 'f5', 'bins': 'nonharm'},
    'ssep_f5_plain': {'call_on': 'ssep', 'ssep': 'f5', 'bins': 'plain'},
}
for _e in ('p3a', 'p3b'):
    for _tail in ('two025', 'two05', 'onepos'):
        for _nb, _sfx in (('delaunay', ''), ('fieldtrip2018', '_fieldtrip2018')):
            CALLS[f'{_e}_{_tail}{_sfx}'] = {'call_on': _e, 'tail': _tail, 'neighbours': _nb}
_CITE = {'ssep': 'SSEP: averaging of 5 and 10 Hz and the ten noise bins NOT STATED',
         'p3a': 'P3a cluster: tail and cluster-level alpha NOT STATED',
         'p3b': 'P3b cluster: tail and cluster-level alpha NOT STATED'}
_CITE_NB = {'delaunay': '; neighbours: Delaunay of the projected layout',
            'fieldtrip2018': '; neighbours: FieldTrip ft_prepare_neighbours.m @ ' + ec.FIELDTRIP_COMMIT[:8]
                             + ' (2018-01-19) l. 198-211, 313-347 (FieldTrip version NOT STATED)'}
VARIANTS = {k: dict(v, cite='cat. §6; ' + _CITE[v['call_on']] + _CITE_NB.get(v.get('neighbours'), ''))
            for k, v in CALLS.items()}
DEFAULT = 'p3b_two025'
RUN_S = 1130.0


def run_s(variant=DEFAULT):
    return RUN_S



def _back_run(g):
    return int(min(BACK_MAX, BACK_MIN + g.geometric(0.5) - 1))


def _draw(start_s, g, targets):
    on, lab, site, blk = [], [], [], []
    t = float(start_s) + LEAD_S
    for b, tw in enumerate(targets):
        wr = g.permutation(np.r_[np.full(g.integers(WRIST_N[0], WRIST_N[1] + 1), LEFT),
                                 np.full(g.integers(WRIST_N[0], WRIST_N[1] + 1), RIGHT)])
        seq = []
        for w in wr:
            seq += [UPPER_BACK] * _back_run(g) + [int(w)]
        seq += [UPPER_BACK] * _back_run(g)
        for s in seq:
            on.append(t)
            site.append(s)
            lab.append(BACK if s == UPPER_BACK else TARGET if s == tw else NONTARGET)
            blk.append(b)
            t += SOA_S
        t += GAP_S
    n = len(on)
    return ec.Overlay(onsets_s=np.array(on), labels=np.array(lab, int), kind='stim',
                      meta={'block': np.array(blk), 'site': np.array(site), 'target_wrist': np.array(targets),
                            'trial': np.arange(n), 'run': np.zeros(n, int)})


def _fits(ov, start_s):
    return ov.onsets_s.max() + EPOCH_S + 5.0 <= float(start_s) + RUN_S


def schedule(start_s, rng, variant=DEFAULT):
    """Right wrist first, then alternating (AAM p. 5)."""
    g = mc.gen(rng)
    targets = np.array([RIGHT if b % 2 == 0 else LEFT for b in range(N_BLOCKS)])
    while True:
        ov = _draw(start_s, g, targets)
        if _fits(ov, start_s):
            return ov


def schedule_randomized(start_s, rng, variant=DEFAULT):
    """Same timing generator; the target wrist per block in a uniformly random balanced order (7 / 7)."""
    g = mc.gen(rng)
    while True:
        targets = mc.balanced_labels((RIGHT, LEFT), (N_BLOCKS // 2, N_BLOCKS // 2), g)
        ov = _draw(start_s, g, targets)
        if _fits(ov, start_s):
            return ov


def extent(ov):
    return float(ov.onsets_s.min()) - PRE_S - 2.5, float(ov.onsets_s.max()) + EPOCH_S + 2.5



def features(rec, ov):
    """-> dict(ep (n, C, T) kept epochs, keep (per onset), names, which, n_pre, fs)."""
    x, used, which = ec.pick(rec, NATIVE_CH, SUB_1020)
    fs = float(rec.fs)
    seg, i0 = mc.span(x, fs, *extent(ov))
    seg = mc.bandpass(seg, fs, *BAND)
    n_tot = int(round(EPOCH_S * fs))
    n_pre = int(round(PRE_S * fs))
    ep = ec.sample_epochs(seg, fs, np.asarray(ov.onsets_s, float) - i0 / fs, n_pre, n_tot - n_pre)
    ep = detrend(ep, axis=-1, type='linear')
    ep = ep - ep[..., :n_pre + 1].mean(-1, keepdims=True)
    keep = np.abs(ep).max(axis=(1, 2)) <= REJECT_UV
    k = kurtosis(ep, axis=-1, fisher=False).max(1)
    kk = k[keep]
    if kk.size > 2 and kk.std() > 0:
        keep &= (k - kk.mean()) / kk.std() <= KURT_SD
    return {'ep': ep[keep], 'keep': keep, 'names': used, 'which': which, 'n_pre': n_pre, 'fs': fs}


def _feat(rec, ov):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: features(rec, ov))



def noise_bins(f0, n_bins, how):
    """The ten bins next to f0 (1-Hz bins): 'plain' = f0 -+ 1..5; 'nonharm' = the ten nearest bins that are neither
    0 Hz nor a multiple of 5 Hz (ties: lower first)."""
    if how == 'plain':
        return np.r_[np.arange(f0 - 5, f0), np.arange(f0 + 1, f0 + 6)]
    cand = [k for k in range(1, n_bins) if k % 5]
    return np.array(sorted(cand, key=lambda k: (abs(k - f0), k))[:10])


def ssep(ep, names, fs, how='avg', bins='nonharm'):
    """-> (F, p) for the Pz average of all kept trials (AAM p. 8)."""
    avg = ep[:, mc.canon(names).index('Pz')].mean(0)
    P = np.abs(np.fft.rfft(avg)) ** 2
    if how == 'f5':
        F = P[5] / P[noise_bins(5, len(P), bins)].mean()
    else:
        nb = np.unique(np.r_[noise_bins(5, len(P), bins), noise_bins(10, len(P), bins)])
        nb = nb[~np.isin(nb, SSEP_HZ)]
        F = P[list(SSEP_HZ)].mean() / P[nb].mean()
    return float(F), float(f_dist.sf(F, 2, 20))



def _graph(C, T, chan_adj):
    """Node (c, t) -> c * T + t; edges: same channel adjacent samples, neighbouring channels same sample."""
    rows, cols = [], []
    idx = np.arange(C * T).reshape(C, T)
    rows.append(idx[:, :-1].ravel())
    cols.append(idx[:, 1:].ravel())
    for i, j in zip(*np.nonzero(np.triu(chan_adj, 1))):
        rows.append(idx[i])
        cols.append(idx[j])
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(C * T, C * T))
    return (A + A.T).tocsr()


def _max_cluster(tvals, mask, A, sign):
    """Largest sign * (sum of t) over connected supra-threshold clusters (0 if none). tvals, mask: flat."""
    nodes = np.nonzero(mask)[0]
    if len(nodes) == 0:
        return 0.0
    _, lab = connected_components(A[nodes][:, nodes], directed=False)
    return float((sign * np.bincount(lab, weights=tvals[nodes])).max())


def cluster_evidence(tvals, crit, A, sign):
    """Continuous evidence for one tail: the largest sign * cluster mass when a supra-threshold cluster
    exists (>= crit), else the largest sign * t (< crit). Deterministic in the t-map and monotone."""
    mass = _max_cluster(tvals, sign * tvals > crit, A, sign)
    return mass if mass > 0 else float((sign * tvals).max())


def indep_t(S1, Q1, n1, S2, Q2, n2):
    """FieldTrip ft_statfun_indepsamplesT from sums and sums of squares (pooled variance)."""
    m1, m2 = S1 / n1, S2 / n2
    v1 = (Q1 - n1 * m1 ** 2) / (n1 - 1)
    v2 = (Q2 - n2 * m2 ** 2) / (n2 - 1)
    df = n1 + n2 - 2
    varc = (1.0 / n1 + 1.0 / n2) * ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    return (m1 - m2) / np.sqrt(varc)


def cluster_test(a, b, A, tail, numrand, g):
    """Trial-level Monte Carlo cluster-mass test of a vs b (each (n, C*T)). -> dict(p_pos, p_neg, pos, neg)."""
    X = np.concatenate([a, b])
    n1, N = len(a), len(a) + len(b)
    df = N - 2
    crit = t_dist.ppf(1 - CLUSTER_ALPHA / (1 if tail == 'onepos' else 2), df)
    M = np.zeros((numrand + 1, N))
    M[0, :n1] = 1.0
    for i in range(1, numrand + 1):
        M[i, g.permutation(N)[:n1]] = 1.0
    S1, Q1 = M @ X, M @ (X * X)
    tot, totq = X.sum(0), (X * X).sum(0)
    T = indep_t(S1, Q1, n1, tot - S1, totq - Q1, N - n1)
    pos = np.array([_max_cluster(T[i], T[i] > crit, A, 1.0) for i in range(numrand + 1)])
    out = {'pos': pos[0], 'p_pos': float((np.sum(pos[1:] > pos[0]) + 1) / (numrand + 1)),
           'ev_pos': cluster_evidence(T[0], crit, A, 1.0)}
    if tail != 'onepos':
        neg = np.array([_max_cluster(T[i], T[i] < -crit, A, -1.0) for i in range(numrand + 1)])
        out.update(neg=neg[0], p_neg=float((np.sum(neg[1:] > neg[0]) + 1) / (numrand + 1)),
                   ev_neg=cluster_evidence(T[0], crit, A, -1.0))
    return out


def _win(F):
    t_ms = (np.arange(F['ep'].shape[-1]) - F['n_pre']) / F['fs'] * 1000.0
    return np.nonzero((t_ms >= WIN_MS[0]) & (t_ms <= WIN_MS[1]))[0]


def _sig(res, tail):
    if tail == 'onepos':
        return res['p_pos'] <= 0.05 and res['pos'] > 0, res['p_pos']
    a = 0.025 if tail == 'two025' else 0.05
    p = min(res['p_pos'] if res['pos'] > 0 else 1.0, res['p_neg'] if res['neg'] > 0 else 1.0)
    return p <= a, p


def _require(n_min, what, n_on):
    if n_min < MIN_PER_CLASS:
        raise NoUsableTrials(f'gibson16: {what} trials survive the +-{REJECT_UV:g} µV and kurtosis rejection '
                             f'({n_on} epochs; each class needs >= {MIN_PER_CLASS})')


def run(rec, overlay, variant=DEFAULT, rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay)
    lab = np.asarray(overlay.labels)[F['keep']]
    ep = F['ep']
    n_on = len(F['keep'])
    if c['call_on'] == 'ssep':
        if len(ep) < MIN_SSEP:
            raise NoUsableTrials(f'gibson16: {len(ep)} of {n_on} epochs survive the +-{REJECT_UV:g} µV and kurtosis '
                                 f'rejection (the SSEP needs >= {MIN_SSEP})')
        Fv, p = ssep(ep, F['names'], F['fs'], c['ssep'], c['bins'])
        return mc.result(Fv, p, Fv >= F_CRIT, size_stat=Fv, size_stat_cont=Fv, n_trials=len(ep), channels=F['which'])
    W = _win(F)
    X = ep[..., W]
    C, Tw = X.shape[1], X.shape[2]
    A = _graph(C, Tw, ec.triangulation_neighbours(F['names'], c.get('neighbours', 'delaunay')))
    if c['call_on'] == 'p3a':
        a, std = X[np.isin(lab, (TARGET, NONTARGET))], X[lab == BACK]
        _require(min(len(a), len(std)), f'{len(a)} wrist and {len(std)} back', n_on)
        b = std[g.choice(len(std), min(len(a), len(std)), replace=False)]
    else:
        a, b = X[lab == TARGET], X[lab == NONTARGET]
        _require(min(len(a), len(b)), f'{len(a)} target and {len(b)} non-target', n_on)
        n = min(len(a), len(b))
        a = a[np.sort(g.choice(len(a), n, replace=False))]
        b = b[np.sort(g.choice(len(b), n, replace=False))]
    res = cluster_test(a.reshape(len(a), -1), b.reshape(len(b), -1), A, c['tail'], NUMRAND, g)
    call, p = _sig(res, c['tail'])
    cont = res['ev_pos'] if c['tail'] == 'onepos' else max(res['ev_pos'], res['ev_neg'])
    return mc.result(res['pos'], p, call, size_stat=1.0 - p, size_stat_cont=cont, n_a=len(a), n_b=len(b),
                     channels=F['which'],
                     cluster=res)


def run_randomized(rec, overlay, variant=DEFAULT, rng=None):
    """Largest positive P3b cluster mass (one-tailed, clusteralpha .05, all kept target / non-target trials); the
    null re-draws the target wrist of every block (balanced 7 / 7), B = 500 with decision-exact early stopping."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay)
    keep = F['keep']
    site = np.asarray(overlay.meta['site'])[keep]
    blk = np.asarray(overlay.meta['block'])[keep]
    tw = np.asarray(overlay.meta['target_wrist'])
    nb = len(tw)
    lo = sum(min(np.sum((blk == b) & (site == LEFT)), np.sum((blk == b) & (site == RIGHT))) for b in range(nb))
    if lo < MIN_PER_CLASS:
        raise NoUsableTrials(f'gibson16: {int(keep.sum())} of {len(keep)} epochs survive rejection; a target-wrist '
                             f'draw can leave {lo} trials in a class (each needs >= {MIN_PER_CLASS})')
    W = _win(F)
    X = F['ep'][..., W]
    C, Tw = X.shape[1], X.shape[2]
    X = X.reshape(len(X), -1)
    A = _graph(C, Tw, ec.triangulation_neighbours(F['names'], c.get('neighbours', 'delaunay')))
    S = {w: np.stack([X[(blk == b) & (site == w)].sum(0) for b in range(nb)]) for w in (LEFT, RIGHT)}
    Q = {w: np.stack([(X[(blk == b) & (site == w)] ** 2).sum(0) for b in range(nb)]) for w in (LEFT, RIGHT)}
    N = {w: np.array([np.sum((blk == b) & (site == w)) for b in range(nb)], float) for w in (LEFT, RIGHT)}
    df_n = sum(N[w].sum() for w in N) - 2
    crit = t_dist.ppf(1 - CLUSTER_ALPHA, df_n)

    def tmap(targets):
        r = (np.asarray(targets) == RIGHT).astype(float)

        def pick(D, w, o):
            return r @ D[w] + (1 - r) @ D[o]
        return indep_t(pick(S, RIGHT, LEFT), pick(Q, RIGHT, LEFT), pick(N, RIGHT, LEFT),
                       pick(S, LEFT, RIGHT), pick(Q, LEFT, RIGHT), pick(N, LEFT, RIGHT))

    def stat(targets):
        T = tmap(targets)
        return _max_cluster(T, T > crit, A, 1.0)

    obs = stat(tw)
    n_r = int(np.sum(tw == RIGHT))
    t = mc.perm_test(obs, lambda: stat(mc.balanced_labels((RIGHT, LEFT), (n_r, nb - n_r), g)),
                     c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'], size_stat=obs,
                     size_stat_cont=cluster_evidence(tmap(tw), crit, A, 1.0))


def stat_for_size_correction(result):
    """size_stat_cont: F (SSEP); for P3a / P3b the observed cluster evidence of the call's own t-map
    (cluster_evidence: positive cluster mass, else the largest t; two-tailed variants take the larger of the two
    tails' evidence); run_randomized: the positive-tail evidence. Falls back to the earlier size_stat."""
    e = result['extra']
    return float(e.get('size_stat_cont', e.get('size_stat', result['stat'])))


def patient_rule(results):
    """One assessment per patient (cat. §6); for k > 1 'any' (NOT STATED). GIB16's 'any of three non-EEG
    assessments' is its reference standard, not an EEG call rule."""
    return any(r['call'] for r in results)
