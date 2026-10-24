"""Edlow et al. 2017 Brain (MGH) hand-squeeze imagery, AS DESCRIBED. Spec: the motor catalogue §5.
- Design (§5.1): 12 squeeze + 12 rest blocks of 24 s; a spoken instruction opens each block and is repeated every 6 s
  ('keep squeezing' / 'keep resting'). One session.
- Signal (§5.2): 19 electrodes; 1-30 Hz third-order zero-phase Butterworth; Hjorth Laplacian; Chronux spectrogram
  with T = 1 s, TW = 1 (one Slepian taper, 1-Hz resolution); absolute power averaged in delta 1-3, theta 4-7,
  alpha 8-13, beta 14-30 Hz -> 76 features per second; the data during repeated instructions and the first second
  after them are excluded (motor matrix 76 x 388 s instead of 576).
- Classifier (§5.3): linear SVM, squeeze vs rest per 1-s column; 20-fold CV on random partitions, repeated 10 times,
  accuracy = mean of the 10. Permutation test with 500 permutations of the on/off labels, the full CV repeated;
  p = #{perm >= observed} / 500 (no +1); positive if p < 0.05. Decision-exact early stopping (p >= 25/500 fixed).
Multiverse (VARIANTS): permutation unit {second, block} (§5.3 NOT STATED) x CV partition unit {second, block}
(§5.3 NOT STATED) x block order {alternating, random} (§5.1 NOT STATED for EEG).
Fixed choices for the other NOT STATED items are in FIXED. The linear SVM is solved for all 200 folds at once by an
exact primal Newton method for the squared-hinge (L2-loss) SVM with a regularized bias, the objective liblinear's
LinearSVC(dual=False) solves (tests check the agreement); folds that do not converge are refitted with liblinear."""
import numpy as np
from scipy.signal.windows import dpss
from sklearn.svm import LinearSVC

from . import motor_common as mc

NAME = 'edlow17'
GRADE = 'AS DESCRIBED'
FAMILY = 'cmd'
NATIVE_CH = list(mc.EEG19)
SQUEEZE, REST = 1, 0
N_BLOCKS, BLOCK_S, CYCLE_S = 24, 24, 6
DROP_PHASE = (0, 1)
BANDS = ((1, 3), (4, 7), (8, 13), (14, 30))
N_FOLD, N_REP, N_PERM, ALPHA = 20, 10, 500, 0.05
C_SVM = 1.0
MIN_BLOCKS = 2
RUN_S = N_BLOCKS * BLOCK_S + 10.0

FIXED = {
    'instructions': 'all four instructions per block (at 0, 6, 12, 18 s) are dropped with the following second (1-s '
                    'instruction assumed): 16 s kept per block, 384 s in all vs the reported 388 [derived; the '
                    'instruction length is NOT STATED]',
    'laplacian': 'Hjorth with 4 (interior) / 3 (edge) nearest sites, inverse-arc weights (the GF11 definition, §4.2; '
                 "ED17 says only 'Hjorth Laplacian')",
    'spectrogram': '1-s columns aligned to the block start, step 1 s (576 columns for 576 s [derived, §5.2]); one '
                   'DPSS taper NW = 1; Chronux pad 0: nfft = next power of 2; bins lo <= f <= hi',
    'artefacts': 'ICA by a neuropsychologist and consensus review (§5.2): automated, drop a 1-s column if any channel '
                 'exceeds |200| µV after 1-30 Hz filtering',
    'scaling': 'features z-scored over all kept columns (label-free; NOT STATED, §5.3)',
    'svm': 'C = 1, squared-hinge linear SVM with a regularized bias (liblinear primal); C, loss, scaling NOT STATED',
    'block_order_alt': 'rest first, then alternating (ED17-S instructions start with rest)',
    'block_cv': '24 blocks randomly split into 20 groups (four groups of two blocks)',
    'minimum': f'>= {N_FOLD} surviving 1-s columns (one per CV fold) and surviving columns in >= {MIN_BLOCKS} blocks '
               'of each class (so block-wise folds always train on both classes); fewer -> mc.NoUsableTrials; '
               'run_randomized re-draws (never zero-fills) block orders that fall below it',
}
VARIANTS = {f'perm{pu}_cv{cu}_{order}': {'perm_unit': pu, 'cv_unit': cu, 'order': order,
                                         'cite': '§5.3 permutation and CV units NOT STATED; §5.1 order NOT STATED'}
            for pu in ('second', 'block') for cu in ('second', 'block') for order in ('alt', 'random')}



def schedule(start_s, rng, variant='permsecond_cvsecond_alt'):
    c = mc.cfg(VARIANTS, variant)
    if c['order'] == 'alt':
        cls = np.tile([REST, SQUEEZE], N_BLOCKS // 2)
    else:
        cls = mc.balanced_labels((REST, SQUEEZE), (N_BLOCKS // 2, N_BLOCKS // 2), rng)
    return _overlay(start_s, cls)


def _overlay(start_s, cls):
    blk = np.repeat(np.arange(N_BLOCKS), BLOCK_S // CYCLE_S)
    on = float(start_s) + blk * BLOCK_S + np.tile(np.arange(0, BLOCK_S, CYCLE_S), N_BLOCKS)
    return mc.Overlay(onsets_s=on.astype(float), labels=np.asarray(cls)[blk], kind='cmd',
                      meta={'trial': np.arange(len(on)), 'block': blk, 'run': np.zeros(len(on), int),
                            'block_class': np.asarray(cls), 'block_start': np.r_[True, np.diff(blk) > 0],
                            'resp_hand': {SQUEEZE: 'right', REST: 'none'}, 'resp_delay_s': 0.0,
                            'resp_dur_s': float(CYCLE_S)})


def schedule_randomized(start_s, rng, variant='permsecond_cvsecond_alt'):
    """Same timing; the class of each block is randomized (12 of each, uniform order)."""
    return _overlay(start_s, mc.balanced_labels((REST, SQUEEZE), (N_BLOCKS // 2, N_BLOCKS // 2), rng))



def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized read; must be clean."""
    starts = np.asarray(ov.onsets_s)[ov.meta['block_start']]
    return float(starts[0]) - 5.0, float(starts[-1]) + BLOCK_S + 5.0



def features(rec, ov):
    """-> (X (n_cols, 76) z-scored, block id per column, keep mask over the 576 columns)."""
    x = np.asarray(rec.x, float)
    nm = mc.canon(rec.names)
    x = x[[nm.index(c) for c in NATIVE_CH]]
    starts = ov.onsets_s[ov.meta['block_start']]
    seg, i0 = mc.span(x, rec.fs, *extent(ov))
    seg = mc.bandpass(seg, rec.fs, 1.0, 30.0, order=3)
    lap = mc.hjorth_matrix(NATIVE_CH) @ seg
    col_on = (starts[:, None] + np.arange(BLOCK_S)[None]).ravel() - i0 / rec.fs
    n = int(round(rec.fs))
    ep_raw = mc.epochs(seg, rec.fs, col_on, 0.0, 1.0)
    ep = mc.epochs(lap, rec.fs, col_on, 0.0, 1.0)
    tap = dpss(n, 1.0, 1)[0]
    nfft = int(2 ** np.ceil(np.log2(n)))
    S = np.abs(np.fft.rfft(ep * tap, n=nfft, axis=-1) / rec.fs) ** 2
    f = np.fft.rfftfreq(nfft, 1 / rec.fs)
    P = np.stack([S[..., (f >= lo) & (f <= hi)].mean(-1) for lo, hi in BANDS], -1)
    phase = np.tile(np.arange(BLOCK_S) % CYCLE_S, N_BLOCKS)
    keep = ~np.isin(phase, DROP_PHASE) & (np.abs(ep_raw).max(axis=(1, 2)) <= 200.0)
    X = P.reshape(len(P), -1)[keep]
    if len(X):
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    return X, np.repeat(np.arange(N_BLOCKS), BLOCK_S)[keep], keep



def l2svm_fit(Xa, s, masks, C=C_SVM, iters=40, tol=1e-8):
    """min 0.5|w|^2 + C sum_{i in fold} max(0, 1 - s_i w.x_i)^2 for every fold (rows of masks) at once, by exact
    Newton steps (Xa carries the bias column). Folds whose gradient does not vanish are refitted with liblinear."""
    F, D = masks.shape[0], Xa.shape[1]
    W = np.zeros((F, D))
    eye = np.eye(D)
    for _ in range(iters):
        marg = 1.0 - (W @ Xa.T) * s
        A = (marg > 0) & masks
        g = W - 2 * C * (np.where(A, marg, 0.0) * s) @ Xa
        H = eye + 2 * C * np.matmul((A[:, :, None] * Xa[None]).transpose(0, 2, 1), Xa)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        W -= step
        if np.abs(step).max() < tol:
            break
    marg = 1.0 - (W @ Xa.T) * s
    A = (marg > 0) & masks
    g = W - 2 * C * (np.where(A, marg, 0.0) * s) @ Xa
    for f in np.nonzero(np.abs(g).max(1) > 1e-6)[0]:
        if len(np.unique(s[masks[f]])) < 2:
            continue
        m = LinearSVC(C=C, dual=False, tol=1e-10, max_iter=10000).fit(Xa[masks[f], :-1], s[masks[f]])
        W[f] = np.r_[m.coef_[0], m.intercept_]
    return W


def cv_partitions(n_cols, blocks, unit, rng):
    """N_REP x N_FOLD test-fold ids per column: random partitions of columns or of blocks."""
    g = mc.gen(rng)
    out = []
    for _ in range(N_REP):
        if unit == 'second':
            out.append(g.permutation(n_cols) % N_FOLD)
        else:
            ub = np.unique(blocks)
            bf = np.empty(ub.max() + 1, int)
            bf[g.permutation(ub)] = np.arange(len(ub)) % N_FOLD
            out.append(bf[blocks])
    return np.array(out)


def cv_accuracy(X, y, folds):
    """Mean over repetitions of the CV accuracy (every column tested once per repetition)."""
    Xa = np.c_[X, np.ones(len(X))]
    s = np.where(y == SQUEEZE, 1.0, -1.0)
    test = np.concatenate([folds[r][None] == np.arange(N_FOLD)[:, None] for r in range(len(folds))])
    test = test[test.any(1)]
    W = l2svm_fit(Xa, s, ~test)
    correct = ((W @ Xa.T) * s > 0) & test
    return float(correct.sum() / test.sum())


def _feat(rec, ov):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: features(rec, ov))


def admissible(bcls, blocks):
    return all(len(np.unique(blocks[bcls[blocks] == cls])) >= MIN_BLOCKS for cls in (REST, SQUEEZE))


def require(bcls, blocks):
    """bcls: class of each of the 24 blocks; blocks: block id of each surviving 1-s column."""
    if len(blocks) < N_FOLD:
        raise mc.NoUsableTrials(f'edlow17: {len(blocks)} 1-s column(s) survive the 200-µV rule, >= {N_FOLD} needed')
    for name, cls in (('rest', REST), ('squeeze', SQUEEZE)):
        k = len(np.unique(blocks[bcls[blocks] == cls]))
        if k < MIN_BLOCKS:
            raise mc.NoUsableTrials(f'edlow17: {k} {name} block(s) keep a 1-s column, >= {MIN_BLOCKS} needed')


def run(rec, overlay, variant='permsecond_cvsecond_alt', rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    X, blocks, keep = _feat(rec, overlay)
    bcls = np.asarray(overlay.meta['block_class'])
    require(bcls, blocks)
    y = bcls[blocks]
    obs = cv_accuracy(X, y, cv_partitions(len(y), blocks, c['cv_unit'], g))

    def draw():
        yp = g.permutation(y) if c['perm_unit'] == 'second' else g.permutation(bcls)[blocks]
        return cv_accuracy(X, yp, cv_partitions(len(y), blocks, c['cv_unit'], g))

    t = mc.perm_test(obs, draw, c.get('n_perm', N_PERM), ALPHA, plus_one=False)
    return mc.result(obs, t['p'], t['p'] < ALPHA, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'], n_cols=int(len(y)))


def run_randomized(rec, overlay, variant='permsecond_cvsecond_alt', rng=None):
    """The variant's features, SVM and CV unit; statistic = mean CV accuracy; the null re-draws the block-level
    randomization (12 + 12, uniform order), B = 500, p = (count + 1)/(B + 1), early stopping. An order leaving a
    class below the minimum among the surviving blocks is re-drawn (conditional randomization test)."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    X, blocks, keep = _feat(rec, overlay)
    bcls = np.asarray(overlay.meta['block_class'])
    require(bcls, blocks)
    stat = lambda cls: cv_accuracy(X, cls[blocks], cv_partitions(len(blocks), blocks, c['cv_unit'], g))
    draw = lambda: mc.balanced_labels((REST, SQUEEZE), (N_BLOCKS // 2, N_BLOCKS // 2), g)
    obs = stat(bcls)
    t = mc.perm_test(obs, lambda: mc.admissible_stat(draw, lambda cls: admissible(cls, blocks), stat),
                     c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'], n_cols=int(len(blocks)))


def stat_for_size_correction(result):
    return result['stat']


def patient_rule(results):
    """One session per patient (§5.1); across sessions any positive (BOD24 'best response', §10)."""
    return any(r['call'] for r in results)
