"""Cornell (Schiff lab) spectral task-vs-rest analysis, Goldfine 2011 / Curley 2018. CORE-EXACT for the statistic
(the Chronux Two Group Test, re-implemented from the catalogue's reading of two_group_test_spectrum.m), AS DESCRIBED
for the rest. Spec: the motor catalogue §4.
- Design (§4.1): a trial = an 'imagine ...' command and a 'stop imagining ...' command 15 s apart; commands ~2 s;
  8 trials = one run (~4 min); 2-4 runs per paradigm.
- Signal (§4.2): 1-50 Hz; Hjorth Laplacian (4 nearest neighbours inside the array, 3 on the edge, weights = 1/arc
  length on a sphere); the response period 1-10 s after the end of each command, detrended, cut into three 3-s
  epochs (24 per condition per run); visual epoch pruning (56-65% of epochs rejected, comparable for task and rest),
  REPLACED here by a deterministic automated rule (a variant, below).
- Spectra (§4.2): multitaper, 5 tapers (TW = 3 on 3 s: 2-Hz resolution), estimates 1/3 Hz apart (nfft = N), 4-24 Hz.
- Two Group Test (§4.3 [code]): samples = epochs x tapers; z = log(mean |J|^2) - (psi(m) - log m); dz = (z1 - z2) /
  sqrt(psi'(m1) + psi'(m2)). Chronux's decision Adz is two-sided at p = 0.05 on dz with this THEORETICAL variance;
  the jackknife variance vdz is computed but unused there, while the papers say 'p <= 0.05 by jackknife method'.
  Both are variants. vdz here = Arvesen two-sample jackknife: leave one sample out of either group (bias and
  psi' at m - 1), (m - 1)/m * sum of squared deviations per group, summed.
- Contiguity (§4.3): a difference counts only if significant at every bin over a range > 2 Hz (last minus first bin
  frequency); narrower = 'trend'.
- OM1 (§4.4): one run has a channel with a significant (> 2 Hz) difference, and a second run shows at least a trend in
  the same channel and frequency range. OM2: all runs combined, at least one difference significant after FDR (0.05)
  over all channels x frequencies. Positive = OM1 and OM2; indeterminate = OM1 only.
- Patient (§4.4): positive if >= 1 paradigm is positive in >= 1 assessment -> any over the patient's results.
  CU18's added later regrouping of runs by state (§4.4, exploratory): extra['call_regrouped'], patient_rule_regrouped.
Multiverse (VARIANTS): variance {theoretical (Chronux Adz, as coded), jackknife (papers)} x FDR {BH 1995, BY 2001}
(§4.3 NOT STATED) x pruning {thresh, keep40} (§4.2 manual) x runs {2, 4} (§4.1 '2 to 4 runs')."""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import detrend
from scipy.signal.windows import dpss
from scipy.special import polygamma, psi
from scipy.stats import norm

from . import motor_common as mc

NAME = 'cornell'
GRADE = 'CORE-EXACT'
FAMILY = 'cmd'
NATIVE_CH = list(mc.EEG19)
TASK, REST = 1, 0
CMD_S, SPACING_S, TRIALS_PER_RUN, RUN_GAP_S = 2.0, 15.0, 8, 30.0
RESP = (1.0, 10.0)
EPOCH_S, N_EP = 3.0, 3
NW, K = 3.0, 5
FPASS = (4.0, 24.0)
P_TGT, Q_FDR, MIN_RANGE_HZ = 0.05, 0.05, 2.0
KEEP_FRAC = 0.4
MIN_EPOCHS, MIN_RUNS = 2, 2
RESPONDER = {'resp_hand': {TASK: 'both', REST: 'none'}, 'resp_delay_s': CMD_S, 'resp_dur_s': 13.0}
RUN_S = 4 * TRIALS_PER_RUN * 2 * SPACING_S + 3 * RUN_GAP_S + 15.0


def run_s(variant):
    """Assessment length for one variant (RUN_S is the maximum over variants)."""
    n = mc.cfg(VARIANTS, variant)['n_runs']
    return n * TRIALS_PER_RUN * 2 * SPACING_S + (n - 1) * RUN_GAP_S + 15.0

FIXED = {
    'channels': '19 10-20 sites (CU18 37, GF11 29/37; names NOT STATED)',
    'timing': 'GF11 timing: 2-s commands every 15 s (CU18: 2-4 s commands, 10-14 s silence); response window from '
              'CU18 (1-10 s after the command end) for both',
    'run_gap': f'{RUN_GAP_S:.0f} s between runs (NOT STATED)',
    'filter': '1-50 Hz zero-phase 4th-order Butterworth on the continuous signal before the Laplacian (CU18 filters '
              'the concatenated clean epochs; filtering continuous data avoids splice edges)',
    'ica': 'GF11 ICA and 60-Hz rmlinesc not applied (CU18 uses no ICA; BDSP data are not 60-Hz contaminated in the '
           '4-24 Hz band)',
    'fpass': '4-24 Hz (GF11 §2.7; CU18 range NOT STATED): 61 bins at 1/3 Hz',
    'prune_thresh': 'reject an epoch if any channel has |x| > 200 µV, SD < 0.5 µV, or a log-variance robust z > 3 '
                    '(median/MAD over the run\'s epochs) on the referential 1-50 Hz signal',
    'prune_keep40': 'per run and condition keep the 40% of epochs (10 of 24) with the lowest max-channel log-variance '
                    'robust z (CU18 rejected 56-65%, comparable for task and rest), chosen among epochs with no '
                    'flat (SD < 0.5 µV) or saturated (|x| > 500 µV, the engine\'s value) channel: a flat epoch has '
                    'the lowest score of all',
    'sign': 'contiguous runs of significant bins must share the sign of dz; the second-run trend must have the '
            'same sign and overlap the first run\'s significant range',
    'om2': "'at least one of the individual spectral differences [...] remained significant after FDR': any bin "
           'rejected by FDR over all channels x frequencies of the combined-runs TGT',
    'regroup': 'CU18 added later state regrouping (visual): runs split into 2 groups by average-linkage clustering of '
               'their mean log spectra (all kept epochs, both conditions); OM1 + OM2 re-applied to each group of '
               '>= 2 runs (only possible with >= 3 runs); only runs meeting the minimum take part',
    'minimum': f'OM1 needs two runs with a TGT: >= {MIN_RUNS} runs keeping >= {MIN_EPOCHS} task and >= {MIN_EPOCHS} '
               'rest epochs after pruning; fewer -> mc.NoUsableTrials (the same rule gates run_randomized, which '
               're-draws, never zero-fills, flip sets that fall below it)',
}
VARIANTS = {f'{var}_{fdr}_{pr}_r{nr}': {'var': var, 'fdr': fdr, 'prune': pr, 'n_runs': nr,
                                        'as_coded': var == 'theoretical',
                                        'cite': '§4.3 variance (code vs papers), FDR NOT STATED; §4.2 pruning '
                                                'manual; §4.1 2-4 runs'}
            for var in ('theoretical', 'jackknife') for fdr in ('bh', 'by') for pr in ('thresh', 'keep40')
            for nr in (2, 4)}



def schedule(start_s, rng=None, variant='theoretical_bh_keep40_r4'):
    c = mc.cfg(VARIANTS, variant)
    run_len = TRIALS_PER_RUN * 2 * SPACING_S
    on, lab, trial, run = [], [], [], []
    for r in range(c['n_runs']):
        t0 = float(start_s) + r * (run_len + RUN_GAP_S)
        for k in range(TRIALS_PER_RUN):
            for j, l in enumerate((TASK, REST)):
                on.append(t0 + (2 * k + j) * SPACING_S)
                lab.append(l)
                trial.append(r * TRIALS_PER_RUN + k)
                run.append(r)
    return mc.Overlay(onsets_s=np.array(on), labels=np.array(lab), kind='cmd',
                      meta={'trial': np.array(trial), 'block': np.array(run), 'run': np.array(run), **RESPONDER})


def schedule_randomized(start_s, rng, variant='theoretical_bh_keep40_r4'):
    """Same timing; within each trial the task and rest commands swap places with probability 0.5."""
    ov = schedule(start_s, None, variant)
    flip = mc.gen(rng).integers(0, 2, len(ov.labels) // 2).astype(bool)
    lab = ov.labels.reshape(-1, 2).copy()
    lab[flip] = lab[flip][:, ::-1]
    ov.labels = lab.ravel()
    ov.meta['flipped'] = flip
    return ov



def extent(ov):
    """Absolute (t0, t1) seconds of the recording this overlay makes run/run_randomized read; must be clean."""
    return float(ov.onsets_s.min()) - 5.0, float(ov.onsets_s.max()) + CMD_S + RESP[1] + 5.0



def features(rec, ov):
    """-> dict(S (n_epochs, K, ch, f) |J|^2, cmd (command of each epoch), score (artefact score), hard (thresh
    reject), f (Hz))."""
    x = np.asarray(rec.x, float)
    nm = mc.canon(rec.names)
    x = x[[nm.index(c) for c in NATIVE_CH]]
    seg, i0 = mc.span(x, rec.fs, *extent(ov))
    seg = mc.bandpass(seg, rec.fs, 1.0, 50.0)
    lap = mc.hjorth_matrix(NATIVE_CH, 4, 3, 'inverse_arc') @ seg
    onl = ov.onsets_s - i0 / rec.fs + CMD_S
    resp = detrend(mc.epochs(lap, rec.fs, onl, *RESP), axis=-1)
    raw = mc.epochs(seg, rec.fs, onl, *RESP)
    n = int(round(EPOCH_S * rec.fs))
    ep = np.concatenate([resp[:, None, :, k * n:(k + 1) * n] for k in range(N_EP)], 1).reshape(-1, len(NATIVE_CH), n)
    rw = np.concatenate([raw[:, None, :, k * n:(k + 1) * n] for k in range(N_EP)], 1).reshape(-1, len(NATIVE_CH), n)
    cmd = np.repeat(np.arange(len(ov.onsets_s)), N_EP)
    tap = dpss(n, NW, K)
    f = np.fft.rfftfreq(n, 1 / rec.fs)
    fm = (f >= FPASS[0] - 1e-9) & (f <= FPASS[1] + 1e-9)
    J = np.fft.rfft(ep[:, None] * tap[None, :, None], n=n, axis=-1)[..., fm] / rec.fs
    S = (np.abs(J) ** 2).astype(np.float64)
    lv = np.log(rw.var(-1) + 1e-12)
    run = np.asarray(ov.meta['run'])[cmd]
    score = np.zeros(len(ep))
    for r in np.unique(run):
        m = run == r
        med = np.median(lv[m], 0)
        mad = 1.4826 * np.median(np.abs(lv[m] - med), 0) + 1e-9
        score[m] = ((lv[m] - med) / mad).max(1)
    flat, peak = rw.std(-1).min(1) < 0.5, np.abs(rw).max(axis=(1, 2))
    return {'S': S, 'cmd': cmd, 'run': run, 'score': score, 'dead': flat | (peak > 500.0),
            'hard': flat | (peak > 200.0) | (score > 3.0), 'f': f[fm]}


def kept(F, lab_ep, prune):
    if prune == 'thresh':
        return ~F['hard']
    keep = np.zeros(len(lab_ep), bool)
    for r in np.unique(F['run']):
        for l in (TASK, REST):
            i = np.nonzero((F['run'] == r) & (lab_ep == l))[0]
            n = int(round(KEEP_FRAC * len(i)))
            i = i[~F['dead'][i]]
            keep[i[np.argsort(F['score'][i], kind='stable')[:n]]] = True
    return keep


def usable_runs(F, lab_ep, keep, runs):
    """Runs keeping >= MIN_EPOCHS task and >= MIN_EPOCHS rest epochs (a per-run TGT exists)."""
    return [r for r in runs if all(np.sum(keep & (F['run'] == r) & (lab_ep == l)) >= MIN_EPOCHS for l in (TASK, REST))]


def _require(F, lab_ep, keep):
    ok = usable_runs(F, lab_ep, keep, sorted(np.unique(F['run']).tolist()))
    if len(ok) < MIN_RUNS:
        raise mc.NoUsableTrials(f'cornell: {len(ok)} runs keep >= {MIN_EPOCHS} task and >= {MIN_EPOCHS} rest epochs '
                                f'after pruning, OM1 needs {MIN_RUNS} ({int(keep.sum())}/{len(keep)} epochs kept)')
    return ok



def tgt(S1, S2):
    """S1 (n1, K, ch, f), S2 likewise -> dz, vdz (ch, f); samples = epochs x tapers."""
    A, B = S1.reshape(-1, *S1.shape[2:]), S2.reshape(-1, *S2.shape[2:])
    m1, m2 = len(A), len(B)
    s1, s2 = A.sum(0), B.sum(0)
    z1 = np.log(s1 / m1) - (psi(m1) - np.log(m1))
    z2 = np.log(s2 / m2) - (psi(m2) - np.log(m2))
    sd = np.sqrt(polygamma(1, m1) + polygamma(1, m2))
    dz = (z1 - z2) / sd
    z1i = np.log((s1 - A) / (m1 - 1)) - (psi(m1 - 1) - np.log(m1 - 1))
    z2j = np.log((s2 - B) / (m2 - 1)) - (psi(m2 - 1) - np.log(m2 - 1))
    d1, d2 = (z1i - z2) / sd, (z1 - z2j) / sd
    vdz = (m1 - 1) / m1 * ((d1 - d1.mean(0)) ** 2).sum(0) + (m2 - 1) / m2 * ((d2 - d2.mean(0)) ** 2).sum(0)
    return dz, vdz


def zscore(dz, vdz, var):
    return dz if var == 'theoretical' else dz / np.sqrt(vdz)


def regions(sig, sign, f):
    """Maximal runs of significant same-sign bins per channel: list of (ch, i0, i1, sign, range_hz)."""
    out = []
    for ch in range(sig.shape[0]):
        i = 0
        while i < sig.shape[1]:
            if sig[ch, i]:
                j = i
                while j + 1 < sig.shape[1] and sig[ch, j + 1] and sign[ch, j + 1] == sign[ch, i]:
                    j += 1
                out.append((ch, i, j, sign[ch, i], f[j] - f[i]))
                i = j + 1
            else:
                i += 1
    return out


def outcome(F, lab_ep, keep, runs, c):
    """OM1, OM2 on the given runs. -> dict(om1, om2, zmax, qmin)."""
    zc = 1.959963984540054
    sigs = {}
    for r in runs:
        m1 = keep & (F['run'] == r) & (lab_ep == TASK)
        m2 = keep & (F['run'] == r) & (lab_ep == REST)
        if m1.sum() < 2 or m2.sum() < 2:
            continue
        z = zscore(*tgt(F['S'][m1], F['S'][m2]), c['var'])
        sigs[r] = (np.abs(z) > zc, np.sign(z))
    om1 = False
    for r1, (sg, sn) in sigs.items():
        for ch, i0, i1, s, rng_hz in regions(sg, sn, F['f']):
            if rng_hz <= MIN_RANGE_HZ:
                continue
            for r2, (sg2, sn2) in sigs.items():
                if r2 != r1 and np.any(sg2[ch, i0:i1 + 1] & (sn2[ch, i0:i1 + 1] == s)):
                    om1 = True
    inr = np.isin(F['run'], list(runs))
    m1, m2 = keep & inr & (lab_ep == TASK), keep & inr & (lab_ep == REST)
    if m1.sum() < 2 or m2.sum() < 2:
        return {'om1': om1, 'om2': False, 'zmax': 0.0, 'qmin': 1.0}
    z = zscore(*tgt(F['S'][m1], F['S'][m2]), c['var'])
    p = 2 * norm.sf(np.abs(z)).ravel()
    rej = mc.fdr_reject(p, Q_FDR, c['fdr'])
    m = len(p)
    cm = 1.0 if c['fdr'] == 'bh' else np.sum(1.0 / np.arange(1, m + 1))
    ps = np.sort(p)
    q = np.minimum.accumulate((ps * m * cm / np.arange(1, m + 1))[::-1])[::-1]
    return {'om1': om1, 'om2': bool(rej.any()), 'zmax': float(np.abs(z).max()), 'qmin': float(min(1.0, q[0]))}


def _feat(rec, ov):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float)), lambda: features(rec, ov))


def _regroup(F, keep, runs, c):
    runs = usable_runs(F, F['lab'], keep, runs)
    if len(runs) < 3:
        return False, None
    spec = np.stack([np.log(F['S'][keep & (F['run'] == r)].mean((0, 1))).ravel() for r in runs])
    grp = fcluster(linkage(spec, 'average'), 2, 'maxclust')
    pos = False
    for gi in np.unique(grp):
        sub = [r for r, gg in zip(runs, grp) if gg == gi]
        if 2 <= len(sub) < len(runs):
            o = outcome(F, F['lab'], keep, sub, c)
            pos |= o['om1'] and o['om2']
    return pos, grp.tolist()


def run(rec, overlay, variant='theoretical_bh_keep40_r4', rng=None):
    c = mc.cfg(VARIANTS, variant)
    F = dict(_feat(rec, overlay))
    lab_ep = np.asarray(overlay.labels)[F['cmd']]
    F['lab'] = lab_ep
    keep = kept(F, lab_ep, c['prune'])
    _require(F, lab_ep, keep)
    runs = sorted(np.unique(F['run']).tolist())
    o = outcome(F, lab_ep, keep, runs, c)
    call = o['om1'] and o['om2']
    reg, grp = _regroup(F, keep, runs, c)
    return mc.result(o['zmax'], o['qmin'], call, om1=o['om1'], om2=o['om2'],
                     indeterminate=bool(o['om1'] and not o['om2']), call_regrouped=bool(call or reg),
                     regroup=grp, n_kept=int(keep.sum()), n_epochs=int(len(keep)))


def run_randomized(rec, overlay, variant='theoretical_bh_keep40_r4', rng=None):
    """Same spectra, pruning and TGT; statistic = max |z| over channels x frequencies of the all-runs TGT (the OM2
    evidence, variance per variant; pruning recomputed per draw); the null re-draws the per-trial task/rest flips,
    B = 500, early stopping. A flip set that leaves fewer than MIN_RUNS usable runs (the minimum the observed
    labelling meets) is re-drawn: a conditional randomization test over the admissible flip sets."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay)
    base = schedule(0.0, None, {**c, 'n_runs': int(np.max(overlay.meta['run'])) + 1}).labels
    runs = sorted(np.unique(F['run']).tolist())

    def ok(lab_cmd):
        lab_ep = lab_cmd[F['cmd']]
        return len(usable_runs(F, lab_ep, kept(F, lab_ep, c['prune']), runs)) >= MIN_RUNS

    def stat(lab_cmd):
        lab_ep = lab_cmd[F['cmd']]
        keep = kept(F, lab_ep, c['prune'])
        m1, m2 = keep & (lab_ep == TASK), keep & (lab_ep == REST)
        return float(np.abs(zscore(*tgt(F['S'][m1], F['S'][m2]), c['var'])).max())

    def flips():
        lab = base.reshape(-1, 2).copy()
        flip = g.integers(0, 2, len(lab)).astype(bool)
        lab[flip] = lab[flip][:, ::-1]
        return lab.ravel()

    lab0 = np.asarray(overlay.labels)[F['cmd']]
    _require(F, lab0, kept(F, lab0, c['prune']))
    obs = stat(np.asarray(overlay.labels))
    t = mc.perm_test(obs, lambda: mc.admissible_stat(flips, ok, stat), c.get('n_perm_rand', 500), 0.05)
    return mc.result(obs, t['p'], t['p'] < 0.05, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_nonfinite_null=t['n_nonfinite_null'])


def stat_for_size_correction(result):
    return result['stat']


def patient_rule(results):
    """>= 1 paradigm positive in >= 1 assessment (CU18 p. 1410, §4.4)."""
    return any(r['call'] for r in results)


def patient_rule_regrouped(results):
    """CU18 added later regrouping by state (§4.4)."""
    return any(r['extra'].get('call_regrouped', r['call']) for r in results)
