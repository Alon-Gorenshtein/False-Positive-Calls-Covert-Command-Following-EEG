"""Sokoliuk 2021 (Ann Neurol 89:646) passive speech tracking: ITPC at the word / phrase / sentence rates. CORE-EXACT:
Script_2 (ITPC) and Script_3 (frequencies of interest, chance-frequency surrogate, FDR) are ported line by line from
the authors' OSF code (wu2vy, CC BY 4.0) and reproduce their saved ITPC
outputs (tests); Script_1's filters are ported, its visual artefact rejection and ICA component removal are manual
and are replaced by the automated choices below. Spec: the erp catalogue §8 (cat. §8).
- Design (cat. §8; Paradigm/CorticalTracking_Paradigm.m l. 60-75, 185-215): 72 trials of 12 four-word sentences
  (0.32-s words: 3.125 / 1.5625 / 0.78125 Hz, 15.36 s); each trial preceded by 1.2 + U(0, 1) s; a pause after trials
  24 and 48 (waits for a key press: length NOT STATED, BREAK_S here). No command.
- Script_1 (l. 38-58, 64-69, 301-329): notch 48-52 and 98-102 Hz (and 148-152 at 512 Hz) and the "0.01-100 Hz band
  pass", all with pop_eegfiltnew(..., revfilt = 1): the last call is in fact a BAND-STOP (passband edges 0.01 and
  100 Hz, transition 12.4 Hz, -6 dB at 6.2 and 93.8 Hz). The paper (p. 650) describes a band-pass. Their saved
  preprocessed data carry it (tests: 102-120 Hz power far above 55-80 Hz power). Epochs -1 to 15.36 s around each
  trial trigger, resampled to 256 Hz, average reference over all channels, bad channels interpolated.
- Script_2 (l. 37-98): 256 Hz; FieldTrip low-pass 25 Hz (Butterworth, default order 6, two-pass, trial mean removed
  and restored); samples nearest 1.28 and 15.36 s kept (3604 at 256 Hz: the first sentence dropped); per trial and
  channel the full-length DFT; ITPC = |mean over trials of exp(i angle)|, per channel; A1/A2/COMNT/SCALE excluded.
- Script_3 (l. 45-66, 156-329): channel-mean ITPC; FOIs = first f >= 0.78 / 1.56 / 3.12 Hz; chance frequencies =
  0 < f <= 39 Hz, not multiples of the sentence FOI (MATLAB mod compensates round-off, so integer bin arithmetic);
  surrogate = a random chance frequency (word) or a chance frequency "and its harmonic" (phrase + sentence), 1000
  draws; p = (reps - #(surr <= obs)) / reps, or 0.5 / reps when every surrogate exceeds obs; fdr.m (BH) over the
  p-values. Their test is a GROUP Spearman correlation with GOSE; here the same surrogate compares one recording's
  ITPC with its own chance-frequency ITPC (our per-recording reading, no per-patient test is in the paper).
- Code quirks reproduced under code = 'as_coded' and corrected under 'fixed':
  (a) the empty-index branch (obs below every surrogate) returns p = 0.5 / reps, the smallest p possible;
  (b) the "harmonic" draw indexes f with the chance-array index (harm = find(f == f(i) * 2) = 2i - 1) and then reads
      itpc_chance at [i, 2i - 1]: the pair is chance-array entries i and 2i - 1, not a frequency and its harmonic.
- Per-recording call (cat. §8: "Per-patient rule: none in Methods"): the outcome-chosen ROC threshold 0.116 on the
  phrase + sentence ITPC (p. 655), or the surrogate p < .05, word or phrase + sentence or either after BH.
  Null ITPC floor: E = sqrt(pi / (4 K)), 0.104 at K = 72 [derived, cat. §8].
Automated substitutes (FIXED): no ICA; trials rejected by amplitude under reject = 'amp' (visual inspection, Script_1
step 2, is manual); only the 19 scalp channels exist in 10-20 data, so the average reference uses those 19 (theirs
also averaged A1, A2, COMNT, SCALE: Script_1 l. 325-329 'all'); fs < 256 is resampled to 256 Hz BEFORE the Script_1
filters (their 100-Hz edge is above a 128-Hz recording's Nyquist and pop_eegfiltnew errors); filters run on the
overlay span with firfilt's edge padding, not the whole recording.
The randomization-based counterpart (run_randomized) is a design addition: each trial is randomly a sentence trial
(label 1) or a random-word-list trial (label 2, Ding 2016-type control, same word rate), and the statistic is the
phrase + sentence ITPC difference between them, re-drawing the balanced trial labels."""
import math

import numpy as np
from scipy.signal import butter

from . import erp_common as ec
from . import motor_common as mc

NAME = 'sokoliuk21'
GRADE = 'CORE-EXACT'
FAMILY = 'erp'
NATIVE_CH = list(ec.EEG19)
SENT, LIST = 1, 2
WORD_S, WORDS_PER_SENT, SENT_PER_TRIAL = 0.32, 4, 12
TRIAL_S = WORD_S * WORDS_PER_SENT * SENT_PER_TRIAL
N_TRIALS, ISI_S, ISI_JIT_S = 72, 1.2, 1.0
BREAK_AFTER, BREAK_S = (24, 48), 30.0
EPOCH = (-1.0, TRIAL_S)
FS_ITPC = 256.0
LP_HZ, LP_ORDER = 25.0, 6
TOI = (1.28, 15.36)
FOI_HZ = (0.78, 1.56, 3.12)
FREQ_LIM_HZ = 39.0
REPS = 1000
ROC_THR = 0.116
ALPHA = 0.05
AMP_UV = 150.0
MIN_TRIALS = 2
EXCLUDE = ('A1', 'A2', 'COMNT', 'SCALE')
LEAD_S, TAIL_S = 12.0, 10.0
RULES = {
    'roc116': 'phrase + sentence ITPC >= 0.116 (outcome-chosen ROC, p. 655)',
    'surr_ps': 'phrase + sentence ITPC vs chance-frequency surrogate, p < .05 (Script_3 l. 239-281)',
    'surr_word': 'word ITPC vs chance-frequency surrogate, p < .05 (Script_3 l. 162-199)',
    'surr_fdr': 'either surrogate test after BH (fdr.m) over the two (Script_3 l. 327-329)',
}
VARIANTS = {}
for _r, _rc in RULES.items():
    for _code in (('as_coded',) if _r == 'roc116' else ('as_coded', 'fixed')):
        for _filt in ('as_coded', 'as_described'):
            for _rej in ('none', 'amp'):
                VARIANTS[f'{_r}_{_code}_{_filt}_{_rej}'] = {
                    'rule': _r, 'code': _code, 'filters': _filt, 'reject': _rej,
                    'as_coded': _r == 'surr_fdr' and _code == 'as_coded' and _filt == 'as_coded' and _rej == 'none',
                    'cite': f'cat. §8; {_rc}; per-patient rule NOT STATED in Methods; filters {_filt} '
                            '(Script_1 l. 49 revfilt = 1 vs paper p. 650 band-pass); trials kept after artefact '
                            'inspection NOT STATED'}
DEFAULT = 'roc116_as_coded_as_coded_none'
RUN_S = LEAD_S + N_TRIALS * (ISI_S + ISI_JIT_S + TRIAL_S) + len(BREAK_AFTER) * BREAK_S + TAIL_S


def run_s(variant=DEFAULT):
    return RUN_S



def _timing(start_s, g):
    isi = ISI_S + g.random(N_TRIALS)
    on, t = np.empty(N_TRIALS), float(start_s) + LEAD_S
    for j in range(N_TRIALS):
        t += isi[j]
        on[j] = t
        t += TRIAL_S
        if j + 1 in BREAK_AFTER:
            t += BREAK_S
    return on, isi


def _overlay(on, isi, lab):
    n = len(on)
    return ec.Overlay(onsets_s=on, labels=np.asarray(lab, int), kind='stim',
                      meta={'trial': np.arange(n), 'block': np.arange(n) // 24, 'run': np.zeros(n, int),
                            'isi_s': isi, 'trial_s': TRIAL_S, 'word_s': WORD_S})


def schedule(start_s, rng, variant=DEFAULT):
    on, isi = _timing(start_s, mc.gen(rng))
    return _overlay(on, isi, np.full(N_TRIALS, SENT))


def schedule_randomized(start_s, rng, variant=DEFAULT):
    """Same timing; each trial a sentence (1) or word-list (2) trial, complete randomization 36 / 36."""
    g = mc.gen(rng)
    on, isi = _timing(start_s, g)
    return _overlay(on, isi, mc.balanced_labels((SENT, LIST), (N_TRIALS // 2, N_TRIALS // 2), g))


def extent(ov):
    return float(ov.onsets_s.min()) + EPOCH[0] - 10.0, float(ov.onsets_s.max()) + TRIAL_S + 8.0



def ft_nearest(t, val):
    """FieldTrip nearest.m (20180121): ties go to the lower index; beyond the range, the end sample."""
    t = np.asarray(t, float)
    if val > t.max():
        return int(len(t) - 1 - np.argmax(t[::-1]))
    if val < t.min():
        return int(np.argmin(t))
    i2 = int(np.nonzero(t <= val)[0][-1])
    i3 = int(np.nonzero(t >= val)[0][0])
    return i2 if abs(t[i2] - val) <= abs(t[i3] - val) else i3


def lowpass_ft(x, fs, hz=LP_HZ, order=LP_ORDER):
    """ft_preproc_lowpassfilter 'but', 'twopass': trial mean removed, butter(N, hz / (fs / 2)) filtfilt, mean back."""
    b, a = butter(order, hz / (fs / 2.0))
    m = x.mean(-1, keepdims=True)
    return ec.filtfilt_ba(b, a, x - m) + m


def script2_itpc(trials, time, fs, labels=None):
    """Script_2 on one patient's epoched data. trials (K, C, T), time (T) s. -> (ITPC (C', N), f (N)) over the
    channels not in EXCLUDE (l. 63-64), in their order."""
    trials = np.asarray(trials, float)
    time = np.asarray(time, float)
    if fs != FS_ITPC:
        trials = ec.ft_resample(trials, fs, FS_ITPC)
        time = time[0] + np.arange(trials.shape[-1]) / FS_ITPC
        fs = FS_ITPC
    trials = lowpass_ft(trials, fs)
    i0, i1 = ft_nearest(time, TOI[0]), ft_nearest(time, TOI[1])
    seg = trials[..., i0:i1 + 1]
    if labels is not None:
        keep = [i for i, c in enumerate(labels) if c not in EXCLUDE]
        seg = seg[:, keep]
    N = seg.shape[-1]
    f = np.arange(N) * fs / N
    ph = np.exp(1j * np.angle(np.fft.fft(seg, axis=-1)))
    return np.abs(ph.mean(0)), f


def script3_bins(f):
    """-> (FOI bins (sentence, phrase, word), chance bins), 0-based (Script_3 l. 50-57)."""
    fois = np.array([int(np.nonzero(f >= h)[0][0]) for h in FOI_HZ])
    lim = int(np.nonzero(f <= FREQ_LIM_HZ)[0][-1])
    k_sent = fois[0]
    chance = np.array([k for k in range(lim + 1) if k % k_sent != 0])
    return fois, chance


def harmonic_entries(chance):
    """For each chance bin k, the chance-array entry of 2k (its true harmonic) or -1 (code = 'fixed')."""
    pos = {int(k): j for j, k in enumerate(chance)}
    return np.array([pos.get(2 * int(k), -1) for k in chance])


def surrogate_p(obs, chance_itpc, g, pair=False, code='as_coded', harm=None, reps=REPS):
    """Chance-frequency surrogate for one recording (Script_3 l. 162-189 single frequency, l. 239-270 pair).
    chance_itpc: channel-mean ITPC at the chance bins; harm: harmonic_entries(chance) for code = 'fixed'."""
    chance_itpc = np.asarray(chance_itpc, float)
    n = len(chance_itpc)
    surr = np.empty(reps)
    for r in range(reps):
        if not pair:
            surr[r] = chance_itpc[g.integers(n)]
            continue
        while True:
            i = int(g.integers(n))
            h = 2 * i if code == 'as_coded' else int(harm[i])
            if 0 <= h <= n - 1:
                surr[r] = 0.5 * (chance_itpc[i] + chance_itpc[h])
                break
    if code == 'as_coded':
        le = int(np.sum(np.sort(surr) <= obs))
        p = 0.5 / reps if le == 0 else (reps - le) / reps
    else:
        p = (np.sum(surr >= obs) + 1.0) / (reps + 1)
    return float(p), surr


def surrogate_population(chance_itpc, pair=False, code='as_coded', harm=None):
    """Every value surrogate_p can draw, equally weighted: the chance-bin ITPCs, or the pair means of the valid
    pairs (as coded: entries i and 2i; fixed: a chance bin and its true harmonic)."""
    ch = np.asarray(chance_itpc, float)
    if not pair:
        return ch
    n = len(ch)
    idx = [(i, 2 * i if code == 'as_coded' else int(harm[i])) for i in range(n)]
    return np.array([0.5 * (ch[i] + ch[h]) for i, h in idx if 0 <= h <= n - 1])


def surrogate_z(obs, population):
    """Deterministic, continuous evidence: obs standardized against the full surrogate population."""
    return float((obs - population.mean()) / population.std())



def script1_filters(x, fs, how):
    """Script_1 l. 38-50 on continuous data (channels x samples)."""
    x = ec.eegfiltnew(x, fs, 48, 52, 1)
    x = ec.eegfiltnew(x, fs, 98, 102, 1)
    if fs == 512:
        x = ec.eegfiltnew(x, fs, 148, 152, 1)
    return ec.eegfiltnew(x, fs, 0.01, 100, 1 if how == 'as_coded' else 0)


def features(rec, ov, filters='as_coded'):
    """Label-free part: epoched trials at 256 Hz after Script_1 filters and average reference. -> dict(trials
    (K, 19, T), time, fs, which)."""
    x, used, which = ec.pick(rec, NATIVE_CH, NATIVE_CH)
    fs = float(rec.fs)
    seg, i0 = mc.span(x, fs, *extent(ov))
    t_on = np.asarray(ov.onsets_s, float) - i0 / fs
    if fs < 256:
        seg = ec.ft_resample(seg, fs, 256.0)
        fs = 256.0
    seg = script1_filters(seg, fs, filters)
    n_pre, n_post = int(round(-EPOCH[0] * fs)), int(round(EPOCH[1] * fs))
    tr = ec.sample_epochs(seg, fs, t_on, n_pre, n_post)
    time = np.arange(-n_pre, n_post) / fs
    if fs != FS_ITPC:
        tr = ec.ft_resample(tr, fs, FS_ITPC)
        time = time[0] + np.arange(tr.shape[-1]) / FS_ITPC
        fs = FS_ITPC
    tr = tr - tr.mean(1, keepdims=True)
    return {'trials': tr, 'time': time, 'fs': fs, 'which': which}


def _feat(rec, ov, filters):
    return mc.memo(mc.rec_key(rec, np.asarray(ov.onsets_s, float), filters), lambda: features(rec, ov, filters))


def _keep(F, reject):
    if reject == 'none':
        return np.ones(len(F['trials']), bool)
    i0, i1 = ft_nearest(F['time'], TOI[0]), ft_nearest(F['time'], TOI[1])
    return np.abs(F['trials'][..., i0:i1 + 1]).max(axis=(1, 2)) <= AMP_UV


def itpc_from_features(F, keep):
    itpc, f = script2_itpc(F['trials'][keep], F['time'], F['fs'])
    return itpc.mean(0), f


def run(rec, overlay, variant=DEFAULT, rng=None):
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay, c['filters'])
    keep = _keep(F, c['reject'])
    K = int(keep.sum())
    if K < MIN_TRIALS:
        raise mc.NoUsableTrials(f'sokoliuk21: {K} of {len(keep)} trials survive rejection (the ITPC needs >= '
                                f'{MIN_TRIALS})')
    avg, f = itpc_from_features(F, keep)
    fois, chance = script3_bins(f)
    ps, word = float(avg[fois[:2]].mean()), float(avg[fois[2]])
    extra = dict(itpc_sentence=float(avg[fois[0]]), itpc_phrase=float(avg[fois[1]]), itpc_word=word, itpc_ps=ps,
                 n_trials=K, noise_floor=math.sqrt(math.pi / (4 * K)), fois_hz=f[fois], channels=F['which'])
    if c['rule'] == 'roc116':
        return mc.result(ps, None, ps >= ROC_THR, size_stat=ps, size_stat_cont=ps, **extra)
    ch, harm = avg[chance], harmonic_entries(chance)
    p_w = surrogate_p(word, ch, g, pair=False, code=c['code'])[0]
    p_ps = surrogate_p(ps, ch, g, pair=True, code=c['code'], harm=harm)[0]
    z_w = surrogate_z(word, surrogate_population(ch))
    z_ps = surrogate_z(ps, surrogate_population(ch, True, c['code'], harm))
    extra.update(p_word=p_w, p_ps=p_ps, z_word=z_w, z_ps=z_ps)
    if c['rule'] == 'surr_ps':
        return mc.result(ps, p_ps, p_ps < ALPHA, size_stat=1.0 - p_ps, size_stat_cont=z_ps, **extra)
    if c['rule'] == 'surr_word':
        return mc.result(word, p_w, p_w < ALPHA, size_stat=1.0 - p_w, size_stat_cont=z_w, **extra)
    thr = ec.bh_threshold([p_w, p_ps], ALPHA)
    pmin = min(p_w, p_ps)
    return mc.result(ps, pmin, pmin <= thr and thr > 0, size_stat=1.0 - pmin, size_stat_cont=max(z_w, z_ps),
                     bh_threshold=thr, **extra)


def _phasors(F, keep):
    """Unit phasors at the sentence and phrase bins: (K, 19, 2), after Script_2 resampling, low-pass and trim."""
    tr = lowpass_ft(F['trials'][keep], F['fs'])
    i0, i1 = ft_nearest(F['time'], TOI[0]), ft_nearest(F['time'], TOI[1])
    seg = tr[..., i0:i1 + 1]
    N = seg.shape[-1]
    f = np.arange(N) * F['fs'] / N
    fois = script3_bins(f)[0][:2]
    n = np.arange(N)
    E = np.exp(-2j * np.pi * np.outer(n, fois) / N)
    return np.exp(1j * np.angle(np.einsum('kct,tf->kcf', seg, E)))


def run_randomized(rec, overlay, variant=DEFAULT, rng=None):
    """Statistic: channel-mean phrase + sentence ITPC of sentence trials minus that of word-list trials; the null
    re-draws the balanced trial labels (complete randomization, cat. §8 design + Ding-type control), B = 500 with
    decision-exact early stopping. A re-draw that leaves fewer than MIN_TRIALS kept trials in a class is re-drawn
    (mc.admissible_stat, a conditional randomization test), never scored."""
    c = mc.cfg(VARIANTS, variant)
    g = mc.gen(rng)
    F = _feat(rec, overlay, c['filters'])
    keep = _keep(F, c['reject'])
    lab_all = np.asarray(overlay.labels)
    n1 = int(np.sum(lab_all == SENT))
    lab = lab_all[keep]

    def ok(y):
        return min(np.sum(y == SENT), np.sum(y == LIST)) >= MIN_TRIALS

    if not ok(lab):
        raise mc.NoUsableTrials(f'sokoliuk21: {int(np.sum(lab == SENT))} sentence and {int(np.sum(lab == LIST))} '
                                f'word-list trials survive rejection (each needs >= {MIN_TRIALS})')
    Z = _phasors(F, keep)

    def stat(y):
        a = np.abs(Z[y == SENT].mean(0)).mean()
        b = np.abs(Z[y == LIST].mean(0)).mean()
        return float(a - b)

    def draw_labels():
        return mc.balanced_labels((SENT, LIST), (n1, len(lab_all) - n1), g)[keep]

    obs = stat(lab)
    t = mc.perm_test(obs, lambda: mc.admissible_stat(draw_labels, ok, stat), c.get('n_perm_rand', 500), ALPHA)
    return mc.result(obs, t['p'], t['p'] < ALPHA, n_perm_run=t['n_run'], stopped_early=t['stopped'],
                     n_trials=int(keep.sum()), size_stat=obs, size_stat_cont=obs)


def stat_for_size_correction(result):
    """size_stat_cont: the phrase + sentence ITPC (roc116); for the surrogate rules the observed ITPC's z
    against the full chance-frequency population the surrogate draws from (surrogate_z: data-only, continuous), the
    larger of the word and phrase + sentence z for surr_fdr; the ITPC difference (run_randomized). Falls back to the
    earlier size_stat."""
    e = result['extra']
    return float(e.get('size_stat_cont', e.get('size_stat', result['stat'])))


def patient_rule(results):
    """One 72-trial assessment per patient in SOK21 (cat. §8); for k > 1 assessments 'any' (NOT STATED)."""
    return any(r['call'] for r in results)
