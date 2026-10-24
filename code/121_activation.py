"""A9 activation-procedure exclusion (part A, A9): flagged intervals per
recording, so that 116 and A1(a) can keep only windows clear of activation procedures, stimulation, reactivity
testing and annotated clinical events. Outside the frozen-code hash; built test-first (code/test_activation.py).

(a) Annotations, where a file carries them: every non-timekeeping annotation is flagged over [onset, onset +
duration] (instantaneous ones at their onset), whatever its text, with a keyword category for reporting only.
- pDoC EDFs: EDF+ "EDF Annotations" signals, read by edf_annotations() from header byte 184 on, so the identity
  and start-date fields (bytes 8-184) are never read; no EEG samples are read either.
- BDSP .mat files carry none (read-only probe of 3 dev files on 2026-10-14: top-level keys '#refs#', 'Fs',
  'channels', 'data' only; '#refs#' holds the 20 channel-name strings and one empty value) and the local metadata
  (sample_500 / bench_extra_500 / train_600: patient_id, dir, file, bytes, n_eligible_segments) has no event column.
(b) BDSP therefore uses the automated detectors below, on x19 (Fp1 Fp2 O1 O2 F3 F4, uV, 128 Hz): photic stimulation
(photic_intervals) and the eyes-open/closed alternation (eoec_intervals). Padding seconds (SD < 0.1 uV on a channel
used) are never read as events. Hyperventilation and reactivity testing have no detector (none was fixed in advance),
and there is no recording-type metadata to stratify on: both are stated limitations.

Thresholds: fixed on synthetic signals and the 20 BDSP dev recordings only, before any analysis-split read.
- Photic. A 4-s two-channel Hann periodogram bin of noise exceeds 20 x its flank median with probability ~1e-13, so
  a PHOTIC_LINE_SNR line is a real narrow-band component; but on dev 38-48% of windows held one (equipment lines,
  ECG harmonic combs, narrow alpha), which is why the alpha band is excluded and the train, stationarity, comb and
  4-step staircase rules exist. The development versions found 7 dev sessions; on occipital spectrograms 1 was a
  visible ascending stepped staircase (3-Hz steps) and 6 were stationary equipment lines (a 7.8-Hz harmonic
  series, a constant 20-Hz line) or ECG combs. Final: the 1 staircase only. Synthetic (3-19 Hz in 2-Hz steps,
  10-s trains, 6-s gaps, pink noise SD 10 uV plus an 8-uV alpha): a session is found in 12/20 runs at a 5-6 uV
  fundamental, 19/20 at 8 uV and 20/20 at 10 uV; no session in 50 x 10 min of noise and alpha, nor on a stationary
  line, an ECG-like pulse train, a single train, equipment lines cycling through four frequencies, or steps borne by
  a harmonic comb (tests). Cost of the comb rule: a real session during strong ECG pick-up can be missed.
- Eyes. A first version (any Fp transient, alpha drop 2 x from the median, >= 2 openings within 60 s) flagged 18/20
  dev recordings, up to 48% of one; review showed frontal slow waves, ECG pick-up at Fp and broadband artefact bursts.
  Each later rule removed one such class: Fp dominance over F3/F4 and the 0.1-0.5 s width (slow waves), the
  periodicity rule (pulse), the 2 x median and 3 x change with an 8-13 Hz rhythm (artefact bursts) and the regular
  alternation with 3-30 s spells (spontaneous drowsy eye closure, which stays in the Morgoth background strata).
  After them: 1 episode in 1 recording, visible alpha switching with frontopolar transients at the switches (mean
  flagged share 0.2%). Synthetic (3 cycles, 10-s spells, pink noise SD 10 uV): found in 17/20 runs with a 30-uV eye
  transient and 20/20 at >= 50 uV (alpha 6-12 uV); no episode in 50 x 5 min each of blinks only, alpha switching
  without transients, and noise, nor for one cycle, long irregular spells, pulse transients, broadband artefact
  bursts or frontal slow waves (tests).
- Margins (photic +-20 s; eyes 20 s before the first closure to 20 s after the last opening) cover steps outside
  2.5-29 Hz or inside the alpha exclusion and cycles next to a run whose switches were missed.

Outputs (git-ignored): output/intermediate/_activation_<source>_<split>.csv (uid, kind, start_s, end_s, detail,
method; seconds from the start of the recording as bench.engine loads it) and ..._recordings.csv (one row per
processed recording). windows_flagged() is the lookup for 116; a recording absent from the table is not clean.
python3 121_activation.py --source bdsp|pdoc --split dev|analysis
"""
import argparse
import os
import re
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
OUT = os.path.join(ROOT, "output", "intermediate")

COLUMNS = ["kind", "start_s", "end_s", "detail", "method"]
EdfAnnotations = namedtuple("EdfAnnotations", "edf_plus n_annotation_signals dur_s events")
CATEGORIES = (
    ("photic", r"photic|\bips\b|flash|strob"),
    ("hyperventilation", r"hyperv|\bhv\b"),
    ("eyes", r"eye|\beo\b|\bec\b|blink"),
    ("stimulation", r"stim|reactiv|pain|noxious|sternal|nail|tactile|touch|clap|sound|name|call|command"
                    r"|light|свет"),
    ("clinical_event", r"seiz|\bsz\b|event|spike|clin|nurs|care|suction|turn|reposition|move|cough|agitat"),
    ("technical", r"start|stop|measur|record|calib|impedance|montage|старт|измерен"),
)


def edf_annotations(path):
    """EDF+ annotations of one file: (edf_plus, number of 'EDF Annotations' signals, duration s, [(onset_s,
    duration_s or None, text)]). Reads from byte 184 on (never the patient/recording identity or start date/time
    fields) and, of the data records, only the annotation signals' bytes; the per-record timekeeping TALs (empty
    text) are dropped."""
    with open(path, "rb") as fh:
        fh.seek(184)
        head = fh.read(72)
        n_head, reserved = int(head[:8]), head[8:52].decode("ascii", "replace").strip()
        n_rec, rec_s, ns = int(head[52:60]), float(head[60:68]), int(head[68:72])
        sig = fh.read(256 * ns)
        labels = [sig[16 * i:16 * (i + 1)].decode("ascii", "replace").strip() for i in range(ns)]
        off = 216 * ns
        spr = [int(sig[off + 8 * i:off + 8 * (i + 1)]) for i in range(ns)]
        ann = [i for i, lab in enumerate(labels) if lab == "EDF Annotations"]
        events = []
        if ann:
            rec_bytes = 2 * sum(spr)
            starts = np.cumsum([0] + [2 * s for s in spr])
            for r in range(max(n_rec, 0)):
                for i in ann:
                    fh.seek(n_head + r * rec_bytes + int(starts[i]))
                    events += _tals(fh.read(2 * spr[i]))
    return EdfAnnotations(reserved.startswith("EDF+"), len(ann), max(n_rec, 0) * rec_s, events)


def _tals(raw):
    out = []
    for tal in raw.split(b"\x00"):
        if not tal:
            continue
        parts = tal.split(b"\x14")
        when = parts[0].decode("ascii", "replace").split("\x15")
        onset, dur = float(when[0]), (float(when[1]) if len(when) > 1 and when[1] else None)
        out += [(onset, dur, _text(t)) for t in parts[1:] if t]
    return out


def _text(raw):
    """UTF-8 annotation text; Windows-1251 bytes that were stored as UTF-8-encoded Latin-1 (the pDoC files) are
    repaired."""
    t = raw.decode("utf-8", "replace")
    try:
        fixed = t.encode("latin-1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t
    return fixed if re.search("[\u0400-\u04ff]", fixed) else t


def category(text):
    t = text.lower()
    return next((name for name, pat in CATEGORIES if re.search(pat, t)), "other")


def annotation_intervals(events, dur_s):
    """One flagged interval per annotation: [onset, onset + duration] clipped to the recording."""
    rows = [dict(kind=category(text), start_s=float(on), end_s=float(min(dur_s, on + (d or 0.0))),
                 detail=text[:60], method="annotation") for on, d, text in events]
    return pd.DataFrame(rows, columns=COLUMNS)



PAD_UV = 0.1

PHOTIC_WIN_S, PHOTIC_STEP_S = 4.0, 1.0
PHOTIC_F = (2.5, 29.0)
PHOTIC_ALPHA_EXCL = (7.5, 13.5)
PHOTIC_FLANK_HZ = (0.75, 2.0)
PHOTIC_LINE_SNR = 20.0
PHOTIC_COMB_SNR, PHOTIC_COMB_LINES = 6.0, 2
PHOTIC_SAME_HZ = 0.5
PHOTIC_MIN_WIN, PHOTIC_MAX_TRAIN_S = 2, 40.0
PHOTIC_STATIONARY_S, PHOTIC_STATIONARY_MAX = 300.0, 40
PHOTIC_GAP_S, PHOTIC_MIN_STEPS, PHOTIC_STEP_HZ = 60.0, 4, 1.0
PHOTIC_MARGIN_S = 20.0

FP_BAND, ALPHA_BAND = (0.5, 8.0), (8.0, 13.0)
BLINK_Z, BLINK_Z_EACH = 5.0, 3.0
BLINK_FP_DOM = 2.0
BLINK_WIDTH_S, BLINK_SEP_S = (0.1, 0.5), 1.0
BLINK_PERIODIC_S, BLINK_PERIODIC_N, BLINK_PERIODIC_CV = 10.0, 6, 0.35
ALPHA_PRE_S, ALPHA_POST_S = (-3.0, -0.5), (0.5, 3.0)
ALPHA_HIGH, ALPHA_DROP = 2.0, 3.0
ALPHA_PEAK = 1.5
EOEC_SPELL_S = (3.0, 30.0)
EOEC_MIN_CYCLES, EOEC_MARGIN_S = 2, 20.0


def _padding(x, fs):
    fs = int(round(fs))
    n = x.shape[1] // fs
    sd = np.asarray(x[:, : n * fs], float).reshape(x.shape[0], n, fs).std(2)
    return (sd < PAD_UV).any(0)


def _frames(y, fs):
    w, st = int(PHOTIC_WIN_S * fs), int(PHOTIC_STEP_S * fs)
    n = 1 + max(0, (y.shape[-1] - w) // st)
    idx = np.arange(w)[None, :] + st * np.arange(n)[:, None]
    return y[..., idx], idx[:, 0] / fs


def photic_intervals(occ, fs):
    """Photic-stimulation sessions from O1/O2 (2 x samples, uV). Per 4-s window (1-s step) the channel-averaged Hann
    periodogram has a line at f when P(f) >= PHOTIC_LINE_SNR x the median of its 0.75-2-Hz flanks (f 2.5-29 Hz,
    outside 7.5-13.5 Hz; the strongest line per window), unless the window also holds >= 2 other peaks >= 6 x their
    flanks off f's harmonics (a comb). Windows with the same f (within 0.5 Hz) form a train (>= 2
    windows, <= 40 s); a train whose line is present in > 40 windows within +-300 s is a stationary artefact (pump,
    ventilator, ECG comb) and is dropped. Trains <= 60 s apart form a session, kept when >= 4 consecutive trains step
    one way by >= 1 Hz each (the stepped protocol). Flagged: the session span +- 20 s."""
    fs = float(fs)
    occ = np.asarray(occ, float)
    pad = _padding(occ, fs)
    fr, t0 = _frames(occ - occ.mean(1, keepdims=True), fs)
    p = (np.abs(np.fft.rfft(fr * np.hanning(fr.shape[-1]), axis=-1)) ** 2).mean(0)
    df = fs / fr.shape[-1]
    k = np.arange(int(round(PHOTIC_F[0] / df)), int(round(PHOTIC_F[1] / df)) + 1)
    lo, hi = int(round(PHOTIC_FLANK_HZ[0] / df)), int(round(PHOTIC_FLANK_HZ[1] / df))
    off = np.r_[-np.arange(hi, lo - 1, -1), np.arange(lo, hi + 1)]
    snr = p[:, k] / np.maximum(np.median(p[:, k[:, None] + off[None, :]], axis=-1), 1e-30)
    line = snr >= PHOTIC_LINE_SNR
    fk = k * df
    cand = np.where(line & ~((fk >= PHOTIC_ALPHA_EXCL[0]) & (fk <= PHOTIC_ALPHA_EXCL[1]))[None, :], snr, 0.0)
    secs = np.floor(t0).astype(int)
    span = int(np.ceil(PHOTIC_WIN_S))
    clean = np.array([not pad[s:s + span].any() for s in secs])
    f = np.where((cand.max(1) > 0) & clean, fk[np.argmax(cand, 1)], np.nan)
    peak = np.zeros_like(snr, bool)
    peak[:, 1:-1] = (snr[:, 1:-1] >= snr[:, :-2]) & (snr[:, 1:-1] >= snr[:, 2:])
    other = peak & (snr >= PHOTIC_COMB_SNR) & ~((fk >= PHOTIC_ALPHA_EXCL[0]) & (fk <= PHOTIC_ALPHA_EXCL[1]))[None, :]
    for w in np.flatnonzero(~np.isnan(f)):
        harm = np.min(np.abs(fk[None, :] - np.array([[1.0], [2.0], [3.0]]) * f[w]), axis=0) <= PHOTIC_SAME_HZ
        if (other[w] & ~harm).sum() >= PHOTIC_COMB_LINES:
            f[w] = np.nan
    trains, i = [], 0
    while i < len(f):
        if np.isnan(f[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(f) and not np.isnan(f[j + 1]) and abs(f[j + 1] - f[j]) <= PHOTIC_SAME_HZ:
            j += 1
        a, b, fm = t0[i], t0[j] + PHOTIC_WIN_S, float(np.median(f[i:j + 1]))
        near = (t0 >= a - PHOTIC_STATIONARY_S) & (t0 <= b + PHOTIC_STATIONARY_S)
        persistent = line[near][:, np.abs(fk - fm) <= PHOTIC_SAME_HZ].any(1).sum() > PHOTIC_STATIONARY_MAX
        if j - i + 1 >= PHOTIC_MIN_WIN and b - a <= PHOTIC_MAX_TRAIN_S and not persistent:
            trains.append((a, b, fm))
        i = j + 1
    rows, dur = [], occ.shape[1] / fs
    for group in _clusters(trains, PHOTIC_GAP_S):
        steps = _staircase([fr_ for _, _, fr_ in group], PHOTIC_STEP_HZ)
        if len(steps) >= PHOTIC_MIN_STEPS:
            freqs = sorted({fr_ for _, _, fr_ in group})
            rows.append(dict(kind="photic", start_s=max(0.0, group[0][0] - PHOTIC_MARGIN_S),
                             end_s=min(dur, group[-1][1] + PHOTIC_MARGIN_S),
                             detail="f=" + ",".join(f"{v:g}" for v in freqs)
                             + f"Hz; trains={len(group)}; steps={len(steps)}", method="detector"))
    return pd.DataFrame(rows, columns=COLUMNS)


def _clusters(events, gap):
    """Split time-sorted (start, end, ...) tuples where the next start is > gap after the previous end."""
    out = []
    for e in sorted(events):
        if out and e[0] - out[-1][-1][1] <= gap:
            out[-1].append(e)
        else:
            out.append([e])
    return out


def _staircase(freqs, step):
    """Longest run of consecutive values that all rise (or all fall) by >= step."""
    best, run = freqs[:1], freqs[:1]
    for sign in (1, -1):
        run = freqs[:1]
        for v in freqs[1:]:
            run = run + [v] if sign * (v - run[-1]) >= step else [v]
            best = run if len(run) > len(best) else best
    return best


def eoec_intervals(x, fs):
    """Eyes-open/closed alternation from Fp1, Fp2, O1, O2, F3, F4 (6 x samples, uV, in that order).
    - Eye transient: in the 0.5-8 Hz band, the Fp1/Fp2 mean beyond BLINK_Z robust SD (1.4826 MAD over non-padding
      samples), each Fp channel beyond BLINK_Z_EACH with the same sign, the Fp mean >= BLINK_FP_DOM x the F3/F4 mean
      (an EOG field falls off steeply; frontal slow waves do not), half-height width 0.1-0.5 s; transients >= 6 in
      +-10 s at regular intervals (CV < 0.35) are pulse/ECG pick-up and dropped.
    - Opening: a transient with O1/O2 8-13 Hz power over 0.5-3 s before it >= ALPHA_DROP x the power 0.5-3 s after
      it, the 'before' power >= ALPHA_HIGH x the recording median and its 8-13 Hz density >= ALPHA_PEAK x that of 4-7
      and 14-20 Hz (a posterior rhythm with the eyes closed, not an artefact burst). Closure: the mirror image.
    - Episode: >= EOEC_MIN_CYCLES consecutive close -> open cycles, each closed and open spell 3-30 s (the
      instructed alternation; isolated or irregular openings are not flagged; a repeat of the same switch < 3 s later
      is the same switch). Flagged: 20 s before the first closure to 20 s after the last opening (cycles next to the
      run whose transitions were missed)."""
    fs = float(fs)
    x = np.asarray(x, float)
    pad = _padding(x, fs)
    n = x.shape[1]
    sec = int(round(fs))
    good = np.r_[np.repeat(~pad, sec), np.zeros(n - len(pad) * sec, bool)]
    if good.sum() < 10 * fs:
        return pd.DataFrame(columns=COLUMNS)
    bp = sosfiltfilt(butter(4, FP_BAND, btype="band", fs=fs, output="sos"), x[[0, 1, 4, 5]], axis=1)
    y, fr = bp[:2].mean(0), bp[2:].mean(0)

    def rsd(v):
        v = v[good]
        return 1.4826 * np.median(np.abs(v - np.median(v)))

    sy, s1, s2 = rsd(y), rsd(bp[0]), rsd(bp[1])
    pk, _ = find_peaks(np.abs(y) * good, height=BLINK_Z * sy, distance=int(BLINK_SEP_S * fs),
                       width=(BLINK_WIDTH_S[0] * fs, BLINK_WIDTH_S[1] * fs), rel_height=0.5)
    keep = ((np.sign(bp[0, pk]) == np.sign(bp[1, pk])) & (np.abs(bp[0, pk]) >= BLINK_Z_EACH * s1)
            & (np.abs(bp[1, pk]) >= BLINK_Z_EACH * s2) & (np.abs(y[pk]) >= BLINK_FP_DOM * np.abs(fr[pk])))
    pk = pk[keep]
    periodic = np.zeros(len(pk), bool)
    for i in range(len(pk)):
        near = pk[np.abs(pk - pk[i]) <= BLINK_PERIODIC_S * fs]
        if len(near) >= BLINK_PERIODIC_N:
            d = np.diff(near)
            periodic[i] = d.std() < BLINK_PERIODIC_CV * d.mean()
    pk = pk[~periodic]
    al = sosfiltfilt(butter(4, ALPHA_BAND, btype="band", fs=fs, output="sos"), x[2:4], axis=1)
    a = (al ** 2).mean(0)
    c, cg = np.r_[0.0, np.cumsum(a)], np.r_[0, np.cumsum(~good)]
    ref = np.median(a[: len(pad) * sec].reshape(-1, sec).mean(1)[~pad])
    occ = x[2:4]

    def rhythmic(lo, hi):
        seg = occ[:, lo:hi] - occ[:, lo:hi].mean(1, keepdims=True)
        p = (np.abs(np.fft.rfft(seg * np.hanning(hi - lo), axis=1)) ** 2).mean(0)
        f = np.fft.rfftfreq(hi - lo, 1 / fs)
        band = p[(f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])].mean()
        flank = p[((f >= 4) & (f <= 7)) | ((f >= 14) & (f <= 20))].mean()
        return band >= ALPHA_PEAK * flank

    events = []
    for t in pk:
        lo1, hi1 = t + int(ALPHA_PRE_S[0] * fs), t + int(ALPHA_PRE_S[1] * fs)
        lo2, hi2 = t + int(ALPHA_POST_S[0] * fs), t + int(ALPHA_POST_S[1] * fs)
        if lo1 < 0 or hi2 > n or cg[hi2] - cg[lo1] > 0:
            continue
        pre, post = (c[hi1] - c[lo1]) / (hi1 - lo1), (c[hi2] - c[lo2]) / (hi2 - lo2)
        if pre >= ALPHA_HIGH * ref and pre >= ALPHA_DROP * post and rhythmic(lo1, hi1):
            events.append((t / fs, "O"))
        elif post >= ALPHA_HIGH * ref and post >= ALPHA_DROP * pre and rhythmic(lo2, hi2):
            events.append((t / fs, "C"))
    rows, dur, run = [], n / fs, []

    def close_run():
        cycles = sum(1 for i in range(1, len(run)) if run[i - 1][1] == "C" and run[i][1] == "O")
        if cycles >= EOEC_MIN_CYCLES:
            rows.append(dict(kind="eoec", start_s=max(0.0, run[0][0] - EOEC_MARGIN_S),
                             end_s=min(dur, run[-1][0] + EOEC_MARGIN_S), detail=f"cycles={cycles}",
                             method="detector"))

    for ev in events:
        if run and ev[1] == run[-1][1] and ev[0] - run[-1][0] < EOEC_SPELL_S[0]:
            continue
        if run and ev[1] != run[-1][1] and EOEC_SPELL_S[0] <= ev[0] - run[-1][0] <= EOEC_SPELL_S[1]:
            run.append(ev)
        else:
            close_run()
            run = [ev] if ev[1] == "C" else []
    close_run()
    return pd.DataFrame(rows, columns=COLUMNS)


def bdsp_intervals(x, names, fs):
    """Both detectors on one BDSP recording (x19 / names19). Needs Fp1, Fp2, O1, O2, F3, F4."""
    up = [str(nm).upper() for nm in names]
    need = ["FP1", "FP2", "O1", "O2", "F3", "F4"]
    miss = [c for c in need if c not in up]
    if miss:
        raise ValueError(f"missing channels {miss} (need Fp1 Fp2 O1 O2 F3 F4)")
    xs = np.asarray(x, float)[[up.index(c) for c in need]]
    parts = [photic_intervals(xs[2:4], fs), eoec_intervals(xs, fs)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values("start_s", ignore_index=True)


def flag_windows(intervals, starts_s, dur_s):
    """True for each window [s, s + dur) that overlaps a flagged interval (an instantaneous one counts when
    s <= t < s + dur)."""
    s = np.asarray(starts_s, float)[:, None]
    if len(intervals) == 0:
        return np.zeros(len(s), bool)
    a = np.asarray(intervals["start_s"], float)[None, :]
    b = np.asarray(intervals["end_s"], float)[None, :]
    return ((a < s + dur_s) & (b >= s)).any(1)



def _union_s(iv):
    spans = sorted(zip(iv["start_s"], iv["end_s"]))
    tot, cur = 0.0, None
    for a, b in spans:
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            tot += (cur[1] - cur[0]) if cur else 0.0
            cur = [a, b]
    return tot + ((cur[1] - cur[0]) if cur else 0.0)


def _tables(per_rec):
    """[(uid, method, dur_s, intervals, extra)] -> (intervals with uid, one row per recording)."""
    ivs, recs = [], []
    for uid, method, dur, iv, extra in per_rec:
        ivs.append(iv.assign(uid=uid))
        recs.append(dict(uid=uid, method=method, dur_s=float(dur), n_intervals=len(iv),
                         n_photic=int((iv.kind == "photic").sum()), n_eoec=int((iv.kind == "eoec").sum()),
                         flagged_s=float(_union_s(iv)), **extra))
    ivs = [v for v in ivs if len(v)]
    iv = pd.concat(ivs, ignore_index=True) if ivs else pd.DataFrame(columns=COLUMNS + ["uid"])
    return iv[["uid"] + COLUMNS], pd.DataFrame(recs)


def annotation_table(paths):
    """A9 (a) over EDF files: every annotation flagged. uid = file name without .edf (as bench.engine)."""
    out = []
    for p in paths:
        a = edf_annotations(p)
        out.append((os.path.basename(p)[:-4], "annotation", a.dur_s, annotation_intervals(a.events, a.dur_s),
                    dict(n_annotations=len(a.events), edf_plus=a.edf_plus)))
    return _tables(out)


def detector_table(recordings):
    """A9 (b) over BDSP recordings (objects with uid, x, names, fs; bench.engine.Recording)."""
    return _tables([(r.uid, "detector", r.x.shape[1] / r.fs, bdsp_intervals(r.x, r.names, r.fs), {})
                    for r in recordings])


def windows_flagged(intervals, recordings, uid, starts_s, dur_s):
    """flag_windows for one recording; KeyError if the recording was not processed (absent is not clean)."""
    if uid not in set(recordings["uid"]):
        raise KeyError(f"{uid}: not processed by 121 (absent is not clean)")
    return flag_windows(intervals[intervals["uid"] == uid], starts_s, dur_s)


def paths(source, split):
    return (os.path.join(OUT, f"_activation_{source}_{split}.csv"),
            os.path.join(OUT, f"_activation_{source}_{split}_recordings.csv"))


def load(source, split):
    """(intervals, recordings) written by main() for a source and split."""
    a, b = paths(source, split)
    return pd.read_csv(a), pd.read_csv(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["bdsp", "pdoc"], required=True)
    ap.add_argument("--split", choices=["dev", "analysis"], required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, CODE)
    from bench import engine as en
    if a.source == "pdoc":
        sp = en.pdoc_split(en.PDOC_RAW, en.DEV_FILE)
        iv, recs = annotation_table([os.path.join(en.PDOC_RAW, f) for f in sp[a.split]])
    else:
        iv, recs = detector_table(en.load_recordings("bdsp", a.split))
    os.makedirs(OUT, exist_ok=True)
    for df, path in zip((iv, recs), paths(a.source, a.split)):
        df.to_csv(path + ".part", index=False)
        os.replace(path + ".part", path)
    print(f"{a.source} {a.split}: {len(recs)} recordings, {int((recs.n_intervals > 0).sum())} with a flag, "
          f"{len(iv)} intervals ({ {k: int(v) for k, v in iv.kind.value_counts().items()} }), flagged share "
          f"{recs.flagged_s.sum() / max(recs.dur_s.sum(), 1e-9):.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
