"""Mechanism analyses of the null benchmark, second stage: where false calls come from (section "Mechanism: where false calls come from", written before any second-stage rate was read). Outside the benchmark
hash; built test-first on synthetic fixtures and the dev split (code/test_mechanism.py). Every module detail
(schedules, epoch grids, ERP windows and baselines, channels, bands) is read from the frozen bench modules.

The predictor X (design-spectrum overlap), per null row:
- Feature series on the module's native channels (on BDSP its stated 10-20 substitute set, as the module picks it)
  after the module's own spatial transform (SPATIAL, the frozen functions and matrices: columbia 112's CSD; cornell
  and edlow17 their Hjorth Laplacians; cruse11 the equal-weight Laplacian over every recorded channel; cruse12p its
  two bipolar derivations; chennu13 and sokoliuk21 the average reference; mbmi, dubois (CSP, data-driven), mbvt and
  gibson16 none), on its own epoch/trial grid (family-generic features; no artefact rule in the primary X):
  - motor: log 8-30 Hz power per epoch (engine.BAND; zero-phase 4th-order Butterworth over the module's extent;
    floor 1e-12 as doc_null.trial_features);
  - ERP: mean 1-20 Hz amplitude per trial in the module's ERP window minus its mean over the module's baseline.
- Grids (grid_windows): columbia 5 sub-epochs per command (112.epoch_events, TMAX); cornell N_EP x EPOCH_S epochs
  from CMD_S + RESP[0]; mbmi CSP_WIN; dubois TASK; cruse11 the variant's action window; edlow17 the kept 1-s
  spectrogram columns (DROP_PHASE dropped); cruse12p 0-EPOCH_S; mbvt every stimulus, the FEATS window, baseline
  (ds7) over the pre-stimulus span features() epochs (the earliest FEATS window start, 100 ms); chennu13 the epoched
  words (labels != DIST), WINDOWS p3b (p3a + p3b for p3_any), baseline EPOCH[0]-0; gibson16 every stimulus, WIN_MS,
  baseline -PRE_S-0; sokoliuk21 every trial, TOI, no baseline. Baselines end at the onset sample (exclusive).
- Label sequence: the variant's own contrast on that grid, w = 1{A} / n_A - 1{B} / n_B (the centred indicator up to
  scale for two classes; CONTRASTS). cruse12p's two comparisons (left and right vs relax) are averaged in E. Variants
  whose test contrasts no trial labels (gibson16 SSEP; sokoliuk21's published single condition) have no label
  spectrum: X = NA with reason "no_contrast" (counted, never 0).
- E|L(f)|^2 (design_spectrum): the mean periodogram of w over N_DRAWS (1000) draws of the variant's schedule()
  (schedule_randomized() for the randomization-based rows), np.random.default_rng(20261014) per variant; exactly the
  realised periodogram when every draw gives the same spectrum (fixed schedules: "fixed"). Designs whose length varies
  between draws (chennu13, gibson16) are averaged on a zero-padded fine frequency grid and read at each row's own
  Fourier frequencies.
- periodogram: |FFT(x - mean)|^2 / N at f = k / N, k = 1 .. N // 2 (cycles per epoch), the same for feature and label.
- S_hat: log S = b - log(k + f^chi), k >= 0, |chi| <= 10, least squares on the log periodogram over the bins with
  E <= 4 x median(E) (the alternation and its harmonics excluded, so the fit never sees the test statistic's own
  in-band power; bins below 1e-12 x max E are the exact zeros of a square wave), then evaluated at every bin.
  X = log(sum S_hat E / (mean S_hat * sum E)), averaged over the channels whose fit succeeds; chi likewise. The
  failure rule applies per channel: a failed channel ("fit_failed": no convergence or an unusable bin;
  "fit_degenerate": chi on its bound, the least-squares infimum of a short white-ish periodogram being a step through
  one or two bins) is left out, and X is NA unless a strict majority of the native channels fit (S4's C3 + C4: both).
  Other NAs: "few_bins" (< 8 bins survive), "no_contrast", "feature_error". NAs are never 0; they are counted by
  stratum and design and left out of the models. The fraction of channels fitted is reported by design, immunity
  type and stratum (channel_use).
- Sensitivity X: whole recording (the periodogram averaged over the row's overlay laid on every non-overlapping clean
  window of the recording, engine.patient_windows) and module bands (MODULE_BANDS from the generated spec, X averaged
  over bands; mbmi and mbvt band-pass inside features() with no constant: the family band, flagged band_source).
- Design table (design_table): per module x variant x design kind, E's peak frequencies, design type (block: some bin
  above 4 x median; randomised: none; no_contrast), X on a reference 1/f series (S = 1 / f, no fit) and the immunity
  type (immune by design if |that X| <= 0.1, exposed otherwise) and the three-way class (x_ref_class: inflated
  > 0.1, immune, deflated < -0.1; mbvt's one-target-per-group-of-8 design is deflated).

Models (BDSP null rows; resp none, native montage, first overlay as in 115.published, status ok, every frozen variant;
Columbia on the analysis split = 113's rows as 115.merge substitutes them):
- M1 (primary): weighted logistic call ~ z(X) + pipeline-variant fixed effects. Every fit of every model (point and
  bootstrap) carries a weak L2 penalty (fe_logit, sklearn C = 100: lam / 2 (|beta|^2 + sum (alpha - mean alpha)^2),
  covariates divided by their weighted SD, intercept unpenalised), so no fit fails and no variant is dropped; the
  plain MLE is refitted on the same draws and its failures reported as a diagnostic. Row weight = 1 / (frozen BDSP
  variants of the pipeline, 114.variant_specs), divided by the patient's rows of that variant (one row per patient x
  variant expected: first overlay; more, from several recordings, share it; counted). z by the weighted mean and SD
  of the model's rows. Patient-cluster bootstrap, B = 2000, relabelled draws (boot_sample), seed [20261014,
  crc32(model name)] (one stream per model); one-sided p = (#{beta* <= 0} + 1) / (B + 1); percentile 95% CI.
- Immunity: published rows of designs immune by design (immunity type), OR per SD of chi, "shown" if the upper 95%
  bound is <= 1.25; the chi x immunity-type slope contrast (exposed minus immune, one-sided > 0); the
  randomization-based rows (rand_call, first overlay, one per timing group) separately as a construction check,
  never merged into it. Deflated group (deflated_report, descriptive, predicted: a rate at or below nominal and a
  negative chi slope): from the deflated rows only, the first-overlay false-call rate of the validity-primary and the
  default variant per pipeline with a patient-bootstrap 95% CI, and the OR per SD of chi from the immunity check's
  model.
- S1 (concentration): one unit per patient x pipeline over every non-overlapping sequence window (s1_units); O = mean
  call over its variants and windows, E = mean predicted probability of the model without X. D = sum over patients
  of (sum_pipelines O - sum_pipelines E)^2 / sum_pipelines E, against a patient-label permutation within pipeline
  (the units' (O, E) pairs permuted among the patients that pipeline ran on; 2000 permutations): within-pipeline
  dispersion is kept by the null, so excess D means different pipelines call the same patients. Lorenz/Gini of the
  patient totals; mediation = share of the excess (D - mean null D) removed when X enters, on the rows with X
  (>= 30% "largely explained").
- S2: organised (normal_awake, sleep) vs disorganised (burst_suppression, iiic_periodic, diffuse_slowing), with the
  transitioning flag as a covariate, before and after z(X); other_awake, mixed and unlabelled excluded and counted;
  weighted per-stratum rates and the transitioning flag. S3: OR per 10-point artifact share, before and after z(X).
  Two-sided bootstrap p (both have stated directions only as predictions; M2-M5 are estimates whatever their sign).
- S4 (pDoC, reduced montage): the b12 reference schedule (doc_null.schedule block 12; exact E) on log 8-30 Hz power
  at C3, C4 (doc_null.trial_features) in consecutive non-overlapping 480-s windows from the start of each recording
  (windows holding a padding second, 108.padding_seconds, are skipped and counted). Patient median X, MCS vs VS:
  Hodges-Lehmann shift with a diagnosis-stratified relabelled bootstrap CI and the two-sided Wilcoxon rank-sum p,
  (a) overall, (b) on lower-delta (wake-like) windows (91_doc_null's lower_delta_half rule: the window's mean C3/C4
  log delta/alpha below the recording median), (c) state-adjusted (OLS of median X on MCS + z(median window log
  delta/alpha)); AUROC gain of X over the resting markers on the same windows (aperiodic exponent and relative
  alpha), leave-one-patient-out ridge logistic, same bootstrap with copies of one patient left out together. The
  markers (fixed here, before any analysis-split S4 run): Welch
  PSD (4-s segments) of C3 and C4 over 2-25 Hz, FOOOF's 'fixed' aperiodic mode (no knee) with its robust refit, and
  relative alpha 8-12 Hz over the same range; the range is narrowed per recording where a corner in the EDF signal
  headers' prefilter fields (C3, C4; edf_prefilters reads the signal header block only, never bytes 8-183) puts its
  transition band inside it (above 2 x the high-pass corner, below 0.8 x the low-pass corner), and logged.
- Holm over S1 (permutation p), S2 and S3 (before X) and S4 (a, Wilcoxon).
- Sensitivity (not tests): random intercept for patient (statsmodels BinomialBayesMixedGLM, variational Bayes) on the
  a-priori default variant of each pipeline only, one row per patient x pipeline (ri_rows: equal pipeline weight, the
  class taking no row weights); whole-recording X, module-band X; flat failed channels: a channel whose fit fails but
  whose fitted bins pass the flatness check (Bartlett's cumulative periodogram test, 5%) enters as X_c = 0 (X_flat0);
  artefact mask: the module's own rejection (MASKS, by its frozen functions; columbia and dubois have none), rejected
  epochs zero-filled in the feature series and in the label weights of every draw (masked_series,
  masked_label_spectrum on design_draws); post-freeze entry: M1 on the BDSP extra listing only (engine.bdsp_extras,
  patient-disjoint from the 108 design sample; same weights, bootstrap and seed); part A8: X of the first
  principal component across the channel-standardised feature series, M1 without the Columbia rows; A9: M1 on the
  windows clear of activation procedures.

part A (estimates, no tests):
- A9 flags (a9_flags): a row's extent overlapping an interval of 121_activation's table for its source and split
  (121.windows_flagged); a recording 121 did not process (KeyError) is flagged (absent is not clean).
- Variants (variant_choices): validity primary = 115.published's selection per null set; default = 114's a-priori
  primary_variant (its as_coded flag reported); both reported wherever a variant is selected (A8).
- A1(a): first-overlay false-call rate in model-assigned awake backgrounds (normal_awake, non-transitioning,
  A9-clean), and in ICU-typical backgrounds (the disorganised strata) for wording; A1(c): the pDoC first-overlay rate
  on lower-delta (wake-like) extents (91's rule on the overlay's extent), A9-clean. Patient-bootstrap 95% CI.
- A1(b): expected false calls per 100 patients = 100 x the patient composite at the published k (read_k_table:
  docs/benchmark-k-table.json, exactly the frozen pipelines, integer k >= 1; k_sensitivity where given): the module's
  patient_rule on the first k non-overlapping BDSP windows (115.composite_curve: patients with >= k windows only;
  those with fewer are counted and their share reported, per pipeline); Columbia, 115's k = 2 bracket; mbvt and mbmi,
  the pDoC best-of-k (rule best_run).
- A2 (transport): the BDSP M1 fit (alpha by pipeline-variant, slope on z(X), z by the BDSP weighted mean and SD)
  predicts the pDoC first-overlay published rows; rows of variants absent from BDSP, without both outcomes there, or
  with X NA are excluded and counted (chennu13 runs 12 blocks on BDSP, 20 on pDoC: none shared). Calibration
  intercept (in the large: y ~ 1 + offset(lp)) and slope (y ~ 1 + lp), weighted, with a joint patient bootstrap
  (BDSP draws refit the model, pDoC draws the evaluation rows).
- On the S4 patient median X: Spearman with the CRS-R Arousal (first assessment) and AUROC for recovery (Consc vs
  PDOC or D), the latter also without deaths; both also adjusted for VS/MCS (partial Spearman on ranks;
  covariate-adjusted AUROC within diagnosis strata, Janes and Pepe 2009); patient-bootstrap 95% CIs; the fixed in
  advance wording.
- A7 (a7_spectra, descriptive): per pipeline (default variant, first overlay) x Morgoth stratum (transitioning its
  own), the mean log feature spectrum (each recording's channel-mean periodogram scaled to mean 1, on the
  pipeline's modal grid), its aperiodic fit and the residual peaks (>= 3 robust SDs and >= 1 dB above the fit), in
  cycles per epoch and approximately in Hz (/ the median epoch spacing); pDoC: the S4 windows by VS/MCS x
  lower-delta.

Splits: dev rows (output/intermediate/_bench_shards_dev) and dev recordings go to output/intermediate/_mechanism_dev;
the analysis split (output/tables/mechanism.json, rows under output/intermediate/_mechanism) runs at the fixed draw counts only. X results
are one file per recording (written whole; files of another version of this script are refused, never mixed). The
CLI prints counts, never an estimate.

X stage: the BDSP rows above and the pDoC first overlays (reduced montage, as 114 reads it: source_channels).

python 116_mechanism.py --split dev|analysis [--stage table|x|s4|stats|all] [--a3-release]
                        [--limit N --draws N --boot N --perm N --no-mixed]   (these five: dev smoke runs only)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import hashlib
import importlib
import json
import pickle
import re
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import welch
from scipy.special import expit
from scipy.stats import kstest, mannwhitneyu

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from bench import motor_common as mc
from bench import locked_spec
import doc_null as dn

rn = importlib.import_module("114_benchmark")
mg = importlib.import_module("115_benchmark_merge")
bn = importlib.import_module("108_bdsp_null")
en = rn.en
wr = rn.wr

SEED = 20261014
N_DRAWS, B_BOOT, N_PERM = 1000, 2000, 2000
PEAK_RATIO, MIN_BINS = 4.0, 8
ROUNDOFF = 1e-12
MOTOR_BAND, ERP_BAND = tuple(en.BAND), (1.0, 20.0)
LOG_FLOOR = 1e-12
CHI_MAX, MAX_NFEV = 10.0, 5000
PAD_FACTOR = 16
IMMUNITY_MAX, LARGELY = 1.25, 0.30
ART_STEP = 0.10
ORGANISED = ("normal_awake", "sleep")
DISORGANISED = ("burst_suppression", "iiic_periodic", "diffuse_slowing")
BDSP_NULLS = ("ncmd_bdsp", "nstim0_bdsp")
PDOC_NULLS = ("ncmd_pdoc", "nstim0_pdoc")
SOURCE = "bdsp"
PIPES = list(locked_spec.PIPES)
EDLOW_COL_S = 1.0
PENALTY_C = 100.0
IMMUNE_XREF = 0.1
FLAT_ALPHA = 0.05
MARKER_HZ, ALPHA_HZ, WELCH_S = (2.0, 25.0), (8.0, 12.0), 4.0
HP_CLEAR, LP_CLEAR = 2.0, 0.8
RIDGE = 1.0
K_TABLE = os.path.join(ROOT, "docs", "benchmark-k-table.json")
PEAK_SD, PEAK_MIN_DB = 3.0, 1.0
A3_MAP = os.path.join(ROOT, "docs", "pdoc-case-map.csv")
A3_MAP_SHA256 = "c959cce7c9a856f7ccf3039b83a0cbca0aa07e005a32590c9d82e6430d04f212"
A3_OUTCOMES_SHA256 = "7f601f6e035d07b6eb0358a6610508facc2c842c6b0050c567ce4dfa17c6c7fe"
A3_OUTCOMES = os.path.join(ROOT, "output", "raw", "pdoc_clinical", "pdoc_outcomes.csv")
A3_AROUSAL, A3_OUTCOME = "crs_r_arousal_first", "outcome"



def periodogram(x):
    """(f, P): |FFT(x - mean)|^2 / N along the last axis at f = k / N, k = 1 .. N // 2."""
    x = np.asarray(x, float)
    n = x.shape[-1]
    P = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2 / n
    return np.arange(1, n // 2 + 1) / n, P[..., 1:n // 2 + 1]


def contrast_weights(labels, cons):
    """[w] per contrast (pos labels, neg labels) whose two sides are both present:
    w = 1{pos} / n_pos - 1{neg} / n_neg."""
    lab = np.asarray(labels)
    out = []
    for pos, neg in cons:
        a, b = np.isin(lab, pos), np.isin(lab, neg)
        if a.any() and b.any():
            out.append(a / a.sum() - b / b.sum())
    return out


def label_spectrum(ws):
    """Mean periodogram of the contrasts' weight sequences (None without a contrast)."""
    if not ws:
        return None
    return np.mean([periodogram(w)[1] for w in ws], 0)


def peak_bins(E):
    """Bins with E > 4 x median(E); values below ROUNDOFF x max(E) count as the exact zeros they are (FFT round-off in
    a square wave's empty bins would otherwise set the median)."""
    E = np.asarray(E, float)
    E = np.where(E < ROUNDOFF * E.max(), 0.0, E)
    return E > PEAK_RATIO * np.median(E)


def aperiodic(f, p):
    b, k, chi = p
    return np.exp(b) / (k + np.asarray(f, float) ** chi)


def fit_aperiodic(f, P):
    """((b, k, chi), None) of log P = b - log(k + f^chi), k >= 0, |chi| <= CHI_MAX, least squares on log P (trf, at
    most FOOOF's 5000 evaluations), or (None, reason). Failed fits: "fit_failed" (a non-positive or non-finite bin, an
    optimizer error, no convergence) and "fit_degenerate" (chi on its bound: on a short white-ish periodogram the
    least-squares infimum is a step through one or two bins and the parameters diverge; no exponent of that size is
    physiological)."""
    f, P = np.asarray(f, float), np.asarray(P, float)
    if len(f) < 3 or not np.all(np.isfinite(P)) or np.any(P <= 0):
        return None, "fit_failed"
    y, lf = np.log(P), np.log(f)
    with np.errstate(all="ignore"):
        slope, icpt = np.polyfit(lf, y, 1)
        x0 = [float(icpt), 0.0, float(np.clip(-slope, -CHI_MAX, CHI_MAX))]

        def res(p):
            return p[0] - np.log(p[1] + f ** p[2]) - y

        def jac(p):
            fc = f ** p[2]
            d = p[1] + fc
            return np.column_stack([np.ones_like(f), -1.0 / d, -fc * lf / d])
        try:
            r = least_squares(res, x0, jac=jac, bounds=([-np.inf, 0.0, -CHI_MAX], [np.inf, np.inf, CHI_MAX]),
                              method="trf", max_nfev=MAX_NFEV)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            return None, "fit_failed"
    if not r.success or not np.all(np.isfinite(r.x)):
        return None, "fit_failed"
    if abs(r.x[2]) >= CHI_MAX - 1e-6:
        return None, "fit_degenerate"
    return tuple(float(v) for v in r.x), None


def _na(reason, n_bins=0):
    return {"X": np.nan, "chi": np.nan, "n_bins": int(n_bins), "reason": reason, "X_flat0": np.nan}


def is_flat(P):
    """Flatness check of a periodogram (Bartlett's cumulative periodogram test at FLAT_ALPHA): under a flat spectrum the
    bins are i.i.d. exponential, so the normalised partial sums are uniform order statistics (Kolmogorov-Smirnov).
    False for fewer than 3 bins or a zero, negative or non-finite periodogram (a constant channel is not flat)."""
    P = np.asarray(P, float)
    if len(P) < 3 or not np.all(np.isfinite(P)) or np.any(P < 0) or P.sum() <= 0:
        return False
    return bool(kstest(np.cumsum(P)[:-1] / P.sum(), "uniform").pvalue >= FLAT_ALPHA)


def overlap_from_periodogram(f, P, E, exclude=True):
    """X and chi from channel periodograms P (channels, bins) and the label spectrum E on the same bins. X_flat0
    (sensitivity): a failed channel whose fitted bins pass the flatness check enters as X_c = 0 (a flat S gives X = 0
    exactly), with the same strict-majority rule over fitted plus flat channels."""
    if E is None:
        return _na("no_contrast")
    E = np.asarray(E, float)
    if E.shape != f.shape:
        raise ValueError(f"label spectrum has {E.shape} bins, the feature periodogram {f.shape}")
    if not np.all(np.isfinite(E)) or E.sum() <= 0:
        return _na("no_contrast")
    keep = ~peak_bins(E) if exclude else np.ones(len(f), bool)
    n = int(keep.sum())
    if n < MIN_BINS:
        return _na("few_bins", n)
    P = np.atleast_2d(P)
    xs, chis, whys, n_flat = [], [], [], 0
    for p in P:
        fit, why = fit_aperiodic(f[keep], p[keep])
        with np.errstate(all="ignore"):
            S = None if fit is None else aperiodic(f, fit)
        if S is None or not np.all(np.isfinite(S)) or np.any(S <= 0):
            whys.append(why or "fit_failed")
            n_flat += is_flat(p[keep])
            continue
        xs.append(np.log(np.sum(S * E) / (S.mean() * E.sum())))
        chis.append(fit[2])
    used = len(xs) + n_flat
    cnt = {"n_ch": int(len(P)), "n_ch_fit": len(xs), "n_ch_flat": int(n_flat),
           "X_flat0": float(np.sum(xs) / used) if 2 * used > len(P) else np.nan}
    if 2 * len(xs) <= len(P):
        return {**_na(max(sorted(set(whys)), key=whys.count), n), **cnt}
    return {"X": float(np.mean(xs)), "chi": float(np.mean(chis)), "n_bins": n, "reason": None, **cnt}


def overlap(F, E, exclude=True):
    """X of a feature series F ((channels) epochs) against the label spectrum E (overlap_from_periodogram)."""
    f, P = periodogram(np.atleast_2d(np.asarray(F, float)))
    return overlap_from_periodogram(f, P, E, exclude)


def x_reference(E):
    """X on a reference 1/f series (S = 1 / f, no fit); NA without a contrast."""
    if E is None or np.sum(E) <= 0:
        return np.nan
    E = np.asarray(E, float)
    S = 1.0 / np.arange(1, len(E) + 1)
    return float(np.log(np.sum(S * E) / (S.mean() * E.sum())))



class DesignSpectrum:
    """E|L(f)|^2 of one design. at(n): E on the Fourier bins of an n-epoch grid (None without a contrast)."""

    def __init__(self, none=False, fixed=False, exact=None, fine=None, ns=(), n=None, n_empty=0):
        self.none, self.fixed, self.exact, self.fine = none, fixed, exact or {}, fine
        self.ns, self.n, self.n_empty = tuple(sorted(set(ns))), n, n_empty

    def at(self, n):
        if self.none:
            return None
        if n in self.exact:
            return self.exact[n]
        if self.fine is None:
            raise KeyError(f"design drew no {n}-epoch grid (lengths {self.ns})")
        return np.interp(np.arange(1, n // 2 + 1) / n, *self.fine)

    def state(self):
        return dict(none=self.none, fixed=self.fixed, exact=self.exact, fine=self.fine, ns=self.ns, n=self.n,
                    n_empty=self.n_empty)


def expected_spectrum(draw, n_draws=N_DRAWS, seed=SEED):
    """DesignSpectrum from n_draws calls of draw(g) -> [contrast weights] with one generator, default_rng(seed).
    Exact (the realised spectrum) when every draw gives the same spectrum; one length: the mean periodogram; varying
    lengths: the mean zero-padded spectrum on a fine grid (PAD_FACTOR x the longest), read off at each grid."""
    g = np.random.default_rng(seed)
    draws, empty = [], 0
    for _ in range(n_draws):
        ws = draw(g)
        if ws:
            draws.append(ws)
        else:
            empty += 1
    if not draws:
        return DesignSpectrum(none=True, n_empty=empty)
    ns = [len(ws[0]) for ws in draws]
    n_mode = int(pd.Series(ns).mode().min())
    if len(set(ns)) == 1:
        P = np.array([label_spectrum(ws) for ws in draws])
        fixed = bool(np.all(P == P[0]))
        return DesignSpectrum(fixed=fixed, exact={ns[0]: P[0] if fixed else P.mean(0)}, ns=ns, n=n_mode,
                              n_empty=empty)
    pad = int(2 ** np.ceil(np.log2(PAD_FACTOR * max(ns))))
    acc = np.zeros(pad // 2 + 1)
    for ws in draws:
        acc += np.mean([np.abs(np.fft.rfft(w - w.mean(), n=pad)) ** 2 / len(w) for w in ws], 0)
    return DesignSpectrum(fine=(np.arange(pad // 2 + 1) / pad, acc / len(draws)), ns=ns, n=n_mode, n_empty=empty)


def design_spectrum(mod, spec, kind, n_draws=N_DRAWS, seed=SEED):
    """E of a variant's published (schedule) or randomized (schedule_randomized) design on its own grid."""
    fn = mod.schedule if kind == "published" else mod.schedule_randomized
    cfg = module_cfg(mod, spec)
    cons = contrasts(mod, cfg)

    def draw(g):
        ov = rn._sched(mod, fn, spec, rn.GROUP_START_S, g)
        return contrast_weights(grid_labels(mod, cfg, ov), cons)
    return expected_spectrum(draw, n_draws, seed)


class Spectra:
    """Design spectra by (pipe, variant, kind), computed on demand and kept in a pickle of plain dicts."""

    def __init__(self, path=None, n_draws=N_DRAWS, seed=SEED):
        self.path, self.n_draws, self.seed, self.d = path, n_draws, seed, {}
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                st = pickle.load(fh)
            if st.get("n_draws") == n_draws and st.get("seed") == seed:
                self.d = {k: DesignSpectrum(**v) for k, v in st["spectra"].items()}

    def get(self, pipe, variant, kind):
        k = (pipe, variant, kind)
        if k not in self.d:
            mod = rn.load_pipe(pipe)
            self.d[k] = design_spectrum(mod, rn._spec(mod, variant), kind, self.n_draws, self.seed)
        return self.d[k]

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path + ".tmp", "wb") as fh:
                pickle.dump({"n_draws": self.n_draws, "seed": self.seed,
                             "spectra": {k: v.state() for k, v in self.d.items()}}, fh)
            os.replace(self.path + ".tmp", self.path)


def design_type(ds):
    if ds.none:
        return "no_contrast"
    return "block" if peak_bins(ds.at(ds.n)).any() else "randomised"


def x_ref_class(x_ref):
    """Three-way design typing from X on the reference 1/f series: "inflated" (> IMMUNE_XREF), "immune" (|X_ref| <=
    IMMUNE_XREF) or "deflated" (< -IMMUNE_XREF); None without a contrast."""
    if x_ref is None or not np.isfinite(x_ref):
        return None
    return "immune" if abs(x_ref) <= IMMUNE_XREF else ("inflated" if x_ref > 0 else "deflated")


def immunity_type(x_ref):
    """Immunity check typing (mechanistic): "immune" by design if |X on the reference 1/f series| <= IMMUNE_XREF,
    "exposed" otherwise (inflated or deflated); None without a contrast. design_type's 4 x median rule stays for the
    S_hat bin exclusion."""
    c = x_ref_class(x_ref)
    return None if c is None else ("immune" if c == "immune" else "exposed")


def table_row(pipe, variant, kind, ds):
    row = {"pipe": pipe, "variant": variant, "kind": kind, "design_type": design_type(ds), "fixed": ds.fixed,
           "n_epochs": ds.n, "n_lengths": len(ds.ns), "n_peak_bins": 0, "peak_ratio": np.nan, "peak_freqs": "",
           "x_ref_1f": np.nan, "immunity_type": None, "x_ref_class": None}
    if ds.none:
        return row
    E = ds.at(ds.n)
    f = np.arange(1, len(E) + 1) / ds.n
    pk = peak_bins(E)
    top = np.argsort(-E[pk], kind="stable")[:10]
    med = np.median(E)
    row.update(n_peak_bins=int(pk.sum()), peak_ratio=float(E.max() / med) if med > 0 else np.inf,
               peak_freqs=";".join(f"{v:.4f}" for v in f[pk][top]), x_ref_1f=x_reference(E),
               immunity_type=immunity_type(x_reference(E)), x_ref_class=x_ref_class(x_reference(E)))
    return row


def design_table(pipes=None, n_draws=N_DRAWS, seed=SEED, spectra=None):
    """Per module x frozen BDSP variant x design kind: E's peak frequencies (cycles per epoch, largest first, up to
    10), the design type and X on a reference 1/f series."""
    sp = spectra or Spectra(None, n_draws, seed)
    rows = []
    for pipe in pipes or PIPES:
        mod = rn.load_pipe(pipe)
        src = {}
        for source in (SOURCE, "pdoc"):
            try:
                names = rn.variant_specs(mod, source) if source == SOURCE or wr.pdoc_overrides(mod) is not None else {}
            except ValueError:
                names = {}
            for v in names:
                src.setdefault(v, []).append(source)
        for v, sources in src.items():
            for kind in ("published", "randomized"):
                rows.append(dict(table_row(pipe, v, kind, sp.get(pipe, v, kind)), sources="+".join(sources)))
    return pd.DataFrame(rows)



def module_cfg(mod, spec):
    return wr._cfg(mod, spec["arg"])


def _on(ov):
    return np.asarray(ov.onsets_s, float)


def _around(on, w):
    return on[:, None] + np.asarray(w, float)[None]


def _g_columbia(mod, cfg, ov, fs):
    ce = mod.ce
    lab = np.repeat(np.asarray(ov.labels), ce.N_SEG)
    if fs is None:
        return lab, None, None
    start, _, _ = ce.epoch_events(ov.onsets_s, np.asarray(ov.meta.get("canonical", ov.labels)), fs)
    t0 = np.asarray(start, float) / fs
    return lab, np.column_stack([t0, t0 + (int(round(ce.TMAX * fs)) + 1) / fs]), None


def _g_cornell(mod, cfg, ov, fs):
    t0 = (_on(ov)[:, None] + mod.CMD_S + mod.RESP[0] + np.arange(mod.N_EP)[None] * mod.EPOCH_S).ravel()
    return np.repeat(np.asarray(ov.labels), mod.N_EP), np.column_stack([t0, t0 + mod.EPOCH_S]), None


def _g_edlow17(mod, cfg, ov, fs):
    starts = _on(ov)[np.asarray(ov.meta["block_start"], bool)]
    cols = (starts[:, None] + np.arange(mod.BLOCK_S)[None]).ravel()
    keep = ~np.isin(np.tile(np.arange(mod.BLOCK_S) % mod.CYCLE_S, mod.N_BLOCKS), mod.DROP_PHASE)
    lab = np.asarray(ov.meta["block_class"])[np.repeat(np.arange(mod.N_BLOCKS), mod.BLOCK_S)]
    return lab[keep], np.column_stack([cols, cols + EDLOW_COL_S])[keep], None


def _g_mbvt(mod, cfg, ov, fs):
    f = mod.FEATS[cfg["feat"]]
    pre = min(v["win"][0] for v in mod.FEATS.values())
    on = _on(ov)
    return np.asarray(ov.labels), _around(on, f["win"]), (_around(on, (pre, 0.0)) if f["base"] else None)


def _g_chennu13(mod, cfg, ov, fs):
    sel = np.asarray(ov.labels) != mod.DIST
    w = mod.WINDOWS["p3b"] if cfg["rule"] != "p3_any" else (mod.WINDOWS["p3a"][0], mod.WINDOWS["p3b"][1])
    on = _on(ov)[sel]
    return np.asarray(ov.labels)[sel], _around(on, np.asarray(w) / 1000.0), _around(on, (mod.EPOCH[0], 0.0))


def _g_window(attr):
    def g(mod, cfg, ov, fs):
        w = attr(mod, cfg)
        return np.asarray(ov.labels), _around(_on(ov), w), None
    return g


def _g_gibson16(mod, cfg, ov, fs):
    on = _on(ov)
    return np.asarray(ov.labels), _around(on, np.asarray(mod.WIN_MS) / 1000.0), _around(on, (-mod.PRE_S, 0.0))


GRIDS = {"columbia": _g_columbia, "cornell": _g_cornell, "edlow17": _g_edlow17, "mbvt": _g_mbvt,
         "chennu13": _g_chennu13, "gibson16": _g_gibson16,
         "mbmi": _g_window(lambda m, c: m.CSP_WIN), "dubois": _g_window(lambda m, c: m.TASK),
         "cruse11": _g_window(lambda m, c: c["action"]), "cruse12p": _g_window(lambda m, c: (0.0, m.EPOCH_S)),
         "sokoliuk21": _g_window(lambda m, c: m.TOI)}


def grid_labels(mod, cfg, ov):
    """The label of every epoch of the module's grid, in time order."""
    return GRIDS[mod.NAME](mod, cfg, ov, None)[0]


def grid_windows(mod, cfg, ov, fs):
    """{lab, win (epochs, 2), base (epochs, 2) or None}: absolute seconds, half-open [t0, t1)."""
    lab, win, base = GRIDS[mod.NAME](mod, cfg, ov, fs)
    return {"lab": lab, "win": win, "base": base}


def _gibson_contrast(m, c):
    on = c.get("call_on")
    if on == "p3b":
        return [((m.TARGET,), (m.NONTARGET,))]
    if on == "p3a":
        return [((m.TARGET, m.NONTARGET), (m.BACK,))]
    return []


CONTRASTS = {"columbia": lambda m, c: [((m.ce.KEEP_L, m.ce.KEEP_R), (m.ce.STOP_L, m.ce.STOP_R))],
             "cornell": lambda m, c: [((m.TASK,), (m.REST,))],
             "mbmi": lambda m, c: [((m.LEFT,), (m.RIGHT,))],
             "dubois": lambda m, c: [((m.CLASS_A,), (m.CLASS_B,))],
             "cruse11": lambda m, c: [((m.HAND,), (m.TOE,))],
             "edlow17": lambda m, c: [((m.SQUEEZE,), (m.REST,))],
             "cruse12p": lambda m, c: [((m.LEFT,), (m.RELAX,)), ((m.RIGHT,), (m.RELAX,))],
             "mbvt": lambda m, c: [((m.TARGET,), (m.NONTARGET, m.DISTRACTOR))],
             "chennu13": lambda m, c: [((m.EXPL,), (m.IMPL,))],
             "gibson16": _gibson_contrast,
             "sokoliuk21": lambda m, c: [((m.SENT,), (m.LIST,))]}


def contrasts(mod, cfg):
    return CONTRASTS[mod.NAME](mod, cfg)


MODULE_BANDS = {"columbia": lambda m: list(m.ce.BANDS), "cornell": lambda m: [tuple(m.FPASS)],
                "dubois": lambda m: list(m.BANDS.values()), "cruse11": lambda m: list(m.BANDS),
                "edlow17": lambda m: list(m.BANDS), "cruse12p": lambda m: list(m.BANDS),
                "chennu13": lambda m: [(m.HP_HZ, m.LP_HZ)], "gibson16": lambda m: [tuple(m.BAND)],
                "sokoliuk21": lambda m: [(None, m.LP_HZ)]}


def module_bands(mod):
    """(bands, source): the module's own feature bands, or the family band where features() hard-codes its filter
    (mbmi, mbvt)."""
    if mod.NAME in MODULE_BANDS:
        return [tuple(b) for b in MODULE_BANDS[mod.NAME](mod)], "module"
    return [ERP_BAND if mod.FAMILY == "erp" else MOTOR_BAND], "family"


def module_rows(mod, rec):
    """Channel rows the module reads: NATIVE_CH when all present, else its SUB_1020 (T3-T6 aliased)."""
    nm = mc.canon(rec.names)
    use = mc.canon(mod.NATIVE_CH)
    if not all(c in nm for c in use):
        if not hasattr(mod, "SUB_1020"):
            raise ValueError(f"{mod.NAME}: native channels missing and no 10-20 substitute set")
        use = mc.canon(mod.SUB_1020)
    return [nm.index(c) for c in use]


def _avg_ref(m, rec, x):
    r = x[module_rows(m, rec)]
    return r - r.mean(0, keepdims=True)


def _bipolar(m, rec, x):
    r = x[module_rows(m, rec)]
    return np.stack([r[0] - r[1], r[2] - r[3]])


SPATIAL = {"columbia": lambda m, rec, x: m.ce._csd(x[module_rows(m, rec)], *m.ce._CSD),
           "cornell": lambda m, rec, x: mc.hjorth_matrix(m.NATIVE_CH, 4, 3, "inverse_arc") @ x[module_rows(m, rec)],
           "edlow17": lambda m, rec, x: mc.hjorth_matrix(m.NATIVE_CH) @ x[module_rows(m, rec)],
           "cruse11": lambda m, rec, x: (mc.hjorth_matrix(rec.names, k_interior=4, k_edge=4, weights="equal") @ x)[
               module_rows(m, rec)],
           "cruse12p": _bipolar, "chennu13": _avg_ref, "sokoliuk21": _avg_ref}


def spatial(mod, rec, x):
    """(x', names): the module's spatial transform (SPATIAL) of x (rows = rec.names, any time span) on the channels it
    reads; the channels unchanged for a module without one."""
    names = [mc.canon(rec.names)[i] for i in module_rows(mod, rec)]
    x = np.asarray(x, float)
    if mod.NAME not in SPATIAL:
        return x[module_rows(mod, rec)], names
    if mod.NAME == "cruse12p":
        names = [f"{names[0]}-{names[1]}", f"{names[2]}-{names[3]}"]
    return SPATIAL[mod.NAME](mod, rec, x), names


def feature_series(rec, mod, cfg, ov, band=None, grid=None):
    """(channels, epochs) family-generic feature series of the overlay on the module's spatially transformed channels
    (module docstring)."""
    fs = float(rec.fs)
    g = grid or grid_windows(mod, cfg, ov, fs)
    erp = mod.FAMILY == "erp"
    lo, hi = band or (ERP_BAND if erp else MOTOR_BAND)
    seg, i0 = mc.span(np.asarray(rec.x), fs, *mod.extent(ov))
    xf = mc.bandpass(spatial(mod, rec, seg)[0], fs, lo, hi)
    cs = np.concatenate([np.zeros((xf.shape[0], 1)), np.cumsum(xf if erp else xf ** 2, axis=1)], axis=1)

    def mean_over(w):
        a = np.round(w[:, 0] * fs).astype(int) - i0
        b = np.round(w[:, 1] * fs).astype(int) - i0
        if a.min() < 0 or b.max() > xf.shape[1] or np.any(b <= a):
            raise ValueError(f"{mod.NAME}: an epoch window falls outside the module's extent")
        return (cs[:, b] - cs[:, a]) / (b - a)
    F = mean_over(g["win"])
    if erp:
        return F - mean_over(g["base"]) if g["base"] is not None else F
    return np.log(F + LOG_FLOOR)



def _mask_cornell(m, cfg, rec, ov):
    return m.kept(m.features(rec, ov), np.repeat(np.asarray(ov.labels), m.N_EP), cfg["prune"])


def _mask_edlow17(m, cfg, rec, ov):
    keep = m.features(rec, ov)[2]
    return keep[~np.isin(np.tile(np.arange(m.BLOCK_S) % m.CYCLE_S, m.N_BLOCKS), m.DROP_PHASE)]


MASKS = {"cornell": _mask_cornell, "edlow17": _mask_edlow17,
         "mbmi": lambda m, c, rec, ov: m.features(rec, ov)[1],
         "cruse11": lambda m, c, rec, ov: m.features(rec, ov, c["bp"], c["action"])[2],
         "cruse12p": lambda m, c, rec, ov: m.features(rec, ov)[1],
         "gibson16": lambda m, c, rec, ov: m.features(rec, ov)["keep"],
         "chennu13": lambda m, c, rec, ov: m._keep(m.features(rec, ov)["ep"], c["reject"]),
         "sokoliuk21": lambda m, c, rec, ov: m._keep(m.features(rec, ov, c["filters"]), c["reject"]),
         "mbvt": lambda m, c, rec, ov: m.features(rec, ov, c["feat"])["keep"][np.asarray(ov.meta["group"])]}


MASK_ARGS = {"cornell": ("prune",), "chennu13": ("reject",), "sokoliuk21": ("filters", "reject")}


def artefact_mask(mod, cfg, rec, ov):
    """Kept epochs (bool, on the module grid) under the module's artefact rule, or None for a module without one."""
    if mod.NAME not in MASKS:
        return None
    keep = np.asarray(MASKS[mod.NAME](mod, cfg, rec, ov), bool)
    n = len(grid_labels(mod, cfg, ov))
    if len(keep) != n:
        raise RuntimeError(f"{mod.NAME}: artefact mask over {len(keep)} epochs, grid of {n}")
    return keep


def masked_series(F, keep):
    """The feature series with rejected epochs zero-filled after centring each channel on its kept epochs."""
    F = np.atleast_2d(np.asarray(F, float))
    out = np.zeros_like(F)
    out[:, keep] = F[:, keep] - F[:, keep].mean(1, keepdims=True)
    return out


def masked_label_spectrum(draws, cons, keep):
    """E|L|^2 with rejected epochs zero-filled in the weights: per draw (label arrays on the grid) the contrasts are
    rebuilt on the kept epochs only (contrast_weights) and their periodograms averaged; draws without a contrast are
    left out. None if no draw has one."""
    L = np.asarray(draws)
    keep = np.asarray(keep, bool)
    acc, nc = np.zeros((len(L), L.shape[1] // 2)), np.zeros(len(L))
    for pos, neg in cons:
        a, b = np.isin(L, pos) & keep, np.isin(L, neg) & keep
        na, nb = a.sum(1), b.sum(1)
        ok = (na > 0) & (nb > 0)
        W = a / np.maximum(na, 1)[:, None] - b / np.maximum(nb, 1)[:, None]
        acc += np.where(ok[:, None], periodogram(W)[1], 0.0)
        nc += ok
    if not nc.any():
        return None
    return (acc[nc > 0] / nc[nc > 0, None]).mean(0)


_DRAWS, _DRAW_SIGS = {}, {}


def _small_int(a):
    return a.astype(np.int8) if a.dtype.kind in "iu" and np.abs(a).max() < 127 else a


def design_draws(mod, spec, kind, n_draws=N_DRAWS, seed=SEED):
    """{n: (draws, n) grid labels}: the same draws as design_spectrum (same generator and calls), by grid length;
    cached for the process, identical draw sets stored once."""
    k = (mod.NAME, json.dumps(spec, sort_keys=True, default=str), kind, n_draws, seed)
    if k not in _DRAW_SIGS:
        fn = mod.schedule if kind == "published" else mod.schedule_randomized
        cfg = module_cfg(mod, spec)
        g = np.random.default_rng(seed)
        out = {}
        for _ in range(n_draws):
            lab = np.asarray(grid_labels(mod, cfg, rn._sched(mod, fn, spec, rn.GROUP_START_S, g)))
            out.setdefault(len(lab), []).append(lab)
        out = {n: _small_int(np.array(v)) for n, v in out.items()}
        sig = hashlib.sha1(b"".join(f"{n}:".encode() + v.tobytes() for n, v in sorted(out.items()))).hexdigest()
        _DRAWS.setdefault(sig, out)
        _DRAW_SIGS[k] = sig
    return _DRAWS[_DRAW_SIGS[k]]



def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def guard_split(split, spec_path=None):
    """114's split guard (the check against the private working copy is not applied here)."""
    rn.guard_split(split, spec_path)


def paths(split):
    if split == "dev":
        base = tables = os.path.join(ROOT, "output", "intermediate", "_mechanism_dev")
        summary = "mechanism_dev.json"
    else:
        base = os.path.join(ROOT, "output", "intermediate", "_mechanism")
        tables = os.path.join(ROOT, "output", "tables")
        summary = "mechanism.json"
    return {"results": os.path.join(base, "x_rows"), "spectra": os.path.join(base, "design_spectra.pkl"),
            "s4": os.path.join(base, "s4_windows.parquet"), "table": os.path.join(tables, "mechanism_design_table.csv"),
            "summary": os.path.join(tables, summary)}


def load_rows(split):
    """115's rows of the split (dev rows refused in analysis and vice versa); analysis: touched recordings dropped and
    Columbia's first overlays replaced by 113's rows, exactly as 115.merge does."""
    df = mg.load_rows(split)
    if split == "dev" or df.empty:
        return df
    df, _ = mg.drop_touched(df)
    c = rn.columbia113()
    col = (df["null"] == "ncmd_bdsp") & (df["pipe"] == "columbia") & (df["resp"] == "none") & (df["j"] == 0)
    mg.check_113(c, df[~col], mg.dev_uids(), mg.touched_uids())
    return pd.concat([df[~col], mg.rows_from_113(c)], ignore_index=True, sort=False)


def s1_rows(df):
    """Published BDSP null rows of every non-overlapping sequence window (115._null_seq: ok, no responder, native)."""
    d = mg._null_seq(df)
    return d[d["null"].isin(BDSP_NULLS)]


def m1_rows(df):
    """The first overlay (j = 0) of every variant: the rows of M1, S2 and S3."""
    d = s1_rows(df)
    return d[d["j"] == 0]


def rand_rows(df):
    """The randomization-based method's first-overlay draws, one per recording x pipeline x timing group."""
    d = m1_rows(df)
    if "rand_call" not in d:
        return d.iloc[:0]
    d = d[d["rand_call"].notna()]
    return d.drop_duplicates(["uid", "pipe", "rand_variant"])


def _is113(row):
    return str(row.get("key", "")).startswith("113|")


def overlay_for_row(mod, specs, row, kind="published"):
    """The row's overlay rebuilt with the runner's own seeds (114 _run_overlay / _rand_fields); refused unless it gives
    the row's extent (and onset count)."""
    start, uid, j = float(row["start_s"]), str(row["uid"]), int(row["j"])
    if _is113(row):
        return mod.schedule(start) if kind == "published" else mod.schedule_randomized(start, 0)
    if kind == "published":
        ov = rn._sched(mod, mod.schedule, specs[row["variant"]], start, rn._rng(mod.NAME, uid, j, 1))
        want, n = (row["extent_t0"], row["extent_t1"]), row.get("n_onsets")
    else:
        g = row["rand_variant"]
        ov = rn._sched(mod, mod.schedule_randomized, specs[g], start, rn._rng(mod.NAME, uid, j, 3, rn._crc(g)))
        want, n = (row["rand_extent_t0"], row["rand_extent_t1"]), None
    got = mod.extent(ov)
    if (abs(got[0] - float(want[0])) > 1e-6 or abs(got[1] - float(want[1])) > 1e-6
            or (n is not None and pd.notna(n) and len(ov.onsets_s) != int(n))):
        raise RuntimeError(f"{uid} {mod.NAME} {row.get('variant')} j={j} ({kind}): the overlay does not rebuild "
                           f"(extent {got} vs row {tuple(want)})")
    return ov


def _shifted(ov, delta):
    return en.Overlay(onsets_s=_on(ov) + delta, labels=ov.labels, kind=ov.kind, meta=ov.meta)


@contextlib.contextmanager
def source_channels(mod, source):
    """The module reads its pDoC reduced-montage channel set on pDoC (bench.wrappers.pdoc_overrides, as 114 runs it)."""
    if source != "pdoc":
        yield
        return
    g = wr.pdoc_overrides(mod)
    if g is None:
        raise ValueError(wr.PDOC_NA_MODULES[mod.NAME])
    with wr.module_globals(mod, g):
        yield


def pc1_series(F):
    """The first principal component across channels of the channel-standardised feature series, unit SD (A8:
    cross-channel coherence). Constant channels are left out."""
    F = np.atleast_2d(np.asarray(F, float))
    sd = F.std(1)
    Z = (F[sd > 0] - F[sd > 0].mean(1, keepdims=True)) / sd[sd > 0, None]
    if len(Z) == 0:
        return np.zeros(F.shape[1])
    if len(Z) == 1:
        return Z[0]
    v = np.linalg.eigh(Z @ Z.T / Z.shape[1])[1][:, -1]
    pc = v @ Z
    return pc / pc.std()


class RecordingX:
    """X of every needed row of one recording; feature series and fits cached by grid and label spectrum."""

    def __init__(self, rec, spectra):
        self.rec, self.sp, self.F, self.X, self.M = rec, spectra, {}, {}, {}

    def features(self, mod, cfg, ov, band=None):
        g = grid_windows(mod, cfg, ov, self.rec.fs)
        k = (mod.NAME, band, g["win"].tobytes(), None if g["base"] is None else g["base"].tobytes())
        if k not in self.F:
            with source_channels(mod, self.rec.source):
                self.F[k] = (feature_series(self.rec, mod, cfg, ov, band, g), k)
        return self.F[k]

    def x(self, mod, cfg, ov, E, band=None, pc1=False):
        try:
            F, fk = self.features(mod, cfg, ov, band)
        except (ValueError, IndexError) as e:
            return dict(_na(f"feature_error: {str(e)[:120]}"), n_epochs=None)
        Ek = None if E is None else hashlib.sha1(np.asarray(E, float).tobytes()).hexdigest()
        if (fk, Ek, pc1) not in self.X:
            if E is not None and len(E) != F.shape[1] // 2:
                raise RuntimeError(f"{mod.NAME}: label spectrum for {2 * len(E)} epochs, grid of {F.shape[1]}")
            self.X[(fk, Ek, pc1)] = dict(overlap(pc1_series(F) if pc1 else F, E), n_epochs=int(F.shape[1]))
        return self.X[(fk, Ek, pc1)]

    def spectrum(self, pipe, variant, kind, n):
        ds = self.sp.get(pipe, variant, kind)
        return None if ds.none else ds.at(n)

    def mask(self, mod, cfg, ov):
        k = (mod.NAME, tuple(str(cfg[a]) for a in MASK_ARGS.get(mod.NAME, ())), _on(ov).tobytes(),
             np.asarray(ov.labels).tobytes(), json.dumps(ov.meta.get("group") if mod.NAME == "mbvt" else None,
                                                         default=lambda v: np.asarray(v).tolist()))
        if k not in self.M:
            with source_channels(mod, self.rec.source):
                self.M[k] = artefact_mask(mod, cfg, self.rec, ov)
        return self.M[k]

    def x_mask(self, mod, spec, kind, cfg, ov, o):
        """Sensitivity: X with the module's rejected epochs zero-filled in the feature series and in the label weights
        (o: the row's X, returned unchanged when the module rejects nothing)."""
        same = {"X": o["X"], "reason": o["reason"], "n_rejected": 0}
        try:
            keep = self.mask(mod, cfg, ov)
        except (ValueError, IndexError, mc.NoUsableTrials) as e:
            return dict(_na(f"mask_error: {str(e)[:120]}"), n_rejected=None, mask="module")
        if keep is None:
            return dict(same, mask="none")
        if keep.all() or o["reason"] == "no_contrast" or str(o["reason"]).startswith("feature_error"):
            return dict(same, n_rejected=int((~keep).sum()), mask="module")
        F, _ = self.features(mod, cfg, ov)
        draws = design_draws(mod, spec, kind, self.sp.n_draws, self.sp.seed).get(len(keep))
        if draws is None:
            return dict(_na("mask_no_draw_of_this_length"), n_rejected=int((~keep).sum()), mask="module")
        r = overlap(masked_series(F, keep), masked_label_spectrum(draws, contrasts(mod, cfg), keep))
        return {"X": r["X"], "reason": r["reason"], "n_rejected": int((~keep).sum()), "mask": "module"}

    def x_whole(self, mod, variant, cfg, ov, E, start_s):
        """X from the periodogram averaged over the overlay laid on every non-overlapping clean window of the
        variant's own length (114.need_seconds) tiled over the recording (engine.patient_windows)."""
        need = rn.need_seconds(mod, [variant], self.rec.source)
        Ps, f = [], None
        for _, s in en.patient_windows([self.rec], need):
            ovt = _shifted(ov, s + rn.LEAD_S - start_s)
            t0, t1 = mod.extent(ovt)
            if t0 < s - 1e-6 or t1 > s + need + 1e-6:
                continue
            try:
                F, _ = self.features(mod, cfg, ovt)
            except (ValueError, IndexError):
                continue
            f, P = periodogram(F)
            Ps.append(P)
        if not Ps:
            return dict(_na("no_window"), n_tiles=0)
        return dict(overlap_from_periodogram(f, np.mean(Ps, 0), E), n_tiles=len(Ps))

    def x_band(self, mod, cfg, ov, E):
        bands, src = module_bands(mod)
        rs = [self.x(mod, cfg, ov, E, b) for b in bands]
        bad = [r["reason"] for r in rs if r["reason"]]
        if bad:
            return dict(_na(bad[0]), band_source=src, n_bands=len(bands))
        return {"X": float(np.mean([r["X"] for r in rs])), "reason": None, "band_source": src, "n_bands": len(bands)}

    def feature_spectrum(self, mod, cfg, ov):
        """A7: the channel-mean feature periodogram (each channel scaled to mean 1), mean spacing of the epochs (s)."""
        F, _ = self.features(mod, cfg, ov)
        P = periodogram(F)[1]
        P = P / np.where(P.mean(1, keepdims=True) > 0, P.mean(1, keepdims=True), 1.0)
        g = grid_windows(mod, cfg, ov, self.rec.fs)
        return P.mean(0), float(np.median(np.diff(g["win"][:, 0]))), int(F.shape[1])

    def lower_delta(self, t0, t1):
        """pDoC: the extent's mean C3/C4 log delta/alpha (30-s windows, 91_doc_null._windows_ratio) below the
        recording median (91's lower_delta_half rule on the overlay's extent)."""
        if not hasattr(self, "_lr"):
            m91 = importlib.import_module("91_doc_null")
            x4 = np.asarray(self.rec.x, float)[[self.rec.names.index(c) for c in m91.CH]].T
            self._lr = np.log(m91._windows_ratio(x4, float(self.rec.fs)) + 1e-12)
        w0, w1 = int(t0 // 30), int(t1 // 30) + 1
        seg = self._lr[w0:w1]
        return bool(seg.mean() < np.median(self._lr)) if len(seg) else None


def recording_results(rec, need, spectra, split):
    """Result rows of one recording. need: its rows with column 'part' in {m1, s1, rand, pdoc}. m1 rows also get the
    whole-recording, module-band and PC1 X and, for the default variant, the A7 feature spectrum; pdoc rows the
    lower-delta flag; 113's Columbia rows the artifact share the runner records. Every row carries the rebuilt
    overlay's extent (for the A9 flags)."""
    rx = RecordingX(rec, spectra)
    out = []
    for pipe, d in need.groupby("pipe"):
        mod = rn.load_pipe(pipe)
        specs = rn.variant_specs(mod, rec.source)
        default = rn.primary_variant(mod, rec.source)
        for _, r in d.iterrows():
            kind = "randomized" if r["part"] == "rand" else "published"
            variant = r["rand_variant"] if kind == "randomized" else r["variant"]
            spec = specs[variant]
            cfg = module_cfg(mod, spec)
            ov = overlay_for_row(mod, specs, r, kind)
            n = len(grid_labels(mod, cfg, ov))
            E = rx.spectrum(pipe, variant, kind, n)
            o = rx.x(mod, cfg, ov, E)
            t0, t1 = (float(v) for v in mod.extent(ov))
            res = {"key": r["key"] if kind == "published" else f"{r['uid']}|{pipe}|{int(r['j'])}|~{variant}|rand",
                   "uid": r["uid"], "patient": r["patient"], "pipe": pipe, "variant": variant, "j": int(r["j"]),
                   "kind": kind, "part": r["part"], "split": split, "source": rec.source, "X": o["X"], "chi": o["chi"],
                   "n_bins": o["n_bins"], "reason": o["reason"], "n_epochs": o.get("n_epochs"), "n_ch": o.get("n_ch"),
                   "n_ch_fit": o.get("n_ch_fit"), "n_ch_flat": o.get("n_ch_flat"), "X_flat0": o.get("X_flat0"),
                   "extent_t0_116": t0, "extent_t1_116": t1}
            if r["part"] == "m1":
                w = rx.x_whole(mod, variant, cfg, ov, E, float(r["start_s"]))
                b = rx.x_band(mod, cfg, ov, E)
                pc = rx.x(mod, cfg, ov, E, pc1=True)
                mk = rx.x_mask(mod, spec, kind, cfg, ov, o)
                res.update(X_whole=w["X"], reason_whole=w["reason"], n_tiles=w["n_tiles"], X_band=b["X"],
                           reason_band=b["reason"], band_source=b["band_source"], X_pc1=pc["X"],
                           reason_pc1=pc["reason"], X_mask=mk["X"], reason_mask=mk["reason"],
                           n_rejected=mk["n_rejected"], mask=mk["mask"])
                if variant == default and o.get("n_epochs"):
                    spc, dt, ne = rx.feature_spectrum(mod, cfg, ov)
                    res.update(spec=[float(v) for v in spc], spec_dt=dt, spec_n=ne)
                if _is113(r):
                    res["artifact_share_116"] = en.artifact_index(rec, t0, t1 - t0)[0]
            if r["part"] == "pdoc":
                res["lower_delta"] = rx.lower_delta(t0, t1)
            out.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res.items()})
    return out


def pdoc_rows(df):
    """pDoC published null rows (reduced montage), the first overlay of every variant (A1(c), A2)."""
    d = mg._null_seq(df)
    return d[d["null"].isin(PDOC_NULLS) & (d["j"] == 0)]


def needed_rows(df):
    parts = [m1_rows(df).assign(part="m1")]
    s1 = s1_rows(df)
    parts.append(s1[s1["j"] != 0].assign(part="s1"))
    parts.append(rand_rows(df).assign(part="rand"))
    parts.append(pdoc_rows(df).assign(part="pdoc"))
    return pd.concat(parts, ignore_index=True, sort=False)


def _result_file(path, uid):
    return os.path.join(path, f"{uid}.json")


def _done_uids(path):
    """Recordings with a result file (each written whole, through a temporary file and a rename). Files written by
    another version of this script are refused (never mixed)."""
    if not os.path.isdir(path):
        return set()
    uids = {f[:-5] for f in os.listdir(path) if f.endswith(".json") and not f.startswith("._")}
    h = own_sha256()
    stale = [u for u in sorted(uids) if json.load(open(_result_file(path, u)))["sha256_116"] != h]
    if stale:
        raise RuntimeError(f"{len(stale)} result file(s) in {path} come from another version of 116_mechanism.py "
                           f"(e.g. {stale[:3]}): move them away before running this version")
    return uids


def _write_result(path, uid, rows):
    fn = _result_file(path, uid)
    with open(fn + ".tmp", "w") as fh:
        json.dump({"uid": uid, "sha256_116": own_sha256(), "rows": rows}, fh)
    os.replace(fn + ".tmp", fn)


def run_x(split, limit=None, n_draws=N_DRAWS):
    p = paths(split)
    df = load_rows(split)
    need = needed_rows(df)
    items = {it["uid"]: it for source in (SOURCE, "pdoc") for it in en._items(source, split)}
    uids = sorted(need["uid"].unique())[:limit]
    missing = sorted(set(uids) - set(items))
    if missing:
        raise RuntimeError(f"{len(missing)} row recording(s) not in the {split} listing: {missing[:5]}")
    sp = Spectra(p["spectra"], n_draws)
    done = _done_uids(p["results"])
    os.makedirs(p["results"], exist_ok=True)
    by_uid = dict(tuple(need.groupby("uid")))
    for i, uid in enumerate(uids):
        if uid in done:
            continue
        rec = en._load(items[uid])
        rn.check_member(rec, split)
        rows = recording_results(rec, by_uid[uid], sp, split)
        del rec
        _write_result(p["results"], uid, rows)
        print(f"{i + 1}/{len(uids)} {uid}: {len(rows)} rows, {sum(r['reason'] is not None for r in rows)} NA",
              flush=True)
    sp.save()


def run_table(split, n_draws=N_DRAWS):
    p = paths(split)
    sp = Spectra(p["spectra"], n_draws)
    tab = design_table(n_draws=n_draws, spectra=sp)
    sp.save()
    os.makedirs(os.path.dirname(p["table"]), exist_ok=True)
    tab.to_csv(p["table"], index=False)
    print(f"design table: {len(tab)} rows -> {p['table']}; types {tab.design_type.value_counts().to_dict()}")
    return tab



def n_frozen_variants(pipe):
    return len(rn.variant_specs(rn.load_pipe(pipe), SOURCE))


def row_weights(pipes, n_variants):
    """1 / (frozen variants of the row's pipeline): every pipeline counts equally whatever its multiverse size."""
    return np.array([1.0 / n_variants[p] for p in np.asarray(pipes)], float)


def zscore(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    m = np.sum(w * v) / w.sum()
    sd = np.sqrt(np.sum(w * (v - m) ** 2) / w.sum())
    return (v - m) / sd, m, sd


def _loglik(y, eta, w):
    return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))


def fe_logit(y, Z, fe, w, start=None, tol=1e-9, max_iter=100, lam=0.0):
    """Weighted logistic fit of y ~ Z + fixed effects (fe codes) by Newton steps, the fixed effects absorbed (Schur
    complement; the penalised fixed-effect block inverted by Sherman-Morrison).
    lam = 0: the MLE; groups without both outcomes are dropped (their intercept diverges; they carry no information on
    Z), and a singular or non-converging fit is returned with converged False.
    lam > 0: the penalised fit, maximising loglik - lam / 2 (|beta|^2 + sum_g (alpha_g - mean alpha)^2): sklearn's
    LogisticRegression(C = 1 / lam) on [Z, one-hot fe] with an unpenalised intercept. Every group is kept.
    -> {beta, alpha (by kept code), converged, n_fe, n_fe_dropped, n_rows, codes}."""
    y, w, fe = np.asarray(y, float), np.asarray(w, float), np.asarray(fe)
    Z = np.asarray(Z, float).reshape(len(y), -1)
    codes, fe = np.unique(fe, return_inverse=True)
    G = len(codes)
    sw, sy = np.bincount(fe, w, G), np.bincount(fe, w * y, G)
    ok = (sy > 1e-12 * sw) & (sy < sw * (1 - 1e-12)) if lam == 0 else sw > 0
    keep = ok[fe]
    remap = np.cumsum(ok) - 1
    y, Z, w, fe = y[keep], Z[keep], w[keep], remap[fe[keep]]
    G, k = int(ok.sum()), Z.shape[1]
    out = {"beta": np.full(k, np.nan), "alpha": None, "converged": False, "n_fe": G, "n_fe_dropped": int((~ok).sum()),
           "n_rows": int(keep.sum()), "codes": codes[ok]}
    if G == 0 or len(y) == 0:
        return out
    if start is not None and start.get("alpha") is not None and len(start["alpha"]) == G:
        a, beta = np.array(start["alpha"], float), np.array(start["beta"], float)
    else:
        q = np.clip(np.bincount(fe, w * y, G) / np.bincount(fe, w, G), 0.01, 0.99)
        a, beta = np.log(q / (1 - q)), np.zeros(k)

    def obj(a, beta):
        return _loglik(y, a[fe] + Z @ beta, w) - lam / 2 * (beta @ beta + np.sum((a - a.mean()) ** 2))
    ll = obj(a, beta)
    for it in range(max_iter):
        p = expit(a[fe] + Z @ beta)
        v, r = w * p * (1 - p), w * (y - p)
        ga, gb = np.bincount(fe, r, G) - lam * (a - a.mean()), Z.T @ r - lam * beta
        Dl = np.bincount(fe, v, G) + lam
        C = np.column_stack([np.bincount(fe, v * Z[:, j], G) for j in range(k)])
        H = (Z * v[:, None]).T @ Z + lam * np.eye(k)
        u = 1.0 / Dl
        c = (lam / G) / (1 - lam / G * u.sum()) if lam > 0 else 0.0

        def Ainv(M):
            M = M.reshape(G, -1)
            return M * u[:, None] + c * np.outer(u, u @ M)
        AC = Ainv(C)
        try:
            db = np.linalg.solve(H - C.T @ AC, gb - AC.T @ ga)
        except np.linalg.LinAlgError:
            return out
        da = Ainv(ga - C @ db)[:, 0]
        if not (np.all(np.isfinite(da)) and np.all(np.isfinite(db))):
            return out
        t = 1.0
        while True:
            ll1 = obj(a + t * da, beta + t * db)
            if ll1 >= ll - 1e-10 * abs(ll) or t < 1e-8:
                break
            t /= 2
        a, beta, ll = a + t * da, beta + t * db, ll1
        if max(np.abs(t * da).max(), np.abs(t * db).max()) < tol:
            out.update(beta=beta, alpha=a, converged=True, iters=it + 1)
            return out
    out.update(beta=beta, alpha=a, iters=max_iter)
    return out


def cluster_index(patients):
    p = np.asarray(patients).astype(str)
    order = np.argsort(p, kind="stable")
    ids, starts, lens = np.unique(p[order], return_index=True, return_counts=True)
    return {"order": order, "starts": starts, "lens": lens, "ids": ids}


def boot_sample(ci, draw):
    """(row indices, cluster labels) of one patient-cluster draw: each drawn copy is its own cluster (relabelled)."""
    lens, starts = ci["lens"][draw], ci["starts"][draw]
    off = np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
    return ci["order"][off], np.repeat(np.arange(len(draw)), lens)


def _seed(name):
    return [SEED, zlib.crc32(str(name).encode())]


def _ptests(est, draws):
    d = draws[np.isfinite(draws)]
    B = len(d)
    if B == 0:
        return {"p_greater": None, "p_less": None, "p_two_sided": None, "ci95": None}
    pg, pl = (np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)
    return {"p_greater": float(pg), "p_less": float(pl), "p_two_sided": float(min(1.0, 2 * min(pg, pl))),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


LAM = 1.0 / PENALTY_C


def fe_model(d, covs, B=B_BOOT, name="model", weights=None, fe_col="fe"):
    """Weighted FE logistic of d.call on the named covariates (arrays aligned with d) with the patient-cluster
    bootstrap. Every fit, point and bootstrap, is penalised (fe_logit, lam = 1 / PENALTY_C) on the covariates divided
    by their weighted SD over d (coefficients reported per original unit), so no fit fails; the plain MLE is fitted on
    the same draws as a diagnostic (its failures counted). -> per covariate: beta, OR, OR 95% CI, one- and two-sided
    bootstrap p; counts."""
    y = d["call"].astype(float).to_numpy()
    Z = np.column_stack([np.asarray(c, float) for c in covs.values()])
    fe = d[fe_col].to_numpy()
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    sd = np.sqrt(np.average((Z - np.average(Z, 0, w)) ** 2, 0, w))
    sd = np.where(sd > 0, sd, 1.0)
    full = fe_logit(y, Z / sd, fe, w, lam=LAM)
    plain = fe_logit(y, Z, fe, w)
    out = {"n_rows": int(len(d)), "n_rows_fit": full["n_rows"], "n_patients": int(d["patient"].nunique()),
           "n_recordings": int(d["uid"].nunique()), "n_variants": int(full["n_fe"] + full["n_fe_dropped"]),
           "converged": full["converged"], "B": B,
           "penalty": {"C": PENALTY_C, "on": "covariates / weighted SD; pipeline-variant effects about their mean"},
           "plain_mle": {"converged": plain["converged"], "n_variants_dropped_one_outcome": plain["n_fe_dropped"],
                         "beta": {c: float(b) for c, b in zip(covs, plain["beta"])}}}
    if not full["converged"]:
        out["note"] = "full-sample fit did not converge"
        return out
    ci = cluster_index(d["patient"].to_numpy())
    draws = mg.boot_idx(len(ci["ids"]), B, _seed(name))
    bd = np.full((B, Z.shape[1]), np.nan)
    plain_fail = 0
    for b in range(B):
        idx, _ = boot_sample(ci, draws[b])
        r = fe_logit(y[idx], Z[idx] / sd, fe[idx], w[idx], lam=LAM)
        if r["converged"]:
            bd[b] = r["beta"] / sd
        plain_fail += not fe_logit(y[idx], Z[idx], fe[idx], w[idx])["converged"]
    out["n_boot_failed"] = int(np.sum(~np.isfinite(bd[:, 0])))
    out["plain_mle"]["n_boot_failed"] = int(plain_fail)
    beta = full["beta"] / sd
    out["coef"] = {}
    for j, cname in enumerate(covs):
        t = _ptests(beta[j], bd[:, j])
        out["coef"][cname] = {"beta": float(beta[j]), "or": float(np.exp(beta[j])),
                              "or_ci95": None if t["ci95"] is None else [float(np.exp(v)) for v in t["ci95"]],
                              **{k: t[k] for k in ("p_greater", "p_less", "p_two_sided")}}
    return out


def _prep(d, n_variants, per_patient=True):
    """Fixed-effect codes and row weights: 1 / (frozen variants of the pipeline), divided by the patient's rows of
    that variant (several recordings or tiles share one patient x variant weight; per_patient=False: per row)."""
    d = d.copy()
    d["fe"] = d["pipe"].astype(str) + "|" + d["variant"].astype(str)
    d["w"] = row_weights(d["pipe"].to_numpy(), n_variants)
    if per_patient and len(d):
        d["w"] = d["w"] / d.groupby(["patient", "pipe", "variant"])["pipe"].transform("size").to_numpy(float)
    return d


def rows_per_patient_variant(d):
    """Rows per patient x pipeline x variant (M1: one expected; more are counted and share the weight)."""
    n = d.groupby(["patient", "pipe", "variant"]).size()
    return {"cells": int(len(n)), "cells_with_more_than_one_row": int((n > 1).sum()), "max_rows": int(n.max())
            if len(n) else 0}


def channel_use(d):
    """Fraction of the native channels whose aperiodic fit succeeded (n_ch_fit / n_ch), by design type, immunity type,
    stratum and design x stratum (rows with a channel count)."""
    e = d[pd.to_numeric(d.get("n_ch"), errors="coerce").fillna(0) > 0] if "n_ch" in d else d.iloc[:0]
    frac = e["n_ch_fit"].astype(float) / e["n_ch"].astype(float)
    by = lambda keys: {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): float(v)
                       for k, v in frac.groupby(keys).mean().items()}
    col = lambda c: e[c].fillna("unlabelled").astype(str) if c in e else pd.Series("na", index=e.index)
    return {"n_rows": int(len(e)), "overall": float(frac.mean()) if len(e) else None,
            "by_design": by(col("design_type")), "by_immunity": by(col("immunity_type")),
            "by_stratum": by(col("stratum")), "by_design_x_stratum": by([col("design_type"), col("stratum")])}


def na_counts(d):
    """NA X by reason, by Morgoth stratum and by design type (left out of the models)."""
    na = d[d["X"].isna()]
    reason = na["reason"].fillna("missing_result").astype(str).str.split(":").str[0]
    st = na.get("stratum", pd.Series("na", index=na.index)).fillna("na").astype(str)
    return {"n_rows": int(len(d)), "n_na": int(len(na)), "by_reason": reason.value_counts().to_dict(),
            "by_stratum": st.value_counts().to_dict(),
            "by_design": na["design_type"].fillna("unknown").value_counts().to_dict() if len(na) else {},
            "by_stratum_x_reason": {f"{a}|{b}": int(n) for (a, b), n in
                                    pd.crosstab(st, reason).stack().items() if n} if len(na) else {}}


def m1_model(d, xcol, B, name):
    e = d[d[xcol].notna()] if xcol in d else d.iloc[:0]
    if e.empty or e["patient"].nunique() < 2:
        return {"n_rows": int(len(e)), "n_patients": int(e["patient"].nunique()) if len(e) else 0,
                "note": "fewer than 2 patients with X: not estimated"}
    z, m, sd = zscore(e[xcol], e["w"])
    out = fe_model(e, {"z_" + xcol: z}, B, name, e["w"])
    out.update(x_weighted_mean=float(m), x_weighted_sd=float(sd))
    return out


def immunity(d, rd, B):
    """OR per SD of chi on the published designs immune by design (immunity_type; shown if the upper bound <= 1.25),
    the chi x immunity-type slope contrast, and the randomization-based rows as a construction check."""
    out = {"typing": f"immune if |X on the reference 1/f series| <= {IMMUNE_XREF}, exposed otherwise",
           "pipelines_immune": sorted(set(d.loc[d["immunity_type"] == "immune", "pipe"]))}
    r = d[(d["immunity_type"] == "immune") & d["chi"].notna()]
    if len(r):
        z, _, _ = zscore(r["chi"], r["w"])
        out["immune_designs"] = fe_model(r, {"z_chi": z}, B, "immunity", r["w"])
        c = out["immune_designs"].get("coef", {}).get("z_chi", {})
        out["shown"] = bool(c.get("or_ci95") and c["or_ci95"][1] <= IMMUNITY_MAX)
        out["margin"] = IMMUNITY_MAX
    a = d[d["immunity_type"].isin(["immune", "exposed"]) & d["chi"].notna()]
    if len(a) and a["immunity_type"].nunique() == 2:
        z, _, _ = zscore(a["chi"], a["w"])
        exp_ = (a["immunity_type"] == "exposed").to_numpy(float)
        out["slope_contrast"] = fe_model(a, {"z_chi": z, "z_chi_x_exposed": z * exp_}, B, "immunity_contrast",
                                         a["w"])
        out["slope_contrast"]["test"] = "z_chi_x_exposed > 0 (immune slope smaller), one-sided: p_greater"
    if len(rd):
        rr = rd[rd["chi"].notna()].copy()
        if len(rr):
            rr["fe"] = rr["pipe"].astype(str) + "|~" + rr["rand_variant"].astype(str)
            ng = rr.groupby("pipe")["rand_variant"].nunique()
            rr["w"] = 1.0 / rr["pipe"].map(ng).to_numpy(float)
            rr = rr.assign(call=rr["rand_call"].astype(float))
            z, _, _ = zscore(rr["chi"], rr["w"])
            out["construction_check"] = fe_model(rr, {"z_chi": z}, B, "construction", rr["w"])
            out["construction_check"]["note"] = ("randomization-based rows: conditional rate <= alpha on every "
                                                 "recording, so a zero slope is guaranteed and proves nothing")
    return out


def deflated_report(d, choices, B):
    """Descriptive, for the deflated designs (x_ref_class; predicted: a false-call rate at or below nominal and a
    negative slope on chi), from the deflated rows only: per pipeline, the first-overlay false-call rate of the
    validity-primary and the default variant with a patient-bootstrap 95% CI; pooled, the OR per SD of chi from the
    immunity check's model. No test."""
    e = d[d["x_ref_class"] == "deflated"]
    out = {"pipelines": sorted(set(e["pipe"])), "nominal": mg.NOMINAL, "rates": {},
           "prediction": "rate at or below nominal; OR per SD of chi below 1 (descriptive, no test)"}
    for (nul, pipe), g in e.groupby(["null", "pipe"]):
        ch = choices.get((nul, pipe), {})
        out["rates"][pipe] = {role: dict(variant=ch.get(role), **_rate_ci(g[g["variant"] == ch.get(role)], B,
                                                                         _seed(f"deflated|{nul}|{pipe}|{role}")))
                              for role in ("validity_primary", "default")}
        out["rates"][pipe]["null"] = nul
    r = e[e["chi"].notna()]
    if len(r):
        z, _, _ = zscore(r["chi"], r["w"])
        out["chi_slope"] = fe_model(r, {"z_chi": z}, B, "deflated", r["w"])
    return out


def s1_units(rows, prob=None):
    """One unit per patient x pipeline: O = mean call over the pipeline's variants and the patient's windows, E = mean
    predicted probability (prob: per-row array; default the model without X, i.e. each variant's mean call)."""
    d = rows.assign(call=rows["call"].astype(float))
    E = d.groupby(["pipe", "variant"])["call"].transform("mean") if prob is None else pd.Series(prob, index=d.index)
    return (d.assign(E=E).groupby(["patient", "pipe"], sort=True)
            .agg(O=("call", "mean"), E=("E", "mean"), n=("call", "size")).reset_index())


def s1_dispersion(units, n_perm=N_PERM, seed=SEED, perms=None):
    """D = sum over patients of (sum O - sum E)^2 / sum E against permutations of the patient labels within pipeline.
    -> D, p, null mean and excess, the permutations used (reusable for the mediation contrast)."""
    pats = np.array(sorted(units["patient"].unique()))
    pi = {p: i for i, p in enumerate(pats)}
    groups = [(np.array([pi[p] for p in g["patient"]]), g["O"].to_numpy(float), g["E"].to_numpy(float))
              for _, g in units.groupby("pipe", sort=True)]

    def D(Ot, Et):
        m = Et > 0
        return np.sum((Ot - Et) ** 2 * m / np.where(m, Et, 1.0), -1)
    O0, E0 = np.zeros(len(pats)), np.zeros(len(pats))
    for i, o, e in groups:
        O0[i] += o
        E0[i] += e
    d_obs = float(D(O0, E0))
    rng = np.random.default_rng(seed)
    if perms is None:
        perms = [np.argsort(rng.random((n_perm, len(i))), axis=1) for i, _, _ in groups]
    Op, Ep = np.zeros((len(perms[0]), len(pats))), np.zeros((len(perms[0]), len(pats)))
    for (i, o, e), pm in zip(groups, perms):
        np.add.at(Op, (slice(None), i), o[pm])
        np.add.at(Ep, (slice(None), i), e[pm])
    null = D(Op, Ep)
    return {"D": d_obs, "p": float((np.sum(null >= d_obs - 1e-12) + 1) / (len(null) + 1)),
            "null_mean": float(null.mean()),
            "excess": float(d_obs - null.mean()), "n_units": int(len(units)), "n_patients": int(len(pats)),
            "n_pipes": len(groups), "n_perm": int(len(null)), "perms": perms, "totals": O0}


def lorenz_gini(totals, n_points=21):
    x = np.sort(np.asarray(totals, float))
    n, s = len(x), x.sum()
    if n == 0 or s <= 0:
        return {"gini": None, "lorenz": None}
    gini = float(2 * np.sum(np.arange(1, n + 1) * x) / (n * s) - (n + 1) / n)
    q = np.linspace(0, 1, n_points)
    L = np.interp(q * n, np.arange(n + 1), np.r_[0, np.cumsum(x)] / s)
    return {"gini": gini, "lorenz": {"patient_share": q.round(4).tolist(), "call_share": L.round(6).tolist()}}


def _public(r):
    return {k: v for k, v in r.items() if k not in ("perms", "totals")}


def s1(rows, n_variants, n_perm):
    """S1: concentration (test on every row), Lorenz/Gini and the mediation by X (rows with X)."""
    t = s1_dispersion(s1_units(rows), n_perm, SEED)
    out = {"test": _public(t), **lorenz_gini(t["totals"])}
    e = _prep(rows[rows["X"].notna()], n_variants, per_patient=False)
    if len(e) and e["patient"].nunique() > 1:
        z, _, _ = zscore(e["X"], e["w"])
        fit = fe_logit(e["call"].astype(float), z[:, None], e["fe"].to_numpy(), e["w"].to_numpy(), lam=LAM)
        if fit["converged"]:
            a = pd.Series(fit["alpha"], index=fit["codes"])
            eta = e["fe"].map(a).to_numpy(float) + z * fit["beta"][0]
            prob = np.where(np.isfinite(eta), expit(eta), e.groupby("fe")["call"].transform("mean").to_numpy(float))
            u0, ux = s1_units(e), s1_units(e, prob)
            r0 = s1_dispersion(u0, n_perm, SEED)
            rx = s1_dispersion(ux, perms=r0["perms"])
            share = 1 - rx["excess"] / r0["excess"] if r0["excess"] > 0 else None
            out["mediation"] = {"without_x": _public(r0), "with_x": _public(rx), "share_removed": share,
                                "largely_explained": None if share is None else bool(share >= LARGELY),
                                "threshold": LARGELY}
    return out


def background_group(stratum):
    st = pd.Series(stratum).fillna("unlabelled").astype(str)
    return pd.Series(np.where(st.isin(ORGANISED), "organised", np.where(st.isin(DISORGANISED), "disorganised",
                                                                        "excluded")), index=st.index)


def _wrate(d):
    return float(np.sum(d["w"] * d["call"].astype(float)) / d["w"].sum()) if len(d) else None


def s2(d, B):
    grp = background_group(d["stratum"])
    tr = d["transitioning"].fillna(False).astype(bool)
    out = {"excluded": {s: {"rows": int(n), "recordings": int(d[d["stratum"].fillna("unlabelled") == s]["uid"]
                                                                .nunique())}
                        for s, n in d[grp == "excluded"]["stratum"].fillna("unlabelled").value_counts().items()},
           "per_stratum": {s: {"rows": int(len(g)), "recordings": int(g["uid"].nunique()), "rate_weighted": _wrate(g)}
                           for s, g in d.groupby(d["stratum"].fillna("unlabelled"))},
           "transitioning": {str(k): {"rows": int(len(g)), "rate_weighted": _wrate(g)} for k, g in d.groupby(tr)}}
    e = d[grp != "excluded"]
    if e.empty or background_group(e["stratum"]).nunique() < 2:
        out["note"] = "both background groups are needed"
        return out
    covs = lambda x: {"organised": (background_group(x["stratum"]) == "organised").to_numpy(float),
                      "transitioning": x["transitioning"].fillna(False).astype(float).to_numpy()}
    out["before_x"] = fe_model(e, covs(e), B, "s2_before", e["w"])
    ex = e[e["X"].notna()]
    out["before_x_rows_with_x"] = fe_model(ex, covs(ex), B, "s2_before_x_rows", ex["w"])
    out["after_x"] = fe_model(ex, {**covs(ex), "z_X": zscore(ex["X"], ex["w"])[0]}, B, "s2_after", ex["w"])
    out["p_test"] = out["before_x"].get("coef", {}).get("organised", {}).get("p_two_sided")
    return out


def s3(d, B):
    share = d["artifact_share"].astype(float)
    e = d[share.notna()]
    per10 = lambda x: (x["artifact_share"].astype(float) / ART_STEP).to_numpy()
    out = {"n_rows_no_share": int(share.isna().sum()), "per": "10 points of artifact share"}
    out["before_x"] = fe_model(e, {"artifact_per10": per10(e)}, B, "s3_before", e["w"])
    ex = e[e["X"].notna()]
    out["before_x_rows_with_x"] = fe_model(ex, {"artifact_per10": per10(ex)}, B, "s3_before_x_rows", ex["w"])
    out["after_x"] = fe_model(ex, {"artifact_per10": per10(ex), "z_X": zscore(ex["X"], ex["w"])[0]}, B, "s3_after",
                              ex["w"])
    out["p_test"] = out["before_x"].get("coef", {}).get("artifact_per10", {}).get("p_two_sided")
    return out


def ri_rows(d):
    """Random-intercept rows: the a-priori default variant (primary_variant) of each pipeline, one row per patient x
    pipeline (the first recording by uid), X present: every pipeline counts once per patient."""
    e = d[(d["variant"] == d["primary_variant"]) & d["X"].notna()]
    keys = [c for c in ("patient", "pipe", "uid") if c in e]
    return e.sort_values(keys, kind="stable").drop_duplicates(["patient", "pipe"])


def random_intercept(d):
    """Sensitivity: M1 with a patient random intercept on ri_rows (statsmodels BinomialBayesMixedGLM, VB, which takes
    no row weights: one row per patient x pipeline gives every pipeline equal weight). Deterministic starts (fit_vb
    otherwise draws random starting SDs)."""
    from scipy import sparse
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    e = ri_rows(d)
    y = e["call"].astype(float).to_numpy()
    fe_codes, fe = np.unique(e["pipe"].astype(str).to_numpy(), return_inverse=True)
    sw, sy = np.bincount(fe, minlength=len(fe_codes)), np.bincount(fe, y, len(fe_codes))
    ok = (sy > 0) & (sy < sw)
    e, y, fe = e[ok[fe]], y[ok[fe]], (np.cumsum(ok) - 1)[fe[ok[fe]]]
    if len(e) == 0 or e["patient"].nunique() < 2:
        return {"n_rows": int(len(e)), "note": "fewer than 2 patients with both outcomes in a pipeline: not estimated"}
    z, _, _ = zscore(e["X"], np.ones(len(e)))
    G = int(ok.sum())
    exog = np.zeros((len(e), G + 1))
    exog[:, 0] = z
    exog[np.arange(len(e)), 1 + fe] = 1.0
    pid, pat = np.unique(e["patient"].astype(str).to_numpy(), return_inverse=True)
    vc = sparse.csr_matrix((np.ones(len(e)), (np.arange(len(e)), pat)), shape=(len(e), len(pid)))
    start = fe_logit(y, z[:, None], fe, np.ones(len(e)), lam=LAM)
    k = G + 1
    mean = np.r_[start["beta"][0], start["alpha"], 0.0, np.zeros(len(pid))] if start["converged"] else None
    sd = np.r_[np.full(k, 0.1), 0.5, np.full(len(pid), 0.5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = BinomialBayesMixedGLM(y, exog, vc, np.zeros(len(pid), int), vcp_p=1, fe_p=10).fit_vb(mean=mean, sd=sd)
    b, s = float(r.fe_mean[0]), float(r.fe_sd[0])
    return {"or": float(np.exp(b)), "or_ci95": [float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))],
            "beta": b, "beta_sd": s, "patient_sd": float(np.exp(r.vcp_mean[0])), "n_rows": int(len(e)),
            "n_patients": int(len(pid)), "n_pipelines": G, "n_pipelines_dropped_one_outcome": int((~ok).sum()),
            "rows": "default variant per pipeline, one row per patient x pipeline (equal pipeline weight)",
            "weights": "none (BinomialBayesMixedGLM takes no row weights)"}


def holm_family(pvals):
    names = [k for k, v in pvals.items() if v is not None]
    adj = mg.holm([pvals[k] for k in names])
    out = {k: {"p": pvals[k], "p_holm": None, "reject": None} for k in pvals}
    for k, a in zip(names, adj):
        out[k].update(p_holm=float(a), reject=bool(a <= 0.05))
    return out



def _open_binary(path):
    return open(path, "rb")


def _corner(s, key):
    m = re.search(key + r"\s*:?\s*([0-9]*\.?[0-9]+)\s*(mHz|Hz)?", s, re.I)
    if not m:
        return None
    return float(m.group(1)) / (1000.0 if m.group(2) == "mHz" else 1.0)


def edf_prefilters(path):
    """{label: {hp, lp}} (Hz; None where not stated, e.g. "HP:DC") from the EDF per-signal prefilter fields. Reads
    the signal count (bytes 252-255) and the signal header block (from byte 256: labels, prefilter fields) only, never
    bytes 8-183 (patient and recording identification, start date and time)."""
    with _open_binary(path) as fh:
        fh.seek(252)
        ns = int(fh.read(4).decode("ascii").strip())
        fh.seek(256)
        labels = fh.read(16 * ns).decode("ascii", "replace")
        fh.seek(256 + 136 * ns)
        pre = fh.read(80 * ns).decode("ascii", "replace")
    return {labels[16 * i:16 * (i + 1)].strip(): {"hp": _corner(pre[80 * i:80 * (i + 1)], "HP"),
                                                   "lp": _corner(pre[80 * i:80 * (i + 1)], "LP")} for i in range(ns)}


def marker_range(filters):
    """((lo, hi) Hz, reason or None): MARKER_HZ, narrowed to clear every stated corner's transition band (above
    HP_CLEAR x a high-pass corner, below LP_CLEAR x a low-pass corner); (None, reason) if nothing is left."""
    lo, hi = MARKER_HZ
    for f in filters:
        if f.get("hp"):
            lo = max(lo, HP_CLEAR * f["hp"])
        if f.get("lp"):
            hi = min(hi, LP_CLEAR * f["lp"])
    if (lo, hi) == MARKER_HZ:
        return (lo, hi), None
    if hi <= lo:
        return None, f"no range clears the filter corners ({filters})"
    return (lo, hi), f"narrowed to {lo:g}-{hi:g} Hz by the EDF prefilter corners {filters}"


def fixed_exponent(f, P):
    """Aperiodic exponent in FOOOF's 'fixed' mode (log10 P = b - chi log10 f, no knee) with its robust step: a first
    fit, then a refit on the bins at or below it, so peaks do not pull it. None with fewer than 3 usable bins."""
    f, P = np.asarray(f, float), np.asarray(P, float)
    if len(f) < 3 or not np.all(np.isfinite(P)) or np.any(P <= 0):
        return None
    lf, y = np.log10(f), np.log10(P)
    b = np.polyfit(lf, y, 1)
    below = y <= np.polyval(b, lf)
    if below.sum() >= 3:
        b = np.polyfit(lf[below], y[below], 1)
    return float(-b[0])


def resting_markers(x, fs, band=MARKER_HZ):
    """(aperiodic exponent, relative alpha) of channels x (channels, samples) over band (Hz), averaged over channels
    (NaN if any channel's fit fails or there is no band)."""
    if band is None:
        return np.nan, np.nan
    f, P = welch(np.asarray(x, float), fs=fs, nperseg=int(round(WELCH_S * fs)), axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    ex = [fixed_exponent(f[m], p[m]) for p in P]
    power = lambda lo, hi: P[:, (f >= lo) & (f <= hi)].sum(-1)
    ra = power(*ALPHA_HZ) / power(*band)
    return (np.nan if any(v is None for v in ex) else float(np.mean(ex))), float(np.mean(ra))


def s4_spectrum():
    """E of the b12 reference schedule (doc_null.schedule block 12, 48 trials): exact (fixed up to the first label)."""
    ds = expected_spectrum(lambda g: contrast_weights(dn.schedule("block", g, block=12)[0], [((1,), (0,))]),
                           n_draws=N_DRAWS)
    assert ds.fixed
    return ds.at(dn.N_TRIALS)


def s4_recording(rec, E, band=MARKER_HZ):
    """One row per 480-s window: X (C3, C4; b12), chi, resting markers (over band, Hz), the lower-delta flag and the
    window's mean C3/C4 log delta/alpha (91_doc_null rules)."""
    m91 = importlib.import_module("91_doc_null")
    fs = float(rec.fs)
    x = np.asarray(rec.x, float)
    ci = [rec.names.index(c) for c in ("C3", "C4")]
    x4 = x[[rec.names.index(c) for c in m91.CH]].T
    run_s = dn.N_TRIALS * m91.TRIAL_S
    wn, step = int(round(run_s * fs)), int(m91.TRIAL_S * fs)
    n_win = x.shape[1] // wn
    if n_win == 0:
        return []
    pad = bn.padding_seconds(rec.x, fs)
    starts = np.arange(n_win) * wn
    F = dn.trial_features(x[ci].T, fs, (starts[:, None] + np.arange(dn.N_TRIALS)[None] * step).ravel())
    lr = np.log(m91._windows_ratio(x4, fs) + 1e-12)
    med = np.median(lr)
    rows = []
    for k, s in enumerate(starts):
        t0 = s / fs
        base = {"uid": rec.uid, "patient": rec.patient, "diagnosis": rec.meta.get("diagnosis", ""), "window": k,
                "start_s": float(t0)}
        if pad[int(t0): int(t0 + run_s)].any():
            rows.append({**base, "skipped": "padding"})
            continue
        o = overlap(F[k * dn.N_TRIALS:(k + 1) * dn.N_TRIALS].T, E)
        w0, w1 = int(s / fs // 30), int((s / fs + run_s) // 30) + 1
        ap, ra = resting_markers(x[ci, s:s + wn], fs, band)
        Pw = periodogram(F[k * dn.N_TRIALS:(k + 1) * dn.N_TRIALS].T)[1]
        spec = (Pw / np.where(Pw.mean(1, keepdims=True) > 0, Pw.mean(1, keepdims=True), 1.0)).mean(0)
        rows.append({**base, "skipped": None, "X": o["X"], "chi": o["chi"], "reason": o["reason"], "aperiodic": ap,
                     "spec": [float(v) for v in spec], "marker_lo_hz": None if band is None else band[0],
                     "marker_hi_hz": None if band is None else band[1],
                     "rel_alpha": ra, "log_delta_alpha": float(lr[w0:w1].mean()),
                     "lower_delta": bool(lr[w0:w1].mean() < med)})
    return rows


def run_s4(split, limit=None):
    p = paths(split)
    E = s4_spectrum()
    rows = []
    items = en._items("pdoc", split)[:limit]
    for i, it in enumerate(items):
        pf = edf_prefilters(it["path"])
        band, why = marker_range([pf.get(c, {}) for c in ("C3", "C4")])
        rec = en._load(it)
        rn.check_member(rec, split)
        r = [dict(x, marker_range_note=why) for x in s4_recording(rec, E, band)]
        del rec
        rows += r
        print(f"S4 {i + 1}/{len(items)} {it['uid']}: {len(r)} windows, "
              f"{sum(1 for x in r if x.get('skipped'))} skipped, {sum(1 for x in r if x.get('reason'))} NA"
              + (f"; marker range {why}" if why else ""), flush=True)
    os.makedirs(os.path.dirname(p["s4"]), exist_ok=True)
    pd.DataFrame(rows).to_parquet(p["s4"], index=False)


def hodges_lehmann(a, b):
    return float(np.median(np.subtract.outer(np.asarray(a, float), np.asarray(b, float))))


def _ridge_logit(X, y, lam=RIDGE, iters=50):
    """Logistic regression on standardised columns, ridge on the slopes (not the intercept)."""
    A = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(A.shape[1])
    pen = np.r_[0.0, np.full(X.shape[1], lam)]
    for _ in range(iters):
        p = expit(A @ beta)
        g = A.T @ (y - p) - pen * beta
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(pen) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def _auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


def loo_auc(X, y, groups):
    """AUROC of leave-one-group-out ridge-logistic predictions (a patient's bootstrap copies are left out together)."""
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            return np.nan
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        b = _ridge_logit((X[tr] - mu) / sd, y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ b
    return _auc(pred, y)


def _s4_patients(w, subset):
    d = w[w["skipped"].isna()] if "skipped" in w else w
    if subset == "lower_delta":
        d = d[d["lower_delta"].fillna(False).astype(bool)]
    d = d[d["X"].notna()]
    g = d.groupby("patient")
    P = pd.DataFrame({"X": g["X"].median(), "aperiodic": g["aperiodic"].median(), "rel_alpha": g["rel_alpha"].median(),
                      "diagnosis": g["diagnosis"].first(), "n_windows": g.size()})
    st = (w[w["skipped"].isna()] if "skipped" in w else w).groupby("patient")["log_delta_alpha"].median()
    P["state"] = st.reindex(P.index)
    P["mcs"] = P["diagnosis"].astype(str).str.startswith("MCS")
    return P[P["diagnosis"].astype(str).str.match(r"^(MCS|VS)")].dropna(subset=["X", "aperiodic", "rel_alpha"])


def _ols_mcs(P):
    A = np.column_stack([np.ones(len(P)), P["mcs"].to_numpy(float), zscore(P["state"], np.ones(len(P)))[0]])
    return float(np.linalg.lstsq(A, P["X"].to_numpy(float), rcond=None)[0][1])


def s4_estimates(P, B, adjust, name):
    """MCS minus VS of patient median X (HL shift, or the OLS coefficient adjusted for state), Wilcoxon p, and the
    AUROC gain of X over the resting markers (+ state when adjusted), with a diagnosis-stratified relabelled
    bootstrap (B draws)."""
    mcs, vs = P[P["mcs"]], P[~P["mcs"]]
    out = {"n_mcs": int(len(mcs)), "n_vs": int(len(vs)), "n_windows_median": float(P["n_windows"].median())
           if len(P) else None}
    if len(mcs) < 2 or len(vs) < 2:
        out["note"] = "fewer than 2 patients in a group"
        return out
    base_cols = ["aperiodic", "rel_alpha"] + (["state"] if adjust else [])
    y = P["mcs"].to_numpy()

    def stats(Q, groups):
        est = _ols_mcs(Q) if adjust else hodges_lehmann(Q[Q["mcs"]]["X"], Q[~Q["mcs"]]["X"])
        yy = Q["mcs"].to_numpy()
        a0 = loo_auc(Q[base_cols].to_numpy(float), yy, groups)
        a1 = loo_auc(Q[base_cols + ["X"]].to_numpy(float), yy, groups)
        return est, a0, a1
    est, a0, a1 = stats(P, np.arange(len(P)))
    rng = np.random.default_rng(_seed(name))
    im, iv = np.flatnonzero(y), np.flatnonzero(~y)
    bd = np.full((B, 3), np.nan)
    for b in range(B):
        idx = np.r_[rng.choice(im, len(im)), rng.choice(iv, len(iv))]
        Q = P.iloc[idx].reset_index(drop=True)
        bd[b] = stats(Q, idx)
    t_est, t_gain = _ptests(est, bd[:, 0]), _ptests(a1 - a0, bd[:, 2] - bd[:, 1])
    out.update({"estimate": est, "estimate_kind": "OLS coefficient of MCS, adjusted for z(state)" if adjust else
                "Hodges-Lehmann shift, MCS minus VS", "estimate_ci95": t_est["ci95"],
                "estimate_p_two_sided_boot": t_est["p_two_sided"],
                "auroc_markers": a0, "auroc_markers_plus_x": a1, "auroc_gain": a1 - a0,
                "auroc_gain_ci95": t_gain["ci95"],
                "auroc_x_alone": _auc(P["X"].to_numpy(float), y), "B": B})
    if not adjust:
        out["wilcoxon_p_two_sided"] = float(mannwhitneyu(mcs["X"], vs["X"], alternative="two-sided").pvalue)
    return out


def s4(w, B):
    out = {"n_recordings": int(w["uid"].nunique()), "n_windows": int(len(w)),
           "n_skipped_padding": int(w["skipped"].eq("padding").sum()) if "skipped" in w else 0,
           "n_na": int(w["reason"].notna().sum()) if "reason" in w else 0,
           "na_by_reason": w["reason"].dropna().value_counts().to_dict() if "reason" in w else {},
           "marker_range": {"default_hz": list(MARKER_HZ), "mode": "fixed (no knee)",
                            "narrowed": w.drop_duplicates("uid")["marker_range_note"].dropna().value_counts().to_dict()
                            if "marker_range_note" in w else {}}}
    for name, subset, adjust in (("a_overall", "all", False), ("b_lower_delta", "lower_delta", False),
                                 ("c_state_adjusted", "all", True)):
        out[name] = s4_estimates(_s4_patients(w, subset), B, adjust, "s4_" + name)
    out["p_test"] = out["a_overall"].get("wilcoxon_p_two_sided")
    return out



def extra_stems(listing=None):
    """Stems of the BDSP extra listing (engine.bdsp_extras()["analysis"]): patient-disjoint from the design sample
    that 108 used (engine.check_disjoint, run by bdsp_extras)."""
    listing = en.bdsp_extras() if listing is None else listing
    return {str(stem) for _, stem in listing["analysis"]}


def disjoint_rows(d, stems):
    """Post-freeze sensitivity: the rows whose recording is in the extra listing only."""
    return d[d["uid"].astype(str).isin(stems)]


def without_columbia(d):
    """A8: M1 without the Columbia rows."""
    return d[d["pipe"] != "columbia"]


def read_k_table(path=None):
    """A1(b): {pipe: {k, k_sensitivity}} from docs/benchmark-k-table.json (generated from the catalogues before any
    rate is read). Refused unless it holds exactly the frozen pipelines, each with an integer k >= 1."""
    with open(path or K_TABLE) as fh:
        d = json.load(fh)
    if set(d) != set(PIPES):
        raise ValueError(f"k table pipelines {sorted(d)} are not the frozen ones {sorted(PIPES)}")
    out = {}
    for pipe, e in d.items():
        ks = {}
        for key in ("k", "k_sensitivity"):
            v = e.get(key)
            if v is None and key == "k_sensitivity":
                ks[key] = None
                continue
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"k table {pipe}.{key} = {v!r}: an integer >= 1 is required")
            ks[key] = v
        out[pipe] = ks
    return out


def a9_flags(rows, intervals, recordings):
    """A9: per row, whether its extent overlaps a flagged interval (121_activation.windows_flagged), and whether 121
    processed its recording at all (windows_flagged raises KeyError for an unprocessed recording: flagged, absent is
    not clean)."""
    act = importlib.import_module("121_activation")
    iv = intervals.assign(uid=intervals["uid"].astype(str))
    recs = recordings.assign(uid=recordings["uid"].astype(str))
    flag, done = np.ones(len(rows), bool), np.zeros(len(rows), bool)
    uid = rows["uid"].astype(str).to_numpy()
    t0, t1 = rows["extent_t0"].astype(float).to_numpy(), rows["extent_t1"].astype(float).to_numpy()
    for i in range(len(rows)):
        try:
            flag[i] = bool(act.windows_flagged(iv, recs, uid[i], [t0[i]], t1[i] - t0[i])[0])
            done[i] = True
        except KeyError:
            pass
    return pd.DataFrame({"a9_flag": flag, "a9_processed": done}, index=rows.index)


def variant_choices(df, B):
    """{(null, pipe): {validity_primary (115.published's selection), default (114.primary_variant, the a-priori
    default), default_as_coded}} for every null set and pipeline with first-overlay rows."""
    out = {}
    for nul, d in mg._null_seq(df).groupby("null"):
        pub = mg.published(d, B)
        for pipe, e in pub.items():
            mod = rn.load_pipe(pipe)
            dflt = rn.primary_variant(mod, d["source"].iloc[0] if "source" in d else SOURCE)
            out[(nul, pipe)] = {"validity_primary": e.get("validity_primary"), "default": dflt,
                                "default_as_coded": bool(mod.VARIANTS.get(dflt, {}).get("as_coded", False))}
    return out


def _rate_ci(d, B, seed):
    d = d.assign(c=d["call"].astype(float))
    if d.empty:
        return {"n": 0, "n_patients": 0, "rate": None, "ci95": None}
    s, n = mg._per_patient(d, "c")
    draws = mg.ratio_draws(s, n, mg.boot_idx(len(s), B, seed))
    return {"n": int(n.sum()), "n_patients": int(len(s)), "rate": float(s.sum() / n.sum()), "ci95": mg._ci(draws)}


def a1_rates(d, choices, B, masks=None):
    """A1(a): false-call rate in model-assigned awake backgrounds (normal_awake, non-transitioning, A9-clean) and,
    for wording (A1(c)), in ICU-typical backgrounds (the disorganised strata), for the validity-primary AND the
    default variant, patient-bootstrap 95% CI. masks: {name: function(rows) -> bool mask} replaces the two
    (pDoC: lower-delta windows)."""
    tr = d["transitioning"].fillna(False).astype(bool) if "transitioning" in d else pd.Series(False, index=d.index)
    clean = ~d["a9_flag"].fillna(True).astype(bool) & ~tr
    masks = masks or {"awake": lambda x: x["stratum"].astype(str) == "normal_awake",
                      "icu_typical": lambda x: x["stratum"].astype(str).isin(DISORGANISED)}
    out = {}
    for (nul, pipe), g in d.groupby(["null", "pipe"]):
        ch = choices.get((nul, pipe), {})
        e = {"null": nul, "n_a9_excluded": int(g["a9_flag"].fillna(True).astype(bool).sum()),
             "n_transitioning_excluded": int(tr[g.index].sum()), "default_as_coded": ch.get("default_as_coded")}
        for name, mk in masks.items():
            keep = clean[g.index] & mk(g)
            e[name] = {role: dict(variant=ch.get(role), **_rate_ci(g[keep & (g["variant"] == ch.get(role))], B,
                                                                    _seed(f"a1|{nul}|{pipe}|{name}|{role}")))
                       for role in ("validity_primary", "default")}
        out[pipe] = e
    return out


def per100_composite(rows, mod, k, B, seed, rule=None):
    """A1(b): expected false calls per 100 patients who cannot follow commands: 100 x the share of patients whose
    composite (the module's patient_rule; mindBEAGLE: rule) over their first k non-overlapping windows is positive.
    Only patients with >= k windows enter (115.patient_outcomes); those with fewer are counted and their share
    reported."""
    fn = mg._rule(mod, rule=rule) if rule else mg._rule(mod)
    c = mg.composite_curve(rows, fn, (k,), B, seed)[0]
    n_any = len(mg.patient_outcomes(rows, fn, 1))
    sc = lambda v: None if v is None else 100.0 * v
    return {"k": int(k), "rule": rule or "patient_rule", "n_patients": c["n"], "per100": sc(c["rate"]),
            "ci95": None if c["ci95"] is None else [sc(v) for v in c["ci95"]], "n_patients_any_window": n_any,
            "n_excluded_fewer_than_k": n_any - c["n"],
            "share_excluded_fewer_than_k": (n_any - c["n"]) / n_any if n_any else None}


def a1b(df, ktab, choices, B):
    """A1(b) per pipeline at its published k (k table): BDSP, the module's own composite; Columbia, the k = 2 bracket
    (115.columbia_k2_bracket); mindBEAGLE type (mbvt, mbmi), the pDoC best-of-k curve. Validity-primary and default
    variant; k_sensitivity where the table gives one."""
    seq = mg._null_seq(df)
    out = {}
    for pipe, kt in ktab.items():
        mod = rn.load_pipe(pipe)
        if pipe == "columbia":
            br = mg.columbia_k2_bracket(df, B).get("ncmd_bdsp", {})
            out[pipe] = {"k": kt["k"], "rule": "BH within patient, k = 2 bracket", "bracket": {
                b: {"per100": None if e.get("rate") is None else 100 * e["rate"], "n_patients": e.get("n_patients"),
                    "ci95": None if e.get("ci95") is None else [100 * v for v in e["ci95"]], "label": e.get("label")}
                for b, e in br.items()}}
            continue
        mb = pipe in rn.MB_PIPES
        nul = ("ncmd_pdoc" if mod.FAMILY == "cmd" else "nstim0_pdoc") if mb else \
            ("ncmd_bdsp" if mod.FAMILY == "cmd" else "nstim0_bdsp")
        ch = choices.get((nul, pipe), {})
        e = {"k": kt["k"], "null": nul, "covert_processing": pipe in mg.COVERT_PROCESSING}
        for role in ("validity_primary", "default"):
            r = seq[(seq["null"] == nul) & (seq["pipe"] == pipe) & (seq["variant"] == ch.get(role))]
            e[role] = dict(variant=ch.get(role), **per100_composite(r, mod, kt["k"], B, _seed(f"a1b|{pipe}|{role}"),
                                                                     "best_run" if mb else None))
            if kt.get("k_sensitivity"):
                e[role]["k_sensitivity"] = per100_composite(r, mod, kt["k_sensitivity"], B,
                                                            _seed(f"a1b|{pipe}|{role}|ks"), "best_run" if mb else None)
        out[pipe] = e
    return out


def _logit_fit(y, A, w, offset=None, iters=100):
    """Weighted logistic MLE of y on the columns of A (no fixed effects), with an optional offset."""
    beta = np.zeros(A.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        p = expit(off + A @ beta)
        H = (A * (w * p * (1 - p))[:, None]).T @ A
        step = np.linalg.solve(H + 1e-12 * np.eye(len(beta)), A.T @ (w * (y - p)))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def calibration(y, lp, w):
    """Calibration of predicted log-odds lp: intercept = calibration-in-the-large (y ~ 1 + offset(lp)), slope = the
    coefficient of lp in y ~ 1 + lp."""
    y, lp, w = (np.asarray(v, float) for v in (y, lp, w))
    one = np.ones((len(y), 1))
    a = _logit_fit(y, one, w, offset=lp)[0]
    b0, b1 = _logit_fit(y, np.column_stack([one, lp]), w)
    return {"intercept": float(a), "slope": float(b1), "slope_model_intercept": float(b0)}


def _m1_predictor(b):
    """The BDSP M1 fit as a predictor: (fit, X mean, X SD); b has fe, w, X, call."""
    z, m, sd = zscore(b["X"], b["w"])
    return fe_logit(b["call"].astype(float), z[:, None], b["fe"].to_numpy(), b["w"].to_numpy(), lam=LAM), m, sd


def _transport_lp(fit, m, sd, p):
    a = pd.Series(fit["alpha"], index=fit["codes"])
    return p["fe"].map(a).to_numpy(float) + fit["beta"][0] * (p["X"].astype(float).to_numpy() - m) / sd


def transport(bdsp, pdoc, nv_bdsp, nv_pdoc, B, name="a2"):
    """A2: the BDSP-fitted M1 model (X slope + pipeline-variant effects) predicts the pDoC published rows of the
    variants present in both arms (the others excluded and counted). Calibration intercept and slope with a joint
    patient bootstrap (BDSP patients refit the model, pDoC patients the evaluation rows)."""
    b = _prep(bdsp[bdsp["X"].notna()], nv_bdsp)
    p = _prep(pdoc, nv_pdoc)
    fit, m, sd = _m1_predictor(b)
    rate = b.groupby("fe")["call"].agg(lambda c: c.astype(float).mean())
    est = set(rate[(rate > 0) & (rate < 1)].index)
    why = np.where(~p["fe"].isin(set(b["fe"])), "variant_not_in_bdsp",
                   np.where(~p["fe"].isin(est), "variant_without_both_outcomes_in_bdsp",
                            np.where(p["X"].isna(), "x_na", "")))
    out = {"n_rows_excluded": {k: int(np.sum(why == k)) for k in ("variant_not_in_bdsp",
                                                                   "variant_without_both_outcomes_in_bdsp", "x_na")},
           "limits": "pDoC: 6-channel reduced montage (F3 F4 C3 C4 O1 O2; module subsets per bench.wrappers.PDOC_CH), "
                     "overnight PSG; BDSP: 19-channel 10-20 clinical EEG"}
    p = p[why == ""]
    out.update(n_rows=int(len(p)), n_patients=int(p["patient"].nunique()), B=B,
               variants_shared=int(p["fe"].nunique()))
    if p.empty or not fit["converged"]:
        out["note"] = "no shared estimable variant" if p.empty else "BDSP fit did not converge"
        return out
    c = calibration(p["call"].astype(float), _transport_lp(fit, m, sd, p), p["w"])
    cib, cip = cluster_index(b["patient"].to_numpy()), cluster_index(p["patient"].to_numpy())
    db = mg.boot_idx(len(cib["ids"]), B, _seed(name + "|bdsp"))
    dp = mg.boot_idx(len(cip["ids"]), B, _seed(name + "|pdoc"))
    bd = np.full((B, 2), np.nan)
    for i in range(B):
        bb = b.iloc[boot_sample(cib, db[i])[0]]
        f2 = fe_logit(bb["call"].astype(float), ((bb["X"].astype(float) - m) / sd).to_numpy()[:, None],
                      bb["fe"].to_numpy(), bb["w"].to_numpy(), lam=LAM)
        pp = p.iloc[boot_sample(cip, dp[i])[0]]
        if not f2["converged"]:
            continue
        lp = _transport_lp(f2, m, sd, pp)
        ok = np.isfinite(lp)
        if ok.sum() > 1 and len(np.unique(pp["call"].to_numpy()[ok])) == 2:
            ci = calibration(pp["call"].astype(float).to_numpy()[ok], lp[ok], pp["w"].to_numpy()[ok])
            bd[i] = ci["intercept"], ci["slope"]
    out.update(calibration_intercept=c["intercept"], calibration_slope=c["slope"],
               slope_model_intercept=c["slope_model_intercept"], n_boot_failed=int(np.isnan(bd[:, 0]).sum()),
               calibration_intercept_ci95=_ptests(0, bd[:, 0])["ci95"],
               calibration_slope_ci95=_ptests(0, bd[:, 1])["ci95"])
    return out


A3_STATUS = ("matched", "ambiguous", "unmatched")
A3_RECOVERED, A3_NOT_RECOVERED = ("Consc",), ("PDOC", "D")
A3_WORDING = ("An exploratory estimate in {n} patients ({k} recovered). The study was not powered for prognosis, and "
              "no inference about prognosis is drawn.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_committed(path):
    """True if the file is tracked by git with no uncommitted change."""
    import subprocess
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True).returncode
    return run("ls-files", "--error-unmatch", rel) == 0 and run("diff", "--quiet", "HEAD", "--", rel) == 0


def _check_sha(path, want):
    h = sha256_file(path)
    if h != want:
        raise RuntimeError(f"{path}: sha256 {h} is not the pinned {want}: refused, not read")


def a3_table(map_path=None, outcome_path=None, map_sha=A3_MAP_SHA256, outcome_sha=A3_OUTCOMES_SHA256,
             committed=is_committed):
    """A3 inputs: the case mapping (columns case, uid, status in A3_STATUS; uid = our recording, set only for a
    deterministic signal match), read only if git-tracked and unchanged and matching its fixed sha256; then the
    outcome table, only if it matches its fixed sha256, of which only the case, A3_AROUSAL and A3_OUTCOME columns
    are read. -> (one row per matched case: case, uid, patient, arousal, outcome; counts by status)."""
    map_path, outcome_path = map_path or A3_MAP, outcome_path or A3_OUTCOMES
    if not committed(map_path):
        raise RuntimeError(f"{map_path}: the case mapping must be committed (tracked, unchanged): refused, not read")
    _check_sha(map_path, map_sha)
    mp = pd.read_csv(map_path, dtype={"case": str, "uid": str, "status": str})
    bad = sorted(set(mp["status"]) - set(A3_STATUS))
    if bad or mp["case"].duplicated().any():
        raise ValueError(f"case mapping: unknown status {bad} or a case listed twice")
    counts = {s: int((mp["status"] == s).sum()) for s in A3_STATUS}
    ok = mp[(mp["status"] == "matched") & mp["uid"].notna()]
    if ok["uid"].duplicated().any():
        raise ValueError("case mapping: a recording matched to two cases")
    _check_sha(outcome_path, outcome_sha)
    out = pd.read_csv(outcome_path, usecols=["case", A3_AROUSAL, A3_OUTCOME], dtype={"case": str})
    unknown = sorted(set(out[A3_OUTCOME].astype(str)) - set(A3_RECOVERED + A3_NOT_RECOVERED))
    if unknown:
        raise ValueError(f"outcome values {unknown} are not {A3_RECOVERED + A3_NOT_RECOVERED}")
    t = ok.merge(out.rename(columns={A3_AROUSAL: "arousal", A3_OUTCOME: "outcome"}), on="case", how="left",
                 validate="one_to_one")
    t["patient"] = t["uid"].str.extract(r"^(Patient[ _]\d+)_", expand=False).str.replace(" ", "_")
    return t, counts


def partial_spearman(x, y, z):
    """Spearman correlation of x and y adjusted for z: Pearson correlation of the rank residuals after regressing
    each rank on [1, z]."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    A = np.column_stack([np.ones(len(rx)), np.asarray(z, float)])
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    ex, ey = res(rx), res(ry)
    den = np.sqrt(np.sum(ex ** 2) * np.sum(ey ** 2))
    return float(np.sum(ex * ey) / den) if den > 0 else np.nan


def adjusted_auroc(score, y, strata):
    """Covariate-adjusted AUROC for a discrete covariate (Janes and Pepe 2009): case-control pairs compared only
    within their stratum, pooled over strata (ties 1/2)."""
    score, y, strata = np.asarray(score, float), np.asarray(y, bool), np.asarray(strata)
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a, b = score[m & y], score[m & ~y]
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            num += np.sum(d > 0) + 0.5 * np.sum(d == 0)
            den += d.size
    return float(num / den) if den else np.nan


def a3_estimates(P, B):
    """A3: on the S4 reference X (patient median): Spearman with the CRS-R Arousal subscale (first assessment) and
    AUROC for recovery (Consc vs PDOC or D), the latter also without deaths; both also adjusted for VS/MCS;
    patient-bootstrap 95% CIs (B draws; draws without both outcomes are left out and counted)."""
    from scipy.stats import spearmanr
    P = P.dropna(subset=["X"]).reset_index(drop=True)
    rec = P["outcome"].isin(A3_RECOVERED).to_numpy()
    nod = (P["outcome"] != "D").to_numpy()
    x, ar, mcs = P["X"].to_numpy(float), P["arousal"].astype(float).to_numpy(), P["mcs"].astype(float).to_numpy()
    has_ar = np.isfinite(ar)
    stats = {
        "spearman": lambda i: spearmanr(x[i][has_ar[i]], ar[i][has_ar[i]])[0],
        "spearman_vs_mcs_adjusted": lambda i: partial_spearman(x[i][has_ar[i]], ar[i][has_ar[i]], mcs[i][has_ar[i]]),
        "auroc_recovery": lambda i: _auc(x[i], rec[i]),
        "auroc_recovery_vs_mcs_adjusted": lambda i: adjusted_auroc(x[i], rec[i], mcs[i]),
        "auroc_recovery_no_deaths": lambda i: _auc(x[i][nod[i]], rec[i][nod[i]]),
        "auroc_recovery_no_deaths_vs_mcs_adjusted": lambda i: adjusted_auroc(x[i][nod[i]], rec[i][nod[i]],
                                                                             mcs[i][nod[i]])}
    ns = {"spearman": int(has_ar.sum()), "spearman_vs_mcs_adjusted": int(has_ar.sum()), "auroc_recovery": len(P),
          "auroc_recovery_vs_mcs_adjusted": len(P), "auroc_recovery_no_deaths": int(nod.sum()),
          "auroc_recovery_no_deaths_vs_mcs_adjusted": int(nod.sum())}
    full = np.arange(len(P))
    draws = mg.boot_idx(len(P), B, _seed("a3"))
    out = {"n": int(len(P)), "n_recovered": int(rec.sum()), "n_deaths": int((~nod).sum()), "B": B,
           "wording": A3_WORDING.format(n=len(P), k=int(rec.sum()))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in stats.items():
            bd = np.array([fn(d) for d in draws], float)
            out[name] = {"estimate": float(fn(full)), "n": ns[name], "ci95": _ptests(0, bd)["ci95"],
                         "n_boot_undefined": int(np.sum(~np.isfinite(bd)))}
    return out


def a3(w, release=False, map_path=None, outcome_path=None, B=B_BOOT):
    """A3 on the S4 windows (patient median X, overall) and the mapped outcome table, only with the release flag (the
    outcome file is not opened, not even hashed, without it); a note while the inputs are not there."""
    if not release:
        return {"note": "not run: A3 runs only with --a3-release (the outcome file is not opened without it)"}
    map_path, outcome_path = map_path or A3_MAP, outcome_path or A3_OUTCOMES
    if not (os.path.exists(map_path) and os.path.exists(outcome_path)):
        return {"note": f"inputs not available yet ({map_path}, {outcome_path})"}
    t, counts = a3_table(map_path, outcome_path)
    P = _s4_patients(w, "all").reset_index()
    P = t.merge(P[["patient", "X", "mcs"]], on="patient", how="left")
    return {"cases": counts, "n_matched_without_x": int(P["X"].isna().sum()), **a3_estimates(P, B)}


def a7_spectra(rows, n_peaks=5):
    """A7 (descriptive): per pipeline x stratum, the mean log feature spectrum over recordings (each spectrum scaled
    to mean 1 and read on the pipeline's modal grid), its aperiodic fit and the residual peaks above it (local maxima
    of the log residual at least 3 robust SDs of the residual and 1 dB above the fit), largest first."""
    from scipy.signal import find_peaks
    out = {}
    for pipe, g in rows.groupby("pipe"):
        n_mode = int(pd.Series([r.get("spec_n") or 2 * len(r["spec"]) for _, r in g.iterrows()]).mode().min())
        f0 = np.arange(1, n_mode // 2 + 1) / n_mode
        dt = float(np.median(g["spec_dt"].astype(float)))
        st = np.where(g.get("transitioning", pd.Series(False, index=g.index)).fillna(False).astype(bool),
                      "transitioning", g["stratum"].fillna("unlabelled").astype(str))
        res = {}
        for s in sorted(set(st)):
            L = []
            for _, r in g[st == s].iterrows():
                P = np.asarray(r["spec"], float)
                n = r.get("spec_n") or 2 * len(P)
                L.append(np.log(np.interp(f0, np.arange(1, len(P) + 1) / n, P) + 1e-300))
            Lm = np.mean(L, 0)
            fit, why = fit_aperiodic(f0, np.exp(Lm))
            e = {"n": len(L), "f_cycles_per_epoch": f0.round(6).tolist(), "mean_log_power": Lm.round(6).tolist(),
                 "spacing_s": dt, "aperiodic": None if fit is None else dict(zip(("b", "k", "chi"), fit)),
                 "fit_reason": why, "peaks": []}
            if fit is not None:
                r_ = Lm - np.log(aperiodic(f0, fit))
                thr = max(PEAK_SD * 1.4826 * np.median(np.abs(r_ - np.median(r_))), PEAK_MIN_DB / (10 * np.log10(np.e)))
                idx, _ = find_peaks(r_, height=thr)
                for i in idx[np.argsort(-r_[idx])][:n_peaks]:
                    e["peaks"].append({"f_cycles_per_epoch": float(f0[i]), "f_hz_approx": float(f0[i] / dt),
                                       "residual_db": float(10 * np.log10(np.e) * r_[i])})
            res[s] = e
        out[pipe] = res
    return out



def read_results(path):
    return pd.DataFrame([r for u in sorted(_done_uids(path)) for r in json.load(open(_result_file(path, u)))["rows"]])


def _with_extent(d):
    """Rows' extents (113's rows: the rebuilt overlay's, recorded by the X stage) for the A9 flags."""
    d = d.copy()
    for c in ("extent_t0", "extent_t1"):
        own = d[c].astype(float) if c in d else pd.Series(np.nan, index=d.index)
        d[c] = own.fillna(d[c + "_116"].astype(float)) if c + "_116" in d else own
    return d


def _a9(d, source, split):
    act = importlib.import_module("121_activation")
    a, b = act.paths(source, split)
    if not (os.path.exists(a) and os.path.exists(b)):
        raise RuntimeError(f"A9 tables missing ({a}): run 121_activation.py --source {source} --split {split}")
    iv, recs = act.load(source, split)
    return pd.concat([d, a9_flags(d, iv, recs)], axis=1)


def run_stats(split, B=B_BOOT, n_perm=N_PERM, n_draws=N_DRAWS, with_mixed=True, a3_release=False):
    p = paths(split)
    df = load_rows(split)
    res = read_results(p["results"])
    if res.empty:
        raise RuntimeError(f"no X rows in {p['results']}: run --stage x first")
    tab = pd.read_csv(p["table"]) if os.path.exists(p["table"]) else run_table(split, n_draws)
    tp = tab[tab["kind"] == "published"].drop_duplicates(["pipe", "variant"]).set_index(["pipe", "variant"])
    dtype, itype, xclass = tp["design_type"], tp["x_ref_1f"].map(immunity_type), tp["x_ref_1f"].map(x_ref_class)
    pipes = sorted(set(m1_rows(df)["pipe"]))
    n_variants = {pp: n_frozen_variants(pp) for pp in pipes}
    if res.duplicated(["key", "pipe", "kind"]).any():
        raise RuntimeError("duplicate X result keys: refused (each null row has one result)")
    pub = res[res["kind"] == "published"].drop(columns=["uid", "patient", "variant", "j", "kind", "split", "source"],
                                               errors="ignore")
    m1 = _prep(m1_rows(df).merge(pub, on=["key", "pipe"], how="left", validate="one_to_one"), n_variants)
    if "artifact_share_116" in m1:
        m1["artifact_share"] = m1["artifact_share"].astype(float).fillna(m1["artifact_share_116"].astype(float))
    m1["design_type"] = [dtype.get((a, b)) for a, b in zip(m1["pipe"], m1["variant"])]
    m1["immunity_type"] = [itype.get((a, b)) for a, b in zip(m1["pipe"], m1["variant"])]
    m1["x_ref_class"] = [xclass.get((a, b)) for a, b in zip(m1["pipe"], m1["variant"])]
    m1["primary_variant"] = m1["pipe"].map({pp: rn.primary_variant(rn.load_pipe(pp), SOURCE) for pp in pipes})
    m1 = _a9(_with_extent(m1), SOURCE, split)
    s1r = s1_rows(df).merge(pub[["key", "pipe", "X"]], on=["key", "pipe"], how="left", validate="one_to_one")
    rd = rand_rows(df)
    rres = res[res["kind"] == "randomized"]
    rd = rd.assign(rkey=rd["uid"] + "|" + rd["pipe"] + "|" + rd["j"].astype(int).astype(str) + "|~"
                   + rd["rand_variant"].astype(str) + "|rand").merge(
        rres[["key", "chi", "X"]].rename(columns={"key": "rkey"}), on="rkey", how="left", validate="one_to_one")
    pr = pdoc_rows(df).merge(pub[[c for c in ("key", "pipe", "X", "chi", "reason", "lower_delta", "extent_t0_116",
                                               "extent_t1_116") if c in pub]], on=["key", "pipe"], how="left",
                             validate="one_to_one")
    if len(pr):
        pr = _a9(_with_extent(pr), "pdoc", split)
    out = {"split": split, "sha256_116": own_sha256(), "B": B, "n_perm": n_perm, "n_draws": n_draws,
           "frozen_variants": n_variants, "design_types": tp["design_type"].value_counts().to_dict(),
           "immunity_types": itype.value_counts().to_dict(), "x_ref_classes": xclass.value_counts().to_dict(),
           "rows": {"m1": int(len(m1)), "s1": int(len(s1r)), "rand": int(len(rd)),
                    "pdoc_first_overlays": int(len(pr))},
           "m1_rows_per_patient_variant": rows_per_patient_variant(m1), "channel_use": channel_use(m1),
           "na": na_counts(m1), "a9": {"bdsp_m1_rows_flagged": int(m1["a9_flag"].sum()),
                                       "bdsp_m1_rows_unprocessed": int((~m1["a9_processed"]).sum())}}
    out["m1"] = m1_model(m1, "X", B, "m1")
    out["immunity"] = immunity(m1, rd, B)
    sens = {"whole_recording_x": m1_model(m1, "X_whole", B, "m1_whole"),
            "module_band_x": m1_model(m1, "X_band", B, "m1_band"),
            "pc1_x": m1_model(m1, "X_pc1", B, "m1_pc1"),
            "disjoint_extra_listing": m1_model(disjoint_rows(m1, extra_stems()), "X", B, "m1"),
            "without_columbia": m1_model(without_columbia(m1), "X", B, "m1_no_columbia"),
            "a9_clean_windows": m1_model(m1[~m1["a9_flag"]], "X", B, "m1_a9_clean"),
            "flat_channel_zero": m1_model(m1, "X_flat0", B, "m1_flat0"),
            "artefact_mask_x": m1_model(m1, "X_mask", B, "m1_mask")}
    sens["flat_channel_zero"]["rows_with_a_flat_failed_channel"] = int((m1["n_ch_flat"].fillna(0) > 0).sum())
    sens["artefact_mask_x"].update(
        rows_with_rejections=int((m1["n_rejected"].fillna(0) > 0).sum()),
        mask_by_pipe=m1.groupby("pipe")["mask"].first().to_dict(),
        na=m1["reason_mask"].dropna().astype(str).str.split(":").str[0].value_counts().to_dict(),
        note="the module's own artefact rule; rejected epochs zero-filled in the feature series and label weights")
    sens["whole_recording_x"]["na"] = m1["reason_whole"].dropna().value_counts().to_dict()
    sens["module_band_x"]["band_source"] = m1.groupby("pipe")["band_source"].first().to_dict()
    sens["pc1_x"]["na"] = m1["reason_pc1"].dropna().value_counts().to_dict()
    sens["disjoint_extra_listing"]["note"] = ("post-freeze sensitivity: the extra BDSP listing only (patient-disjoint "
                                              "from the 108 design sample); not a test")
    if with_mixed:
        sens["random_intercept"] = random_intercept(m1)
    out["sensitivity"] = sens
    out["S1"] = s1(s1r, n_variants, n_perm)
    out["S2"] = s2(m1, B)
    out["S3"] = s3(m1, B)
    w4 = pd.read_parquet(p["s4"]) if os.path.exists(p["s4"]) else None
    if w4 is not None:
        out["S4"] = s4(w4, B)
    out["holm"] = holm_family({"S1": out["S1"]["test"]["p"], "S2": out["S2"].get("p_test"),
                               "S3": out["S3"].get("p_test"), "S4": out.get("S4", {}).get("p_test")})
    choices = variant_choices(df[df["null"].isin(BDSP_NULLS + PDOC_NULLS)], B)
    out["deflated_group"] = deflated_report(m1, choices, B)
    amend = {"variant_choices": {f"{n}|{pp}": v for (n, pp), v in choices.items()}}
    amend["A1a_awake"] = a1_rates(m1, choices, B)
    if len(pr):
        amend["A1c_pdoc_lower_delta"] = a1_rates(pr, choices, B, masks={
            "lower_delta": lambda x: x["lower_delta"].fillna(False).astype(bool)})
    amend["A1b_per100"] = a1b(df, read_k_table(), choices, B)
    nv_pdoc = {pp: len(rn.variant_specs(rn.load_pipe(pp), "pdoc")) for pp in sorted(set(pr["pipe"]))} if len(pr) else {}
    amend["A2_transport"] = transport(m1, pr, n_variants, nv_pdoc, B) if len(pr) else {"note": "no pDoC rows"}
    amend["A3"] = a3(w4, release=a3_release, B=B) if w4 is not None else {"note": "S4 windows missing"}
    sp = m1[m1["spec"].notna()] if "spec" in m1 else m1.iloc[:0]
    amend["A7_feature_spectra"] = {"bdsp_default_variant": a7_spectra(sp) if len(sp) else {}}
    if w4 is not None and "spec" in w4:
        k = w4[w4["spec"].notna()] if "skipped" not in w4 else w4[w4["skipped"].isna() & w4["spec"].notna()]
        dx = np.where(k["diagnosis"].astype(str).str.startswith("MCS"), "MCS", "VS")
        trial_s = float(importlib.import_module("91_doc_null").TRIAL_S)
        k = k.assign(pipe="pdoc_b12_c3c4", transitioning=False, spec_dt=trial_s,
                     stratum=[f"{a}|{'lower_delta' if b else 'other'}" for a, b in zip(dx, k["lower_delta"])])
        amend["A7_feature_spectra"]["pdoc_s4_windows"] = a7_spectra(k)
    out["part_A"] = amend
    os.makedirs(os.path.dirname(p["summary"]), exist_ok=True)
    with open(p["summary"], "w") as fh:
        json.dump(mg._jsonable(out), fh, indent=1, default=float)
    print(f"mechanism {split}: {len(m1)} M1 rows ({out['na']['n_na']} NA X), {len(s1r)} S1 rows, {len(rd)} "
          f"randomization rows, {len(pr)} pDoC first overlays -> {p['summary']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "analysis"))
    ap.add_argument("--stage", default="all", choices=("table", "x", "s4", "stats", "all"))
    ap.add_argument("--limit", type=int, default=None, help="dev only: first N recordings")
    ap.add_argument("--draws", type=int, default=N_DRAWS, help="dev only: label-spectrum draws")
    ap.add_argument("--boot", type=int, default=B_BOOT, help="dev only: bootstrap draws")
    ap.add_argument("--perm", type=int, default=N_PERM, help="dev only: S1 permutations")
    ap.add_argument("--no-mixed", action="store_true", help="dev only: skip the random-intercept sensitivity")
    ap.add_argument("--a3-release", action="store_true", help="run A3 (reads the hash-pinned outcome table)")
    a = ap.parse_args()
    guard_split(a.split)
    if a.split == "analysis" and (a.limit or a.draws != N_DRAWS or a.boot != B_BOOT or a.perm != N_PERM or a.no_mixed):
        raise SystemExit("the analysis split runs at the fixed in advance settings only (no --limit/--draws/--boot/"
                         "--perm/--no-mixed)")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.stage in ("table", "all"):
        run_table(a.split, a.draws)
    if a.stage in ("x", "all"):
        run_x(a.split, a.limit, a.draws)
    if a.stage in ("s4", "all"):
        run_s4(a.split, a.limit)
    if a.stage in ("stats", "all"):
        run_stats(a.split, a.boot, a.perm, a.draws, not a.no_mixed, a.a3_release)


if __name__ == "__main__":
    main()
