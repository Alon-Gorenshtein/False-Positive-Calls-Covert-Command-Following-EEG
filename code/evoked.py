"""Evoked-channel detectors.

ERP:
- band-pass 1-20 Hz on the continuous recording;
- epochs 0.25-0.8 s after each event, resampled to 100 Hz;
- xDAWN covariances (2 filters, OAS) + tangent space + standardised logistic regression with balanced classes.

SSVEP: canonical correlation with sine/cosine references at f and 2f (training-free)."""
from fractions import Fraction
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ERP_WIN = (0.25, 0.8)
ERP_BAND = (1.0, 20.0)
ERP_FS = 100
SSVEP_BAND = (4.0, 40.0)


def bandpass(x, fs, lo, hi):
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, np.float64), axis=0)


def cut(x, fs, onsets, tmin, tmax, out_fs=None):
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    onsets = np.asarray(onsets, int)
    keep = (onsets + a >= 0) & (onsets + b <= len(x))
    ep = np.stack([x[o + a: o + b].T for o in onsets[keep]]) if keep.any() else np.zeros((0, x.shape[1], b - a))
    if out_fs and int(round(fs)) != int(out_fs) and len(ep):
        f = Fraction(int(out_fs), int(round(fs))).limit_denominator(1000)
        ep = resample_poly(ep, f.numerator, f.denominator, axis=-1)
    return ep, keep


def erp_model():
    return make_pipeline(XdawnCovariances(nfilter=2, estimator="oas"), TangentSpace(metric="riemann"),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))


def _cancorr(A, B):
    qa, _ = np.linalg.qr(A - A.mean(0))
    qb, _ = np.linalg.qr(B - B.mean(0))
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False)[0])


def cca_scores(E, fs, freqs, n_harm=2):
    t = np.arange(E.shape[-1]) / fs
    refs = [np.column_stack([fn(2 * np.pi * h * f * t) for h in range(1, n_harm + 1) for fn in (np.sin, np.cos)])
            for f in freqs]
    return np.array([[_cancorr(e.T, R) for R in refs] for e in E])
