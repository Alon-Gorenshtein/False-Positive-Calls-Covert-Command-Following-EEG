"""Load EEGMMIDB (PhysioNet, via MNE eegbci) per subject into imagery-vs-rest
epochs for the covert-command-following contrast. Imagery = imagined movement
(annotations T1/T2 in the imagery runs); rest = T0. Band-pass 8-30 Hz."""
import warnings, numpy as np
import mne
mne.set_log_level("ERROR")
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, events_from_annotations, pick_types

IMAGERY_RUNS = [4, 8, 12, 6, 10, 14]
EXCLUDE = {88, 89, 92, 100}
FMIN, FMAX = 8.0, 30.0
TMIN, TMAX = 0.0, 2.0


def all_subjects():
    return [s for s in range(1, 110) if s not in EXCLUDE]


def load_subject(subject, runs=IMAGERY_RUNS, fmin=FMIN, fmax=FMAX,
                 tmin=TMIN, tmax=TMAX, broadband=False):
    """Return (X[n_epochs, n_ch, n_times], y[n_epochs], ch_names, sfreq).
    y: 1=imagery (T1/T2), 0=rest (T0). If broadband, skip the 8-30 band-pass
    (used by the EMG-robustness path which needs >30 Hz content)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn = eegbci.load_data(subject, runs, update_path=True)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fn])
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")
        if not broadband:
            raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
        else:
            raw.filter(1.0, min(fmax * 3, raw.info["sfreq"] / 2 * 0.98),
                       fir_design="firwin", verbose=False)
        events, eid = events_from_annotations(raw)
        code = {v: k for k, v in eid.items()}
        picks = pick_types(raw.info, eeg=True)
        ep = Epochs(raw, events, eid, tmin=tmin, tmax=tmax, picks=picks,
                    baseline=None, preload=True, verbose=False)
        ev = ep.events[:, 2]
        y = np.array([0 if code[c] == "T0" else 1 for c in ev], dtype=int)
        X = ep.get_data(copy=True).astype(np.float32)
        ch = [ep.ch_names[i] for i in range(len(ep.ch_names))]
        return X, y, ch, float(ep.info["sfreq"])


def epoch_runs(subject, runs=IMAGERY_RUNS, tmin=TMIN, tmax=TMAX):
    """Run number and label of each epoch load_subject returns, in the same order. Metadata only (no filtering;
    filtering does not change which epochs survive), so E6 can replay the cached epochs in recorded order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raws = [read_raw_edf(f, preload=False) for f in eegbci.load_data(subject, runs, update_path=True)]
        bounds = np.cumsum([r.n_times for r in raws])
        raw = concatenate_raws(raws)
        eegbci.standardize(raw)
        events, eid = events_from_annotations(raw)
        code = {v: k for k, v in eid.items()}
        ep = Epochs(raw, events, eid, tmin=tmin, tmax=tmax, picks=pick_types(raw.info, eeg=True),
                    baseline=None, preload=False, verbose=False)
        ep.drop_bad()
        run = np.asarray(runs)[np.searchsorted(bounds, ep.events[:, 0] - raw.first_samp, side="right")]
        y = np.array([0 if code[c] == "T0" else 1 for c in ep.events[:, 2]], dtype=int)
    return run, y


def chronological_order(run):
    """Indices that put epochs in recorded order: runs ascending (4, 6, ..., 14), each run's epochs kept in order."""
    return np.lexsort((np.arange(len(run)), np.asarray(run)))
