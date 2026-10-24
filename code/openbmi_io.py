"""Lee 2019 OpenBMI (CC0, GigaDB 100542). MATLAB v5, fields x (samples x 62, uV),
t (onset samples), fs, y_dec, chan. ERP: 1 target / 2 non-target. MI: 1 right / 2 left. SSVEP: 1-4 = 12/8.57/6.67/5.45 Hz."""
import os
import shutil
import numpy as np
import requests
from scipy.io import loadmat
import evoked
import montage

BASE = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542"
SSVEP_HZ = [12.0, 8.57, 6.67, 5.45]
MI_FS = 250


def url(session, subj, paradigm):
    return f"{BASE}/session{session}/s{subj}/sess{session:02d}_subj{subj:02d}_EEG_{paradigm}.mat"


def fetch(session, subj, paradigm, dest):
    if shutil.disk_usage(dest if os.path.exists(dest) else ".").free < 5e9:
        raise RuntimeError("less than 5 GB free")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, os.path.basename(url(session, subj, paradigm)))
    with requests.get(url(session, subj, paradigm), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path + ".part", "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
    os.replace(path + ".part", path)
    return path


def load(path, paradigm, phase):
    key = f"EEG_{paradigm}_{phase}"
    m = loadmat(path, variable_names=[key])[key][0, 0]
    chan = [str(np.squeeze(c).item()) for c in np.ravel(m["chan"])]
    return (m["x"].astype(np.float64), float(m["fs"].item()), m["t"].ravel().astype(int),
            m["y_dec"].ravel().astype(int), chan)


def _kit(x, chan):
    return x[:, montage.kit8_indices(chan)]


def erp_epochs(x, fs, t, y, chan):
    xx = evoked.bandpass(_kit(x, chan), fs, *evoked.ERP_BAND)
    E, ok = evoked.cut(xx, fs, t, *evoked.ERP_WIN, evoked.ERP_FS)
    return E, (y[ok] == 1).astype(int)


def mi_epochs(x, fs, t, chan):
    xx = evoked.bandpass(_kit(x, chan), fs, 8.0, 30.0)
    imag, ok1 = evoked.cut(xx, fs, t, 1.0, 3.5, MI_FS)
    rest, ok2 = evoked.cut(xx, fs, t, -2.5, 0.0, MI_FS)
    n = min(len(imag), len(rest))
    X = np.stack([v for pair in zip(rest[:n], imag[:n]) for v in pair])
    return X, np.tile([0, 1], n)


def ssvep_scores(x, fs, t, y, chan):
    xx = evoked.bandpass(_kit(x, chan), fs, *evoked.SSVEP_BAND)
    E, ok = evoked.cut(xx, fs, t, 0.5, 4.0, MI_FS)
    return evoked.cca_scores(E, MI_FS, SSVEP_HZ), y[ok] - 1
