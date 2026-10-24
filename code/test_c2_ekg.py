"""Tests for c2_ekg.py (EKG of the fresh 400 for part D1). Synthetic v7.3 files only; no BDSP data.
python3 -m pytest test_c2_ekg.py -q"""
import contextlib
import hashlib
import importlib
import io
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdsp_stream as bs
import c2_ekg as ce
import c2_worker as cw

ek = importlib.import_module("131_ekg_design")
NAMES20 = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
           "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
META = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "bdsp", "meta")


def _src(root, f, dur, seed, names=NAMES20):
    os.makedirs(f"{root}/EEGs", exist_ok=True)
    d = np.random.default_rng(seed).normal(0, 20, (len(names), dur * 128))
    d[-1] = np.sin(np.arange(d.shape[1]) / 7.0) * 800 + 0.123456789
    bs.write_mat(f"{root}/EEGs/{f}", d, 128, names)
    return d


def test_pinned_list_only():
    assert ce.PINNED_LIST == "216c6a2977264d2c3f5e354df6185daa83782a69a7e176de8fbb1ede92c45d39"
    real = os.path.join(META, "c2_fresh_400.csv")
    if os.path.exists(real):
        assert len(ce.pinned_list(real)) == 400
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "l.csv")
        with open(p, "w") as fh:
            fh.write(open(real).read().replace(",0,", ",1,", 1) if os.path.exists(real) else "rank\n0\n")
        with pytest.raises(RuntimeError, match="pinned"):
            ce.pinned_list(p)
    assert ce.WINDOW_S == 1800


def test_same_reader_as_the_check():
    with tempfile.TemporaryDirectory() as tmp:
        d = _src(tmp, "Bonobo00005_2.mat", 2300, 1)
        path = f"{tmp}/EEGs/Bonobo00005_2.mat"
        status, ch, x, fs = ce.ekg_window(path, 400)
        assert (status, ch, fs) == ("ok", "EKG", 128.0) and x.dtype == np.float32 and x.shape == (230400,)
        assert x.tobytes() == d[19, 400 * 128:2200 * 128].astype(np.float32).tobytes()
        out = os.path.join(tmp, "nogo")
        r = SimpleNamespace(uid="Bonobo00005_2", dir="EEGs", file="Bonobo00005_2.mat", start_s=400, fs=128.0,
                            channel="EKG")
        dest, _ = ek.pull_one(r, out=out, ssh=["bash", "-c"], root=tmp, py=sys.executable)
        with np.load(dest) as z:
            assert z["ekg"].tobytes() == x.tobytes()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ce.main(["design-check", path, "400"])
        assert hashlib.sha256(x.tobytes()).hexdigest() in buf.getvalue()
        assert sorted(os.listdir(os.path.join(tmp, "EEGs"))) == ["Bonobo00005_2.mat"]


def test_aligned_with_the_stored_c2_window():
    with tempfile.TemporaryDirectory() as tmp:
        _src(tmp, "Bonobo00006_1.mat", 2500, 2)
        path = f"{tmp}/EEGs/Bonobo00006_1.mat"
        z = cw.window_npz(path, 600, "Bonobo00006_1", "Bonobo00006")
        p = ce.window(path, ["C3", "EKG"], 600 * 128, 1800 * 128)
        assert p["x"][0].tobytes() == z["x19"][2].tobytes()


def test_ekg_status():
    with tempfile.TemporaryDirectory() as tmp:
        _src(tmp, "Bonobo00001_1.mat", 2000, 3, names=NAMES20[:-1] + ["X1"])
        _src(tmp, "Bonobo00002_1.mat", 2000, 4, names=NAMES20 + ["ECG2"])
        _src(tmp, "Bonobo00003_1.mat", 2000, 5, names=NAMES20[:-1] + ["ECG EKG+-EKG-"])
        assert ce.ekg_window(f"{tmp}/EEGs/Bonobo00001_1.mat", 0)[:3] == ("no_ekg", "", None)
        assert ce.ekg_window(f"{tmp}/EEGs/Bonobo00002_1.mat", 0)[:3] == ("several_ekg", "EKG|ECG2", None)
        assert ce.ekg_window(f"{tmp}/EEGs/Bonobo00003_1.mat", 100)[:2] == ("ok", "ECG EKG+-EKG-")
        with pytest.raises(ValueError):
            ce.ekg_window(f"{tmp}/EEGs/Bonobo00003_1.mat", 300)


def _c2(tmp, specs):
    """A fixed-list stand-in plus stored C2 windows; returns (base, bdsp, list path, its sha256)."""
    bdsp, base = os.path.join(tmp, "bdsp"), os.path.join(tmp, "c2")
    os.makedirs(f"{base}/sig")
    rows = []
    for rank, (uid, names, start) in enumerate(specs):
        _src(bdsp, f"{uid}.mat", 2200, 10 + rank, names=names)
        z = cw.window_npz(f"{bdsp}/EEGs/{uid}.mat", start, uid, uid[:-2]) if "EKG" in names else dict(
            window_start_s=start, uid=uid)
        np.savez(f"{base}/sig/{uid}.npz", window_start_s=z["window_start_s"], uid=z["uid"])
        rows.append(dict(rank=rank, patient_id=uid[:-2], uid=uid, dir="EEGs", file=f"{uid}.mat", seg_rank=0,
                         fs=128.0, n_samples=2200 * 128, window_start_s=start, n_eligible_starts=1))
    lst = os.path.join(tmp, "list.csv")
    pd.DataFrame(rows).to_csv(lst, index=False)
    return base, bdsp, lst, ce.sha256_file(lst)


def test_run_writes_files_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        noekg = NAMES20[:-1] + ["X1"]
        base, bdsp, lst, sha = _c2(tmp, [("Bonobo00011_1", NAMES20, 100), ("Bonobo00012_1", noekg, 50),
                                         ("Bonobo00013_1", NAMES20, 400)])
        with pytest.raises(RuntimeError):
            ce.run(base, bdsp, lst)
        ce.run(base, bdsp, lst, expect=sha)
        man = pd.read_csv(f"{base}/ekg/manifest.csv", keep_default_na=False)
        assert list(man.columns) == ce.MANIFEST_COLS and list(man.status) == ["ok", "no_ekg", "ok"]
        assert sorted(os.listdir(f"{base}/ekg")) == ["Bonobo00011_1.npz", "Bonobo00013_1.npz", "manifest.csv"]
        with np.load(f"{base}/ekg/Bonobo00013_1.npz") as z:
            assert set(z.files) == {"ekg", "fs", "start_s", "channel", "uid"} and int(z["start_s"]) == 400
            assert hashlib.sha256(z["ekg"].tobytes()).hexdigest() == man.ekg_sha256[2]
            assert z["ekg"].shape == (230400,) and man.n_samples[2] == 230400
        m = os.path.getmtime(f"{base}/ekg/Bonobo00011_1.npz")
        first = open(f"{base}/ekg/manifest.csv").read()
        ce.run(base, bdsp, lst, expect=sha)
        assert os.path.getmtime(f"{base}/ekg/Bonobo00011_1.npz") == m
        assert open(f"{base}/ekg/manifest.csv").read() == first
        assert sorted(os.listdir(f"{bdsp}/EEGs")) == ["Bonobo00011_1.mat", "Bonobo00012_1.mat", "Bonobo00013_1.mat"]


def test_run_refuses_a_window_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        base, bdsp, lst, sha = _c2(tmp, [("Bonobo00021_1", NAMES20, 100)])
        np.savez(f"{base}/sig/Bonobo00021_1.npz", window_start_s=101, uid="Bonobo00021_1")
        with pytest.raises(ValueError, match="does not match"):
            ce.run(base, bdsp, lst, expect=sha)
        assert not os.path.exists(f"{base}/ekg/Bonobo00021_1.npz")


def test_extension_list_only_once_pinned(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "ext.csv")
        pd.DataFrame(dict(rank=[1125], uid=["Bonobo00099_0"])).to_csv(p, index=False)
        assert ce.EXT_LIST is None and ce.pinned_lists() == {ce.PINNED_LIST}
        with pytest.raises(RuntimeError):
            ce.pinned_list(p)
        monkeypatch.setattr(ce, "EXT_LIST", ce.sha256_file(p))
        assert len(ce.pinned_list(p)) == 1
        assert ce.PINNED_LIST in ce.pinned_lists() and len(ce.pinned_lists()) == 2
        real = os.path.join(META, "c2_fresh_400.csv")
        if os.path.exists(real):
            assert len(ce.pinned_list(real)) == 400
        q = os.path.join(tmp, "other.csv")
        pd.DataFrame(dict(rank=[1126], uid=["Bonobo00098_0"])).to_csv(q, index=False)
        with pytest.raises(RuntimeError):
            ce.pinned_list(q)
