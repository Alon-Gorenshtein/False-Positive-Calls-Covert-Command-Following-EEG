"""Tests for 134_ekg_phase1.py (Phase-1 EKG for part D1). Synthetic v7.3 files, plus the local benchmark
metadata for the pool. python3 -m pytest test_ekg_phase1.py -q"""
import hashlib
import importlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdsp_stream as bs

ep = importlib.import_module("134_ekg_phase1")
bn = importlib.import_module("108_bdsp_null")
NAMES20 = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
           "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
FAKE = dict(ssh=["bash", "-c"], py=sys.executable)
HAVE_META = os.path.exists(os.path.join(ep.META, "touched_before_freeze.json"))


def test_constants():
    assert (ep.SEED, ep.STREAM, ep.POOL_N, ep.WINDOW_S) == (20261016, 4, 974, 1800)
    assert ep.ek.WINDOW_S == 1800


@pytest.mark.skipif(not HAVE_META, reason="benchmark metadata not present")
def test_pool_is_the_analysis_split_minus_touched():
    from bench import engine as en
    p = ep.pool()
    uids = [u for u, *_ in p]
    assert len(p) == 974 and uids == sorted(uids) and len(set(uids)) == 974
    touched = importlib.import_module("114_benchmark").touched_uids()
    assert not set(uids) & touched and len(touched) == 6
    allu = {it["uid"] for it in en._items("bdsp", "analysis")}
    assert set(uids) | touched == allu
    assert {s for _, _, s, _ in p} == {"design", "extra"}
    src = ep.sources()
    assert all(u in src and src[u][1] == f"{u}.mat" for u in uids)
    with pytest.raises(RuntimeError):
        ep.pool(n=973)


def _z(n_s, seed, names=NAMES20, pad_s=0):
    d = np.random.default_rng(seed).normal(0, 20, (20, n_s * 128))
    d[:19, :pad_s * 128] = 0.0
    return {"all_names": np.array(names), "x19": d[:19].astype(np.float32), "fs": np.float64(128)}


def test_plan_rule():
    noekg = NAMES20[:-1] + ["X1"]
    spec = {"p0": _z(2000, 0), "p1": _z(1700, 1), "p2": _z(2100, 2, names=noekg),
            "p3": _z(2000, 3, names=NAMES20 + ["ECG2"]), "p4": _z(2400, 4, pad_s=500), "p5": _z(1900, 5, names=noekg)}
    spec["p5"]["x19"][18, :] = 0.0
    entries = [(u, u.upper(), "design", u) for u in sorted(spec)]
    src = {u: ("EEGs", f"{u}.mat") for u in spec}
    rows = ep.plan(entries, src, spec.__getitem__)
    assert [r["status"] for r in rows] == ["ok", "no_window", "no_ekg", "several_ekg", "ok", "no_ekg"]
    assert [r["position"] for r in rows] == list(range(6)) and [r["uid"] for r in rows] == sorted(spec)
    for r in rows:
        el = bn.eligible_starts(spec[r["uid"]]["x19"], 128.0, run_s=1800)
        assert r["n_eligible_starts"] == len(el)
        if r["status"] == "ok":
            assert r["start_s"] == np.random.default_rng([20261016, 4, r["position"]]).choice(el)
            assert r["channel"] == "EKG" and r["fs"] == 128.0
        else:
            assert r["start_s"] is None
    assert rows[4]["start_s"] >= 530 and rows[3]["channel"] == "EKG|ECG2"
    assert ep.plan(entries[4:5], src, spec.__getitem__)[0]["start_s"] == int(
        np.random.default_rng([20261016, 4, 0]).choice(bn.eligible_starts(spec["p4"]["x19"], 128.0, run_s=1800)))


def test_run_streams_with_the_check_reader():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "bdsp")
        os.makedirs(f"{root}/EEGs")
        data, spec = {}, {}
        for i, (dur, names) in enumerate([(2300, NAMES20), (1500, NAMES20), (2600, NAMES20[:-1] + ["X1"]),
                                          (2000, NAMES20)]):
            u = f"Bonobo0004{i}_1"
            d = np.random.default_rng(30 + i).normal(0, 20, (20, dur * 128))
            d[19] += 0.123456789
            bs.write_mat(f"{root}/EEGs/{u}.mat", d, 128, names)
            data[u] = d
            spec[u] = {"all_names": np.array(names), "x19": d[:19].astype(np.float32), "fs": np.float64(128)}
        entries = [(u, u[:-2], "extra", u) for u in sorted(spec)]
        src = {u: ("EEGs", f"{u}.mat") for u in spec}
        out = os.path.join(tmp, "ekg")
        before = sorted(os.listdir(f"{root}/EEGs"))
        df, sha = ep.run(entries, src, load=spec.__getitem__, out=out, root=root, **FAKE)
        assert list(df.columns) == ep.COLS and list(df.status) == ["ok", "no_window", "no_ekg", "ok"]
        assert sorted(os.listdir(out)) == ["Bonobo00040_1.npz", "Bonobo00043_1.npz", "manifest.csv"]
        assert sha == hashlib.sha256(open(os.path.join(out, "manifest.csv"), "rb").read()).hexdigest()
        man = pd.read_csv(os.path.join(out, "manifest.csv"), keep_default_na=False)
        assert man.start_s[1] == "" and man.ekg_sha256[1] == ""
        for r in df[df.status == "ok"].itertuples():
            with np.load(os.path.join(out, f"{r.uid}.npz")) as z:
                a = int(r.start_s) * 128
                assert z["ekg"].tobytes() == data[r.uid][19, a:a + 230400].astype(np.float32).tobytes()
                assert set(z.files) == {"ekg", "fs", "start_s", "channel", "uid"} and int(z["start_s"]) == r.start_s
                assert hashlib.sha256(z["ekg"].tobytes()).hexdigest() == r.ekg_sha256
        m = os.path.getmtime(os.path.join(out, "Bonobo00040_1.npz"))
        df2, sha2 = ep.run(entries, src, load=spec.__getitem__, out=out, root=root, **FAKE)
        assert sha2 == sha and os.path.getmtime(os.path.join(out, "Bonobo00040_1.npz")) == m
        assert sorted(os.listdir(f"{root}/EEGs")) == before


def test_run_refuses_a_stale_file():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "bdsp")
        os.makedirs(f"{root}/EEGs")
        d = np.random.default_rng(7).normal(0, 20, (20, 2200 * 128))
        bs.write_mat(f"{root}/EEGs/Bonobo00050_1.mat", d, 128, NAMES20)
        spec = {"Bonobo00050_1": {"all_names": np.array(NAMES20), "x19": d[:19].astype(np.float32),
                                  "fs": np.float64(128)}}
        out = os.path.join(tmp, "ekg")
        os.makedirs(out)
        np.savez(os.path.join(out, "Bonobo00050_1.npz"), ekg=np.zeros(230400, np.float32), fs=128.0,
                 start_s=-1, channel="EKG", uid="Bonobo00050_1")
        with pytest.raises(ValueError, match="does not match"):
            ep.run([("Bonobo00050_1", "Bonobo00050", "design", "Bonobo00050_1")],
                   {"Bonobo00050_1": ("EEGs", "Bonobo00050_1.mat")}, load=spec.__getitem__, out=out, root=root,
                   **FAKE)
