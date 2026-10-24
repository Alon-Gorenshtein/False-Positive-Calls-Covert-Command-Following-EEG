"""Tests for bench/engine.py (Engine section).
Synthetic data except the pinned pDoC development draw (file names only, no signal is read).
cd code && /usr/bin/python3 -m pytest bench/tests/test_engine.py -q"""
import glob
import importlib
import os
import warnings

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.signal import periodogram

from bench import engine as en

cn = importlib.import_module("113_columbia_null")
NAMES19 = "Fp1 F3 C3 P3 F7 T3 T5 O1 Fz Cz Pz Fp2 F4 C4 P4 F8 T4 T6 O2".split()
_POS = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]



def _pink(n, fs, rng):
    f = np.fft.rfftfreq(n, 1 / fs)
    g = np.zeros_like(f)
    g[1:] = 1 / np.sqrt(f[1:])
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * g, n=n)


def _alpha(n, fs, rng, lo=8.0, hi=13.0):
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * ((f >= lo) & (f <= hi)), n=n)


def vc_eeg(fs=128, dur=80, seed=0, names=NAMES19):
    """Volume-conducted synthetic EEG (uV): one independent pink + alpha + beta source under every 10-20 site, mixed by
    a Gaussian kernel of half a neighbour distance, plus 0.5 uV white sensor noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    P = np.array([_POS[c] for c in names])
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    M = np.exp(-D ** 2 / (2 * 0.035 ** 2))
    S = np.array([_pink(n, fs, rng) for _ in names])
    S = 10 * S / S.std(1, keepdims=True)
    S += np.array([12 * _alpha(n, fs, rng) / _alpha(n, fs, rng).std() for _ in names])
    S += np.array([4 * _alpha(n, fs, rng, 14, 30) / _alpha(n, fs, rng, 14, 30).std() for _ in names])
    return (np.einsum("ij,jt->it", M, S) + 0.5 * rng.standard_normal((len(names), n))).astype(np.float32)


def rec_of(x, fs=128, names=NAMES19, source="pdoc", meta=None):
    return en.Recording(source=source, uid="u", patient="p", x=x, fs=float(fs), names=list(names), meta=meta or {})


def band_power(x, fs, lo, hi):
    f, p = periodogram(np.asarray(x, float), fs=fs, window="hann", axis=-1)
    return p[..., (f >= lo) & (f < hi)].sum(-1)



def _fake_morgoth(path, dur):
    """Morgoth columns that 108.row_labels reads; every window normal awake."""
    z = np.zeros(dur)
    pd.DataFrame({"t_s": np.arange(dur), "valid": np.ones(dur, bool), "normal_p": z + 0.9, "bs_p": z,
                  "iiic_other": z + 1.0, "slowing_generalized": z, "sleep_n1": z, "sleep_n2": z, "sleep_n3": z,
                  "sleep_rem": z}).to_parquet(path)


def _fake_bdsp(tmp_path, n=48, seed=5):
    """A fake design sample: most recordings clean and long, some too short, some without x19 or Morgoth."""
    sig, mor, meta = (tmp_path / d for d in ("sig", "mor", "meta"))
    for d in (sig, mor, meta):
        d.mkdir()
    rng = np.random.default_rng(seed)
    files = [f"P{k:03d}_0.mat" for k in range(n)]
    pd.DataFrame({"patient_id": [f"P{k:03d}" for k in range(n)], "file": files}).to_csv(meta / "sample_500.csv",
                                                                                         index=False)
    pd.DataFrame({"patient_id": [f"T{k:03d}" for k in range(10)],
                  "file": [f"T{k:03d}_0.mat" for k in range(10)]}).to_csv(meta / "train_600.csv", index=False)
    pd.DataFrame({"patient_id": [], "file": []}).to_csv(meta / "bench_extra_500.csv", index=False)
    fs = 4
    kind = {}
    for k, f in enumerate(files):
        stem = f[:-4]
        kind[stem] = ("short" if k % 7 == 3 else "nox19" if k % 11 == 5 else "nomor" if k % 13 == 6 else
                      "missing" if k % 17 == 8 else "ok")
        if kind[stem] == "missing":
            continue
        dur = 1200 if kind[stem] == "short" else 1700
        x = (10 * rng.standard_normal((19, dur * fs))).astype(np.float32)
        arrs = {"x": x[:4], "names": np.array(["F3", "F4", "C3", "C4"]), "fs": float(fs)}
        if kind[stem] != "nox19":
            arrs |= {"x19": x, "names19": np.array(NAMES19)}
        np.savez(sig / f"{stem}.npz", **arrs)
        if kind[stem] != "nomor":
            _fake_morgoth(mor / f"{stem}.parquet", dur)
    return str(sig), str(mor), str(meta), kind


def _save_x19(sig, stem, dur, seed):
    x = (10 * np.random.default_rng(seed).standard_normal((19, dur * 4))).astype(np.float32)
    np.savez(os.path.join(sig, f"{stem}.npz"), x=x[:4], names=np.array(["F3", "F4", "C3", "C4"]), fs=4.0, x19=x,
             names19=np.array(NAMES19))


def _complete(sig, mor, meta, kind):
    """Make every recording of the fake sample usable (x19 + Morgoth), as after the pull and the x19 backfill."""
    s = pd.read_csv(os.path.join(meta, "sample_500.csv"))
    for r, stem in enumerate(s.file.str[:-4]):
        if kind[stem] in ("nox19", "nomor", "missing"):
            _save_x19(sig, stem, 1700, r)
            _fake_morgoth(os.path.join(mor, f"{stem}.parquet"), 1700)
    return s, cn.sample_order(len(s))


def test_bdsp_dev_is_split_dev_of_the_columbia_eligible_list(tmp_path, monkeypatch):
    sig, mor, meta, kind = _fake_bdsp(tmp_path)
    monkeypatch.setitem(cn.SOURCES, "design", (sig, mor, os.path.join(meta, "sample_500.csv"), cn.SEED))
    analysis_113, dev_113 = cn.split_dev(cn.eligible_list())
    sp = en.bdsp_split(sig, mor, meta)
    assert len(dev_113) == 20
    assert sp["dev"] == dev_113
    assert set(analysis_113) <= set(sp["analysis"])
    assert not {s for _, _, s in sp["dev"]} & {s for _, _, s in sp["analysis"]}
    usable = {s for s, k in kind.items() if k in ("ok", "short")}
    assert {s for _, _, s in sp["analysis"]} | {s for _, _, s in sp["dev"]} == usable
    assert [r for r, _, _ in sp["analysis"]] == sorted(r for r, _, _ in sp["analysis"])


def test_bdsp_dev_is_provisional_while_a_later_ranked_recording_is_incomplete(tmp_path):
    sig, mor, meta, kind = _fake_bdsp(tmp_path)
    s, order = _complete(sig, mor, meta, kind)
    assert en.bdsp_split(sig, mor, meta)["final"]
    last = s.file[order[-1]][:-4]
    os.remove(os.path.join(mor, f"{last}.parquet"))
    sp = en.bdsp_split(sig, mor, meta)
    assert not sp["final"] and sp["pending"] == 1
    assert last not in {t for _, _, t in sp["dev"]}


def _fake_pdoc(tmp_path):
    raw = tmp_path / "pdoc"
    raw.mkdir()
    for f in [f"Patient_{k}_VS.edf" for k in range(1, 43)] + ["._Patient_3_VS.edf"]:
        (raw / f).write_bytes(b"")
    return str(raw)


def _frozen_setup(tmp_path, monkeypatch):
    """A complete fake sample, frozen, with the engine pointed at it; then data that would shift a fresh scan."""
    sig, mor, meta, kind = _fake_bdsp(tmp_path)
    s, order = _complete(sig, mor, meta, kind)
    raw = _fake_pdoc(tmp_path)
    path = os.path.join(meta, "dev_split.json")
    for k, v in (("SIG", sig), ("MOR", mor), ("META", meta), ("PDOC_RAW", raw), ("DEV_FILE", path),
                 ("EXTRA_SIG", str(tmp_path / "none")), ("EXTRA_MOR", str(tmp_path / "none")),
                 ("EXTRA_LOG", str(tmp_path / "none.log"))):
        monkeypatch.setattr(en, k, v)
    live_b, live_p = en.bdsp_split(sig, mor, meta), en.pdoc_split(raw)
    out = en.freeze_dev_split(path)
    assert os.path.exists(path) and out["bdsp"] == [list(t) for t in live_b["dev"]] and out["pdoc"] == live_p["dev"]
    _save_x19(sig, live_b["dev"][-1][2], 1200, 99)
    (tmp_path / "pdoc" / "Patient_43_VS.edf").write_bytes(b"")
    return sig, mor, meta, raw, path, live_b, live_p


def test_freeze_refuses_while_the_bdsp_dev_split_is_pending(tmp_path):
    sig, mor, meta, kind = _fake_bdsp(tmp_path)
    s, order = _complete(sig, mor, meta, kind)
    os.remove(os.path.join(mor, f"{s.file[order[-1]][:-4]}.parquet"))
    path = os.path.join(meta, "dev_split.json")
    with pytest.raises(RuntimeError):
        en.freeze_dev_split(path, sig=sig, mor=mor, meta=meta, raw=_fake_pdoc(tmp_path))
    assert not os.path.exists(path)


def test_freeze_refuses_to_overwrite(tmp_path, monkeypatch):
    sig, mor, meta, raw, path, _, _ = _frozen_setup(tmp_path, monkeypatch)
    before = open(path).read()
    with pytest.raises(FileExistsError):
        en.freeze_dev_split(path)
    assert open(path).read() == before


def test_frozen_membership_is_used_after_writing(tmp_path, monkeypatch):
    sig, mor, meta, raw, path, live_b, live_p = _frozen_setup(tmp_path, monkeypatch)
    assert en.bdsp_split(sig, mor, meta)["dev"] != live_b["dev"]
    assert en.pdoc_split(raw)["dev"] != live_p["dev"]
    assert en.bdsp_split(sig, mor, meta, dev_file=path)["dev"] == live_b["dev"]
    assert en.pdoc_split(raw, dev_file=path)["dev"] == live_p["dev"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recs = list(en.load_recordings("bdsp", "dev"))
    assert [r.uid for r in recs] == [t for _, _, t in live_b["dev"]]
    assert all(r.meta["dev"] and r.meta["split_frozen"] for r in recs)
    assert [it["uid"] for it in en._items("pdoc", "dev")] == [f[:-4] for f in live_p["dev"]]


def test_frozen_analysis_never_overlaps_dev(tmp_path, monkeypatch):
    sig, mor, meta, raw, path, live_b, live_p = _frozen_setup(tmp_path, monkeypatch)
    fresh = {t for _, _, t in en.bdsp_split(sig, mor, meta)["dev"]}
    frozen = {t for _, _, t in live_b["dev"]}
    for src in ("bdsp", "pdoc"):
        dev = {it["uid"] for it in en._items(src, "dev")}
        ana = {it["uid"] for it in en._items(src, "analysis")}
        assert dev and ana and not dev & ana
        assert {it["uid"] for it in en._items(src, "all")} == dev | ana
    ana_b = {it["uid"] for it in en._items("bdsp", "analysis")}
    assert fresh - frozen <= ana_b
    assert "Patient_43_VS" in {it["uid"] for it in en._items("pdoc", "analysis")}



def _fake_extras(tmp_path, meta, n=8, partial=(), absent=(), log=None, patients=None):
    """Extras in the design npz/parquet format (x19 at 128 Hz). partial: only a .npz.part, or an npz without its
    Morgoth file; absent: not pulled yet. log: lines of the pull log."""
    xs, xm = tmp_path / "xsig", tmp_path / "xmor"
    xs.mkdir(exist_ok=True)
    xm.mkdir(exist_ok=True)
    pats = patients or [f"E{k:03d}" for k in range(n)]
    pd.DataFrame({"patient_id": pats, "file": [f"{p}_0.mat" for p in pats]}).to_csv(
        os.path.join(meta, "bench_extra_500.csv"), index=False)
    for k, p in enumerate(pats):
        stem = f"{p}_0"
        if stem in absent:
            continue
        if stem in partial and k % 2 == 0:
            (xs / f"{stem}.npz.part").write_bytes(b"\0" * 64)
            continue
        x = (10 * np.random.default_rng(100 + k).standard_normal((19, 128 * 60))).astype(np.float32)
        np.savez(xs / f"{stem}.npz", x=x[[1, 12, 2, 13]], names=np.array(["F3", "F4", "C3", "C4"]), fs=128.0, x19=x,
                 names19=np.array(NAMES19))
        if stem not in partial:
            _fake_morgoth(xm / f"{stem}.parquet", 60)
    lf = tmp_path / "extra_pull.log"
    lf.write_text("\n".join(log or []) + "\n")
    return str(xs), str(xm), str(lf)


def _with_extras(tmp_path, monkeypatch, **kw):
    """A frozen fake design sample plus fake extras, with the engine pointed at both."""
    sig, mor, meta, raw, path, live_b, live_p = _frozen_setup(tmp_path, monkeypatch)
    xs, xm, lf = _fake_extras(tmp_path, meta, **kw)
    for k, v in (("EXTRA_SIG", xs), ("EXTRA_MOR", xm), ("EXTRA_LOG", lf)):
        monkeypatch.setattr(en, k, v)
    return meta, live_b


def _assert_disjoint_listing(meta):
    """End-to-end check on what the engine lists: no listed BDSP patient is in train_600.csv, and no patient appears
    under two uids."""
    items = en._items("bdsp", "all")
    train = set(pd.read_csv(os.path.join(meta, "train_600.csv")).patient_id.astype(str))
    pats = [it["patient"] for it in items]
    if set(pats) & train or len(set(pats)) != len(pats):
        raise AssertionError(f"overlapping BDSP patients listed: {sorted(set(pats) & train)}")


@pytest.mark.parametrize("clash", ["design", "train", "twice"])
def test_extras_must_be_patient_disjoint(tmp_path, monkeypatch, clash):
    meta, _ = _with_extras(tmp_path, monkeypatch)
    pats = [f"E{k:03d}" for k in range(7)] + [{"design": "P010", "train": "T003", "twice": "E002"}[clash]]
    _fake_extras(tmp_path, meta, patients=pats)
    with pytest.raises(RuntimeError, match="patient"):
        en.bdsp_extras()
    with pytest.raises(RuntimeError, match="patient"):
        en._items("bdsp", "analysis")


def test_a_broken_disjointness_check_is_caught(tmp_path, monkeypatch):
    meta, _ = _with_extras(tmp_path, monkeypatch)
    _assert_disjoint_listing(meta)
    _fake_extras(tmp_path, meta, patients=[f"E{k:03d}" for k in range(7)] + ["T003"])
    monkeypatch.setattr(en, "check_disjoint", lambda lists: None)
    with pytest.raises(AssertionError, match="T003"):
        _assert_disjoint_listing(meta)


def test_extras_leave_the_dev_split_unchanged(tmp_path, monkeypatch):
    meta, live_b = _with_extras(tmp_path, monkeypatch)
    dev = [it["uid"] for it in en._items("bdsp", "dev")]
    assert dev == [t for _, _, t in live_b["dev"]]
    sp = en.bdsp_split(en.SIG, en.MOR, en.META, en.DEV_FILE)
    design = [t for _, _, t in sp["analysis"]]
    extras = [f"E{k:03d}_0" for k in range(8)]
    ana = [it["uid"] for it in en._items("bdsp", "analysis")]
    assert ana == design + extras
    whole = [t for _, _, t in sorted(sp["dev"] + sp["analysis"])]
    assert [it["uid"] for it in en._items("bdsp", "all")] == whole + extras
    assert not set(dev) & set(ana)
    assert all(it["meta"]["dev"] is False and it["meta"]["extra"] for it in en._items("bdsp", "analysis")[-8:])
    assert [it["patient"] for it in en._items("bdsp", "analysis")[-8:]] == [f"E{k:03d}" for k in range(8)]


def test_an_extras_only_recording_loads_with_x19_at_128_hz(tmp_path, monkeypatch):
    _with_extras(tmp_path, monkeypatch)
    assert not os.path.exists(os.path.join(en.MOR, "E003_0.parquet"))
    rec = [r for r in en.load_recordings("bdsp", "analysis") if r.uid == "E003_0"][0]
    assert rec.source == "bdsp" and rec.patient == "E003" and rec.fs == 128.0
    assert rec.x.shape == (19, 128 * 60) and rec.x.dtype == np.float32 and rec.names == NAMES19
    assert rec.meta["extra"] and rec.meta["dev"] is False and len(rec.meta["morgoth_lab"]) == 60
    assert en.stratum(rec, 0, 30) == ("normal_awake", False)


def test_extras_are_listed_only_when_complete_on_disk(tmp_path, monkeypatch):
    part, absent = ("E001_0", "E002_0"), ("E005_0",)
    head = ["2026-10-13 16:34:38 8 sampled, 0 done, 8 to do", "16:35:56 [1/8] E000_0.mat 18.4 MB fetch 46 s"]
    meta, _ = _with_extras(tmp_path, monkeypatch, partial=part, absent=absent, log=head)
    ex = en.bdsp_extras()
    listed = [t for _, t in ex["analysis"]]
    assert listed == ["E000_0", "E003_0", "E004_0", "E006_0", "E007_0"]
    assert ex["n_listed"] == 8 and ex["n_present"] == 5 and not ex["complete"]
    assert sorted(ex["missing"]) == ["E001_0", "E002_0", "E005_0"]
    assert not {"E001_0", "E002_0", "E005_0"} & {it["uid"] for it in en._items("bdsp", "analysis")}
    fail = head + ["16:40:00 [4/8] FAIL E005_0.mat: TimeoutError()"]
    done = fail + ["2026-10-13 18:00:00 BDSP_EXTRA_PULL_DONE 7 ok, 1 failed, 1.5 h"]
    xs, xm, lf = _fake_extras(tmp_path, meta, partial=part, absent=absent, log=done)
    assert not en.bdsp_extras()["complete"]
    for f in os.listdir(xs):
        if f.endswith(".part"):
            os.remove(os.path.join(xs, f))
    _fake_extras(tmp_path, meta, absent=absent, log=fail)
    assert not en.bdsp_extras()["complete"]
    _fake_extras(tmp_path, meta, absent=absent, log=done + [head[0]] + fail[1:])
    assert not en.bdsp_extras()["complete"]
    _fake_extras(tmp_path, meta, absent=absent, log=done)
    ex = en.bdsp_extras()
    assert ex["complete"] and ex["n_present"] == 7 and ex["failed"] == ["E005_0"]
    assert "E005_0" not in {it["uid"] for it in en._items("bdsp", "analysis")}
    _fake_extras(tmp_path, meta, log=[])
    assert en.bdsp_extras()["complete"] and en.bdsp_extras()["n_present"] == 8


def test_incomplete_extras_warn_in_the_analysis_listing(tmp_path, monkeypatch):
    _with_extras(tmp_path, monkeypatch, absent=("E005_0",))
    with pytest.warns(UserWarning, match="extra"):
        en._items("bdsp", "analysis")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        en._items("bdsp", "dev")


def test_the_touched_exclusion_reaches_extras(tmp_path, monkeypatch):
    _with_extras(tmp_path, monkeypatch)
    rn = importlib.import_module("114_benchmark")
    uids = [it["uid"] for it in rn.list_items({"source": "bdsp"}, "analysis", skip={"E003_0"})]
    assert "E003_0" not in uids and "E004_0" in uids


def test_pdoc_dev_is_three_files_drawn_with_the_seed_from_the_sorted_list(tmp_path):
    names = [f"Patient_{k}_VS.edf" for k in range(1, 43)] + ["Patient 1_VS.edf"]
    for f in names + ["._Patient_3_VS.edf", "._x.edf"]:
        (tmp_path / f).write_bytes(b"")
    sp = en.pdoc_split(str(tmp_path))
    files = sorted(names)
    idx = np.random.default_rng(20261023).choice(len(files), 3, replace=False)
    assert sp["dev"] == sorted(files[i] for i in idx)
    assert len(sp["dev"]) == 3 and not any(f.startswith("._") for f in sp["dev"] + sp["analysis"])
    assert not set(sp["dev"]) & set(sp["analysis"])
    assert sorted(sp["dev"] + sp["analysis"]) == files


def test_pdoc_dev_draw_on_the_real_file_list_is_pinned():
    if not os.path.isdir(en.PDOC_RAW):
        pytest.skip("pDoC PSGs not on disk")
    sp = en.pdoc_split()
    assert sp["dev"] == ["Patient_21_MCS+.edf", "Patient_26_VS.edf", "Patient_2_MCS+.edf"]
    assert len(sp["analysis"]) == 38 and not set(sp["dev"]) & set(sp["analysis"])


def test_icare_has_no_dev_split_and_skips_appledouble(tmp_path):
    for f in ("0284_010.npz", "0284_050.npz", "0301_002.npz", "._0284_010.npz"):
        (tmp_path / f).write_bytes(b"")
    sp = en.icare_split(str(tmp_path))
    assert sp["dev"] == [] and sp["analysis"] == ["0284_010.npz", "0284_050.npz", "0301_002.npz"]


def test_bad_source_or_split_is_refused():
    with pytest.raises(ValueError):
        next(iter(en.load_recordings("bdsp", "train")))
    with pytest.raises(ValueError):
        next(iter(en.load_recordings("eicu", "dev")))





def test_clean_spans_keeps_30s_from_padding():
    fs, n = 10, 400
    x = np.random.default_rng(1).normal(0, 10, (4, n * fs)).astype(np.float32)
    x[2, 200 * fs: 210 * fs] = 0.0
    st = en.clean_spans(rec_of(x, fs, ["F3", "F4", "C3", "C4"]), 50)
    assert set(st.tolist()) == set(range(0, 121)) | set(range(240, 351))


def test_stratum_uses_morgoth_windows_and_na_elsewhere():
    t = np.arange(600)
    lab = np.array(["normal_awake"] * 300 + ["burst_suppression"] * 300, object)
    r = rec_of(np.zeros((19, 600 * 128), np.float32), source="bdsp", meta={"morgoth_t": t, "morgoth_lab": lab})
    assert en.stratum(r, 0, 200) == ("normal_awake", False)
    assert en.stratum(r, 350, 100) == ("burst_suppression", False)
    assert en.stratum(r, 200, 200) == ("mixed", True)
    assert en.stratum(r, 400, 4) == ("burst_suppression", False)
    assert en.stratum(r, 299.5, 1) == ("normal_awake", False)
    assert en.stratum(rec_of(np.zeros((19, 1280), np.float32)), 0, 10) == ("na", False)



def test_artifact_index_flat_saturated_and_emg():
    fs, n = 128, 300
    x = vc_eeg(fs, n, seed=2)
    rng = np.random.default_rng(3)
    x[5, 110 * fs: 120 * fs] = 0.01 * rng.standard_normal(10 * fs)
    x[12, 130 * fs: 145 * fs] += 700.0
    t = np.arange(20 * fs) / fs
    x[16, 150 * fs: 170 * fs] += 60 * np.sin(2 * np.pi * 37 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    r = rec_of(x, fs)
    share, label = en.artifact_index(r, 100, 100)
    assert share == pytest.approx(0.45, abs=0.011) and label == ">30%"
    assert en.artifact_index(r, 110, 10) == (1.0, ">30%")
    assert en.artifact_index(r, 0, 100) == (0.0, "<10%")
    assert en.artifact_index(r, 105, 100)[0] == pytest.approx(0.45, abs=0.011)
    s, lab = en.artifact_index(r, 160, 50)
    assert s == pytest.approx(0.2, abs=0.021) and lab == "10-30%"


def test_artifact_index_labels_at_the_boundaries():
    assert en.artifact_label(0.0999) == "<10%"
    assert en.artifact_label(0.10) == "10-30%"
    assert en.artifact_label(0.30) == "10-30%"
    assert en.artifact_label(0.3001) == ">30%"


def test_artifact_index_median_ignores_padding():
    fs, n = 128, 300
    x = vc_eeg(fs, n, seed=4)
    x[:, 100 * fs:] = 0.0
    r = rec_of(x, fs)
    assert en.artifact_index(r, 0, 100)[0] == 0.0



C3_NEIGH = ("F3", "P3", "Cz", "T3")
FAR = ("Fp2", "F8", "T4", "T6", "O2")


def _cmd_overlay(onsets, dur=6.0, hand="right", labels=None):
    onsets = np.asarray(onsets, float)
    labels = np.ones(len(onsets), int) if labels is None else np.asarray(labels)
    return en.Overlay(onsets_s=onsets, labels=labels, kind="cmd", meta={"resp_dur_s": dur, "resp_hand": hand})


def _plateau_mask(n, fs, onsets, dur):
    m = np.zeros(n, bool)
    for o in onsets:
        m[int((o + 1.5) * fs): int((o + dur - 0.5) * fs)] = True
    return m


def test_erd_reduces_band_power_under_the_template_and_spreads_to_neighbours():
    fs = 128
    x = vc_eeg(fs, 90, seed=6)
    r = rec_of(x, fs)
    on = np.arange(5, 85, 10.0)
    y = en.inject(r, _cmd_overlay(on), np.ones(len(on), bool), "erd", 0.5, np.random.default_rng(7))
    m = _plateau_mask(x.shape[1], fs, on, 6.0)
    ratio = {c: band_power(y[NAMES19.index(c), m], fs, 8, 30) / band_power(x[NAMES19.index(c), m], fs, 8, 30)
             for c in NAMES19}
    drop = {c: 1 - v for c, v in ratio.items()}
    neigh = np.mean([drop[c] for c in C3_NEIGH])
    far = np.max([drop[c] for c in FAR])
    assert drop["C3"] > 0.10
    assert drop["C3"] > neigh > 0.25 * drop["C3"]
    assert far < 0.25 * drop["C3"]
    assert drop["C3"] > 3 * abs(drop["C4"])
    yr = en.inject(r, _cmd_overlay(on, hand="left"), np.ones(len(on), bool), "erd", 0.5, np.random.default_rng(7))
    d4 = 1 - band_power(yr[NAMES19.index("C4"), m], fs, 8, 30) / band_power(x[NAMES19.index("C4"), m], fs, 8, 30)
    assert d4 > 0.10


def test_ers_increases_band_power():
    fs = 128
    x = vc_eeg(fs, 60, seed=8)
    on = np.arange(5, 55, 10.0)
    y = en.inject(rec_of(x, fs), _cmd_overlay(on), np.ones(len(on), bool), "ers", 0.5, np.random.default_rng(1))
    m = _plateau_mask(x.shape[1], fs, on, 6.0)
    c = NAMES19.index("C3")
    assert band_power(y[c, m], fs, 8, 30) > 1.3 * band_power(x[c, m], fs, 8, 30)


def _hf_change(x, y, fs, onsets, dur, ch):
    """Relative 30-45 Hz power change at one channel over the response windows (ramps included)."""
    m = np.zeros(x.shape[1], bool)
    for o in onsets:
        m[int(o * fs): int((o + dur) * fs)] = True
    return abs(band_power(y[ch, m], fs, 30, 45) / band_power(x[ch, m], fs, 30, 45) - 1)


def _hf_share_of_injected(x, y, fs):
    d = np.asarray(y, float) - x
    return band_power(d, fs, 30, 64).sum() / band_power(d, fs, 0.5, 64).sum()


def test_erd_has_no_broadband_step_transient():
    fs = 128
    x = vc_eeg(fs, 90, seed=9)
    r = rec_of(x, fs)
    on = np.arange(5, 85, 10.0)
    mask = np.ones(len(on), bool)
    c = NAMES19.index("C3")
    y = en.inject(r, _cmd_overlay(on), mask, "erd", 0.8, np.random.default_rng(2))
    assert _hf_change(x, y, fs, on, 7.0, c) < 0.05
    smooth = _hf_share_of_injected(x, y, fs)
    assert smooth < 0.01
    ys = en.inject(r, _cmd_overlay(on), mask, "erd", 0.8, np.random.default_rng(2), ramp_s=0.0)
    assert _hf_share_of_injected(x, ys, fs) > 3 * smooth


def test_erd_onset_jitter_is_0_to_1_s_and_nothing_changes_outside_responses():
    fs = 128
    x = vc_eeg(fs, 60, seed=10)
    on = np.arange(5, 55, 10.0)
    y = en.inject(rec_of(x, fs), _cmd_overlay(on, dur=6.0), np.ones(len(on), bool), "erd", 0.5,
                  np.random.default_rng(3))
    ch = np.abs(y - x).max(0) > 0
    starts = []
    for o in on:
        seg = np.flatnonzero(ch[int(o * fs): int((o + 6.0) * fs)])
        assert len(seg)
        starts.append(seg[0] / fs)
        assert seg[-1] / fs <= 6.0
    assert all(0.0 <= s <= 1.0 + 2 / fs for s in starts) and np.ptp(starts) > 0.05
    outside = np.ones(x.shape[1], bool)
    for o in on:
        outside[int(o * fs): int((o + 6.0) * fs) + 1] = False
    assert not ch[outside].any()


def test_resp_mask_is_respected_and_x_is_not_modified_in_place():
    fs = 128
    x = vc_eeg(fs, 60, seed=11)
    x0 = x.copy()
    r = rec_of(x, fs)
    on = np.arange(5, 55, 10.0)
    mask = np.array([True, False, True, False, False])
    y = en.inject(r, _cmd_overlay(on), mask, "erd", 0.5, np.random.default_rng(4))
    assert np.array_equal(r.x, x0) and y is not r.x and y.dtype == np.float32 and y.shape == x.shape
    for o, m in zip(on, mask):
        w = slice(int(o * fs), int((o + 6.0) * fs))
        assert (np.abs(y[:, w] - x[:, w]).max() > 0) == m


def test_no_response_in_burst_suppression():
    fs = 128
    x = vc_eeg(fs, 60, seed=12)
    t = np.arange(60)
    lab = np.array(["burst_suppression"] * 30 + ["normal_awake"] * 30, object)
    r = rec_of(x, fs, source="bdsp", meta={"morgoth_t": t, "morgoth_lab": lab})
    on = np.array([5.0, 15.0, 35.0, 45.0])
    y = en.inject(r, _cmd_overlay(on, dur=5.0), np.ones(4, bool), "erd", 0.5, np.random.default_rng(5))
    assert np.array_equal(y[:, : 30 * fs], x[:, : 30 * fs])
    assert np.abs(y[:, 30 * fs:] - x[:, 30 * fs:]).max() > 0
    ov = en.Overlay(onsets_s=on, labels=np.ones(4, int), kind="stim", meta={})
    ye = en.inject(r, ov, np.ones(4, bool), "erp", 10.0, np.random.default_rng(5))
    assert np.array_equal(ye[:, : 30 * fs], x[:, : 30 * fs]) and np.abs(ye[:, 30 * fs:] - x[:, 30 * fs:]).max() > 0


def test_erp_adds_a_jittered_p3_template_with_centroparietal_maximum():
    fs = 128
    x = np.zeros((19, 60 * fs), np.float32)
    on = np.arange(2, 58, 2.0)
    ov = en.Overlay(onsets_s=on, labels=np.ones(len(on), int), kind="stim", meta={})
    y = en.inject(rec_of(x, fs), ov, np.ones(len(on), bool), "erp", 8.0, np.random.default_rng(6))
    top = np.abs(y).max(1)
    assert NAMES19[int(np.argmax(top))] in ("Cz", "Pz")
    assert top.max() == pytest.approx(8.0, rel=0.02)
    assert top[NAMES19.index("Pz")] > top[NAMES19.index("C3")] > top[NAMES19.index("Fp1")]
    pz = y[NAMES19.index("Pz")]
    lat = np.array([np.argmax(pz[int(o * fs): int((o + 1) * fs)]) / fs for o in on])
    assert (pz >= 0).all()
    assert lat.min() >= 0.35 - 1 / fs and lat.max() <= 0.55 + 1 / fs
    assert np.ptp(lat) > 0.1
    yy = en.inject(rec_of(x, fs), ov, np.r_[True, np.zeros(len(on) - 1, bool)], "erp", 8.0,
                   np.random.default_rng(6))
    assert np.abs(yy[:, int(3.9 * fs):]).max() == 0


def test_inject_needs_response_duration_and_hand_for_erd():
    x = np.zeros((19, 128 * 20), np.float32)
    ov = en.Overlay(onsets_s=np.array([2.0]), labels=np.array([1]), kind="cmd", meta={"resp_hand": "left"})
    with pytest.raises(KeyError):
        en.inject(rec_of(x), ov, np.array([True]), "erd", 0.3, np.random.default_rng(0))
    ov.meta = {"resp_dur_s": 4.0}
    with pytest.raises(KeyError):
        en.inject(rec_of(x), ov, np.array([True]), "erd", 0.3, np.random.default_rng(0))


def test_resp_hand_may_map_label_codes():
    fs = 128
    x = vc_eeg(fs, 40, seed=13)
    on = np.array([5.0, 15.0, 25.0])
    ov = _cmd_overlay(on, hand={1: "left", 2: "right", 0: "none"}, labels=[1, 2, 0])
    y = en.inject(rec_of(x, fs), ov, np.ones(3, bool), "erd", 0.6, np.random.default_rng(8))
    c3, c4 = NAMES19.index("C3"), NAMES19.index("C4")
    w = [slice(int((o + 1.5) * fs), int((o + 5.5) * fs)) for o in on]
    d = lambda c, k: 1 - band_power(y[c, w[k]], fs, 8, 30) / band_power(x[c, w[k]], fs, 8, 30)
    assert d(c4, 0) > d(c3, 0) and d(c3, 1) > d(c4, 1)
    assert np.array_equal(y[:, w[2]], x[:, w[2]])


def test_template_topographies():
    a = en.hand_topography(NAMES19, "right")
    assert NAMES19[int(np.argmax(a))] == "C3" and a.max() == pytest.approx(1.0)
    assert all(0.4 < a[NAMES19.index(c)] < 0.75 for c in C3_NEIGH)
    assert all(a[NAMES19.index(c)] < 0.15 for c in FAR)
    b = en.hand_topography(["F3", "F4", "C3", "C4", "O1", "O2"], "left")
    assert b[3] == pytest.approx(1.0) and b[1] > b[5]
    e = en.erp_topography(NAMES19)
    assert {NAMES19[i] for i in np.argsort(e)[-2:]} == {"Cz", "Pz"}



def test_patient_windows_do_not_overlap_and_stay_clean():
    fs = 10
    rng = np.random.default_rng(14)
    x1 = rng.normal(0, 10, (4, 2000 * fs)).astype(np.float32)
    x1[1, 700 * fs: 760 * fs] = 0.0
    x2 = rng.normal(0, 10, (4, 480 * fs)).astype(np.float32)
    r1, r2 = rec_of(x1, fs, ["F3", "F4", "C3", "C4"]), rec_of(x2, fs, ["F3", "F4", "C3", "C4"])
    r2.uid = "u2"
    run_s = 150
    w = en.patient_windows([r1, r2], run_s)
    for rec in (r1, r2):
        st = sorted(s for r, s in w if r is rec)
        assert all(b - a >= run_s for a, b in zip(st, st[1:]))
        ok = set(en.clean_spans(rec, run_s).tolist())
        assert all(s in ok for s in st)
    assert sum(r is r1 for r, _ in w) == 12 and sum(r is r2 for r, _ in w) == 3
    k = en.patient_windows([r1, r2], run_s, k=5, rng=np.random.default_rng(0))
    assert len(k) == 5 and set(k) <= set(w)
    assert k == sorted(k, key=w.index)
    assert len(en.patient_windows([r1, r2], run_s, k=50)) == len(w)


def test_icare_loader_reorders_to_names19_in_uv(tmp_path):
    """Keys as icare_null_pull._save writes them (x, names, sf, hour, seg, util, pid); I-CARE channel order."""
    icare_order = "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz".split()
    x = np.arange(19, dtype=np.float32)[:, None] * np.ones((1, 480 * 100), np.float32)
    np.savez(tmp_path / "0284_010.npz", x=x, names=np.array(icare_order), sf=100, hour=10, seg=3, util=60.0,
             pid="0284")
    it = dict(source="icare", path=str(tmp_path / "0284_010.npz"), uid="0284_010", patient="0284",
              meta={"file": "0284_010.npz", "dev": False})
    r = en._load(it)
    assert r.names == NAMES19 and r.fs == 100.0 and r.x.dtype == np.float32 and r.x.shape == (19, 48000)
    assert [int(r.x[i, 0]) for i in range(19)] == [icare_order.index(c) for c in NAMES19]
    assert r.patient == "0284" and r.meta["hour"] == 10 and r.meta["pid"] == "0284"


