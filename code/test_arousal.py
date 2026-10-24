"""Tests for 135_arousal.py (part D): synthetic recordings, ECG and rows only."""
import importlib
import math

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

ar = importlib.import_module("135_arousal")
kt, mech, rn, en = ar.kt, ar.mech, ar.rn, ar.en
NAMES = list(en.NAMES19)
FS = 128.0



def slow(n, fs, tau_s, rng):
    """Unit-variance Ornstein-Uhlenbeck process (correlation time tau_s)."""
    a = math.exp(-1.0 / (tau_s * fs))
    return lfilter([math.sqrt(1.0 - a * a)], [1.0, -a], rng.normal(size=n), zi=[a * rng.normal()])[0]


def pink(n, rng):
    f = np.fft.rfftfreq(n)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** -0.5
    x = np.fft.irfft(amp * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))), n=n)
    return x / x.std()


def x19(seconds, rng, drive=None, log_sd=0.0, alpha=1.0, alpha_drive=None):
    """(19, n) uV: shared pink sources, an 8-12 Hz posterior rhythm, sensor noise; amplitude exp(log_sd * drive)."""
    n = int(seconds * FS)
    src = np.array([pink(n, rng) for _ in range(4)])
    a = ar.mc.bandpass(rng.normal(size=n), FS, 8.0, 12.0)
    a = alpha * a / a.std()
    if alpha_drive is not None:
        a = a * np.exp(0.5 * alpha_drive)
    post = np.array([1.0 if c in ("O1", "O2", "P3", "P4", "Pz") else 0.2 for c in NAMES])
    x = rng.normal(size=(19, 4)) @ src + post[:, None] * a[None] + 0.2 * rng.normal(size=(19, n))
    env = np.ones(n) if drive is None else np.exp(log_sd * drive - log_sd ** 2 / 2)
    return (15.0 * x * env[None]).astype(np.float32)


def ecg(seconds, rng, fs=FS, hr=None, hr0=70.0, noise=0.03):
    """ECG whose heart rate follows hr (bpm array at fs) or hr0."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    hr = np.full(n, hr0) if hr is None else hr
    ph = np.cumsum(hr / 60.0 / fs)
    beats = t[np.flatnonzero(np.diff(np.floor(ph)) > 0) + 1]
    x = np.zeros(n)
    for b in beats:
        for dt, w, a in ((-0.03, 0.008, -0.1), (0.0, 0.012, 1.0), (0.03, 0.008, -0.2), (0.25, 0.04, 0.3)):
            c = b + dt
            i0, i1 = max(0, int((c - 5 * w) * fs)), min(n, int((c + 5 * w) * fs) + 1)
            x[i0:i1] += a * np.exp(-0.5 * ((t[i0:i1] - c) / w) ** 2)
    return (x + noise * rng.normal(size=n)).astype(np.float32)


def recording(x, uid="R1", labels=None):
    n_s = x.shape[1] // int(FS)
    t = np.arange(0, n_s - 9)
    lab = np.array(["normal_awake"] * len(t), object) if labels is None else np.asarray(labels, object)
    return en.Recording(source="bdsp", uid=uid, patient="P" + uid, x=x, fs=FS, names=list(NAMES),
                        meta={"morgoth_t": t, "morgoth_lab": lab})


def coupled(seconds, rng, k_eeg, k_hr, tau=40.0):
    """A recording and an ECG sharing one slow drive (EEG amplitude log-SD k_eeg, heart rate SD k_hr bpm)."""
    n = int(seconds * FS)
    g = slow(n, FS, tau, rng)
    x = x19(seconds, rng, drive=g, log_sd=k_eeg)
    e = ecg(seconds, rng, hr=70.0 + k_hr * g)
    return x, e



def test_morgoth_transitions_per_10_min():
    lab = np.array(["normal_awake"] * 1791, object)
    t = np.arange(1791)
    seq = ["a", "a", "b", "b", "invalid", "b", "a"]
    for k, s in enumerate(seq):
        lab[k * 10:(k + 1) * 10] = s
    lab[70:] = "a"
    rec = en.Recording(source="bdsp", uid="T", patient="T", x=np.zeros((19, 128 * 1800), np.float32), fs=FS,
                       names=list(NAMES), meta={"morgoth_t": t, "morgoth_lab": lab})
    assert ar.morgoth_transitions(rec, 0.0, 1800.0) == pytest.approx(2 * 600 / 1800)
    assert ar.morgoth_transitions(rec, 0.0, 60.0) == pytest.approx(1 * 600 / 60)


def test_alpha_peak_rule():
    rng = np.random.default_rng(1)
    with_alpha = x19(600, rng, alpha=1.5)
    no_alpha = x19(600, rng, alpha=0.0)
    assert ar.alpha_marker(with_alpha, NAMES, FS)["alpha_peak"] is True
    r = ar.alpha_marker(no_alpha, NAMES, FS)
    assert r["alpha_peak"] is False and r["alpha_env_logvar"] is None and r["reason"] == "no_posterior_alpha_peak"


def test_alpha_envelope_variance_rises_with_slow_alpha_modulation():
    rng = np.random.default_rng(2)
    n = int(1800 * FS)
    st = ar.alpha_marker(x19(1800, rng, alpha=1.5), NAMES, FS)["alpha_env_logvar"]
    mo = ar.alpha_marker(x19(1800, rng, alpha=1.5, alpha_drive=slow(n, FS, 30.0, rng)), NAMES, FS)["alpha_env_logvar"]
    assert mo > st + 1.0


def test_alpha_envelope_variance_is_read_in_the_slow_band_only():
    rng = np.random.default_rng(20)
    tt = np.arange(int(1800 * FS)) / FS
    drive = lambda f: 2.0 * np.sin(2 * np.pi * f * tt)
    sl = ar.alpha_marker(x19(1800, rng, alpha=1.5, alpha_drive=drive(0.03)), NAMES, FS)["alpha_env_logvar"]
    fa = ar.alpha_marker(x19(1800, rng, alpha=1.5, alpha_drive=drive(0.5)), NAMES, FS)["alpha_env_logvar"]
    assert sl > fa + 1.0


def test_coherence_beats_the_shift_null_only_when_coupled():
    rng = np.random.default_rng(3)
    x, e = coupled(1800, rng, k_eeg=0.6, k_hr=6.0)
    c = ar.hr_envelope_coherence(x, NAMES, FS, e, FS, "C1")
    assert c["coh"] > c["coh_null_p95"] and c["n_shifts"] == 100
    x2 = x19(1800, rng, drive=slow(int(1800 * FS), FS, 40.0, rng), log_sd=0.6)
    c2 = ar.hr_envelope_coherence(x2, NAMES, FS, e, FS, "C2")
    assert c2["coh"] < c["coh"] and abs(c2["coh"] - c2["coh_null_mean"]) < 0.15
    assert ar.hr_envelope_coherence(x, NAMES, FS, e, FS, "C1") == c


def test_window_measures_fields_and_sources():
    rng = np.random.default_rng(4)
    x, e = coupled(1800, rng, k_eeg=0.4, k_hr=4.0)
    rec = recording(x)
    w = ar.window_measures(rec, {"ekg": e, "fs": FS}, 0.0, 1800.0)
    h, v = kt.a_hr(e, FS), kt.ver_ref(x, NAMES, FS, "R1")
    assert w["a_hr"] == h["a_hr"] and w["log_ver_ref"] == v["log_ver_ref"]
    assert (w["stratum"], w["transitioning"]) == en.stratum(rec, 0.0, 1800.0)
    assert w["artifact_share"] == en.artifact_index(rec, 0.0, 1800.0)[0]
    assert w["transitions_per_10min"] == 0.0 and "coh" in w and "alpha_env_logvar" in w


def test_window_measures_cut_the_window_from_a_longer_recording():
    rng = np.random.default_rng(5)
    x = x19(2400, rng)
    e = ecg(1800, rng)
    lab = np.where(np.arange(2391) < 1000, "sleep", "normal_awake").astype(object)
    rec = recording(x, labels=lab)
    assert en.stratum(rec, 300.0, 1800.0)[0] != en.stratum(rec, 0.0, 1800.0)[0]
    w = ar.window_measures(rec, {"ekg": e, "fs": FS}, 300.0, 1800.0)
    assert w["log_ver_ref"] == kt.ver_ref(x[:, 300 * 128:2100 * 128], NAMES, FS, "R1")["log_ver_ref"]
    assert (w["stratum"], w["transitioning"]) == en.stratum(rec, 300.0, 1800.0)
    with pytest.raises(ValueError, match="1800"):
        ar.window_measures(rec, {"ekg": e[:-5], "fs": FS}, 300.0, 1800.0)



def _table(rng, n, beta):
    a = rng.normal(size=n)
    st = rng.choice(["normal_awake", "sleep", "diffuse_slowing"], n, p=[0.5, 0.3, 0.2])
    art = rng.uniform(0, 0.3, n)
    y = 5.0 * (beta * a + 0.5 * (st == "sleep") + 1.0 * art + math.sqrt(1 - beta ** 2) * rng.normal(size=n))
    return pd.DataFrame({"uid": [f"U{i}" for i in range(n)], "a_hr": a, "log_ver_ref": y, "stratum": st,
                         "artifact_share": art})


def test_d1_model_recovers_a_standardised_beta():
    out = ar.d1_model(_table(np.random.default_rng(6), 400, 0.3), "a_hr")
    assert out["beta"] == pytest.approx(0.3, abs=0.08) and out["se_type"] == "HC3" and out["n"] == 400
    assert out["lower95_one_sided"] == pytest.approx(out["beta"] - 1.6448536 * out["se"], abs=1e-6)
    assert set(out["covariates"]) >= {"artifact_share", "stratum[sleep]", "stratum[diffuse_slowing]"}
    assert out["stratum_reference"] == "normal_awake"
    import statsmodels.api as sm
    d = _table(np.random.default_rng(6), 400, 0.3)
    z = lambda v: (v - v.mean()) / v.std()
    X = pd.DataFrame({"a": z(d["a_hr"]), "art": d["artifact_share"], "s": (d["stratum"] == "diffuse_slowing") * 1.0,
                      "l": (d["stratum"] == "sleep") * 1.0})
    ref = sm.OLS(z(d["log_ver_ref"]), sm.add_constant(X)).fit(cov_type="HC3")
    assert out["se"] == pytest.approx(float(ref.bse["a"]), rel=1e-9)


def test_d1_model_counts_and_drops_na():
    d = _table(np.random.default_rng(7), 100, 0.2)
    d.loc[:4, "a_hr"] = np.nan
    d.loc[5:6, "log_ver_ref"] = np.nan
    out = ar.d1_model(d, "a_hr")
    assert out["n"] == 93 and out["n_na"] == {"a_hr": 5, "log_ver_ref": 2}


def test_d1_bar():
    b = lambda beta, lo: ar.d1_bar({"beta": beta, "lower95_one_sided": lo})["confirmed"]
    assert b(0.15, 0.01) is True and b(0.30, 0.10) is True
    assert b(0.149, 0.05) is False and b(0.30, 0.0) is False and b(0.30, -0.02) is False
    assert ar.d1_bar({"note": "not estimated"})["confirmed"] is False



def _rows(rng, n_pat, a_to_v, v_to_call, a_to_call):
    rows = []
    for i in range(n_pat):
        a = rng.normal()
        v = a_to_v * a + math.sqrt(max(0.0, 1 - a_to_v ** 2)) * rng.normal()
        for pipe, nv in (("cornell", 2), ("mbvt", 2)):
            for k in range(nv):
                lin = -2.0 + v_to_call * v + a_to_call * a
                rows.append({"patient": f"P{i}", "uid": f"P{i}", "pipe": pipe, "variant": f"{pipe}{k}",
                             "a_hr": a, "log_ver": v, "call": bool(rng.random() < 1 / (1 + math.exp(-lin)))})
    return pd.DataFrame(rows)


def test_mediation_share_is_high_when_the_effect_runs_through_ver():
    d = _rows(np.random.default_rng(8), 600, 0.7, 1.0, 0.0)
    m = ar.mediation(d, {"cornell": 2, "mbvt": 2}, B=100, name="t")
    assert m["total"]["beta"] > 0.3 and abs(m["direct"]["beta"]) < 0.2 and m["share_mediated"] > 0.6
    assert m["B"] == 100 and len(m["share_ci95"]) == 2


def test_mediation_share_is_low_for_a_direct_effect():
    d = _rows(np.random.default_rng(9), 600, 0.0, 0.0, 0.6)
    m = ar.mediation(d, {"cornell": 2, "mbvt": 2}, B=100, name="t")
    assert m["direct"]["beta"] > 0.3 and abs(m["share_mediated"]) < 0.25


def test_mediation_total_effect_is_the_weighted_fe_model():
    d = _rows(np.random.default_rng(21), 200, 0.5, 0.8, 0.2)
    nv = {"cornell": 2, "mbvt": 5}
    m = ar.mediation(d, nv, B=5, name="t")
    e = mech._prep(d, nv)
    z = mech.zscore(e["a_hr"].astype(float), e["w"])[0]
    ref = mech.fe_model(e, {"z": z}, 5, "t", e["w"])
    assert m["total"]["beta"] == pytest.approx(ref["coef"]["z"]["beta"], rel=1e-6)


def test_mediation_uses_one_set_of_draws_for_both_models():
    d = _rows(np.random.default_rng(10), 200, 0.5, 0.8, 0.2)
    m1 = ar.mediation(d, {"cornell": 2, "mbvt": 2}, B=40, name="t")
    m2 = ar.mediation(d, {"cornell": 2, "mbvt": 2}, B=40, name="t")
    assert m1 == m2 and m1["n_boot_failed"] == 0



def test_pdoc_ecg_channel_rule():
    assert ar.pdoc_ecg_name(["C3", "ECG", "O1"]) == ("ECG", None)
    assert ar.pdoc_ecg_name(["C3", "EKG1"]) == ("EKG1", None)
    assert ar.pdoc_ecg_name(["C3", "O1"]) == (None, "no_ecg")
    assert ar.pdoc_ecg_name(["ECG1", "ECG2"]) == (None, "several_ecg")


def test_pdoc_window_is_seeded_by_position_and_108_clean():
    rng = np.random.default_rng(11)
    x = (20 * rng.normal(size=(19, int(2400 * 256)))).astype(np.float32)
    s = ar.pdoc_window(x, 256.0, 3)
    el = ar.bn.eligible_starts(x, 256.0, run_s=1800)
    assert s == int(np.random.default_rng([20261016, 5, 3]).choice(el)) and s != ar.pdoc_window(x, 256.0, 4)
    x[:, int(1000 * 256):int(1100 * 256)] = 0.0
    assert ar.pdoc_window(x, 256.0, 3) is None


def test_d3_measures_with_and_without_ecg():
    rng = np.random.default_rng(12)
    x = x19(2000, rng, drive=slow(int(2000 * FS), FS, 40.0, rng), log_sd=0.4)
    rec = en.Recording(source="pdoc", uid="Patient 3_MCS+", patient="Patient 3", x=x, fs=FS, names=list(NAMES),
                       meta={"diagnosis": "MCS+"})
    e = ecg(2000, rng)
    with_ecg = ar.d3_measures(rec, e, FS, 0)
    s = with_ecg["start_s"]
    assert with_ecg["log_ver_ref"] == kt.ver_ref(x[:, s * 128:(s + 1800) * 128], NAMES, FS, rec.uid)["log_ver_ref"]
    assert with_ecg["a_hr"] == kt.a_hr(e[s * 128:(s + 1800) * 128], FS)["a_hr"]
    no_ecg = ar.d3_measures(rec, None, None, 0, ecg_reason="no_ecg")
    assert no_ecg["a_hr"] is None and no_ecg["a_hr_reason"] == "no_ecg" and no_ecg["log_ver_ref"] is not None
    assert with_ecg["patient_key"] == "Patient_3" and with_ecg["diagnosis"] == "MCS+"


def test_d3_summary_is_descriptive():
    t = pd.DataFrame({"log_ver_ref": [0.1, 0.5, 0.9, 1.3, np.nan], "a_hr": [1.0, np.nan, 1.5, 2.0, 1.1],
                      "diagnosis": ["VS", "VS", "MCS+", "MCS-", "VS"], "arousal": [1, 2, 3, 3, 1],
                      "outcome": ["PDOC", "D", "Consc", "Consc", "PDOC"]})
    s = ar.d3_summary(t)
    assert s["by_diagnosis"]["VS"]["log_ver_ref"]["n"] == 2 and s["by_diagnosis"]["MCS"]["log_ver_ref"]["n"] == 2
    assert s["by_outcome"]["recovered"]["log_ver_ref"]["median"] == pytest.approx(1.1)
    assert s["spearman_with_arousal"]["log_ver_ref"]["n"] == 4 and "p" not in s["spearman_with_arousal"]["log_ver_ref"]
    assert s["note"].startswith("descriptive")


def test_d3_summary_leaves_a_missing_outcome_out_of_the_outcome_split():
    t = pd.DataFrame({"log_ver_ref": [0.1, 0.5, 0.9, 1.3], "a_hr": [1.0, 1.2, 1.5, 2.0],
                      "diagnosis": ["VS", "VS", "MCS+", "MCS-"], "arousal": [1, np.nan, 3, 3],
                      "outcome": ["PDOC", np.nan, "Consc", np.nan]})
    s = ar.d3_summary(t)
    assert s["by_outcome"]["not_recovered"]["log_ver_ref"]["n"] == 1
    assert s["by_outcome"]["recovered"]["log_ver_ref"]["n"] == 1
    assert s["n_outcome_missing"] == 2 and s["n_arousal_missing"] == 1 and set(s["by_outcome"]) == {"recovered",
                                                                                                    "not_recovered"}


def test_window_measures_without_an_ekg():
    rng = np.random.default_rng(13)
    x = x19(1800, rng)
    w = ar.window_measures(recording(x), None, 0.0, 1800.0, ekg_reason="several_ekg")
    assert w["a_hr"] is None and w["a_hr_reason"] == "several_ekg" and w["coh"] is None
    assert w["log_ver_ref"] == kt.ver_ref(x, NAMES, FS, "R1")["log_ver_ref"]



def test_phase1_selection_and_manifest_check():
    uids = [f"Bonobo{i:05d}_0" for i in range(8)]
    g = lambda u: np.random.default_rng(int(u[6:11]))

    def load(u):
        x = (20 * g(u).normal(size=(19, int(1900 * FS)))).astype(np.float32)
        if u == uids[2]:
            x[:, int(900 * FS):int(1000 * FS)] = 0.0
        names = NAMES + (["EKG", "ECG2"] if u == uids[5] else [] if u == uids[6] else ["EKG"])
        return {"x19": x, "fs": FS, "all_names": np.array(names)}
    rows = ar.phase1_selection(list(reversed(uids)), load)
    by = {r["uid"]: r for r in rows}
    assert [r["position"] for r in rows] == list(range(8)) and [r["uid"] for r in rows] == sorted(uids)
    assert by[uids[2]]["status"] == "no_window" and by[uids[5]]["status"] == "several_ekg"
    assert by[uids[6]]["status"] == "no_ekg" and by[uids[6]]["start_s"] is None
    ok = by[uids[0]]
    el = ar.bn.eligible_starts(load(uids[0])["x19"], FS, run_s=1800)
    assert ok["status"] == "ok" and ok["start_s"] == int(np.random.default_rng([20261016, 4, 0]).choice(el))
    man = pd.DataFrame(rows)
    ar.check_phase1_manifest(man, rows)
    for col, val in (("status", "ok"), ("start_s", ok["start_s"] + 1), ("uid", "X")):
        bad = man.copy()
        bad.loc[0 if col != "status" else 2, col] = val
        with pytest.raises(RuntimeError, match="manifest"):
            ar.check_phase1_manifest(bad, rows)


def test_ekg_file_checks(tmp_path):
    e = np.zeros(int(1800 * FS), np.float32)
    f = tmp_path / "U1.npz"
    np.savez(f, ekg=e, fs=np.array(FS), start_s=np.array(30), channel=np.array("EKG"), uid=np.array("U1"))
    import hashlib
    row = {"uid": "U1", "start_s": 30, "fs": FS, "channel": "EKG",
           "ekg_sha256": hashlib.sha256(e.tobytes()).hexdigest()}
    got = ar.load_ekg(str(f), row)
    assert got["fs"] == FS and len(got["ekg"]) == len(e)
    with pytest.raises(RuntimeError, match="sha256"):
        ar.load_ekg(str(f), dict(row, ekg_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="start_s"):
        ar.load_ekg(str(f), dict(row, start_s=31, ekg_sha256=None))



def test_icare_ecg_lead_rule():
    a, b, c = (np.arange(10, dtype=np.float32) * k for k in (1.0, 3.0, 5.0))
    sig, why = ar.icare_ecg_signal(np.vstack([a]), ["ECG"])
    assert why is None and np.array_equal(sig, a)
    sig, why = ar.icare_ecg_signal(np.vstack([b, a]), ["ECGL", "ECGR"])
    assert why is None and np.allclose(sig, b - a)
    sig, why = ar.icare_ecg_signal(np.vstack([c, b, a]), ["SpO2", "ECG1", "ECG2"])
    assert why is None and np.allclose(sig, b - a)
    assert ar.icare_ecg_signal(np.vstack([a, b]), ["LAT1", "RAT1"]) == (None, "no_ecg_lead")
    assert ar.icare_ecg_signal(np.vstack([a, b, c]), ["ECG1", "ECG2", "ECG3"]) == (None, "more_than_two_ecg_leads")
    bad = np.vstack([b, a])
    bad[1, 3] = np.nan
    assert ar.icare_ecg_signal(bad, ["ECG1", "ECG2"]) == (None, "ecg_invalid_samples")


def test_d2_measures_from_kept_files(tmp_path):
    rng = np.random.default_rng(14)
    x = x19(1800, rng)
    e = ecg(1800, rng, fs=250.0)
    win = {"segment": {"seg": 3, "hour": 22, "bytes": 1}, "start_s": 120, "rosc_hours": 22.6}
    import json
    kw, ke = tmp_path / "keep_windows", tmp_path / "keep_ecg"
    kw.mkdir(), ke.mkdir()
    np.savez(kw / "0001.npz", x=x, names=np.array(NAMES), window=json.dumps(win))
    np.savez(ke / "0001.npz", ecg=np.vstack([np.zeros_like(e), e]).astype(np.float32), fs=np.array(250),
             names=np.array(["SpO2", "ECG"]), window=json.dumps(win))
    (ke / "0001.json").write_text(json.dumps({"patient": "0001", "status": "ok"}))
    (ke / "0002.json").write_text(json.dumps({"patient": "0002", "status": "no_ecg_file"}))
    np.savez(kw / "0002.npz", x=x, names=np.array(NAMES), window=json.dumps(dict(win, start_s=60)))
    r1 = ar.d2_measures("0001", str(kw), str(ke), flagged=set())
    assert r1["a_hr"] == kt.a_hr(e, 250.0)["a_hr"]
    assert r1["log_ver_ref"] == kt.ver_ref(x, NAMES, FS, "0001")["log_ver_ref"]
    assert r1["rosc_hours"] == 22.6
    r2 = ar.d2_measures("0002", str(kw), str(ke), flagged=set())
    assert r2["a_hr"] is None and r2["a_hr_reason"] == "no_ecg_file" and r2["log_ver_ref"] is not None
    with pytest.raises(RuntimeError, match="flagged"):
        ar.d2_measures("0001", str(kw), str(ke), flagged={"0001"})
    common = rng.normal(size=len(e)).astype(np.float32)
    np.savez(ke / "0001.npz", ecg=np.vstack([e + common, common]).astype(np.float32), fs=np.array(250),
             names=np.array(["ECGL", "ECGR"]), window=json.dumps(win))
    r3 = ar.d2_measures("0001", str(kw), str(ke), flagged=set())
    assert r3["a_hr"] == kt.a_hr(np.asarray((e + common) - common, float), 250.0)["a_hr"]
    assert r3["ecg_lead"] == "ECGL-ECGR"
    np.savez(ke / "0001.npz", ecg=np.vstack([np.zeros_like(e), e]).astype(np.float32), fs=np.array(250),
             names=np.array(["SpO2", "ECG"]), window=json.dumps(win))
    np.savez(ke / "0001.npz", ecg=np.vstack([np.zeros_like(e), e]).astype(np.float32), fs=np.array(250),
             names=np.array(["SpO2", "ECG"]), window=json.dumps(dict(win, start_s=121)))
    with pytest.raises(RuntimeError, match="window"):
        ar.d2_measures("0001", str(kw), str(ke), flagged=set())


def _patients(rng, n, coef):
    hosp = rng.choice(list("ABCD"), n)
    lv = rng.normal(size=n)
    S = rng.uniform(0, 1, n)
    grade = np.where(S < 0.1, "continuous", np.where(S < 0.5, "discontinuous", np.where(S < 0.99,
                                                                                          "burst_suppression",
                                                                                          "suppressed")))
    lin = -0.5 + coef * lv - 1.5 * (S >= 0.5)
    good = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    cpc = np.where(good == 1, rng.choice([1, 2], n), rng.choice([3, 4, 5], n))
    return pd.DataFrame({"patient": [f"{i:04d}" for i in range(n)], "status": "included", "hospital": hosp,
                         "age": rng.normal(60, 10, n), "ohca": rng.random(n) < 0.8, "shockable": rng.random(n) < 0.4,
                         "ttm": rng.choice([33.0, 36.0, np.nan], n), "rosc_hours": rng.uniform(12, 36, n),
                         "xref": rng.normal(size=n), "stratum": "diffuse_slowing", "transitioning": False,
                         "acns_grade": grade, "acns_S": S, "acns_voltage": rng.normal(2, 0.3, n),
                         "acns_periodic": rng.random(n) < 0.1, "outcome_good": good, "cpc": cpc,
                         "log_ver_ref": lv, "a_hr": 0.5 * lv + rng.normal(size=n)})


def test_d2b_reuses_the_b3_model_on_log_ver_and_the_s_population():
    t = _patients(np.random.default_rng(15), 400, 0.8)
    t.loc[:9, "xref"] = np.nan
    t.loc[10, "age"] = np.nan
    t.loc[11, "log_ver_ref"] = np.nan
    out = ar.d2b(t, B=50)
    assert out["exclusions"] == {"age_missing": 1, "log_ver_ref_na": 1}
    t = t.drop(index=[10, 11])
    p = out["primary_s_below_0.5"]
    assert p["n_patients"] == int((t["acns_S"] < 0.5).sum()) and p["or_per_sd"] > 1.5
    assert out["full_sample"]["n_patients"] == 398 and "z_xref" in p["model"]["coef"]
    assert out["predictor"] == "z(log VER_ref) in place of z(X_ref)"
    assert out["ordinal_cpc"]["or_per_sd_better"] > 1.2 and "with_a_hr" in out


def test_d2a_coupling_model():
    t = _patients(np.random.default_rng(16), 300, 0.0)
    out = ar.d2a(t)
    assert 0.25 < out["primary_s_below_0.5"]["beta"] < 0.7 and out["primary_s_below_0.5"]["positive_point_estimate"]
    p = out["primary_s_below_0.5"]
    assert p["se_type"] == "HC3" and sum(c.startswith("hospital[") for c in p["covariates"]) == 3
    assert out["full_sample"]["n"] == 300


def test_refetch_flags_read_the_last_line_per_patient(tmp_path):
    import json
    for pid in ("0001", "0002", "0003"):
        np.savez(tmp_path / f"{pid}.npz", x=np.zeros(3))
    sha = {p: ar.sha256_file(str(tmp_path / f"{p}.npz")) for p in ("0001", "0002", "0003")}
    lines = [{"patient": "0001", "file": "0001.npz", "sha256": sha["0001"], "verified": False},
             {"patient": "0001", "file": "0001.npz", "sha256": sha["0001"], "verified": True},
             {"patient": "0002", "file": "0002.npz", "sha256": sha["0002"], "verified": True},
             {"patient": "0002", "file": "0002.npz", "sha256": sha["0002"], "verified": False},
             {"patient": "0003", "file": "0003.npz", "sha256": "0" * 64, "verified": True}]
    (tmp_path / "refetch_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert ar.refetch_flagged(str(tmp_path)) == {"0002", "0003"}






def _windows(rng, n, beta):
    d = _table(rng, n, beta)
    d["transitions_per_10min"] = rng.poisson(2, n).astype(float)
    d["alpha_env_logvar"] = np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n))
    d["coh"], d["coh_null_mean"], d["coh_null_p95"] = rng.uniform(0.2, 0.6, n), 0.3, 0.5
    d["patient"] = d["uid"]
    return d


def test_d1_tables_primary_bar_secondaries_and_coherence():
    w = _windows(np.random.default_rng(17), 300, 0.3)
    out = ar.d1_tables(w, label="test", bar=True)
    assert out["primary"]["marker"] == "a_hr" and out["bar"]["confirmed"] is True
    assert set(out["secondary_eeg_derived"]) == {"transitions_per_10min", "alpha_env_logvar"}
    assert out["secondary_eeg_derived"]["alpha_env_logvar"]["n"] < 300
    c = out["coherence"]
    assert c["n"] == 300 and c["share_above_own_null_p95"] == pytest.approx(float((w["coh"] > 0.5).mean()))
    assert "bar" not in ar.d1_tables(w, label="phase1", bar=False)


def test_d1_fresh_windows_check_the_ekg_against_the_list(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)
    import json
    lst = pd.DataFrame({"rank": [1, 2], "patient_id": ["Bonobo1", "Bonobo2"], "uid": ["Bonobo1_0", "Bonobo2_0"],
                        "window_start_s": [100, 200]})
    lst.to_csv(tmp_path / "list.csv", index=False)
    ekg = tmp_path / "ekg"
    ekg.mkdir()
    rows = []
    for u, s in zip(lst["uid"], lst["window_start_s"]):
        np.savez(ekg / f"{u}.npz", ekg=ecg(1800, rng), fs=np.array(FS), start_s=np.array(s), channel=np.array("EKG"),
                 uid=np.array(u))
        with np.load(ekg / f"{u}.npz") as z:
            h = __import__("hashlib").sha256(z["ekg"].tobytes()).hexdigest()
        rows.append({"rank": 1, "patient_id": u[:7], "uid": u, "start_s": s, "status": "ok", "channel": "EKG",
                     "fs": FS, "n_samples": 230400, "ekg_sha256": h})
    pd.DataFrame(rows).to_csv(ekg / "manifest.csv", index=False)
    monkeypatch.setattr(ar, "FRESH_EKG_MANIFEST_SHA256", ar.sha256_file(str(ekg / "manifest.csv")))
    monkeypatch.setattr(ar.ns, "fresh_items", lambda root, man: {u: {"uid": u} for u in lst["uid"]})
    monkeypatch.setattr(ar.en, "_load", lambda it: recording(x19(1800, rng), uid=it["uid"]))
    out = ar.d1_fresh_windows(str(tmp_path), str(tmp_path / "list.csv"), str(ekg))
    assert len(out) == 2 and out["a_hr"].notna().all() and set(out["uid"]) == set(lst["uid"])
    lst.loc[1, "window_start_s"] = 201
    lst.to_csv(tmp_path / "list.csv", index=False)
    with pytest.raises(RuntimeError, match="start_s"):
        ar.d1_fresh_windows(str(tmp_path), str(tmp_path / "list.csv"), str(ekg))
    monkeypatch.setattr(ar, "FRESH_EKG_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        ar.d1_fresh_windows(str(tmp_path), str(tmp_path / "list.csv"), str(ekg))


def test_d3_driver_reads_the_ecg_only_through_the_header_rule(monkeypatch):
    rng = np.random.default_rng(19)
    items = [{"uid": "Patient 1_VS", "path": "p1"}, {"uid": "Patient 2_MCS+", "path": "p2"}]
    recs = {it["uid"]: en.Recording(source="pdoc", uid=it["uid"], patient=it["uid"][:9], x=x19(2000, rng), fs=FS,
                                    names=list(NAMES), meta={"diagnosis": it["uid"].split("_")[1]}) for it in items}
    monkeypatch.setattr(ar.en, "_items", lambda src, split: items)
    monkeypatch.setattr(ar.en, "_load", lambda it: recs[it["uid"]])
    monkeypatch.setattr(ar, "edf_channel_names", lambda path: ["C3", "ECG"] if path == "p1" else ["C3"])
    monkeypatch.setattr(ar, "edf_signal", lambda path, name: (ecg(2000, rng), FS))
    out = ar.d3_windows()
    by = out.set_index("uid")
    assert by.loc["Patient 1_VS", "a_hr"] is not None and by.loc["Patient 2_MCS+", "a_hr_reason"] == "no_ecg"
    assert list(out["k"]) == [0, 1]




def test_phase1_mediation_keeps_rows_inside_the_heart_rate_window(monkeypatch):
    seen = {}
    monkeypatch.setattr(ar, "mediation", lambda d, nv, B, name: seen.setdefault("keys", sorted(d["key"])) and {})
    w = pd.DataFrame({"uid": ["A", "B"], "a_hr": [0.1, 0.2], "start_s": [100.0, 0.0]})
    rows = pd.DataFrame({"key": ["a1", "a2", "a3", "b1"], "uid": ["A", "A", "A", "B"],
                         "extent_t0": [150.0, 50.0, 1800.0, 10.0], "extent_t1": [900.0, 600.0, 1950.0, 1700.0],
                         "start_s": [130.0, 30.0, 1780.0, 0.0]})
    ver = pd.DataFrame({"key": ["a1", "a2", "a3", "b1"], "log_ver": [0.0, 0.1, 0.2, 0.3]})
    ar.d1_mediation(w, rows, ver, {}, B=1, within_window=True)
    assert seen["keys"] == ["a1", "b1"]
    seen.clear()
    ar.d1_mediation(w, rows, ver, {}, B=1)
    assert seen["keys"] == ["a1", "a2", "a3", "b1"]
