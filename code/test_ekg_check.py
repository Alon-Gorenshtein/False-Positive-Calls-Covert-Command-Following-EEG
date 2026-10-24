"""Tests for 132_ekg_check.py: synthetic ECG and EEG only."""
import importlib
import math

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

kt = importlib.import_module("132_ekg_check")



def ecg(seconds, rng, fs=200.0, hr0=70.0, amp=0.0, f_mod=0.03, jitter=0.5, noise=0.03, invert=False, drop=0.0):
    """ECG with heart rate hr0 + amp sin(2 pi f_mod t) + white jitter (bpm): Gaussian Q, R, S and T waves, baseline
    wander and sensor noise."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    hr = hr0 + amp * np.sin(2 * np.pi * f_mod * t + rng.uniform(0, 2 * np.pi))
    beats, phase, k = [], 0.0, 0
    for i in range(n):
        phase += hr[i] / 60.0 / fs
        if phase >= 1.0:
            phase -= 1.0
            beats.append(t[i] + rng.normal(0, jitter / 1000.0))
    beats = np.array(beats)
    if drop:
        beats = beats[rng.random(len(beats)) >= drop]
    x = np.zeros(n)
    for b in beats:
        for dt, w, a in ((-0.03, 0.008, -0.1), (0.0, 0.01, 1.0), (0.03, 0.008, -0.2), (0.25, 0.04, 0.3)):
            c = b + dt
            i0, i1 = max(0, int((c - 5 * w) * fs)), min(n, int((c + 5 * w) * fs) + 1)
            x[i0:i1] += a * np.exp(-0.5 * ((t[i0:i1] - c) / w) ** 2)
    x += 0.2 * np.sin(2 * np.pi * 0.3 * t) + noise * rng.normal(size=n)
    return (-x if invert else x), np.array(beats)


def envelope(n, fs, log_sd, tau_s, rng):
    if log_sd == 0:
        return np.ones(n)
    a = math.exp(-1.0 / (tau_s * fs))
    g = lfilter([math.sqrt(1.0 - a * a)], [1.0, -a], rng.normal(size=n), zi=[a * rng.normal()])[0]
    return np.exp(log_sd * g - log_sd ** 2 / 2.0)


def pink(n, rng):
    f = np.fft.rfftfreq(n)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** -0.5
    x = np.fft.irfft(amp * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))), n=n)
    return x / x.std()


def x19(seconds, rng, log_sd=0.0, tau_s=30.0, fs=128.0):
    """(19, n) EEG-like noise (shared pink sources plus an 8-12 Hz rhythm) times one log-normal envelope."""
    n = int(seconds * fs)
    src = np.array([pink(n, rng) for _ in range(4)])
    alpha = kt.mc.bandpass(rng.normal(size=n), fs, 8.0, 12.0)
    x = rng.normal(size=(19, 4)) @ src + 0.5 * alpha / alpha.std() + 0.2 * rng.normal(size=(19, n))
    return (20.0 * x * envelope(n, fs, log_sd, tau_s, rng)[None]).astype(np.float32)


NAMES = list(kt.en.NAMES19)



def test_r_peaks_find_every_beat_in_either_polarity():
    rng = np.random.default_rng(1)
    for inv in (False, True):
        x, beats = ecg(120, rng, invert=inv)
        p = kt.r_peaks(x, 200.0) / 200.0
        assert abs(len(p) - len(beats)) <= 1
        d = np.abs(p[:, None] - beats[None]).min(1)
        assert np.median(d) < 0.03


@pytest.mark.parametrize("hr0", [40.0, 90.0, 150.0, 180.0])
def test_r_peaks_hold_across_heart_rates_at_128_hz(hr0):
    rng = np.random.default_rng(int(hr0))
    x, beats = ecg(300, rng, fs=128.0, hr0=hr0, amp=3.0)
    p = kt.r_peaks(x, 128.0) / 128.0
    hit = (np.abs(p[:, None] - beats[None]).min(1) < 0.05).sum()
    assert len(beats) - hit <= 0.01 * len(beats) and len(p) - hit <= 0.01 * len(beats)


def test_a_hr_recovers_a_known_heart_rate_modulation():
    rng = np.random.default_rng(2)
    for amp in (3.0, 8.0):
        x, _ = ecg(900, rng, amp=amp, f_mod=0.03)
        r = kt.a_hr(x, 200.0)
        assert r["reason"] is None and r["rr_ok_share"] > 0.99
        assert math.exp(r["a_hr"]) == pytest.approx(amp / math.sqrt(2), rel=0.15)


def test_a_hr_rises_with_the_modulation_and_ignores_out_of_band_ones():
    rng = np.random.default_rng(3)
    small = kt.a_hr(ecg(900, rng, amp=2.0)[0], 200.0)["a_hr"]
    big = kt.a_hr(ecg(900, rng, amp=8.0)[0], 200.0)["a_hr"]
    fast = kt.a_hr(ecg(900, rng, amp=8.0, f_mod=0.25)[0], 200.0)["a_hr"]
    assert big > small + 1.0 and fast < big - 1.0


def test_a_hr_is_na_when_rr_intervals_are_implausible():
    rng = np.random.default_rng(4)
    slow = kt.a_hr(ecg(600, rng, hr0=25.0)[0], 200.0)
    assert slow["a_hr"] is None and slow["reason"] == "rr_implausible" and slow["rr_ok_share"] < kt.RR_MIN_SHARE
    lost = kt.a_hr(ecg(600, rng, hr0=40.0, drop=0.2)[0], 200.0)
    assert lost["a_hr"] is None and lost["reason"] == "rr_implausible" and 0.5 < lost["rr_ok_share"] < 0.9
    flat = kt.a_hr(np.zeros(200 * 600), 200.0)
    assert flat["a_hr"] is None and flat["reason"] == "rr_implausible" and flat["n_beats"] == 0


def test_a_hr_at_the_native_128_hz():
    rng = np.random.default_rng(11)
    x, beats = ecg(900, rng, fs=128.0, amp=6.0)
    r = kt.a_hr(x, 128.0)
    assert r["reason"] is None and abs(r["n_beats"] - len(beats)) <= 1
    assert math.exp(r["a_hr"]) == pytest.approx(6.0 / math.sqrt(2), rel=0.15)


def test_hr_series_is_4_hz_and_drops_implausible_intervals():
    peaks = np.round(np.array([0.0, 1.0, 2.0, 4.5, 5.5, 6.5]) * 100).astype(int)
    t, hr, share = kt.hr_series(peaks, 100.0)
    assert np.allclose(np.diff(t), 0.25) and t[0] == 1.0 and t[-1] == 6.5
    assert np.allclose(hr, 60.0) and share == pytest.approx(4 / 5)



def test_ver_ref_about_zero_when_stationary_and_positive_under_an_envelope():
    rng = np.random.default_rng(5)
    st = [kt.ver_ref(x19(600, rng), NAMES, 128.0, f"S{i}")["log_ver_ref"] for i in range(4)]
    mo = [kt.ver_ref(x19(600, rng, log_sd=0.5), NAMES, 128.0, f"M{i}")["log_ver_ref"] for i in range(4)]
    assert abs(np.mean(st)) < 0.15 and np.mean(mo) > 0.5 and min(mo) > max(st)


def test_ver_ref_uses_c3_c4_2s_epochs_and_ten_surrogates_seeded_by_uid(monkeypatch):
    rng = np.random.default_rng(6)
    x = x19(300, rng, log_sd=0.5)
    seen = {}
    real = kt.ns.surrogates

    def spy(seg, uid, i0, i1, n=10):
        seen.update(shape=seg.shape, uid=uid, n=n, span=(i0, i1))
        return real(seg, uid, i0, i1, n)
    monkeypatch.setattr(kt.ns, "surrogates", spy)
    r = kt.ver_ref(x, NAMES, 128.0, "U1")
    assert seen == {"shape": (2, 300 * 128), "uid": "U1", "n": 10, "span": (0, 300 * 128)}
    assert r["n_epochs"] == 150 and r["n_ch_used"] == 2
    assert kt.ver_ref(x, NAMES, 128.0, "U1")["log_ver_ref"] == r["log_ver_ref"]
    assert kt.ver_ref(x, NAMES, 128.0, "U2")["log_ver_ref"] != r["log_ver_ref"]
    x2 = x.copy()
    x2[[i for i, c in enumerate(NAMES) if c not in ("C3", "C4")]] = 0.0
    assert kt.ver_ref(x2, NAMES, 128.0, "U1")["log_ver_ref"] == r["log_ver_ref"]


def test_ver_ref_feature_is_8_30_hz_log_power():
    rng = np.random.default_rng(7)
    x = x19(120, rng)
    F = kt.c3c4_log_power(x[[NAMES.index("C3"), NAMES.index("C4")]].astype(float), 128.0)
    xf = kt.mc.bandpass(x[NAMES.index("C4")].astype(float), 128.0, 8.0, 30.0)
    assert F.shape == (2, 60)
    assert F[1, 5] == pytest.approx(np.log(np.mean(xf[5 * 256:6 * 256] ** 2) + kt.LOG_FLOOR))



def _cohort(rng, n, coupled, stationary=False):
    a_hr, ver = [], []
    for i in range(n):
        arousal = rng.uniform(0, 0.75)
        eeg_sd = 0.0 if stationary else (arousal if coupled else rng.uniform(0, 0.75))
        amp = 0.0 if stationary else 1.0 + 8.0 * arousal
        ver.append(kt.ver_ref(x19(300, rng, log_sd=eeg_sd), NAMES, 128.0, f"C{i}")["log_ver_ref"])
        a_hr.append(kt.a_hr(ecg(600, rng, amp=amp, f_mod=rng.uniform(0.02, 0.08))[0], 200.0)["a_hr"])
    return np.array(a_hr), np.array(ver)


def test_a_planted_coupling_gives_a_positive_rho():
    a, v = _cohort(np.random.default_rng(8), 40, coupled=True)
    out = kt.spearman_ci(a, v)
    assert out["rho"] > 0.5 and out["ci95"][0] > 0.1 and out["n"] == 40
    assert kt.verdict(out)["verdict"] == "GO"


def test_a_stationary_pair_gives_rho_about_zero():
    a, v = _cohort(np.random.default_rng(9), 40, coupled=False, stationary=True)
    out = kt.spearman_ci(a, v)
    assert abs(out["rho"]) < 0.35 and out["ci95"][0] < 0 < out["ci95"][1]
    assert kt.verdict(out)["verdict"] == "NOGO"


def test_spearman_ci_bootstrap_is_seeded_and_drops_na():
    rng = np.random.default_rng(10)
    a = rng.normal(size=50)
    b = a + rng.normal(size=50)
    a[3] = np.nan
    o1, o2 = kt.spearman_ci(a, b), kt.spearman_ci(a, b)
    assert o1 == o2 and o1["n"] == 49 and o1["B"] == 2000 and o1["seed"] == 20261016
    assert o1["ci95"][0] < o1["rho"] < o1["ci95"][1]


def test_verdict_rule():
    v = lambda r, lo, hi: kt.verdict({"rho": r, "ci95": [lo, hi]})["verdict"]
    assert v(0.12, 0.01, 0.3) == "GO"
    assert v(0.09, 0.01, 0.2) == "NOGO"
    assert v(0.30, -0.01, 0.5) == "NOGO"
    assert v(-0.30, -0.5, -0.1) == "NOGO"



def _pool(n=8, fs=128.0, seconds=1900, no_window=(), seed=12):
    """Fake cached design npz: x19 noise; a recording in no_window has a 200-s zero stretch mid-way."""
    uids = [f"Bonobo{i:05d}_0" for i in range(n)]

    def load(uid):
        g = np.random.default_rng([seed, int(uid[6:11])])
        x = (20.0 * g.normal(size=(19, int(seconds * fs)))).astype(np.float32)
        if uid in no_window:
            x[:, int(900 * fs):int(1100 * fs)] = 0.0
        return {"x19": x, "fs": fs, "names19": np.array(NAMES), "all_names": np.array(NAMES + ["EKG"])}
    return uids, load


def test_selection_follows_the_clarified_rule():
    uids, load = _pool(no_window=("Bonobo00003_0",))
    rows, fail = kt.select_rows(sorted(uids, reverse=True), load, n=5)
    order = np.random.default_rng([20261016, 3]).permutation(sorted(uids))
    taken = [u for u in order if u != "Bonobo00003_0"][:5]
    assert [r["uid"] for r in rows] == taken and fail.get("no_30min_window", 0) == int("Bonobo00003_0" in order[:6])
    for r in rows:
        el = kt.bn.eligible_starts(load(r["uid"])["x19"], 128.0, run_s=1800)
        assert r["start_s"] == int(np.random.default_rng([20261016, 3, r["position"]]).choice(el))
        assert list(order).index(r["uid"]) == r["position"]


def test_selection_skips_a_recording_without_one_ekg_channel():
    uids, load = _pool(n=4)
    no_ekg = lambda u: dict(load(u), all_names=np.array(NAMES)) if u == uids[1] else load(u)
    rows, fail = kt.select_rows(uids, no_ekg, n=3)
    assert uids[1] not in [r["uid"] for r in rows] and fail["no_ekg"] == 1


def test_manifest_must_equal_the_recomputed_selection():
    uids, load = _pool(n=6)
    rows, _ = kt.select_rows(uids, load, n=4)
    man = pd.DataFrame(rows)
    kt.check_manifest(man, rows)
    for col, val in (("start_s", man["start_s"].iloc[1] + 1), ("uid", "Bonobo99999_0"), ("position", 99)):
        bad = man.copy()
        bad.loc[1, col] = val
        with pytest.raises(RuntimeError, match="manifest"):
            kt.check_manifest(bad, rows)
    with pytest.raises(RuntimeError, match="manifest"):
        kt.check_manifest(man.iloc[:3], rows)


def test_recording_measures_cut_the_same_window(tmp_path):
    rng = np.random.default_rng(13)
    fs = 128.0
    x = x19(1900, rng, log_sd=0.5)
    e, _ = ecg(1900, rng, fs=fs, amp=5.0)
    start = 40
    row = {"position": 0, "uid": "Bonobo00001_0", "start_s": start, "channel": "EKG", "fs": fs}
    ekg = {"ekg": e[int(start * fs):int((start + 1800) * fs)].astype(np.float32), "fs": fs, "start_s": start,
           "channel": "EKG", "uid": "Bonobo00001_0"}
    sig = {"x19": x, "fs": fs, "names19": np.array(NAMES)}
    r = kt.recording_measures(row, ekg, sig)
    w = x[:, int(start * fs):int((start + 1800) * fs)]
    assert r["log_ver_ref"] == kt.ver_ref(w, NAMES, fs, "Bonobo00001_0")["log_ver_ref"]
    assert r["a_hr"] == kt.a_hr(ekg["ekg"], fs)["a_hr"] and r["reason"] is None
    for k, v in (("start_s", start + 1), ("uid", "X"), ("fs", 256.0), ("channel", "ECG")):
        with pytest.raises(RuntimeError, match="manifest"):
            kt.recording_measures(row, dict(ekg, **{k: v}), sig)
    short = dict(ekg, ekg=ekg["ekg"][:-10])
    with pytest.raises(RuntimeError, match="1800"):
        kt.recording_measures(row, short, sig)



@pytest.mark.parametrize("kind", ["white", "pink"])
def test_noise_is_na_by_the_beat_template_guard(kind):
    rng = np.random.default_rng(14)
    x = rng.normal(size=128 * 1800) if kind == "white" else pink(128 * 1800, rng)
    r = kt.a_hr(x, 128.0)
    assert r["a_hr"] is None and r["reason"] == "low_template_correlation" and r["template_corr"] < kt.TEMPLATE_MIN


def test_a_clean_ecg_passes_the_guard():
    rng = np.random.default_rng(15)
    r = kt.a_hr(ecg(900, rng, fs=128.0, amp=4.0)[0], 128.0)
    assert r["reason"] is None and r["template_corr"] > 0.8 and 60 < r["median_hr"] < 80


def test_template_correlation_uses_100_ms_around_each_peak():
    fs = 100.0
    x = np.zeros(1000)
    peaks = np.array([5, 200, 400, 600, 995])
    for p in peaks[1:4]:
        x[p - 10:p + 11] = np.hanning(21)
    c, n = kt.template_corr(x, peaks, fs)
    assert n == 3 and c == pytest.approx(1.0)
    x[600 - 10:600 + 11] = np.random.default_rng(16).normal(size=21)
    assert kt.template_corr(x, peaks, fs)[0] < 1.0


def test_quality_rule_order_and_ranges():
    q = kt.quality_reason
    assert q(0.95, 0.9, 70.0) is None
    assert q(0.80, 0.9, 70.0) == "rr_implausible"
    assert q(0.95, 0.59, 70.0) == "low_template_correlation"
    assert q(0.95, 0.9, 29.0) == "median_hr_out_of_range" and q(0.95, 0.9, 201.0) == "median_hr_out_of_range"
    assert q(0.95, 0.9, 30.0) is None and q(0.95, 0.9, 200.0) is None and q(0.95, 0.6, 70.0) is None


def test_a_70_bpm_ecg_with_moderate_noise_passes_the_guard():
    rng = np.random.default_rng(17)
    r = kt.a_hr(ecg(900, rng, fs=128.0, hr0=70.0, amp=4.0, noise=0.1)[0], 128.0)
    assert r["reason"] is None and r["template_corr"] >= kt.TEMPLATE_MIN and 60 < r["median_hr"] < 80
