"""Tests for 140_cardiac_control.py: synthetic EEG, ECG and tables only."""
import importlib

import numpy as np
import pandas as pd
import pytest

cc = importlib.import_module("140_cardiac_control")
ex, rn, en = cc.ex, cc.ex.rn, cc.ex.en
FS = 128.0
NAMES = list(en.NAMES19)


def _qrs(n_t=27):
    t = np.arange(n_t) - n_t // 2
    return np.exp(-0.5 * (t / 1.5) ** 2) - 0.3 * np.exp(-0.5 * ((t - 4) / 2.0) ** 2)


def _eeg_with_artifact(seconds, amp, rng, n_ch=6, hr=70.0):
    n = int(seconds * FS)
    x = 20.0 * rng.normal(size=(n_ch, n))
    beats = np.arange(int(FS), n - int(FS), int(round(60.0 / hr * FS)))
    k = _qrs()
    for b in beats:
        x[:, b - 13:b + 14] += amp * k
    return x, beats


def test_contamination_is_about_zero_without_artifact_and_grows_with_it():
    rng = np.random.default_rng(1)
    x0, b = _eeg_with_artifact(300, 0.0, rng)
    c0 = cc.cardiac_contamination(x0, FS, b, np.random.default_rng(2))
    assert abs(c0["contam_ratio"]) < 0.15 and c0["contam_amp"] < 0.03
    lo = cc.cardiac_contamination(_eeg_with_artifact(300, 5.0, rng)[0], FS, b, np.random.default_rng(2))
    hi = cc.cardiac_contamination(_eeg_with_artifact(300, 20.0, rng)[0], FS, b, np.random.default_rng(2))
    assert hi["contam_ratio"] > lo["contam_ratio"] > 0.3 and hi["contam_amp"] > lo["contam_amp"] > 0.05


def test_debiased_contamination_does_not_grow_with_the_number_of_beats():
    rng = np.random.default_rng(3)
    xs, bs = _eeg_with_artifact(120, 8.0, rng)
    xl, bl = _eeg_with_artifact(900, 8.0, rng)
    s = cc.cardiac_contamination(xs, FS, bs, np.random.default_rng(4))
    l = cc.cardiac_contamination(xl, FS, bl, np.random.default_rng(4))
    assert l["contam_amp"] == pytest.approx(s["contam_amp"], rel=0.3)
    assert l["contam_ratio"] > s["contam_ratio"]


def test_count_contrast_sees_more_beats_in_one_label_class():
    onsets = np.arange(0, 600, 10.0)
    labels = np.tile([1, 0], 30)
    rng = np.random.default_rng(5)
    beats = []
    for a, l in zip(onsets, labels):
        rate = 1.4 if l == 1 else 1.1
        beats += list(np.sort(a + rng.uniform(0, 10, int(round(10 * rate)) + rng.integers(-1, 2))))
    w = cc.count_contrast(np.array(beats), onsets, labels, [((1,), (0,))])
    assert w["abs_t"] > 5
    flat = np.arange(0.4, 600, 0.8) + rng.normal(0, 0.02, 750)
    assert cc.count_contrast(flat, onsets, labels, [((1,), (0,))])["abs_t"] < 3


def _rows(rng, n_rec=300, b=0.8):
    out = []
    for i in range(n_rec):
        for pipe in ("cornell", "edlow17"):
            for v in range(2):
                a, p = abs(rng.normal()), abs(rng.normal())
                out.append({"uid": f"U{i}", "patient": f"U{i}", "pipe": pipe, "variant": f"{pipe}{v}",
                            "abs_t_aligned": a, "abs_t_placebo": p,
                            "call": bool(rng.random() < 1 / (1 + np.exp(2.0 - b * (a - 1))))})
    return pd.DataFrame(out)


def test_shared_draw_contrasts():
    d = _rows(np.random.default_rng(6))
    nv = {"cornell": 2, "edlow17": 2}
    fits = cc.shared_draw_fits({"m": (d, nv, ("abs_t_aligned", "abs_t_placebo"), "call", None),
                                "same": (d, nv, ("abs_t_aligned", "abs_t_placebo"), "call", None)}, B=100, seed=[1, 2])
    c = cc.contrast(fits, ("m", "z_abs_t_aligned"), ("m", "z_abs_t_placebo"))
    assert c["diff"] > 0.3 and c["ci95"][0] > 0
    z = cc.contrast(fits, ("m", "z_abs_t_aligned"), ("same", "z_abs_t_aligned"))
    assert z["diff"] == 0 and z["ci95"] == [0.0, 0.0]


def test_shared_draw_point_estimate_equals_the_sign_model():
    d = _rows(np.random.default_rng(7))
    nv = {"cornell": 2, "edlow17": 2}
    fits = cc.shared_draw_fits({"m": (d, nv, ("abs_t_aligned",), "call", None)}, B=20, seed=[1, 3])
    ref = ex.sign_model(d, nv, B=20, name="ref", cols=("abs_t_aligned",))
    assert fits["m"]["beta"]["z_abs_t_aligned"] == pytest.approx(ref["coef"]["z_abs_t_aligned"]["beta"], abs=1e-9)


def test_recording_cardiac_sees_an_artifact_injected_into_the_modules_channels():
    uid = "Bonobo00042_0"
    mod = rn.load_pipe("cornell")
    v = "jackknife_by_keep40_r4"
    spec = rn.variant_specs(mod, "bdsp")[v]
    start = 60.0
    ov = rn._sched(mod, mod.schedule, spec, start, rn._rng("cornell", uid, 0, 1))
    t0, t1 = mod.extent(ov)
    dur = 2600.0
    n = int(dur * FS)
    rng = np.random.default_rng(8)
    t = np.arange(n) / FS
    beats = np.arange(int(FS), n - int(FS), int(round(0.85 * FS)))
    e = 0.03 * rng.normal(size=n)
    k = _qrs()
    for b in beats:
        e[b - 13:b + 14] += k
    out = {}
    topo = np.random.default_rng(9).normal(0, 1, 19)
    for amp in (0.0, 15.0):
        x = (20 * rng.normal(size=(19, n))).astype(np.float32)
        for b in beats:
            x[:, b - 13:b + 14] += (amp * k * topo[:, None]).astype(np.float32)
        rec = en.Recording(source="bdsp", uid=uid, patient="P", x=x, fs=FS, names=NAMES, meta={})
        row = {"key": f"{uid}|{v}", "uid": uid, "patient": "P", "pipe": "cornell", "variant": v, "j": 0,
               "start_s": start, "extent_t0": t0, "extent_t1": t1, "n_onsets": len(ov.onsets_s), "rand_variant": None}
        out[amp] = cc.recording_cardiac(pd.DataFrame([row]), rec, e, FS)[0]
    assert out[15.0]["contam_ratio"] > out[0.0]["contam_ratio"] + 0.5
    assert abs(out[0.0]["contam_ratio"]) < 0.2 and out[0.0]["count_abs_t"] is not None


def test_an_extent_past_the_recording_end_is_clipped():
    x = np.arange(2 * 1000, dtype=float).reshape(2, 1000)
    seg, i0 = cc.eeg_span(x, 100.0, 2.96, 10.002)
    assert i0 == 296 and seg.shape == (2, 704) and seg[0, 0] == 296
    seg, i0 = cc.eeg_span(x, 100.0, -0.5, 1.0)
    assert i0 == 0 and seg.shape == (2, 100)
