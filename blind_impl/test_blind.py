"""Tests for the blind re-implementations (dubois_blind, p300_blind).

Run from this folder with:  /usr/bin/python3 -m pytest -s
Observed noise call rates are printed (use -s) and stored in noise_rates.txt.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dubois_blind
import p300_blind

CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4",
        "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
HERE = os.path.dirname(os.path.abspath(__file__))


def _log_rate(name, rate, n):
    line = "%s noise call rate = %.3f over %d seeds" % (name, rate, n)
    print("\n" + line)
    with open(os.path.join(HERE, "noise_rates.txt"), "a") as fh:
        fh.write(line + "\n")


def make_mi_run(rng, effect_uv, fs=128, n_ch=19, ch_names=CH19, n_blocks=6,
                per_block=15, cue_gap=8.0, block_gap=20.0, t0=20.0):
    """One assessment run: 6 blocks x 15 trials, alternating left/right,
    beep every 8 s.  Imagery of 'left' adds a 10 Hz burst at C3-side
    channels, 'right' at C4-side channels (0.5-5.5 s after the cue)."""
    n_trials = n_blocks * per_block
    dur = t0 + n_trials * cue_gap + (n_blocks - 1) * block_gap + 20.0
    n = int(dur * fs)
    eeg = rng.standard_normal((n_ch, n)) * 5.0
    events = []
    t = t0
    tt = np.arange(n) / fs
    for b in range(n_blocks):
        label = "left" if b % 2 == 0 else "right"
        for _ in range(per_block):
            events.append((t, label, b))
            if effect_uv > 0:
                i0, i1 = int((t + 0.5) * fs), int((t + 5.5) * fs)
                chans = [ch_names.index("C3"), ch_names.index("Cz")] \
                    if label == "left" else \
                    [ch_names.index("C4"), ch_names.index("Cz")]
                ph = rng.uniform(0, 2 * np.pi)
                burst = effect_uv * np.sin(2 * np.pi * 10.0 * tt[i0:i1] + ph)
                for c in chans[:1]:
                    eeg[c, i0:i1] += burst
            t += cue_gap
        t += block_gap
    return eeg, fs, list(ch_names), events


def test_dubois_returns_contract():
    rng = np.random.default_rng(1)
    eeg, fs, chn, ev = make_mi_run(rng, 0.0)
    out = dubois_blind.assess(eeg, fs, chn, ev, seed=0)
    assert set(["call", "peak_da", "p_t", "notes"]) <= set(out)
    assert isinstance(out["call"], bool)
    assert 0.0 <= out["peak_da"] <= 1.0
    assert 0.0 <= out["p_t"] <= 1.0
    assert isinstance(out["notes"], str)


def test_dubois_detects_strong_effect():
    rng = np.random.default_rng(2)
    eeg, fs, chn, ev = make_mi_run(rng, 15.0)
    t0 = time.time()
    out = dubois_blind.assess(eeg, fs, chn, ev, seed=0)
    print("\nstrong effect:", out["call"], out["peak_da"], out["p_t"],
          "%.1fs" % (time.time() - t0))
    assert out["call"] is True
    assert out["peak_da"] > 0.70
    assert out["p_t"] < 0.05


def test_dubois_runtime_30min_19ch_128hz():
    rng = np.random.default_rng(3)
    eeg, fs, chn, ev = make_mi_run(rng, 5.0)
    pad = 30 * 60 * fs - eeg.shape[1]
    if pad > 0:
        eeg = np.concatenate([eeg, rng.standard_normal((19, pad)) * 5.0], 1)
    t0 = time.time()
    dubois_blind.assess(eeg, fs, chn, ev, seed=0)
    dt = time.time() - t0
    print("\n30-min runtime %.1fs" % dt)
    assert dt < 60.0


def test_dubois_noise_call_rate():
    n_seeds = 30
    calls, peaks = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        eeg, fs, chn, ev = make_mi_run(rng, 0.0)
        out = dubois_blind.assess(eeg, fs, chn, ev, seed=s)
        calls.append(out["call"])
        peaks.append(out["peak_da"])
    rate = float(np.mean(calls))
    _log_rate("dubois_blind", rate, n_seeds)
    print("mean noise peak DA = %.3f" % np.mean(peaks))
    assert 0.5 < np.mean(peaks) < 1.0


def test_dubois_edge_cases():
    rng = np.random.default_rng(4)
    eeg, fs, chn, ev = make_mi_run(rng, 0.0)
    with pytest.raises(ValueError):
        dubois_blind.assess(eeg[:-1], fs, chn, ev)
    with pytest.raises(ValueError):
        dubois_blind.assess(eeg[0], fs, chn, ev)
    out = dubois_blind.assess(eeg, fs, chn, ev[:4])
    assert out["call"] is False and "trial" in out["notes"].lower()
    out = dubois_blind.assess(eeg, fs, chn, [e for e in ev if e[1] == "left"])
    assert out["call"] is False
    out = dubois_blind.assess(eeg, fs, chn, [])
    assert out["call"] is False
    bad = list(ev) + [(1e6, "left", 9), (-5.0, "right", 9)]
    out = dubois_blind.assess(eeg, fs, chn, bad, seed=0)
    assert isinstance(out["call"], bool)
    e2 = eeg.copy()
    e2[3, 100] = np.nan
    out = dubois_blind.assess(e2, fs, chn, ev)
    assert out["call"] is False and "finite" in out["notes"].lower()
    e3 = eeg.copy()
    e3[5] = 0.0
    out = dubois_blind.assess(e3, fs, chn, ev, seed=0)
    assert isinstance(out["call"], bool)
    sub = ["Fz", "C3", "Cz", "C4", "Pz"]
    idx = [chn.index(c) for c in sub]
    out = dubois_blind.assess(eeg[idx], fs, sub, ev, seed=0)
    assert isinstance(out["call"], bool)
    out = dubois_blind.assess(np.round(eeg).astype(np.int32), fs, chn, ev, seed=0)
    assert isinstance(out["call"], bool)


def test_dubois_deterministic_given_seed():
    rng = np.random.default_rng(5)
    eeg, fs, chn, ev = make_mi_run(rng, 3.0)
    a = dubois_blind.assess(eeg, fs, chn, ev, seed=7)
    b = dubois_blind.assess(eeg, fs, chn, ev, seed=7)
    assert a["peak_da"] == b["peak_da"] and a["p_t"] == b["p_t"]


def make_vt3_run(rng, effect_uv, target="left", fs=128, ch_names=CH19,
                 n_groups=30, soa=0.2, group_gap=3.0, t0=10.0, sd=8.0):
    """30 groups of 8 stimuli: one left, one right, six distractors,
    100 ms on / 100 ms off.  The attended wrist gets a P300-like
    deflection (peak ~ 350 ms, Cz/C3/C4/Pz)."""
    n_ev = n_groups * 8
    dur = t0 + n_groups * (8 * soa + group_gap) + 10.0
    n = int(dur * fs)
    eeg = rng.standard_normal((len(ch_names), n)) * sd
    drift = np.cumsum(rng.standard_normal(n)) * 0.05
    eeg += drift[None, :]
    events = []
    t = t0
    sites_group = ["left", "right"] + ["distractor"] * 6
    chans = [ch_names.index(c) for c in ("Cz", "C3", "C4", "Pz")
             if c in ch_names]
    for g in range(n_groups):
        order = rng.permutation(8)
        for k in range(8):
            site = sites_group[order[k]]
            events.append((t, site))
            if effect_uv > 0 and site == target:
                i = int(round(t * fs))
                w = np.arange(int(0.7 * fs)) / fs - 0.1
                erp = effect_uv * np.exp(-0.5 * ((w - 0.35) / 0.06) ** 2)
                for c in chans:
                    eeg[c, i - int(0.1 * fs): i - int(0.1 * fs) + len(erp)] += erp
            t += soa
        t += group_gap
    return eeg, fs, list(ch_names), events


def test_p300_returns_contract():
    rng = np.random.default_rng(1)
    eeg, fs, chn, ev = make_vt3_run(rng, 0.0)
    out = p300_blind.assess(eeg, fs, chn, ev, target="left", seed=0)
    assert set(["call", "accuracy", "notes"]) <= set(out)
    assert isinstance(out["call"], bool)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert isinstance(out["notes"], str)


@pytest.mark.parametrize("target", ["left", "right"])
def test_p300_detects_strong_effect(target):
    rng = np.random.default_rng(2)
    eeg, fs, chn, ev = make_vt3_run(rng, 20.0, target=target)
    out = p300_blind.assess(eeg, fs, chn, ev, target=target, seed=0)
    print("\nstrong effect:", target, out["call"], out["accuracy"])
    assert out["call"] is True
    assert out["accuracy"] > 0.5


def test_p300_wrong_target_is_not_detected():
    rng = np.random.default_rng(2)
    eeg, fs, chn, ev = make_vt3_run(rng, 20.0, target="left")
    out = p300_blind.assess(eeg, fs, chn, ev, target="right", seed=0)
    assert out["accuracy"] < 0.5


def test_p300_noise_call_rate():
    n_seeds = 100
    calls, accs = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(2000 + s)
        eeg, fs, chn, ev = make_vt3_run(rng, 0.0)
        out = p300_blind.assess(eeg, fs, chn, ev, target="left", seed=s)
        calls.append(out["call"])
        accs.append(out["accuracy"])
    rate = float(np.mean(calls))
    _log_rate("p300_blind", rate, n_seeds)
    print("mean noise accuracy = %.3f" % np.mean(accs))
    assert 0.05 < np.mean(accs) < 0.25


def test_p300_edge_cases():
    rng = np.random.default_rng(4)
    eeg, fs, chn, ev = make_vt3_run(rng, 0.0)
    with pytest.raises(ValueError):
        p300_blind.assess(eeg[:-1], fs, chn, ev, target="left")
    with pytest.raises(ValueError):
        p300_blind.assess(eeg, fs, chn, ev, target="foot")
    out = p300_blind.assess(eeg, fs, chn, ev[:10], target="left")
    assert out["call"] is False
    out = p300_blind.assess(eeg, fs, chn, [], target="left")
    assert out["call"] is False
    out = p300_blind.assess(eeg, fs, chn, [e for e in ev if e[1] != "left"],
                            target="left")
    assert out["call"] is False
    bad = list(ev) + [(1e6, "left"), (-3.0, "left")]
    out = p300_blind.assess(eeg, fs, chn, bad, target="left", seed=0)
    assert isinstance(out["call"], bool)
    e2 = eeg.copy()
    e2[2, 50] = np.inf
    out = p300_blind.assess(e2, fs, chn, ev, target="left")
    assert out["call"] is False and "finite" in out["notes"].lower()
    sub = ["Fz", "Cz", "Pz"]
    idx = [chn.index(c) for c in sub]
    out = p300_blind.assess(eeg[idx], fs, sub, ev, target="left", seed=0)
    assert isinstance(out["call"], bool)
    with pytest.raises(ValueError):
        p300_blind.assess(eeg[:2], fs, ["Fp1", "Fp2"], ev, target="left")
    big = eeg * 1000.0
    out = p300_blind.assess(big, fs, chn, ev, target="left", seed=0)
    assert out["call"] is False


def test_p300_deterministic_given_seed():
    rng = np.random.default_rng(5)
    eeg, fs, chn, ev = make_vt3_run(rng, 5.0)
    a = p300_blind.assess(eeg, fs, chn, ev, target="left", seed=7)
    b = p300_blind.assess(eeg, fs, chn, ev, target="left", seed=7)
    assert a["accuracy"] == b["accuracy"]
