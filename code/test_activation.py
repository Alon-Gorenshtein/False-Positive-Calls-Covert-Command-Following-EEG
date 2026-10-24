import importlib
import zlib

import numpy as np
import pandas as pd
import pytest

act = importlib.import_module("121_activation")
FS = 128


def _edf(path, signals, annotations=None, rec_s=1, junk_identity=True):
    """Minimal EDF(+) writer: signals = {label: int16 array}, one data record per rec_s seconds; annotations =
    [(onset_s, duration_s or None, text)] go into an 'EDF Annotations' signal (EDF+C). The identity and date fields
    hold non-ASCII junk so a reader that decodes them would fail."""
    n_rec = len(next(iter(signals.values()))) // (FS * rec_s)
    labels = list(signals) + (["EDF Annotations"] if annotations is not None else [])
    spr = [FS * rec_s] * len(signals) + ([60] if annotations is not None else [])
    ns = len(labels)

    def f(v, n):
        return str(v).encode("ascii").ljust(n)[:n]

    ident = (bytes(range(128, 208)) if junk_identity else b" " * 80)
    head = f("0", 8) + ident + ident + bytes([200] * 16) + f(256 * (ns + 1), 8)
    head += f("EDF+C" if annotations is not None else "", 44) + f(n_rec, 8) + f(rec_s, 8) + f(ns, 4)
    head += b"".join(f(lab, 16) for lab in labels) + b" " * 80 * ns
    head += b"".join(f("uV" if lab != "EDF Annotations" else "", 8) for lab in labels)
    head += b"".join(f(-3276.8, 8) for _ in labels) + b"".join(f(3276.7, 8) for _ in labels)
    head += b"".join(f(-32768, 8) for _ in labels) + b"".join(f(32767, 8) for _ in labels)
    head += b" " * 80 * ns + b"".join(f(s, 8) for s in spr) + b" " * 32 * ns
    recs = []
    for r in range(n_rec):
        body = b"".join(np.asarray(sig[r * FS * rec_s:(r + 1) * FS * rec_s], "<i2").tobytes()
                        for sig in signals.values())
        if annotations is not None:
            tal = f"+{r * rec_s}\x14\x14\x00".encode()
            for on, dur, text in annotations:
                if r * rec_s <= on < (r + 1) * rec_s:
                    d = f"\x15{dur}" if dur is not None else ""
                    tal += f"+{on}{d}\x14{text}\x14\x00".encode()
            body += tal.ljust(2 * 60, b"\x00")
        recs.append(body)
    path.write_bytes(head + b"".join(recs))
    return path


def test_edf_annotations_reads_tals_without_the_identity_fields(tmp_path):
    sig = {"F3": np.zeros(FS * 10, np.int16), "O1": np.zeros(FS * 10, np.int16)}
    p = _edf(tmp_path / "a.edf", sig, [(2.5, 10, "Photic 5 Hz"), (7, None, "Eyes open")])
    a = act.edf_annotations(str(p))
    assert a.n_annotation_signals == 1 and a.edf_plus
    assert a.events == [(2.5, 10.0, "Photic 5 Hz"), (7.0, None, "Eyes open")]
    plain = act.edf_annotations(str(_edf(tmp_path / "b.edf", sig)))
    assert plain.n_annotation_signals == 0 and plain.events == [] and not plain.edf_plus


def test_annotation_intervals_flag_every_event_with_a_category():
    ev = [(2.5, 10.0, "Photic 5 Hz"), (7.0, None, "Eyes open"), (40.0, 5.0, "seizure onset"), (50.0, 1.0, "xyz")]
    iv = act.annotation_intervals(ev, dur_s=100)
    assert list(iv.kind) == ["photic", "eyes", "clinical_event", "other"]
    assert list(iv.start_s) == [2.5, 7.0, 40.0, 50.0] and list(iv.end_s) == [12.5, 7.0, 45.0, 51.0]


def _pink(n, rng, sd=10.0):
    f = np.fft.rfftfreq(n, 1 / FS)
    s = np.zeros_like(f)
    s[1:] = 1 / np.sqrt(f[1:])
    x = np.fft.irfft(s * (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))), n)
    return sd * x / x.std()


def _occipital(n, rng, alpha_uv=8.0):
    t = np.arange(n) / FS
    return np.stack([_pink(n, rng) + alpha_uv * np.sin(2 * np.pi * 10.1 * t + rng.uniform(0, 6)) for _ in range(2)])


def _photic_trains(n, freqs, start_s=60, train_s=10, gap_s=6, amp=10.0):
    t = np.arange(n) / FS
    y, spans = np.zeros(n), []
    for i, f in enumerate(freqs):
        a = start_s + i * (train_s + gap_s)
        on = (t >= a) & (t < a + train_s)
        y[on] += amp * np.sin(2 * np.pi * f * t[on]) + 0.5 * amp * np.sin(2 * np.pi * 2 * f * t[on])
        spans.append((a, a + train_s))
    return y, spans


def test_photic_detector_flags_a_stepped_session_and_nothing_else():
    rng = np.random.default_rng(1)
    n = FS * 600
    occ = _occipital(n, rng)
    y, spans = _photic_trains(n, [3, 5, 7, 9, 11, 13, 15])
    iv = act.photic_intervals(occ + y, FS)
    assert len(iv) == 1
    assert iv.start_s[0] <= spans[0][0] and iv.end_s[0] >= spans[-1][1]
    assert iv.start_s[0] >= spans[0][0] - act.PHOTIC_MARGIN_S - act.PHOTIC_WIN_S
    assert iv.end_s[0] <= spans[-1][1] + act.PHOTIC_MARGIN_S + act.PHOTIC_WIN_S
    found = [float(f) for f in iv.detail[0].split("Hz")[0].split("=")[1].split(",")]
    assert {3.0, 5.0, 7.0, 15.0} <= set(found)
    assert not any(act.PHOTIC_ALPHA_EXCL[0] <= f <= act.PHOTIC_ALPHA_EXCL[1] for f in found)


@pytest.mark.parametrize("case", ["noise", "one_train", "stationary_line", "pulse_train", "cycling_lines",
                                  "comb_borne_steps"])
def test_photic_detector_ignores_background_single_trains_and_stationary_lines(case):
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    n = FS * 600
    t = np.arange(n) / FS
    occ = _occipital(n, rng)
    if case == "one_train":
        occ = occ + _photic_trains(n, [7])[0]
    elif case == "stationary_line":
        occ = occ + 10 * np.sin(2 * np.pi * 6 * t) + 5 * np.sin(2 * np.pi * 12 * t)
    elif case == "pulse_train":
        pulse = np.zeros(n)
        pulse[(np.arange(0, 600, 1 / 1.2) * FS).astype(int)] = 1
        occ = occ + 40 * np.convolve(pulse, np.hanning(8), "same")
    elif case == "cycling_lines":
        for a in range(0, 600, 40):
            occ = occ + _photic_trains(n, [4, 6, 15, 20], start_s=a, train_s=8, gap_s=2)[0]
    elif case == "comb_borne_steps":
        comb = sum(1.2 * np.sin(2 * np.pi * 1.25 * h * t) for h in range(2, 24))
        boost, _ = _photic_trains(n, [3.75, 6.25, 16.25, 21.25, 23.75], start_s=200, gap_s=10)
        occ = occ + comb + boost
    assert len(act.photic_intervals(occ, FS)) == 0


def test_photic_detector_is_quiet_on_noise_across_seeds():
    n = FS * 600
    assert sum(len(act.photic_intervals(_occipital(n, np.random.default_rng(s)), FS)) for s in range(20)) == 0


def _eyes(n, rng, plan, blink_uv=80.0, alpha_uv=12.0):
    """plan: [(time_s, "close" | "open" | "blink")]; eyes start open (no alpha). Returns Fp1, Fp2, O1, O2, F3, F4;
    every event puts a 0.3-s eye transient on Fp1/Fp2, reaching F3/F4 at 30% (EOG fall-off)."""
    t = np.arange(n) / FS
    blinks, closed = np.zeros(n), np.zeros(n, bool)
    w = np.hanning(int(0.3 * FS))
    for when, what in sorted(plan):
        k = int(when * FS)
        blinks[k:k + len(w)] += blink_uv * w[: n - k]
        if what in ("close", "open"):
            closed[k:] = what == "close"
    env = np.convolve(closed.astype(float), np.hanning(int(0.4 * FS)) / np.hanning(int(0.4 * FS)).sum(), "same")
    fp = np.stack([_pink(n, rng) + blinks for _ in range(2)])
    occ = np.stack([_pink(n, rng) + alpha_uv * env * np.sin(2 * np.pi * 10.1 * t + p) for p in (0.0, 0.7)])
    front = np.stack([_pink(n, rng) + 0.3 * blinks for _ in range(2)])
    return np.vstack([fp, occ, front])


def test_eoec_detector_flags_instructed_alternation():
    rng = np.random.default_rng(3)
    n = FS * 300
    plan = [(60, "close"), (70, "open"), (80, "close"), (90, "open"), (100, "close"), (110, "open")]
    iv = act.eoec_intervals(_eyes(n, rng, plan), FS)
    assert len(iv) == 1 and "cycles=3" in iv.detail[0]
    assert 60 - act.EOEC_MARGIN_S - 1 <= iv.start_s[0] <= 60 - act.EOEC_MARGIN_S + 1
    assert 110 + act.EOEC_MARGIN_S - 1 <= iv.end_s[0] <= 110 + act.EOEC_MARGIN_S + 1


@pytest.mark.parametrize("case", ["blinks_only", "alpha_only", "single_opening", "one_cycle", "irregular", "noise",
                                  "pulse_transients", "artefact_bursts", "frontal_slow_waves"])
def test_eoec_detector_ignores_blinks_alpha_changes_and_one_opening(case):
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    n = FS * 300
    if case == "blinks_only":
        x = _eyes(n, rng, [(s, "blink") for s in np.arange(10, 290, 4.3)])
    elif case == "alpha_only":
        x = _eyes(n, rng, [(60, "close"), (70, "open"), (80, "close"), (90, "open")], blink_uv=0.0)
    elif case == "single_opening":
        x = _eyes(n, rng, [(60, "close"), (120, "open")])
    elif case == "one_cycle":
        x = _eyes(n, rng, [(60, "close"), (70, "open")])
    elif case == "irregular":
        x = _eyes(n, rng, [(20, "close"), (70, "open"), (120, "close"), (170, "open"), (220, "close"), (270, "open")])
    elif case == "pulse_transients":
        plan = [(t, "close" if k % 2 == 0 else "open") for k, t in enumerate(range(60, 140, 10))]
        toggles = _eyes(n, rng, plan, blink_uv=0.0)
        pulse = _eyes(n, rng, [(s, "blink") for s in np.arange(5, 295, 1.1)])
        x = np.vstack([pulse[:2], toggles[2:4], pulse[4:]])
    elif case == "artefact_bursts":
        plan = [(60, "close"), (70, "open"), (80, "close"), (90, "open"), (100, "close"), (110, "open")]
        x = _eyes(n, rng, plan, alpha_uv=0.0)
        on = np.zeros(n, bool)
        for a_, b_ in [(60, 70), (80, 90), (100, 110)]:
            on[a_ * FS:b_ * FS] = True
        x[2:4, on] += 40 * rng.normal(size=(2, on.sum()))
    elif case == "frontal_slow_waves":
        plan = [(60, "close"), (70, "open"), (80, "close"), (90, "open"), (100, "close"), (110, "open")]
        x = _eyes(n, rng, plan)
        x[4:6] = x[0:2] + rng.normal(0, 1, size=(2, n))
    else:
        x = _eyes(n, rng, [])
    assert len(act.eoec_intervals(x, FS)) == 0


def test_detectors_skip_padding_and_bdsp_table_and_window_flags():
    rng = np.random.default_rng(5)
    n = FS * 300
    names = ["Fp1", "Fp2", "O1", "O2", "F3", "F4", "Cz"]
    x = np.vstack([_eyes(n, rng, [(60, "close"), (70, "open"), (80, "close"), (90, "open")]), _pink(n, rng)[None]])
    x[:, : FS * 30] = 0.0
    iv = act.bdsp_intervals(x.astype(np.float32), names, FS)
    assert list(iv.kind) == ["eoec"] and (iv.method == "detector").all()
    flags = act.flag_windows(iv, np.array([0.0, 20.0, 35.0, 75.0, 120.0]), 10.0)
    assert list(flags) == [False, False, True, True, False]
    with pytest.raises(ValueError, match="Fp1"):
        act.bdsp_intervals(x[2:], names[2:], FS)


def test_flag_windows_catches_instantaneous_annotations():
    iv = pd.DataFrame({"start_s": [7.0], "end_s": [7.0]})
    assert list(act.flag_windows(iv, np.array([0.0, 5.5, 7.0, 7.5]), 2.0)) == [False, True, True, False]
    assert list(act.flag_windows(iv.iloc[:0], np.array([0.0]), 2.0)) == [False]


def test_tal_text_repairs_double_encoded_cp1251_and_categories():
    raw = "Свет Вкл".encode("cp1251").decode("latin-1").encode("utf-8")
    assert act._text(raw) == "Свет Вкл" and act._text("Eyes open".encode()) == "Eyes open"
    assert act.category("Свет Вкл") == "stimulation"
    assert act.category("Старт измерения") == "technical"
    assert act.category("Recording start") == "technical"


def test_tables_list_every_recording_and_window_lookup_refuses_unknown_ones(tmp_path):
    from types import SimpleNamespace
    sig = {"F3": np.zeros(FS * 20, np.int16)}
    paths = [str(_edf(tmp_path / "p1.edf", sig, [(3.0, None, "Свет Вкл")])),
             str(_edf(tmp_path / "p2.edf", sig, []))]
    iv, recs = act.annotation_table(paths)
    assert list(recs.uid) == ["p1", "p2"] and list(recs.n_intervals) == [1, 0] and list(recs.dur_s) == [20.0, 20.0]
    assert list(iv.uid) == ["p1"] and iv.kind[0] == "stimulation" and iv.method[0] == "annotation"
    rng = np.random.default_rng(7)
    n = FS * 300
    x = np.vstack([_eyes(n, rng, [(60, "close"), (70, "open"), (80, "close"), (90, "open")]), _pink(n, rng)[None]])
    rec = SimpleNamespace(uid="b1", x=x, names=["Fp1", "Fp2", "O1", "O2", "F3", "F4", "Cz"], fs=FS)
    iv2, recs2 = act.detector_table([rec])
    assert list(recs2.uid) == ["b1"] and recs2.n_eoec[0] == 1 and recs2.flagged_s[0] == 70.0
    assert list(act.windows_flagged(iv2, recs2, "b1", [0.0, 50.0], 10.0)) == [False, True]
    with pytest.raises(KeyError, match="not processed"):
        act.windows_flagged(iv2, recs2, "b2", [0.0], 10.0)
