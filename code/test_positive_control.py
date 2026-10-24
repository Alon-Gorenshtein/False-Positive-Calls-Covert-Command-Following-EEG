"""Tests for code/118_positive_control.py (part A, A5: the reinstated
positive control). Synthetic fixtures first; the real-data checks read EEGMMIDB annotations and the Kojima event index
only (no module is run on real data here, no rate is computed). Every guard is also run against a deliberately broken
input, and the test checks that the guard catches it.
cd code && /usr/bin/python3 -m pytest test_positive_control.py -q"""
import importlib
import json
import os

import numpy as np
import pytest

pc = importlib.import_module("118_positive_control")
mc = importlib.import_module("bench.motor_common")

MOTOR = ("columbia", "cornell", "mbmi", "dubois", "cruse11", "edlow17", "cruse12p")
ERP = ("mbvt", "chennu13", "gibson16")
FS = 160.0



def one_over_f(n_ch, n, rng, chi=1.6):
    """channels x samples, power ~ 1 / f^chi (random phases), unit SD per channel."""
    f = np.fft.rfftfreq(n)
    amp = np.zeros(len(f))
    amp[1:] = f[1:] ** (-chi / 2)
    x = np.fft.irfft(amp * (rng.normal(size=(n_ch, len(f))) + 1j * rng.normal(size=(n_ch, len(f)))), n=n, axis=1)
    return x / x.std(1, keepdims=True)


def motor_person(rng, per_class=40, classes=("left", "right", "both", "feet", "rest"), effect=0.0, offsets=200.0,
                 fs=FS, names=None):
    """A synthetic EEGMMIDB-like person: one 'run' per source event (4.1 s), 1/f noise with a 10-Hz rhythm, a
    per-event DC offset (so a splice without a crossfade is a step), and (effect > 0) the 8-30 Hz band of every
    non-rest event scaled by (1 - effect)."""
    names = list(pc.MMIDB64) if names is None else list(names)
    n = int(round(4.1 * fs))
    runs, sources = [], []
    for c in classes:
        for k in range(per_class):
            x = 10.0 * one_over_f(len(names), n, rng)
            mu = mc.bandpass(rng.normal(size=(len(names), n)), fs, 9.0, 11.0, order=2)
            x += 8.0 * mu / mu.std(1, keepdims=True)
            if effect and c != "rest":
                band = mc.bandpass(x, fs, 8.0, 30.0)
                x = x - effect * band
            x += rng.uniform(-offsets, offsets, size=(len(names), 1))
            runs.append(x.astype(np.float32))
            sources.append(pc.Source(run=len(runs) - 1, start=0, n=n, cls=c, tag=f"syn:{c}:{k}"))
    return pc.Person(pid=f"syn{rng.integers(1 << 30)}", fs=fs, names=names, runs=runs, sources=sources)


def erp_person(rng, n_trials=24, fs=250.0, p3=0.0, names=None):
    """A synthetic Kojima-like person: n_trials trials of 150 attended tones at 0.6 s (15 counted targets, 15 other
    deviants, 120 standards, deviants >= 3 tones apart), target identity balanced low / high; 1/f noise; p3 > 0
    plants a P3-like wave after every counted target at Cz / Pz."""
    names = list(pc.ERP_CHANNELS) if names is None else names
    runs, trials = [], []
    for t in range(n_trials):
        target = ("low", "high")[t % 2]
        extra = rng.multinomial(150 - 2 - (29 * 3 + 1), np.full(31, 1 / 31))
        dev = 2 + extra[0] + np.cumsum(np.r_[0, 3 + extra[1:30]])
        kind = np.array(["standard"] * 150, dtype=object)
        kind[dev] = rng.permutation(["target"] * 15 + ["other"] * 15)
        kind = kind.astype(str)
        other = "high" if target == "low" else "low"
        ident = np.where(kind == "target", target, np.where(kind == "other", other, ""))
        lead, soa = 12.0, 0.6
        n = int(round((lead + 150 * soa + 12.0) * fs))
        x = 8.0 * one_over_f(len(names), n, rng)
        on = np.round((lead + np.arange(150) * soa) * fs).astype(int)
        if p3:
            w = np.exp(-((np.arange(int(0.8 * fs)) / fs - 0.4) ** 2) / (2 * 0.08 ** 2))
            rows = [i for i, c in enumerate(names) if c in ("Cz", "Pz", "CPz", "CP1", "CP2", "P3", "P4")]
            for o in on[kind == "target"]:
                x[rows, o:o + len(w)] += p3 * w
        x += rng.uniform(-100, 100, size=(len(names), 1))
        runs.append(x.astype(np.float32))
        code = np.where(kind == "standard", 1, np.where(kind == "target", 111, 101))
        trials.append(pc.Trial(run=len(runs) - 1, tid=f"syn-trial{t}", target=target, on=on, kind=kind,
                               ident=ident, code=code, lo=0, hi=n))
    return pc.ErpPerson(pid=f"synE{rng.integers(1 << 30)}", fs=fs, names=names, runs=runs, trials=trials)


def mod_of(name):
    return importlib.import_module(f"bench.pipe_{name}")


def variants_to_check(mod):
    """DEFAULT (or run()'s default) plus every variant whose schedule differs in length or design."""
    vs = list(mod.VARIANTS)
    prim = pc.primary_variant(mod)
    picks = [prim] + [v for v in vs if v != prim]
    seen, out = set(), []
    for v in picks:
        ov = mod.schedule(pc.START_S, np.random.default_rng(5), v)
        sig = (len(ov.onsets_s), round(float(ov.onsets_s.max() - ov.onsets_s.min()), 3),
               mc.cfg(mod.VARIANTS, v).get("paradigm"))
        if sig not in seen:
            seen.add(sig)
            out.append(v)
    return out



def _periods(mod, ov):
    """(start, end, class) of every onset's response period, truncated at the next one, in s."""
    on = np.asarray(ov.onsets_s, float)
    d = np.broadcast_to(np.asarray(ov.meta.get("resp_delay_s", 0.0), float), on.shape)
    dur = np.broadcast_to(np.asarray(ov.meta["resp_dur_s"], float), on.shape)
    a, b = on + d, on + d + dur
    nxt = np.r_[a[1:], np.inf]
    b = np.minimum(b, nxt)
    return [(float(x), float(y), pc.class_of(mod, int(l))) for x, y, l in zip(a, b, ov.labels)]


@pytest.mark.parametrize("name", MOTOR)
def test_motor_session_follows_the_module_schedule(name):
    mod = mod_of(name)
    rng = np.random.default_rng(1)
    person = motor_person(rng, per_class=30)
    for v in variants_to_check(mod):
        for design in ("published", "randomized"):
            fn = mod.schedule if design == "published" else mod.schedule_randomized
            ov = fn(pc.START_S, np.random.default_rng(2), v)
            s = pc.motor_session(person, mod, ov, np.random.default_rng(3))
            assert s.overlay is ov
            fs = person.fs
            slots = s.plan
            T = s.rec.x.shape[1] / fs
            t0, t1 = mod.extent(ov)
            assert 0 <= t0 and t1 <= T, (name, v, design)
            per = _periods(mod, ov)
            for a, b, c in per:
                inside = [p for p in slots if min(p["b"] / fs, b) - max(p["a"] / fs, a) > 1.0 / fs]
                assert inside and all(p["cls"] == c for p in inside), (name, v, design, a, c)
                assert any(abs(p["a"] / fs - a) <= 1.0 / fs for p in slots)
                assert any(abs(p["b"] / fs - b) <= 1.0 / fs for p in slots)
            for p in slots:
                mid = (p["a"] + p["b"]) / 2 / fs
                if not any(a <= mid < b for a, b, _ in per):
                    assert p["cls"] == "rest", (name, v, design, p)


def test_motor_session_breaks_when_the_schedule_is_ignored():
    """Deliberately broken: a session built on a shifted schedule must fail the coverage check."""
    mod = mod_of("cornell")
    person = motor_person(np.random.default_rng(1), per_class=30)
    ov = mod.schedule(pc.START_S, np.random.default_rng(2), pc.primary_variant(mod))
    shifted = mc.Overlay(onsets_s=ov.onsets_s + 7.5, labels=ov.labels, kind=ov.kind, meta=dict(ov.meta))
    s = pc.motor_session(person, mod, shifted, np.random.default_rng(3))
    per = _periods(mod, ov)
    wrong = 0
    for a, b, c in per:
        inside = [p for p in s.plan if min(p["b"] / FS, b) - max(p["a"] / FS, a) > 1.0 / FS]
        wrong += any(p["cls"] != c for p in inside)
    assert wrong > 0


@pytest.mark.parametrize("name", ERP)
def test_erp_session_follows_the_module_schedule(name):
    mod = mod_of(name)
    person = erp_person(np.random.default_rng(4))
    for v in variants_to_check(mod):
        for design in ("published", "randomized"):
            fn = mod.schedule if design == "published" else mod.schedule_randomized
            tmpl = fn(pc.START_S, np.random.default_rng(6), v)
            s = pc.erp_session(person, mod, tmpl, np.random.default_rng(7))
            ov, meta = s.overlay, s.overlay.meta
            T = s.rec.x.shape[1] / person.fs
            t0, t1 = mod.extent(ov)
            assert 0 <= t0 and t1 <= T, (name, v, design)
            assert np.all(np.diff(ov.onsets_s) > 0)
            lab = np.asarray(ov.labels)
            if name == "mbvt":
                c = mc.cfg(mod.VARIANTS, v)
                grp = np.asarray(meta["group"])
                assert grp.max() + 1 == mod.N_GROUPS and np.all(np.bincount(grp) == mod.GROUP)
                np.testing.assert_array_equal(meta["target_wrist"], tmpl.meta["target_wrist"])
                for gi in range(mod.N_GROUPS):
                    lg = lab[grp == gi]
                    assert (lg == mod.TARGET).sum() == 1
                    if c["paradigm"] == "vt3":
                        assert (lg == mod.NONTARGET).sum() == 1 and (lg == mod.DISTRACTOR).sum() == 6
                    else:
                        assert (lg == mod.NONTARGET).sum() == 7
                assert meta["paradigm"] == c["paradigm"]
            elif name == "chennu13":
                np.testing.assert_array_equal(meta["explicit_word"], tmpl.meta["explicit_word"])
                blk = np.asarray(meta["block"])
                assert blk.max() + 1 == len(tmpl.meta["explicit_word"])
                for b, w in enumerate(meta["explicit_word"]):
                    words = np.asarray(meta["word"])[(blk == b) & (lab == mod.EXPL)]
                    assert len(words) and np.all(words == w)
            else:
                np.testing.assert_array_equal(meta["target_wrist"], tmpl.meta["target_wrist"])
                blk = np.asarray(meta["block"])
                assert blk.max() + 1 == len(tmpl.meta["target_wrist"])
                for b, w in enumerate(meta["target_wrist"]):
                    sites = np.asarray(meta["site"])[(blk == b) & (lab == mod.TARGET)]
                    assert len(sites) and np.all(sites == w)



def splice_report(x, plan, fs):
    """(step ratio, rms ratio). Step: per channel, the 95th percentile of |first difference| across the junction
    samples (one per splice) over the 95th percentile of |first difference| at all other samples; max over channels.
    RMS: 8-30 Hz band, median RMS in windows of the crossfade length centred on the splices over the median RMS of
    such windows centred between splices (slot middles); median over channels."""
    x = np.asarray(x, float)
    d = np.abs(np.diff(x, axis=1))
    edges = np.array([p["a"] for p in plan[1:]])
    at = np.zeros(d.shape[1], bool)
    at[edges - 1] = True
    step = float(np.max(np.percentile(d[:, at], 95, axis=1) / np.percentile(d[:, ~at], 95, axis=1)))
    h = max(2, pc.xfade_samples(fs) // 2)
    band = mc.bandpass(x, fs, 8.0, 30.0)
    rms = lambda i: np.sqrt((band[:, i - h: i + h] ** 2).mean(1))
    mids = [(p["a"] + p["b"]) // 2 for p in plan[1:-1]]
    near = np.median([rms(e) for e in edges if h <= e < x.shape[1] - h], axis=0)
    away = np.median([rms(m) for m in mids if h <= m < x.shape[1] - h], axis=0)
    return step, float(np.median(near / away))


def _spliced(xfade_s):
    mod = mod_of("columbia")
    person = motor_person(np.random.default_rng(11), per_class=60, offsets=300.0)
    ov = mod.schedule(pc.START_S, None, "as_coded")
    return pc.motor_session(person, mod, ov, np.random.default_rng(12), xfade_s=xfade_s), person.fs


def test_no_splice_creates_a_step_or_a_variance_dip():
    s, fs = _spliced(pc.XFADE_S)
    assert len(s.plan) > 300
    step, rms = splice_report(s.rec.x, s.plan, fs)
    assert step < 1.5, step
    assert 0.9 < rms < 1.1, rms


def test_the_step_check_fails_without_the_crossfade():
    """Deliberately broken: the same session with the crossfade disabled must fail the step check."""
    s, fs = _spliced(0.0)
    step, _ = splice_report(s.rec.x, s.plan, fs)
    assert step >= 1.5, step


def test_a_linear_crossfade_would_dip_the_variance():
    """Deliberately broken: a linear (equal-gain) crossfade of independent segments dips the RMS at the splice."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(8, 4000)), rng.normal(size=(8, 4000))
    c = 400
    lin = pc.join([a, b], c, kind="linear")
    eq = pc.join([a, b], c, kind="equal_power")
    mid = slice(4000 - c // 2 - 50, 4000 - c // 2 + 50)
    assert np.sqrt((lin[:, mid] ** 2).mean()) < 0.8
    assert 0.9 < np.sqrt((eq[:, mid] ** 2).mean()) < 1.1



def test_every_slot_is_filled_from_a_source_of_its_own_class():
    mod = mod_of("cruse11")
    person = motor_person(np.random.default_rng(21), per_class=50)
    ov = mod.schedule(pc.START_S, np.random.default_rng(22), pc.primary_variant(mod))
    s = pc.motor_session(person, mod, ov, np.random.default_rng(23))
    c = pc.xfade_samples(person.fs)
    rows, names = pc.module_channels(mod, person.names)
    assert list(s.rec.names) == names
    for p in s.plan:
        src = person.sources[p["src"]]
        assert src.cls == p["cls"]
        piece = person.runs[src.run][rows, src.start + p["off"]: src.start + p["off"] + (p["b"] - p["a"]) + c]
        piece = piece - piece.mean(1, keepdims=True)
        got = s.rec.x[:, p["a"] + c // 2: p["b"] - c // 2]
        np.testing.assert_allclose(got, piece[:, c: c + got.shape[1]], atol=1e-3)


def test_a_relabelled_source_is_refused():
    """Deliberately broken: a source whose class does not match its event is caught by check_provenance."""
    person = motor_person(np.random.default_rng(31), per_class=5)
    pc.check_provenance(person)
    bad = pc.Person(pid=person.pid, fs=person.fs, names=person.names, runs=person.runs,
                    sources=[pc.Source(run=s.run, start=s.start, n=s.n, cls="left" if s.cls == "rest" else s.cls,
                                       tag=s.tag) for s in person.sources])
    with pytest.raises(ValueError):
        pc.check_provenance(bad)


MMIDB_DOC = {(4, "T1"): "left", (4, "T2"): "right", (8, "T1"): "left", (8, "T2"): "right", (12, "T1"): "left",
             (12, "T2"): "right", (6, "T1"): "both", (6, "T2"): "feet", (10, "T1"): "both", (10, "T2"): "feet",
             (14, "T1"): "both", (14, "T2"): "feet"}


@pytest.mark.skipif(not os.path.exists(pc.mmidb_file(1, 4)), reason="EEGMMIDB S001 not on disk")
def test_real_eegmmidb_sources_match_the_annotations():
    import mne
    person = pc.mmidb_person(1, "imagery")
    assert len(person.sources) == 6 * 30
    for s in person.sources:
        run = person.meta["run_numbers"][s.run]
        ann = mne.read_annotations(pc.mmidb_file(1, run))
        t = s.start / person.fs
        k = int(np.argmin(np.abs(ann.onset - t)))
        assert abs(ann.onset[k] - t) < 1.0 / person.fs
        want = "rest" if ann.description[k] == "T0" else MMIDB_DOC[(run, ann.description[k])]
        assert s.cls == want, (run, t, s.cls, want)
        assert s.n <= int(round(ann.duration[k] * person.fs)) + 1


@pytest.mark.skipif(not os.path.exists(pc.KOJIMA_EVENTS), reason="Kojima event index not on disk")
def test_real_kojima_trials_match_the_event_codes():
    import pandas as pd
    ev = pd.read_parquet(pc.KOJIMA_EVENTS)
    trials = pc.kojima_trials(ev, "L")
    assert len(trials) == 24
    for t in trials:
        assert len(t.on) == 150 and (t.kind == "target").sum() == 15 and (t.kind == "other").sum() == 15
        assert np.all(np.isin(t.code[t.kind == "target"], (111, 112, 113, 114)))
        assert np.all(np.isin(t.code[t.kind == "other"], (101, 102, 103, 104)))
        assert np.all(np.isin(t.code[t.kind == "standard"], (1, 2)))
        low = {111: "low", 113: "low", 112: "high", 114: "high"}
        assert {low[c] for c in t.code[t.kind == "target"]} == {t.target}


def test_erp_target_labels_are_the_counted_deviants():
    """Every TARGET / EXPL label in the built overlay sits on a stimulus that was really counted (kind 'target')."""
    person = erp_person(np.random.default_rng(41))
    for name in ERP:
        mod = mod_of(name)
        v = pc.primary_variant(mod)
        tmpl = mod.schedule(pc.START_S, np.random.default_rng(42), v)
        s = pc.erp_session(person, mod, tmpl, np.random.default_rng(43))
        tgt = getattr(mod, "EXPL", None) if name == "chennu13" else mod.TARGET
        lab = np.asarray(s.overlay.labels)
        assert np.all(s.overlay.meta["real_kind"][lab == tgt] == "target")
        assert np.all(s.overlay.meta["real_kind"][lab != tgt] != "target")


def test_erp_control_moves_the_target_label_off_the_counted_deviant():
    """The null-shuffled ERP control: the instruction (which deviant was counted) is permuted over trials, so about
    half of the TARGET labels sit on deviants that were not counted."""
    person = erp_person(np.random.default_rng(51))
    mod = mod_of("gibson16")
    tmpl = mod.schedule(pc.START_S, np.random.default_rng(52), pc.primary_variant(mod))
    s = pc.erp_session(person, mod, tmpl, np.random.default_rng(53), control=True)
    lab = np.asarray(s.overlay.labels)
    share = np.mean(s.overlay.meta["real_kind"][lab == mod.TARGET] == "target")
    assert 0.2 < share < 0.8



def _trial_logpower(x, names, fs, ov, mod):
    on = np.asarray(ov.onsets_s, float)
    d = np.broadcast_to(np.asarray(ov.meta.get("resp_delay_s", 0.0), float), on.shape)
    dur = np.broadcast_to(np.asarray(ov.meta["resp_dur_s"], float), on.shape)
    band = mc.bandpass(np.asarray(x, float), fs, 8.0, 30.0)
    rows = [list(names).index(c) for c in ("C3", "Cz", "C4")]
    return np.array([np.log((band[rows, int((o + a) * fs): int((o + a + u) * fs)] ** 2).mean())
                     for o, a, u in zip(on, d, dur)])


def _perm_p(f, task, g, B=199):
    obs = f[~task].mean() - f[task].mean()
    null = [f[~t].mean() - f[t].mean() for t in (g.permutation(task) for _ in range(B))]
    return (1 + np.sum(np.asarray(null) >= obs)) / (B + 1)


def test_null_shuffled_control_is_near_nominal_and_the_true_session_is_detected():
    mod = mod_of("cornell")
    v = "theoretical_bh_keep40_r2"
    n_people, hits, false = 200, 0, 0
    for i in range(n_people):
        g = np.random.default_rng(1000 + i)
        person = motor_person(g, per_class=120, classes=("both", "rest"), effect=0.5, names=pc.EEG19)
        ov = mod.schedule(pc.START_S, g, v)
        task = np.asarray(ov.labels) == mod.TASK
        true = pc.motor_session(person, mod, ov, g)
        ctrl = pc.motor_session(person, mod, ov, g, control=True)
        assert ctrl.info["control"] and not true.info["control"]
        assert true.info["reuse_max"] == 1 and ctrl.info["reuse_max"] == 1
        hits += _perm_p(_trial_logpower(true.rec.x, true.rec.names, person.fs, ov, mod), task, g) < 0.05
        false += _perm_p(_trial_logpower(ctrl.rec.x, ctrl.rec.names, person.fs, ov, mod), task, g) < 0.05
    assert hits / n_people > 0.9
    assert false < 20, false


def test_control_keeps_the_reuse_structure_and_shuffles_class_at_the_source():
    """The control draws from pools whose CLASS labels are permuted over source events (counts kept), so every copy of
    a reused source carries the same pseudo-class, as in the true session."""
    mod = mod_of("edlow17")
    person = motor_person(np.random.default_rng(61), per_class=12, classes=("right", "rest"))
    ov = mod.schedule(pc.START_S, np.random.default_rng(62), pc.primary_variant(mod))
    true = pc.motor_session(person, mod, ov, np.random.default_rng(63))
    ctrl = pc.motor_session(person, mod, ov, np.random.default_rng(63), control=True)
    assert true.info["reuse_max"] > 1 and ctrl.info["reuse_max"] > 1
    pseudo = {}
    for p in ctrl.plan:
        pseudo.setdefault(p["src"], set()).add(p["cls"])
    assert all(len(v) == 1 for v in pseudo.values())
    real = np.array([person.sources[p["src"]].cls == p["cls"] for p in ctrl.plan])
    assert 0.2 < real.mean() < 0.8
    assert np.all([person.sources[p["src"]].cls == p["cls"] for p in true.plan])



def test_bars_are_read_from_the_k_table():
    with open(pc.K_TABLE) as fh:
        kt = json.load(fh)
    assert set(pc.BARS) == set(MOTOR + ERP)
    for pipe, bars in pc.BARS.items():
        for b in bars:
            assert f"{b['k']}/{b['n']}" in kt[pipe]["control_rate"], (pipe, b)
    for pipe, b in pc.SECONDARY_BARS.items():
        assert f"{b['k']}/{b['n']}" in kt[pipe]["control_rate"], (pipe, b)


def test_bars_cover_every_variant():
    for name in MOTOR + ERP:
        mod = mod_of(name)
        for v in mod.VARIANTS:
            assert pc.bar_for(name, v) is not None, (name, v)


def test_detection_rate_ci_and_fidelity_flag():
    r = pc.detection(np.array([True] * 3 + [False] * 17), bar=(9, 12))
    assert r["k"] == 3 and r["n"] == 20
    lo, hi = mc.cp_interval(3, 20, 0.95)
    assert abs(r["ci"][0] - lo) < 1e-12 and abs(r["ci"][1] - hi) < 1e-12
    assert r["fidelity_concern"] is True
    ok = pc.detection(np.array([True] * 14 + [False] * 6), bar=(9, 12))
    assert ok["fidelity_concern"] is False


def test_summary_leaves_out_dev_people_and_unusable_rows():
    rows = [{"person": "S001", "dev": True, "pipe": "cruse11", "variant": "cr11_b4_hilbert_ci99", "design": "published",
             "arm": "imagery", "version": "true", "status": "ok", "call": True}]
    rows += [{"person": f"S{i:03d}", "dev": False, "pipe": "cruse11", "variant": "cr11_b4_hilbert_ci99",
              "design": "published", "arm": "imagery", "version": "true", "status": "ok", "call": i % 2 == 0}
             for i in range(10, 30)]
    rows += [{"person": "S099", "dev": False, "pipe": "cruse11", "variant": "cr11_b4_hilbert_ci99",
              "design": "published", "arm": "imagery", "version": "true", "status": "na: no_usable_trials",
              "call": None}]
    out = pc.summarize(rows)
    cell = out["cells"]["imagery|cruse11|published|cr11_b4_hilbert_ci99|true"]
    assert cell["n"] == 20 and cell["k"] == 10 and cell["n_unusable"] == 1 and cell["n_dev_excluded"] == 1




def test_write_summary_headline_and_dev_guard(tmp_path, monkeypatch):
    out_dir = tmp_path / "pc"
    out_dir.mkdir()
    monkeypatch.setitem(pc.OUT, False, str(out_dir))
    mod = mod_of("edlow17")
    prim = pc.primary_variant(mod)
    rep = next(g for g, vs in pc.rand_groups(mod).items() if prim in vs)
    rows = []
    for i in range(12):
        for design, v in (("published", prim), ("randomized", rep)):
            for ver in ("true", "control"):
                rows.append({"key": f"imagery|S{i:03d}|edlow17|{design}|{v}|{ver}", "arm": "imagery",
                             "person": f"S{i:03d}", "dev": False, "pipe": "edlow17", "design": design, "variant": v,
                             "version": ver, "status": "ok", "call": ver == "true" and i < 3,
                             "session_reuse_max": 4})
    with open(out_dir / "imagery__edlow17.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    path = tmp_path / "summary.json"
    s = pc.write_summary(str(path))
    h = s["headline"]["imagery|edlow17"]
    assert h["default_published"]["k"] == 3 and h["default_published"]["n"] == 12
    assert h["default_published"]["fidelity_concern"] is True
    assert h["control_randomized"]["k"] == 0 and h["control_randomized"]["reuse_or_splice_calls"] is False
    assert h["reuse_max_median"] == 4
    with open(out_dir / "imagery__edlow17.jsonl", "a") as fh:
        fh.write(json.dumps(dict(rows[0], key="x", dev=True)) + "\n")
    with pytest.raises(RuntimeError):
        pc.write_summary(str(path))


def test_erp_reuse_is_counted_per_unit():
    """A unit drawn once is counted once (unit ids are unique across the low / high pools); 24 trials fill 20 Chennu
    blocks without reuse, and 8 trials cannot."""
    mod = mod_of("chennu13")
    tmpl = mod.schedule(pc.START_S, np.random.default_rng(71), "p1_samples10_none_b20")
    s = pc.erp_session(erp_person(np.random.default_rng(72)), mod, tmpl, np.random.default_rng(73))
    assert s.info["reuse_max"] == 1 and s.info["n_sources"] == 20
    small = erp_person(np.random.default_rng(74), n_trials=8)
    s2 = pc.erp_session(small, mod, tmpl, np.random.default_rng(75))
    assert s2.info["reuse_max"] > 1 and s2.info["n_sources"] == 8



def _overlaps(plan, person, c):
    """Pairs of slots whose source excerpts share a sample (same source, overlapping [off, off + need))."""
    by = {}
    for p in plan:
        by.setdefault(p["src"], []).append((p["off"], p["off"] + p["b"] - p["a"] + c))
    n = 0
    for iv in by.values():
        iv.sort()
        n += sum(b0 > a1 for (a0, b0), (a1, b1) in zip(iv, iv[1:]))
    return n


def test_session_is_reuse_free_when_the_data_suffice():
    mod = mod_of("edlow17")
    person = motor_person(np.random.default_rng(81), per_class=120, classes=("right", "rest"))
    ov = mod.schedule(pc.START_S, np.random.default_rng(82), pc.primary_variant(mod))
    s = pc.motor_session(person, mod, ov, np.random.default_rng(83))
    c = pc.xfade_samples(person.fs)
    assert _overlaps(s.plan, person, c) == 0
    assert s.info["reuse_max"] == 1 and s.info["demand_ratio_max"] <= 1.0


def test_long_sources_host_several_disjoint_excerpts():
    """A 30-s rest source is cut into disjoint excerpts (packing), so a few long sources fill many rest slots without
    reusing a sample; random whole-source draws would reuse."""
    mod = mod_of("edlow17")
    rng = np.random.default_rng(84)
    person = motor_person(rng, per_class=120, classes=("right",))
    n = int(30 * FS)
    for k in range(14):
        person.runs.append((10 * one_over_f(len(person.names), n, rng)).astype(np.float32))
        person.sources.append(pc.Source(run=len(person.runs) - 1, start=0, n=n, cls="rest", tag=f"syn:rest:{k}"))
    ov = mod.schedule(pc.START_S, np.random.default_rng(85), pc.primary_variant(mod))
    s = pc.motor_session(person, mod, ov, np.random.default_rng(86))
    rest = [p for p in s.plan if p["cls"] == "rest"]
    assert len({p["src"] for p in rest}) < len(rest)
    assert _overlaps(s.plan, person, pc.xfade_samples(FS)) == 0 and s.info["reuse_max"] == 1


def test_reuse_is_counted_per_sample_when_the_data_run_out():
    mod = mod_of("edlow17")
    person = motor_person(np.random.default_rng(87), per_class=12, classes=("right", "rest"))
    ov = mod.schedule(pc.START_S, np.random.default_rng(88), pc.primary_variant(mod))
    s = pc.motor_session(person, mod, ov, np.random.default_rng(89))
    assert s.info["reuse_max"] > 1 and s.info["demand_ratio_max"] > 1.0
    assert _overlaps(s.plan, person, pc.xfade_samples(FS)) > 0


def test_plan_only_allocation_matches_the_session():
    """allocate() on source lengths alone (the feasibility table) gives the same reuse as the built session."""
    mod = mod_of("columbia")
    person = motor_person(np.random.default_rng(90), per_class=30)
    ov = mod.schedule(pc.START_S, None, "as_coded")
    s = pc.motor_session(person, mod, ov, np.random.default_rng(91))
    f = pc.feasibility(mod, ov, person.sources, person.fs, np.random.default_rng(91))
    assert f["reuse_max"] == s.info["reuse_max"]
    assert abs(f["demand_ratio_max"] - s.info["demand_ratio_max"]) < 1e-9



def test_sample_is_seeded_excludes_dev_and_is_stable():
    a = pc.people("imagery", dev=False, sample=20)
    assert len(a) == 20 and a == pc.people("imagery", dev=False, sample=20)
    dev = {f"S{s:03d}" for s in pc.dev_people("imagery")}
    assert not set(a) & dev
    full = pc.people("imagery", dev=False)
    g = np.random.default_rng(pc.SAMPLE_SEED)
    assert a == sorted(full[i] for i in g.choice(len(full), 20, replace=False))
    assert pc.SAMPLE_SEED == 20261014
    with pytest.raises(ValueError):
        pc.people("imagery", dev=False, sample=len(full) + 1)



def test_openbmi_sources_follow_the_trial_timing():
    """OpenBMI MI trial: 3-s fixation, cue at t, 4-s imagery, 6 +- 1.5-s blank. Imagery = [t, t + 4 s) of its class
    (1 right, 2 left); rest = the contiguous no-imagery stretch [t + 4 s, next cue) (blank + the next fixation), the
    lead-in [0, first cue) and the tail [last t + 4 s, end), each kept only if at least 4 s long."""
    fs = 250.0
    t = np.array([5000, 8500, 12000])
    y = np.array([1, 2, 1])
    src = pc.openbmi_sources(t, y, n=15000, fs=fs, run=0, tag="tr")
    img = [s for s in src if s.cls in ("right", "left")]
    assert [s.cls for s in img] == ["right", "left", "right"]
    assert all(s.n == 1000 for s in img) and [s.start for s in img] == list(t)
    rest = sorted((s.start, s.start + s.n) for s in src if s.cls == "rest")
    assert rest == [(0, 5000), (6000, 8500), (9500, 12000), (13000, 15000)]
    short = pc.openbmi_sources(t, y, n=13500, fs=fs, run=0, tag="tr")
    assert max(s.start + s.n for s in short if s.cls == "rest") == 12000
    allv = sorted((s.start, s.start + s.n) for s in src)
    assert all(b0 <= a1 for (a0, b0), (a1, b1) in zip(allv, allv[1:]))


def test_one_short_source_does_not_shrink_every_slot():
    """The slot grid's longest slot comes from the typical (median) source length per class, not the single shortest
    source; a short source still hosts the short slots it can hold."""
    mod = mod_of("edlow17")
    person = motor_person(np.random.default_rng(95), per_class=120, classes=("right", "rest"))
    short = pc.Source(run=0, start=0, n=int(1.0 * FS), cls="rest", tag="syn:rest:short")
    person.sources.append(short)
    ov = mod.schedule(pc.START_S, np.random.default_rng(96), pc.primary_variant(mod))
    s = pc.motor_session(person, mod, ov, np.random.default_rng(97))
    longest = max(p["b"] - p["a"] for p in s.plan)
    assert longest > int(3.0 * FS)


def test_control_has_the_same_splice_grid_as_the_true_session():
    mod = mod_of("columbia")
    person = motor_person(np.random.default_rng(98), per_class=30)
    person.runs.append((10 * one_over_f(len(person.names), int(9.0 * FS), np.random.default_rng(1))).astype(np.float32))
    person.sources += [pc.Source(run=len(person.runs) - 1, start=0, n=int(9.0 * FS), cls="rest", tag="syn:rest:long")]
    ov = mod.schedule(pc.START_S, None, "as_coded")
    true = pc.motor_session(person, mod, ov, np.random.default_rng(99))
    ctrl = pc.motor_session(person, mod, ov, np.random.default_rng(99), control=True)
    assert [(p["a"], p["b"]) for p in true.plan] == [(p["a"], p["b"]) for p in ctrl.plan]



def test_stream_fetches_runs_and_purges(tmp_path, monkeypatch):
    calls = {"fetch": [], "load": [], "run": [], "purge": []}
    monkeypatch.setitem(pc.OUT, False, str(tmp_path / "out"))
    monkeypatch.setattr(pc, "OPENBMI_DIR", str(tmp_path / "obmi"))
    os.makedirs(pc.OPENBMI_DIR)
    monkeypatch.setattr(pc, "OPENBMI_SESSIONS", {"edlow17": (1,), "cornell": (1, 2)})
    monkeypatch.setattr(pc, "PLAN", {"edlow17": ("primary", "primary", None), "cornell": ("primary", "primary", 2)})

    def fetch(subj, sess):
        calls["fetch"].append((subj, sess))
        open(pc.openbmi_npz(subj, sess), "w").close()
        return pc.openbmi_npz(subj, sess)

    def load(subj, sessions):
        calls["load"].append((subj, tuple(sessions)))
        return pc.Person(pid=f"OpenBMI_S{subj:02d}", fs=250.0, names=[], runs=[], sources=[],
                         meta={"source": "openbmi"})

    def runner(person, mod, arm, dev, versions, done, write, scope, control_scope):
        calls["run"].append((person.pid, mod.NAME))
        for d, v in pc.jobs_for(mod, "true", scope, control_scope):
            write({"key": f"{arm}|{person.pid}|{mod.NAME}|{d}|{v}|true", "status": "ok"})

    monkeypatch.setattr(pc, "openbmi_fetch", fetch)
    monkeypatch.setattr(pc, "openbmi_person", load)
    monkeypatch.setattr(pc, "run_person", runner)
    ppl = pc.people("openbmi", dev=False)[:4]
    monkeypatch.setattr(pc, "people", lambda arm, dev, given=None, limit=None, sample=None:
                        (ppl[:sample] if sample else ppl))
    pc.run_stream("openbmi", ["edlow17", "cornell"], versions=("true",))
    subs = [int(p[1:]) for p in ppl]
    assert sorted(calls["fetch"]) == sorted([(s, 1) for s in subs] + [(s, 2) for s in subs[:2]])
    assert ("OpenBMI_S%02d" % subs[3], "cornell") not in calls["run"]
    assert (subs[0], (1, 2)) in calls["load"] and (subs[0], (1,)) in calls["load"]
    assert os.listdir(pc.OPENBMI_DIR) == []
    calls["fetch"].clear()
    pc.run_stream("openbmi", ["edlow17", "cornell"], versions=("true",))
    assert calls["fetch"] == []


def test_headline_marks_primary_secondary_and_reuse_confounded(tmp_path, monkeypatch):
    out_dir = tmp_path / "pc"
    out_dir.mkdir()
    monkeypatch.setitem(pc.OUT, False, str(out_dir))
    rows = []
    for arm, pipe, reuse in (("openbmi", "edlow17", 1), ("imagery", "edlow17", 4), ("imagery", "cruse11", 4)):
        mod = mod_of(pipe)
        prim = pc.primary_variant(mod)
        rep = next(g for g, vs in pc.rand_groups(mod).items() if prim in vs)
        for i in range(10):
            for design, v in (("published", prim), ("randomized", rep)):
                rows.append({"key": f"{arm}|S{i:02d}|{pipe}|{design}|{v}|true", "arm": arm, "person": f"S{i:02d}",
                             "dev": False, "pipe": pipe, "design": design, "variant": v, "version": "true",
                             "status": "ok", "call": True, "session_reuse_max": reuse})
    for arm in ("openbmi", "imagery"):
        with open(out_dir / f"{arm}.jsonl", "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows if r["arm"] == arm) + "\n")
    h = pc.write_summary(str(tmp_path / "s.json"))["headline"]
    assert h["openbmi|edlow17"]["role"] == "primary" and h["openbmi|edlow17"]["verdict"] == "consistent with the bar"
    assert h["imagery|edlow17"]["role"] == "secondary"
    assert h["imagery|cruse11"]["role"] == "secondary" and h["imagery|cruse11"]["verdict"] == "reuse-confounded"
    assert pc.PRIMARY["cruse11"] is None
    bad = [dict(r, session_reuse_max=2) if r["arm"] == "openbmi" else r for r in rows]
    with open(out_dir / "openbmi.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in bad if r["arm"] == "openbmi") + "\n")
    h = pc.write_summary(str(tmp_path / "s.json"))["headline"]
    assert h["openbmi|edlow17"]["verdict"] == "reuse-confounded"
