"""Tests for code/124_patient_control.py (patient-level positive control: aware patients with brain or cord disease
doing instructed tasks). Synthetic fixtures first; the real-data checks read only event structure, labels and a few
samples (no module is run on these people, no rate is computed).
cd code && /usr/bin/python3 -m pytest test_patient_control.py -q"""
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

pt = importlib.import_module("124_patient_control")
pc = importlib.import_module("118_positive_control")

MOTOR = ("columbia", "cornell", "mbmi", "dubois", "cruse11", "edlow17", "cruse12p")


def mod_of(name):
    return importlib.import_module(f"bench.pipe_{name}")



def test_ucl_sources_follow_the_trial_structs():
    """UCL ALS: each struct element holds one left (L), right (R) and rest (Re) trial; all-zero elements are padding."""
    rng = np.random.default_rng(1)
    el = [SimpleNamespace(L=rng.normal(size=(1288, 22)), R=rng.normal(size=(1288, 22)), Re=rng.normal(size=(256, 22)))
          for _ in range(3)]
    el.append(SimpleNamespace(L=np.zeros((1288, 22)), R=np.zeros((1288, 22)), Re=np.zeros((256, 22))))
    runs, src = pt.ucl_sources(el)
    assert len(src) == 9 and len(runs) == 9
    assert [s.cls for s in src] == ["left", "right", "rest"] * 3
    assert [s.n for s in src] == [1288, 1288, 256] * 3
    assert all(r.shape[0] == 19 for r in runs)


def test_liu_sources_follow_the_trial_timing():
    """Liu 2024: 8-s trials (2-s instruction, 4-s imagery, 2-s break); imagery [8k + 2, 8k + 6) s of the trial's hand
    (1 left, 2 right); rest = break + next instruction [8k + 6, 8k + 10) s; the last break (2 s) is shorter than an
    imagery period and is dropped."""
    fs = 250.0
    src = pt.liu_sources(np.array([1, 2, 1]), n=int(24 * fs), fs=fs)
    img = [(s.start, s.n, s.cls) for s in src if s.cls != "rest"]
    assert img == [(500, 1000, "left"), (2500, 1000, "right"), (4500, 1000, "left")]
    rest = [(s.start, s.start + s.n) for s in src if s.cls == "rest"]
    assert rest == [(1500, 2500), (3500, 4500)]


def test_sci_sources_follow_the_cues():
    """BNCI 001-2019: cue (776/777/779/925/926) at trial start + 2 s, attempted movement for 3 s; rest = the lead-in
    [0, first cue) and the inter-trial interval plus the next fixation [cue + 3 s, next cue), kept only if >= 3 s."""
    fs = 256.0
    ev = {"768": [10.0, 17.2, 24.5], "777": [12.0], "925": [19.2], "779": [26.5]}
    src = pt.sci_sources(ev, n=int(40 * fs), fs=fs, run=0, tag="r3")
    mv = [(round(s.start / fs, 2), round(s.n / fs, 2), s.cls) for s in src if s.cls != "rest"]
    assert mv == [(12.0, 3.0, "move"), (19.2, 3.0, "move"), (26.5, 3.0, "move")]
    rest = [(round(s.start / fs, 2), round((s.start + s.n) / fs, 2)) for s in src if s.cls == "rest"]
    assert rest == [(0.0, 12.0), (15.0, 19.2), (22.2, 26.5), (29.5, 40.0)]



def test_contrast_table_covers_every_pair():
    for ds in pt.MOTOR_DATASETS:
        for p in MOTOR:
            st, note = pt.contrast(ds, p)
            assert st in ("match", "mismatch") and note
    assert pt.contrast("ucl_als", "cruse11")[0] == "mismatch"
    assert pt.contrast("liu_stroke", "mbmi")[0] == "match"
    assert pt.contrast("sci_bnci", "mbmi")[0] == "mismatch"
    assert pt.UPPER_BOUND == {"sci_bnci"}


def test_class_map_reaches_the_dataset_classes():
    """Every published class of a matching pipeline maps to a class the dataset holds."""
    for ds in pt.MOTOR_DATASETS:
        for p in MOTOR:
            if pt.contrast(ds, p)[0] != "match":
                continue
            mod = mod_of(p)
            ov = mod.schedule(pc.START_S, np.random.default_rng(0), pc.primary_variant(mod))
            got = {c for _, _, c in pc.periods(mod, ov, ds)}
            assert got <= set(pt.CLASSES[ds]), (ds, p, got)


def test_channels_follow_each_module_montage():
    sci = ["AFz", "F3", "F1", "Fz", "F2", "F4", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
           "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P3", "P1", "Pz", "P2", "P4", "POz"]
    ok = {p: pt.channels_ok(mod_of(p), sci)[0] for p in MOTOR}
    assert ok["mbmi"] and ok["cruse12p"]
    assert not ok["columbia"] and not ok["cornell"] and not ok["edlow17"] and not ok["dubois"] and not ok["cruse11"]
    assert all(pt.channels_ok(mod_of(p), list(pc.EEG19) + ["FC3", "FC4", "CP3", "CP4"])[0] for p in MOTOR)



def test_feasibility_row_reports_reuse_contrast_and_channels():
    rng = np.random.default_rng(3)
    names = list(pc.EEG19) + ["FC3", "FC4", "CP3", "CP4"]
    runs, src = [], []
    for c, n_ev, dur in (("left", 30, 4.0), ("right", 30, 4.0), ("rest", 40, 4.0)):
        for k in range(n_ev):
            runs.append((10 * rng.normal(size=(len(names), int(dur * 250)))).astype(np.float32))
            src.append(pc.Source(run=len(runs) - 1, start=0, n=int(dur * 250), cls=c, tag=f"syn:{c}:{k}"))
    person = pc.Person(pid="syn", fs=250.0, names=names, runs=runs, sources=src,
                       meta={"source": "liu_stroke", "units_uv": True})
    rows = pt.feasibility_rows_for(person, "liu_stroke")
    by = {r["pipe"]: r for r in rows}
    assert set(by) == set(MOTOR)
    assert by["cruse11"]["contrast"] == "mismatch"
    assert by["mbmi"]["primary_reuse_max"] >= 1 and by["columbia"]["primary_reuse_max"] > 1
    assert by["mbmi"]["max_slot_s"] > 0
    unk = pc.Person(pid="u", fs=250.0, names=names, runs=runs, sources=src, meta={"source": "ucl_als",
                                                                                   "units_uv": False})
    assert {r["amp_usable"] for r in pt.feasibility_rows_for(unk, "ucl_als") if r["pipe"] == "mbmi"} == {None}



UCL = os.path.join(pt.UCL_DIR, "S21.mat")


@pytest.mark.skipif(not os.path.exists(UCL), reason="UCL ALS not on disk")
def test_real_ucl_person():
    p = pt.ucl_person("S21")
    cls = [s.cls for s in p.sources]
    assert cls.count("left") == cls.count("right") == cls.count("rest") == 159
    assert p.fs == 256.0 and len(p.names) == 19 and "Cz" in p.names


@pytest.mark.skipif(not os.path.exists(pt.LIU_SOURCE), reason="Liu stroke not on disk")
def test_real_liu_person_labels_and_timing():
    p = pt.liu_person(1)
    img = [s for s in p.sources if s.cls != "rest"]
    assert len(img) == 40 and [s.cls for s in img][:4] == ["left", "right", "left", "right"]
    assert p.fs == pt.LIU_FS and all(c in p.names for c in pc.EEG19)


@pytest.mark.skipif(not os.path.exists(os.path.join(pt.SCI_DIR, "P01.zip")), reason="SCI BNCI not on disk")
def test_real_sci_person_counts():
    p = pt.sci_person(1)
    mv = [s for s in p.sources if s.cls == "move"]
    assert len(mv) == 9 * 40 and len(p.runs) == 9
    assert all(abs(s.n / p.fs - 3.0) < 0.01 for s in mv)


def test_feasibility_table_aggregates_people():
    rows = [{"dataset": "liu_stroke", "person": f"p{i}", "pipe": "mbmi", "contrast": "match", "contrast_note": "x",
             "channels": "ok", "avail_s": {"left": 80.0, "rest": 156.0}, "primary_reuse_max": 1 + (i == 0),
             "worst_reuse_max": 2, "primary_demand_ratio": 0.9, "worst_demand_ratio": 1.4, "max_slot_s": 3.75,
             "amp_p": 0.5, "amp_usable": i > 0, "upper_bound": False, "primary_demand_s": {"left": 120.0}}
            for i in range(4)]
    rows.append(dict(rows[0], pipe="cruse11", contrast="mismatch", primary_reuse_max=None, worst_reuse_max=None,
                     primary_demand_ratio=None, worst_demand_ratio=None, max_slot_s=None, amp_usable=None))
    t = pt.feasibility_table(rows)
    c = t["liu_stroke|mbmi"]
    assert c["n_people"] == 4 and c["primary_reuse_max"] == [1, 2] and c["share_reuse_free_primary"] == 0.75
    assert c["amp_usable_share"] == 0.75 and c["channels"] == "ok"
    assert t["liu_stroke|cruse11"]["contrast"] == "mismatch" and "primary_reuse_max" not in t["liu_stroke|cruse11"]



def test_liu_plan_follows_the_decision():
    assert pt.ARM == "liu_stroke"
    assert pt.LIU_PIPES == ("cornell", "mbmi", "dubois", "edlow17", "cruse12p")
    assert pt.LIU_PLAN == {"cornell": ("all", "all"), "mbmi": ("all", "all"), "cruse12p": ("all", "all"),
                           "edlow17": ("all", "primary"), "dubois": ("all", "primary")}
    assert all(pt.contrast("liu_stroke", p)[0] == "match" for p in pt.LIU_PIPES)
    assert pt.people("liu_stroke") == list(range(1, 51))


def _liu_like(rng, pid="Liu_sub-99"):
    """A synthetic person in the Liu layout (40 alternating 4-s hand trials, 4-s rests) on the Liu channels."""
    names = list(pc.EEG19) + ["FCz", "FC3", "FC4", "FT7", "FT8", "CPz", "CP3", "CP4", "TP7", "TP8", "Oz"]
    fs = pt.LIU_FS
    labels = np.tile([1, 2], 20)
    x = (10 * rng.normal(size=(len(names), int(40 * pt.LIU_TRIAL_S * fs)))).astype(np.float32)
    src = pt.liu_sources(labels, x.shape[1], fs)
    return pc.Person(pid=pid, fs=fs, names=names, runs=[x], sources=src,
                     meta={"source": "liu_stroke", "units_uv": True, "events": {(s.run, s.start): s.cls for s in src}})


def test_run_writes_rows_and_resumes(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    loads = []

    def fake_load(dataset, sid):
        loads.append(sid)
        return _liu_like(rng, f"Liu_sub-{sid:02d}")
    monkeypatch.setattr(pt, "load", fake_load)
    out = str(tmp_path)
    pt.run(pipes=("cornell",), given=[7], out_dir=out, versions=("true",))
    rows = [__import__("json").loads(l) for l in open(os.path.join(out, "liu_stroke__cornell.jsonl"))]
    mod = mod_of("cornell")
    assert {r["key"] for r in rows} == {f"liu_stroke|Liu_sub-07|cornell|{d}|{v}|true"
                                        for d, v in pc.jobs_for(mod, "true", "all", "all")}
    assert all(r["arm"] == "liu_stroke" and r["dev"] is False for r in rows)
    assert all(r["status"] == "ok" or r["status"].startswith("na") for r in rows), {r["status"] for r in rows}
    assert all(r["session_reuse_max"] >= 1 for r in rows if r["status"] == "ok")
    pt.run(pipes=("cornell",), given=[7], out_dir=out, versions=("true",))
    assert loads == [7] and len(open(os.path.join(out, "liu_stroke__cornell.jsonl")).readlines()) == len(rows)




@pytest.mark.skipif(not os.path.exists(pt.LIU_EDF), reason="Liu stroke not on disk")
def test_real_liu_edf_is_the_labelled_trials():
    """The continuous EDF is the 40 labelled sourcedata trials end to end (trial k = samples 4000k-4000k+3999 at
    500 Hz, uV, offsets removed): 1-40 Hz correlation on Cz, C3, C4 > 0.95 for the first, a middle and the last trial,
    so the sourcedata labels apply to the EDF trial by trial. No extracted file is left behind."""
    from scipy.signal import butter, sosfiltfilt
    a, b = pt.liu_check(1, trials=(0, 17, 39))
    sos = butter(4, [1, 40], btype="band", fs=500, output="sos")
    for k in range(a.shape[0]):
        fa, fb = sosfiltfilt(sos, a[k], axis=-1), sosfiltfilt(sos, b[k], axis=-1)
        for i in range(fa.shape[0]):
            assert np.corrcoef(fa[i, 200:-200], fb[i, 200:-200])[0, 1] > 0.95
            assert 0.8 < np.std(fa[i]) / np.std(fb[i]) < 1.25
    assert not [f for f in os.listdir(pt.TMP) if not f.startswith("._")] if os.path.isdir(pt.TMP) else True
