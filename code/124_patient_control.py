"""Patient-level positive control: aware patients with brain or cord disease doing instructed tasks, run through the
frozen published pipelines and their randomization-based variants in pseudo-sessions built in each module's own
schedule. Everything about sessions, packing, crossfades, controls, rates and verdicts is IMPORTED from
code/118_positive_control.py (fixed A5 code); this file adds the patient datasets, their contrast match and the
feasibility table. Outside the frozen-code hash; built test-first (code/test_patient_control.py).

Datasets (read-only; zip members are extracted to a temporary folder, read as data with mne / scipy, and deleted):
- ucl_als: UCL RDR 28156016, 8 people with ALS (ucl_io): per struct element one left-hand (L), right-hand (R) and rest
  (Re) trial, 19 EEG columns then 3 EOG, 256 Hz, linked ears. L / R ~5 s, Re ~1 s. Units are NOT microvolts (1-40 Hz
  amplitudes of thousands): amplitude rules cannot be applied (usable = None, "units unknown").
- liu_stroke: Liu 2024 (figshare 21679035, CC BY 4.0), 50 acute stroke patients, one run of 40 8-s trials (2-s
  instruction, 4-s video-guided left / right hand imagery, 2-s break), 500 Hz -> 250 Hz, CPz reference, uV. Labels
  from sourcedata (1 left, 2 right); signal from the continuous EDF.
- sci_bnci: BNCI 001-2019 (CC BY 4.0), 9 people with cervical SCI, ATTEMPTED movements of one arm (5 classes, cue +
  0-3 s) in runs 3-7 and 10-13, 256 Hz, 61 EEG sites (no Fp / F7 / F8 / T / O sites). An UPPER BOUND: attempted
  movement is not imagery.
- BCI-FIT (ds007720, ALS P300 calibration) is not on disk (streamed and deleted by the pathway work): see the
  feasibility report; not built here.

Sources (classes from the real events): ucl_sources, liu_sources, sci_sources. Rest is each contiguous no-task
stretch at least as long as one task period (as 118.openbmi_sources), so one short stretch cannot set the slot grid.
Contrast (CONTRAST): "match" where the dataset's classes realise the published task's contrast (CLASS_MAP maps each
published class to a dataset class through 118.DATASET_CLASS), else "mismatch". Channels (channels_ok): Columbia
(USE_CH), Cornell / Edlow (19 sites), Cruse 2011 (its Laplacian runs over the recording's channels, the 19 sites as on
BDSP) and du Bois (CHANSETS full) need the 19 10-20 sites; mbmi and Cruse 2012 their native or 10-20 set.
Rule (fixed): a pipeline x dataset pair is a positive control only if its sessions are reuse-free
(reuse_max = 1) and the contrast matches; otherwise "reuse-confounded" or "contrast mismatch", reported, never pass or
fail. No pair met it (patient_control_feasibility.json).

Run: ONLY the Liu stroke arm, as an UPPER-BOUND sensitivity (its sessions reuse each trial 3-12
times): cornell, mbmi, dubois, edlow17, cruse12p (LIU_PLAN) on all 50 patients (no development people: no module was
run on them while building). True version: every variant and every timing group's randomized method; control: the
same for the cheap modules, the default variant and its randomized group for edlow17 and du Bois. No Columbia; no UCL,
SCI or BCI-FIT; no montage substitution. Rows, sessions and controls are 118.run_person's; cells are 118.summarize's
(rate, Clopper-Pearson CI, the published healthy-control bar). The verdict is always "upper bound (reuse-inflated)",
never pass; each default-variant call or miss is listed per person with paralysis side and NIHSS.
The Liu EDF is the 40 labelled sourcedata trials end to end in uV (liu_check; the sourcedata keep DC offsets).
python 124_patient_control.py --feasibility   -> output/tables/patient_control_feasibility.json
python 124_patient_control.py --run [--pipe cornell,mbmi] [--people 1,2]   -> output/intermediate/_patient_control/
python 124_patient_control.py --summarize     -> output/tables/patient_control.json (never printed)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import collections
import importlib
import io
import json
import re
import time
import sys
import warnings
import zipfile

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
pc = importlib.import_module("118_positive_control")
mc = pc.mc
ROOT = pc.ROOT

UCL_DIR = os.path.join(ROOT, "output", "intermediate", "raw_ucl")
UCL_FS = 256.0
UCL_EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
           "O1", "O2"]
LIU_DIR = os.path.join(ROOT, "output", "raw", "stroke_liu")
LIU_SOURCE = os.path.join(LIU_DIR, "sourcedata.zip")
LIU_EDF = os.path.join(LIU_DIR, "edffile.zip")
LIU_FS, LIU_TRIAL_S, LIU_MI = 250.0, 8.0, (2.0, 6.0)
LIU_CLASS = {1: "left", 2: "right"}
SCI_DIR = os.path.join(ROOT, "output", "raw", "sci_bnci")
SCI_RUNS = (3, 4, 5, 6, 7, 10, 11, 12, 13)
SCI_MOVE = ("776", "777", "779", "925", "926")
SCI_MOVE_S = 3.0
TMP = os.path.join(ROOT, "output", "intermediate", "_patient_tmp")
FEASIBILITY = os.path.join(ROOT, "output", "tables", "patient_control_feasibility.json")
OUT = os.path.join(ROOT, "output", "intermediate", "_patient_control")
SUMMARY = os.path.join(ROOT, "output", "tables", "patient_control.json")
ARM = "liu_stroke"
LIU_PIPES = ("cornell", "mbmi", "dubois", "edlow17", "cruse12p")
LIU_PLAN = {"cornell": ("all", "all"), "mbmi": ("all", "all"), "cruse12p": ("all", "all"),
            "edlow17": ("all", "primary"), "dubois": ("all", "primary")}
VERDICT = "upper bound (reuse-inflated)"

MOTOR_DATASETS = ("ucl_als", "liu_stroke", "sci_bnci")
UPPER_BOUND = {"sci_bnci"}
CLASSES = {"ucl_als": ("left", "right", "rest"), "liu_stroke": ("left", "right", "rest"), "sci_bnci": ("move", "rest")}
_HANDS = {"left": "left", "right": "right", "both": "right", "feet": None, "rest": "rest"}
CLASS_MAP = {"ucl_als": _HANDS, "liu_stroke": _HANDS,
             "sci_bnci": {"left": "move", "right": "move", "both": "move", "feet": None, "rest": "rest"}}
_IMAG = {
    "columbia": ("match", "left / right hand imagery vs rest for keep / stop (controls moved: imagery is a harder bar)"),
    "cornell": ("match", "right-hand imagery vs rest (CU18 'open/close right hand')"),
    "mbmi": ("match", "left vs right hand imagery"),
    "dubois": ("match", "left vs right hand imagery (two imagined movements)"),
    "cruse11": ("mismatch", "no toe imagery"),
    "edlow17": ("match", "right-hand imagery vs rest"),
    "cruse12p": ("match", "left / right hand imagery vs relax (controls squeezed: imagery is a harder bar)"),
}
_SCI = {
    "columbia": ("mismatch", "one arm only: the published keep / stop alternates hands"),
    "cornell": ("match", "attempted hand / arm movement vs rest (upper bound)"),
    "mbmi": ("mismatch", "one arm only: no left vs right"),
    "dubois": ("mismatch", "one arm only: the published pair is two limbs"),
    "cruse11": ("mismatch", "no toe movement"),
    "edlow17": ("match", "attempted hand movement vs rest (upper bound)"),
    "cruse12p": ("mismatch", "one arm only: no left vs right"),
}
CONTRAST = {(ds, p): v for ds, tab in (("ucl_als", _IMAG), ("liu_stroke", _IMAG), ("sci_bnci", _SCI))
            for p, v in tab.items()}
NINETEEN = {"columbia", "cornell", "edlow17", "cruse11"}


def contrast(dataset, pipe):
    return CONTRAST[(dataset, pipe)]


def _register_classes():
    """118.DATASET_CLASS entries for every matching pipeline: each published label -> its dataset class."""
    for ds in MOTOR_DATASETS:
        for p in pc.MOTOR_PIPES:
            if CONTRAST[(ds, p)][0] != "match":
                continue
            mod = pc.load_pipe(p)
            ov = mod.schedule(pc.START_S, np.random.default_rng(0), pc.primary_variant(mod))
            for lab in ov.meta["resp_hand"]:
                pc.DATASET_CLASS[(ds, p, int(lab))] = CLASS_MAP[ds][pc.class_of(mod, int(lab))]


_register_classes()


def channels_ok(mod, names):
    """(ok, reason): the module's montage on these channels (module docstring)."""
    have = set(mc.canon(names))
    if mod.NAME in NINETEEN:
        want = list(pc.EEG19)
    elif mod.NAME == "dubois":
        want = mc.canon(mod.CHANSETS["full"])
    else:
        nat, sub = mc.canon(mod.NATIVE_CH), mc.canon(mod.SUB_1020)
        want = nat if set(nat) <= have else sub
    miss = [c for c in want if c not in have]
    return (not miss, "ok" if not miss else f"missing {', '.join(miss)}")



def _rest_ok(n, task):
    return n >= task


def ucl_sources(elements):
    """(runs, sources): one run per real trial (19 x n, as stored), class from its field; padding elements (all zero)
    and trials flat on a channel dropped (ucl_io)."""
    runs, src = [], []
    for i, el in enumerate(elements):
        if not any(np.asarray(getattr(el, f)).any() for f in ("L", "R", "Re")):
            continue
        for field, cls in (("L", "left"), ("R", "right"), ("Re", "rest")):
            x = np.asarray(getattr(el, field), float)
            if x.ndim != 2 or x.shape[0] < 2 or not x[:, :19].std(0).all():
                continue
            runs.append(x[:, :19].T.astype(np.float32))
            src.append(pc.Source(run=len(runs) - 1, start=0, n=int(x.shape[0]), cls=cls, tag=f"el{i}:{field}"))
    return runs, src


def ucl_person(sid):
    import scipy.io as sio
    m = sio.loadmat(os.path.join(UCL_DIR, f"{sid}.mat"), squeeze_me=True, struct_as_record=False)
    runs, src = ucl_sources([v for k, v in m.items() if not k.startswith("__")][0])
    return pc.Person(pid=f"UCL_{sid}", fs=UCL_FS, names=list(UCL_EEG), runs=runs, sources=src,
                     meta={"source": "ucl_als", "units_uv": False, "events": {(s.run, s.start): s.cls for s in src}})


def liu_sources(labels, n, fs):
    """Imagery [8k + 2, 8k + 6) s of each trial's hand; rest = break + next instruction [8k + 6, 8k + 10) s, kept only
    if as long as an imagery period (the last break is not)."""
    task = int(round((LIU_MI[1] - LIU_MI[0]) * fs))
    src = []
    for k, lab in enumerate(labels):
        t0 = k * LIU_TRIAL_S
        a = int(round((t0 + LIU_MI[0]) * fs))
        src.append(pc.Source(run=0, start=a, n=task, cls=LIU_CLASS[int(lab)], tag=f"trial{k}:mi"))
        r0, r1 = a + task, min(n, int(round((t0 + LIU_TRIAL_S + LIU_MI[0]) * fs)))
        if _rest_ok(r1 - r0, task):
            src.append(pc.Source(run=0, start=r0, n=r1 - r0, cls="rest", tag=f"trial{k}:rest"))
    return src


def _member(zpath, member, name):
    """One zip member written as a flat file under TMP (no folder tree: exFAT '._' sidecars break folder removal);
    the caller deletes it."""
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, name)
    with open(path, "wb") as fh:
        fh.write(zipfile.ZipFile(zpath).read(member))
    return path


def _remove(path):
    for f in (path, os.path.join(os.path.dirname(path), "._" + os.path.basename(path))):
        if os.path.exists(f):
            os.remove(f)


def _liu_labels(sid):
    import scipy.io as sio
    zs = zipfile.ZipFile(LIU_SOURCE)
    return sio.loadmat(io.BytesIO(zs.read(f"sourcedata/sub-{sid:02d}/sub-{sid:02d}_task-motor-imagery_eeg.mat")),
                       squeeze_me=True, struct_as_record=False)["eeg"]


def _liu_edf(sid):
    """(x uV, channel names as recorded, sfreq) of the continuous EDF, EEG and EOG only."""
    import mne
    path = _member(LIU_EDF, f"edffile/sub-{sid:02d}/eeg/sub-{sid:02d}_task-motor-imagery_eeg.edf", f"liu{sid:02d}.edf")
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        keep = [i for i, c in enumerate(raw.ch_names) if c.strip()]
        x = raw.get_data(picks=keep) * 1e6
        names, fs = [raw.ch_names[i] for i in keep], float(raw.info["sfreq"])
        del raw
    finally:
        _remove(path)
    return x, names, fs


def liu_check(sid, trials=(0, 17, 39), sites=("Cz", "C3", "C4")):
    """(EDF, sourcedata): the same trials (trials x sites x 4000 samples at 500 Hz) from both files, for the alignment
    test."""
    x, names, fs = _liu_edf(sid)
    raw = np.asarray(_liu_labels(sid).rawdata, float)
    n = raw.shape[2]
    rows = [names.index(c) for c in sites]
    return (np.stack([x[rows, k * n:(k + 1) * n] for k in trials]), np.stack([raw[k][rows] for k in trials]))


def liu_person(sid):
    """One Liu stroke patient: labels from sourcedata, the continuous EDF (signals, sfreq, channel names only)."""
    labels = np.asarray(_liu_labels(sid).label, int)
    x, ch, fs0 = _liu_edf(sid)
    keep = [i for i, c in enumerate(ch) if not c.upper().startswith(("HEO", "VEO"))]
    names = mc.canon([{"FP1": "Fp1", "FP2": "Fp2"}.get(ch[i], ch[i]) for i in keep])
    x = mc.resample(x[keep], fs0, LIU_FS).astype(np.float32)
    src = liu_sources(labels, x.shape[1], LIU_FS)
    return pc.Person(pid=f"Liu_sub-{sid:02d}", fs=LIU_FS, names=names, runs=[x], sources=src,
                     meta={"source": "liu_stroke", "units_uv": True, "events": {(s.run, s.start): s.cls for s in src}})


def sci_sources(ev, n, fs, run, tag):
    """Attempted movement [cue, cue + 3 s) of any of the 5 classes ('move'); rest = the lead-in [0, first cue) and each
    [cue + 3 s, next cue) (inter-trial interval + the next fixation) and the tail, kept only if >= 3 s."""
    task = int(round(SCI_MOVE_S * fs))
    cues = sorted(int(round(t * fs)) for c in SCI_MOVE for t in ev.get(c, []))
    src = []
    if cues and _rest_ok(cues[0], task):
        src.append(pc.Source(run=run, start=0, n=int(cues[0]), cls="rest", tag=f"{tag}:lead"))
    for k, a in enumerate(cues):
        src.append(pc.Source(run=run, start=a, n=min(task, n - a), cls="move", tag=f"{tag}:move@{a}"))
        end = cues[k + 1] if k + 1 < len(cues) else n
        if _rest_ok(end - a - task, task):
            src.append(pc.Source(run=run, start=a + task, n=end - a - task, cls="rest", tag=f"{tag}:rest@{a}"))
    return src


def sci_person(sid):
    """One SCI participant: the 9 movement runs (signals, sfreq, channel names, annotations only; NaN tail trimmed)."""
    import mne
    np._set_promotion_state("legacy")
    zp = os.path.join(SCI_DIR, f"P{sid:02d}.zip")
    members = {int(re.search(r"Run (\d+)", n).group(1)): n for n in zipfile.ZipFile(zp).namelist()
               if n.endswith(".gdf") and "__MACOSX" not in n}
    runs, src, names, fs = [], [], None, None
    for r in SCI_RUNS:
        path = _member(zp, members[r], f"sci{sid:02d}_run{r}.gdf")
        try:
            raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
            eeg = [c for c in raw.ch_names if not c.lower().startswith("eog")]
            x = raw.get_data(picks=eeg) * 1e6
            ev = {}
            for o, desc in zip(raw.annotations.onset, raw.annotations.description):
                ev.setdefault(str(desc), []).append(float(o))
            f = float(raw.info["sfreq"])
            del raw
        finally:
            _remove(path)
        bad = np.flatnonzero(np.isnan(x).any(0))
        x = x[:, :bad[0]] if len(bad) else x
        if names is None:
            names, fs = eeg, f
        if eeg != names or f != fs:
            raise ValueError(f"SCI P{sid:02d}: runs differ in channels or sampling rate")
        runs.append(x.astype(np.float32))
        src += sci_sources(ev, x.shape[1], fs, len(runs) - 1, f"run{r}")
    return pc.Person(pid=f"SCI_P{sid:02d}", fs=fs, names=mc.canon(names), runs=runs, sources=src,
                     meta={"source": "sci_bnci", "units_uv": True, "upper_bound": True,
                           "events": {(s.run, s.start): s.cls for s in src}})


def people(dataset):
    if dataset == "ucl_als":
        return sorted(f[:-4] for f in os.listdir(UCL_DIR) if f.endswith(".mat") and not f.startswith("._"))
    if dataset == "liu_stroke":
        return list(range(1, 51))
    return sorted(int(re.search(r"P(\d+)\.zip", f).group(1)) for f in os.listdir(SCI_DIR)
                  if re.fullmatch(r"P0[1-9]\.zip", f))


def load(dataset, pid):
    return {"ucl_als": ucl_person, "liu_stroke": liu_person, "sci_bnci": sci_person}[dataset](pid)



def feasibility_rows_for(person, dataset):
    """One row per motor pipeline: contrast, channels, reuse (118.feasibility on the person's real source lengths, for
    every schedule), the longest slot, seconds per class, amplitude usability (None when the units are not uV)."""
    avail = collections.Counter()
    for s in person.sources:
        avail[s.cls] += s.n / person.fs
    rows = []
    for p in pc.MOTOR_PIPES:
        mod = pc.load_pipe(p)
        st, note = contrast(dataset, p)
        ok, why = channels_ok(mod, person.names)
        row = {"dataset": dataset, "person": person.pid, "pipe": p, "contrast": st, "contrast_note": note,
               "channels": why, "avail_s": {k: round(v, 1) for k, v in avail.items()},
               "primary_reuse_max": None, "worst_reuse_max": None, "primary_demand_ratio": None,
               "worst_demand_ratio": None, "max_slot_s": None, "amp_p": None, "amp_usable": None,
               "upper_bound": dataset in UPPER_BOUND}
        if st == "match":
            scheds, of = pc._schedules(mod)
            res = {k: pc.feasibility(mod, ov, person.sources, person.fs, np.random.default_rng([pc.SEED, 4]),
                                     dataset=dataset) for k, ov in scheds.items()}
            prim = res[of[pc.primary_variant(mod)]]
            used = {c for _, _, c in pc.periods(mod, scheds[of[pc.primary_variant(mod)]], dataset)} | {"rest"}
            c = pc.xfade_samples(person.fs)
            med = min(np.median([s.n for s in person.sources if s.cls == k]) for k in used)
            row.update(primary_reuse_max=prim["reuse_max"], primary_demand_ratio=round(prim["demand_ratio_max"], 3),
                       worst_reuse_max=max(r["reuse_max"] for r in res.values()),
                       worst_demand_ratio=round(max(r["demand_ratio_max"] for r in res.values()), 3),
                       max_slot_s=round((med - c) / person.fs, 2), primary_demand_s=prim["demand_s"])
        if ok and person.meta.get("units_uv"):
            p_ok, usable = pc.amplitude_usable(person, p)
            row.update(amp_p=round(p_ok, 4), amp_usable=usable)
        rows.append(row)
    return rows


def feasibility_table(rows):
    out = {}
    for ds, p in sorted({(r["dataset"], r["pipe"]) for r in rows}):
        rr = [r for r in rows if (r["dataset"], r["pipe"]) == (ds, p)]
        r0 = rr[0]
        cell = {"n_people": len(rr), "contrast": r0["contrast"], "contrast_note": r0["contrast_note"],
                "channels": collections.Counter(r["channels"] for r in rr).most_common(1)[0][0],
                "upper_bound": r0["upper_bound"],
                "avail_s_median": {k: float(np.median([r["avail_s"].get(k, 0) for r in rr]))
                                   for k in sorted({k for r in rr for k in r["avail_s"]})}}
        if r0["contrast"] == "match":
            cell.update(primary_reuse_max=[min(r["primary_reuse_max"] for r in rr),
                                           max(r["primary_reuse_max"] for r in rr)],
                        worst_reuse_max=max(r["worst_reuse_max"] for r in rr),
                        primary_demand_ratio_median=float(np.median([r["primary_demand_ratio"] for r in rr])),
                        share_reuse_free_primary=float(np.mean([r["primary_reuse_max"] == 1 for r in rr])),
                        share_reuse_free_all=float(np.mean([r["worst_reuse_max"] == 1 for r in rr])),
                        max_slot_s_median=float(np.median([r["max_slot_s"] for r in rr])),
                        demand_s_primary=r0.get("primary_demand_s"))
        us = [r["amp_usable"] for r in rr if r["amp_usable"] is not None]
        cell["amp_usable_share"] = float(np.mean(us)) if us else None
        cell["amp_rule"] = (pc.USABILITY.get(p) or {}).get("rule", "none")
        out[f"{ds}|{p}"] = cell
    return out


def write_feasibility(path=FEASIBILITY):
    rows = []
    for ds in MOTOR_DATASETS:
        for pid in people(ds):
            person = load(ds, pid)
            rows += feasibility_rows_for(person, ds)
            print(ds, person.pid, flush=True)
            del person
    with open(path + ".rows.json", "w") as fh:
        json.dump(rows, fh)
    out = {"table": feasibility_table(rows), "rows": rows}
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
    return out



def out_file(pipe, out_dir=OUT):
    return os.path.join(out_dir, f"{ARM}__{pipe}.jsonl")


def job_keys(pid, mod, versions):
    return [f"{ARM}|{pid}|{mod.NAME}|{d}|{v}|{ver}" for ver in versions
            for d, v in pc.jobs_for(mod, ver, *LIU_PLAN[mod.NAME])]


def run(pipes=LIU_PIPES, given=None, out_dir=OUT, versions=pc.VERSIONS):
    """Person by person (each EDF read once), every pending pipeline with its LIU_PLAN scope; resumes from the rows."""
    bad = [p for p in pipes if p not in LIU_PIPES]
    if bad:
        raise SystemExit(f"{bad}: not in the pinned Liu run {LIU_PIPES}")
    who = [int(p) for p in given] if given else people(ARM)
    if not set(who) <= set(people(ARM)):
        raise ValueError(f"unknown Liu patients {sorted(set(who) - set(people(ARM)))}")
    mods = {p: pc.load_pipe(p) for p in pipes}
    os.makedirs(out_dir, exist_ok=True)
    dones, files = {}, {}
    for p in mods:
        dones[p] = pc._done(out_file(p, out_dir))
        files[p] = pc._writer(out_file(p, out_dir), dones[p])
    try:
        for sid in who:
            pid = f"Liu_sub-{sid:02d}"
            todo = [p for p in mods if not all(k in dones[p] for k in job_keys(pid, mods[p], versions))]
            if not todo:
                continue
            person = load(ARM, sid)
            for p in todo:
                t0 = time.perf_counter()
                pc.run_person(person, mods[p], ARM, False, versions, dones[p], files[p][1], *LIU_PLAN[p])
                print(f"{ARM} {p} {person.pid} {time.perf_counter() - t0:.1f} s", flush=True)
            del person
    finally:
        for fh, _ in files.values():
            fh.close()


def _outcome(r):
    st = (r or {}).get("status") or ""
    if r is None:
        return "not run"
    if st == "ok":
        return "call" if r["call"] else "miss"
    return "unusable" if st.startswith("na") else "error"


def _participants():
    import csv
    with open(os.path.join(LIU_DIR, "participants.tsv"), newline="") as fh:
        return {r["Participant_ID"]: r for r in csv.DictReader(fh, delimiter="\t")}


def write_summary(path=SUMMARY, out_dir=OUT):
    """118.summarize cells; per pipeline the default variant and its randomized group beside the bar, reuse, the
    upper-bound verdict, and every patient's outcome on the default variant (true version)."""
    rows = []
    if os.path.isdir(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".jsonl") and not f.startswith("._"):
                with open(os.path.join(out_dir, f)) as fh:
                    rows += [json.loads(l) for l in fh if l.strip()]
    if any(r.get("dev") for r in rows):
        raise RuntimeError("development rows in the patient-control output")
    cells = pc.summarize(rows)["cells"]
    part = _participants()
    head, per, misses = {}, {}, {}
    for pipe in LIU_PIPES:
        mine = [r for r in rows if r["pipe"] == pipe]
        if not mine:
            continue
        mod = pc.load_pipe(pipe)
        prim = pc.primary_variant(mod)
        rep = next(g for g, vs in pc.rand_groups(mod).items() if prim in vs)
        get = lambda d, v, ver: cells.get(f"{ARM}|{pipe}|{d}|{v}|{ver}")
        reuse = [r["session_reuse_max"] for r in mine if r.get("version") == "true"
                 and r.get("session_reuse_max") is not None]
        dp = get("published", prim, "true")
        pub = [c for k, c in cells.items() if k.startswith(f"{ARM}|{pipe}|published|") and k.endswith("|true")]
        rates = [c["rate"] for c in pub if c["rate"] is not None]
        head[pipe] = {"verdict": VERDICT if dp and dp.get("n") else "no usable people",
                      "contrast": contrast(ARM, pipe)[1], "bar": pc.BARS[pipe], "default_variant": prim,
                      "randomized_group": rep, "default_published": dp,
                      "default_randomized": get("randomized", rep, "true"),
                      "control_published": get("published", prim, "control"),
                      "control_randomized": get("randomized", rep, "control"),
                      "variant_rate_range": [min(rates), max(rates)] if rates else None, "n_variants": len(pub),
                      "reuse_max_any": max(reuse) if reuse else None,
                      "reuse_max_median": float(np.median(reuse)) if reuse else None}
        by = {(r["person"], r["design"], r["variant"]): r for r in mine if r["version"] == "true"}
        per[pipe] = {}
        for pid in sorted({r["person"] for r in mine}):
            pr = part.get(pid.split("_", 1)[1], {})
            per[pipe][pid] = {"published": _outcome(by.get((pid, "published", prim))),
                              "randomized": _outcome(by.get((pid, "randomized", rep))),
                              "paralysis": pr.get("ParalysisSide"),
                              "nihss": float(pr["NIHSS"]) if pr.get("NIHSS") not in (None, "") else None}
        misses[pipe] = [pid for pid, o in per[pipe].items() if o["published"] == "miss"]
    out = {"arm": ARM, "role": "upper-bound sensitivity (reuse-inflated); never pass or fail", "cells": cells,
           "headline": head, "per_person": per, "misses": misses, "plan": LIU_PLAN}
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--run", action="store_true", help="the pinned Liu stroke run (LIU_PLAN)")
    ap.add_argument("--pipe", default=",".join(LIU_PIPES))
    ap.add_argument("--people", help="comma-separated Liu patient numbers (1-50)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.feasibility:
        write_feasibility()
        print(FEASIBILITY)
    elif a.run:
        run(tuple(a.pipe.split(",")), a.people.split(",") if a.people else None)
    elif a.summarize:
        write_summary()
        print(SUMMARY)


if __name__ == "__main__":
    main()
