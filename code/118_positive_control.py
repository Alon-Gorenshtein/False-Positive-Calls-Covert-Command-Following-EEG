"""Positive control of the common null benchmark (real
responders; part A, A5, reinstated). The frozen published pipelines (bench/pipe_*.py, imported, never
edited) and their randomization-based variants run on REAL healthy responders, in pseudo-sessions built in each
module's own published schedule from real trials. Outside the frozen-code hash; built test-first on synthetic data and
development people (code/test_positive_control.py).

Data (read-only):
- Motor modules, PRIMARY where reuse-free (team-decision: reuse can manufacture detections): OpenBMI MI (Lee 2019,
  CC0), left / right hand imagery, 100 trials per class per session (3-s fixation, 4-s imagery, 6 +- 1.5-s blank),
  streamed per person (openbmi_fetch: the .mat is cut to MOTOR_CHANNELS at 250 Hz and deleted; run_stream deletes the
  person's files after its pipelines), sessions per pipeline in OPENBMI_SESSIONS (one session where every schedule is
  reuse-free, else 1 + 2). cruse11 is N/A there (no toe imagery). The feasibility table (--feasibility) gives each
  pipeline's reuse and amplitude usability on every dataset.
- Motor modules, SECONDARY: EEGMMIDB (PhysioNet, ODC-By), 160 Hz.
  Arm "imagery" = runs 4/8/12 (T1 left fist, T2 right fist) and 6/10/14 (T1 both fists, T2 both feet), from the
  local MNE cache; arm "execution" = runs 3/7/11 and 5/9/13 (same codes, real movement), reported separately. T0 =
  rest in every run. Subjects 88, 89, 92, 100 are excluded (eeg_io.EXCLUDE: 128 Hz or corrupt).
- ERP modules (mbvt, chennu13, gibson16): Kojima & Kanoh 2024B two-stream task (CC0; output/raw/passive), where the
  participant COUNTS one deviant of one stream per ~90-s trial (24 trials per person, 12 with the low and 12 with the
  high deviant counted). Only the attended stream is the stimulus sequence (SOA 0.6 s); the interleaved unattended
  stream stays in the signal as background. Signals are resampled to 250 Hz (Chennu's rate), the 27 channels any ERP
  module reads.
- sokoliuk21: not applicable (passive speech tracking needs isochronous speech; no counting data carry it).

Pseudo-sessions (one per schedule draw):
- Motor: the module's own schedule() / schedule_randomized() overlay is passed to the module UNCHANGED. Every onset's
  response period [onset + resp_delay_s, onset + resp_delay_s + resp_dur_s] (the module's own responder spec,
  truncated at the next period) is filled with real trials of its class (class_of: the overlay's resp_hand, 'none'
  = rest; cruse11 toe imagery = both feet); everything else (commands, pauses) with real rest. The splice grid is a
  function of the schedule's timing only: every period start and end is a splice, whatever the classes on either
  side, and stretches longer than a source trial allows are cut into equal parts. So splices fall at the same
  positions for every class (and both designs) and cannot carry the label.
- ERP: the module's schedule fixes the design (mbvt: 60 groups of 8 and the target wrist per instruction (VT3) or per
  group (randomized); chennu13: n_blocks and the explicit word per block; gibson16: 14 blocks and the target wrist
  per block); each group / block is filled with a CONTIGUOUS real stretch from a trial whose REAL instruction matches
  it (low deviant = left wrist / YES, high = right wrist / NO), with the timing as recorded (the module's SOA cannot be
  imposed on real ERPs). mbvt groups are 8 consecutive attended tones with exactly 1 counted target + 7 standards
  (VT2) or 1 low + 1 high deviant + 6 standards (VT3); chennu13 / gibson16 blocks are whole trials. Labels: the
  counted deviant = target / explicit, the other deviant of the attended stream = non-target / implicit, standards =
  non-targets (VT2) / distractors / back.
- Every source piece is demeaned and joined by an equal-power (sin / cos) crossfade of XFADE_S: no step, and no
  variance dip for independent pieces (tests check both, and that both appear when the crossfade is removed or made
  linear).
- Reuse-free packing (allocate): every slot takes a disjoint excerpt from the start of a randomly drawn source's
  unused stretch; only when no source of the class has room does the pool start a new cycle (reuse). reuse_max = the
  most times any real sample is used; demand_ratio = demanded / available seconds. The longest slot is the shortest
  real class's median source length (the same for the control). EEGMMIDB holds ~92 s of imagery per class per person
  (reuse 3-8 for every schedule); OpenBMI 400 s per class per session.
- Null-shuffled control (version "control"): the same construction with the class labels permuted over SOURCE units
  (motor: trial events, counts kept; ERP: which deviant each trial counted), so every copy of a reused source carries
  the same pseudo-class, as in the true session. Splices, reuse and timing are as in the true session; only the link
  between label and real task is broken. A randomized-design control rate above 5% means the construction itself
  (reuse, splicing) creates calls; oddball contrasts (VT2 target vs standards, gibson16 P3a and SSEP, chennu13 vs its
  own baseline) stay positive under it by design (a physical deviant), so the control is a volition null only for
  the counterbalanced contrasts.

Runs: per person, pipeline and version, every native variant's run() on its published-schedule session (variants
whose schedule draw is identical share one session; the draw is seeded per person and pipeline, as in 114), and the
randomization-based method (bench.wrappers.run_randomized, B = 200) once per timing group (114.timing_groups; mbvt:
per paradigm, the timing variants are moot under recorded timing) on its schedule_randomized() session.
Bars: each pipeline's published healthy-control rate, read from docs/benchmark-k-table.json (BARS cites the
fraction; a test checks it appears in the table). Detection rate = called / usable people, Clopper-Pearson 95% CI;
fidelity concern if the upper bound is below the bar. A person whose session the module cannot analyse (its own
NoUsableTrials, e.g. a +-100 uV artefact rule on EEGMMIDB's high-amplitude recordings) is counted as unusable and left
out of the denominator, as in the benchmark. Development people (3 EEGMMIDB subjects drawn with SEED_DEV;
Kojima L and O, the frozen passive dev split) are run for plumbing and never reported.
Scope: --scope all (default) runs every native variant and every timing group; --scope primary only the primary
variant and its group's randomized method; --control-scope the same for the control version; --sample N a seeded
(SAMPLE_SEED) subsample of the analysis people. PLAN holds the per-pipeline scope and sample of the fixed run plan
(run_stream applies it on OpenBMI).
python 118_positive_control.py --arm imagery|execution|erp --pipe NAME [--dev] [--people S001,S002] [--limit N]
                               [--versions true,control] [--scope all|primary] [--control-scope all|primary]
python 118_positive_control.py --dev|... --stream --arm openbmi --pipe cornell,mbmi,cruse12p,edlow17,dubois,columbia
python 118_positive_control.py --fetch-openbmi 11,23 --sessions 1,2 | --feasibility --sessions 1,2 | --summarize
-> output/intermediate/_positive_control[_dev]/<arm>__<pipe>.jsonl; output/tables/positive_control.json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import collections
import importlib
import json
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from bench import engine as en
from bench import motor_common as mc
from bench import wrappers as wr

rn = importlib.import_module("114_benchmark")
eeg_io = importlib.import_module("eeg_io")

K_TABLE = os.path.join(ROOT, "docs", "benchmark-k-table.json")
MMIDB_DIR = os.path.join(os.path.expanduser("~"), "mne_data", "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
EXEC_DIR = os.path.join(ROOT, "output", "raw", "eegmmidb_exec")
KOJIMA_DIR = os.path.join(ROOT, "output", "raw", "passive", "kojima2024b_2stream")
KOJIMA_EVENTS = os.path.join(KOJIMA_DIR, "events_index.parquet")
OUT = {False: os.path.join(ROOT, "output", "intermediate", "_positive_control"),
       True: os.path.join(ROOT, "output", "intermediate", "_positive_control_dev")}
SUMMARY = os.path.join(ROOT, "output", "tables", "positive_control.json")
FEASIBILITY = os.path.join(ROOT, "output", "tables", "positive_control_feasibility.json")

SEED, SEED_DEV, DEV_N = 20261028, 20261029, 3
START_S, TAIL_S = 10.0, 10.0
XFADE_S = 0.25
ERP_FS = 250.0
ERP_PAD_S, ERP_EDGE_S = {"mbvt": (1.0, 1.6), "chennu13": (4.0, 4.0), "gibson16": (4.0, 4.0)}, 10.0
ARMS = ("imagery", "execution", "openbmi", "erp")
VERSIONS = ("true", "control")
MOTOR_PIPES = ("columbia", "cornell", "mbmi", "dubois", "cruse11", "edlow17", "cruse12p")
ERP_PIPES = ("mbvt", "chennu13", "gibson16")
NA_PIPES = {"sokoliuk21": "N/A: passive speech tracking needs isochronous speech; the counting data carry none "
                          "(as the adapters rule for N-passive)"}
RUNS = {"imagery": {4: ("left", "right"), 8: ("left", "right"), 12: ("left", "right"),
                    6: ("both", "feet"), 10: ("both", "feet"), 14: ("both", "feet")},
        "execution": {3: ("left", "right"), 7: ("left", "right"), 11: ("left", "right"),
                      5: ("both", "feet"), 9: ("both", "feet"), 13: ("both", "feet")}}
CLASS_OVERRIDE = {("cruse11", 2): "feet"}
DATASET_CLASS = {("openbmi", "cornell", 1): "right"}
SAMPLE_SEED = 20261014
EEG19 = mc.canon(mc.EEG19)
MMIDB64 = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3',
           'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
           'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3',
           'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']
ERP_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC1', 'FC2', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP1', 'CPz',
                'CP2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'PO8', 'O1', 'Oz', 'O2']
KOJIMA_TONES = (1, 2, 101, 102, 103, 104, 111, 112, 113, 114)
KOJIMA_STREAM = {201: 1, 202: 1, 203: 2, 204: 2}
KOJIMA_TARGET_ID = {201: "low", 202: "high", 203: "low", 204: "high"}
KOJIMA_IDENT = {101: "low", 103: "low", 111: "low", 113: "low", 102: "high", 104: "high", 112: "high", 114: "high"}

BARS = {
    "columbia": [{"k": 10, "n": 10, "applies": None, "note": "NEJM19: controls moved for real (execution is the "
                                                             "matched arm; imagery is a harder bar)"}],
    "cornell": [{"k": 15, "n": 15, "applies": None, "note": "CU18 imagery, >= 1 paradigm"}],
    "mbmi": [{"k": 3, "n": 3, "applies": None, "note": "GU17 (erratum: 3/3 participants, 5/6 runs); "
                                                       "CH18 2/10 reported beside it"}],
    "dubois": [{"k": 2, "n": 2, "applies": None, "note": "DB26 able-bodied benchmarks, Phase I"}],
    "cruse11": [{"k": 9, "n": 12, "applies": None, "note": "CR11 controls"}],
    "edlow17": [{"k": 12, "n": 16, "applies": None, "note": "ED17 (erratum: 12/16)"}],
    "cruse12p": [{"k": 5, "n": 6, "applies": None, "note": "CR12P controls squeezed for real (execution is the "
                                                           "matched arm)"}],
    "mbvt": [{"k": 3, "n": 3, "applies": None, "note": "GUG17 regular VT2 / VT3 runs [derived]"}],
    "chennu13": [{"k": 8, "n": 8, "applies": None, "note": "CH13 P3b to explicit targets"}],
    "gibson16": [{"k": 10, "n": 15, "applies": {"call_on": "p3b"}, "note": "GIB16 P3b"},
                 {"k": 15, "n": 15, "applies": {"call_on": "p3a"}, "note": "GIB16 P3a"},
                 {"k": 15, "n": 15, "applies": {"call_on": "ssep"}, "note": "GIB16 SSEP"}],
}
PRIMARY = {"columbia": "openbmi", "cornell": "openbmi", "mbmi": "openbmi", "dubois": "openbmi", "edlow17": "openbmi",
           "cruse12p": "openbmi", "cruse11": None, "mbvt": "erp", "chennu13": "erp", "gibson16": "erp"}
SECONDARY_BARS = {"mbmi": {"k": 2, "n": 10, "note": "CH18 (ICU, mindBEAGLE)"},
                  "chennu13": {"k": 5, "n": 8, "note": "CH13 P3a"}}



@dataclass
class Source:
    """One real trial: samples [start, start + n) of run `run`, class from its task event, tag = provenance."""
    run: int
    start: int
    n: int
    cls: str
    tag: str


@dataclass
class Person:
    pid: str
    fs: float
    names: list
    runs: list
    sources: list
    meta: dict = field(default_factory=dict)


@dataclass
class Trial:
    """One Kojima trial: attended-stream tones (samples of run `run` at the person's fs), their real kind ('target' =
    counted, 'other' = the other deviant of the attended stream, 'standard'), deviant identity and event codes;
    target = the identity of the counted deviant; [lo, hi) = the run's samples."""
    run: int
    tid: str
    target: str
    on: np.ndarray
    kind: np.ndarray
    ident: np.ndarray
    code: np.ndarray
    lo: int = 0
    hi: int = 0


@dataclass
class ErpPerson:
    pid: str
    fs: float
    names: list
    runs: list
    trials: list
    meta: dict = field(default_factory=dict)


@dataclass
class Session:
    rec: object
    overlay: object
    plan: list
    info: dict


def _crc(s):
    return zlib.crc32(str(s).encode())


def _rng(*parts):
    return np.random.default_rng([SEED] + [p if isinstance(p, int) else _crc(p) for p in parts])



def load_pipe(name):
    return importlib.import_module(f"bench.pipe_{name}")


def primary_variant(mod):
    """The module's DEFAULT, else run()'s default (114.primary_variant on a non-BDSP source)."""
    return rn.primary_variant(mod, "eegmmidb")


_CLASS_MAP = {}


def class_of(mod, label, dataset=None):
    """Real-trial class that fills the response period of a label: the schedule's resp_hand ('none' = rest), with
    CLASS_OVERRIDE (cruse11 toe imagery -> both feet) and DATASET_CLASS (a dataset without the class)."""
    if mod.NAME not in _CLASS_MAP:
        ov = mod.schedule(START_S, np.random.default_rng(0), primary_variant(mod))
        hands = ov.meta["resp_hand"]
        _CLASS_MAP[mod.NAME] = {int(k): ("rest" if h == "none" else h) for k, h in hands.items()}
    if (dataset, mod.NAME, int(label)) in DATASET_CLASS:
        return DATASET_CLASS[(dataset, mod.NAME, int(label))]
    if (mod.NAME, int(label)) in CLASS_OVERRIDE:
        return CLASS_OVERRIDE[(mod.NAME, int(label))]
    return _CLASS_MAP[mod.NAME][int(label)]


def module_channels(mod, names):
    """Channels the module's Recording carries: its native set when every native site is present, else its 10-20
    substitute, always within the 19 10-20 sites plus those (cruse11's Laplacian runs over every channel of the
    recording, as on BDSP's 19)."""
    have = mc.canon(names)
    nat = mc.canon(mod.NATIVE_CH)
    if mod.FAMILY == "cmd":
        extra = [c for c in nat if c not in EEG19] if all(c in have for c in nat) else []
        want = EEG19 + extra
    else:
        sub = mc.canon(getattr(mod, "SUB_1020", mod.NATIVE_CH))
        want = nat if all(c in have for c in nat) else sub
    missing = [c for c in want if c not in have]
    if missing:
        raise ValueError(f"{mod.NAME}: channels {missing} missing")
    return [have.index(c) for c in want], want



def xfade_samples(fs, xfade_s=XFADE_S):
    return 2 * int(round(xfade_s * fs / 2))


def _ramps(c, kind):
    u = (np.arange(c) + 0.5) / c
    if kind == "linear":
        return u, 1 - u
    return np.sin(np.pi / 2 * u), np.cos(np.pi / 2 * u)


def join(pieces, c, kind="equal_power"):
    """Pieces (channels x samples) joined in order, consecutive pieces overlapping by c samples under a crossfade
    (equal_power: sin / cos, constant power for independent pieces; linear: for tests). Length = sum - c (n - 1)."""
    n = sum(p.shape[1] for p in pieces) - c * (len(pieces) - 1)
    out = np.zeros((pieces[0].shape[0], n))
    up, dn = _ramps(c, kind) if c else (None, None)
    pos = 0
    for k, p in enumerate(pieces):
        w = np.ones(p.shape[1])
        if c and k > 0:
            w[:c] = up
        if c and k < len(pieces) - 1:
            w[-c:] = dn
        out[:, pos:pos + p.shape[1]] += p * w
        pos += p.shape[1] - c
    return out


def _place(out, piece, a, c, first, last, up, dn):
    """Add one demeaned piece whose sample c/2 falls on output sample a (crossfade ramps at inner ends)."""
    p = piece - piece.mean(1, keepdims=True)
    w = np.ones(p.shape[1])
    if c and not first:
        w[:c] = up
    if c and not last:
        w[-c:] = dn
    i0 = a - c // 2
    j0, j1 = max(0, -i0), p.shape[1] - max(0, i0 + p.shape[1] - out.shape[1])
    out[:, i0 + j0: i0 + j1] += (p[:, j0:j1] * w[j0:j1]).astype(out.dtype)



def periods(mod, ov, dataset=None):
    """[(start s, end s, class)] of every onset's response period, truncated at the next period's start."""
    on = np.asarray(ov.onsets_s, float)
    d = np.broadcast_to(np.asarray(ov.meta.get("resp_delay_s", 0.0), float), on.shape)
    dur = np.broadcast_to(np.asarray(ov.meta["resp_dur_s"], float), on.shape)
    a = on + d
    order = np.argsort(a, kind="stable")
    a, b, lab = a[order], (on + d + dur)[order], np.asarray(ov.labels)[order]
    b = np.minimum(b, np.r_[a[1:], np.inf])
    return [(float(x), float(y), class_of(mod, int(l), dataset)) for x, y, l in zip(a, b, lab)]


def motor_slots(per, fs, t_end_s, max_len, min_len):
    """Contiguous slots [a, b) (samples) from 0 to t_end: a splice at every period start and end; class = the period's
    or rest; stretches longer than max_len cut into equal parts. Depends on timing only."""
    T = int(np.ceil(t_end_s * fs))
    pts = sorted({0, T} | {int(round(x * fs)) for a, b, _ in per for x in (a, b) if 0 < round(x * fs) < T})
    starts = np.array([int(round(a * fs)) for a, _, _ in per])
    ends = np.array([int(round(b * fs)) for _, b, _ in per])
    slots = []
    for a, b in zip(pts[:-1], pts[1:]):
        mid = (a + b) / 2
        k = np.flatnonzero((starts <= mid) & (mid < ends))
        cls = per[k[0]][2] if len(k) else "rest"
        m = int(np.ceil((b - a) / max_len))
        cuts = np.round(np.linspace(a, b, m + 1)).astype(int)
        for x, y in zip(cuts[:-1], cuts[1:]):
            if y - x < min_len:
                raise ValueError(f"slot of {y - x} samples at {x / fs:.2f} s is shorter than {min_len} (crossfade)")
            slots.append({"a": int(x), "b": int(y), "cls": cls})
    return slots


def pools_for(sources, classes, rng, control):
    """{class: [source index]} over the sources of the classes; control: class labels permuted over the sources
    (counts kept), so each source sits in exactly one pseudo-class pool."""
    idx = [i for i, s in enumerate(sources) if s.cls in classes]
    lab = np.array([sources[i].cls for i in idx])
    if control:
        lab = rng.permutation(lab)
    pools = {c: [i for i, l in zip(idx, lab) if l == c] for c in classes}
    empty = [c for c, p in pools.items() if not p]
    if empty:
        raise ValueError(f"no source trials of class(es) {empty}")
    return pools


class _Drawer:
    """Sources of each pool in a seeded order without replacement; a used-up pool restarts in a fresh order."""

    def __init__(self, pools, rng):
        self.pools, self.rng = pools, rng
        self.queue = {c: collections.deque() for c in pools}
        self.uses = collections.Counter()

    def next(self, c):
        if not self.queue[c]:
            self.queue[c].extend(int(i) for i in self.rng.permutation(self.pools[c]))
        i = self.queue[c].popleft()
        self.uses[i] += 1
        return i


def allocate(slots, sources, pools, rng, c):
    """Reuse-free packing: every slot of class k takes a disjoint excerpt of need = (b - a) + c samples from a source of
    pool k, cut from the start of that source's unused stretch (so a long source hosts several excerpts, a 4-s trial
    one); the source is drawn at random among those with room. Only when no source of the pool has room does the pool
    start a new cycle with every source free again (reuse). Sets slot src / off. -> stats: reuse_max = the most times
    any source sample is used (the highest cycle), demand_ratio per class = demanded / available samples."""
    used, cycle, need_tot = {}, {k: 1 for k in pools}, collections.Counter()
    for s in slots:
        k, need = s["cls"], s["b"] - s["a"] + c
        room = [i for i in pools[k] if sources[i].n - used.get(i, 0) >= need]
        if not room:
            cycle[k] += 1
            for i in pools[k]:
                used[i] = 0
            room = [i for i in pools[k] if sources[i].n >= need]
            if not room:
                raise ValueError(f"no {k} source holds {need} samples")
        i = room[int(rng.integers(len(room)))]
        off = used.get(i, 0)
        used[i] = off + need
        need_tot[k] += need
        s.update(src=int(i), off=int(off))
    avail = {k: sum(sources[i].n for i in p) for k, p in pools.items()}
    ratio = {k: need_tot[k] / avail[k] for k in need_tot}
    return {"reuse_max": int(max(cycle[k] for k in need_tot)), "demand_ratio": ratio,
            "demand_ratio_max": float(max(ratio.values())), "n_sources": len({s["src"] for s in slots})}


def check_provenance(person):
    """Refuse a person whose source classes do not follow its recorded events (meta 'events': {(run, start): class};
    synthetic people carry their class in the tag 'syn:<class>:<k>')."""
    ev = person.meta.get("events")
    for s in person.sources:
        want = ev.get((s.run, s.start)) if ev is not None else s.tag.split(":")[1] if s.tag.startswith("syn:") else None
        if want is None or want != s.cls:
            raise ValueError(f"{person.pid}: source {s.tag} labelled {s.cls!r}, its event says {want!r}")


def _motor_layout(mod, ov, sources, fs, rng, control, xfade_s, dataset=None):
    """(slots, pools, c) of a motor overlay on a person's sources (shared by motor_session and feasibility)."""
    c = xfade_samples(fs, xfade_s)
    per = periods(mod, ov, dataset)
    classes = sorted({p[2] for p in per} | {"rest"})
    real = pools_for(sources, classes, rng, False)
    pools = pools_for(sources, classes, rng, True) if control else real
    max_len = int(min(np.median([sources[i].n for i in p]) for p in real.values())) - c
    t_end = max(float(mod.extent(ov)[1]), per[-1][1]) + TAIL_S
    return motor_slots(per, fs, t_end, max_len, max(2 * c, 2)), pools, c


def feasibility(mod, ov, sources, fs, rng, control=False, xfade_s=XFADE_S, dataset=None):
    """allocate() on source lengths alone: the reuse a session of this overlay would have (no signal read)."""
    slots, pools, c = _motor_layout(mod, ov, sources, fs, rng, control, xfade_s, dataset)
    st = allocate(slots, sources, pools, rng, c)
    demand = collections.Counter()
    for s in slots:
        demand[s["cls"]] += (s["b"] - s["a"]) / fs
    return dict(st, demand_s={k: round(v, 1) for k, v in demand.items()},
                avail_s={k: round(sum(sources[i].n for i in p) / fs, 1) for k, p in pools.items()})


def motor_session(person, mod, ov, rng, control=False, xfade_s=XFADE_S):
    """The pseudo-session of a motor module's overlay (module docstring); the overlay itself goes to the module."""
    check_provenance(person)
    fs = person.fs
    dataset = person.meta.get("source")
    slots, pools, c = _motor_layout(mod, ov, person.sources, fs, rng, control, xfade_s, dataset)
    st = allocate(slots, person.sources, pools, rng, c)
    rows, names = module_channels(mod, person.names)
    T = slots[-1]["b"]
    x = np.zeros((len(rows), T), np.float32)
    up, dn = _ramps(c, "equal_power") if c else (None, None)
    for k, s in enumerate(slots):
        src = person.sources[s["src"]]
        piece = np.asarray(person.runs[src.run][rows, src.start + s["off"]: src.start + s["off"] + s["b"] - s["a"] + c],
                           float)
        _place(x, piece, s["a"], c, k == 0, k == len(slots) - 1, up, dn)
    demand = collections.Counter()
    for s in slots:
        demand[s["cls"]] += (s["b"] - s["a"]) / fs
    avail = {cl: sum(person.sources[i].n for i in p) / fs for cl, p in pools.items()}
    info = {"control": bool(control), "n_slots": len(slots), "n_sources": st["n_sources"], "reuse_max": st["reuse_max"],
            "demand_ratio_max": float(st["demand_ratio_max"]),
            "demand_s": {k: round(v, 1) for k, v in demand.items()}, "avail_s": {k: round(v, 1) for k, v in avail.items()},
            "xfade_s": xfade_s, "seconds": T / fs}
    rec = en.Recording(source=dataset or "eegmmidb", uid=f"{person.pid}#{mod.NAME}", patient=person.pid, x=x, fs=fs,
                       names=names, meta={"pseudo_session": True, "control": bool(control)})
    return Session(rec, ov, slots, info)



WRIST_ID = {1: "low", 2: "high"}
WORD_ID = {1: "low", 2: "high"}


def _relabel(trials, rng, control):
    """Per trial (kind, counted identity): real, or (control) the counted identity permuted over trials (counts kept)
    and the kinds re-derived from it."""
    real = [t.target for t in trials]
    tgt = list(rng.permutation(real)) if control else real
    out = []
    for t, g in zip(trials, tgt):
        kind = np.where(t.ident == g, "target", np.where(t.ident == "", "standard", "other"))
        out.append((kind, g))
    return out


def _windows(ok, n, L):
    out, i = [], 0
    while i + L <= n:
        if ok(i, i + L):
            out.append(i)
            i += L
        else:
            i += 1
    return out


def _mbvt_units(trials, lab, paradigm, mod):
    """{identity: [(trial index, first tone, last tone + 1)]}: VT2 = 1 counted target + 7 standards (from trials
    counting the high deviant, the 'right wrist'); VT3 = 1 low + 1 high deviant + 6 standards, keyed by the trial's
    counted identity. Earliest-first, non-overlapping within a trial."""
    G = mod.GROUP
    units = {"low": [], "high": []}
    for ti, (t, (kind, g)) in enumerate(zip(trials, lab)):
        if paradigm == "vt2":
            if g != "high":
                continue
            ok = lambda a, b: (kind[a:b] == "target").sum() == 1 and (kind[a:b] == "standard").sum() == G - 1
        else:
            ok = lambda a, b: ((t.ident[a:b] == "low").sum() == 1 and (t.ident[a:b] == "high").sum() == 1
                               and (kind[a:b] == "standard").sum() == G - 2)
        units[g] += [(ti, a, a + G) for a in _windows(ok, len(t.on), G)]
    return units


def _block_units(trials, lab):
    units = {"low": [], "high": []}
    for ti, (t, (kind, g)) in enumerate(zip(trials, lab)):
        units[g].append((ti, 0, len(t.on)))
    return units


def erp_session(person, mod, tmpl, rng, control=False, xfade_s=XFADE_S):
    """The pseudo-session of an ERP module's schedule draw `tmpl` (module docstring): -> Session whose overlay carries
    the module's design structure on the real recorded onsets."""
    name, fs = mod.NAME, person.fs
    c = xfade_samples(fs, xfade_s)
    lab = _relabel(person.trials, rng, control)
    if name == "mbvt":
        para = tmpl.meta["paradigm"]
        tw = np.asarray(tmpl.meta["target_wrist"])
        want = ["high"] * len(tw) if para == "vt2" else [WRIST_ID[int(w)] for w in tw]
        units = _mbvt_units(person.trials, lab, para, mod)
    elif name == "chennu13":
        expl = np.asarray(tmpl.meta["explicit_word"])
        want = [WORD_ID[int(w)] for w in expl]
        units = _block_units(person.trials, lab)
    elif name == "gibson16":
        tw = np.asarray(tmpl.meta["target_wrist"])
        want = [WRIST_ID[int(w)] for w in tw]
        units = _block_units(person.trials, lab)
    else:
        raise ValueError(f"{name}: no ERP pseudo-session ({NA_PIPES.get(name, 'not an ERP module here')})")
    pre, post = (int(round(s * fs)) for s in ERP_PAD_S[name])
    edge = int(round(ERP_EDGE_S * fs))
    flat = [u for g in sorted(units) for u in units[g]]
    ids, k0 = {}, 0
    for g in sorted(units):
        ids[g] = list(range(k0, k0 + len(units[g])))
        k0 += len(units[g])
    draw = _Drawer(ids, rng)
    picked = []
    for k, g in enumerate(want):
        if not units[g]:
            raise ValueError(f"{person.pid}: no {name} unit from trials counting the {g} deviant")
        tried = 0
        while True:
            j = draw.next(g)
            ti, a, b = flat[j]
            t = person.trials[ti]
            lo = t.on[a] - (edge if k == 0 else pre) - c // 2
            hi = t.on[b - 1] + (edge if k == len(want) - 1 else post) + c // 2
            if lo >= t.lo and hi <= t.hi:
                break
            draw.uses[j] -= 1
            tried += 1
            if tried > 2 * len(units[g]):
                raise ValueError(f"{person.pid}: no {name} unit with {ERP_EDGE_S} s of data at the session edge")
        picked.append((k, g, ti, a, b, lo, hi))
    rows, names = module_channels(mod, person.names)
    T = sum(hi - lo - c for _, _, _, _, _, lo, hi in picked)
    x = np.zeros((len(rows), T), np.float32)
    up, dn = _ramps(c, "equal_power") if c else (None, None)
    on, labels, meta, plan = [], [], collections.defaultdict(list), []
    pos = 0
    for k, g, ti, a, b, lo, hi in picked:
        t = person.trials[ti]
        kind, cnt = lab[ti]
        piece = np.asarray(person.runs[t.run][rows, lo:hi], float)
        _place(x, piece, pos, c, k == 0, k == len(picked) - 1, up, dn)
        for q in range(a, b):
            on.append((pos + (t.on[q] - lo) - c // 2) / fs)
            labels.append(_label(mod, tmpl, k, kind[q], t.ident[q]))
            meta["real_kind"].append(t.kind[q])
            meta["ident"].append(t.ident[q])
            meta["unit"].append(k)
            meta["pos"].append(q - a)
        plan.append({"a": pos, "b": pos + hi - lo - c, "unit": k, "want": g, "trial": t.tid, "counted": cnt,
                     "real_target": t.target, "n_stim": b - a})
        pos += hi - lo - c
    ov = _erp_overlay(mod, tmpl, np.array(on), np.array(labels, int), meta)
    t0, t1 = mod.extent(ov)
    if t0 < 0 or t1 > T / fs:
        raise ValueError(f"{name}: extent {t0:.1f}-{t1:.1f} s outside the {T / fs:.1f}-s session")
    info = {"control": bool(control), "n_units": len(picked), "reuse_max": int(max(draw.uses.values())),
            "n_sources": int(sum(v > 0 for v in draw.uses.values())), "xfade_s": xfade_s, "seconds": T / fs,
            "n_trials_used": len({p["trial"] for p in plan})}
    rec = en.Recording(source="kojimaB", uid=f"{person.pid}#{name}", patient=person.pid, x=x, fs=fs, names=names,
                       meta={"pseudo_session": True, "control": bool(control)})
    return Session(rec, ov, plan, info)


def _label(mod, tmpl, unit, kind, ident):
    if mod.NAME == "mbvt":
        if tmpl.meta["paradigm"] == "vt2":
            return mod.TARGET if kind == "target" else mod.NONTARGET
        return mod.TARGET if kind == "target" else mod.NONTARGET if kind == "other" else mod.DISTRACTOR
    if mod.NAME == "chennu13":
        return mod.EXPL if kind == "target" else mod.IMPL if kind == "other" else mod.DIST
    return mod.TARGET if kind == "target" else mod.NONTARGET if kind == "other" else mod.BACK


def _erp_overlay(mod, tmpl, on, labels, m):
    n = len(on)
    unit, ident = np.array(m["unit"]), np.array(m["ident"])
    meta = {"real_kind": np.array(m["real_kind"]), "dataset": "kojimaB", "run": np.zeros(n, int)}
    if mod.NAME == "mbvt":
        para = tmpl.meta["paradigm"]
        if para == "vt2":
            site = np.where(labels == mod.TARGET, mod.RIGHT, mod.LEFT)
        else:
            site = np.where(ident == "low", mod.LEFT, np.where(ident == "high", mod.RIGHT, mod.OTHER))
        per = mod.N_GROUPS // mod.N_INSTR
        meta |= {"group": unit, "pos": np.array(m["pos"]), "site": site, "block": unit // per, "trial": unit,
                 "target_wrist": np.asarray(tmpl.meta["target_wrist"]).copy(), "paradigm": para}
    elif mod.NAME == "chennu13":
        word = np.where(ident == "low", mod.YES, np.where(ident == "high", mod.NO, 0))
        trial = np.array(m["pos"])
        snum = np.zeros(n, int)
        for i in range(n):
            snum[i] = 0 if word[i] else (snum[i - 1] if trial[i] else 0) + 1
        meta |= {"block": unit, "trial": trial, "word": word, "snum": snum,
                 "explicit_word": np.asarray(tmpl.meta["explicit_word"]).copy()}
    else:
        site = np.where(ident == "low", mod.LEFT, np.where(ident == "high", mod.RIGHT, mod.UPPER_BACK))
        meta |= {"block": unit, "site": site, "target_wrist": np.asarray(tmpl.meta["target_wrist"]).copy(),
                 "trial": np.arange(n)}
    return mc.Overlay(onsets_s=on, labels=labels, kind="stim", meta=meta)



def mmidb_subjects():
    return [s for s in range(1, 110) if s not in eeg_io.EXCLUDE]


def mmidb_file(sid, run):
    """The run's EDF: the local MNE cache (imagery runs), else output/raw/eegmmidb_exec (execution runs)."""
    for d in (MMIDB_DIR, EXEC_DIR):
        f = os.path.join(d, f"S{sid:03d}", f"S{sid:03d}R{run:02d}.edf")
        if os.path.exists(f):
            return f
    return os.path.join(MMIDB_DIR, f"S{sid:03d}", f"S{sid:03d}R{run:02d}.edf")


MOTOR_CHANNELS = EEG19 + [c for c in ("FC3", "FC4", "C5", "C1", "C2", "C6", "CP1", "CPz", "CP2", "CP3", "CP4")
                          if c not in EEG19]


def mmidb_person(sid, arm):
    """One EEGMMIDB person: the arm's six runs (signals, annotations; nothing else is read), MOTOR_CHANNELS in µV, one
    Source per T0 / T1 / T2 event, class from the run's task (RUNS)."""
    import mne
    from mne.datasets import eegbci
    runs, sources, events, nums = [], [], {}, []
    fs = None
    for r, (c1, c2) in RUNS[arm].items():
        f = mmidb_file(sid, r)
        if not os.path.exists(f):
            raise FileNotFoundError(f"S{sid:03d} run {r}: {f} missing")
        raw = mne.io.read_raw_edf(f, preload=False, verbose="ERROR")
        eegbci.standardize(raw)
        fs_r = float(raw.info["sfreq"])
        if fs is None:
            fs = fs_r
        if fs_r != fs:
            raise ValueError(f"S{sid:03d}: runs differ in sampling rate")
        x = (raw.get_data(picks=[raw.ch_names.index(c) for c in MOTOR_CHANNELS]) * 1e6).astype(np.float32)
        ann = raw.annotations
        k = len(runs)
        for on, du, d in zip(ann.onset, ann.duration, ann.description):
            cls = {"T0": "rest", "T1": c1, "T2": c2}[str(d)]
            start = int(round(on * fs))
            n = min(int(round(du * fs)), x.shape[1] - start)
            sources.append(Source(run=k, start=start, n=n, cls=cls, tag=f"R{r:02d}:{d}@{on:.2f}"))
            events[(k, start)] = cls
        runs.append(x)
        nums.append(r)
        del raw
    return Person(pid=f"S{sid:03d}", fs=fs, names=list(MOTOR_CHANNELS), runs=runs, sources=sources,
                  meta={"events": events, "run_numbers": nums, "arm": arm, "source": "eegmmidb"})


OPENBMI_DIR = os.path.join(ROOT, "output", "raw", "openbmi_mi")
OPENBMI_FS = 250.0
OPENBMI_IMAG_S = 4.0
OPENBMI_CLASS = {1: "right", 2: "left"}
OPENBMI_NA = {"cruse11": "N/A on OpenBMI: no toe imagery (left / right hand only)"}
OPENBMI_SESSIONS = {"columbia": (1,), "dubois": (1,), "edlow17": (1,), "cornell": (1, 2), "mbmi": (1, 2),
                    "cruse12p": (1, 2)}
PLAN = {"mbvt": ("all", "all", None), "gibson16": ("all", "all", None), "chennu13": ("all", "all", None),
        "cornell": ("all", "all", None), "mbmi": ("all", "all", None), "cruse11": ("all", "all", None),
        "cruse12p": ("all", "all", None), "edlow17": ("all", "primary", None), "dubois": ("all", "primary", None),
        "columbia": ("all", "primary", 20)}


def openbmi_sources(t, y, n, fs, run, tag):
    """Sources of one OpenBMI MI phase (Lee 2019: 3-s fixation, cue at t, 4-s imagery, 6 +- 1.5-s blank): imagery
    [t, t + 4 s) of class y (1 right, 2 left); rest = each contiguous no-imagery stretch [t + 4 s, next cue) (the blank
    and the next fixation), the lead-in [0, first cue) and the tail [last t + 4 s, end), each kept only if it is at
    least as long as an imagery trial (a short stretch would set the slot grid for every class)."""
    img = int(round(OPENBMI_IMAG_S * fs))
    t = np.asarray(t, int)
    out = []
    if t[0] >= img:
        out.append(Source(run=run, start=0, n=int(t[0]), cls="rest", tag=f"{tag}:lead"))
    for k, (on, cls) in enumerate(zip(t, y)):
        out.append(Source(run=run, start=int(on), n=int(min(img, n - on)), cls=OPENBMI_CLASS[int(cls)],
                          tag=f"{tag}:img@{on}"))
        end = t[k + 1] if k + 1 < len(t) else n
        if end - on - img >= img:
            out.append(Source(run=run, start=int(on + img), n=int(end - on - img), cls="rest", tag=f"{tag}:rest@{on}"))
    return out


def openbmi_npz(subj, sess):
    return os.path.join(OPENBMI_DIR, f"S{subj:02d}_sess{sess}.npz")


def openbmi_fetch(subj, sess):
    """Download one OpenBMI MI session file (CC0, GigaDB 100542) into OPENBMI_DIR/_download, keep MOTOR_CHANNELS at
    OPENBMI_FS (anti-aliased polyphase) with the cue onsets and classes of both phases, delete the .mat. Reads the
    fields x, t, y_dec, fs and chan only."""
    out = openbmi_npz(subj, sess)
    if os.path.exists(out):
        return out
    ob = importlib.import_module("openbmi_io")
    from scipy.io import loadmat
    dl = os.path.join(OPENBMI_DIR, "_download")
    os.makedirs(dl, exist_ok=True)
    path = _download(ob.url(sess, subj, "MI"), dl)
    try:
        arrays = {}
        for phase in ("train", "test"):
            key = f"EEG_MI_{phase}"
            m = loadmat(path, variable_names=[key])[key][0, 0]
            chan = [str(np.squeeze(c).item()) for c in np.ravel(m["chan"])]
            low = {c.lower(): i for i, c in enumerate(chan)}
            fs0 = float(m["fs"].item())
            x = np.asarray(m["x"], float)[:, [low[c.lower()] for c in MOTOR_CHANNELS]].T
            arrays[f"x_{phase}"] = mc.resample(x, fs0, OPENBMI_FS).astype(np.float32)
            arrays[f"t_{phase}"] = np.round(m["t"].ravel() * OPENBMI_FS / fs0).astype(int)
            arrays[f"y_{phase}"] = m["y_dec"].ravel().astype(int)
            del m, x
        np.savez(out + ".tmp.npz", names=np.array(MOTOR_CHANNELS), fs=OPENBMI_FS, **arrays)
        os.replace(out + ".tmp.npz", out)
    finally:
        os.remove(path)
    return out


def _download(url, dest, tries=3):
    """Stream url into dest; the byte count must equal Content-Length (a short read is retried, then refused)."""
    import shutil
    import requests
    if shutil.disk_usage(dest).free < 21e9:
        raise RuntimeError("less than 21 GB free on the SSD (20 GB must stay free)")
    path = os.path.join(dest, url.rsplit("/", 1)[1])
    for k in range(tries):
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            want = int(r.headers.get("Content-Length", -1))
            with open(path + ".part", "wb") as fh:
                for chunk in r.iter_content(1 << 22):
                    fh.write(chunk)
        got = os.path.getsize(path + ".part")
        if want < 0 or got == want:
            os.replace(path + ".part", path)
            return path
        os.remove(path + ".part")
    raise IOError(f"{url}: {tries} short downloads")


def openbmi_person(subj, sessions=(1,)):
    """One OpenBMI person from the extracted sessions: one run per phase (train, test), sources by openbmi_sources."""
    runs, sources, events, nums = [], [], {}, []
    for sess in sessions:
        z = np.load(openbmi_npz(subj, sess))
        if list(z["names"]) != list(MOTOR_CHANNELS):
            raise ValueError(f"OpenBMI S{subj:02d}: unexpected channels")
        for phase in ("train", "test"):
            x = np.asarray(z[f"x_{phase}"], np.float32)
            k = len(runs)
            src = openbmi_sources(z[f"t_{phase}"], z[f"y_{phase}"], x.shape[1], OPENBMI_FS, k, f"sess{sess}:{phase}")
            sources += src
            events.update({(k, q.start): q.cls for q in src})
            runs.append(x)
            nums.append(f"sess{sess}-{phase}")
    return Person(pid=f"OpenBMI_S{subj:02d}", fs=OPENBMI_FS, names=list(MOTOR_CHANNELS), runs=runs, sources=sources,
                  meta={"events": events, "run_numbers": nums, "source": "openbmi", "sessions": list(sessions)})


def kojima_trials(ev, subject, fs=ERP_FS, run_index=None):
    """The subject's 24 trials from the event index: attended stream = the stream of the trial's start code (201-204);
    kinds from the codes (111-114 counted target, 101-104 other deviant of that stream, 1 / 2 standards)."""
    ev = ev[ev["subject"].astype(str) == str(subject)]
    runs = sorted(int(r) for r in ev["run"].unique())
    run_index = run_index or {r: i for i, r in enumerate(runs)}
    out = []
    for (r, t), g in ev.groupby(["run", "trial"], sort=True):
        tc = int(g["target_code"].dropna().iloc[0])
        st = KOJIMA_STREAM[tc]
        a = g[g["code"].isin(KOJIMA_TONES) & (g["stream"].astype("float") == st)].sort_values("sample")
        code = a["code"].to_numpy(int)
        kind = np.where(code >= 111, "target", np.where(code >= 101, "other", "standard"))
        ident = np.array([KOJIMA_IDENT.get(int(x), "") for x in code])
        on = np.round(a["sample"].to_numpy(float) / a["sfreq"].to_numpy(float) * fs).astype(int)
        out.append(Trial(run=run_index[int(r)], tid=f"run-{int(r)} trial-{int(t)}", target=KOJIMA_TARGET_ID[tc],
                         on=on, kind=kind, ident=ident, code=code))
    return out


def kojima_person(subject):
    """One Kojima person: the six 2stream runs (signal, sfreq and channel names only), ERP_CHANNELS in µV, resampled
    to ERP_FS; trials from the event index."""
    import mne
    import pandas as pd
    ev = pd.read_parquet(KOJIMA_EVENTS)
    ev = ev[ev["subject"].astype(str) == str(subject)]
    runs_n = sorted(int(r) for r in ev["run"].unique())
    runs = []
    for r in runs_n:
        f = os.path.join(KOJIMA_DIR, f"sub-{subject}", "eeg", f"sub-{subject}_task-2stream_run-{r}_eeg.vhdr")
        raw = mne.io.read_raw_brainvision(f, preload=False, verbose="ERROR")
        low = {c.lower(): i for i, c in enumerate(raw.ch_names)}
        x = raw.get_data(picks=[low[c.lower()] for c in ERP_CHANNELS]) * 1e6
        fs0 = float(raw.info["sfreq"])
        del raw
        runs.append(mc.resample(x, fs0, ERP_FS).astype(np.float32))
        del x
    trials = kojima_trials(ev, subject, ERP_FS, {r: i for i, r in enumerate(runs_n)})
    for t in trials:
        t.lo, t.hi = 0, runs[t.run].shape[1]
    return ErpPerson(pid=f"kojimaB_sub-{subject}", fs=ERP_FS, names=list(ERP_CHANNELS), runs=runs, trials=trials,
                     meta={"subject": subject, "run_numbers": runs_n, "source": "kojimaB"})


def kojima_subjects():
    import pandas as pd
    return sorted(pd.read_parquet(KOJIMA_EVENTS, columns=["subject"])["subject"].astype(str).unique())


def openbmi_subjects():
    return list(range(1, 55))


def all_people(arm):
    """Every person id of an arm, as given on the command line: Kojima letters, EEGMMIDB 'S021', OpenBMI 'S01'."""
    if arm == "erp":
        return kojima_subjects()
    if arm == "openbmi":
        return [f"S{s:02d}" for s in openbmi_subjects()]
    return [f"S{s:03d}" for s in mmidb_subjects()]


def dev_people(arm):
    """Development people, never reported: 3 EEGMMIDB subjects (SEED_DEV draw over the sorted list; the same for both
    motor arms); 3 OpenBMI subjects (the same draw rule over 1-54); Kojima: the frozen passive dev split (L, O)."""
    if arm == "erp":
        passive = importlib.import_module("bench.passive")
        return sorted(passive.passive_split("kojimaB")["dev"])
    subs = openbmi_subjects() if arm == "openbmi" else mmidb_subjects()
    return sorted(int(subs[i]) for i in np.random.default_rng(SEED_DEV).choice(len(subs), DEV_N, replace=False))


def _dev_ids(arm):
    devs = dev_people(arm)
    return set(devs) if arm == "erp" else {f"S{s:02d}" if arm == "openbmi" else f"S{s:03d}" for s in devs}



def _specs(mod):
    return {v: {"arg": v, "globals": {}, "twin_of": None, "short": False} for v in mod.VARIANTS}


def rand_groups(mod):
    """{representative: [variants]} for the randomization-based method: 114.timing_groups over every native variant;
    mbvt per paradigm (its timing variants are moot under recorded timing)."""
    specs, prim = _specs(mod), primary_variant(mod)
    if mod.NAME == "mbvt":
        by = collections.defaultdict(list)
        for v in specs:
            by[mod.VARIANTS[v]["paradigm"]].append(v)
        return {rn._representative(vs, specs, prim, mod): vs for vs in by.values()}
    return rn.timing_groups(mod, specs, list(specs), prim)


def _sig(mod, ov):
    if mod.FAMILY == "cmd":
        return (np.asarray(ov.onsets_s, float).tobytes(), np.asarray(ov.labels).tobytes())
    m = ov.meta
    key = {"mbvt": lambda: (m["paradigm"], np.asarray(m["target_wrist"]).tobytes()),
           "chennu13": lambda: np.asarray(m["explicit_word"]).tobytes(),
           "gibson16": lambda: np.asarray(m["target_wrist"]).tobytes()}[mod.NAME]
    return key()


def _sig_id(sig):
    """A seed for the session of a schedule signature, so equal draws give the same session after a resume."""
    design, key = sig
    parts = key if isinstance(key, tuple) else (key,)
    return zlib.crc32(design.encode() + b"".join(p if isinstance(p, bytes) else str(p).encode() for p in parts))


def build(person, mod, ov, rng, control):
    return (motor_session if mod.FAMILY == "cmd" else erp_session)(person, mod, ov, rng, control=control)


def _result(res, err, sec, mod):
    out = {"status": rn._status(err), "stat": None, "p": None, "call": None, "size_stat": None, "seconds": sec}
    if err and err.startswith(rn.NA_NO_USABLE):
        out["na_detail"] = err[len(rn.NA_NO_USABLE):].strip(" []")
    if res is not None:
        size, _, _ = rn._call(wr.size_stat, mod, res)
        out.update(stat=rn._scalar(res["stat"]), p=rn._scalar(res["p"]), call=bool(res["call"]),
                   size_stat=rn._scalar(size), p_censored=wr.p_censored(res))
        out.update(rn.scalars(res.get("extra")))
    return out


SCOPES = ("all", "primary")


def jobs_for(mod, version, scope="all", control_scope="all"):
    """[(design, variant)]: scope "all" = every native variant's run() and every timing group's randomized method;
    "primary" = the primary variant and the randomized method of its timing group. The control version uses
    control_scope."""
    groups = rand_groups(mod)
    sc = control_scope if version == "control" else scope
    if sc == "all":
        return [("published", v) for v in mod.VARIANTS] + [("randomized", g) for g in groups]
    prim = primary_variant(mod)
    return [("published", prim), ("randomized", next(g for g, vs in groups.items() if prim in vs))]


def run_person(person, mod, arm, dev, versions, done, write, scope="all", control_scope="all"):
    """Every row of one person and pipeline (module docstring)."""
    pid, name = person.pid, mod.NAME
    groups = rand_groups(mod)
    prim = primary_variant(mod)
    for version in versions:
        control = version == "control"
        jobs = jobs_for(mod, version, scope, control_scope)
        if all(f"{arm}|{pid}|{name}|{d}|{v}|{version}" in done for d, v in jobs):
            continue
        sessions = {}
        for design, v in jobs:
            key = f"{arm}|{pid}|{name}|{design}|{v}|{version}"
            if key in done:
                continue
            if design == "published":
                ov = mod.schedule(START_S, _rng(pid, name, 1), v)
            else:
                ov = mod.schedule_randomized(START_S, _rng(pid, name, 3, v), v)
            sig = (design, _sig(mod, ov))
            if sig not in sessions:
                mc._CACHE.clear()
                sessions.clear()
                s, err, _ = rn._call(build, person, mod, ov, _rng(pid, name, version, "session", _sig_id(sig)),
                                     control)
                sessions[sig] = (s, err)
            s, err = sessions[sig]
            row = {"key": key, "arm": arm, "person": pid, "dev": dev, "pipe": name, "design": design, "variant": v,
                   "version": version, "primary_variant": prim,
                   "as_coded": bool(mod.VARIANTS.get(v, {}).get("as_coded", False)),
                   "group_members": ",".join(groups[v]) if design == "randomized" else None}
            if s is None:
                row.update(status=f"error: session: {err}", call=None)
            else:
                row.update({f"session_{k}": v_ for k, v_ in s.info.items() if not isinstance(v_, dict)})
                for k in ("demand_s", "avail_s"):
                    row[f"session_{k}"] = json.dumps(s.info[k]) if k in s.info else None
                g = _rng(pid, name, 2, design, v, version)
                if design == "published":
                    res, e, sec = rn._call_run(mod, mod.run, s.rec, s.overlay, v, g)
                else:
                    res, e, sec = rn._call_run(mod, wr.run_randomized, mod, s.rec, s.overlay, v, g)
                row.update(_result(res, e, sec, mod))
                mp, _, _ = rn._call(wr.min_attainable_p, mod, v)
                row.update(min_attainable_p=rn._scalar(mp), cannot_call=bool(mp is not None and mp > 0.05))
            write(row)
        sessions.clear()
        mc._CACHE.clear()


def load_person(arm, pid, pipe=None):
    if arm == "erp":
        return kojima_person(pid)
    if arm == "openbmi":
        return openbmi_person(int(str(pid).lstrip("S")), OPENBMI_SESSIONS[pipe])
    return mmidb_person(int(str(pid).lstrip("S")), arm)


def person_id(arm, pid):
    """The pid a loaded person carries (row keys), without loading it."""
    if arm == "erp":
        return f"kojimaB_sub-{pid}"
    if arm == "openbmi":
        return f"OpenBMI_S{int(str(pid).lstrip('S')):02d}"
    return f"S{int(str(pid).lstrip('S')):03d}"


def job_keys(arm, pid, mod, versions, scope="all", control_scope="all"):
    return [f"{arm}|{person_id(arm, pid)}|{mod.NAME}|{d}|{v}|{ver}" for ver in versions
            for d, v in jobs_for(mod, ver, scope, control_scope)]


def people(arm, dev, given=None, limit=None, sample=None):
    """The arm's people: dev people, or every other person; given = an explicit list (checked against dev); sample = a
    seeded subsample (SAMPLE_SEED) of the analysis people, sorted."""
    allp, devset = all_people(arm), _dev_ids(arm)
    if given:
        want = [str(p) for p in given]
        bad = [p for p in want if p not in allp]
        if bad:
            raise ValueError(f"unknown people {bad}")
        wrong = [p for p in want if (p in devset) != dev]
        if wrong:
            raise ValueError(f"{wrong}: {'not ' if dev else ''}development people (use --dev accordingly)")
        out = want
    else:
        out = [p for p in allp if (p in devset) == dev]
    if sample is not None:
        if dev or given:
            raise ValueError("--sample draws from the analysis people only")
        if sample > len(out):
            raise ValueError(f"sample {sample} > {len(out)} people")
        g = np.random.default_rng(SAMPLE_SEED)
        out = sorted(out[i] for i in g.choice(len(out), sample, replace=False))
    return out[:limit] if limit else out


def out_file(arm, pipe, dev):
    return os.path.join(OUT[dev], f"{arm}__{pipe}.jsonl")


def _done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {json.loads(l)["key"] for l in fh if l.strip()}


def run(arm, pipe, dev=False, given=None, limit=None, versions=VERSIONS, scope="all", control_scope="all",
        sample=None):
    if pipe in NA_PIPES:
        raise SystemExit(f"{pipe}: {NA_PIPES[pipe]}")
    fam = ERP_PIPES if arm == "erp" else MOTOR_PIPES
    if pipe not in fam:
        raise SystemExit(f"{pipe} does not run on arm {arm}")
    if arm == "openbmi" and pipe in OPENBMI_NA:
        raise SystemExit(f"{pipe}: {OPENBMI_NA[pipe]}")
    mod = load_pipe(pipe)
    path = out_file(arm, pipe, dev)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _done(path)
    with open(path, "a") as fh:
        def write(row):
            row = {k: rn._scalar(v) if not isinstance(v, (str, type(None))) else v for k, v in row.items()}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done.add(row["key"])
        for pid in people(arm, dev, given, limit, sample):
            if all(k in done for k in job_keys(arm, pid, mod, versions, scope, control_scope)):
                continue
            t0 = time.perf_counter()
            person = load_person(arm, pid, pipe)
            run_person(person, mod, arm, dev, versions, done, write, scope, control_scope)
            print(f"{arm} {pipe} {person.pid} {time.perf_counter() - t0:.1f} s", flush=True)
            del person


def _writer(path, done):
    fh = open(path, "a")

    def write(row):
        row = {k: rn._scalar(v) if not isinstance(v, (str, type(None))) else v for k, v in row.items()}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        done.add(row["key"])
    return fh, write


def run_stream(arm, pipes, dev=False, versions=VERSIONS, purge=True):
    """OpenBMI, person by person (the disk holds one person at a time): fetch the sessions the person's pending
    pipelines need (OPENBMI_SESSIONS), run each pipeline with its PLAN scope (and only on its PLAN sample), then delete
    the person's extracted files."""
    if arm != "openbmi":
        raise ValueError("run_stream is for the OpenBMI arm (other arms keep their data on disk: use run)")
    mods = {p: load_pipe(p) for p in pipes if p not in OPENBMI_NA}
    who = {p: people(arm, dev, sample=PLAN[p][2]) for p in mods}
    files, dones = {}, {}
    for p in mods:
        path = out_file(arm, p, dev)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dones[p] = _done(path)
        files[p] = _writer(path, dones[p])
    try:
        for pid in dict.fromkeys(q for p in mods for q in who[p]):
            todo = [p for p in mods if pid in who[p] and not all(
                k in dones[p] for k in job_keys(arm, pid, mods[p], versions, *PLAN[p][:2]))]
            if not todo:
                continue
            subj = int(str(pid).lstrip("S"))
            for sess in sorted({q for p in todo for q in OPENBMI_SESSIONS[p]}):
                openbmi_fetch(subj, sess)
            for p in todo:
                t0 = time.perf_counter()
                person = openbmi_person(subj, OPENBMI_SESSIONS[p])
                run_person(person, mods[p], arm, dev, versions, dones[p], files[p][1], *PLAN[p][:2])
                print(f"{arm} {p} {person.pid} {time.perf_counter() - t0:.1f} s", flush=True)
                del person
            if purge:
                for sess in (1, 2):
                    f = openbmi_npz(subj, sess)
                    if os.path.exists(f):
                        os.remove(f)
    finally:
        for fh, _ in files.values():
            fh.close()



CONTRAST = {
    "columbia": "keep moving / stop, right and left hand by block (NEJM19; controls moved)",
    "cornell": "sustained motor imagery vs stop (CU18 swim / tennis / open-close hand; GF11)",
    "mbmi": "left vs right hand imagery, cued per trial (GU17 / CH18 / OR17)",
    "dubois": "two imagined movements (e.g. left vs right arm), blocks (DB26)",
    "cruse11": "right-hand squeeze vs toe imagery, blocks of 15 tones (CR11)",
    "edlow17": "right-hand squeeze imagery vs rest, 24-s blocks (ED17)",
    "cruse12p": "attempted right / left hand squeeze vs relax, per trial (CR12P; controls squeezed)",
}
OFFERED = {"imagery": "EEGMMIDB imagery: left / right fist (runs 4/8/12), both fists / feet (6/10/14), rest T0",
           "execution": "EEGMMIDB execution: real left / right fist, both fists / feet, rest T0",
           "openbmi": "OpenBMI MI: left / right hand imagery (4 s), fixation + blank rest"}
USABILITY = {
    "mbmi": {"filt": (0.1, 30.0, 4), "win": 8.0, "thr": 100.0, "n": 30, "min": 10, "per": 1,
             "rule": "OR17 +-100 uV on 0.1-30 Hz, 8 s after the beep; >= 10 surviving per hand (gu17: 30 per hand)"},
    "cruse11": {"filt": (1.0, 40.0, 4), "win": 5.5, "thr": 100.0, "n": 60, "min": 2, "per": 1, "ch": "sub",
                "rule": "+-100 uV on 1-40 Hz, -1.5 to 4 s; >= 2 surviving per class (b4: 60 per class)"},
    "cruse12p": {"filt": "eegfilt", "win": 6.0, "thr": 100.0, "n": 60, "min": 3, "per": 1,
                 "rule": "+-100 uV after eegfilt 1-40 Hz at 100 Hz, 0-6 s; >= 3 surviving per instruction (60)"},
    "edlow17": {"filt": (1.0, 30.0, 3), "win": 1.0, "thr": 200.0, "n": 384, "min": 20, "per": 1, "ch": "native",
                "rule": "+-200 uV per 1-s column on 1-30 Hz; >= 20 surviving columns of 384"},
    "cornell": {"filt": (1.0, 50.0, 4), "win": 3.0, "thr": 200.0, "n": 24, "min": 2, "per": 1, "ch": "native",
                "rule": "thresh variants: +-200 uV per 3-s epoch (the robust-z part not modelled); >= 2 per condition "
                        "and run (24); keep40 variants keep 40% always"},
    "columbia": None, "dubois": None,
}


def amplitude_usable(person, pipe):
    """(p, usable) of the USABILITY proxy; (1.0, True) for a module without an amplitude rule."""
    spec = USABILITY.get(pipe)
    if spec is None:
        return 1.0, True
    mod = load_pipe(pipe)
    nm = mc.canon(person.names)
    if spec.get("ch") == "sub":
        want = mc.canon(mod.SUB_1020)
    elif spec.get("ch") == "native":
        want = mc.canon(mod.NATIVE_CH)
    else:
        nat = mc.canon(mod.NATIVE_CH)
        want = nat if all(c in nm for c in nat) else mc.canon(mod.SUB_1020)
    rows = [nm.index(c) for c in want]
    ok = tot = 0
    for x in person.runs:
        x = np.asarray(x[rows], float)
        fs = person.fs
        if spec["filt"] == "eegfilt":
            x = mc.resample(x, fs, mod.FS)
            fs = mod.FS
            x = mod._eegfilt(mod._eegfilt(x, fs, lo=1.0), fs, hi=40.0)
        else:
            x = mc.bandpass(x, fs, spec["filt"][0], spec["filt"][1], order=spec["filt"][2])
        w = int(round(spec["win"] * fs))
        for a in range(0, x.shape[1] - w + 1, max(1, w // 2)):
            tot += 1
            ok += np.abs(x[:, a:a + w]).max() <= spec["thr"]
    p = ok / tot if tot else 0.0
    return p, bool(spec["n"] * p ** spec["per"] >= spec["min"])


def _schedules(mod):
    """{label: overlay} of every distinct published schedule (one seed for all variants, as run_person draws them)
    and the randomized schedule of every timing group; {variant: label} maps each variant to its schedule."""
    out, of, seen = {}, {}, {}
    for v in mod.VARIANTS:
        ov = mod.schedule(START_S, np.random.default_rng([SEED, 1]), v)
        sig = _sig(mod, ov)
        if sig not in seen:
            seen[sig] = f"published:{v}"
            out[seen[sig]] = ov
        of[v] = seen[sig]
    for g in rand_groups(mod):
        out[f"randomized:{g}"] = mod.schedule_randomized(START_S, np.random.default_rng([SEED, 3]), g)
    return out, of


def feasibility_rows(arm, pids, sessions=(1,)):
    """One row per person and motor pipeline: reuse (allocate on the person's real source lengths) for the primary
    variant, the worst schedule over variants and designs, seconds per class, and the amplitude proxy."""
    rows = []
    for pid in pids:
        person = (openbmi_person(int(str(pid).lstrip("S")), sessions) if arm == "openbmi"
                  else mmidb_person(int(str(pid).lstrip("S")), arm))
        avail = collections.Counter()
        for q in person.sources:
            avail[q.cls] += q.n / person.fs
        for pipe in MOTOR_PIPES:
            if arm == "openbmi" and pipe in OPENBMI_NA:
                rows.append({"arm": arm, "person": person.pid, "pipe": pipe, "status": OPENBMI_NA[pipe]})
                continue
            mod = load_pipe(pipe)
            prim = primary_variant(mod)
            res = {}
            scheds, of = _schedules(mod)
            for label, ov in scheds.items():
                res[label] = feasibility(mod, ov, person.sources, person.fs, np.random.default_rng([SEED, 4]),
                                         dataset=person.meta.get("source"))
            pk = of[prim]
            p, usable = amplitude_usable(person, pipe)
            rows.append({"arm": arm, "person": person.pid, "pipe": pipe, "status": "ok",
                         "sessions": ",".join(map(str, sessions)) if arm == "openbmi" else None,
                         "avail_s": {k: round(v, 1) for k, v in avail.items()},
                         "primary_reuse_max": res[pk]["reuse_max"], "primary_demand_ratio": res[pk]["demand_ratio_max"],
                         "primary_demand_s": res[pk]["demand_s"],
                         "worst_reuse_max": max(r["reuse_max"] for r in res.values()),
                         "worst_demand_ratio": max(r["demand_ratio_max"] for r in res.values()),
                         "worst_schedule": max(res, key=lambda k: res[k]["demand_ratio_max"]),
                         "amp_p": round(p, 4), "amp_usable": usable})
        del person
    return rows


def feasibility_table(rows):
    """Per arm x pipeline: medians and ranges over people."""
    out = {}
    for key in sorted({(r["arm"], r["pipe"], r.get("sessions")) for r in rows}):
        rr = [r for r in rows if (r["arm"], r["pipe"], r.get("sessions")) == key and r["status"] == "ok"]
        arm, pipe, sess = key
        name = f"{arm}{'' if sess is None else ' sess ' + sess}|{pipe}"
        if not rr:
            out[name] = {"status": next(r["status"] for r in rows if (r["arm"], r["pipe"], r.get("sessions")) == key)}
            continue
        med = lambda k: float(np.median([r[k] for r in rr]))
        cls = sorted({c for r in rr for c in r["avail_s"]})
        out[name] = {
            "contrast_needed": CONTRAST[pipe], "contrast_offered": OFFERED[arm], "n_people": len(rr),
            "avail_s_median": {c: float(np.median([r["avail_s"].get(c, 0) for r in rr])) for c in cls},
            "demand_s_primary": rr[0]["primary_demand_s"],
            "primary_reuse_max": [min(r["primary_reuse_max"] for r in rr), max(r["primary_reuse_max"] for r in rr)],
            "primary_demand_ratio_median": round(med("primary_demand_ratio"), 3),
            "primary_demand_ratio_max": round(max(r["primary_demand_ratio"] for r in rr), 3),
            "worst_reuse_max": max(r["worst_reuse_max"] for r in rr),
            "worst_demand_ratio_max": round(max(r["worst_demand_ratio"] for r in rr), 3),
            "share_reuse_free_primary": round(float(np.mean([r["primary_reuse_max"] == 1 for r in rr])), 3),
            "share_reuse_free_all": round(float(np.mean([r["worst_reuse_max"] == 1 for r in rr])), 3),
            "amp_rule": (USABILITY.get(pipe) or {}).get("rule", "none"),
            "amp_usable_share": round(float(np.mean([r["amp_usable"] for r in rr])), 3),
            "amp_p_median": round(med("amp_p"), 3)}
    return out


def write_feasibility(sessions=(1,), path=FEASIBILITY):
    """Rows and table over every EEGMMIDB person (imagery; execution when its runs are on disk) and every OpenBMI
    person extracted so far, for each session set up to `sessions`."""
    rows = feasibility_rows("imagery", all_people("imagery"))
    ex = [p for p in all_people("execution") if all(os.path.exists(mmidb_file(int(p[1:]), r)) for r in RUNS["execution"])]
    rows += feasibility_rows("execution", ex)
    for k in range(1, max(sessions) + 1):
        ss = tuple(range(1, k + 1))
        have = [p for p in all_people("openbmi") if all(os.path.exists(openbmi_npz(int(p[1:]), q)) for q in ss)]
        rows += feasibility_rows("openbmi", have, ss)
    out = {"table": feasibility_table(rows), "rows": rows, "n_execution_people": len(ex)}
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
    return out



def bar_for(pipe, variant):
    """The published healthy-control bar that applies to a variant: {k, n, note}."""
    if pipe in NA_PIPES:
        return None
    cfg = load_pipe(pipe).VARIANTS.get(variant, {})
    for b in BARS[pipe]:
        if b["applies"] is None or all(cfg.get(k) == v for k, v in b["applies"].items()):
            return b
    return None


def detection(calls, bar=None, conf=0.95):
    """Called share of usable people with its Clopper-Pearson CI; fidelity concern if the upper bound < the bar."""
    calls = np.asarray(calls, bool)
    k, n = int(calls.sum()), int(len(calls))
    lo, hi = mc.cp_interval(k, n, conf) if n else (float("nan"), float("nan"))
    out = {"k": k, "n": n, "rate": k / n if n else None, "ci": [lo, hi]}
    if bar is not None:
        br = bar[0] / bar[1]
        out.update(bar=f"{bar[0]}/{bar[1]}", bar_rate=br, fidelity_concern=bool(n and hi < br))
    return out


def summarize(rows):
    """Per arm x pipe x design x variant x version: detection (true) or control rate, unusable and dev counts; the
    randomized-design control is tested against 5% (one-sided binomial): reuse_or_splice_calls if p < .05."""
    from scipy.stats import binomtest
    cells = collections.defaultdict(lambda: {"calls": [], "n_unusable": 0, "n_error": 0, "n_dev_excluded": 0})
    for r in rows:
        key = f"{r['arm']}|{r['pipe']}|{r['design']}|{r['variant']}|{r['version']}"
        c = cells[key]
        if r.get("dev"):
            c["n_dev_excluded"] += 1
            continue
        st = r.get("status") or ""
        if st == "ok":
            c["calls"].append(bool(r["call"]))
        elif st.startswith("na"):
            c["n_unusable"] += 1
        else:
            c["n_error"] += 1
    out = {}
    for key, c in cells.items():
        arm, pipe, design, variant, version = key.split("|")
        b = bar_for(pipe, variant)
        d = detection(c["calls"], (b["k"], b["n"]) if (b and version == "true") else None)
        d.update(n_unusable=c["n_unusable"], n_error=c["n_error"], n_dev_excluded=c["n_dev_excluded"])
        if version == "control" and d["n"]:
            d["p_above_5pct"] = float(binomtest(d["k"], d["n"], 0.05, alternative="greater").pvalue)
            d["reuse_or_splice_calls"] = bool(design == "randomized" and d["p_above_5pct"] < 0.05)
        out[key] = d
    return {"cells": out}


def read_rows(dev=False):
    rows = []
    d = OUT[dev]
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl") and not f.startswith("._"):
                with open(os.path.join(d, f)) as fh:
                    rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def headline(rows, cells):
    """Per arm x pipeline: the a-priori default variant (primary_variant) and the randomization-based method of its
    timing group, true and control, beside the bar; the range over variants and how many are flagged; reuse."""
    out = {}
    for arm in ARMS:
        for pipe in (ERP_PIPES if arm == "erp" else MOTOR_PIPES):
            mine = [r for r in rows if r["arm"] == arm and r["pipe"] == pipe and not r.get("dev")]
            if not mine:
                continue
            mod = load_pipe(pipe)
            prim = primary_variant(mod)
            rep = next(g for g, vs in rand_groups(mod).items() if prim in vs)
            get = lambda d, v, ver: cells.get(f"{arm}|{pipe}|{d}|{v}|{ver}")
            pub = [(v, cells[k]) for v in mod.VARIANTS for k in [f"{arm}|{pipe}|published|{v}|true"] if k in cells]
            rates = [c["rate"] for _, c in pub if c["rate"] is not None]
            reuse = [r.get("session_reuse_max") for r in mine if r.get("version") == "true"
                     and r.get("session_reuse_max") is not None]
            role = "primary" if PRIMARY.get(pipe) == arm else "secondary"
            dp = get("published", prim, "true")
            if PRIMARY.get(pipe) is None or (reuse and max(reuse) > 1):
                verdict = "reuse-confounded"
            elif dp is None or not dp.get("n"):
                verdict = "no usable people"
            else:
                verdict = "fidelity concern" if dp.get("fidelity_concern") else "consistent with the bar"
            out[f"{arm}|{pipe}"] = {
                "role": role, "verdict": verdict, "reuse_max_any": max(reuse) if reuse else None,
                "bar": BARS[pipe], "default_variant": prim, "randomized_group": rep,
                "default_published": get("published", prim, "true"), "default_randomized": get("randomized", rep, "true"),
                "control_published": get("published", prim, "control"),
                "control_randomized": get("randomized", rep, "control"),
                "variant_rate_range": [min(rates), max(rates)] if rates else None,
                "n_variants": len(pub), "n_variants_fidelity_concern": sum(bool(c.get("fidelity_concern")) for _, c in pub),
                "reuse_max_median": float(np.median(reuse)) if reuse else None}
    return out


def write_summary(path=SUMMARY):
    rows = read_rows(False)
    if any(r.get("dev") for r in rows):
        raise RuntimeError("development rows in the analysis output")
    out = summarize(rows)
    out["headline"] = headline(rows, out["cells"])
    out["bars"] = BARS
    out["secondary_bars"] = SECONDARY_BARS
    out["not_applicable"] = NA_PIPES
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--pipe")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--people")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--scope", choices=SCOPES, default="all")
    ap.add_argument("--control-scope", choices=SCOPES, default="all")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--fetch-openbmi", help="comma-separated subject numbers (1-54)")
    ap.add_argument("--sessions", default="1")
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--stream", action="store_true", help="OpenBMI: person by person over --pipe a,b,c with PLAN")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        write_summary()
        print(SUMMARY)
        return
    sessions = tuple(int(x) for x in a.sessions.split(","))
    if a.fetch_openbmi:
        for subj in (int(x) for x in a.fetch_openbmi.split(",")):
            for sess in sessions:
                t0 = time.perf_counter()
                print(openbmi_fetch(subj, sess), f"{time.perf_counter() - t0:.0f} s", flush=True)
        return
    if a.feasibility:
        write_feasibility(sessions)
        print(FEASIBILITY)
        return
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.stream:
        run_stream(a.arm, a.pipe.split(","), a.dev, tuple(a.versions.split(",")))
        return
    run(a.arm, a.pipe, a.dev, a.people.split(",") if a.people else None, a.limit, tuple(a.versions.split(",")),
        a.scope, a.control_scope, a.sample)


if __name__ == "__main__":
    main()
