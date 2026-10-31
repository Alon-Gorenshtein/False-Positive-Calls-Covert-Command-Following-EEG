"""Detection at a common patient-level false-call rate. Each analysis's evidence statistic (the one 115's size
correction uses: bench.wrappers.size_stat) is thresholded on recordings without study commands so that a patient
assessed k times is called positive by chance with probability alpha = 5%, and the same threshold is applied to REAL
instructed responses (the healthy positive-control sessions of 118, the acute-stroke control of 163, the stroke and
tactile sessions of 172 and 173). Beside it: the randomization-based method (p <= the per-assessment level); the
published rule's own rate is any_positive(rate, k).

Calibration (as 115's statistic-based correction, extended to k assessments):
- per-assessment level a_k = 1 - (1 - alpha)^(1 / k), the level at which k independent assessments give alpha (Sidak);
  repeated recordings of one patient are not independent, and under positive within-patient association of calls the
  any-positive rate at a_k is at most alpha (158's 'upper_bound' reading, there for the published rules);
- threshold = the (1 - a_k) quantile ('higher') of the null statistic, a call strictly above it, CROSS-FITTED by
  patient: halves by crc32(patient) parity (column 'patient', else 'unit'; 115.crossfit_calls), each half's null
  calls judged by the other half's threshold (false_call_heldout = the mean of the two held-out rates); a task unit with
  a 'patient' is judged by the threshold of the half NOT holding its patient (held out even when the null rows come
  from the same patients, as the 172 / 173 shams do), else by both (detection = the mean); +-inf capped, NaN missing;
- a threshold transfers only between recordings of the same design (trial count, runs, channels): the statistic's
  scale depends on them. Each task design is therefore calibrated on its OWN null rows (the 172 / 173 shams for their
  sessions; the command-free windows for the published schedules), never on another design's;
- a task unit whose statistic is missing is left out and counted (n_task_missing), never called negative.
k: 1, 2 (acute ICU: median 2 recordings per patient, Egbebike 2022), 5 (prolonged DoC: five assessments, EAN 2020 /
Wannez 2017) and 10 (daily sessions, Murovec 2020). Curve: detection against held-out false calls over a grid of levels.

python 174_matched_calibration.py --all -> output/tables/matched_calibration.json (stored rows: published schedules from
  the clinical-EEG null shards and 118's healthy responders; the 172 / 173 sessions against their own shams)
python 174_matched_calibration.py --null NULL.csv --task TASK.csv [--rand RAND.csv]
(CSV columns: pipe, reading, unit, stat for NULL and TASK; pipe, unit, p for RAND; the adapters that write them from the
stored rows are added with the port to the working copy)."""
import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
ALPHA = 0.05
KS = (1, 2, 5, 10)
LEVELS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def per_assessment_alpha(alpha, k):
    return 1.0 - (1.0 - alpha) ** (1.0 / k)


def any_positive(rate, k):
    """Patient-level any-positive rate over k independent assessments at a per-assessment rate."""
    return 1.0 - (1.0 - rate) ** k


def half_of(units):
    """0 / 1 per row: crc32(unit) parity, so a patient's rows always fall in one half."""
    return np.array([zlib.crc32(str(u).encode()) % 2 for u in units])


CAP = 1e6


def _clean(stats):
    """+-inf capped at +-CAP (a zero-variance paired t is the strongest evidence, not missing); NaN stays missing."""
    return np.clip(np.asarray(stats, float), -CAP, CAP)


def threshold(stats, a):
    """The (1 - a) 'higher' quantile; a statistic calls when STRICTLY above it, so ties at the threshold can never
    push the null rate above a."""
    s = _clean(stats)
    s = s[np.isfinite(s)]
    return float(np.quantile(s, 1.0 - a, method="higher")) if len(s) else np.inf


def crossfit(null, task, alpha, k):
    """-> dict(false_call_heldout, detection, n_null, n_task, n_task_missing, level)."""
    a = per_assessment_alpha(alpha, k)
    h = half_of(null["patient"] if "patient" in null else null["unit"])
    ns = _clean(null["stat"])
    thr = [threshold(ns[h == j], a) for j in (0, 1)]
    held = [np.mean(ns[(h == j) & np.isfinite(ns)] > thr[1 - j]) for j in (0, 1) if np.any(h == j)]
    ts = _clean(task["stat"])
    ok = np.isfinite(ts)
    if "patient" in task and len(task):
        ht = half_of(task["patient"])
        tt = np.where(ht == 0, thr[1], thr[0])
        det = float(np.mean(ts[ok] > tt[ok])) if ok.any() else None
    else:
        det = float(np.mean([np.mean(ts[ok] > t) for t in thr])) if ok.any() else None
    hk = int(sum(np.sum(ns[(h == j) & np.isfinite(ns)] > thr[1 - j]) for j in (0, 1) if np.any(h == j)))
    hn = int(np.isfinite(ns).sum())
    return {"level": a, "k": k, "false_call_heldout": float(np.mean(held)), "detection": det,
            "n_null": hn, "n_task": int(ok.sum()), "n_task_missing": int((~ok).sum()),
            "thresholds": thr, "false_call_heldout_k": hk, "false_call_heldout_n": hn,
            "false_call_heldout_ci95": _cp(hk, hn) if hn else None}


def _cp(k, n):
    from scipy.stats import beta
    return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]


def direct_patient_level(runs, task, k, alpha=ALPHA):
    """Patient-level calibration measured on repeated null runs instead of projected under independence.
    runs: DataFrame(patient, j, stat); a patient enters if it has a finite statistic on every run j < k (others are
    counted in n_excluded, never padded). Halves by crc32(patient) as crossfit. Direct: each half's threshold is the
    (1 - alpha) 'higher' quantile of its patients' maximum over the k runs; the other half's patients are called if
    their maximum is strictly above it (held-out patient-level false calls, pooled over the halves, exact CI).
    Sidak: each half's threshold is the per-assessment-level quantile of its single-run statistics (j = 0), applied
    to the other half's k runs the same way. Detection: single-assessment task units judged as crossfit judges them.
    -> dict(n_patients, n_excluded, direct, sidak)."""
    r = runs[runs["j"] < k].copy()
    r["stat"] = _clean(r["stat"])
    r = r[np.isfinite(r["stat"])]
    cnt = r.groupby("patient")["j"].nunique()
    keep = cnt.index[cnt == k]
    allp = runs["patient"].nunique()
    r = r[r["patient"].isin(keep)]
    mx = r.groupby("patient")["stat"].max()
    one = r[r["j"] == 0].set_index("patient")["stat"].reindex(mx.index)
    h = half_of(mx.index)
    ts = _clean(task["stat"]) if len(task) else np.array([])
    ok = np.isfinite(ts)
    ht = half_of(task["patient"]) if "patient" in task and len(task) else None
    out = {"k": k, "n_patients": int(len(mx)), "n_excluded": int(allp - len(mx))}
    for name, thr in (("direct", [threshold(mx.values[h == j], alpha) for j in (0, 1)]),
                      ("sidak", [threshold(one.values[h == j], per_assessment_alpha(alpha, k)) for j in (0, 1)])):
        hk = int(sum(np.sum(mx.values[h == j] > thr[1 - j]) for j in (0, 1)))
        hn = int(len(mx))
        if not ok.any():
            det = None
        elif ht is not None:
            det = float(np.mean(ts[ok] > np.where(ht == 0, thr[1], thr[0])[ok]))
        else:
            det = float(np.mean([np.mean(ts[ok] > t) for t in thr]))
        out[name] = {"thresholds": thr, "heldout_k": hk, "heldout_n": hn, "rate": hk / hn if hn else None,
                     "ci95": _cp(hk, hn) if hn else None, "detection": det}
    return out


def load_null_runs(pipe):
    """Every overlay (run) of the published null rows of the frozen variant: DataFrame(patient, j, stat)."""
    null_set, variant, _, _ = PUBLISHED[pipe]
    R = [r for r in _jsonl(os.path.join(INTER, "_bench_shards", f"{null_set}__{pipe}__native__none", "*.jsonl"))
         if r.get("variant") == variant and r.get("status") == "ok" and r.get("resp") in (None, "none")
         and r.get("split") == "analysis"]
    return pd.DataFrame({"patient": [r["patient"] for r in R], "j": [int(r["j"]) for r in R],
                         "stat": [r.get("size_stat") for r in R]}).drop_duplicates(["patient", "j"])


def p_rule_rate(rows, level):
    p = np.asarray(rows["p"], float)
    p = p[np.isfinite(p)]
    return float(np.mean(p <= level + 1e-12)) if len(p) else None


def curve(null, task, levels=LEVELS):
    return [crossfit(null, task, lv, 1) for lv in levels]


def table(null, task, rand=None, alpha=ALPHA, ks=KS):
    out = {}
    for (pipe, reading), nd in null.groupby(["pipe", "reading"]):
        td = task[(task["pipe"] == pipe) & (task["reading"] == reading)]
        rd = rand[rand["pipe"] == pipe] if rand is not None else None
        out[f"{pipe}|{reading}"] = {
            "calibrated": {str(k): crossfit(nd, td, alpha, k) for k in ks},
            "curve": curve(nd, td),
            "randomization_detection": ({str(k): p_rule_rate(rd, per_assessment_alpha(alpha, k)) for k in ks}
                                        if rd is not None and len(rd) else None)}
    return out


PUBLISHED = {
    "cornell": ("ncmd_bdsp", "jackknife_bh_keep40_r4", "openbmi", "theoretical_bh_keep40_r4"),
    "dubois": ("ncmd_bdsp", "assessment_blocked_phase1", "openbmi", "assessment_random_phase1"),
    "mbvt": ("nstim0_bdsp", "vt3_mur20_seg8_median_gt23", "erp", "vt3_mur20_seg8_full_gt23")}
INTER = os.path.join(ROOT, "output", "intermediate")


def _jsonl(pattern):
    import glob
    import json as _json
    out = []
    for f in sorted(glob.glob(pattern)):
        if os.path.basename(f).startswith("._"):
            continue
        out += [_json.loads(l) for l in open(f)]
    return out


def load_published(pipe):
    """The published schedule: null = the clinical-EEG first-overlay rows of the frozen variant (analysis split, no
    responder; statistic = size_stat, rand_p for the randomized method); task = 118's healthy responders in the same
    schedule (true version; size_stat; the randomized method's p on its own randomized session)."""
    null_set, variant, arm, rvar = PUBLISHED[pipe]
    R = [r for r in _jsonl(os.path.join(INTER, "_bench_shards", f"{null_set}__{pipe}__native__none", "*.jsonl"))
         if r.get("variant") == variant and r.get("j") == 0 and r.get("status") == "ok"
         and r.get("resp") in (None, "none") and r.get("split") == "analysis"]
    null = pd.DataFrame({"pipe": pipe, "reading": variant, "unit": [r["uid"] for r in R],
                         "patient": [r["patient"] for r in R], "stat": [r.get("size_stat") for r in R]})
    rnull = pd.DataFrame({"pipe": pipe, "unit": [r["uid"] for r in R], "p": [r.get("rand_p") for r in R]})
    P = _jsonl(os.path.join(INTER, "_positive_control", f"{arm}__{pipe}.jsonl"))
    T = [r for r in P if r["version"] == "true" and r["design"] == "published" and r["variant"] == variant
         and not r.get("dev") and r.get("status") == "ok"]
    task = pd.DataFrame({"pipe": pipe, "reading": variant, "unit": [r["person"] for r in T],
                         "patient": [r["person"] for r in T], "stat": [r.get("size_stat") for r in T],
                         "published_call": [bool(r.get("call")) for r in T]})
    Q = [r for r in P if r["version"] == "true" and r["design"] == "randomized" and r["variant"] == rvar
         and not r.get("dev") and r.get("status") == "ok"]
    rtask = pd.DataFrame({"pipe": pipe, "unit": [r["person"] for r in Q], "p": [r.get("p") for r in Q]})
    return null, task, rnull, rtask


def load_braille(path_glob=None):
    """Tactile control (173): unit = one run (A or B) of a person-session; null = the sham runs' LDA margin, task =
    the true runs' margin; randomization p alike. Patient = participant."""
    rows = _jsonl(path_glob or os.path.join(INTER, "braille_p300_rows_s*.jsonl"))
    out = {}
    for version in ("sham", "true"):
        m = [r for r in rows if r["reading"] == "mbvt_primary" and r["version"] == version and r["run"] in ("A", "B")
             and r.get("call") is not None]
        out[version] = pd.DataFrame({"pipe": "mbvt_braille", "reading": "mbvt_primary",
                                     "unit": [f"{r['subject']}_{r['session']}_{r['run']}" for r in m],
                                     "patient": [str(r["subject"]) for r in m], "stat": [r["margin"] for r in m],
                                     "published_call": [bool(r["call"]) for r in m]})
        q = [r for r in rows if r["reading"] == "randomization" and r["version"] == version and r["run"] in ("A", "B")
             and r.get("call") is not None]
        out["rand_" + version] = pd.DataFrame({"pipe": "mbvt_braille", "unit": [f"{r['subject']}_{r['session']}_{r['run']}"
                                                                                  for r in q], "p": [r["p"] for r in q]})
    return out["sham"], out["true"], out["rand_sham"], out["rand_true"]


def load_stroke(path_glob=None):
    """Stroke control (172): unit = session; Cornell c3 statistic = max |z| (stat), du Bois reading B = paired t;
    null = the rest-only sham draws (Cornell) or the relabelled draws (du Bois) of the same sessions. Patient =
    subject. -> {reading: (null, task)}."""
    rows = _jsonl(path_glob or os.path.join(INTER, "stroke_native_rows_s*.jsonl"))
    import importlib
    dup = importlib.import_module("172_stroke_native").duplicate_sessions(rows)
    rows = [r for r in rows if r["session_id"] not in dup]
    key = {"cornell_c3": "stat", "cornell_cmd_t": "stat", "dubois_reading_b": "t"}
    out = {}
    for reading, col in key.items():
        def df(version):
            m = [r for r in rows if r["reading"] == reading and r["version"] == version and r.get(col) is not None]
            return pd.DataFrame({"pipe": "stroke", "reading": reading,
                                 "unit": [f"{r['session_id']}_{r['draw']}" for r in m],
                                 "patient": [str(r["subject"]) for r in m], "stat": [r[col] for r in m],
                                 "published_call": [bool(r["call"]) for r in m]})
        out[reading] = (df("sham"), df("true"))
    return out


def design_table(null, task, rnull=None, rtask=None, ks=KS):
    """One design: calibrated detection at k, the published rule's own null and detection rates, the randomization
    comparator at k, and the curve."""
    res = {"n_null": int(len(null)), "n_task": int(len(task)), "n_null_patients": int(null["patient"].nunique()),
           "calibrated": {str(k): crossfit(null, task, ALPHA, k) for k in ks}, "curve": curve(null, task)}
    if "published_call" in task:
        res["published_detection"] = float(task["published_call"].mean()) if len(task) else None
    if "published_call" in null:
        res["published_null_rate"] = float(null["published_call"].mean()) if len(null) else None
    if rnull is not None and len(rnull):
        res["randomization"] = {str(k): {"null": p_rule_rate(rnull, per_assessment_alpha(ALPHA, k)),
                                         "detection": p_rule_rate(rtask, per_assessment_alpha(ALPHA, k))
                                         if rtask is not None and len(rtask) else None} for k in ks}
    return res


def run_all(out=os.path.join(ROOT, "output", "tables", "matched_calibration.json")):
    res = {"what": __doc__.split("\n")[0], "designs": {}}
    for pipe in PUBLISHED:
        n, t, rn, rt = load_published(pipe)
        res["designs"][f"published_schedule|{pipe}"] = design_table(n, t, rn, rt)
    try:
        n, t, rn, rt = load_braille()
        res["designs"]["braille_native|mbvt_primary"] = design_table(n, t, rn, rt)
    except (FileNotFoundError, KeyError, ValueError) as e:
        res["designs"]["braille_native|mbvt_primary"] = {"status": f"unavailable: {e}"[:160]}
    for reading, (n, t) in load_stroke().items():
        res["designs"][f"stroke_native|{reading}"] = design_table(n, t) if len(n) and len(t) else {"status": "no rows"}
    res["direct"] = {}
    for pipe, ks_direct in (("mbvt", (2, 5, 10)), ("dubois", (2,))):
        _, task, _, _ = load_published(pipe)
        runs = load_null_runs(pipe)
        res["direct"][f"published_schedule|{pipe}"] = {str(k): direct_patient_level(runs, task, k) for k in ks_direct}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--null")
    ap.add_argument("--task")
    ap.add_argument("--rand")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "tables", "matched_calibration.json"))
    a = ap.parse_args()
    if a.all:
        res = run_all()
        for d, v in res["designs"].items():
            if "calibrated" in v:
                print(d, {k: (round(c["false_call_heldout"], 3), None if c["detection"] is None else round(c["detection"], 3))
                          for k, c in v["calibrated"].items()}, "published det", v.get("published_detection"),
                      "rand", {k: x["detection"] for k, x in v.get("randomization", {}).items()})
        return
    res = {"what": __doc__.split("\n")[0],
           "pipes": table(pd.read_csv(a.null), pd.read_csv(a.task), pd.read_csv(a.rand) if a.rand else None)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: {kk: v["calibrated"][kk]["detection"] for kk in v["calibrated"]} for k, v in res["pipes"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
