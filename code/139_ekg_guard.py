"""Re-aligned beat-template guard for the EKG heart-rate measures, and its check on DISCOVERY data (the Phase-1
full-span EKG). Built test-first (code/test_ekg_guard.py).

Why: the check's detector (132) marks the steepest slope of the 5-25 Hz band-passed EKG. On a symmetric R wave
the up- and down-stroke are equally steep, so the mark flips between them from beat to beat; the +/- 100 ms beats
then misalign and their correlation with the median template falls below 0.6 on clean ECG. 132's guard would call
such a span noise.

Guard (a_hr_guarded, guard="aligned"): each detected peak is moved to the apex of the band-passed EKG within +/- 50 ms
in the span's dominant polarity (realign_peaks: the sign whose median beat extreme is the larger) before the
beat-template correlation; NA below TEMPLATE_MIN_ALIGNED = 0.85. Re-centring any signal on its own peaks builds a
common shape: re-aligned noise reaches 0.62-0.66 (white, pink, brown) and 0.75 (high-passed white, 60-s spans), so
132's 0.6 would admit it; 0.85 keeps a 0.1 margin over the highest. The RR rule and the median heart rate (30-200 bpm)
are 132's, in 132's order; a heart-rate series spanning under 100 s is NA ("short_hr_series"). Heart rate stays on the detector's peaks (A_HR is unchanged wherever both guards pass);
hr_peaks="aligned" also moves the beats (reported as a variant only). Not caught by either template guard: narrow-band
5-25 Hz interference (a sinusoid re-aligns to r ~0.9-1.0; 132's guard rejects it only because its sign flips). Two
extra rules, applied to spans that pass every other one (extra_rule_reason, rules=): "rr_floor" (metronomic beats:
RR SD of the re-aligned peaks below 5 ms, or median RR within 10 ms of the 0.3-s refractory) and "prominence" (the
detector signal at the beats over its largest value between them, median over beats, below 2: a sinusoid gives
1.0-1.13 and its "RR" SD is 30-100 ms, so the RR floor does not catch it; white and pink noise 1.25).

Check (--check, --check-summary): every Phase-1 recording of the full-span EKG (136_ekg_span; checked by 137's
load_full_ekg), consecutive non-overlapping spans of 1800 s (A_HR's window) and 300 s (tile scale) from the start;
per span 132's guard, the re-aligned guard and the moved-beat variant. Reported: NA by reason under each guard, the
old-by-new reason table, spans recovered and lost, the median heart rate of each group, A_HR agreement where both
pass, and the re-aligned guard with each extra rule (VARIANTS).
python 139_ekg_guard.py --check [--shard i --nshards n] | --check-summary"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import hashlib
import importlib
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

kt = importlib.import_module("132_ekg_check")
ex = importlib.import_module("137_arousal_explore")
mc = kt.mc

ALIGN_S = 0.05
TEMPLATE_MIN_ALIGNED = 0.85
RR_SD_MIN_MS, LOCK_IN_S, LOCK_IN_TOL_S = 5.0, 0.30, 0.010
PROMINENCE_MIN = 2.0
CHECK_SPANS_S = (1800, 300)
OUT = os.path.join(ROOT, "output", "intermediate", "_ekg_guard")


def realign_peaks(x, peaks, fs):
    """Detected peaks moved to the apex of the band-passed EKG x within +/- 50 ms, in the span's dominant polarity
    (the sign whose median beat extreme is the larger); peaks whose search window leaves x are dropped."""
    a = int(round(ALIGN_S * fs))
    p = np.asarray(peaks, int)
    p = p[(p - a >= 0) & (p + a + 1 <= len(x))]
    if len(p) == 0:
        return p
    S = np.array([x[q - a:q + a + 1] for q in p], float)
    sign = 1.0 if np.median(S.max(1)) >= np.median(-S.min(1)) else -1.0
    return p - a + np.argmax(sign * S, 1)


def beat_diagnostics(p, x, q, fs):
    """RR SD (ms) and median RR (s) over the kept intervals (0.3-2.0 s) of the re-aligned peaks q (the apex times:
    the detector's mark adds +/- 1 sample of jitter, and a fixed RR on the 128-Hz grid alone gives 3.5 ms); beat prominence
    of the detector's peaks p: the median of the detector signal |diff x| at the beats over the median of its largest
    value between beats (outside +/- 100 ms of each); NaN when undefined."""
    rr = np.diff(np.asarray(q, float)) / fs
    rr = rr[(rr >= kt.RR_OK[0]) & (rr <= kt.RR_OK[1])]
    d = np.abs(np.diff(x))
    h = int(round(kt.TEMPLATE_HALF_S * fs))
    p = np.asarray(p, int)
    gaps = [d[a + h:b - h].max() for a, b in zip(p[:-1], p[1:]) if b - h > a + h]
    den = float(np.median(gaps)) if gaps else float("nan")
    return {"rr_sd_ms": float(1000 * rr.std(ddof=1)) if len(rr) > 1 else float("nan"),
            "rr_median_s": float(np.median(rr)) if len(rr) else float("nan"),
            "beat_prominence": float(np.median(d[p]) / den) if len(p) and den > 0 else float("nan")}


def extra_rule_reason(diag, rules):
    """The first failing extra rule, applied to a span that passes every other one: "rr_floor" (metronomic beats:
    RR SD below 5 ms, or the median RR within 10 ms of the 0.3-s refractory), "prominence" (the beats do not stand
    out of the cardiac cycle: beat prominence below 2)."""
    if "rr_floor" in rules and (not diag["rr_sd_ms"] >= RR_SD_MIN_MS
                                or abs(diag["rr_median_s"] - LOCK_IN_S) <= LOCK_IN_TOL_S):
        return "metronomic_rr"
    if "prominence" in rules and not diag["beat_prominence"] >= PROMINENCE_MIN:
        return "low_beat_prominence"
    return None


def a_hr_guarded(ekg, fs, guard="old", hr_peaks="detector", rules=()):
    """132's a_hr under its own guard ("old": beats at the detector's peaks, template >= 0.6) or the re-aligned one
    ("aligned": beats at realign_peaks, template >= TEMPLATE_MIN_ALIGNED); 132's rule order (RR rule, template, median
    heart rate), then "short_hr_series" when the kept beats span under one period of the band's low edge (100 s; 132
    has no such rule and fails on it). Heart rate from the detector's peaks (hr_peaks="aligned": the re-aligned ones)."""
    p, x = kt._detect(ekg, fs)
    q = realign_peaks(x, p, fs)
    hp = p if hr_peaks == "detector" else q
    t, hr, share = kt.hr_series(hp, fs)
    tc = kt.template_corr(x, p if guard == "old" else q, fs)[0]
    mhr = float(60.0 / np.median(np.diff(hp) / fs)) if len(hp) > 1 else float("nan")
    out = {"n_beats": int(len(hp)), "rr_ok_share": share, "template_corr": tc, "median_hr": mhr,
           **beat_diagnostics(p, x, q, fs)}
    if len(hr) < 2 or not share >= kt.RR_MIN_SHARE:
        return dict(out, a_hr=None, reason="rr_implausible")
    if not tc >= (kt.TEMPLATE_MIN if guard == "old" else TEMPLATE_MIN_ALIGNED):
        return dict(out, a_hr=None, reason="low_template_correlation")
    if not kt.MEDIAN_HR[0] <= mhr <= kt.MEDIAN_HR[1]:
        return dict(out, a_hr=None, reason="median_hr_out_of_range")
    if t[-1] - t[0] < 1.0 / kt.HR_BAND[0]:
        return dict(out, a_hr=None, reason="short_hr_series")
    sd = float(mc.bandpass(hr - hr.mean(), kt.HR_FS, *kt.HR_BAND).std())
    if not sd > 0:
        return dict(out, a_hr=None, reason="no_variance")
    why = extra_rule_reason(out, rules)
    return dict(out, a_hr=None if why else math.log(sd), reason=why)



def recording_check(ekg, fs, uid):
    """Per consecutive span (CHECK_SPANS_S) of one recording: 132's guard, the re-aligned guard, the moved-beat
    variant."""
    rows = []
    for L in CHECK_SPANS_S:
        n = int(round(L * fs))
        for k, a in enumerate(range(0, len(ekg) - n + 1, n)):
            seg = ekg[a:a + n]
            r = {"uid": uid, "span_s": L, "k": k, "t0": a / fs}
            for tag, g, hp in (("old", "old", "detector"), ("new", "aligned", "detector"),
                               ("moved", "aligned", "aligned")):
                o = a_hr_guarded(seg, fs, g, hp)
                r.update({f"{tag}_{c}": o[c] for c in ("reason", "a_hr", "template_corr", "median_hr", "rr_ok_share",
                                                       "rr_sd_ms", "rr_median_s", "beat_prominence")})
            rows.append(r)
    return rows


def own_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _path(uid):
    return os.path.join(OUT, "rec", f"{uid}.json")


def _uids():
    pool = ex.phase1_pool()
    man = pd.read_csv(os.path.join(ex.EKG_FULL, "manifest.csv"), dtype={"uid": str, "status": str, "channel": str})
    return pool, man.set_index("uid"), sorted(set(man["uid"]) & set(pool))


def run_check(shard, nshards):
    pool, man, uids = _uids()
    for i, uid in enumerate(uids):
        if i % nshards != shard or os.path.exists(_path(uid)):
            continue
        ex.assert_discovery(pool[uid])
        r = man.loc[uid]
        if r["status"] != "ok":
            rows = [{"uid": uid, "status": r["status"]}]
        else:
            n = ex.x19_length(pool[uid]["path"])
            ekg = ex.load_full_ekg(os.path.join(ex.EKG_FULL, f"{uid}.npz"), dict(r.to_dict(), uid=uid), n)
            rows = recording_check(ekg, float(r["fs"]), uid)
        os.makedirs(os.path.dirname(_path(uid)), exist_ok=True)
        with open(_path(uid) + ".tmp", "w") as fh:
            json.dump({"sha256_139": own_sha256(), "uid": uid, "rows": rows}, fh, default=float)
        os.replace(_path(uid) + ".tmp", _path(uid))
        print(f"{i + 1}/{len(uids)} {uid}: {len(rows)} spans", flush=True)


def agreement(a, b):
    """Mean and max |b - a| over the pairs with both values; with 3 or more, Pearson and Spearman r and the SD of
    b - a."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = np.isfinite(a) & np.isfinite(b)
    if not k.any():
        return {"n": 0}
    d = b[k] - a[k]
    out = {"n": int(k.sum()), "mean_diff": float(d.mean()), "max_abs_diff": float(np.abs(d).max())}
    if k.sum() >= 3:
        out.update(pearson=float(np.corrcoef(a[k], b[k])[0, 1]), spearman=float(spearmanr(a[k], b[k]).correlation),
                   sd_diff=float(d.std(ddof=1)))
    return out


VARIANTS = {"aligned": (), "aligned_rr_floor": ("rr_floor",), "aligned_prominence": ("prominence",),
            "aligned_rr_floor_prominence": ("rr_floor", "prominence")}


def variant_reason(d, rules):
    """The re-aligned guard's reason per span with the extra rules applied to the spans it passes."""
    diag = d[["new_rr_sd_ms", "new_rr_median_s", "new_beat_prominence"]].astype(float).to_numpy()
    out = [r if isinstance(r, str) else extra_rule_reason(
        {"rr_sd_ms": a, "rr_median_s": b, "beat_prominence": c}, rules) for r, (a, b, c) in zip(d["new_reason"], diag)]
    return pd.Series(out, index=d.index, dtype=object).fillna("ok")


def variant_summary(d, old, rules):
    v = variant_reason(d, rules)
    grp = np.select([(old == "ok") & (v == "ok"), (old != "ok") & (v == "ok"), (old == "ok") & (v != "ok")],
                    ["both_pass", "recovered", "lost"], "both_na")
    return {"na": v[v != "ok"].value_counts().to_dict(), "na_rate": float((v != "ok").mean()),
            "groups_vs_old": pd.Series(grp).value_counts().to_dict(),
            "recordings_with_a_usable_span": int(d.loc[v == "ok", "uid"].nunique()),
            "median_hr_recovered": d.loc[grp == "recovered", "new_median_hr"].median()}


def span_summary(d):
    """One span length: NA by reason under each guard, the old-by-new table, recovered and lost spans, the median
    heart rate of each group, and A_HR agreement where both pass."""
    lab = lambda c: d[c].fillna("ok").astype(str)
    old, new = lab("old_reason"), lab("new_reason")
    grp = np.select([(old == "ok") & (new == "ok"), (old != "ok") & (new == "ok"), (old == "ok") & (new != "ok")],
                    ["both_pass", "recovered", "lost"], "both_na")
    mhr = lambda m: d.loc[m, "new_median_hr"].describe(percentiles=[0.1, 0.5, 0.9]).to_dict()
    both = (old == "ok") & (new == "ok")
    return {"n_spans": int(len(d)), "n_recordings": int(d["uid"].nunique()),
            "na_old": old[old != "ok"].value_counts().to_dict(), "na_new": new[new != "ok"].value_counts().to_dict(),
            "na_rate_old": float((old != "ok").mean()), "na_rate_new": float((new != "ok").mean()),
            "old_by_new": pd.crosstab(old, new).to_dict(), "groups": pd.Series(grp).value_counts().to_dict(),
            "recordings_with_a_usable_span": {"old": int(d.loc[old == "ok", "uid"].nunique()),
                                              "new": int(d.loc[new == "ok", "uid"].nunique())},
            "median_hr_by_group": {g: mhr(grp == g) for g in ("both_pass", "recovered", "lost", "both_na")},
            "new_template_corr_by_group": {g: d.loc[grp == g, "new_template_corr"].describe(
                percentiles=[0.1, 0.5, 0.9]).to_dict() for g in ("both_pass", "recovered", "lost", "both_na")},
            "a_hr_where_both_pass": agreement(d.loc[both, "old_a_hr"], d.loc[both, "new_a_hr"]),
            "a_hr_moved_beats_where_both_pass": agreement(d.loc[both & d["moved_reason"].isna(), "old_a_hr"],
                                                          d.loc[both & d["moved_reason"].isna(), "moved_a_hr"]),
            "na_moved": lab("moved_reason")[lab("moved_reason") != "ok"].value_counts().to_dict(),
            "variants": {k: variant_summary(d, old, r) for k, r in VARIANTS.items()},
            "beat_prominence_by_group": {g: d.loc[grp == g, "new_beat_prominence"].describe(
                percentiles=[0.01, 0.05, 0.5]).to_dict() for g in ("both_pass", "recovered", "lost", "both_na")},
            "rr_sd_ms_by_group": {g: d.loc[grp == g, "new_rr_sd_ms"].describe(percentiles=[0.01, 0.05, 0.5]).to_dict()
                                  for g in ("both_pass", "recovered", "lost", "both_na")}}


def check_summary():
    pool, man, uids = _uids()
    missing = [u for u in uids if not os.path.exists(_path(u))]
    if missing:
        raise RuntimeError(f"{len(missing)} recording(s) not checked (e.g. {missing[:3]})")
    rows = []
    for u in uids:
        o = json.load(open(_path(u)))
        if o["sha256_139"] != own_sha256():
            raise RuntimeError(f"{u}: check result of another version of 139")
        rows += o["rows"]
    d = pd.DataFrame(rows)
    st = d["status"] if "status" in d else pd.Series(np.nan, index=d.index)
    out = {"label": "re-aligned beat-template guard, check on the Phase-1 full-span EKG (discovery data)",
           "sha256_139": own_sha256(), "template_min_aligned": TEMPLATE_MIN_ALIGNED, "align_s": ALIGN_S,
           "n_recordings": len(uids), "ekg_not_ok": st.dropna().value_counts().to_dict(),
           "by_span_s": {str(L): span_summary(d[d["span_s"] == L]) for L in CHECK_SPANS_S}}
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, "check_rows.csv"), index=False)
    with open(os.path.join(OUT, "check_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({L: {k: v[k] for k in ("n_spans", "na_rate_old", "na_rate_new", "groups")}
                      for L, v in out["by_span_s"].items()}, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--check", action="store_true")
    m.add_argument("--check-summary", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if a.check:
        run_check(a.shard, a.nshards)
    else:
        check_summary()


if __name__ == "__main__":
    main()
