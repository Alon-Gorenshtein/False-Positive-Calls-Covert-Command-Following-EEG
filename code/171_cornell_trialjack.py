"""Cornell with a dependence-aware variance. The study's Cornell reading (130, c3_jackknife_bh_keep40_r4_om2contig)
takes the Two Group Test's jackknife over samples = epochs x tapers, as if every 3-s epoch and every taper were
independent. The three epochs of one command's response period, and the five tapers of one epoch, share the command's
state (arousal, artefact, slow amplitude drift), so a leave-one-sample-out jackknife understates the variance of the
log-spectral difference. Here the jackknife deletes one COMMAND at a time (all its kept epochs x tapers), the
delete-a-group jackknife (G - 1) / G * sum of squared deviations per group, the two groups' variances summed (the
module's Arvesen form: each leave-one-out statistic divided by the full-sample SD, as pipe_cornell.tgt). Everything
else is the c3 reading unchanged: features, pruning (keep40), BH over channels x 4-24 Hz on the all-runs TGT, OM1 on
the per-run TGTs, OM2 as FDR-surviving same-sign bins in one channel spanning > 2 Hz, call = OM1 and OM2.

Reference distribution of z: 'normal' (as the c3 reading; the only change is the deletion unit) or 't' with
G1 + G2 - 2 degrees of freedom (the jackknife variance from G commands has about G - 1 df; secondary).
unit='sample' reproduces the c3 reading exactly (tested), so a difference between the two readings on the same
windows is the deletion unit alone.

Readings fixed here (descriptive, reported whatever they show), on the development windows of the c3 reading and on
the positive-control sessions: 'calibrated' if the upper bound of the exact 95% CI of the first-overlay call rate is
<= 10% (the paper's specificity bound for the control analyses); 'detection kept' if, on real-responder sessions,
the command-unit reading calls >= 80% of the people the c3 reading calls.

python 171_cornell_trialjack.py --synthetic N [--drift D] [--workers W]
python 171_cornell_trialjack.py --dev [--limit N] [--workers W]       (the c3 reading's development windows)
python 171_cornell_trialjack.py --merge ROWS.jsonl [--out PATH] -> output/tables/cornell_trialjack[_synthetic].json"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import importlib
import json
import sys

import numpy as np
from scipy.special import polygamma, psi
from scipy.stats import beta, norm, t as tdist

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from bench import motor_common as mc
from bench import pipe_cornell as pc

BASE = "jackknife_bh_keep40_r4"
READING = "c3_jackknife_bh_keep40_r4_om2contig"
OUT_DIR = os.path.join(ROOT, "output", "tables")


def _z(s, m):
    return np.log(s / m) - (psi(m) - np.log(m))


def tgt_grouped(S1, S2, g1, g2):
    """Two Group Test with a delete-a-group jackknife. S (n, K, ch, f) |J|^2 of n epochs x K tapers; g (n) group
    (command) of each epoch. -> dz (ch, f), vdz (ch, f), df = G1 + G2 - 2."""
    n1, K = S1.shape[:2]
    n2 = S2.shape[0]
    m1, m2 = n1 * K, n2 * K
    s1, s2 = S1.sum((0, 1)), S2.sum((0, 1))
    z1, z2 = _z(s1, m1), _z(s2, m2)
    sd = np.sqrt(polygamma(1, m1) + polygamma(1, m2))
    dz = (z1 - z2) / sd

    def part(Sx, gx, sx, mx, other, sign):
        ids = np.unique(gx)
        d = []
        for j in ids:
            sel = gx == j
            mj = int(sel.sum()) * K
            zj = _z(sx - Sx[sel].sum((0, 1)), mx - mj)
            d.append(sign * (zj - other) / sd)
        d = np.asarray(d)
        G = len(ids)
        return (G - 1) / G * ((d - d.mean(0)) ** 2).sum(0), G
    v1, G1 = part(S1, np.asarray(g1), s1, m1, z2, 1.0)
    v2, G2 = part(S2, np.asarray(g2), s2, m2, z1, -1.0)
    return dz, v1 + v2, G1 + G2 - 2


def _zp(S1, S2, g1, g2, unit, ref):
    """-> (z, two-sided p, critical |z| at 0.05)."""
    if unit == "sample":
        dz, vdz = pc.tgt(S1, S2)
        df = None
    else:
        dz, vdz, df = tgt_grouped(S1, S2, g1, g2)
    z = dz / np.sqrt(vdz)
    if ref == "t" and df is not None and df > 0:
        return z, 2 * tdist.sf(np.abs(z), df), float(tdist.ppf(0.975, df))
    return z, 2 * norm.sf(np.abs(z)), 1.959963984540054


def cornell_unit(rec, ov, unit="command", ref="normal", base_variant=BASE, grid="full"):
    """The c3 reading with the jackknife deletion unit `unit` ('sample' | 'command') and reference `ref`."""
    F = dict(pc._feat(rec, ov))
    return decide(F, np.asarray(ov.labels)[F["cmd"]], unit, ref, base_variant, grid=grid)


POSTERIOR = ("O1", "O2", "P7", "P8", "T5", "T6")


DEC = 6


def decimate(F, step=DEC):
    """F on every step-th frequency bin (4, 6, ..., 24 Hz at the 1/3-Hz grid): one bin per 2W = 2-Hz resolution
    element of the 5-taper estimate on 3-s epochs, so FDR and the contiguity rule count independent bins."""
    return {**F, "S": F["S"][..., ::step], "f": F["f"][::step]}


def decide(F, lab_ep, unit="command", ref="normal", base_variant=BASE, names=None, grid="full"):
    """cornell_unit on precomputed features F (pc.features layout; epochs of several runs may be merged, with
    distinct F['cmd'] per command and F['run'] per run) and per-epoch labels. grid 'dec': decimate(F) first.
    extra['vif']: median over channels of the band-mean command-level jackknife variance of dz on all runs (the
    variance inflation over the test's theoretical variance, which is 1 in dz units)."""
    if grid == "dec":
        F = decimate(F)
    c = mc.cfg(pc.VARIANTS, base_variant)
    keep = pc.kept(F, lab_ep, c["prune"])
    pc._require(F, lab_ep, keep)
    runs = sorted(np.unique(F["run"]).tolist())
    sigs = {}
    for r in runs:
        m1 = keep & (F["run"] == r) & (lab_ep == pc.TASK)
        m2 = keep & (F["run"] == r) & (lab_ep == pc.REST)
        if m1.sum() < 2 or m2.sum() < 2:
            continue
        z, _, zc = _zp(F["S"][m1], F["S"][m2], F["cmd"][m1], F["cmd"][m2], unit, ref)
        sigs[r] = (np.abs(z) > zc, np.sign(z))
    om1 = False
    for r1, (sg, sn) in sigs.items():
        for ch, i0, i1, s, rng_hz in pc.regions(sg, sn, F["f"]):
            if rng_hz <= pc.MIN_RANGE_HZ:
                continue
            for r2, (sg2, sn2) in sigs.items():
                if r2 != r1 and np.any(sg2[ch, i0:i1 + 1] & (sn2[ch, i0:i1 + 1] == s)):
                    om1 = True
    inr = np.isin(F["run"], runs)
    m1, m2 = keep & inr & (lab_ep == pc.TASK), keep & inr & (lab_ep == pc.REST)
    z, p, _ = _zp(F["S"][m1], F["S"][m2], F["cmd"][m1], F["cmd"][m2], unit, ref)
    vif = float(np.median(tgt_grouped(F["S"][m1], F["S"][m2], F["cmd"][m1], F["cmd"][m2])[1].mean(1)))
    rej = mc.fdr_reject(p.ravel(), pc.Q_FDR, c["fdr"]).reshape(z.shape)
    contig = any(rg[4] > pc.MIN_RANGE_HZ for rg in pc.regions(rej, np.sign(z), F["f"]))
    nm = mc.canon(names if names is not None else pc.NATIVE_CH)
    post = np.array([n in POSTERIOR for n in nm])
    return mc.result(float(np.abs(z).max()), float(p.min()), om1 and contig, om1=bool(om1), om2_contig=bool(contig),
                     n_sig_allruns=int(rej.sum()), n_sig_posterior=int(rej[post].sum()), unit=unit, ref=ref,
                     grid=grid, vif=vif, n_kept=int(keep.sum()))


READINGS = {"sample_normal": ("sample", "normal", "full"), "command_normal": ("command", "normal", "full"),
            "command_t": ("command", "t", "full"), "sample_normal_dec": ("sample", "normal", "dec"),
            "command_t_dec": ("command", "t", "dec")}


def run_unit(rec, ov, unit_id):
    rows = []
    for name, (u, r, gr) in READINGS.items():
        try:
            res = cornell_unit(rec, ov, u, r, grid=gr)
            rows.append({"unit": unit_id, "reading": name, "call": res["call"], "om1": res["extra"]["om1"],
                         "om2_contig": res["extra"]["om2_contig"], "zmax": res["stat"], "vif": res["extra"]["vif"]})
        except mc.NoUsableTrials as e:
            rows.append({"unit": unit_id, "reading": name, "call": None, "status": "no_usable", "why": str(e)[:120]})
    return rows


def _synthetic_unit(args):
    i, drift = args
    from bench.tests import _synth as S
    g = np.random.default_rng([20261029, 171, i])
    ov = pc.schedule(20.0, None, BASE)
    rec = S.rec(pc.extent(ov)[1] + 10.0, seed=int(g.integers(2 ** 31)), drift=drift)
    return run_unit(rec, ov, f"syn{i}")


def _cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def summarize(rows):
    out = {"readings": {}}
    for name in READINGS:
        c = [r["call"] for r in rows if r["reading"] == name and r.get("call") is not None]
        k, n = int(sum(c)), len(c)
        out["readings"][name] = {"k": k, "n": n, "rate": k / n if n else None, "ci95": _cp(k, n) if n else None,
                                 "calibrated_upper_le_10pct": bool(n and _cp(k, n)[1] <= 0.10)}
    base = [r for r in rows if r["reading"] == "sample_normal" and r.get("vif") is not None]
    if base:
        v = np.array([r["vif"] for r in base], float)
        pos = np.array([bool(r["call"]) for r in base])
        out["vif"] = {"n": len(v), "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)),
                      "q75": float(np.quantile(v, 0.75)), "share_ge_2": float(np.mean(v >= 2)),
                      "median_sample_positive": float(np.median(v[pos])) if pos.any() else None,
                      "median_sample_negative": float(np.median(v[~pos])) if (~pos).any() else None}
    st = [r["matches_stored"] for r in base if "matches_stored" in r]
    if st:
        out["stored_calls_reproduced"] = {"k": int(sum(st)), "n": len(st)}
    return out


def _pool(fn, items, workers):
    if workers <= 1:
        for it in items:
            yield fn(it)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(fn, items)


def dev_windows():
    """[(uid, stored c3 call)]: every development window with a stored first-overlay c3 Cornell row on clinical EEG
    whose recording is on this machine (161's selection rule, without its 200 subsample)."""
    b161 = importlib.import_module("161_blind_agreement")
    rp, st = b161.rp, b161.st
    c = st.c3_rows()
    c = c[(c["null"] == "ncmd_bdsp") & (c["pipe"] == "cornell") & (c["reading"] == READING) & (c["status"] == "ok")
          & (c["j"].fillna(-1).astype(int) == 0)]
    local = {f[:-4] for f in os.listdir(os.path.join(rp.DEV_SIG, "sig")) if f.endswith(".npz") and not f.startswith("._")}
    stored = dict(zip(c["uid"], c["call"].astype(bool)))
    return [(u, stored[u]) for u in sorted(set(c["uid"]) & local)]


def _dev_unit(item):
    """One development window of the c3 reading: its first published Cornell overlay (130.phase1_jobs); the
    sample-level reading must reproduce the stored c3 call (matches_stored)."""
    uid, stored = item
    rp = importlib.import_module("153_replication_rows")
    fr = importlib.import_module("130_fidelity_readings")
    rec = rp.b.en._load(rp.dev_item(uid))
    mod = rp.b.load_pipe("cornell")
    jobs, _ = fr.phase1_jobs(rec, mod, rp.rep_spec("cornell"), 974, first_only=True)
    jb = [j for j in jobs if j["reading"] == READING and j["ov"] is not None]
    if not jb:
        return []
    rows = run_unit(rec, jb[0]["ov"], uid)
    for r in rows:
        if r["reading"] == "sample_normal" and r.get("call") is not None:
            r["stored_call"], r["matches_stored"] = bool(stored), bool(r["call"]) == bool(stored)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rows")
    ap.add_argument("--merge")
    ap.add_argument("--out", help="merge output (default output/tables/cornell_trialjack[_synthetic].json)")
    a = ap.parse_args()
    if a.merge:
        rows = [json.loads(l) for l in open(a.merge)]
        res = {"what": __doc__.split("\n")[0], **summarize(rows)}
        tag = "_synthetic" if all(r["unit"].startswith("syn") for r in rows) else ""
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump(res, open(a.out or os.path.join(OUT_DIR, f"cornell_trialjack{tag}.json"), "w"), indent=1, default=str)
        print(json.dumps(res, indent=1, default=str))
        return
    if a.synthetic:
        items, fn = [(i, a.drift) for i in range(a.synthetic)], _synthetic_unit
        default = os.path.join(ROOT, "output", "intermediate", f"cornell_trialjack_syn_d{a.drift:g}.jsonl")
        key = lambda it: f"syn{it[0]}"
    else:
        items, fn = dev_windows(), _dev_unit
        default = os.path.join(ROOT, "output", "intermediate", "cornell_trialjack_dev.jsonl")
        key = lambda it: it[0]
    items = items[:a.limit] if a.limit else items
    path = a.rows or default
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {json.loads(l)["unit"] for l in open(path)} if os.path.exists(path) else set()
    todo = [it for it in items if key(it) not in done]
    with open(path, "a") as fh:
        for k, rows in enumerate(_pool(fn, todo, a.workers)):
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(k + 1, "/", len(todo), flush=True)


if __name__ == "__main__":
    main()
