"""Fidelity check by an independent implementation: du Bois (Phase I advancement) and the vibrotactile P300 (accuracy
above 23%) written from the publication texts alone, without access to this study's code (blind_impl/), run on the
same clinical EEG development windows and the same sham schedules as this study's primary readings. Bars fixed here:
the blind rate lies inside the range of readings consistent with the publication (eTable 21), and call agreement
kappa >= 0.6. Reported whatever they show. The P300 schedule switches the target wrist between instructions, so each
stimulus is relabeled target ("left"), non-target wrist ("right") or distractor, with target="left"; the blind method
is not changed. -> output/tables/blind_agreement.json (no identifiers)."""
import importlib
import json
import os
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
for p in (CODE, os.path.join(ROOT, "blind_impl")):
    if p not in sys.path:
        sys.path.insert(0, p)
rp = importlib.import_module("153_replication_rows")
b, fid, st = rp.b, rp.fid, rp.st
OUT = os.path.join(ROOT, "output", "tables", "blind_agreement.json")
N, SEED, MARGIN_S = 200, 20261022, 5.0
DUBOIS_READING = "c3_b_assessment_blocked_phase1"
HAND = {1: "left", 2: "right"}
SITE = {1: "left", 2: "right", 3: "distractor"}


def dubois_events(ov, start_s):
    return [(float(t) - start_s, HAND[int(l)], int(k)) for t, l, k in zip(ov.onsets_s, ov.labels, ov.meta["block"])]


def p300_events(ov, start_s):
    return [(round(float(t) - start_s, 9), SITE[int(l)]) for t, l in zip(ov.onsets_s, ov.labels)]


def kappa(a, c):
    a, c = np.asarray(a, bool), np.asarray(c, bool)
    po = float((a == c).mean())
    pe = float(a.mean() * c.mean() + (1 - a.mean()) * (1 - c.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _crop(rec, ov, mod):
    t0, t1 = (float(v) for v in mod.extent(ov))
    a = max(0.0, t0 - MARGIN_S)
    i0, i1 = int(round(a * rec.fs)), min(rec.x.shape[1], int(round((t1 + MARGIN_S) * rec.fs)))
    return rec.x[:, i0:i1], a


def windows():
    c = st.c3_rows()
    c = c[(c["null"] == "ncmd_bdsp") & (c["pipe"] == "dubois") & (c["reading"] == DUBOIS_READING)
          & (c["status"] == "ok") & (c["j"].fillna(-1).astype(int) == 0)]
    local = {f[:-4] for f in os.listdir(os.path.join(rp.DEV_SIG, "sig")) if f.endswith(".npz") and not f.startswith("._")}
    uids = sorted(set(c["uid"]) & local)
    rng = np.random.default_rng(SEED)
    return [uids[i] for i in rng.permutation(len(uids))[:N]], dict(zip(c["uid"], c["call"].astype(bool)))


def run_one(uid, ours_dubois):
    import dubois_blind
    import p300_blind
    out = {}
    rec = b.en._load(rp.dev_item(uid))
    mod = b.load_pipe("dubois")
    jobs, _ = fid.phase1_jobs(rec, mod, rp.rep_spec("dubois"), 974, first_only=True)
    jb = [j for j in jobs if j["reading"] == DUBOIS_READING and j["ov"] is not None]
    if jb:
        x, a = _crop(rec, jb[0]["ov"], mod)
        r = dubois_blind.assess(x, rec.fs, rec.names, dubois_events(jb[0]["ov"], a), seed=0)
        out["dubois"] = (bool(ours_dubois), bool(r["call"]))
    mt = b.load_pipe("mbvt")
    cap, run0 = [], mt.run

    def spy(rec_, ov, variant=None, rng=None, *aa, **kk):
        res = run0(rec_, ov, variant, rng, *aa, **kk)
        cap.append((ov, variant, res))
        return res
    mt.run = spy
    try:
        rows = rp.rows_for_item(rp.dev_item(uid), "mbvt", n_recs=974, split="test")
    finally:
        mt.run = run0
    first = [r for r in rows if r.get("j") == 0 and r.get("status") == "ok"
             and r.get("variant") == rp.FROZEN["mbvt"]["variant"]]
    hit = [c for c in cap if c[1] == rp.FROZEN["mbvt"]["variant"]]
    if first and hit:
        ov = hit[0][0]
        x, a = _crop(rec, ov, mt)
        r = p300_blind.assess(x, rec.fs, rec.names, p300_events(ov, a), seed=0, target="left")
        out["mbvt"] = (bool(first[0]["call"]), bool(r["call"]))
    b._clear_cache()
    return out


def main(out=OUT):
    uids, ours = windows()
    pairs = {"dubois": [], "mbvt": []}
    for i, u in enumerate(uids):
        try:
            for k, v in run_one(u, ours[u]).items():
                pairs[k].append(v)
        except Exception as e:
            print("error", i, type(e).__name__, str(e)[:120], flush=True)
        print(i, {k: len(v) for k, v in pairs.items()}, flush=True)
    cats = json.load(open(os.path.join(ROOT, "output", "tables", "transport_bounds.json")))["categories"]
    res = {"what": __doc__.split("\n")[0], "n_windows": len(uids), "pipes": {}}
    for k, v in pairs.items():
        o, bl = np.array([p[0] for p in v], bool), np.array([p[1] for p in v], bool)
        rng_ = cats[k]["text_consistent"]
        res["pipes"][k] = {"n": int(len(v)), "ours": st.exact_cell(int(o.sum()), len(o)),
                           "blind": st.exact_cell(int(bl.sum()), len(bl)), "kappa": kappa(o, bl),
                           "agree": float((o == bl).mean()), "reading_range": [rng_["min"], rng_["max"]],
                           "blind_inside_range": bool(rng_["min"] <= bl.mean() <= rng_["max"]),
                           "kappa_at_least_0_6": bool(kappa(o, bl) >= 0.6)}
    rp.sr.assert_no_identifiers(res)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps(res["pipes"], indent=1, default=float))


if __name__ == "__main__":
    main()
