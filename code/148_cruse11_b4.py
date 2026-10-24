"""cruse11 b4 natives on b4-sized windows (added later). Built test-first
(code/test_148_cruse11_b4.py). 114 ran each pipeline's variants in one tier, so cruse11's 4-block natives only ran
where the 6-block window (3,264 s) fits; here they run in their own tier (114.need_seconds: 2,186 s).
- 114's machinery unchanged: 114.run_recording with 114.variant_specs restricted, inside this process, to cruse11's
  BDSP natives with n_blocks 4 (b4_natives), so tiers, timing groups (their randomised arms), overlays, seeds, keys
  and rows are 114's for that tier; resp none and resp eval (the frozen eval points of the b4 groups).
- Recordings: the frozen analysis list 114 used (114.list_items over the split minus 114.touched_uids), checked equal
  to the recordings of 114's own cruse11 rows; n_recs = its length (114.m_overlays) and the eval subsamples
  (114.subsamples) are taken over it, as 114.run_shard does. Only the recordings whose clean spans hold the b4 window
  (114.plan_overlays) are staged; the others give 114's no_fit, which 115's coverage does not count as attempted.
- --stage OUT: writes OUT/manifest.json (the staged recordings with the sha256 of each signal and Morgoth file, the
  split context) and OUT/files.txt (the files to copy, relative to the project root); identifiers stay there.
  --manifest M --data-root R: runs the staged recordings, every file's sha256 checked before it is loaded.
- Rows -> output/intermediate/_bench_shards_cruse11_b4/ncmd_bdsp__cruse11__native__<resp>/rowsNN.jsonl (never 114's
  shards); 145 reads cruse11's b4 rows from there. The seeded start depends on the window, so the earlier b4 readings
  (on the 6-block window) are replaced, not extended.
python3 148_cruse11_b4.py --stage OUT | --manifest M --data-root R --resp none|eval [--shard i --nshards n]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import glob
import hashlib
import importlib
import json
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
rn = importlib.import_module("114_benchmark")
fr = importlib.import_module("130_fidelity_readings")
en, wr = rn.en, rn.wr

ROOT = en.ROOT
PIPE, NULL, SPLIT = "cruse11", "ncmd_bdsp", "analysis"
RESPS = ("none", "eval")
B4_ROOT = os.path.join(ROOT, "output", "intermediate", "_bench_shards_cruse11_b4")



def b4_natives(mod):
    specs = rn.variant_specs(mod, "bdsp")
    return sorted(v for v, s in specs.items() if not s["short"] and wr._cfg(mod, s["arg"])["n_blocks"] == 4)


@contextlib.contextmanager
def b4_only(mod):
    """114.variant_specs gives cruse11's BDSP specs restricted to the b4 natives inside the block."""
    orig = rn.variant_specs
    keep = b4_natives(mod)

    def restricted(m, source):
        sp = orig(m, source)
        return {v: s for v, s in sp.items() if v in keep} if (m.NAME == PIPE and source == "bdsp") else sp
    rn.variant_specs = restricted
    try:
        yield keep
    finally:
        rn.variant_specs = orig



def ref_uids(root=None):
    """The recordings of 114's cruse11 resp-none rows (the 145 rows)."""
    d = os.path.join(root or rn.SH[SPLIT], f"{NULL}__{PIPE}__native__none")
    out = set()
    for fn in sorted(glob.glob(os.path.join(d, "rows*.jsonl"))):
        with open(fn) as fh:
            out |= {json.loads(l)["uid"] for l in fh if l.strip()}
    return out


def check_frozen(items, uids):
    got = {it["uid"] for it in items}
    if got != set(uids) or len(got) != len(items):
        raise RuntimeError(f"the frozen list ({len(items)} recordings) is not the {len(uids)} recordings of 114's "
                           "cruse11 rows: refused")
    return len(items)


def frozen_items():
    items = rn.list_items(rn.parse_null(NULL), SPLIT, rn.touched_uids())
    check_frozen(items, ref_uids())
    return items


def context(resp, n_recs, locator, uncorrected):
    """114.run_shard's ctx for the null set."""
    if resp not in RESPS:
        raise ValueError(f"resp must be one of {RESPS}")
    return {"null": NULL, "split": SPLIT, "montage": "native", "resp": resp, "max_overlays": None,
            "locator": set(locator), "uncorrected": set(uncorrected), "col113": None, "spec_path": None,
            "n_recs": int(n_recs)}



def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage(out_dir):
    """The manifest of the frozen recordings whose clean spans hold the b4 window (signals read, no pipeline call)."""
    rn.guard_split(SPLIT)
    items = frozen_items()
    loc, unc = rn.subsamples([it["patient"] for it in items], "bdsp")
    mod = rn.load_pipe(PIPE)
    staged = []
    with b4_only(mod) as v4:
        need = rn.need_seconds(mod, v4, "bdsp")
        for it in items:
            rec = en._load(it)
            if rn.plan_overlays(rec, mod, "bdsp", len(items), v4):
                staged.append({"uid": it["uid"], "patient": it["patient"], "meta": it["meta"],
                               "sig": os.path.relpath(it["path"], ROOT), "mor": os.path.relpath(it["morgoth"], ROOT),
                               "sig_sha256": _sha256(it["path"]), "mor_sha256": _sha256(it["morgoth"]),
                               "in_locator": it["patient"] in loc, "in_uncorrected": it["patient"] in unc})
            del rec
    m = {"null": NULL, "pipe": PIPE, "split": SPLIT, "n_recs": len(items), "n_frozen": len(items),
         "n_staged": len(staged), "b4_variants": v4, "need_s": need, "sha256_148": _sha256(os.path.abspath(__file__)),
         "items": staged}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(m, fh, indent=1)
    with open(os.path.join(out_dir, "files.txt"), "w") as fh:
        fh.write("".join(f"{x[k]}\n" for x in staged for k in ("sig", "mor")))
    return m


def staged_items(m, data_root):
    """The manifest's recordings as engine descriptors under data_root; each file's sha256 checked."""
    out = []
    for x in m["items"]:
        sig, mor = os.path.join(data_root, x["sig"]), os.path.join(data_root, x["mor"])
        for p, want in ((sig, x["sig_sha256"]), (mor, x["mor_sha256"])):
            if _sha256(p) != want:
                raise RuntimeError(f"{os.path.basename(p)}: sha256 differs from the manifest: refused")
        out.append({"source": "bdsp", "path": sig, "morgoth": mor, "uid": x["uid"], "patient": x["patient"],
                    "meta": dict(x["meta"])})
    return out


def locator_sets(m):
    return ({x["patient"] for x in m["items"] if x["in_locator"]},
            {x["patient"] for x in m["items"] if x["in_uncorrected"]})



def shard_file(resp, shard, root=None):
    if resp not in RESPS:
        raise ValueError(f"resp must be one of {RESPS}")
    root = os.path.abspath(root or B4_ROOT)
    for d in (rn.SH[SPLIT],) + tuple(fr.BENCH_SHARDS):
        d = os.path.abspath(d)
        if root == d or root.startswith(d + os.sep):
            raise RuntimeError(f"{root}: never 114's shards")
    return os.path.join(root, f"{NULL}__{PIPE}__native__{resp}", f"rows{shard:02d}.jsonl")


def run(manifest, data_root, resp, shard=0, nshards=1, root=None):
    rn.guard_split(SPLIT)
    with open(manifest) as fh:
        m = json.load(fh)
    if (m["null"], m["pipe"], m["split"]) != (NULL, PIPE, SPLIT):
        raise RuntimeError("manifest of another null set, pipeline or split: refused")
    items = [it for i, it in enumerate(staged_items(m, data_root)) if i % nshards == shard]
    loc, unc = locator_sets(m)
    ctx = context(resp, m["n_recs"], loc, unc)
    mod = rn.load_pipe(PIPE)
    fn = shard_file(resp, shard, root)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    done = rn._done(fn, SPLIT)
    with b4_only(mod) as v4, open(fn, "a") as fh:
        if v4 != m["b4_variants"]:
            raise RuntimeError("the b4 variants differ from the manifest's: refused")

        def write(row):
            fh.write(json.dumps(row, allow_nan=False) + "\n")
            fh.flush()
            done.add(row["key"])
        for it in items:
            t = time.perf_counter()
            rec = en._load(it)
            rn.check_member(rec, SPLIT)
            rn.run_recording(rec, mod, rn.parse_null(NULL), ctx, done, write)
            print(f"{NULL} {PIPE} b4 {resp} recording done in {time.perf_counter() - t:.1f} s", flush=True)
            del rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage")
    ap.add_argument("--manifest")
    ap.add_argument("--data-root")
    ap.add_argument("--resp", default="none", choices=RESPS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    if a.stage:
        m = stage(a.stage)
        print(f"staged {m['n_staged']} of {m['n_frozen']} frozen recordings (b4 need {m['need_s']} s)")
    elif a.manifest and a.data_root:
        run(a.manifest, a.data_root, a.resp, a.shard, a.nshards)
    else:
        ap.error("--stage OUT, or --manifest M --data-root R")


if __name__ == "__main__":
    main()
