import glob
import hashlib
import importlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

fr = importlib.import_module("127_fresh_rows")
b = fr.b
CODE = os.path.dirname(os.path.abspath(__file__))
HAVE_DATA = os.path.isdir(b.en.SIG) and os.path.exists(b.TOUCHED_FILE)
SLOW = os.environ.get("FRESH_IDENTITY") == "1"
TIMING = lambda k: k.endswith("seconds")
RELABEL = ("null", "split")


def _manifest(tmp_path, rows, root=None):
    root = root or tmp_path / "c2_fresh"
    for d in ("sig", "morgoth"):
        (root / d).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["rank", "patient_id", "uid"])
    for s in df.uid:
        (root / "sig" / f"{s}.npz").write_bytes(b"")
        (root / "morgoth" / f"{s}.parquet").write_bytes(b"")
    path = tmp_path / "windows.csv"
    df.to_csv(path, index=False)
    return root, path


def test_spec_runs_the_bdsp_rules_under_the_fresh_null_names():
    assert fr.fresh_spec("cornell") == {"name": "ncmd_bdsp_fresh", "kind": "ncmd", "family": "cmd",
                                        "source": "bdsp"}
    assert fr.fresh_spec("mbvt") == {"name": "nstim0_bdsp_fresh", "kind": "nstim0", "family": "erp",
                                     "source": "bdsp"}
    assert set(fr.FRESH_PIPES) == {"cornell", "cruse11", "cruse12p", "edlow17", "mbvt", "chennu13", "gibson16"}
    for p in ("columbia", "dubois", "mbmi", "sokoliuk21"):
        with pytest.raises(ValueError):
            fr.fresh_spec(p)
    for p in fr.FRESH_PIPES:
        assert b.load_pipe(p).FAMILY == fr.fresh_spec(p)["family"]


def test_rows_go_to_their_own_folder_never_into_the_benchmark_shards(tmp_path):
    fn = fr.shard_file("cornell", 3)
    assert fn == os.path.join(fr.SHARDS, "ncmd_bdsp_fresh__cornell__native__none", "rows03.jsonl")
    assert os.path.basename(fr.SHARDS) == "_fresh_shards"
    for sh in b.SH.values():
        assert not os.path.abspath(fn).startswith(os.path.abspath(sh) + os.sep)
        with pytest.raises(ValueError):
            fr.shard_file("cornell", 0, root=os.path.join(sh, "x"))
        with pytest.raises(ValueError):
            fr.shard_file("cornell", 0, root=sh)


def test_items_follow_the_manifest_in_rank_order(tmp_path):
    root, path = _manifest(tmp_path, [(1, "Bonobo00002", "Bonobo00002_3"), (0, "Bonobo00001", "Bonobo00001_0")])
    its = fr.fresh_items(root, path, excluded=set())
    assert [it["uid"] for it in its] == ["Bonobo00001_0", "Bonobo00002_3"]
    it = its[0]
    assert it["source"] == "bdsp" and it["patient"] == "Bonobo00001"
    assert it["path"] == os.path.join(str(root), "sig", "Bonobo00001_0.npz")
    assert it["morgoth"] == os.path.join(str(root), "morgoth", "Bonobo00001_0.parquet")
    assert it["meta"] == {"dev": False, "fresh": True, "fresh_rank": 0}


def test_items_refuse_missing_files_duplicates_and_touched_patients(tmp_path):
    root, path = _manifest(tmp_path, [(0, "Bonobo00001", "Bonobo00001_0"), (1, "Bonobo00002", "Bonobo00002_0")])
    with pytest.raises(RuntimeError, match="touched"):
        fr.fresh_items(root, path, excluded={"Bonobo00002"})
    os.remove(root / "morgoth" / "Bonobo00002_0.parquet")
    with pytest.raises(FileNotFoundError):
        fr.fresh_items(root, path, excluded=set())
    _, dup = _manifest(tmp_path, [(0, "Bonobo00001", "Bonobo00001_0"), (1, "Bonobo00001", "Bonobo00001_1")], root)
    with pytest.raises(RuntimeError, match="twice"):
        fr.fresh_items(root, dup, excluded=set())
    _, dup = _manifest(tmp_path, [(0, "Bonobo00001", "Bonobo00001_0"), (1, "Bonobo00003", "Bonobo00001_0")], root)
    with pytest.raises(RuntimeError, match="twice"):
        fr.fresh_items(root, dup, excluded=set())


def test_manifest_hash_must_match_the_pinned_one(tmp_path):
    _, path = _manifest(tmp_path, [(0, "Bonobo00001", "Bonobo00001_0")])
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert fr.check_sha256(path, sha) == sha
    with pytest.raises(RuntimeError, match="sha256"):
        fr.check_sha256(path, "0" * 64)


def test_excluded_patients_are_the_benchmark_and_training_listings(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    for name, pids in [("sample_500.csv", ["Bonobo00001"]), ("bench_extra_500.csv", ["Bonobo00002"]),
                       ("train_600.csv", ["Bonobo00003"])]:
        pd.DataFrame({"patient_id": pids}).to_csv(meta / name, index=False)
    (meta / "touched_before_freeze.json").write_text(json.dumps({"excluded": ["Bonobo00004_1"],
                                                                 "loaded_only_not_excluded": ["Bonobo00005_0"]}))
    assert fr.excluded_patients(str(meta)) == {f"Bonobo0000{i}" for i in range(1, 6)}
    os.remove(meta / "train_600.csv")
    with pytest.raises(FileNotFoundError):
        fr.excluded_patients(str(meta))




def _copy_recording(src_item, root, stem):
    for d, k, ext in (("sig", "path", "npz"), ("morgoth", "morgoth", "parquet")):
        os.makedirs(os.path.join(root, d), exist_ok=True)
        shutil.copy(src_item[k], os.path.join(root, d, f"{stem}.{ext}"))


def _rows(fn):
    with open(fn) as fh:
        return [json.loads(l) for l in fh if l.strip()]


@pytest.mark.skipif(not HAVE_DATA, reason="needs the BDSP analysis recordings")
def test_fresh_rows_are_labelled_resumable_and_sharded(tmp_path):
    spec = b.parse_null("ncmd_bdsp")
    src = b.list_items(spec, "analysis", b.touched_uids())[:2]
    root = tmp_path / "c2_fresh"
    for k, it in enumerate(src):
        _copy_recording(it, str(root), f"Bonobo9999{k}_0")
    man = tmp_path / "windows.csv"
    pd.DataFrame({"rank": [0, 1], "patient_id": ["Bonobo99990", "Bonobo99991"],
                  "uid": ["Bonobo99990_0", "Bonobo99991_0"]}).to_csv(man, index=False)
    its = fr.fresh_items(root, man, excluded=set())
    out = tmp_path / "_fresh_shards"
    fn = fr.run("cruse12p", its, n_recs=400, shard=1, nshards=2, root=str(out))
    rows = _rows(fn)
    assert rows and {r["uid"] for r in rows} == {"Bonobo99991_0"}
    assert {r["null"] for r in rows} == {"ncmd_bdsp_fresh"} and {r["split"] for r in rows} == {fr.SPLIT}
    assert {r["source"] for r in rows} == {"bdsp"} and not any(r["dev"] for r in rows)
    assert {r["resp"] for r in rows} == {"none"}
    n = len(rows)
    fr.run("cruse12p", its, n_recs=400, shard=1, nshards=2, root=str(out))
    assert len(_rows(fn)) == n
    other = os.path.join(os.path.dirname(fn), "rows00.jsonl")
    with open(other, "w") as fh:
        fh.write(json.dumps({"key": "x", "split": "analysis"}) + "\n")
    with pytest.raises(RuntimeError, match="split"):
        fr.run("cruse12p", its, n_recs=400, shard=0, nshards=2, root=str(out))
    assert not glob.glob(os.path.join(b.SH["analysis"], "*fresh*"))


@pytest.mark.skipif(not (HAVE_DATA and SLOW), reason="slow: set FRESH_IDENTITY=1 (needs the stored Phase-1 rows)")
@pytest.mark.parametrize("pipe", sorted(fr.FRESH_PIPES))
def test_rows_on_already_run_analysis_recordings_equal_the_stored_114_rows(tmp_path, pipe):
    """The first 3 listed analysis recordings through 127: the same keys, and every field but the null / split labels
    and the timings equal to the stored 114 --resp none rows (only counts are reported, never a value)."""
    null = "ncmd_bdsp" if fr.FRESH_PIPES[pipe] == "ncmd" else "nstim0_bdsp"
    spec = b.parse_null(null)
    listing = b.list_items(spec, "analysis", b.touched_uids())
    its = listing[:3]
    fn = fr.run(pipe, its, n_recs=len(listing), root=str(tmp_path / "_fresh_shards"))
    new = {r["key"]: r for r in _rows(fn)}
    want = {it["uid"] for it in its}
    old = {}
    for f in glob.glob(os.path.join(b.shard_dir(null, pipe, "native", "none", "analysis"), "rows*.jsonl")):
        for r in _rows(f):
            if r["uid"] in want:
                old[r["key"]] = r
    assert len(old) > 0
    assert len(set(new) ^ set(old)) == 0
    assert {r["null"] for r in new.values()} == {fr.fresh_spec(pipe)["name"]}
    diff = {}
    for k, r in new.items():
        for f in set(r) | set(old[k]):
            if f in RELABEL or TIMING(f):
                continue
            a, c = r.get(f), old[k].get(f)
            same = a == c or (isinstance(a, float) and isinstance(c, float) and np.isnan(a) and np.isnan(c))
            if not same:
                diff[f] = diff.get(f, 0) + 1
    assert sorted(diff) == [], f"fields that differ (row counts): {diff}"
