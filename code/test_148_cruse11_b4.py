import hashlib
import importlib
import json
import os

import numpy as np
import pytest

b4 = importlib.import_module("148_cruse11_b4")
rn, en, wr = b4.rn, b4.en, b4.wr



def test_b4_natives_are_the_four_block_native_variants_with_their_own_window():
    mod = rn.load_pipe("cruse11")
    specs = rn.variant_specs(mod, "bdsp")
    v4 = b4.b4_natives(mod)
    assert len(v4) == 8 and v4 == sorted(v4)
    assert all(not specs[v]["short"] and wr._cfg(mod, specs[v]["arg"])["n_blocks"] == 4 for v in v4)
    assert set(v4) == {v for v in specs if not specs[v]["short"] and wr._cfg(mod, specs[v]["arg"])["n_blocks"] == 4}
    assert rn.need_seconds(mod, v4, "bdsp") == 2186 < rn.need_seconds(mod, list(specs), "bdsp")


def test_b4_only_restricts_cruse11_bdsp_specs_within_the_block_and_restores_them():
    mod, other = rn.load_pipe("cruse11"), rn.load_pipe("edlow17")
    full, pdoc, oth = rn.variant_specs(mod, "bdsp"), rn.variant_specs(mod, "pdoc"), rn.variant_specs(other, "bdsp")
    with b4.b4_only(mod) as v4:
        assert sorted(rn.variant_specs(mod, "bdsp")) == v4
        assert rn.variant_specs(mod, "pdoc") == pdoc and rn.variant_specs(other, "bdsp") == oth
    assert rn.variant_specs(mod, "bdsp") == full
    with pytest.raises(ZeroDivisionError):
        with b4.b4_only(mod):
            1 / 0
    assert rn.variant_specs(mod, "bdsp") == full



def test_the_frozen_list_must_be_the_recordings_of_114s_rows():
    items = [{"uid": "a"}, {"uid": "b"}]
    assert b4.check_frozen(items, {"a", "b"}) == 2
    for uids in ({"a"}, {"a", "b", "c"}):
        with pytest.raises(RuntimeError, match="frozen"):
            b4.check_frozen(items, uids)



def _bdsp_rec(dur=2400.0, seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(0, dur, 10.0)
    x = (20 * g.standard_normal((19, int(dur * 128)))).astype(np.float32)
    return en.Recording(source="bdsp", uid="syn_0", patient="syn", x=x, fs=128.0, names=list(en.NAMES19),
                        meta={"dev": False, "split_final": True, "morgoth_t": t,
                              "morgoth_lab": np.array(["normal_awake"] * len(t))})


def test_a_synthetic_recording_runs_only_the_b4_tier_at_114s_seeded_start_for_its_window():
    mod = rn.load_pipe("cruse11")
    rec = _bdsp_rec()
    rows = []
    ctx = b4.context("none", n_recs=974, locator=set(), uncorrected=set())
    with b4.b4_only(mod) as v4:
        plan = rn.plan_overlays(rec, mod, "bdsp", 974, v4)
        rn.run_recording(rec, mod, rn.parse_null(b4.NULL), ctx, set(), rows.append)
    pub = [r for r in rows if r.get("variant")]
    assert pub and {r["variant"] for r in pub} == set(v4)
    assert {r["j"] for r in pub} == {0} and {r["start_s"] for r in pub} == {plan[0]["start_s"]}
    assert all(r["rand_variant"] for r in pub) and all(r["split"] == "analysis" for r in pub)
    full = rn.plan_overlays(rec, mod, "bdsp", 974, list(rn.variant_specs(mod, "bdsp")))
    assert full == []


def test_eval_points_of_the_b4_groups_are_run_by_the_restricted_tier():
    if not os.path.exists(rn.EVAL_FILE["analysis"]):
        pytest.skip("frozen eval points not on this disk")
    mod = rn.load_pipe("cruse11")
    with b4.b4_only(mod) as v4:
        groups = rn.timing_groups(mod, rn.variant_specs(mod, "bdsp"), v4, rn.primary_variant(mod, "bdsp"))
        pts = rn.eval_points(b4.NULL, "cruse11", "analysis")
    mine = [(gp, v, g) for gp, v, g, _ in pts if g in groups]
    assert mine and all(v in v4 for _, v, _ in mine)



def _manifest(tmp_path):
    root = tmp_path / "data"
    (root / "sig").mkdir(parents=True)
    (root / "mor").mkdir()
    (root / "sig" / "r1.npz").write_bytes(b"signal")
    (root / "mor" / "r1.parquet").write_bytes(b"morgoth")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return root, {"null": b4.NULL, "pipe": "cruse11", "split": "analysis", "n_recs": 974,
                  "items": [{"uid": "r1", "patient": "p1", "meta": {"dev": False}, "sig": "sig/r1.npz",
                             "mor": "mor/r1.parquet", "sig_sha256": sha(root / "sig" / "r1.npz"),
                             "mor_sha256": sha(root / "mor" / "r1.parquet"), "in_locator": True,
                             "in_uncorrected": False}]}


def test_staged_items_are_rebased_and_every_file_hash_checked(tmp_path):
    root, m = _manifest(tmp_path)
    (it,) = b4.staged_items(m, str(root))
    assert it["path"] == os.path.join(str(root), "sig/r1.npz") and it["morgoth"] == os.path.join(str(root),
                                                                                                "mor/r1.parquet")
    assert (it["uid"], it["patient"], it["source"]) == ("r1", "p1", "bdsp")
    assert b4.locator_sets(m) == ({"p1"}, set())
    (root / "mor" / "r1.parquet").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="sha256"):
        b4.staged_items(m, str(root))


def test_rows_go_to_their_own_folder_never_to_114s_shards():
    fn = b4.shard_file("none", 0)
    assert fn.startswith(os.path.abspath(b4.B4_ROOT) + os.sep)
    assert os.path.basename(os.path.dirname(fn)) == f"{b4.NULL}__cruse11__native__none"
    with pytest.raises(RuntimeError):
        b4.shard_file("none", 0, root=rn.SH["analysis"])
    with pytest.raises(ValueError):
        b4.shard_file("grid", 0)


def test_run_passes_114_the_frozen_context_inside_the_b4_tier_and_resumes(monkeypatch, tmp_path):
    root, m = _manifest(tmp_path)
    m["items"].append(dict(m["items"][0], uid="r2", patient="p2", in_locator=False, in_uncorrected=True))
    m["b4_variants"] = b4.b4_natives(rn.load_pipe("cruse11"))
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps(m))
    seen = []
    monkeypatch.setattr(rn, "guard_split", lambda split, spec_path=None: None)
    monkeypatch.setattr(en, "_load", lambda it: en.Recording(source="bdsp", uid=it["uid"], patient=it["patient"],
                                                             x=np.zeros((19, 128), np.float32), fs=128.0,
                                                             names=list(en.NAMES19), meta=dict(it["meta"])))

    def fake_run(rec, mod, spec, ctx, done, write):
        seen.append((rec.uid, dict(ctx), sorted(rn.variant_specs(mod, "bdsp"))))
        k = f"{rec.uid}|k"
        if k not in done:
            write({"key": k, "split": "analysis", "uid": rec.uid})
    monkeypatch.setattr(rn, "run_recording", fake_run)
    out = tmp_path / "out"
    b4.run(str(mf), str(root), "eval", shard=0, nshards=1, root=str(out))
    assert [u for u, _, _ in seen] == ["r1", "r2"]
    for _, ctx, specs in seen:
        assert ctx["n_recs"] == 974 and ctx["resp"] == "eval" and ctx["null"] == b4.NULL
        assert ctx["locator"] == {"p1"} and ctx["uncorrected"] == {"p2"} and specs == m["b4_variants"]
    fn = b4.shard_file("eval", 0, root=str(out))
    assert len(open(fn).read().splitlines()) == 2
    seen.clear()
    b4.run(str(mf), str(root), "eval", shard=1, nshards=2, root=str(out))
    assert [u for u, _, _ in seen] == ["r2"]
    b4.run(str(mf), str(root), "eval", shard=0, nshards=1, root=str(out))
    assert len(open(fn).read().splitlines()) == 2
