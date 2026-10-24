import glob
import importlib
import json
import os

import pytest

rp = importlib.import_module("153_replication_rows")
ROOT = rp.ROOT


def _stored(null, pipe, uid, variant, j=0):
    pat = os.path.join(ROOT, "output", "intermediate", "_bench_shards", f"{null}__{pipe}__native__none", "rows*.jsonl")
    for fn in sorted(glob.glob(pat)):
        if os.path.basename(fn).startswith("._"):
            continue
        for line in open(fn):
            r = json.loads(line)
            if r.get("uid") == uid and r.get("variant") == variant and str(r.get("j")) == str(j) \
                    and r.get("resp") == "none":
                return r
    return None






def test_missing_is_not_negative():
    v = rp.FROZEN["gibson16"]["variant"]
    rows = [{"pipe": "gibson16", "uid": "u1", "patient": "p1", "j": 0, "variant": v, "status": "ok", "call": True},
            {"pipe": "gibson16", "uid": "u2", "patient": "p2", "j": 0, "variant": v, "status": "error", "call": None},
            {"pipe": "gibson16", "uid": "u3", "patient": "p3", "j": 0, "variant": v,
             "status": "na: no_usable_trials", "call": None}]
    c = rp.first_overlay_cell(rows, "gibson16")
    assert (c["k"], c["n"]) == (1, 1)


def test_families_and_prediction_intervals():
    cells = {p: {"k": 0, "n": 100} for p in rp.FROZEN}
    cells.update(cornell={"k": 25, "n": 100}, dubois={"k": 22, "n": 100}, mbvt={"k": 14, "n": 100})
    t = rp.tests(cells)
    assert set(t["confirmatory"]) == {"cornell", "dubois", "mbvt"}
    assert set(t["specificity_controls"]) == {"columbia", "edlow17", "mbmi", "gibson16"}
    assert t["confirmatory"]["cornell"]["holm_p"] < 1e-6
    assert t["specificity_controls"]["edlow17"]["holm_p"] < 0.01
    assert t["confirmatory"]["cornell"]["inside_prediction_interval"] is True




def test_items_without_morgoth_keep_the_exclusion_checks(tmp_path):
    import pandas as pd
    (tmp_path / "sig").mkdir()
    pd.DataFrame({"rank": [2, 1], "uid": ["P2_0", "P1_0"], "patient_id": ["P2", "P1"]}).to_csv(tmp_path / "l.csv",
                                                                                              index=False)
    for u in ("P1_0", "P2_0"):
        (tmp_path / "sig" / f"{u}.npz").write_bytes(b"")
    its = rp.items_without_morgoth(str(tmp_path), str(tmp_path / "l.csv"), excluded=set())
    assert [i["uid"] for i in its] == ["P1_0", "P2_0"] and all(i["morgoth"] is None for i in its)
    with pytest.raises(RuntimeError):
        rp.items_without_morgoth(str(tmp_path), str(tmp_path / "l.csv"), excluded={"P2"})


def test_items_can_skip_windows_not_yet_copied(tmp_path):
    import pandas as pd
    (tmp_path / "sig").mkdir()
    pd.DataFrame({"rank": [1, 2], "uid": ["P1_0", "P2_0"], "patient_id": ["P1", "P2"]}).to_csv(tmp_path / "l.csv",
                                                                                              index=False)
    (tmp_path / "sig" / "P2_0.npz").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        rp.items_without_morgoth(str(tmp_path), str(tmp_path / "l.csv"), excluded=set())
    its = rp.items_without_morgoth(str(tmp_path), str(tmp_path / "l.csv"), excluded=set(), allow_missing=True)
    assert [i["uid"] for i in its] == ["P2_0"] and its[0]["meta"]["fresh_rank"] == 2
