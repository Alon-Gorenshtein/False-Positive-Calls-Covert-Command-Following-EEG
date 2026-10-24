import glob
import importlib
import json
import os

import pytest
from scipy.stats import binomtest, fisher_exact

cp = importlib.import_module("166_columbia_dev_placement")
ROOT = cp.ROOT


def test_compare_pairs_windows_and_uses_exact_mcnemar():
    stored = {"a": True, "b": True, "c": False, "d": False, "e": True}
    planned = {"a": False, "b": True, "c": False, "d": True, "f": True}
    out = cp.compare(stored, planned, {"k": 25, "n": 730})
    assert out["n_paired"] == 4 and out["discordant"] == {"stored_only": 1, "planned_only": 1}
    assert (out["stored"]["k"], out["planned"]["k"]) == (2, 2)
    assert out["mcnemar_exact_p"] == pytest.approx(binomtest(1, 2, 0.5).pvalue)
    assert out["fisher_planned_vs_replication_p"] == pytest.approx(fisher_exact([[2, 2], [25, 705]])[1])
    assert out["placement_explains"] is False


def test_placement_explains_needs_a_lower_planned_rate_and_no_replication_difference():
    stored = {f"u{i}": i < 40 for i in range(400)}
    planned = {f"u{i}": i < 12 for i in range(400)}
    out = cp.compare(stored, planned, {"k": 25, "n": 730})
    assert out["mcnemar_exact_p"] < 0.05 and out["fisher_planned_vs_replication_p"] >= 0.05
    assert out["placement_explains"] is True


def test_planner_calls_keep_ok_first_overlays_only_and_refuse_a_missing_call():
    v = cp.rp.FROZEN["columbia"]["variant"].split(":")[0]
    base = {"pipe": "columbia", "variant": v}
    rows = [dict(base, uid="a", j=0, status="ok", x_call_paper=True),
            dict(base, uid="b", j=0, status="error: ValueError", x_call_paper=None),
            dict(base, uid="c", j=1, status="ok", x_call_paper=True),
            dict(base, uid="d", j=0, status="ok", x_call_paper=False)]
    assert cp.planner_calls(rows) == {"a": True, "d": False}
    with pytest.raises(RuntimeError):
        cp.planner_calls(rows + [dict(base, uid="e", j=0, status="ok", x_call_paper=None)])




def _stored_rows(pipe, j, uids):
    pat = os.path.join(ROOT, "output", "intermediate", "_bench_shards", f"ncmd_bdsp__{pipe}__native__none", "rows*.jsonl")
    for fn in sorted(glob.glob(pat)):
        if os.path.basename(fn).startswith("._"):
            continue
        for line in open(fn):
            r = json.loads(line)
            if r.get("uid") in uids and str(r.get("j")) == str(j) and r.get("status") == "ok" and r.get("resp") == "none":
                yield r




