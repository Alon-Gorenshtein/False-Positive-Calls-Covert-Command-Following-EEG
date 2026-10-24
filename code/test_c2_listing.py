import importlib
import json

import numpy as np
import pandas as pd
import pytest

c2 = importlib.import_module("125_c2_listing")
BIG = 18_432_000


def _meta(tmp_path, listing, design=(), extras=(), train=(), touched=(), files=()):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "spark_listing.txt").write_text("\n".join(listing) + "\n")
    for name, pids in [("sample_500.csv", design), ("bench_extra_500.csv", extras), ("train_600.csv", train)]:
        pd.DataFrame({"patient_id": list(pids), "dir": "EEGs", "file": [f"{p}_0.mat" for p in pids],
                      "bytes": BIG, "n_eligible_segments": 1}).to_csv(meta / name, index=False)
    (meta / "touched_before_freeze.json").write_text(json.dumps({"excluded": list(touched),
                                                                  "loaded_only_not_excluded": []}))
    out = tmp_path / "output"
    for f in files:
        p = out / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return meta, out


def _listing(n=40, size=BIG):
    return [f"EEGs Bonobo{i:05d}_0.mat {size}" for i in range(n)]


def test_listing_excludes_every_touched_patient(tmp_path):
    lst = _listing() + ["negative_EEGs Bonobo00001_1.mat %d" % BIG, "EEGs Bonobo00039_1.mat %d" % BIG]
    meta, out = _meta(tmp_path, lst, design=["Bonobo00000", "Bonobo00001"], extras=["Bonobo00002"],
                      train=["Bonobo00003"], touched=["Bonobo00004_0"],
                      files=["bdsp/refcheck/Bonobo00005_1.parquet", "intermediate/_x/deep/Bonobo00006_0.npz"])
    df, info = c2.listing(meta, out)
    gone = {f"Bonobo{i:05d}" for i in range(7)}
    assert not gone & set(df.patient_id)
    assert info["excluded_patients"] == 7 and info["pool_patients"] == 33
    assert df["rank"].is_monotonic_increasing and df.groupby("patient_id")["rank"].nunique().eq(1).all()
    two = df[df.patient_id == "Bonobo00039"]
    assert len(two) == 2 and sorted(two.seg_rank) == [0, 1]


def test_listing_is_a_deterministic_full_permutation(tmp_path):
    meta, out = _meta(tmp_path, _listing(200))
    a, _ = c2.listing(meta, out)
    b, _ = c2.listing(meta, out)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.patient_id) == {f"Bonobo{i:05d}" for i in range(200)} and a.patient_id.is_unique
    assert list(a.patient_id[:20]) != sorted(a.patient_id[:20])


def test_files_too_small_for_30_min_are_not_candidates(tmp_path):
    lst = _listing(5) + [f"EEGs Bonobo00100_0.mat {BIG - 1}", f"EEGs Bonobo00101_0.mat {BIG - 1}",
                         f"EEGs Bonobo00101_1.mat {BIG}"]
    meta, out = _meta(tmp_path, lst)
    df, info = c2.listing(meta, out)
    assert "Bonobo00100" not in set(df.patient_id) and list(df[df.patient_id == "Bonobo00101"].file) == [
        "Bonobo00101_1.mat"]
    assert info["patients_without_a_30min_file"] == 1


def test_listing_refuses_when_an_exclusion_source_is_missing(tmp_path):
    meta, out = _meta(tmp_path, _listing())
    (meta / "train_600.csv").unlink()
    with pytest.raises(FileNotFoundError, match="train_600"):
        c2.listing(meta, out)


def test_threshold_and_generator_are_the_pinned_ones(tmp_path):
    assert c2.MIN_BYTES_30MIN == BIG and (c2.SEED, c2.STREAM, c2.C2_N, c2.WINDOW_S) == (20261016, 2, 400, 1800)
    meta, out = _meta(tmp_path, _listing(50))
    df, _ = c2.listing(meta, out)
    expect = [f"Bonobo{i:05d}" for i in np.random.default_rng([20261016, 2]).permutation(50)]
    assert list(df.patient_id) == expect
