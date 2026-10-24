import tags


def test_name_inserts_the_tag_before_the_extension():
    assert tags.name("../t/ai1_kojimaB.parquet", "_v2labels") == "../t/ai1_kojimaB_v2labels.parquet"
    assert tags.name("../i/_ai1_kojimaB_shards", "_v2labels") == "../i/_ai1_kojimaB_shards_v2labels"
    assert tags.name("../t/ai1_kojimaB.parquet") == "../t/ai1_kojimaB.parquet"


def test_untagged_skips_tagged_files_and_resource_forks(tmp_path):
    for n in ("ai1_kojimaA.parquet", "ai1_kojimaB.parquet", "ai1_kojimaB_v2labels.parquet", "._ai1_kojimaA.parquet",
              "pathway_sci_c3czc4_mi.parquet", "d3_netbci_v1.json"):
        (tmp_path / n).write_text("x")
    names = lambda fs: [f.split("/")[-1] for f in fs]
    assert names(tags.untagged(f"{tmp_path}/ai1_*.parquet")) == ["ai1_kojimaA.parquet", "ai1_kojimaB.parquet"]
    assert names(tags.untagged(f"{tmp_path}/pathway_*.parquet")) == ["pathway_sci_c3czc4_mi.parquet"]
    assert tags.untagged(f"{tmp_path}/d3_*.json") == []


def test_pick_swaps_in_the_tagged_file_only_where_it_exists(tmp_path):
    for n in ("ai1_kojimaA.parquet", "ai1_kojimaB.parquet", "ai1_kojimaB_v2labels.parquet"):
        (tmp_path / n).write_text("x")
    a, b = f"{tmp_path}/ai1_kojimaA.parquet", f"{tmp_path}/ai1_kojimaB.parquet"
    assert tags.pick(b, "_v2labels") == f"{tmp_path}/ai1_kojimaB_v2labels.parquet"
    assert tags.pick(a, "_v2labels") == a
    assert tags.pick(b) == b
