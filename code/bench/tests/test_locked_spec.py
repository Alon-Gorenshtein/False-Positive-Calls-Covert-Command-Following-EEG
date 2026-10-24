import importlib

from bench import locked_spec as sp






def test_code_files_cover_runner_merge_and_113_chain():
    fs = sp.code_files()
    for f in ("114_benchmark.py", "115_benchmark_merge.py", "113_columbia_null.py", "112_columbia_exact.py",
              "bench/engine.py", "bench/adapters.py", "bench/wrappers.py"):
        assert f in fs


def test_code_files_skip_macos_resource_forks(tmp_path, monkeypatch):
    """AppleDouble ._ files are filesystem metadata, never in git: hashing them makes the freeze irreproducible."""
    (tmp_path / "bench" / "tests").mkdir(parents=True)
    for f in ("bench/engine.py", "bench/._engine.py", "bench/tests/test_x.py", "bench/tests/._test_x.py"):
        (tmp_path / f).write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    fs = sp.code_files()
    assert "bench/engine.py" in fs and "bench/tests/test_x.py" in fs
    assert not [f for f in fs if f.split("/")[-1].startswith("._")], fs
