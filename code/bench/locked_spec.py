"""The frozen benchmark's module list (PIPES) and the sha256 over its frozen code files (code_files,
code_hash). Run from code/: /usr/bin/python3 -m bench.locked_spec prints the hash.

The table is generated, never hand-edited, so the fixed in advance specs are exactly what the code runs: each module's
grade, family, native channels, run length, scalar module constants, and every multiverse variant with its config and
catalogue citation. Run from code/: /usr/bin/python3 -m bench.locked_spec
"""
import hashlib
import importlib
import json
import os

PIPES = ["columbia", "cornell", "mbmi", "dubois", "cruse11", "edlow17", "cruse12p",
         "mbvt", "chennu13", "gibson16", "sokoliuk21"]
HASHED = ["113_columbia_null.py", "112_columbia_exact.py", "108_bdsp_null.py", "doc_null.py", "betting.py",
          "114_benchmark.py", "115_benchmark_merge.py", "kojima_io.py"]


def _scalar(v):
    return isinstance(v, (int, float, str, bool)) or (isinstance(v, tuple) and all(_scalar(x) for x in v))


def constants(mod):
    return {k: v for k, v in vars(mod).items() if k.isupper() and k not in ("VARIANTS",) and _scalar(v)
            and not k.startswith("_")}


def _fmt(v):
    return json.dumps(v, default=str, sort_keys=True)


def code_files():
    """Every file whose bytes the frozen-code hash covers: the bench package (modules, engine, adapters, wrappers,
    tests) plus the runner, merge and the 113 / 112 / 108 chain."""
    fs = sorted(os.path.join("bench", f) for f in os.listdir("bench") if f.endswith(".py") and not f.startswith("._"))
    fs += sorted(os.path.join("bench", "tests", f) for f in os.listdir(os.path.join("bench", "tests"))
                 if f.endswith(".py") and not f.startswith("._"))
    return fs + HASHED


def code_hash(files=None):
    """sha256 over (path, file sha256) pairs in sorted order: one number for the frozen code."""
    h = hashlib.sha256()
    for f in sorted(files or code_files()):
        h.update(f.encode() + b"\0" + hashlib.sha256(open(f, "rb").read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def render():
    """Not available here: the module-spec table is built from files that are not distributed with this code."""
    raise RuntimeError("the module-spec table is built from files that are not distributed with this code")


if __name__ == "__main__":
    print(code_hash())
