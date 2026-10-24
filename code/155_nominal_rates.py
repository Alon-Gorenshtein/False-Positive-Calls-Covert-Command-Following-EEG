"""Each analysis's own reference rate, beside the 5% benchmark. NOMINAL holds, per command-following analysis, the
error rate its published call rule states (stated_rate) or, where none is stated, the rate its threshold implies
(implied_rate), with the basis in words. own_rate_table() tests each primary development rate against that rate
(one-sided exact binomial). White noise: the primary readings of the 3 analyses above 5% (Cornell, du Bois, vibrotactile
P300) on 500 further white-noise draws, run exactly as 147.run_unit runs its 60 (same generator, length and seeds
scheme; replicate indices R_NEW, disjoint from 147's 0-59 and 149's 10**6 + i). Synthetic data only.
-> rows output/intermediate/_white500_shards/rows_<pipe>.jsonl; --merge -> output/tables/nominal_rates.json.
python3 155_nominal_rates.py --run PIPE | --merge"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import importlib
import json
import re
import subprocess
import sys

from scipy.stats import binom, fisher_exact

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
wn = importlib.import_module("147_white_noise_eligible")
st = wn.st

T = os.path.join(ROOT, "output", "tables")
SH = os.path.join(ROOT, "output", "intermediate", "_white500_shards")
OUT = os.path.join(T, "nominal_rates.json")
R_NEW = range(2 * 10 ** 6, 2 * 10 ** 6 + 500)
WHITE_PIPES = ("cornell", "dubois", "mbvt")
NULL = {"mbvt": "nstim0_bdsp", "gibson16": "nstim0_bdsp"}


MIN_FREE_PCT = 15


def free_percent(text):
    m = re.search(r"free percentage:\s*(\d+)%", text)
    return int(m.group(1)) if m else None


def memory_ok(text=None):
    text = text if text is not None else subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    f = free_percent(text)
    return f is not None and f >= MIN_FREE_PCT

NOMINAL = {
    "columbia": {"rule": "AUC above 0.5 and permutation P < .05", "stated_rate": 0.05, "implied_rate": None,
                 "basis": "stated significance level"},
    "cornell": {"rule": "spectral difference P <= .05 by jackknife, false discovery rate .05, contiguity",
                "stated_rate": 0.05, "implied_rate": None, "basis": "stated significance level"},
    "cruse11": {"rule": "accuracy above 50% with the 99% binomial interval above 50%", "stated_rate": 0.005,
                "implied_rate": None, "basis": "one-sided level of a two-sided 99% interval"},
    "cruse12p": {"rule": "either of 2 permutation comparisons P < .025", "stated_rate": 0.05, "implied_rate": None,
                 "basis": "stated level .05 split over 2 comparisons"},
    "edlow17": {"rule": "permutation P < .05", "stated_rate": 0.05, "implied_rate": None,
                "basis": "stated significance level"},
    "mbmi": {"rule": "accuracy above the binomial 95% limit", "stated_rate": 0.025, "implied_rate": None,
             "basis": "one-sided level of a two-sided 95% limit; no correction for the search over time steps"},
    "dubois": {"rule": "peak accuracy above 70% or paired t test P < .05", "stated_rate": 0.05, "implied_rate": 0.06,
               "basis": "t test at .05, plus at most .01 for an accuracy threshold the publication places above the "
                        "99% binomial bound for 90 trials"},
    "mbvt": {"rule": "accuracy above 23%", "stated_rate": None, "implied_rate": float(binom.sf(2, 10, 1 / 8)),
             "basis": "no error rate stated; chance of more than 23% (3 or more of 10 splits correct, each 1 in 8) if "
                      "the 10 splits were independent"},
    "gibson16": {"rule": "cluster permutation P < .05 for the target-versus-nontarget P3b", "stated_rate": 0.05,
                 "implied_rate": None, "basis": "stated significance level"},
}


def _load(name):
    with open(os.path.join(T, name)) as fh:
        return json.load(fh)


def primary(pipe):
    return _load("stated_rule.json")["entries"][f"{NULL.get(pipe, 'ncmd_bdsp')}|{pipe}"]


def reference_rate(pipe):
    n = NOMINAL[pipe]
    return n["implied_rate"] if n["implied_rate"] is not None else n["stated_rate"]


def fisher_vs_white(real, white):
    """Two-sided Fisher exact P of the real-EEG rate against the white-noise rate of the same reading."""
    return float(fisher_exact([[real["k"], real["n"] - real["k"]], [white["k"], white["n"] - white["k"]]])[1])


def own_rate_table():
    out = {}
    for pipe in NOMINAL:
        s = primary(pipe)["stated_rule"]
        ref = reference_rate(pipe)
        out[pipe] = {**NOMINAL[pipe], "development": {"k": s["k"], "n": s["n"], "rate": s["rate"]},
                     "reference_rate": ref, "p_vs_own": float(binom.sf(s["k"] - 1, s["n"], ref)),
                     "p_vs_benchmark": float(binom.sf(s["k"] - 1, s["n"], 0.05))}
    return out


def _need(pipe, v):
    return _load("white_noise_eligible.json")["variants"][pipe][v]["need_s"]


def white_rows(pipe, rs, write=None, draw="check"):
    v = primary(pipe)["variant"]
    rows = []
    sink = write or rows.append
    for r in rs:
        wn.run_unit(pipe, r, wn.R_for(pipe, [v]), _need(pipe, v), draw, set(), sink)
    return rows


def stored_white(pipe, rs):
    v = primary(pipe)["variant"]
    rows = {}
    for fn in sorted(glob.glob(os.path.join(ROOT, "output", "intermediate", "_white_shards", "rows*.jsonl"))):
        if os.path.basename(fn).startswith("._"):
            continue
        for l in open(fn):
            r = json.loads(l)
            if r["pipe"] == pipe and r["variant"] == v and r["r"] in rs:
                rows[r["r"]] = r
    return [rows[r] for r in rs]


def run(pipe, limit=None):
    fn = os.path.join(SH, f"rows_{pipe}.jsonl")
    os.makedirs(SH, exist_ok=True)
    done = wn._done(fn)
    v = primary(pipe)["variant"]
    with open(fn, "a") as fh:
        def write(row):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
        for i, r in enumerate(R_NEW):
            if limit is not None and i >= limit:
                break
            if f"{pipe}|{wn.CELL}|{r}|w500|{v}" in done:
                continue
            if i % 25 == 0 and not memory_ok():
                raise SystemExit(f"free memory below {MIN_FREE_PCT}%: stopped (rerun resumes)")
            white_rows(pipe, [r], write, draw="w500")


def merge():
    w60 = _load("white_noise_eligible.json")["variants"]
    out = {"pipes": own_rate_table()}
    for pipe, d in out["pipes"].items():
        v = primary(pipe)["variant"]
        c = w60.get(pipe, {}).get(v)
        d["white60"] = None if c is None else {"k": c["k"], "n": c["n"], "rate": c["rate"]}
        d["white500"] = None
        fn = os.path.join(SH, f"rows_{pipe}.jsonl")
        if pipe in WHITE_PIPES and os.path.exists(fn):
            rows = [json.loads(l) for l in open(fn) if l.strip()]
            ok = [r for r in rows if r["status"] == "ok"]
            if any(r["call"] is None for r in ok):
                raise RuntimeError(f"{pipe}: an ok row without a call")
            d["white500"] = dict(st.exact_cell(sum(bool(r["call"]) for r in ok), len(ok)),
                                 n_not_ok=len(rows) - len(ok))
            d["p_real_vs_white500"] = fisher_vs_white(d["development"], d["white500"])
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({p: {k: d[k] for k in ("reference_rate", "p_vs_own", "white60", "white500")}
                      for p, d in out["pipes"].items()}, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=WHITE_PIPES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    merge() if a.merge else run(a.run, a.limit)


if __name__ == "__main__":
    main()
