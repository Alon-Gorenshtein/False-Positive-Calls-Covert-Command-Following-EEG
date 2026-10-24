"""The Cornell two-group test against the public toolbox (Chronux 2.12, two_group_test_spectrum.m; source and sha256
in test_157_cornell_chronux.py). chronux_two_group is a literal transliteration of the .m file; tgt_toolbox is the
same computation on pipe_cornell's array layout. The module's test statistic and its theoretical-variance decision
equal the toolbox's; its jackknife variance divides each leave-one-out statistic by the full-sample variance where the
toolbox uses the leave-one-out variance, which makes the module's variance larger and its calls fewer.
Sensitivity: the fidelity-corrected Cornell primary reading re-run with the toolbox's variance on the development
windows held on this machine (the unpatched run must first reproduce the stored row).
-> output/tables/cornell_chronux_equivalence.json (no identifiers).
python3 157_cornell_chronux.py [--limit N]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import importlib
import json
import sys

import numpy as np
from scipy.special import polygamma, psi
from scipy.stats import norm

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
OUT = os.path.join(ROOT, "output", "tables", "cornell_chronux_equivalence.json")
SOURCE = {"url": "https://raw.githubusercontent.com/jsiegle/chronux/a3b39837d76309908f3ea91bb9feb72b2c749714/"
                 "spectral_analysis/statistical_tests/two_group_test_spectrum.m",
          "sha256": "1608f946b7fc495c8ddce334e7596b8b2d3727dc0810d507e85c7ce8799c2ce6"}
READING = "c3_jackknife_bh_keep40_r4_om2contig"


def chronux_two_group(J1, J2, p=0.05):
    """two_group_test_spectrum.m, line for line. J1, J2: frequencies x samples (complex or real amplitude)."""
    m1, m2 = J1.shape[1], J2.shape[1]
    dof1, dof2 = m1, m2
    S1 = np.conj(J1) * J1
    S2 = np.conj(J2) * J2
    S1, S2 = S1.real, S2.real
    Sm1 = np.mean(S1, 1)
    Sm2 = np.mean(S2, 1)
    bias1 = psi(dof1) - np.log(dof1)
    bias2 = psi(dof2) - np.log(dof2)
    var1 = polygamma(1, dof1)
    var2 = polygamma(1, dof2)
    z1 = np.log(Sm1) - bias1
    z2 = np.log(Sm2) - bias2
    dz = (z1 - z2) / np.sqrt(var1 + var2)
    bias11 = psi(dof1 - 1) - np.log(dof1 - 1)
    var11 = polygamma(1, dof1 - 1)
    ps1 = np.zeros((len(dz), m1))
    for i in range(m1):
        ikeep = np.setdiff1d(np.arange(m1), [i])
        Sm1i = np.mean(S1[:, ikeep], 1)
        z1i = np.log(Sm1i) - bias11
        dz1i = (z1i - z2) / np.sqrt(var11 + var2)
        ps1[:, i] = m1 * dz - (m1 - 1) * dz1i
    ps1m = np.mean(ps1, 1)
    bias21 = psi(dof2 - 1) - np.log(dof2 - 1)
    var21 = polygamma(1, dof2 - 1)
    ps2 = np.zeros((len(dz), m2))
    for j in range(m2):
        jkeep = np.setdiff1d(np.arange(m2), [j])
        Sm2j = np.mean(S2[:, jkeep], 1)
        z2j = np.log(Sm2j) - bias21
        dz2j = (z1 - z2j) / np.sqrt(var1 + var21)
        ps2[:, j] = m2 * dz - (m2 - 1) * dz2j
    ps2m = np.mean(ps2, 1)
    vdz = (np.sum((ps1 - ps1m[:, None]) ** 2, 1) / (m1 * (m1 - 1))
           + np.sum((ps2 - ps2m[:, None]) ** 2, 1) / (m2 * (m2 - 1)))
    x = norm.ppf([p / 2, 1 - p / 2])
    Adz = ((dz >= x[0]) & (dz <= x[1])).astype(int)
    return dz, vdz, Adz


def tgt_toolbox(S1, S2, parts=False):
    """pipe_cornell.tgt's layout (n, K, ch, f) with the toolbox's leave-one-out variances."""
    A, B = S1.reshape(-1, *S1.shape[2:]), S2.reshape(-1, *S2.shape[2:])
    m1, m2 = len(A), len(B)
    s1, s2 = A.sum(0), B.sum(0)
    z1 = np.log(s1 / m1) - (psi(m1) - np.log(m1))
    z2 = np.log(s2 / m2) - (psi(m2) - np.log(m2))
    dz = (z1 - z2) / np.sqrt(polygamma(1, m1) + polygamma(1, m2))
    z1i = np.log((s1 - A) / (m1 - 1)) - (psi(m1 - 1) - np.log(m1 - 1))
    z2j = np.log((s2 - B) / (m2 - 1)) - (psi(m2 - 1) - np.log(m2 - 1))
    d1 = (z1i - z2) / np.sqrt(polygamma(1, m1 - 1) + polygamma(1, m2))
    d2 = (z1 - z2j) / np.sqrt(polygamma(1, m1) + polygamma(1, m2 - 1))
    p1 = (m1 - 1) / m1 * ((d1 - d1.mean(0)) ** 2).sum(0)
    p2 = (m2 - 1) / m2 * ((d2 - d2.mean(0)) ** 2).sum(0)
    return (dz, p1 + p2, (p1, p2)) if parts else (dz, p1 + p2)


@contextlib.contextmanager
def toolbox_variance(pc):
    """pipe_cornell (and 130's Cornell reading, which calls pc.tgt) with the toolbox's jackknife variance."""
    orig = pc.tgt
    pc.tgt = tgt_toolbox
    try:
        yield
    finally:
        pc.tgt = orig


def equivalence(n=30):
    from bench import pipe_cornell as pc
    t = importlib.import_module("test_157_cornell_chronux")
    out = {"max_abs_diff_dz": 0.0, "max_abs_diff_vdz_toolbox": 0.0, "decisions_equal": True, "n_inputs": n,
           "module_over_toolbox_variance": [np.inf, 0.0]}
    for i in range(n):
        S1, S2 = t._inputs(i)
        dz, v_mod = pc.tgt(S1, S2)
        tdz, tv, tA = t._toolbox(S1, S2)
        out["max_abs_diff_dz"] = max(out["max_abs_diff_dz"], float(np.max(np.abs(dz - tdz))))
        out["max_abs_diff_vdz_toolbox"] = max(out["max_abs_diff_vdz_toolbox"],
                                              float(np.max(np.abs(tgt_toolbox(S1, S2)[1] - tv))))
        out["decisions_equal"] &= bool(np.array_equal((np.abs(dz) <= 1.959963984540054).astype(int), tA))
        r = v_mod / tv
        out["module_over_toolbox_variance"] = [min(out["module_over_toolbox_variance"][0], float(r.min())),
                                               max(out["module_over_toolbox_variance"][1], float(r.max()))]
    return out


def sensitivity(limit=None):
    """Development windows on this machine with a stored first-overlay row of the Cornell primary reading."""
    b = importlib.import_module("114_benchmark")
    fid = importlib.import_module("130_fidelity_readings")
    st = importlib.import_module("143_stratum_table")
    from bench import pipe_cornell as pc
    c3 = st.c3_rows()
    c3 = c3[(c3["null"] == "ncmd_bdsp") & (c3["pipe"] == "cornell") & (c3["reading"] == READING)
            & (c3["j"].fillna(-1).astype(int) == 0) & (c3["status"] == "ok")]
    sig = os.path.join(ROOT, "output", "bdsp")
    rows = [r for r in c3.to_dict("records") if os.path.exists(os.path.join(sig, "sig", f"{r['uid']}.npz"))]
    mod, spec = b.load_pipe("cornell"), b.parse_null("ncmd_bdsp")
    n_recs = 974
    res = {"n": 0, "calls_module": 0, "calls_toolbox": 0, "not_reproduced": 0, "gained": 0, "lost": 0}
    for r in rows[:limit]:
        it = dict(source="bdsp", path=os.path.join(sig, "sig", f"{r['uid']}.npz"),
                  morgoth=os.path.join(sig, "morgoth", f"{r['uid']}.parquet"), uid=r["uid"], patient=r["patient"],
                  meta={"dev": False})
        rec = b.en._load(it)
        jobs, _ = fid.phase1_jobs(rec, mod, spec, n_recs, first_only=True)
        jb = next((j for j in jobs if j["reading"] == READING and j["ov"] is not None), None)
        if jb is None or float(jb["start_s"]) != float(r["start_s"]):
            res["not_reproduced"] += 1
            continue
        a, err_a, _ = b._call_run(mod, fid.run, "cornell", rec, jb["ov"], READING, jb["rng"])
        if a is None or bool(a["call"]) != bool(r["call"]):
            res["not_reproduced"] += 1
            continue
        with toolbox_variance(pc):
            t, err_t, _ = b._call_run(mod, fid.run, "cornell", rec, jb["ov"], READING, jb["rng"])
        b._clear_cache()
        res["n"] += 1
        res["calls_module"] += int(bool(a["call"]))
        res["calls_toolbox"] += int(bool(t["call"]))
        res["gained"] += int(bool(t["call"]) and not bool(a["call"]))
        res["lost"] += int(bool(a["call"]) and not bool(t["call"]))
        print(f"{res['n']}: module {res['calls_module']} toolbox {res['calls_toolbox']}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    out = {"source": SOURCE, "equivalence": equivalence(), "sensitivity_development_subset": sensitivity(a.limit),
           "reading": READING}
    s = json.dumps(out, indent=1, default=float)
    if "Bonobo" in s:
        raise RuntimeError("identifier in output")
    with open(OUT, "w") as fh:
        fh.write(s)
    print(s)


if __name__ == "__main__":
    main()
