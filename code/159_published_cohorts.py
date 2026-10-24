"""Published positive rates with delivered commands against the rate of the same call rule on EEG recorded without
study commands, at the published number of assessments per patient. A published rate above the no-command rate is
signal beyond chance behavior of the rule; a rate within it cannot be told apart from chance behavior by these data.
It does not say which published patients were true or false positives.
Comparisons fixed here:
- du Bois 2026, advancement after Phase I (either of sessions 1 and 2), by diagnosis: primary null = the two-session
  criterion on clinical EEG (19 channels); secondary = the same on prolonged-DoC EEG (6 channels). One-sided Fisher
  exact test (published rate greater).
- Claassen 2019, EEG brain activation (Benjamini-Hochberg across a patient's recordings; mean 2.3 recordings):
  primary null = the 2-recording upper bound (independent recordings) in post-arrest coma; secondary = the same on
  clinical EEG. One-sided exact binomial test against that rate, which assumes independent recordings; because it is
  estimated, P is also given at the ends of its 95% CI, and against the 1-recording rate (identical recordings).
Holm over the 4 primary comparisons. -> output/tables/published_cohorts.json"""
import importlib
import json
import os
import sys

from scipy.stats import binom, fisher_exact

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
st = importlib.import_module("143_stratum_table")
T = os.path.join(ROOT, "output", "tables")
OUT = os.path.join(T, "published_cohorts.json")
DUBOIS_Q = ("Following Phase I (assessment), 8 of 14 UWS, 13 of 17 MCS, 10 of 11 LIS patients, and both AB participants "
            "progressed to Phase II (training and feedback; N = 33, 73.8% of patients passed assessment).")
CLAASSEN_Q = ("A total of 16 of 104 unresponsive patients (15%) had brain activation detected by EEG at a median of 4 "
              "days after injury.")
PUBLISHED = {
    "du Bois UWS": {"k": 8, "n": 14, "rule": "dubois", "source": "du Bois 2026, Results", "quote": DUBOIS_Q},
    "du Bois MCS": {"k": 13, "n": 17, "rule": "dubois", "source": "du Bois 2026, Results", "quote": DUBOIS_Q},
    "du Bois LIS": {"k": 10, "n": 11, "rule": "dubois", "source": "du Bois 2026, Results", "quote": DUBOIS_Q},
    "Claassen 2019": {"k": 16, "n": 104, "rule": "columbia", "source": "Claassen 2019, Abstract", "quote": CLAASSEN_Q},
}


def vs_counts(pub, null):
    return float(fisher_exact([[pub["k"], pub["n"] - pub["k"]], [null["k"], null["n"] - null["k"]]],
                              alternative="greater")[1])


def vs_rate(pub, rate):
    return float(binom.sf(pub["k"] - 1, pub["n"], rate))


def _test(pub, null):
    return vs_counts(pub, null) if isinstance(null, dict) else vs_rate(pub, null)


def compare(nulls):
    prim = {"dubois": nulls["dubois_two_session_clinical"], "columbia": nulls["columbia_k2_upper_coma"]}
    sec = {"dubois": nulls["dubois_two_session_doc"], "columbia": nulls["columbia_k2_upper_clinical"]}
    out = {"primary": {}, "secondary": {}}
    for name, pub in PUBLISHED.items():
        out["primary"][name] = {"published": st.exact_cell(pub["k"], pub["n"]), "null": prim[pub["rule"]],
                                "p": _test(pub, prim[pub["rule"]])}
        if pub["rule"] == "columbia" and "columbia_k2_ci_coma" in nulls:
            out["primary"][name]["p_over_null_ci"] = [vs_rate(pub, r) for r in nulls["columbia_k2_ci_coma"]]
        if pub["rule"] == "columbia" and "columbia_k1_coma" in nulls:
            out["primary"][name]["p_identical_recordings"] = vs_rate(pub, nulls["columbia_k1_coma"])
        out["secondary"][name] = {"null": sec[pub["rule"]], "p": _test(pub, sec[pub["rule"]])}
    names = list(out["primary"])
    for n, h in zip(names, st.holm([out["primary"][n]["p"] for n in names])):
        out["primary"][n]["holm_p"] = float(h)
    return out


def nulls_from_tables():
    pp = json.load(open(os.path.join(T, "stated_rule.json")))["per_patient_at_published_k"]
    rb = json.load(open(os.path.join(T, "repeat_bounds.json")))["nulls"]
    cell = lambda c: {"k": c["k"], "n": c["n"]}
    return {"dubois_two_session_clinical": cell(pp["ncmd_bdsp|dubois"]["composite"]),
            "dubois_two_session_doc": cell(pp["ncmd_pdoc|dubois"]["composite"]),
            "columbia_k2_upper_coma": rb["ncmd_icare"]["columbia"]["k"]["2"]["upper_bound"],
            "columbia_k2_ci_coma": rb["ncmd_icare"]["columbia"]["k"]["2"]["ci95"],
            "columbia_k1_coma": rb["ncmd_icare"]["columbia"]["k"]["1"]["upper_bound"],
            "columbia_k2_upper_clinical": rb["ncmd_bdsp"]["columbia"]["k"]["2"]["upper_bound"]}


def main(out=OUT):
    res = {"what": __doc__.split("\n")[0], "published": PUBLISHED, **compare(nulls_from_tables())}
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({n: (round(v["p"], 4), round(v["holm_p"], 4), round(res["secondary"][n]["p"], 4))
                      for n, v in res["primary"].items()}, indent=1))


if __name__ == "__main__":
    main()
