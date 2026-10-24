import importlib

import pytest
from scipy.stats import binom, fisher_exact

pc = importlib.import_module("159_published_cohorts")


def test_published_counts_are_the_quoted_ones():
    p = pc.PUBLISHED
    assert (p["du Bois UWS"]["k"], p["du Bois UWS"]["n"]) == (8, 14)
    assert (p["du Bois MCS"]["k"], p["du Bois MCS"]["n"]) == (13, 17)
    assert (p["du Bois LIS"]["k"], p["du Bois LIS"]["n"]) == (10, 11)
    assert (p["Claassen 2019"]["k"], p["Claassen 2019"]["n"]) == (16, 104)
    assert all(v["quote"] for v in p.values())


def test_count_null_uses_one_sided_fisher():
    got = pc.vs_counts({"k": 13, "n": 17}, {"k": 30, "n": 95})
    assert got == pytest.approx(fisher_exact([[13, 4], [30, 65]], alternative="greater")[1])


def test_rate_null_uses_one_sided_binomial():
    assert pc.vs_rate({"k": 16, "n": 104}, 0.09) == pytest.approx(binom.sf(15, 104, 0.09))


def test_primary_family_and_holm():
    nulls = {"dubois_two_session_clinical": {"k": 30, "n": 95}, "dubois_two_session_doc": {"k": 18, "n": 38},
             "columbia_k2_upper_coma": 0.09, "columbia_k2_upper_clinical": 0.086}
    out = pc.compare(nulls)
    assert set(out["primary"]) == {"du Bois UWS", "du Bois MCS", "du Bois LIS", "Claassen 2019"}
    hs = [v["holm_p"] for v in out["primary"].values()]
    assert all(h >= v["p"] for h, v in zip(hs, out["primary"].values()))
    assert set(out["secondary"]) == set(out["primary"])


def test_rate_null_reported_over_its_ci_and_for_identical_recordings():
    nulls = {"dubois_two_session_clinical": {"k": 30, "n": 95}, "dubois_two_session_doc": {"k": 18, "n": 38},
             "columbia_k2_upper_coma": 0.093, "columbia_k2_upper_clinical": 0.086,
             "columbia_k2_ci_coma": [0.07, 0.12], "columbia_k1_coma": 0.082}
    c = pc.compare(nulls)["primary"]["Claassen 2019"]
    assert c["p_over_null_ci"] == pytest.approx([pc.vs_rate({"k": 16, "n": 104}, 0.07),
                                                 pc.vs_rate({"k": 16, "n": 104}, 0.12)])
    assert c["p_identical_recordings"] == pytest.approx(pc.vs_rate({"k": 16, "n": 104}, 0.082))
    assert "p_over_null_ci" not in pc.compare(nulls)["primary"]["du Bois UWS"]
