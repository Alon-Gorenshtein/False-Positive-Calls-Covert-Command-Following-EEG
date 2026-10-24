import importlib

from scipy.stats import binom

nr = importlib.import_module("155_nominal_rates")


def test_p300_implied_rate():
    assert abs(nr.NOMINAL["mbvt"]["implied_rate"] - binom.sf(2, 10, 1 / 8)) < 1e-12
    assert nr.NOMINAL["mbvt"]["stated_rate"] is None


def test_every_command_analysis_has_a_reference_rate():
    assert set(nr.NOMINAL) == {"columbia", "cornell", "cruse11", "cruse12p", "edlow17", "mbmi", "dubois", "mbvt",
                               "gibson16"}
    for v in nr.NOMINAL.values():
        assert (v["stated_rate"] or v["implied_rate"]) and v["basis"]






def test_new_draws_are_disjoint_from_147_and_149():
    assert min(nr.R_NEW) >= 2 * 10 ** 6 and len(nr.R_NEW) == 500


def test_memory_guard_reads_free_percentage():
    assert nr.free_percent("Pageouts: 1\n\nSystem-wide memory free percentage: 29%\n") == 29
    assert nr.memory_ok("System-wide memory free percentage: 29%")
    assert not nr.memory_ok("System-wide memory free percentage: 9%")


def test_fisher_against_white_noise():
    from scipy.stats import fisher_exact
    assert abs(nr.fisher_vs_white({"k": 117, "n": 850}, {"k": 76, "n": 500})
               - fisher_exact([[117, 733], [76, 424]])[1]) < 1e-12
