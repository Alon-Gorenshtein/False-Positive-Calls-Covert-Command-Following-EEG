import importlib

dp = importlib.import_module("154_dubois_pathway")


def _c3(uid, peak, p, t):
    return {"uid": uid, "patient": uid, "j": 0, "status": "ok", "x_peak_da": peak, "p": p, "x_t_peak_s": t,
            "start_s": 10.0}


def _pm(uid, pp, t, s=10.0):
    return {"uid": uid, "patient": uid, "j": 0, "status": "ok", "x_p_perm": pp, "x_t_peak_s": t, "start_s": s}


def test_gate_and_mismatch():
    c3 = [_c3("a", .75, .30, 4.2), _c3("b", .60, .01, 3.0), _c3("c", .60, .40, 2.0), _c3("d", .80, .2, 1.0)]
    pm = [_pm("a", .01, 4.2), _pm("b", .20, 3.0), _pm("c", .01, 2.0), _pm("d", .01, 1.5)]
    out = dp.pathway(c3, pm)
    assert (out["advancement"]["k"], out["advancement"]["n"]) == (3, 4)
    assert (out["significant_run"]["k"], out["significant_run"]["n"]) == (1, 4)
    assert (out["permutation"]["k"], out["permutation"]["n"]) == (3, 4)
    assert (out["advancement_and_permutation"]["k"], out["advancement_and_permutation"]["n"]) == (1, 3)
    assert out["unmatched"] == 1 and out["matched"] == 3


def test_other_window_is_unmatched():
    out = dp.pathway([_c3("a", .75, .30, 4.2)], [_pm("a", .01, 4.2, s=99.0)])
    assert out["matched"] == 0 and out["unmatched"] == 1 and out["advancement_and_permutation"]["n"] == 0


def test_error_rows_leave_the_denominator():
    bad = dict(_c3("e", None, None, None), status="error: ValueError")
    out = dp.pathway([_c3("a", .75, .30, 4.2), bad], [_pm("a", .01, 4.2)])
    assert out["advancement"]["n"] == 1




def test_other_configuration_is_unmatched():
    pm = dict(_pm("a", .01, 4.2), x_peak_da=.60)
    out = dp.pathway([_c3("a", .75, .30, 4.2)], [pm])
    assert out["matched"] == 0 and out["unmatched"] == 1


def test_gate_bounds_count_unmatched_both_ways():
    base = {"status": "ok", "j": 0, "start_s": 0.0, "x_t_peak_s": 1.0, "x_peak_da": 0.6}
    c3 = [dict(base, uid="a", p=0.01), dict(base, uid="b", p=0.01), dict(base, uid="c", p=0.5)]
    pm = [dict(base, uid="a", x_p_perm=0.01), dict(base, uid="b", x_p_perm=0.01, x_t_peak_s=2.0),
          dict(base, uid="c", x_p_perm=0.01)]
    out = dp.pathway(c3, pm)
    assert out["unmatched"] == 1
    assert (out["gate_bounds"]["unmatched_as_negative"]["k"], out["gate_bounds"]["unmatched_as_negative"]["n"]) == (1, 3)
    assert (out["gate_bounds"]["unmatched_as_call"]["k"], out["gate_bounds"]["unmatched_as_call"]["n"]) == (2, 3)


def test_pairs_matched_without_a_peak_accuracy_check_are_counted():
    c3 = [_c3("a", .75, .30, 4.2), _c3("b", .75, .30, 4.2)]
    pm = [_pm("a", .01, 4.2), dict(_pm("b", .01, 4.2), x_peak_da=.75)]
    out = dp.pathway(c3, pm)
    assert out["matched"] == 2 and out["matched_without_peak_check"] == 1
