from bench import motor_common as mc


def test_memo_never_shares_results_between_call_sites_with_equal_keys():
    mc._CACHE.clear()
    a = mc.memo(("uid", 1.0), lambda: "features of module A")
    b = mc.memo(("uid", 1.0), lambda: "features of module B")
    assert a == "features of module A" and b == "features of module B"


def test_memo_still_caches_repeat_calls_from_one_call_site():
    mc._CACHE.clear()
    calls = []

    def get():
        return mc.memo(("uid", 2.0), lambda: calls.append(1) or len(calls))

    assert get() == 1 and get() == 1 and calls == [1]
