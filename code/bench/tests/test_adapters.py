"""bench/adapters.py on synthetic stimulus sequences: mbvt groups of 8
(VT2 1 rare + 7 standards, VT3 one of each rare + 6 standards; consecutive, one train, maximal, at most 60),
chennu13 blocks (module block length, explicit = rare) and its C5 / C6 -> T7 / T8 substitution, gibson16 wrists /
back and the SSEP periodicity skip, sokoliuk21 N/A, edge trimming, the published / randomized designs (the randomized
design's assignment drawn as the module's schedule_randomized draws it; a published per-instruction assignment is
refused on the randomized path), and each adapted overlay accepted by ad.run() / ad.run_randomized() (integration on
synthetic noise only: no statistic or call is checked). check_groups is run against a deliberately broken grouping.
cd code && /usr/bin/python3 -m pytest bench/tests/test_adapters.py -q"""
import mne
import numpy as np
import pytest

from bench import adapters as ad
from bench import engine as en
from bench import pipe_chennu13 as pc
from bench import pipe_gibson16 as pg
from bench import pipe_mbvt as pm
from bench.tests import _synth as sy

ERPCORE30 = ["Fp1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz", "Fp2",
             "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2"]


def stream(n, rng, rare_p=0.15, excl_p=0.0, two=False, soa=0.6, jitter=0.0, start=12.0, pauses=(), pattern=None):
    """Synthetic stream: kinds 0 standard, 1 / 2 rare (types), -1 excluded; never two non-standards in a row.
    -> (kinds, onsets, seq, train) of the whole stream."""
    if pattern is not None:
        kinds = np.tile(np.asarray(pattern), n // len(pattern) + 1)[:n]
    else:
        kinds, prev = [], 0
        for _ in range(n):
            u = rng.random()
            if prev != 0 or u >= rare_p + excl_p:
                k = 0
            else:
                k = (int(rng.integers(1, 3)) if two else 1) if u < rare_p else -1
            kinds.append(k)
            prev = k
        kinds = np.array(kinds)
    gaps = soa + (jitter * rng.standard_normal(n) if jitter else np.zeros(n))
    train = np.zeros(n, int)
    for p in pauses:
        gaps[p] += 20.0
        train[p:] += 1
    on = start + np.r_[0.0, np.cumsum(gaps[1:])]
    return kinds, on, np.arange(n), train


def overlay(kinds, on, seq, train, contrast="oddball", dataset="erpcore_mmn"):
    """The passive_overlay layout (labels 1 rare / 0 standard; rare_type, seq, train) of the non-excluded stimuli."""
    k = kinds >= 0
    lab = (kinds[k] > 0).astype(int)
    n = int(k.sum())
    return en.Overlay(onsets_s=on[k], labels=lab, kind="stim",
                      meta={"dataset": dataset, "contrast": contrast, "rare_type": kinds[k].astype(int),
                            "seq": seq[k], "train": train[k], "trial": np.arange(n), "block": train[k].copy(),
                            "run": np.zeros(n, int), "code": kinds[k], "soa_s": 0.6, "excluded": {}})


def rec_for(ov, fs=128.0, names=sy.NAMES19, seed=0, tail=12.0):
    r = sy.rec(float(ov.onsets_s.max()) + tail, fs=fs, seed=seed, names=names)
    return en.Recording(source="synthetic", uid=f"syn{seed}", patient=f"p{seed}", x=r.x, fs=fs, names=list(names),
                        meta={"dataset": ov.meta["dataset"]})


def src_index(ad_ov, src):
    j = np.searchsorted(src.onsets_s, ad_ov.onsets_s)
    assert np.allclose(src.onsets_s[j], ad_ov.onsets_s)
    return j


def max_groups(src, L, valid):
    """Independent optimum: DP over start positions of the largest number of non-overlapping valid windows."""
    seq, tr = src.meta["seq"], src.meta["train"]
    n = len(seq)
    ok = [i + L <= n and np.all(np.diff(seq[i:i + L]) == 1) and len(set(tr[i:i + L])) == 1 and valid(i, i + L)
          for i in range(n)]
    f = np.zeros(n + L + 1, int)
    for i in range(n - 1, -1, -1):
        f[i] = max(f[i + 1], 1 + f[i + L] if ok[i] else 0)
    return int(f[0])


def check_groups(a, src, paradigm):
    """Every mbvt group: 8 stimuli, consecutive in the stream, one train, one rare (VT2) or one of each rare type
    (VT3), labels and sites as mapped; no stimulus in two groups."""
    g, pos = a.meta["group"], a.meta["pos"]
    G = int(g.max()) + 1
    j = src_index(a, src)
    assert len(np.unique(j)) == len(j)
    assert np.array_equal(np.unique(g), np.arange(G)) and G <= pm.N_GROUPS
    tw = a.meta["target_wrist"]
    assert len(tw) == G
    for gi in range(G):
        m = g == gi
        jj = j[m]
        assert m.sum() == 8 and np.array_equal(pos[m], np.arange(8)) and (np.diff(a.onsets_s[m]) > 0).all()
        assert np.all(np.diff(src.meta["seq"][jj]) == 1), "a group spans an excluded stimulus"
        assert len(set(src.meta["train"][jj])) == 1, "a group spans a pause"
        rt, lab, site = src.meta["rare_type"][jj], a.labels[m], a.meta["site"][m]
        if paradigm == "vt2":
            assert (src.labels[jj] == 1).sum() == 1
            assert np.array_equal(lab == pm.TARGET, src.labels[jj] == 1)
            assert (lab[src.labels[jj] == 0] == pm.NONTARGET).all()
            assert np.array_equal(site == pm.RIGHT, src.labels[jj] == 1)
        else:
            assert sorted(rt) == [0] * 6 + [1, 2]
            assert np.array_equal(site, np.where(rt == 1, pm.LEFT, np.where(rt == 2, pm.RIGHT, pm.OTHER)))
            assert np.array_equal(lab, np.where(site == pm.OTHER, pm.DISTRACTOR,
                                                np.where(site == tw[gi], pm.TARGET, pm.NONTARGET)))



def test_mbvt_vt2_groups():
    rng = np.random.default_rng(1)
    src = overlay(*stream(500, rng, rare_p=0.18, excl_p=0.06, pauses=(250,)))
    a = ad.adapt("mbvt", rec_for(src), src)
    check_groups(a, src, "vt2")
    G = int(a.meta["group"].max()) + 1
    lab = src.labels
    assert G == min(pm.N_GROUPS, max_groups(src, 8, lambda i, k: lab[i:k].sum() == 1))
    info = a.meta["adapter"]
    assert info["usable"]["n_groups"] == G and info["usable"]["n_rare"] == G and info["usable"]["n_standard"] == 7 * G
    assert info["dropped"]["not_in_group"] + info["dropped"]["beyond_run"] + 8 * G == len(src.onsets_s)
    assert set(info["skip_variants"]) == {v for v in pm.VARIANTS if v.startswith("vt3")}
    assert set(ad.usable_variants("mbvt", a)) == {v for v in pm.VARIANTS if v.startswith("vt2")}
    assert info["design"] == "published" and info["run"] is True and info["randomized"] is False
    assert a.meta["paradigm"] == "vt2" and (a.meta["target_wrist"] == pm.RIGHT).all() and info["changed"]
    b = ad.adapt("mbvt", rec_for(src), src, design="randomized")
    check_groups(b, src, "vt2")
    ib = b.meta["adapter"]
    assert np.array_equal(b.onsets_s, a.onsets_s) and np.array_equal(b.labels, a.labels)
    assert ib["design"] == "randomized" and ib["run"] is False and ib["randomized"] is True
    assert ib["rand_variant"] == "vt2_mur20_seg8_full_gt23"


def test_mbvt_cap_at_one_run():
    src = overlay(*stream(800, None, pattern=[0, 0, 1, 0, 0, 0, 0, 0]))
    a = ad.adapt("mbvt", rec_for(src), src)
    assert a.meta["group"].max() + 1 == pm.N_GROUPS
    assert a.meta["adapter"]["dropped"]["beyond_run"] == 8 * (100 - pm.N_GROUPS)


def test_mbvt_vt3_groups_and_runs():
    rng = np.random.default_rng(2)
    src = overlay(*stream(600, rng, rare_p=0.2, two=True, pauses=(150, 300, 450)), contrast="equiprobable",
                  dataset="kojimaB")
    a = ad.adapt("mbvt", rec_for(src), src, rng=np.random.default_rng(5))
    check_groups(a, src, "vt3")
    rt = src.meta["rare_type"]
    G = int(a.meta["group"].max()) + 1
    one_each = lambda i, k: (rt[i:k] == 1).sum() == 1 and (rt[i:k] == 2).sum() == 1
    assert G == min(pm.N_GROUPS, max_groups(src, 8, one_each))
    tw = a.meta["target_wrist"]
    per = pm.N_GROUPS // pm.N_INSTR
    assert all(len(set(tw[b * per:(b + 1) * per])) == 1 for b in range(pm.N_INSTR) if b * per < G)
    info = a.meta["adapter"]
    assert set(info["skip_variants"]) == {v for v in pm.VARIANTS if v.startswith("vt2")}
    assert info["randomized"] is False and "published" in info["randomized_reason"]
    full = overlay(*stream(560, None, pattern=[0, 1, 0, 0, 2, 0, 0, 0]), contrast="equiprobable", dataset="kojimaB")
    b = ad.adapt("mbvt", rec_for(full), full, rng=np.random.default_rng(5))
    check_groups(b, full, "vt3")
    assert b.meta["group"].max() + 1 == pm.N_GROUPS and b.meta["adapter"]["randomized"] is False
    t = b.meta["target_wrist"][::per]
    assert sorted(t) == [pm.LEFT, pm.LEFT, pm.RIGHT, pm.RIGHT]


def test_broken_grouping_is_caught(monkeypatch):
    rng = np.random.default_rng(3)
    src = overlay(*stream(500, rng, rare_p=0.15, excl_p=0.15, pauses=(200,)))
    monkeypatch.setattr(ad, "_links", lambda seq, train: np.ones(max(len(seq) - 1, 0), bool))
    a = ad.adapt("mbvt", rec_for(src), src)
    with pytest.raises(AssertionError):
        check_groups(a, src, "vt2")


def test_mbvt_vt3_randomized_design():
    src = overlay(*stream(600, np.random.default_rng(2), rare_p=0.2, two=True, pauses=(150, 300, 450)),
                  contrast="equiprobable", dataset="kojimaB")
    r = rec_for(src)
    pub = ad.adapt("mbvt", r, src, rng=np.random.default_rng(5))
    a = ad.adapt("mbvt", r, src, rng=np.random.default_rng(5), design="randomized")
    check_groups(a, src, "vt3")
    G = int(a.meta["group"].max()) + 1
    assert 4 <= G < pm.N_GROUPS
    assert np.array_equal(a.onsets_s, pub.onsets_s) and np.array_equal(a.meta["site"], pub.meta["site"])
    expect = np.random.default_rng(5).permutation(np.repeat([pm.LEFT, pm.RIGHT], [G // 2, G - G // 2]))
    assert np.array_equal(a.meta["target_wrist"], expect)
    assert len(set(expect[:pm.N_GROUPS // pm.N_INSTR])) == 2
    info = a.meta["adapter"]
    assert info["design"] == "randomized" and info["randomized"] is True and info["run"] is False
    assert info["rand_variant"] == pm.DEFAULT and info["randomized_reason"] is None
    out = ad.run_randomized("mbvt", r, a, np.random.default_rng(1),
                            variant=dict(pm.VARIANTS[pm.DEFAULT], reps=2, n_perm_rand=10))
    assert set(out) == {"stat", "p", "call", "extra"} and not out["extra"].get("too_few_groups")


def test_per_instruction_refused_on_randomized_path():
    full = overlay(*stream(560, None, pattern=[0, 1, 0, 0, 2, 0, 0, 0]), contrast="equiprobable", dataset="kojimaB")
    r = rec_for(full)
    small = dict(pm.VARIANTS[pm.DEFAULT], reps=2, n_perm_rand=10)
    pub = ad.adapt("mbvt", r, full, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        ad.run_randomized("mbvt", r, pub, np.random.default_rng(1), variant=small)
    rnd = ad.adapt("mbvt", r, full, rng=np.random.default_rng(0), design="randomized")
    forged = en.Overlay(onsets_s=rnd.onsets_s, labels=pub.labels, kind="stim",
                        meta=dict(rnd.meta, target_wrist=pub.meta["target_wrist"]))
    with pytest.raises(ValueError):
        ad.run_randomized("mbvt", r, forged, np.random.default_rng(1), variant=small)
    with pytest.raises(ValueError):
        ad.run("mbvt", r, rnd, small, np.random.default_rng(1))
    with pytest.raises(ValueError):
        ad.run("mbvt", r, pub, "vt2_mur20_seg8_full_gt23", np.random.default_rng(1))


def test_mbvt_wrappers_run():
    src = overlay(*stream(560, None, pattern=[0, 1, 0, 0, 2, 0, 0, 0]), contrast="equiprobable", dataset="kojimaB")
    r = rec_for(src)
    small = dict(pm.VARIANTS[pm.DEFAULT], reps=2, n_perm_rand=10)
    out = ad.run("mbvt", r, ad.adapt("mbvt", r, src, rng=np.random.default_rng(0)), small, np.random.default_rng(1))
    assert set(out) == {"stat", "p", "call", "extra"} and not out["extra"].get("too_few_groups")
    out = ad.run_randomized("mbvt", r, ad.adapt("mbvt", r, src, design="randomized"), np.random.default_rng(1),
                            variant=small)
    assert set(out) == {"stat", "p", "call", "extra"}
    o2 = overlay(*stream(500, None, pattern=[0, 0, 1, 0, 0, 0, 0, 0]))
    r2 = rec_for(o2)
    v2 = dict(pm.VARIANTS["vt2_ann18_ds7_full_gt23"], reps=2, n_perm_rand=10)
    out = ad.run("mbvt", r2, ad.adapt("mbvt", r2, o2), v2, np.random.default_rng(1))
    assert set(out) == {"stat", "p", "call", "extra"} and not out["extra"].get("too_few_groups")
    out = ad.run_randomized("mbvt", r2, ad.adapt("mbvt", r2, o2, design="randomized"), np.random.default_rng(1),
                            variant=v2)
    assert set(out) == {"stat", "p", "call", "extra"}



def test_chennu_blocks_oddball():
    rng = np.random.default_rng(4)
    src = overlay(*stream(800, rng, rare_p=0.18, excl_p=0.05, pauses=(400,)))
    a = ad.adapt("chennu13", rec_for(src), src, rng=np.random.default_rng(6))
    L = ad.chennu_block_len()
    assert L == 97
    n = len(src.onsets_s)
    nb = n // L
    blk, word, expl = a.meta["block"], a.meta["word"], a.meta["explicit_word"]
    assert len(expl) == nb and np.array_equal(blk, np.repeat(np.arange(nb), L))
    j = src_index(a, src)
    assert np.array_equal(j, np.arange(nb * L))
    rare = src.labels[j] == 1
    assert np.array_equal(a.labels == pc.EXPL, rare) and np.array_equal(a.labels == pc.IMPL, ~rare)
    assert np.array_equal(word[rare], expl[blk[rare]])
    assert np.array_equal(word[~rare], (pc.YES + pc.NO) - expl[blk[~rare]])
    assert sorted(expl) == sorted([pc.YES] * (nb // 2) + [pc.NO] * (nb - nb // 2))
    info = a.meta["adapter"]
    assert info["dropped"]["partial_block"] == n - nb * L and info["usable"]["n_blocks"] == nb
    assert info["usable"]["n_explicit"] == int(rare.sum()) and info["usable"]["n_distractor"] == 0
    assert info["skip_variants"] == [] and info["randomized"] is False and info["run"] is True
    b = ad.adapt("chennu13", rec_for(src), src, rng=np.random.default_rng(6), design="randomized")
    ib = b.meta["adapter"]
    assert ib["randomized"] is True and ib["run"] is False and ib["rand_variant"] == pc.DEFAULT
    assert np.array_equal(b.meta["explicit_word"], pc._explicit_order(nb, np.random.default_rng(6)))


def test_chennu_blocks_equiprobable_and_run():
    rng = np.random.default_rng(5)
    src = overlay(*stream(700, rng, rare_p=0.2, two=True, pauses=(150, 300, 450)), contrast="equiprobable",
                  dataset="kojimaB")
    r = rec_for(src)
    a = ad.adapt("chennu13", r, src, rng=np.random.default_rng(7))
    j = src_index(a, src)
    rt = src.meta["rare_type"][j]
    word, expl, blk = a.meta["word"], a.meta["explicit_word"], a.meta["block"]
    assert np.array_equal(word, np.where(rt == 1, pc.YES, np.where(rt == 2, pc.NO, 0)))
    assert np.array_equal(a.labels == pc.DIST, rt == 0)
    assert np.array_equal(a.labels == pc.EXPL, (rt > 0) & (word == expl[blk]))
    out = ad.run("chennu13", r, a, pc.DEFAULT, np.random.default_rng(1))
    assert set(out) == {"stat", "p", "call", "extra"} and not out["extra"].get("too_few_trials")
    b = ad.adapt("chennu13", r, src, rng=np.random.default_rng(7), design="randomized")
    out = ad.run_randomized("chennu13", r, b, np.random.default_rng(1),
                            variant=dict(pc.VARIANTS[pc.DEFAULT], n_perm_rand=10))
    assert set(out) == {"stat", "p", "call", "extra"}


def test_chennu_c5_c6_substitution():
    src = overlay(*stream(500, np.random.default_rng(11), rare_p=0.18))
    r = rec_for(src, names=ERPCORE30)
    a = ad.adapt("chennu13", r, src, rng=np.random.default_rng(0))
    info = a.meta["adapter"]
    assert info["channel_substitution"] == {"T7": "C5", "T8": "C6"}
    assert any("C5" in c and "T7" in c and "C6" in c and "T8" in c for c in info["changed"])
    v = ad.recording_for("chennu13", r)
    assert v.names == [{"C5": "T7", "C6": "T8"}.get(c, c) for c in ERPCORE30] and r.names == ERPCORE30
    assert v.x is r.x and len(v.names) == len(ERPCORE30)
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    for t, s in (("T7", "C5"), ("T8", "C6")):
        d = {c: np.linalg.norm(pos[t] - pos[c]) for c in ERPCORE30}
        assert min(d, key=d.get) == s and abs(d[s] - 0.0388) < 0.001
    out = ad.run("chennu13", r, a, pc.DEFAULT, np.random.default_rng(1))
    assert set(out) == {"stat", "p", "call", "extra"} and out["extra"]["channels"] == "native"
    plain = rec_for(src)
    assert ad.recording_for("chennu13", plain).names == list(sy.NAMES19)
    assert "channel_substitution" not in ad.adapt("chennu13", plain, src).meta["adapter"]
    assert ad.recording_for("gibson16", r).names == ERPCORE30
    no_c5 = [c for c in ERPCORE30 if c != "C5"]
    with pytest.raises(ad.NotApplicable):
        ad.adapt("chennu13", rec_for(src, names=no_c5), src)


def test_recording_view_cached_per_pipe(monkeypatch):
    import gc
    import weakref
    from bench import motor_common as mc
    src = overlay(*stream(500, np.random.default_rng(13), rare_p=0.18))
    r = rec_for(src)
    v = ad.recording_for("chennu13", r)
    assert ad.recording_for("chennu13", r) is v and mc.rec_key(ad.recording_for("chennu13", r)) == mc.rec_key(v)
    assert ad.recording_for("gibson16", r) is not v
    n = {"features": 0}
    orig = pc.features

    def counting(rec, ov):
        n["features"] += 1
        return orig(rec, ov)
    monkeypatch.setattr(pc, "features", counting)
    a = ad.adapt("chennu13", r, src, rng=np.random.default_rng(0))
    for var in ("p3b_base_samples10_none_b20", "p3_any_samples10_none_b20"):
        ad.run("chennu13", r, a, var, np.random.default_rng(1))
    assert n["features"] == 1
    ins = ad.instruction_overlay("chennu13", r, src, np.random.default_rng(0))
    for k in range(2):
        ad.run_instruction("chennu13", r, ins, np.random.default_rng(k), n_perm=20)
    assert n["features"] == 2
    ref = weakref.ref(r)
    del r, v, a, ins
    gc.collect()
    assert ref() is None


def test_recording_view_per_pipe():
    src = overlay(*stream(300, np.random.default_rng(12)))
    r = rec_for(src)
    uids = {ad.recording_for(p, r).uid for p in ("mbvt", "chennu13", "gibson16")}
    assert len(uids) == 3 and r.uid not in uids
    assert all(ad.recording_for(p, r).x is r.x for p in ("mbvt", "chennu13", "gibson16"))



def test_gibson_oddball_and_ssep_skip():
    rng = np.random.default_rng(6)
    jit = overlay(*stream(400, rng, rare_p=0.2, jitter=0.03))
    r = rec_for(jit)
    a = ad.adapt("gibson16", r, jit)
    j = src_index(a, jit)
    rare = jit.labels[j] == 1
    assert np.array_equal(a.labels == pg.TARGET, rare) and np.array_equal(a.labels == pg.BACK, ~rare)
    assert not (a.labels == pg.NONTARGET).any() and np.array_equal(a.meta["site"] == pg.RIGHT, rare)
    info = a.meta["adapter"]
    skip = set(info["skip_variants"])
    assert {v for v in pg.VARIANTS if v.startswith("p3b") or v.startswith("ssep")} == skip
    assert info["periodic"] is False and info["randomized"] is False
    per = overlay(*stream(400, np.random.default_rng(7), rare_p=0.2))
    rp = rec_for(per)
    b = ad.adapt("gibson16", rp, per)
    assert b.meta["adapter"]["periodic"] is True
    assert set(b.meta["adapter"]["skip_variants"]) == {v for v in pg.VARIANTS if v.startswith("p3b")}
    for v in ("p3a_two025", "ssep_avg_nonharm"):
        out = ad.run("gibson16", rp, b, v, np.random.default_rng(1))
        assert set(out) == {"stat", "p", "call", "extra"} and not out["extra"].get("too_few_trials")
    c = ad.adapt("gibson16", rp, per, design="randomized")
    assert c.meta["adapter"]["randomized"] is False
    with pytest.raises(ValueError):
        ad.run_randomized("gibson16", rp, c, np.random.default_rng(1))


def test_gibson_equiprobable():
    rng = np.random.default_rng(8)
    src = overlay(*stream(600, rng, rare_p=0.2, two=True, pauses=(150, 300, 450)), contrast="equiprobable",
                  dataset="kojimaB")
    r = rec_for(src)
    a = ad.adapt("gibson16", r, src)
    j = src_index(a, src)
    rt = src.meta["rare_type"][j]
    blk, tw = a.meta["block"], a.meta["target_wrist"]
    assert np.array_equal(blk, src.meta["train"][j]) and list(tw) == [pg.RIGHT, pg.LEFT, pg.RIGHT, pg.LEFT]
    site = a.meta["site"]
    assert np.array_equal(site, np.where(rt == 1, pg.LEFT, np.where(rt == 2, pg.RIGHT, pg.UPPER_BACK)))
    assert np.array_equal(a.labels, np.where(rt == 0, pg.BACK, np.where(site == tw[blk], pg.TARGET, pg.NONTARGET)))
    info = a.meta["adapter"]
    assert info["randomized"] is False and not any(v.startswith("p3b") for v in info["skip_variants"])
    out = ad.run("gibson16", r, a, "p3b_onepos", np.random.default_rng(1))
    assert set(out) == {"stat", "p", "call", "extra"} and not out["extra"].get("too_few_trials")
    b = ad.adapt("gibson16", r, src, rng=np.random.default_rng(3), design="randomized")
    expect = np.random.default_rng(3).permutation(np.repeat([pg.RIGHT, pg.LEFT], [2, 2]))
    assert np.array_equal(b.meta["target_wrist"], expect) and b.meta["adapter"]["randomized"] is True
    assert np.array_equal(b.labels, np.where(rt == 0, pg.BACK, np.where(site == expect[blk], pg.TARGET, pg.NONTARGET)))
    out = ad.run_randomized("gibson16", r, b, np.random.default_rng(1),
                            variant=dict(pg.VARIANTS[pg.DEFAULT], n_perm_rand=10))
    assert set(out) == {"stat", "p", "call", "extra"}



def test_not_applicable():
    src = overlay(*stream(300, np.random.default_rng(9)))
    r = rec_for(src)
    with pytest.raises(ad.NotApplicable):
        ad.adapt("sokoliuk21", r, src)
    bare = rec_for(src, names=[c for c in ERPCORE30 if c not in ("C5", "C6")])
    with pytest.raises(ad.NotApplicable):
        ad.adapt("chennu13", bare, src)
    assert ad.adapt("gibson16", bare, src).meta["adapter"]["usable"]["n_target"] > 0
    with pytest.raises(ValueError):
        ad.adapt("mbvt", r, src, design="adaptive")
    ko = overlay(*stream(300, np.random.default_rng(9), two=True), contrast="oddball", dataset="kojimaB")
    with pytest.raises(ad.NotApplicable):
        ad.adapt("chennu13", rec_for(ko), ko)
    few = overlay(*stream(40, None, pattern=[0, 0, 1, 1]))
    with pytest.raises(ad.NotApplicable):
        ad.adapt("mbvt", rec_for(few), few)


def test_edge_trimming():
    src = overlay(*stream(900, np.random.default_rng(10), rare_p=0.2, start=1.0))
    r = rec_for(src, tail=2.0)
    dur = r.x.shape[1] / r.fs
    for p, m in (("chennu13", pc), ("gibson16", pg), ("mbvt", pm)):
        a = ad.adapt(p, r, src, rng=np.random.default_rng(0))
        lo, hi = m.extent(a)
        assert lo >= 0 and hi <= dur
        lo1, hi1 = m.extent(en.Overlay(onsets_s=np.array([0.0]), labels=np.array([1]), kind="stim"))
        n_edge = int(np.sum((src.onsets_s + lo1 < 0) | (src.onsets_s + hi1 > (r.x.shape[1] - 1) / r.fs)))
        assert n_edge > 0 and a.meta["adapter"]["dropped"]["edge"] == n_edge


def test_contrasts_for():
    assert ad.contrasts_for("mbvt", "kojimaB") == ("oddball", "equiprobable")
    assert ad.contrasts_for("chennu13", "kojimaB") == ("equiprobable",) == ad.contrasts_for("gibson16", "kojimaB")
    for d in ("erpcore_mmn", "ds003061"):
        for p in ("mbvt", "chennu13", "gibson16"):
            assert ad.contrasts_for(p, d) == ("oddball",)
        assert ad.contrasts_for("sokoliuk21", d) == ()
    assert ad.contrasts_for(pm, "erpcore_mmn") == ("oddball",)
