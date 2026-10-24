"""Instruction randomization on N-passive data (adapters.instruction_overlay / run_instruction): blocks of the real stimulus stream (the module's block length, else 8 blocks), sham "count" / "ignore"
instructions per block (Kojima B: a sham attended stream per block), the module's own deviant-minus-standard feature
difference contrasted between count and ignore blocks, and the exact null re-drawing the block instructions.
- Validity: 200 draws per pipeline (20 synthetic recordings x 10 instruction draws, B = 500) on noise with a real
  deviant response that habituates over the recording; false calls inside the binomial 95% interval.
- A planted "attention" effect (a larger deviant response in count blocks) is called.
- A deliberately broken null that re-draws the stimulus labels instead of the instructions fails the validity check.
- Pooled per subject (the primary unit): blocks stay within runs, instructions drawn balanced over all the
  subject's blocks; validity on 200 draws (20 synthetic three-run subjects x 10 draws) and a planted effect called.
- Unusable is never negative: an inadmissible observed labelling or too few kept trials raise NoUsableTrials.
- Inadmissible null draws are re-drawn (a conditional randomization test, motor_common.admissible_stat): validity
  on 200 draws under heavy label-independent rejection (LOST of every run's 8 blocks lost, half the other epochs), the
  rejection-sampled p matching the exact conditional p by enumeration, and two deliberately broken nulls that let the
  inadmissible draws through (as non-finite, or zero-filled) missing it.
Synthetic data only.
cd code && /usr/bin/python3 -m pytest bench/tests/test_instruction.py -q"""
import itertools
import time

import numpy as np
import pytest
from scipy.stats import binom

from bench import adapters as ad
from bench import engine as en
from bench import motor_common as mc
from bench import pipe_mbvt as pm
from bench.tests import _synth as sy

PIPES = ("mbvt", "chennu13", "gibson16")
SOA, START, TAIL = 0.6, 12.0, 12.0


def stream_overlay(n, rng, rare_p=0.15, trains=1, kojima=False):
    """The passive_overlay layout (labels 1 rare / 0 standard; never two rares in a row; trains split by 20-s
    pauses). kojima: the analysed (really unattended) stream alternates 2, 1, 2, 1 over the trains, with codes 1 / 2
    (standards) and 101-104 (deviants), as passive.stream_table gives them."""
    lab, prev = [], 0
    for _ in range(n):
        k = int(prev == 0 and rng.random() < rare_p)
        lab.append(k)
        prev = k
    lab = np.array(lab)
    train = np.repeat(np.arange(trains), int(np.ceil(n / trains)))[:n]
    on = START + np.arange(n) * SOA + 20.0 * train
    rt = np.where(lab == 1, rng.integers(1, 3, n), 0)
    if kojima:
        stream = np.where(train % 2 == 0, 2, 1)
        code = np.where(lab == 0, stream, np.where(stream == 1, 100 + rt, 102 + rt))
    else:
        code = lab.copy()
    return en.Overlay(onsets_s=on, labels=lab, kind="stim",
                      meta={"dataset": "kojimaB" if kojima else "erpcore_mmn", "contrast": "oddball", "code": code,
                            "rare_type": rt, "seq": np.arange(n), "train": train, "trial": np.arange(n),
                            "block": train.copy(), "run": np.zeros(n, int), "soa_s": SOA, "excluded": {}})


def noise_rec(ov, seed):
    r = sy.rec(float(ov.onsets_s.max()) + TAIL, seed=seed)
    return en.Recording(source="synthetic", uid=f"syn{seed}", patient=f"p{seed}", x=r.x, fs=r.fs, names=list(r.names),
                        meta={"dataset": ov.meta["dataset"]})


def add_erp(rec, onsets, amps, tag):
    """A copy with a P3-like response (peak 400 ms, SD 80 ms, Cz-Pz topography) of amps[i] µV after onsets[i]."""
    x = np.array(rec.x, float)
    a = en.erp_topography(rec.names)
    t = np.arange(int(0.8 * rec.fs)) / rec.fs
    w = np.exp(-(t - 0.4) ** 2 / (2 * 0.08 ** 2))
    for o, amp in zip(onsets, amps):
        i = int(round(o * rec.fs))
        x[:, i:i + len(t)] += amp * a[:, None] * w[None]
    return en.Recording(source=rec.source, uid=f"{rec.uid}{tag}", patient=rec.patient, x=x.astype(np.float32),
                        fs=rec.fs, names=list(rec.names), meta=dict(rec.meta))


def habituating(rec, ov, top=8.0):
    """Real deviant response, decreasing linearly from top to 0 µV over the recording (no instruction effect)."""
    rare = ov.labels == 1
    on = ov.onsets_s[rare]
    return add_erp(rec, on, top * (1 - (on - on.min()) / (on.max() - on.min())), "+hab")


def check_validity(calls):
    n = int(np.sum(calls))
    lo, hi = binom.interval(0.95, len(calls), 0.05)
    assert lo <= n <= hi, f"{n}/{len(calls)} false calls outside [{lo:.0f}, {hi:.0f}]"



def test_blocks_and_instructions():
    ov = stream_overlay(1200, np.random.default_rng(0))
    rec = noise_rec(ov, 0)
    for pipe, rule, nb in (("mbvt", "module", 10), ("chennu13", "module", 12), ("gibson16", "fallback", 8)):
        a = ad.instruction_overlay(pipe, rec, ov, np.random.default_rng(3))
        info = a.meta["adapter"]
        assert info["design"] == "instruction" and info["block_rule"] == rule and info["usable"]["n_blocks"] == nb
        blk = a.meta["block"]
        assert np.array_equal(np.unique(blk), np.arange(nb)) and (np.diff(blk) >= 0).all()
        if rule == "module":
            assert (np.bincount(blk) == ad.instruction_block_len(pipe)).all()
        else:
            assert np.bincount(blk).max() - np.bincount(blk).min() <= 1
        j = np.searchsorted(ov.onsets_s, a.onsets_s)
        assert np.allclose(ov.onsets_s[j], a.onsets_s) and np.array_equal(a.labels, ov.labels[j])
        inst = a.meta["instruction"]
        expect = np.random.default_rng(3).permutation(np.repeat([ad.COUNT, ad.IGNORE], [nb // 2, nb - nb // 2]))
        assert np.array_equal(inst, expect)
        assert info["usable"]["n_count"] == nb // 2 and min(info["usable"]["rare_per_block"]) >= 2
    assert ad.instruction_block_len("mbvt") == 120 and ad.instruction_block_len("chennu13") == 97
    assert ad.instruction_block_len("gibson16") > 150


def test_kojima_sham_attended_stream():
    ov = stream_overlay(600, np.random.default_rng(1), trains=4, kojima=True)
    rec = noise_rec(ov, 1)
    for pipe in PIPES:
        a = ad.instruction_overlay(pipe, rec, ov, np.random.default_rng(4))
        info = a.meta["adapter"]
        assert info["usable"]["n_blocks"] == 8 and info["block_rule"] == "fallback"
        j = np.searchsorted(ov.onsets_s, a.onsets_s)
        blk, tr = a.meta["block"], ov.meta["train"][j]
        assert all(len(set(tr[blk == b])) == 1 for b in range(8))
        assert (np.bincount(tr[np.r_[True, np.diff(blk) > 0]]) == 2).all()
        an, sham, inst = a.meta["analysed_stream"], a.meta["sham_stream"], a.meta["instruction"]
        assert np.array_equal(an, np.where(tr[np.r_[True, np.diff(blk) > 0]] % 2 == 0, 2, 1))
        assert np.array_equal(sham, np.where(inst == ad.COUNT, an, 3 - an)) and inst.sum() == 4
        assert info["usable"]["n_conflict"] == 4



def _validity_task(args):
    pipe, seed, top = args
    ov = stream_overlay(1000, np.random.default_rng(seed))
    rec = habituating(noise_rec(ov, seed), ov, top)
    calls = []
    for k in range(10):
        a = ad.instruction_overlay(pipe, rec, ov, np.random.default_rng(1000 * seed + k))
        calls.append(ad.run_instruction(pipe, rec, a, np.random.default_rng(7 + k))["call"])
    return calls


@pytest.mark.parametrize("pipe", PIPES)
def test_instruction_validity(pipe):
    t = time.time()
    calls = sum(sy.pool_map(_validity_task, [(pipe, s, 8.0) for s in range(20)], workers=4), [])
    print(f"{pipe} instruction randomization: {int(np.sum(calls))}/200 false calls ({time.time() - t:.0f} s)")
    assert len(calls) == 200
    check_validity(calls)


@pytest.mark.parametrize("pipe", PIPES)
def test_planted_attention_is_called(pipe):
    ov = stream_overlay(1000, np.random.default_rng(5))
    rec = noise_rec(ov, 5)
    a = ad.instruction_overlay(pipe, rec, ov, np.random.default_rng(6))
    j = np.searchsorted(ov.onsets_s, a.onsets_s)
    rare = a.labels == 1
    count = a.meta["instruction"][a.meta["block"]] == ad.COUNT
    planted = add_erp(rec, a.onsets_s[rare], np.where(count[rare], 10.0, 3.0), "+att")
    out = ad.run_instruction(pipe, planted, a, np.random.default_rng(2))
    assert out["call"] is True and out["p"] < 0.05 and np.isfinite(out["stat"])
    assert len(j) == len(a.onsets_s)


def test_broken_stimulus_label_null_fails(monkeypatch):
    def stimulus_label_null(m, F, meta, g, tally=None):
        rare = F["rare"].copy()
        for b in range(F["nb"]):
            i = np.nonzero(F["block"] == b)[0]
            rare[i] = rare[g.permutation(i)]
        return ad._instr_stat(m, ad._instr_sums(F, rare), meta["instruction"])
    good = sum((_validity_task(("mbvt", s, 30.0)) for s in range(4)), [])
    check_validity(good)
    monkeypatch.setattr(ad, "_null_draw", stimulus_label_null)
    bad = sum((_validity_task(("mbvt", s, 30.0)) for s in range(4)), [])
    assert np.sum(bad) >= 2 * binom.interval(0.95, len(bad), 0.05)[1]
    with pytest.raises(AssertionError):
        check_validity(bad)



def test_instruction_refusals():
    ov = stream_overlay(1000, np.random.default_rng(8))
    rec = noise_rec(ov, 8)
    with pytest.raises(ad.NotApplicable):
        ad.instruction_overlay("sokoliuk21", rec, ov)
    short = stream_overlay(60, np.random.default_rng(8))
    with pytest.raises(ad.NotApplicable):
        ad.instruction_overlay("mbvt", noise_rec(short, 8), short)
    pub = ad.adapt("mbvt", rec, ov)
    with pytest.raises(ValueError):
        ad.run_instruction("mbvt", rec, pub)
    a = ad.instruction_overlay("mbvt", rec, ov, np.random.default_rng(0))
    assert ad.instruction_overlay("mbvt", rec, ov).meta["adapter"]["design"] == "instruction"
    out = ad.run_instruction("mbvt", rec, a, np.random.default_rng(0), n_perm=50)
    assert set(out) == {"stat", "p", "call", "extra"} and out["extra"]["n_blocks"] == 8
    assert mc.stop_count(500, 0.05) == 25 and pm.GROUP == 8



def subject_runs(seed, n_runs=3, n=700, top=8.0):
    """Synthetic three-run subject: per run a stream (single train), noise with a habituating deviant response."""
    out = []
    for r in range(n_runs):
        ov = stream_overlay(n, np.random.default_rng(100 * seed + r))
        rec = noise_rec(ov, 100 * seed + r)
        rec = en.Recording(source=rec.source, uid=f"subj{seed}_run{r}", patient=f"subj{seed}", x=rec.x, fs=rec.fs,
                           names=rec.names, meta=rec.meta)
        out.append((habituating(rec, ov, top) if top else rec, ov))
    return out


def test_pooled_blocks_and_balance():
    runs = subject_runs(0, top=0.0)
    for pipe in PIPES:
        ins = [(r, ad.instruction_overlay(pipe, r, ov, np.random.default_rng(1))) for r, ov in runs]
        nb = [a.meta["adapter"]["usable"]["n_blocks"] for _, a in ins]
        F = ad.pool_instruction_features(pipe, iter(ins))
        N = sum(nb)
        assert F["nb"] == N and F["n_runs"] == 3 and F["patient"] == "subj0"
        assert [k for _, k in F["run_blocks"]] == nb and [u for u, _ in F["run_blocks"]] == [r.uid for r, _ in runs]
        inst = ad.pooled_instruction(N, np.random.default_rng(5))
        assert np.array_equal(inst, np.random.default_rng(5).permutation(
            np.repeat([ad.COUNT, ad.IGNORE], [N // 2, N - N // 2])))
        out = ad.run_instruction_pooled(pipe, F, np.random.default_rng(5))
        assert set(out) == {"stat", "p", "call", "extra"} and out["extra"]["n_blocks"] == N
        assert np.array_equal(out["extra"]["instruction"], inst) and out["extra"]["n_runs"] == 3
        per_run = out["extra"]["instruction_by_run"]
        assert [len(v) for v in per_run.values()] == nb and np.array_equal(np.concatenate(list(per_run.values())), inst)
    (r0, o0), (r1, o1) = runs[0], runs[1]
    with pytest.raises(ValueError, match="not a chennu13 instruction_overlay"):
        ad.pool_instruction_features("chennu13", iter([(r0, ad.instruction_overlay("chennu13", r0, o0)),
                                                       (r1, ad.instruction_overlay("gibson16", r1, o1))]))
    fewer = en.Recording(source=r1.source, uid="fewer", patient="subj0", x=r1.x[1:], fs=r1.fs, names=r1.names[1:],
                         meta=r1.meta)
    with pytest.raises(ValueError, match="runs differ"):
        ad.pool_instruction_features("gibson16", iter([(r0, ad.instruction_overlay("gibson16", r0, o0)),
                                                       (fewer, ad.instruction_overlay("gibson16", fewer, o1))]))
    with pytest.raises(ValueError, match="no runs"):
        ad.pool_instruction_features("mbvt", iter([]))


def _pooled_validity_task(args):
    pipe, seed = args
    runs = subject_runs(seed)
    F = ad.pool_instruction_features(pipe, ((r, ad.instruction_overlay(pipe, r, ov)) for r, ov in runs))
    return [ad.run_instruction_pooled(pipe, F, np.random.default_rng(1000 * seed + k))["call"] for k in range(10)]


@pytest.mark.parametrize("pipe", PIPES)
def test_pooled_validity(pipe):
    t = time.time()
    calls = sum(sy.pool_map(_pooled_validity_task, [(pipe, s) for s in range(20)], workers=4), [])
    print(f"{pipe} pooled instruction randomization: {int(np.sum(calls))}/200 false calls ({time.time() - t:.0f} s)")
    assert len(calls) == 200
    check_validity(calls)


@pytest.mark.parametrize("pipe", PIPES)
def test_pooled_planted_attention_is_called(pipe):
    runs = subject_runs(7, top=0.0)
    ins = [(r, ad.instruction_overlay(pipe, r, ov, np.random.default_rng(2))) for r, ov in runs]
    N = sum(a.meta["adapter"]["usable"]["n_blocks"] for _, a in ins)
    inst = ad.pooled_instruction(N, np.random.default_rng(11))
    planted, k = [], 0
    for r, a in ins:
        nb = a.meta["adapter"]["usable"]["n_blocks"]
        rare = a.labels == 1
        count = inst[k:k + nb][a.meta["block"]] == ad.COUNT
        planted.append((add_erp(r, a.onsets_s[rare], np.where(count[rare], 6.0, 2.0), "+att"), a))
        k += nb
    F = ad.pool_instruction_features(pipe, iter(planted))
    out = ad.run_instruction_pooled(pipe, F, np.random.default_rng(11))
    assert np.array_equal(out["extra"]["instruction"], inst)
    assert out["call"] is True and out["p"] < 0.05



def with_artefacts(rec, inst, lost, frac, rng, tag):
    """Label-independent artefacts: a 250-µV 3-Hz oscillation over every stimulus of the lost blocks (whole blocks
    rejected by the +-100 µV rules) and a 400-µV, 0.1-s bump 0.3 s after a random `frac` of the other stimuli."""
    x = np.array(rec.x, float)
    t = np.arange(x.shape[1]) / rec.fs
    blk, on = inst.meta["block"], inst.onsets_s
    for b in lost:
        m = (t >= on[blk == b].min() - 0.5) & (t <= on[blk == b].max() + 1.0)
        x[:, m] += 250.0 * np.sin(2 * np.pi * 3.0 * t[m])
    w = int(0.1 * rec.fs)
    for o in on[(rng.random(len(on)) < frac) & ~np.isin(blk, lost)]:
        i = int(round((o + 0.3) * rec.fs))
        x[:, i:i + w] += 400.0
    return en.Recording(source=rec.source, uid=f"{rec.uid}{tag}", patient=rec.patient, x=x.astype(np.float32),
                        fs=rec.fs, names=list(rec.names), meta=dict(rec.meta))


def test_unusable_raises_never_negative():
    ov = stream_overlay(1000, np.random.default_rng(20))
    rec = noise_rec(ov, 20)
    for pipe in ("mbvt", "gibson16"):
        a = ad.instruction_overlay(pipe, rec, ov, np.random.default_rng(1))
        count = np.nonzero(a.meta["instruction"] == ad.COUNT)[0]
        ignore = np.nonzero(a.meta["instruction"] == ad.IGNORE)[0]
        bad = with_artefacts(rec, a, ignore, 0.0, np.random.default_rng(0), "+noignore")
        with pytest.raises(mc.NoUsableTrials):
            ad.run_instruction(pipe, bad, a, np.random.default_rng(2))
        pooled_ignore = np.nonzero(ad.pooled_instruction(8, np.random.default_rng(2)) == ad.IGNORE)[0]
        bad_p = with_artefacts(rec, a, pooled_ignore, 0.0, np.random.default_rng(0), "+noignore_p")
        F = ad.pool_instruction_features(pipe, iter([(bad_p, a)]))
        with pytest.raises(mc.NoUsableTrials):
            ad.run_instruction_pooled(pipe, F, np.random.default_rng(2), n_perm=20)
        dead = with_artefacts(rec, a, np.r_[count, ignore], 0.0, np.random.default_rng(0), "+dead")
        with pytest.raises(mc.NoUsableTrials):
            ad.run_instruction(pipe, dead, a, np.random.default_rng(2))
    a = ad.instruction_overlay("mbvt", rec, ov, np.random.default_rng(1))
    out = ad.run_instruction("mbvt", rec, a, np.random.default_rng(2), n_perm=50)
    assert "too_few_trials" not in out["extra"] and out["extra"]["n_inadmissible_draws"] == 0


LOST = {"mbvt": 6, "gibson16": 5}


def _rejection_task(args):
    pipe, seed = args
    runs = []
    for r in range(3):
        ov = stream_overlay(700, np.random.default_rng(100 * seed + r))
        rec = noise_rec(ov, 100 * seed + r)
        rec = en.Recording(source=rec.source, uid=f"rj{seed}_{r}", patient=f"rj{seed}", x=rec.x, fs=rec.fs,
                           names=rec.names, meta=rec.meta)
        inst = ad.instruction_overlay(pipe, rec, ov, np.random.default_rng(0))
        g = np.random.default_rng(10 * seed + r)
        lost = g.choice(8, LOST[pipe], replace=False)
        rec = with_artefacts(habituating(rec, ov), inst, lost, 0.5, g, "+rej")
        runs.append((rec, inst))
    F = ad.pool_instruction_features(pipe, iter(runs))
    calls, redrawn = [], 0
    for k in range(10):
        try:
            out = ad.run_instruction_pooled(pipe, F, np.random.default_rng(1000 * seed + k))
        except mc.NoUsableTrials:
            calls.append(None)
            continue
        calls.append(out["call"])
        redrawn += out["extra"]["n_inadmissible_draws"]
    return calls, redrawn, int(F["usable"].sum())


@pytest.mark.parametrize("pipe", ("mbvt", "gibson16"))
def test_validity_under_heavy_rejection(pipe):
    t = time.time()
    res = sy.pool_map(_rejection_task, [(pipe, s) for s in range(20)], workers=4)
    allc = sum((c for c, _, _ in res), [])
    calls = [c for c in allc if c is not None]
    redrawn, usable = sum(r for _, r, _ in res), [u for _, _, u in res]
    print(f"{pipe} under rejection: {int(np.sum(calls))}/{len(calls)} false calls among usable assessments "
          f"({len(allc) - len(calls)} unusable: inadmissible observed draw), {redrawn} inadmissible null draws "
          f"re-drawn, usable blocks per subject {min(usable)}-{max(usable)} of 24 ({time.time() - t:.0f} s)")
    assert len(allc) == 200 and len(calls) >= 150 and max(usable) <= 3 * (8 - LOST[pipe]) and redrawn > 0
    check_validity(calls)


def _exact_p(m, F, obs_lab):
    """Exact conditional p by enumeration over every balanced labelling, admissible ones only."""
    nb = F["nb"]
    obs = ad._instr_stat(m, F, obs_lab)
    vals = []
    for cnt in itertools.combinations(range(nb), nb // 2):
        lab = np.full(nb, ad.IGNORE)
        lab[list(cnt)] = ad.COUNT
        if ad._instr_ok(F, lab == ad.COUNT):
            vals.append(ad._instr_stat(m, F, lab))
    vals = np.array(vals)
    return float(np.mean(vals >= obs - 1e-12)), len(vals), obs


def test_rejection_sampling_matches_exact_and_broken_nulls_do_not():
    m = ad._module("mbvt")
    ov = stream_overlay(1000, np.random.default_rng(30))
    rec = noise_rec(ov, 30)
    a = ad.instruction_overlay("mbvt", rec, ov, np.random.default_rng(1))
    lost = np.random.default_rng(3).choice(8, 6, replace=False)
    bad = with_artefacts(habituating(rec, ov), a, lost, 0.0, np.random.default_rng(0), "+six")
    F = ad.pool_instruction_features("mbvt", iter([(bad, a)]))
    assert int(F["usable"].sum()) == 2
    labs = [lab for lab in (np.isin(np.arange(8), c).astype(int) for c in itertools.combinations(range(8), 4))
            if ad._instr_ok(F, lab == ad.COUNT)]
    obs_lab = max(labs, key=lambda lab: ad._instr_stat(m, F, lab))
    p_exact, n_adm, obs = _exact_p(m, F, obs_lab)
    assert obs > 0 and n_adm < 70 and abs(p_exact - 0.5) < 1e-9
    B, meta = 4000, {"instruction": obs_lab}

    def p_of(draw):
        g = np.random.default_rng(9)
        return mc.perm_test(obs, lambda: draw(m, F, meta, g), B, 0.05, early_stop=False)["p"]
    se = np.sqrt(p_exact * (1 - p_exact) / B)
    p_ok = p_of(ad._null_draw)
    assert abs(p_ok - p_exact) < 4 * se

    def raw(m_, F_, meta_, g_):
        lab = ad._redraw_instruction(meta_, g_)
        return ad._instr_stat(m_, F_, lab) if ad._instr_ok(F_, lab == ad.COUNT) else np.nan

    def zero(m_, F_, meta_, g_):
        v = raw(m_, F_, meta_, g_)
        return v if np.isfinite(v) else 0.0
    p_nan, p_zero = p_of(raw), p_of(zero)
    assert p_nan - p_exact > 0.15
    assert p_exact - p_zero > 0.15
