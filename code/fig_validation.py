"""Manuscript Figures 3 and 4 (genuine-task controls, sources of the excess, calibration within each design).

Reads only aggregate JSON tables. Exact Clopper-Pearson CIs from k/n where a count is plotted; cluster-bootstrap CIs
are read from the tables. House style of fig_style.py (17 cm, Arial 7 pt and above, Okabe-Ito semantic colours:
vermilion = the rule as published, blue = the command-level variance, green = block-wise or randomization-based, grey
= sham labels). Categorical axes name each point (true / sham), so panels need no legend; Figure 4 has one panel per
analysis, each design its own series, never pooled; a Figure 4 cell whose achieved held-out rate is above its target
(one-sided exact binomial p < 0.05) carries an asterisk and a vermilion frame, and a cell whose target is finer than
the null's half can resolve (target < 1 / (null units per half + 1)) a dagger. Every plotted value goes through put()
for the test.
python fig_validation.py -> output/figures/null_benchmark/Figure{3,4}.{pdf,png}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import fig_null_benchmark as base
import fig_style as fs

TAB, OUT, cp, load = base.TAB, base.OUT, base.cp, base.load
PUBLISHED, CORRECTED, RANDOMIZED, SHAM, INK, MUTED = fs.PUBLISHED, fs.CORRECTED, fs.RANDOMIZED, fs.SHAM, fs.INK, fs.MUTED
PUT = []
FLAGS = {}


def put(fig, panel, series, value, k=None, n=None, lo=None, hi=None):
    PUT.append(dict(fig=fig, panel=panel, series=series, value=value, k=k, n=n, lo=lo, hi=hi))
    return value


def _kn(ax, x, k, n, color, fig, panel, series, marker="o", filled=True):
    lo, hi = cp(k, n)
    put(fig, panel, series, k / n, k=k, n=n, lo=lo, hi=hi)
    fs.errpoint(ax, x, 100 * k / n, 100 * lo, 100 * hi, color, marker, filled)
    return 100 * k / n


def _ci(ax, x, est, lo, hi, color, fig, panel, series, marker="o", filled=True):
    put(fig, panel, series, est, lo=lo, hi=hi)
    fs.errpoint(ax, x, 100 * est, 100 * lo, 100 * hi, color, marker, filled)
    return 100 * est


def _groups(ax, ticks, groups, y=-0.2):
    """Two-level categorical axis: point labels as ticks, group names centred below with a hairline bracket."""
    ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks])
    ax.tick_params(axis="x", length=0, pad=3)
    for x0, x1, name in groups:
        if x1 > x0:
            ax.annotate("", xy=(x0 - 0.15, y + 0.06), xytext=(x1 + 0.15, y + 0.06),
                        xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color=fs.AXIS, lw=0.6))
        ax.text((x0 + x1) / 2, y, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7,
                color=INK)


def _dumbbell(ax, rows, xmax, fig, panel, label_side=None):
    """Horizontal dumbbells: one row per condition, two estimates with 95% CIs joined by a grey line, value printed
    beside each. rows: [(label, (est, lo, hi, colour, marker, series), (est, lo, hi, colour, marker, series))]."""
    for y, (lab, *pts) in enumerate(rows):
        xs = [pt[0] for pt in pts if pt is not None]
        if len(xs) == 2:
            ax.plot([100 * xs[0], 100 * xs[1]], [y, y], color="#C9CED6", lw=2.4, zorder=1, solid_capstyle="round")
        for pt in pts:
            if pt is None:
                continue
            est, lo, hi, col, mk, series = pt
            put(fig, panel, series, est, lo=lo, hi=hi)
            fs.errpoint(ax, y, 100 * est, 100 * lo, 100 * hi, col, mk, ms=6.5, horizontal=True)
            near0 = 100 * est < xmax * 0.05
            ax.text(100 * est + (xmax * 0.012 if near0 else 0), y - 0.3, f"{100 * est:.1f}",
                    ha="left" if near0 else "center", va="bottom", fontsize=8, color=col, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)
    fs.grid(ax, "x")


def _kn_pt(k, n, col, mk, series):
    lo, hi = cp(k, n)
    return (k / n, lo, hi, col, mk, series)


def _key(ax, items, extra=()):
    hs = [mlines.Line2D([], [], marker=mk, ls="none", ms=6, mfc=col, mec="white", label=lab) for lab, col, mk in items]
    hs += list(extra)
    ax.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(hs), fontsize=8 if not extra else 7.5,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2, handlelength=1.2)


def figure3():
    fr, dev = load("cornell_frozen_replication.json"), load("cornell_trialjack.json")
    cl, sn = load("clustered_inference.json"), load("stroke_native.json")
    v2, v2s = load("dubois_reconcile_v2_dev.json"), load("dubois_reconcile_v2S_dev.json")
    rec = load("dubois_reconcile.json")
    fig, axes = plt.subplots(2, 2, figsize=(fs.WIDTH_IN, 4.5),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.62, "left": 0.155, "right": 0.985,
                                          "top": 0.88, "bottom": 0.1, "height_ratios": [3, 4]})
    (a, b), (c, d) = axes

    rows = []
    for lab, r in (("Development", dev["readings"]), ("Replication", fr["cohorts"]["rep"]["readings"]),
                   ("Post-arrest", fr["cohorts"]["icare"]["readings"])):
        s_, c_ = r["sample_normal"], r["command_normal"]
        rows.append((f"{lab} (n = {c_['n']})", _kn_pt(s_["k"], s_["n"], PUBLISHED, "o", f"{lab}|sample"),
                     _kn_pt(c_["k"], c_["n"], CORRECTED, "s", f"{lab}|command")))
    for pt in rows:
        pass
    _dumbbell(a, rows, 32, 3, "A")
    for lab, r in (("Development", dev["readings"]), ("Replication", fr["cohorts"]["rep"]["readings"]),
                   ("Post-arrest", fr["cohorts"]["icare"]["readings"])):
        for rd, sx in (("sample_normal", "sample"), ("command_normal", "command")):
            PUT.append(dict(fig=3, panel="A", series=f"{lab}|{sx}", value=r[rd]["k"] / r[rd]["n"], k=r[rd]["k"],
                            n=r[rd]["n"], lo=cp(r[rd]["k"], r[rd]["n"])[0], hi=cp(r[rd]["k"], r[rd]["n"])[1]))
    fs.bench(a, axis="x")
    a.set_xlabel("False calls without commands, %")
    _key(a, [("Published jackknife", PUBLISHED, "o"), ("Command-level", CORRECTED, "s")])
    fs.panel(a, "A", "Goldfine 2011", x=-0.03, y=1.17)

    st = cl["stroke"]["readings"]
    rows = []
    for key, lab in (("cornell_c3", "Published jackknife"), ("cornell_cmd", "Command-level")):
        sh = sn["readings"][key]["nulls"]["sham"]["rate"]
        rows.append((lab, (st[key]["detection"], *st[key]["ci95"], PUBLISHED if key == "cornell_c3" else CORRECTED,
                           "o" if key == "cornell_c3" else "s", f"{key}|true"),
                     (sh["rate"], *sh["ci95"], SHAM, "o" if key == "cornell_c3" else "s", f"{key}|sham")))
    _dumbbell(b, rows, 104, 3, "B")
    b.set_xlabel("Stroke sessions called, %")
    _key(b, [("Imagery vs rest", INK, "o"), ("Rest-only sham", SHAM, "o")])
    fs.panel(b, "B", "Goldfine 2011, stroke", x=-0.03, y=1.17)

    rows = []
    for lab, rk, bk, src in (("Configuration search", "F_Rxs", "F_Bxs", v2), ("Fixed configuration", "F_Rxf", "F_Bxf", v2),
                             ("Permutation, literal", "P_random_perm", "P_blocked_perm", v2),
                             ("Permutation, search repeated", "P_random_permS", "P_blocked_permS", v2s)):
        r_, b_ = src["steps"][rk], src["steps"][bk]
        rows.append((lab, _kn_pt(r_["k"], r_["n"], fs.ORANGE, "o", f"{lab}|random"),
                     _kn_pt(b_["k"], b_["n"], RANDOMIZED, "s", f"{lab}|blocked")))
    _dumbbell(c, rows, 104, 3, "C")
    ours = rec["steps"]["study_blocked"]
    put(3, "C", "ours", ours["k"] / ours["n"], k=ours["k"], n=ours["n"])
    c.axvline(100 * ours["k"] / ours["n"], ls=":", color=MUTED, lw=1.0, zorder=0)

    fs.bench(c, axis="x")
    c.set_xlabel("Windows without commands called, %")
    _key(c, [("Random folds", fs.ORANGE, "o"), ("Block-wise", RANDOMIZED, "s")],
         [mlines.Line2D([], [], ls=":", color=MUTED, lw=1.0, label="Our implementation")])
    fs.panel(c, "C", "du Bois 2026", x=-0.03, y=1.12)

    t = cl["tactile"]
    rows = []
    for lab, key, unit in (("Published, sessions", "mbvt_primary", "session"), ("Published, runs", "mbvt_primary", "run"),
                           ("Randomization test", "randomization", "session")):
        tr, sh = t[key][f"{unit}_true"], t[key][f"{unit}_sham"]
        rows.append((lab, (tr["estimate"], *tr["ci95"], CORRECTED, "o", f"{key}|{unit}_true"),
                     (sh["estimate"], *sh["ci95"], SHAM, "o", f"{key}|{unit}_sham")))
    _dumbbell(d, rows, 104, 3, "D")
    d.plot([12, 12], [0.7, 1.3], color=INK, lw=1.6, zorder=5)
    d.text(13.5, 1.36, "chance", ha="left", va="center", fontsize=7.5, color=INK)
    d.set_xlabel("Tactile sessions or runs called, %")
    _key(d, [("Attended finger", CORRECTED, "o"), ("Unattended (sham)", SHAM, "o")])
    fs.panel(d, "D", "Guger 2018, tactile", x=-0.03, y=1.12)
    return fig


DESIGNS = [("Goldfine 2011", [("published_schedule|cornell", "Volunteers, 51"),
                              ("stroke_native|cornell_c3", "Stroke, published, 61"),
                              ("stroke_native|cornell_cmd_t", "Stroke, command-level, 61")]),
           ("du Bois 2026", [("published_schedule|dubois", "Volunteers, 51"),
                             ("stroke_native|dubois_reading_b", "Stroke, 61")]),
           ("Guger 2018", [("published_schedule|mbvt", "Auditory counting, 13"),
                           ("braille_native|mbvt_primary", "Tactile runs, 76")])]
KS = ("1", "2", "5", "10")
TARGETS = {"1": 5.0, "2": 2.53, "5": 1.02, "10": 0.51}


def calibration_flags(k_false, n_false, k_assess, n_null):
    """(above, unresolvable) for a calibrated cell: the achieved held-out rate is above the per-assessment target
    1 - 0.95^(1/k) by a one-sided exact binomial test at 0.05, or the target is finer than the null half can resolve
    (a threshold set on n/2 null units cannot reach a rate below 1 / (n/2 + 1))."""
    from scipy.stats import binomtest
    target = 1 - 0.95 ** (1 / int(k_assess))
    above = binomtest(k_false, n_false, target, alternative="greater").pvalue < 0.05
    return above, target < 1 / (n_null // 2 + 1)
GROUP_COL = ["#2C5FAA", "#C65102", "#6A4C93"]


VOLUNTEER_NULL = {"cornell": "ncmd_bdsp|cornell", "dubois": "ncmd_bdsp|dubois", "mbvt": "nstim0_bdsp|mbvt"}


def volunteer_null_rate(pipe, n_null):
    """A volunteer design's null is the clinical development EEG: the published rule's rate there is Figure 1's
    clinical cell (its denominator must be the design's null count)."""
    sr = load("stated_rule.json")["entries"][VOLUNTEER_NULL[pipe]]["stated_rule"]
    if sr["n"] != n_null:
        raise RuntimeError(f"{pipe}: clinical cell n {sr['n']} != the design's null n {n_null}")
    put(4, "A", f"published_schedule|{pipe}|own", sr["k"] / sr["n"], k=sr["k"], n=sr["n"])
    return sr["k"] / sr["n"]


def figure4():
    import matplotlib as mpl
    m = load("matched_calibration.json")["designs"]
    rows = [(g, key, lab) for g, ds in DESIGNS for key, lab in ds]
    ncol = 1 + len(KS)
    fig, ax = plt.subplots(figsize=(fs.WIDTH_IN, 3.85))
    fig.subplots_adjust(left=0.35, right=0.85, top=0.8, bottom=0.12)
    cmap = mpl.colormaps["viridis"]
    for i, (g, key, lab) in enumerate(rows):
        v = m[key]
        det, nul = v["published_detection"], v.get("published_null_rate")
        if nul is None and key.startswith("published_schedule|"):
            nul = volunteer_null_rate(key.split("|")[1], v["n_null"])
        cells = [("pub", det, nul)] + [(k, v["calibrated"][k]["detection"],
                                         v["calibrated"][k]["false_call_heldout_k"] / v["calibrated"][k]["false_call_heldout_n"])
                                        for k in KS]
        for j, (code, dval, fval) in enumerate(cells):
            ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=cmap(dval), lw=0, zorder=1))
            tc = fs.text_on(100 * dval, 0, 100)
            ax.text(j, i - 0.13, f"{100 * dval:.1f}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color=tc, zorder=3)
            sub = f"{100 * fval:.1f}%"
            if code != "pub":
                c = v["calibrated"][code]
                put(4, "A", f"{key}|k{code}|false", fval, k=c["false_call_heldout_k"], n=c["false_call_heldout_n"])
                put(4, "A", f"{key}|k{code}|detection", dval)
                above, unres = calibration_flags(c["false_call_heldout_k"], c["false_call_heldout_n"], code, v["n_null"])
                FLAGS[(key, code)] = (above, unres)
                sub += ("*" if above else "") + ("\u2020" if unres else "")
                if above:
                    ax.add_patch(mpatches.Rectangle((j - 0.44, i - 0.42), 0.88, 0.84, fill=False, ec=PUBLISHED,
                                                    lw=1.8, zorder=5))
            ax.text(j, i + 0.25, sub, ha="center", va="center", fontsize=7, color=tc, zorder=3)
    nrow = len(rows)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -1.0)
    for x in range(1, ncol):
        ax.axvline(x - 0.5, color="white", lw=1.2, zorder=2)
    for y in range(1, nrow):
        ax.axhline(y - 0.5, color="white", lw=1.2, zorder=2)
    ax.axvline(0.5, color=INK, lw=1.2, zorder=4)
    ax.text(0, -0.78, "own rate", ha="center", va="center", fontsize=7.5, color=MUTED)
    for j, k in enumerate(KS, start=1):
        ax.text(j, -0.78, f"target {TARGETS[k]:.1f}%", ha="center", va="center", fontsize=7.5, color=MUTED)
    for j, lab in enumerate(["As published"] + [f"k = {k}" for k in KS]):
        ax.text(j, -1.35, lab, ha="center", va="center", fontsize=8.5, fontweight="bold" if j else "normal",
                clip_on=False)
    ax.plot([0.58, ncol - 0.58], [-1.78, -1.78], color=INK, lw=0.8, clip_on=False)
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.set_yticks(range(nrow), [lab for _, _, lab in rows], fontsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    y0 = 0
    for gi, (g, ds) in enumerate(DESIGNS):
        y1 = y0 + len(ds) - 1
        ax.plot([-3.25, -3.25], [y0 - 0.4, y1 + 0.4], color=GROUP_COL[gi], lw=3.4, clip_on=False, solid_capstyle="butt")
        ax.text(-3.45, (y0 + y1) / 2, g, ha="center", va="center", rotation=90, fontsize=8, fontweight="bold",
                color=GROUP_COL[gi], clip_on=False)
        if y0:
            ax.axhline(y0 - 0.5, color=INK, lw=1.2, zorder=4)
        y0 = y1 + 1
    ax.text(2.5, -2.05, "Thresholds targeting a 5% chance of any false call over k assessments", ha="center",
            va="center", fontsize=8.5, fontweight="bold", clip_on=False)
    fig.text(0.02, 0.055, "Small number: achieved false-call rate per assessment on held-out null recordings.   "
             "*  Above its target (one-sided exact p < 0.05).", fontsize=7, color=INK, ha="left", va="center")
    halves = {m[key]["n_null"] // 2 for (key, _), (_, unres) in FLAGS.items() if unres}
    if len(halves) != 1:
        raise RuntimeError(f"dagger cells come from nulls of different sizes: {halves}")
    fig.text(0.02, 0.02, f"\u2020  Target finer than the null can resolve ({halves.pop()} sham runs per half).   "
             "Each row has its own task and null; rows are not pooled.", fontsize=7, color=INK, ha="left",
             va="center")
    cax = fig.add_axes([0.86, 0.12, 0.018, 0.62])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap="viridis"), cax=cax,
                      ticks=[0, 25, 50, 75, 100])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.set_label("Single-assessment detection, %", fontsize=8.5)
    return fig


def main():
    for n, make in ((3, figure3), (4, figure4)):
        fs.save(make(), OUT, f"Figure{n}")
    print(f"Figures 3 and 4 written; {len(PUT)} plotted values")


if __name__ == "__main__":
    main()
