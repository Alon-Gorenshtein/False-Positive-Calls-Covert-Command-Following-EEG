"""Null-benchmark manuscript Figures 1-3.

Reads only aggregate JSON tables (no patient rows), recomputes every error bar as an exact
Clopper-Pearson 95% CI from k/n, and writes output/figures/null_benchmark/Figure{1,2,3}.{pdf,png}.
Style: Helvetica/Arial, Okabe-Ito colours, 600 DPI, no in-figure titles, top/right spines off.

Every plotted number is routed through `put()` so the test can compare it to the JSON.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "output" / "tables"
OUT = ROOT / "output" / "figures" / "null_benchmark"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#6B7280", "axes.linewidth": 0.8,
    "xtick.color": "#2B2B2B", "ytick.color": "#2B2B2B",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE, YELLOW = "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
INK, MUTED, GRID = "#2B2B2B", "#6B7280", "#E5E7EB"

NAMES = {
    "cornell": "Cornell spectral test",
    "dubois": "du Bois configuration search",
    "mbvt": "Vibrotactile P300 (mindBEAGLE-type)",
    "columbia": "Columbia SVM",
    "gibson16": "Gibson 2016",
    "cruse12p": "Cruse 2012",
    "mbmi": "Motor imagery (mindBEAGLE-type)",
    "cruse11": "Cruse 2011",
    "edlow17": "Edlow 2017",
    "chennu13": "Chennu 2013",
    "sokoliuk21": "Covert processing (Sokoliuk 2021)",
}
COMMAND = ["cornell", "dubois", "mbvt", "columbia", "gibson16", "cruse12p", "mbmi", "cruse11",
           "edlow17", "chennu13"]
SEPARATE = "sokoliuk21"
NULLKIND = {p: "ncmd" for p in ["cornell", "dubois", "columbia", "cruse12p", "mbmi", "cruse11", "edlow17"]}
NULLKIND.update({p: "nstim0" for p in ["mbvt", "gibson16", "chennu13", "sokoliuk21"]})

PUT = []


def put(fig, panel, row, series, k=None, n=None, x=None, y=None, **kw):
    rec = dict(fig=fig, panel=panel, row=row, series=series, k=k, n=n, x=x, y=y, **kw)
    PUT.append(rec)
    return rec


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def load(name):
    with open(TAB / name) as f:
        return json.load(f)


def fig1_data():
    sr = load("stated_rule.json")["entries"]
    wn = load("white_noise_eligible.json")["variants"]
    rand = {}
    for fn in ("benchmark_ncmd_bdsp.json", "benchmark_nstim0_bdsp.json"):
        pv = load(fn)["proposed_validity"]
        for pipe, v in pv.items():
            agg = {}
            for u in v["families"]["null_set"]:
                nul = u["unit"].split("|")[0]
                a = agg.setdefault(nul, [0, 0])
                a[0] += u["calls"]
                a[1] += u["n"]
            rand[pipe] = agg
    ic = load("icare_stated_rule.json")
    aw = load("icare_stated_rule_all_windows.json")["entries"]
    for pipe, v in ic["randomised_pooled"].items():
        if not isinstance(v, dict) or "families" not in v:
            continue
        for u in v["families"]["null_set"]:
            nul = u["unit"].split("|")[0]
            a = rand.setdefault(pipe, {}).setdefault(nul, [0, 0])
            a[0] += u["calls"]
            a[1] += u["n"]
    w500 = {p: v["white500"] for p, v in load("nominal_rates.json")["pipes"].items() if v.get("white500")}
    rep = load("replication.json")["pipes"] if (TAB / "replication.json").exists() else {}
    rows = {}
    for pipe in COMMAND + [SEPARATE]:
        kind = NULLKIND[pipe]
        r = {}
        for ds, suffix in (("clin", "bdsp"), ("doc", "pdoc"), ("icare", "icare")):
            null = f"{kind}_{suffix}"
            e = (ic["entries"] if ds == "icare" else sr).get(f"{null}|{pipe}")
            d = {"primary": None, "planned": None, "random": None, "replication": None, "inflated": False,
                 "wn": None, "present": e is not None}
            if e is not None:
                if e.get("stated_rule"):
                    d["primary"] = (e["stated_rule"]["k"], e["stated_rule"]["n"])
                if ds == "icare" and d["primary"] is not None:
                    a = aw[f"{null}|{pipe}"]
                    if a["variant"] != e.get("variant"):
                        raise RuntimeError(f"{pipe}: all-windows reading differs from the post-arrest primary")
                    d["primary"] = (a["all_windows"]["k"], a["all_windows"]["n"])
                vp = e["beside"]["validity_primary"]
                d["planned"] = (vp["k"], vp["n"])
                d["inflated"] = bool((e.get("held_out_test") or {}).get("inflated", False))
                var = e.get("variant")
                if var is not None and ds != "icare":
                    w = w500[pipe] if ds == "clin" and pipe in w500 else wn[pipe][var]
                    d["wn"] = (w["k"], w["n"])
                if ds == "clin" and pipe in rep:
                    d["replication"] = (rep[pipe]["cell"]["k"], rep[pipe]["cell"]["n"])
            ra = rand.get(pipe, {}).get(null)
            if ra:
                d["random"] = tuple(ra)
            r[ds] = d
        rows[pipe] = r
    return rows


def order_rows(rows):
    def key(p):
        pr = rows[p]["clin"]["primary"]
        return -(pr[0] / pr[1]) if pr else 1.0
    return sorted(COMMAND, key=key)


def _errdot(ax, fig, panel, row, series, y, kn, marker, color, filled, ms=4.6, ci=True):
    k, n = kn
    lo, hi = cp(k, n)
    x = 100 * k / n
    if ci:
        ax.plot([100 * lo, 100 * hi], [y, y], color=color, lw=0.9, solid_capstyle="butt", zorder=2)
    ax.plot([x], [y], marker=marker, ms=ms, mfc=color if filled else "white", mec=color, mew=1.0,
            ls="none", zorder=3)
    put(fig, panel, row, series, k=k, n=n, x=x, y=y, ci=ci)


def make_fig1():
    rows = fig1_data()
    order = order_rows(rows)
    ypos = {p: i for i, p in enumerate(order)}
    ypos[SEPARATE] = len(order) + 0.9
    divider = len(order) + 0.0
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 5.2), sharey=True,
                             gridspec_kw={"wspace": 0.10, "left": 0.27, "right": 0.995,
                                          "top": 0.94, "bottom": 0.175})
    offs = {"primary": -0.22, "replication": 0.0, "random": 0.22}
    spec = {"primary": ("o", VERM, True), "replication": ("D", INK, True), "random": ("^", GREEN, True)}
    heads = {0: ("A", "Clinical EEG"), 1: ("B", "Prolonged DoC EEG"),
             2: ("C", "Post-arrest coma EEG"), 3: ("D", "Synthetic white noise")}
    for j, ax in enumerate(axes):
        ax.set_xlim(-1.5, 50)
        ax.set_xticks(range(0, 51, 25))
        ax.set_xticklabels([f"{t}" for t in range(0, 51, 25)])
        ax.set_xticks(range(0, 51, 5), minor=True)
        ax.grid(axis="x", which="minor", color=GRID, lw=0.3, zorder=0)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(5, color="#9CA3AF", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhline(divider, color="#D1D5DB", lw=0.8, zorder=1)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")
        L, T = heads[j]
        ax.text(0.0, 1.035, L, transform=ax.transAxes, fontsize=11, fontweight="bold",
                color=INK, ha="left", va="bottom")
        ax.text(0.12, 1.045, T, transform=ax.transAxes, fontsize=7, color=INK,
                ha="left", va="bottom")
    fig.text((0.27 + 0.995) / 2, 0.118, "False-call rate, %", ha="center", va="center", fontsize=8)
    for p in COMMAND + [SEPARATE]:
        y0 = ypos[p]
        for j, ds in ((0, "clin"), (1, "doc"), (2, "icare")):
            ax = axes[j]
            d = rows[p][ds]
            panel = "ABC"[j]
            if not d["present"]:
                ax.text(25, y0, "not evaluated", ha="center", va="center", fontsize=7,
                        style="italic", color=MUTED)
                continue
            for s in ("primary", "replication", "random"):
                if d[s] is None:
                    continue
                m, c, filled = spec[s]
                _errdot(ax, 1, panel, p, s, y0 + offs[s], d[s], m, c, filled, ci=(s != "random"))
            if d["primary"] is None:
                ax.text(47.5, y0 - 0.26, "does not fit 30 min" if ds == "icare" else "no consistent reading", ha="right", va="center",
                        fontsize=7, style="italic", color=MUTED)
        ax = axes[3]
        for ds, yo, mk, filled in (("clin", -0.13, "o", True), ("doc", 0.13, "s", False)):
            d = rows[p][ds]
            if d["wn"] is None:
                continue
            _errdot(ax, 1, "D", p, ds, y0 + yo, d["wn"], mk, VERM, filled, ms=4.4)
        if rows[p]["clin"]["wn"] is None and p != SEPARATE:
            ax.text(47.5, y0 - 0.13, "no consistent reading", ha="right", va="center",
                    fontsize=7, style="italic", color=MUTED)
    nom = load("nominal_rates.json")["pipes"]
    for p in COMMAND:
        v = nom.get(p)
        if not v:
            continue
        ref = 100 * (v["implied_rate"] if v.get("implied_rate") is not None else v["stated_rate"])
        for j, a in enumerate(axes):
            ds = ("clin", "doc", "icare", None)[j]
            if (ds and not rows[p][ds]["present"]) or (ds is None and rows[p]["clin"]["wn"] is None
                                                       and rows[p]["doc"]["wn"] is None):
                continue
            a.plot([ref], [ypos[p]], marker="|", ms=14, mew=1.5, color=INK, ls="none", zorder=4)
            put(1, "ABCD"[j], p, "reference", x=ref, y=ypos[p], ms=14)
    ax = axes[0]
    ax.set_ylim(ypos[SEPARATE] + 0.6, -0.6)
    ax.set_yticks([ypos[p] for p in COMMAND + [SEPARATE]])
    labels = ax.set_yticklabels([NAMES[p] for p in COMMAND + [SEPARATE]], fontsize=7.5)
    rep = load("replication.json") if (TAB / "replication.json").exists() else None
    conf = rep["tests"]["confirmatory"] if rep else {}
    for lab, p in zip(labels, COMMAND + [SEPARATE]):
        bold = p in conf and conf[p]["holm_p"] < 0.05
        if bold:
            lab.set_fontweight("bold")
        put(1, "label", p, "bold", value=bold)
    ax.tick_params(axis="y", length=0, pad=4)
    for a in axes[1:]:
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
    h1 = [mlines.Line2D([], [], marker="o", ls="none", ms=4.6, mfc=VERM, mec=VERM, mew=1.0,
                        label="Primary reading"),
          mlines.Line2D([], [], marker="^", ls="none", ms=4.6, mfc=GREEN, mec=GREEN, mew=1.0,
                        label="Randomization-based"),
          mlines.Line2D([], [], color="#9CA3AF", lw=0.9, ls=(0, (4, 3)), label="5% benchmark"),
          mlines.Line2D([], [], color=INK, marker="|", ms=14, mew=1.5, ls="none",
                        label="Rate the rule states or implies")]
    h1.insert(1, mlines.Line2D([], [], marker="D", ls="none", ms=4.6, mfc=INK, mec=INK, mew=1.0,
                               label="Same reading, replication set"))
    h1.insert(3, mlines.Line2D([], [], marker="s", ls="none", ms=4.4, mfc="white", mec=VERM,
                               mew=1.0, label="Primary reading, DoC pick"))
    fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.27, 0.0), ncol=3, frameon=False,
               handletextpad=0.3, columnspacing=1.4, fontsize=7.5)
    fig.text(0.005, 0.0, "Bold label: above 5% in the replication set\n(Holm-adjusted over 3 analyses).",
             fontsize=7, style="italic", color=MUTED, ha="left", va="bottom")
    _save(fig, "Figure1")


def fig2_data():
    sr = load("stated_rule.json")
    pp = sr["per_patient_at_published_k"]
    out = {"mbvt": {}, "dubois": {}}
    for ds, null in (("clin", "bdsp"), ("doc", "pdoc")):
        rows = []
        for r in pp[f"nstim0_{null}|mbvt"]["best_of_k_descriptive"]:
            if r["k"] <= 7 and r["n"] > 0:
                k = int(round(r["rate"] * r["n"]))
                assert abs(k - r["rate"] * r["n"]) < 1e-6
                rows.append((r["k"], k, r["n"], r["reading"]))
        out["mbvt"][ds] = rows
        e = sr["entries"][f"ncmd_{null}|dubois"]["stated_rule"]
        c = pp[f"ncmd_{null}|dubois"]["composite"]
        out["dubois"][ds] = {1: (e["k"], e["n"]), 2: (c["k"], c["n"])}
    comp = pp["nstim0_bdsp|mbvt"]["composite"]
    out["tworun"] = {"clin": (comp["k"], comp["n"]),
                     "doc": (pp["nstim0_pdoc|mbvt"]["composite"]["k"],
                             pp["nstim0_pdoc|mbvt"]["composite"]["n"])}
    fc = load("fixed_cohort_runs.json")
    out["fixed"] = [(r["runs"], r["k"], r["n"]) for r in fc["fixed_cohort"]]
    return out


def make_fig2():
    d = fig2_data()
    fig = plt.figure(figsize=(7.2, 3.4))
    axA = fig.add_axes([0.08, 0.14, 0.50, 0.80])
    col = {"clin": BLUE, "doc": VERM}
    axA.axhline(5, color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for ds in ("clin", "doc"):
        rows = d["mbvt"][ds]
        ks = np.array([r[0] for r in rows], float)
        rate = np.array([100 * r[1] / r[2] for r in rows])
        ci = np.array([cp(r[1], r[2]) for r in rows]) * 100
        axA.fill_between(ks, ci[:, 0], ci[:, 1], color=col[ds], alpha=0.16, lw=0, zorder=1)
        axA.plot(ks, rate, color=col[ds], lw=1.3, marker="o", ms=3.4, zorder=3)
        for r in rows:
            put(2, "A", "mbvt", ds, k=r[1], n=r[2], x=r[0], y=100 * r[1] / r[2])
    fx = [r for r in d["fixed"] if r[0] in (1, 2, 3, 5, 7)]
    axA.plot([r[0] for r in fx], [100 * r[1] / r[2] for r in fx], color=BLUE, lw=1.1, ls=(0, (3, 2)),
             marker="o", ms=3.0, mfc="white", mec=BLUE, zorder=3)
    for r in fx:
        put(2, "A", "mbvt_fixed", "clin", k=r[1], n=r[2], x=r[0], y=100 * r[1] / r[2])
    for ds, dx in (("clin", -0.3), ("doc", 0.3)):
        for kk in (1, 2):
            k, n = d["dubois"][ds][kk]
            lo, hi = cp(k, n)
            x = kk + dx
            axA.plot([x, x], [100 * lo, 100 * hi], color=col[ds], lw=0.9, zorder=4)
            axA.plot([x], [100 * k / n], marker="D", ms=4.6, mfc="white", mec=col[ds], mew=1.1,
                     ls="none", zorder=5)
            put(2, "A", "dubois", f"{ds}_k{kk}", k=k, n=n, x=x, y=100 * k / n)
    for ds, dx in (("clin", -0.1), ("doc", 0.1)):
        k, n = d["tworun"][ds]
        lo, hi = cp(k, n)
        x = 2 + dx
        axA.plot([x, x], [100 * lo, 100 * hi], color=col[ds], lw=0.9, zorder=4)
        axA.plot([x], [100 * k / n], marker="s", ms=4.4, mfc=col[ds], mec="white", mew=0.6,
                 ls="none", zorder=5)
        put(2, "A", "mbvt_two_run", ds, k=k, n=n, x=x, y=100 * k / n)
    axA.set_xlim(0.5, 7.5)
    axA.set_ylim(0, 100)
    axA.set_xticks(range(1, 8))
    axA.set_xlabel("Runs per patient, k")
    axA.set_ylabel("Patients with at least one positive run, %")
    axA.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axA.set_axisbelow(True)
    hA = [mlines.Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=3.4,
                        label="Vibrotactile P300, clinical EEG\n(patients with at least k runs)"),
          mlines.Line2D([], [], color=BLUE, lw=1.1, ls=(0, (3, 2)), marker="o", ms=3.0, mfc="white",
                        label="Vibrotactile P300, clinical EEG\n(fixed cohort of 639 with 7 runs)"),
          mlines.Line2D([], [], color=VERM, lw=1.3, marker="o", ms=3.4,
                        label="Vibrotactile P300, prolonged DoC EEG"),
          mlines.Line2D([], [], color=MUTED, lw=0, marker="s", ms=4.4, mfc=MUTED,
                        label="Vibrotactile P300, published two-run rule"),
          mlines.Line2D([], [], color=MUTED, lw=0, marker="D", ms=4.6, mfc="white", mew=1.1,
                        label="du Bois, one session and two-session criterion"),
          mlines.Line2D([], [], color="#9CA3AF", lw=0.8, ls=(0, (4, 3)), label="5% benchmark")]
    fig.legend(handles=hA, loc="center left", bbox_to_anchor=(0.61, 0.55), ncol=1, frameon=False,
               handletextpad=0.5, labelspacing=0.9, fontsize=7.5)
    _save(fig, "Figure2")


SHORT3 = {"columbia": "Columbia", "cornell": "Cornell", "edlow17": "Edlow 2017", "dubois": "du Bois"}


def fig3_data():
    st = load("positive_control_stated.json")["variants"]
    os_ = load("positive_control_onesource.json")["cells"]
    pts = [
        ("Cornell, primary reading", "cornell", "stated", st["cornell"]["c3_jackknife_bh_keep40_r4_om2contig"]),
        ("du Bois, primary reading", "dubois", "stated", st["dubois"]["c3_b_assessment_blocked_phase1"]),
        ("Edlow 2017, leave-block-out", "edlow17", "stated", st["edlow17"]["permsecond_cvblock_alt"]),
        ("du Bois, leave-block-out permutation", "dubois", "stated", st["dubois"]["assessment_blocked_perm"]),
    ]
    out = []
    for lab, pipe, kind, v in pts:
        out.append((lab, pipe, kind, (v["true"]["k"], v["true"]["n"]), (v["control"]["k"], v["control"]["n"])))
    for pipe, key in (("columbia", "columbia|randomized|as_coded"),
                      ("cornell", "cornell|randomized|theoretical_bh_keep40_r4"),
                      ("edlow17", "edlow17|randomized|permsecond_cvsecond_alt"),
                      ("dubois", "dubois|randomized|assessment_random_phase1")):
        t, c = os_[f"openbmi_onesource|{key}|true"], os_[f"openbmi_onesource|{key}|control"]
        out.append((f"{SHORT3[pipe]}, randomization-based", pipe, "random",
                    (t["k"], t["n"]), (c["k"], c["n"])))
    return out


def make_fig3():
    pts = fig3_data()
    colors = {"cornell": VERM, "dubois": BLUE, "edlow17": GREEN, "columbia": PURPLE}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.17)
    ax.hlines(5, 60, 100, color="#9CA3AF", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.hlines([10, 20, 30, 40], 60, 100, color=GRID, lw=0.6, zorder=0)
    ax.vlines(range(60, 101, 5), -2, 47, color=GRID, lw=0.6, zorder=0)
    pos = {
        "Cornell, primary reading": ("dataR", 98.6, 24.9, False),
        "du Bois, primary reading": ("pt", -6, 8, "right"),
        "Edlow 2017, leave-block-out": ("col", 103.0, 3.5),
        "du Bois, leave-block-out permutation": ("dataR", 79.0, 8.4, True),
        "Columbia, randomization-based": ("col", 103.0, -1.4),
        "Cornell, randomization-based": ("col", 103.0, 11.0),
        "Edlow 2017, randomization-based": ("col", 103.0, 7.2),
        "du Bois, randomization-based": ("dataR", 87.4, 11.2, True),
    }
    for lab, pipe, kind, (tk, tn), (ck, cn) in pts:
        x, y = 100 * tk / tn, 100 * ck / cn
        xl, xh = (100 * v for v in cp(tk, tn))
        yl, yh = (100 * v for v in cp(ck, cn))
        c = colors[pipe]
        ax.plot([xl, xh], [y, y], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
        ax.plot([x, x], [yl, yh], color=c, lw=0.9, zorder=2, solid_capstyle="butt")
        if kind == "stated":
            ax.plot([x], [y], marker="o", ms=6, mfc=c, mec="white", mew=0.7, ls="none", zorder=4)
        else:
            ax.plot([x], [y], marker="^", ms=6.5, mfc="white", mec=c, mew=1.2, ls="none", zorder=4)
        sp = pos[lab]
        if sp[0] == "pt":
            ax.annotate(lab, (x, y), xytext=(sp[1], sp[2]), textcoords="offset points",
                        ha=sp[3], va="center", fontsize=7.5, color=INK)
        elif sp[0] == "dataR":
            ax.annotate(lab, (x, y), xytext=(sp[1], sp[2]), textcoords="data", ha="right",
                        va="center", fontsize=7.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.6, shrinkA=1,
                                        shrinkB=4, relpos=(1, 0.5)) if sp[3] else None)
        else:
            ax.annotate(lab, (x, y), xytext=(sp[1], sp[2]), textcoords="data", ha="left",
                        va="center", fontsize=7.5, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.6, shrinkA=1,
                                        shrinkB=4, relpos=(0, 0.5)))
        put(3, "-", lab, "detection", k=tk, n=tn, x=x, y=y)
        put(3, "-", lab, "control", k=ck, n=cn, x=x, y=y)
    ax.set_xlim(60, 128)
    ax.set_ylim(-2, 47)
    ax.set_xticks(range(60, 101, 5))
    ax.spines["bottom"].set_bounds(60, 100)
    ax.xaxis.set_label_coords(0.29, -0.11)
    ax.set_xlabel("Detection of true responders, %")
    ax.set_ylabel("Calls on label-shuffled sessions, %")
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    h = [mlines.Line2D([], [], marker="o", ls="none", ms=6, mfc=MUTED, mec="white",
                       label="Stated-rule reading"),
         mlines.Line2D([], [], marker="^", ls="none", ms=6.5, mfc="white", mec=MUTED, mew=1.2,
                       label="Randomization-based reading"),
         mlines.Line2D([], [], color="#9CA3AF", lw=0.9, ls=(0, (4, 3)), label="5% benchmark")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.8, fontsize=7.5)
    _save(fig, "Figure3")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    PUT.clear()
    make_fig1()
    make_fig2()
    make_fig3()
    return PUT


if __name__ == "__main__":
    main()
    print(f"wrote {len(PUT)} plotted values to {OUT}")
