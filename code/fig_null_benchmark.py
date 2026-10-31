"""Null-benchmark manuscript Figures 1-3.

Reads only aggregate JSON tables (no patient rows), recomputes every error bar as an exact
Clopper-Pearson 95% CI from k/n, and writes output/figures/null_benchmark/Figure{1,2,3}.{pdf,png}.
Style: the house style of fig_style.py (17 cm, Arial 7 pt and above, Okabe-Ito semantic colours, 600 DPI).

Every plotted number is routed through `put()` so the test can compare it to the JSON.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "output" / "tables"
OUT = ROOT / "output" / "figures" / "null_benchmark"

import fig_style as fs

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE, YELLOW = "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
INK, MUTED, GRID = fs.INK, fs.MUTED, fs.GRID

NAMES = dict(fs.NAMES)
COMMAND = ["cornell", "dubois", "mbvt", "columbia", "gibson16", "cruse12p", "mbmi", "cruse11",
           "edlow17", "chennu13"]
SEPARATE = "sokoliuk21"
NULLKIND = {p: "ncmd" for p in ["cornell", "dubois", "columbia", "cruse12p", "mbmi", "cruse11", "edlow17"]}
NULLKIND.update({p: "nstim0" for p in ["mbvt", "gibson16", "chennu13", "sokoliuk21"]})

PUT = []
GREY = []


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
    ax.plot([x], [y], marker=marker, ms=ms, mfc=color if filled else "white", mec="white" if filled else color,
            mew=0.6 if filled else 1.0, ls="none", zorder=3)
    put(fig, panel, row, series, k=k, n=n, x=x, y=y, ci=ci)


HEAT_COLS = [("stated", "Stated"), ("A", "Clinical"), ("rep", "Replic-\nation"), ("B", "Prolonged\nDoC"),
             ("C", "Post-\narrest"), ("Dclin", "Clinical\npick"), ("Ddoc", "DoC\npick")]
RAND_COLS = [("A", "Clinical"), ("B", "Prolonged\nDoC"), ("C", "Post-\narrest")]
VMAX = 40.0


def _row_order():
    return [p for _, ps, _ in fs.FAMILY for p in ps]


def long_cells():
    """{(pipe, column): (k, n)} of the cells run later. The full-length schedules on the recordings that hold them
    (178-180): Chennu 2013 on clinical EEG and its white-noise rate, Cruse 2011 / 2012 and Chennu 2013 in the
    replication set and after arrest, Sokoliuk 2021 in the replication set. Claassen 2019 on the six DoC sites (181):
    its published and randomized (column "rB") DoC rates, and the white-noise rate of its one reading."""
    out = {}
    if (TAB / "claassen_six.json").exists():
        doc = load("claassen_six.json")["doc"]
        out[("columbia", "B")] = (doc["published"]["k"], doc["published"]["n"])
        out[("columbia", "rB")] = (doc["randomized"]["k"], doc["randomized"]["n"])
        w = load("white_noise_eligible.json")["variants"]["columbia"]["as_coded:call_paper"]
        out[("columbia", "Ddoc")] = (w["k"], w["n"])
    if not (TAB / "long_schedule.json").exists():
        return out
    ls = load("long_schedule.json")
    dev = load("long_schedule_dev.json")["chennu13"]["stated_rule"]
    out[("chennu13", "A")] = (dev["k"], dev["n"])
    w = load("white_noise_eligible.json")["variants"]["chennu13"][ls["frozen"]["chennu13"]["variant"]]
    out[("chennu13", "Dclin")] = (w["k"], w["n"])
    for key, c in ls["cells"].items():
        cohort, pipe = key.split("|")
        if c.get("n"):
            out[(pipe, "rep" if cohort == "replication" else "C")] = (c["k"], c["n"])
    return out


def _heat_cell(ax, j, i, kn, put_args, na=None, outline=False, later=False):
    """One heatmap cell: viridis by rate with the value printed; light grey with a short reason when absent; later:
    a dagger for a full-length schedule run later on the recordings that hold it."""
    if kn is None:
        GREY.append((j, i, na))
        ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#ECEEF1", lw=0, zorder=1))
        if na:
            ax.text(j, i, na, ha="center", va="center", fontsize=7, color=MUTED, zorder=3)
        return
    k, n = kn
    v = 100 * k / n
    ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=matplotlib.colormaps["viridis"](min(v, VMAX) / VMAX),
                                    lw=0, zorder=1))
    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=fs.text_on(min(v, VMAX), 0, VMAX), zorder=3)
    if outline:
        ax.add_patch(mpatches.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec="white", lw=1.6, zorder=4))
    if later:
        inset = 0.38 if outline else 0.44
        ax.text(j + inset, i - inset + 0.02, "†", ha="right", va="top", fontsize=7,
                color=fs.text_on(min(v, VMAX), 0, VMAX), zorder=5)
    fig_, panel, row, series = put_args
    put(fig_, panel, row, series, k=k, n=n, x=v, y=i, ci=False)


def _heat_frame(ax, ncol, nrow, labels, top_groups=None):
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    for x in range(1, ncol):
        ax.axvline(x - 0.5, color="white", lw=1.0, zorder=2)
    for y in range(1, nrow):
        ax.axhline(y - 0.5, color="white", lw=1.0, zorder=2)
    ax.set_xticks(range(ncol), labels, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for lab, a, b in (top_groups or []):
        ax.plot([a - 0.42, b + 0.42], [-1.55, -1.55], color=INK, lw=0.8, clip_on=False)
        ax.text((a + b) / 2, -1.68, lab, ha="center", va="bottom", fontsize=8, fontweight="bold", clip_on=False)


def make_fig1():
    rows = fig1_data()
    order = _row_order()
    late = long_cells()
    nom = load("nominal_rates.json")["pipes"]
    rep = load("replication.json") if (TAB / "replication.json").exists() else None
    conf = rep["tests"]["confirmatory"] if rep else {}
    fig = plt.figure(figsize=(fs.WIDTH_IN, 6.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[7, 3], height_ratios=[3.25, 1.0], left=0.225, right=0.985,
                          top=0.87, bottom=0.115, wspace=0.08, hspace=0.42)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    sub = gs[1, :].subgridspec(1, 3, wspace=0.12)
    axC = [fig.add_subplot(sub[0, j]) for j in range(3)]

    for i, p in enumerate(order):
        r = rows[p]
        v = nom.get(p)
        if v:
            ref = v["implied_rate"] if v.get("implied_rate") is not None else v["stated_rate"]
            axA.add_patch(mpatches.Rectangle((-0.5, i - 0.5), 1, 1, color="#DDE1E6", lw=0, zorder=1))
            axA.text(0, i, f"{100 * ref:.0f}" if ref > 0.1 else f"{100 * ref:g}", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK,
                     zorder=3)
            put(1, "stated", p, "reference", x=100 * ref, y=i)
        else:
            _heat_cell(axA, 0, i, None, None, "–")
        for j, (code, _) in enumerate(HEAT_COLS[1:], start=1):
            if code in ("A", "B", "C"):
                ds = {"A": "clin", "B": "doc", "C": "icare"}[code]
                d = r[ds]
                if not d["present"] and (p, code) not in late:
                    _heat_cell(axA, j, i, None, None, "not run")
                    continue
                kn, later = d["primary"], False
                if kn is None and (p, code) in late:
                    kn, later = late[(p, code)], True
                sig = kn is not None and cp(*kn)[0] > 0.05
                _heat_cell(axA, j, i, kn, (1, code, p, "primary_later" if later else "primary"),
                           None if kn else "too long", outline=sig, later=later)
            elif code == "rep":
                kn, later = r["clin"]["replication"], False
                if kn is None and (p, "rep") in late:
                    kn, later = late[(p, "rep")], True
                sig = kn is not None and cp(*kn)[0] > 0.05
                _heat_cell(axA, j, i, kn, (1, "A", p, "replication_later" if later else "replication"),
                           "too long" if p in ("cruse11", "cruse12p") else "–", outline=sig, later=later)
            else:
                ds = "clin" if code == "Dclin" else "doc"
                kn, later = r[ds]["wn"], False
                if kn is None and (p, code) in late:
                    kn, later = late[(p, code)], True
                _heat_cell(axA, j, i, kn, (1, "D", p, ds + "_later" if later else ds),
                           "not run" if not r[ds]["present"] else "too long", later=later)
    _heat_frame(axA, len(HEAT_COLS), len(order), [lab for _, lab in HEAT_COLS],
                [("Rule", 0, 0), ("EEG without commands", 1, 4), ("White noise", 5, 6)])
    axA.set_yticks(range(len(order)), [NAMES[p] for p in order], fontsize=8)
    for lab, p in zip(axA.get_yticklabels(), order):
        bold = p in conf and conf[p]["holm_p"] < 0.05
        lab.set_fontweight("bold" if bold else "normal")
        put(1, "label", p, "bold", value=bold)
    y0 = 0
    for name, ps, col in fs.FAMILY:
        y1 = y0 + len(ps) - 1
        axA.plot([-2.75, -2.75], [y0 - 0.38, y1 + 0.38], color=col, lw=3.2, clip_on=False, solid_capstyle="butt")
        short = {"Event-related potential": "ERP", "Covert processing": "Covert"}.get(name, name)
        if len(ps) > 1:
            axA.text(-3.0, (y0 + y1) / 2, short, rotation=90, ha="center", va="center", fontsize=7.5, color=col,
                     fontweight="bold", clip_on=False)
        else:
            axA.text(-2.9, y0, short, ha="right", va="center", fontsize=7.5, color=col, fontweight="bold",
                     clip_on=False)
        if y0:
            for ax in (axA, axB):
                ax.axhline(y0 - 0.5, color=INK, lw=1.0, zorder=5)
        y0 = y1 + 1
    fs.panel(axA, "A", "False-call rate of the published rule, %", x=-0.03, y=1.2)

    for i, p in enumerate(order):
        for j, (code, _) in enumerate(RAND_COLS):
            ds = {"A": "clin", "B": "doc", "C": "icare"}[code]
            d = rows[p][ds]
            kn, later = d["random"] if d["present"] else None, False
            if kn is None and (p, "r" + code) in late:
                kn, later = late[(p, "r" + code)], True
            _heat_cell(axB, j, i, kn, (1, code, p, "random_later" if later else "random"),
                       "not run" if not d["present"] else "–", later=later)
    _heat_frame(axB, len(RAND_COLS), len(order), [lab for _, lab in RAND_COLS],
                [("EEG without commands", 0, 2)])
    axB.set_yticks([])
    fs.panel(axB, "B", "Randomization-based, %", x=-0.03, y=1.2)

    cax = fig.add_axes([0.43, 0.355, 0.36, 0.012])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, VMAX), cmap="viridis"),
                      cax=cax, orientation="horizontal", ticks=[0, 5, 10, 20, 30, 40])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.set_label("False-call rate, %  (white frame: 95% CI above 5%)", fontsize=7.5)

    for j, p in enumerate(["cornell", "dubois", "mbvt"]):
        ax = axC[j]
        r = rows[p]
        items = [("Clinical", r["clin"]["primary"], PUBLISHED_MARK), ("Replication", r["clin"]["replication"], REPL_MARK),
                 ("Post-arrest", r["icare"]["primary"], PUBLISHED_MARK)]
        for y, (lab, kn, (mk, col)) in enumerate(items):
            lo, hi = cp(*kn)
            fs.errpoint(ax, y, 100 * kn[0] / kn[1], 100 * lo, 100 * hi, col, mk, horizontal=True)
            put("1F", p, lab, "ci", k=kn[0], n=kn[1], x=100 * kn[0] / kn[1], y=y, ci=True)
            ax.text(100 * hi + 1.2, y, f"{100 * kn[0] / kn[1]:.1f}", ha="left", va="center", fontsize=7.5, color=col)
        for y, ds in ((0, "clin"), (2, "icare")):
            k, n = r[ds]["random"]
            ax.plot([100 * k / n], [y + 0.32], marker="^", ms=5, mfc=fs.RANDOMIZED, mec="white", mew=0.5, ls="none")
        v = nom[p]
        ref = 100 * (v["implied_rate"] if v.get("implied_rate") is not None else v["stated_rate"])
        ax.plot([ref, ref], [-0.5, 2.5], color=INK, lw=1.4, zorder=2)
        put("1F", p, "stated", "reference", x=ref)
        fs.bench(ax, axis="x")
        ax.set_ylim(2.6, -0.6)
        ax.set_xlim(0, 42)
        ax.set_xticks([0, 5, 10, 20, 30, 40])
        ax.set_yticks(range(3), [lab for lab, _, _ in items] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        fs.grid(ax, "x")
        ax.set_title(NAMES[p], fontsize=8.5, fontweight="bold", loc="left", pad=3)

    fs.panel(axC[0], "C", None, x=-0.36, y=1.08)
    h = [mlines.Line2D([], [], marker="o", ls="none", ms=5.5, mfc=PUBLISHED_MARK[1], mec="white", label="Published rule"),
         mlines.Line2D([], [], marker="D", ls="none", ms=5, mfc=REPL_MARK[1], mec="white", label="Same rule, new patients"),
         mlines.Line2D([], [], marker="^", ls="none", ms=5.5, mfc=fs.RANDOMIZED, mec="white", label="Randomization-based"),
         mlines.Line2D([], [], color=INK, lw=1.4, label="Rate the rule states or implies"),
         mlines.Line2D([], [], color=fs.BENCH, lw=0.8, ls=(0, (4, 3)), label="5% benchmark")]
    fig.text(0.6, 0.06, "False-call rate, % (exact 95% CI)", ha="center", va="center", fontsize=8.5)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.56, -0.012), ncol=5, fontsize=7.5,
               handletextpad=0.3, columnspacing=1.0)
    _save(fig, "Figure1")


PUBLISHED_MARK, REPL_MARK = ("o", fs.PUBLISHED), ("D", INK)


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
    fig = plt.figure(figsize=(fs.WIDTH_IN, 3.0))
    axA = fig.add_axes([0.085, 0.155, 0.905, 0.815])
    col = {"clin": BLUE, "doc": VERM}
    axA.axhline(5, color=fs.BENCH, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for ds in ("doc", "clin"):
        rows = d["mbvt"][ds]
        ks = np.array([r[0] for r in rows], float)
        rate = np.array([100 * r[1] / r[2] for r in rows])
        ci = np.array([cp(r[1], r[2]) for r in rows]) * 100
        axA.fill_between(ks, ci[:, 0], ci[:, 1], color=col[ds], alpha=0.10 if ds == "doc" else 0.22, lw=0,
                         zorder=1 if ds == "doc" else 2)
        axA.plot(ks, rate, color=col[ds], lw=2.2, marker="o", ms=5, mec="white", mew=0.6, zorder=3)
        axA.text(ks[-1] + 0.12, rate[-1] + (2.5 if ds == "clin" else -3.0), f"{rate[-1]:.1f}", fontsize=8,
                 color=col[ds], va="center", fontweight="bold")
        for r in rows:
            put(2, "A", "mbvt", ds, k=r[1], n=r[2], x=r[0], y=100 * r[1] / r[2])
    fx = [r for r in d["fixed"] if r[0] in (1, 2, 3, 5, 7)]
    axA.plot([r[0] for r in fx], [100 * r[1] / r[2] for r in fx], color=BLUE, lw=1.4, ls=(0, (3, 2)),
             marker="o", ms=4.0, mfc="white", mec=BLUE, zorder=3)
    for r in fx:
        put(2, "A", "mbvt_fixed", "clin", k=r[1], n=r[2], x=r[0], y=100 * r[1] / r[2])
    for ds, dx in (("clin", -0.3), ("doc", 0.3)):
        for kk in (1, 2):
            k, n = d["dubois"][ds][kk]
            lo, hi = cp(k, n)
            x = kk + dx
            axA.plot([x, x], [100 * lo, 100 * hi], color=col[ds], lw=1.3, zorder=4)
            axA.plot([x], [100 * k / n], marker="D", ms=6, mfc="white", mec=col[ds], mew=1.4,
                     ls="none", zorder=5)
            put(2, "A", "dubois", f"{ds}_k{kk}", k=k, n=n, x=x, y=100 * k / n)
    for ds, dx in (("clin", -0.1), ("doc", 0.1)):
        k, n = d["tworun"][ds]
        lo, hi = cp(k, n)
        x = 2 + dx
        axA.plot([x, x], [100 * lo, 100 * hi], color=col[ds], lw=1.3, zorder=4)
        axA.plot([x], [100 * k / n], marker="s", ms=6, mfc=col[ds], mec="white", mew=0.6,
                 ls="none", zorder=5)
        put(2, "A", "mbvt_two_run", ds, k=k, n=n, x=x, y=100 * k / n)
    axA.set_xlim(0.5, 7.5)
    axA.set_ylim(0, 100)
    axA.set_xticks(range(1, 8))
    axA.set_xlabel("Runs per patient, k")
    axA.set_ylabel("Patients with at least one positive run, %")
    fs.grid(axA)
    hA = [mlines.Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=5,
                        label="Guger 2018, clinical EEG (patients with at least k runs)"),
          mlines.Line2D([], [], color=BLUE, lw=1.4, ls=(0, (3, 2)), marker="o", ms=4, mfc="white",
                        label="Guger 2018, clinical EEG (fixed cohort of 639 with 7 runs)"),
          mlines.Line2D([], [], color=VERM, lw=2.2, marker="o", ms=5,
                        label="Guger 2018, prolonged DoC EEG"),
          mlines.Line2D([], [], color=MUTED, lw=0, marker="s", ms=6, mfc=MUTED,
                        label="Guger 2018, published two-run rule"),
          mlines.Line2D([], [], color=MUTED, lw=0, marker="D", ms=6, mfc="white", mew=1.4,
                        label="du Bois 2026, one and two sessions"),
          mlines.Line2D([], [], color=fs.BENCH, lw=0.8, ls=(0, (4, 3)), label="5% benchmark")]
    axA.legend(handles=hA, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False,
               handletextpad=0.5, labelspacing=0.45, columnspacing=1.4, fontsize=8)
    _save(fig, "Figure2")


SHORT3 = {p: fs.NAMES[p] for p in ("columbia", "cornell", "edlow17", "dubois")}


def fig3_data():
    st = load("positive_control_stated.json")["variants"]
    os_ = load("positive_control_onesource.json")["cells"]
    pts = [
        ("Goldfine 2011, primary reading", "cornell", "stated", st["cornell"]["c3_jackknife_bh_keep40_r4_om2contig"]),
        ("du Bois 2026, primary reading", "dubois", "stated", st["dubois"]["c3_b_assessment_blocked_phase1"]),
        ("Edlow 2017, leave-block-out", "edlow17", "stated", st["edlow17"]["permsecond_cvblock_alt"]),
        ("du Bois 2026, leave-block-out permutation", "dubois", "stated", st["dubois"]["assessment_blocked_perm"]),
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


S1_GROUPS = [("cornell", ["Goldfine 2011, primary reading", "Goldfine 2011, randomization-based"]),
             ("dubois", ["du Bois 2026, primary reading", "du Bois 2026, leave-block-out permutation",
                         "du Bois 2026, randomization-based"]),
             ("edlow17", ["Edlow 2017, leave-block-out", "Edlow 2017, randomization-based"]),
             ("columbia", ["Claassen 2019, randomization-based"])]


def make_fig3():
    """Supplementary Figure S1: motor imagery positive control. One row per reading, grouped by analysis; detection of
    true responders (A) beside calls on label-shuffled sessions (B), exact 95% CIs, value printed on each point."""
    pts = {lab: (pipe, kind, t, c) for lab, pipe, kind, t, c in fig3_data()}
    rows = [lab for _, labs in S1_GROUPS for lab in labs]
    fig = plt.figure(figsize=(fs.WIDTH_IN, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.3, right=0.985, top=0.86, bottom=0.2, wspace=0.12)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    style = {"stated": ("o", fs.PUBLISHED), "random": ("^", fs.RANDOMIZED)}
    for y, lab in enumerate(rows):
        pipe, kind, (tk, tn), (ck, cn) = pts[lab]
        mk, col = style[kind]
        for ax, (k, n), series in ((axA, (tk, tn), "detection"), (axB, (ck, cn), "control")):
            lo, hi = (100 * v for v in cp(k, n))
            v = 100 * k / n
            fs.errpoint(ax, y, v, lo, hi, col, mk, ms=6.5, horizontal=True)
            ax.text(hi + (1.2 if ax is axA else 1.0), y, f"{v:.1f}", ha="left", va="center", fontsize=8, color=col,
                    fontweight="bold")
            put(3, "-", lab, series, k=k, n=n, x=100 * tk / tn, y=100 * ck / cn)
    reading = {lab: {"leave-block-out permutation": "Block-wise permutation"}.get(lab.split(", ", 1)[1],
                                                                            lab.split(", ", 1)[1].capitalize())
               for lab in rows}
    for ax in (axA, axB):
        ax.set_ylim(len(rows) - 0.45, -0.75)
        ax.tick_params(axis="y", length=0)
        fs.grid(ax, "x")
    axA.set_yticks(range(len(rows)), [reading[lab] for lab in rows])
    axB.set_yticks(range(len(rows)), [])
    y0 = 0
    for gi, (pipe, labs) in enumerate(S1_GROUPS):
        y1 = y0 + len(labs) - 1
        if y0:
            for ax in (axA, axB):
                ax.axhline(y0 - 0.5, color="#D1D5DB", lw=0.8, zorder=0)
        axA.annotate(NAMES[pipe], xy=(-0.66, (y0 + y1) / 2), xycoords=("axes fraction", "data"), ha="right",
                     va="center", fontsize=8.5, fontweight="bold", annotation_clip=False)
        axA.plot([-0.63, -0.63], [y0 - 0.32, y1 + 0.32], color=INK, lw=1.6, clip_on=False,
                 transform=matplotlib.transforms.blended_transform_factory(axA.transAxes, axA.transData))
        y0 = y1 + 1
    axA.set_xlim(55, 109)
    axA.set_xticks(range(60, 101, 10))
    axA.set_xlabel("Detection of true responders, %")
    axB.set_xlim(-1, 51)
    axB.set_xticks([0, 5, 10, 20, 30, 40])
    fs.bench(axB, axis="x")
    axB.set_xlabel("Calls on label-shuffled sessions, %")
    fs.panel(axA, "A", "True labels", x=-0.02, y=1.04)
    fs.panel(axB, "B", "Shuffled labels, same sessions", x=-0.02, y=1.04)
    h = [mlines.Line2D([], [], marker="o", ls="none", ms=6, mfc=fs.PUBLISHED, mec="white", label="Stated-rule reading"),
         mlines.Line2D([], [], marker="^", ls="none", ms=6.5, mfc=fs.RANDOMIZED, mec="white",
                       label="Randomization-based reading"),
         mlines.Line2D([], [], color=fs.BENCH, lw=0.8, ls=(0, (4, 3)), label="5% benchmark")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.8, fontsize=8)
    _save(fig, "FigureS1")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fs.save(fig, OUT, name)


def main():
    PUT.clear()
    GREY.clear()
    make_fig1()
    make_fig2()
    make_fig3()
    return PUT


if __name__ == "__main__":
    main()
    print(f"wrote {len(PUT)} plotted values to {OUT}")
