"""House figure style for the null-benchmark manuscript (Figures 1-4, Supplementary Figure S1): one width, one type
scale, one semantic palette, one panel grammar, so the figures read as a set.

- Width 17 cm (Annals full page; Nature double column is 18.3 cm), Arial, text 7.5-9 pt at final size (never below
  the house minimum of 7), panel letters 12 pt bold upright with a 9-pt bold header beside them.
- Analyses are named only by first author and year (NAMES); magnitude heatmaps use viridis with the value printed in
  every cell (hochberg-figure-style), not-applicable cells are light grey with a short reason.
- Semantic colours (Okabe-Ito, colour-blind safe), used with the same meaning in every figure:
  PUBLISHED vermilion = the rule as published / primary reading; CORRECTED blue = the command-level variance or a
  clinical-EEG series; RANDOMIZED green = randomization-based or block-wise alternative; SHAM grey = sham or null
  labels; INK = replication or reference marks. Orange is a second series colour where two are needed.
- Hairline axes (left / bottom), outward 2.5-pt ticks, near-invisible grid, dashed 5% benchmark."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH_IN = 17 / 2.54
PUBLISHED, CORRECTED, RANDOMIZED = "#D55E00", "#0072B2", "#009E73"
ORANGE, SKY, PURPLE = "#E69F00", "#56B4E9", "#CC79A7"
SHAM, INK, MUTED, GRID, AXIS, BENCH = "#8C96A3", "#1F2933", "#5F6B7A", "#EEF0F3", "#6B7280", "#9CA3AF"
HILITE = "#F6F1EC"

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.direction": "out", "ytick.direction": "out",
    "lines.linewidth": 1.4, "errorbar.capsize": 0, "legend.frameon": False, "legend.handlelength": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
}
plt.rcParams.update(RC)


NAMES = {"columbia": "Claassen 2019", "cornell": "Goldfine 2011", "cruse11": "Cruse 2011", "cruse12p": "Cruse 2012",
         "edlow17": "Edlow 2017", "mbmi": "Guger 2017", "dubois": "du Bois 2026", "mbvt": "Guger 2018",
         "chennu13": "Chennu 2013", "gibson16": "Gibson 2016", "sokoliuk21": "Sokoliuk 2021"}
FAMILY = [("Motor command", ["cornell", "dubois", "columbia", "cruse12p", "mbmi", "cruse11", "edlow17"], "#2C5FAA"),
          ("Event-related potential", ["mbvt", "gibson16", "chennu13"], "#6A4C93"),
          ("Covert processing", ["sokoliuk21"], "#117A8B")]


def panel(ax, letter, header=None, x=-0.02, y=1.05):
    """Upright bold panel letter at the top left of the axes, with an optional short bold header beside it."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom",
            color=INK)
    if header:
        ax.text(x + 0.012, y + 0.006, header, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left",
                va="bottom", color=INK)


def text_on(value, vmin, vmax, cmap="viridis"):
    """White text on dark cells, near-black on light ones (relative luminance of the cell colour)."""
    import matplotlib as mpl
    r, g, b, _ = mpl.colormaps[cmap]((value - vmin) / ((vmax - vmin) or 1))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.5 else INK


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def bench(ax, value=5, axis="y", label=None):
    """The 5% benchmark: a dashed grey reference line."""
    kw = dict(color=BENCH, lw=0.8, ls=(0, (4, 3)), zorder=1, label=label)
    return ax.axhline(value, **kw) if axis == "y" else ax.axvline(value, **kw)


def errpoint(ax, x, est, lo, hi, color, marker="o", filled=True, ms=5.6, horizontal=False, zorder=4):
    """A point estimate with a hairline 95% CI; white-edged filled markers or hollow markers."""
    if horizontal:
        ax.plot([lo, hi], [x, x], color=color, lw=1.4, solid_capstyle="butt", zorder=zorder - 1)
        y, xx = x, est
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=1.4, solid_capstyle="butt", zorder=zorder - 1)
        y, xx = est, x
    ax.plot([xx], [y], marker=marker, ms=ms, ls="none", zorder=zorder,
            mfc=color if filled else "white", mec="white" if filled else color, mew=0.6 if filled else 1.0)


def save(fig, out_dir, name, pad=0.04):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
