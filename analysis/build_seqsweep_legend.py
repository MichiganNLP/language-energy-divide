"""Standalone legend for the batch-size sweep subfigures.

Renders ONLY the 8-language color key (same colors/markers as
build_seqsweep_figs.py) to a tightly-cropped PDF, so it can be placed once as a
shared legend above/below the L40S + RTX subfigures on Overleaf.

Layout: NCOL columns, column-major in matplotlib, but handles are pre-
transposed so the displayed grid reads row-major -> HIGH-res on the top row,
LOW-res on the bottom row (with the default NCOL=4).

Output: paper/paper-update-1/extracted/figures/fig_seqsweep_legend.crop.pdf
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(".")
OUT = REPO / "paper" / "paper-update-1" / "extracted" / "figures" / "fig_seqsweep_legend.pdf"

HIGH = ["eng_Latn", "zho_Hans", "rus_Cyrl", "fra_Latn"]
LOW = ["pbt_Arab", "tir_Ethi", "shn_Mymr", "bod_Tibt"]
LANGS = HIGH + LOW
COLOR = {
    "eng_Latn": "#08519c", "zho_Hans": "#3182bd", "rus_Cyrl": "#6baed6", "fra_Latn": "#9ecae1",
    "pbt_Arab": "#a50f15", "tir_Ethi": "#de2d26", "shn_Mymr": "#fb6a4a", "bod_Tibt": "#fdae6b",
}
NCOL = 4  # 4 -> two rows (HIGH on top, LOW on bottom); set 8 for a single row
# Readable name + writing system for each language; rendered with a mathtext
# subscript, e.g. English_(Latin).
LABEL = {
    "eng_Latn": ("English", "Latin"),
    "zho_Hans": ("Chinese", "Simplified"),
    "rus_Cyrl": ("Russian", "Cyrillic"),
    "fra_Latn": ("French", "Latin"),
    "pbt_Arab": ("Southern Pashto", "Arabic"),
    "tir_Ethi": ("Tigrinya", "Ethiopic/Ge’ez"),
    "shn_Mymr": ("Shan", "Myanmar/Burmese"),
    "bod_Tibt": ("Tibetan", "Tibetan"),
}


def fancy(code):
    name, script = LABEL[code]
    return f"{name}$_{{\\mathrm{{({script})}}}}$"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "legend.fontsize": 8,
    "pdf.fonttype": 42,
})


def col_major(items, ncol):
    """Reorder a row-major list so matplotlib's column-major legend displays it
    row-major (top row first)."""
    nrow = math.ceil(len(items) / ncol)
    out = []
    for c in range(ncol):
        for r in range(nrow):
            i = r * ncol + c
            if i < len(items):
                out.append(items[i])
    return out


def main():
    order = col_major(LANGS, NCOL)
    handles = [
        Line2D([], [], color=COLOR[L], marker="o", ms=4, lw=1.6,
               label=fancy(L))
        for L in order
    ]
    fig = plt.figure(figsize=(7.0, 0.6))
    leg = fig.legend(handles=handles, loc="center", ncol=NCOL, frameon=False,
                     handlelength=1.6, columnspacing=1.4, handletextpad=0.5)
    fig.canvas.draw()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # crop tightly to just the legend
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT, bbox_inches=bbox, pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
