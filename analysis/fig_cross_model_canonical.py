"""Cross-model figure on the 12-language stratified subset (canonical Belebele).

Same 12 languages used for the cross-GPU and cross-dataset experiments.
Grouped bars: 12 languages (grouped LOW/MID/HIGH) x 3 single-GPU models.
Qwen3-14B on bod_Tibt is right-censored (run hit the 6h cap before steady
state); we draw its whole-run lower bound as a hatched bar.

Writes paper/figures/fig_tier_heatmap.pdf (+ .crop.pdf) to keep the existing
\\label/\\ref and \\includegraphics path unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("./"
           "paper/figures")
SRC = Path("./"
           "research/lmarena_translate")

SUB = ["eng_Latn", "spa_Latn", "zho_Hans", "vie_Latn",      # LOW
       "rus_Cyrl", "hin_Deva", "tha_Thai", "fra_Latn",      # MID
       "pbt_Arab", "tir_Ethi", "shn_Mymr", "bod_Tibt"]      # HIGH
# Qwen3-14B bod_Tibt: whole-run J/tok over the completed fraction (censored).
CENSORED = {("Qwen3-14B", "bod_Tibt"): 2.90}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42,
})


def load(label):
    d = pd.read_csv(SRC / f"belebele_canonical_{label}.csv")
    d = d[d.steady_state_duration > 1.0]
    return d.set_index("language").steady_energy_per_token_J


def main():
    models = {"Qwen3-8B": load("Qwen3-8B"),
              "Qwen3-14B": load("Qwen3-14B"),
              "Llama-3.1-8B": load("Llama-3.1-8B")}
    colors = {"Qwen3-8B": "#4c72b0", "Qwen3-14B": "#dd8452",
              "Llama-3.1-8B": "#55a868"}

    x = np.arange(len(SUB))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.1))

    for i, (mname, s) in enumerate(models.items()):
        vals, hatched = [], []
        for L in SUB:
            if (mname, L) in CENSORED:
                vals.append(CENSORED[(mname, L)])
                hatched.append(True)
            elif L in s.index:
                vals.append(s.loc[L])
                hatched.append(False)
            else:
                vals.append(0.0)
                hatched.append(False)
        bars = ax.bar(x + (i - 1) * w, vals, w, label=mname,
                      color=colors[mname])
        for b, h in zip(bars, hatched):
            if h:
                b.set_hatch("////")
                b.set_edgecolor("black")
                b.set_linewidth(0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([L.replace("_", "\\_") for L in SUB],
                       rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Steady-state J / output token")
    ax.set_yscale("log")
    # Tier separators / labels
    for xx in (3.5, 7.5):
        ax.axvline(xx, color="#999", ls=":", lw=0.7)
    ax.text(1.5, ax.get_ylim()[1] * 0.92, "LOW", ha="center", fontsize=7,
            color="#555")
    ax.text(5.5, ax.get_ylim()[1] * 0.92, "MID", ha="center", fontsize=7,
            color="#555")
    ax.text(9.5, ax.get_ylim()[1] * 0.92, "HIGH", ha="center", fontsize=7,
            color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend incl. a hatched proxy for the censored bar.
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[m], label=m) for m in models]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="////",
                          label="censored (whole-run lower bound)"))
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2,
              fontsize=6.5)

    fig.tight_layout()
    out = OUT / "fig_tier_heatmap.pdf"
    fig.savefig(out)
    import shutil
    shutil.copyfile(out, OUT / "fig_tier_heatmap.crop.pdf")
    plt.close(fig)

    # Stats for the caption / text, on the 11 clean langs.
    clean = [L for L in SUB
             if all(L in s.index for s in models.values())]
    b = models["Qwen3-8B"].loc[clean]
    msg = []
    for n in ("Qwen3-14B", "Llama-3.1-8B"):
        rho, p = stats.spearmanr(b.values, models[n].loc[clean].values)
        msg.append(f"Qwen3-8B vs {n}: rho={rho:.2f} (p={p:.2g})")
    print("Wrote", out, "and .crop.pdf")
    print(f"clean langs (3-model) = {len(clean)} (bod_Tibt censored for Qwen3-14B)")
    print("  " + " | ".join(msg))


if __name__ == "__main__":
    main()
