"""Batch-size sweep figures: TWO separate single-GPU figures (L40S, RTX 6000).

Qwen3-8B, 0-shot CoT, 8-language Belebele subset, max_num_seqs in
{16,32,64,128,256,512}. Per-token energy = steady-state J / output token
(request-count invariant, so the differing repeat counts are comparable).

Sources:
  L40S : research/lmarena_translate/belebele_seqsweep_l40s_0shot.csv
  RTX  : research/cross_gpu/seqsweep_belebele_v1_qwen8b_RTX6000_all8.csv

Outputs (each a standalone figure, auto-scaled y so each GPU's batch trend is
readable on its own; the cross-GPU magnitude is carried by the captions/table):
  paper/paper-update-1/extracted/figures/fig_seqsweep_l40s.crop.pdf
  paper/paper-update-1/extracted/figures/fig_seqsweep_rtx.crop.pdf
Set SHARED_YLIM = True to put both on the same y-axis for a visual comparison.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(".")
FIGDIR = REPO / "paper" / "paper-update-1" / "extracted" / "figures"
L40S_CSV = REPO / "research" / "lmarena_translate" / "belebele_seqsweep_l40s_0shot.csv"
RTX_CSV = REPO / "research" / "cross_gpu" / "seqsweep_belebele_v1_qwen8b_RTX6000_all8.csv"

SEQS = [16, 32, 64, 128, 256, 512]
HIGH = ["eng_Latn", "zho_Hans", "rus_Cyrl", "fra_Latn"]
LOW = ["pbt_Arab", "tir_Ethi", "shn_Mymr", "bod_Tibt"]
LANGS = HIGH + LOW
COLOR = {
    "eng_Latn": "#08519c", "zho_Hans": "#3182bd", "rus_Cyrl": "#6baed6", "fra_Latn": "#9ecae1",
    "pbt_Arab": "#a50f15", "tir_Ethi": "#de2d26", "shn_Mymr": "#fb6a4a", "bod_Tibt": "#fdae6b",
}
SHARED_YLIM = False  # set True to force both figures onto the same y-axis

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
})


def load(csv_path, lang_key, bsz_key, jpt_key):
    d = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            d[(r[lang_key], int(r[bsz_key]))] = float(r[jpt_key])
    return d


def render_one(data, title, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(3.7, 1.5))
    for L in LANGS:
        ys = [data.get((L, b)) for b in SEQS]
        ax.plot(SEQS, ys, marker="o", ms=3, lw=1.4, color=COLOR[L],
                label=f"{L} ({'HIGH' if L in HIGH else 'LOW'}-res)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQS)
    ax.set_xticklabels(SEQS)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("J / output token")
    ax.set_ylim(0, ylim) if ylim else ax.set_ylim(bottom=0)
    # ax.set_title(title)
    # ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    l40s = load(L40S_CSV, "language", "max_num_seqs", "jpt")
    rtx = load(RTX_CSV, "lang", "bsz", "jpt")
    top = None
    if SHARED_YLIM:
        top = 1.02 * max(max(l40s.values()), max(rtx.values()))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    top = 1.0
    render_one(l40s, "L40S", FIGDIR / "fig_seqsweep_l40s.pdf", ylim=top)
    render_one(rtx, "RTX 6000 Pro Blackwell", FIGDIR / "fig_seqsweep_rtx.pdf", ylim=top)
    for name, d in [("L40S", l40s), ("RTX", rtx)]:
        v = {b: [d[(l, b)] for l in LANGS if (l, b) in d] for b in SEQS}
        print(f"  {name} disparity: " + " ".join(
            f"bs{b}={max(v[b])/min(v[b]):.1f}x" for b in SEQS if v[b]))


if __name__ == "__main__":
    main()
