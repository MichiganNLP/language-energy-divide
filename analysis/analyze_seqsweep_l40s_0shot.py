"""L40S zero-shot Belebele max_num_seqs sweep -> fig_seqsweep (paper-update-1).

Qwen3-8B, single L40S, 0-shot CoT, 8-language subset, max_num_seqs in
{16,32,64,128,256,512}. Each batch size is a separate run dir
belebele_0shot_seqs256_Qwen__Qwen3-8B_sweep_bs{N}; bs=512 uses the rep=2
(1800-request) run so the 2*512 steady-state-window threshold is cleared.

Single-panel L40S figure now; an RTX panel will be added beside it when the
RTX sweep lands (user plan 2026-05-22).

Outputs:
  - research/lmarena_translate/belebele_seqsweep_l40s_0shot.csv
  - paper/paper-update-1/extracted/figures/fig_seqsweep.crop.pdf
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

RUN = Path("./benchmark/"
           "run/llm/lmarena_qwen8b_L40S_12langs")
REPO = Path(".")
OUT_DIR = REPO / "paper" / "paper-update-1" / "extracted" / "figures"
CSV = REPO / "research" / "lmarena_translate" / "belebele_seqsweep_l40s_0shot.csv"

SEQS = [16, 32, 64, 128, 256, 512]
LANGS = ["eng_Latn", "zho_Hans", "rus_Cyrl", "fra_Latn",
         "pbt_Arab", "tir_Ethi", "shn_Mymr", "bod_Tibt"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times",
                   "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})


def collect() -> pd.DataFrame:
    rows = []
    for bs in SEQS:
        d = RUN / f"belebele_0shot_seqs256_Qwen__Qwen3-8B_sweep_bs{bs}"
        for rj in d.glob("*/local-jsonl-bench/results/**/results.json"):
            if rj.stat().st_size == 0:
                continue
            lang = rj.relative_to(d).parts[0]
            if lang not in LANGS:
                continue
            m = re.search(r"max_num_seqs\+(\d+)", str(rj))
            r = json.loads(rj.read_text())
            rows.append({
                "language": lang,
                "max_num_seqs": int(m.group(1)) if m else bs,
                "jpt": r.get("steady_state_energy_per_token"),
                "steady_state_duration": r.get("steady_state_duration"),
                "completed": r.get("completed"),
            })
    df = pd.DataFrame(rows)
    df = df[(df["steady_state_duration"] > 5) & (df["jpt"] > 0)]
    # Per (lang, batch) keep the run with the longest steady-state window.
    df = (df.sort_values("steady_state_duration")
            .groupby(["language", "max_num_seqs"], as_index=False).last())
    return df.sort_values(["language", "max_num_seqs"]).reset_index(drop=True)


def make_figure(piv: pd.DataFrame) -> None:
    cheapest = piv[256].idxmin()
    priciest = piv[256].idxmax()
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for lang in piv.index:
        emph = lang in (cheapest, priciest)
        ax.plot(SEQS, piv.loc[lang, SEQS].values, marker="o", ms=3,
                lw=1.8 if emph else 0.8,
                color=("#FB7185" if lang == priciest
                       else "#60A5FA" if lang == cheapest else "#bbb"),
                label=lang if emph else None, zorder=3 if emph else 1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQS)
    ax.set_xticklabels(SEQS)
    ax.set_xlabel("Serving batch size")
    ax.set_ylabel("J / output token")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = OUT_DIR / "fig_seqsweep.crop.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\nWrote {out}  (cheapest={cheapest}, priciest={priciest})")


def main() -> None:
    df = collect()
    if df.empty:
        print("No sweep results found.")
        return
    df.to_csv(CSV, index=False)
    print(f"Wrote {CSV}")
    piv = df.pivot_table(index="language", columns="max_num_seqs",
                         values="jpt").reindex(columns=SEQS).reindex(LANGS)
    print("\nJ/token by language x batch size:")
    print(piv.round(3).to_string())
    print("\nDisparity (max/min across 8 langs) and mean at each batch size:")
    for s in SEQS:
        c = piv[s]
        print(f"  bs={s:4d}: min={c.min():.3f} max={c.max():.3f} "
              f"ratio={c.max()/c.min():.2f}x mean={c.mean():.3f}")
    print("\nAdjacent-config Spearman (per-language ordering):")
    for a, b in zip(SEQS, SEQS[1:]):
        rho, _ = stats.spearmanr(piv[a].values, piv[b].values)
        print(f"  {a:4d} -> {b:4d}: rho={rho:.3f}")
    make_figure(piv)


if __name__ == "__main__":
    main()
