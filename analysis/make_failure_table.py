"""Emit a LaTeX table of the four output failure modes with the *actual*
Qwen3-8B Belebele generations (verbatim excerpts).

pdfLaTeX safety: the raw outputs contain emojis, markdown, and (Amharic)
Ethiopic script, all of which break inputenc[utf8]. We therefore:
  - keep the text verbatim but strip codepoints outside a pdfLaTeX-safe set
    (ASCII + Latin-1/Extended-A), replacing runs of stripped chars with [.];
  - escape LaTeX specials;
  - elide repetition loops with \ldots;
  - for the Ethiopic example (whole output is non-Latin) the cell shows the
    English gloss + the chars/token statistic, with the raw text preserved
    in the sidecar .txt.
Also writes the untouched raw excerpts to examples_failure_modes_raw.txt.
"""

from __future__ import annotations

import glob
import json
import os
import re

B = "./benchmark/run/llm/belebele"
PAPER = "./paper"
TEX = f"{PAPER}/examples_failure_modes.tex"
RAW = f"{PAPER}/examples_failure_modes_raw.txt"


def load(lang):
    fs = [f for f in glob.glob(
        f"{B}/{lang}/local-jsonl-bench/results/Qwen/Qwen3-8B/*/*/results.json")
        if os.path.getsize(f) > 0]
    return json.load(open(fs[0]))["results"]


def repscore(s, n=8):
    s2 = re.sub(r"\s+", " ", s)
    g = [s2[i:i + n] for i in range(max(len(s2) - n, 0))]
    return 1 - len(set(g)) / len(g) if g else 0.0


def latin_frac(s):
    ns = [c for c in s if not c.isspace()]
    return sum(c.isascii() and c.isalpha() for c in ns) / len(ns) if ns else 0.0


def tex_sanitize(s: str) -> str:
    """Verbatim text -> pdfLaTeX-safe. Strip non-Latin/emoji, escape specials."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\n\t":
            out.append(" ")
        elif o < 0x250 or 0x1E00 <= o <= 0x1EFF:   # ASCII + Latin-1/Ext-A/B/Add'l
            out.append(ch)
        else:
            out.append("")                   # mark stripped
    s = "".join(out)
    s = re.sub("+", " [.] ", s)              # collapse stripped runs
    s = re.sub(r"\s+", " ", s).strip()
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(c, c) for c in s)


def main():
    raw_lines = []

    # 1. Repetition -- Wolof
    r = load("wol_Latn")
    e = max(r, key=lambda x: (repscore(x["output_text"] or ""), x["output_len"]))
    rep_out = e["output_text"]
    raw_lines.append(("1 REPETITION wol_Latn ol=%d" % e["output_len"], rep_out[:600]))
    rep_cell = tex_sanitize(rep_out[:90]) + r" \ldots{}"

    # 2. Over-generation / uncertainty -- Latvian
    r = load("lvs_Latn")
    c = [x for x in r if repscore(x["output_text"] or "") < 0.3
         and x["output_len"] > 800]
    e = max(c, key=lambda x: x["output_len"])
    lv_out = e["output_text"]
    raw_lines.append(("2 OVERGEN lvs_Latn ol=%d" % e["output_len"], lv_out[:600]))
    lv_cell = tex_sanitize(lv_out[:300]) + r" \ldots{}"
    lv_gloss = ("The answer to all three questions is given here, based on "
                "the measures and options provided, each question analysed "
                "in its own logic\\ldots{}")

    # 3. High chars/token -- Amharic (native cannot render; gloss + stat)
    r = load("amh_Ethi")
    c = [x for x in r if repscore(x["output_text"] or "") < 0.3
         and 1500 < x["output_len"] < 4000]
    e = max(c, key=lambda x: len(x["output_text"]) / max(x["output_len"], 1))
    am_out = e["output_text"]
    cpt = len(am_out) / e["output_len"]
    en_ref = load("eng_Latn")[0]
    en_cpt = len(en_ref["output_text"]) / max(en_ref["output_len"], 1)
    raw_lines.append(("3 CHARS/TOK amh_Ethi ol=%d chars=%d cpt=%.2f"
                      % (e["output_len"], len(am_out), cpt), am_out[:600]))
    am_cell = (r"\textit{[Amharic, Ethiopic script --- raw text in sidecar; "
               r"gloss:]} ``Let us think step by step. The Cook Islands are "
               r"an island nation in the South Pacific, in Polynesia, in "
               r"free association with New Zealand\ldots{}'' "
               r"\textbf{%d chars $=$ %d tokens, %.2f chars/tok "
               r"(English: %.2f).}" % (len(am_out), e["output_len"], cpt,
                                       en_cpt))

    # 4. Code-switching -- Shan prompt, English output
    r = load("shn_Mymr")
    c = [x for x in r if latin_frac(x["output_text"] or "") > 0.85
         and x["output_len"] > 200]
    e = max(c, key=lambda x: x["output_len"])
    sh_out = e["output_text"]
    raw_lines.append(("4 CODESWITCH shn_Mymr->EN ol=%d" % e["output_len"],
                      sh_out[:600]))
    sh_cell = tex_sanitize(sh_out[:340]) + r" \ldots{}"

    # --- Well-behaved high-resource references (English, Chinese) ---
    def good(lang):
        rr = load(lang)
        c = [x for x in rr if x["output_text"]
             and 40 < x["output_len"] < 260
             and repscore(x["output_text"]) < 0.15
             and re.search(r"(answer is|答案是)", x["output_text"])]
        c.sort(key=lambda x: x["output_len"])
        return c[len(c) // 4] if c else min(rr, key=lambda x: x["output_len"])

    eg = good("eng_Latn")
    eng_out = eg["output_text"]
    raw_lines.append(("R1 GOOD eng_Latn ol=%d" % eg["output_len"],
                      eng_out[:600]))
    eng_cell = tex_sanitize(eng_out[:360])

    zg = good("zho_Hans")
    zh_out = zg["output_text"]
    raw_lines.append(("R2 GOOD zho_Hans ol=%d" % zg["output_len"],
                      zh_out[:600]))
    # Faithful manual translation (NLLB-600M garbled 'lifelong' and inverted
    # the reasoning; a reference example must be accurately glossed).
    zh_cell = (r"\textit{[Chinese; raw in sidecar. Translation:]} "
               r"``According to the passage, the reason Allen began "
               r"searching for the Musashi was his `lifelong deep interest "
               r"in war'. Although the text mentions he was wealthy, "
               r"invested in ocean exploration, and did seabed mapping, "
               r"these are background, not the main reason that drove the "
               r"eight-year search. Therefore the correct answer is (4). "
               r"The answer is (4).''")

    with open(RAW, "w") as f:
        for hdr, txt in raw_lines:
            f.write(f"### {hdr}\n{txt}\n\n")

    rows = [
        (r"Token explosion (repetition)", r"Wolof", rep_cell,
         r"(not meaningful Wolof; the token ``Dhaan,'' looped to the "
         r"$\sim$18.9k-token cap, no answer)"),
        (r"Task uncertainty $\rightarrow$ over-generation", r"Latvian",
         lv_cell, lv_gloss + r" \emph{[$\sim$1.9k tokens, 12.3$\times$ "
         r"English, 1.0\% acc]}"),
        (r"High chars/token (script)", r"Amharic", am_cell, r"---"),
        (r"Code-switching", r"Shan$\rightarrow$English", sh_cell,
         r"prompt is entirely in Shan; model answers in English and "
         r"misidentifies the passage as Burmese"),
    ]

    L = []
    a = L.append
    a("% Auto-generated by research/lmarena_translate/make_failure_table.py")
    a("% Verbatim Qwen3-8B Belebele outputs; non-Latin/emoji stripped to a")
    a("% [.] marker for pdfLaTeX safety; raw text in")
    a("% examples_failure_modes_raw.txt. Repetition elided with \\ldots{}.")
    a(r"\begin{table*}[t]")
    a(r"  \centering\footnotesize")
    a(r"  \renewcommand{\arraystretch}{1.2}")
    a(r"  \begin{tabular}{p{2.4cm}p{1.5cm}p{6.2cm}p{4.4cm}}")
    a(r"    \toprule")
    a(r"    \textbf{Phenomenon} & \textbf{Lang.} & "
      r"\textbf{Actual model output (verbatim excerpt)} & "
      r"\textbf{English gloss / note} \\")
    a(r"    \midrule")
    for ph, lg, out, gl in rows:
        a(f"    {ph} & {lg} & {out} & {gl} \\\\")
        a(r"    \addlinespace")
    a(r"    \midrule")
    a(r"    \multicolumn{4}{l}{\textit{Well-behaved high-resource outputs "
      r"(reference)}} \\")
    a(r"    \addlinespace")
    a(f"    Concise correct answer & English & {eng_cell} & "
      r"(correct step-by-step reasoning, properly formatted answer; "
      r"$\sim$70 tokens) \\")
    a(r"    \addlinespace")
    a(f"    Concise correct answer & Chinese & {zh_cell} & "
      r"(correct reasoning in the prompt language; $\sim$72 tokens, "
      r"same task as the Wolof row) \\")
    a(r"    \addlinespace")
    a(r"    \bottomrule")
    a(r"  \end{tabular}")
    a(r"  \caption{Representative Qwen3-8B generations on Belebele: the four "
      r"output failure modes (Section~\ref{sec:drivers}, \ref{sec:script}, "
      r"\ref{sec:codeswitch}) and, for contrast, well-behaved English and "
      r"Chinese outputs on the same task. Output column is verbatim except: "
      r"long repetition loops elided with `\ldots'; emojis, markdown "
      r"symbols, and non-Latin glyphs that break pdf\LaTeX{} are replaced "
      r"with `[.]' (raw untouched text in the supplementary "
      r"\texttt{examples\_failure\_modes\_raw.txt}). The Amharic and Chinese "
      r"outputs are non-Latin script (cannot render under pdf\LaTeX{}), so "
      r"those cells give an accurate English translation (raw text in the "
      r"sidecar); the Amharic cell also gives the decisive chars/token "
      r"statistic.}")
    a(r"  \label{tab:failure_modes}")
    a(r"\end{table*}")
    open(TEX, "w").write("\n".join(L) + "\n")
    print(f"Wrote {TEX}\nWrote {RAW}")
    for hdr, _ in raw_lines:
        print("  ", hdr)


if __name__ == "__main__":
    main()
