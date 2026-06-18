"""Mine concrete output examples for the four failure phenomena.

Source: canonical Belebele Qwen3-8B runs (output_text is stored per request).
CPU-only (JSON + string metrics); does not touch the GPUs.

Phenomena:
  1. token explosion (repetition)         -> high out_len, high char-ngram repeat
  2. uncertain-about-task (verbose, varied)-> high out_len, LOW repeat
  3. high character-to-token (script)      -> coherent attempt, non-Latin script
  4. code-switching                        -> non-Latin-script lang, high Latin frac
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter

BASE = "./benchmark/run/llm/belebele"

# script kind per language (for Latin-fraction interpretation)
LANGS = {
    "bod_Tibt": "Tibetan", "shn_Mymr": "Myanmar", "tir_Ethi": "Ethiopic",
    "amh_Ethi": "Ethiopic", "mya_Mymr": "Myanmar", "hin_Deva": "Devanagari",
    "wol_Latn": "Latin", "grn_Latn": "Latin", "lvs_Latn": "Latin",
    "luo_Latn": "Latin", "pbt_Arab": "Arabic", "kan_Knda": "Kannada",
}


def char_ngrams(s, n=8):
    s = re.sub(r"\s+", " ", s)
    return [s[i:i + n] for i in range(0, max(len(s) - n, 0))]


def repeat_score(s):
    ng = char_ngrams(s, 8)
    if not ng:
        return 0.0
    return 1.0 - len(set(ng)) / len(ng)          # ~1 => extremely repetitive


def latin_frac(s):
    nonspace = [c for c in s if not c.isspace()]
    if not nonspace:
        return 0.0
    return sum(c.isascii() and c.isalpha() for c in nonspace) / len(nonspace)


def load(lang):
    fs = glob.glob(f"{BASE}/{lang}/local-jsonl-bench/results/Qwen/Qwen3-8B/"
                   f"*/*/results.json")
    fs = [f for f in fs if __import__("os").path.getsize(f) > 0]
    if not fs:
        return []
    d = json.load(open(fs[0]))
    return d["results"]


def main():
    for lang, script in LANGS.items():
        rows = load(lang)
        if not rows:
            print(f"\n### {lang}: NO DATA")
            continue
        scored = []
        for r in rows:
            ot = r.get("output_text") or ""
            if not ot:
                continue
            scored.append({
                "ol": r["output_len"], "rep": repeat_score(ot),
                "lat": latin_frac(ot), "ot": ot, "pr": r["prompt"],
            })
        if not scored:
            continue
        n = len(scored)
        rep_hi = max(scored, key=lambda x: (x["rep"], x["ol"]))
        verbose_lowrep = max((s for s in scored if s["rep"] < 0.3),
                             key=lambda x: x["ol"], default=None)
        lat_hi = max(scored, key=lambda x: x["lat"])
        print(f"\n### {lang} ({script})  n={n}")
        print(f"  [repetition]  ol={rep_hi['ol']} rep={rep_hi['rep']:.2f} "
              f"lat={rep_hi['lat']:.2f}")
        print(f"     out[:160]: {rep_hi['ot'][:160]!r}")
        if verbose_lowrep:
            print(f"  [verbose/low-rep] ol={verbose_lowrep['ol']} "
                  f"rep={verbose_lowrep['rep']:.2f} "
                  f"lat={verbose_lowrep['lat']:.2f}")
            print(f"     out[:160]: {verbose_lowrep['ot'][:160]!r}")
        print(f"  [max-latin]   ol={lat_hi['ol']} lat={lat_hi['lat']:.2f} "
              f"rep={lat_hi['rep']:.2f}")
        print(f"     out[:160]: {lat_hi['ot'][:160]!r}")


if __name__ == "__main__":
    main()
