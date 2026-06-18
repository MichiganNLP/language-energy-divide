"""Aggregate the 0-shot Belebele canonical run into a per-model CSV with
energy, output-token, and parsed-accuracy columns.

Source layout (set ``REPO_ROOT`` / run root via environment):
  <RUN_ROOT>/lmarena_qwen8b_L40S_12langs/
    belebele_0shot_seqs256/<lang>/local-jsonl-bench/
    results/<model>/<gpu>/<params>/results.json

Each results.json carries the per-request outputs in `results[]`. For
accuracy we apply the locked 0-shot parser (last `####\\s*([1-4])` regex
match in `output_text`) and compare to the gold `correct_answer` in
research/belebele/<lang>.jsonl. Format-failures (no `#### N` match) count
as INCORRECT and are tracked separately as a reportable degeneration
metric (per the design memo for the 0-shot rerun).

Output: research/lmarena_translate/belebele_canonical_<MODEL>_0shot.csv
with columns:
  language, language_name, resource_level,
  steady_energy_per_token_J, total_output_tokens, total_input_tokens,
  completed, duration, steady_state_duration, whole_gpu_energy_J,
  accuracy, num_correct, num_format_failures, num_evaluated

Extends NLLB binary resource labels to the 6 romanized variants by
inheriting from their native-script counterpart (arb_Latn = arb_Arab's
label, etc.). New totals: 56 High + 66 Low = 122.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(".")
DEFAULT_OUT_ROOT = (
    "./"
    "benchmark/run/llm/lmarena_qwen8b_L40S_12langs/"
    "belebele_0shot_seqs256"
)
BELEBELE_DIR = REPO / "research" / "belebele"
NLLB_CSV = REPO / "research" / "Qwen3-8B" / "resource_levels_matched.csv"

# Extend NLLB binary resource labels to the 6 romanized variants by their
# native-script counterpart (same language, just Latin spelling).
ROMANIZED_INHERITANCE = {
    "arb_Latn": "arb_Arab",
    "ben_Latn": "ben_Beng",
    "hin_Latn": "hin_Deva",
    "npi_Latn": "npi_Deva",
    "sin_Latn": "sin_Sinh",
    "urd_Latn": "urd_Arab",
}

ROMANIZED_NAMES = {
    "arb_Latn": "Modern Standard Arabic (Latin)",
    "ben_Latn": "Bengali (Latin)",
    "hin_Latn": "Hindi (Latin)",
    "npi_Latn": "Nepali (Latin)",
    "sin_Latn": "Sinhala (Latin)",
    "urd_Latn": "Urdu (Latin)",
}

# English names extracted verbatim from the existing tab:full_results in
# paper/paper-update/_extracted/latex/acl_latex.tex so we don't drift on
# casing/diacritics. Keys are FLORES/Belebele codes.
LANG_NAMES = {
    "acm_Arab": "Mesopotamian Arabic", "afr_Latn": "Afrikaans",
    "als_Latn": "Tosk Albanian", "amh_Ethi": "Amharic",
    "apc_Arab": "North Levantine Arabic", "arb_Arab": "Modern Standard Arabic",
    "ars_Arab": "Najdi Arabic", "ary_Arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic", "asm_Beng": "Assamese",
    "azj_Latn": "N. Azerbaijani", "bam_Latn": "Bambara",
    "ben_Beng": "Bengali", "bod_Tibt": "Tibetan",
    "bul_Cyrl": "Bulgarian", "cat_Latn": "Catalan",
    "ceb_Latn": "Cebuano", "ces_Latn": "Czech",
    "ckb_Arab": "Central Kurdish", "dan_Latn": "Danish",
    "deu_Latn": "German", "ell_Grek": "Greek",
    "eng_Latn": "English", "est_Latn": "Estonian",
    "eus_Latn": "Basque", "fin_Latn": "Finnish",
    "fra_Latn": "French", "fuv_Latn": "Fulfulde",
    "gaz_Latn": "West Central Oromo", "grn_Latn": "Guarani",
    "guj_Gujr": "Gujarati", "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa", "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi", "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian", "hye_Armn": "Armenian",
    "ibo_Latn": "Igbo", "ilo_Latn": "Ilocano",
    "ind_Latn": "Indonesian", "isl_Latn": "Icelandic",
    "ita_Latn": "Italian", "jav_Latn": "Javanese",
    "jpn_Jpan": "Japanese", "kac_Latn": "Jingpho",
    "kan_Knda": "Kannada", "kat_Geor": "Georgian",
    "kaz_Cyrl": "Kazakh", "kea_Latn": "Kabuverdianu",
    "khk_Cyrl": "Halh Mongolian", "khm_Khmr": "Khmer",
    "kin_Latn": "Kinyarwanda", "kir_Cyrl": "Kyrgyz",
    "kor_Hang": "Korean", "lao_Laoo": "Lao",
    "lin_Latn": "Lingala", "lit_Latn": "Lithuanian",
    "lug_Latn": "Ganda", "luo_Latn": "Luo",
    "lvs_Latn": "Latvian", "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi", "mkd_Cyrl": "Macedonian",
    "mlt_Latn": "Maltese", "mri_Latn": "Maori",
    "mya_Mymr": "Burmese", "nld_Latn": "Dutch",
    "nob_Latn": "Norwegian Bokmal",
    "npi_Deva": "Nepali", "nso_Latn": "Northern Sotho",
    "nya_Latn": "Nyanja", "ory_Orya": "Odia",
    "pan_Guru": "Eastern Panjabi", "pbt_Arab": "Southern Pashto",
    "pes_Arab": "Western Persian", "plt_Latn": "Plateau Malagasy",
    "pol_Latn": "Polish", "por_Latn": "Portuguese",
    "ron_Latn": "Romanian", "rus_Cyrl": "Russian",
    "shn_Mymr": "Shan", "sin_Sinh": "Sinhala",
    "slk_Latn": "Slovak", "slv_Latn": "Slovenian",
    "sna_Latn": "Shona", "snd_Arab": "Sindhi",
    "som_Latn": "Somali", "sot_Latn": "Southern Sotho",
    "spa_Latn": "Spanish", "srp_Cyrl": "Serbian",
    "ssw_Latn": "Swati", "sun_Latn": "Sundanese",
    "swe_Latn": "Swedish", "swh_Latn": "Swahili",
    "tam_Taml": "Tamil", "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik", "tgl_Latn": "Tagalog",
    "tha_Thai": "Thai", "tir_Ethi": "Tigrinya",
    "tsn_Latn": "Tswana", "tso_Latn": "Tsonga",
    "tur_Latn": "Turkish", "ukr_Cyrl": "Ukrainian",
    "urd_Arab": "Urdu", "uzn_Latn": "Northern Uzbek",
    "vie_Latn": "Vietnamese", "war_Latn": "Waray",
    "wol_Latn": "Wolof", "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba", "zho_Hans": "Simplified Chinese",
    "zho_Hant": "Traditional Chinese", "zsm_Latn": "Standard Malay",
    "zul_Latn": "Zulu",
    **ROMANIZED_NAMES,
}

# Parser for the locked 0-shot format token: last `####\s*([1-4])` match.
TOKEN_RE = re.compile(r"####\s*([1-4])")
# Lenient fallback: scan the LAST non-empty lines of the output for an
# isolated 1-4. Catches the common cases where Qwen3-8B emits the right
# answer without the `#### N` literal (e.g. `**2**`, `Answer: 2`, ` 2 `).
# Strict accuracy still excludes these from `num_correct`; lenient accuracy
# counts them. The gap between the two is the "format-failure" rate.
LENIENT_RE = re.compile(r"(?<![0-9])([1-4])(?![0-9])")

# Resource-level patches for langs missing from the NLLB CSV but used by
# the existing paper. Pre-existing pre-paper-update inconsistency; we
# match the paper's existing convention here so cross-cutting numbers stay
# self-consistent.
RESOURCE_PATCH = {
    "bod_Tibt": "Low",
}


def load_resource_levels() -> dict[str, str]:
    rl = {r["language"]: r["resource_level"] for r in csv.DictReader(open(NLLB_CSV))}
    # Apply patches for langs the existing paper relies on but that aren't in
    # the NLLB CSV (e.g. bod_Tibt). Then inherit for romanized variants.
    rl.update(RESOURCE_PATCH)
    for rom, native in ROMANIZED_INHERITANCE.items():
        if native in rl:
            rl[rom] = rl[native]
    return rl


def parse_lenient(output_text: str) -> str | None:
    """Find an isolated 1-4 in the last few lines of the output. Used only
    as a fallback when the strict `#### N` parser misses."""
    if not output_text:
        return None
    # Take the last 5 non-empty lines (usually where the answer lives) so
    # we don't pick up an "answer: 2" mention from the prompt echo.
    tail = [ln for ln in output_text.splitlines() if ln.strip()][-5:]
    matches = LENIENT_RE.findall("\n".join(tail))
    return matches[-1] if matches else None


def load_gold(lang: str) -> list[str]:
    """Gold answer ('1'..'4') in row order, matching how prepare_belebele_0shot.py
    iterates the file."""
    out = []
    with open(BELEBELE_DIR / f"{lang}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(str(r["correct_answer"]).strip())
    return out


def parse_answer(output_text: str) -> str | None:
    matches = TOKEN_RE.findall(output_text or "")
    return matches[-1] if matches else None


def score(results: list[dict], gold: list[str]) -> tuple[int, int, int, int]:
    """(strict_correct, lenient_correct, num_format_failures, num_evaluated).

    strict: counts only rows whose output ends with the literal `#### N` token
            (matches the locked 0-shot parser design).
    lenient: also counts rows where the strict parser missed but a fallback
             (last 1-4 in the tail of the output) recovers a digit equal to
             gold. This is the metric closest to what the 2-shot run with
             its own "The answer is (X)" parser was measuring.
    format_failures: rows whose strict parse returned None. The lenient/strict
             accuracy gap is bounded by this number."""
    strict_c = lenient_c = n_fmt_fail = n_eval = 0
    for i, r in enumerate(results):
        if i >= len(gold):
            break
        ot = r.get("output_text", "")
        pred = parse_answer(ot)
        if pred is None:
            n_fmt_fail += 1
            lp = parse_lenient(ot)
            if lp == gold[i]:
                lenient_c += 1
        else:
            if pred == gold[i]:
                strict_c += 1
                lenient_c += 1
        n_eval += 1
    return strict_c, lenient_c, n_fmt_fail, n_eval


def find_results_json(out_root: Path, lang: str, model_id: str, gpu: str) -> Path | None:
    base = out_root / lang / "local-jsonl-bench" / "results" / model_id / gpu
    if not base.exists():
        return None
    cands = sorted(base.glob("**/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [p for p in cands if p.stat().st_size > 0]
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-tag", default="Qwen3-8B",
                    help="short tag for the output filename")
    ap.add_argument("--gpu", default="L40S")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--csv-out", default=None,
                    help="default: research/lmarena_translate/belebele_canonical_<tag>_0shot.csv")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    csv_out = Path(args.csv_out) if args.csv_out else (
        REPO / "research" / "lmarena_translate" / f"belebele_canonical_{args.model_tag}_0shot.csv")

    rl = load_resource_levels()
    langs = sorted({p.name for p in out_root.iterdir() if p.is_dir()})

    rows = []
    skipped = []
    for lg in langs:
        rj = find_results_json(out_root, lg, args.model_id, args.gpu)
        if rj is None:
            skipped.append((lg, "no results.json"))
            continue
        d = json.loads(rj.read_text())
        gold = load_gold(lg)
        strict_c, lenient_c, n_fmt_fail, n_eval = score(d.get("results") or [], gold)
        rows.append({
            "language": lg,
            "language_name": LANG_NAMES.get(lg, lg),
            "resource_level": rl.get(lg, "Unknown"),
            "steady_energy_per_token_J": d.get("steady_state_energy_per_token"),
            "total_output_tokens": d.get("total_output_tokens"),
            "total_input_tokens": d.get("total_input_tokens"),
            "completed": d.get("completed"),
            "duration": d.get("duration"),
            "steady_state_duration": d.get("steady_state_duration"),
            "whole_gpu_energy_J": (d.get("entire_benchmark_measurement") or {}).get("gpu_energy"),
            "accuracy_strict": (strict_c / n_eval) if n_eval else None,
            "accuracy_lenient": (lenient_c / n_eval) if n_eval else None,
            "num_correct_strict": strict_c,
            "num_correct_lenient": lenient_c,
            "num_format_failures": n_fmt_fail,
            "num_evaluated": n_eval,
        })

    rows.sort(key=lambda r: r["language"])
    if not rows:
        print("no rows aggregated; check --out-root and --model-id")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # quick summary
    n_high = sum(r["resource_level"] == "High" for r in rows)
    n_low = sum(r["resource_level"] == "Low" for r in rows)
    n_unk = len(rows) - n_high - n_low
    print(f"wrote {csv_out} | {len(rows)} langs ({n_high} High + {n_low} Low + {n_unk} Unknown)")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:6]}")
    censored = [r for r in rows if (r["completed"] or 0) != 900]
    print(f"right-censored (completed != 900): {len(censored)}")
    fmt_fail_heavy = sorted(rows, key=lambda r: -(r["num_format_failures"] or 0))[:5]
    print("top-5 format-failure langs:",
          [(r["language"], r["num_format_failures"], r["num_evaluated"]) for r in fmt_fail_heavy])


if __name__ == "__main__":
    main()
