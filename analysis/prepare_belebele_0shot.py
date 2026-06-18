"""Build 0-shot CoT Belebele prompts for the ml.energy multilingual rerun.

0-shot replacement for the 2-shot research/prepareBelebeleForMlEnergy.py:
header + few-shot examples are dropped; each prompt is the per-language
translated instruction (already ending with the verbatim parse token
"#### N") + the localized Passage/Question/Options block + the per-language
primer that elicits chain-of-thought.

All inputs are local (no network / load_dataset):
  - instructions_all_languages.json  -> per-lang `instruction` + `primer`
  - diverse_prompts_db.jsonl         -> localized P/Q/O labels (116 langs)
  - romanized_labels_TO_FILL.json    -> P/Q/O labels for the 6 romanized
                                        variants (db labels are blank there)
  - belebele/{lang}.jsonl            -> 900 questions/lang

Output: {"prompt": ...} jsonl, one file per language (mlenergy workload
contract, identical to the 2-shot generator).
"""

import argparse
import json
import os

ROOT = "./research"
INSTR_PATH = f"{ROOT}/lmarena_translate/instructions_all_languages.json"
DB_PATH = f"{ROOT}/diverse_prompts_db.jsonl"
ROMAN_LABELS_PATH = f"{ROOT}/lmarena_translate/romanized_labels_TO_FILL.json"
BELEBELE_DIR = f"{ROOT}/belebele"
OUT_DIR = os.environ.get("OUT_DIR", "belebele_0shot")

# The 6 romanized Belebele variants whose db labels are blank (NLLB had no
# code) -> labels come from the hand-filled override file instead.
ROMANIZED = {"arb_Latn", "ben_Latn", "hin_Latn", "npi_Latn", "sin_Latn", "urd_Latn"}


def load_inputs():
    instr = json.load(open(INSTR_PATH, encoding="utf-8"))["languages"]
    db = {}
    with open(DB_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            db[d["lang"]] = d
    roman = json.load(open(ROMAN_LABELS_PATH, encoding="utf-8"))["labels"]
    return instr, db, roman


def resolve_labels(lang, db, roman):
    """Localized {P,Q,O}, stripped. Romanized variants use the override file."""
    src = roman[lang] if lang in ROMANIZED else db[lang]["labels"]
    lbl = {k: (src[k] or "").strip() for k in ("P", "Q", "O")}
    for k, v in lbl.items():
        if not v:  # defensive: no empty label should ever reach a prompt
            raise ValueError(f"empty {k} label for {lang} (unexpected blank-label language?)")
    return lbl


def build_prompt(instruction, primer, lbl, row):
    task = (
        f"{lbl['P']} {row['flores_passage']}\n"
        f"{lbl['Q']} {row['question']}\n"
        f"{lbl['O']} 1. {row['mc_answer1']} 2. {row['mc_answer2']} "
        f"3. {row['mc_answer3']} 4. {row['mc_answer4']}\n"
        f"{primer}"
    )
    # `instruction` already ends with the verbatim final line "#### N".
    return f"{instruction}\n\n{task}"


def gen_lang(lang, instr, db, roman, out_dir, write=True):
    instruction = instr[lang]["instruction"]
    primer = instr[lang]["primer"]
    lbl = resolve_labels(lang, db, roman)
    prompts = []
    with open(f"{BELEBELE_DIR}/{lang}.jsonl", encoding="utf-8") as f:
        for line in f:
            prompts.append(build_prompt(instruction, primer, lbl, json.loads(line)))
    if write:
        os.makedirs(out_dir, exist_ok=True)
        outp = os.path.join(out_dir, f"{lang}_bench.jsonl")
        with open(outp, "w", encoding="utf-8") as g:
            for i, p in enumerate(prompts):
                # Full DataRequest schema (mlenergy/llm/datasets.py): prompt +
                # required completion/multimodal_contents/multimodal_content_paths.
                # completion="" matches the gsm8k/lmarena crosstask jsonl that
                # ran successfully (Belebele is scored downstream vs the gold
                # `correct_answer`, not from this field).
                g.write(json.dumps({
                    "prompt": p,
                    "completion": "",
                    "multimodal_contents": [],
                    "multimodal_content_paths": [],
                    "id": f"belebele_{lang}_{i}",
                }, ensure_ascii=False) + "\n")
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=None, help="subset; default = all 122")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="build but write nothing")
    args = ap.parse_args()

    instr, db, roman = load_inputs()
    langs = args.langs or sorted(instr.keys())
    total = 0
    for lang in langs:
        ps = gen_lang(lang, instr, db, roman, args.out_dir, write=not args.dry_run)
        total += len(ps)
        tail = "" if args.dry_run else f" -> {args.out_dir}/{lang}_bench.jsonl"
        print(f"{lang}: {len(ps)} prompts{tail}")
    note = " (dry-run, nothing written)" if args.dry_run else ""
    print(f"\nTOTAL: {len(langs)} langs, {total} prompts{note}")


if __name__ == "__main__":
    main()
