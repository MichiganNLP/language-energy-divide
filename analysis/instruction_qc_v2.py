"""v2: translate ONLY the clean task+CoT sentence (no inline '####' literal,
which is what NLLB chokes on), then deterministically append a fixed
universal '#### N' format token. Re-run back-translation QC and emit the
final per-language instruction JSON.

Design:
  localized_instruction = NLLB( TASK )  +  "\\n"  +  FIXED_SPEC
  localized_primer       = NLLB( PRIMER )
TASK / PRIMER are short and NLLB-friendly (the first QC showed BERT median
0.96 on the prose; only the '#### N, where N is ...' clause tanked COMET).
FIXED_SPEC is identical ASCII in every language -> parser-bulletproof.

Outputs:
  research/lmarena_translate/instruction_qc_v2.jsonl   (per-lang QC)
  research/lmarena_translate/instructions_all_languages.json  (all 122)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(".")
BELE = REPO / "research" / "belebele"
QC_OUT = REPO / "research" / "lmarena_translate" / "instruction_qc_v2.jsonl"
JSON_OUT = REPO / "research" / "lmarena_translate" / "instructions_all_languages.json"
MANUAL_CSV = REPO / "research" / "lmarena_translate" / "instruction_manual.csv"

TASK = ("Read the passage and answer the multiple-choice question. "
        "Reason step by step. Then, on a new last line, write the number "
        "of the correct option (1, 2, 3, or 4) in exactly this format:")
FIXED_SPEC = "#### N"          # appended verbatim, identical in every language
PRIMER = "Let's think step by step."

ROMANIZED = ["arb_Latn", "ben_Latn", "hin_Latn",
             "npi_Latn", "sin_Latn", "urd_Latn"]   # NLLB has no code -> manual
BERT_T, COMET_T = 0.85, 0.75


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def translate(texts, model, tok, src, tgt, device, max_length=160):
    import torch
    tok.src_lang = src
    tgt_id = tok.convert_tokens_to_ids(tgt)
    out = []
    for b in batched(texts, 16):
        enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        with torch.inference_mode():
            ids = model.generate(**enc, forced_bos_token_id=tgt_id,
                                  max_length=max_length, num_beams=1)
        out.extend(tok.batch_decode(ids, skip_special_tokens=True))
    return out


def main():
    assert os.environ.get("HF_HOME") and os.environ.get("HF_TOKEN")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from bert_score import score as bert_score_fn
    from comet import download_model, load_from_checkpoint

    langs = sorted(p.stem for p in BELE.glob("*.jsonl"))
    targets = [l for l in langs if l != "eng_Latn" and l not in ROMANIZED]
    print(f"{len(langs)} Belebele langs | {len(targets)} NLLB targets "
          f"| {len(ROMANIZED)} romanized->manual | eng_Latn=source")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("loading NLLB-200-3.3B ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", torch_dtype=torch.bfloat16).to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    rows = {}
    t0 = time.time()
    for i, lg in enumerate(targets):
        fwd = translate([TASK, PRIMER], model, tok, "eng_Latn", lg, device)
        back = translate(fwd, model, tok, lg, "eng_Latn", device)
        rows[lg] = {"task_tgt": fwd[0], "primer_tgt": fwd[1],
                    "task_back": back[0], "primer_back": back[1]}
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(targets)} ({time.time()-t0:.0f}s)")
    print(f"translations done in {time.time()-t0:.0f}s")

    del model
    torch.cuda.empty_cache()

    tb = [rows[l]["task_back"] for l in targets]
    print("BERTScore ...")
    _, _, F1 = bert_score_fn(tb, [TASK] * len(targets), lang="en",
                             rescale_with_baseline=False, device=device,
                             batch_size=64, verbose=False)
    print("COMET ...")
    cm = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    ci = cm.predict([{"src": rows[l]["task_tgt"], "mt": rows[l]["task_back"],
                      "ref": TASK} for l in targets],
                    batch_size=32, gpus=1 if device == "cuda" else 0,
                    progress_bar=False)["scores"]

    with QC_OUT.open("w") as f:
        for k, lg in enumerate(targets):
            r = rows[lg]
            rec = {"lang": lg, "task_tgt": r["task_tgt"],
                   "task_back_en": r["task_back"],
                   "primer_tgt": r["primer_tgt"],
                   "primer_back_en": r["primer_back"],
                   "bertscore_f1": float(F1[k]), "comet": float(ci[k])}
            rec["pass"] = (rec["bertscore_f1"] >= BERT_T
                           and rec["comet"] >= COMET_T)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    qc = {json.loads(x)["lang"]: json.loads(x) for x in QC_OUT.open()}

    # ---- assemble the final per-language instruction JSON (all 122) --------
    out = {}
    out["eng_Latn"] = {
        "instruction": TASK + "\n" + FIXED_SPEC, "primer": PRIMER,
        "source": "english_canonical", "bertscore_f1": None, "comet": None,
        "task_back_en": TASK}
    for lg in ROMANIZED:
        out[lg] = {
            "instruction": None, "primer": None,
            "source": "MANUAL_PENDING (fill instruction_manual.csv; "
                      "NLLB has no romanized code)",
            "english_instruction": TASK + "\n" + FIXED_SPEC,
            "english_primer": PRIMER,
            "bertscore_f1": None, "comet": None, "task_back_en": None}
    for lg in targets:
        r = qc[lg]
        out[lg] = {
            "instruction": r["task_tgt"] + "\n" + FIXED_SPEC,
            "primer": r["primer_tgt"],
            "source": "native_nllb" if r["pass"]
                      else "native_nllb_LOW_QC_review",
            "bertscore_f1": round(r["bertscore_f1"], 3),
            "comet": round(r["comet"], 3),
            "task_back_en": r["task_back_en"]}
    JSON_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    recs = sorted((qc[l] for l in targets),
                  key=lambda r: (r["bertscore_f1"] + r["comet"]) / 2)
    n_pass = sum(r["pass"] for r in recs)
    fail = [r for r in recs if not r["pass"]]
    print(f"\n=== v2 SUMMARY ({len(recs)} NLLB targets) ===")
    print(f"pass (BERT>={BERT_T} AND COMET>={COMET_T}): {n_pass}/{len(recs)}")
    print(f"need review/manual (failed): {len(fail)} -> "
          f"{[r['lang'] for r in fail]}")
    print(f"\n--- all failures (BERT, COMET, back-translation) ---")
    for r in fail:
        print(f"{r['lang']:9s} {r['bertscore_f1']:.2f} {r['comet']:5.2f}  "
              f"{r['task_back_en'][:95]}")
    print(f"\n--- 5 best ---")
    for r in recs[-5:]:
        print(f"{r['lang']:9s} {r['bertscore_f1']:.2f} {r['comet']:5.2f}  "
              f"{r['task_back_en'][:95]}")
    print(f"\nwrote {QC_OUT}\nwrote {JSON_OUT}")
    print(f"manual CSV (6 romanized{' + add failures' if fail else ''}): "
          f"{MANUAL_CSV}")


if __name__ == "__main__":
    main()
