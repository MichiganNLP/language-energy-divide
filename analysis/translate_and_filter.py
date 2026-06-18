"""Per-language translation worker.

For one target language, on one GPU:
  1. Load NLLB-200-3.3B.
  2. For each candidate prompt: forward-translate en -> target.
  3. Back-translate target -> en.
  4. Score (original_en, back_en) with BERTScore (roberta-large baseline).
  5. Score (target, back_en, original_en) with COMET wmt22-comet-da.
  6. Write a JSONL with all prompts and their QC scores.

Filtering is NOT applied here. The orchestrator computes the
intersection of survivors across languages in a separate step.

Run:
    CUDA_VISIBLE_DEVICES=0 python3 translate_and_filter.py \\
        --target-lang zho_Hans \\
        --candidates research/lmarena_benchmark/candidates.jsonl \\
        --out research/lmarena_translate/zho_Hans_scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_candidates(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def translate(texts, model, tokenizer, src_lang, tgt_lang, device, batch_size=8,
              max_length=512):
    """Translate a list of strings using NLLB."""
    import torch
    tokenizer.src_lang = src_lang
    out = []
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    for batch in batched(texts, batch_size):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(device)
        with torch.inference_mode():
            ids = model.generate(
                **enc,
                forced_bos_token_id=tgt_token_id,
                max_length=max_length,
                num_beams=1,  # greedy for speed
            )
        out.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-lang", required=True,
                    help="NLLB target lang code, e.g. zho_Hans, rus_Cyrl, wol_Latn, bod_Tibt")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of candidates (for quick testing).")
    args = ap.parse_args()

    assert os.environ.get("HF_HOME"), "Set HF_HOME."
    assert os.environ.get("HF_TOKEN"), "Set HF_TOKEN."

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from bert_score import score as bert_score_fn
    from comet import download_model, load_from_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.target_lang}] device={device}, "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES','all')}")

    candidates = load_candidates(args.candidates)
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"[{args.target_lang}] candidates: {len(candidates)}")

    print(f"[{args.target_lang}] loading NLLB-3.3B ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B",
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[{args.target_lang}]   loaded in {time.time() - t0:.1f}s")

    prompts = [c["prompt"] for c in candidates]

    print(f"[{args.target_lang}] forward translate en -> {args.target_lang} ...")
    t0 = time.time()
    tgt_texts = translate(prompts, model, tokenizer,
                          src_lang="eng_Latn", tgt_lang=args.target_lang,
                          device=device, batch_size=args.batch_size,
                          max_length=args.max_length)
    print(f"[{args.target_lang}]   done in {time.time() - t0:.1f}s")

    print(f"[{args.target_lang}] back translate {args.target_lang} -> en ...")
    t0 = time.time()
    back_texts = translate(tgt_texts, model, tokenizer,
                           src_lang=args.target_lang, tgt_lang="eng_Latn",
                           device=device, batch_size=args.batch_size,
                           max_length=args.max_length)
    print(f"[{args.target_lang}]   done in {time.time() - t0:.1f}s")

    # Free NLLB before loading COMET (both are big).
    del model
    torch.cuda.empty_cache()

    print(f"[{args.target_lang}] BERTScore ...")
    t0 = time.time()
    P, R, F1 = bert_score_fn(back_texts, prompts, lang="en", verbose=False,
                             rescale_with_baseline=False, device=device,
                             batch_size=32)
    bert_f1 = F1.tolist()
    print(f"[{args.target_lang}]   done in {time.time() - t0:.1f}s")

    print(f"[{args.target_lang}] COMET ...")
    t0 = time.time()
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)
    comet_inputs = [
        {"src": tgt, "mt": back, "ref": orig}
        for tgt, back, orig in zip(tgt_texts, back_texts, prompts)
    ]
    comet_out = comet_model.predict(comet_inputs, batch_size=16,
                                     gpus=1 if device == "cuda" else 0,
                                     progress_bar=False)
    comet_scores = comet_out["scores"]
    print(f"[{args.target_lang}]   done in {time.time() - t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for c, tgt, back, b, co in zip(candidates, tgt_texts, back_texts,
                                        bert_f1, comet_scores):
            rec = {
                "id": c["id"],
                "prompt_en": c["prompt"],
                "prompt_tgt": tgt,
                "back_en": back,
                "bertscore_f1": float(b),
                "comet": float(co),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_pass_bert = sum(1 for b in bert_f1 if b >= 0.85)
    n_pass_comet = sum(1 for co in comet_scores if co >= 0.75)
    n_pass_both = sum(1 for b, co in zip(bert_f1, comet_scores)
                       if b >= 0.85 and co >= 0.75)
    print(f"[{args.target_lang}] passing BERTScore>=0.85: {n_pass_bert}/{len(candidates)}")
    print(f"[{args.target_lang}] passing COMET>=0.75:     {n_pass_comet}/{len(candidates)}")
    print(f"[{args.target_lang}] passing BOTH:            {n_pass_both}/{len(candidates)}")
    print(f"[{args.target_lang}] wrote {out_path}")


if __name__ == "__main__":
    main()
