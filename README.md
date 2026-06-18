<div align="center">

# 🌍⚡ The Language–Energy Divide

### Measuring Energy Costs of Multilingual LLM Inference

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-HuggingFace-ff9d00.svg)](https://huggingface.co/datasets/MichiganNLP/language-energy-divide)
[![Leaderboard](https://img.shields.io/badge/⚡_ML.ENERGY-Leaderboard-23d175.svg)](https://ml.energy/leaderboard)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776ab.svg)](https://www.python.org/)

</div>

---

LLMs are increasingly deployed in multilingual settings, yet the **energy cost** of serving
them across languages is poorly understood. We present the first systematic measurement of
inference energy across **122 languages**, and uncover a stark, persistent **language–energy divide**.

## 🔑 Key findings

| | Finding |
|---|---|
| ⚡ | **Per-token energy** varies by up to **8.3×** across languages (single model). |
| 🔋 | **Total energy** for a fixed request set varies by up to **179×** — English (17.6 kJ) vs. Pashto (3,147 kJ). |
| 📉 | **Double penalty:** the most energy-expensive languages are also the *least* accurate. |
| 🌐 | The divide **persists across 5 models, 2 GPUs, 6 batch sizes, and 3 tasks**. |

The disparity compounds two factors: higher **per-token energy** for complex/rare scripts, and
**more tokens generated** for low-resource languages.

## 📊 What's in this repo

```
.
├── analysis/   # Scripts that build the paper's tables & figures, plus the
│               #   prompt-construction / NLLB translation / QC pipeline
├── results/    # Canonical per-language result CSVs (the numbers in the paper)
└── README.md
```

The **full data** — per-language prompts, energy/accuracy/token measurements, and the
translated GSM8K & LM-Arena prompts — lives on the 🤗 [Hugging Face dataset](https://huggingface.co/datasets/MichiganNLP/language-energy-divide).
Energy is measured with the [ML.ENERGY Benchmark](https://ml.energy/leaderboard) (vLLM serving + the [Zeus](https://ml.energy/zeus) energy-measurement library), reporting **steady-state** per-token energy.

### `results/` files

| File | Contents |
|------|----------|
| `belebele_canonical_Qwen3-8B_0shot.csv` | Main experiment: 122 languages × {energy/token, output tokens, total energy, accuracy} |
| `belebele_canonical_{Qwen3-14B,Qwen3-8B,Llama-3.1-8B,Llama-3.1-70B}.csv` | Cross-model per-language energy |
| `cross_model_12lang_table.csv` | Cross-model comparison table |
| `crosstask_seqs256_summary.csv` | Cross-task (Belebele / GSM8K / LM-Arena), 8-language subset |
| `belebele_seqsweep_l40s_0shot.csv` | Batch-size sweep, L40S, 8 languages |
| `seqsweep_belebele_v1_qwen8b_RTX6000_all8.csv` | Batch-size sweep, RTX 6000 Pro Blackwell, 8 languages |

## 🔬 Experimental setup

- **Models:** Qwen3-8B / 14B / 32B, gemma-3-27b-it, Llama-3.1-8B-Instruct (cross-model).
- **Serving:** vLLM, batch size (`max_num_seqs`) = 256 unless noted; 32B & 27B use pipeline parallelism (degree 2).
- **Hardware:** NVIDIA L40S (48 GB) and RTX 6000 Pro Blackwell (96 GB).
- **Tasks:** Belebele (reading comprehension, 122 langs), translated GSM8K (math), LM-Arena (open chat).
- **Prompting:** zero-shot chain-of-thought; per-language instructions/primers translated with NLLB-200 and quality-controlled by back-translation (BERTScore ≥ 0.85, COMET ≥ 0.75), hand-curated where the automated pipeline failed.

## ▶️ Reproducing the tables & figures

```bash
pip install -r requirements.txt
# Scripts read run outputs from a configurable root (set REPO_ROOT / RUN_ROOT),
# or operate on the CSVs in results/. Examples:
python analysis/build_cross_model_table.py
python analysis/build_crosstask_table_0shot.py
python analysis/build_cross_gpu_table.py
python analysis/build_seqsweep_figs.py
```

To re-run the energy measurements from scratch, use the
[ML.ENERGY Benchmark](https://github.com/ml-energy/leaderboard) with the models, GPUs, and
batch sizes above.

## 📚 Citation

```bibtex
@article{language-energy-divide,
  title  = {The Language--Energy Divide: Measuring Energy Costs of Multilingual LLM Inference},
  author = {Deng, Naihao and Shen, Alissa and Feng, Yiming and Nwatu, Joan and
            Chung, Jae-Won and Chowdhury, Mosharaf and Chen, Yulong and Mihalcea, Rada},
  year   = {2026}
}
```

This work builds on the **ML.ENERGY Benchmark** ([Chung et al., NeurIPS 2025 D&B](https://arxiv.org/abs/2505.06371))
and the **Belebele** dataset ([Bandarkar et al., ACL 2024](https://aclanthology.org/2024.acl-long.44/)).

## 🙏 Acknowledgements

Energy measurement is powered by [Zeus](https://ml.energy/zeus) and the
[ML.ENERGY](https://ml.energy) project. Multilingual data from
[Belebele](https://github.com/facebookresearch/belebele); translation via
[NLLB-200](https://arxiv.org/abs/2207.04672).
