# Third-party notices

This pack contains no ported third-party code apart from one prompt string (below). Its test fixtures are synthetic
rows in GSM8K's shape, not GSM8K data. The lock files (`runtime/uv.lock`, `eval/uv.lock`, `eval-requirements.txt`) name
third-party packages that are installed from PyPI under their own licenses; nothing of them is copied here.

## NeMo Evaluator GSM8K prompt

- **What:** `src/gatedtrain_pack_text_gsm8k/prompt.txt` holds NeMo Evaluator's GSM8K prompt text, with its
  `{question}` placeholder written `{}`, so that the training prompt equals the prompt gatedtrain uses for GSM8K.
- **Source:** NeMo Evaluator, <https://github.com/NVIDIA-NeMo/Evaluator>, `src/nemo_evaluator/benchmarks/gsm8k.py`
  lines 22-26 at commit `ea35ba36`. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
- **License:** Apache-2.0 (the same license as this pack, text in `LICENSE`). The project has no NOTICE file at that
  commit.
