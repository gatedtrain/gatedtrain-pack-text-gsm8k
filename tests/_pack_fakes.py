"""Fake container work for the pack's tests: what the training job and the keeper's inference container write.

- `play_training` plays `train_grpo.py` in a job's container: it writes `samples.jsonl` through the script's own
  `GSM8KReward._log`, `memory.jsonl` with the memory callback's keys, and `checkpoint-<N>/` directories whose files
  come out 0600, then (with `chmod`) applies the script's own `chmod_checkpoint`. A model file holds
  `<profile> step <N> <name>`, so the full run's `model.safetensors` starts with `full step`.
- `play_inference` plays `generate_vllm.py` in the keeper's inference container: one canonical prediction line per
  seed and item in `/out/predictions.jsonl`, "The answer is <label>" for a model whose `model.safetensors` starts
  with `full step` (the candidate) and "The answer is 0" for any other model (the base). The labels come from an
  intake data file (`{"input", "label"}` lines) keyed by input, since the container's items carry inputs only.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from gatedtrain_pack_text_gsm8k import job

SIX = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
       "chat_template.jinja")
CANDIDATE_MARK = b"full step"
# The files of Qwen/Qwen3-0.6B at the pinned commit c1899de: no chat_template.jinja, the template is the string field
# `chat_template` of tokenizer_config.json.
HF_FILES = ("config.json", "generation_config.json", "LICENSE", "merges.txt", "model.safetensors",
            "tokenizer_config.json", "tokenizer.json", "vocab.json")
HF_TEMPLATE = ("{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
               "{%- endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{%- if enable_thinking is defined "
               "and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}{%- endif %} — café")


def write_hf_hub(hub: Path, template=HF_TEMPLATE) -> None:
    """A model directory in the layout of the real hub download (HF_FILES): `tokenizer_config.json` carries
    `chat_template` (left out when `template` is None), a Qwen3 `config.json`, the other files placeholders."""
    hub.mkdir(parents=True, exist_ok=True)
    for name in HF_FILES:
        (hub / name).write_text(f"hub {name}\n")
    (hub / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    tok = {"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 131072}
    if template is not None:
        tok["chat_template"] = template
    (hub / "tokenizer_config.json").write_text(json.dumps(tok, ensure_ascii=False, indent=2), encoding="utf-8")


def load_script():
    """The pack's train_grpo.py by path (its module level imports the standard library only)."""
    spec_ = importlib.util.spec_from_file_location("pack_train_grpo_for_fakes", job.TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


SCRIPT = load_script()


def play_training(run_spec, host_path, *, rewards=(1.0, 0.0) * 4, texts=None, peak_gb=19.8, chmod=True) -> int:
    """The files `train_grpo.py --profile <smoke|full>` writes at the top of /out (module docstring)."""
    argv = list(run_spec.argv)
    profile = argv[argv.index("--profile") + 1]
    out = host_path("/out")
    assert host_path("/inputs/base-model/model.safetensors").is_file()
    steps, save = (10, 5) if profile == "smoke" else (100, 25)
    reward = SCRIPT.GSM8KReward(out / "samples.jsonl", 8)
    for step in range(steps):
        rows = [r for r in range(2) for _ in range(8)]
        step_texts = texts or [f"step {step} row {r} completion {i}" for r in range(2) for i in range(8)]
        reward._log(rows=rows, rewards=list(rewards) * 2, state=SimpleNamespace(global_step=step, epoch=0.0),
                    texts=step_texts)
        with open(out / "memory.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step + 1, "utc": "2026-11-23T00:00:00Z", "seconds": 40.0,
                                "device_used_gb_at_step_end": 12.0, "torch_max_allocated_gb": 10.0,
                                "torch_max_reserved_gb": peak_gb if step == 3 else 11.0}) + "\n")
        if (step + 1) % save == 0:
            cut = out / f"checkpoint-{step + 1}"
            os.mkdir(cut, 0o700)
            for name in SIX + ("optimizer.pt", "trainer_state.json"):
                fd = os.open(cut / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.write(fd, f"{profile} step {step + 1} {name}\n".encode())
                os.close(fd)
            os.chmod(cut, 0o700)
            if chmod:
                SCRIPT.chmod_checkpoint(cut)
    print("trained", profile)
    return 0


def labels_by_input(data: Path) -> dict:
    """`{input: label}` of an intake data file (split on "\\n" only)."""
    return {row["input"]: row["label"] for row in
            (json.loads(line) for line in data.read_text(encoding="utf-8").split("\n") if line)}


def play_inference(run_spec, host_path, labels: dict, sampler_log: Path | None = None) -> int:
    """The predictions `generate_vllm.py` writes (module docstring); exit 2 for a sampler that keeps thinking on. With
    `sampler_log`, the sampler document the keeper passed is appended to it as one JSON line."""
    argv = list(run_spec.argv)
    arg = lambda flag: host_path(argv[argv.index(flag) + 1])          # noqa: E731
    doc = json.loads(arg("--sampler").read_text(encoding="utf-8"))
    if sampler_log is not None:
        with open(sampler_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, sort_keys=True) + "\n")
    if doc["sampler"].get("enable_thinking") is not False:
        return 2
    items = [json.loads(line) for line in arg("--items").read_text(encoding="utf-8").split("\n") if line]
    candidate = (arg("--model") / "model.safetensors").read_bytes().startswith(CANDIDATE_MARK)
    lines = []
    for seed in doc["seeds"]:
        for item in items:
            answer = labels[item["input"]] if candidate else "0"
            lines.append(json.dumps({"item_id": item["item_id"], "prediction": f"The answer is {answer}",
                                     "seed": seed}, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False))
    (arg("--out") / "predictions.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return 0
