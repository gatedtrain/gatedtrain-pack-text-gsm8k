"""The GSM8K pack's batch generator (gatedtrain AN-25): standard library plus vLLM, run in the keeper's inference
container as

    python -I generate_vllm.py --items ITEMS --sampler SAMPLER --model MODEL_DIR --out OUT_DIR

It is copied into the container on its own, so it imports nothing from the pack and embeds the prompt. It reads the
items (canonical `{"item_id", "input"}` lines, the input being the GSM8K question), the sampler and seeds
(`{"sampler", "seeds"}`) and the model directory, and writes `OUT_DIR/predictions.jsonl`: one canonical JSON line
`{"item_id", "prediction", "seed"}` per seed and item, in (seed, item) order, the prediction being the completion
text. The lines use the writer recipe of `gatedtrain.ainative.batch`, canonical for string predictions.

The container root is read-only with only /tmp writable and no network, so every cache is pointed below /tmp before
vLLM is imported. A sampler outside the deployment keys, or one that does not turn thinking off, is refused with
exit code 2 before vLLM is loaded, and nothing is written. Results are matched to items by position: `LLM.generate`
returns its outputs in prompt order whatever order the requests finish in.

Once the predictions file is written, flushed, synced and closed, the process ends at once with `os._exit(0)`:
vLLM's engine teardown at interpreter exit is known to hang, and the keeper's time limit for the container would be
spent waiting for it. The refusal (exit code 2) and errors (an exception, exit code 1) exit as usual.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROMPT = ("Solve the following math problem step by step. Put your final numerical answer after 'The answer is'."
          "\n\nProblem: {}")
SAMPLER_KEYS = frozenset({"name", "temperature", "top_p", "max_new_tokens", "enable_thinking"})
PREDICTIONS = "predictions.jsonl"

_ENV_DEFAULTS = (
    ("VLLM_CACHE_ROOT", "/tmp/cache/vllm"),
    ("TRITON_CACHE_DIR", "/tmp/cache/triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "/tmp/cache/inductor"),
    ("XDG_CACHE_HOME", "/tmp/cache"),
    ("HF_HOME", "/tmp/cache/hf"),
    ("VLLM_USE_FLASHINFER_SAMPLER", "0"),
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
)


class _Refusal(Exception):
    pass


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_sampler(doc) -> tuple[dict, list]:
    """The sampler and seeds of `doc`, or _Refusal with a message that quotes no item."""
    if not isinstance(doc, dict):
        raise _Refusal("the sampler file is not an object with sampler and seeds")
    sampler, seeds = doc.get("sampler"), doc.get("seeds")
    if not isinstance(sampler, dict):
        raise _Refusal("the sampler is not an object")
    extra = sorted(set(sampler) - SAMPLER_KEYS)
    if extra:
        raise _Refusal(f"the sampler has keys outside {sorted(SAMPLER_KEYS)}: {extra}")
    if sampler.get("enable_thinking") is not False:
        raise _Refusal("the sampler must set enable_thinking to false")
    for key in ("temperature", "top_p"):
        if not _number(sampler.get(key)):
            raise _Refusal(f"the sampler's {key} must be a number")
    max_new = sampler.get("max_new_tokens")
    if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 1:
        raise _Refusal("the sampler's max_new_tokens must be a positive integer")
    if not isinstance(seeds, list) or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise _Refusal("the seeds must be a list of integers")
    return sampler, seeds


def _line(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    for name in ("--items", "--sampler", "--model", "--out"):
        ap.add_argument(name, required=True)
    args = ap.parse_args(argv)
    with open(args.sampler, encoding="utf-8") as f:
        sampler_doc = json.load(f)
    try:
        sampler, seeds = _check_sampler(sampler_doc)
    except _Refusal as exc:
        print(f"generate_vllm: refused: {exc}", file=sys.stderr)
        return 2
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f.read().split("\n") if line]

    for key, value in _ENV_DEFAULTS:
        os.environ.setdefault(key, value)
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", generation_config="vllm", max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True, seed=0)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": PROMPT.format(item["input"])}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for item in items]
    lines = []
    for seed in seeds:
        params = SamplingParams(temperature=sampler["temperature"], top_p=sampler["top_p"],
                                max_tokens=sampler["max_new_tokens"], seed=seed)
        outputs = llm.generate(prompts, params)
        if len(outputs) != len(items):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(items)} prompts")
        lines.extend(_line({"item_id": item["item_id"], "prediction": out.outputs[0].text, "seed": seed})
                     for item, out in zip(items, outputs))

    fd = os.open(os.path.join(args.out, PREDICTIONS), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(b"".join(lines))
        f.flush()
        os.fsync(f.fileno())
    return 0


if __name__ == "__main__":
    code = main()
    if code == 0:
        # Nothing after the write matters; skip vLLM's teardown (module docstring).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    sys.exit(code)
