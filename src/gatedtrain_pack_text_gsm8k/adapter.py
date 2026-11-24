"""Ext adapter `text-gsm8k-grpo`: the launch rules of the pack's GRPO training job (`job.train_job`).

gatedtrain's local backend checks every job against the launch rules of the ext adapter it names
(`gatedtrain.ainative.backend.job_problems`). This module registers that adapter through the `gatedtrain.adapters`
entry point (or in process with `install()`): the training row, modelled on the row of the recipe
gatedtrain-recipe-trl-grpo-gsm8k (`rows.py`), an evaluator row for the pack's scorer, and a small adapter class.

The row differs from the recipe's in three ways: no spec binding (the pack's stages bind the inputs, not an ext
spec), the trainer id `rl.trl_grpo@0.1`, and the input root `{output_root}`, since the backend's directories (the
job's inputs, the base model) live under `{output_root}/gatedtrain-<hex>/`. The other input roots, the interpreter,
its trees and the scan skips are rowkit's host rows for the runtime directory: `GATEDTRAIN_PACK_TEXT_GSM8K_RUNTIME`,
else this package's `runtime/` (which holds the lock the owner builds the venv from). gatedtrain's
`GATEDTRAIN_EXT_*` overrides move them for another layout.

Runs of this adapter are read by the pack's own stages; `gatedtrain ext-collect` does not read them, so the class's
run-reading methods refuse. No framework is imported here (MM-12).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, NoReturn

from gatedtrain.errors import Problem, Refused
from gatedtrain.ext.rowkit import CONTAINER_PATHS, HF_LLM_COMPONENTS, runtime_dir, venv_runtime_rows

NAME = "text-gsm8k-grpo"
TRAINER_ID = "rl.trl_grpo@0.1"
EVALUATOR_ID = "gsm8k.gsm8k_answer@0.1"
FRAMEWORK_PACKAGE = "trl"
RUNTIME_ENV = "GATEDTRAIN_PACK_TEXT_GSM8K_RUNTIME"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__, parents=0)
HOST_ROWS = venv_runtime_rows(RUNTIME, NAME)
OUTPUT_ROOT = "{output_root}"

# The recipe's 18 environment names (its runtime/README.md "Environment variable names"); none is a credential.
ENV_NAMES = ("DO_NOT_TRACK", "FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HF_HUB_OFFLINE", "HOME", "MASTER_ADDR",
             "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM", "TORCHINDUCTOR_CACHE_DIR",
             "TORCH_CUDA_ARCH_LIST", "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
             "VLLM_CONFIG_ROOT", "VLLM_NO_USAGE_STATS", "VLLM_USE_FLASHINFER_SAMPLER", "XDG_CACHE_HOME")
# The names whose value is a directory (homes, caches); each value lies under /tmp or /out (EXT-114).
ENV_PATHS = ("FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
             "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "XDG_CACHE_HOME")

TRAINER_ROWS: dict[str, dict[str, Any]] = {NAME: {
    "uses": (TRAINER_ID,),
    "pins": ("trl", "vllm"),
    "outputs": ("checkpoint-<step>/ HF directories", "trainer_state.json", "samples.jsonl", "memory.jsonl"),
    "components": HF_LLM_COMPONENTS,
    "consumption": {"target": "stage", "fallback": None},
    "gpu_h_estimate": 1.5,                           # the full profile (job.PROFILES)
    "evaluators": (EVALUATOR_ID,),
    "framework": FRAMEWORK_PACKAGE,
    # EXT-79: the interpreter install directory and the runtime directory (rowkit host rows), and the whole output
    # root, under which the backend keeps the job's inputs and the base model. The filter drops rowkit's
    # `{output_root}/<name>` (a recipe's own directory under the output root), which `{output_root}` already covers.
    "input_roots": tuple(r for r in HOST_ROWS["input_roots"] if not r.startswith(OUTPUT_ROOT)) + (OUTPUT_ROOT,),
    "spec_binding": False,
    "env_names": ENV_NAMES,
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],
    "container_paths": CONTAINER_PATHS,
    "env_paths": ENV_PATHS,
}}

# The pack's scorer (answer.py `score`): 1 when the response's number equals the label's, else 0.
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {EVALUATOR_ID: {
    "metric": "accuracy",
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "scorer": {"name": "gsm8k_answer", "version": "gatedtrain-pack-text-gsm8k@0.1", "weights_sha256": None,
               "code_sha256": "sha256:" + hashlib.sha256((Path(__file__).parent / "answer.py").read_bytes()).hexdigest()},
    "tier": "T3",                                    # not the GRPO training reward (math-verify)
    "binary": True,
    "error_classes": ("missing_score",),
    "score_outcomes": ("wrong_number", "no_number_in_response"),
    "payload_text_field": "question",
    "item_payload": ("question", "answer"),
    "normalizer": False,
    "env_version": False,
    "framework": "gatedtrain-pack-text-gsm8k",
    "samples": "items n, 4 seeds",
}}

PLUGIN = {"adapters": {NAME: "gatedtrain_pack_text_gsm8k.adapter:TextGsm8kGrpoAdapter"},
          "trainer_rows": TRAINER_ROWS, "evaluator_rows": EVALUATOR_ROWS}


def install() -> None:
    """Register the plugin in this process when the entry point has not (idempotent)."""
    from gatedtrain import spec
    from gatedtrain.ext import registry

    if registry.ADAPTERS.get(NAME) != PLUGIN["adapters"][NAME]:
        registry.register(PLUGIN, "gatedtrain_pack_text_gsm8k.adapter.PLUGIN")
    spec._external_ids()


def _not_collected(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise Refused([Problem("EXT-12", "adapter", f"runs of the ext adapter {NAME} are read by the "
                           "gatedtrain-pack-text-gsm8k stages, not by gatedtrain ext-collect",
                           "run the pack's pipeline (gatedtrain's AI-native commands)")])


class TextGsm8kGrpoAdapter:
    """The adapter interface (MM-12) for the pack's training job: its identity and metric; the run-reading methods
    refuse (module docstring)."""

    name = NAME
    trainer_id = TRAINER_ID
    evaluator_ids = (EVALUATOR_ID,)
    framework_name = FRAMEWORK_PACKAGE
    binds_spec = False
    checkpoint_completeness = {
        "library": "transformers Trainer (TRL GRPOTrainer subclass)",
        "rule": "not claimed: ext-collect does not read this adapter's runs",
        "complete_dir_pattern": r"^checkpoint-(\d+)$",
        "sources": [],
        "verified": False,
    }
    seed_derivation = ("the job's training_seed is passed to train_grpo.py --seed, GRPOConfig.seed, which TRL hands "
                       "to transformers set_seed before the model is built")

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    identity = model_dir_problem = checkpoint_dirs = find_cuts = cut_problem = checks = _not_collected
    consumption = framework_identity = script_problems = mount_binding_problems = _not_collected
