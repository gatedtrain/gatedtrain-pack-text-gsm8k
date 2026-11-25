"""Stages of the GSM8K scenario pack. Every stage's registry name is prefixed `gsm8k_`.

`gsm8k_choose_task_and_base` copies the six deployable files of the base model into a backend output directory;
`gsm8k_smoke` and `gsm8k_train` run the pack's `train_grpo.py` as GPU jobs of the project's backend (`ctx.backend`,
job.py `train_job`): smoke applies the MM-38 checks to the files the script writes at the top of /out, train hands the
last checkpoint's six deployable files off as the candidate. Pack refusals use the rules `TXT-<n>`: TXT-1 training
rows, TXT-2 base model, TXT-3 training output, TXT-4 job parameters (as job.py), TXT-5 recipe; smoke refusals MM-38.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import stat
from decimal import Decimal
from pathlib import Path
from typing import Any

from gatedtrain.ainative.project import ArtifactRef
from gatedtrain.ainative.stage import Proposal, StageContext
from gatedtrain.ainative.trust_stages import SCHEMA_DEPLOYMENT, SCHEMA_MODEL_REF, SCHEMA_TRAIN_PART
from gatedtrain.errors import Problem, Refused
from gatedtrain.ext import launch
from gatedtrain.ext.smoke import GPU_MEM_LIMIT_GB

from . import job as job_mod
from .answer import label_ok, normalise

SCHEMA_VERSION = 1
SCHEMA_PROFILE = "gatedtrain_pack_text_gsm8k.profile"
SCHEMA_TRAIN_SET = "gatedtrain_pack_text_gsm8k.train_set"
SCHEMA_RECIPE = "gatedtrain_pack_text_gsm8k.recipe"
SCHEMA_SMOKE = "gatedtrain_pack_text_gsm8k.smoke"
PERCENTILES = (("p50", 50), ("p90", 90), ("p99", 99))

GENERATOR = "text_gsm8k_vllm"           # the batch generator the scenario registers
ARCHITECTURE = "Qwen3ForCausalLM"
# The only files a model reference lists (and the base copy holds).
DEPLOYABLE = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja")
# A source without it (the hub's Qwen3-0.6B at the pinned commit) has its template as the string `chat_template`
# of tokenizer_config.json; ChooseTaskAndBase writes that string to the base copy's chat_template.jinja.
TEMPLATE = "chat_template.jinja"
# The deployment decodes greedily by default (temperature 0, one evaluation seed, one sample per item): gatedtrain
# registers it as `decoding: greedy` and the keeper checks it with a control seed (AN-26). Sampled decoding
# (temperature 0.7 over four seeds, the pack's 0.1.0 mode) stays available with `decoding: sampled`.
GREEDY_SAMPLER = {"name": "vllm", "decoding": "greedy", "temperature": 0, "top_p": 1.0, "max_new_tokens": 1024,
                  "enable_thinking": False}
GREEDY_SEEDS = [11]
SAMPLED_SAMPLER = {"name": "vllm", "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 1024, "enable_thinking": False}
SAMPLED_SEEDS = [11, 12, 13, 14]
DECODINGS = {"greedy": (GREEDY_SAMPLER, GREEDY_SEEDS), "sampled": (SAMPLED_SAMPLER, SAMPLED_SEEDS)}
SAMPLER, EVAL_SEEDS = GREEDY_SAMPLER, GREEDY_SEEDS
# train_grpo.py PROFILES (max_steps); `train_job` passes no --max-steps, so these are the only steps a job runs.
PROFILE_STEPS = {"smoke": 10, "full": 100}
TRAINING_SEED = 42                      # train_grpo.py COMMON["seed"]
JOB_POLL_S = 30.0
# MM-38 smoke rules. Memory: memory.jsonl holds GB of 10^9 bytes (train_grpo.py make_memory_callback: `/ 1e9`), the
# bound is in the same unit, so no conversion; the peak is the largest of the device memory in use at a step's end
# and torch's peak allocated and reserved memory of the step.
MEMORY_LIMIT_GB = GPU_MEM_LIMIT_GB       # gatedtrain's default smoke bound (a 24 GB GPU's); the owner's is the host file's
MEMORY_KEYS = ("device_used_gb_at_step_end", "torch_max_allocated_gb", "torch_max_reserved_gb")
MIN_SAMPLES = 8
INPUTS_DIR = "train-inputs"
BASE_DIR = "base-model"
_CHECKPOINT = re.compile(r"checkpoint-([0-9]+)")
_CHUNK = 1 << 20


def _refuse(rule: str, field_: str, message: str, fix: str) -> Refused:
    return Refused([Problem(rule, field_, message, fix)])


def _percentile(sorted_vals: list[int], pct: int) -> int:
    """Nearest-rank percentile of a non-empty sorted list."""
    rank = -(-pct * len(sorted_vals) // 100)
    return sorted_vals[max(rank, 1) - 1]


class ProfileData:
    name = "gsm8k_profile_data"
    version = "0.1"
    inputs = {"train_part": SCHEMA_TRAIN_PART}
    outputs = {"profile": SCHEMA_PROFILE}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def _profile(self, ctx: StageContext) -> dict:
        rec = ctx.project.read_artifact(ctx.inputs["train_part"])
        rows = rec.get("rows") if isinstance(rec, dict) else None
        if not (isinstance(rows, list) and rows and all(isinstance(r, dict) and isinstance(r.get("input"), str)
                                                        and "label" in r for r in rows)):
            raise _refuse("TXT-1", "inputs.train_part.rows", "the training rows are missing, empty or malformed",
                          "rerun the stage that writes the train part")
        bad = [i for i, r in enumerate(rows) if not label_ok(r["label"])]
        if bad:
            shown = ", ".join(str(i) for i in bad[:10])
            raise _refuse("TXT-1", "inputs.train_part.rows", f"{len(bad)} of {len(rows)} labels are not numbers "
                          f"(row indexes from 0: {shown}{', ...' if len(bad) > 10 else ''})",
                          "fix the labels in the source data and intake it again")
        values = [Decimal(normalise(r["label"])) for r in rows]
        lengths = sorted(len(r["input"]) for r in rows)
        return {"schema": SCHEMA_PROFILE, "schema_version": SCHEMA_VERSION, "n": len(rows),
                "label_min": normalise(format(min(values), "f")), "label_max": normalise(format(max(values), "f")),
                "n_non_integer_labels": sum(1 for v in values if v != v.to_integral_value()),
                "question_chars": {"min": lengths[0], "max": lengths[-1],
                                   **{k: _percentile(lengths, p) for k, p in PERCENTILES}}}

    def plan(self, ctx: StageContext) -> Proposal:
        p = self._profile(ctx)
        return Proposal(self.name, f"profile {p['n']} training rows: labels {p['label_min']} to {p['label_max']}, "
                        f"question length p50 {p['question_chars']['p50']} chars", {}, {"n": p["n"]})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"profile": ctx.project.write_artifact(ctx.stage or self.name, "profile", self._profile(ctx),
                                                      SCHEMA_PROFILE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        p = ctx.project.read_artifact(outputs["profile"])
        n = ctx.project.read_artifact(ctx.inputs["train_part"]).get("rows")
        if not isinstance(p, dict) or p.get("n") != (len(n) if isinstance(n, list) else None):
            return [Problem("TXT-1", "profile.n", "the row count differs from the train part", "rerun gsm8k_profile_data")]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------

def _meta(schema: str) -> dict:
    return {"schema": schema, "schema_version": SCHEMA_VERSION}


def _num(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:                 # an integer too large for a float
        return False


def _read(ctx: StageContext, name: str, rule: str) -> dict:
    rec = ctx.project.read_artifact(ctx.inputs[name])
    if not isinstance(rec, dict):
        raise _refuse(rule, f"inputs.{name}", "the input record is not an object", "rerun the stage that writes it")
    return rec


def _sha256_path(path: Path | str, *, follow: bool = True) -> str | None:
    """`sha256:<hex>` of a regular file, None when what the open finds is not one (or, unless `follow`, a link)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | (0 if follow else os.O_NOFOLLOW))
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _same(path: Path, sha256: str) -> bool:
    """`path` is a regular file of mode 0644 (not a link) whose content has this hash."""
    try:
        st = os.lstat(path)
        return (stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o644
                and _sha256_path(path, follow=False) == sha256)
    except OSError:
        return False


def _replace(dst: Path, write) -> str:
    """`dst` replaced (never through a link) by a mode 0644 file that `write(f, h)` fills while updating the hash
    `h`: a temporary file in the same directory, synced and renamed. The new content's `sha256:`."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    h = hashlib.sha256()
    with os.fdopen(fd, "wb") as f:
        write(f, h)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, dst)
    return "sha256:" + h.hexdigest()


def _write_bytes(dst: Path, data: bytes) -> None:
    """`dst` holds `data` (mode 0644). An identical file is left as it is: same inode and mtime, so a job that mounts
    it keeps its key (large inputs are fingerprinted by size and mtime)."""
    if _same(dst, _sha256_bytes(data)):
        return

    def write(f, h):
        f.write(data)
        h.update(data)
    _replace(dst, write)


def _copy(src: Path, dst: Path) -> str:
    """`dst` a mode 0644 copy of `src` (left as it is when already identical); the copy's `sha256:`. TXT-2 when the
    source changed while it was copied."""
    want = _sha256_path(src)
    if want is None:
        raise _refuse("TXT-2", "params.base_model_dir", f"{src.name} is not a regular file",
                      "point base_model_dir at the downloaded model directory")
    if _same(dst, want):
        return want

    def write(f, h):
        with open(src, "rb") as fi:
            for chunk in iter(lambda: fi.read(_CHUNK), b""):
                f.write(chunk)
                h.update(chunk)
    got = _replace(dst, write)
    if got != want:
        raise _refuse("TXT-2", "params.base_model_dir", f"{src.name} changed while it was copied",
                      "leave the base model directory unchanged and run the stage again")
    return got


def _backend(ctx: StageContext, stage: str) -> Any:
    if ctx.backend is None:
        raise _refuse("TXT-4", "backend", f"stage {stage!r} needs the project's job backend (ctx.backend)",
                      "run the stage through the stage machine")
    return ctx.backend


def train_jsonl(rows: list[dict]) -> bytes:
    """The training file train_grpo.py reads (`load_rows`: one JSON object `{question, answer}` per line). JSON
    escapes every line break inside a value (U+2028 and U+2029 too, as ASCII escapes), so lines split on "\n" only."""
    return "".join(json.dumps({"question": r["question"], "answer": r["answer"]}, ensure_ascii=True, sort_keys=True,
                              separators=(",", ":")) + "\n" for r in rows).encode("ascii")


def _package_hashes() -> dict[str, str]:
    return {"script_sha256": _sha256_path(job_mod.TRAIN_SCRIPT), "prompt_sha256": _sha256_path(job_mod.PROMPT_FILE)}


def _model_files(ref: dict, field_: str, rule: str, fix: str) -> Path:
    """The directory holding the six deployable files a model reference lists (in DEPLOYABLE order)."""
    files = ref.get("files")
    names = [Path(str(f.get("path"))).name for f in files if isinstance(f, dict)] if isinstance(files, list) else []
    if names != list(DEPLOYABLE) or not all(isinstance(f.get("sha256"), str) for f in files):
        raise _refuse(rule, f"{field_}.files", f"the model reference does not list the six deployable files "
                      f"({', '.join(DEPLOYABLE)}) with their hashes", fix)
    dirs = {os.path.dirname(str(f["path"])) for f in files}
    if len(dirs) != 1 or not os.path.isabs(next(iter(dirs))):
        raise _refuse(rule, f"{field_}.files", "the model files do not lie in one absolute directory", fix)
    return Path(next(iter(dirs)))


def _changed_files(ref: dict) -> list[str]:
    """The names of the listed files that are no longer regular files with the listed hash."""
    bad = []
    for f in ref["files"]:
        try:
            ok = _sha256_path(f["path"], follow=False) == f["sha256"]
        except OSError:
            ok = False
        if not ok:
            bad.append(Path(f["path"]).name)
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# Base model and deployment
# ---------------------------------------------------------------------------------------------------------------------

class ChooseTaskAndBase:
    """Task: GSM8K final numeric answers. Base: Qwen3 from `params.base_model_dir` (a model directory downloaded by
    the owner); its six deployable files are copied (mode 0644) into the backend output directory `base-model`, where
    the keeper user can read them, and listed with their hashes in the `base` model reference. A source without
    chat_template.jinja (the hub's download at the pinned commit) gets it from the non-empty string `chat_template` of
    its tokenizer_config.json, written byte for byte (UTF-8); the source directory is not changed."""

    name = "gsm8k_choose_task_and_base"
    version = "0.1"
    inputs = {"profile": SCHEMA_PROFILE}
    outputs = {"deployment": SCHEMA_DEPLOYMENT, "base": SCHEMA_MODEL_REF}
    needs: frozenset[str] = frozenset()
    params = frozenset({"base_model_dir", "decoding"})
    trains = False

    def _source(self, ctx: StageContext) -> Path:
        d = ctx.params.get("base_model_dir")
        fix = "set base_model_dir to the absolute path of the Qwen3 model directory downloaded on the host"
        if not (isinstance(d, str) and os.path.isabs(d) and os.path.isdir(d)):
            raise _refuse("TXT-2", "params.base_model_dir", f"base_model_dir {d!r} is not an absolute path of a "
                          "directory", fix)
        src = Path(d)
        missing = [n for n in DEPLOYABLE if not (src / n).is_file() and n != TEMPLATE]
        if not missing and not (src / TEMPLATE).is_file():
            self._template(src)                 # refuses when tokenizer_config.json has no template either
        if missing:
            raise _refuse("TXT-2", "params.base_model_dir", f"the base model directory lacks {', '.join(missing)} "
                          f"(a deployable model has {', '.join(DEPLOYABLE)})",
                          "download the complete model at the pinned commit")
        try:
            arch = json.loads((src / "config.json").read_text(encoding="utf-8")).get("architectures")
        except (OSError, ValueError, AttributeError):
            arch = None
        if not (isinstance(arch, list) and ARCHITECTURE in arch):
            raise _refuse("TXT-2", "params.base_model_dir", f"config.json does not name {ARCHITECTURE} in "
                          "`architectures`", "point base_model_dir at the Qwen3 model directory")
        return src

    @staticmethod
    def _template(src: Path) -> bytes:
        """The UTF-8 bytes of the string `chat_template` of `src/tokenizer_config.json`, for a source without
        chat_template.jinja (the hub's Qwen3-0.6B at the pinned commit keeps its template there). TXT-2 when the
        file is unreadable or the field is not a non-empty string."""
        fix = ("download the complete model at the pinned commit: its tokenizer_config.json holds the chat template "
               f"as the string `chat_template` (or the directory holds {TEMPLATE})")
        try:
            tok = json.loads((src / "tokenizer_config.json").read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            raise _refuse("TXT-2", "params.base_model_dir", f"the base model directory has no {TEMPLATE} and its "
                          "tokenizer_config.json is not readable JSON", fix) from None
        template = tok.get("chat_template") if isinstance(tok, dict) else None
        if not (isinstance(template, str) and template):
            raise _refuse("TXT-2", "params.base_model_dir", f"the base model directory has no {TEMPLATE} and its "
                          "tokenizer_config.json has no non-empty string `chat_template`", fix)
        return template.encode("utf-8")

    @staticmethod
    def _decoding(ctx: StageContext) -> tuple[dict, list]:
        d = ctx.params.get("decoding", "greedy")
        if d not in DECODINGS:
            raise _refuse("TXT-2", "params.decoding", f"decoding {d!r} is not one of {', '.join(DECODINGS)}",
                          "set decoding to greedy (the default) or sampled")
        sampler, seeds = DECODINGS[d]
        return dict(sampler), list(seeds)

    def plan(self, ctx: StageContext) -> Proposal:
        n = ctx.project.read_artifact(ctx.inputs["profile"]).get("n")
        src = self._source(ctx)
        sampler, seeds = self._decoding(ctx)
        how = "" if (src / TEMPLATE).is_file() else \
            f" ({TEMPLATE} written from the chat_template string of tokenizer_config.json)"
        return Proposal(self.name, f"task: GSM8K final numeric answer ({n} training rows); base model: "
                        f"{ARCHITECTURE} from {src}, its {len(DEPLOYABLE)} deployable files copied for the keeper"
                        f"{how}",
                        {"task": "gsm8k", "sampler": sampler, "eval_seeds": seeds},
                        {"gpu_hours": 0.0})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        src = self._source(ctx)
        sampler, seeds = self._decoding(ctx)
        out = _backend(ctx, self.name).new_output_dir(BASE_DIR)
        files = []
        for n in DEPLOYABLE:
            if n == TEMPLATE and not (src / n).is_file():
                data = self._template(src)
                _write_bytes(out / n, data)
                sha = _sha256_bytes(data)
            else:
                sha = _copy(src / n, out / n)
            files.append({"path": str(out / n), "sha256": sha})
        stage = ctx.stage or self.name
        deployment = {**_meta(SCHEMA_DEPLOYMENT), "sampler": sampler, "eval_seeds": seeds}
        base = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "base", "files": files}
        return {"deployment": ctx.project.write_artifact(stage, "deployment", deployment, SCHEMA_DEPLOYMENT),
                "base": ctx.project.write_artifact(stage, "base", base, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        fix = "rerun gsm8k_choose_task_and_base"
        base = ctx.project.read_artifact(outputs["base"])
        dep = ctx.project.read_artifact(outputs["deployment"])
        sampler, seeds = self._decoding(ctx)
        if dep.get("sampler") != sampler or dep.get("eval_seeds") != seeds:
            return [Problem("TXT-2", "deployment", "the deployment is not the pack's sampler and seeds", fix)]
        if base.get("generator") != GENERATOR or base.get("label") != "base":
            return [Problem("TXT-2", "base", f"the base reference is not a {GENERATOR} base model", fix)]
        try:
            _model_files(base, "base", "TXT-2", fix)
        except Refused as exc:
            return list(exc.problems)
        bad = _changed_files(base) + [Path(f["path"]).name for f in base["files"] if os.path.exists(f["path"])
                                      and stat.S_IMODE(os.lstat(f["path"]).st_mode) != 0o644]
        if bad:
            return [Problem("TXT-2", "base.files", f"{', '.join(sorted(set(bad)))}: not the listed mode 0644 file",
                            fix)]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# Training set and recipe
# ---------------------------------------------------------------------------------------------------------------------

class BuildTrainSet:
    """The keeper's training part rows `{input, label}` as the recipe's `{question, answer}` rows (the label
    normalised as the scorer does); the train stages write them as train.jsonl (`train_jsonl`), whose hash the
    record keeps."""

    name = "gsm8k_build_train_set"
    version = "0.1"
    inputs = {"train_part": SCHEMA_TRAIN_PART}
    outputs = {"train_set": SCHEMA_TRAIN_SET}
    needs: frozenset[str] = frozenset()
    params: frozenset[str] = frozenset()
    trains = False

    def _build(self, ctx: StageContext) -> dict:
        rows = _read(ctx, "train_part", "TXT-1").get("rows")
        if not (isinstance(rows, list) and rows and all(isinstance(r, dict) and isinstance(r.get("input"), str)
                                                        and "label" in r for r in rows)):
            raise _refuse("TXT-1", "inputs.train_part.rows", "the training rows are missing, empty or malformed",
                          "rerun the stage that writes the train part")
        bad = [i for i, r in enumerate(rows) if not label_ok(r["label"])]
        if bad:
            raise _refuse("TXT-1", "inputs.train_part.rows", f"{len(bad)} of {len(rows)} labels are not numbers "
                          f"(first row index from 0: {bad[0]})",
                          "fix the labels in the source data and intake it again")
        out = [{"question": r["input"], "answer": normalise(r["label"])} for r in rows]
        return {**_meta(SCHEMA_TRAIN_SET), "n": len(out), "train_jsonl_sha256": _sha256_bytes(train_jsonl(out)),
                "rows": out}

    def plan(self, ctx: StageContext) -> Proposal:
        n = self._build(ctx)["n"]
        return Proposal(self.name, f"write {n} training rows as the recipe's {{question, answer}} lines", {},
                        {"gpu_hours": 0.0, "n": n})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"train_set": ctx.project.write_artifact(ctx.stage or self.name, "train_set", self._build(ctx),
                                                        SCHEMA_TRAIN_SET)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        t = ctx.project.read_artifact(outputs["train_set"])
        rows = t.get("rows")
        if not isinstance(rows, list) or t.get("n") != len(rows) \
                or t.get("train_jsonl_sha256") != _sha256_bytes(train_jsonl(rows)):
            return [Problem("TXT-1", "train_set", "the row count or the train.jsonl hash does not match the rows",
                            "rerun gsm8k_build_train_set")]
        return []


class WriteRecipe:
    """The recipe's settings: its fixed profiles (`train_grpo.py` PROFILES, which `train_job` runs as they are), the
    training seed, and the hashes of the script and prompt file the jobs mount. Only the template parameter
    `profile_steps` is accepted, and only with the script's values: the job passes no other step count."""

    name = "gsm8k_write_recipe"
    version = "0.1"
    inputs = {"train_set": SCHEMA_TRAIN_SET}
    outputs = {"recipe": SCHEMA_RECIPE}
    needs: frozenset[str] = frozenset()
    params = frozenset({"profile_steps"})
    trains = False

    def _steps(self, ctx: StageContext) -> dict:
        steps = ctx.params.get("profile_steps", PROFILE_STEPS)
        ok = (isinstance(steps, dict) and set(steps) == set(PROFILE_STEPS)
              and all(type(v) is int for v in steps.values()) and steps == PROFILE_STEPS)
        if not ok:
            raise _refuse("TXT-5", "params.profile_steps", f"profile_steps {steps!r} is not the recipe's "
                          f"{PROFILE_STEPS}: the training job runs train_grpo.py's fixed profiles",
                          f"leave profile_steps out, or set it to {PROFILE_STEPS}")
        return dict(steps)

    def _recipe(self, ctx: StageContext) -> dict:
        return {**_meta(SCHEMA_RECIPE), "profile_steps": self._steps(ctx), "seed": TRAINING_SEED,
                "gpu_hours": {p: v["gpu_hours_est"] for p, v in job_mod.PROFILES.items()},
                "train_set_sha256": ctx.inputs["train_set"].sha256, **_package_hashes()}

    def plan(self, ctx: StageContext) -> Proposal:
        r = self._recipe(ctx)
        return Proposal(self.name, f"GRPO recipe: smoke {r['profile_steps']['smoke']} steps, full "
                        f"{r['profile_steps']['full']} steps, seed {r['seed']}", {"profile_steps": r["profile_steps"]},
                        {"gpu_hours": sum(r["gpu_hours"].values())})

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        return {"recipe": ctx.project.write_artifact(ctx.stage or self.name, "recipe", self._recipe(ctx),
                                                     SCHEMA_RECIPE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        r = ctx.project.read_artifact(outputs["recipe"])
        if r.get("train_set_sha256") != ctx.inputs["train_set"].sha256 or \
                any(r.get(k) != v for k, v in _package_hashes().items()):
            return [Problem("TXT-5", "recipe", "the recipe is not for this training set and this pack's script",
                            "rerun gsm8k_write_recipe")]
        return []


# ---------------------------------------------------------------------------------------------------------------------
# The training jobs
# ---------------------------------------------------------------------------------------------------------------------

class _JobStage:
    """What smoke and train share: the job parameters, the checked inputs, and one run of `train_job`."""

    version = "0.1"
    needs = frozenset({"gpu"})
    # `cuda_arch` and `memory_limit_gb` are optional (see `_params`): `memory_limit_gb` can only lower the owner's bound.
    params = frozenset({"image", "runtime_dir", "python_dir", "cuda_arch", "memory_limit_gb"})
    PROFILE = ""
    RULE = ""

    def _params(self, ctx: StageContext) -> dict:
        image = ctx.params.get("image")
        if not (isinstance(image, str) and image):
            raise _refuse("TXT-4", "params.image", f"image {image!r} is not an image reference",
                          f"set image to the recipe's CUDA image pinned by digest ({job_mod.IMAGE})")
        out = {"image": image}
        for key in ("runtime_dir", "python_dir"):
            v = ctx.params.get(key)
            if not (isinstance(v, str) and os.path.isabs(v)):
                raise _refuse("TXT-4", f"params.{key}", f"{key} {v!r} is not an absolute path",
                              f"set {key} (the pipeline YAML says how)")
            out[key] = Path(v)
        arch = ctx.params.get("cuda_arch")
        if arch is not None:
            if not job_mod.valid_cuda_arch(arch):
                raise _refuse("TXT-4", "params.cuda_arch", f"cuda_arch {arch!r} is not a list of compute capabilities",
                              "set cuda_arch to your GPU's compute capability such as 8.9, or leave it null")
            out["cuda_arch"] = arch
        # The memory bound is the owner's: the host file's `gpu_mem_limit_gb` (else gatedtrain's default). A pipeline,
        # which an agent may have written, can ask for a lower bound and never for a higher one: a value above the
        # owner's is refused, not applied.
        owner = launch.owner_gpu_mem_limit_gb()
        limit = ctx.params.get("memory_limit_gb")
        if limit is None:
            limit = owner
        elif not (_num(limit) and limit > 0):
            raise _refuse("TXT-4", "params.memory_limit_gb", f"memory_limit_gb {limit!r} is not a number above 0",
                          "set memory_limit_gb to the GB (10^9 bytes) of GPU memory a smoke run may peak at, at most "
                          "the owner's bound, or leave it null")
        elif limit > owner:
            raise _refuse("TXT-4", "params.memory_limit_gb", f"memory_limit_gb {limit:.15g} is above the owner's bound of "
                          f"{owner:.15g} GB (the host file's gpu_mem_limit_gb, else gatedtrain's default); a pipeline can "
                          "only lower it", f"set memory_limit_gb to at most {owner:.15g}, or leave it null; the owner "
                          "raises the bound by setting gpu_mem_limit_gb in the host file")
        out["memory_limit_gb"] = limit
        out["owner_memory_limit_gb"] = owner
        # The adapter's runtime (the host file's [packs.text-gsm8k] runtime_dir), its venv's python and lock: plan refuses as
        # train_job would, before a GPU job is submitted.
        job_mod.check_runtime(out["runtime_dir"])
        return out

    def _base(self, ctx: StageContext, *, hashes: bool) -> Path:
        fix = "rerun gsm8k_choose_task_and_base"
        base = _read(ctx, "base", "TXT-2")
        d = _model_files(base, "inputs.base", "TXT-2", fix)
        bad = _changed_files(base) if hashes else [n for n in DEPLOYABLE if not (d / n).is_file()]
        if bad:
            raise _refuse("TXT-2", "inputs.base.files", f"{', '.join(bad)}: not the base model file its reference "
                          "lists (changed or removed since gsm8k_choose_task_and_base)", fix)
        return d

    def _recipe(self, ctx: StageContext) -> tuple[dict, dict]:
        recipe, ts = _read(ctx, "recipe", "TXT-5"), _read(ctx, "train_set", "TXT-1")
        fix = "rerun gsm8k_write_recipe"
        if recipe.get("train_set_sha256") != ctx.inputs["train_set"].sha256:
            raise _refuse("TXT-5", "inputs.recipe", "the recipe was written for another training set", fix)
        if any(recipe.get(k) != v for k, v in _package_hashes().items()):
            raise _refuse("TXT-5", "inputs.recipe", "the recipe names another train_grpo.py or prompt file than this "
                          "pack's", fix)
        rows = ts.get("rows")
        if not isinstance(rows, list) or ts.get("train_jsonl_sha256") != _sha256_bytes(train_jsonl(rows)):
            raise _refuse("TXT-1", "inputs.train_set", "the training set's rows do not match its hash",
                          "rerun gsm8k_build_train_set")
        return recipe, ts

    def _estimate(self) -> dict:
        return {"gpu_hours": job_mod.PROFILES[self.PROFILE]["gpu_hours_est"]}

    def _run_job(self, ctx: StageContext) -> tuple[Any, str]:
        """Write the job's inputs, build the job, ask for its output directory by its key, submit (a job of the same
        key reattaches) and wait. The backend and the job id of a succeeded job."""
        p = self._params(ctx)
        base_dir = self._base(ctx, hashes=True)
        recipe, ts = self._recipe(ctx)
        backend = _backend(ctx, self.name)
        digest = ctx.inputs["train_set"].sha256.removeprefix("sha256:")[:12]
        inputs = backend.new_output_dir(f"{INPUTS_DIR}-{digest}")
        _write_bytes(inputs / "train.jsonl", train_jsonl(ts["rows"]))
        _write_bytes(inputs / "prompt.txt", job_mod.PROMPT_FILE.read_bytes())
        _write_bytes(inputs / "train_grpo.py", job_mod.TRAIN_SCRIPT.read_bytes())
        j = job_mod.train_job(profile=self.PROFILE, image=p["image"], python_dir=p["python_dir"], inputs_dir=inputs,
                              base_dir=base_dir, seed=recipe["seed"], runtime_dir=p["runtime_dir"],
                              cuda_arch=p.get("cuda_arch"))
        j = dataclasses.replace(j, output_dir=str(backend.new_output_dir(j.label, key=j.key())))
        job_id = backend.submit(j)
        status = backend.wait(job_id, poll_s=JOB_POLL_S)
        if status.state != "succeeded":
            raise _refuse(self.RULE, f"{self.PROFILE}.job", f"the {self.PROFILE} training job {job_id} ended "
                          f"{status.state}" + (f" ({status.reason})" if status.reason else ""),
                          "read `gatedtrain project job logs`, fix the cause and run the stage again")
        return backend, job_id


def _jsonl(path: Path, listed: dict, name: str) -> list:
    """The JSON lines of the job's output file `name` (split on "\n" only), checked against the backend's listing."""
    f = listed.get(name)
    if f is None:
        raise _refuse("MM-38", f"smoke.{name}", f"the smoke job wrote no {name}", "rerun smoke with the pack's script")
    data = (path / name).read_bytes()
    if _sha256_bytes(data) != f["sha256"]:
        raise _refuse("MM-38", f"smoke.{name}", f"{name} changed after the job ended", "rerun smoke")
    try:
        return [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    except (UnicodeDecodeError, ValueError):
        raise _refuse("MM-38", f"smoke.{name}", f"{name} is not JSON lines", "rerun smoke with the pack's script") \
            from None


def smoke_measures(samples: list, memory: list) -> dict:
    """The MM-38 measurements from train_grpo.py's samples.jsonl lines (GSM8KReward._log: per step and prompt row,
    `completions`, `reward_mean` and the pack's `completion_sha256`) and memory.jsonl lines (make_memory_callback:
    `step`, `seconds` and MEMORY_KEYS in GB). Raises ValueError on a malformed line."""
    n = 0
    reward_sum = 0.0
    digests: set[str] = set()
    for i, r in enumerate(samples):
        hs = r.get("completion_sha256") if isinstance(r, dict) else None
        k = r.get("completions") if isinstance(r, dict) else None
        if not (isinstance(hs, list) and all(isinstance(h, str) for h in hs) and type(k) is int and k == len(hs)
                and _num(r.get("reward_mean"))):
            raise ValueError(f"samples.jsonl line {i + 1} has no completion hashes, completion count and mean reward")
        n += k
        reward_sum += r["reward_mean"] * k
        digests.update(hs)
    peaks, seconds, steps = [], [], []
    for i, r in enumerate(memory):
        if not isinstance(r, dict):
            raise ValueError(f"memory.jsonl line {i + 1} is not an object")
        peaks += [r[k] for k in MEMORY_KEYS if _num(r.get(k))]
        seconds += [r["seconds"]] if _num(r.get("seconds")) else []
        steps += [r["step"]] if type(r.get("step")) is int else []
    if not peaks:
        raise ValueError("memory.jsonl holds no GPU memory measurement")
    return {"n_samples": n, "n_distinct_completions": len(digests),
            "reward_mean": reward_sum / n if n else 0.0, "all_rewards_zero": n > 0 and reward_sum == 0,
            "peak_gpu_mem_gb": max(peaks), "steps": max(steps, default=0),
            "seconds_per_step": sum(seconds) / len(seconds) if seconds else None}


def smoke_reasons(m: dict, memory_limit_gb: float = MEMORY_LIMIT_GB) -> list[str]:
    """The MM-38 reasons the measurements give, as sentences; [] is a PASS. `memory_limit_gb` is the bound the smoke
    run was judged against (the owner's `memory_limit_gb` parameter, recorded in the smoke record)."""
    out = []
    if m["n_samples"] < MIN_SAMPLES:
        out.append(f"only {m['n_samples']} samples (at least {MIN_SAMPLES} are needed to judge)")
    else:
        if m["all_rewards_zero"]:
            out.append(f"all rewards are 0 across {m['n_samples']} samples")
        d = m["n_distinct_completions"]
        if d < 2:
            out.append(f"{d} distinct completion{'' if d == 1 else 's'} across {m['n_samples']} samples")
    if m["peak_gpu_mem_gb"] > memory_limit_gb:
        out.append(f"peak GPU memory {m['peak_gpu_mem_gb']:g} GB is over the bound of {memory_limit_gb:g} GB")
    return out


class Smoke(_JobStage):
    """The recipe's smoke profile (`PROFILE_STEPS`) as a GPU job, then MM-38: refused when every reward is 0, when fewer
    than 2 distinct completions appear across `MIN_SAMPLES` samples, or when the peak GPU memory is over the bound
    (`memory_limit_gb`, by default gatedtrain's)."""

    name = "gsm8k_smoke"
    inputs = {"base": SCHEMA_MODEL_REF, "train_set": SCHEMA_TRAIN_SET, "recipe": SCHEMA_RECIPE}
    outputs = {"smoke": SCHEMA_SMOKE}
    trains = False
    PROFILE = "smoke"
    RULE = "MM-38"

    def plan(self, ctx: StageContext) -> Proposal:
        p = self._params(ctx)
        self._base(ctx, hashes=False)
        recipe, ts = self._recipe(ctx)
        return Proposal(self.name, f"smoke-train {recipe['profile_steps']['smoke']} GRPO steps on {ts['n']} rows in "
                        f"{p['image']}; refuse all-zero rewards, one repeated completion or peak GPU memory over "
                        f"{p['memory_limit_gb']:g} GB", {}, self._estimate())

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        backend, job_id = self._run_job(ctx)
        arts = backend.artifacts(job_id)
        listed = {f["path"]: f for f in arts["files"]}
        top = Path(arts["output_dir"])
        samples, memory = _jsonl(top, listed, "samples.jsonl"), _jsonl(top, listed, "memory.jsonl")
        try:
            m = smoke_measures(samples, memory)
        except ValueError as exc:
            raise _refuse("MM-38", "smoke", f"the smoke output cannot be judged: {exc}",
                          "rerun smoke with the pack's train_grpo.py (without --no-sample-log)") from None
        reasons = smoke_reasons(m, self._params(ctx)["memory_limit_gb"])
        if reasons:
            raise _refuse("MM-38", "smoke", f"the smoke run {job_id} is degenerate or over the memory bound: "
                          f"{'; '.join(reasons)}",
                          "change the recipe, the data or the base model so the smoke run learns from varied "
                          "completions within the GPU's memory")
        del m["all_rewards_zero"]
        out = {**_meta(SCHEMA_SMOKE), "verdict": "PASS", "reasons": [], "job_id": job_id, **m,
               "memory_unit": "GB (10^9 bytes)",
               "memory_limit_gb": self._params(ctx)["memory_limit_gb"],
               "owner_memory_limit_gb": self._params(ctx)["owner_memory_limit_gb"],
               "base_sha256": ctx.inputs["base"].sha256, "train_set_sha256": ctx.inputs["train_set"].sha256,
               "recipe_sha256": ctx.inputs["recipe"].sha256}
        return {"smoke": ctx.project.write_artifact(ctx.stage or self.name, "smoke", out, SCHEMA_SMOKE)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        s = ctx.project.read_artifact(outputs["smoke"])
        try:
            bound = s["memory_limit_gb"]
            if not (_num(bound) and 0 < bound <= launch.owner_gpu_mem_limit_gb()):
                raise TypeError("memory_limit_gb")        # not a bound the owner's host allows (it may have been lowered)
            again = smoke_reasons({**s, "all_rewards_zero": s.get("reward_mean") == 0}, bound)
        except (KeyError, TypeError):
            again = ["incomplete record"]
        if s.get("verdict") != "PASS" or s.get("reasons") != [] or again:
            return [Problem("MM-38", "smoke.verdict", "the smoke record is not a PASS its measurements support",
                            "rerun smoke")]
        return []


class Train(_JobStage):
    """The recipe's full profile (`PROFILE_STEPS`) as a GPU job after a smoke PASS of the same inputs; the candidate is the
    last checkpoint's six deployable files, handed off by the backend (each `o+r` in `o+x` directories)."""

    name = "gsm8k_train"
    params = _JobStage.params - {"memory_limit_gb"}          # the memory bound is the smoke stage's
    inputs = {"base": SCHEMA_MODEL_REF, "train_set": SCHEMA_TRAIN_SET, "recipe": SCHEMA_RECIPE,
              "smoke": SCHEMA_SMOKE}
    outputs = {"model": SCHEMA_MODEL_REF}
    trains = True
    PROFILE = "full"
    RULE = "TXT-3"

    def _smoke_passed(self, ctx: StageContext) -> None:
        s = _read(ctx, "smoke", "MM-38")
        if s.get("verdict") != "PASS":
            raise _refuse("MM-38", "inputs.smoke", "the smoke run did not pass", "rerun gsm8k_smoke")
        for key in ("base", "train_set", "recipe"):
            if s.get(f"{key}_sha256") != ctx.inputs[key].sha256:
                raise _refuse("MM-38", "inputs.smoke", f"the smoke run passed with another {key} than this "
                              "training run's", "rerun gsm8k_smoke on these inputs")
        bound = s.get("memory_limit_gb")           # the bound it was judged against must still be one the owner allows
        if not (_num(bound) and 0 < bound <= launch.owner_gpu_mem_limit_gb()):
            raise _refuse("MM-38", "inputs.smoke", f"the smoke run was judged against a memory bound ({bound!r} GB) "
                          "that the host's gpu_mem_limit_gb no longer allows", "rerun gsm8k_smoke")

    def plan(self, ctx: StageContext) -> Proposal:
        self._smoke_passed(ctx)
        p = self._params(ctx)
        self._base(ctx, hashes=False)
        recipe, ts = self._recipe(ctx)
        return Proposal(self.name, f"GRPO-train {recipe['profile_steps']['full']} steps on {ts['n']} rows in "
                        f"{p['image']}; the candidate is the last checkpoint", {}, self._estimate())

    def run(self, ctx: StageContext, decision: dict) -> dict[str, ArtifactRef]:
        self._smoke_passed(ctx)
        backend, job_id = self._run_job(ctx)
        paths = {f["path"] for f in backend.artifacts(job_id)["files"]}
        cuts = {int(m.group(1)) for p in paths if "/" in p
                for m in [_CHECKPOINT.fullmatch(p.split("/", 1)[0])] if m}
        if not cuts:
            raise _refuse("TXT-3", "train.checkpoints", f"the training job {job_id} wrote no checkpoint-<N> directory",
                          "read `gatedtrain project job logs` and run the stage again")
        last = max(cuts)
        missing = [n for n in DEPLOYABLE if f"checkpoint-{last}/{n}" not in paths]
        if missing:
            raise _refuse("TXT-3", f"train.checkpoint-{last}", f"the last checkpoint checkpoint-{last} of job {job_id} "
                          f"lacks {', '.join(missing)}", "fix the training script's save and run the stage again")
        files = backend.hand_off(job_id, [f"checkpoint-{last}/{n}" for n in DEPLOYABLE])
        model = {**_meta(SCHEMA_MODEL_REF), "generator": GENERATOR, "label": "candidate", "job_id": job_id,
                 "checkpoint": last, "files": files}
        return {"model": ctx.project.write_artifact(ctx.stage or self.name, "model", model, SCHEMA_MODEL_REF)}

    def check(self, ctx: StageContext, outputs: dict[str, ArtifactRef]) -> list[Problem]:
        fix = "rerun gsm8k_train"
        m = ctx.project.read_artifact(outputs["model"])
        if m.get("generator") != GENERATOR or m.get("label") != "candidate":
            return [Problem("TXT-3", "model", f"the model is not a {GENERATOR} candidate", fix)]
        try:
            d = _model_files(m, "model", "TXT-3", fix)
        except Refused as exc:
            return list(exc.problems)
        if d.name != f"checkpoint-{m.get('checkpoint')}":
            return [Problem("TXT-3", "model.files", "the files are not those of the recorded checkpoint", fix)]
        bad = _changed_files(m) + [Path(f["path"]).name for f in m["files"] if os.path.exists(f["path"])
                                   and not os.lstat(f["path"]).st_mode & stat.S_IROTH]
        if bad:
            return [Problem("TXT-3", "model.files", f"{', '.join(sorted(set(bad)))}: not the listed file readable by "
                            "other users", fix)]
        return []
