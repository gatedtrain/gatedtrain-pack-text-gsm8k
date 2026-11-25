"""The GRPO training job: a gatedtrain backend `Job` of the ext adapter `text-gsm8k-grpo` (adapter.py).

The job runs the pack's `train_grpo.py` (the recipe gatedtrain-recipe-trl-grpo-gsm8k's script) in the recipe's CUDA
devel image with the runtime directory `<runtime>` (the owner-built venv `<runtime>/.venv`, from `runtime/uv.lock`, and
the lock files) mounted read-only at its own path, as is the interpreter tree the venv was made from. The directory, not
the venv, is the mount: the adapter row's credential-scan skip names `<runtime>/.venv`, and a scan skips a listed
sub-directory of a mount but always walks the mount root (a venv holds library code named like `credentials`). The inputs are mounted read-only under /inputs: the training
file, the prompt file, the script and the base model directory. The output is /out; /tmp is a tmpfs; no network.

The runtime directory `<runtime>` is the adapter's (adapter.py `RUNTIME`): `GATEDTRAIN_PACK_TEXT_GSM8K_RUNTIME`, else
this package's `runtime/`. It holds copies of `runtime/pyproject.toml` and `runtime/uv.lock`; the owner builds the
venv there on the host (no GPU), without bytecode, from the uv-managed CPython `PYTHON_VERSION` in `<python>`
(`<platform>` is the platform tag `uv python install` gives the directory, e.g. what `uv python find` prints):

    cd <runtime>
    env -u UV_COMPILE_BYTECODE UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \\
        uv sync --frozen --python <python>/cpython-<PYTHON_VERSION>-<platform>/bin/python3.12

(no `--compile-bytecode`). The backend's job key fingerprints every entry of the read-only mounts and refuses past
`gatedtrain.ainative.backend.MAX_INPUT_ENTRIES` (100,000); a venv with compiled bytecode exceeds it. The mounts are
read-only and the job sets `PYTHONDONTWRITEBYTECODE=1`, so the run writes no bytecode either. `train_job` counts the
entries before it returns and refuses (TXT-4) past the limit.

`train_job` leaves `output_dir` empty: the caller fills in the backend's new output directory for the job's key
(`dataclasses.replace(job, output_dir=str(backend.new_output_dir(job.label, key=job.key())))`).
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

from gatedtrain.ainative import backend
from gatedtrain.ainative.backend import Job
from gatedtrain.ainative.runtime import Mount
from gatedtrain.errors import Problem, Refused

from . import adapter
from .adapter import FRAMEWORK_PACKAGE, NAME

RULE = "TXT-4"
PYTHON_VERSION = "3.12.16"           # the CPython the runtime venv is built from (runtime/README, the recipe's pin)
BUILD = ("cd <runtime>; env -u UV_COMPILE_BYTECODE UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy "
         "UV_PYTHON_DOWNLOADS=never uv sync --frozen "
         f"--python <python>/cpython-{PYTHON_VERSION}-<platform>/bin/python3.12")

PACKAGE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PACKAGE_DIR / "train_grpo.py"
PROMPT_FILE = PACKAGE_DIR / "prompt.txt"
LOCK_FILE = PACKAGE_DIR / "runtime" / "uv.lock"

# The recipe's base image (its runtime/README.md "Pins"), pinned by digest.
IMAGE = ("docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04"
         "@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266")


def _locked_version(package: str) -> str:
    """The version `runtime/uv.lock` pins for `package`: the lock is the one place a pin lives."""
    import tomllib

    try:
        found = [p["version"] for p in tomllib.loads(LOCK_FILE.read_text(encoding="utf-8"))["package"]
                 if p["name"] == package]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"cannot read the pinned {package} version from {LOCK_FILE}: {exc}") from exc
    if len(found) != 1:
        raise RuntimeError(f"{LOCK_FILE} must pin {package} exactly once; it lists {len(found)} entries")
    return found[0]


FRAMEWORK = {"name": FRAMEWORK_PACKAGE, "version": _locked_version(FRAMEWORK_PACKAGE), "commit": None}   # trl

# The recipe's environment (its runtime/README.md "Command shape"). It names no GPU: `TORCH_CUDA_ARCH_LIST` is the
# architecture of the owner's GPU, so a stage parameter (`cuda_arch`, `train_job`'s argument) adds it.
ENV = {
    "DO_NOT_TRACK": "1",
    "FLASHINFER_WORKSPACE_BASE": "/tmp/cache",
    "HF_HOME": "/tmp/hf",
    "HF_HUB_OFFLINE": "1",
    "HOME": "/tmp/home",
    "MASTER_ADDR": "127.0.0.1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCHINDUCTOR_CACHE_DIR": "/tmp/cache/inductor",
    "TRANSFORMERS_OFFLINE": "1",
    "TRITON_CACHE_DIR": "/tmp/cache/triton",
    "VLLM_CACHE_ROOT": "/tmp/cache/vllm",
    "VLLM_CONFIG_ROOT": "/tmp/cache/vllm-config",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "XDG_CACHE_HOME": "/tmp/cache",
}

# train_grpo.py PROFILES: smoke 10 steps, full 100 steps, both 32 prompts x 8 generations per step.
PROFILES = {
    "smoke": {"gpu_hours_est": adapter.GPU_H_ESTIMATE["smoke"], "time_limit_s": 3600},
    "full": {"gpu_hours_est": adapter.GPU_H_ESTIMATE["full"], "time_limit_s": 14400},
}


def lock_sha256() -> str:
    """The hex SHA-256 of the runtime lock the venv is built from."""
    return hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest()


def _refused(message: str, fix: str) -> Refused:
    return Refused([Problem(RULE, "job", message, fix)])


def input_entries(mounts: tuple[Mount, ...]) -> int:
    """The entries the job key fingerprints (`backend.input_fingerprint`): per read-only mount, the mount itself and
    every entry below its resolved path, links not followed. Names only: nothing is read or hashed."""
    count = 0
    for m in mounts:
        if not m.read_only:
            continue
        count += 1
        stack = [os.path.realpath(m.host)]
        while stack:
            path = stack.pop()
            try:
                if not stat.S_ISDIR(os.lstat(path).st_mode):
                    continue
                with os.scandir(path) as it:
                    names = [e.name for e in it]
            except OSError:
                continue
            count += len(names)
            stack += [os.path.join(path, n) for n in names]
    return count


def check_runtime(runtime_dir: Path | str | None = None) -> str:
    """The adapter's runtime directory (adapter.py `RUNTIME`), checked: `runtime_dir` (default: that directory) must be
    it, `<runtime>/.venv/bin/python` must exist, and `<runtime>/uv.lock` must be this package's `runtime/uv.lock`
    (same SHA-256). Refused (TXT-4) otherwise. The stages call it in plan as well, so the owner hears it before a GPU
    job is submitted."""
    runtime = adapter.RUNTIME            # the row's interpreter is `<RUNTIME>/.venv/bin/python`, written exactly so
    if runtime_dir is not None and os.path.realpath(runtime_dir) != os.path.realpath(runtime):
        raise _refused(f"runtime directory {runtime_dir} is not the adapter's {runtime}, whose venv the adapter row "
                       "names as the interpreter",
                       f"set `runtime_dir = \"{runtime_dir}\"` in the [packs.{adapter.PACK}] table of the host file "
                       "(the adapter reads it at import)")
    if not os.path.isfile(Path(runtime) / ".venv" / "bin" / "python"):
        raise _refused(f"the runtime directory {runtime} has no .venv/bin/python",
                       f"set `runtime_dir` in the [packs.{adapter.PACK}] table of the host file to the directory holding "
                       f"the runtime venv, built with: {BUILD}")
    lock = Path(runtime) / "uv.lock"
    try:
        got = hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        got = None
    if got != lock_sha256():
        raise _refused(f"{lock} is {'missing' if got is None else f'not the pack runtime/uv.lock (sha256 {got})'}; "
                       f"the venv must be built from the pack's runtime/uv.lock (sha256 {lock_sha256()})",
                       f"copy runtime/pyproject.toml and runtime/uv.lock out of the installed package into {runtime} "
                       f"and build the venv there: {BUILD}")
    return runtime


# One capability is `<major>.<minor>` with an optional arch suffix (`9.0a`) or `+PTX`; several are separated by single
# spaces, the one separator both PyTorch and FlashInfer read (`;` is PyTorch's alone).
_CUDA_ARCH = re.compile(r"[0-9]+\.[0-9]+[a-z]?(?:\+PTX)?(?: [0-9]+\.[0-9]+[a-z]?(?:\+PTX)?)*")
MAX_CUDA_ARCH_CHARS = 64


def valid_cuda_arch(value: object) -> bool:
    """Whether `value` is a `TORCH_CUDA_ARCH_LIST`: compute capabilities such as `8.9`, `9.0a`, `9.0+PTX`, separated by
    single spaces, at most `MAX_CUDA_ARCH_CHARS` long."""
    return isinstance(value, str) and len(value) <= MAX_CUDA_ARCH_CHARS and _CUDA_ARCH.fullmatch(value) is not None


def train_job(*, profile: str, image: str, python_dir: Path, inputs_dir: Path, base_dir: Path, seed: int,
              runtime_dir: Path | None = None, cuda_arch: str | None = None) -> Job:
    """The training job of `profile` (smoke or full). `inputs_dir` holds `train.jsonl`, `prompt.txt` and
    `train_grpo.py`; `base_dir` is the base model directory; `python_dir` holds the interpreter tree the venv was made
    from. `runtime_dir` defaults to the adapter's runtime directory and must be that directory. ValueError for another
    profile; Refused (TXT-4) for a runtime directory `check_runtime` refuses, or read-only inputs over the job key's
    entry limit. The runtime's `venv_lock` is `<runtime>/uv.lock`, the lock the venv was built from. `cuda_arch`, the
    GPU architecture list of the owner's host, sets `TORCH_CUDA_ARCH_LIST` (left out when None); ValueError when it
    is not a list of compute capabilities."""
    if profile not in PROFILES:
        raise ValueError(f"profile {profile!r} is not one of {', '.join(PROFILES)}")
    if cuda_arch is not None and not valid_cuda_arch(cuda_arch):
        raise ValueError(f"cuda_arch {cuda_arch!r} is not a list of compute capabilities such as '8.9'")
    runtime = check_runtime(runtime_dir)
    python, inputs = Path(python_dir), Path(inputs_dir)
    venv = f"{runtime}/.venv"             # the interpreter's venv; the mount is the runtime directory holding it
    job = Job(
        label=f"grpo-{profile}",
        adapter=NAME,
        runtime={"kind": "venv_in_container", "image": image, "venv_lock": str(Path(runtime) / "uv.lock")},
        framework=dict(FRAMEWORK),
        argv=(f"{venv}/bin/python", "/inputs/train_grpo.py", "--profile", profile, "--out", "/out",
              "--seed", str(seed)),
        env={**ENV, **({"TORCH_CUDA_ARCH_LIST": cuda_arch} if cuda_arch is not None else {})},
        mounts=(Mount(runtime, runtime), Mount(str(python), str(python)),
                Mount(str(inputs / "train.jsonl"), "/inputs/train.jsonl"),
                Mount(str(inputs / "prompt.txt"), "/inputs/prompt.txt"),
                Mount(str(inputs / "train_grpo.py"), "/inputs/train_grpo.py"),
                Mount(str(Path(base_dir)), "/inputs/base-model")),
        output_dir="",
        output_container="/out",
        tmpfs=("/tmp",),
        network="none",
        gpu=True,
        time_limit_s=PROFILES[profile]["time_limit_s"],
        gpu_hours_est=PROFILES[profile]["gpu_hours_est"],
        training_seed=seed,
        profile=profile,
    )
    limit = backend.MAX_INPUT_ENTRIES
    count = input_entries(job.mounts)
    if count > limit:
        raise _refused(f"the read-only inputs hold {count} entries, over the job key's limit of {limit} "
                       "(gatedtrain backend MAX_INPUT_ENTRIES); the runtime directory, its venv above all, is most of them",
                       f"build the runtime venv without bytecode (no --compile-bytecode, UV_COMPILE_BYTECODE unset): "
                       f"{BUILD}")
    return job
