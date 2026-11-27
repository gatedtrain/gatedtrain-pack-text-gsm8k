# gatedtrain-pack-text-gsm8k

A scenario pack for [gatedtrain](https://github.com/gatedtrain/gatedtrain) (`gatedtrain>=0.7.2,<0.8`): it teaches a small
language model grade-school math and lets gatedtrain decide, by a rule fixed before training, whether the trained
model really is better.

It trains Qwen3-0.6B with TRL GRPO on GSM8K questions on your GPU. Before training, gatedtrain's keeper seals part of
your questions away; after training, the keeper answers those sealed questions with the base and the trained model and
scores each response by the number it gives as its answer. The gate passes only when the trained model beats the base by the
registered rule.

It is a prototype: the models it trains are not promotable.

## How it works

The owner (you) prepares data and trains; the keeper, a service of gatedtrain, holds the sealed questions and runs the
evaluation. Whether the keeper's own OS user keeps the sealed questions from your account depends on the keeper mode
(see "Keeper modes"). The boxes below name the stages of the pack's pipeline.

```mermaid
flowchart LR
    subgraph owner["Owner's account"]
        data["GSM8K rows"] --> intake["intake_seal"]
        intake --> train_part["Training part"]
        train_part --> prep["profile_data, choose_task_and_base,<br/>build_train_set, write_recipe"]
        prep --> smoke["smoke: 10 GRPO steps (GPU)"]
        smoke --> train["train: 100 GRPO steps (GPU)"]
    end
    subgraph keeper["Keeper's account"]
        sealed["Sealed questions and labels"]
        reg["preregister: criterion fixed before training"]
        eval["evaluate_sealed: base and trained model<br/>answer in a container (GPU, no network)"]
        gate{"gate"}
        report["report (signed)"]
    end
    intake --> sealed
    intake --> reg
    train --> eval
    sealed --> eval
    reg --> gate
    eval --> gate
    gate --> report
```

The container gets the sealed questions, never their labels; the keeper scores the answers itself.

## Changes in 0.3.0

- Needs gatedtrain 0.7 (`gatedtrain>=0.7,<0.8`). The pack's code needed no change for it: its CPU tests (123) pass
  unchanged against 0.6.0 and against the 0.7.0 candidate; only the dependency range, the version and the documentation
  moved.
- **Keeper modes.** gatedtrain 0.7 sets up a same-user keeper by default (no sudo) and states the mode in every result.
  See "Keeper modes" below; a result of this pack is quoted with the mode line of its report.
- The pack uses no gatedtrain image: its trainer image is the recipe's CUDA devel image pinned by digest in the
  pipeline params, so the image lock of gatedtrain 0.7 (`gatedtrain image list`) does not concern it.
- Known limits of this release: no GPU end to end run was made with gatedtrain 0.7 (the CPU tests use fake containers);
  the keeper's eval tree still comes from `gatedtrain setup host`, a script for root, in both keeper modes (a user-owned
  eval tree for a same-user keeper has not been tried); the pack's own paths (`/opt/gatedtrain-keeper/...` for the eval
  interpreter, `GATEDTRAIN_PACK_TEXT_GSM8K_EVAL_PYTHON`) are unchanged.

## Changes in 0.2.0

- Needs gatedtrain 0.6 (the pack and the VLM pack can now share one keeper environment).
- **Greedy decoding is the default** (`decoding: greedy`, temperature 0, top_p 1.0, one evaluation seed, one sample per
  item); `decoding: sampled` in the `choose_task_and_base` params keeps the 0.1.0 mode (temperature 0.7, four seeds).
  See "Decoding". No GPU end to end run of this release was made: the run below is the 0.1.0 run (sampled decoding),
  and the greedy path is covered by the CPU tests (fake containers) and the determinism probe described below.

## Decoding

gatedtrain 0.6 registers greedy decoding as an explicit mode and its keeper checks it: every sealed item is generated
once more under a seed the registration does not name, and an attempt whose predictions differ is refused after it
spent its alpha (AN-26). vLLM's default batching is not deterministic, so the batch generator runs greedy decoding one
request at a time (`max_num_seqs=1`, prefix caching off). Probe of this pack (the keeper's container flags and eval
environment, vLLM 0.31.0, 399 GSM8K rows from the training part of an earlier project, never sealed items, two seeds in one
call, 1,024 new tokens):

| Model | Default batching: items that differ | `max_num_seqs=1`: items that differ | Time for both passes |
|---|---|---|---|
| Base (Qwen3-0.6B) | 239/399 (59.9 %) | **0/399** | 17 min 42 s |
| Trained (100 GRPO steps) | 205/399 (51.4 %) | **0/399** | 17 min 33 s |

**Time limit.** One request at a time is slow: about 17.5 minutes per arm against the generator's limit of 20 minutes
(the keeper allows at most 1,200 s), a margin of about 10 %. A slower GPU or a busier host can exceed it; the attempt is
then a crash that costs its alpha. Use `decoding: sampled` (under two minutes per arm in the default-batching probe) when the margin is too small for
your host. A change of the generator, the vLLM version or the GPU needs the probe again.

## A tested run (0.1.0, sampled decoding)

One run on one 24 GB GPU, with 7,074 GSM8K training-split rows for training and 399 rows sealed for the test. Scores are
at temperature 0.7 over four seeds, so they are not the scores of the greedy mode:

| | Score on the sealed questions |
|---|---|
| Base (Qwen3-0.6B) | 0.659 |
| Trained (100 GRPO steps) | 0.762 |
| Gate | **PASS**: +0.103, 95.25 % interval [0.0739, 0.1335] |

About 0.64 GPU hours in all (training 0.51 h). Your numbers will differ with other data, seeds and hardware.

## What you need

- Linux x86_64, one NVIDIA GPU, a CUDA 13.0 driver; CPython 3.12 for both venvs.
- gatedtrain 0.7 with a keeper and rootless Docker for the owner and the keeper (see gatedtrain's
  [Quickstart without sudo](https://github.com/gatedtrain/gatedtrain/blob/main/docs/sudo-free-quickstart.md) and
  [AI-native projects](https://github.com/gatedtrain/gatedtrain/blob/main/docs/ai-native.md) guides; `gatedtrain setup
  check` lists what the host must provide).
- A same-user keeper (the default) needs no sudo for the pack: you build the keeper's eval tree yourself with
  `gatedtrain setup eval-tree` ([the keeper's eval tree](#the-keepers-eval-tree)), and the pack is installed in the
  environment the keeper runs from. A separate-user keeper needs root, once, for the root-owned eval tree
  (`gatedtrain setup host`) and to install the pack in the keeper's environment. The host's own prerequisites (rootless
  Docker, `uidmap`, the NVIDIA driver and CDI specification; `gatedtrain setup check` lists them) need an administrator
  on a machine that lacks them.

## Keeper modes

gatedtrain 0.7 has two keeper modes and every result states the one that produced it. **Same-user** (the default of
`gatedtrain setup keeper`): the keeper runs under your own OS user and follows the procedure, but anything running as
you, such as an agent with a shell, can read the sealed questions, the labels and the keeper's key; the report says the
data was not protected from your account. **Separate-user** (`gatedtrain setup keeper --mode separate-user`, one sudo):
the keeper runs under its own OS user and the sealed questions are out of your account's reach. This pack works in both.
Quote the mode line of the report when you quote a result, and use `--require-keeper-mode separate-user` at `project
create` when a weaker keeper must be refused. The run described below was made before gatedtrain 0.7 existed and does not state a keeper mode.

## Install

The pack is not on PyPI. Install it from this repository at a release tag, into the environment of the `gatedtrain` you
run, as the owner:

```sh
pip install "gatedtrain-pack-text-gsm8k @ git+https://github.com/gatedtrain/gatedtrain-pack-text-gsm8k@v0.3.0"
```

The keeper's own environment needs the same pack at the same version (see "The keeper's environment" below).

What the pack registers with gatedtrain:

| Entry point group | Name | Object |
|---|---|---|
| `gatedtrain.scenarios` | `text-gsm8k` | `scenario.py` `SCENARIO`: the batch generator `text_gsm8k_vllm`, the scorer `gsm8k_answer`, `pipeline.yaml` |
| `gatedtrain.stages` | `gsm8k_profile_data`, `gsm8k_choose_task_and_base`, `gsm8k_build_train_set`, `gsm8k_write_recipe`, `gsm8k_smoke`, `gsm8k_train` | `stages.py` |
| `gatedtrain.adapters` | `text-gsm8k-grpo` | `adapter.py` `PLUGIN`: the training job's trainer row and the evaluator row |
| console script | `gatedtrain-pack-text-gsm8k-data` | `data.py`: GSM8K rows to the intake format |

The pack derives from the recipe
[`gatedtrain-recipe-trl-grpo-gsm8k`](https://github.com/gatedtrain/gatedtrain-recipe-trl-grpo-gsm8k) (its training
script and its runtime lock). It has no per-item generator: a project needs a keeper, which runs the batch generator
`text_gsm8k_vllm` (`generate_vllm.py`, vLLM) in its inference container on the GPU. The scorer `gsm8k_answer`
(`answer.py`, standard library only) takes the number of the first `\boxed{}`, else the first number after the last
"The answer is", else the last number in the response, and compares it with the label as a decimal.

## Setting up, in order

1. [Prepare the host](#preparing-the-host-the-owner-no-sudo) as the owner: the base model, the training runtime, the
   data and the image.
2. Install the pack in [the keeper's environment](#the-keepers-environment) (root in a separate-user keeper).
3. Build [the keeper's eval tree](#the-keepers-eval-tree) (root).
4. Fill in [the pipeline params](#the-pipeline-params) and [run the project](#running-a-project).

## Preparing the host (the owner, no sudo)

Everything in this section runs as the owner. The paths are examples; `/srv/gatedtrain` is the owner's tree.

1. Fetch the base model at the pinned commit. The directory is a pipeline param (`choose_task_and_base.base_model_dir`):

   ```sh
   uvx --from huggingface_hub hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
       --local-dir /srv/gatedtrain/models/Qwen3-0.6B
   ```

   The stage checks that the directory holds a Qwen3 `config.json`, the weights and the tokenizer files, and copies
   the six deployable files (`config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`,
   `tokenizer_config.json`, `chat_template.jinja`) for the keeper. The download at this commit has no
   `chat_template.jinja`: its chat template is the string field `chat_template` of `tokenizer_config.json`. The stage
   then writes that string, byte for byte as UTF-8, to the copy's `chat_template.jinja` (the download is left as it
   is); a directory with neither a `chat_template.jinja` nor a non-empty string `chat_template` is refused.

2. Build the training runtime venv with a uv-managed CPython 3.12. The runtime directory holds the pack's copies of
   `runtime/pyproject.toml` and `runtime/uv.lock` (the recipe's lock). Copy them out of the installed package into a
   directory of yours, **and** set `runtime_dir` in the `[packs.text-gsm8k]` table of your host file
   (`~/.config/gatedtrain/host.toml`) to that directory (the default is the package's own `runtime/`). The adapter reads
   the host file and `<runtime>/.venv/pyvenv.cfg` when it is imported, so the venv must exist before that process
   starts, and the directory must equal the pipeline params `smoke.runtime_dir` and `train.runtime_dir`. `smoke` and
   `train` refuse in their plan, before any GPU job, when the directory differs, when `<runtime>/.venv/bin/python` is
   missing, or when `<runtime>/uv.lock` is not the pack's `runtime/uv.lock`. Build **without bytecode**:

   ```sh
   uv python install 3.12.16 --install-dir /srv/gatedtrain/python
   mkdir -p /srv/gatedtrain/runtime-gsm8k
   cp <site-packages>/gatedtrain_pack_text_gsm8k/runtime/pyproject.toml \
      <site-packages>/gatedtrain_pack_text_gsm8k/runtime/uv.lock /srv/gatedtrain/runtime-gsm8k/
   cd /srv/gatedtrain/runtime-gsm8k
   env -u UV_COMPILE_BYTECODE UV_PROJECT_ENVIRONMENT=$PWD/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
       uv sync --frozen --python /srv/gatedtrain/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12
   ```
   and in `~/.config/gatedtrain/host.toml`:
   ```toml
   [packs.text-gsm8k]
   runtime_dir = "/srv/gatedtrain/runtime-gsm8k"
   ```
   (The variable `GATEDTRAIN_PACK_TEXT_GSM8K_RUNTIME` still works as a deprecated workaround and prints a notice; it
   is removed with gatedtrain 0.9.0.)

   Do not add `--compile-bytecode`: the job's key fingerprints every entry of the read-only runtime mount and refuses
   more than 100,000 entries (`gatedtrain.ainative.backend.MAX_INPUT_ENTRIES`), and a venv with compiled bytecode
   exceeds that. The runs write no bytecode either (`PYTHONDONTWRITEBYTECODE=1`). The job mounts the whole runtime
   directory read-only (not just `.venv`), so the adapter row's credential-scan skip for the venv applies.

3. Convert the GSM8K rows (JSONL with `question` and `answer`, the answer ending with a `#### <number>` line) to the
   intake format. Hugging Face ships GSM8K (`openai/gsm8k`, config `main`) as parquet; export the rows you want as
   JSONL first, for example the train split at the recipe's dataset commit:

   ```sh
   uv run --no-project --with datasets python -c "from datasets import load_dataset; load_dataset('openai/gsm8k', \
       'main', split='train', revision='740312add88f781978c0658806c59bc2815b9866').to_json('gsm8k-main.jsonl')"
   ```

   The command refuses with line numbers only, never row content, and writes nothing when any row is refused or the
   source holds no row:

   ```sh
   gatedtrain-pack-text-gsm8k-data --src gsm8k-main.jsonl --dst intake.jsonl
   ```

4. Pull the recipe's CUDA devel image into the owner's rootless daemon (no job pulls an image):
   `docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266`.

Rules: TXT-2 (the base model files), TXT-4 (the runtime directory and the 100,000-entry limit).

## The keeper's environment

The keeper checks, before it starts a generator, that the generator's script is a file of a distribution in the
keeper's own environment record (gatedtrain, the scenario packs and their dependencies), so the pack must be installed
in the keeper's own environment at the same version as the owner's. In the default same-user mode the keeper runs from
the environment you run `gatedtrain` from, so the install above is enough (restart the keeper afterwards). In
separate-user mode it is the root-owned venv `gatedtrain setup keeper --mode separate-user` installs
(`/opt/gatedtrain-keeper/venv` by default). `gatedtrain setup keeper` prints that install line with the packs
installed for the owner at their versions (`--packages gatedtrain-pack-text-gsm8k==<version>` names it); the printed
`pip install ... gatedtrain-pack-text-gsm8k==<version>` resolves only from PyPI, where the pack is not published. In
separate-user mode root installs it from this repository at the same tag as the owner's instead:

```sh
sudo /opt/gatedtrain-keeper/venv/bin/pip install --no-cache-dir \
    "gatedtrain-pack-text-gsm8k @ git+https://github.com/gatedtrain/gatedtrain-pack-text-gsm8k@v0.3.0"
```

Restart the keeper after installing or upgrading the pack.

## The keeper's eval tree

The keeper's inference container runs `generate_vllm.py` with the python of an eval venv that the keeper mounts
read-only. The keeper accepts a tree only when every node and every directory above it is owned by the keeper's own uid
or by root and writable by no one else. The pack ships the inputs for it: `eval-requirements.txt` (hash-pinned, binary wheels only: vllm 0.31.0, transformers 5.17.0,
torch 2.13.0 and their locked dependencies, the versions of the runtime lock) and the lock it was exported from
(`eval/pyproject.toml`, `eval/uv.lock`; the comment in the pyproject says how to regenerate it). Find them in the
installed package.

**Same-user keeper (the default, no sudo).** Print the build script, read it, run it as yourself (it needs `uv`), then
give the keeper the options it prints; the script changes only files under its root, `gatedtrain-keeper-eval` beside
the keeper's state directory (`--root DIR` changes it):

```sh
gatedtrain setup eval-tree \
    --requirements <site-packages>/gatedtrain_pack_text_gsm8k/eval-requirements.txt \
    --python-version 3.12 > gatedtrain-eval-tree.sh
sh gatedtrain-eval-tree.sh
gatedtrain setup keeper --mode same-user --inference-image <image ID> \
    --eval-tree <root>/python --eval-tree <root>/eval-venv --eval-python <root>/eval-venv/bin/python
```

The tree is yours, so like the rest of a same-user keeper it is not sealed from your account, and the keeper checks it at
start only (restart the keeper after any change). A separate-user keeper refuses a tree you own.

**Separate-user keeper (one sudo).** Print the host setup script and read it before running it as root (`gatedtrain setup
host --help` lists the other flags):

```sh
gatedtrain setup host \
    --inference-image docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 \
    --eval-requirements <site-packages>/gatedtrain_pack_text_gsm8k/eval-requirements.txt \
    --eval-python /opt/gatedtrain-keeper/python/<cpython-3.12>/bin/python3.12
```

The script builds `/opt/gatedtrain-keeper/eval-venv` from that interpreter, which must already sit under
`/opt/gatedtrain-keeper/python`: root installs a standalone CPython there before running the script, with
`env UV_PYTHON_INSTALL_DIR=/opt/gatedtrain-keeper/python uv python install 3.12` (the script checks that the tree is
root-owned and not group- or other-writable, and does not install it itself). The keeper serves the eval venv
read-only to its containers. The generator's interpreter is `bin/python` of the eval venv `gatedtrain setup host` builds; start the keeper with
`--eval-python PATH` when the eval venv lives elsewhere (the value is part of the keeper's signed environment record).
The variable `GATEDTRAIN_PACK_TEXT_GSM8K_EVAL_PYTHON` still works as a deprecated workaround, prints a notice, and is
removed in pack 0.4.0. The keeper still runs it only when that interpreter is a regular file inside one of its `--eval-tree`
directories.

## The pipeline params

`pipeline.yaml` leaves these params null; a stage refuses to run until they are set. Copy the file, fill them in and pass
the copy to `gatedtrain project create --pipeline <copy> --keeper <socket>`:

| Param | Value |
|---|---|
| `choose_task_and_base.base_model_dir` | the Qwen3-0.6B directory fetched above, e.g. `/srv/gatedtrain/models/Qwen3-0.6B` |
| `choose_task_and_base.decoding` | optional: `greedy` (the default) or `sampled` |
| `smoke.image`, `train.image` | the CUDA devel image pinned by digest, held by the owner's rootless daemon |
| `smoke.runtime_dir`, `train.runtime_dir` | the runtime directory holding the owner-built `.venv`, e.g. `/srv/gatedtrain/runtime-gsm8k`; the host file's `[packs.text-gsm8k] runtime_dir` |
| `smoke.python_dir`, `train.python_dir` | the directory of uv-managed interpreters the venv was made from, e.g. `/srv/gatedtrain/python` |

`smoke` and `train` take the same three values. Two optional params describe the owner's GPU:

| Param | Value |
|---|---|
| `smoke.cuda_arch`, `train.cuda_arch` | the GPU's compute capability, such as `8.9` (`nvidia-smi --query-gpu=compute_cap`); it becomes `TORCH_CUDA_ARCH_LIST` in the job. Without it the job sets none and the toolchain reads the GPU. |
| `smoke.memory_limit_gb` | optional: a smoke bound **lower** than the owner's, in GB (10^9 bytes). The owner's bound is the host file's `gpu_mem_limit_gb` (default 22.5, a 24 GB card's); a pipeline value above it is refused, never applied, so an agent-written pipeline cannot raise the bound. The smoke record keeps the bound used and the owner's, and `check` and `train` refuse a record whose bound the host no longer allows. |

`write_recipe.profile_steps` is optional (the recipe's value is the only
one accepted).

## Running a project

A project runs the pipeline one stage at a time with `gatedtrain project run`; `gatedtrain project plan` shows what a
stage would do first, and `gatedtrain project report` prints the gate report at the end. Creating a project with a
keeper, confirming the metric and checking the signed records with `verify` are described in gatedtrain's
[AI-native projects](https://github.com/gatedtrain/gatedtrain/blob/main/docs/ai-native.md) guide. `smoke` and `train`
need the GPU and count against the project's GPU-hour budget; `evaluate_sealed` runs on the GPU too, in the keeper's
container.

## What the keeper checks and what is cooperative

This section follows gatedtrain's
[AI-native projects](https://github.com/gatedtrain/gatedtrain/blob/main/docs/ai-native.md) guide. What the keeper
guarantees about the sealed data depends on its mode ("Keeper modes"): in same-user mode these checks show that the
procedure was followed, not that the data stayed unseen.

The keeper checks, from its own signed records: the models it evaluates (it copies the files a model reference lists
into its own state and checks their hashes), the sealed set and its labels (the container gets the sealed inputs, never
the labels), the container it runs the generator in (rootless daemon, `--network none`, `--read-only`,
`--cap-drop ALL`, one container per arm), the alpha spent per attempt, and the container seconds of the report's
`compute` block. A `/out` that does not hold exactly one well-formed `predictions.jsonl` is a crash that has spent its
alpha.

What is cooperative: the training side. The job records, the GPU budget, the GPU log and the `ctx.backend` rules hold
for drivers and stages that use the backend; anything running as the owner can start containers on the owner's daemon
itself or edit the project's records. The budget is enforced while a driver polls the job. The backend's job totals the
driver sends with the report request are "the driver's job records (cooperative, not checked by the keeper)". The
rootful-access check does not cover sudo, and `rootless = true` in the host file is the host file's word.

## Limits

- Linux x86_64, one NVIDIA GPU, a CUDA 13.0 driver; CPython 3.12 for both venvs.
- The base model is Qwen3-0.6B only; the stage checks its architecture.
- The training script is the recipe's: TRL GRPO, `smoke` (10 steps) and `full` (100 steps) profiles, no resume (a
  preemption restarts from step 0), `beta=0.0` (no reference model).
- The scorer reads the final number only; the training reward (math-verify) is a different function, so a gain on the
  sealed set is measured by `gsm8k_answer`, not by the training reward.
- The runtime venv must be built without bytecode and kept under 100,000 entries.
- A prototype: the models it trains are not promotable.

## Settings that depend on the GPU

The batch generator (`generate_vllm.py`) runs in the keeper's container with `max_model_len=2048`,
`gpu_memory_utilization=0.85`, bfloat16 and, for greedy decoding, `max_num_seqs=1` with prefix caching off. The training
script (`train_grpo.py`) uses a vLLM memory share of 0.30 and a micro-batch of 8. They were chosen and probed on one
24 GB GPU and are part of the pack's code (the generator's hash is registered); they are not settings. `max_num_seqs=1` is
what makes greedy decoding deterministic and must stay.

**Re-probe on another GPU.** On a different GPU, or a different vLLM or driver, run the smoke stage and the greedy
determinism probe described under "Decoding" again before trusting the numbers: the peak memory against your bound, the
time of one arm against the generator's time limit, and the items that differ between the two seeds (it must be 0).
Raise `gpu_mem_limit_gb` in your host file only after measuring. The host file is written by you; a process
running as your own OS user (an agent with your shell) can edit it like any other file of yours, so the bound is recorded
and re-checked, not sealed.

## Changes

### 0.3.2 (Unreleased)

Needs gatedtrain 0.7.2 (`gatedtrain>=0.7.2,<0.8`), which adds `gatedtrain setup eval-tree`. No code of the pack
changed.

- **No sudo for the pack in a same-user keeper.** The keeper's eval tree is built by the user-level script
  `gatedtrain setup eval-tree` prints, from this pack's `eval-requirements.txt` (see "The keeper's eval tree"); before,
  only the root script of `gatedtrain setup host` built it. A separate-user keeper still needs that root script.
- Not changed: the trainer image, the pack's decoding and its other params.
- Checked on a GPU (one RTX 4090, gatedtrain 0.7.2 from source, same-user keeper, eval tree built by `gatedtrain setup
  eval-tree`): smoke, train (100 steps), sealed evaluation and gate ran end to end, gate PASS, base 0.6642 to trained
  0.7794 on 399 sealed items (a single run; the report states "same-user keeper: procedure followed; sealed data not
  protected from the user's account"). The tree was built by the version of the script before its last review fixes.

### 0.3.1

Needs gatedtrain 0.7.1 (`gatedtrain>=0.7.1,<0.8`): it uses the host file's `[packs]` tables, the function `launch.owner_gpu_mem_limit_gb` (it reads the host file's key `gpu_mem_limit_gb`) and the
keeper's `--eval-python`.


- **GPU facts are the owner's params.** `TORCH_CUDA_ARCH_LIST` is no longer fixed at `8.9` in the job: set `cuda_arch` on
  `smoke` and `train` (without it the job exports none and the toolchain reads the GPU). A pipeline copy that relied on
  the old behaviour on an 8.9 GPU adds `cuda_arch: "8.9"`. The smoke memory bound is the owner's (the host file's
  `gpu_mem_limit_gb`, default 22.5 GB, unchanged); the `memory_limit_gb` param of `smoke` can only lower it, a higher value is
  refused; the smoke record keeps both bounds. The runtime directory is the host file's `[packs.text-gsm8k] runtime_dir`
  and the eval interpreter the keeper's `--eval-python`; the two variables are deprecated workarounds (removed with
  gatedtrain 0.9.0 and pack 0.4.0). These need gatedtrain 0.7.1, which adds them.
- Values the pack shared with gatedtrain are imported from it (the generator's time limit, the eval venv's location, the
  memory bound); the GPU-hour estimates, the trl version (read from `runtime/uv.lock`) and the interpreter version in the
  build advice are each written once, and `tests/test_single_source.py` checks the copies that must stay in
  `train_grpo.py`.

## License

Apache License, Version 2.0 (`LICENSE`). See `NOTICE` for the copyright notice and `THIRD_PARTY_NOTICES.md` for the
third-party text the pack includes.

Names of third-party projects and products are trademarks of their owners; this pack is not affiliated with or
endorsed by them.
