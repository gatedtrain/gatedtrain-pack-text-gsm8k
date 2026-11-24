import ast
import difflib
import hashlib
import importlib.util
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import spec
from gatedtrain.ext import conformance, launch, registry
from gatedtrain.ext.rowkit import CONTAINER_PATHS

import gatedtrain_pack_text_gsm8k
from gatedtrain_pack_text_gsm8k import adapter

PACKAGE = Path(gatedtrain_pack_text_gsm8k.__file__).resolve().parent
SCRIPT = PACKAGE / "train_grpo.py"
LOCK_SHA256 = "392e60284236dd963a52a15bcd684fc225e5b7492f5c482108f64f7328cd68e7"
PROMPT = ("Solve the following math problem step by step. Put your final numerical answer after 'The answer is'."
          "\n\nProblem: {}")
RECIPE_ENV_NAMES = ("DO_NOT_TRACK", "FLASHINFER_WORKSPACE_BASE", "HF_HOME", "HF_HUB_OFFLINE", "HOME", "MASTER_ADDR",
                    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM", "TORCHINDUCTOR_CACHE_DIR",
                    "TORCH_CUDA_ARCH_LIST", "TRANSFORMERS_OFFLINE", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
                    "VLLM_CONFIG_ROOT", "VLLM_NO_USAGE_STATS", "VLLM_USE_FLASHINFER_SAMPLER", "XDG_CACHE_HOME")
BEGIN, END = "# pack: begin", "# pack: end"

_PATCHES = []


def setUpModule():
    """Register the pack's plugin for this module only (the way gatedtrain's tests install the stand-in)."""
    for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS,
                  spec.EVALUATOR_BACKENDS):
        _PATCHES.append(mock.patch.dict(table))
        _PATCHES[-1].start()
    adapter.install()


def tearDownModule():
    while _PATCHES:
        _PATCHES.pop().stop()


def load_script():
    """Import the pack's train_grpo.py by path (its module level imports the standard library only)."""
    spec_ = importlib.util.spec_from_file_location("pack_train_grpo", SCRIPT)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def marked_blocks(lines):
    """[(begin index, end index)] of the `# pack: begin` / `# pack: end` lines."""
    blocks, start = [], None
    for i, line in enumerate(lines):
        if line.strip() == BEGIN:
            assert start is None, f"nested {BEGIN} at line {i + 1}"
            start = i
        elif line.strip() == END:
            assert start is not None, f"{END} without {BEGIN} at line {i + 1}"
            blocks.append((start, i))
            start = None
    assert start is None, "unterminated pack block"
    return blocks


class AdapterRowTests(unittest.TestCase):
    def test_conforms(self):
        self.assertEqual(conformance.adapter_problems(adapter.NAME), [])

    def test_install_is_idempotent(self):
        adapter.install()
        self.assertEqual(registry.ADAPTERS[adapter.NAME], adapter.PLUGIN["adapters"][adapter.NAME])

    def test_row(self):
        row = registry.TRAINER_ROWS[adapter.NAME]
        self.assertEqual(adapter.NAME, "text-gsm8k-grpo")
        self.assertEqual(row["uses"], ("rl.trl_grpo@0.1",))
        self.assertIs(row["spec_binding"], False)
        self.assertEqual(row["env_names"], RECIPE_ENV_NAMES)
        self.assertEqual(row["container_paths"], CONTAINER_PATHS)
        self.assertIn("{output_root}", row["input_roots"])
        self.assertIn(adapter.RUNTIME, row["input_roots"])
        self.assertEqual(row["interpreter"], f"{adapter.RUNTIME}/.venv/bin/python")
        self.assertIs(registry.adapter(adapter.NAME).binds_spec, False)

    def test_output_root_is_an_input_root(self):
        cfg = launch.adapter_config(launch.HostConfig(host_work="/srv/gatedtrain/work",
                                                      output_root="/srv/gatedtrain/out"), adapter.NAME)
        self.assertIn("/srv/gatedtrain/out", cfg.input_roots)
        self.assertNotIn("{output_root}", cfg.input_roots)

    def test_collect_interface_refuses(self):
        from gatedtrain.errors import Refused

        ad = registry.adapter(adapter.NAME)
        with self.assertRaises(Refused):
            ad.find_cuts(Path("/nonexistent"))

    def test_no_framework_imports(self):
        for name in ("adapter.py", "job.py"):
            tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
            mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(mods & {"torch", "trl", "vllm", "transformers", "datasets"}, name)


class RuntimeFilesTests(unittest.TestCase):
    def test_lock_verbatim(self):
        self.assertEqual(hashlib.sha256((PACKAGE / "runtime" / "uv.lock").read_bytes()).hexdigest(), LOCK_SHA256)

    def test_runtime_pyproject_verbatim(self):
        recipe = os.environ.get("GATEDTRAIN_PACK_RECIPE_DIR")
        if not recipe:
            self.skipTest("set GATEDTRAIN_PACK_RECIPE_DIR to the recipe checkout")
        self.assertEqual((PACKAGE / "runtime" / "pyproject.toml").read_bytes(),
                         (Path(recipe) / "runtime" / "pyproject.toml").read_bytes())

    def test_prompt_file(self):
        # The recipe reads the prompt file whole (`read_text`, no strip), so the file has no trailing newline.
        self.assertEqual((PACKAGE / "prompt.txt").read_bytes(), PROMPT.encode("utf-8"))


class TrainScriptTests(unittest.TestCase):
    def test_train_script_matches_recipe_except_io(self):
        recipe = os.environ.get("GATEDTRAIN_PACK_RECIPE_DIR")
        if not recipe:
            self.skipTest("set GATEDTRAIN_PACK_RECIPE_DIR to the recipe checkout")
        old = (Path(recipe) / "runtime" / "train_grpo.py").read_text(encoding="utf-8").split("\n")
        new = SCRIPT.read_text(encoding="utf-8").split("\n")
        blocks = marked_blocks(new)
        self.assertTrue(blocks)
        hunks = [op for op in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if op[0] != "equal"]
        self.assertTrue(hunks)
        for tag, i1, i2, j1, j2 in hunks:
            if tag == "insert":     # the diff may place a block's surrounding blank lines inside the insertion
                while j1 < j2 and not new[j1].strip():
                    j1 += 1
                while j2 > j1 and not new[j2 - 1].strip():
                    j2 -= 1
                if j1 == j2:
                    continue
            if j2 > j1:     # pack lines j1 .. j2-1 lie in one block, markers included
                inside = any(b <= j1 and j2 - 1 <= e for b, e in blocks)
            else:           # a deletion before pack line j1
                inside = any(b < j1 <= e for b, e in blocks)
            self.assertTrue(inside, f"{tag} recipe lines {i1 + 1}-{i2}, pack lines {j1 + 1}-{j2} outside pack blocks")

    def test_inputs_and_profiles(self):
        mod = load_script()
        self.assertEqual((mod.DEFAULT_MODEL, mod.DEFAULT_TRAIN, mod.DEFAULT_PROMPT),
                         ("/inputs/base-model", "/inputs/train.jsonl", "/inputs/prompt.txt"))
        self.assertEqual(mod.PROFILES["smoke"]["max_steps"], 10)
        self.assertEqual(mod.PROFILES["full"]["max_steps"], 100)
        c = mod.COMMON
        self.assertEqual((c["prompts_per_step"], c["num_generations"], c["per_device_train_batch_size"],
                          c["learning_rate"], c["beta"], c["max_completion_length"]), (32, 8, 8, 5e-6, 0.0, 512))
        for profile in ("smoke", "full"):
            s = mod.resolve_settings(mod.parse_args(["--profile", profile, "--out", "/out"]))
            self.assertEqual(s["gradient_accumulation_steps"], 32)
        args = mod.parse_args(["--profile", "smoke", "--out", "/out", "--max-steps", "5", "--save-steps", "5"])
        self.assertEqual(mod.resolve_settings(args)["max_steps"], 5)

    def test_load_rows_uses_prompt(self):
        mod = load_script()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "train.jsonl").write_text('{"question": "What is 2 + 3?", "answer": "5"}\n', encoding="utf-8")
        shutil.copyfile(PACKAGE / "prompt.txt", tmp / "prompt.txt")
        rows = mod.load_rows(tmp / "train.jsonl", tmp / "prompt.txt")
        self.assertEqual(rows, [{"prompt": [{"role": "user", "content": PROMPT.format("What is 2 + 3?")}],
                                 "answer": "5", "row": 0}])

    def test_chmod_checkpoint(self):
        mod = load_script()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        cut = tmp / "checkpoint-5"
        (cut / "nested").mkdir(parents=True)
        for p in (cut / "model.safetensors", cut / "nested" / "x.json"):
            p.write_text("x")
            p.chmod(0o600)
        (cut / "nested").chmod(0o700)
        cut.chmod(0o700)
        mod.chmod_checkpoint(cut)
        for p in (cut, cut / "nested"):
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o755, p)
        for p in (cut / "model.safetensors", cut / "nested" / "x.json"):
            self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o644, p)

    def test_chmod_callback_wired(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("def on_save(self, args, state, control, **kwargs):", text)
        self.assertIn("make_chmod_callback()]", text)


if __name__ == "__main__":
    unittest.main()
