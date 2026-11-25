"""One name, one place: values the pack shares with gatedtrain or with its own container scripts, and the settings that
belong to the owner's host."""

from __future__ import annotations

import ast
import io
import os
import re
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr
from pathlib import Path, PurePosixPath
from unittest import mock

from gatedtrain.ainative import hostsetup
from gatedtrain.ainative.batch import DEFAULT_TIME_LIMIT_S
from gatedtrain.ext.smoke import GPU_MEM_LIMIT_GB

from gatedtrain_pack_text_gsm8k import adapter, job, scenario, stages

PACKAGE = Path(stages.__file__).resolve().parent


def _literal(path: Path, name: str):
    """The value of the module-level assignment `name` in `path`, read without importing it (train_grpo.py imports
    torch and runs inside the job's container)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class SharedWithGatedtrain(unittest.TestCase):
    def test_the_generator_time_limit_is_gatedtrains_default(self) -> None:
        self.assertEqual(scenario.TIME_LIMIT_S, DEFAULT_TIME_LIMIT_S)

    def test_the_eval_python_is_in_the_eval_venv_gatedtrain_builds(self) -> None:
        venv = PurePosixPath(hostsetup.DEFAULT_EVAL_VENV)
        self.assertEqual(scenario.DEFAULT_EVAL_PYTHON, str(venv / "bin" / "python"))

    def test_the_memory_bound_is_gatedtrains(self) -> None:
        self.assertEqual(stages.MEMORY_LIMIT_GB, GPU_MEM_LIMIT_GB)


class SharedWithTheTrainingScript(unittest.TestCase):
    """train_grpo.py runs in the job's container and cannot import the pack; the pack's copies are checked here."""

    def test_the_steps_per_profile(self) -> None:
        profiles = _literal(PACKAGE / "train_grpo.py", "PROFILES")
        self.assertEqual(stages.PROFILE_STEPS, {k: v["max_steps"] for k, v in profiles.items()})

    def test_the_training_seed(self) -> None:
        self.assertEqual(stages.TRAINING_SEED, _literal(PACKAGE / "train_grpo.py", "COMMON")["seed"])

    def test_the_gpu_estimate_has_one_home(self) -> None:
        self.assertEqual(adapter.TRAINER_ROWS[adapter.NAME]["gpu_h_estimate"], job.PROFILES["full"]["gpu_hours_est"])

    def test_the_lock_pins_trl_exactly_once(self) -> None:
        lock = tomllib.loads(job.LOCK_FILE.read_text())
        self.assertEqual([p["name"] for p in lock["package"]].count("trl"), 1)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            job._locked_version("no-such-package")

    def test_the_trl_version_is_the_locks(self) -> None:
        lock = tomllib.loads(job.LOCK_FILE.read_text())
        locked = next(p["version"] for p in lock["package"] if p["name"] == "trl")
        self.assertEqual(job.FRAMEWORK["version"], locked)


class BuildAdviceNamesNoPlatform(unittest.TestCase):
    def test_one_interpreter_version(self) -> None:
        self.assertRegex(job.PYTHON_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertIn(job.PYTHON_VERSION, job.BUILD)
        self.assertEqual(len(re.findall(r"\d+\.\d+\.\d+", job.BUILD)), 1, job.BUILD)
        self.assertNotIn("x86_64", job.BUILD)
        self.assertIn("<platform>", job.BUILD)


class GpuArchitecture(unittest.TestCase):
    """TORCH_CUDA_ARCH_LIST names a GPU of the owner's host: a stage parameter, left out unless set."""

    def test_the_job_env_has_no_architecture_by_default(self) -> None:
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", job.ENV)

    def test_format(self) -> None:
        for good in ("8.9", "9.0", "9.0a", "12.0a", "8.9+PTX", "7.5 8.0"):
            self.assertTrue(job.valid_cuda_arch(good), good)
        for bad in ("", "89", "8.9;", "8.0;8.6", "a.b", "8.9\n", "8.9; rm", "8.9,9.0", "8.9  9.0", "8.9 " * 40, None, 8.9):
            self.assertFalse(job.valid_cuda_arch(bad), bad)

    def test_the_stage_params_name_it(self) -> None:
        for cls in (stages.Smoke, stages.Train):
            self.assertIn("cuda_arch", cls.params)
            self.assertEqual("memory_limit_gb" in cls.params, cls is stages.Smoke)

    def test_the_pipeline_file_documents_the_optional_params(self) -> None:
        text = (PACKAGE / "pipeline.yaml").read_text()
        self.assertIn("cuda_arch", text)
        self.assertIn("memory_limit_gb", text)


class OwnerChannels(unittest.TestCase):
    """The owner's settings come from the host file and the keeper's options; a variable is a named, deprecated
    workaround that says so."""

    def host_file(self, text: str) -> dict[str, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "host.toml"
        path.write_text(text, encoding="utf-8")
        return {"GATEDTRAIN_HOST_CONFIG": str(path)}

    def runtime(self, env: dict[str, str]) -> str:
        from gatedtrain.ext.rowkit import runtime_dir
        with mock.patch.dict(os.environ, env), redirect_stderr(io.StringIO()):
            os.environ.pop(adapter.RUNTIME_ENV, None) if adapter.RUNTIME_ENV not in env else None
            return runtime_dir(adapter.RUNTIME_ENV, adapter.__file__, parents=0, pack=adapter.PACK)

    def test_the_runtime_directory_is_a_host_file_setting(self) -> None:
        self.assertEqual(adapter.PACK, "text-gsm8k")
        env = self.host_file('[packs.text-gsm8k]\nruntime_dir = "/owner/runtime"\n')
        self.assertEqual(self.runtime(env), "/owner/runtime")
        self.assertTrue(self.runtime(self.host_file("")).endswith("runtime"))

    def test_the_runtime_variable_is_deprecated_and_says_so(self) -> None:
        err = io.StringIO()
        from gatedtrain.ext import rowkit
        rowkit._WARNED.discard(adapter.RUNTIME_ENV)                 # the notice is printed once per process
        with mock.patch.dict(os.environ, {adapter.RUNTIME_ENV: "/e"}), redirect_stderr(err):
            from gatedtrain.ext.rowkit import runtime_dir
            self.assertEqual(runtime_dir(adapter.RUNTIME_ENV, adapter.__file__, parents=0, pack=adapter.PACK), "/e")
        self.assertIn("deprecated", err.getvalue())
        self.assertIn("[packs.text-gsm8k]", err.getvalue())

    def test_the_eval_python_variable_is_deprecated_and_says_so(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(scenario.eval_python({scenario.EVAL_PYTHON_ENV: "/e/bin/python"}), "/e/bin/python")
            self.assertEqual(scenario.eval_python({}), scenario.DEFAULT_EVAL_PYTHON)
        self.assertEqual(err.getvalue().count("deprecated"), 1)
        self.assertIn("--eval-python", err.getvalue())


if __name__ == "__main__":
    unittest.main()
