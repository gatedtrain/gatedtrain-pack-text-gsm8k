import dataclasses
import hashlib
import importlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gatedtrain import spec
from gatedtrain.ainative import backend
from gatedtrain.ainative.backend import job_problems
from gatedtrain.ainative.runtime import Mount
from gatedtrain.errors import Refused
from gatedtrain.ext import launch, registry

from gatedtrain_pack_text_gsm8k import adapter, job

IMAGE = ("docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04"
         "@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266")
LOCK_SHA256 = "392e60284236dd963a52a15bcd684fc225e5b7492f5c482108f64f7328cd68e7"
FAKE_GID = 2**31 - 3                     # the owner's primary group: no Docker socket's group on any host
TREE = "cpython-3.12.16-linux-x86_64-gnu"
RECIPE_ENV = {
    "DO_NOT_TRACK": "1", "FLASHINFER_WORKSPACE_BASE": "/tmp/cache", "HF_HOME": "/tmp/hf", "HF_HUB_OFFLINE": "1",
    "HOME": "/tmp/home", "MASTER_ADDR": "127.0.0.1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false", "TORCHINDUCTOR_CACHE_DIR": "/tmp/cache/inductor", "TORCH_CUDA_ARCH_LIST": "8.9",
    "TRANSFORMERS_OFFLINE": "1", "TRITON_CACHE_DIR": "/tmp/cache/triton", "VLLM_CACHE_ROOT": "/tmp/cache/vllm",
    "VLLM_CONFIG_ROOT": "/tmp/cache/vllm-config", "VLLM_NO_USAGE_STATS": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "XDG_CACHE_HOME": "/tmp/cache",
}


def reload_adapter(environ):
    """Re-import adapter.py under `environ` (its rows read the runtime directory at import) and register its plugin in
    fresh registry tables; the caller's cleanups restore both."""
    with mock.patch.dict(os.environ, environ, clear=True):
        importlib.reload(adapter)
    for table in (registry.ADAPTERS, registry.TRAINER_ROWS):
        table.pop(adapter.NAME, None)
    registry.EVALUATOR_ROWS.pop(adapter.EVALUATOR_ID, None)
    adapter.install()


class _Layout(unittest.TestCase):
    """A host layout in a temporary directory: runtime venv (linked to an interpreter tree via pyvenv.cfg), output
    root with the job's inputs and base model, host work directory."""

    def setUp(self):
        for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS,
                      spec.EVALUATOR_BACKENDS):
            p = mock.patch.dict(table)
            p.start()
            self.addCleanup(p.stop)
        # The identity mounts of the venv and the interpreter may not lie under the job's /tmp tmpfs (mounts never
        # nest), so the layout lives in a temporary directory beside this file rather than in the system's /tmp.
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="tmp-job-", dir=Path(__file__).resolve().parent)))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work, self.out_root = self.tmp / "work", self.tmp / "out"
        self.runtime, self.python = self.tmp / "runtime", self.tmp / "python"
        self.inputs = self.out_root / "gatedtrain-0123abcd" / "train-inputs"
        self.base = self.out_root / "gatedtrain-0123abcd" / "base-model"
        self.output = self.out_root / "gatedtrain-0123abcd" / "train"
        for d in (self.work, self.out_root, self.inputs, self.base, self.output, self.runtime / ".venv" / "bin",
                  self.python / TREE / "bin"):
            d.mkdir(parents=True, exist_ok=True)
        interp = self.python / TREE / "bin" / "python3.12"
        interp.write_text("fake interpreter")
        (self.runtime / ".venv" / "bin" / "python").symlink_to(interp)
        (self.runtime / ".venv" / "pyvenv.cfg").write_text(f"home = {self.python / TREE / 'bin'}\n")
        shutil.copyfile(job.LOCK_FILE, self.runtime / "uv.lock")
        (self.inputs / "train.jsonl").write_text('{"question": "q", "answer": "1"}\n')
        shutil.copyfile(job.PROMPT_FILE, self.inputs / "prompt.txt")
        shutil.copyfile(job.TRAIN_SCRIPT, self.inputs / "train_grpo.py")
        (self.base / "config.json").write_text("{}")
        self.addCleanup(reload_adapter, dict(os.environ))
        reload_adapter({**os.environ, adapter.RUNTIME_ENV: str(self.runtime)})
        # Input roots, interpreter and interpreter trees stay None: launch.adapter_config fills them from the row.
        self.cfg = launch.HostConfig(host_work=str(self.work), output_root=str(self.out_root),
                                     primary_gid=lambda uid: FAKE_GID, rootless=True)

    def job(self, profile="smoke", seed=42, **kw):
        j = job.train_job(profile=profile, image=IMAGE, runtime_dir=kw.pop("runtime_dir", self.runtime),
                          python_dir=self.python, inputs_dir=self.inputs, base_dir=self.base, seed=seed)
        return dataclasses.replace(j, output_dir=str(self.output), **kw)

    def assertRefusedTxt4(self, cm, *words):
        self.assertEqual([p.rule for p in cm.exception.problems], ["TXT-4"])
        text = str(cm.exception)
        for w in words:
            self.assertIn(w, text)


class JobTests(_Layout):
    def test_row_follows_the_runtime_variable(self):
        row = registry.TRAINER_ROWS[adapter.NAME]
        self.assertEqual(row["interpreter"], f"{self.runtime}/.venv/bin/python")
        self.assertEqual(row["interpreter_trees"], (str(self.python / TREE),))
        cfg = launch.adapter_config(self.cfg, adapter.NAME)
        self.assertEqual(set(cfg.input_roots), {str(self.python), str(self.runtime), str(self.out_root)})

    def test_job_problems_clean(self):
        self.assertEqual(job_problems(self.job(), self.cfg), [])
        self.assertEqual(job_problems(self.job("full"), self.cfg), [])

    def test_job_problems_clean_with_a_library_named_credentials(self):
        """A real venv holds library code whose path matches the credential name scan (a `credentials` package in
        site-packages); the row's scan skip applies only when the runtime directory, not the venv, is the mount."""
        cred = self.runtime / ".venv" / "lib" / "python3.12" / "site-packages" / "examplelib" / "lib" / "credentials"
        cred.mkdir(parents=True)
        (cred / "_auth.py").write_text("")
        self.assertEqual(job_problems(self.job(), self.cfg), [])

    def test_runtime_directory_is_mounted_not_the_venv(self):
        runtime, venv = str(self.runtime), str(self.runtime / ".venv")
        mounts = self.job().mounts
        self.assertEqual(mounts[0], Mount(runtime, runtime, True))
        self.assertNotIn(venv, [m.host for m in mounts] + [m.container for m in mounts])

    def test_job_problems_refuse_a_foreign_env_name(self):
        j = self.job()
        bad = dataclasses.replace(j, env={**j.env, "HF_TOKEN": "x"})
        self.assertTrue(job_problems(bad, self.cfg))

    def test_runtime_defaults_to_the_rows(self):
        j = job.train_job(profile="smoke", image=IMAGE, python_dir=self.python, inputs_dir=self.inputs,
                          base_dir=self.base, seed=42)
        self.assertEqual(j.argv[0], f"{self.runtime}/.venv/bin/python")

    def test_other_runtime_refused(self):
        other = self.tmp / "other-runtime"
        (other / ".venv" / "bin").mkdir(parents=True)
        with self.assertRaises(Refused) as cm:
            self.job(runtime_dir=other)
        self.assertRefusedTxt4(cm, str(other), adapter.RUNTIME_ENV)

    def test_runtime_without_venv_refused(self):
        shutil.rmtree(self.runtime / ".venv")
        with self.assertRaises(Refused) as cm:
            self.job()
        self.assertRefusedTxt4(cm, adapter.RUNTIME_ENV, "uv sync")

    def test_runtime_without_venv_python_refused(self):
        (self.runtime / ".venv" / "bin" / "python").unlink()
        with self.assertRaises(Refused) as cm:
            self.job()
        self.assertRefusedTxt4(cm, ".venv/bin/python", adapter.RUNTIME_ENV, "uv sync")

    def test_runtime_without_lock_refused(self):
        (self.runtime / "uv.lock").unlink()
        with self.assertRaises(Refused) as cm:
            self.job()
        self.assertRefusedTxt4(cm, str(self.runtime / "uv.lock"), "runtime/uv.lock")

    def test_runtime_with_another_lock_refused(self):
        (self.runtime / "uv.lock").write_text("version = 1\n")
        with self.assertRaises(Refused) as cm:
            self.job()
        self.assertRefusedTxt4(cm, str(self.runtime / "uv.lock"), LOCK_SHA256)

    def test_check_runtime_is_the_job_refusal(self):
        """The stages' plan calls `check_runtime`: the same refusal as `train_job`, before any job exists."""
        other = self.tmp / "other-runtime"
        (other / ".venv" / "bin").mkdir(parents=True)
        for runtime in (other, self.runtime):
            if runtime == self.runtime:
                (self.runtime / "uv.lock").unlink()
            with self.assertRaises(Refused) as a:
                job.check_runtime(runtime)
            with self.assertRaises(Refused) as b:
                self.job(runtime_dir=runtime)
            self.assertEqual(a.exception.problems, b.exception.problems)

    def test_unset_variable_and_packaged_runtime_without_venv_refused(self):
        reload_adapter({k: v for k, v in os.environ.items() if k != adapter.RUNTIME_ENV})
        self.assertEqual(adapter.RUNTIME, str(job.PACKAGE_DIR / "runtime"))
        self.assertFalse((job.PACKAGE_DIR / "runtime" / ".venv").exists())
        with self.assertRaises(Refused) as cm:
            job.train_job(profile="smoke", image=IMAGE, python_dir=self.python, inputs_dir=self.inputs,
                          base_dir=self.base, seed=42)
        self.assertRefusedTxt4(cm, adapter.RUNTIME_ENV, "uv sync")

    def test_job_fields(self):
        j = job.train_job(profile="smoke", image=IMAGE, runtime_dir=self.runtime, python_dir=self.python,
                          inputs_dir=self.inputs, base_dir=self.base, seed=7)
        self.assertEqual(j.adapter, adapter.NAME)
        self.assertEqual(dict(j.runtime), {"kind": "venv_in_container", "image": IMAGE,
                                           "venv_lock": str(self.runtime / "uv.lock")})
        self.assertEqual(hashlib.sha256(job.LOCK_FILE.read_bytes()).hexdigest(), LOCK_SHA256)
        self.assertEqual(j.argv, (f"{self.runtime}/.venv/bin/python", "/inputs/train_grpo.py", "--profile", "smoke",
                                  "--out", "/out", "--seed", "7"))
        runtime = str(self.runtime)
        self.assertEqual(j.mounts, (
            Mount(runtime, runtime, True), Mount(str(self.python), str(self.python), True),
            Mount(str(self.inputs / "train.jsonl"), "/inputs/train.jsonl", True),
            Mount(str(self.inputs / "prompt.txt"), "/inputs/prompt.txt", True),
            Mount(str(self.inputs / "train_grpo.py"), "/inputs/train_grpo.py", True),
            Mount(str(self.base), "/inputs/base-model", True)))
        self.assertEqual(dict(j.env), RECIPE_ENV)
        self.assertEqual((j.gpu, j.network, j.tmpfs, j.output_container, j.profile, j.training_seed),
                         (True, "none", ("/tmp",), "/out", "smoke", 7))
        self.assertEqual(j.framework["name"], registry.adapter(adapter.NAME).framework_name)

    def test_profile_values(self):
        for profile, est, limit in (("smoke", 0.25, 3600), ("full", 1.5, 14400)):
            j = self.job(profile)
            self.assertEqual((j.profile, j.gpu_hours_est, j.time_limit_s), (profile, est, limit))
            self.assertEqual(j.argv[2:4], ("--profile", profile))
        with self.assertRaises(ValueError):
            self.job("medium")

    def test_build_command_documented(self):
        self.assertIn("uv sync --frozen", job.__doc__)
        self.assertIn("UV_COMPILE_BYTECODE", job.__doc__)


class EntryCountTests(_Layout):
    def fill(self, n):
        lib = self.runtime / ".venv" / "lib"
        lib.mkdir(exist_ok=True)
        for i in range(n):
            (lib / f"f{i:05d}.py").write_text("")

    def test_job_key_below_the_limit(self):
        self.fill(2000)
        with mock.patch.object(backend, "MAX_INPUT_ENTRIES", 3000):
            key = self.job().key()
        self.assertRegex(key, r"^sha256:[0-9a-f]{64}$")

    def test_count_matches_the_key_walk(self):
        self.fill(50)
        j = self.job()
        n = job.input_entries(j.mounts)
        with mock.patch.object(backend, "MAX_INPUT_ENTRIES", n):
            j.key()
        with mock.patch.object(backend, "MAX_INPUT_ENTRIES", n - 1):
            with self.assertRaises(Refused):
                j.key()

    def test_over_the_limit_refused_before_key(self):
        self.fill(3500)
        with mock.patch.object(backend, "MAX_INPUT_ENTRIES", 3000):
            with self.assertRaises(Refused) as cm:
                self.job()
        self.assertRefusedTxt4(cm, "3000", "without bytecode", "uv sync")

    def test_documented_limit(self):
        self.assertEqual(backend.MAX_INPUT_ENTRIES, 100_000)


if __name__ == "__main__":
    unittest.main()
