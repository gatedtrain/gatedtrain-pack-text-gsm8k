"""The pack stages that run through the backend: base files, train set, recipe, smoke and train.

No test starts Docker or reads a GPU: the backend is gatedtrain's `LocalBackend` on `FakeRuntime`, whose handler
plays the training job by writing the files the pack's `train_grpo.py` writes at the top of /out (`samples.jsonl`
through the script's own `GSM8KReward._log`, `memory.jsonl` with its memory callback's keys) and `checkpoint-<N>/`
directories whose files come out 0600, then, unless a test says otherwise, applies the script's own
`chmod_checkpoint`.
"""

import datetime
import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gatedtrain import spec
from gatedtrain.ainative.backend import LocalBackend, StageBackend
from gatedtrain.ainative.project import create_project
from gatedtrain.ainative.stage import StageContext
from gatedtrain.ainative.trust_stages import SCHEMA_DEPLOYMENT, SCHEMA_MODEL_REF, SCHEMA_TRAIN_PART
from gatedtrain.errors import Refused
from gatedtrain.ext import launch, registry
from gatedtrain.testing.fake_runtime import FakeRuntime

from _pack_fakes import HF_TEMPLATE, SCRIPT, SIX, play_training, write_hf_hub
from gatedtrain_pack_text_gsm8k import adapter, job, stages
from gatedtrain_pack_text_gsm8k.stages import (BuildTrainSet, ChooseTaskAndBase, ProfileData, Smoke, Train,
                                               WriteRecipe)

IMAGE = ("docker.io/nvidia/cuda:13.0.3-devel-ubuntu24.04"
         "@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266")
FAKE_GID = 2**31 - 3                     # the owner's primary group: no Docker socket's group on any host
TREE = "cpython-3.12.16-linux-x86_64-gnu"
CLOCK = datetime.datetime(2026, 11, 23, tzinfo=datetime.timezone.utc)
SAMPLER = {"name": "vllm", "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 1024, "enable_thinking": False}




def reload_adapter(environ):
    """Re-import adapter.py under `environ` (its rows read the runtime directory at import) and register its plugin
    in the (patched) registry tables."""
    with mock.patch.dict(os.environ, environ, clear=True):
        importlib.reload(adapter)
    for table in (registry.ADAPTERS, registry.TRAINER_ROWS):
        table.pop(adapter.NAME, None)
    registry.EVALUATOR_ROWS.pop(adapter.EVALUATOR_ID, None)
    adapter.install()


def mode(path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


class _Layout(unittest.TestCase):
    """A host layout beside this file (the job's identity mounts may not lie under its /tmp tmpfs): a runtime venv
    linked to an interpreter tree, the output root, the host work directory, a hub download of the base model; a
    project and a `LocalBackend` on `FakeRuntime` whose handler plays `train_grpo.py`."""

    def setUp(self):
        for table in (registry.ADAPTERS, registry.TRAINER_ROWS, registry.EVALUATOR_ROWS, spec.TRAINER_BACKENDS,
                      spec.EVALUATOR_BACKENDS):
            p = mock.patch.dict(table)
            p.start()
            self.addCleanup(p.stop)
        self.tmp = Path(os.path.realpath(tempfile.mkdtemp(prefix="tmp-job-", dir=Path(__file__).resolve().parent)))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.work, self.out_root = self.tmp / "work", self.tmp / "out"
        self.runtime, self.python, self.hub = self.tmp / "runtime", self.tmp / "python", self.tmp / "hub"
        for d in (self.work, self.out_root, self.runtime / ".venv" / "bin", self.python / TREE / "bin", self.hub):
            d.mkdir(parents=True, exist_ok=True)
        self.out_root.chmod(0o755)
        interp = self.python / TREE / "bin" / "python3.12"
        interp.write_text("fake interpreter")
        (self.runtime / ".venv" / "bin" / "python").symlink_to(interp)
        (self.runtime / ".venv" / "pyvenv.cfg").write_text(f"home = {self.python / TREE / 'bin'}\n")
        shutil.copyfile(job.LOCK_FILE, self.runtime / "uv.lock")
        for name in SIX + ("LICENSE", "README.md", "merges.txt", "vocab.json"):
            (self.hub / name).write_text(f"hub {name}\n")
        (self.hub / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
        self.addCleanup(reload_adapter, dict(os.environ))
        reload_adapter({**os.environ, adapter.RUNTIME_ENV: str(self.runtime)})
        self.cfg = launch.HostConfig(host_work=str(self.work), output_root=str(self.out_root),
                                     primary_gid=lambda uid: FAKE_GID, rootless=True)
        data = self.tmp / "data.jsonl"
        data.write_text("{}\n")
        self.proj = create_project(self.tmp / "proj", goal="g", data=data, pipeline="p.yaml",
                                   budget={"gpu_hours": 10.0, "llm_usd": 0})
        # What the fake training job writes.
        self.rewards = [1.0, 0.0] * 4                     # per completion of each prompt row
        self.texts = None                                 # None: a distinct text per completion
        self.peak_gb = 19.8
        self.chmod = True
        self.seen_train = []                              # the /inputs/train.jsonl each run saw
        self.rt = FakeRuntime({IMAGE: self.play})
        self.backend = LocalBackend(self.proj, self.rt, host=self.cfg, sleep=lambda s: None, clock=lambda: CLOCK)
        self.params = {"image": IMAGE, "runtime_dir": str(self.runtime), "python_dir": str(self.python)}

    # -- the fake training job ---------------------------------------------------------------------------------------

    def play(self, run_spec, host_path) -> int:
        self.seen_train.append(host_path("/inputs/train.jsonl").read_bytes())
        return play_training(run_spec, host_path, rewards=self.rewards, texts=self.texts, peak_gb=self.peak_gb,
                             chmod=self.chmod)

    # -- stage driving -----------------------------------------------------------------------------------------------

    def ctx(self, ref, inputs, params=None, *, gpu=False, planning=False):
        sb = StageBackend(self.backend, stage=ref, allow_gpu=gpu, planning=planning)
        return StageContext(self.proj, inputs, dict(params or {}), None, None, stage=ref, backend=sb)

    def train_part(self, n=12):
        rows = [{"input": f"Question {i}: what is {i} + {i}?", "label": str(2 * i)} for i in range(n)]
        rec = {"schema": SCHEMA_TRAIN_PART, "schema_version": 1, "n": n, "rows": rows}
        return self.proj.write_artifact("intake_seal", "train_part", rec, SCHEMA_TRAIN_PART)

    def base(self):
        tp = self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub)})
        out = ChooseTaskAndBase().run(ctx, {})
        return tp, ctx, out

    def upto_recipe(self):
        tp, _, out = self.base()
        ts = BuildTrainSet().run(self.ctx("build_train_set", {"train_part": tp}), {})["train_set"]
        recipe = WriteRecipe().run(self.ctx("write_recipe", {"train_set": ts}), {})["recipe"]
        return {"base": out["base"], "train_set": ts, "recipe": recipe}

    def smoke(self, inputs):
        ctx = self.ctx("smoke", inputs, self.params, gpu=True)
        return ctx, Smoke().run(ctx, {})

    def train(self, inputs, smoke_ref):
        ctx = self.ctx("train", {**inputs, "smoke": smoke_ref}, self.params, gpu=True)
        return ctx, Train().run(ctx, {})

    def job_ids(self):
        return self.backend._job_ids()

    def assertRule(self, cm, rule, *words):
        self.assertEqual({p.rule for p in cm.exception.problems}, {rule}, str(cm.exception))
        for w in words:
            self.assertIn(w, str(cm.exception))


class SampleLogTests(unittest.TestCase):
    def test_reward_call_logs_completion_hashes(self):
        """The script's reward call (TRL's entry, math-verify replaced) passes the completion texts to `_log`."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        reward = SCRIPT.GSM8KReward(tmp / "samples.jsonl", 2)
        comps = [[{"role": "assistant", "content": t}] for t in ("The answer is 4", "4", "The answer is 4")]
        with mock.patch.object(SCRIPT, "math_reward", lambda response, gt: float(response.endswith("is 4"))):
            got = reward(prompts=[None] * 3, completions=comps, answer=["4"] * 3, row=[0, 0, 1],
                         trainer_state=SimpleNamespace(global_step=0, epoch=0.0))
        self.assertEqual(got, [1.0, 0.0, 1.0])
        lines = [json.loads(x) for x in (tmp / "samples.jsonl").read_text().split("\n") if x]
        h = [hashlib.sha256(t.encode()).hexdigest() for t in ("The answer is 4", "4")]
        self.assertEqual([(r["row"], r["completions"], r["reward_mean"], r["completion_sha256"]) for r in lines],
                         [(0, 2, 0.5, h), (1, 1, 1.0, [h[0]])])
        m = stages.smoke_measures(lines, [{"step": 1, "torch_max_reserved_gb": 1.0}])
        self.assertEqual((m["n_samples"], m["n_distinct_completions"]), (3, 2))


class StageShapeTests(unittest.TestCase):
    def test_names_and_declarations(self):
        self.assertEqual([c.name for c in (ChooseTaskAndBase, BuildTrainSet, WriteRecipe, Smoke, Train)],
                         ["gsm8k_choose_task_and_base", "gsm8k_build_train_set", "gsm8k_write_recipe", "gsm8k_smoke",
                          "gsm8k_train"])
        self.assertEqual(ChooseTaskAndBase.outputs, {"deployment": SCHEMA_DEPLOYMENT, "base": SCHEMA_MODEL_REF})
        self.assertEqual(ChooseTaskAndBase.params, frozenset({"base_model_dir"}))
        self.assertEqual(WriteRecipe.params, frozenset({"profile_steps"}))
        for cls in (Smoke, Train):
            self.assertEqual(cls.needs, frozenset({"gpu"}))
            self.assertEqual(cls.params, frozenset({"image", "runtime_dir", "python_dir"}))
        self.assertIs(Train.trains, True)
        self.assertIs(Smoke.trains, False)
        self.assertEqual(Train.outputs, {"model": SCHEMA_MODEL_REF})
        self.assertEqual(stages.MEMORY_LIMIT_GB, 22.5)


class BaseTests(_Layout):
    def test_base_copies_only_deployable(self):
        _, ctx, out = self.base()
        base = self.proj.read_artifact(out["base"])
        self.assertEqual((base["generator"], base["label"]), ("text_gsm8k_vllm", "base"))
        self.assertEqual([Path(f["path"]).name for f in base["files"]], list(SIX))
        [d] = {Path(f["path"]).parent for f in base["files"]}
        self.assertEqual(sorted(os.listdir(d)), sorted(SIX))
        self.assertTrue(str(d).startswith(str(self.out_root) + "/"))
        for f in base["files"]:
            self.assertEqual(mode(f["path"]), 0o644)
            data = Path(f["path"]).read_bytes()
            self.assertEqual(data, (self.hub / Path(f["path"]).name).read_bytes())
            self.assertEqual(f["sha256"], "sha256:" + hashlib.sha256(data).hexdigest())
        dep = self.proj.read_artifact(out["deployment"])
        self.assertEqual((dep["sampler"], dep["eval_seeds"]), (SAMPLER, [11, 12, 13, 14]))
        self.assertEqual(ChooseTaskAndBase().check(ctx, out), [])
        # A re-run leaves identical copies as they are (same inode: a job that mounts them keeps its key).
        inode = os.stat(base["files"][2]["path"]).st_ino
        ChooseTaskAndBase().run(ctx, {})
        self.assertEqual(os.stat(base["files"][2]["path"]).st_ino, inode)

    def use_hf_hub(self, template=HF_TEMPLATE):
        shutil.rmtree(self.hub)
        write_hf_hub(self.hub, template)

    def test_base_real_hf_layout_writes_the_template(self):
        """The hub download at the pinned commit has no chat_template.jinja: the stage writes it from the string
        `chat_template` of tokenizer_config.json, byte for byte, and the six-file reference holds."""
        self.use_hf_hub()
        self.assertFalse((self.hub / "chat_template.jinja").exists())
        _, ctx, out = self.base()
        base = self.proj.read_artifact(out["base"])
        self.assertEqual([Path(f["path"]).name for f in base["files"]], list(SIX))
        [d] = {Path(f["path"]).parent for f in base["files"]}
        self.assertEqual(sorted(os.listdir(d)), sorted(SIX))
        jinja = d / "chat_template.jinja"
        self.assertEqual(jinja.read_bytes(), HF_TEMPLATE.encode("utf-8"))
        self.assertEqual(mode(jinja), 0o644)
        self.assertEqual(base["files"][5]["sha256"],
                         "sha256:" + hashlib.sha256(HF_TEMPLATE.encode("utf-8")).hexdigest())
        for f in base["files"][:5]:
            self.assertEqual(Path(f["path"]).read_bytes(), (self.hub / Path(f["path"]).name).read_bytes())
        self.assertEqual(ChooseTaskAndBase().check(ctx, out), [])
        self.assertFalse((self.hub / "chat_template.jinja").exists())        # the source is left as it is
        inode = jinja.stat().st_ino
        ChooseTaskAndBase().run(ctx, {})
        self.assertEqual(jinja.stat().st_ino, inode)
        self.assertIn("chat_template", ChooseTaskAndBase().plan(ctx).summary)

    def test_base_real_hf_layout_runs_smoke(self):
        self.use_hf_hub()
        inputs = self.upto_recipe()
        _, out = self.smoke(inputs)
        self.assertEqual(self.proj.read_artifact(out["smoke"])["verdict"], "PASS")

    def test_base_without_any_template_refused(self):
        for template in (None, "", ["named"], {"default": "x"}, 3):
            with self.subTest(template=template):
                self.use_hf_hub(template)
                tp = self.train_part()
                prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
                ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub)})
                for call in (lambda: ChooseTaskAndBase().plan(ctx), lambda: ChooseTaskAndBase().run(ctx, {})):
                    with self.assertRaises(Refused) as cm:
                        call()
                    self.assertRule(cm, "TXT-2", "chat_template")
                self.assertFalse(any(self.out_root.rglob("config.json")))

    def test_base_unreadable_tokenizer_config_refused(self):
        self.use_hf_hub()
        (self.hub / "tokenizer_config.json").write_text("not json")
        ctx = self.ctx("choose_task_and_base", {}, {"base_model_dir": str(self.hub)})
        with self.assertRaises(Refused) as cm:
            ChooseTaskAndBase()._source(ctx)
        self.assertRule(cm, "TXT-2", "tokenizer_config.json")

    def test_base_missing_file_refused(self):
        (self.hub / "tokenizer.json").unlink()
        tp = self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub)})
        for call in (lambda: ChooseTaskAndBase().plan(ctx), lambda: ChooseTaskAndBase().run(ctx, {})):
            with self.assertRaises(Refused) as cm:
                call()
            self.assertRule(cm, "TXT-2", "tokenizer.json")
        self.assertFalse(any((self.out_root).rglob("config.json")))

    def test_base_other_architecture_refused(self):
        (self.hub / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
        ctx = self.ctx("choose_task_and_base", {}, {"base_model_dir": str(self.hub)})
        with self.assertRaises(Refused) as cm:
            ChooseTaskAndBase()._source(ctx)
        self.assertRule(cm, "TXT-2", "Qwen3ForCausalLM")

    def test_base_dir_param_refused(self):
        ctx = self.ctx("choose_task_and_base", {}, {"base_model_dir": "relative/dir"})
        with self.assertRaises(Refused) as cm:
            ChooseTaskAndBase()._source(ctx)
        self.assertRule(cm, "TXT-2", "base_model_dir")

    def test_plan_creates_nothing(self):
        tp = self.train_part()
        prof = ProfileData().run(self.ctx("profile_data", {"train_part": tp}), {})["profile"]
        ctx = self.ctx("choose_task_and_base", {"profile": prof}, {"base_model_dir": str(self.hub)}, planning=True)
        prop = ChooseTaskAndBase().plan(ctx)
        self.assertEqual(prop.estimate, {"gpu_hours": 0.0})
        self.assertEqual(prop.choices["sampler"], SAMPLER)
        self.assertEqual(list(self.out_root.iterdir()), [])


class TrainSetAndRecipeTests(_Layout):
    def test_build_train_set(self):
        tp = self.train_part(3)
        ctx = self.ctx("build_train_set", {"train_part": tp})
        out = BuildTrainSet().run(ctx, {})
        ts = self.proj.read_artifact(out["train_set"])
        self.assertEqual(ts["n"], 3)
        self.assertEqual(ts["rows"][1], {"question": "Question 1: what is 1 + 1?", "answer": "2"})
        data = stages.train_jsonl(ts["rows"])
        self.assertEqual(ts["train_jsonl_sha256"], "sha256:" + hashlib.sha256(data).hexdigest())
        self.assertEqual([json.loads(line) for line in data.decode().split("\n") if line], ts["rows"])
        self.assertEqual(BuildTrainSet().check(ctx, out), [])
        self.assertEqual(BuildTrainSet().plan(ctx).estimate, {"gpu_hours": 0.0, "n": 3})

    def test_train_jsonl_keeps_line_separators_inside_a_line(self):
        data = stages.train_jsonl([{"question": "a b\rc", "answer": "1"}])
        self.assertEqual(data.count(b"\n"), 1)
        self.assertEqual(json.loads(data)["question"], "a b\rc")

    def test_bad_label_refused(self):
        rec = {"schema": SCHEMA_TRAIN_PART, "schema_version": 1, "n": 1, "rows": [{"input": "q", "label": "x"}]}
        tp = self.proj.write_artifact("intake_seal", "train_part", rec, SCHEMA_TRAIN_PART)
        with self.assertRaises(Refused) as cm:
            BuildTrainSet().run(self.ctx("build_train_set", {"train_part": tp}), {})
        self.assertRule(cm, "TXT-1")

    def test_write_recipe(self):
        ts = BuildTrainSet().run(self.ctx("build_train_set", {"train_part": self.train_part(3)}), {})["train_set"]
        ctx = self.ctx("write_recipe", {"train_set": ts})
        out = WriteRecipe().run(ctx, {})
        r = self.proj.read_artifact(out["recipe"])
        self.assertEqual(r["profile_steps"], {"smoke": 10, "full": 100})
        self.assertEqual(r["train_set_sha256"], ts.sha256)
        self.assertEqual(r["script_sha256"], "sha256:" + hashlib.sha256(job.TRAIN_SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(r["prompt_sha256"], "sha256:" + hashlib.sha256(job.PROMPT_FILE.read_bytes()).hexdigest())
        self.assertEqual(r["seed"], 42)
        self.assertEqual(WriteRecipe().check(ctx, out), [])
        self.assertEqual(WriteRecipe().plan(ctx).estimate, {"gpu_hours": 1.75})

    def test_write_recipe_refuses_steps_the_job_cannot_run(self):
        ts = BuildTrainSet().run(self.ctx("build_train_set", {"train_part": self.train_part(3)}), {})["train_set"]
        for bad in ({"smoke": 10, "full": 200}, {"smoke": 10}, {"smoke": 10, "full": 100, "extra": 1}, "10",
                    {"smoke": True, "full": 100}):
            ctx = self.ctx("write_recipe", {"train_set": ts}, {"profile_steps": bad})
            with self.assertRaises(Refused) as cm:
                WriteRecipe().plan(ctx)
            self.assertRule(cm, "TXT-5", "profile_steps")


class SmokeTests(_Layout):
    def test_smoke_pass(self):
        inputs = self.upto_recipe()
        ctx, out = self.smoke(inputs)
        s = self.proj.read_artifact(out["smoke"])
        self.assertEqual((s["verdict"], s["reasons"]), ("PASS", []))
        self.assertEqual(s["n_samples"], 10 * 2 * 8)
        self.assertEqual(s["n_distinct_completions"], 160)
        self.assertEqual(s["reward_mean"], 0.5)
        self.assertEqual(s["peak_gpu_mem_gb"], 19.8)
        self.assertEqual(s["steps"], 10)
        self.assertEqual(Smoke().check(ctx, out), [])
        [jid] = self.job_ids()
        self.assertEqual(s["job_id"], jid)
        rec = json.loads((self.proj.root / "jobs" / jid / "job.json").read_text())
        self.assertEqual((rec["stage"], rec["job"]["gpu"], rec["job"]["profile"]), ("smoke", True, "smoke"))
        ts = self.proj.read_artifact(inputs["train_set"])
        self.assertEqual(self.seen_train, [stages.train_jsonl(ts["rows"])])

    def test_inputs_directory_named_after_the_train_set(self):
        """Two training sets never share an inputs directory: its name carries the training set's hash."""
        inputs = self.upto_recipe()
        self.smoke(inputs)
        [jid] = self.job_ids()
        rec = json.loads((self.proj.root / "jobs" / jid / "job.json").read_text())
        digest = inputs["train_set"].sha256.removeprefix("sha256:")[:12]
        [train] = [Path(m["host"]) for m in rec["job"]["mounts"] if Path(m["host"]).name == "train.jsonl"]
        self.assertEqual(train.parent.name, f"train-inputs-{digest}")

    def test_smoke_degenerate_refused(self):
        inputs = self.upto_recipe()
        self.rewards = [0.0] * 8
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "MM-38", "all rewards are 0")

    def test_smoke_one_completion_refused(self):
        inputs = self.upto_recipe()
        self.texts = ["The answer is 4"] * 16
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "MM-38", "1 distinct completion")

    def test_smoke_memory_refused(self):
        inputs = self.upto_recipe()
        self.peak_gb = 23.1
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "MM-38", "23.1", "22.5")

    def test_smoke_plan_has_no_side_effects(self):
        inputs = self.upto_recipe()
        before = sorted(p.name for p in next(self.out_root.iterdir()).iterdir())
        passed = self.proj.write_artifact("smoke", "smoke", {
            "schema": stages.SCHEMA_SMOKE, "schema_version": 1, "verdict": "PASS", "reasons": [],
            **{f"{k}_sha256": inputs[k].sha256 for k in ("base", "train_set", "recipe")}}, stages.SCHEMA_SMOKE)
        for cls, ref, extra in ((Smoke, "smoke", {}), (Train, "train", {"smoke": passed})):
            ctx = self.ctx(ref, {**inputs, **extra}, self.params, gpu=True, planning=True)
            prop = cls().plan(ctx)
            self.assertEqual(prop.estimate, {"gpu_hours": job.PROFILES["smoke" if cls is Smoke else "full"]
                                             ["gpu_hours_est"]})
        self.assertEqual(self.job_ids(), [])
        self.assertEqual(sorted(p.name for p in next(self.out_root.iterdir()).iterdir()), before)

    def test_smoke_bad_params_refused_in_plan(self):
        inputs = self.upto_recipe()
        for key, bad in (("image", ""), ("python_dir", "rel"), ("runtime_dir", None)):
            ctx = self.ctx("smoke", inputs, {**self.params, key: bad}, gpu=True, planning=True)
            with self.assertRaises(Refused) as cm:
                Smoke().plan(ctx)
            self.assertRule(cm, "TXT-4", key)


    def plan_and_run_refusals(self, inputs, params):
        """The refusal of Smoke's and Train's plan and of Smoke's run under `params`; no job is submitted."""
        passed = self.proj.write_artifact("smoke", "smoke", {
            "schema": stages.SCHEMA_SMOKE, "schema_version": 1, "verdict": "PASS", "reasons": [],
            **{f"{k}_sha256": inputs[k].sha256 for k in ("base", "train_set", "recipe")}}, stages.SCHEMA_SMOKE)
        got = []
        for cls, ref, extra in ((Smoke, "smoke", {}), (Train, "train", {"smoke": passed})):
            ctx = self.ctx(ref, {**inputs, **extra}, params, gpu=True, planning=True)
            with self.assertRaises(Refused) as cm:
                cls().plan(ctx)
            got.append(cm.exception.problems)
        with self.assertRaises(Refused) as cm:
            Smoke().run(self.ctx("smoke", inputs, params, gpu=True), {})
        got.append(cm.exception.problems)
        self.assertEqual(self.job_ids(), [])
        self.assertEqual(self.rt.runs, [])
        return got

    def test_plan_refuses_another_runtime_dir(self):
        """runtime_dir is not the adapter's runtime (GATEDTRAIN_PACK_TEXT_GSM8K_RUNTIME): plan refuses with the
        message of the run-time refusal, before a GPU job exists."""
        inputs = self.upto_recipe()
        other = self.tmp / "other-runtime"
        (other / ".venv" / "bin").mkdir(parents=True)
        (other / ".venv" / "bin" / "python").symlink_to(self.python / TREE / "bin" / "python3.12")
        shutil.copyfile(job.LOCK_FILE, other / "uv.lock")
        plan_smoke, plan_train, run = self.plan_and_run_refusals(inputs, {**self.params, "runtime_dir": str(other)})
        self.assertEqual(plan_smoke, plan_train)
        self.assertEqual(plan_smoke, run)
        self.assertEqual([p.rule for p in run], ["TXT-4"])
        self.assertIn(adapter.RUNTIME_ENV, str(run))
        self.assertIn(str(other), str(run))

    def test_plan_refuses_a_runtime_without_venv_python(self):
        inputs = self.upto_recipe()
        (self.runtime / ".venv" / "bin" / "python").unlink()
        plan_smoke, plan_train, run = self.plan_and_run_refusals(inputs, self.params)
        self.assertEqual(plan_smoke, run)
        self.assertEqual(plan_train, run)
        self.assertEqual([p.rule for p in run], ["TXT-4"])
        self.assertIn(".venv/bin/python", str(run))

    def test_plan_refuses_a_runtime_without_the_lock(self):
        inputs = self.upto_recipe()
        (self.runtime / "uv.lock").unlink()
        plan_smoke, plan_train, run = self.plan_and_run_refusals(inputs, self.params)
        self.assertEqual(plan_smoke, run)
        self.assertEqual(plan_train, run)
        self.assertEqual([p.rule for p in run], ["TXT-4"])
        self.assertIn("uv.lock", str(run))


class TrainTests(_Layout):
    def ready(self):
        inputs = self.upto_recipe()
        _, out = self.smoke(inputs)
        return inputs, out["smoke"]

    def test_train_hands_off_last_checkpoint(self):
        inputs, smoke_ref = self.ready()
        ctx, out = self.train(inputs, smoke_ref)
        m = self.proj.read_artifact(out["model"])
        self.assertEqual((m["generator"], m["label"], m["checkpoint"]), ("text_gsm8k_vllm", "candidate", 100))
        self.assertEqual([Path(f["path"]).name for f in m["files"]], list(SIX))
        for f in m["files"]:
            p = Path(f["path"])
            self.assertEqual(p.parent.name, "checkpoint-100")
            self.assertEqual(p.read_text(), f"full step 100 {p.name}\n")
            self.assertEqual(mode(p), 0o644)
            self.assertEqual(f["sha256"], "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(Train().check(ctx, out), [])
        self.assertEqual(m["job_id"], self.job_ids()[-1])

    def test_train_without_chmod_refused_by_hand_off(self):
        inputs, smoke_ref = self.ready()
        self.chmod = False
        with self.assertRaises(Refused) as cm:
            self.train(inputs, smoke_ref)
        self.assertRule(cm, "AN-22", "checkpoint-100", "o+r")

    def test_train_missing_file_refused(self):
        inputs, smoke_ref = self.ready()
        real = self.play

        def play(run_spec, host_path):
            rc = real(run_spec, host_path)
            if "full" in run_spec.argv:
                (host_path("/out") / "checkpoint-100" / "chat_template.jinja").unlink()
            return rc
        self.rt.handlers[IMAGE] = play
        with self.assertRaises(Refused) as cm:
            self.train(inputs, smoke_ref)
        self.assertRule(cm, "TXT-3", "checkpoint-100", "chat_template.jinja")

    def test_train_refuses_without_smoke_pass(self):
        inputs = self.upto_recipe()
        fail = self.proj.write_artifact("smoke", "smoke", {"schema": stages.SCHEMA_SMOKE, "schema_version": 1,
                                                           "verdict": "FAIL", "reasons": ["gpu_memory"]},
                                        stages.SCHEMA_SMOKE)
        with self.assertRaises(Refused) as cm:
            self.train(inputs, fail)
        self.assertRule(cm, "MM-38", "smoke")
        self.assertEqual(self.job_ids(), [])
        self.assertEqual(self.rt.runs, [])

    def test_train_refuses_a_smoke_of_another_recipe(self):
        inputs, smoke_ref = self.ready()
        ts = BuildTrainSet().run(self.ctx("build_train_set", {"train_part": self.train_part(5)}), {})["train_set"]
        recipe = WriteRecipe().run(self.ctx("write_recipe", {"train_set": ts}), {})["recipe"]
        with self.assertRaises(Refused) as cm:
            self.train({**inputs, "train_set": ts, "recipe": recipe}, smoke_ref)
        self.assertRule(cm, "MM-38", "smoke")

    def test_reattach_same_inputs(self):
        inputs, smoke_ref = self.ready()
        _, first = self.train(inputs, smoke_ref)
        _, second = self.train(inputs, smoke_ref)
        a, b = (self.proj.read_artifact(o["model"]) for o in (first, second))
        self.assertEqual(a["job_id"], b["job_id"])
        self.assertEqual(a["files"], b["files"])
        self.assertEqual(len(self.job_ids()), 2)                     # smoke + one full job
        self.assertEqual([("full" in s.argv) for s in self.rt.runs], [False, True])

    def test_base_changed_after_choice_refused(self):
        inputs = self.upto_recipe()
        base = self.proj.read_artifact(inputs["base"])
        path = Path(base["files"][2]["path"])
        path.write_text("other weights\n")
        with self.assertRaises(Refused) as cm:
            self.smoke(inputs)
        self.assertRule(cm, "TXT-2", "model.safetensors")
        self.assertEqual(self.job_ids(), [])


if __name__ == "__main__":
    unittest.main()
