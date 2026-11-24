"""The pack's scenario (`scenario.SCENARIO`), its pipeline file and its entry points, as gatedtrain loads them."""

import hashlib
import importlib
import os
import unittest
from importlib import metadata
from pathlib import Path
from unittest import mock

from gatedtrain.ainative import scorers, stage as stage_mod, trust_stages
from gatedtrain.ainative.batch import BatchGenerator
from gatedtrain.ainative.pipeline import load_pipeline, pipeline_problems
from gatedtrain.errors import Refused

import gatedtrain_pack_text_gsm8k
from gatedtrain_pack_text_gsm8k import adapter, answer, scenario, stages

PACKAGE = Path(gatedtrain_pack_text_gsm8k.__file__).resolve().parent
STAGE_CLASSES = (stages.ProfileData, stages.ChooseTaskAndBase, stages.BuildTrainSet, stages.WriteRecipe,
                 stages.Smoke, stages.Train)
STEPS = {"profile_data": "gsm8k_profile_data", "choose_task_and_base": "gsm8k_choose_task_and_base",
         "build_train_set": "gsm8k_build_train_set", "write_recipe": "gsm8k_write_recipe", "smoke": "gsm8k_smoke",
         "train": "gsm8k_train"}


class _EP:
    def __init__(self, name, obj, value):
        self.name, self.value, self._obj, self.dist = name, value, obj, None

    def load(self):
        return self._obj


def scenario_eps(obj=None):
    return [_EP("text-gsm8k", scenario.SCENARIO if obj is None else obj,
                "gatedtrain_pack_text_gsm8k.scenario:SCENARIO")]


def stage_eps():
    return [_EP(c.name, c, f"gatedtrain_pack_text_gsm8k.stages:{c.__name__}") for c in STAGE_CLASSES]


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        trust_stages._reset_scenario_snapshot()
        self.addCleanup(trust_stages._reset_scenario_snapshot)
        p = mock.patch.object(trust_stages, "_scenario_entry_points", scenario_eps)
        p.start()
        self.addCleanup(p.stop)

    def test_scenario_loads(self):
        self.assertEqual(scenario.SCENARIO.name, "text-gsm8k")
        self.assertEqual(dict(scenario.SCENARIO.generators), {})
        self.assertEqual(dict(trust_stages.scenario_generators()), {})
        self.assertEqual(dict(trust_stages.scenario_batch_generators()), {"text_gsm8k_vllm": scenario.BATCH})
        self.assertEqual(trust_stages.scenario_pack_of("text_gsm8k_vllm"), "text-gsm8k")
        spec = scorers.resolve("gsm8k_answer")
        self.assertEqual((spec.name, spec.evaluator_id, spec.direction, spec.planning_sd, spec.score_range),
                         ("gsm8k_answer", "gsm8k_answer@0.1", "higher_is_better", 0.5, (0.0, 1.0)))
        self.assertLessEqual(spec.planning_sd, scorers.PLANNING_SD_BOUND)
        self.assertIs(spec.fn, answer.score)
        self.assertIs(spec.label_ok, answer.label_ok)
        decl = scenario.SCENARIO.scorers["gsm8k_answer"]
        self.assertEqual(decl["code_sha256"],
                         "sha256:" + hashlib.sha256((PACKAGE / "answer.py").read_bytes()).hexdigest())
        self.assertEqual((decl["what"], decl["unit"], list(decl["range"])),
                         ("final numeric answer equals the label", "accuracy", [0, 1]))
        self.assertEqual(adapter.EVALUATOR_ROWS[adapter.EVALUATOR_ID]["scorer"]["code_sha256"], decl["code_sha256"])

    def test_changed_scorer_code_refused(self):
        changed = scenario.Scenario(scenario.NAME, {}, scenario.SCENARIO.pipeline,
                                    {"gsm8k_answer": {**scenario.SCORERS["gsm8k_answer"],
                                                      "code_sha256": "sha256:" + "0" * 64}},
                                    {"text_gsm8k_vllm": scenario.BATCH})
        with mock.patch.object(trust_stages, "_scenario_entry_points", lambda: scenario_eps(changed)):
            trust_stages._reset_scenario_snapshot()
            with self.assertRaises(Refused) as cm:
                scorers.resolve("gsm8k_answer")
        self.assertEqual({p.rule for p in cm.exception.problems}, {"AN-17"})

    def test_batch_generator(self):
        gen = scenario.BATCH
        self.assertIsInstance(gen, BatchGenerator)
        self.assertEqual(gen.script, PACKAGE / "generate_vllm.py")
        self.assertTrue(gen.script.is_file())
        self.assertEqual((gen.gpu, gen.time_limit_s, gen.eval_trees), (True, 1200, True))

    def test_eval_python_default_and_override(self):
        self.assertEqual(scenario.eval_python({}), "/opt/gatedtrain-keeper/eval-venv/bin/python")
        self.assertEqual(scenario.eval_python({scenario.EVAL_PYTHON_ENV: ""}),
                         "/opt/gatedtrain-keeper/eval-venv/bin/python")
        self.assertEqual(scenario.eval_python({scenario.EVAL_PYTHON_ENV: "/srv/eval/venv/bin/python"}),
                         "/srv/eval/venv/bin/python")
        with self.assertRaises(ValueError):
            scenario.eval_python({scenario.EVAL_PYTHON_ENV: "venv/bin/python"})

    def test_eval_python_read_at_import(self):
        env = {k: v for k, v in os.environ.items() if k != scenario.EVAL_PYTHON_ENV}

        def reload(environ):
            with mock.patch.dict(os.environ, environ, clear=True):
                importlib.reload(scenario)
        self.addCleanup(reload, dict(os.environ))
        reload({**env, scenario.EVAL_PYTHON_ENV: "/srv/eval/venv/bin/python"})
        self.assertEqual(scenario.BATCH.interpreter, "/srv/eval/venv/bin/python")
        self.assertEqual(scenario.SCENARIO.batch_generators["text_gsm8k_vllm"].interpreter,
                         "/srv/eval/venv/bin/python")
        with self.assertRaises(ValueError):
            reload({**env, scenario.EVAL_PYTHON_ENV: "relative/python"})
        reload(env)
        self.assertEqual(scenario.BATCH.interpreter, "/opt/gatedtrain-keeper/eval-venv/bin/python")

    def test_pipeline(self):
        path = scenario.SCENARIO.pipeline
        self.assertEqual(path, PACKAGE / "pipeline.yaml")
        spec = load_pipeline(path)
        with mock.patch.object(stage_mod, "_plugin_entry_points", stage_eps):
            registry = stage_mod.stage_registry()
        self.assertEqual(pipeline_problems(spec, registry), [])
        refs = {r.name: r for r in spec.stages}
        self.assertEqual([r.name for r in spec.stages],
                         ["intake_seal", "profile_data", "choose_task_and_base", "preregister", "build_train_set",
                          "write_recipe", "smoke", "train", "evaluate_sealed", "gate", "report"])
        self.assertEqual({k: refs[k].stage for k in STEPS}, STEPS)
        for name in ("smoke", "train"):
            self.assertEqual(refs[name].inputs["base"], "choose_task_and_base.base")
            self.assertEqual(refs[name].params, {"image": None, "runtime_dir": None, "python_dir": None})
        self.assertEqual(refs["train"].inputs["smoke"], "smoke.smoke")
        self.assertEqual(refs["choose_task_and_base"].params, {"base_model_dir": None})
        self.assertEqual(refs["evaluate_sealed"].inputs["baseline_model"], "choose_task_and_base.base")
        self.assertEqual(refs["evaluate_sealed"].inputs["candidate_model"], "train.model")
        text = path.read_text()
        for key in ("image", "runtime_dir", "python_dir", "base_model_dir", "--pipeline"):
            self.assertIn(key, text)


class EntryPointTests(unittest.TestCase):
    """The installed distribution's entry points (after `pip install -e .`)."""

    def eps(self, group):
        return {ep.name: ep for ep in metadata.entry_points(group=group)
                if ep.dist is not None and ep.dist.metadata["Name"] == "gatedtrain-pack-text-gsm8k"}

    def test_entry_points(self):
        self.assertEqual({n: ep.value for n, ep in self.eps("gatedtrain.scenarios").items()},
                         {"text-gsm8k": "gatedtrain_pack_text_gsm8k.scenario:SCENARIO"})
        self.assertIs(self.eps("gatedtrain.scenarios")["text-gsm8k"].load(), scenario.SCENARIO)
        st = {n: ep.load() for n, ep in self.eps("gatedtrain.stages").items()}
        self.assertEqual(st, {c.name: c for c in STAGE_CLASSES})
        self.assertEqual(set(st), set(STEPS.values()))
        ad = self.eps("gatedtrain.adapters")
        self.assertEqual(set(ad), {"text-gsm8k-grpo"})
        self.assertIs(ad["text-gsm8k-grpo"].load(), adapter.PLUGIN)


if __name__ == "__main__":
    unittest.main()
