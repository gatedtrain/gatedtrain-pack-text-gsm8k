import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gatedtrain.ainative.batch import read_predictions

import gatedtrain_pack_text_gsm8k

SCRIPT = Path(gatedtrain_pack_text_gsm8k.__file__).parent / "generate_vllm.py"
FAKE = Path(__file__).parent / "fakevllm"
SAMPLER = {"name": "vllm", "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 1024, "enable_thinking": False}
GREEDY = {"name": "vllm", "decoding": "greedy", "temperature": 0, "top_p": 1.0, "max_new_tokens": 1024,
          "enable_thinking": False}
QUESTIONS = ["How many apples?", "What is 2 + 3?", "Héllo – a café costs 4 euros; how much for 3?"]


def _canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.model = self.tmp / "model"
        self.model.mkdir()
        self.record = self.tmp / "record.jsonl"
        (self.tmp / "items.jsonl").write_text(
            "".join(_canonical({"item_id": i, "input": q}) for i, q in enumerate(QUESTIONS)), encoding="utf-8")

    def run_script(self, sampler=SAMPLER, seeds=(11, 12), extra_env=None, timeout=60):
        (self.tmp / "sampler.json").write_text(json.dumps({"sampler": sampler, "seeds": list(seeds)}),
                                               encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(FAKE), "FAKEVLLM_RECORD": str(self.record),
               "HOME": str(self.tmp), **(extra_env or {})}
        argv = [sys.executable, str(SCRIPT), "--items", str(self.tmp / "items.jsonl"),
                "--sampler", str(self.tmp / "sampler.json"), "--model", str(self.model), "--out", str(self.out)]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd=self.tmp)

    def events(self, kind):
        if not self.record.exists():
            return []
        docs = [json.loads(x) for x in self.record.read_text(encoding="utf-8").split("\n") if x]
        return [d for d in docs if d["event"] == kind]

    def test_predictions_contract(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(os.listdir(self.out), ["predictions.jsonl"])
        raw = (self.out / "predictions.jsonl").read_text(encoding="utf-8")
        lines = raw.split("\n")[:-1]
        self.assertEqual(len(lines), 6)
        docs = [json.loads(x) for x in lines]
        self.assertEqual([(d["seed"], d["item_id"]) for d in docs],
                         [(11, 0), (11, 1), (11, 2), (12, 0), (12, 1), (12, 2)])
        self.assertTrue(all(_canonical(d) == x + "\n" for d, x in zip(docs, lines)))
        prompts = [e["messages"][0]["content"] for e in self.events("chat_template")]
        for d in docs:
            templated = "<|user|>" + prompts[d["item_id"]] + "<|assistant|>"
            self.assertEqual(d["prediction"], f"The answer is {len(templated) % 7}")
        preds = read_predictions(self.out, 3, [11, 12])
        self.assertEqual(len(preds), 6)

    def test_engine_and_sampling_args(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        [llm] = self.events("llm")
        self.assertEqual(llm["model"], str(self.model))
        self.assertEqual(llm["kwargs"], {"dtype": "bfloat16", "generation_config": "vllm", "max_model_len": 2048,
                                         "gpu_memory_utilization": 0.85, "enforce_eager": True, "seed": 0})
        gens = self.events("generate")
        self.assertEqual([g["n"] for g in gens], [3, 3])
        self.assertEqual([g["sampling"] for g in gens],
                         [{"temperature": 0.7, "top_p": 1.0, "max_tokens": 1024, "seed": s} for s in (11, 12)])

    def test_thinking_off_and_prompt(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        calls = self.events("chat_template")
        self.assertEqual(len(calls), 3)
        for call, q in zip(calls, QUESTIONS):
            self.assertEqual(call["kwargs"], {"tokenize": False, "add_generation_prompt": True,
                                              "enable_thinking": False})
            [msg] = call["messages"]
            self.assertEqual(msg["role"], "user")
            self.assertTrue(msg["content"].startswith("Solve the following"))
            self.assertEqual(msg["content"],
                             "Solve the following math problem step by step. Put your final numerical answer after "
                             "'The answer is'.\n\nProblem: " + q)

    def test_env_set_before_import(self):
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        [imp] = self.events("import")
        self.assertEqual(imp["env"], {
            "VLLM_CACHE_ROOT": "/tmp/cache/vllm", "TRITON_CACHE_DIR": "/tmp/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/cache/inductor", "XDG_CACHE_HOME": "/tmp/cache",
            "HF_HOME": "/tmp/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    def test_greedy_ignores_the_seed_and_decodes_one_request_at_a_time(self):
        res = self.run_script(sampler=GREEDY, seeds=(11, 987654))
        self.assertEqual(res.returncode, 0, res.stderr)
        [llm] = self.events("llm")
        self.assertEqual(llm["kwargs"]["enable_prefix_caching"], False)
        self.assertEqual(llm["kwargs"]["max_num_seqs"], 1)
        gens = self.events("generate")
        self.assertEqual(len(gens), 2)
        for gen in gens:
            self.assertEqual(gen["sampling"], {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": None})
        docs = [json.loads(x) for x in (self.out / "predictions.jsonl").read_text(encoding="utf-8").split("\n")[:-1]]
        self.assertEqual([d["seed"] for d in docs], [11] * 3 + [987654] * 3)
        self.assertEqual(len(read_predictions(self.out, 3, [11, 987654])), 6)

    def test_sampled_engine_args_are_unchanged(self):
        self.run_script()
        [llm] = self.events("llm")
        self.assertNotIn("enable_prefix_caching", llm["kwargs"])
        self.assertNotIn("max_num_seqs", llm["kwargs"])

    def test_bad_greedy_sampler_refused(self):
        bad = [dict(GREEDY, temperature=0.3), dict(GREEDY, top_p=0.9), dict(GREEDY, decoding="beam"),
               dict(SAMPLER, decoding="greedy"), dict(GREEDY, temperature=True),
               dict(SAMPLER, temperature=0)]
        for sampler in bad:
            with self.subTest(sampler=sampler):
                self.record.unlink(missing_ok=True)
                res = self.run_script(sampler=sampler)
                self.assertEqual(res.returncode, 2)
                self.assertIn("sampler", res.stderr)
                self.assertEqual(os.listdir(self.out), [])
                self.assertEqual(self.events("import"), [])

    def test_bad_sampler_refused(self):
        bad = [
            dict(SAMPLER, top_k=20),
            dict(SAMPLER, enable_thinking=True),
            {k: v for k, v in SAMPLER.items() if k != "enable_thinking"},
            {k: v for k, v in SAMPLER.items() if k != "temperature"},
            dict(SAMPLER, max_new_tokens="1024"),
            "vllm",
        ]
        for sampler in bad:
            with self.subTest(sampler=sampler):
                self.record.unlink(missing_ok=True)
                res = self.run_script(sampler=sampler)
                self.assertEqual(res.returncode, 2)
                self.assertIn("sampler", res.stderr)
                self.assertEqual(os.listdir(self.out), [])
                self.assertEqual(self.events("import"), [])
        for seeds in ([True], ["11"], [1.5]):
            with self.subTest(seeds=seeds):
                res = self.run_script(seeds=seeds)
                self.assertEqual(res.returncode, 2)
                self.assertIn("seed", res.stderr)
                self.assertEqual(os.listdir(self.out), [])

    def test_exits_without_engine_teardown(self):
        """vLLM's teardown can hang: after the predictions file is written, synced and closed the process ends at
        once (os._exit), so no exit handler runs and the keeper's time limit is not spent waiting."""
        res = self.run_script(extra_env={"FAKEVLLM_EXIT_HANG_S": "600"}, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.events("atexit"), [])
        self.assertEqual(len(read_predictions(self.out, 3, [11, 12])), 6)
        self.assertTrue((self.out / "predictions.jsonl").read_bytes().endswith(b"\n"))

    def test_error_exit_codes_kept(self):
        res = self.run_script(extra_env={"FAKEVLLM_SHORT": "1"})
        self.assertEqual(res.returncode, 1)
        self.assertIn("vLLM returned 2 outputs for 3 prompts", res.stderr)
        self.assertEqual(os.listdir(self.out), [])
        res = self.run_script(sampler=dict(SAMPLER, enable_thinking=True))
        self.assertEqual(res.returncode, 2)

    def test_stdlib_only(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                self.assertEqual(n.level, 0)
                mods.add(n.module.split(".")[0])
        self.assertIn("vllm", mods)
        self.assertLessEqual(mods - {"__future__", "vllm"}, set(sys.stdlib_module_names))


if __name__ == "__main__":
    unittest.main()
