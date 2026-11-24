import shutil
import tempfile
import unittest
from pathlib import Path

from gatedtrain.ainative.project import create_project
from gatedtrain.ainative.stage import StageContext
from gatedtrain.ainative.trust_stages import SCHEMA_TRAIN_PART
from gatedtrain.errors import Refused

from gatedtrain_pack_text_gsm8k.stages import SCHEMA_PROFILE, ProfileData


class ProfileDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        data = self.tmp / "data.jsonl"
        data.write_text("{}\n")
        self.proj = create_project(self.tmp / "proj", goal="g", data=data, pipeline="p.yaml",
                                   budget={"gpu_hours": 0, "llm_usd": 0})

    def ctx(self, rows):
        rec = {"schema": SCHEMA_TRAIN_PART, "schema_version": 1, "n": len(rows), "rows": rows}
        ref = self.proj.write_artifact("intake_seal", "train_part", rec, SCHEMA_TRAIN_PART)
        return StageContext(self.proj, {"train_part": ref}, {}, None, None, stage="profile_data")

    def test_profile_data_counts(self):
        rows = [{"input": "x" * n, "label": lab} for n, lab in
                [(10, "3"), (20, "-4"), (30, "4.5"), (40, "100"), (50, "7")]]
        stage, ctx = ProfileData(), self.ctx(rows)
        self.assertEqual(stage.name, "gsm8k_profile_data")
        prop = stage.plan(ctx)
        self.assertEqual(prop.estimate, {"n": 5})
        out = stage.run(ctx, {})
        self.assertEqual(out["profile"].schema, SCHEMA_PROFILE)
        p = self.proj.read_artifact(out["profile"])
        self.assertEqual(p["n"], 5)
        self.assertEqual(p["label_min"], "-4")
        self.assertEqual(p["label_max"], "100")
        self.assertEqual(p["n_non_integer_labels"], 1)
        self.assertEqual(p["question_chars"]["p50"], 30)
        self.assertEqual(p["question_chars"]["max"], 50)
        self.assertEqual(stage.check(ctx, out), [])

    def test_small_and_large_labels_keep_their_value(self):
        """str(Decimal) gives an exponent ("1E-7"), which normalise refuses; the range is written in plain notation."""
        for labels, lo, hi in ((["0.0000001", "0.5"], "0.0000001", "0.5"),
                               (["-0.00000005", "3"], "-0.00000005", "3"),
                               (["1", "1000000000000000000000000000000"], "1", "1000000000000000000000000000000")):
            with self.subTest(labels=labels):
                ctx = self.ctx([{"input": f"q{i}", "label": lab} for i, lab in enumerate(labels)])
                p = self.proj.read_artifact(ProfileData().run(ctx, {})["profile"])
                self.assertEqual((p["label_min"], p["label_max"]), (lo, hi))
                self.assertIn(f"labels {lo} to {hi}", ProfileData().plan(ctx).summary)

    def test_bad_label_refused_without_echo(self):
        ctx = self.ctx([{"input": "SECRETQ", "label": "3"}, {"input": "y", "label": "SECRETL"}])
        with self.assertRaises(Refused) as cm:
            ProfileData().run(ctx, {})
        text = str(cm.exception) + repr(cm.exception.problems)
        self.assertEqual(cm.exception.problems[0].rule, "TXT-1")
        self.assertNotIn("SECRETL", text)
        self.assertNotIn("SECRETQ", text)
        with self.assertRaises(Refused):
            ProfileData().plan(ctx)

    def test_empty_train_part_refused(self):
        with self.assertRaises(Refused):
            ProfileData().plan(self.ctx([]))


if __name__ == "__main__":
    unittest.main()
