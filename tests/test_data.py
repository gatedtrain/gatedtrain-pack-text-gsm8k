import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from gatedtrain.errors import Refused

from gatedtrain_pack_text_gsm8k.data import convert, main

FIXTURE = Path(__file__).parent / "fixtures" / "gsm8k-mini.jsonl"


def _line(q, a):
    return json.dumps({"question": q, "answer": a}) + "\n"


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def src(self, text):
        p = self.tmp / "src.jsonl"
        p.write_text(text, encoding="utf-8")
        return p

    def test_convert_fixture(self):
        dst = self.tmp / "out.jsonl"
        res = convert(FIXTURE, dst)
        self.assertEqual(res["rows"], 11)
        self.assertEqual(res["duplicates"], 1)
        self.assertEqual(res["refused"], [])
        raw = dst.read_bytes().decode("utf-8")
        out = [json.loads(x) for x in raw.split("\n") if x]
        self.assertEqual(len(out), 11)
        self.assertTrue(all(list(r) == ["input", "label"] for r in out))
        self.assertIn("3082", [r["label"] for r in out])
        self.assertIn("3000", [r["label"] for r in out])
        src_qs = [json.loads(x)["question"] for x in FIXTURE.read_text(encoding="utf-8").split("\n") if x]
        sep = [q for q in src_qs if " " in q]
        self.assertEqual(len(sep), 1)
        self.assertIn(sep[0], [r["input"] for r in out])
        self.assertEqual(stat.S_IMODE(dst.stat().st_mode), 0o644)

    def test_conflicting_labels_refused(self):
        s = self.src(_line("q1", "x\n#### 5") + _line("q1", "x\n#### 6"))
        dst = self.tmp / "out.jsonl"
        res = convert(s, dst)
        self.assertEqual([r["line"] for r in res["refused"]], [2])
        self.assertFalse(dst.exists())
        self.assertNotIn("q1", json.dumps(res))

    def test_bad_answer_refused_by_line(self):
        s = self.src(_line("a", "x\n#### 1") + _line("b", "x\n#### 2") + _line("c", "x\n#### 3")
                     + _line("d", "no marker here 4") + _line("e", "x\n#### 5"))
        dst = self.tmp / "out.jsonl"
        res = convert(s, dst)
        self.assertEqual(res["refused"][0]["line"], 4)
        self.assertEqual(len(res["refused"]), 1)
        self.assertFalse(dst.exists())

    def test_bad_json_and_blank_and_types_refused(self):
        s = self.src(_line("a", "x\n#### 1") + "not json\n" + "\n" + json.dumps({"question": 1, "answer": "#### 2"}) + "\n"
                     + _line("z", "x\n#### abc"))
        res = convert(s, self.tmp / "o.jsonl")
        self.assertEqual([r["line"] for r in res["refused"]], [2, 3, 4, 5])

    def test_existing_dst_refused(self):
        dst = self.tmp / "out.jsonl"
        dst.write_text("keep")
        with self.assertRaises(Refused):
            convert(FIXTURE, dst)
        self.assertEqual(dst.read_text(), "keep")

    def test_empty_source_refused(self):
        s = self.src("")
        dst = self.tmp / "out.jsonl"
        with self.assertRaises(Refused) as cm:
            convert(s, dst)
        self.assertEqual([p.rule for p in cm.exception.problems], ["TXT-1"])
        self.assertIn("no row", str(cm.exception))
        self.assertFalse(dst.exists())
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            self.assertEqual(main(["--src", str(s), "--dst", str(dst)]), 3)
        self.assertIn("no row", err.getvalue())
        self.assertFalse(dst.exists())

    def test_main_summary_and_exit_codes(self):
        dst = self.tmp / "out.jsonl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--src", str(FIXTURE), "--dst", str(dst)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buf.getvalue())["rows"], 11)
        bad = self.src("garbage\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--src", str(bad), "--dst", str(self.tmp / "o2.jsonl")])
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(buf.getvalue())["refused"][0]["line"], 1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--src", str(FIXTURE), "--dst", str(dst)]), 3)


if __name__ == "__main__":
    unittest.main()
