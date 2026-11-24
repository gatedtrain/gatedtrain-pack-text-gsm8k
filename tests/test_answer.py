import unittest

from gatedtrain_pack_text_gsm8k import answer


class AnswerTests(unittest.TestCase):
    def test_extract(self):
        table = [
            ("The answer is 3,082.", "3082"),
            ("\\boxed{1,000}", "1000"),
            ("so 5 apples ... The answer is 12", "12"),
            ("x = 2.50", "2.5"),
            ("no digits", None),
            ("-7 then 9", "9"),
        ]
        for text, want in table:
            with self.subTest(text=text):
                self.assertEqual(answer.extract(text), want)

    def test_extract_precedence(self):
        self.assertEqual(answer.extract("\\boxed{4} The answer is 5"), "4")
        self.assertEqual(answer.extract("the answer is 1 ... THE ANSWER IS 2 and 3"), "2")
        self.assertEqual(answer.extract("The answer is -0.0"), "0")

    def test_normalise(self):
        self.assertEqual(answer.normalise(" -1,200.500 "), "-1200.5")
        self.assertEqual(answer.normalise("7."), "7")
        self.assertIsNone(answer.normalise("1/2"))
        self.assertEqual(answer.normalise("-0"), "0")

    def test_long_digits_are_exact(self):
        self.assertEqual(answer.normalise("9" * 30), "9" * 30)
        self.assertEqual(answer.normalise("1" * 40), "1" * 40)
        self.assertNotEqual(answer.normalise("1" * 40), answer.normalise("1" * 39 + "2"))
        self.assertEqual(answer.normalise("0." + "0" * 40 + "1"), "0." + "0" * 40 + "1")
        self.assertEqual(answer.score("1" * 39 + "2", "1" * 40), 0.0)

    def test_hyphen_is_not_a_minus(self):
        self.assertEqual(answer.extract("10-5"), "5")
        self.assertEqual(answer.extract("on 2023-05"), "5")
        self.assertEqual(answer.extract("temp -5"), "-5")
        self.assertEqual(answer.extract("--5"), "5")

    def test_ascii_digits_only(self):
        self.assertIsNone(answer.normalise("\uff13"))
        self.assertIsNone(answer.extract("\uff13\uff12"))

    def test_leading_dot(self):
        self.assertEqual(answer.normalise(".5"), "0.5")
        self.assertEqual(answer.extract("it is .5"), "0.5")
        self.assertEqual(answer.normalise("-.50"), "-0.5")

    def test_odd_shapes(self):
        self.assertIsNone(answer.normalise("1e5"))
        self.assertEqual(answer.extract("1e5"), "5")
        self.assertIsNone(answer.normalise("--5"))
        self.assertIsNone(answer.normalise("1.2.3"))
        self.assertEqual(answer.extract("1.2.3"), "1.2")
        self.assertEqual(answer.extract("\\boxed{} then 7"), "7")
        self.assertIsNone(answer.extract("\\boxed{}"))

    def test_huge_input_never_raises(self):
        huge = "1" + "0" * 1100000
        self.assertIsNone(answer.normalise(huge))
        self.assertEqual(answer.score(huge, huge), 0.0)
        self.assertEqual(answer.score("9" * 1100000, "5"), 0.0)

    def test_score(self):
        self.assertEqual(answer.score("The answer is 18", "18"), 1.0)
        self.assertEqual(answer.score("18.0", "18"), 1.0)
        self.assertEqual(answer.score("1", "10"), 0.0)
        self.assertEqual(answer.score(None, "3"), 0.0)

    def test_label_ok(self):
        self.assertTrue(answer.label_ok("3,082"))
        self.assertTrue(answer.label_ok("-5"))
        self.assertFalse(answer.label_ok(""))
        self.assertFalse(answer.label_ok("1/2"))


if __name__ == "__main__":
    unittest.main()
