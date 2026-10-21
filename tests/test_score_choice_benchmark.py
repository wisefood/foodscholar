"""MMLU conversion and choice scoring (scripts/mmlu_to_questions.py, score_choice_benchmark.py)."""

import importlib.util
import os
import unittest
from types import SimpleNamespace

import pandas as pd

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mmlu = _load("mmlu_to_questions")
scorer = _load("score_choice_benchmark")

LABELS = ["A", "B", "C", "D"]


class FakeGroq:
    def __init__(self, content):
        self.calls = []
        outer = self

        class Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                message = SimpleNamespace(content=content)
                return SimpleNamespace(
                    model=kwargs["model"],
                    choices=[SimpleNamespace(message=message, finish_reason="stop")],
                )

        self.chat = SimpleNamespace(completions=Completions())


def _question(qid, gold):
    return {"question_id": qid, "question_type": "nutrition", "gold": gold,
            "answer_labels": LABELS, "foodscholar_question": "Q?\n\nA. a\nB. b\nC. c\nD. d"}


def _record(answer=None, *, status="ok", clarification=False):
    response = {"needs_clarification": clarification,
                "primary_answer": {"answer": answer or ""}}
    return {"status": status, "response": response if status == "ok" else None}


class ConvertTests(unittest.TestCase):
    def test_question_carries_lettered_options_and_gold(self):
        df = pd.DataFrame([
            {"question": "Which vitamin?", "subject": "nutrition", "choices": ["A1", "B12", "C", "D"], "answer": 1},
            {"question": "x" * 990, "subject": "nutrition", "choices": ["a", "b", "c", "d"], "answer": 0},
        ])
        questions, too_long = mmlu.convert(df)
        self.assertEqual(too_long, ["mmlu-nutrition-0002"])
        q = questions[0]
        self.assertEqual((q["question_id"], q["gold"], q["persona_id"]), ("mmlu-nutrition-0001", "B", "MMLU"))
        self.assertIn("A. A1\nB. B12\nC. C\nD. D", q["foodscholar_question"])
        self.assertNotIn("B12\n\nB", q["foodscholar_question"])  # gold is never in the text


class LeadingChoiceTests(unittest.TestCase):
    def test_clear_openings_are_read(self):
        for answer, expected in [
            ("C. Fruits and vegetables are eaten less.", "C"),
            ("**C** – fruits and vegetables", "C"),
            ("**Answer: B**\n\nBecause...", "B"),
            ("The correct answer is **D**.", "D"),
            ("(A) Meat", "A"),
            ("\n\nB\n\nexplanation", "B"),
        ]:
            self.assertEqual(scorer.leading_choice(answer, LABELS), expected, answer)

    def test_a_word_is_not_an_option(self):
        for answer in ["A balanced diet keeps...", "Option B is correct because", "## Answer\nC."]:
            self.assertIsNone(scorer.leading_choice(answer, LABELS), answer)

    def test_yes_no_labels(self):
        self.assertEqual(scorer.leading_choice("**Yes** — it fits.", ["yes", "no"]), "yes")
        self.assertIsNone(scorer.leading_choice("Not really clear.", ["yes", "no"]))


class ScoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_outcomes_and_accuracy_count_every_question_that_ran(self):
        questions = [_question(f"q{i}", "C") for i in range(1, 6)]
        records = {
            "q1": _record("C. Right."),
            "q2": _record("A. Wrong."),
            "q3": _record("Fruit and vegetables, so option C."),  # needs the model
            "q4": _record(clarification=True),
            "q5": _record(status="error"),
        }
        groq = FakeGroq('{"choice": "C"}')
        rows = await scorer.score(questions, records, client=groq, model="m")
        by_id = {r["question_id"]: r for r in rows}
        self.assertEqual((by_id["q1"]["method"], by_id["q1"]["correct"]), ("pattern", True))
        self.assertFalse(by_id["q2"]["correct"])
        self.assertEqual((by_id["q3"]["method"], by_id["q3"]["choice"]), ("model", "C"))
        self.assertEqual(by_id["q4"]["outcome"], "clarification")
        self.assertEqual(by_id["q5"]["outcome"], "error")
        self.assertEqual(len(groq.calls), 1)  # only the unclear answer
        enum = groq.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["choice"]["enum"]
        self.assertEqual(enum, LABELS + ["none"])

        summary = scorer.summarize(rows)
        self.assertEqual((summary["ran"], summary["correct"]), (5, 2))
        self.assertEqual(summary["accuracy"], 0.4)  # no answer counts as wrong
        self.assertEqual(summary["accuracy_when_answered"], round(2 / 3, 4))

    async def test_unclear_answers_without_a_key_fail_loudly(self):
        with self.assertRaises(RuntimeError):
            await scorer.score([_question("q1", "A")], {"q1": _record("Hard to say.")}, client=None, model="m")

    async def test_model_saying_none_is_no_choice(self):
        rows = await scorer.score(
            [_question("q1", "A")], {"q1": _record("It depends.")}, client=FakeGroq('{"choice": "none"}'), model="m"
        )
        self.assertEqual((rows[0]["outcome"], rows[0]["correct"]), ("no_choice", False))


class CompareTests(unittest.TestCase):
    def test_paired_counts_and_mcnemar(self):
        def rows(correct):
            return [{"question_id": f"q{i}", "outcome": "answered", "correct": c} for i, c in enumerate(correct)]

        result = scorer.compare(rows([True] * 8 + [False] * 2), rows([True] * 3 + [False] * 7))
        self.assertEqual(
            (result["both_correct"], result["only_this_correct"], result["only_other_correct"], result["neither_correct"]),
            (3, 5, 0, 2),
        )
        self.assertEqual(result["mcnemar_exact_p"], 0.0625)  # 5 of 5 discordant: 2 * 0.5**5


if __name__ == "__main__":
    unittest.main()
