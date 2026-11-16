"""Combining benchmark runs into one result set (scripts/combine_runs.py)."""

import importlib.util
import json
import os
import tempfile
import unittest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "combine_runs.py")
_spec = importlib.util.spec_from_file_location("combine_runs", _SCRIPT)
combine_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine_runs)

PERSONA = [{"question_id": f"P01-Q{i}"} for i in range(1, 4)]
MMLU = [{"question_id": f"mmlu-nutrition-000{i}"} for i in range(1, 3)]


def _record(qid, status="ok", **extra):
    return {"question_id": qid, "status": status, "response": {"needs_clarification": False}, **extra}


class CombineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run(self, name, records, *, retriever=None, commit="new", reused=(), scores=None):
        run_dir = os.path.join(self.tmp.name, name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"run_id": name, "options": {"retriever": retriever},
                       "runtime_info_end": {"git_commit": commit}, "reused": list(reused)}, handle)
        with open(os.path.join(run_dir, "responses.jsonl"), "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in records)
        if scores:
            with open(os.path.join(run_dir, "scores.jsonl"), "w", encoding="utf-8") as handle:
                handle.writelines(json.dumps(s) + "\n" for s in scores)
        return run_dir

    def test_runs_are_placed_merged_and_attributed(self):
        rag = self._run("rag-persona", [
            _record("P01-Q1", reused_from="old-run"),
            _record("P01-Q2"),
            _record("P01-Q3"),
        ], reused=[{"run_id": "old-run", "git_commit": "old"}])
        statements = self._run("st-persona", [_record("P01-Q1"), _record("P01-Q2", status="error")], retriever="statements")
        mmlu = self._run("rag-mmlu", [_record("mmlu-nutrition-0001")],
                         scores=[{"question_id": "mmlu-nutrition-0001", "correct": True}])
        cells, runs, coverage = combine_runs.combine({"persona": PERSONA, "mmlu": MMLU}, [rag, statements, mmlu])

        self.assertEqual({(r["dataset"], r["setting"]) for r in runs},
                         {("persona", "rag"), ("persona", "statements"), ("mmlu", "rag")})
        rag_cell = coverage["persona"]["rag"]
        self.assertEqual((rag_cell["answered"], rag_cell["missing"]), (3, 0))
        self.assertEqual(rag_cell["commits"], {"old": 1, "new": 2})  # a reused answer keeps its commit
        st_cell = coverage["persona"]["statements"]
        self.assertEqual((st_cell["answered"], st_cell["errors"], st_cell["missing_ids"]), (1, 1, ["P01-Q3"]))
        self.assertTrue(cells[("mmlu", "rag")]["mmlu-nutrition-0001"]["score"]["correct"])
        self.assertFalse(combine_runs.complete(coverage, {"persona": PERSONA, "mmlu": MMLU}))

    def test_later_run_wins_on_overlap(self):
        first = self._run("a", [_record("P01-Q1", status="error")])
        second = self._run("b", [_record("P01-Q1")])
        cells, _, coverage = combine_runs.combine({"persona": PERSONA}, [first, second])
        self.assertEqual(cells[("persona", "rag")]["P01-Q1"]["source_run"], "b")
        self.assertEqual(coverage["persona"]["rag"]["errors"], 0)

    def test_a_run_outside_every_dataset_is_refused(self):
        stray = self._run("stray", [_record("unknown-question")])
        with self.assertRaises(ValueError):
            combine_runs.combine({"persona": PERSONA}, [stray])


if __name__ == "__main__":
    unittest.main()
