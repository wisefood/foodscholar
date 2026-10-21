"""Score a run_qa_benchmark.py run against a question set with known answers.

For closed-form benchmarks (MMLU multiple choice; yes/no sets work the same
way): each question in the set carries ``gold`` and ``answer_labels``. For
each question the run answered, the choice is read from FoodScholar's answer:
first by pattern, from how the answer opens ("C.", "**C**", "Answer: C"),
and only when that finds nothing, by gpt-oss-20b on Groq, constrained to the
labels or "none" (no single option committed to).

A question counts as wrong unless a choice was read and matches: an API
error, an unanswered clarification and an answer that picks no option are
all reported separately, never dropped, so accuracy stays comparable between
runs that fail differently.

Writes ``scores.jsonl`` (one line per question) and ``scores.json`` (summary)
into the run directory. ``--compare`` pairs this run with another scored run
of the same questions, e.g. FoodScholar's retrieval against ``--retriever
no_rag``, with an exact McNemar test on the questions only one got right.

Usage:
    python scripts/score_choice_benchmark.py benchmark-runs/<run_id> \\
        --questions mmlu-nutrition-test.json --groq-api-key <key>
    python scripts/score_choice_benchmark.py benchmark-runs/<rag_run> \\
        --questions mmlu-nutrition-test.json --compare benchmark-runs/<no_rag_run>
"""
import argparse
import asyncio
import collections
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_qa_benchmark import (  # noqa: E402
    CLARIFIER_MODEL,
    file_digest,
    groq_client,
    groq_structured,
    load_json,
    read_recorded,
)

EXTRACT_SYSTEM = (
    "You read an answer to a multiple-choice question and report which single "
    "option the answer commits to as correct. Report 'none' if it commits to "
    "no option, to more than one, or declines to answer. Judge only what the "
    "answer says, not which option is actually correct."
)


def leading_choice(answer: str, labels: List[str]) -> Optional[str]:
    """The label the answer opens with, or None when the opening is not a clear pick.

    Strict on purpose: "A balanced diet..." opens with a word, not option A,
    so a letter only counts when punctuation or the end of the line follows
    it. Anything less clear goes to the model.
    """
    first = next((line for line in answer.splitlines() if line.strip()), "")
    if all(len(label) == 1 and label.isupper() for label in labels):
        letters = "".join(labels)
        match = re.match(
            r"^[\s>#*_(\[]*"
            r"(?:(?i:(?:the\s+)?(?:correct\s+)?(?:answer|option)(?:\s+is)?)\s*[:\-–]?\s*[*_(\[]*)?"
            rf"([{letters}])(?:[.):\]*_,]|\s*$|\s+[-–—])",
            first,
        )
    else:  # word labels, e.g. yes/no
        words = "|".join(re.escape(label) for label in labels)
        match = re.match(rf"^[\s>#*_(\[]*({words})\b", first, re.IGNORECASE)
        if match:
            return next(label for label in labels if label.lower() == match.group(1).lower())
    return match.group(1) if match else None


async def model_choice(client: Any, model: str, question: str, answer: str, labels: List[str]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": labels + ["none"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    prompt = f"Question:\n{question}\n\nAnswer to read:\n{answer}"
    try:
        data, served_by = await groq_structured(
            client, model=model, system=EXTRACT_SYSTEM, prompt=prompt,
            schema=schema, name="answer_choice",
        )
    except Exception as exc:
        return {"choice": None, "error": f"{exc.__class__.__name__}: {exc}"}
    choice = data.get("choice")
    return {"choice": None if choice == "none" else choice, "model": served_by}


def outcome_of(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a record has no answer to read, or None when it has one."""
    if record is None:
        return "not_run"
    if record.get("status") != "ok":
        return "error"
    response = record.get("response") or {}
    if response.get("needs_clarification"):
        return "clarification"
    if not ((response.get("primary_answer") or {}).get("answer") or "").strip():
        return "empty_answer"
    return None


async def score(questions, records, *, client, model, concurrency=4) -> List[Dict[str, Any]]:
    rows, pending = [], []
    for q in questions:
        record = records.get(q["question_id"])
        row = {"question_id": q["question_id"], "question_type": q.get("question_type"), "gold": q["gold"]}
        missing = outcome_of(record)
        if missing:
            row.update(choice=None, method=None, outcome=missing)
        else:
            answer = record["response"]["primary_answer"]["answer"]
            choice = leading_choice(answer, q["answer_labels"])
            row.update(choice=choice, method="pattern" if choice else "model", outcome=None)
            if not choice:
                pending.append((row, q, answer))
        rows.append(row)

    if pending and client is None:
        raise RuntimeError(
            f"{len(pending)} answer(s) do not open with a clear choice and need "
            "the model to read them: pass --groq-api-key (or set GROQ_API_KEY)"
        )
    semaphore = asyncio.Semaphore(concurrency)

    async def read(row, q, answer):
        async with semaphore:
            result = await model_choice(client, model, q["foodscholar_question"], answer, q["answer_labels"])
        row["choice"] = result["choice"]
        if "error" in result:
            row["outcome"] = "extract_error"
            row["extract_error"] = result["error"]

    await asyncio.gather(*(read(*item) for item in pending))
    for row in rows:
        if row["outcome"] is None:
            row["outcome"] = "no_choice" if row["choice"] is None else "answered"
        row["correct"] = row["choice"] is not None and row["choice"] == row["gold"]
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ran = [r for r in rows if r["outcome"] != "not_run"]
    answered = [r for r in ran if r["outcome"] == "answered"]
    correct = sum(r["correct"] for r in ran)
    by_type = collections.defaultdict(lambda: [0, 0])
    for r in ran:
        by_type[r["question_type"]][0] += r["correct"]
        by_type[r["question_type"]][1] += 1
    return {
        "questions": len(rows),
        "ran": len(ran),
        "correct": correct,
        # Every question that ran is in the denominator: no answer = wrong.
        "accuracy": round(correct / len(ran), 4) if ran else None,
        "accuracy_when_answered": round(correct / len(answered), 4) if answered else None,
        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
        "read_by": dict(collections.Counter(r["method"] for r in ran if r["method"])),
        "by_question_type": {
            t: {"correct": c, "ran": n, "accuracy": round(c / n, 4)} for t, (c, n) in sorted(by_type.items())
        },
    }


def compare(rows: List[Dict[str, Any]], other_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Paired outcome on the questions both runs ran, with an exact McNemar test."""
    other = {r["question_id"]: r for r in other_rows if r["outcome"] != "not_run"}
    pairs = [(r["correct"], other[r["question_id"]]["correct"])
             for r in rows if r["outcome"] != "not_run" and r["question_id"] in other]
    only_this = sum(a and not b for a, b in pairs)
    only_other = sum(b and not a for a, b in pairs)
    result = {
        "paired_questions": len(pairs),
        "both_correct": sum(a and b for a, b in pairs),
        "only_this_correct": only_this,
        "only_other_correct": only_other,
        "neither_correct": sum(not a and not b for a, b in pairs),
        "mcnemar_exact_p": None,
    }
    if only_this + only_other:
        from scipy.stats import binomtest

        result["mcnemar_exact_p"] = round(binomtest(only_this, only_this + only_other, 0.5).pvalue, 4)
    return result


def read_scores(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "scores.jsonl")
    if not os.path.exists(path):
        raise RuntimeError(f"{run_dir} has not been scored yet (no scores.jsonl)")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def main_async(args: argparse.Namespace) -> int:
    questions = [q for q in load_json(args.questions) if "gold" in q]
    if not questions:
        raise RuntimeError(f"{args.questions} has no questions with a gold answer")
    records = read_recorded(os.path.join(args.run_dir, "responses.jsonl"))
    if not records:
        raise RuntimeError(f"{args.run_dir} has no recorded responses")
    client = None
    if args.groq_api_key or os.environ.get("GROQ_API_KEY"):
        client = groq_client(args.groq_api_key, "reading answers with the model")

    rows = await score(questions, records, client=client, model=args.extract_model)
    summary = {
        "run_dir": os.path.abspath(args.run_dir),
        "questions": {"path": os.path.abspath(args.questions), "sha256": file_digest(args.questions)},
        "extract_model": args.extract_model,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        **summarize(rows),
    }
    if args.compare:
        summary["compare"] = {"run_dir": os.path.abspath(args.compare), **compare(rows, read_scores(args.compare))}

    with open(os.path.join(args.run_dir, "scores.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.run_dir, "scores.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("ran", "correct", "accuracy", "accuracy_when_answered", "outcomes", "read_by")}))
    if args.compare:
        print(json.dumps(summary["compare"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", help="A run directory from run_qa_benchmark.py")
    parser.add_argument("--questions", required=True, help="The question set with gold + answer_labels")
    parser.add_argument("--groq-api-key", help="Groq key for answers the pattern cannot read; defaults to GROQ_API_KEY")
    parser.add_argument("--extract-model", default=CLARIFIER_MODEL, help="Groq model that reads unclear answers")
    parser.add_argument("--compare", help="Another scored run of the same questions to pair with")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
