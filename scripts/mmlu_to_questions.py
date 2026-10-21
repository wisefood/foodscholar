"""Turn MMLU multiple-choice questions into a question set for run_qa_benchmark.py.

Each question becomes one /qa/ask question: the stem, the four lettered
options, and an instruction to lead with the letter. The correct letter is
kept in the file as ``gold`` (with ``answer_labels``) for
score_choice_benchmark.py; the runner never sends it. A question that would
exceed the API's 1000-character limit is left out and reported, not cut.

A single neutral persona asks every question: no country, region or age, so
nothing personal steers retrieval, and a profile that lets the runner's
persona clarifier answer "in general" if FoodScholar asks for detail.

Usage:
    python scripts/mmlu_to_questions.py --parquet .../mmlu/all/test-00000-of-00001.parquet \\
        --subject nutrition --out-questions mmlu-nutrition.json --out-personas mmlu-persona.json
"""
import argparse
import json
import random
import sys

import pandas as pd

LABELS = ["A", "B", "C", "D"]
MAX_QUESTION_CHARS = 1000  # QARequest.question max_length
INSTRUCTION = (
    "This is a multiple-choice question. Start your answer with the letter of "
    "the single correct option, then explain briefly."
)

PERSONA = {
    "persona_id": "MMLU",
    "name": "Exam candidate",
    "occupation": "Student answering a general nutrition exam",
    "household": "Not relevant to the question",
    "short_persona_description": (
        "Answers a general knowledge exam question; has no particular country, "
        "age group or personal health situation in mind."
    ),
    "api_mapping": {"expertise_level": "intermediate", "language": "en"},
}


def question_text(stem: str, choices) -> str:
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices))
    return f"{stem.strip()}\n\n{options}\n\n{INSTRUCTION}"


def convert(df: pd.DataFrame):
    questions, too_long = [], []
    for subject, rows in df.groupby("subject", sort=True):
        for n, row in enumerate(rows.itertuples(index=False), start=1):
            qid = f"mmlu-{subject}-{n:04d}"
            text = question_text(row.question, row.choices)
            if len(text) > MAX_QUESTION_CHARS:
                too_long.append(qid)
                continue
            questions.append({
                "question_id": qid,
                "persona_id": PERSONA["persona_id"],
                "question_type": subject,
                "language": "en",
                "foodscholar_question": text,
                "answer_labels": LABELS[: len(row.choices)],
                "gold": LABELS[int(row.answer)],
            })
    return questions, too_long


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", required=True, help="An MMLU split (cais/mmlu parquet)")
    parser.add_argument("--subject", action="append", help="MMLU subject (repeatable); default nutrition")
    parser.add_argument("--out-questions", required=True)
    parser.add_argument("--out-personas", required=True)
    parser.add_argument("--sample", type=int, help="Keep a random N of the questions (seeded, reproducible)")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for --sample")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    subjects = args.subject or ["nutrition"]
    unknown = sorted(set(subjects) - set(df["subject"]))
    if unknown:
        print(f"error: unknown subject(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    questions, too_long = convert(df[df["subject"].isin(subjects)])
    if args.sample is not None and args.sample < len(questions):
        keep = set(random.Random(args.seed).sample(range(len(questions)), args.sample))
        questions = [q for i, q in enumerate(questions) if i in keep]  # file order kept

    with open(args.out_questions, "w", encoding="utf-8") as handle:
        json.dump(questions, handle, ensure_ascii=False, indent=1)
    with open(args.out_personas, "w", encoding="utf-8") as handle:
        json.dump([PERSONA], handle, ensure_ascii=False, indent=1)
    print(f"{len(questions)} question(s) -> {args.out_questions}", file=sys.stderr)
    if too_long:
        print(f"left out {len(too_long)} over {MAX_QUESTION_CHARS} chars: {', '.join(too_long)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
