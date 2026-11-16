"""Combine benchmark runs into one result set: every dataset x setting x question.

Each run directory (from run_qa_benchmark.py) is placed by its manifest: the
setting is its retriever (unset = rag), and the dataset is the given question
set that contains every question it recorded. Where two runs cover the same
dataset, setting and question, the later run on the command line wins.

Writes into --out:

* ``responses.jsonl`` -- one line per dataset x setting x question: the
  run's record, plus ``dataset``, ``setting``, ``source_run`` and
  ``git_commit`` (the commit that answer actually ran on: a reused answer
  keeps its original run's), and ``score`` when the run was scored
  (score_choice_benchmark.py).
* ``manifest.json`` -- the datasets (path, sha256, size), the runs, and per
  dataset x setting: answered, missing (with ids), errors, unanswered
  clarifications, degraded retrieval, follow-ups without context, and the
  commits the answers came from.

Exits 1 when any dataset x setting is incomplete, so a script can tell.

Usage:
    python scripts/combine_runs.py --out benchmark-runs/combined \\
        --dataset persona=20260904-questions-v2.json --dataset mmlu=mmlu-nutrition-test.json \\
        benchmark-runs/<run> [benchmark-runs/<run> ...]
"""
import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_qa_benchmark import (  # noqa: E402
    file_digest,
    lacks_thread_context,
    load_json,
    read_recorded,
    search_failures,
    setting_of,
    write_manifest,
)


def answer_commit(record: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    """The commit a record was answered on: its original run's, if it was reused."""
    if record.get("reused_from"):
        for source in manifest.get("reused") or []:
            if source.get("run_id") == record["reused_from"]:
                return source.get("git_commit") or "unknown"
        return "unknown"
    runtime = manifest.get("runtime_info_end") or manifest.get("runtime_info_start") or {}
    return runtime.get("git_commit") or "unknown"


def read_scores(run_dir: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(run_dir, "scores.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return {row["question_id"]: row for row in map(json.loads, filter(str.strip, handle))}


def combine(datasets: Dict[str, List[Dict[str, Any]]], run_dirs: List[str]):
    ids = {name: {q["question_id"] for q in questions} for name, questions in datasets.items()}
    cells: Dict[tuple, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    runs = []
    for run_dir in run_dirs:
        manifest = load_json(os.path.join(run_dir, "manifest.json"))
        records = read_recorded(os.path.join(run_dir, "responses.jsonl"))
        dataset = next((name for name, known in ids.items() if records and set(records) <= known), None)
        if dataset is None:
            raise ValueError(f"{run_dir}: its questions are not all in any one of the given datasets")
        setting = setting_of(manifest.get("options") or {})["retriever"]
        scores = read_scores(run_dir)
        for qid, record in records.items():
            row = {
                "dataset": dataset,
                "setting": setting,
                "source_run": manifest.get("run_id"),
                "git_commit": answer_commit(record, manifest),
                **record,
            }
            if qid in scores:
                row["score"] = scores[qid]
            cells[(dataset, setting)][qid] = row
        runs.append({"run_dir": os.path.abspath(run_dir), "run_id": manifest.get("run_id"),
                     "dataset": dataset, "setting": setting, "records": len(records),
                     "scored": bool(scores)})

    coverage: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    for (dataset, setting), rows in sorted(cells.items()):
        missing = sorted(ids[dataset] - set(rows))
        values = rows.values()
        coverage[dataset][setting] = {
            "answered": sum(r.get("status") == "ok" for r in values),
            "missing": len(missing),
            "missing_ids": missing[:50],
            "errors": sum(r.get("status") == "error" for r in values),
            "skipped": sum(r.get("status") == "skipped" for r in values),
            "unanswered_clarifications": sum(bool((r.get("response") or {}).get("needs_clarification")) for r in values),
            "degraded_retrieval": sum(r.get("status") == "ok" and search_failures(r) > 0 for r in values),
            "follow_up_without_context": sum(lacks_thread_context(r) for r in values),
            "commits": dict(collections.Counter(r["git_commit"] for r in values)),
        }
    return cells, runs, coverage


def complete(coverage: Dict[str, Dict[str, Any]], datasets) -> bool:
    return all(
        cell["missing"] == 0 and cell["answered"] == len(datasets[dataset])
        for dataset, settings in coverage.items() for cell in settings.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="Run directories, later ones winning on overlap")
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=QUESTIONS_JSON",
                        help="A full question set the runs belong to (repeatable)")
    parser.add_argument("--out", required=True, help="Directory for the combined result set")
    args = parser.parse_args()

    dataset_files = dict(spec.split("=", 1) for spec in args.dataset)
    datasets = {name: load_json(path) for name, path in dataset_files.items()}
    try:
        cells, runs, coverage = combine(datasets, args.run_dirs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "responses.jsonl"), "w", encoding="utf-8") as handle:
        for key in sorted(cells):
            for qid in sorted(cells[key]):
                handle.write(json.dumps(cells[key][qid], ensure_ascii=False) + "\n")
    write_manifest(os.path.join(args.out, "manifest.json"), {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {name: {"path": os.path.abspath(path), "sha256": file_digest(path), "questions": len(datasets[name])}
                     for name, path in dataset_files.items()},
        "runs": runs,
        "coverage": coverage,
    })

    for dataset, settings in coverage.items():
        for setting, cell in settings.items():
            print(f"{dataset:10s} {setting:11s} answered {cell['answered']:4d}/{len(datasets[dataset])}"
                  f"  missing {cell['missing']:3d}  errors {cell['errors']}  degraded {cell['degraded_retrieval']}"
                  f"  no-context follow-ups {cell['follow_up_without_context']}  commits {cell['commits']}")
    print(args.out)
    return 0 if complete(coverage, datasets) else 1


if __name__ == "__main__":
    sys.exit(main())
