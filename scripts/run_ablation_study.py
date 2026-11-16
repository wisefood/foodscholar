"""Run an ablation study of the QA pipeline and report what each part is worth.

A study file (``experiments/ablation/study.json``) names the configurations:

- the **full system** (the reference every other configuration is paired with),
  run more than once so the report knows how much two runs of the *same*
  configuration disagree (the noise floor);
- **baselines**: simpler systems the full one has to beat (no sources at all;
  the single-pass flow; an earlier recorded production run);
- **ablations**: the full system with one part switched off;
- **variants**: one setting changed the other way;
- **sweeps**: one setting over several values (agent rounds 0-3, sub-questions,
  candidates per search, per-guide cap, evidence budget), to see where more
  stops paying for itself; the value the full system uses is the full system.

Each configuration has a ``stage`` (default 1): run stage 1 (baselines and the
main ablations) first and stage 2 (sweeps, single ranking signals, models)
where stage 1 shows something worth decomposing.

Every configuration is the same question set through ``run_qa_benchmark.py``
against one API started with ``QA_ABLATION_ENABLED=true``; a configuration is
a set of per-request ``ablation`` settings, so all of them come from the same
code, index and models. Runs resume: running the study again asks only what
is missing.

    python scripts/run_ablation_study.py run    experiments/ablation/study.json --api http://127.0.0.1:18011
    python scripts/run_ablation_study.py score  experiments/ablation/study.json
    python scripts/run_ablation_study.py report experiments/ablation/study.json

``report`` writes ``<out>/<study>/report.json`` and ``report.md``. Per
dataset and configuration:

- accuracy (closed-form datasets, from ``score_choice_benchmark.py``) and the
  mean judge score when a run has ``judgments.jsonl`` (one
  ``{"question_id", "score"}`` per line, from any judge);
- the paired difference from the full system on the questions both answered,
  with a bootstrap 95% interval and an exact McNemar test (sign test for
  scores), Holm-corrected across the configurations of a dataset;
- what each answer cost: latency, searches, evidence checks, sources, and
  how often the answer fell back to general knowledge or asked a question.

Imports nothing from ``src/``, like the runner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_qa_benchmark.py")
SCORER = os.path.join(HERE, "score_choice_benchmark.py")

ROLES = ("system", "baseline", "ablation", "variant", "sweep")
GENERAL_KNOWLEDGE = re.compile(r"\[general knowledge\]", re.IGNORECASE)
BOOTSTRAP_SAMPLES = 2000
ALPHA = 0.05


# --- The study file ---------------------------------------------------------


def load_study(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        study = json.load(handle)
    base = os.path.dirname(os.path.abspath(path))
    study.setdefault("out", "benchmark-runs")
    study.setdefault("repeats", {})
    study["_sweeps"] = []
    for sweep in study.get("sweeps") or []:
        labels = []
        for value in sweep["values"]:
            if value == sweep.get("default"):
                labels.append((value, study["reference"]))
                continue
            overlay = {**(sweep.get("ablation") or {}), sweep["setting"]: value}
            same = next((c for c in study["configs"]
                         if c.get("ablation") == overlay and not c.get("retriever")), None)
            if same:  # an ablation already answers this point
                labels.append((value, same["name"]))
                continue
            name = f"{sweep['name']}={value}"
            labels.append((value, name))
            study["configs"].append({
                "name": name, "role": "sweep", "stage": sweep.get("stage", 2),
                "ablation": {**(sweep.get("ablation") or {}), sweep["setting"]: value},
                "datasets": sweep.get("datasets"),
                "what": f"{sweep.get('what') or sweep['setting']}: {value}",
            })
        study["_sweeps"].append({**sweep, "labels": labels})
    names = [c["name"] for c in study["configs"]]
    if len(set(names)) != len(names):
        raise ValueError("configuration names must be unique")
    if study.get("reference") not in names:
        raise ValueError("'reference' must name one of the configurations")
    for config in study["configs"]:
        role = config.setdefault("role", "ablation")
        config.setdefault("stage", 1)
        if role not in ROLES:
            raise ValueError(f"{config['name']}: role must be one of {ROLES}")
        unknown = set(config.get("datasets") or []) - set(study["datasets"])
        if unknown:
            raise ValueError(f"{config['name']}: unknown dataset(s) {sorted(unknown)}")
    for name, dataset in study["datasets"].items():
        dataset.setdefault("scorer", "none")
        for key in ("personas", "questions"):
            if not os.path.isabs(dataset[key]) and not os.path.exists(dataset[key]):
                candidate = os.path.join(base, dataset[key])
                if os.path.exists(candidate):
                    dataset[key] = candidate
    study["_path"] = os.path.abspath(path)
    return study


def runs(study: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every (dataset, configuration, repeat) the study answers or reuses."""
    out = []
    for dataset in study["datasets"]:
        for config in study["configs"]:
            if config.get("datasets") and dataset not in config["datasets"]:
                continue
            existing = (config.get("existing_runs") or {}).get(dataset)
            if config.get("existing_runs") and not existing:
                continue  # a recorded run for other datasets only
            for repeat in range(1, int(study["repeats"].get(config["name"], 1)) + 1):
                label = config["name"] if repeat == 1 else f"{config['name']}#{repeat}"
                run_id = existing or f"{study['name']}-{dataset}-{config['name']}" + (
                    "" if repeat == 1 else f"-r{repeat}"
                )
                out.append({
                    "dataset": dataset,
                    "config": config,
                    "label": label,
                    "repeat": repeat,
                    "run_id": run_id,
                    "run_dir": existing if existing else os.path.join(study["out"], run_id),
                    "recorded": bool(existing),
                })
    return out


def runner_command(study: Dict[str, Any], run: Dict[str, Any], api: str, extra: Sequence[str]) -> List[str]:
    dataset = study["datasets"][run["dataset"]]
    config = run["config"]
    cmd = [
        sys.executable, RUNNER, "--api", api,
        "--personas", dataset["personas"], "--questions", dataset["questions"],
        "--out", study["out"], "--run-id", run["run_id"],
        "--config-name", run["label"],
    ]
    if config.get("retriever"):
        cmd += ["--retriever", config["retriever"]]
    if config.get("ablation"):
        cmd += ["--ablation-json", json.dumps(config["ablation"], sort_keys=True)]
    for option in ("limit",):
        if dataset.get(option) is not None:
            cmd += [f"--{option}", str(dataset[option])]
    for value in dataset.get("only") or []:
        cmd += ["--only", value]
    return cmd + list(extra)


# --- Reading runs -----------------------------------------------------------


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def final_records(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """The last record per question (the runner's file is append-only)."""
    return {r["question_id"]: r for r in read_jsonl(os.path.join(run_dir, "responses.jsonl"))}


def answer_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """What one answer cost and how it was built, from the recorded response."""
    response = record.get("response") or {}
    steps = response.get("reasoning_steps") or []
    answer = (response.get("primary_answer") or {}).get("answer") or ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    tagged = sum(1 for s in sentences if GENERAL_KNOWLEDGE.search(s))
    return {
        "ok": record.get("status") == "ok",
        "latency_ms": record.get("latency_ms"),
        "searches": sum(1 for s in steps if s.get("kind") == "search"),
        "evidence_checks": sum(1 for s in steps if s.get("kind") == "evaluate"),
        "sources": len(response.get("retrieved_sources") or []),
        "citations": len((response.get("primary_answer") or {}).get("citations") or []),
        "general_knowledge": tagged > 0,
        "general_knowledge_share": (tagged / len(sentences)) if sentences else 0.0,
        "asked_clarification": bool(record.get("clarification_turns")),
        "unanswered_clarification": bool(response.get("needs_clarification")),
    }


def outcomes(run_dir: str) -> Dict[str, Any]:
    """Per question: correct (closed-form) and judge score, when scored."""
    correct = {r["question_id"]: bool(r["correct"]) for r in read_jsonl(os.path.join(run_dir, "scores.jsonl"))}
    scores = {}
    for row in read_jsonl(os.path.join(run_dir, "judgments.jsonl")):
        if isinstance(row.get("score"), (int, float)):
            scores[row["question_id"]] = float(row["score"])
    return {"correct": correct, "score": scores}


# --- Statistics -------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def mean(values: Iterable[float]) -> Optional[float]:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test at p = 0.5 (McNemar's exact test, sign test)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(p for p in probs if p <= observed * (1 + 1e-9)))


def bootstrap_interval(
    diffs: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 7
) -> Tuple[Optional[float], Optional[float]]:
    """95% percentile interval of the mean paired difference."""
    if not diffs:
        return None, None
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[min(int(0.975 * samples), samples - 1)]


def paired(this: Dict[str, float], reference: Dict[str, float]) -> Dict[str, Any]:
    """This configuration minus the reference, on the questions both have."""
    shared = sorted(set(this) & set(reference))
    diffs = [float(this[q]) - float(reference[q]) for q in shared]
    better = sum(d > 0 for d in diffs)
    worse = sum(d < 0 for d in diffs)
    low, high = bootstrap_interval(diffs)
    return {
        "paired": len(shared),
        "delta": mean(diffs),
        "ci95": [low, high],
        "better": better,
        "worse": worse,
        "p": binomial_two_sided(better, better + worse) if better + worse else None,
    }


def by_type(this: Dict[str, float], reference: Dict[str, float], types: Dict[str, str]) -> Dict[str, Any]:
    """The paired difference within each question type (no test: groups are small)."""
    groups: Dict[str, List[float]] = {}
    for qid in set(this) & set(reference):
        groups.setdefault(types.get(qid) or "?", []).append(float(this[qid]) - float(reference[qid]))
    return {t: {"paired": len(d), "delta": mean(d)} for t, d in sorted(groups.items())}


def holm(pvalues: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Holm-Bonferroni adjusted p-values (None stays None)."""
    ranked = sorted((p, name) for name, p in pvalues.items() if p is not None)
    m = len(ranked)
    adjusted: Dict[str, Optional[float]] = {name: None for name in pvalues}
    running = 0.0
    for i, (p, name) in enumerate(ranked):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def verdict(role: str, comparison: Dict[str, Any], noise: Optional[float]) -> str:
    """Plain words for one comparison with the full system."""
    p = comparison.get("p_holm")
    delta = comparison.get("delta")
    if delta is None or comparison.get("paired", 0) == 0:
        return "not enough data"
    within_noise = noise is not None and abs(delta) <= noise
    if p is None or p >= ALPHA or within_noise:
        return "no detectable difference"
    if role == "ablation":
        return "this part helps" if delta < 0 else "this part hurts"
    if role == "baseline":
        return "full system is better" if delta < 0 else "baseline is better"
    if role == "sweep":
        return "better than the default" if delta > 0 else "worse than the default"
    return "better than full" if delta > 0 else "worse than full"


# --- The report -------------------------------------------------------------


def build_report(study: Dict[str, Any]) -> Dict[str, Any]:
    reference = study["reference"]
    report: Dict[str, Any] = {"study": study["name"], "reference": reference, "datasets": {}}
    for dataset, spec in study["datasets"].items():
        rows: Dict[str, Dict[str, Any]] = {}
        per_label_outcomes: Dict[str, Dict[str, Any]] = {}
        question_types: Dict[str, str] = {}
        for run in [r for r in runs(study) if r["dataset"] == dataset]:
            records = final_records(run["run_dir"])
            for qid, record in records.items():
                if record.get("question_type"):
                    question_types.setdefault(qid, record["question_type"])
            metrics = [answer_metrics(r) for r in records.values()]
            answered = [m for m in metrics if m["ok"]]
            result = outcomes(run["run_dir"])
            per_label_outcomes[run["label"]] = result
            row = {
                "config": run["config"]["name"],
                "role": run["config"]["role"],
                "what": run["config"].get("what"),
                "run_dir": run["run_dir"],
                "recorded_elsewhere": run["recorded"],
                "questions": len(records),
                "errors": sum(not m["ok"] for m in metrics),
                "latency_p50_ms": percentile([m["latency_ms"] for m in answered], 0.5),
                "latency_p95_ms": percentile([m["latency_ms"] for m in answered], 0.95),
                "searches": mean(m["searches"] for m in answered),
                "evidence_checks": mean(m["evidence_checks"] for m in answered),
                "sources": mean(m["sources"] for m in answered),
                "citations": mean(m["citations"] for m in answered),
                "general_knowledge_rate": mean(float(m["general_knowledge"]) for m in answered),
                "general_knowledge_share": mean(m["general_knowledge_share"] for m in answered),
                "clarification_rate": mean(float(m["asked_clarification"]) for m in answered),
                "unanswered_clarifications": sum(m["unanswered_clarification"] for m in answered),
            }
            if result["correct"]:
                values = [float(v) for v in result["correct"].values()]
                row["accuracy"] = mean(values)
                row["accuracy_ci95"] = list(bootstrap_interval(values))
            if result["score"]:
                row["judge_score"] = mean(result["score"].values())
            rows[run["label"]] = row

        metric = "correct" if any(o["correct"] for o in per_label_outcomes.values()) else (
            "score" if any(o["score"] for o in per_label_outcomes.values()) else None
        )
        noise = None
        comparisons: Dict[str, Dict[str, Any]] = {}
        if metric and reference in per_label_outcomes:
            ref = {q: float(v) for q, v in per_label_outcomes[reference][metric].items()}
            repeats = [label for label in per_label_outcomes if label.startswith(f"{reference}#")]
            for label in repeats:
                other = {q: float(v) for q, v in per_label_outcomes[label][metric].items()}
                rep = paired(other, ref)
                rep["disagree_rate"] = (rep["better"] + rep["worse"]) / rep["paired"] if rep["paired"] else None
                rows[label]["noise"] = rep
                noise = max(noise or 0.0, abs(rep["delta"] or 0.0))
            for label, result in per_label_outcomes.items():
                if label == reference or label in repeats:
                    continue
                other = {q: float(v) for q, v in result[metric].items()}
                comparisons[label] = paired(other, ref)
                comparisons[label]["by_type"] = by_type(other, ref, question_types)
            adjusted = holm({label: c["p"] for label, c in comparisons.items()})
            for label, comparison in comparisons.items():
                comparison["p_holm"] = adjusted[label]
                comparison["verdict"] = verdict(rows[label]["role"], comparison, noise)
                rows[label]["vs_reference"] = comparison
        report["datasets"][dataset] = {
            "metric": {"correct": "accuracy", "score": "judge score"}.get(metric) if metric else None,
            "noise_floor": noise,
            "question_types": sorted(set(question_types.values())),
            "rows": rows,
            "sweeps": [
                {"name": sw["name"], "setting": sw["setting"], "default": sw.get("default"),
                 "points": [{"value": v, "label": label} for v, label in sw["labels"] if label in rows]}
                for sw in study["_sweeps"]
            ],
        }
    return report


def _fmt(value: Optional[float], digits: int = 1, scale: float = 1.0, signed: bool = False) -> str:
    if value is None:
        return "–"
    text = f"{value * scale:+.{digits}f}" if signed else f"{value * scale:.{digits}f}"
    return text


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Ablation study `{report['study']}`", ""]
    lines.append(
        f"Every configuration is paired with **{report['reference']}** on the questions both "
        "answered. Δ is this configuration minus the full system; p is an exact McNemar "
        "test (sign test for judge scores), Holm-corrected within a dataset. A difference "
        "no larger than two runs of the full system differ by is reported as no difference."
    )
    for dataset, block in report["datasets"].items():
        metric = block["metric"]
        scale = 100.0 if metric == "accuracy" else 1.0
        unit = " (points)" if metric == "accuracy" else ""
        lines += ["", f"## {dataset}", ""]
        if not metric:
            lines.append("_Not scored yet: only the cost columns are filled._")
            lines.append("")
        else:
            lines.append(
                f"Noise floor (largest Δ between runs of the full system): "
                f"{_fmt(block['noise_floor'], 1, scale)}{unit}"
                if block["noise_floor"] is not None else
                "Noise floor: not measured (run the full system more than once)."
            )
            lines.append("")
        header = "| configuration | role | n | " + (f"{metric} | Δ vs full | 95% CI | p (Holm) | verdict | " if metric else "") + \
            "p50 s | p95 s | searches | checks | sources | gen. knowledge | clarified |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (header.count("|") - 1)) + "|")
        order = {role: i for i, role in enumerate(ROLES)}
        for label, row in sorted(block["rows"].items(), key=lambda kv: (order[kv[1]["role"]], kv[0])):
            cells = [f"`{label}`", row["role"], str(row["questions"])]
            if metric:
                value = row.get("accuracy") if metric == "accuracy" else row.get("judge_score")
                cells.append(_fmt(value, 1, scale))
                comparison = row.get("vs_reference") or row.get("noise") or {}
                low, high = (comparison.get("ci95") or [None, None])
                cells += [
                    _fmt(comparison.get("delta"), 1, scale, signed=True) if comparison else "",
                    f"{_fmt(low, 1, scale, True)} … {_fmt(high, 1, scale, True)}" if comparison else "",
                    _fmt(comparison.get("p_holm"), 3) if "p_holm" in comparison else "",
                    comparison.get("verdict") or ("repeat of full" if "noise" in row else ""),
                ]
            cells += [
                _fmt(row["latency_p50_ms"], 1, 0.001), _fmt(row["latency_p95_ms"], 1, 0.001),
                _fmt(row["searches"], 1), _fmt(row["evidence_checks"], 1), _fmt(row["sources"], 1),
                f"{_fmt(row['general_knowledge_rate'], 0, 100)}%",
                f"{_fmt(row['clarification_rate'], 0, 100)}%",
            ]
            lines.append("| " + " | ".join(cells) + " |")

        for sweep in block.get("sweeps") or []:
            if len(sweep["points"]) < 2:
                continue
            lines += ["", f"### Sweep: {sweep['name']} (`{sweep['setting']}`, default {sweep['default']})", ""]
            lines.append(f"| value | {metric or 'score'} | Δ vs default | p (Holm) | p50 s | searches | checks | sources |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for point in sweep["points"]:
                row = block["rows"][point["label"]]
                comparison = row.get("vs_reference") or {}
                value = row.get("accuracy") if metric == "accuracy" else row.get("judge_score")
                lines.append("| " + " | ".join([
                    f"{point['value']}" + (" (default)" if point["value"] == sweep["default"] else ""),
                    _fmt(value, 1, scale),
                    _fmt(comparison.get("delta"), 1, scale, signed=True) if comparison else "",
                    _fmt(comparison.get("p_holm"), 3) if "p_holm" in comparison else "",
                    _fmt(row["latency_p50_ms"], 1, 0.001), _fmt(row["searches"], 1),
                    _fmt(row["evidence_checks"], 1), _fmt(row["sources"], 1),
                ]) + " |")

        types = block.get("question_types") or []
        compared = [(label, row) for label, row in block["rows"].items() if row.get("vs_reference")]
        if metric and len(types) > 1 and compared:
            lines += ["", "### Δ vs full by question type", ""]
            lines.append("| configuration | " + " | ".join(types) + " |")
            lines.append("|---|" + "|".join(["---"] * len(types)) + "|")
            for label, row in sorted(compared, key=lambda kv: (order[kv[1]["role"]], kv[0])):
                cells = [
                    _fmt((row["vs_reference"]["by_type"].get(t) or {}).get("delta"), 0, scale, signed=True)
                    for t in types
                ]
                lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# --- Commands ---------------------------------------------------------------


def command_run(study: Dict[str, Any], args: argparse.Namespace) -> int:
    pending = [r for r in runs(study) if not r["recorded"]]
    if args.only_config:
        pending = [r for r in pending if r["config"]["name"] in args.only_config]
    if args.only_dataset:
        pending = [r for r in pending if r["dataset"] in args.only_dataset]
    if args.stage:
        # The full system is every stage's reference.
        pending = [
            r for r in pending
            if r["config"]["stage"] in args.stage or r["config"]["name"] == study["reference"]
        ]
    extra = ["--concurrency", str(args.concurrency)] + (args.runner_arg or [])
    commands = [(r, runner_command(study, r, args.api, extra)) for r in pending]
    if args.dry_run:
        for _, cmd in commands:
            print(" ".join(json.dumps(c) if " " in c or "{" in c else c for c in cmd))
        return 0
    os.makedirs(study["out"], exist_ok=True)

    def execute(item):
        run, cmd = item
        log = os.path.join(study["out"], f"{run['run_id']}.log")
        with open(log, "a", encoding="utf-8") as handle:
            code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT)
        print(f"{'ok' if code == 0 else 'FAILED':>6}  {run['dataset']} / {run['label']}  ({log})", file=sys.stderr)
        return code

    # Configurations run side by side, so a slow hour of the provider falls on
    # all of them rather than on whichever ran then.
    with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as pool:
        codes = list(pool.map(execute, commands))
    failed = sum(code != 0 for code in codes)
    if failed:
        print(f"{failed} run(s) need attention; run the same command again to resume them", file=sys.stderr)
    return 1 if failed else 0


def command_score(study: Dict[str, Any], args: argparse.Namespace) -> int:
    failed = 0
    for run in runs(study):
        spec = study["datasets"][run["dataset"]]
        if spec["scorer"] != "choice" or not os.path.exists(os.path.join(run["run_dir"], "responses.jsonl")):
            continue
        if os.path.exists(os.path.join(run["run_dir"], "scores.jsonl")) and not args.rescore:
            continue
        cmd = [sys.executable, SCORER, run["run_dir"], "--questions", spec["questions"]]
        print(f"score  {run['dataset']} / {run['label']}", file=sys.stderr)
        failed += subprocess.call(cmd) != 0
    return 1 if failed else 0


def command_report(study: Dict[str, Any], args: argparse.Namespace) -> int:
    report = build_report(study)
    out_dir = os.path.join(study["out"], study["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    markdown = render_markdown(report)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Answer every configuration (resumes)")
    run.add_argument("study")
    run.add_argument("--api", required=True, help="FoodScholar started with QA_ABLATION_ENABLED=true")
    run.add_argument("--parallel", type=int, default=3, help="Configurations answered at once")
    run.add_argument("--concurrency", type=int, default=1, help="Questions in flight per configuration")
    run.add_argument("--only-config", action="append", help="Only this configuration (repeatable)")
    run.add_argument("--only-dataset", action="append", help="Only this dataset (repeatable)")
    run.add_argument("--stage", action="append", type=int, help="Only configurations of this stage (repeatable)")
    run.add_argument("--runner-arg", action="append", help="Passed through to run_qa_benchmark.py (repeatable)")
    run.add_argument("--dry-run", action="store_true", help="Print the runner commands only")
    score = sub.add_parser("score", help="Score closed-form datasets (needs GROQ_API_KEY for unclear answers)")
    score.add_argument("study")
    score.add_argument("--rescore", action="store_true")
    report = sub.add_parser("report", help="Compare every configuration with the full system")
    report.add_argument("study")
    args = parser.parse_args(argv)
    study = load_study(args.study)
    return {"run": command_run, "score": command_score, "report": command_report}[args.command](study, args)


if __name__ == "__main__":
    sys.exit(main())
