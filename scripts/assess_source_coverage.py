"""Can the agentic QA retrieval reach every source? Measure it.

Reads the ``articles`` and ``guidelines`` indexes (read-only: search, count,
point-in-time; nothing is written) and runs three layers, each reported per
source type, per guide / journal and per publication decade:

  census         every document checked against the gates that make it
                 unreachable whatever the question (editorial filters, rule
                 status, rule-text floor, zero-score tiers, missing vectors)
  probe          every eligible document searched for with its own title,
                 text and keywords through the real legs, fusion, ranking and
                 selection; resumable (results are appended as they finish)
  concentration  real questions through both branches: how much of the
                 corpus answers actually draw on, and how unevenly

Run it where the service runs (same env vars, same settings), e.g. in the
FoodScholar pod or with the service's env and a tunnel to Elasticsearch:

  python scripts/assess_source_coverage.py --out-dir /tmp/coverage
  python scripts/assess_source_coverage.py --out-dir /tmp/coverage --sample 3000
  python scripts/assess_source_coverage.py --out-dir /tmp/coverage \\
      --questions benchmarks/questions.jsonl --layers census,concentration

Needs ELASTIC_HOST (default http://elastic:9200) and, for the vector legs, the
sentence-transformers model the service embeds with (QA_EMBED_MODEL, default
sentence-transformers/all-MiniLM-L6-v2; pass --no-embed for keyword legs only).
The ranking settings (QA_RRF_CANDIDATES, QA_MIN_SCORE, QA_PER_DOC_CAP, ...)
come from the same config module the service reads, so set them as the
deployment does or the probe measures a different pipeline.

Outputs in --out-dir: report.json, report.md, census_docs.jsonl and
probe_results.jsonl (rerunning with the same --out-dir skips documents
already probed; delete the file to start over).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embed_model_name() -> str:
    if os.getenv("QA_EMBED_MODEL"):
        return os.environ["QA_EMBED_MODEL"]
    try:  # the service's own constant, when its dependencies are installed
        from services.qa_service import QA_EMBED_MODEL

        return QA_EMBED_MODEL
    except Exception:
        return DEFAULT_EMBED_MODEL


def _make_embedder(name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    lock = threading.Lock()

    def embed(texts):
        with lock:
            vectors = model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    return embed


def _fail(message: str) -> None:
    print(f"\nCannot run the coverage assessment: {message}\n", file=sys.stderr)
    sys.exit(2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="where the report and the resumable results go")
    parser.add_argument("--layers", default="census,probe", help="any of census,probe,concentration")
    parser.add_argument("--sample", type=int, default=None, help="probe a stratified sample of N documents (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--workers", type=int, default=4, help="concurrent searches (keep it gentle on a live cluster)")
    parser.add_argument("--expertise", default="intermediate", choices=["beginner", "intermediate", "expert"])
    parser.add_argument("--top-k", type=int, default=5, help="studies per answer, as the request's top_k")
    parser.add_argument("--questions", help="questions for the concentration layer: lines, JSONL or a JSON list")
    parser.add_argument("--no-embed", action="store_true", help="keyword legs only (skips the vector legs)")
    parser.add_argument("--reuse-census", action="store_true", help="reuse census_docs.jsonl from --out-dir")
    parser.add_argument("--limit-per-index", type=int, default=None, help="census at most N docs per index (smoke runs)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("elasticsearch", "elastic_transport", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    layers = {layer.strip() for layer in args.layers.split(",") if layer.strip()}
    unknown = layers - {"census", "probe", "concentration"}
    if unknown:
        _fail(f"unknown layer(s): {', '.join(sorted(unknown))}")
    if "concentration" in layers and not args.questions:
        _fail("the concentration layer needs --questions PATH")

    from backend.elastic import ELASTIC_CLIENT, ELASTIC_HOST
    from services import qa_coverage as cov

    client = ELASTIC_CLIENT.client
    try:
        if not client.ping():
            raise RuntimeError("ping returned false")
        for index in (cov.ARTICLES_INDEX, cov.GUIDELINES_INDEX):
            if not client.indices.exists(index=index):
                raise RuntimeError(f"index '{index}' does not exist")
    except Exception as exc:
        _fail(
            f"Elasticsearch at ELASTIC_HOST={ELASTIC_HOST} is not reachable or incomplete ({exc}). "
            "Set ELASTIC_HOST to the cluster the service uses (run inside the FoodScholar pod, or "
            "tunnel to it and export ELASTIC_HOST=http://localhost:9200)."
        )

    embed = None
    if not args.no_embed and ({"probe", "concentration"} & layers):
        name = _embed_model_name()
        try:
            embed = _make_embedder(name)
            embed(["warm-up"])
        except Exception as exc:
            _fail(
                f"cannot load the embedding model '{name}' ({exc}). Install sentence-transformers and make "
                "the model available (the service image has it), set QA_EMBED_MODEL, or pass --no-embed "
                "to measure the keyword legs only."
            )

    os.makedirs(args.out_dir, exist_ok=True)
    census_path = os.path.join(args.out_dir, "census_docs.jsonl")
    started = time.monotonic()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elastic_host": ELASTIC_HOST,
        "options": vars(args),
        "config": cov.runtime_config(),
        "embedding": None if embed is None else _embed_model_name(),
    }

    # ---- Census (always: the other layers need the eligible set) ----------
    if args.reuse_census and os.path.exists(census_path):
        with open(census_path, encoding="utf-8") as fh:
            records = [cov.DocRecord.from_json(json.loads(line)) for line in fh if line.strip()]
        logging.info("Reused census: %d documents", len(records))
    else:
        logging.info("Census: reading both indexes (read-only)...")
        records = cov.run_census(client, limit=args.limit_per_index)
        with open(census_path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        logging.info("Census: %d documents classified", len(records))
    report["census"] = cov.summarize_census(records)

    ctx = cov.ProbeContext(embed=embed, expertise_level=args.expertise, top_k=args.top_k)

    # ---- Probe -------------------------------------------------------------
    if "probe" in layers:
        eligible = [r for r in records if r.status != "unreachable" and r.queries]
        chosen = cov.stratified_sample(eligible, args.sample, seed=args.seed)
        logging.info("Probe: %d of %d eligible documents", len(chosen), len(eligible))
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if now - last[0] > 15 or done == total:
                last[0] = now
                logging.info("Probe: %d/%d", done, total)

        rows = cov.run_probe(
            chosen,
            ctx,
            results_path=os.path.join(args.out_dir, "probe_results.jsonl"),
            workers=args.workers,
            progress=progress,
        )
        report["probe"] = cov.summarize_probe(
            rows, candidates=cov.retrieval._candidates(), vectors=cov.vector_population(records)
        )
        report["probe"]["sampled"] = args.sample is not None

    # ---- Concentration -----------------------------------------------------
    if "concentration" in layers:
        questions = cov.load_questions(args.questions)
        logging.info("Concentration: %d questions", len(questions))
        report["concentration"] = cov.run_concentration(questions, records, ctx, workers=args.workers)

    report["elapsed_s"] = round(time.monotonic() - started, 1)
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    markdown = cov.render_markdown(report)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown)
    print(markdown)
    print(f"Full report: {os.path.join(args.out_dir, 'report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
