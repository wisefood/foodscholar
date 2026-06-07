"""Seed Langfuse with the QA + enrichment prompts from the in-code registry.

Idempotent: creating a prompt with the same name adds a new version only when
the text changed. Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (+ optional
LANGFUSE_BASE_URL) in the environment. The in-code fallbacks in
``backend.prompts`` are the single source of truth for the initial seed; after
seeding, Langfuse becomes the editable source and the fallbacks remain the
safety net.

Usage:
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py
"""
import os
import sys

# Allow running both as `python scripts/seed_langfuse_prompts.py` (with
# PYTHONPATH=src) and from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.langfuse import langfuse_enabled, get_langfuse_client  # noqa: E402
from backend.prompts import sync_prompts, ALL_PROMPTS  # noqa: E402


def main() -> int:
    if not langfuse_enabled():
        print(
            "Langfuse disabled. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "(and optionally LANGFUSE_BASE_URL) to seed prompts."
        )
        return 1

    client = get_langfuse_client()
    if client is None:
        print("Could not initialize Langfuse client.")
        return 1

    # Idempotent: creates missing prompts and re-creates only those whose live
    # text differs from the in-code fallback; identical prompts are skipped.
    result = sync_prompts(client=client)
    client.flush()
    print(
        f"Done ({len(ALL_PROMPTS)} registry prompts): "
        f"created={result['created']} skipped={result['skipped']} "
        f"failed={result['failed']}"
    )
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
