"""Seed Langfuse with the QA + enrichment prompts from the in-code registry.

Two modes:

* Default — create ONLY prompts that don't exist in Langfuse yet (the same
  create-only sync app startup runs). Existing prompts are never touched, so
  edits made in the Langfuse UI always win.
* ``--force`` — additionally publish the in-code fallback as a NEW VERSION
  (with the production label) for every prompt whose live text differs. Use
  this after a release that deliberately changed the fallbacks; without it,
  the stale managed versions keep overriding the code. Nothing is destroyed:
  Langfuse keeps all prior versions and the UI can move the label back.

Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (+ optional
LANGFUSE_BASE_URL) in the environment.

Usage:
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py           # create missing
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py --force   # also update changed
"""
import argparse
import os
import sys

# Allow running both as `python scripts/seed_langfuse_prompts.py` (with
# PYTHONPATH=src) and from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.langfuse import langfuse_enabled, get_langfuse_client  # noqa: E402
from backend.prompts import sync_prompts, ALL_PROMPTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Publish the in-code fallback as a new production version for "
            "every prompt whose live Langfuse text differs. UI edits stop "
            "being served (but stay recoverable as prior versions)."
        ),
    )
    args = parser.parse_args()

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

    result = sync_prompts(client=client, force=args.force)
    client.flush()
    print(
        f"Done ({len(ALL_PROMPTS)} registry prompts, force={args.force}): "
        f"created={result['created']} updated={result['updated']} "
        f"skipped={result['skipped']} failed={result['failed']}"
    )
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
