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
from backend import prompts as P  # noqa: E402

_REGISTRY = [
    P.ENRICHMENT_ANNOTATION,
    P.ENRICHMENT_KEYWORDS_SYSTEM,
    P.ENRICHMENT_KEYWORDS_USER,
    P.QA_ANSWER_RAG_SYSTEM,
    P.QA_ANSWER_RAG_USER,
    P.QA_ANSWER_NORAG_SYSTEM,
    P.QA_ANSWER_NORAG_USER,
    P.QA_CLARIFIER_SYSTEM,
    P.QA_STARTER_QUESTIONS,
    P.QA_TIPS_FROM_GUIDELINES,
    P.QA_TIPS_FROM_ARTICLES,
    P.QA_TIP_REWRITE,
]


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

    for prompt in _REGISTRY:
        client.create_prompt(
            name=prompt.name,
            type="text",
            prompt=prompt.fallback,
            labels=[prompt.label],
        )
        print(f"seeded {prompt.name}")

    client.flush()
    print(f"Done: {len(_REGISTRY)} prompts seeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
