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

``--name`` limits either mode to the named prompts (short or namespaced
name, repeatable), so one deliberately changed fallback can be pushed without
touching UI edits on the others. ``--dry-run`` reads Langfuse and reports what
would be created or updated without writing anything.

Requires LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (+ optional
LANGFUSE_BASE_URL) in the environment.

Usage:
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py           # create missing
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py --force   # also update changed
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py --force --name qa-memory-extractor
    PYTHONPATH=src python scripts/seed_langfuse_prompts.py --force --dry-run
"""
import argparse
import os
import sys

# Allow running both as `python scripts/seed_langfuse_prompts.py` (with
# PYTHONPATH=src) and from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.langfuse import langfuse_enabled, get_langfuse_client  # noqa: E402
from backend.prompts import sync_prompts, ALL_PROMPTS, _PROMPT_NAMESPACE  # noqa: E402


def select_prompts(registry, names):
    """The registry entries whose short or namespaced name is in ``names``.

    Raises ValueError naming anything that matched nothing, so a typo cannot
    silently turn into "nothing to do".
    """
    if not names:
        return list(registry)
    wanted = {
        n if n.startswith(_PROMPT_NAMESPACE) else _PROMPT_NAMESPACE + n
        for n in names
    }
    chosen = [p for p in registry if p.name in wanted]
    missing = sorted(wanted - {p.name for p in chosen})
    if missing:
        raise ValueError(f"Unknown prompt name(s): {', '.join(missing)}")
    return chosen


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
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        metavar="PROMPT",
        help=(
            "Only this prompt (short name like 'qa-memory-extractor' or the "
            "namespaced name). Repeatable. Default: every registry prompt."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be created or updated; write nothing.",
    )
    args = parser.parse_args()

    try:
        registry = select_prompts(ALL_PROMPTS, args.name)
    except ValueError as exc:
        print(exc)
        return 2

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

    # A fetch failure is not "prompt missing". Refuse to guess against a host
    # that cannot be reached: from inside the cluster the in-cluster service
    # name may not resolve, and the public host may be down.
    base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    reachable = getattr(client, "auth_check", None)
    if reachable is not None:
        try:
            ok = bool(reachable())
        except Exception as exc:
            ok = False
            print(f"Langfuse auth check raised: {exc}")
        if not ok:
            print(
                f"Langfuse is unreachable or rejected the keys at {base_url}. "
                "Nothing was read or written. Point LANGFUSE_BASE_URL at a "
                "running instance (the public host, if the in-cluster service "
                "does not resolve) and retry."
            )
            return 1

    result = sync_prompts(
        client=client, registry=registry, force=args.force, dry_run=args.dry_run
    )
    client.flush()
    print(
        f"{'Dry run' if args.dry_run else 'Done'} "
        f"({len(registry)} of {len(ALL_PROMPTS)} registry prompts, "
        f"force={args.force}): "
        f"created={result['created']} updated={result['updated']} "
        f"skipped={result['skipped']} failed={result['failed']}"
    )
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
