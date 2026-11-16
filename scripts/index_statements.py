"""Index the FSKG statement workbooks for ``retriever="statements"`` (experimental).

Reads every ``*.xlsx`` in a folder (README / Statements / Sample Relations
layout), turns each Statements row into a document
(``services.qa_pipeline.statements.statement_document``), embeds the English
text with the same model the ``rag`` retriever uses, and bulk-indexes it by
Statement ID, so re-running updates in place.

The corpus is experimental and stays local: the target is ELASTIC_HOST, and
anything that is not localhost is refused unless ``--allow-remote`` is given.
The Sample Relations sheets are not indexed (illustrative, not exhaustive).

Usage:
    ELASTIC_HOST=http://127.0.0.1:19200 PYTHONPATH=src \\
        python scripts/index_statements.py OneDrive_1_10-21-2026 [--recreate]
"""
import argparse
import glob
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
from elasticsearch import helpers  # noqa: E402

from backend.elastic import ELASTIC_CLIENT  # noqa: E402
from config import config  # noqa: E402
from services.qa_pipeline.statements import (  # noqa: E402
    INDEX_MAPPING,
    statement_document,
    statements_index,
)
from services.qa_service import QA_EMBED_MODEL  # noqa: E402


def read_documents(folder: str):
    docs, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(folder, "*.xlsx"))):
        workbook = os.path.basename(path)
        rows = pd.read_excel(path, sheet_name="Statements").to_dict("records")
        for row in rows:
            doc = statement_document(row, workbook=workbook)
            if doc is None:
                skipped += 1
            else:
                docs.append(doc)
    return docs, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="Folder holding the FSKG_*.xlsx workbooks")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the index first")
    parser.add_argument("--allow-remote", action="store_true", help="Permit a non-localhost ELASTIC_HOST")
    parser.add_argument("--batch", type=int, default=256, help="Embedding batch size")
    args = parser.parse_args()

    host = urlparse(str(config.settings["ELASTIC_HOST"])).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1") and not args.allow_remote:
        print(
            f"error: ELASTIC_HOST is {config.settings['ELASTIC_HOST']}; the statements corpus is "
            "local-only (pass --allow-remote to override)",
            file=sys.stderr,
        )
        return 2

    docs, skipped = read_documents(args.folder)
    if not docs:
        print(f"error: no statements found in {args.folder}", file=sys.stderr)
        return 2
    print(f"{len(docs)} statements read ({skipped} rows without text or id skipped)", file=sys.stderr)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(QA_EMBED_MODEL)
    vectors = model.encode(
        [doc["text"] for doc in docs], batch_size=args.batch,
        normalize_embeddings=True, show_progress_bar=True,
    )

    client = ELASTIC_CLIENT.client
    index = statements_index()
    if args.recreate and client.indices.exists(index=index):
        client.indices.delete(index=index)
    if not client.indices.exists(index=index):
        # One local node: a replica could never be placed and would leave the
        # index yellow forever.
        client.indices.create(index=index, mappings=INDEX_MAPPING, settings={"number_of_replicas": 0})

    actions = (
        {"_index": index, "_id": doc["statement_id"], "_source": {**doc, "embedding": vector.tolist()}}
        for doc, vector in zip(docs, vectors)
    )
    indexed, errors = helpers.bulk(client, actions, chunk_size=500, raise_on_error=False)
    client.indices.refresh(index=index)
    count = client.count(index=index)["count"]
    print(f"indexed {indexed} into {index} ({len(errors)} errors); index now holds {count}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
