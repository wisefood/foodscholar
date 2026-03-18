#!/usr/bin/env python3
"""CLI wrapper for the shared guideline extraction service."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.guideline_extractor import (  # noqa: E402
    GuidelineExtractionError,
    get_default_dpi,
    get_default_model,
    GuidelineExtractorService,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PDF extraction."""
    parser = argparse.ArgumentParser(
        description="Extract dietary guidelines from relevant pages in a PDF."
    )
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument(
        "-o",
        "--output",
        default="dietary_guidelines.json",
        help="Output JSON path (default: dietary_guidelines.json)",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional markdown output path",
    )
    parser.add_argument(
        "--model",
        default=get_default_model(),
        help=f"Model to use (default: {get_default_model()})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=get_default_dpi(),
        help=f"Render DPI for page images (default: {get_default_dpi()})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the shared guideline extraction pipeline from the command line."""
    args = parse_args()
    extractor = GuidelineExtractorService()

    try:
        result = extractor.process_pdf(
            pdf_path=args.pdf,
            model=args.model,
            dpi=args.dpi,
        )
    except GuidelineExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    write_json(args.output, result)
    print(f"Wrote JSON: {args.output}", file=sys.stderr)

    if args.markdown:
        write_markdown(args.markdown, result)
        print(f"Wrote Markdown: {args.markdown}", file=sys.stderr)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
