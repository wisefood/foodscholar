#!/usr/bin/env python3
"""Seed the integrator's backlog from the project's source catalogue.

`Sources_Catalogue.ods` in wisefood-client holds 219 candidate sources across
five sheets whose columns differ per sheet — a dietary guide row knows its
country and population group, a journal row knows its CiteScore and publisher.
Rather than flatten them into a shape that fits none, the columns each sheet
actually has are kept verbatim in `attributes`, and only the handful the queue
sorts and filters on are lifted out.

Idempotent: `external_key` is sheet + title + url, so re-running after the
spreadsheet grows adds the new rows and leaves the rest alone.

    PYTHONPATH=src python scripts/seed_integrator_backlog.py \\
        "../wisefood-client/Sources_Catalogue (1).ods"
"""
from __future__ import annotations

import html
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

#: Sheet → the catalog entity kind it feeds.
SHEET_KINDS = {
    "National_Dietary_Guides": "guide",
    "Nutrition_Journals": "article",
    "Food_Composition_Tables": "fctable",
    "Textbooks": "textbook",
    "Recipe_Collections": "rcollection",
}

#: Column headers that mean the same thing across sheets, normalised.
COLUMN_ALIASES = {
    "country": "country",
    "title": "title",
    "source": "title",
    "population group": "population_group",
    "language": "language",
    "url": "url",
}


def read_ods(path: Path) -> Dict[str, List[List[str]]]:
    """Every sheet as a list of rows. No pandas — one zip and some regex."""
    xml = zipfile.ZipFile(path).read("content.xml").decode()
    sheets: Dict[str, List[List[str]]] = {}
    for table in re.finditer(
        r'<table:table table:name="([^"]+)"(.*?)</table:table>', xml, re.S
    ):
        rows: List[List[str]] = []
        for raw in re.findall(
            r"<table:table-row[^>]*>(.*?)</table:table-row>", table.group(2), re.S
        ):
            cells: List[str] = []
            for cell in re.finditer(
                r"<table:table-cell([^>]*)>(.*?)</table:table-cell>"
                r"|<table:table-cell([^>]*)/>", raw, re.S
            ):
                attrs = cell.group(1) or cell.group(3) or ""
                body = cell.group(2) or ""
                repeat = re.search(r'number-columns-repeated="(\d+)"', attrs)
                text = " ".join(re.findall(r"<text:p>(.*?)</text:p>", body, re.S))
                text = html.unescape(re.sub(r"<[^>]+>", "", text)).strip()
                cells += [text] * min(int(repeat.group(1)) if repeat else 1, 16)
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                rows.append(cells)
        sheets[table.group(1)] = rows
    return sheets


#: What to call a row on a sheet that has no Title column. The food-composition
#: sheet is Country / Format / Free Access / URL — the country *is* the
#: identity there, and "Austria" alone is not a name a curator can act on.
SYNTHETIC_TITLES = {
    "fctable": lambda r: " ".join(
        x for x in [r.get("Country", ""), "food composition table"] if x
    ).strip(),
}


def to_items(sheets: Dict[str, List[List[str]]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Every source the spreadsheet names, and a note for every row skipped.

    Skips are reported rather than swallowed: a seeder that quietly imports
    194 of 219 rows looks like it worked.
    """
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for sheet, rows in sheets.items():
        kind = SHEET_KINDS.get(sheet)
        if not kind:
            skipped.append(f"{sheet}: not a sheet this seeder knows")
            continue
        if len(rows) < 2:
            continue
        headers = [h.strip() for h in rows[0]]
        for number, row in enumerate(rows[1:], start=2):
            record = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
            lifted = {
                COLUMN_ALIASES[k.strip().lower()]: v
                for k, v in record.items()
                if k.strip().lower() in COLUMN_ALIASES and v
            }
            title = lifted.pop("title", "") or record.get("Title") or ""
            if not title and kind in SYNTHETIC_TITLES:
                title = SYNTHETIC_TITLES[kind](record)
            if not title:
                skipped.append(
                    f"{sheet} row {number}: no title and nothing to build one from "
                    f"({record or 'empty row'})"
                )
                continue
            url = lifted.pop("url", "") or ""
            items.append({
                "external_key": f"{sheet}|{title}|{url}"[:300],
                "kind": kind,
                "title": title,
                "url": url or None,
                "source_sheet": sheet,
                # Everything the sheet said, kept as it was said. The assistant
                # reads these: a journal's CiteScore and a textbook's declared
                # licence are exactly the kind of thing it should weigh.
                "attributes": {k: v for k, v in record.items() if v},
                **lifted,
            })
    return items, skipped


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no such file: {path}")
        return 1

    items, skipped = to_items(read_ods(path))
    by_kind: Dict[str, int] = {}
    for item in items:
        by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
    print(f"parsed {len(items)} sources: " +
          ", ".join(f"{k} {n}" for k, n in sorted(by_kind.items())))
    if skipped:
        print(f"skipped {len(skipped)} row(s):")
        for note in skipped:
            print("  -", note[:160])

    if "--dry-run" in sys.argv:
        for item in items[:5]:
            print("  ", item["kind"], "|", item["title"][:60], "|", item.get("country"))
        return 0

    from backend.db_init import init_db
    from integrator import service

    init_db()
    result = service.seed_backlog(items)
    print(f"added {result['added']}, already present {result['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
