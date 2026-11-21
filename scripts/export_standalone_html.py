#!/usr/bin/env python3
"""Export a verified single-file HTML, preserving inline CSS/JS and all notes."""

import argparse
import json
from pathlib import Path

from chicago_food.slide_validation import (
    SOURCE_HTML_NAME,
    github_blob_base,
    standalone_html,
    validate_deck,
    verify_aggregate_sources,
    verify_note_source,
    write_utf8_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / SOURCE_HTML_NAME)
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/metrics/slide_metrics.json")
    parser.add_argument("--notes", type=Path, default=ROOT / "reports/metrics/slide_notes_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/exports/Chicago-Food-Inspections-EN-standalone.html")
    args = parser.parse_args()
    if args.output.resolve() == args.html.resolve():
        raise ValueError("Standalone output must differ from the relative-image source HTML")
    source = args.html.read_bytes().decode("utf-8")
    metrics, notes = json.loads(args.metrics.read_text()), json.loads(args.notes.read_text())
    verify_aggregate_sources(ROOT, metrics)
    verify_note_source(ROOT, notes)
    validate_deck(source, args.html.resolve(), ROOT, metrics, notes)
    result = standalone_html(source, args.html.resolve(), ROOT, blob_base=github_blob_base(ROOT))
    summary = validate_deck(result, args.output.resolve(), ROOT, metrics, notes, standalone=True)
    write_utf8_atomic(args.output, result)
    summary.update(output=str(args.output), bytes=args.output.stat().st_size)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
