#!/usr/bin/env python3
"""Bind authoritative display text and AP bars without changing speaker notes."""

import argparse
import json
from pathlib import Path

from chicago_food.slide_validation import (
    SOURCE_HTML_NAME,
    bind_html_metrics,
    validate_deck,
    validate_notes,
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
    args = parser.parse_args()
    source = args.html.read_bytes().decode("utf-8")
    metrics, notes = json.loads(args.metrics.read_text()), json.loads(args.notes.read_text())
    verify_aggregate_sources(ROOT, metrics)
    verify_note_source(ROOT, notes)
    validate_notes(source, notes)
    bound = bind_html_metrics(source, metrics)
    result = validate_deck(bound, args.html.resolve(), ROOT, metrics, notes)
    write_utf8_atomic(args.html, bound)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
