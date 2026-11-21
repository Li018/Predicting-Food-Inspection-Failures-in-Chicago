#!/usr/bin/env python3
"""Check the 13-slide HTML against figure hashes, metrics and original notes."""

import argparse
import json
from pathlib import Path

from chicago_food.slide_validation import (
    SOURCE_HTML_NAME, validate_deck, verify_aggregate_sources, verify_note_source,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / SOURCE_HTML_NAME)
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/metrics/slide_metrics.json")
    parser.add_argument("--notes", type=Path, default=ROOT / "reports/metrics/slide_notes_manifest.json")
    parser.add_argument("--standalone", action="store_true", help="Require embedded image data URIs.")
    parser.add_argument("--report", type=Path, help="Optionally write the compact validation JSON.")
    args = parser.parse_args()
    source = args.html.read_bytes().decode("utf-8")
    metrics, notes = json.loads(args.metrics.read_text()), json.loads(args.notes.read_text())
    source_count = verify_aggregate_sources(ROOT, metrics)
    verify_note_source(ROOT, notes)
    result = validate_deck(source, args.html.resolve(), ROOT, metrics, notes, standalone=args.standalone)
    result["notes_verified_against_git"] = notes["source_commit"]
    result["aggregate_sources_verified"] = source_count
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
