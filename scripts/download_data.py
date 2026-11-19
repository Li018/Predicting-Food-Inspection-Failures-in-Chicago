#!/usr/bin/env python3
"""Download a reproducible, compact projection of the two official Chicago datasets.

Usage: python scripts/download_data.py
The committed snapshots are sufficient for an identical-data rerun. Live refreshes
can differ after City corrections; use --refresh only to intentionally replace them.
Raw paginated CSV files are cached below data/raw and must not be committed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_food.data import (  # noqa: E402
    INSPECTION_COLUMNS,
    LICENSE_COLUMNS,
    load_inspections,
    load_licenses,
    normalize_license,
    sha256_file,
    verify_snapshot,
)

HOST = "https://data.cityofchicago.org"
SOURCES = {
    "inspections": {
        "dataset": "4ijn-s7e5", "key": "inspection_id", "date": "inspection_date",
        "columns": INSPECTION_COLUMNS,
        "url": f"{HOST}/Health-Human-Services/Food-Inspections/4ijn-s7e5",
    },
    "licenses": {
        "dataset": "r5kz-chrr", "key": "license_id", "date": "date_issued",
        "columns": LICENSE_COLUMNS,
        "url": f"{HOST}/Community-Economic-Development/Business-Licenses/r5kz-chrr",
    },
}


def request_bytes(url: str) -> bytes:
    headers = {"User-Agent": "ChicagoFoodInspectionResearch/1.0", "Accept-Encoding": "gzip"}
    if os.environ.get("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                content = response.read()
                return gzip.decompress(content) if response.headers.get("Content-Encoding") == "gzip" else content
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("Unreachable download retry state")


def query_url(dataset: str, params: dict, suffix: str = "json") -> str:
    return f"{HOST}/resource/{dataset}.{suffix}?{urllib.parse.urlencode(params)}"


def download_source(
    name: str, end_exclusive: str, raw_dir: Path, page_size: int,
    keep_licenses: set[str] | None = None, refresh: bool = False, workers: int = 1,
) -> tuple[pd.DataFrame, dict]:
    source = SOURCES[name]
    dataset, key = source["dataset"], source["key"]
    where = f"{source['date']} < '{end_exclusive}T00:00:00'"
    if name == "inspections":
        where += " AND inspection_date >= '2010-01-01T00:00:00'"
    selection = {"$select": ",".join(source["columns"]), "$where": where, "$order": f"{key} ASC"}
    cache_identity = selection if workers == 1 else {"selection": selection, "workers": workers}
    fingerprint = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = raw_dir / dataset / fingerprint
    if refresh and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    header_path = cache / "request.json"
    if header_path.exists():
        header = json.loads(header_path.read_text())
        if header["page_size"] != page_size:
            raise ValueError("Cached page size differs; rerun with the same --page-size or --refresh")
    else:
        bounds = json.loads(request_bytes(query_url(dataset, {
            "$select": f"count(*) as n,max({key}) as max_key", "$where": where,
        })))[0]
        missing_date = json.loads(request_bytes(query_url(dataset, {
            "$select": "count(*) as n", "$where": f"{source['date']} IS NULL",
        })))[0]
        metadata = json.loads(request_bytes(f"{HOST}/api/views/{dataset}.json"))
        header = {
            "dataset": dataset, "source_url": source["url"], "selection": selection,
            "expected_rows": int(bounds["n"]), "max_key": int(bounds["max_key"]),
            "source_rows_without_event_date": int(missing_date["n"]),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "page_size": page_size,
            "source_rows_updated_at": metadata.get("rowsUpdatedAt"),
            "source_description": metadata.get("description"),
            "field_definitions": [
                {"field": c["fieldName"], "name": c["name"], "description": c.get("description")}
                for c in metadata["columns"] if c["fieldName"] in source["columns"]
            ],
        }
        header_path.write_text(json.dumps(header, indent=2, ensure_ascii=False) + "\n")
    def download_shard(shard: int) -> tuple[list[pd.DataFrame], list[dict], int]:
        cursor = header["max_key"] * shard // workers
        upper = header["max_key"] * (shard + 1) // workers
        page_number, shard_total = 0, 0
        kept, pages = [], []
        while True:
            params = dict(selection)
            params["$where"] += f" AND {key} > {cursor} AND {key} <= {upper}"
            params["$limit"] = page_size
            prefix = "" if workers == 1 else f"shard_{shard:02d}_"
            path = cache / f"{prefix}page_{page_number:04d}.csv.gz"
            if not path.exists():
                body = request_bytes(query_url(dataset, params, "csv"))
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(gzip.compress(body, mtime=0))
                temporary.replace(path)
            frame = pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())), dtype=str, keep_default_na=False)
            count = len(frame)
            pages.append({"shard": shard, "page": page_number, "after_key": cursor,
                          "upper_key": upper, "rows": count, "sha256": sha256_file(path)})
            if not count:
                break
            values = pd.to_numeric(frame[key], errors="raise")
            if (values.duplicated().any() or not values.is_monotonic_increasing
                    or values.iloc[0] <= cursor or values.iloc[-1] > upper):
                raise ValueError(f"Invalid {key} page ordering/uniqueness on shard {shard}, page {page_number}")
            cursor = int(values.iloc[-1])
            shard_total += count
            if keep_licenses is not None:
                frame = frame.loc[frame["license_number"].map(normalize_license).isin(keep_licenses)]
            kept.append(frame)
            print(f"{name} shard {shard + 1}/{workers}: {shard_total:,} source rows; "
                  f"{sum(len(p) for p in kept):,} retained", flush=True)
            if count < page_size:
                break
            page_number += 1
        return kept, pages, shard_total

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(download_shard, range(workers)))
    kept = [frame for frames, _, _ in parts for frame in frames]
    pages = [page for _, shard_pages, _ in parts for page in shard_pages]
    total = sum(n for _, _, n in parts)
    if total != header["expected_rows"]:
        raise ValueError(f"{name}: expected {header['expected_rows']} rows, received {total}; use --refresh if source changed")
    result = pd.concat(kept, ignore_index=True)
    header["downloaded_rows"] = total
    header["retained_rows"] = len(result)
    header["pages"] = pages
    return result, header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "snapshots")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--page-size", type=int, default=50000)
    parser.add_argument("--license-workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.page_size <= 50000:
        parser.error("--page-size must be from 1 through 50000")
    if not 1 <= args.license_workers <= 8:
        parser.error("--license-workers must be from 1 through 8")
    end_exclusive = (datetime.strptime(args.end_date, "%Y-%m-%d") + timedelta(days=1)).date().isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists() and not args.refresh:
        manifest = verify_snapshot(args.output_dir)
        if manifest["data_end_date"] != args.end_date:
            raise ValueError("Existing snapshot uses a different end date; choose another output directory or --refresh")
        print("Existing snapshot checksums verified; no live download required.")
        return
    inspections, inspections_source = download_source(
        "inspections", end_exclusive, args.raw_dir, args.page_size, refresh=args.refresh,
    )
    inspection_path = args.output_dir / "inspections.parquet"
    inspections.to_parquet(inspection_path, index=False, compression="zstd")
    licenses_used = set(inspections["license_"].map(normalize_license)) - {""}
    licenses, license_source = download_source(
        "licenses", end_exclusive, args.raw_dir, args.page_size,
        keep_licenses=licenses_used, refresh=args.refresh, workers=args.license_workers,
    )
    license_path = args.output_dir / "licenses.parquet"
    licenses.to_parquet(license_path, index=False, compression="zstd")
    license_source["selection_note"] = (
        "Keep transactions whose public license_number occurs in the downloaded inspection projection. "
        "This only reduces storage; feature construction still applies the per-row issue-date cutoff. "
        "No current status filter is applied."
    )
    cleaned_inspections, cleaned_licenses = load_inspections(inspection_path), load_licenses(license_path)
    manifest = {
        "schema_version": 1, "data_end_date": args.end_date,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_kind": "Official retrospective projection; not a historical publication-time archive",
        "sources": {"inspections": inspections_source, "licenses": license_source},
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in [inspection_path, license_path]},
        "cleaning_audit": {"inspections": cleaned_inspections.attrs["audit"],
                           "licenses": cleaned_licenses.attrs["audit"]},
        "limitations": [
            "Event and issue dates do not establish public availability dates.",
            "The City backfilled additional license addresses on 2025-09-25 (pnmi-a3z5).",
            "License term start is not a verified business opening date.",
            "Final license status is retained for auditing only and must not be backfilled into features.",
            "Violation definitions changed on 2018-07-01; violation text is not downloaded or modeled.",
            "Fresh downloads may differ after retrospective source corrections.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    verify_snapshot(args.output_dir)
    print(json.dumps({"files": manifest["files"], "cleaning_audit": manifest["cleaning_audit"]}, indent=2))


if __name__ == "__main__":
    main()
