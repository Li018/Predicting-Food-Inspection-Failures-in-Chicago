#!/usr/bin/env python3
"""Create compact Topic 1 data aggregates; keep row-level feature data local.

Run from the repository root with ``PYTHONPATH=src python scripts/profile_data.py``.
The official projection is verified before using the existing cleaning and
point-in-time feature modules. No source rows or long text are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_food.data import load_inspections, load_licenses, sha256_file, verify_snapshot
from chicago_food.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    VALID_RESULTS,
    build_features,
    select_targets,
)
from chicago_food.models import temporal_splits


TEMPORAL_CONFIG = {
    "train_start": "2019-01-01", "train_end": "2022-12-31",
    "validation_start": "2023-01-01", "validation_end": "2023-12-31",
    "calibration_start": "2024-01-01", "calibration_end": "2024-12-31",
    "test_start": "2025-01-01", "test_end": "2025-12-31",
}


def rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def outcome_summary(records: pd.DataFrame, targets: pd.DataFrame) -> dict:
    recognized = records["results"].isin(VALID_RESULTS)
    return {
        "inspection_records": int(len(records)),
        "recognized_outcome_records": int(recognized.sum()),
        "all_recognized_outcomes_fail_rate": rate(
            int(records.loc[recognized, "results"].eq("Fail").sum()), int(recognized.sum())
        ),
        "canvass_records": int(records["inspection_type"].eq("Canvass").sum()),
        "eligible_target_days": int(len(targets)),
        "target_failures": int(targets["target"].sum()),
        "target_fail_rate": rate(int(targets["target"].sum()), len(targets)),
        "target_entities": int(targets["entity"].nunique()),
    }


def schema_summary(path: Path) -> dict:
    metadata = pq.ParquetFile(path)
    return {
        "rows": metadata.metadata.num_rows,
        "columns": len(metadata.schema_arrow),
        "schema": {field.name: str(field.type) for field in metadata.schema_arrow},
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def missing_summary(frame: pd.DataFrame, columns: list[str]) -> dict:
    result = {}
    for name in columns:
        series = frame[name]
        missing = series.isna()
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            missing = missing | series.astype("string").str.strip().eq("").fillna(False)
        count = int(missing.sum())
        result[name] = {"missing": count, "missing_rate": rate(count, len(frame))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "data/snapshots")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/metrics/data_summary.json")
    parser.add_argument("--feature-cache", type=Path, default=ROOT / ".local/features.parquet")
    parser.add_argument("--lag-days", type=int, default=1)
    args = parser.parse_args()

    manifest = verify_snapshot(args.snapshot_dir)
    inspection_path = args.snapshot_dir / "inspections.parquet"
    license_path = args.snapshot_dir / "licenses.parquet"
    inspections = load_inspections(inspection_path)
    licenses = load_licenses(license_path)
    inspection_audit = inspections.attrs.get("audit", {}).copy()
    license_audit = licenses.attrs.get("audit", {}).copy()
    targets, target_audit = select_targets(inspections)
    print(f"Verified source rows: inspections={len(inspections):,}; relevant license transactions={len(licenses):,}", flush=True)

    features, feature_audit = build_features(
        inspections, licenses, lag_days=args.lag_days,
        start_date=TEMPORAL_CONFIG["train_start"], end_date=TEMPORAL_CONFIG["test_end"],
    )
    splits = temporal_splits(features, TEMPORAL_CONFIG)

    license_numbers = set(licenses["license_number"]) - {""}
    license_keys = pd.MultiIndex.from_frame(
        licenses.loc[licenses["address_key"].ne(""), ["license_number", "address_key"]]
    ).unique()
    target_identity = targets.set_index("inspection_id")[["license_number", "address_key"]]

    def matching(frame: pd.DataFrame) -> dict:
        identity = target_identity.loc[frame["inspection_id"]]
        number_match = identity["license_number"].isin(license_numbers)
        pair_match = pd.MultiIndex.from_frame(identity).isin(license_keys)
        visible_match = frame["license_history_available"].eq(1)
        return {
            "rows": len(frame),
            "license_number_only_matches": int(number_match.sum()),
            "license_number_only_match_rate": rate(int(number_match.sum()), len(frame)),
            "license_address_matches_ignoring_time": int(pair_match.sum()),
            "license_address_match_rate_ignoring_time": rate(int(pair_match.sum()), len(frame)),
            "point_in_time_visible_matches": int(visible_match.sum()),
            "point_in_time_visible_match_rate": rate(int(visible_match.sum()), len(frame)),
            "lost_to_time_or_missing_term_start": int(pair_match.sum() - visible_match.sum()),
        }

    annual = []
    years = sorted(inspections["inspection_date"].dt.year.unique())
    for year in years:
        row = outcome_summary(
            inspections.loc[inspections["inspection_date"].dt.year.eq(year)],
            targets.loc[targets["inspection_date"].dt.year.eq(year)],
        )
        annual.append({"year": int(year), **row})

    periods = {
        "all_pre_change": ("2010-01-01", "2018-06-30"),
        "all_post_change": ("2018-07-01", TEMPORAL_CONFIG["test_end"]),
        "adjacent_12_months_pre": ("2017-07-01", "2018-06-30"),
        "adjacent_12_months_post": ("2018-07-01", "2019-06-30"),
        "2018_first_half": ("2018-01-01", "2018-06-30"),
        "2018_second_half": ("2018-07-01", "2018-12-31"),
    }
    rule_change = {}
    for label, (start, end) in periods.items():
        rule_change[label] = {
            "start": start, "end": end,
            **outcome_summary(
                inspections.loc[inspections["inspection_date"].between(start, end)],
                targets.loc[targets["inspection_date"].between(start, end)],
            ),
        }

    split_summary = {}
    train_entities = set(splits["train"]["entity_id"])
    for name, frame in splits.items():
        count = len(frame)
        split_summary[name] = {
            "start": TEMPORAL_CONFIG[f"{name}_start"], "end": TEMPORAL_CONFIG[f"{name}_end"],
            "rows": count, "entities": int(frame["entity_id"].nunique()),
            "failures": int(frame["target"].sum()), "fail_rate": float(frame["target"].mean()),
            "cold_start_rows": int(frame["cold_start"].sum()),
            "cold_start_rate": float(frame["cold_start"].mean()),
            "no_scored_history_rate": float(frame["no_scored_history"].mean()),
            "row_share_with_entity_seen_in_training": float(frame["entity_id"].isin(train_entities).mean()),
            "license_matching": matching(frame),
            "numeric_feature_missingness": missing_summary(frame, NUMERIC_FEATURES),
        }
        print(f"{name}: n={count:,}; Fail={frame['target'].mean():.4%}; visible license={frame['license_history_available'].mean():.4%}; cold start={frame['cold_start'].mean():.4%}", flush=True)

    source_summary = {}
    for name, path in (("inspections", inspection_path), ("licenses", license_path)):
        source = manifest["sources"][name]
        source_summary[name] = {
            "dataset": source["dataset"], "source_url": source["source_url"],
            "downloaded_rows": source["downloaded_rows"], "retained_rows": source["retained_rows"],
            "expected_source_rows": source["expected_rows"], "selection": source["selection"],
            "snapshot": schema_summary(path),
        }
    summary = {
        "schema_version": 1,
        "data_end_date": manifest["data_end_date"],
        "snapshot_completed_at_utc": manifest["completed_at_utc"],
        "snapshot_kind": manifest["snapshot_kind"],
        "sources": source_summary,
        "source_cleaning_audit": manifest.get("cleaning_audit", {}),
        "cleaning_audit": {"inspections": inspection_audit, "licenses": license_audit},
        "all_year_target_audit": target_audit,
        "modeling_feature_audit": feature_audit,
        "overall": outcome_summary(inspections, targets),
        "annual": annual,
        "rule_change": {
            "date": "2018-07-01", "periods": rule_change,
            "interpretation": "Descriptive comparison, not a causal estimate: rules, inspected population, and time may all differ.",
        },
        "temporal_config": TEMPORAL_CONFIG,
        "splits": split_summary,
        "license_matching_modeling_cohort": matching(features),
        "license_matching_definition": {
            "number_only": "License number appears anywhere in the cleaned retained source; ignores address and time.",
            "number_and_address": "Exact normalized license number plus address appears in the source; ignores visibility time.",
            "point_in_time": "Exact number/address match with issue date <= scoring date minus lag and term start <= scoring date.",
            "denominator": "Eligible Canvass entity-day targets in the named split, including unmatched and cold-start targets.",
            "tenure": "Years since earliest visible recorded term start; a left-censored proxy, not verified business age.",
        },
        "feature_columns": FEATURE_COLUMNS,
        "numeric_feature_missingness": missing_summary(features, NUMERIC_FEATURES),
        "categorical_missingness": missing_summary(features, CATEGORICAL_FEATURES),
        "limitations": manifest.get("limitations", []) + [
            "Publication timestamps are unavailable: event-time correctness cannot prove historical publication-time availability.",
            "Observed failure risk is conditional on an inspection taking place; annual Top-K is retrospective, not a prospective scheduling trial.",
        ],
    }
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.feature_cache, index=False, compression="zstd")
    cache_metadata = {
        "rows": len(features), "columns": len(features.columns),
        "sha256": sha256_file(args.feature_cache),
        "source_sha256": {name: entry["sha256"] for name, entry in manifest["files"].items()},
        "code_sha256": {
            name: sha256_file(ROOT / "src/chicago_food" / name)
            for name in ("data.py", "features.py")
        },
        "lag_days": args.lag_days, "temporal_config": TEMPORAL_CONFIG,
    }
    args.feature_cache.with_suffix(".meta.json").write_text(json.dumps(cache_metadata, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved aggregate report: {args.output.relative_to(ROOT)}", flush=True)
    print(f"Local feature cache: {len(features):,} rows x {len(features.columns)} columns; {args.feature_cache.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
