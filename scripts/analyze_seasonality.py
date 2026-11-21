#!/usr/bin/env python3
"""Describe monthly seasonality and the exact Complaint comparison cohort."""

from __future__ import annotations

import json
from pathlib import Path

from chicago_food.data import load_inspections, sha256_file, verify_snapshot
from chicago_food.seasonality import aggregate_trends, comparison_targets

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"


def main() -> None:
    manifest = verify_snapshot(ROOT / "data/snapshots")
    snapshots = ROOT / "data/snapshots"
    inspections = load_inspections(snapshots / "inspections.parquet")
    targets, audit = comparison_targets(inspections)
    annual, monthly, seasonal, findings = aggregate_trends(targets)
    summary = {
        "schema_version": 1,
        "cohort_policy": {
            "inspection_types": ["Canvass (exact)", "Complaint (exact)"],
            "positive": "Fail", "negative": ["Pass", "Pass w/ Conditions"],
            "denominator": "Recognized binary outcomes; one entity/day/type after conflict exclusion",
            "seasonal_profile_period": ["2019-01-01", "2025-12-31"],
            "rule_change": "2018-07-01; old/new rule periods remain explicitly labeled",
            "year_adjustment": "Equal-year mean monthly rate and mean within-year rate deviation are descriptive",
        },
        "cohort_audit": audit,
        **findings,
        "interpretation": {
            "finding_2": "Month variation challenges universal time-invariant labels in retrospective rescheduling; it does not identify a causal month effect",
            "finding_3": "Complaint is an observational comparison, not a randomized control; cohort mix and selection differ",
            "historical_comparison": "These are Fail outcomes under both rule eras; the 2019 paper studied pre-2018 critical violations, so this is a related check rather than an exact replication",
            "limits": "No seasonal-significance test, causal deployment effect, or complaint-signal model is estimated",
        },
        "sources": [
            "https://github.com/Chicago/food-inspections-evaluation",
            "https://arxiv.org/abs/1910.04906",
        ],
        "provenance": {
            "inspections_sha256": manifest["files"]["inspections.parquet"]["sha256"],
            "source_manifest_sha256": sha256_file(snapshots / "manifest.json"),
            "data_code_sha256": sha256_file(ROOT / "src/chicago_food/data.py"),
            "feature_code_sha256": sha256_file(ROOT / "src/chicago_food/features.py"),
        },
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    annual.to_csv(METRICS / "inspection_type_annual.csv", index=False)
    monthly.to_csv(METRICS / "inspection_type_monthly.csv", index=False)
    seasonal.to_csv(METRICS / "inspection_type_seasonality.csv", index=False)
    (METRICS / "seasonality_complaint.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(f"Saved annual ({len(annual)}), monthly ({len(monthly)}) and seasonal ({len(seasonal)}) aggregate rows.")
    for kind, row in findings["seasonality"].items():
        print(f"{kind}: 2019-2025 n={row['n_2019_2025']:,}; monthly Fail range={row['pooled_month_range_pp']:.2f} pp.")


if __name__ == "__main__":
    main()
