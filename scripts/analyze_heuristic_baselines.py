#!/usr/bin/env python3
"""Add three historical business-rule baselines without fitting any model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chicago_food.data import sha256_file
from chicago_food.heuristics import add_heuristic_slide_metrics, align_test_data, evaluate_rules


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    root = ROOT
    metrics = root / "reports/metrics"
    bundle = joblib.load(root / ".local/models/logistic_regression.joblib")
    provenance = bundle["provenance"]
    expected = {
        ".local/features.parquet": provenance["feature_sha256"],
        "config/topic1.json": provenance["config_sha256"],
        **{f"src/chicago_food/{name}": digest for name, digest in {
            **provenance["feature_code_sha256"], **provenance["model_code_sha256"],
        }.items()},
        **{f"data/snapshots/{name}": digest for name, digest in provenance["source_sha256"].items()},
    }
    manifest = json.loads((root / "data/snapshots/manifest.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        if manifest["files"][name]["sha256"] != digest:
            raise ValueError(f"Snapshot manifest differs from the frozen LR: {name}")
    for name, digest in expected.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Frozen input differs from the original LR: {name}")
    for name in (".local/models/logistic_regression.joblib", ".local/predictions/logistic_regression.parquet",
                 "data/snapshots/manifest.json"):
        expected[name] = sha256_file(root / name)
    columns = ["inspection_id", "entity_id", "inspection_date", "target", "last_result",
               "historical_failure_rate", "prior_fail_days"]
    features = pd.read_parquet(root / ".local/features.parquet", columns=columns)
    saved = pd.read_parquet(root / ".local/predictions/logistic_regression.parquet")
    frame, probabilities = align_test_data(features, saved)
    table, summary = evaluate_rules(frame, probabilities)
    primary = table.loc[table["model"].eq("logistic_regression")].iloc[0]
    comparison = pd.read_csv(metrics / "model_comparison.csv")
    reference = comparison.loc[comparison["model"].eq("logistic_regression") & comparison["probability_kind"].eq("calibrated")].iloc[0]
    checked = ("roc_auc", "average_precision", "top_10pct_precision", "top_10pct_selected_n",
               "daily_top_10pct_expected_failures_found", "daily_top_10pct_selected_n",
               "daily_top_10pct_lift_vs_daily_random")
    for key in checked:
        if not np.isclose(primary[key], reference[key], rtol=0, atol=1e-10):
            raise ValueError(f"Rule evaluation differs from the original LR convention: {key}")
    summary.update(
        schema_version=1, test_year=2025, n=len(frame), failures=int(frame["target"].sum()),
        primary_model="logistic_regression", refitted=False, input_sha256=expected,
        original_lr_metrics_verified=list(checked),
        records=table.to_dict("records"),
        outputs=["heuristic_baselines.csv", "heuristic_baselines.json"],
    )
    if any(sha256_file(root / name) != digest for name, digest in expected.items()):
        raise RuntimeError("A frozen input changed during rule evaluation")
    table.to_csv(metrics / "heuristic_baselines.csv", index=False, float_format="%.12g")
    (metrics / "heuristic_baselines.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    slide_path = metrics / "slide_metrics.json"
    if slide_path.is_file():
        slides = add_heuristic_slide_metrics(json.loads(slide_path.read_text()), root)
        slide_path.write_text(json.dumps(slides, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(table[["model", "roc_auc", "average_precision", "top_10pct_precision",
                 "daily_top_10pct_expected_failures_found", "daily_top_10pct_lift_vs_daily_random"]].to_string(index=False))
    print(json.dumps({"best_heuristic": summary["best_heuristic"], "n": len(frame),
                      "annual_slots": int(primary["top_10pct_selected_n"]),
                      "daily_slots": int(primary["daily_top_10pct_selected_n"]),
                      "protected_inputs_unchanged": True}))


if __name__ == "__main__":
    main()
