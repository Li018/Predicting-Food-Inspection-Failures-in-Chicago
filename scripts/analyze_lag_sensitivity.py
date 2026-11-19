#!/usr/bin/env python3
"""Stress-test observation delays using frozen models on the same 2025 targets.

Only the available inspection/license history changes. Models, preprocessing,
2024 calibrators, target rows, outcomes, and scoring dates remain fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from chicago_food.data import load_inspections, load_licenses, sha256_file, verify_snapshot
from chicago_food.features import FEATURE_COLUMNS, NUMERIC_FEATURES, build_features
from chicago_food.metrics import capacity_weights, evaluate_probabilities
from run_model import load_verified_features


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("logistic_regression", "lightgbm")
ALIGNMENT = ["inspection_id", "inspection_date", "entity_id", "target"]


def changed_features(baseline: pd.DataFrame, variant: pd.DataFrame) -> dict:
    changed = pd.DataFrame(index=baseline.index)
    for feature in FEATURE_COLUMNS:
        left, right = baseline[feature], variant[feature]
        if feature in NUMERIC_FEATURES:
            changed[feature] = ~np.isclose(left.to_numpy(float), right.to_numpy(float),
                                          rtol=0, atol=1e-12, equal_nan=True)
        else:
            changed[feature] = ~(left.eq(right) | (left.isna() & right.isna()))
    return {
        "rows_with_any_feature_change": int(changed.any(axis=1).sum()),
        "row_feature_change_fraction": float(changed.any(axis=1).mean()),
        "cell_feature_change_fraction": float(changed.to_numpy().mean()),
        "change_rate_by_feature": {name: float(changed[name].mean()) for name in FEATURE_COLUMNS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "reports/metrics")
    parser.add_argument("--snapshot-dir", type=Path, default=ROOT / "data/snapshots")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/topic1.json").read_text())
    if config["observation_lag_days"] != 1:
        raise ValueError("This fixed comparison expects the existing one-day-lag baseline")
    all_features, baseline_metadata = load_verified_features(ROOT / ".local/features.parquet", config)
    baseline = all_features.loc[all_features["inspection_date"].between(config["test_start"], config["test_end"])].copy()
    baseline = baseline.sort_values("inspection_id").reset_index(drop=True)
    if baseline["inspection_id"].duplicated().any():
        raise ValueError("Baseline targets require unique inspection identifiers")
    manifest = verify_snapshot(args.snapshot_dir)
    source_hashes = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    if source_hashes != baseline_metadata["source_sha256"]:
        raise ValueError("Lag variants must use exactly the baseline source snapshot")
    selection_path = args.metrics_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["selected_model"] != "logistic_regression":
        raise ValueError("Expected the frozen validation-selected LR primary model")
    comparison = pd.read_csv(args.metrics_dir / "model_comparison.csv").set_index(["model", "probability_kind"])
    models, model_hashes = {}, {}
    for family in FAMILIES:
        path = ROOT / ".local/models" / f"{family}.joblib"
        bundle = joblib.load(path)
        if bundle["provenance"]["feature_sha256"] != baseline_metadata["sha256"]:
            raise ValueError(f"Model and baseline feature cache differ: {family}")
        if bundle["config"] != config:
            raise ValueError(f"Frozen model configuration differs: {family}")
        if bundle["summary"]["calibration"]["fit_period"] != ["2024-01-01", "2024-12-31"]:
            raise ValueError("Frozen calibrator must have been fitted on 2024 only")
        models[family], model_hashes[family] = bundle, sha256_file(path)

    inspections = load_inspections(args.snapshot_dir / "inspections.parquet")
    licenses = load_licenses(args.snapshot_dir / "licenses.parquet")
    variants = {1: baseline}
    variant_details = {}
    print(f"Frozen cohort: n={len(baseline):,}, failures={int(baseline['target'].sum()):,}; no fitting or model selection", flush=True)
    for lag in (7, 30):
        frame, audit = build_features(
            inspections, licenses, lag_days=lag,
            start_date=config["test_start"], end_date=config["test_end"],
        )
        frame = frame.sort_values("inspection_id").reset_index(drop=True)
        if not baseline[ALIGNMENT].equals(frame[ALIGNMENT]):
            raise ValueError(f"Lag {lag} changed target identity, date, entity or outcome")
        path = ROOT / ".local" / f"lag_{lag}_features.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        metadata = {
            "lag_days": lag, "rows": len(frame), "columns": len(frame.columns),
            "sha256": sha256_file(path), "source_sha256": source_hashes,
            "feature_code_sha256": baseline_metadata["code_sha256"],
            "target_window": [config["test_start"], config["test_end"]],
            "baseline_feature_sha256": baseline_metadata["sha256"],
            "target_alignment_verified": ALIGNMENT,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        variants[lag] = frame
        variant_details[str(lag)] = {"feature_cache": str(path.relative_to(ROOT)), "metadata": metadata, "feature_audit": audit}
        print(f"Built lag {lag} days: n={len(frame):,}; visible license={frame['license_history_available'].mean():.4%}", flush=True)

    rows, baseline_predictions, baseline_metrics = [], {}, {}
    capacity = int(np.ceil(len(baseline) * 0.1))
    baseline_weights = {}
    for lag, frame in variants.items():
        changes = changed_features(baseline, frame)
        variant_details.setdefault(str(lag), {})["feature_changes_vs_lag_1"] = changes
        variant_details[str(lag)]["visible_license_rate"] = float(frame["license_history_available"].mean())
        for family, bundle in models.items():
            raw = bundle["pipeline"].predict_proba(frame)[:, 1]
            probabilities = bundle["calibrator"].predict(raw)
            metrics = evaluate_probabilities(
                frame["target"], probabilities, dates=frame["inspection_date"],
                top_fractions=tuple(config["top_fractions"]),
            )
            if lag == 1:
                for name, value in metrics.items():
                    if value is not None and not np.isclose(value, comparison.loc[(family, "calibrated"), name], rtol=0, atol=1e-12):
                        raise ValueError(f"Frozen-model baseline differs from committed metrics: {family}/{name}")
                saved = pd.read_parquet(ROOT / ".local/predictions" / f"{family}.parquet")
                saved = saved.loc[saved["split"].eq("test")].sort_values("inspection_id").reset_index(drop=True)
                if not baseline[ALIGNMENT].equals(saved[ALIGNMENT]):
                    raise ValueError(f"Frozen baseline prediction alignment differs: {family}")
                if not np.allclose(probabilities, saved["calibrated_probability"], rtol=0, atol=1e-12):
                    raise ValueError(f"Frozen baseline probabilities differ from the saved predictions: {family}")
                baseline_predictions[family] = probabilities
                baseline_metrics[family] = metrics
                baseline_weights[family] = capacity_weights(probabilities, capacity)
            prediction_delta = probabilities - baseline_predictions[family]
            weights = capacity_weights(probabilities, capacity)
            retention = float(np.minimum(weights, baseline_weights[family]).sum() / capacity)
            row = {
                "model": family, "primary_model": family == selection["selected_model"],
                "lag_days": lag, "split": "test", "year": 2025,
                "probability_kind": "frozen_2024_calibrated",
                "license_history_available_rate": float(frame["license_history_available"].mean()),
                "cold_start_rate": float(frame["cold_start"].mean()),
                "row_feature_change_fraction": changes["row_feature_change_fraction"],
                "cell_feature_change_fraction": changes["cell_feature_change_fraction"],
                "prediction_mean_absolute_change_vs_lag_1": float(np.abs(prediction_delta).mean()),
                "prediction_mean_change_vs_lag_1": float(prediction_delta.mean()),
                "prediction_p95_absolute_change_vs_lag_1": float(np.quantile(np.abs(prediction_delta), 0.95)),
                "prediction_max_absolute_change_vs_lag_1": float(np.abs(prediction_delta).max()),
                "prediction_changed_fraction_vs_lag_1": float((np.abs(prediction_delta) > 1e-12).mean()),
                "top_10pct_member_retention_vs_lag_1": retention,
                "top_10pct_capacity": capacity,
                **metrics,
            }
            for name, value in metrics.items():
                if value is not None and (
                    name in {"roc_auc", "average_precision", "brier_score", "log_loss"}
                    or name.startswith(("top_", "daily_top_"))
                ):
                    row[f"delta_{name}_vs_lag_1"] = value - baseline_metrics[family][name]
            rows.append(row)
            print(f"lag={lag}, {family}: AP={metrics['average_precision']:.6f}, Brier={metrics['brier_score']:.6f}, "
                  f"Top10 retained={retention:.4%}, prediction MAD={row['prediction_mean_absolute_change_vs_lag_1']:.6f}", flush=True)
    table = pd.DataFrame(rows)
    csv_path = args.metrics_dir / "lag_sensitivity.csv"
    table.to_csv(csv_path, index=False)
    summary = {
        "schema_version": 1, "scope": "Frozen-model observation-lag sensitivity on identical 2025 Canvass targets",
        "selected_model": selection["selected_model"], "lag_days": [1, 7, 30],
        "cohort": {"n": len(baseline), "entities": int(baseline["entity_id"].nunique()),
                   "failures": int(baseline["target"].sum()), "alignment_fields": ALIGNMENT},
        "fit_performed": False, "model_selection_performed": False,
        "calibrators": "The original 2024 calibrators are frozen and applied unchanged to every lag; no variant is recalibrated.",
        "source_sha256": source_hashes, "baseline_feature_sha256": baseline_metadata["sha256"],
        "model_sha256": model_hashes, "model_selection_sha256": sha256_file(selection_path),
        "feature_code_sha256": baseline_metadata["code_sha256"],
        "variants": variant_details,
        "definitions": {
            "available_history": "Inspection event dates and license issue dates must be <= scoring date minus lag. License term start must also be <= scoring date, as in the existing feature module.",
            "feature_change_fraction": "Share of identical test rows with any of the 39 allowlisted predictors changed versus lag 1; NaN equals NaN; numeric absolute tolerance 1e-12.",
            "prediction_change": "Absolute or signed changes in frozen-calibrator probabilities relative to lag 1 for the same model and target.",
            "top_10pct_retention": "sum(min(lag-1 allocation weight, lag-k allocation weight)) / ceil(0.1*n); the existing expected fractional allocation handles boundary ties.",
            "delta_metrics": "Variant metric minus the same model's lag-1 metric, with identical outcomes, dates and ranking conventions.",
            "confidence_intervals": "Not estimated for these lag perturbations; no significance or causal claim is made.",
        },
        "interpretation_limits": [
            "This tests robustness of the frozen deployed estimator to delayed records, not the performance of a separately trained lag-specific model.",
            "Same-day and future-dated events remain excluded by the existing point-in-time builder; longer lags remove additional information rather than adding future events.",
            "Retrospective publication delays, address backfills and corrections are not fully reconstructable from the snapshot; 7/30-day lags do not prove historical publication-time availability.",
            "Stable aggregate scores do not imply unchanged individual risks; the report includes per-row change rates, probability shifts and Top10 retention.",
        ],
        "metric_rows": rows,
        "aggregate_csv": str(csv_path.relative_to(ROOT)), "aggregate_csv_sha256": sha256_file(csv_path),
    }
    (args.metrics_dir / "lag_sensitivity.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} aggregate model/lag rows; row-level lag feature caches remain local", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
