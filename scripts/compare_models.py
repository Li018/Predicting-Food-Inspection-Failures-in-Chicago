#!/usr/bin/env python3
"""Compare saved model runs without fitting or changing the frozen selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chicago_food.data import sha256_file
from chicago_food.metrics import evaluate_probabilities

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("prevalence", "logistic_regression", "lightgbm")
PAIRED_METRICS = (
    "roc_auc", "average_precision", "brier_score", "log_loss", "top_10pct_precision",
)


def paired_differences(predictions_dir: Path, *, seed: int, replicates: int) -> dict:
    """Resample the same establishment clusters for both saved probability vectors."""
    frames = {}
    for family in ("logistic_regression", "lightgbm"):
        frame = pd.read_parquet(predictions_dir / f"{family}.parquet")
        frames[family] = frame.loc[frame["split"].eq("test")].set_index(
            "inspection_id", verify_integrity=True
        ).sort_index()
    left, right = frames["logistic_regression"], frames["lightgbm"]
    alignment_columns = ["inspection_date", "entity_id", "target"]
    if not left[alignment_columns].equals(right[alignment_columns]):
        raise ValueError("Paired predictions must have identical inspection IDs, dates, entities and labels")
    y = left["target"].to_numpy(dtype=int)
    probabilities = {
        family: frame["calibrated_probability"].to_numpy(dtype=float)
        for family, frame in frames.items()
    }
    codes, entities = pd.factorize(left["entity_id"], sort=True)
    estimates = {
        family: evaluate_probabilities(y, p, top_fractions=(0.1,))
        for family, p in probabilities.items()
    }
    samples = {metric: [] for metric in PAIRED_METRICS}
    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(y))
    for _ in range(replicates):
        draw = rng.integers(0, len(entities), size=len(entities))
        multiplicities = np.bincount(draw, minlength=len(entities))
        indices = np.repeat(row_indices, multiplicities[codes])
        # Crucially, both models see the very same resampled rows and labels.
        resampled = {
            family: evaluate_probabilities(y[indices], p[indices], top_fractions=(0.1,))
            for family, p in probabilities.items()
        }
        for metric in PAIRED_METRICS:
            lhs = resampled["logistic_regression"][metric]
            rhs = resampled["lightgbm"][metric]
            if lhs is not None and rhs is not None and np.isfinite([lhs, rhs]).all():
                samples[metric].append(float(rhs - lhs))
    intervals = {}
    for metric in PAIRED_METRICS:
        lower, upper = np.quantile(samples[metric], [0.025, 0.975])
        intervals[metric] = {
            "estimate": estimates["lightgbm"][metric] - estimates["logistic_regression"][metric],
            "lower": float(lower), "upper": float(upper),
            "valid_replicates": len(samples[metric]),
            "contains_zero": bool(lower <= 0 <= upper),
            "direction_favoring_lightgbm": "negative" if metric in {"brier_score", "log_loss"} else "positive",
        }
    return {
        "schema_version": 1,
        "difference": "LightGBM calibrated minus Logistic Regression calibrated",
        "split": "test", "year": 2025,
        "alignment": {
            "key": "inspection_id", "verified_fields": alignment_columns,
            "n": len(y), "n_entities": len(entities), "failures": int(y.sum()),
        },
        "bootstrap": {
            "method": "Paired establishment-cluster percentile bootstrap",
            "n_replicates": replicates, "seed": seed, "confidence": 0.95,
            "same_cluster_draw_for_both_models": True,
            "within_cluster_rows": "Retain all inspection rows for each sampled establishment",
        },
        "intervals": intervals,
        "limitations": [
            "Conditional on fixed estimators, source snapshot and historical inspection scheduling.",
            "No retraining, calendar-drift, publication-lag or policy-effect uncertainty is included.",
            "Percentile intervals describe paired uncertainty; no multiplicity-adjusted hypothesis test was performed.",
            "Annual pooled Top 10% ranking; these intervals are not intervals for daily inspection queues.",
            "Test differences do not alter the model selected using 2023 validation AP.",
        ],
    }


def compare(metrics_dir: Path, predictions_dir: Path) -> dict:
    summaries = {
        family: json.loads((metrics_dir / f"{family}_metrics.json").read_text())
        for family in FAMILIES
    }
    reference = summaries[FAMILIES[0]]
    for family, summary in summaries.items():
        if summary["model"] != family:
            raise ValueError("Model summary family mismatch")
        for field in ("config", "numeric_features", "categorical_features", "split_summary"):
            if summary[field] != reference[field]:
                raise ValueError(f"Cannot compare models with different {field}")
        for field in ("feature_sha256", "source_sha256", "feature_code_sha256", "model_code_sha256"):
            if summary["provenance"][field] != reference["provenance"][field]:
                raise ValueError(f"Cannot compare models with different provenance: {field}")
    validation_ap = {
        family: summary["selection"]["validation_average_precision"]
        for family, summary in summaries.items()
    }
    # Tuple order is fixed before scoring; held-out scores never resolve ties.
    selected = max(FAMILIES, key=lambda family: validation_ap[family])
    rows = []
    for family, summary in summaries.items():
        for metrics in summary["metrics"]:
            if metrics["split"] != "test":
                continue
            kind = metrics["probability_kind"]
            bootstrap = summary["bootstrap"][kind]
            row = {
                **metrics,
                "validation_average_precision": validation_ap[family],
                "selected_family": family == selected,
                "selected_for_delivery": family == selected and kind == "calibrated",
                "selected_parameters": json.dumps(summary["selected_parameters"], sort_keys=True),
                "pr_auc_convention": "average_precision; non-interpolated stepwise AP",
                "confidence": bootstrap["confidence"],
                "bootstrap_replicates": bootstrap["n_replicates"],
                "bootstrap_entities": bootstrap["n_entities"],
                "daily_confidence_intervals_available": False,
            }
            for metric, interval in bootstrap["intervals"].items():
                if not np.isclose(metrics[metric], interval["estimate"], rtol=0, atol=1e-12):
                    raise ValueError(f"Summary and bootstrap point estimate differ: {family}/{kind}/{metric}")
                row[f"{metric}_ci_lower"] = interval["lower"]
                row[f"{metric}_ci_upper"] = interval["upper"]
                row[f"{metric}_ci_valid_replicates"] = interval["valid_replicates"]
            rows.append(row)
    comparison = pd.DataFrame(rows)
    calibrated = comparison.loc[comparison["probability_kind"].eq("calibrated")]
    descriptive_leader = calibrated.loc[calibrated["average_precision"].idxmax()]
    selection = {
        "schema_version": 1,
        "selected_model": selected,
        "probability_kind": "calibrated",
        "selection_metric": "2023 validation average_precision (non-interpolated AP)",
        "selected_validation_average_precision": validation_ap[selected],
        "validation_average_precision_by_family": validation_ap,
        "fixed_tie_order": list(FAMILIES),
        "selected_parameters": summaries[selected]["selected_parameters"],
        "estimator_fit_period": summaries[selected]["estimator_fit_period"],
        "calibrator_fit_period": summaries[selected]["calibration"]["fit_period"],
        "refitted_after_selection": False,
        "test_used_for_model_selection": False,
        "descriptive_test_ap_leader": {
            "model": descriptive_leader["model"],
            "average_precision": float(descriptive_leader["average_precision"]),
            "role": "Descriptive held-out comparison only; does not replace the validation-selected model",
        },
        "requested_additional_families": {
            "xgboost": "Not implemented in the remote baseline; no new model was added in this finishing scope",
            "catboost": "Not implemented in the remote baseline; no new model was added in this finishing scope",
        },
        "source_metrics": {
            family: {
                "path": f"reports/metrics/{family}_metrics.json",
                "sha256": sha256_file(metrics_dir / f"{family}_metrics.json"),
            } for family in FAMILIES
        },
        "reported_population": "2025 observed Canvass inspections, conditional on historical scheduling",
        "pr_auc_definition": "AP uses non-interpolated stepwise area; it is not trapezoidal integration of the plotted PR curve",
        "daily_capacity_definition": "ceil(fraction * n_day); selected_n and selected_fraction report effective capacity",
        "confidence_interval_scope": "Annual pooled fixed-model establishment bootstrap; no daily-queue intervals",
    }
    paired = paired_differences(
        predictions_dir, seed=int(reference["config"]["random_seed"]),
        replicates=int(reference["config"]["bootstrap_replicates"]),
    )
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    for name, payload in (("model_selection.json", selection), ("paired_model_differences.json", paired)):
        (metrics_dir / name).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"Saved model_comparison.csv: {len(comparison)} rows, {len(comparison.columns)} columns")
    print(f"Frozen selection: {selected} calibrated; 2023 AP={validation_ap[selected]:.6f}")
    print(f"2025 descriptive AP leader: {descriptive_leader['model']}; selection unchanged")
    for metric, interval in paired["intervals"].items():
        print(f"LGBM-LR {metric}: {interval['estimate']:.6f}; "
              f"95% paired CI [{interval['lower']:.6f}, {interval['upper']:.6f}]")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "reports/metrics")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / ".local/predictions")
    args = parser.parse_args()
    compare(args.metrics_dir.resolve(), args.predictions_dir.resolve())


if __name__ == "__main__":
    main()
