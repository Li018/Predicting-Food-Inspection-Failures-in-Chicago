#!/usr/bin/env python3
"""Run one frozen model family; publish aggregates and keep row data local.

Example: PYTHONPATH=src python scripts/run_model.py --family logistic_regression
The estimator is fitted only on 2019--2022, selected by 2023 AP, and calibrated
on 2024. Each family is saved separately so milestones can be committed promptly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from chicago_food.data import sha256_file, verify_snapshot
from chicago_food.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from chicago_food.metrics import cluster_bootstrap_ci, evaluate_probabilities
from chicago_food.models import (
    LIGHTGBM_CANDIDATES,
    LOGISTIC_CANDIDATES,
    MonotoneSigmoidCalibrator,
    fit_pipeline,
    make_pipeline,
    temporal_splits,
)

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {
    "prevalence": [{}],
    "logistic_regression": LOGISTIC_CANDIDATES,
    "lightgbm": LIGHTGBM_CANDIDATES,
}


def load_verified_features(path: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    """Require the Step 2 cache to match data, feature code, lag and date windows."""
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    manifest = verify_snapshot(ROOT / "data/snapshots")
    source_hashes = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    code_hashes = {
        name: sha256_file(ROOT / "src/chicago_food" / name)
        for name in ("data.py", "features.py")
    }
    windows = {
        f"{split}_{bound}": config[f"{split}_{bound}"]
        for split in ("train", "validation", "calibration", "test")
        for bound in ("start", "end")
    }
    checks = {
        "feature_file": sha256_file(path) == metadata["sha256"],
        "source_data": source_hashes == metadata["source_sha256"],
        "feature_code": code_hashes == metadata["code_sha256"],
        "observation_lag": config["observation_lag_days"] == metadata["lag_days"],
        "temporal_windows": windows == metadata["temporal_config"],
    }
    failed = [name for name, matches in checks.items() if not matches]
    if failed:
        raise ValueError(f"Rebuild Step 2 feature cache; mismatched: {', '.join(failed)}")
    frame = pd.read_parquet(path)
    if frame.shape != (metadata["rows"], metadata["columns"]):
        raise ValueError("Feature cache shape does not match its metadata")
    return frame, metadata


def run_family(family: str, config_path: Path, feature_path: Path, output_dir: Path) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    frame, feature_metadata = load_verified_features(feature_path, config)
    splits = temporal_splits(frame, config)
    numeric, categorical = list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)
    seed = int(config["random_seed"])
    fractions = tuple(config["top_fractions"])
    train, validation = splits["train"], splits["validation"]
    selected, selected_index, selected_parameters, best_ap = None, None, None, -np.inf
    candidates, warning_rows = [], []
    fit_started = time.perf_counter()
    for index, parameters in enumerate(GRIDS[family]):
        candidate_started = time.perf_counter()
        pipeline = make_pipeline(
            family, numeric, categorical, seed=seed,
            n_jobs=int(config["n_jobs"]), parameters=parameters,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit_pipeline(pipeline, family, train, train["target"], numeric, categorical)
            probabilities = pipeline.predict_proba(validation)[:, 1]
        counts = Counter((w.category.__name__, str(w.message)) for w in captured)
        for (category, message), count in counts.items():
            warning_rows.append({
                "candidate_index": index, "category": category, "count": count,
                "message": message[:600],
            })
        metrics = evaluate_probabilities(
            validation["target"], probabilities, top_fractions=fractions
        )
        candidates.append({
            "model": family, "candidate_index": index,
            "parameters": json.dumps(parameters, sort_keys=True),
            "fit_and_validation_seconds": time.perf_counter() - candidate_started,
            "warning_count": len(captured), **metrics,
        })
        # Fixed candidate order resolves ties; no calibration/test label is used here.
        if metrics["average_precision"] > best_ap:
            selected, selected_index, selected_parameters = pipeline, index, parameters
            best_ap = metrics["average_precision"]
        print(f"{family}: candidate {index + 1}/{len(GRIDS[family])}; "
              f"validation AP={metrics['average_precision']:.6f}; warnings={len(captured)}",
              flush=True)
    search_seconds = time.perf_counter() - fit_started
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame["selected"] = candidate_frame["candidate_index"].eq(selected_index)
    calibration_started = time.perf_counter()
    calibration = splits["calibration"]
    calibrator = MonotoneSigmoidCalibrator().fit(
        selected.predict_proba(calibration)[:, 1], calibration["target"]
    )
    calibration_seconds = time.perf_counter() - calibration_started
    metric_rows, prediction_frames, split_summary = [], [], []
    for split, subset in splits.items():
        raw = selected.predict_proba(subset)[:, 1]
        variants = {"raw": raw}
        # Calibration fitted in 2024 is not retroactively exposed in 2019--2023.
        calibrated = np.full(len(subset), np.nan)
        if split in {"calibration", "test"}:
            calibrated = calibrator.predict(raw)
            variants["calibrated"] = calibrated
        for kind, probabilities in variants.items():
            metric_rows.append({
                "split": split, "model": family, "probability_kind": kind,
                "estimator_fitted_on_this_split": split == "train",
                "calibrator_fitted_on_this_split": split == "calibration" and kind == "calibrated",
                **evaluate_probabilities(
                    subset["target"], probabilities, dates=subset["inspection_date"],
                    top_fractions=fractions,
                ),
            })
        predictions = subset[["inspection_id", "inspection_date", "entity_id", "target"]].copy()
        predictions["split"], predictions["model"] = split, family
        predictions["raw_probability"] = raw
        predictions["calibrated_probability"] = calibrated
        prediction_frames.append(predictions)
        split_summary.append({
            "split": split, "n": len(subset), "entities": int(subset["entity_id"].nunique()),
            "failures": int(subset["target"].sum()), "prevalence": float(subset["target"].mean()),
            "first_date": str(subset["inspection_date"].min().date()),
            "last_date": str(subset["inspection_date"].max().date()),
        })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test_predictions = predictions.loc[predictions["split"].eq("test")]
    test_metrics = {row["probability_kind"]: row for row in metric_rows if row["split"] == "test"}
    ranking_fields = [
        name for name in test_metrics["raw"]
        if name in {"roc_auc", "average_precision"}
        or (name.startswith(("top_", "daily_top_")) and name.endswith(("precision", "recall", "lift")))
    ]
    maximum_ranking_difference = max(
        abs(test_metrics["raw"][name] - test_metrics["calibrated"][name])
        for name in ranking_fields
    )
    bootstrap_started = time.perf_counter()
    bootstrap, interval_rows = {}, []
    for kind in ("raw", "calibrated"):
        result = cluster_bootstrap_ci(
            test_predictions["target"], test_predictions[f"{kind}_probability"],
            test_predictions["entity_id"], n_replicates=int(config["bootstrap_replicates"]),
            seed=seed, confidence=0.95, top_fractions=fractions,
        )
        bootstrap[kind] = result
        for metric, interval in result["intervals"].items():
            interval_rows.append({
                "model": family, "split": "test", "probability_kind": kind,
                "metric": metric, "confidence": result["confidence"],
                "n_replicates": result["n_replicates"], "n_entities": result["n_entities"],
                **interval,
            })
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    selection = {
        "metric": "2023 validation average_precision; non-interpolated AP",
        "candidate_count": len(candidates), "candidate_index": int(selected_index),
        "validation_average_precision": best_ap,
        "tie_policy": "First candidate in the prespecified grid",
        "refitted_after_selection": False,
    }
    provenance = {
        "feature_cache": str(feature_path.relative_to(ROOT)),
        "feature_sha256": feature_metadata["sha256"],
        "source_sha256": feature_metadata["source_sha256"],
        "feature_code_sha256": feature_metadata["code_sha256"],
        "model_code_sha256": {
            name: sha256_file(ROOT / "src/chicago_food" / name)
            for name in ("models.py", "metrics.py")
        },
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    versions = {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib", "pyarrow"
    )}
    versions["python"] = platform.python_version()
    summary = {
        "schema_version": 1, "model": family, "started_at_utc": started_at,
        "config": config, "provenance": provenance, "versions": versions,
        "numeric_features": numeric, "categorical_features": categorical,
        "preprocessing_fit_period": [config["train_start"], config["train_end"]],
        "estimator_fit_period": [config["train_start"], config["train_end"]],
        "selected_parameters": selected_parameters, "selection": selection,
        "calibration": {
            "method": "Positive-slope sigmoid of raw-probability log-odds",
            "fit_period": [config["calibration_start"], config["calibration_end"]],
            "n": calibrator.n_samples_, "slope": calibrator.slope_,
            "intercept": calibrator.intercept_, "clip": calibrator.clip,
            "optimization_message": calibrator.optimization_message_,
            "test_ranking_max_absolute_difference": maximum_ranking_difference,
            "test_ranking_preserved": bool(maximum_ranking_difference < 1e-12),
        },
        "split_summary": split_summary, "metrics": metric_rows, "bootstrap": bootstrap,
        "timing_seconds": {
            "candidate_search": search_seconds, "calibration_fit": calibration_seconds,
            "bootstrap": bootstrap_seconds, "total_before_saving": time.perf_counter() - started,
        },
        "warnings": warning_rows,
        "metric_definitions": {
            "pr_auc": "average_precision (non-interpolated stepwise AP), not trapezoidal PR area",
            "top_k": "Pooled annual ranking with expected allocation at boundary-score ties",
            "daily_top_k": "Separate observed-inspection queues; ceil(fraction*n_day) capacity",
            "confidence_intervals": "95% establishment-cluster percentile bootstrap, fixed model",
            "calibration_split": "Calibrated 2024 metrics are in-sample to the calibrator",
            "population": "Observed Canvass inspections, conditional on historical scheduling",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir, predictions_dir = ROOT / ".local/models", ROOT / ".local/predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(output_dir / f"{family}_validation.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(output_dir / f"{family}_confidence_intervals.csv", index=False)
    (output_dir / f"{family}_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    predictions.to_parquet(predictions_dir / f"{family}.parquet", index=False)
    joblib.dump({
        "model": family, "pipeline": selected, "calibrator": calibrator,
        "config": config, "numeric_features": numeric, "categorical_features": categorical,
        "selected_parameters": selected_parameters, "selection": selection,
        "provenance": provenance, "summary": summary,
    }, models_dir / f"{family}.joblib", compress=3)
    calibrated = test_metrics["calibrated"]
    print(f"{family}: test n={calibrated['n']}; ROC-AUC={calibrated['roc_auc']:.6f}; "
          f"AP={calibrated['average_precision']:.6f}; Brier={calibrated['brier_score']:.6f}; "
          f"LogLoss={calibrated['log_loss']:.6f}", flush=True)
    print(f"{family}: saved aggregate metrics; local model/predictions retained; "
          f"ranking_preserved={summary['calibration']['test_ranking_preserved']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(GRIDS), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic1.json")
    parser.add_argument("--features", type=Path, default=ROOT / ".local/features.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/metrics")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    with threadpool_limits(limits=int(config["n_jobs"])):
        run_family(args.family, args.config.resolve(), args.features.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
