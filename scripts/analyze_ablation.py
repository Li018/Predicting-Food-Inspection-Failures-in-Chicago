#!/usr/bin/env python3
"""Run additive LR feature ablations without retraining the frozen main model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from chicago_food.ablation import run_ablation
from chicago_food.data import sha256_file
from chicago_food.feature_groups import FEATURE_GROUPS, GROUP_LABELS
from run_model import load_verified_features

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"


def main() -> None:
    config_path = ROOT / "config/topic1.json"
    config = json.loads(config_path.read_text())
    frame, metadata = load_verified_features(ROOT / ".local/features.parquet", config)
    model_path = ROOT / ".local/models/logistic_regression.joblib"
    original_model_hash = sha256_file(model_path)
    reference = joblib.load(model_path)
    provenance = reference["provenance"]
    if provenance["feature_sha256"] != metadata["sha256"] or reference["config"] != config:
        raise ValueError("Frozen model and verified feature/config provenance differ")
    for name, digest in {**provenance["feature_code_sha256"], **provenance["model_code_sha256"]}.items():
        if sha256_file(ROOT / "src/chicago_food" / name) != digest:
            raise ValueError(f"Frozen model core implementation changed: {name}")
    with threadpool_limits(limits=int(config["n_jobs"])):
        results, search, selections = run_ablation(frame, config, reference=reference)
    if sha256_file(model_path) != original_model_hash:
        raise ValueError("The frozen model artifact changed during ablation")
    summary = {
        "schema_version": 1,
        "protocol": {
            "estimator_training": "2019-2022 only",
            "parameter_selection": "2023 raw non-interpolated AP; original four LR candidates",
            "probability_calibration": "2024 only; separate positive-slope sigmoid per ablation",
            "evaluation": "2025 only; no test-driven feature or hyperparameter selection",
            "validation_probability_kind": "raw only; the later 2024 calibrator is not applied to 2023",
            "nested_feature_counts": [13, 27, 37],
            "reference_feature_count": 39,
            "reference_policy": "Load the frozen 39-field primary LR, never fit or overwrite it",
            "facility_risk_policy": "Excluded from all three nested ablations; present only in frozen reference",
            "delta_policy": "Current minus previous nested stage, same split/probability kind; lower Brier is better",
            "uncertainty": "Descriptive point estimates; no significance or causal license-effect claim",
        },
        "feature_groups": {key: {"label": GROUP_LABELS[key], "features": list(features)}
                           for key, features in FEATURE_GROUPS.items()},
        "selections": selections,
        "test_calibrated": json.loads(results.loc[
            results["split"].eq("test") & results["probability_kind"].eq("calibrated")
        ].to_json(orient="records", double_precision=15)),
        "provenance": {
            "source_sha256": metadata["source_sha256"], "feature_sha256": metadata["sha256"],
            "frozen_primary_model_sha256": original_model_hash,
            "config_sha256": sha256_file(config_path),
            "core_code_sha256": {**provenance["feature_code_sha256"], **provenance["model_code_sha256"]},
        },
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS / "feature_ablation.csv", index=False)
    search.to_csv(METRICS / "feature_ablation_validation.csv", index=False)
    (METRICS / "feature_ablation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    display = results.loc[results["split"].eq("test") & results["probability_kind"].eq("calibrated"),
                          ["stage", "feature_count", "average_precision", "roc_auc", "brier_score"]]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved three aggregate files; frozen primary model checksum unchanged.")


if __name__ == "__main__":
    main()
