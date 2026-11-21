"""Incremental LR ablation on the existing temporal protocol.

Only ablated estimators are fitted here. The frozen 39-feature primary model is an
optional read-only reference. A 2024 calibrator is never applied to 2023 validation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .feature_groups import FEATURE_GROUPS
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from .models import (
    LOGISTIC_CANDIDATES, MonotoneSigmoidCalibrator, fit_pipeline, make_pipeline,
    temporal_splits,
)

ABLATION_STAGES = (
    ("history_only", "Historical outcomes", ("historical_failure_evidence",)),
    ("history_time", "+ Inspection exposure and time",
     ("historical_failure_evidence", "inspection_exposure_time")),
    ("history_time_license", "+ Business license background",
     ("historical_failure_evidence", "inspection_exposure_time", "license_background")),
)
METRIC_NAMES = ("average_precision", "roc_auc", "brier_score")


def stage_features(groups: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Retain the frozen preprocessing field order within each feature type."""
    wanted = {feature for group in groups for feature in FEATURE_GROUPS[group]}
    numeric = [feature for feature in NUMERIC_FEATURES if feature in wanted]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in wanted]
    return numeric, categorical


def probability_metrics(labels, probabilities) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def add_stage_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare adjacent nested ablations within the same split/probability scale.

    The full model is a separate reference, so it never receives an ablation delta.
    """
    result = frame.copy()
    for metric in METRIC_NAMES:
        result[f"delta_{metric}"] = np.nan
    subset = result.loc[~result["is_frozen_reference"]]
    for _, rows in subset.groupby(["split", "probability_kind"], sort=False):
        ordered = rows.sort_values("stage_order")
        for metric in METRIC_NAMES:
            result.loc[ordered.index, f"delta_{metric}"] = ordered[metric].diff()
    return result


def run_ablation(
    frame: pd.DataFrame,
    config: dict,
    *,
    reference: dict | None = None,
    candidates: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Select each nested feature set on 2023, calibrate on 2024, score 2025."""
    splits = temporal_splits(frame, config)
    train, validation, calibration = (splits[k] for k in ("train", "validation", "calibration"))
    candidates = LOGISTIC_CANDIDATES if candidates is None else candidates
    if not candidates:
        raise ValueError("At least one prespecified LR candidate is required")
    rows, search_rows, selections = [], [], []

    def record(stage, label, order, numeric, categorical, pipeline, calibrator, frozen):
        columns = numeric + categorical
        for split, kind in (("validation", "raw"), ("test", "raw"), ("test", "calibrated")):
            subset = splits[split]
            probabilities = pipeline.predict_proba(subset[columns])[:, 1]
            if kind == "calibrated":
                probabilities = calibrator.predict(probabilities)
            rows.append({
                "stage": stage, "stage_label": label, "stage_order": order,
                "feature_count": len(columns), "split": split, "probability_kind": kind,
                "n": len(subset), "failures": int(subset["target"].sum()),
                "is_frozen_reference": frozen,
                **probability_metrics(subset["target"], probabilities),
            })

    for order, (stage, label, groups) in enumerate(ABLATION_STAGES, 1):
        numeric, categorical = stage_features(groups)
        columns = numeric + categorical
        best_ap, selected = -np.inf, None
        for index, parameters in enumerate(candidates):
            candidate = make_pipeline(
                "logistic_regression", numeric, categorical,
                seed=int(config["random_seed"]), n_jobs=int(config.get("n_jobs", 2)),
                parameters=parameters,
            )
            fit_pipeline(candidate, "logistic_regression", train[columns], train["target"],
                         numeric, categorical)
            metrics = probability_metrics(
                validation["target"], candidate.predict_proba(validation[columns])[:, 1]
            )
            search_rows.append({
                "stage": stage, "candidate_index": index,
                "parameters": json.dumps(parameters, sort_keys=True), **metrics,
            })
            # Prespecified order resolves AP ties, without consulting test labels.
            if metrics["average_precision"] > best_ap:
                selected, best_ap = candidate, metrics["average_precision"]
                selected_index, selected_parameters = index, parameters.copy()
        calibrator = MonotoneSigmoidCalibrator().fit(
            selected.predict_proba(calibration[columns])[:, 1], calibration["target"]
        )
        selections.append({
            "stage": stage, "stage_label": label, "groups": list(groups),
            "feature_count": len(columns), "numeric_features": numeric,
            "categorical_features": categorical, "selected_candidate_index": selected_index,
            "selected_parameters": selected_parameters, "validation_ap": best_ap,
            "calibrator": {"slope": calibrator.slope_, "intercept": calibrator.intercept_,
                           "n": calibrator.n_samples_},
        })
        record(stage, label, order, numeric, categorical, selected, calibrator, False)
    if reference is not None:
        if set(reference["numeric_features"] + reference["categorical_features"]) != set(FEATURE_COLUMNS):
            raise ValueError("The frozen reference must contain all 39 original feature fields")
        record("frozen_primary_39", "Frozen primary LR (includes facility/risk)", 4,
               reference["numeric_features"], reference["categorical_features"],
               reference["pipeline"], reference["calibrator"], True)
    return add_stage_deltas(pd.DataFrame(rows)), pd.DataFrame(search_rows), selections
