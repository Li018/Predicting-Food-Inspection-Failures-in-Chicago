"""Frozen temporal model selection and strictly separate probability calibration.

All fitted transforms and estimators see 2019--2022 training rows only. Candidate
selection uses 2023 average precision. Calibration uses 2024; 2025 is only scored.
No estimator is refitted after selection, including during lag sensitivity checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .metrics import cluster_bootstrap_ci, evaluate_probabilities, subgroup_metrics


FORBIDDEN_FEATURES = {
    "target", "results", "result", "violations", "inspection_id", "inspection_date",
    "entity_id", "entity", "license_", "license_number", "license_id", "dba_name",
    "aka_name", "address", "facility_type", "risk", "license_status", "target_year",
}

# Declared before any model fit. The search is deliberately finite and auditable.
LOGISTIC_CANDIDATES = [
    {"C": c, "class_weight": weight}
    for c in (0.1, 1.0)
    for weight in (None, "balanced")
]
LIGHTGBM_CANDIDATES = [
    {**parameters, "class_weight": weight}
    for parameters in (
        {"num_leaves": 15, "n_estimators": 400, "learning_rate": 0.04,
         "min_child_samples": 80, "reg_lambda": 10.0},
        {"num_leaves": 31, "n_estimators": 400, "learning_rate": 0.04,
         "min_child_samples": 80, "reg_lambda": 5.0},
        {"num_leaves": 31, "n_estimators": 700, "learning_rate": 0.03,
         "min_child_samples": 80, "reg_lambda": 10.0},
    )
    for weight in (None, "balanced")
]


def _categorical_strings(frame):
    """Normalize missing categories without learning from held-out observations."""
    return pd.DataFrame(frame).astype("string").fillna("__MISSING__").astype(object)


def _validate_feature_names(numeric_features, categorical_features):
    columns = list(numeric_features) + list(categorical_features)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Use a nonempty, nonoverlapping explicit feature allowlist")
    forbidden = set(columns) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden current-event/identity features: {sorted(forbidden)}")
    return columns


def make_pipeline(
    family: str,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    seed: int,
    n_jobs: int = 2,
    parameters: dict | None = None,
) -> Pipeline:
    """Build an unfitted estimator with an explicit feature boundary."""
    _validate_feature_names(numeric_features, categorical_features)
    parameters = parameters or {}
    if family == "prevalence":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    category_normalizer = FunctionTransformer(
        _categorical_strings, feature_names_out="one-to-one"
    )
    if family == "logistic_regression":
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                     keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("normalize", category_normalizer),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                     sparse_output=True)),
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ])
        estimator = LogisticRegression(
            solver="lbfgs", max_iter=2000, tol=1e-6, random_state=seed, **parameters
        )
    elif family == "lightgbm":
        categorical = Pipeline([
            ("normalize", category_normalizer),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                     dtype=np.float64)),
        ])
        preprocessor = ColumnTransformer([
            ("num", "passthrough", numeric_features),
            ("cat", categorical, categorical_features),
        ], sparse_threshold=0).set_output(transform="pandas")
        estimator = LGBMClassifier(
            objective="binary", random_state=seed, n_jobs=n_jobs,
            deterministic=True, force_col_wise=True, verbosity=-1,
            subsample=1.0, colsample_bytree=1.0, **parameters,
        )
    else:
        raise ValueError(f"Unknown model family {family!r}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def fit_pipeline(pipeline, family, x_train, y_train, numeric_features, categorical_features):
    """Fit on training data, declaring ordinal columns as native categories."""
    fit_parameters = {}
    if family == "lightgbm":
        fit_parameters["model__categorical_feature"] = list(range(
            len(numeric_features), len(numeric_features) + len(categorical_features)
        ))
    return pipeline.fit(x_train, y_train, **fit_parameters)


class MonotoneSigmoidCalibrator(BaseEstimator):
    """Positive-slope sigmoid of raw probability log-odds.

    The lower slope bound retains model order, even if the calibration-year
    relationship is weak. The fixed tiny L2 penalty prevents separated numerical
    fits; neither constraint is tuned on test labels.
    """

    def __init__(self, clip: float = 1e-7, slope_min: float = 1e-6):
        self.clip = clip
        self.slope_min = slope_min

    def fit(self, probabilities, labels):
        probabilities = np.asarray(probabilities, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if probabilities.ndim != 1 or labels.ndim != 1 or len(probabilities) != len(labels):
            raise ValueError("Calibration probabilities and labels must have matching lengths")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Invalid calibration probabilities")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Calibration requires both outcome classes")
        x = logit(np.clip(probabilities, self.clip, 1 - self.clip))

        def objective(parameters):
            slope, intercept = parameters
            z = slope * x + intercept
            error = expit(z) - labels
            loss = np.mean(np.logaddexp(0, z) - labels * z) + 1e-8 * slope**2
            gradient = np.array([np.mean(error * x) + 2e-8 * slope, error.mean()])
            return float(loss), gradient

        initial = [1.0, float(logit(labels.mean()) - x.mean())]
        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            bounds=[(self.slope_min, 100.0), (None, None)],
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Calibration optimization failed: {result.message}")
        self.slope_, self.intercept_ = (float(value) for value in result.x)
        self.n_samples_ = len(labels)
        self.optimization_message_ = str(result.message)
        return self

    def predict(self, probabilities):
        if not hasattr(self, "slope_"):
            raise ValueError("Calibrator has not been fitted")
        p = np.asarray(probabilities, dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probabilities")
        return expit(self.slope_ * logit(np.clip(p, self.clip, 1 - self.clip)) + self.intercept_)

    def predict_proba(self, probabilities):
        p = self.predict(probabilities)
        return np.column_stack([1 - p, p])


def temporal_splits(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Validate chronological windows and assign data without random splitting."""
    dates = pd.to_datetime(frame["inspection_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Inspection dates cannot be missing")
    splits = {}
    previous_end = None
    for name in ("train", "validation", "calibration", "test"):
        start, end = pd.Timestamp(config[f"{name}_start"]), pd.Timestamp(config[f"{name}_end"])
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("Temporal windows must be ordered and nonoverlapping")
        subset = frame.loc[dates.between(start, end)].copy()
        if subset.empty or set(subset["target"].dropna().unique()) != {0, 1}:
            raise ValueError(f"{name} split requires nonempty data with both outcome classes")
        if subset["target"].isna().any():
            raise ValueError("Targets cannot be missing")
        subset = subset.sort_values(["inspection_date", "inspection_id"], kind="stable")
        splits[name] = subset.reset_index(drop=True)
        previous_end = end
    return splits


def _predict_record(frame, probabilities, split, model, probability_kind):
    columns = ["inspection_id", "entity_id", "inspection_date", "target"]
    result = frame[columns].copy()
    result["split"] = split
    result["model"] = model
    result["probability_kind"] = probability_kind
    result["probability"] = probabilities
    return result


def evaluate_fixed_model(
    model: Pipeline,
    calibrator: MonotoneSigmoidCalibrator,
    frame: pd.DataFrame,
    *,
    top_fractions=(0.05, 0.10, 0.20),
) -> dict:
    """Score a sensitivity/subgroup cohort without fitting anything."""
    raw = model.predict_proba(frame)[:, 1]
    calibrated = calibrator.predict(raw)
    return {
        "raw_probabilities": raw,
        "calibrated_probabilities": calibrated,
        "raw": evaluate_probabilities(frame["target"], raw,
                                      dates=frame["inspection_date"],
                                      top_fractions=top_fractions),
        "calibrated": evaluate_probabilities(frame["target"], calibrated,
                                             dates=frame["inspection_date"],
                                             top_fractions=top_fractions),
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    config: dict,
    output_dir="reports/metrics",
    *,
    numeric_features=None,
    categorical_features=None,
) -> dict:
    """Run the prespecified experiment and write machine-readable numeric results.

    Return the fitted estimators for artifact persistence and SHAP. ``comparison``
    includes raw probabilities for each family's validation-best model and a
    calibrated row only for the overall selected model. Its calibration-split
    calibrated metrics are in-sample to the calibrator, explicitly marked as such.
    """
    if numeric_features is None or categorical_features is None:
        from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
        numeric_features = list(NUMERIC_FEATURES)
        categorical_features = list(CATEGORICAL_FEATURES)
    columns = _validate_feature_names(numeric_features, categorical_features)
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    splits = temporal_splits(frame, config)
    train, validation = splits["train"], splits["validation"]
    seed = int(config["random_seed"])
    fractions = tuple(config["top_fractions"])
    models = {}
    search_rows = []
    selected_parameters = {}
    candidate_grids = {
        "prevalence": [{}],
        "logistic_regression": LOGISTIC_CANDIDATES,
        "lightgbm": LIGHTGBM_CANDIDATES,
    }
    for family, candidates in candidate_grids.items():
        best_ap = -np.inf
        for index, parameters in enumerate(candidates):
            model = make_pipeline(
                family, numeric_features, categorical_features, seed=seed,
                n_jobs=int(config.get("n_jobs", 2)), parameters=parameters,
            )
            fit_pipeline(model, family, train, train["target"],
                         numeric_features, categorical_features)
            probabilities = model.predict_proba(validation)[:, 1]
            metrics = evaluate_probabilities(
                validation["target"], probabilities, top_fractions=fractions
            )
            search_rows.append({
                "model": family, "candidate_index": index,
                "parameters": json.dumps(parameters, sort_keys=True), **metrics,
            })
            # Ties retain the first prespecified candidate; no test metric breaks ties.
            if metrics["average_precision"] > best_ap:
                best_ap = metrics["average_precision"]
                models[family] = model
                selected_parameters[family] = parameters
    validation_search = pd.DataFrame(search_rows)
    family_order = list(candidate_grids)
    best_ap_by_family = validation_search.groupby("model")["average_precision"].max()
    selected_name = max(family_order, key=lambda name: best_ap_by_family[name])
    selected = models[selected_name]
    calibration = splits["calibration"]
    calibrator = MonotoneSigmoidCalibrator().fit(
        selected.predict_proba(calibration)[:, 1], calibration["target"]
    )
    comparison_rows, prediction_frames = [], []
    for split, subset in splits.items():
        for name, model in models.items():
            raw = model.predict_proba(subset)[:, 1]
            variants = {"raw": raw}
            # A calibrator learned in 2024 is not available for earlier dates.
            if name == selected_name and split in {"calibration", "test"}:
                variants["calibrated"] = calibrator.predict(raw)
            for kind, probabilities in variants.items():
                comparison_rows.append({
                    "split": split, "model": name, "probability_kind": kind,
                    "estimator_fitted_on_this_split": split == "train",
                    "calibrator_fitted_on_this_split": split == "calibration" and kind == "calibrated",
                    **evaluate_probabilities(
                        subset["target"], probabilities, dates=subset["inspection_date"],
                        top_fractions=fractions,
                    ),
                })
                prediction_frames.append(_predict_record(subset, probabilities, split, name, kind))
    comparison = pd.DataFrame(comparison_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test = splits["test"]
    test_probabilities = calibrator.predict(selected.predict_proba(test)[:, 1])
    bootstrap = cluster_bootstrap_ci(
        test["target"], test_probabilities, test["entity_id"],
        n_replicates=int(config.get("bootstrap_replicates", 500)), seed=seed,
        top_fractions=fractions,
    )
    split_summary = [{
        "split": split, "n": len(subset), "entities": int(subset["entity_id"].nunique()),
        "failures": int(subset["target"].sum()), "prevalence": float(subset["target"].mean()),
        "first_date": str(pd.Timestamp(subset["inspection_date"].min()).date()),
        "last_date": str(pd.Timestamp(subset["inspection_date"].max()).date()),
    } for split, subset in splits.items()]
    subgroup_frames = []
    for column in (
        "cold_start", "no_scored_history", "license_history_available",
        "license_match", "historical_facility_type", "historical_risk",
        "prior_facility_type", "prior_risk", "license_term_status",
    ):
        if column in test:
            result = subgroup_metrics(test, test_probabilities, column, top_fractions=fractions)
            result["small_sample"] = (result["n"] < 100) | (result["failures"] < 10)
            subgroup_frames.append(result)
    subgroups = pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
    summary = {
        "selected_model": selected_name,
        "selection_metric": "validation average_precision (non-interpolated AP)",
        "selected_parameters": selected_parameters,
        "estimator_fit_period": [config["train_start"], config["train_end"]],
        "calibration_fit_period": [config["calibration_start"], config["calibration_end"]],
        "calibration": {"slope": calibrator.slope_, "intercept": calibrator.intercept_,
                        "n": calibrator.n_samples_, "clip": calibrator.clip},
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "split_summary": split_summary,
        "tie_policy": "stable sort; fractional expected allocation at boundary score ties",
        "daily_capacity": "ceil(fraction * daily cohort size); report effective selection fraction",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_search.to_csv(output_dir / "validation_search.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    pd.DataFrame(split_summary).to_csv(output_dir / "split_summary.csv", index=False)
    subgroups.to_csv(output_dir / "subgroups.csv", index=False)
    for filename, payload in (("summary.json", summary), ("bootstrap.json", bootstrap)):
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    return {
        "selected_name": selected_name, "selected_pipeline": selected, "calibrator": calibrator,
        "models": models, "predictions": predictions, "comparison": comparison,
        "validation_search": validation_search, "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features), "split_summary": split_summary,
        "bootstrap": bootstrap, "subgroups": subgroups, "summary": summary, "splits": splits,
    }
