"""Read-only interpretation and retrospective allocation diagnostics.

These helpers never fit an estimator, choose parameters, or change an operating
threshold. Business-group SHAP preserves signed local additivity. Subgroup
selection uses the same annual pooled capacity for every group. The calendar
reordering is a historical reference, not an executable scheduling policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .metrics import capacity_weights


def _probability_arrays(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be matching nonempty vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return y.astype(int), p


def grouped_shap(
    values, feature_names: Sequence[str], groups: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Sum signed contributions per row before measuring group magnitude."""
    phi = np.asarray(values, dtype=float)
    names = list(feature_names)
    assigned = [name for members in groups.values() for name in members]
    if (len(names) != len(set(names)) or len(assigned) != len(set(assigned))
            or set(names) != set(assigned)):
        raise ValueError("Business groups must partition all original features exactly once")
    if phi.ndim != 2 or phi.shape[1] != len(names) or not len(phi):
        raise ValueError("SHAP rows and original feature columns do not align")
    if not np.isfinite(phi).all():
        raise ValueError("SHAP values must be finite")
    rows, columns = [], []
    for group, members in groups.items():
        block = phi[:, [names.index(name) for name in members]]
        signed = block.sum(axis=1)
        magnitude = float(np.abs(signed).mean())
        unsigned = float(np.abs(block).mean(axis=0).sum())
        columns.append(signed)
        rows.append({
            "group": group, "feature_count": len(members), "n": len(phi),
            "mean_abs_group_shap": magnitude,
            "sum_feature_mean_abs_shap": unsigned,
            "within_group_cancellation_fraction": 1 - magnitude / unsigned if unsigned else 0.0,
            "mean_signed_shap": float(signed.mean()),
        })
    grouped = np.column_stack(columns)
    total_magnitude = float(np.abs(grouped.sum(axis=1)).mean())
    group_magnitudes = float(np.abs(grouped).mean(axis=0).sum())
    summary = {
        "n": len(phi), "feature_count": len(names), "group_count": len(groups),
        "sum_group_mean_abs_shap": group_magnitudes,
        "mean_abs_total_shap": total_magnitude,
        "cross_group_cancellation_fraction": (
            1 - total_magnitude / group_magnitudes if group_magnitudes else 0.0
        ),
        "sum_max_absolute_error": float(np.max(np.abs(grouped.sum(axis=1) - phi.sum(axis=1)))),
    }
    return grouped, pd.DataFrame(rows), summary


def verify_explanation(explanation: dict, saved: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, dict]:
    """Reject stale or misaligned saved SHAP/predictions without refitting."""
    frame = explanation["test_frame"].reset_index(drop=True)
    expected = saved.loc[saved["split"].eq("test")].copy()
    if frame["inspection_id"].duplicated().any() or expected["inspection_id"].duplicated().any():
        raise ValueError("Test prediction identifiers must be unique")
    if set(frame["inspection_id"]) != set(expected["inspection_id"]):
        raise ValueError("SHAP and prediction test identifiers differ")
    expected = expected.set_index("inspection_id").loc[frame["inspection_id"]].reset_index()
    for column in ("inspection_date", "entity_id", "target"):
        if not np.array_equal(frame[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"SHAP and saved predictions disagree on {column}")
    if explanation["feature_names"] != bundle["numeric_features"] + bundle["categorical_features"]:
        raise ValueError("Original SHAP features differ from the frozen model")
    raw = bundle["pipeline"].predict_proba(frame)[:, 1]
    calibrated = bundle["calibrator"].predict(raw)
    reconstructed = explanation["base_value"] + np.asarray(explanation["shap_values"]).sum(axis=1)
    verification = {
        "saved_prediction_max_absolute_error": float(np.max(np.abs(
            calibrated - expected["calibrated_probability"].to_numpy()))),
        "saved_raw_prediction_max_absolute_error": float(np.max(np.abs(
            raw - expected["raw_probability"].to_numpy()))),
        "shap_probability_max_absolute_error": float(np.max(np.abs(expit(reconstructed) - calibrated))),
        "shap_logit_max_absolute_error": float(np.max(np.abs(reconstructed - logit(calibrated)))),
        "stored_shap_probability_max_absolute_error": float(np.max(np.abs(
            explanation["calibrated_probability"] - calibrated))),
    }
    if max(verification.values()) > 1e-9:
        raise ValueError("Frozen model, predictions and SHAP do not agree")
    return expected, verification


def odds_ratio_tables(bundle: dict, groups: Mapping[str, Sequence[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export model coefficients with explicit numeric and encoded units.

    Numeric coefficients use the training SD after median imputation. Missing
    indicators use a 0-to-1 change (undoing their StandardScaler scale). One-hot
    coefficients describe a single encoded coordinate, not a category contrast:
    the fitted encoder retains all categories and has no omitted reference.
    """
    pipeline, calibrator = bundle["pipeline"], bundle["calibrator"]
    preprocessor = pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out().tolist()
    coefficients = pipeline.named_steps["model"].coef_[0]
    scale = preprocessor.named_transformers_["num"].named_steps["scale"].scale_
    lookup = {feature: group for group, features in groups.items() for feature in features}
    numeric_rows, encoded_rows = [], []
    numeric_index = 0
    for name, coefficient in zip(names, coefficients, strict=True):
        transformer, field = name.split("__", 1)
        if transformer == "num":
            current_scale = float(scale[numeric_index])
            numeric_index += 1
            missing = field.startswith("missingindicator_")
            original = field.removeprefix("missingindicator_")
            change = float(coefficient / current_scale) if missing else float(coefficient)
            kind = "missing_indicator" if missing else "numeric"
            unit = "indicator 0 to 1" if missing else "one training SD after median imputation"
            if not missing:
                beta = float(coefficient)
                numeric_rows.append({
                    "feature": original, "business_group": lookup[original],
                    "train_sd": current_scale, "unit": unit,
                    "raw_beta_per_train_sd": beta,
                    "raw_odds_ratio_per_train_sd": float(np.exp(beta)),
                    "calibrated_beta_per_train_sd": float(calibrator.slope_ * beta),
                    "calibrated_odds_ratio_per_train_sd": float(np.exp(calibrator.slope_ * beta)),
                    "raw_beta_per_observed_unit": beta / current_scale,
                })
        elif transformer == "cat":
            matches = [feature for feature in bundle["categorical_features"] if field.startswith(feature + "_")]
            original = max(matches, key=len)
            kind, change = "one_hot_coordinate", float(coefficient)
            unit = "encoded coordinate 0 to 1; no omitted reference category"
        else:
            raise ValueError(f"Unrecognized transformed column {name}")
        encoded_rows.append({
            "transformed_feature": name, "original_feature": original,
            "business_group": lookup[original], "kind": kind, "unit": unit,
            "raw_coefficient_in_fitted_coordinates": float(coefficient),
            "raw_log_odds_change": change, "raw_odds_multiplier": float(np.exp(change)),
            "calibrated_log_odds_change": float(calibrator.slope_ * change),
            "calibrated_odds_multiplier": float(np.exp(calibrator.slope_ * change)),
        })
    return pd.DataFrame(numeric_rows), pd.DataFrame(encoded_rows)


def calibration_error(labels, probabilities, *, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """ECE with common equal-width [0, 1] bins; p=1 belongs to the last bin."""
    y, p = _probability_arrays(labels, probabilities)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    bin_ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for number in range(n_bins):
        selected = bin_ids == number
        if selected.any():
            mean_p, mean_y = float(p[selected].mean()), float(y[selected].mean())
            rows.append({"bin": number, "left": number / n_bins, "right": (number + 1) / n_bins,
                         "n": int(selected.sum()), "mean_probability": mean_p,
                         "observed_rate": mean_y, "absolute_error": abs(mean_p - mean_y)})
    bins = pd.DataFrame(rows)
    ece = float(np.dot(bins["n"], bins["absolute_error"]) / len(y))
    return ece, bins


def subgroup_audit(
    frame: pd.DataFrame, probabilities, *,
    columns: Sequence[str] = ("prior_risk", "prior_facility_type"),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit descriptive operational slices under one global annual allocation."""
    y, p = _probability_arrays(frame["target"], probabilities)
    fraction = 0.1
    capacity = int(np.ceil(len(frame) * fraction))
    selected = capacity_weights(p, capacity)
    working = frame.reset_index(drop=True)
    rows, bin_tables = [], []
    for column in columns:
        values = working[column].astype("string").fillna("Unknown").replace("", "Unknown")
        counts = values.value_counts()
        facility_major = set(counts.loc[~counts.index.isin(["Unknown", "Ambiguous"])].head(5).index)
        for value, indices in values.groupby(values, sort=True).indices.items():
            ids = np.asarray(indices, dtype=int)
            gy, gp, weights = y[ids], p[ids], selected[ids]
            positives, n = int(gy.sum()), len(ids)
            ece, bins = calibration_error(gy, gp)
            bins.insert(0, "value", str(value))
            bins.insert(0, "feature", column)
            bin_tables.append(bins)
            has_both = 0 < positives < n
            rows.append({
                "feature": column, "value": str(value), "n": n, "failures": positives,
                "entities": int(working.iloc[ids]["entity_id"].nunique()) if "entity_id" in working else n,
                "prevalence": float(gy.mean()),
                "average_precision": float(average_precision_score(gy, gp)) if has_both else None,
                "roc_auc": float(roc_auc_score(gy, gp)) if has_both else None,
                "brier_score": float(brier_score_loss(gy, gp)),
                "ece_10_equal_width": ece,
                "mean_probability": float(gp.mean()),
                "calibration_bias": float(gp.mean() - gy.mean()),
                "global_top10_selected_n": float(weights.sum()),
                "global_top10_selected_rate": float(weights.mean()),
                "global_top10_recall": float(weights @ gy / positives) if positives else None,
                "small_sample": n < 100 or positives < 10 or n - positives < 10,
                "major_group": column == "prior_risk" or value in facility_major or value == "Unknown",
            })
    table = pd.DataFrame(rows)
    summary = {
        "n": len(y), "failures": int(y.sum()), "nominal_fraction": fraction,
        "global_capacity": capacity, "effective_selection_fraction": capacity / len(y),
        "policy": "Same annual pooled Top10% weights for all slices; boundary ties averaged",
        "group_fields": list(columns),
        "group_field_timing": "Last visible historical inspection, not the current event's attributes",
        "ece_definition": "10 fixed equal-width probability bins on [0,1]; final right endpoint included",
        "calibration_bias_definition": "Mean predicted probability minus observed Fail rate",
        "ap_single_class": "Undefined (null/blank) when a slice lacks either outcome class",
        "major_facility_selection": "Five largest known historical facility types by test count plus Unknown",
        "limitations": [
            "Descriptive operational slices do not establish demographic fairness or causal bias.",
            "Excluding sanitarian inputs does not remove bias in historical labels or proxies.",
            "Small-sample flags mark n<100, Fail<10, or Non-Fail<10; no subgroup confidence intervals.",
        ],
    }
    return table, pd.concat(bin_tables, ignore_index=True), summary


def fixed_window_reordering(dates, probabilities, labels, *, year: int = 2025) -> tuple[pd.DataFrame, dict]:
    """Adapt the official calendar-slot mean to six fixed two-month windows.

    Score ties receive the expected original date slot and delayed/advanced
    probabilities under uniform random permutation within the tie. This keeps
    daily capacity in expectation and removes source-row ordering effects.
    All Fail events enter the average, including negative (delayed) outcomes.
    """
    y, p = _probability_arrays(labels, probabilities)
    dates = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize()
    if len(dates) != len(y) or dates.isna().any() or not dates.dt.year.eq(year).all():
        raise ValueError("Every score needs a valid actual date in the specified test year")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    advances, delayed, earlier = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    rows, max_capacity_error = [], 0.0
    for month in range(1, 13, 2):
        start = pd.Timestamp(year, month, 1)
        end = start + pd.DateOffset(months=2)
        ids = np.flatnonzero((dates >= start).to_numpy() & (dates < end).to_numpy())
        ordered = ids[np.argsort(-p[ids], kind="stable")]
        slots = np.sort(days[ids])
        boundaries = np.r_[0, np.flatnonzero(np.diff(p[ordered]) != 0) + 1, len(ids)]
        capacity_expected = {}
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            if right == left:
                continue
            tied, block = ordered[left:right], slots[left:right]
            advances[tied] = days[tied] - block.mean()
            delayed[tied] = (len(block) - np.searchsorted(block, days[tied], side="right")) / len(block)
            earlier[tied] = np.searchsorted(block, days[tied], side="left") / len(block)
            values, counts = np.unique(block, return_counts=True)
            for day, count in zip(values, counts, strict=True):
                capacity_expected[day] = capacity_expected.get(day, 0.0) + len(tied) * count / len(block)
        values, counts = np.unique(slots, return_counts=True)
        if len(slots):
            max_capacity_error = max(max_capacity_error, max(
                abs(capacity_expected[day] - count) for day, count in zip(values, counts, strict=True)))
        positive_ids = ids[y[ids] == 1]
        failures = len(positive_ids)
        random_mean = float(days[positive_ids].mean() - slots.mean()) if failures else None
        mean = float(advances[positive_ids].mean()) if failures else None
        rows.append({
            "window_start": start.date().isoformat(),
            "window_end": (end - pd.Timedelta(days=1)).date().isoformat(),
            "n": len(ids), "positive_count": failures,
            "days_earlier": mean, "random_days_earlier": random_mean,
            "improvement_over_random_days": mean - random_mean if failures else None,
            "delayed_positive_fraction": float(delayed[positive_ids].mean()) if failures else None,
            "advanced_positive_fraction": float(earlier[positive_ids].mean()) if failures else None,
        })
    positive = y == 1
    table = pd.DataFrame(rows)
    summary = {
        "year": year, "window_count": 6, "n": len(y), "positive_count": int(y.sum()),
        "pooled_days_earlier": float(advances[positive].mean()) if positive.any() else None,
        "pooled_delayed_positive_fraction": float(delayed[positive].mean()) if positive.any() else None,
        "pooled_advanced_positive_fraction": float(earlier[positive].mean()) if positive.any() else None,
        "random_days_earlier": float(np.nansum(table["random_days_earlier"] * table["positive_count"]) / y.sum()) if y.sum() else None,
        "daily_capacity_max_absolute_error": float(max_capacity_error),
        "calendar_window": "Jan-Feb, Mar-Apr, May-Jun, Jul-Aug, Sep-Oct, Nov-Dec; calendar months",
        "definition": "Mean actual date minus expected rescheduled date over every Fail; calendar days",
        "pooling": "Positive-count weighted mean of six windows; negative advances retained",
        "tie_policy": "Uniform expected date-slot assignment within exact score ties",
        "original_day_score_retrospective": True,
        "official_reference": "https://github.com/Chicago/food-inspections-evaluation/blob/master/CODE/functions/simulated_date_diff_mean.R",
        "audit_reference": "https://arxiv.org/abs/1910.04906",
        "limitations": [
            "2015 used Sep-Oct 2014 and critical violations; this is a 2025 Fail-label adaptation, not a performance contest.",
            "2019 Finding 2: an inspection's outcome need not remain unchanged after moving its date.",
            "Scores use each actual inspection date's information; an earlier assigned date may precede input availability.",
            "The observed inspection pool excludes uninspected candidates and ignores routing, legal and staffing constraints.",
            "This is a retrospective reference, not evidence of realized earlier detection or public-health benefit.",
        ],
    }
    return table, summary
