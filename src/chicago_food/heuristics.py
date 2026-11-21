"""Simple historical ranking rules under the frozen LR capacity conventions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import sha256_file
from .metrics import capacity_weights
from .slide_validation import presentation_metric, validate_metric_contract


RULE_SPECS = (
    {"model": "prior_last_fail", "label": "Rule A: Last inspection failed",
     "source": "last_result == 'Fail'", "missing_history": "Unknown and every non-Fail result score 0"},
    {"model": "prior_fail_rate", "label": "Rule B: Historical Fail rate",
     "source": "historical_failure_rate", "missing_history": "No scored history scores 0"},
    {"model": "prior_failed_days", "label": "Rule C: Historical Fail count",
     "source": "prior_fail_days", "missing_history": "No history has 0 failed inspection-days"},
)


def rule_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Use only previously visible history, retaining unknown-history cases."""
    rate = frame["historical_failure_rate"].fillna(0).to_numpy(dtype=float)
    count = frame["prior_fail_days"].to_numpy(dtype=float)
    if not np.isfinite(rate).all() or ((rate < 0) | (rate > 1)).any():
        raise ValueError("Historical Fail rates must be in [0, 1] or missing")
    if not np.isfinite(count).all() or (count < 0).any() or not np.equal(count, np.floor(count)).all():
        raise ValueError("Historical Fail days must be nonnegative counts")
    return {
        "prior_last_fail": frame["last_result"].eq("Fail").to_numpy(dtype=float),
        "prior_fail_rate": rate,
        "prior_failed_days": count,
    }


def align_test_data(features: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Match the exact 2025 saved LR test cohort, including labels and dates."""
    test = predictions.loc[predictions["split"].eq("test")].copy()
    visible = features.loc[pd.to_datetime(features["inspection_date"]).dt.year.eq(2025)].copy()
    if test.empty or not pd.to_datetime(test["inspection_date"]).dt.year.eq(2025).all():
        raise ValueError("The saved LR test cohort must be nonempty and entirely in 2025")
    for frame in (test, visible):
        if frame["inspection_id"].duplicated().any():
            raise ValueError("Inspection IDs must be unique within the test cohort")
    if set(test["inspection_id"]) != set(visible["inspection_id"]):
        raise ValueError("Feature and prediction test cohorts differ")
    test = test.sort_values("inspection_id").reset_index(drop=True)
    visible = visible.set_index("inspection_id").loc[test["inspection_id"]].reset_index()
    for name in ("entity_id", "inspection_date", "target"):
        left, right = visible[name], test[name]
        if name == "inspection_date":
            left, right = pd.to_datetime(left), pd.to_datetime(right)
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise ValueError(f"Feature/prediction test alignment differs: {name}")
    probabilities = test["calibrated_probability"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Saved calibrated LR probabilities must be finite and in [0, 1]")
    return visible, probabilities


def ranking_metrics(labels, scores, dates) -> dict:
    """Evaluate arbitrary ranking scores with exact expected yield at score ties.

    Annual capacity is ceil(10% of all test rows). Daily capacity is separately
    rounded up for each actual inspection date. Daily lift uses those same daily
    slots allocated uniformly at random, matching the original LR evaluation.
    """
    y, s = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("Labels and scores must be matching nonempty vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all():
        raise ValueError("Labels must be binary and ranking scores finite")
    days = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize().reset_index(drop=True)
    if len(days) != len(y) or days.isna().any():
        raise ValueError("Every score needs an inspection date")
    annual_k = int(np.ceil(0.1 * len(y)))
    annual = capacity_weights(s, annual_k)
    boundary = float(np.min(s[annual > 0]))
    daily = np.zeros(len(y), dtype=float)
    random = np.zeros(len(y), dtype=float)
    partial_tie_days = 0
    for indices in days.groupby(days, sort=True).indices.values():
        capacity = int(np.ceil(0.1 * len(indices)))
        daily[indices] = capacity_weights(s[indices], capacity)
        random[indices] = capacity / len(indices)
        partial_tie_days += int(((daily[indices] > 0) & (daily[indices] < 1)).any())
    found, random_found = float(daily @ y), float(random @ y)
    lift = found / random_found if random_found else None
    return {
        "n": len(y), "failures": int(y.sum()), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, s)) if y.sum() else 0.0,
        "top_10pct_selected_n": annual_k,
        "top_10pct_expected_failures_found": float(annual @ y),
        "top_10pct_precision": float(annual @ y) / annual_k,
        "daily_top_10pct_selected_n": int(round(daily.sum())),
        "daily_top_10pct_expected_failures_found": found,
        "daily_top_10pct_precision": found / daily.sum(),
        "daily_top_10pct_random_expected_failures_found": random_found,
        "daily_top_10pct_lift_vs_daily_random": lift,
        "daily_top_10pct_lift_gain": lift - 1 if lift is not None else None,
        "unique_score_count": len(np.unique(s)),
        "annual_boundary_score": boundary,
        "annual_boundary_tie_size": int((s == boundary).sum()),
        "annual_boundary_slots": annual_k - int((s > boundary).sum()),
        "daily_partial_boundary_tie_days": partial_tie_days,
    }


def evaluate_rules(frame: pd.DataFrame, lr_probabilities) -> tuple[pd.DataFrame, dict]:
    """Evaluate three fixed rules and the already-scored primary LR."""
    scores = rule_scores(frame)
    rows = [{"model": rule["model"], "label": rule["label"], "kind": "heuristic",
             **ranking_metrics(frame["target"], scores[rule["model"]], frame["inspection_date"])}
            for rule in RULE_SPECS]
    best = max(rows, key=lambda row: row["average_precision"])
    primary = {"model": "logistic_regression", "label": "Logistic regression", "kind": "frozen_reference",
               **ranking_metrics(frame["target"], lr_probabilities, frame["inspection_date"])}
    rows.append(primary)
    summary = {
        "best_heuristic": best["model"], "best_heuristic_label": best["label"],
        "display_selection": "Highest 2025 AP among the three prespecified rules; exact ties retain A/B/C order. This selects a display row, not a replacement deployment model.",
        "lr_minus_best_rule": {
            "average_precision": primary["average_precision"] - best["average_precision"],
            "roc_auc": primary["roc_auc"] - best["roc_auc"],
            "annual_top10_precision": primary["top_10pct_precision"] - best["top_10pct_precision"],
            "daily_top10_expected_failures": primary["daily_top_10pct_expected_failures_found"] - best["daily_top_10pct_expected_failures_found"],
        },
        "history_policy": list(RULE_SPECS),
        "history_rate_missing_count": int(frame["historical_failure_rate"].isna().sum()),
        "rules_are_ranking_scores": True,
        "probability_metrics": "Brier and log loss are omitted for uncalibrated heuristic scores.",
        "tie_policy": "Exact fractional expectation over uniform tie breaking at every annual/daily capacity boundary; all tied rows receive the same remaining-slots / tie-size weight.",
        "annual_capacity": "ceil(0.10 * test cohort size); precision is expected Fail found divided by slots",
        "daily_capacity": "Sum of ceil(0.10 * daily cohort size); daily lift divides expected Fail found by equal-capacity same-day random expected Fail found",
        "roc_auc_ap": "sklearn ROC-AUC and non-interpolated average precision on the original unbounded ranking scores",
    }
    return pd.DataFrame(rows), summary


def add_heuristic_slide_metrics(metrics: dict, root: Path) -> dict:
    """Append rule facts from aggregate outputs, preserving every existing fact.

    Shared by the incremental evaluator and the figure exporter's metric builder.
    This function reads only aggregate files and never draws or modifies figures.
    """
    source = root / "reports/metrics/heuristic_baselines.json"
    if not source.is_file():
        return metrics
    summary = json.loads(source.read_text())
    rows = {record["model"]: record for record in summary["records"]}
    rule_ids = [rule["model"] for rule in RULE_SPECS]
    if set(rows) != {*rule_ids, "logistic_regression"} or summary["best_heuristic"] not in rule_ids:
        raise ValueError("Slide rule metrics require the three rules and frozen LR reference")
    result = deepcopy(metrics)

    def put(key, value, fmt, field, *, scale=1, suffix=""):
        value, display, spec = presentation_metric(value, fmt, scale=scale, suffix=suffix)
        result["values"][key], result["display"][key], result["format_specs"][key] = value, display, spec
        result["source_refs"][key] = f"heuristic_baselines.json:{field}"

    for model in rule_ids:
        for name, field, fmt, scale, suffix in (
            ("auc", "roc_auc", ".3f", 1, ""),
            ("ap", "average_precision", ".3f", 1, ""),
            ("top10_precision", "top_10pct_precision", ".1f", 100, "%"),
            ("daily_top10_failures", "daily_top_10pct_expected_failures_found", ".1f", 1, ""),
            ("daily_top10_lift_gain", "daily_top_10pct_lift_gain", "+.1f", 100, "%"),
        ):
            put(f"heuristics.{model}.{name}", rows[model][field], fmt,
                f"records:{model}/{field}", scale=scale, suffix=suffix)
    put("heuristics.rule_count", len(rule_ids), "d", "history_policy:count")
    put("heuristics.best_rule", summary["best_heuristic"], "text", "best_heuristic")
    put("heuristics.best_label", summary["best_heuristic_label"], "text", "best_heuristic_label")
    for extension in ("csv", "json"):
        relative = f"reports/metrics/heuristic_baselines.{extension}"
        result.setdefault("source_hashes", {})[relative] = sha256_file(root / relative)
    validate_metric_contract(result)
    return result
