"""Probability and capacity metrics with explicit tie and bootstrap conventions.

Average precision is sklearn's non-interpolated AP, not trapezoidal PR area.
At a score tie crossing the capacity boundary, every tied observation receives the
same fractional selection weight. This is the expectation over random tie-breaking;
in particular a constant prevalence predictor never acquires spurious ranking lift.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


DEFAULT_FRACTIONS = (0.05, 0.10, 0.20)


def _validate(y_true, probabilities):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Labels and probabilities must be equally sized nonempty 1-D arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must contain only binary 0/1 outcomes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities must be finite and between 0 and 1")
    return y.astype(int), p


def capacity_weights(scores, capacity: int) -> np.ndarray:
    """Allocate exactly ``capacity`` slots, averaging any boundary-score tie.

    Sorting is stable. Fractional boundary weights remove input-order dependence;
    they describe expected yield rather than a realizable fractional inspection.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite 1-D array")
    if not isinstance(capacity, (int, np.integer)) or not 0 <= capacity <= len(scores):
        raise ValueError("Capacity must be an integer between zero and the queue length")
    weights = np.zeros(len(scores), dtype=float)
    if capacity == 0:
        return weights
    order = np.argsort(-scores, kind="stable")
    boundary = scores[order[capacity - 1]]
    above = scores > boundary
    tied = scores == boundary
    weights[above] = 1.0
    weights[tied] = (capacity - int(above.sum())) / int(tied.sum())
    return weights


def _capacity_result(y: np.ndarray, weights: np.ndarray) -> dict:
    selected = float(weights.sum())
    positives = int(y.sum())
    found = float(weights @ y)
    prevalence = float(y.mean())
    precision = found / selected if selected else None
    return {
        "selected_n": int(round(selected)),
        "selected_fraction": selected / len(y),
        "expected_failures_found": found,
        "precision": precision,
        "recall": found / positives if positives else None,
        "lift": precision / prevalence if prevalence else None,
    }


def _fraction_name(fraction: float) -> str:
    return f"{100 * fraction:g}pct"


def evaluate_probabilities(
    y_true,
    probabilities,
    *,
    dates=None,
    top_fractions: Iterable[float] = DEFAULT_FRACTIONS,
) -> dict:
    """Evaluate pooled ranking and optionally a separate queue for every date.

    Daily capacity is ``ceil(fraction * n_that_day)``. Its effective selection rate
    therefore generally exceeds the nominal fraction, especially in small queues.
    Daily precision is pooled across allocated daily slots, not an unweighted
    average of daily precision. Single-class ROC-AUC and undefined recall/lift are
    reported as None, never as a fabricated zero.
    """
    y, p = _validate(y_true, probabilities)
    fractions = tuple(float(f) for f in top_fractions)
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Top fractions must lie in (0, 1]")
    result = {
        "n": int(len(y)),
        "failures": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
    }
    day_indices = None
    if dates is not None:
        parsed = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize()
        if len(parsed) != len(y) or parsed.isna().any():
            raise ValueError("Dates must be present for every observation")
        day_indices = parsed.groupby(parsed, sort=True).indices.values()
        result["inspection_days"] = int(parsed.nunique())
    for fraction in fractions:
        label = _fraction_name(fraction)
        weights = capacity_weights(p, int(np.ceil(fraction * len(y))))
        for key, value in _capacity_result(y, weights).items():
            result[f"top_{label}_{key}"] = value
        if day_indices is not None:
            daily_weights = np.zeros(len(y))
            for indices in day_indices:
                daily_weights[indices] = capacity_weights(
                    p[indices], int(np.ceil(fraction * len(indices)))
                )
            for key, value in _capacity_result(y, daily_weights).items():
                result[f"daily_top_{label}_{key}"] = value
    return result


def cluster_bootstrap_ci(
    y_true,
    probabilities,
    entities,
    *,
    n_replicates: int = 500,
    seed: int = 20261119,
    confidence: float = 0.95,
    top_fractions: Iterable[float] = DEFAULT_FRACTIONS,
) -> dict:
    """Percentile intervals by resampling establishments, retaining all their rows.

    These intervals condition on the fitted model, source snapshot and observed
    inspection policy. They do not account for retraining or calendar drift.
    Daily allocation is intentionally excluded because a bootstrap pseudo-queue
    is not the historical daily candidate queue.
    """
    y, p = _validate(y_true, probabilities)
    groups = pd.Series(entities, dtype="string")
    if len(groups) != len(y) or groups.isna().any():
        raise ValueError("Every row needs a nonmissing entity identifier")
    if n_replicates < 2 or not 0 < confidence < 1:
        raise ValueError("At least two replicates and confidence in (0, 1) are required")
    codes, unique = pd.factorize(groups, sort=True)
    fractions = tuple(top_fractions)
    estimate = evaluate_probabilities(y, p, top_fractions=fractions)
    metrics = [
        "average_precision", "roc_auc", "brier_score", "log_loss",
        *[
            f"top_{_fraction_name(f)}_{metric}"
            for f in fractions
            for metric in ("precision", "recall", "lift")
        ],
    ]
    samples = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(y))
    for _ in range(n_replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        multiplicities = np.bincount(sampled, minlength=len(unique))
        indices = np.repeat(row_indices, multiplicities[codes])
        values = evaluate_probabilities(y[indices], p[indices], top_fractions=fractions)
        for metric in metrics:
            value = values[metric]
            if value is not None and np.isfinite(value):
                samples[metric].append(float(value))
    alpha = (1 - confidence) / 2
    intervals = {}
    for metric in metrics:
        values = samples[metric]
        bounds = np.quantile(values, [alpha, 1 - alpha]) if values else [None, None]
        intervals[metric] = {
            "estimate": estimate[metric],
            "lower": float(bounds[0]) if bounds[0] is not None else None,
            "upper": float(bounds[1]) if bounds[1] is not None else None,
            "valid_replicates": len(values),
        }
    return {
        "method": "establishment-cluster percentile bootstrap",
        "n_entities": int(len(unique)),
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "intervals": intervals,
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    probabilities,
    column: str,
    *,
    target_column: str = "target",
    min_n: int = 1,
    top_fractions: Iterable[float] = DEFAULT_FRACTIONS,
) -> pd.DataFrame:
    """Descriptive subgroup metrics; no subgroup-dependent threshold fitting."""
    p = np.asarray(probabilities)
    if len(p) != len(frame):
        raise ValueError("Probability and frame lengths differ")
    working = frame.reset_index(drop=True)
    groups = working[column].astype("string").fillna("Missing")
    rows = []
    for group, indices in groups.groupby(groups, sort=True).indices.items():
        if len(indices) >= min_n:
            rows.append({
                "subgroup": column,
                "value": str(group),
                **evaluate_probabilities(
                    working.iloc[indices][target_column], p[indices],
                    top_fractions=top_fractions,
                ),
            })
    return pd.DataFrame(rows)
