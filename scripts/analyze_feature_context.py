"""Aggregate correlation and attribution context without fitting another model."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chicago_food.data import sha256_file
from chicago_food.models import temporal_splits
from run_model import load_verified_features


CONTEXT_FEATURES = [
    "observed_history_years", "prior_scored_days", "prior_fail_days",
    "historical_failure_rate", "prior_routine_days", "routine_failure_rate",
    "license_term_elapsed_days", "days_since_license_issue",
    "license_observed_tenure_years", "days_since_last_inspection",
]


def feature_group(name: str) -> str:
    if name.startswith("license_") or name in {
        "days_since_license_issue", "days_to_license_term_end",
    }:
        return "license_history"
    if name.startswith("month_"):
        return "calendar"
    if name in {"prior_facility_type", "prior_risk"}:
        return "prior_facility_and_risk"
    if name.startswith("days_since_"):
        return "inspection_recency"
    return "inspection_history"


def main() -> None:
    config = json.loads((ROOT / "config/topic1.json").read_text())
    features, metadata = load_verified_features(ROOT / ".local/features.parquet", config)
    train = temporal_splits(features, config)["train"]
    shap_path = ROOT / ".local/shap/logistic_regression.joblib"
    explanation = joblib.load(shap_path)
    if explanation["provenance"]["feature_sha256"] != metadata["sha256"]:
        raise ValueError("SHAP and feature context use different source features")

    correlations = []
    for first, second in itertools.combinations(CONTEXT_FEATURES, 2):
        pair = train[[first, second]].dropna()
        correlations.append({
            "feature_a": first, "feature_b": second, "n_complete_train": len(pair),
            "pearson": pair[first].corr(pair[second], method="pearson"),
            "spearman": pair[first].corr(pair[second], method="spearman"),
        })

    names = explanation["feature_names"]
    phi = np.asarray(explanation["shap_values"])
    groups = {group: [i for i, name in enumerate(names) if feature_group(name) == group]
              for group in sorted({feature_group(name) for name in names})}
    if sum(map(len, groups.values())) != len(names):
        raise ValueError("Every original feature must belong to exactly one context group")
    group_rows = []
    for scope, indices in [("global_plot_sample", explanation["sample_indices"]),
                           ("all_2025_targets", np.arange(len(phi)))]:
        subset = phi[indices]
        reconstructed = sum(subset[:, columns].sum(axis=1) for columns in groups.values())
        np.testing.assert_allclose(reconstructed, subset.sum(axis=1), atol=1e-12, rtol=0)
        for group, columns in groups.items():
            values = subset[:, columns]
            signed_sum = values.sum(axis=1)
            absolute_parts = float(np.abs(values).mean(axis=0).sum())
            absolute_group = float(np.abs(signed_sum).mean())
            group_rows.append({
                "scope": scope, "n": len(indices), "feature_group": group,
                "feature_count": len(columns), "mean_signed_group_contribution": float(signed_sum.mean()),
                "mean_absolute_group_contribution": absolute_group,
                "sum_individual_mean_absolute_contributions": absolute_parts,
                "within_group_cancellation_fraction": 1 - absolute_group / absolute_parts if absolute_parts else 0.0,
            })

    output = ROOT / "reports/metrics"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correlations).to_csv(output / "feature_correlations.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "shap_feature_groups.csv", index=False)
    summary = {
        "correlation_reference": {"split": "train", "n": len(train), "missingness": "Pairwise complete observations"},
        "groups": {group: [names[i] for i in columns] for group, columns in groups.items()},
        "units": "LR-logit SHAP scaled onto the affine calibration-logit scale",
        "cancellation_definition": "1 - mean(abs(sum(phi_j within group))) / sum(mean(abs(phi_j)))",
        "interpretation": [
            "Summing signed contributions before taking absolute values exposes offsetting terms.",
            "Correlations and cancellation describe the fitted model and covariate distribution; they are not a causal explanation of coefficient signs.",
            "These are attribution summaries, not feature-ablation estimates of predictive value.",
            "In particular, a negative conditional historical-failure-rate coefficient must not be interpreted as a protective effect of failures.",
        ],
        "feature_sha256": metadata["sha256"], "local_shap_sha256": sha256_file(shap_path),
        "correlation_pairs": len(correlations), "group_summary_rows": len(group_rows),
    }
    (output / "feature_context.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(correlations)} training correlation pairs and {len(group_rows)} SHAP group summaries.")
    for row in group_rows:
        if row["scope"] == "global_plot_sample":
            print(f"{row['feature_group']}: mean absolute signed-group SHAP={row['mean_absolute_group_contribution']:.4f}; cancellation={row['within_group_cancellation_fraction']:.1%}")


if __name__ == "__main__":
    main()
