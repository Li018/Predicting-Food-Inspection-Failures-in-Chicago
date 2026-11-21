#!/usr/bin/env python3
"""Supplement frozen LR results with business groups, ORs and allocation audits.

Requires the existing local model, prediction and analytic SHAP artifacts. Never
calls fit or rebuilds the main model. Only new summary CSV/JSON files are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from chicago_food.audit import (
    fixed_window_reordering,
    grouped_shap,
    odds_ratio_tables,
    subgroup_audit,
    verify_explanation,
)
from chicago_food.data import sha256_file
from chicago_food.feature_groups import FEATURE_GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _verify_sources(root: Path, bundle: dict, explanation: dict) -> dict[str, str]:
    """Bind the read-only audit to the original frozen study and model inputs."""
    provenance = bundle["provenance"]
    if explanation["provenance"] != provenance:
        raise ValueError("SHAP provenance differs from the frozen LR provenance")
    expected = {
        ".local/features.parquet": provenance["feature_sha256"],
        "config/topic1.json": provenance["config_sha256"],
        "scripts/run_model.py": provenance["runner_sha256"],
        **{f"src/chicago_food/{name}": digest for name, digest in {
            **provenance["feature_code_sha256"], **provenance["model_code_sha256"],
        }.items()},
        **{f"data/snapshots/{name}": digest for name, digest in provenance["source_sha256"].items()},
    }
    manifest = json.loads((root / "data/snapshots/manifest.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        if manifest["files"][name]["sha256"] != digest:
            raise ValueError(f"Frozen study manifest differs from fitted model: {name}")
    for name, digest in expected.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Frozen input differs from original model: {name}")
    for name in (
        ".local/models/logistic_regression.joblib",
        ".local/shap/logistic_regression.joblib",
        ".local/predictions/logistic_regression.parquet",
        "data/snapshots/manifest.json",
    ):
        expected[name] = sha256_file(root / name)
    return expected


def run_audit(root: Path = ROOT) -> dict:
    metrics = root / "reports/metrics"
    bundle = joblib.load(root / ".local/models/logistic_regression.joblib")
    explanation = joblib.load(root / ".local/shap/logistic_regression.joblib")
    if json.loads((metrics / "model_selection.json").read_text())["selected_model"] != "logistic_regression":
        raise ValueError("This audit requires the original selected LR")
    inputs = _verify_sources(root, bundle, explanation)
    saved = pd.read_parquet(root / ".local/predictions/logistic_regression.parquet")
    aligned, verification = verify_explanation(explanation, saved, bundle)
    frame, values = explanation["test_frame"].reset_index(drop=True), explanation["shap_values"]
    grouped, full_table, full_summary = grouped_shap(values, explanation["feature_names"], FEATURE_GROUPS)
    _, sample_table, sample_summary = grouped_shap(
        values[explanation["sample_indices"]], explanation["feature_names"], FEATURE_GROUPS,
    )
    verification["grouped_probability_max_absolute_error"] = float(np.max(np.abs(
        expit(explanation["base_value"] + grouped.sum(axis=1)) - aligned["calibrated_probability"].to_numpy()
    )))
    if verification["grouped_probability_max_absolute_error"] > 1e-9:
        raise ValueError("Business-group SHAP does not reproduce the frozen probabilities")
    sample_table.insert(0, "scope", "fixed_explanation_sample")
    full_table.insert(0, "scope", "full_test")
    importance = pd.concat([sample_table, full_table], ignore_index=True)
    importance.insert(2, "label", importance["group"].map(GROUP_LABELS))
    provenance = {"model": "logistic_regression", "refitted": False, "input_sha256": inputs}
    shap_summary = {
        "schema_version": 1, **provenance,
        "method": "Signed sum of existing analytic calibrated LR-logit SHAP within each business group",
        "units": "Calibrated log-odds", "feature_groups": FEATURE_GROUPS,
        "fixed_explanation_sample": sample_summary, "full_test": full_summary,
        "verification": verification,
        "limitations": [
            "Business grouping makes cancellation explicit; it does not uniquely resolve correlated attribution.",
            "Mean absolute group contributions are descriptive, not causal effects or ablation gains.",
            "Sample scope matches the original fixed 1,500-row explanation; full_test uses all test rows.",
        ],
    }
    numeric, encoded = odds_ratio_tables(bundle, FEATURE_GROUPS)
    subgroup, calibration_bins, subgroup_summary = subgroup_audit(frame, aligned["calibrated_probability"])
    subgroup_summary.update(schema_version=1, **provenance)
    windows, reordering = fixed_window_reordering(
        frame["inspection_date"], aligned["calibrated_probability"], frame["target"],
    )
    reordering.update(schema_version=1, **provenance)
    or_summary = {
        "schema_version": 1, **provenance,
        "numeric_count": len(numeric), "transformed_count": len(encoded),
        "calibrator_slope": float(bundle["calibrator"].slope_),
        "numeric_unit": "One training SD of the numeric coordinate after training-median imputation",
        "missing_indicator_unit": "0 to 1 after undoing the fitted scaler",
        "categorical_unit": "Single encoded coordinate 0 to 1; full one-hot has no omitted reference",
        "interpretation": "Conditional associations holding other coordinates fixed; not risk ratios or causal effects",
        "uncertainty": "No coefficient confidence intervals are estimated for this regularized frozen model",
        "historical_context": {
            "purple_cluster_coefficient": 1.555,
            "purple_cluster_odds_multiplier": float(np.exp(1.555)),
            "source": "https://arxiv.org/abs/1910.04906",
            "scope": "2019 Table 3 audit of the old official model; dummy-coordinate OR, not a probability multiplier or Purple-vs-Brown contrast",
            "comparison": "Modern numeric one-SD ORs use different units, outcome and period; compare model inputs rather than effect-size rankings",
        },
    }
    # No original study metric or image is overwritten by this supplementary run.
    csv_outputs = {
        "grouped_shap_importance.csv": importance,
        "lr_standardized_odds_ratios.csv": numeric,
        "lr_transformed_odds_ratios.csv": encoded,
        "subgroup_audit.csv": subgroup,
        "subgroup_audit_calibration_bins.csv": calibration_bins,
        "days_earlier_windows.csv": windows,
    }
    json_outputs = {
        "grouped_shap_summary.json": shap_summary,
        "lr_odds_ratio_summary.json": or_summary,
        "subgroup_audit_summary.json": subgroup_summary,
        "days_earlier_summary.json": reordering,
    }
    for name, table in csv_outputs.items():
        table.to_csv(metrics / name, index=False, float_format="%.12g")
    for name, payload in json_outputs.items():
        _write_json(metrics / name, payload)
    if any(sha256_file(root / name) != digest for name, digest in inputs.items()):
        raise RuntimeError("A protected input changed during the supplementary audit")
    result = {"test_n": len(frame), "group_count": len(FEATURE_GROUPS),
              "subgroup_rows": len(subgroup), "days_earlier": reordering["pooled_days_earlier"],
              "probability_reconstruction_error": verification["grouped_probability_max_absolute_error"],
              "protected_inputs_unchanged": True, "outputs": list(csv_outputs) + list(json_outputs)}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run_audit()
