#!/usr/bin/env python3
"""Exact grouped interventional SHAP for the frozen calibrated logistic model."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import shap

from chicago_food.data import sha256_file
from chicago_food.models import temporal_splits
from plot_model_performance import COLORS, FIGURES, METRICS, ROOT, TEXT, save_figure, style


def display_name(name: str) -> str:
    names = {
        "historical_failure_rate": "Historical failure rate",
        "days_since_last_inspection": "Days since last inspection",
        "license_observed_tenure_years": "Observed license tenure (years)",
        "historical_facility_type": "Historical facility type",
        "historical_risk": "Historical risk category",
        "last_result": "Last observed inspection result",
        "last_routine_result": "Last observed routine result",
        "license_term_status": "Recorded license term status",
        "recency_weighted_failure_rate": "Recency-weighted failure rate",
        "historical_pre2018_share": "Share of history before July 2018",
        "days_to_license_term_end": "Days to recorded license term end",
    }
    return names.get(name, name.replace("_", " ").capitalize())


def map_transformed_columns(preprocessor, numeric: list, categorical: list) -> tuple[list, list]:
    transformed_names = preprocessor.get_feature_names_out().tolist()
    mapping = []
    for name in transformed_names:
        transformer, output_name = name.split("__", 1)
        if transformer == "num":
            source = output_name.removeprefix("missingindicator_")
            if source not in numeric:
                raise ValueError(f"Unmapped numeric transform {name}")
        elif transformer == "cat":
            matches = [feature for feature in categorical if output_name.startswith(feature + "_")]
            if not matches:
                raise ValueError(f"Unmapped categorical transform {name}")
            source = max(matches, key=len)
        else:
            raise ValueError(f"Unexpected preprocessing output {name}")
        mapping.append(source)
    return transformed_names, mapping


def build_explanation() -> dict:
    selection = json.loads((METRICS / "model_selection.json").read_text())
    if selection["selected_model"] != "logistic_regression":
        raise ValueError("This exact explanation implementation requires the selected linear model")
    model_path = ROOT / ".local/models/logistic_regression.joblib"
    bundle = joblib.load(model_path)
    feature_path = ROOT / ".local/features.parquet"
    if sha256_file(feature_path) != bundle["provenance"]["feature_sha256"]:
        raise ValueError("Explanation feature cache differs from the fitted model's source")
    for name, digest in {**bundle["provenance"]["feature_code_sha256"], **bundle["provenance"]["model_code_sha256"]}.items():
        if sha256_file(ROOT / "src/chicago_food" / name) != digest:
            raise ValueError(f"Core code has changed since fitting: {name}")
    frame = pd.read_parquet(feature_path)
    splits = temporal_splits(frame, bundle["config"])
    train, test = splits["train"], splits["test"]
    numeric, categorical = bundle["numeric_features"], bundle["categorical_features"]
    feature_names = numeric + categorical
    pipeline, calibrator = bundle["pipeline"], bundle["calibrator"]
    preprocessor, estimator = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    transformed_train = preprocessor.transform(train)
    transformed_test = preprocessor.transform(test)
    if hasattr(transformed_train, "toarray"):
        transformed_train, transformed_test = transformed_train.toarray(), transformed_test.toarray()
    reference_mean = np.asarray(transformed_train).mean(axis=0)
    coef, intercept = estimator.coef_[0], float(estimator.intercept_[0])
    raw_base = float(intercept + np.dot(coef, reference_mean))
    raw_phi = (transformed_test - reference_mean) * coef
    # Explicit sufficient statistics preserve the full-training mean. Passing an
    # ordinary background matrix would allow the masker's default subsampling.
    explainer = shap.LinearExplainer((coef, intercept), (reference_mean, None))
    library_phi = np.asarray(explainer.shap_values(transformed_test))
    library_error = float(np.max(np.abs(raw_phi - library_phi)))
    library_base_error = float(abs(raw_base - float(explainer.expected_value)))
    if library_error > 1e-10 or library_base_error > 1e-10:
        raise ValueError("Analytic SHAP differs from shap.LinearExplainer")
    transformed_names, mapping = map_transformed_columns(preprocessor, numeric, categorical)
    if len(mapping) != len(coef):
        raise ValueError("Transformed feature mapping is incomplete")
    grouped_raw = np.column_stack([
        raw_phi[:, np.asarray(mapping) == feature].sum(axis=1) for feature in feature_names
    ])
    raw_probability = pipeline.predict_proba(test)[:, 1]
    clip_count = int(((raw_probability < calibrator.clip) | (raw_probability > 1-calibrator.clip)).sum())
    if clip_count:
        raise ValueError("Calibration clipping is active; affine logit SHAP would not be exact")
    phi = grouped_raw * calibrator.slope_
    base = float(calibrator.slope_ * raw_base + calibrator.intercept_)
    calibrated_probability = calibrator.predict(raw_probability)
    reconstructed_logit = base + phi.sum(axis=1)
    logit_error = float(np.max(np.abs(reconstructed_logit - logit(calibrated_probability))))
    probability_error = float(np.max(np.abs(expit(reconstructed_logit) - calibrated_probability)))
    saved = pd.read_parquet(ROOT / ".local/predictions/logistic_regression.parquet")
    saved = saved.loc[saved["split"].eq("test")].set_index("inspection_id").loc[test["inspection_id"]]
    saved_error = float(np.max(np.abs(calibrated_probability - saved["calibrated_probability"].to_numpy())))
    if max(logit_error, probability_error, saved_error) > 1e-9:
        raise ValueError("SHAP reconstruction or saved prediction parity failed")
    rng = np.random.default_rng(int(bundle["config"]["random_seed"]))
    sample_indices = np.sort(rng.choice(len(test), size=min(int(bundle["config"]["shap_sample_size"]), len(test)), replace=False))
    return {
        "model": "logistic_regression", "feature_names": feature_names,
        "numeric_features": numeric, "categorical_features": categorical,
        "test_frame": test, "shap_values": phi, "base_value": base,
        "raw_base_value": raw_base, "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability, "sample_indices": sample_indices,
        "transformed_names": transformed_names, "transformed_to_original": mapping,
        "reference_mean_transformed": reference_mean, "reference_n": len(train),
        "calibrator_slope": calibrator.slope_, "calibrator_intercept": calibrator.intercept_,
        "numeric_scales": preprocessor.named_transformers_["num"].named_steps["scale"].scale_,
        "transformed_coefficients": coef, "config": bundle["config"],
        "provenance": bundle["provenance"],
        "verification": {
            "shap_library_max_absolute_error": library_error,
            "shap_library_base_absolute_error": library_base_error,
            "calibrated_logit_max_absolute_error": logit_error,
            "calibrated_probability_max_absolute_error": probability_error,
            "saved_probability_max_absolute_error": saved_error,
            "test_calibration_clip_count": clip_count,
            "test_raw_probability_min": float(raw_probability.min()),
            "test_raw_probability_max": float(raw_probability.max()),
        },
    }


def importance_table(explanation: dict) -> pd.DataFrame:
    indices = explanation["sample_indices"]
    sample = explanation["test_frame"].iloc[indices]
    phi = explanation["shap_values"][indices]
    rows = []
    for column, feature in enumerate(explanation["feature_names"]):
        values = sample[feature]
        numeric = feature in explanation["numeric_features"]
        missing = values.isna().to_numpy()
        row = {
            "feature": feature, "display_name": display_name(feature),
            "feature_type": "numeric" if numeric else "categorical",
            "encoded_columns_grouped": explanation["transformed_to_original"].count(feature),
            "sample_n": len(sample), "mean_abs_shap": float(np.abs(phi[:, column]).mean()),
            "mean_shap": float(phi[:, column].mean()),
            "positive_contribution_fraction": float((phi[:, column] > 0).mean()),
            "missing_n": int(missing.sum()),
            "mean_shap_missing": float(phi[missing, column].mean()) if missing.any() else np.nan,
            "mean_shap_observed": float(phi[~missing, column].mean()) if (~missing).any() else np.nan,
        }
        if numeric:
            transform_index = explanation["transformed_names"].index("num__" + feature)
            row["calibrated_logit_slope_per_observed_unit"] = float(
                explanation["calibrator_slope"] * explanation["transformed_coefficients"][transform_index]
                / explanation["numeric_scales"][transform_index]
            )
        else:
            row["calibrated_logit_slope_per_observed_unit"] = np.nan
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result)+1))
    return result


def global_plot(explanation: dict, importance: pd.DataFrame) -> dict:
    indices = explanation["sample_indices"]
    sample, phi = explanation["test_frame"].iloc[indices], explanation["shap_values"][indices]
    top = importance.head(15)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 8.1), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.29, right=0.98, top=0.83, bottom=0.20, wspace=0.16)
    rng = np.random.default_rng(int(explanation["config"]["random_seed"]))
    cmap = plt.get_cmap("coolwarm")
    labels = []
    for position, row in enumerate(top.itertuples(index=False)):
        column = explanation["feature_names"].index(row.feature)
        values = sample[row.feature]
        y = np.full(len(sample), position) + rng.uniform(-0.27, 0.27, len(sample))
        color = np.tile(matplotlib.colors.to_rgba("#9AA4AC"), (len(sample), 1))
        if row.feature_type == "numeric":
            observed = values.notna().to_numpy()
            observed_values = values[observed].to_numpy(dtype=float)
            low, high = np.quantile(observed_values, [0.05, 0.95]) if len(observed_values) else (0, 1)
            scaled = np.full(len(observed_values), 0.5) if high <= low else np.clip((observed_values-low)/(high-low), 0, 1)
            color[observed] = cmap(scaled)
        axes[0].scatter(phi[:, column], y, c=color, s=5, alpha=0.55, linewidths=0, rasterized=True)
        labels.append(textwrap.fill(row.display_name + (" †" if row.feature_type == "categorical" else ""), 33))
    positions = np.arange(len(top))
    axes[0].set(yticks=positions, yticklabels=labels, ylim=(len(top)-0.4, -0.7),
                xlabel="Contribution to calibrated log-odds", title="A · Per-inspection contribution distribution")
    axes[0].axvline(0, color="#61778B", lw=1)
    axes[0].grid(axis="x", alpha=0.65)
    axes[0].tick_params(axis="y", labelsize=9)
    axes[1].barh(positions, top["mean_abs_shap"], color=COLORS["logistic_regression"], height=0.65)
    axes[1].set(yticks=positions, yticklabels=[], ylim=(len(top)-0.4, -0.7),
                xlabel="Mean absolute grouped contribution", title="B · Global importance")
    axes[1].grid(axis="x", alpha=0.65)
    axes[1].set_axisbelow(True)
    cax = fig.add_axes([0.37, 0.105, 0.28, 0.014])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1], labels=["Low numeric value", "High numeric value"])
    cb.ax.tick_params(labelsize=8)
    fig.suptitle("Why the primary model assigns risk · global SHAP", x=0.04, y=0.97, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.91, f"Frozen calibrated logistic regression · {len(indices):,} seeded 2025 samples · "
             f"reference = all {explanation['reference_n']:,} training rows (2019–2022)", fontsize=10.5)
    fig.text(0.04, 0.025, "Exact LR-logit SHAP with affine calibration scaling; signed one-hot and missing-indicator terms are grouped first.\n"
             "† Categorical features and missing values are gray. Numeric colors use each feature's 5th–95th percentiles.\n"
             "Units are calibrated log-odds, not probability points. Correlated predictors and model associations are not causal effects.",
             fontsize=8.8, linespacing=1.4)
    return save_figure(fig, "shap_global_summary.png")


def dependence_plots(explanation: dict, importance: pd.DataFrame) -> tuple[dict, pd.DataFrame, list]:
    selected = ["historical_failure_rate", "days_since_last_inspection", "license_observed_tenure_years"]
    selected += [feature for feature in importance["feature"] if feature not in selected][:3]
    indices = explanation["sample_indices"]
    sample = explanation["test_frame"].iloc[indices]
    phi = explanation["shap_values"][indices]
    fig = plt.figure(figsize=(14.5, 9.4))
    grid = fig.add_gridspec(2, 3, left=0.07, right=0.98, bottom=0.18, top=0.82, hspace=0.64, wspace=0.42)
    summary_rows = []
    rng = np.random.default_rng(int(explanation["config"]["random_seed"]))
    for position, feature in enumerate(selected):
        subgrid = grid[position // 3, position % 3].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.10)
        ax = fig.add_subplot(subgrid[0, 0])
        missing_ax = fig.add_subplot(subgrid[0, 1], sharey=ax)
        column = explanation["feature_names"].index(feature)
        values, contribution = sample[feature], phi[:, column]
        missing = values.isna().to_numpy()
        numeric = feature in explanation["numeric_features"]
        if numeric:
            ax.scatter(values[~missing], contribution[~missing], s=7, color=COLORS["logistic_regression"], alpha=0.45, linewidths=0)
            if feature == "days_since_last_inspection":
                ax.set_xscale("symlog", linthresh=100)
                ax.set_xlabel("Observed days (symlog scale)", fontsize=8.5)
            else:
                ax.set_xlabel("Observed feature value", fontsize=8.5)
            observed = values[~missing]
            if observed.nunique() > 1:
                bins = pd.qcut(observed, q=5, duplicates="drop")
                groups = [(str(label), indices.index) for label, indices in observed.groupby(bins, observed=True)]
            else:
                groups = [("Observed", observed.index)]
        else:
            counts = values.value_counts()
            levels = counts.head(5).index.tolist()
            plot_levels = levels + (["Other levels"] if len(counts) > len(levels) else [])
            grouped = values.where(values.isin(levels), "Other levels")
            for level_index, level in enumerate(plot_levels):
                membership = grouped.eq(level).to_numpy() & ~missing
                ax.scatter(level_index+rng.uniform(-0.15, 0.15, membership.sum()), contribution[membership],
                           s=7, color="#9AA4AC", alpha=0.5, linewidths=0)
            ax.set_xticks(range(len(plot_levels)), [textwrap.fill(str(value).title(), 11) for value in plot_levels], rotation=25, ha="right", fontsize=7)
            ax.set_xlabel("Observed category (gray; no ordinal color)", fontsize=8)
            groups = [(str(level), sample.index[grouped.eq(level)]) for level in plot_levels]
        if missing.any():
            missing_ax.scatter(rng.uniform(-0.12, 0.12, missing.sum()), contribution[missing], s=8, color="#9AA4AC", alpha=0.5, linewidths=0)
        missing_ax.set(xlim=(-0.5, 0.5), xticks=[0], xticklabels=[f"Missing\nn={int(missing.sum())}"])
        missing_ax.tick_params(axis="x", labelsize=7.5)
        missing_ax.tick_params(axis="y", labelleft=False, left=False)
        missing_ax.spines["left"].set_color("#D5DEE5")
        for current in (ax, missing_ax):
            current.axhline(0, color="#9AA4AC", lw=0.8, ls="--")
            current.grid(axis="y", alpha=0.5)
        ax.set_title(f"{chr(65+position)} · {textwrap.fill(display_name(feature), 30)}", fontsize=10.5)
        ax.set_ylabel("Calibrated log-odds SHAP", fontsize=9)
        # Aggregates retain no establishment IDs or row-level source values.
        for label, group_indices in groups + [("Missing", sample.index[missing])]:
            membership = sample.index.isin(group_indices)
            if not membership.any():
                continue
            summary_rows.append({
                "feature": feature, "feature_type": "numeric" if numeric else "categorical",
                "group": label, "n": int(membership.sum()),
                "mean_feature_value": float(values[membership].mean()) if numeric and label != "Missing" else np.nan,
                "mean_shap": float(contribution[membership].mean()),
                "median_shap": float(np.median(contribution[membership])),
                "mean_calibrated_probability": float(explanation["calibrated_probability"][indices][membership].mean()),
                "observed_fail_rate": float(sample.loc[membership, "target"].mean()),
            })
    fig.suptitle("How observed features contribute to the primary score", x=0.04, y=0.97, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.91, "2025 explanation sample · three prespecified history/license features plus the three strongest additional grouped features", fontsize=10.5)
    fig.text(0.04, 0.035, "Missing values are displayed in separate strips; no zero-value substitution is shown. Categorical levels have no numeric order.\n"
             "Grouped LR-logit SHAP scaled to calibrated log-odds; no clipping occurs on explained samples; observed-value effects are linear.\n"
             "These are model associations conditional on correlated predictors, not evidence that changing a feature will change food safety.",
             fontsize=9, linespacing=1.5)
    return save_figure(fig, "shap_feature_dependence.png"), pd.DataFrame(summary_rows), selected


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    explanation = build_explanation()
    importance = importance_table(explanation)
    global_image = global_plot(explanation, importance)
    dependence_image, dependence, dependence_features = dependence_plots(explanation, importance)
    summary = {
        "schema_version": 1, "model": explanation["model"], "selected_on": "2023 validation AP",
        "explanation_method": "Exact interventional LR-logit SHAP with affine calibration-logit scaling, checked with shap.LinearExplainer",
        "units": "Calibrated log-odds; not probability percentage points",
        "reference": {
            "period": [explanation["config"]["train_start"], explanation["config"]["train_end"]],
            "n": explanation["reference_n"], "statistic": "Mean transformed training covariates; no background subsampling",
            "test_data_used_to_fit_reference": False,
        },
        "sample": {"year": 2025, "available_n": len(explanation["test_frame"]),
                   "n": len(explanation["sample_indices"]), "seed": explanation["config"]["random_seed"],
                   "method": "Uniform random sample without replacement; sorted indices"},
        "features": {"original": len(explanation["feature_names"]), "transformed": len(explanation["transformed_names"]),
                     "numeric": len(explanation["numeric_features"]), "categorical": len(explanation["categorical_features"]),
                     "missing_indicators": sum("__missingindicator_" in name for name in explanation["transformed_names"]),
                     "grouping_rule": "Sum signed one-hot/missing-indicator contributions into original features, then take mean absolute values"},
        "affine_calibration": {"slope": explanation["calibrator_slope"], "intercept": explanation["calibrator_intercept"],
                               "raw_logit_base": explanation["raw_base_value"], "calibrated_logit_base": explanation["base_value"],
                               "formula": "phi_cal=a*phi_raw; base_cal=a*base_raw+b; p=expit(base_cal+sum(phi_cal))",
                               "base_interpretation": "Expected calibrated logit under the training reference; expit(base) is not mean predicted probability"},
        "verification": explanation["verification"],
        "top_features": importance.head(10)[["rank", "feature", "feature_type", "mean_abs_shap", "mean_shap"]].to_dict("records"),
        "dependence_features": dependence_features,
        "figures": [global_image, dependence_image],
        "shap_version": shap.__version__,
        "provenance": explanation["provenance"],
        "limitations": ["Interventional feature contributions use a training reference, not a causal intervention model.",
                        "The affine-logit explanation is exact on all explained test inputs; clipping makes the calibrator non-affine outside its unclipped input domain.",
                        "Correlated predictors can divide or offset credit; coefficient direction is not a marginal causal effect.",
                        "Global ranking is descriptive on a seeded 1500-row held-out sample and is not a refit or feature-selection step."],
    }
    importance.to_csv(METRICS / "shap_global_importance.csv", index=False)
    dependence.to_csv(METRICS / "shap_dependence_summary.csv", index=False)
    (METRICS / "shap_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    local_dir = ROOT / ".local/shap"
    local_dir.mkdir(parents=True, exist_ok=True)
    explanation["summary"] = summary
    joblib.dump(explanation, local_dir / "logistic_regression.joblib", compress=3)
    print(f"Exact grouped SHAP: {len(explanation['test_frame'])} test rows; {len(explanation['feature_names'])} original features; reference n={explanation['reference_n']}")
    print(f"Global sample n={len(explanation['sample_indices'])}; clip count={explanation['verification']['test_calibration_clip_count']}; max logit error={explanation['verification']['calibrated_logit_max_absolute_error']:.3g}")
    print("Top features: " + ", ".join(importance.head(5)["feature"]))
    for image in (global_image, dependence_image):
        print(f"{Path(image['path']).name}: {image['width_px']}×{image['height_px']} px; {image['bytes']/1024:.1f} KiB")


if __name__ == "__main__":
    main()
