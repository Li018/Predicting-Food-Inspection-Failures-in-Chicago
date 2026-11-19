#!/usr/bin/env python3
"""Describe errors at a fixed capacity and explain anonymous cases from saved SHAP."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from chicago_food.data import sha256_file
from chicago_food.metrics import capacity_weights, evaluate_probabilities
from explain_model import display_name
from plot_model_performance import COLORS, FIGURES, METRICS, ROOT, TEXT, save_figure, style

GROUP_ORDER = ("TP", "FP", "FN", "TN")


def confusion(y: np.ndarray, weights: np.ndarray) -> dict:
    tp, fp = float(np.dot(weights, y)), float(np.dot(weights, 1-y))
    fn, tn = float(np.dot(1-weights, y)), float(np.dot(1-weights, 1-y))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "selected_n": float(weights.sum()), "selected_fraction": float(weights.mean()),
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None,
            "false_negative_rate": fn/(tp+fn) if tp+fn else None}


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    total = weights[valid].sum()
    return float(np.dot(values[valid], weights[valid])/total) if total else np.nan


def group_profiles(frame: pd.DataFrame, p: np.ndarray, y: np.ndarray, selection: np.ndarray) -> pd.DataFrame:
    group_weights = {"TP": selection*y, "FP": selection*(1-y), "FN": (1-selection)*y, "TN": (1-selection)*(1-y)}
    rows = []
    for group in GROUP_ORDER:
        w = group_weights[group]
        row = {"group": group, "expected_n": float(w.sum()), "rows_with_positive_weight": int((w>0).sum()),
               "mean_probability": weighted_mean(p, w)}
        for feature in ("prior_scored_days", "prior_fail_days", "historical_failure_rate", "days_since_last_inspection",
                        "observed_history_years", "license_observed_tenure_years"):
            row["mean_"+feature] = weighted_mean(frame[feature], w)
            row[feature+"_missing_fraction"] = weighted_mean(frame[feature].isna(), w)
        flags = {
            "cold_start_fraction": frame["cold_start"].eq(1),
            "no_scored_history_fraction": frame["no_scored_history"].eq(1),
            "license_history_available_fraction": frame["license_history_available"].eq(1),
            "zero_prior_fail_days_fraction": frame["prior_fail_days"].eq(0),
            "last_result_pass_fraction": frame["last_result"].eq("Pass"),
            "last_result_fail_fraction": frame["last_result"].eq("Fail"),
            "last_result_pass_within_365d_fraction": frame["last_result"].eq("Pass") & frame["days_since_last_inspection"].le(365),
            "actual_pass_fraction": frame["actual_result"].eq("Pass"),
            "actual_pass_with_conditions_fraction": frame["actual_result"].eq("Pass w/ Conditions"),
        }
        row.update({name: weighted_mean(values, w) for name, values in flags.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(frame: pd.DataFrame, p: np.ndarray, y: np.ndarray, selection: np.ndarray) -> pd.DataFrame:
    rows = []
    for feature in ("prior_facility_type", "prior_risk", "license_history_available", "cold_start", "no_scored_history", "license_term_status"):
        for value, indices in frame.groupby(feature, dropna=False, sort=True).groups.items():
            ids = np.asarray(indices, dtype=int)
            metrics = evaluate_probabilities(y[ids], p[ids], top_fractions=(0.1,))
            # Keep subgroup discrimination/calibration, but never the helper's
            # within-subgroup Top-K. Operating decisions retain the global weights.
            row = {"feature": feature, "value": str(value), "entities": int(frame.loc[ids, "entity_id"].nunique()),
                   **{key: metrics[key] for key in ("n", "failures", "prevalence", "roc_auc", "average_precision", "brier_score", "log_loss")}}
            counts = confusion(y[ids], selection[ids])
            row.update({"global_top10_"+key: value for key, value in counts.items()})
            row["small_sample"] = row["n"] < 100 or row["failures"] < 10 or row["n"]-row["failures"] < 10
            row["threshold_scope"] = "Same overall annual Top10% allocation; no subgroup-specific threshold"
            rows.append(row)
    return pd.DataFrame(rows)


def choose_cases(frame: pd.DataFrame, p: np.ndarray, y: np.ndarray, selection: np.ndarray, explanation: dict) -> tuple[list, pd.DataFrame]:
    memberships = {"FP": (selection == 1) & (y == 0), "FN": (selection == 0) & (y == 1), "TP": (selection == 1) & (y == 1)}
    cases, rows, used_entities = [], [], set()
    for group in ("FP", "FN", "TP"):
        candidates = np.flatnonzero(memberships[group])
        order = candidates[np.argsort(p[candidates] if group == "FN" else -p[candidates], kind="stable")]
        chosen = []
        for index in order:
            entity = frame.iloc[index]["entity_id"]
            if entity in used_entities:
                continue
            used_entities.add(entity)
            chosen.append(int(index))
            if len(chosen) == 3:
                break
        if len(chosen) != 3:
            raise ValueError(f"Not enough unambiguous distinct-entity cases for {group}")
        for number, index in enumerate(chosen, start=1):
            phi = explanation["shap_values"][index]
            positive = [i for i in np.argsort(-phi) if phi[i] > 0][:3]
            negative = [i for i in np.argsort(phi) if phi[i] < 0][:3]
            case_id = f"{group}-{number:02d}"
            cases.append({"case_id": case_id, "group": group, "row_index": index})
            row = {"case_id": case_id, "group": group, "probability": float(p[index]),
                   "actual_result": frame.iloc[index]["actual_result"],
                   "selected_weight": float(selection[index]),
                   "base_calibrated_logit": explanation["base_value"],
                   "reconstructed_calibrated_logit": float(explanation["base_value"] + phi.sum()),
                   "probability_reconstruction_error": float(abs(expit(explanation["base_value"] + phi.sum()) - p[index])),
                   "top_positive_contributions": "; ".join(f"{explanation['feature_names'][i]}:{phi[i]:+.6f}" for i in positive),
                   "top_negative_contributions": "; ".join(f"{explanation['feature_names'][i]}:{phi[i]:+.6f}" for i in negative)}
            for feature in ("prior_scored_days", "prior_fail_days", "days_since_last_inspection", "observed_history_years",
                            "license_history_available", "license_observed_tenure_years", "cold_start", "last_result"):
                row[feature] = frame.iloc[index][feature]
            rows.append(row)
    return cases, pd.DataFrame(rows)


def waterfall(case: dict, frame: pd.DataFrame, p: np.ndarray, explanation: dict) -> tuple[dict, float]:
    index = case["row_index"]
    phi = explanation["shap_values"][index]
    top = np.argsort(-np.abs(phi), kind="stable")[:8]
    remaining = np.ones(len(phi), dtype=bool)
    remaining[top] = False
    contributions = np.r_[phi[top], phi[remaining].sum()]
    labels = []
    for column in top:
        feature = explanation["feature_names"][column]
        value = frame.iloc[index][feature]
        value_label = "Missing" if pd.isna(value) else f"{value:.3g}" if isinstance(value, (float, int, np.number)) else str(value)
        labels.append(textwrap.fill(f"{display_name(feature)} = {value_label}", 39))
    labels.append(f"Other {int(remaining.sum())} features")
    cumulative = explanation["base_value"] + np.r_[0, np.cumsum(contributions)]
    closure_error = float(abs(cumulative[-1] - (explanation["base_value"] + phi.sum())))
    fig, ax = plt.subplots(figsize=(11.9, 7.6))
    fig.subplots_adjust(left=0.36, right=0.91, top=0.80, bottom=0.17)
    positions = np.arange(1, len(contributions)+1)
    for position, value, before, after in zip(positions, contributions, cumulative[:-1], cumulative[1:], strict=True):
        color = COLORS["lightgbm"] if value >= 0 else COLORS["logistic_regression"]
        ax.barh(position, abs(value), left=min(before, after), height=0.63, color=color)
        ax.plot([after, after], [position+0.33, position+0.67], color="#8798A5", ls="--", lw=0.8)
        ax.text(1.015, position, f"{value:+.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.scatter([cumulative[0], cumulative[-1]], [0, len(contributions)+1], color=TEXT, s=35, zorder=3)
    ax.text(1.015, 0, f"{cumulative[0]:+.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.text(1.015, len(contributions)+1, f"{cumulative[-1]:+.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    all_labels = ["Training-reference logit base"] + labels + [f"Final logit → risk {p[index]:.1%}"]
    ax.set(yticks=np.arange(len(all_labels)), yticklabels=all_labels, ylim=(len(all_labels)-0.3, -0.7),
           xlabel="Calibrated log-odds")
    span = float(np.ptp(cumulative))
    ax.set_xlim(float(cumulative.min())-0.07*span, float(cumulative.max())+0.07*span)
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(cumulative[0], color="#9AA4AC", lw=0.9, ls="--")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    titles = {"FP": "High-score false positive", "FN": "Low-score missed failure", "TP": "High-risk correct priority"}
    fig.suptitle(f"{titles[case['group']]} · {case['case_id']}", x=0.04, y=0.96, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.90, f"Observed result: {frame.iloc[index]['actual_result']}  |  Predicted Fail risk: {p[index]:.1%}  |  "
             "Primary model: calibrated logistic regression", fontsize=10.3)
    fig.text(0.04, 0.84, "Annual pooled Top 10% priority rule; anonymous representative chosen after scoring", fontsize=9.7)
    fig.text(0.04, 0.035, "Orange raises model log-odds; blue lowers them. The remaining signed contributions close all 39 features.\n"
             "This explains the model's score; it cannot establish the real-world cause of the inspection outcome.",
             fontsize=9, linespacing=1.5)
    return save_figure(fig, f"shap_local_{case['group'].lower()}.png"), closure_error


def profile_plot(profiles: pd.DataFrame) -> dict:
    features = ["mean_probability", "zero_prior_fail_days_fraction", "last_result_pass_fraction",
                "license_history_available_fraction", "cold_start_fraction"]
    labels = ["Mean predicted\nrisk", "No prior\nfailed days", "Last result\nwas Pass", "License history\navailable", "Cold start"]
    matrix = profiles.set_index("group").loc[list(GROUP_ORDER), features].to_numpy()
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    fig.subplots_adjust(left=0.16, right=0.92, top=0.77, bottom=0.26)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row,column]:.1%}", ha="center", va="center",
                    color="white" if matrix[row,column] > 0.55 else TEXT, fontsize=12)
    counts = profiles.set_index("group")["expected_n"]
    ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
           yticks=np.arange(4), yticklabels=[f"{group}  (n={counts[group]:,.0f})" for group in GROUP_ORDER])
    ax.tick_params(axis="both", length=0)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label="Within-group share / mean probability")
    fig.suptitle("What distinguishes priority hits and misses?", x=0.04, y=0.96, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.87, "2025 · primary calibrated logistic regression · one fixed annual pooled Top 10% allocation", fontsize=10.5)
    fig.text(0.04, 0.055, "FN means a Fail outside the priority quota, not necessarily a predicted probability below 50%.\n"
             "Shares describe observed associations in the complete error groups; the separately selected extreme cases are illustrative.",
             fontsize=9, linespacing=1.5)
    return save_figure(fig, "error_profiles.png")


def main() -> None:
    style()
    source_path = ROOT / ".local/shap/logistic_regression.joblib"
    explanation = joblib.load(source_path)
    frame = explanation["test_frame"].reset_index(drop=True).copy()
    p = np.asarray(explanation["calibrated_probability"])
    y = frame["target"].to_numpy(dtype=int)
    capacity = int(np.ceil(0.1*len(frame)))
    selection = capacity_weights(p, capacity)
    current_results = pd.read_parquet(ROOT / "data/snapshots/inspections.parquet", columns=["inspection_id", "results"])
    if current_results["inspection_id"].duplicated().any():
        raise ValueError("Outcome association requires unique inspection IDs")
    frame["actual_result"] = frame["inspection_id"].map(current_results.set_index("inspection_id")["results"])
    if frame["actual_result"].isna().any() or not np.array_equal(frame["actual_result"].eq("Fail").astype(int), y):
        raise ValueError("Inspection outcome association disagrees with frozen labels")
    profiles = group_profiles(frame, p, y, selection)
    subgroups = subgroup_table(frame, p, y, selection)
    cases, case_table = choose_cases(frame, p, y, selection, explanation)
    images, closure_errors = [], []
    for group in ("FP", "FN", "TP"):
        representative = next(case for case in cases if case["group"] == group)
        image, error = waterfall(representative, frame, p, explanation)
        images.append(image)
        closure_errors.append(error)
    images.append(profile_plot(profiles))
    if max(closure_errors) > 1e-10 or case_table["probability_reconstruction_error"].max() > 1e-10:
        raise ValueError("Local explanation does not reconstruct the frozen score")
    profile = profiles.set_index("group")
    summary = {
        "schema_version": 1, "model": "logistic_regression", "probability_kind": "calibrated", "test_year": 2025,
        "primary_operating_point": {
            "rule": "Annual pooled Top10% capacity; no outcome-based threshold optimization",
            "nominal_fraction": 0.1, "capacity": capacity, "n": len(frame),
            "boundary_probability": float(np.sort(p)[-capacity]),
            "fractional_boundary_rows": int(((selection>0)&(selection<1)).sum()),
            "tie_policy": "capacity_weights expected fractional allocation at boundary-score ties",
            "confusion": confusion(y, selection),
            "error_definitions": {"FP": "Non-Fail allocated to priority quota", "FN": "Fail outside priority quota",
                                  "TP": "Fail allocated to priority quota", "TN": "Non-Fail outside priority quota"},
        },
        "separate_fixed_probability_operating_point": {
            "rule": "p >= 0.5; supplied as a distinct diagnostic, not used for case or subgroup selection",
            "confusion": confusion(y, (p>=0.5).astype(float)),
        },
        "case_selection": {
            "n": len(cases), "groups": {"FP": 3, "FN": 3, "TP": 3},
            "method": "Highest-probability FP and TP; lowest-probability FN; stable order; no fractional membership cases",
            "unique_entities_across_all_cases": len({frame.iloc[case["row_index"]]["entity_id"] for case in cases}),
            "public_identifiers": "Anonymous case_id only; no inspection IDs, entity IDs, names, addresses or licenses exported",
            "caveat": "Extreme illustrative cases are selected after observing test outcomes; they do not represent group frequencies",
        },
        "profile_definitions": {
            "counts_and_shares": "Weighted by expected global priority allocation when ties occur",
            "prior_counts": "Entity inspection days, not raw event rows",
            "recent_pass": "Last observed result is Pass and days since last inspection <=365",
            "missing_means": "Numeric means use observed values only; corresponding missing fractions are reported",
            "small_subgroups": "n<100 or fewer than 10 Fail or 10 Non-Fail cases; no comparative ranking claims",
        },
        "descriptive_patterns": [
            {"pattern": "Missed failures and previously observed Pass results",
             "fn_last_result_pass_fraction": float(profile.loc["FN", "last_result_pass_fraction"]),
             "tp_last_result_pass_fraction": float(profile.loc["TP", "last_result_pass_fraction"]),
             "fn_zero_prior_fail_days_fraction": float(profile.loc["FN", "zero_prior_fail_days_fraction"])},
            {"pattern": "Priority false positives retain historical failure evidence",
             "fp_mean_prior_fail_days": float(profile.loc["FP", "mean_prior_fail_days"]),
             "tn_mean_prior_fail_days": float(profile.loc["TN", "mean_prior_fail_days"]),
             "fp_actual_pass_with_conditions_fraction": float(profile.loc["FP", "actual_pass_with_conditions_fraction"])},
            {"pattern": "History coverage and misses despite observed history",
             "fn_cold_start_fraction": float(profile.loc["FN", "cold_start_fraction"]),
             "fn_license_history_available_fraction": float(profile.loc["FN", "license_history_available_fraction"]),
             "fp_license_history_available_fraction": float(profile.loc["FP", "license_history_available_fraction"])},
        ],
        "verification": {"local_waterfall_max_logit_closure_error": max(closure_errors),
                         "all_nine_case_max_probability_reconstruction_error": float(case_table["probability_reconstruction_error"].max()),
                         "all_test_max_probability_reconstruction_error": float(np.max(np.abs(expit(explanation["base_value"]+explanation["shap_values"].sum(axis=1))-p)))},
        "figures": images,
        "local_shap_source_sha256": sha256_file(source_path),
        "limitations": ["SHAP explains the fitted score, not the real-world cause of an inspection result.",
                        "Current outcomes are read only to describe errors and conditional Pass, never as predictive inputs.",
                        "Associations cannot distinguish changed practices, incomplete history, inspector effects or unobserved hazards.",
                        "Capacity misses are not interchangeable with probability-threshold errors or newly proven safety risks."],
    }
    case_table.to_csv(METRICS / "error_case_summary.csv", index=False)
    profiles.to_csv(METRICS / "error_group_summary.csv", index=False)
    subgroups.to_csv(METRICS / "error_subgroup_metrics.csv", index=False)
    (METRICS / "error_analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    local_dir = ROOT / ".local/errors"
    local_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cases": cases, "test_frame": frame, "selection_weights": selection,
                 "probabilities": p, "summary": summary}, local_dir / "analysis.joblib", compress=3)
    print("Annual Top10% confusion:", summary["primary_operating_point"]["confusion"])
    print("Separate p>=0.5 confusion:", summary["separate_fixed_probability_operating_point"]["confusion"])
    print(f"Anonymous cases: {len(cases)}; distinct entities={summary['case_selection']['unique_entities_across_all_cases']}; fractional cases excluded")
    for image in images:
        print(f"{Path(image['path']).name}: {image['width_px']}×{image['height_px']} px; {image['bytes']/1024:.1f} KiB")


if __name__ == "__main__":
    main()
