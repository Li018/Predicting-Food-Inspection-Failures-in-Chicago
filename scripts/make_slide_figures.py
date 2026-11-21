#!/usr/bin/env python3
"""Export projection figures and rounded slide facts from frozen Topic 1 artifacts.

This is a read-only consumer of fitted models, predictions and analytic SHAP.
The original reports/figures directory is never an output destination.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.metrics import average_precision_score, precision_recall_curve

from chicago_food.heuristics import add_heuristic_slide_metrics
from chicago_food.slide_validation import presentation_metric, validate_metric_contract


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("logistic_regression", "lightgbm", "prevalence")
SHORT = {"logistic_regression": "lr", "lightgbm": "lgbm", "prevalence": "prevalence"}
LABELS = {"logistic_regression": "LR · primary", "lightgbm": "LightGBM", "prevalence": "Base rate"}
COLORS = {"logistic_regression": "#1B365D", "lightgbm": "#B85A35", "prevalence": "#8E969D"}
PAPER, INK, LINE = "#F7F6F2", "#181C20", "#DCE1E5"
FIGURE_NAMES = (
    "01_seasonality_drift.png", "02_precision_recall.png", "03_calibration.png",
    "04_daily_capacity.png", "05_grouped_shap.png", "06_shap_dependence.png",
    "07_error_fp.png", "08_error_fn.png", "09_subgroup_audit.png",
)
FONT_SIZES = {"title": 18, "label": 15, "tick": 13.5, "legend": 13, "annotation": 13}
FEATURE_LABELS = {
    "observed_history_years": "Visible history years",
    "prior_scored_days": "Scored history days",
    "prior_fail_days": "Failed history days",
    "prior_inspection_days": "All inspection days",
    "prior_conditions_days": "Conditional-pass days",
    "prior_routine_days": "Routine history days",
    "historical_failure_rate": "Historical failure rate",
    "historical_conditions_rate": "Conditional-pass rate",
    "license_observed_tenure_years": "Visible license years",
    "days_to_license_term_end": "Days to license end",
    "license_term_elapsed_days": "Elapsed license days",
    "days_since_last_inspection": "Days since last visit",
    "days_since_last_failure": "Days since last Fail",
    "days_since_license_issue": "Days since issuance",
    "historical_pre2018_share": "Pre-2018 history share",
    "last_result": "Last observed result",
    "last_routine_result": "Last routine result",
    "prior_facility_type": "Prior facility type",
    "prior_risk": "Prior risk category",
    "scored_days_365d": "Scored days in last year",
    "license_records_observed": "Observed license records",
    "license_category": "License category",
    "license_term_status": "Recorded term status",
    "recent_failure_streak": "Consecutive failed days",
    "recency_weighted_failure_rate": "Recency-weighted Fail rate",
}
GROUP_LABELS = {
    "historical_failure_evidence": "Failure evidence",
    "inspection_exposure_time": "Exposure & timing",
    "facility_risk": "Facility & risk",
    "license_background": "License background",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_SIZES["label"],
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.titlesize": FONT_SIZES["title"], "axes.titleweight": "bold",
        "axes.labelsize": FONT_SIZES["label"], "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"], "legend.fontsize": FONT_SIZES["legend"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": LINE, "axes.facecolor": PAPER, "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER, "grid.color": LINE, "grid.linewidth": 0.8,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def protected_hashes(root: Path) -> dict[str, str]:
    """Protect the fitted model and its immutable inputs, not only the manifest."""
    paths = [root / "data/snapshots/manifest.json", root / ".local/features.parquet"]
    for directory in ("src/chicago_food", ".local/models", ".local/predictions", ".local/shap",
                      ".local/errors", "reports/figures"):
        paths.extend(p for p in (root / directory).glob("*") if p.is_file())
    paths.extend(root / "data/snapshots" / name for name in ("inspections.parquet", "licenses.parquet"))
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def check_output_directory(root: Path, output: Path) -> None:
    original = (root / "reports/figures").resolve()
    destination = output.resolve()
    if destination == original or original in destination.parents:
        raise ValueError("Slide exports must not overwrite reports/figures or its descendants")


def load_inputs(root: Path) -> dict:
    metrics = root / "reports/metrics"
    manifest = json.loads((root / "data/snapshots/manifest.json").read_text())
    for name, record in manifest["files"].items():
        if sha256(root / "data/snapshots" / name) != record["sha256"]:
            raise ValueError(f"Snapshot is not the frozen research input: {name}")
    json_names = (
        "data_summary", "model_selection", "calibration_summary", "paired_model_differences",
        "error_analysis", "feature_ablation", "seasonality_complaint", "grouped_shap_summary",
        "subgroup_audit_summary", "days_earlier_summary",
    )
    csv_names = (
        "model_comparison", "calibration_bins", "error_case_summary", "error_group_summary",
        "lag_sensitivity", "figure_daily_capacity_yield", "shap_global_importance",
        "feature_ablation", "inspection_type_annual", "inspection_type_seasonality",
        "grouped_shap_importance", "lr_standardized_odds_ratios", "subgroup_audit",
        "days_earlier_windows",
    )
    data = {name: json.loads((metrics / f"{name}.json").read_text()) for name in json_names}
    data.update({f"{name}_table": pd.read_csv(metrics / f"{name}.csv") for name in csv_names})
    if data["model_selection"]["selected_model"] != "logistic_regression":
        raise ValueError("The presentation must follow the frozen validation-selected LR")
    data["predictions"] = {}
    reference = None
    table = data["model_comparison_table"].set_index(["model", "probability_kind"])
    for family in FAMILIES:
        predictions = pd.read_parquet(root / ".local/predictions" / f"{family}.parquet")
        test = predictions.loc[predictions["split"].eq("test")].sort_values("inspection_id").reset_index(drop=True)
        alignment = test[["inspection_id", "entity_id", "inspection_date", "target"]]
        if reference is None:
            reference = alignment
        elif not reference.equals(alignment):
            raise ValueError("Saved model predictions have different test cohorts")
        if not test["inspection_date"].dt.year.eq(2025).all():
            raise ValueError("Projection figures require the 2025 holdout")
        for kind in ("raw", "calibrated"):
            measured = average_precision_score(test["target"], test[f"{kind}_probability"])
            if not np.isclose(measured, table.loc[(family, kind), "average_precision"], atol=1e-12, rtol=0):
                raise ValueError(f"Saved prediction AP differs from its summary: {family}/{kind}")
        data["predictions"][family] = test
    data["shap"] = joblib.load(root / ".local/shap/logistic_regression.joblib")
    data["errors"] = joblib.load(root / ".local/errors/analysis.joblib")
    explanation = data["shap"]
    reconstructed = expit(explanation["base_value"] + explanation["shap_values"].sum(axis=1))
    if not np.allclose(reconstructed, explanation["calibrated_probability"], atol=1e-12, rtol=0):
        raise ValueError("Saved analytic SHAP does not reconstruct the frozen probabilities")
    saved = data["predictions"]["logistic_regression"].set_index("inspection_id")
    aligned = saved.loc[explanation["test_frame"]["inspection_id"], "calibrated_probability"].to_numpy()
    if not np.allclose(reconstructed, aligned, atol=1e-12, rtol=0):
        raise ValueError("Saved analytic SHAP differs from the frozen LR prediction file")
    error_frame = data["errors"]["test_frame"]
    keys = ["inspection_id", "entity_id", "inspection_date", "target"]
    if not error_frame[keys].equals(explanation["test_frame"][keys]):
        raise ValueError("Anonymous error cases and analytic SHAP have different row ordering")
    if not np.allclose(data["errors"]["probabilities"], reconstructed, atol=1e-12, rtol=0):
        raise ValueError("Anonymous error-case scores differ from analytic SHAP")
    if sha256(root / ".local/features.parquet") != explanation["provenance"]["feature_sha256"]:
        raise ValueError("Feature cache differs from the analytic SHAP provenance")
    data["source_hashes"] = {
        str(p.relative_to(root)): sha256(p)
        for p in [*(metrics / f"{name}.json" for name in json_names),
                  *(metrics / f"{name}.csv" for name in csv_names)]
    }
    return data


def save(fig, output: Path, name: str, root: Path) -> dict:
    fig.savefig(output / name, dpi=180, pil_kwargs={"optimize": True})
    plt.close(fig)
    path = output / name
    with Image.open(path) as image:
        width, height = image.size
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        relative = str(path)
    return {"path": relative, "width_px": width, "height_px": height,
            "dpi": 180, "sha256": sha256(path), "bytes": path.stat().st_size}


def trend_figure(data: dict):
    annual = data["inspection_type_annual_table"]
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.84)
    colors = {"Canvass": COLORS["logistic_regression"], "Complaint": COLORS["lightgbm"]}
    for kind in ("Canvass", "Complaint"):
        rows = annual.loc[annual["inspection_type"].eq(kind)].sort_values("year")
        ax.plot(rows["year"], rows["fail_rate"], marker="o", markersize=5.5,
                lw=2.7, color=colors[kind], label=kind)
    ax.axvspan(2018.5, 2025.5, color="#DDE8EC", alpha=0.42, zorder=0)
    ax.axvline(2018.5, color=COLORS["lightgbm"], lw=1.6, linestyle="--")
    ax.text(2018.65, 0.97, "July 2018\nrule change", transform=ax.get_xaxis_transform(),
            fontsize=13, va="top", color=COLORS["lightgbm"])
    ax.set(title="Canvass and Complaint fail rates", xlabel="Inspection year", ylabel="Observed Fail rate")
    ax.set_xlim(2009.6, 2025.5)
    ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
    ax.set_ylim(0, max(0.45, float(annual["fail_rate"].max()) * 1.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", alpha=0.85)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.04))
    return fig


def pr_figure(data: dict):
    fig, ax = plt.subplots(figsize=(8.0, 3.95))
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.21, top=0.88)
    comparison = data["model_comparison_table"].set_index(["model", "probability_kind"])
    for family in FAMILIES:
        frame = data["predictions"][family]
        ap = float(comparison.loc[(family, "calibrated"), "average_precision"])
        if family == "prevalence":
            ax.axhline(frame["target"].mean(), color=COLORS[family], ls="--", lw=2,
                       label=f"Base rate {ap:.3f}")
        else:
            precision, recall, _ = precision_recall_curve(frame["target"], frame["calibrated_probability"])
            ax.step(recall, precision, where="post", color=COLORS[family], lw=2.1,
                    label=f"{LABELS[family]}  AP {ap:.3f}")
    ax.set(title="2025 precision–recall", xlabel="Recall of all Fail cases", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(alpha=0.75)
    ax.legend(loc="upper right", fontsize=13)
    return fig


def calibration_figure(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.1))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.81, wspace=0.35)
    bins = data["calibration_bins_table"]
    limit = max(0.6, float(bins[["raw_mean_probability", "calibrated_mean_probability", "observed_fail_rate"]].max().max()) * 1.08)
    for ax, family in zip(axes, FAMILIES[:2], strict=True):
        rows = bins.loc[bins["model"].eq(family)]
        ax.plot([0, limit], [0, limit], color=COLORS["prevalence"], ls="--", lw=1.5, label="Ideal")
        for kind, color, marker, label in (("raw", "#A97144", "o", "Before"),
                                          ("calibrated", COLORS["logistic_regression"], "s", "After")):
            ax.plot(rows[f"{kind}_mean_probability"], rows["observed_fail_rate"],
                    color=color, marker=marker, markersize=6, lw=2.2, label=label)
        ax.set(title=LABELS[family], xlabel="Predicted Fail risk", xlim=(0, limit), ylim=(0, limit))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.8)
    axes[0].set_ylabel("Observed Fail rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 0.995), fontsize=14)
    return fig


def daily_figure(data: dict):
    rows = data["figure_daily_capacity_yield_table"]
    rows = rows.loc[rows["model"].eq("logistic_regression")].sort_values("nominal_budget_fraction")
    failures = int(data["predictions"]["logistic_regression"]["target"].sum())
    fig, ax = plt.subplots(figsize=(8.8, 5.35))
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.18, top=0.83)
    x = np.arange(len(rows))
    for offset, column, color, label in ((-0.2, "random_expected_failures_found", "#BFC8CF", "Same-day random"),
                                         (0.2, "expected_failures_found", COLORS["logistic_regression"], "LR priority")):
        values = rows[column].to_numpy()
        bars = ax.bar(x + offset, values, 0.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.0f}\n{value/failures:.1%}" for value in values],
                     padding=5, fontsize=14, linespacing=1.3)
    ax.set_xticks(x, [f"Daily Top {100*r.nominal_budget_fraction:.0f}%\n{r.selected_n:,.0f} slots"
                         for r in rows.itertuples(index=False)])
    ax.set(title="Failures found within daily priority slots", ylabel="Observed Fail cases",
           ylim=(0, rows["expected_failures_found"].max() * 1.32))
    ax.grid(axis="y", alpha=0.7)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.03), fontsize=14)
    return fig


def grouped_shap_figure(data: dict):
    groups = data["grouped_shap_importance_table"]
    groups = groups.loc[groups["scope"].eq("fixed_explanation_sample")].sort_values("mean_abs_group_shap", ascending=False)
    top = data["shap_global_importance_table"].sort_values("rank").head(8)
    labels = [GROUP_LABELS[row.group] for row in groups.itertuples(index=False)]
    labels += [FEATURE_LABELS.get(feature, feature.replace("_", " ")) for feature in top["feature"]]
    positions = np.r_[np.arange(4), np.arange(8) + 5.2]
    values = np.r_[groups["mean_abs_group_shap"], top["mean_abs_shap"]]
    fig, ax = plt.subplots(figsize=(7.1, 5.05))
    fig.subplots_adjust(left=0.385, right=0.91, bottom=0.135, top=0.89)
    bars = ax.barh(positions, values, height=0.65,
                   color=[COLORS["logistic_regression"]] * 4 + ["#5B879E"] * 8)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=12.5)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, labelsize=12.5, pad=6)
    ax.set(ylim=(13.05, -0.7), xlim=(0, max(values)*1.22), xlabel="Mean absolute log-odds contribution")
    ax.xaxis.label.set_size(13)
    ax.tick_params(axis="x", labelsize=12)
    ax.axhline(4.1, color=LINE, lw=1.5)
    ax.grid(axis="x", alpha=0.7)
    fig.text(0.025, 0.95, "Business groups and Top 8 features", fontsize=18, fontweight="bold")
    return fig


def dependence_figure(data: dict):
    explanation = data["shap"]
    ids = explanation["sample_indices"]
    frame = explanation["test_frame"].iloc[ids]
    phi = explanation["shap_values"][ids]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 5.05))
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.19, top=0.82, wspace=0.4)
    choices = (("historical_failure_rate", "Historical\nfailure rate", "Historical Fail share"),
               ("license_observed_tenure_years", "Visible license\ntenure", "Observed years"))
    for ax, (feature, title, xlabel) in zip(axes, choices, strict=True):
        column = explanation["feature_names"].index(feature)
        observed = frame[feature].notna().to_numpy()
        x = frame.loc[observed, feature].to_numpy()
        y = phi[observed, column]
        ax.scatter(x, y, s=12, alpha=0.3, color=COLORS["logistic_regression"], edgecolors="none", rasterized=True)
        ax.axhline(0, color=COLORS["prevalence"], lw=1.1)
        ax.set(title=title, xlabel=xlabel)
        ax.title.set_size(17)
        ax.xaxis.label.set_size(13.5)
        ax.tick_params(labelsize=12.5)
        if feature == "historical_failure_rate":
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xticks([0, 0.5, 1])
        ax.grid(alpha=0.65)
    axes[0].set_ylabel("SHAP contribution (log-odds)", fontsize=13.5)
    return fig


def reduced_waterfall(explanation: dict, index: int, top_n: int = 5) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Keep signed residual contributions so truncation still closes the score."""
    phi = np.asarray(explanation["shap_values"][index], dtype=float)
    top = np.argsort(-np.abs(phi), kind="stable")[:top_n]
    remaining = np.ones(len(phi), dtype=bool)
    remaining[top] = False
    values = np.r_[phi[top], phi[remaining].sum()]
    labels = [FEATURE_LABELS.get(explanation["feature_names"][i], explanation["feature_names"][i].replace("_", " ")) for i in top]
    labels.append(f"Other {int(remaining.sum())} features")
    cumulative = float(explanation["base_value"]) + np.r_[0, np.cumsum(values)]
    if not np.isclose(cumulative[-1], float(explanation["base_value"]) + phi.sum(), atol=1e-12, rtol=0):
        raise ValueError("Top-five waterfall no longer closes the original score")
    return labels, values, cumulative


def waterfall_figure(data: dict, case_id: str):
    case = next(case for case in data["errors"]["cases"] if case["case_id"] == case_id)
    index = int(case["row_index"])
    labels, values, cumulative = reduced_waterfall(data["shap"], index)
    probability = float(data["errors"]["probabilities"][index])
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    fig.subplots_adjust(left=0.40, right=0.88, bottom=0.17, top=0.87)
    for position, value, before, after in zip(range(1, 7), values, cumulative[:-1], cumulative[1:], strict=True):
        ax.barh(position, abs(value), left=min(before, after), height=0.58,
                color=COLORS["lightgbm"] if value >= 0 else COLORS["logistic_regression"])
        ax.plot([after, after], [position+0.31, position+0.69], color="#8798A5", ls="--", lw=0.8)
        ax.text(1.03, position, f"{value:+.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=13)
    ax.scatter([cumulative[0], cumulative[-1]], [0, 7], color=INK, s=33, zorder=3)
    for value, position in ((cumulative[0], 0), (cumulative[-1], 7)):
        ax.text(1.03, position, f"{value:+.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=13)
    ax.set_yticks(range(8), ["Training reference"] + labels + ["Final score"])
    ax.tick_params(axis="y", labelsize=12.5, length=0, pad=7)
    span = max(float(np.ptp(cumulative)), 0.1)
    ax.set(xlim=(cumulative.min()-span*0.08, cumulative.max()+span*0.08),
           ylim=(7.55, -0.6), xlabel="Calibrated log-odds")
    ax.xaxis.label.set_size(13)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", alpha=0.7)
    fig.text(0.025, 0.945, f"{case_id} · predicted Fail risk {probability:.1%}", fontsize=18, fontweight="bold")
    return fig


def subgroup_rows(data: dict) -> pd.DataFrame:
    table = data["subgroup_audit_table"]
    risk = table.loc[table["feature"].eq("prior_risk")].copy()
    risk_order = {"Risk 1 (High)": 0, "Risk 2 (Medium)": 1, "Risk 3 (Low)": 2, "Unknown": 3}
    risk = risk.loc[risk["value"].isin(risk_order)]
    risk["plot_order"] = risk["value"].map(risk_order)
    facilities = table.loc[table["feature"].eq("prior_facility_type")]
    facility = facilities.loc[~facilities["value"].isin(["Unknown", "Ambiguous"])].nlargest(5, "n").copy()
    unknown = facilities.loc[facilities["value"].eq("Unknown")]
    return pd.concat([risk.sort_values("plot_order"), facility, unknown], ignore_index=True)


def subgroup_figure(data: dict):
    rows = subgroup_rows(data)
    columns = ["average_precision", "ece_10_equal_width", "global_top10_selected_rate"]
    values = rows[columns].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 5.25))
    fig.subplots_adjust(left=0.405, right=0.985, bottom=0.04, top=0.79)
    ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=0.65)
    names = {"Risk 1 (High)": "Risk 1", "Risk 2 (Medium)": "Risk 2", "Risk 3 (Low)": "Risk 3*",
             "Unknown": "Unknown*", "Children's Services Facility": "Children's services"}
    labels = []
    for row in rows.itertuples(index=False):
        label = names.get(str(row.value), str(row.value))
        if row.value == "Unknown":
            label = "Unknown risk*" if row.feature == "prior_risk" else "Unknown facility*"
        labels.append(f"{label}  ({int(row.n):,})")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks(range(3), ["AP", "ECE ↓", "Selected"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10, labelsize=13.5)
    for row in range(len(rows)):
        for col in range(3):
            value = values[row, col]
            label = f"{value:.3f}" if col == 0 else f"{value:.1%}"
            ax.text(col, row, label, va="center", ha="center", fontsize=15,
                    color="white" if value > 0.4 else INK)
    ax.axhline(3.5, color=PAPER, lw=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.025, 0.95, "Historical categories · global Top 10%", fontsize=18, fontweight="bold")
    fig.text(0.025, 0.89, "Group size in parentheses   * Small subgroup", fontsize=12.5)
    return fig


def rounded(value):
    if isinstance(value, dict):
        return {str(key): rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_slide_metrics(data: dict, figures: list[dict], tests_passed: int | None = None, *, root: Path = ROOT) -> dict:
    values, display, refs, format_specs = {}, {}, {}, {}

    def put(key, value, fmt=".3f", source="", *, scale=1, suffix=""):
        values[key], display[key], format_specs[key] = presentation_metric(value, fmt, scale=scale, suffix=suffix)
        refs[key] = source

    summary = data["data_summary"]
    split = summary["splits"]
    match = split["test"]["license_matching"]
    put("data.targets", sum(x["rows"] for x in split.values()), ",.0f", "data_summary.json:splits.*.rows")
    for key, value in (("test_n", split["test"]["rows"]), ("test_failures", split["test"]["failures"]),
                       ("source_inspections", summary["overall"]["inspection_records"]),
                       ("numeric_features", 32), ("categorical_features", 7), ("features", 39), ("sanitarian_features", 0)):
        put(f"data.{key}", value, ",.0f", "data_summary.json")
    manifest_sources = summary["sources"]
    for key, value in (("license_transactions", manifest_sources["licenses"]["downloaded_rows"]),
                       ("license_retained", manifest_sources["licenses"]["retained_rows"])):
        put(f"data.{key}", value, ",.0f", "data_summary.json:sources.licenses")
    put("data.fail_rate", split["test"]["fail_rate"], ".1%", "data_summary.json:splits.test.fail_rate")
    for key, field in (("match_number", "license_number_only_match_rate"),
                       ("match_address", "license_address_match_rate_ignoring_time"),
                       ("match_visible", "point_in_time_visible_match_rate")):
        put(f"data.{key}", match[field], ".1%", f"data_summary.json:splits.test.license_matching.{field}")
    put("data.rule_change", "2018-07-01", "text", "data_summary.json:rule_change.date")
    put("data.backfill_date", "2025-09-25", "text", "data/snapshots/manifest.json:sources.licenses.source_description")
    for key, value, source in (
        ("data.study_years", "2019–2025", "config/topic1.json:train_start/test_end"),
        ("data.test_year", "2025", "config/topic1.json:test_start/test_end"),
        ("data.history_years", "2010–2025", "config/topic1.json:history_start/data_end_date"),
        ("labels.post_code", "Post-2018 Food Code", "data_summary.json:rule_change.date; config/topic1.json:train_start"),
        ("labels.top5", "Top 5%", "config/topic1.json:top_fractions"),
        ("labels.top10", "Top 10%", "config/topic1.json:top_fractions"),
        ("labels.top20", "Top 20%", "config/topic1.json:top_fractions"),
        ("labels.history_lag", "≥1 day", "config/topic1.json:observation_lag_days"),
        ("labels.complaints", "311 complaint signals", "reports/topic1_report.md:future work"),
        ("errors.fp01_id", "FP-01", "error_case_summary.csv:case_id"),
        ("errors.fn01_id", "FN-01", "error_case_summary.csv:case_id"),
        ("labels.bootstrap_replicates", "500", "config/topic1.json:bootstrap_replicates"),
        ("presentation.slide_count", "13", "User-specified presentation layout"),
    ):
        put(key, value, "text", source)
    for index in range(1, 14):
        put(f"presentation.slide_{index:02d}", f"{index:02d} / 13", "text", "Presentation navigation metadata")
    for index in range(1, 4):
        put(f"presentation.step_{index:02d}", f"{index:02d}", "text", "Presentation roadmap metadata")
    for days in (1, 7, 30):
        put(f"lag.{days}.days", f"{days} day" + ("s" if days != 1 else ""), "text", "config/topic1.json:observation_lag_days/sensitivity_lags")
    for name, period in (("pre", "2018_first_half"), ("post", "2018_second_half")):
        put(f"data.rule_{name}_fail_rate", summary["rule_change"]["periods"][period]["target_fail_rate"], ".1%", "data_summary.json:rule_change.periods")
    for name, record in split.items():
        years = record["start"][:4] if record["start"][:4] == record["end"][:4] else f"{record['start'][:4]}–{record['end'][:4]}"
        put(f"splits.{name}.years", years, "text", f"data_summary.json:splits.{name}")
        put(f"splits.{name}.n", record["rows"], ",.0f", f"data_summary.json:splits.{name}.rows")
        put(f"splits.{name}.fail_rate", record["fail_rate"], ".1%", f"data_summary.json:splits.{name}.fail_rate")
    comparison = data["model_comparison_table"].set_index(["model", "probability_kind"])
    for family in FAMILIES:
        row = comparison.loc[(family, "calibrated")]
        for key, field in (("ap", "average_precision"), ("auc", "roc_auc"), ("brier", "brier_score"),
                           ("log_loss", "log_loss"), ("top10_precision", "top_10pct_precision")):
            put(f"{SHORT[family]}.{key}", row[field], ".1%" if key == "top10_precision" else ".3f", f"model_comparison.csv:{family}/calibrated/{field}")
        put(f"{SHORT[family]}.validation_ap", data["model_selection"]["validation_average_precision_by_family"][family], ".4f", "model_selection.json:validation_average_precision_by_family")
    interval = data["paired_model_differences"]["intervals"]["average_precision"]
    for key, field in (("ap_delta", "estimate"), ("ap_ci_lower", "lower"), ("ap_ci_upper", "upper")):
        put(f"paired.{key}", interval[field], "+.4f", f"paired_model_differences.json:intervals.average_precision.{field}")
    for family in FAMILIES[:2]:
        record = data["calibration_summary"]["models"][family]["probability_variants"]
        for short_kind, kind in (("raw", "raw"), ("cal", "calibrated")):
            for key, field, fmt in (("mean_p", "mean_probability", ".1%"), ("brier", "brier_score", ".4f"),
                                    ("ece", "ece_on_raw_score_deciles", ".1%")):
                put(f"calibration.{SHORT[family]}.{short_kind}_{key}", record[kind][field], fmt,
                    f"calibration_summary.json:models.{family}.probability_variants.{kind}.{field}")
    daily = data["figure_daily_capacity_yield_table"]
    daily = daily.loc[daily["model"].eq("logistic_regression")]
    for row in daily.itertuples(index=False):
        budget = int(round(100*row.nominal_budget_fraction))
        mapping = {"slots": (row.selected_n, ",.0f"), "selected_rate": (row.selected_fraction, ".1%"),
                   "failures": (row.expected_failures_found, ",.0f"),
                   "random_expected": (row.random_expected_failures_found, ".1f"),
                   "random_rounded": (row.random_expected_failures_found, ",.0f"),
                   "extra_failures": (row.expected_failures_found-row.random_expected_failures_found, ".1f"),
                   "lift_gain": (row.lift_vs_daily_random-1, "+.1%"), "recall": (row.recall, ".1%")}
        for key, (value, fmt) in mapping.items():
            put(f"capacity.daily{budget}.{key}", value, fmt, f"figure_daily_capacity_yield.csv:logistic_regression/{budget}%")
    stages = {"history_only": "history", "history_time": "exposure", "history_time_license": "license", "frozen_primary_39": "reference"}
    ablation = data["feature_ablation_table"]
    ablation = ablation.loc[ablation["split"].eq("test") & ablation["probability_kind"].eq("calibrated")]
    for row in ablation.itertuples(index=False):
        prefix = f"ablation.{stages[row.stage]}"
        for key, field in (("ap", "average_precision"), ("auc", "roc_auc"), ("brier", "brier_score"),
                           ("delta_ap", "delta_average_precision"), ("count", "feature_count")):
            put(f"{prefix}.{key}", getattr(row, field), ",.0f" if key == "count" else "+.4f" if key == "delta_ap" else ".4f", f"feature_ablation.csv:{row.stage}/test/calibrated/{field}")
    for kind, record in data["seasonality_complaint"]["seasonality"].items():
        prefix = f"seasonality.{kind.lower()}"
        for key, field, fmt in (("low_rate", "pooled_min_fail_rate", ".1%"), ("high_rate", "pooled_max_fail_rate", ".1%"),
                                ("range_pp", "pooled_month_range_pp", ".1f"),
                                ("change_2019_2025_pp", "fail_rate_change_2019_2025_pp", "+.1f"),
                                ("change_2023_2025_pp", "fail_rate_change_2023_2025_pp", "+.1f")):
            put(f"{prefix}.{key}", record[field], fmt, f"seasonality_complaint.json:seasonality.{kind}.{field}")
        for key, field in (("low_month", "pooled_min_month"), ("high_month", "pooled_max_month")):
            put(f"{prefix}.{key}", calendar.month_abbr[int(record[field])], "text", f"seasonality_complaint.json:seasonality.{kind}.{field}")
    grouped = data["grouped_shap_importance_table"]
    for row in grouped.loc[grouped["scope"].eq("fixed_explanation_sample")].itertuples(index=False):
        for key, field, fmt in (("count", "feature_count", ",.0f"), ("mean_abs", "mean_abs_group_shap", ".3f"),
                                ("cancellation", "within_group_cancellation_fraction", ".1%")):
            put(f"groups.{row.group}.{key}", getattr(row, field), fmt, f"grouped_shap_importance.csv:{row.group}/{field}")
    odds = data["lr_standardized_odds_ratios_table"].set_index("feature")
    for feature in ("prior_fail_days", "prior_scored_days", "observed_history_years", "historical_failure_rate", "license_observed_tenure_years"):
        for kind in ("raw", "calibrated"):
            put(f"or.{feature}.{kind}", odds.loc[feature, f"{kind}_odds_ratio_per_train_sd"], ".2f", f"lr_standardized_odds_ratios.csv:{feature}/{kind}")
    risk = data["subgroup_audit_table"]
    risk = risk.loc[risk["feature"].eq("prior_risk")].set_index("value")
    for key, label in (("risk1", "Risk 1 (High)"), ("risk2", "Risk 2 (Medium)"), ("risk3", "Risk 3 (Low)")):
        for field, column, fmt in (("n", "n", ",.0f"), ("ap", "average_precision", ".3f"),
                                   ("ece", "ece_10_equal_width", ".1%"), ("selection", "global_top10_selected_rate", ".1%")):
            put(f"subgroups.{key}.{field}", risk.loc[label, column], fmt, f"subgroup_audit.csv:{label}/{column}")
    earlier = data["days_earlier_summary"]
    put("days_earlier.mean", earlier["pooled_days_earlier"], "+.2f", "days_earlier_summary.json:pooled_days_earlier")
    windows = data["days_earlier_windows_table"]
    delay = np.average(windows["delayed_positive_fraction"], weights=windows["positive_count"])
    put("days_earlier.delayed_fraction", float(delay), ".1%", "days_earlier_windows.csv:Fail-weighted delayed_positive_fraction")
    put("days_earlier.windows", len(windows), ",.0f", "days_earlier_windows.csv:window count")
    profiles = data["error_group_summary_table"].set_index("group")
    for key, group, field in (("fn_last_pass", "FN", "last_result_pass_fraction"),
                              ("fn_cold_start", "FN", "cold_start_fraction"),
                              ("fp_conditions", "FP", "actual_pass_with_conditions_fraction")):
        put(f"errors.{key}", profiles.loc[group, field], ".1%", f"error_group_summary.csv:{group}/{field}")
    put("errors.annual_slots", data["error_analysis"]["primary_operating_point"]["confusion"]["selected_n"], ",.0f", "error_analysis.json:primary_operating_point.confusion.selected_n")
    cases = data["error_case_summary_table"].set_index("case_id")
    for key, case in (("fp01_probability", "FP-01"), ("fn01_probability", "FN-01")):
        put(f"errors.{key}", cases.loc[case, "probability"], ".1%", f"error_case_summary.csv:{case}/probability")
    lags = data["lag_sensitivity_table"]
    lags = lags.loc[lags["model"].eq("logistic_regression")]
    for row in lags.itertuples(index=False):
        for key, column, fmt in (("ap", "average_precision", ".4f"), ("delta_ap", "delta_average_precision_vs_lag_1", "+.4f"),
                                 ("retention", "top_10pct_member_retention_vs_lag_1", ".1%"),
                                 ("max_probability_delta", "prediction_max_absolute_change_vs_lag_1", ".1%")):
            binding = f"lag.{int(row.lag_days)}.{key}"
            source = f"lag_sensitivity.csv:logistic_regression/{row.lag_days}/{column}"
            if key == "max_probability_delta":
                put(binding, getattr(row, column), ".1f", source, scale=100, suffix=" pp")
            else:
                put(binding, getattr(row, column), fmt, source)
    put("literature.purple_or", float(np.exp(1.555)), ".2f", "Kannan et al. (2019), Table 3: exp(Purple coefficient 1.555)")
    put("literature.official_year", 2015, "d", "Chicago/food-inspections-evaluation")
    put("literature.audit_year", 2019, "d", "Kannan, Shapiro & Bilgic (2019)")
    put("validation.original_tests", 55, "d", "Original unmodified tests/test_{data,features,metrics,models}.py")
    put("validation.tests_passed", tests_passed, "d", "Verified pytest result supplied through --tests-passed")
    result = {
        "schema_version": 2, "primary_model": "logistic_regression", "test_year": 2025,
        "display": display, "values": values, "source_refs": refs, "format_specs": format_specs,
        "figures": figures, "source_hashes": data["source_hashes"],
        "figure_data": {
            "annual_fail_rates": data["inspection_type_annual_table"][["inspection_type", "year", "n", "failures", "fail_rate"]].to_dict("records"),
            "calibration_bins": data["calibration_bins_table"][["model", "raw_score_decile", "n", "observed_fail_rate", "raw_mean_probability", "calibrated_mean_probability"]].to_dict("records"),
            "daily_capacity": daily.to_dict("records"),
            "shap_top8": data["shap_global_importance_table"].sort_values("rank").head(8)[["feature", "sample_n", "mean_abs_shap"]].to_dict("records"),
            "grouped_shap": grouped.loc[grouped["scope"].eq("fixed_explanation_sample")].to_dict("records"),
            "subgroup_rows": subgroup_rows(data).drop(columns=["plot_order"]).to_dict("records"),
            "pr_curves": "Full curves are rendered directly from the frozen prediction files; per-row values remain local.",
        },
        "font_sizes_pt": FONT_SIZES,
        "method": {
            "fit_performed": False, "analytic_shap_recomputed": False,
            "shap_sample_n": int(len(data["shap"]["sample_indices"])),
            "grouped_shap": "Signed within-row sums of the existing 39 analytic feature contributions, then mean absolute group contribution.",
            "waterfalls": "Five largest absolute feature contributions plus the signed sum of all remaining features.",
            "subgroup_selection": "One overall annual Top 10% allocation, not a separate top decile within each group.",
            "days_earlier": "Retrospective fixed-window reference using original-day scores; time-invariant outcomes and earlier feature availability are not established.",
            "no_inspector_features": "No sanitarian identifiers or clusters are inputs. Label and proxy bias may remain.",
            "rounding": "Values are quantized once from full source metrics at each field's display precision, in the original units. format_specs derives display from values; percentage and probability-change displays apply scale=100. Exact research values remain in source_refs. Figure-data summaries separately retain at most four decimals.",
        },
    }
    result = add_heuristic_slide_metrics(result, root)
    validate_metric_contract(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: <root>/reports/slide_figures")
    parser.add_argument("--metrics-output", type=Path, help="Default: <root>/reports/metrics/slide_metrics.json")
    parser.add_argument("--tests-passed", type=int, help="Passing count from the latest completed full pytest run")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "reports/slide_figures").resolve()
    metric_output = (args.metrics_output or root / "reports/metrics/slide_metrics.json").resolve()
    check_output_directory(root, output)
    baseline = protected_hashes(root)
    data = load_inputs(root)
    style()
    output.mkdir(parents=True, exist_ok=True)
    builders = (trend_figure, pr_figure, calibration_figure, daily_figure, grouped_shap_figure,
                dependence_figure, lambda data: waterfall_figure(data, "FP-01"),
                lambda data: waterfall_figure(data, "FN-01"), subgroup_figure)
    figures = [save(builder(data), output, name, root) for builder, name in zip(builders, FIGURE_NAMES, strict=True)]
    metrics = build_slide_metrics(data, figures, args.tests_passed, root=root)
    if baseline != protected_hashes(root):
        raise RuntimeError("Frozen inputs or original figures changed during projection export")
    metrics["protected_inputs_unchanged"] = True
    metric_output.parent.mkdir(parents=True, exist_ok=True)
    payload = rounded(metrics)
    # Presentation values already have their declared precision; do not round them again.
    payload["values"], payload["format_specs"] = metrics["values"], metrics["format_specs"]
    validate_metric_contract(payload)
    metric_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Exported {len(figures)} projection figures; {len(metrics['display'])} rounded display bindings.")
    print("Frozen models, source snapshots and original figures: unchanged.")
    for figure in figures:
        print(f"{Path(figure['path']).name}: {figure['width_px']} x {figure['height_px']} px")


if __name__ == "__main__":
    main()
