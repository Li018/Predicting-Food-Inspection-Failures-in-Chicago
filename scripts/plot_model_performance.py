#!/usr/bin/env python3
"""Render held-out ranking and capacity figures from frozen local predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from chicago_food.data import sha256_file
from chicago_food.metrics import evaluate_probabilities

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"
FIGURES = ROOT / "reports/figures"
FAMILIES = ("logistic_regression", "lightgbm", "prevalence")
COLORS = {"logistic_regression": "#247BA0", "lightgbm": "#E47B34", "prevalence": "#9AA4AC"}
LABELS = {"logistic_regression": "Logistic regression · primary", "lightgbm": "LightGBM · comparison", "prevalence": "Prevalence / random"}
TEXT = "#19324D"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.edgecolor": "#D5DEE5", "axes.labelsize": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "#FAFBFC", "figure.facecolor": "white",
        "grid.color": "#E3E9EE", "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 9,
        "savefig.facecolor": "white", "savefig.dpi": 300,
    })


def figure_header(fig, title: str, subtitle: str, footer: str) -> None:
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.06, y=0.975, ha="left")
    fig.text(0.06, 0.905, subtitle, fontsize=10.5, ha="left")
    fig.text(0.06, 0.035, footer, fontsize=9, va="bottom", linespacing=1.5)


def save_figure(fig, filename: str) -> dict:
    path = FIGURES / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.18, pil_kwargs={"optimize": True})
    plt.close(fig)
    with Image.open(path) as image:
        width, height = image.size
    return {"path": f"reports/figures/{filename}", "dpi": 300,
            "width_px": width, "height_px": height, "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def load_inputs() -> tuple[dict, dict, dict]:
    selection = json.loads((METRICS / "model_selection.json").read_text())
    comparison = pd.read_csv(METRICS / "model_comparison.csv")
    calibrated = comparison.loc[comparison["probability_kind"].eq("calibrated")].set_index("model")
    predictions, summaries = {}, {}
    for family in FAMILIES:
        frame = pd.read_parquet(ROOT / ".local/predictions" / f"{family}.parquet")
        frame = frame.loc[frame["split"].eq("test")].sort_values("inspection_id").reset_index(drop=True)
        predictions[family] = frame
        calculated = evaluate_probabilities(
            frame["target"], frame["calibrated_probability"],
            dates=frame["inspection_date"], top_fractions=(0.05, 0.1, 0.2),
        )
        for key, value in calculated.items():
            if value is not None and not np.isclose(value, calibrated.loc[family, key], rtol=0, atol=1e-10):
                raise ValueError(f"Prediction/table mismatch: {family}/{key}")
        summaries[family] = calibrated.loc[family].to_dict()
    aligned = ["inspection_id", "inspection_date", "entity_id", "target"]
    for family in FAMILIES[1:]:
        if not predictions[family][aligned].equals(predictions[FAMILIES[0]][aligned]):
            raise ValueError("Figures require the same held-out cohort for every model")
    if selection["selected_model"] != "logistic_regression":
        raise ValueError("Update primary-model legend labels to match the frozen selection")
    return predictions, summaries, selection


def roc_pr_figure(predictions: dict, summaries: dict) -> tuple[dict, list]:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.80, wspace=0.25)
    aggregate_rows = []
    for family in FAMILIES:
        frame, summary = predictions[family], summaries[family]
        y, p = frame["target"], frame["calibrated_probability"]
        fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(y, p, drop_intermediate=False)
        linestyle = "--" if family == "prevalence" else "-"
        label = LABELS[family]
        axes[0].plot(fpr, tpr, color=COLORS[family], lw=2.1, ls=linestyle,
                     label=f"{label}  |  AUC {summary['roc_auc']:.3f}")
        if family == "prevalence":
            axes[1].axhline(summary["prevalence"], color=COLORS[family], lw=1.8, ls="--",
                            label=f"Random baseline  |  Fail rate {summary['prevalence']:.1%}")
        else:
            # Recall is descending; this step convention agrees with AP's stepwise area.
            axes[1].step(recall, precision, where="post", color=COLORS[family], lw=1.9,
                         label=f"{label}  |  AP {summary['average_precision']:.3f}")
        aggregate_rows.append({
            "model": family, "n": len(frame), "failures": int(y.sum()),
            "prevalence": float(y.mean()), "roc_auc": summary["roc_auc"],
            "average_precision": summary["average_precision"],
            "roc_curve_points_rendered": len(fpr), "pr_curve_points_rendered": len(recall),
            "probability_kind": "calibrated",
        })
    axes[0].set(title="A · ROC discrimination", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="B · Precision–recall", xlabel="Recall of observed failures", ylabel="Precision")
    for axis in axes:
        axis.set(xlim=(0, 1), ylim=(0, 1.02))
        axis.grid(alpha=0.7)
        axis.legend(loc="lower right" if axis is axes[0] else "upper right", fontsize=8.7)
    reference = summaries["logistic_regression"]
    figure_header(
        fig, "Held-out ranking performance · 2025",
        f"{int(reference['n']):,} routine inspections · {int(reference['failures']):,} failures "
        "· estimators fitted on 2019–2022; probabilities calibrated on 2024",
        "Primary model selected using 2023 validation AP; test results never change that selection.\n"
        "AP is non-interpolated stepwise PR area. Full empirical curves are rendered; dense curve coordinates remain local.",
    )
    return save_figure(fig, "roc_pr_curves.png"), aggregate_rows


def topk_figure(predictions: dict, summaries: dict) -> tuple[dict, list]:
    fractions = np.unique(np.r_[np.linspace(0.02, 0.50, 25), 0.05, 0.10, 0.20])
    aggregate_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.80, wspace=0.25)
    for family in FAMILIES:
        frame = predictions[family]
        result = evaluate_probabilities(frame["target"], frame["calibrated_probability"], top_fractions=fractions)
        family_rows = []
        for fraction in fractions:
            prefix = f"top_{100 * fraction:g}pct_"
            row = {"model": family, "budget_fraction": float(fraction)}
            row.update({name: result[prefix + name] for name in (
                "selected_n", "selected_fraction", "expected_failures_found", "precision", "recall", "lift"
            )})
            family_rows.append(row)
        aggregate_rows.extend(family_rows)
        for axis, metric in zip(axes, ("precision", "recall"), strict=True):
            axis.plot(fractions, [row[metric] for row in family_rows], color=COLORS[family],
                      lw=2.1, ls="--" if family == "prevalence" else "-", label=LABELS[family])
            for fraction in (0.05, 0.1, 0.2):
                key = f"top_{100 * fraction:g}pct_{metric}"
                value = summaries[family][key]
                lower, upper = (summaries[family][key + suffix] for suffix in ("_ci_lower", "_ci_upper"))
                axis.errorbar(fraction, value, yerr=[[value - lower], [upper - value]],
                              color=COLORS[family], fmt="o" if family == "logistic_regression" else "s",
                              markersize=3.8, elinewidth=1, capsize=3, alpha=0.75)
    axes[0].set(title="A · Hit rate among prioritized inspections", ylabel="Precision")
    axes[1].set(title="B · Share of observed failures captured", ylabel="Recall")
    for axis in axes:
        axis.set(xlabel="Annual pooled inspection budget", xlim=(0, 0.51), ylim=(0, 0.8))
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(alpha=0.7)
        axis.legend(loc="upper left" if axis is axes[1] else "upper right")
    figure_header(
        fig, "Inspection priority versus capacity · 2025",
        "Annual pooled ranking · identical held-out cohort · boundary-score ties use expected fractional allocation",
        "95% establishment-cluster percentile intervals at 5%, 10%, and 20% capacity; 500 replicates; fixed estimators.\n"
        "These curves describe the observed inspected cohort; a pooled annual queue is distinct from daily scheduling.",
    )
    return save_figure(fig, "topk_performance.png"), aggregate_rows


def daily_figure(summaries: dict) -> tuple[dict, list]:
    fractions = (0.05, 0.1, 0.2)
    aggregate_rows = []
    reference = summaries["logistic_regression"]
    x = np.arange(len(fractions))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.1))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.23, top=0.80, wspace=0.25)
    actual = [reference[f"daily_top_{100*f:g}pct_selected_n"] for f in fractions]
    pooled = [reference[f"top_{100*f:g}pct_selected_n"] for f in fractions]
    bars = axes[0].bar(x - 0.18, pooled, 0.34, color="#D5DEE5", label="Annual pooled capacity")
    axes[0].bar_label(bars, labels=[f"{int(v):,}" for v in pooled], padding=4, fontsize=9)
    bars = axes[0].bar(x + 0.18, actual, 0.34, color=COLORS["logistic_regression"], label="Sum of daily ceiling capacities")
    axes[0].bar_label(bars, labels=[f"{int(v):,}\n({v/reference['n']:.2%})" for v in actual], padding=4, fontsize=9)
    axes[0].set(title="A · Daily ceilings increase effective capacity", ylabel="Inspection slots", ylim=(0, max(actual)*1.36))
    axes[0].legend(loc="upper left", fontsize=8.7)
    for offset, family in zip((-0.25, 0.0, 0.25), FAMILIES, strict=True):
        values = []
        for fraction in fractions:
            summary = summaries[family]
            prefix = f"daily_top_{100*fraction:g}pct_"
            row = {"model": family, "nominal_budget_fraction": fraction,
                   "inspection_days": int(summary["inspection_days"])}
            row.update({name: summary[prefix + name] for name in (
                "selected_n", "selected_fraction", "expected_failures_found", "precision", "recall",
                "random_expected_failures_found", "lift_vs_daily_random",
            )})
            aggregate_rows.append(row)
            values.append(row["expected_failures_found"])
        label = "Same-day random expectation" if family == "prevalence" else LABELS[family]
        bars = axes[1].bar(x + offset, values, 0.23, color=COLORS[family], label=label)
        axes[1].bar_label(bars, labels=[f"{v:.1f}" if family == "prevalence" else f"{v:.0f}" for v in values],
                          padding=4, fontsize=9)
    maximum_yield = max(row["expected_failures_found"] for row in aggregate_rows)
    axes[1].set(title="B · Yield at the same daily inspection budget", ylabel="Expected observed failures captured", ylim=(0, maximum_yield * 1.33))
    axes[1].legend(loc="upper left", fontsize=8.7)
    for axis in axes:
        axis.set(xticks=x, xticklabels=[f"{f:.0%}" for f in fractions], xlabel="Nominal daily budget")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.7)
    daily_ten = next(row for row in aggregate_rows if row["model"] == "logistic_regression" and row["nominal_budget_fraction"] == 0.1)
    figure_header(
        fig, "Daily inspection queues · capacity and yield",
        f"2025 retrospective comparison across {int(reference['inspection_days'])} inspection days "
        "· daily capacity = ceiling(nominal budget × observed daily queue size)",
        f"At nominal 10%: {int(daily_ten['selected_n']):,} slots ({daily_ten['selected_fraction']:.2%} effective); "
        f"primary LR captures {daily_ten['expected_failures_found']:.0f} failures versus "
        f"{daily_ten['random_expected_failures_found']:.1f} under same-day random allocation.\n"
        "Expected retrospective yield conditional on historical scheduling; no causal policy effect or daily-queue confidence interval is claimed.",
    )
    return save_figure(fig, "daily_capacity_yield.png"), aggregate_rows


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    predictions, summaries, selection = load_inputs()
    outputs = []
    for function, arguments, csv_name in (
        (roc_pr_figure, (predictions, summaries), "figure_model_performance_summary.csv"),
        (topk_figure, (predictions, summaries), "figure_topk_performance.csv"),
        (daily_figure, (summaries,), "figure_daily_capacity_yield.csv"),
    ):
        image, aggregate = function(*arguments)
        pd.DataFrame(aggregate).to_csv(METRICS / csv_name, index=False)
        image["aggregate_data"] = f"reports/metrics/{csv_name}"
        outputs.append(image)
        print(f"{Path(image['path']).name}: {image['width_px']}×{image['height_px']} px; "
              f"{image['bytes']/1024:.1f} KiB; aggregate rows={len(aggregate)}")
    manifest = {
        "schema_version": 1, "selected_model": selection["selected_model"],
        "selection_year": 2023, "calibration_year": 2024, "evaluation_year": 2025,
        "probability_kind": "calibrated", "figures": outputs,
        "source_files": {
            name: sha256_file(METRICS / name)
            for name in ("model_comparison.csv", "model_selection.json")
        },
        "local_prediction_hashes": {
            family: sha256_file(ROOT / ".local/predictions" / f"{family}.parquet")
            for family in FAMILIES
        },
        "curve_data_policy": "Full ROC/PR curves are calculated locally; only scalar summaries are exported",
        "topk_uncertainty": "Existing 500-replicate establishment-cluster intervals at 5%, 10%, 20%; annual pooled ranking only",
    }
    (METRICS / "figure_performance_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
