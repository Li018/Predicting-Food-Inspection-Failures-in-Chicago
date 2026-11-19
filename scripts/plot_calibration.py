#!/usr/bin/env python3
"""Compare saved 2025 raw/calibrated probabilities without fitting any model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from chicago_food.metrics import evaluate_probabilities


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("logistic_regression", "lightgbm")
LABELS = {"logistic_regression": "Logistic regression · primary", "lightgbm": "LightGBM · comparison"}
COLORS = {"text": "#19324D", "raw": "#247BA0", "calibrated": "#E47B34", "gray": "#9AA4AC"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "reports/metrics")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / ".local/predictions")
    parser.add_argument("--figure", type=Path, default=ROOT / "reports/figures/calibration_curves.png")
    args = parser.parse_args()
    selection_path = args.metrics_dir / "model_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["selected_model"] != "logistic_regression":
        raise ValueError("Update primary labels to match the frozen validation-selected model")
    comparison_path = args.metrics_dir / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path).set_index(["model", "probability_kind"])
    frames, model_summary, all_bins = {}, {}, []
    histogram_edges = np.linspace(0, 1, 21)
    for family in FAMILIES:
        path = args.predictions_dir / f"{family}.parquet"
        predictions = pd.read_parquet(path)
        frame = predictions.loc[predictions["split"].eq("test")].sort_values("inspection_id").reset_index(drop=True)
        if not frame["inspection_date"].dt.year.eq(2025).all():
            raise ValueError("Calibration figure may contain only held-out 2025 predictions")
        for kind in ("raw", "calibrated"):
            probabilities = frame[f"{kind}_probability"]
            if not (probabilities.notna() & probabilities.between(0, 1)).all():
                raise ValueError(f"Invalid saved probabilities: {family}/{kind}")
        # Both curves use exactly the same raw-score bins, including tied scores.
        frame["score_decile"] = pd.qcut(frame["raw_probability"], q=10, labels=False, duplicates="drop") + 1
        family_bins = []
        for decile, group in frame.groupby("score_decile", observed=True, sort=True):
            row = {
                "model": family, "split": "test", "year": 2025, "raw_score_decile": int(decile),
                "n": len(group), "failures": int(group["target"].sum()),
                "observed_fail_rate": float(group["target"].mean()),
                "raw_mean_probability": float(group["raw_probability"].mean()),
                "calibrated_mean_probability": float(group["calibrated_probability"].mean()),
                "raw_probability_min": float(group["raw_probability"].min()),
                "raw_probability_max": float(group["raw_probability"].max()),
            }
            family_bins.append(row)
        bins = pd.DataFrame(family_bins)
        if bins["n"].sum() != len(frame) or bins["failures"].sum() != frame["target"].sum():
            raise ValueError("Deciles must partition the complete held-out cohort")
        kind_summary = {}
        for kind in ("raw", "calibrated"):
            probabilities = frame[f"{kind}_probability"].to_numpy()
            metrics = evaluate_probabilities(frame["target"], probabilities)
            for name in ("n", "prevalence", "roc_auc", "average_precision", "brier_score", "log_loss"):
                if not np.isclose(metrics[name], comparison.loc[(family, kind), name], rtol=0, atol=1e-12):
                    raise ValueError(f"Saved probabilities differ from model comparison: {family}/{kind}/{name}")
            ece = float(np.sum(bins["n"] / len(frame) * np.abs(bins[f"{kind}_mean_probability"] - bins["observed_fail_rate"])))
            counts, _ = np.histogram(probabilities, bins=histogram_edges)
            kind_summary[kind] = {
                "mean_probability": float(probabilities.mean()),
                "observed_fail_rate": float(frame["target"].mean()),
                "mean_probability_minus_observed_pp": float(100 * (probabilities.mean() - frame["target"].mean())),
                "brier_score": metrics["brier_score"], "log_loss": metrics["log_loss"],
                "ece_on_raw_score_deciles": ece,
                "histogram_counts": counts.tolist(),
            }
        run_path = args.metrics_dir / f"{family}_metrics.json"
        run = json.loads(run_path.read_text())
        if run["calibration"]["fit_period"] != ["2024-01-01", "2024-12-31"]:
            raise ValueError("Expected calibration fit period is 2024 only")
        if run["calibration"]["slope"] <= 0:
            raise ValueError("Expected an order-preserving sigmoid calibrator")
        model_summary[family] = {
            "n": len(frame), "failures": int(frame["target"].sum()),
            "observed_fail_rate": float(frame["target"].mean()),
            "actual_decile_count": len(bins), "decile_n_min": int(bins["n"].min()),
            "decile_n_max": int(bins["n"].max()),
            "primary_model": family == selection["selected_model"],
            "calibrator_fit_period": run["calibration"]["fit_period"],
            "calibrator_method": run["calibration"]["method"],
            "calibrator_positive_slope": run["calibration"]["slope"],
            "probability_variants": kind_summary,
            "prediction_file": str(path.relative_to(ROOT)), "prediction_sha256": sha256(path),
            "run_summary_sha256": sha256(run_path),
        }
        frames[family] = (frame, bins)
        all_bins.extend(family_bins)

    alignment = ["inspection_id", "inspection_date", "entity_id", "target"]
    if not frames[FAMILIES[0]][0][alignment].equals(frames[FAMILIES[1]][0][alignment]):
        raise ValueError("Both calibration figures must use the identical test cohort")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"], "axes.titlecolor": COLORS["text"],
        "xtick.color": COLORS["text"], "ytick.color": COLORS["text"],
        "axes.edgecolor": "#DCE1E5", "figure.facecolor": "#FAFBFC",
        "axes.facecolor": "white", "savefig.facecolor": "#FAFBFC",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5), gridspec_kw={"height_ratios": [1.25, 0.8]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.825, bottom=0.17, hspace=0.45, wspace=0.22)
    fig.suptitle("Probability calibration on held-out 2025 inspections", x=0.075, y=0.975,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.075, 0.929, "Models fitted on 2019–2022 · primary model selected on 2023 · sigmoid calibrators fitted only on 2024",
             fontsize=10, color="#536B80", ha="left")
    curve_limit = max(
        float(bins[["raw_mean_probability", "calibrated_mean_probability", "observed_fail_rate"]].max().max())
        for _, bins in frames.values()
    )
    curve_limit = max(0.5, np.ceil(curve_limit * 1.15 / 0.05) * 0.05)
    for col, family in enumerate(FAMILIES):
        frame, bins = frames[family]
        info = model_summary[family]
        ax, histogram = axes[:, col]
        ax.plot([0, curve_limit], [0, curve_limit], color=COLORS["gray"], lw=1.2,
                linestyle="--", label="Perfect calibration", zorder=1)
        for kind, label, marker in (("raw", "Raw", "o"), ("calibrated", "2024-calibrated", "s")):
            ax.plot(bins[f"{kind}_mean_probability"], bins["observed_fail_rate"], color=COLORS[kind],
                    marker=marker, markersize=4.6, linewidth=1.8, label=label,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            histogram.stairs(info["probability_variants"][kind]["histogram_counts"], histogram_edges,
                             color=COLORS[kind], linewidth=1.8, label=label, fill=False)
        ax.set(title=LABELS[family], xlabel="Mean predicted Fail probability", ylabel="Observed Fail rate",
               xlim=(0, curve_limit), ylim=(0, curve_limit))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        raw, cal = (info["probability_variants"][kind] for kind in ("raw", "calibrated"))
        metrics_text = (
            f"Mean p: {raw['mean_probability']:.1%} → {cal['mean_probability']:.1%} | actual {info['observed_fail_rate']:.1%}\n"
            f"Brier: {raw['brier_score']:.4f} → {cal['brier_score']:.4f}\n"
            f"Log loss: {raw['log_loss']:.4f} → {cal['log_loss']:.4f}"
        )
        ax.text(0.97, 0.035, metrics_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
                color="#536B80", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 3})
        histogram.set(title=f"Saved probability distribution · N = {len(frame):,}",
                      xlabel="Predicted Fail probability", ylabel="Inspection count", xlim=(0, 1))
        histogram.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        histogram.legend(loc="upper right", frameon=False, fontsize=8.5)
        histogram.text(0.98, 0.56, f"Calibration curve uses {len(bins)} raw-score deciles\n"
                       f"n per decile: {info['decile_n_min']}–{info['decile_n_max']}",
                       transform=histogram.transAxes, ha="right", va="top", fontsize=8.5, color="#536B80")
        for axis in (ax, histogram):
            axis.grid(color="#E5E9ED", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
            axis.tick_params(length=0, pad=6)

    fig.text(0.075, 0.075,
             "Curves use the same raw-score decile membership before and after calibration; each point is a bin mean. Exact counts are in calibration_bins.csv.",
             fontsize=8.8, color="#536B80", ha="left")
    fig.text(0.075, 0.044,
             "All outcomes shown are from 2025. Calibration is a probability adjustment, not a refit or a change to model ranking; curves are descriptive, without confidence bands.",
             fontsize=8.8, color="#536B80", ha="left")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300, pil_kwargs={"optimize": True})
    dimensions = [round(value * 300) for value in fig.get_size_inches()]
    plt.close(fig)
    bins_path = args.metrics_dir / "calibration_bins.csv"
    pd.DataFrame(all_bins).to_csv(bins_path, index=False)
    summary = {
        "schema_version": 1, "split": "test", "year": 2025,
        "selected_model": selection["selected_model"],
        "binning": "pandas.qcut of each model's 2025 raw probabilities into 10 quantile bins; duplicate edges dropped; raw and calibrated use identical membership.",
        "ece_definition": "sum over raw-score bins of (bin n / test n) * abs(bin mean probability - bin observed Fail rate), separately for each probability variant.",
        "ece_interpretation": "Descriptive bin-dependent calibration statistic on 2025; not used to choose a model, calibrator or threshold.",
        "fit_performed_by_this_script": False,
        "confidence_bands": "Not estimated; bin rates and curves are descriptive.",
        "histogram_probability_edges": histogram_edges.tolist(),
        "models": model_summary,
        "source_selection_sha256": sha256(selection_path),
        "source_comparison_sha256": sha256(comparison_path),
        "bin_table": str(bins_path.relative_to(ROOT)), "bin_table_sha256": sha256(bins_path),
        "figure": str(args.figure.relative_to(ROOT)), "figure_sha256": sha256(args.figure),
        "figure_bytes": args.figure.stat().st_size, "pixel_dimensions": dimensions, "dpi": 300,
    }
    output = args.metrics_dir / "calibration_summary.json"
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.figure.relative_to(ROOT)}: {dimensions[0]} x {dimensions[1]} px; {args.figure.stat().st_size / 1e6:.3f} MB")
    print(f"Saved {len(all_bins)} calibration-bin aggregates and calibration_summary.json; no fitting performed")
    for family, info in model_summary.items():
        raw, cal = (info["probability_variants"][kind] for kind in ("raw", "calibrated"))
        print(f"{family}: mean p {raw['mean_probability']:.6f} → {cal['mean_probability']:.6f}; "
              f"observed {info['observed_fail_rate']:.6f}; ECE {raw['ece_on_raw_score_deciles']:.6f} → {cal['ece_on_raw_score_deciles']:.6f}")


if __name__ == "__main__":
    main()
