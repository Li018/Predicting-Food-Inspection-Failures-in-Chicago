#!/usr/bin/env python3
"""Plot annual inspection aggregates already saved by Step 2; never read source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"text": "#19324D", "blue": "#247BA0", "orange": "#E47B34", "gray": "#9AA4AC"}
WINDOWS = (
    (2019, 2023, "Train\n2019–22", COLORS["blue"], 0.055),
    (2023, 2024, "Val\n2023", COLORS["orange"], 0.08),
    (2024, 2025, "Calib\n2024", COLORS["gray"], 0.13),
    (2025, 2026, "Test\n2025", COLORS["blue"], 0.11),
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_aggregates(summary: dict, annual: pd.DataFrame) -> None:
    """Check each plotted count against the saved all-year and split aggregates."""
    for annual_column, overall_column in (
        ("inspection_records", "inspection_records"),
        ("eligible_target_days", "eligible_target_days"),
        ("target_failures", "target_failures"),
    ):
        if int(annual[annual_column].sum()) != summary["overall"][overall_column]:
            raise ValueError(f"Annual and overall aggregate differ: {annual_column}")
    if not np.allclose(
        annual["target_failures"] / annual["eligible_target_days"], annual["target_fail_rate"],
        rtol=0, atol=1e-12,
    ):
        raise ValueError("Annual target counts do not reproduce failure rates")
    for name, split in summary["splits"].items():
        start, end = int(split["start"][:4]), int(split["end"][:4])
        subset = annual.loc[annual["year"].between(start, end)]
        if int(subset["eligible_target_days"].sum()) != split["rows"]:
            raise ValueError(f"Annual and split target count differ: {name}")
        if int(subset["target_failures"].sum()) != split["failures"]:
            raise ValueError(f"Annual and split failure count differ: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/metrics/data_summary.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "reports/figures/annual_drift.png")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    annual = pd.DataFrame(summary["annual"]).sort_values("year").reset_index(drop=True)
    validate_aggregates(summary, annual)
    metric_dir = args.summary.parent
    csv_path = metric_dir / "annual_drift.csv"
    annual.to_csv(csv_path, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"], "axes.edgecolor": "#DCE1E5",
        "axes.titlecolor": COLORS["text"], "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"], "figure.facecolor": "#FAFBFC",
        "axes.facecolor": "white", "savefig.facecolor": "#FAFBFC",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.18, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.845, bottom=0.18, hspace=0.27)
    fig.suptitle("Chicago food inspections: volume and failure-rate drift", x=0.09, y=0.975,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(0.09, 0.93, "Official retrospective projection, 2010–2025  |  Annual aggregates of observed inspections",
             ha="left", fontsize=10.5, color="#536B80")

    years = annual["year"].to_numpy()
    centers = years + 0.5
    for ax in axes:
        for start, end, label, color, alpha in WINDOWS:
            ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0, zorder=0)
        ax.axvline(2018.5, color=COLORS["gray"], linestyle=(0, (4, 3)), linewidth=1.3, zorder=1)
        ax.grid(axis="y", color="#E5E9ED", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_xlim(int(years.min()), int(years.max()) + 1)
        ax.tick_params(axis="both", length=0, pad=6)

    axes[0].bar(centers - 0.16, annual["inspection_records"], width=0.29,
                color=COLORS["blue"], label="All inspection records", zorder=3)
    axes[0].bar(centers + 0.16, annual["eligible_target_days"], width=0.29,
                color=COLORS["orange"], label="Eligible Canvass targets", zorder=3)
    axes[0].set_ylabel("Annual count")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    axes[0].set_ylim(0, 31000)
    axes[0].set_yticks([0, 10000, 20000, 30000])
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.5, ncol=1,
                   bbox_to_anchor=(0.005, 0.99), borderaxespad=0.25)
    axes[0].text(2018.5 - 0.12, 28800, "1 Jul 2018\nrule change", ha="right", va="top",
                 fontsize=9, color="#657987", linespacing=1.25)
    for start, end, label, _, _ in WINDOWS:
        axes[0].text((start + end) / 2, 29500, label, ha="center", va="top", fontsize=9,
                     fontweight="bold", linespacing=1.25)

    axes[1].plot(centers, annual["target_fail_rate"], color=COLORS["blue"], linewidth=2.1,
                 marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    axes[1].set_ylabel("Canvass target Fail rate")
    axes[1].set_ylim(0, 0.345)
    axes[1].set_yticks([0, 0.1, 0.2, 0.3])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_xticks(centers, labels=[str(year) for year in years], fontsize=9)
    axes[1].set_xlabel("Inspection calendar year", labelpad=9)
    for year in (2019, 2022, 2023, 2024, 2025):
        value = float(annual.loc[annual["year"].eq(year), "target_fail_rate"].iloc[0])
        axes[1].annotate(f"{value:.1%}", xy=(year + 0.5, value), xytext=(0, 10),
                         textcoords="offset points", ha="center", fontsize=8.5,
                         color=COLORS["text"])

    fig.text(0.09, 0.075,
             "Eligible target: exact Canvass; Fail vs Pass / Pass w/ Conditions; one target per establishment-day.",
             ha="left", fontsize=9, color="#536B80")
    fig.text(0.09, 0.045,
             "Annual comparisons are descriptive. They do not identify a causal effect of the July 2018 rule change.",
             ha="left", fontsize=9, color="#536B80")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=300, pil_kwargs={"optimize": True}, metadata={"Title": "Annual inspection drift, 2010–2025"})
    pixel_dimensions = [round(v * 300) for v in fig.get_size_inches()]
    plt.close(fig)
    manifest = {
        "schema_version": 1,
        "figure": str(args.figure.relative_to(ROOT)),
        "source": str(args.summary.relative_to(ROOT)),
        "source_sha256": file_sha256(args.summary),
        "aggregate_csv": str(csv_path.relative_to(ROOT)),
        "aggregate_csv_sha256": file_sha256(csv_path),
        "figure_sha256": file_sha256(args.figure),
        "figure_bytes": args.figure.stat().st_size,
        "pixel_dimensions": pixel_dimensions, "dpi": 300,
        "calendar_years": [int(years.min()), int(years.max())],
        "annual_rows": len(annual), "rule_change_date": "2018-07-01",
        "year_geometry": "Annual bars and rates are centered at year + 0.5; period boundaries are integer years.",
        "validation": "Annual record/target/failure sums match overall and all four split aggregates.",
        "interpretation": "Descriptive observed-inspection drift; not a causal rule-change estimate.",
    }
    manifest_path = metric_dir / "annual_drift_figure.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.figure.relative_to(ROOT)}: {pixel_dimensions[0]} x {pixel_dimensions[1]} px; {args.figure.stat().st_size / 1e6:.3f} MB")
    print(f"Validated {len(annual)} annual aggregate rows against overall and all temporal splits")
    print(f"Saved {csv_path.relative_to(ROOT)} and {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
