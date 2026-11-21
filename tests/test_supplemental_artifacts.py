"""Verify published supplemental evidence without requiring private row artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from chicago_food.feature_groups import FEATURE_GROUPS
from chicago_food.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"


def payload(name):
    return json.loads((METRICS / name).read_text())


def table(document, marker):
    text = (ROOT / document).read_text()
    block = text.split(f"<!-- {marker} -->", 1)[1].split(f"<!-- /{marker} -->", 1)[0]
    lines = [line for line in block.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def numeric(text):
    return float(text.replace(",", "").replace("%", "").replace("pp", "").strip())


def test_published_provenance_preserves_frozen_sources_and_core_code():
    manifest_path = ROOT / "data/snapshots/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_hashes = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    original = payload("shap_summary.json")["provenance"]
    assert source_hashes == original["source_sha256"]
    assert payload("feature_ablation.json")["provenance"]["source_sha256"] == source_hashes
    seasonality = payload("seasonality_complaint.json")["provenance"]
    assert seasonality["inspections_sha256"] == source_hashes["inspections.parquet"]
    assert seasonality["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for name in ("grouped_shap_summary.json", "lr_odds_ratio_summary.json",
                 "subgroup_audit_summary.json", "days_earlier_summary.json"):
        summary = payload(name)
        assert summary["refitted"] is False
        inputs = summary["input_sha256"]
        for filename, digest in source_hashes.items():
            assert inputs[f"data/snapshots/{filename}"] == digest
        for filename, digest in {**original["feature_code_sha256"], **original["model_code_sha256"]}.items():
            path = f"src/chicago_food/{filename}"
            assert inputs[path] == digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert inputs[".local/features.parquet"] == original["feature_sha256"]
        assert inputs[".local/models/logistic_regression.joblib"] == payload("feature_ablation.json")["provenance"]["frozen_primary_model_sha256"]


def test_frozen_ablation_reference_equals_original_published_primary_metrics():
    ablation = pd.read_csv(METRICS / "feature_ablation.csv")
    original = pd.DataFrame(payload("logistic_regression_metrics.json")["metrics"])
    reference = ablation.loc[ablation["is_frozen_reference"]]
    assert len(reference) == 3
    for _, row in reference.iterrows():
        existing = original.loc[
            original["model"].eq("logistic_regression") & original["split"].eq(row["split"])
            & original["probability_kind"].eq(row["probability_kind"])
        ].iloc[0]
        for metric in ("n", "failures", "average_precision", "roc_auc", "brier_score"):
            assert row[metric] == pytest.approx(existing[metric], abs=1e-12)
        assert row["feature_count"] == 39
        assert pd.isna(row["delta_average_precision"])


def test_ablation_tables_keep_nested_features_and_comparable_deltas():
    summary = payload("feature_ablation.json")
    published = {name: item["features"] for name, item in summary["feature_groups"].items()}
    assert published == {name: list(fields) for name, fields in FEATURE_GROUPS.items()}
    features = [selection["numeric_features"] + selection["categorical_features"]
                for selection in summary["selections"]]
    assert [len(row) for row in features] == [13, 27, 37]
    assert set(features[0]) < set(features[1]) < set(features[2])
    assert set(FEATURE_COLUMNS) - set(features[2]) == {"prior_facility_type", "prior_risk"}
    rows = pd.read_csv(METRICS / "feature_ablation.csv")
    assert not (rows["split"].eq("validation") & rows["probability_kind"].eq("calibrated")).any()
    for _, grouped in rows.loc[~rows["is_frozen_reference"]].groupby(["split", "probability_kind"]):
        grouped = grouped.sort_values("stage_order")
        for metric in ("average_precision", "roc_auc", "brier_score"):
            assert np.allclose(grouped[f"delta_{metric}"].iloc[1:], grouped[metric].diff().iloc[1:])


def test_seasonal_aggregates_reconcile_to_months_and_original_canvass_targets():
    annual = pd.read_csv(METRICS / "inspection_type_annual.csv")
    monthly = pd.read_csv(METRICS / "inspection_type_monthly.csv")
    seasonal = pd.read_csv(METRICS / "inspection_type_seasonality.csv")
    rolled = monthly.groupby(["inspection_type", "year"])[["n", "failures"]].sum().sort_index()
    pd.testing.assert_frame_equal(rolled, annual.set_index(["inspection_type", "year"])[["n", "failures"]].sort_index())
    for frame in (annual, monthly, seasonal):
        assert np.allclose(frame["fail_rate"], frame["failures"] / frame["n"])
    original = pd.read_csv(METRICS / "annual_drift.csv").set_index("year")
    canvass = annual.loc[annual["inspection_type"].eq("Canvass")].set_index("year")
    assert np.array_equal(canvass["n"], original.loc[canvass.index, "eligible_target_days"])
    assert np.array_equal(canvass["failures"], original.loc[canvass.index, "target_failures"])
    post = monthly.loc[monthly["year"].between(2019, 2025)]
    expected = post.groupby(["inspection_type", "month"])[["n", "failures"]].sum().sort_index()
    pd.testing.assert_frame_equal(expected, seasonal.set_index(["inspection_type", "month"])[["n", "failures"]].sort_index())
    equal_year = post.groupby(["inspection_type", "month"])["fail_rate"].mean()
    assert np.allclose(equal_year, seasonal.set_index(["inspection_type", "month"])["equal_year_mean_fail_rate"])


def test_grouped_shap_tables_preserve_partition_and_cancellation_accounting():
    summary = payload("grouped_shap_summary.json")
    assert summary["feature_groups"] == {name: list(fields) for name, fields in FEATURE_GROUPS.items()}
    importance = pd.read_csv(METRICS / "grouped_shap_importance.csv")
    for scope, rows in importance.groupby("scope"):
        assert rows["feature_count"].sum() == 39
        assert set(rows["group"]) == set(FEATURE_GROUPS)
        expected = 1 - rows["mean_abs_group_shap"] / rows["sum_feature_mean_abs_shap"]
        assert np.allclose(rows["within_group_cancellation_fraction"], expected)
        assert rows["mean_abs_group_shap"].sum() == pytest.approx(summary[scope]["sum_group_mean_abs_shap"])
        assert summary[scope]["sum_max_absolute_error"] < 1e-12
        assert np.all(rows["mean_abs_group_shap"] + 1e-12 >= rows["mean_signed_shap"].abs())
    assert summary["verification"]["grouped_probability_max_absolute_error"] < 1e-12


def test_standardized_odds_ratios_are_exponentiated_calibrated_coefficients():
    rows = pd.read_csv(METRICS / "lr_standardized_odds_ratios.csv")
    summary = payload("lr_odds_ratio_summary.json")
    assert len(rows) == summary["numeric_count"] == 32
    assert np.allclose(rows["calibrated_beta_per_train_sd"], rows["raw_beta_per_train_sd"] * summary["calibrator_slope"])
    assert np.allclose(rows["raw_odds_ratio_per_train_sd"], np.exp(rows["raw_beta_per_train_sd"]))
    assert np.allclose(rows["calibrated_odds_ratio_per_train_sd"], np.exp(rows["calibrated_beta_per_train_sd"]))
    assert np.all(rows["train_sd"] > 0)


def test_subgroup_capacity_and_ece_reconcile_from_published_bin_statistics():
    summary = payload("subgroup_audit_summary.json")
    rows = pd.read_csv(METRICS / "subgroup_audit.csv")
    bins = pd.read_csv(METRICS / "subgroup_audit_calibration_bins.csv")
    for _, group in rows.groupby("feature"):
        assert group["n"].sum() == summary["n"] == 8363
        assert group["failures"].sum() == summary["failures"] == 1886
        assert group["global_top10_selected_n"].sum() == pytest.approx(summary["global_capacity"])
    assert summary["global_capacity"] == 837
    for _, row in rows.iterrows():
        b = bins.loc[bins["feature"].eq(row["feature"]) & bins["value"].eq(row["value"])]
        assert b["n"].sum() == row["n"]
        assert np.average(b["absolute_error"], weights=b["n"]) == pytest.approx(row["ece_10_equal_width"])
        assert np.average(b["mean_probability"], weights=b["n"]) == pytest.approx(row["mean_probability"])
        assert (b["observed_rate"] * b["n"]).sum() == pytest.approx(row["failures"])
        assert row["global_top10_selected_rate"] * row["n"] == pytest.approx(row["global_top10_selected_n"])
        assert row["calibration_bias"] == pytest.approx(row["mean_probability"] - row["prevalence"])
        if row["failures"] in (0, row["n"]):
            assert pd.isna(row["average_precision"]) and pd.isna(row["roc_auc"])


def test_days_earlier_uses_all_fail_weighting_and_six_contiguous_windows():
    rows = pd.read_csv(METRICS / "days_earlier_windows.csv")
    summary = payload("days_earlier_summary.json")
    assert len(rows) == summary["window_count"] == 6
    assert rows["n"].sum() == summary["n"] == 8363
    assert rows["positive_count"].sum() == summary["positive_count"] == 1886
    for column, field in [("days_earlier", "pooled_days_earlier"),
                          ("delayed_positive_fraction", "pooled_delayed_positive_fraction"),
                          ("advanced_positive_fraction", "pooled_advanced_positive_fraction"),
                          ("random_days_earlier", "random_days_earlier")]:
        assert np.average(rows[column], weights=rows["positive_count"]) == pytest.approx(summary[field])
    assert summary["pooled_delayed_positive_fraction"] > 0
    assert summary["daily_capacity_max_absolute_error"] == 0
    assert summary["original_day_score_retrospective"] is True
    starts, ends = pd.to_datetime(rows["window_start"]), pd.to_datetime(rows["window_end"])
    assert starts.iloc[0] == pd.Timestamp("2025-01-01")
    assert ends.iloc[-1] == pd.Timestamp("2025-12-31")
    assert np.array_equal(starts.iloc[1:].to_numpy(), (ends.iloc[:-1] + pd.Timedelta(days=1)).to_numpy())


@pytest.mark.parametrize("document", ["README.md", "reports/topic1_report.md"])
def test_documented_ablation_table_matches_exact_published_values(document):
    rows = pd.read_csv(METRICS / "feature_ablation.csv")
    stages = ["history_only", "history_time", "history_time_license", "frozen_primary_39"]
    documented = table(document, "supplemental-ablation-table")
    assert len(documented) == 4
    for cells, stage in zip(documented, stages):
        actual = rows.loc[rows["stage"].eq(stage) & rows["split"].eq("test") & rows["probability_kind"].eq("calibrated")].iloc[0]
        validation = rows.loc[rows["stage"].eq(stage) & rows["split"].eq("validation")].iloc[0]
        assert numeric(cells[1]) == actual["feature_count"]
        for index, value in [(2, validation["average_precision"]), (3, actual["average_precision"]),
                             (5, actual["roc_auc"]), (6, actual["brier_score"])]:
            assert numeric(cells[index]) == round(value, 6)
        if pd.isna(actual["delta_average_precision"]):
            assert cells[4] == "—"
        else:
            assert numeric(cells[4]) == round(actual["delta_average_precision"], 6)


def test_documented_group_and_or_tables_match_published_values():
    rows = pd.read_csv(METRICS / "grouped_shap_importance.csv")
    documented = table("reports/topic1_report.md", "supplemental-grouped-shap-table")
    for cells, group in zip(documented, FEATURE_GROUPS):
        fixed = rows.loc[rows["group"].eq(group) & rows["scope"].eq("fixed_explanation_sample")].iloc[0]
        full = rows.loc[rows["group"].eq(group) & rows["scope"].eq("full_test")].iloc[0]
        assert numeric(cells[1]) == fixed["feature_count"]
        assert numeric(cells[2]) == round(fixed["mean_abs_group_shap"], 6)
        assert numeric(cells[3]) == round(full["mean_abs_group_shap"], 6)
        assert numeric(cells[4]) == round(100 * fixed["within_group_cancellation_fraction"], 2)
    odds = pd.read_csv(METRICS / "lr_standardized_odds_ratios.csv").set_index("feature")
    for cells in table("reports/topic1_report.md", "supplemental-odds-ratio-table"):
        row = odds.loc[cells[0].strip("`")]
        for index, field in [(1, "raw_beta_per_train_sd"), (2, "calibrated_beta_per_train_sd"),
                             (3, "calibrated_odds_ratio_per_train_sd")]:
            assert numeric(cells[index]) == round(row[field], 6)


def test_documented_subgroup_and_days_tables_match_published_values():
    rows = pd.read_csv(METRICS / "subgroup_audit.csv")
    for cells in table("reports/topic1_report.md", "supplemental-subgroup-table"):
        feature, value = cells[0].split("／", 1)
        row = rows.loc[rows["feature"].eq(feature.strip("`")) & rows["value"].eq(value)].iloc[0]
        assert numeric(cells[1]) == row["n"]
        assert numeric(cells[2]) == row["failures"]
        assert numeric(cells[3]) == round(row["average_precision"], 4)
        assert numeric(cells[4]) == round(100 * row["ece_10_equal_width"], 2)
        assert numeric(cells[5]) == round(100 * row["global_top10_selected_rate"], 2)
        assert (cells[6] == "是") == row["small_sample"]
    windows = pd.read_csv(METRICS / "days_earlier_windows.csv")
    documented = table("reports/topic1_report.md", "supplemental-days-earlier-table")
    assert len(documented) == 7
    for cells, (_, row) in zip(documented, windows.iterrows()):
        assert numeric(cells[1]) == row["n"]
        assert numeric(cells[2]) == row["positive_count"]
        assert numeric(cells[3]) == round(row["days_earlier"], 3)
        assert numeric(cells[4]) == round(100 * row["delayed_positive_fraction"], 2)
    summary = payload("days_earlier_summary.json")
    assert numeric(documented[-1][3]) == round(summary["pooled_days_earlier"], 3)


def test_supplemental_document_links_resolve_in_a_source_only_checkout():
    for document, marker in [("README.md", "## 9."), ("reports/topic1_report.md", "## 14.")]:
        path = ROOT / document
        text = path.read_text().split(marker, 1)[1]
        destinations = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        for destination in destinations:
            if "://" not in destination and not destination.startswith("#"):
                assert (path.parent / destination.split("#", 1)[0]).exists(), destination
