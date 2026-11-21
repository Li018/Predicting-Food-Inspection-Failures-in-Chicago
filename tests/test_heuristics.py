"""Rule baselines share the original capacity policy and fair tie treatment."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chicago_food.heuristics import add_heuristic_slide_metrics, align_test_data, ranking_metrics, rule_scores
from chicago_food.metrics import evaluate_probabilities
from chicago_food.slide_validation import validate_metric_contract, verify_aggregate_sources


def test_rules_use_visible_history_and_keep_unknown_cases():
    frame = pd.DataFrame({
        "last_result": ["Fail", "Pass", "Pass w/ Conditions", "Unknown"],
        "historical_failure_rate": [0.5, 0.25, 1.0, np.nan],
        "prior_fail_days": [2.0, 4.0, 1.0, 0.0],
    })
    actual = rule_scores(frame)
    np.testing.assert_array_equal(actual["prior_last_fail"], [1, 0, 0, 0])
    np.testing.assert_array_equal(actual["prior_fail_rate"], [0.5, 0.25, 1, 0])
    np.testing.assert_array_equal(actual["prior_failed_days"], [2, 4, 1, 0])


def test_constant_rule_has_random_expected_yield_not_row_order_yield():
    labels = np.zeros(20)
    labels[[0, 1, 10]] = 1
    dates = ["2025-01-01"] * 10 + ["2025-01-02"] * 10
    result = ranking_metrics(labels, np.zeros(20), dates)
    assert result["roc_auc"] == 0.5
    assert result["average_precision"] == pytest.approx(0.15)
    assert result["top_10pct_precision"] == pytest.approx(0.15)
    assert result["daily_top_10pct_expected_failures_found"] == pytest.approx(0.3)
    assert result["daily_top_10pct_lift_vs_daily_random"] == pytest.approx(1)
    assert result["annual_boundary_tie_size"] == 20
    assert result["annual_boundary_slots"] == 2


def test_boundary_tie_allocates_the_remaining_slot_uniformly():
    scores = np.array([5, 4, 4, 4, 4] + [1] * 15)
    labels = np.array([1, 1, 1, 0, 0] + [0] * 15)
    result = ranking_metrics(labels, scores, ["2025-01-01"] * 20)
    assert result["top_10pct_selected_n"] == 2
    assert result["top_10pct_expected_failures_found"] == pytest.approx(1.5)
    assert result["top_10pct_precision"] == pytest.approx(0.75)
    assert result["daily_top_10pct_expected_failures_found"] == pytest.approx(1.5)
    assert result["daily_top_10pct_lift_vs_daily_random"] == pytest.approx(5)
    assert result["annual_boundary_tie_size"] == 4
    assert result["annual_boundary_slots"] == 1


def test_tie_aware_results_are_invariant_to_input_permutation():
    scores = np.array([0, 0, 1, 1, 1, 2, 2, 2, 4, 4] * 3)
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 3)
    dates = np.array(["2025-01-01"] * 7 + ["2025-01-02"] * 11 + ["2025-01-03"] * 12)
    expected = ranking_metrics(labels, scores, dates)
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(scores))
        actual = ranking_metrics(labels[order], scores[order], dates[order])
        assert actual == pytest.approx(expected)


def test_ranking_metrics_match_original_lr_daily_capacity_conventions():
    labels = np.array([1, 1, 1, 0, 0] + [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0] + [0, 0, 0])
    scores = np.array([9, 5, 5, 2, 0] + [4, 4, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0] + [5, 5, 5])
    dates = ["2025-01-01"] * 5 + ["2025-01-02"] * 12 + ["2025-01-03"] * 3
    actual = ranking_metrics(labels, scores, dates)
    original = evaluate_probabilities(labels, scores / scores.max(), dates=dates, top_fractions=(0.1,))
    for key in ("roc_auc", "average_precision", "top_10pct_precision", "top_10pct_selected_n",
                "daily_top_10pct_selected_n", "daily_top_10pct_expected_failures_found",
                "daily_top_10pct_random_expected_failures_found", "daily_top_10pct_lift_vs_daily_random"):
        assert actual[key] == pytest.approx(original[key])
    assert actual["daily_top_10pct_selected_n"] == 4
    assert actual["daily_top_10pct_random_expected_failures_found"] == pytest.approx(1.1)


def sample_alignment_frames():
    features = pd.DataFrame({"inspection_id": [2, 1], "entity_id": ["b", "a"],
                             "inspection_date": pd.to_datetime(["2025-01-02", "2025-01-01"]),
                             "target": [0, 1], "last_result": ["Unknown", "Fail"],
                             "historical_failure_rate": [np.nan, 0.5], "prior_fail_days": [0.0, 2.0]})
    predictions = features[["inspection_id", "entity_id", "inspection_date", "target"]].iloc[::-1].copy()
    predictions["split"] = "test"
    predictions["calibrated_probability"] = [0.4, 0.2]
    return features, predictions


def test_saved_predictions_align_by_id_instead_of_source_order():
    features, predictions = sample_alignment_frames()
    aligned, probabilities = align_test_data(features, predictions)
    assert aligned["inspection_id"].tolist() == [1, 2]
    np.testing.assert_array_equal(probabilities, [0.4, 0.2])


@pytest.mark.parametrize("change", ["drop_row", "target", "inspection_date"])
def test_saved_prediction_cohort_or_label_drift_is_rejected(change):
    features, predictions = sample_alignment_frames()
    if change == "drop_row":
        features = features.iloc[:1]
    elif change == "target":
        features.loc[0, "target"] = 1
    else:
        features.loc[0, "inspection_date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="cohorts differ|alignment differs"):
        align_test_data(features, predictions)


def test_published_rule_results_preserve_lr_reference_and_common_capacities():
    metrics = Path(__file__).resolve().parents[1] / "reports/metrics"
    table = pd.read_csv(metrics / "heuristic_baselines.csv").set_index("model")
    summary = json.loads((metrics / "heuristic_baselines.json").read_text())
    original = pd.read_csv(metrics / "model_comparison.csv")
    lr = original.loc[original["model"].eq("logistic_regression") & original["probability_kind"].eq("calibrated")].iloc[0]
    assert set(table.index) == {"prior_last_fail", "prior_fail_rate", "prior_failed_days", "logistic_regression"}
    assert table["n"].eq(8363).all() and table["failures"].eq(1886).all()
    assert table["top_10pct_selected_n"].eq(837).all()
    assert table["daily_top_10pct_selected_n"].eq(948).all()
    for key in ("roc_auc", "average_precision", "top_10pct_precision", "daily_top_10pct_expected_failures_found", "daily_top_10pct_lift_vs_daily_random"):
        assert table.loc["logistic_regression", key] == pytest.approx(lr[key], abs=1e-10)
    assert summary["best_heuristic"] == table.drop("logistic_regression")["average_precision"].idxmax()
    assert summary["refitted"] is False
    assert set(record["model"] for record in summary["records"]) == set(table.index)


def test_slide_rule_append_preserves_all_existing_facts_and_is_idempotent():
    root = Path(__file__).resolve().parents[1]
    published = json.loads((root / "reports/metrics/slide_metrics.json").read_text())
    original = deepcopy(published)
    contract_maps = ("values", "display", "source_refs", "format_specs")
    for name in contract_maps:
        original[name] = {key: value for key, value in original[name].items() if not key.startswith("heuristics.")}
    original["source_hashes"] = {key: value for key, value in original["source_hashes"].items()
                                 if not key.startswith("reports/metrics/heuristic_baselines.")}
    before = deepcopy(original)
    actual = add_heuristic_slide_metrics(original, root)
    assert original == before
    for name, value in before.items():
        if name in (*contract_maps, "source_hashes"):
            assert all(actual[name][key] == existing for key, existing in value.items())
        else:
            assert actual[name] == value
    assert add_heuristic_slide_metrics(actual, root) == actual
    assert validate_metric_contract(actual) == len(before["display"]) + 18
    assert verify_aggregate_sources(root, actual) == len(before["source_hashes"]) + 2
    expected = {
        "prior_last_fail": ("0.500", "0.225", "22.4%", "213.2", "−0.5%"),
        "prior_fail_rate": ("0.583", "0.283", "31.6%", "307.1", "+43.4%"),
        "prior_failed_days": ("0.549", "0.260", "30.2%", "293.8", "+37.2%"),
    }
    fields = ("auc", "ap", "top10_precision", "daily_top10_failures", "daily_top10_lift_gain")
    for model, displays in expected.items():
        for field, display in zip(fields, displays, strict=True):
            key = f"heuristics.{model}.{field}"
            assert actual["display"][key] == display
            assert all(published[name][key] == actual[name][key] for name in contract_maps)
    assert actual["values"]["heuristics.best_rule"] == "prior_fail_rate"
    assert actual["values"]["heuristics.rule_count"] == 3
