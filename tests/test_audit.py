"""Independent examples for supplemental interpretation and allocation semantics."""

import numpy as np
import pandas as pd
import pytest

from chicago_food.audit import (
    calibration_error,
    fixed_window_reordering,
    grouped_shap,
    odds_ratio_tables,
    subgroup_audit,
    verify_explanation,
)
from chicago_food.models import MonotoneSigmoidCalibrator, make_pipeline


def test_grouped_shap_sums_signed_values_before_absolute_magnitude():
    phi = np.array([[4.0, -3.0, 2.0], [-2.0, 1.0, -3.0]])
    grouped, table, summary = grouped_shap(phi, ["a", "b", "c"], {"history": ("a", "b"), "other": ("c",)})
    np.testing.assert_allclose(grouped, [[1, 2], [-1, -3]])
    assert table.loc[0, "mean_abs_group_shap"] == 1
    assert table.loc[0, "sum_feature_mean_abs_shap"] == 5
    assert table.loc[0, "within_group_cancellation_fraction"] == pytest.approx(0.8)
    assert summary["sum_max_absolute_error"] == 0
    np.testing.assert_allclose(grouped.sum(axis=1), phi.sum(axis=1))


@pytest.mark.parametrize("groups", [{"g": ("a",)}, {"g": ("a", "a", "b")}, {"g": ("a", "b", "extra")}])
def test_grouped_shap_rejects_omission_overlap_and_unknown_features(groups):
    with pytest.raises(ValueError, match="partition"):
        grouped_shap([[1, 2]], ["a", "b"], groups)


def test_equal_width_ece_uses_last_endpoint_and_all_observations():
    ece, bins = calibration_error([0, 1, 0, 1], [0, 0.1, 0.2, 1])
    assert bins["n"].sum() == 4
    assert bins["bin"].tolist() == [0, 1, 2, 9]
    assert ece == pytest.approx((0 + 0.9 + 0.2 + 0) / 4)


def test_subgroups_use_one_global_allocation_not_each_groups_top_k():
    frame = pd.DataFrame({"target": [0, 1, 0, 1], "prior_risk": ["Risk 1", "Risk 1", "Risk 2", "Risk 2"]})
    table, _, summary = subgroup_audit(frame, [0.9, 0.8, 0.2, 0.1], columns=("prior_risk",))
    by_group = table.set_index("value")
    assert summary["global_capacity"] == 1
    assert by_group.loc["Risk 1", "global_top10_selected_rate"] == 0.5
    assert by_group.loc["Risk 2", "global_top10_selected_rate"] == 0
    assert table["global_top10_selected_n"].sum() == 1


def test_subgroup_single_class_ap_is_undefined_and_ties_are_fractional():
    frame = pd.DataFrame({"target": [0, 0, 1, 1], "prior_risk": ["A", "A", "B", "B"]})
    table, _, _ = subgroup_audit(frame, [0.5] * 4, columns=("prior_risk",))
    assert table["average_precision"].isna().all()
    assert table["global_top10_selected_rate"].tolist() == [0.25, 0.25]


def test_reordering_keeps_negative_advances_and_calendar_day_units():
    table, summary = fixed_window_reordering(["2025-01-01", "2025-02-01"], [0.1, 0.9], [1, 0])
    assert table.iloc[0]["days_earlier"] == -31
    assert summary["pooled_days_earlier"] == -31
    assert summary["pooled_delayed_positive_fraction"] == 1
    assert summary["daily_capacity_max_absolute_error"] == 0
    assert summary["window_count"] == 6


def test_reordering_uses_fixed_windows_and_positive_weighted_pooling():
    table, summary = fixed_window_reordering(
        ["2025-01-01", "2025-02-01", "2025-03-01", "2025-03-03", "2025-03-05"],
        [0.1, 0.9, 0.1, 0.8, 0.9], [0, 1, 0, 1, 1],
    )
    assert table.iloc[0]["days_earlier"] == 31
    assert table.iloc[1]["days_earlier"] == 2
    assert summary["pooled_days_earlier"] == pytest.approx(35 / 3)
    assert table.iloc[2:]["days_earlier"].isna().all()
    assert table["n"].sum() == 5


def test_reordering_ties_use_uniform_expected_dates_without_row_order_bias():
    dates, scores, labels = np.array(["2025-01-01", "2025-01-10", "2025-01-20"]), np.array([0.4, 0.4, 0.1]), np.array([1, 0, 0])
    _, first = fixed_window_reordering(dates, scores, labels)
    permutation = [2, 1, 0]
    _, second = fixed_window_reordering(dates[permutation], scores[permutation], labels[permutation])
    assert first["pooled_days_earlier"] == -4.5
    assert first["pooled_delayed_positive_fraction"] == 0.5
    assert first == second


def test_reordering_empty_positive_set_remains_undefined():
    table, summary = fixed_window_reordering(["2025-01-02", "2025-01-03"], [0.8, 0.2], [0, 0])
    assert summary["pooled_days_earlier"] is None
    assert summary["random_days_earlier"] is None
    assert table["days_earlier"].isna().all()


def test_numeric_odds_ratios_equal_a_one_training_sd_prediction_change():
    # A small independent fit checks units only; the production audit never fits.
    frame = pd.DataFrame({"history": [0, 1, 2, 3, 4, 5, 6, 7]})
    model = make_pipeline("logistic_regression", ["history"], [], seed=1)
    model.fit(frame, [0, 0, 1, 0, 1, 1, 1, 1])
    calibrator = MonotoneSigmoidCalibrator()
    calibrator.slope_, calibrator.intercept_ = 0.7, -0.1
    bundle = {"pipeline": model, "calibrator": calibrator, "numeric_features": ["history"], "categorical_features": []}
    numeric, _ = odds_ratio_tables(bundle, {"history": ("history",)})
    sd = numeric.iloc[0]["train_sd"]
    pair = pd.DataFrame({"history": [2, 2 + sd]})
    logits = model.decision_function(pair)
    assert numeric.iloc[0]["raw_beta_per_train_sd"] == pytest.approx(logits[1] - logits[0])
    assert numeric.iloc[0]["calibrated_odds_ratio_per_train_sd"] == pytest.approx(np.exp(0.7 * (logits[1] - logits[0])))


def test_missing_indicator_odds_multiplier_undoes_standardization():
    frame = pd.DataFrame({"history": [0, 1, np.nan, 3, 4, 5, 6, np.nan]})
    model = make_pipeline("logistic_regression", ["history"], [], seed=1)
    model.fit(frame, [0, 0, 1, 0, 1, 1, 1, 1])
    calibrator = MonotoneSigmoidCalibrator()
    calibrator.slope_, calibrator.intercept_ = 0.7, -0.1
    bundle = {"pipeline": model, "calibrator": calibrator, "numeric_features": ["history"], "categorical_features": []}
    _, encoded = odds_ratio_tables(bundle, {"history": ("history",)})
    indicator = encoded.loc[encoded["kind"].eq("missing_indicator")].iloc[0]
    scaler = model.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    assert indicator["raw_log_odds_change"] == pytest.approx(model.named_steps["model"].coef_[0, 1] / scaler.scale_[1])


def test_saved_explanation_aligns_by_id_without_fitting_and_rejects_wrong_labels(monkeypatch):
    frame = pd.DataFrame({"history": [0, 1, 2, 3], "inspection_id": ["a", "b", "c", "d"],
                          "entity_id": ["w", "x", "y", "z"],
                          "inspection_date": pd.to_datetime(["2025-01-01"] * 4),
                          "target": [0, 0, 1, 1]})
    model = make_pipeline("logistic_regression", ["history"], [], seed=1)
    model.fit(frame, frame["target"])
    calibrator = MonotoneSigmoidCalibrator()
    calibrator.slope_, calibrator.intercept_ = 0.7, -0.1
    raw = model.predict_proba(frame)[:, 1]
    transformed = model.named_steps["preprocess"].transform(frame)
    phi = transformed * model.named_steps["model"].coef_[0] * calibrator.slope_
    base = calibrator.slope_ * model.named_steps["model"].intercept_[0] + calibrator.intercept_
    bundle = {"pipeline": model, "calibrator": calibrator, "numeric_features": ["history"], "categorical_features": []}
    explanation = {"test_frame": frame, "feature_names": ["history"], "shap_values": phi,
                   "base_value": base, "calibrated_probability": calibrator.predict(raw)}
    saved = frame.drop(columns="history").assign(
        split="test", raw_probability=raw, calibrated_probability=calibrator.predict(raw))
    monkeypatch.setattr(model, "fit", lambda *_args, **_kwargs: pytest.fail("Frozen audit refitted a model"))
    aligned, verification = verify_explanation(explanation, saved.iloc[::-1], bundle)
    assert aligned["inspection_id"].tolist() == ["a", "b", "c", "d"]
    assert max(verification.values()) < 1e-12
    saved.loc[0, "target"] = 1
    with pytest.raises(ValueError, match="target"):
        verify_explanation(explanation, saved, bundle)
