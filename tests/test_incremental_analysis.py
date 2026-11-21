"""Regression checks for additive analysis, without editing the original tests."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from chicago_food.ablation import ABLATION_STAGES, add_stage_deltas, run_ablation, stage_features
from chicago_food.data import clean_inspections
from chicago_food.feature_groups import FEATURE_GROUPS
from chicago_food.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, select_targets
from chicago_food.seasonality import aggregate_trends, comparison_targets


def test_business_groups_cover_every_original_field_once():
    flattened = [feature for fields in FEATURE_GROUPS.values() for feature in fields]
    assert len(flattened) == len(set(flattened)) == 39
    assert set(flattened) == set(FEATURE_COLUMNS)
    assert [len(fields) for fields in FEATURE_GROUPS.values()] == [13, 14, 2, 10]


def test_ablation_is_nested_without_silently_adding_facility_or_risk():
    stages = [sum(stage_features(groups), []) for _, _, groups in ABLATION_STAGES]
    assert [len(stage) for stage in stages] == [13, 27, 37]
    assert set(stages[0]) < set(stages[1]) < set(stages[2])
    assert set(FEATURE_COLUMNS) - set(stages[-1]) == {"prior_facility_type", "prior_risk"}
    assert not any(name.startswith("license_") for name in stages[0] + stages[1])


def _synthetic_temporal_frame():
    rng = np.random.default_rng(91)
    frames = []
    for year in (2020, 2023, 2024, 2025):
        count = 40
        values = rng.normal(size=(count, len(NUMERIC_FEATURES)))
        frame = pd.DataFrame(values, columns=NUMERIC_FEATURES)
        for name in CATEGORICAL_FEATURES:
            frame[name] = np.where(np.arange(count) % 2, "A", "B")
        frame["inspection_id"] = [f"{year}-{i}" for i in range(count)]
        frame["entity_id"] = [f"entity-{i}" for i in range(count)]
        frame["inspection_date"] = pd.date_range(f"{year}-01-01", periods=count)
        frame["target"] = (frame["prior_fail_days"] + rng.normal(size=count) > 0).astype(int)
        frames.append(frame)
    config = {
        "random_seed": 1, "n_jobs": 1,
        **{f"{name}_{bound}": f"{year}-{'01-01' if bound == 'start' else '12-31'}"
           for name, year in [("train", 2020), ("validation", 2023), ("calibration", 2024), ("test", 2025)]
           for bound in ("start", "end")},
    }
    return pd.concat(frames, ignore_index=True), config


def test_test_labels_do_not_change_ablation_selection_or_calibration():
    frame, config = _synthetic_temporal_frame()
    candidates = [{"C": 0.1, "class_weight": None}, {"C": 1.0, "class_weight": None}]
    result, search, selections = run_ablation(frame, config, candidates=candidates)
    changed = frame.copy()
    test = changed["inspection_date"].dt.year.eq(2025)
    changed.loc[test, "target"] = 1 - changed.loc[test, "target"]
    _, changed_search, changed_selections = run_ablation(changed, config, candidates=candidates)
    assert_frame_equal(search, changed_search)
    assert selections == changed_selections
    assert not ((result["split"] == "validation") & (result["probability_kind"] == "calibrated")).any()
    for _, rows in result.loc[result["split"].eq("test")].groupby("stage"):
        assert rows["average_precision"].nunique() == 1
        assert rows["roc_auc"].nunique() == 1


def test_ablation_deltas_do_not_mix_probability_scales_or_frozen_reference():
    rows = []
    for stage in (1, 2, 3, 4):
        for kind, base in [("raw", 0.1), ("calibrated", 0.2)]:
            rows.append({"stage_order": stage, "split": "test", "probability_kind": kind,
                         "is_frozen_reference": stage == 4,
                         "average_precision": base + stage / 100,
                         "roc_auc": base + stage / 10, "brier_score": base - stage / 100})
    result = add_stage_deltas(pd.DataFrame(rows))
    middle = result.loc[result["stage_order"].between(2, 3)]
    assert np.allclose(middle["delta_average_precision"], 0.01)
    assert np.allclose(middle["delta_brier_score"], -0.01)
    assert result.loc[result["stage_order"].isin([1, 4]), "delta_average_precision"].isna().all()


def _inspection_rows():
    # Current identities are synthetic and only test cleaning/cohort rules.
    specifications = [
        (1, "2019-01-01", "Canvass", "Fail", 10),
        (2, "2019-01-01", "Canvass", "Pass", 10),  # binary conflict
        (3, "2019-01-02", "Canvass", "Pass", 11),
        (4, "2019-01-02", "Canvass", "Pass w/ Conditions", 11),  # same binary label
        (5, "2019-02-01", "Canvass", "Fail", 12),
        (6, "2019-02-01", "Canvass", "No Entry", 13),
        (7, "2019-01-01", "Complaint", "Fail", 10),  # distinct type retained
        (8, "2019-02-01", "Complaint", "Pass", 14),
        (9, "2019-02-02", "Complaint Re-Inspection", "Fail", 15),
        (10, "2019-02-03", "Canvass Re-Inspection", "Fail", 16),
    ]
    return clean_inspections(pd.DataFrame([
        {"inspection_id": identity, "inspection_date": date, "inspection_type": kind,
         "results": result, "license_": license_id, "address": f"{license_id} TEST ST"}
        for identity, date, kind, result, license_id in specifications
    ]))


def test_complaint_cohort_uses_exact_type_and_retains_conditions_as_negative():
    targets, audit = comparison_targets(_inspection_rows())
    assert len(targets) == 4
    assert targets["inspection_type"].value_counts().to_dict() == {"Canvass": 2, "Complaint": 2}
    assert audit["Canvass"]["conflicting_rows_excluded"] == 2
    assert audit["Canvass"]["same_label_rows_collapsed"] == 1
    assert targets["target"].sum() == 2


def test_canvass_trend_cohort_matches_existing_target_definition_and_order_invariance():
    inspections = _inspection_rows()
    expected, _ = select_targets(inspections)
    actual, _ = comparison_targets(inspections.sample(frac=1, random_state=3))
    assert sorted(expected["inspection_id"]) == sorted(
        actual.loc[actual["inspection_type"].eq("Canvass"), "inspection_id"]
    )


def test_seasonal_denominators_are_pooled_while_equal_year_means_are_distinct():
    rows = []
    for year, labels in [(2019, [1]), (2020, [0, 0, 0])]:
        for label in labels:
            rows.append({"inspection_type": "Canvass", "inspection_date": f"{year}-01-01", "target": label})
    annual, monthly, seasonal, summary = aggregate_trends(pd.DataFrame(rows))
    assert len(annual) == len(monthly) == 2
    assert seasonal.iloc[0]["fail_rate"] == pytest.approx(0.25)
    assert seasonal.iloc[0]["equal_year_mean_fail_rate"] == pytest.approx(0.5)
    assert seasonal.iloc[0]["years_observed"] == 2
    assert summary["monthly_rate_correlation_2019_2025"] is None


def test_rule_change_is_split_at_july_not_at_year_end():
    frame = pd.DataFrame([
        {"inspection_type": "Canvass", "inspection_date": "2018-06-30", "target": 1},
        {"inspection_type": "Canvass", "inspection_date": "2018-07-01", "target": 0},
        {"inspection_type": "Canvass", "inspection_date": "2019-01-01", "target": 0},
    ])
    annual, monthly, seasonal, summary = aggregate_trends(frame)
    assert annual.loc[annual["year"].eq(2018), "regime"].item() == "transition_year"
    assert monthly.loc[monthly["year_month"].eq("2018-06"), "regime"].item() == "pre_2018_07"
    assert monthly.loc[monthly["year_month"].eq("2018-07"), "regime"].item() == "post_2018_07"
    assert seasonal["n"].sum() == 1
    assert [row["fail_rate"] for row in summary["rule_change_2018_halves"]] == [1.0, 0.0]
