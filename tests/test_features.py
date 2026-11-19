"""Counterfactual tests for point-in-time validity, not model-score smoke tests."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from chicago_food.data import clean_inspections, clean_licenses
from chicago_food.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
    build_features, score_candidates,
)


def inspection(identity, date, result="Pass", kind="Canvass", **overrides):
    return {
        "inspection_id": identity, "inspection_date": date, "results": result,
        "inspection_type": kind, "license_number": "123", "address": "100 W MAIN ST",
        "dba_name": "Example Cafe", "facility_type": "Restaurant", "risk": "Risk 1 (High)",
        **overrides,
    }


def license_row(identity="1", **overrides):
    return {
        "license_id": identity, "license_number": "123", "address": "100 W MAIN STREET 1ST",
        "date_issued": "2018-01-01", "license_start_date": "2018-02-01",
        "expiration_date": "2020-01-31", "license_description": "Retail Food Establishment",
        "application_type": "ISSUE", "license_status": "AAI",
        **overrides,
    }


def features(rows, licenses=None, **kwargs):
    source = clean_inspections(pd.DataFrame(rows))
    license_frame = None if licenses is None else clean_licenses(pd.DataFrame(licenses))
    return build_features(source, license_frame, **kwargs)


def row(frame, identity):
    return frame.set_index("inspection_id").loc[str(identity)]


def test_only_exact_canvass_and_resolved_outcomes_are_targets():
    frame, audit = features([
        inspection(1, "2019-01-01", "Fail"),
        inspection(2, "2019-02-01", "Pass w/ Conditions"),
        inspection(3, "2019-03-01", "No Entry"),
        inspection(4, "2019-04-01", "Fail", "Canvass Re-Inspection"),
        inspection(5, "2019-05-01", "Fail", "Complaint"),
    ])
    assert frame["inspection_id"].tolist() == ["1", "2"]
    assert frame["target"].tolist() == [1, 0]
    assert audit["routine_labeled_rows"] == 2


def test_current_attributes_are_never_used_as_predictors():
    original = [inspection(1, "2018-12-01"), inspection(2, "2019-01-01", "Fail")]
    changed = [original[0], {**original[1], "facility_type": "FUTURE SECRET", "risk": "TARGET LEAK"}]
    before, _ = features(original)
    after, _ = features(changed)
    assert_frame_equal(before, after)
    assert row(before, 2)["prior_facility_type"] == "Restaurant"


def test_future_rows_cannot_change_past_features():
    past = [inspection(1, "2018-12-01"), inspection(2, "2019-01-01", "Fail")]
    before, _ = features(past)
    after, _ = features(past + [inspection(3, "2025-01-01", "Fail", facility_type="SECRET")])
    assert_frame_equal(before, after.iloc[:2].reset_index(drop=True))


def test_same_day_rows_do_not_supply_history():
    frame, _ = features([
        inspection(1, "2018-12-01", "Pass"),
        inspection(2, "2019-01-01", "Fail", "Complaint"),
        inspection(3, "2019-01-01", "Pass"),
    ])
    candidate = row(frame, 3)
    assert candidate["prior_inspection_days"] == 1
    assert candidate["historical_failure_rate"] == 0
    assert candidate["days_since_last_inspection"] == 31


def test_conflicting_same_day_targets_excluded_and_same_label_collapsed():
    frame, audit = features([
        inspection(1, "2019-01-01", "Pass"), inspection(2, "2019-01-01", "Fail"),
        inspection(3, "2019-02-01", "Pass"), inspection(4, "2019-02-01", "Pass w/ Conditions"),
    ])
    assert frame["inspection_id"].tolist() == ["3"]
    assert audit["conflicting_target_days"] == 1
    assert audit["conflicting_target_rows_excluded"] == 2
    assert audit["same_day_same_label_targets_collapsed"] == 1


def test_past_same_day_outcomes_aggregate_without_artificial_order():
    records = [
        inspection(10, "2018-12-01", "Fail", "Complaint"),
        inspection(2, "2018-12-01", "Pass", "License"),
        inspection(3, "2019-01-01"),
    ]
    before, _ = features(records)
    after, _ = features(records[::-1])
    assert_frame_equal(before, after)
    candidate = row(before, 3)
    assert candidate["prior_scored_days"] == 1
    assert candidate["prior_fail_days"] == 1
    assert candidate["last_result"] == "Mixed"


def test_unknown_results_do_not_count_as_passes():
    frame, _ = features([
        inspection(1, "2018-01-01", "Fail", "Complaint"),
        inspection(2, "2018-11-01", "No Entry"),
        inspection(3, "2019-01-01", "Pass"),
    ])
    candidate = row(frame, 3)
    assert candidate["prior_inspection_days"] == 2
    assert candidate["prior_scored_days"] == 1
    assert candidate["historical_failure_rate"] == 1
    assert candidate["last_result"] == "Unknown"


@pytest.mark.parametrize("lag,expected", [(1, 2), (7, 1), (30, 0)])
def test_observation_lag_has_exact_calendar_day_boundary(lag, expected):
    frame, _ = features([
        inspection(1, "2019-01-24", "Fail", "Complaint"),
        inspection(2, "2019-01-30", "Pass", "Complaint"),
        inspection(3, "2019-01-31"),
    ], lag_days=lag)
    assert row(frame, 3)["prior_scored_days"] == expected


@pytest.mark.parametrize("lag", [0, -1, 1.5])
def test_nonpositive_or_fractional_lag_rejected(lag):
    with pytest.raises(ValueError, match="positive integer"):
        features([inspection(1, "2019-01-01")], lag_days=lag)


def test_cold_start_is_missing_history_not_a_zero_failure_rate():
    frame, _ = features([inspection(1, "2019-01-01", "Fail")])
    candidate = row(frame, 1)
    assert candidate["cold_start"] == 1
    assert candidate["no_scored_history"] == 1
    assert np.isnan(candidate["historical_failure_rate"])
    assert candidate["prior_facility_type"] == "Unknown"


def test_license_entity_does_not_cross_different_addresses():
    frame, _ = features([
        inspection(1, "2018-01-01", "Fail", address="200 W MAIN ST"),
        inspection(2, "2019-01-01"),
    ], [license_row(address="200 W MAIN ST")])
    assert row(frame, 2)["cold_start"] == 1
    assert row(frame, 2)["license_history_available"] == 0


def test_missing_license_entities_use_name_and_address_without_chain_merging():
    frame, _ = features([
        inspection(1, "2018-01-01", "Fail", license_number=""),
        inspection(2, "2019-01-01", license_number=""),
        inspection(3, "2019-01-01", license_number="", address="200 W MAIN ST"),
    ])
    assert row(frame, 2)["prior_fail_days"] == 1
    assert row(frame, 3)["cold_start"] == 1


def test_license_not_visible_until_issued_and_started():
    records = [inspection(1, "2019-01-01")]
    frame, _ = features(records, [
        license_row(date_issued="2019-01-02", license_start_date="2005-01-01"),
        license_row("2", date_issued="2018-01-01", license_start_date="2019-02-01"),
    ])
    assert row(frame, 1)["license_history_available"] == 0
    assert np.isnan(row(frame, 1)["license_observed_tenure_years"])


def test_future_license_transactions_cannot_change_past_tenure_or_term():
    records = [inspection(1, "2019-01-01")]
    before, _ = features(records, [license_row()])
    after, _ = features(records, [license_row(), license_row(
        "2", date_issued="2025-01-01", license_start_date="2003-01-01",
        expiration_date="2030-01-01", license_description="FUTURE CATEGORY",
    )])
    assert_frame_equal(before, after)


def test_license_tenure_uses_earliest_eligible_term_and_status_is_interval_only():
    records = [inspection(1, "2019-01-01")]
    before, _ = features(records, [license_row(license_status="AAI")])
    after, _ = features(records, [license_row(license_status="REV", license_status_change_date="2019-01-01")])
    assert_frame_equal(before, after)
    candidate = row(before, 1)
    assert candidate["license_term_status"] == "In recorded term"
    assert candidate["license_observed_tenure_years"] == pytest.approx(334 / 365.25)


def test_license_issued_on_scoring_day_is_not_visible():
    frame, _ = features([inspection(1, "2019-01-01")], [license_row(date_issued="2019-01-01")])
    assert row(frame, 1)["license_history_available"] == 0


def test_current_day_scheduled_term_can_start_if_it_was_issued_beforehand():
    frame, _ = features([inspection(1, "2019-01-01")], [
        license_row(date_issued="2018-12-01", license_start_date="2019-01-01")
    ])
    assert row(frame, 1)["license_history_available"] == 1
    assert row(frame, 1)["license_observed_tenure_years"] == 0


def test_date_filter_keeps_earlier_history():
    frame, audit = features([
        inspection(1, "2010-01-01", "Fail"), inspection(2, "2019-01-01")
    ], start_date="2019-01-01", end_date="2019-12-31")
    assert len(frame) == 1
    assert row(frame, 2)["prior_fail_days"] == 1
    assert audit["target_count_before_date_filter"] == 2


def test_explicit_model_feature_allowlist_excludes_identifiers_and_target():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert not (set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES))
    assert not ({"results", "target", "inspection_id", "entity_id", "violations", "risk", "facility_type", "license_number", "address"} & set(FEATURE_COLUMNS))


def test_scoring_queue_has_training_feature_parity_and_ignores_untrusted_attributes():
    inspections = clean_inspections(pd.DataFrame([
        inspection(1, "2018-12-01", "Fail"), inspection(2, "2019-01-01")
    ]))
    licenses = clean_licenses(pd.DataFrame([license_row()]))
    historical, _ = build_features(inspections, licenses)
    candidates = pd.DataFrame([{
        "candidate_id": "queue-1", "license_number": "123", "address": "100 W MAIN STREET 1ST",
        "facility_type": "IGNORE ME", "risk": "IGNORE ME", "results": "Fail",
    }])
    scoring = score_candidates(candidates, inspections, licenses, as_of="2019-01-01")
    expected = historical.loc[historical["inspection_id"].eq("2"), FEATURE_COLUMNS].reset_index(drop=True)
    assert_frame_equal(expected, scoring[FEATURE_COLUMNS])
    assert "target" not in scoring
    assert scoring["inspection_id"].tolist() == ["queue-1"]


def test_entity_input_order_cannot_change_features():
    records = [
        inspection(1, "2018-12-01", "Fail"), inspection(2, "2019-01-01"),
        inspection(3, "2018-12-01", license_number="456"),
        inspection(4, "2019-01-01", "Fail", license_number="456"),
    ]
    before, _ = features(records, [license_row(), license_row("2", license_number="456")])
    after, _ = features(records[::-1], [license_row("2", license_number="456"), license_row()])
    assert_frame_equal(before, after)


def test_empty_target_cohort_still_has_explicit_schema():
    frame, audit = features([inspection(1, "2019-01-01", "No Entry")])
    assert len(frame) == 0
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert audit["target_count"] == 0
