"""Business groups for incremental ablation and signed SHAP aggregation.

The partition covers the 39 original source fields, before missing indicators or
one-hot expansion. Outcome summaries can contain time windows; the exposure/time
group adds denominators, elapsed intervals, observation coverage and calendar terms.
These descriptive groups do not remove dependence between features or groups.
"""

FEATURE_GROUPS = {
    "historical_failure_evidence": (
        "prior_fail_days", "prior_conditions_days", "historical_failure_rate",
        "historical_conditions_rate", "routine_failure_rate", "last3_failure_rate",
        "last5_failure_rate", "recency_weighted_failure_rate", "failures_365d",
        "failure_rate_365d", "recent_failure_streak", "last_result", "last_routine_result",
    ),
    "inspection_exposure_time": (
        "prior_inspection_days", "prior_scored_days", "prior_routine_days",
        "scored_days_365d", "days_since_last_inspection",
        "days_since_last_scored_inspection", "days_since_last_routine",
        "days_since_last_failure", "observed_history_years", "cold_start",
        "no_scored_history", "historical_pre2018_share", "month_sin", "month_cos",
    ),
    "facility_risk": ("prior_facility_type", "prior_risk"),
    "license_background": (
        "license_history_available", "license_observed_tenure_years",
        "days_since_license_issue", "license_records_observed", "license_terms_observed",
        "days_to_license_term_end", "license_term_elapsed_days", "license_term_status",
        "license_category", "license_application_type",
    ),
}

GROUP_LABELS = {
    "historical_failure_evidence": "Historical failure evidence",
    "inspection_exposure_time": "Inspection exposure and time",
    "facility_risk": "Facility type and risk",
    "license_background": "Business license background",
}
