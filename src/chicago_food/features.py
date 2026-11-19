"""Point-in-time features for the next routine inspection.

Every feature is computed relative to an explicit scoring date. Inspection histories
are summarized by calendar day: multiple observations on one day have no meaningful
within-day ordering in this source. A scored day contains any recognized outcome;
a failure day contains at least one Fail. Conflicting labels on a *target* day are
excluded altogether. No global means, current inspection attributes, or future
license transactions are used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

VALID_RESULTS = {"Fail", "Pass", "Pass w/ Conditions"}
NUMERIC_FEATURES = [
    "prior_inspection_days", "prior_scored_days", "prior_fail_days",
    "prior_conditions_days", "historical_failure_rate", "historical_conditions_rate",
    "prior_routine_days", "routine_failure_rate", "last3_failure_rate",
    "last5_failure_rate", "recency_weighted_failure_rate", "scored_days_365d",
    "failures_365d", "failure_rate_365d", "days_since_last_inspection",
    "days_since_last_scored_inspection", "days_since_last_routine",
    "days_since_last_failure", "observed_history_years", "cold_start",
    "no_scored_history", "recent_failure_streak", "historical_pre2018_share",
    "license_history_available", "license_observed_tenure_years",
    "days_since_license_issue", "license_records_observed", "license_terms_observed",
    "days_to_license_term_end", "license_term_elapsed_days", "month_sin", "month_cos",
]
CATEGORICAL_FEATURES = [
    "last_result", "last_routine_result", "prior_facility_type", "prior_risk",
    "license_term_status", "license_category", "license_application_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
META_COLUMNS = ["inspection_id", "entity_id", "inspection_date", "target"]
RULE_CHANGE = pd.Timestamp("2018-07-01").toordinal()


def _date(values: pd.Series) -> pd.Series:
    """Parse mixed portal formats without depending on the input row order."""
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def _mode_or_ambiguous(values: pd.Series) -> str:
    unique = sorted({str(x).strip() for x in values.dropna() if str(x).strip()})
    if not unique:
        return "Unknown"
    return unique[0] if len(unique) == 1 else "Ambiguous"


def _result(values: pd.Series) -> str:
    known = set(values) & VALID_RESULTS
    return next(iter(known)) if len(known) == 1 else ("Mixed" if known else "Unknown")


def _prepare_inspections(inspections: pd.DataFrame) -> pd.DataFrame:
    from .data import normalize_address, normalize_license

    frame = inspections.copy()
    aliases = {"license_": "license_number", "entity_id": "entity"}
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
    required = {"inspection_id", "inspection_date", "inspection_type", "results"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing inspection columns: {sorted(missing)}")
    frame["inspection_date"] = _date(frame["inspection_date"])
    frame = frame.loc[frame["inspection_date"].notna()].copy()
    frame["inspection_id"] = frame["inspection_id"].astype(str)
    for col in ["license_number", "address", "dba_name", "risk", "facility_type"]:
        if col not in frame:
            frame[col] = ""
    frame["license_number"] = frame["license_number"].map(normalize_license)
    if "address_key" not in frame:
        frame["address_key"] = frame["address"].map(normalize_address)
    frame["address_key"] = frame["address_key"].fillna("").astype(str)
    if "entity" not in frame:
        frame["entity"] = [
            _entity_key(row.license_number, row.address_key, row.dba_name, row.inspection_id)
            for row in frame.itertuples()
        ]
    frame["entity"] = frame["entity"].astype(str)
    frame["inspection_type"] = frame["inspection_type"].fillna("").astype(str).str.strip()
    frame["results"] = frame["results"].fillna("").astype(str).str.strip()
    # Input order must not resolve duplicate identifiers or same-day ordering.
    frame = frame.drop_duplicates().sort_values(
        ["entity", "inspection_date", "inspection_id"], kind="mergesort"
    )
    duplicate_ids = frame["inspection_id"].duplicated(keep=False)
    if duplicate_ids.any():
        # Contradictory records for one ID cannot be repaired using a future snapshot.
        raise ValueError("Conflicting duplicate inspection_id rows; clean the source first")
    return frame.reset_index(drop=True)


def _entity_key(license_number: str, address_key: str, name: Any, identity: Any) -> str:
    if license_number and address_key:
        return f"license:{license_number}|{address_key}"
    normalized_name = " ".join(str(name).upper().split()) if pd.notna(name) else ""
    if not license_number and address_key and normalized_name:
        return f"name:{normalized_name}|{address_key}"
    return f"unknown:{identity}"


def select_targets(inspections: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select exact Canvass outcomes and remove ambiguous entity/day targets."""
    routine = inspections.loc[
        inspections["inspection_type"].eq("Canvass")
        & inspections["results"].isin(VALID_RESULTS)
    ].copy()
    routine["target"] = routine["results"].eq("Fail").astype("int8")
    keys = ["entity", "inspection_date"]
    conflict = routine.groupby(keys, sort=False)["target"].transform("nunique").gt(1)
    excluded = routine.loc[conflict]
    valid = routine.loc[~conflict].sort_values(keys + ["inspection_id"], kind="mergesort")
    targets = valid.drop_duplicates(keys, keep="first").reset_index(drop=True)
    audit = {
        "routine_labeled_rows": int(len(routine)),
        "conflicting_target_days": int(excluded[keys].drop_duplicates().shape[0]),
        "conflicting_target_rows_excluded": int(conflict.sum()),
        "same_day_same_label_targets_collapsed": int(len(valid) - len(targets)),
        "target_count_before_date_filter": int(len(targets)),
        "target_type": "Canvass (exact)",
        "positive_label": "Fail",
        "negative_labels": ["Pass", "Pass w/ Conditions"],
    }
    return targets, audit


@dataclass
class _History:
    days: np.ndarray
    scored: np.ndarray
    failed: np.ndarray
    conditions: np.ndarray
    routine: np.ndarray
    routine_failed: np.ndarray
    results: list[str]
    routine_results: list[str]
    facilities: list[str]
    risks: list[str]


def _histories(frame: pd.DataFrame) -> dict[str, _History]:
    # Vectorized aggregation avoids a Python callback for every inspection day.
    work = frame.copy()
    work["scored"] = work["results"].isin(VALID_RESULTS)
    work["failed"] = work["results"].eq("Fail")
    work["conditions"] = work["results"].eq("Pass w/ Conditions")
    work["routine"] = work["inspection_type"].eq("Canvass") & work["scored"]
    work["routine_failed"] = work["routine"] & work["failed"]
    work["routine_result"] = work["results"].where(work["routine"], "")
    daily = work.groupby(["entity", "inspection_date"], sort=True).agg(
        scored=("scored", "max"), failed=("failed", "max"),
        conditions=("conditions", "max"), routine=("routine", "max"),
        routine_failed=("routine_failed", "max"), results=("results", _result),
        routine_results=("routine_result", _result),
        facilities=("facility_type", _mode_or_ambiguous), risks=("risk", _mode_or_ambiguous),
    ).reset_index()
    histories = {}
    for entity, group in daily.groupby("entity", sort=False):
        histories[str(entity)] = _History(
            days=np.array([d.toordinal() for d in group["inspection_date"]], dtype=np.int32),
            scored=group["scored"].to_numpy(bool), failed=group["failed"].to_numpy(bool),
            conditions=group["conditions"].to_numpy(bool),
            routine=group["routine"].to_numpy(bool),
            routine_failed=group["routine_failed"].to_numpy(bool),
            results=group["results"].tolist(),
            routine_results=group["routine_results"].tolist(),
            facilities=group["facilities"].tolist(), risks=group["risks"].tolist(),
        )
    return histories


def _empty_history() -> dict[str, Any]:
    features = {name: np.nan for name in NUMERIC_FEATURES}
    for name in [
        "prior_inspection_days", "prior_scored_days", "prior_fail_days",
        "prior_conditions_days", "prior_routine_days", "scored_days_365d",
        "failures_365d", "recent_failure_streak",
    ]:
        features[name] = 0.0
    features.update(
        cold_start=1.0, no_scored_history=1.0, last_result="Unknown",
        last_routine_result="Unknown", prior_facility_type="Unknown", prior_risk="Unknown",
    )
    return features


def _history_features(history: _History | None, day: int, cutoff: int) -> dict[str, Any]:
    result = _empty_history()
    stop = 0 if history is None else int(np.searchsorted(history.days, cutoff, side="right"))
    if stop == 0:
        return result
    dates = history.days[:stop]
    scored = history.scored[:stop]
    failed = history.failed[:stop]
    condition = history.conditions[:stop]
    routine = history.routine[:stop]
    recent = (dates >= day - 365) & scored
    known_failed = failed[scored]
    n_scored = int(scored.sum())
    n_routine = int(routine.sum())
    result.update(
        prior_inspection_days=float(stop), prior_scored_days=float(n_scored),
        prior_fail_days=float(failed.sum()), prior_conditions_days=float(condition.sum()),
        prior_routine_days=float(n_routine), scored_days_365d=float(recent.sum()),
        failures_365d=float(failed[recent].sum()), cold_start=0.0,
        no_scored_history=float(n_scored == 0),
        days_since_last_inspection=float(day - dates[-1]),
        observed_history_years=float((day - dates[0]) / 365.25),
        historical_pre2018_share=float((dates < RULE_CHANGE).mean()),
        last_result=history.results[stop - 1],
        prior_facility_type=history.facilities[stop - 1], prior_risk=history.risks[stop - 1],
    )
    if n_scored:
        scored_dates = dates[scored]
        weights = np.exp2(-(day - scored_dates) / 365.25)
        result.update(
            historical_failure_rate=float(known_failed.mean()),
            historical_conditions_rate=float(condition[scored].mean()),
            last3_failure_rate=float(known_failed[-3:].mean()),
            last5_failure_rate=float(known_failed[-5:].mean()),
            recency_weighted_failure_rate=float(np.average(known_failed, weights=weights)),
            days_since_last_scored_inspection=float(day - scored_dates[-1]),
            recent_failure_streak=float(
                next((i for i, value in enumerate(known_failed[::-1]) if not value), n_scored)
            ),
        )
    if recent.any():
        result["failure_rate_365d"] = float(failed[recent].mean())
    if failed.any():
        result["days_since_last_failure"] = float(day - dates[failed][-1])
    if n_routine:
        last_index = np.flatnonzero(routine)[-1]
        result.update(
            routine_failure_rate=float(history.routine_failed[:stop][routine].mean()),
            days_since_last_routine=float(day - dates[last_index]),
            last_routine_result=history.routine_results[last_index],
        )
    return result


def _license_groups(licenses: pd.DataFrame | None) -> dict[tuple[str, str], pd.DataFrame]:
    from .data import normalize_address, normalize_license

    if licenses is None or licenses.empty:
        return {}
    frame = licenses.copy()
    aliases = {
        "license_start_date": "term_start", "license_term_start_date": "term_start",
        "expiration_date": "term_end", "license_term_expiration_date": "term_end",
    }
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
    required = {"license_number", "date_issued", "term_start"}
    if required - set(frame):
        raise ValueError(f"Missing license columns: {sorted(required - set(frame))}")
    frame["license_number"] = frame["license_number"].map(normalize_license)
    if "address_key" not in frame:
        frame["address_key"] = frame.get("address", pd.Series("", index=frame.index)).map(
            normalize_address
        )
    for col in ["date_issued", "term_start", "term_end"]:
        frame[col] = _date(frame.get(col, pd.Series(pd.NaT, index=frame.index)))
    for col in ["license_description", "application_type", "license_id"]:
        if col not in frame:
            frame[col] = ""
    frame = frame.loc[
        frame["license_number"].ne("") & frame["address_key"].fillna("").ne("")
        & frame["date_issued"].notna() & frame["term_start"].notna()
    ].copy()
    frame["license_id"] = frame["license_id"].fillna("").astype(str)
    frame = frame.sort_values(["term_start", "date_issued", "license_id"], kind="mergesort")
    return {key: group for key, group in frame.groupby(["license_number", "address_key"], sort=False)}


def _license_features(records: pd.DataFrame | None, date: pd.Timestamp, cutoff: pd.Timestamp) -> dict:
    result = {
        "license_history_available": 0.0, "license_observed_tenure_years": np.nan,
        "days_since_license_issue": np.nan, "license_records_observed": 0.0,
        "license_terms_observed": 0.0, "days_to_license_term_end": np.nan,
        "license_term_elapsed_days": np.nan, "license_term_status": "No history",
        "license_category": "Unknown", "license_application_type": "Unknown",
    }
    if records is None:
        return result
    eligible = records.loc[(records["date_issued"] <= cutoff) & (records["term_start"] <= date)]
    if eligible.empty:
        return result
    latest = eligible.iloc[-1]
    ending = latest["term_end"]
    result.update(
        license_history_available=1.0,
        license_observed_tenure_years=float((date - eligible["term_start"].min()).days / 365.25),
        days_since_license_issue=float((date - eligible["date_issued"].max()).days),
        license_records_observed=float(len(eligible)),
        license_terms_observed=float(eligible[["term_start", "term_end"]].drop_duplicates().shape[0]),
        license_term_elapsed_days=float((date - latest["term_start"]).days),
        license_category=str(latest["license_description"]) if pd.notna(latest["license_description"]) else "Unknown",
        license_application_type=str(latest["application_type"]) if pd.notna(latest["application_type"]) else "Unknown",
    )
    if pd.isna(ending):
        result["license_term_status"] = "End unknown"
    else:
        result["days_to_license_term_end"] = float((ending - date).days)
        result["license_term_status"] = "In recorded term" if ending >= date else "Recorded term expired"
    return result


def _score_rows(
    candidates: pd.DataFrame, inspections: pd.DataFrame, licenses: pd.DataFrame | None, lag_days: int,
) -> pd.DataFrame:
    if not isinstance(lag_days, int) or lag_days < 1:
        raise ValueError("lag_days must be a positive integer; same-day records are never available")
    histories = _histories(inspections)
    license_groups = _license_groups(licenses)
    rows = []
    for row in candidates.itertuples(index=False):
        date = row.inspection_date
        cutoff = date - pd.Timedelta(days=lag_days)
        features = _history_features(histories.get(row.entity), date.toordinal(), cutoff.toordinal())
        features.update(_license_features(
            license_groups.get((row.license_number, row.address_key)), date, cutoff,
        ))
        features.update(
            month_sin=float(np.sin(2 * np.pi * (date.month - 1) / 12)),
            month_cos=float(np.cos(2 * np.pi * (date.month - 1) / 12)),
        )
        record = {
            "inspection_id": row.inspection_id, "entity_id": row.entity,
            "inspection_date": date, **features,
        }
        if hasattr(row, "target"):
            record["target"] = int(row.target)
        rows.append(record)
    if not rows:
        cols = ["inspection_id", "entity_id", "inspection_date"]
        if "target" in candidates:
            cols.append("target")
        return pd.DataFrame(columns=cols + FEATURE_COLUMNS)
    frame = pd.DataFrame(rows)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype(float)
    frame[CATEGORICAL_FEATURES] = frame[CATEGORICAL_FEATURES].fillna("Unknown").astype(str)
    return frame.sort_values(["inspection_date", "entity_id", "inspection_id"], kind="mergesort").reset_index(drop=True)


def build_features(
    inspections: pd.DataFrame, licenses: pd.DataFrame | None = None, lag_days: int = 1,
    start_date: str | None = None, end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build retrospective routine targets and event-time-valid feature rows.

    History filtering is local to each target. Year filtering controls target rows,
    never the earlier history from which those rows are constructed.
    """
    cleaned = _prepare_inspections(inspections)
    targets, audit = select_targets(cleaned)
    if start_date is not None:
        targets = targets.loc[targets["inspection_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        targets = targets.loc[targets["inspection_date"] <= pd.Timestamp(end_date)]
    features = _score_rows(targets, cleaned, licenses, lag_days)
    audit.update(
        observation_lag_days=lag_days, target_count=int(len(features)),
        history_unit="calendar day; fail if any recognized Fail that day",
        feature_count=len(FEATURE_COLUMNS),
        numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
        cold_start_count=int(features["cold_start"].sum()) if len(features) else 0,
        license_history_available_count=int(features["license_history_available"].sum()) if len(features) else 0,
    )
    return features, audit


def score_candidates(
    candidates: pd.DataFrame, inspections: pd.DataFrame, licenses: pd.DataFrame | None,
    as_of: str | pd.Timestamp, lag_days: int = 1,
) -> pd.DataFrame:
    """Build deployment features for an externally supplied inspection queue.

    Only the candidate's license/address/name are read. A candidate's inspection
    result, violations, facility and risk columns are intentionally ignored.
    This function produces features; the saved estimator produces probabilities.
    """
    from .data import normalize_address, normalize_license

    date = pd.Timestamp(as_of).normalize()
    if pd.isna(date):
        raise ValueError("as_of must be a valid date")
    cleaned = _prepare_inspections(inspections)
    frame = candidates.copy().reset_index(drop=True)
    for col in ["license_number", "address", "dba_name"]:
        if col not in frame:
            frame[col] = ""
    if "license_" in frame:
        frame["license_number"] = frame["license_number"].where(
            frame["license_number"].fillna("").astype(str).ne(""), frame["license_"]
        )
    frame["license_number"] = frame["license_number"].map(normalize_license)
    frame["address_key"] = frame["address"].map(normalize_address)
    frame["inspection_id"] = frame.get(
        "candidate_id", pd.Series([f"candidate-{i}" for i in range(len(frame))])
    ).astype(str)
    frame["entity"] = [
        _entity_key(row.license_number, row.address_key, row.dba_name, row.inspection_id)
        for row in frame.itertuples()
    ]
    frame["inspection_date"] = date
    frame = frame[["inspection_id", "entity", "inspection_date", "license_number", "address_key"]]
    return _score_rows(frame, cleaned, licenses, lag_days)
