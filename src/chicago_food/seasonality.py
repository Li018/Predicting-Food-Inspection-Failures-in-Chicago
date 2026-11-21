"""Descriptive Canvass/Complaint trends with the existing binary target policy."""

from __future__ import annotations

import pandas as pd

from .features import VALID_RESULTS

INSPECTION_TYPES = ("Canvass", "Complaint")
RULE_DATE = pd.Timestamp("2018-07-01")


def comparison_targets(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Use exact types and one unambiguous binary target per entity/day/type.

    ``cleaned`` is produced by the existing ``load_inspections``. Complaint
    Re-Inspection and other similarly named types are deliberately excluded.
    """
    subset = cleaned.loc[
        cleaned["inspection_type"].isin(INSPECTION_TYPES)
        & cleaned["results"].isin(VALID_RESULTS)
    ].copy()
    subset["target"] = subset["results"].eq("Fail").astype(int)
    keys = ["inspection_type", "entity", "inspection_date"]
    conflict = subset.groupby(keys, sort=False)["target"].transform("nunique").gt(1)
    valid = subset.loc[~conflict].sort_values(keys + ["inspection_id"], kind="stable")
    targets = valid.drop_duplicates(keys, keep="first").copy()
    audit = {}
    for kind in INSPECTION_TYPES:
        original = subset["inspection_type"].eq(kind)
        retained = targets["inspection_type"].eq(kind)
        audit[kind] = {
            "recognized_outcome_rows": int(original.sum()),
            "conflicting_entity_days": int(subset.loc[original & conflict, keys].drop_duplicates().shape[0]),
            "conflicting_rows_excluded": int((original & conflict).sum()),
            "same_label_rows_collapsed": int((original & ~conflict).sum() - retained.sum()),
            "target_count": int(retained.sum()),
        }
    return targets.reset_index(drop=True), audit


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.groupby(keys, sort=True, observed=True).agg(
        n=("target", "size"), failures=("target", "sum"),
    ).reset_index()
    result["fail_rate"] = result["failures"] / result["n"]
    return result


def aggregate_trends(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Keep pooled seasonal profiles separate from equal-year descriptive means."""
    work = targets.copy()
    dates = pd.to_datetime(work["inspection_date"], errors="raise")
    work["year"] = dates.dt.year
    work["month"] = dates.dt.month
    annual = _aggregate(work, ["inspection_type", "year"])
    annual["regime"] = annual["year"].map(
        lambda year: "pre_2018_07" if year < 2018 else (
            "transition_year" if year == 2018 else "post_2018_07"
        )
    )
    monthly = _aggregate(work, ["inspection_type", "year", "month"])
    monthly["year_month"] = (
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    monthly["regime"] = monthly["year_month"].map(
        lambda month: "pre_2018_07" if month < "2018-07" else "post_2018_07"
    )
    monthly = monthly.merge(
        annual[["inspection_type", "year", "fail_rate"]].rename(columns={"fail_rate": "annual_fail_rate"}),
        on=["inspection_type", "year"], how="left", validate="many_to_one",
    )
    monthly["deviation_from_annual_rate"] = monthly["fail_rate"] - monthly["annual_fail_rate"]
    post = work.loc[work["year"].between(2019, 2025)]
    seasonal = _aggregate(post, ["inspection_type", "month"])
    post_monthly = monthly.loc[monthly["year"].between(2019, 2025)]
    by_month = post_monthly.groupby(["inspection_type", "month"], observed=True).agg(
        years_observed=("year", "nunique"), equal_year_mean_fail_rate=("fail_rate", "mean"),
        mean_deviation_from_annual_rate=("deviation_from_annual_rate", "mean"),
        minimum_year_month_rate=("fail_rate", "min"), maximum_year_month_rate=("fail_rate", "max"),
    ).reset_index()
    seasonal = seasonal.merge(by_month, on=["inspection_type", "month"], validate="one_to_one")
    seasonal["period_start"], seasonal["period_end"] = "2019-01-01", "2025-12-31"

    summaries = {}
    for kind in INSPECTION_TYPES:
        rows = seasonal.loc[seasonal["inspection_type"].eq(kind)]
        if rows.empty:
            continue
        low = rows.loc[rows["fail_rate"].idxmin()]
        high = rows.loc[rows["fail_rate"].idxmax()]
        yr = annual.loc[annual["inspection_type"].eq(kind)].set_index("year")
        summary = {
            "n_2019_2025": int(rows["n"].sum()),
            "fail_rate_2019_2025": float(rows["failures"].sum() / rows["n"].sum()),
            "pooled_min_month": int(low["month"]), "pooled_min_fail_rate": float(low["fail_rate"]),
            "pooled_max_month": int(high["month"]), "pooled_max_fail_rate": float(high["fail_rate"]),
            "pooled_month_range_pp": float(100 * (high["fail_rate"] - low["fail_rate"])),
            "equal_year_month_range_pp": float(100 * (
                rows["equal_year_mean_fail_rate"].max() - rows["equal_year_mean_fail_rate"].min()
            )),
        }
        for first, last in ((2019, 2025), (2023, 2025)):
            if first in yr.index and last in yr.index:
                summary[f"fail_rate_change_{first}_{last}_pp"] = float(
                    100 * (yr.loc[last, "fail_rate"] - yr.loc[first, "fail_rate"])
                )
        summaries[kind] = summary
    around_break = work.loc[work["year"].eq(2018)].copy()
    around_break["half"] = around_break["month"].map(lambda month: "H1" if month <= 6 else "H2")
    break_rows = _aggregate(around_break, ["inspection_type", "half"])
    common = post_monthly.pivot(index=["year", "month"], columns="inspection_type", values="fail_rate")
    correlation = None
    if all(kind in common for kind in INSPECTION_TYPES):
        paired = common[list(INSPECTION_TYPES)].dropna()
        if len(paired) >= 2 and paired.nunique().min() > 1:
            correlation = float(paired["Canvass"].corr(paired["Complaint"]))
    summary = {
        "seasonality": summaries,
        "rule_change_2018_halves": break_rows.to_dict("records"),
        "monthly_rate_correlation_2019_2025": correlation,
    }
    return annual, monthly, seasonal, summary
