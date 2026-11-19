"""Deterministic input cleaning and conservative establishment identity.

The loaders never learn a cross-row entity mapping. License numbers are public
license identifiers, not transaction IDs. Address normalization removes an explicit
unit/floor suffix only after a recognized street suffix; it does not fuzzy-match
business names or merge different license numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


INSPECTION_COLUMNS = [
    "inspection_id", "dba_name", "aka_name", "license_", "facility_type", "risk",
    "address", "city", "state", "zip", "inspection_date", "inspection_type",
    "results", "latitude", "longitude",
]
LICENSE_COLUMNS = [
    "license_id", "license_number", "account_number", "site_number", "legal_name",
    "doing_business_as_name", "address", "city", "state", "zip_code", "license_code",
    "license_description", "application_type", "application_created_date",
    "application_requirements_complete", "license_start_date", "expiration_date",
    "license_approved_for_issuance", "date_issued", "license_status",
    "license_status_change_date",
]

_SUFFIXES = {
    "STREET": "ST", "STR": "ST", "ST": "ST", "AVENUE": "AVE", "AV": "AVE",
    "AVE": "AVE", "BOULEVARD": "BLVD", "BLVD": "BLVD", "ROAD": "RD", "RD": "RD",
    "DRIVE": "DR", "DR": "DR", "COURT": "CT", "CT": "CT", "PLACE": "PL", "PL": "PL",
    "PARKWAY": "PKWY", "PKWY": "PKWY", "LANE": "LN", "LN": "LN", "TERRACE": "TER",
    "TER": "TER", "HIGHWAY": "HWY", "HWY": "HWY", "CIRCLE": "CIR", "CIR": "CIR",
}
_DIRECTIONS = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}


def normalize_text(value: object) -> str:
    """Normalize case/whitespace without treating a missing value as a name."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).strip()).upper()


def normalize_license(value: object) -> str:
    """Return a positive integral identifier, or an empty string for invalid IDs."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return ""
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        return ""
    return str(int(number))


def normalize_address(value: object) -> str:
    """Canonical street address used *with* license/name, never as the only key.

    Unit/floor strings are inconsistent between BACP and CDPH. When a normal street
    address has a recognized suffix, retain the street through that suffix, keeping
    the complete house-number range. Missing/redacted/unknown addresses stay empty.
    """
    text = normalize_text(value)
    if text in {"", "UNKNOWN", "N/A", "NA", "NONE", "NAN"} or "REDACTED" in text:
        return ""
    text = re.sub(r"[.,]", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    words = text.split()
    if len(words) > 1 and re.fullmatch(r"\d+[A-Z]?(?:-\d+[A-Z]?)?", words[0]):
        words[1] = _DIRECTIONS.get(words[1], words[1])
        indices = [i for i, word in enumerate(words[2:], 2) if word in _SUFFIXES]
        if indices:
            last = indices[-1]
            words[last] = _SUFFIXES[words[last]]
            words = words[: last + 1]
    return " ".join(words)


def _read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".json":
        return pd.read_json(path, dtype=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")


def _canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    names = {}
    for name in frame.columns:
        normalized = str(name).strip().lower().replace(" ", "_")
        names[name] = {
            "license_#": "license_number", "license_": "license_number",
            "license_term_start_date": "license_start_date",
            "license_term_expiration_date": "expiration_date",
        }.get(normalized, normalized)
    return frame.rename(columns=names)


def clean_inspections(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean rows; preserve all outcomes/types for strictly historical features.

    Outcome filtering and same-day conflicting-label policy belong to the feature
    builder. No mutable current-visit fields become model features in this loader.
    """
    frame = _canonical_columns(frame.copy())
    _require(frame, ["inspection_id", "inspection_date", "results", "inspection_type"])
    initial = len(frame)
    frame = frame.drop_duplicates()
    duplicate_rows = initial - len(frame)
    frame["inspection_id"] = frame["inspection_id"].map(normalize_license)
    if frame["inspection_id"].eq("").any():
        raise ValueError("Inspection IDs must be positive integers")
    if frame["inspection_id"].duplicated().any():
        raise ValueError("Conflicting rows share an inspection_id")
    frame["inspection_date"] = pd.to_datetime(
        frame["inspection_date"], errors="coerce", format="mixed"
    ).dt.normalize()
    invalid_dates = int(frame["inspection_date"].isna().sum())
    frame = frame.loc[frame["inspection_date"].notna()].copy()
    for column in ["license_number", "address", "dba_name", "facility_type", "risk", "zip"]:
        if column not in frame:
            frame[column] = ""
    frame["license_number"] = frame["license_number"].map(normalize_license)
    frame["address"] = frame["address"].fillna("").astype(str).str.strip()
    frame["address_key"] = frame["address"].map(normalize_address)
    frame["name_key"] = frame["dba_name"].map(normalize_text)
    for column in ["results", "inspection_type", "dba_name", "facility_type", "risk", "zip"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    result_names = {
        "FAIL": "Fail", "PASS": "Pass", "PASS W/ CONDITIONS": "Pass w/ Conditions",
        "OUT OF BUSINESS": "Out of Business", "NO ENTRY": "No Entry",
        "NOT READY": "Not Ready", "BUSINESS NOT LOCATED": "Business Not Located",
    }
    frame["results"] = frame["results"].map(lambda v: result_names.get(v.upper(), v))
    frame["inspection_type"] = frame["inspection_type"].map(
        lambda v: "Canvass" if v.upper() == "CANVASS" else v
    )
    has_address = frame["address_key"].ne("")
    has_license = frame["license_number"].ne("")
    has_name = frame["name_key"].ne("")
    frame["entity"] = "unknown:" + frame["inspection_id"]
    named = has_address & ~has_license & has_name
    frame.loc[named, "entity"] = (
        "name:" + frame.loc[named, "name_key"] + "|" + frame.loc[named, "address_key"]
    )
    licensed = has_address & has_license
    frame.loc[licensed, "entity"] = (
        "license:" + frame.loc[licensed, "license_number"] + "|"
        + frame.loc[licensed, "address_key"]
    )
    frame = frame.sort_values(["inspection_date", "inspection_id"], kind="stable").reset_index(drop=True)
    frame.attrs["audit"] = {
        "input_rows": initial, "exact_duplicates_removed": duplicate_rows,
        "invalid_date_rows_removed": invalid_dates, "output_rows": len(frame),
        "missing_license_rows": int(frame["license_number"].eq("").sum()),
        "missing_address_rows": int(frame["address_key"].eq("").sum()),
        "isolated_unknown_entity_rows": int(frame["entity"].str.startswith("unknown:").sum()),
    }
    return frame


def load_inspections(path: str | Path) -> pd.DataFrame:
    return clean_inspections(_read_frame(path))


def clean_licenses(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse license transactions without manufacturing historical legal status."""
    frame = _canonical_columns(frame.copy())
    _require(frame, ["license_number", "date_issued", "license_start_date", "expiration_date"])
    initial = len(frame)
    frame = frame.drop_duplicates()
    if "license_id" in frame and frame["license_id"].duplicated().any():
        raise ValueError("Conflicting rows share a license_id transaction key")
    frame["license_number"] = frame["license_number"].map(normalize_license)
    for column in [
        "date_issued", "license_start_date", "expiration_date", "license_status_change_date",
        "application_created_date", "application_requirements_complete",
        "license_approved_for_issuance",
    ]:
        if column not in frame:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce", format="mixed").dt.normalize()
    for column in ["address", "license_code", "license_description", "application_type"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["address_key"] = frame["address"].map(normalize_address)
    invalid = frame["license_number"].eq("") | frame["date_issued"].isna()
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    sort_columns = ["date_issued", "license_number"]
    if "license_id" in frame:
        sort_columns.append("license_id")
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    frame.attrs["audit"] = {
        "input_rows": initial, "output_rows": len(frame), "invalid_identity_or_issue_rows_removed": invalid_count,
        "missing_term_start_rows": int(frame["license_start_date"].isna().sum()),
        "missing_expiration_rows": int(frame["expiration_date"].isna().sum()),
        "expiration_before_start_rows": int((frame["expiration_date"] < frame["license_start_date"]).sum()),
        "missing_address_rows": int(frame["address_key"].eq("").sum()),
    }
    return frame


def load_licenses(path: str | Path) -> pd.DataFrame:
    return clean_licenses(_read_frame(path))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(directory: str | Path) -> dict:
    """Validate the committed snapshot against its manifest before experimentation."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
    return manifest
