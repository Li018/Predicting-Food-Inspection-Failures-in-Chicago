import pandas as pd
import pytest

from chicago_food.data import clean_inspections, clean_licenses, normalize_address, normalize_license


@pytest.mark.parametrize("value,expected", [(123, "123"), ("00123.0", "123"), (None, ""),
    (float("nan"), ""), ("0", ""), ("-1", ""), ("12.3", ""), ("bad", "")])
def test_license_identifiers(value, expected):
    assert normalize_license(value) == expected


def test_address_normalization_preserves_street_and_range():
    assert normalize_address("123 North Main Street 1ST FLOOR") == "123 N MAIN ST"
    assert normalize_address("123 N MAIN ST STE 5") == "123 N MAIN ST"
    assert normalize_address("100 - 104 S ST LOUIS AVENUE") == "100-104 S ST LOUIS AVE"
    assert normalize_address("[REDACTED FOR PRIVACY]") == ""
    assert normalize_address(None) == ""
    assert normalize_address("100-104 S MAIN ST") != normalize_address("100 S MAIN ST")


def inspection_rows():
    return pd.DataFrame({
        "inspection_id": ["1", "2", "3", "4"],
        "inspection_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
        "license_": ["123", "123", "456", "123"],
        "address": ["123 N MAIN ST", "123 NORTH MAIN STREET 1ST", "123 N MAIN ST", ""],
        "dba_name": ["FIRST", "RENAMED", "OTHER", "FIRST"],
        "inspection_type": ["Canvass"] * 4, "results": ["Pass", "Fail", "Pass", "Fail"],
    })


def test_identity_is_conservative_and_row_local():
    frame = clean_inspections(inspection_rows())
    assert frame.loc[0, "entity"] == frame.loc[1, "entity"]
    assert frame.loc[0, "entity"] != frame.loc[2, "entity"]
    assert frame.loc[3, "entity"] == "unknown:4"
    prefix = clean_inspections(inspection_rows().iloc[:2])
    assert prefix["entity"].tolist() == frame["entity"].iloc[:2].tolist()


def test_duplicate_ids_fail_loudly():
    frame = inspection_rows()
    assert len(clean_inspections(pd.concat([frame, frame]))) == 4
    frame.loc[1, "inspection_id"] = "1"
    with pytest.raises(ValueError, match="Conflicting"):
        clean_inspections(frame)


def test_unknown_results_are_not_recoded_as_pass():
    frame = inspection_rows()
    frame.loc[0, "results"] = "No Entry"
    assert clean_inspections(frame).loc[0, "results"] == "No Entry"


def test_license_transactions_keep_terms_and_status_separate():
    raw = pd.DataFrame({
        "license_id": ["1", "2"], "license_number": ["123", "123"],
        "date_issued": ["2020-02-01", "2022-03-01"],
        "license_start_date": ["2020-01-01", "2022-01-01"],
        "expiration_date": ["2021-12-31", "2023-12-31"],
        "license_status": ["AAI", "AAC"],
        "license_status_change_date": ["", "2023-05-01"],
    })
    frame = clean_licenses(raw)
    assert len(frame) == 2
    assert frame.loc[0, "date_issued"] > frame.loc[0, "license_start_date"]
    assert frame.loc[1, "license_status"] == "AAC"
    assert frame.loc[1, "license_status_change_date"] == pd.Timestamp("2023-05-01")
