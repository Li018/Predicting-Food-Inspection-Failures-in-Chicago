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


def test_lettered_avenues_and_unknown_address_tails_do_not_collide():
    assert normalize_address("13335 S AVENUE O") == "13335 S AVE O"
    assert normalize_address("13335 S AVENUE L") == "13335 S AVE L"
    assert normalize_address("13335 S AVENUE O STE 2") == "13335 S AVE O"
    assert normalize_address("13335 AVENUE O") == "13335 AVE O"
    assert normalize_address("123 N MAIN ST EXTENSION") == "123 N MAIN ST EXTENSION"
    assert normalize_address("123 N MAIN ST EXTENSION") != normalize_address("123 N MAIN ST")
    assert normalize_address("3101 W IRVING PARK RD A") == "3101 W IRVING PARK RD"


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


def test_impossible_license_term_end_is_unknown_and_audited():
    raw = pd.DataFrame({
        "license_number": ["123"], "date_issued": ["2020-02-01"],
        "license_start_date": ["2020-01-01"], "expiration_date": ["2019-12-31"],
    })
    clean = clean_licenses(raw)
    assert len(clean) == 1
    assert pd.isna(clean.loc[0, "expiration_date"])
    assert clean.loc[0, "license_start_date"] == pd.Timestamp("2020-01-01")
    assert clean.attrs["audit"]["expiration_before_start_rows"] == 1
    assert clean.attrs["audit"]["invalid_term_end_dates_masked"] == 1
    assert raw.loc[0, "expiration_date"] == "2019-12-31"


def test_parallel_download_covers_sparse_ids_and_resumes_without_network(tmp_path, monkeypatch):
    """Multiple key ranges must neither skip nor repeat records at their boundaries."""
    import importlib.util
    import json
    import re
    from pathlib import Path
    from urllib.parse import parse_qs, urlsplit

    path = Path(__file__).parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = pd.DataFrame({"license_id": [1, 24, 25, 49, 50, 74, 75, 99],
                         "license_number": [101, 102, 101, 102, 101, 102, 101, 102]})

    def fake_request(url):
        query = parse_qs(urlsplit(url).query)
        if "/api/views/" in url:
            return json.dumps({"columns": [], "description": "fixture"}).encode()
        select = query["$select"][0]
        where = query["$where"][0]
        if select == "count(*) as n":
            return b'[{"n":"0"}]'
        if "max(" in select:
            return json.dumps([{"n": len(rows), "max_key": 99}]).encode()
        lower = int(re.search(r"license_id > (\d+)", where).group(1))
        upper = int(re.search(r"license_id <= (\d+)", where).group(1))
        chosen = rows.loc[rows.license_id.gt(lower) & rows.license_id.le(upper)]
        chosen = chosen.head(int(query["$limit"][0]))
        return chosen.to_csv(index=False).encode()

    monkeypatch.setattr(module, "request_bytes", fake_request)
    result, source = module.download_source(
        "licenses", "2026-01-01", tmp_path, page_size=2, workers=4, keep_licenses={"101"}
    )
    assert result.license_id.tolist() == ["1", "25", "50", "75"]
    assert source["downloaded_rows"] == 8
    assert source["retained_rows"] == 4
    monkeypatch.setattr(module, "request_bytes", lambda url: pytest.fail("cache resumed over network"))
    resumed, resumed_source = module.download_source(
        "licenses", "2026-01-01", tmp_path, page_size=2, workers=4, keep_licenses={"101"}
    )
    pd.testing.assert_frame_equal(result, resumed)
    assert source["pages"] == resumed_source["pages"]
