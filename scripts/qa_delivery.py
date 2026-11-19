#!/usr/bin/env python3
"""Verify the completed Topic 1 delivery without retraining or printing data rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit

from chicago_food.data import sha256_file, verify_snapshot
from chicago_food.features import FEATURE_COLUMNS, NUMERIC_FEATURES
from chicago_food.metrics import capacity_weights, evaluate_probabilities
from chicago_food.models import temporal_splits
from analyze_errors import confusion
from analyze_lag_sensitivity import changed_features
from run_model import load_verified_features


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"
LOCAL = ROOT / ".local"
FAMILIES = ("prevalence", "logistic_regression", "lightgbm")
ALIGNMENT = ["inspection_id", "inspection_date", "entity_id", "target"]
START_COMMIT = "80f92b7c8a2f43cb2478ba15d0574a287fac670e"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, label: str, atol: float = 1e-10) -> None:
    if isinstance(expected, (str, bool)) or expected is None:
        require(actual == expected, label)
    else:
        require(np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True), label)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def test_fingerprint() -> dict:
    paths = sorted([
        *ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"),
        ROOT / "scripts/download_data.py", ROOT / "pyproject.toml", ROOT / "requirements-lock.txt",
    ])
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.is_file()}


def pytest_check(force: bool = False) -> dict:
    receipt_path, log_path = LOCAL / "pytest_receipt.json", LOCAL / "pytest_final.log"
    current = test_fingerprint()
    receipt = read_json(receipt_path) if receipt_path.is_file() else {}
    reusable = (receipt.get("source_fingerprint") == current and log_path.is_file()
                and receipt.get("log_sha256") == sha256_file(log_path)
                and receipt.get("exit_code") == 0 and "warning_summaries" in receipt)
    if force or not reusable:
        LOCAL.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
        started = time.perf_counter()
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src"},
                                    stdout=log, stderr=subprocess.STDOUT)
        lines = log_path.read_text().splitlines()
        summary = next((line.strip() for line in reversed(lines)
                        if re.search(r"\b\d+ (?:passed|failed|error|errors)\b", line)), "No pytest summary")
        receipt = {
            "command": command, "exit_code": result.returncode,
            "elapsed_seconds": time.perf_counter() - started, "summary": summary,
            "warning_summaries": [line.strip() for line in lines if re.search(r"\b\w+Warning:", line)],
            "source_fingerprint": current,
            "fingerprint_scope": "src/**/*.py, tests/**/*.py, scripts/download_data.py, pyproject.toml, requirements-lock.txt",
            "log_sha256": sha256_file(log_path),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    require(receipt.get("exit_code") == 0, "Full pytest did not pass; inspect the local log")
    require(receipt["source_fingerprint"] == current, "Pytest source fingerprint changed")
    passed = re.search(r"(\d+) passed", receipt["summary"])
    require(passed is not None and int(passed.group(1)) > 0, "No passing pytest suite recorded")
    warnings = re.search(r"(\d+) warnings?", receipt["summary"])
    return {
        "summary": receipt["summary"], "passed_tests": int(passed.group(1)),
        "warnings": int(warnings.group(1)) if warnings else 0,
        "warning_summaries": receipt["warning_summaries"],
        "elapsed_seconds": receipt["elapsed_seconds"], "reused_matching_receipt": reusable and not force,
        "fingerprint_files": len(current), "source_fingerprint": current,
        "log_sha256": receipt["log_sha256"],
        "warning_interpretation": "The recorded PytestRemovedIn10Warning, if present, is from the installed AnyIO pytest plugin; it is not a model fit or leakage-test failure.",
    }


class DeliveryQA:
    def __init__(self):
        self.config = read_json(ROOT / "config/topic1.json")
        self.features, self.feature_metadata = load_verified_features(LOCAL / "features.parquet", self.config)
        self.splits = temporal_splits(self.features, self.config)
        self.test = self.splits["test"].sort_values("inspection_id").reset_index(drop=True)
        self.comparison = pd.read_csv(METRICS / "model_comparison.csv").set_index(["model", "probability_kind"])
        self.selection = read_json(METRICS / "model_selection.json")
        self.runs = {family: read_json(METRICS / f"{family}_metrics.json") for family in FAMILIES}
        self.predictions = {}
        for family in FAMILIES:
            frame = pd.read_parquet(LOCAL / "predictions" / f"{family}.parquet")
            self.predictions[family] = frame.loc[frame["split"].eq("test")].sort_values("inspection_id").reset_index(drop=True)

    def data_and_splits(self) -> dict:
        manifest = verify_snapshot(ROOT / "data/snapshots")
        summary = read_json(METRICS / "data_summary.json")
        for name, split in self.splits.items():
            reported = summary["splits"][name]
            for field, actual in {
                "rows": len(split), "entities": split["entity_id"].nunique(),
                "failures": split["target"].sum(), "fail_rate": split["target"].mean(),
                "cold_start_rows": split["cold_start"].sum(), "cold_start_rate": split["cold_start"].mean(),
            }.items():
                close(actual, reported[field], f"Data summary split mismatch: {name}/{field}")
            close(split["license_history_available"].mean(), reported["license_matching"]["point_in_time_visible_match_rate"], f"PIT match: {name}")
            for feature in NUMERIC_FEATURES:
                close(split[feature].isna().sum(), reported["numeric_feature_missingness"][feature]["missing"], f"Missingness: {name}/{feature}")
        annual = pd.read_csv(METRICS / "annual_drift.csv")
        pd.testing.assert_frame_equal(annual, pd.DataFrame(summary["annual"]), check_dtype=False, atol=1e-12, rtol=0)
        for name, source in summary["sources"].items():
            close(source["snapshot"]["sha256"], manifest["files"][f"{name}.parquet"]["sha256"], f"Source hash: {name}")
        return {"feature_rows": len(self.features), "features": len(FEATURE_COLUMNS),
                "split_rows": {name: len(frame) for name, frame in self.splits.items()},
                "annual_rows": len(annual), "source_sha256": self.feature_metadata["source_sha256"]}

    def models_and_intervals(self) -> dict:
        require(len(self.comparison) == 2 * len(FAMILIES), "Expected raw/calibrated comparison rows for all fitted families")
        validation_ap = {name: run["selection"]["validation_average_precision"] for name, run in self.runs.items()}
        chosen = max(FAMILIES, key=validation_ap.get)
        close(chosen, self.selection["selected_model"], "Validation-selected model differs")
        require(not self.selection["test_used_for_model_selection"], "Selection used test outcomes")
        interval_count = 0
        for family, run in self.runs.items():
            prediction = self.predictions[family]
            require(self.test[ALIGNMENT].equals(prediction[ALIGNMENT]), f"Test target alignment: {family}")
            require(run["provenance"]["feature_sha256"] == self.feature_metadata["sha256"], f"Feature provenance: {family}")
            require(run["config"] == self.config, f"Model configuration: {family}")
            require(run["estimator_fit_period"] == ["2019-01-01", "2022-12-31"], f"Estimator period: {family}")
            require(run["calibration"]["fit_period"] == ["2024-01-01", "2024-12-31"], f"Calibration period: {family}")
            require(run["calibration"]["slope"] > 0 and run["calibration"]["test_ranking_preserved"], f"Calibration ranking: {family}")
            intervals = pd.read_csv(METRICS / f"{family}_confidence_intervals.csv").set_index(["probability_kind", "metric"])
            for kind in ("raw", "calibrated"):
                actual = evaluate_probabilities(prediction["target"], prediction[f"{kind}_probability"],
                                                dates=prediction["inspection_date"], top_fractions=self.config["top_fractions"])
                reported = next(row for row in run["metrics"] if row["split"] == "test" and row["probability_kind"] == kind)
                for metric, value in actual.items():
                    close(value, reported[metric], f"Prediction/summary: {family}/{kind}/{metric}")
                    close(self.comparison.loc[(family, kind), metric], value, f"Comparison: {family}/{kind}/{metric}")
                for metric, interval in run["bootstrap"][kind]["intervals"].items():
                    close(actual[metric], interval["estimate"], f"Bootstrap point estimate: {family}/{kind}/{metric}")
                    for field in ("estimate", "lower", "upper", "valid_replicates"):
                        close(intervals.loc[(kind, metric), field], interval[field], f"CI CSV: {family}/{kind}/{metric}/{field}")
                    for bound in ("lower", "upper"):
                        close(self.comparison.loc[(family, kind), f"{metric}_ci_{bound}"], interval[bound], f"Comparison CI: {family}/{kind}/{metric}/{bound}")
                    require(interval["lower"] <= interval["upper"], "CI endpoints reversed")
                    interval_count += 1
                require(not self.comparison.loc[(family, kind), "daily_confidence_intervals_available"], "Daily intervals must not inherit pooled intervals")
        paired = read_json(METRICS / "paired_model_differences.json")
        require(paired["alignment"]["n"] == len(self.test) and paired["bootstrap"]["same_cluster_draw_for_both_models"], "Paired bootstrap alignment")
        for metric, interval in paired["intervals"].items():
            close(interval["estimate"], self.comparison.loc[("lightgbm", "calibrated"), metric] - self.comparison.loc[("logistic_regression", "calibrated"), metric], f"Paired difference: {metric}")
        return {"comparison_rows": len(self.comparison), "confidence_intervals": interval_count,
                "selected_by_2023_ap": chosen, "test_rows_per_model": len(self.test)}

    def calibration_and_capacity(self) -> dict:
        summary = read_json(METRICS / "calibration_summary.json")
        bins = pd.read_csv(METRICS / "calibration_bins.csv")
        require(not summary["fit_performed_by_this_script"], "Calibration plotting unexpectedly fitted a model")
        for family, group in bins.groupby("model"):
            prediction = self.predictions[family]
            close(group["n"].sum(), len(prediction), f"Calibration bin total: {family}")
            close(group["failures"].sum(), prediction["target"].sum(), f"Calibration bin failures: {family}")
            for kind in ("raw", "calibrated"):
                reported = summary["models"][family]["probability_variants"][kind]
                close(np.average(group[f"{kind}_mean_probability"], weights=group["n"]), prediction[f"{kind}_probability"].mean(), f"Calibration mean: {family}/{kind}")
                close(reported["mean_probability"], prediction[f"{kind}_probability"].mean(), f"Calibration summary mean: {family}/{kind}")
                close(np.average(np.abs(group[f"{kind}_mean_probability"] - group["observed_fail_rate"]), weights=group["n"]), reported["ece_on_raw_score_deciles"], f"Calibration ECE: {family}/{kind}")
        topk = pd.read_csv(METRICS / "figure_topk_performance.csv")
        for family, group in topk.groupby("model"):
            prediction = self.predictions[family]
            computed = evaluate_probabilities(prediction["target"], prediction["calibrated_probability"], top_fractions=group["budget_fraction"].tolist())
            for row in group.to_dict("records"):
                prefix = f"top_{100 * row['budget_fraction']:g}pct_"
                for metric in ("selected_n", "selected_fraction", "expected_failures_found", "precision", "recall", "lift"):
                    close(row[metric], computed[prefix + metric], f"Top-K figure: {family}/{prefix}/{metric}")
        daily = pd.read_csv(METRICS / "figure_daily_capacity_yield.csv")
        for row in daily.to_dict("records"):
            prefix = f"daily_top_{100 * row['nominal_budget_fraction']:g}pct_"
            for metric in ("selected_n", "selected_fraction", "expected_failures_found", "precision", "recall", "random_expected_failures_found", "lift_vs_daily_random"):
                close(row[metric], self.comparison.loc[(row["model"], "calibrated"), prefix + metric], f"Daily capacity: {row['model']}/{metric}")
        return {"calibration_bins": len(bins), "pooled_curve_rows": len(topk), "daily_capacity_rows": len(daily)}

    def shap_and_errors(self) -> dict:
        cache_path = LOCAL / "shap/logistic_regression.joblib"
        cache = joblib.load(cache_path)
        frame, values = cache["test_frame"], cache["shap_values"]
        require(cache["feature_names"] == FEATURE_COLUMNS and values.shape == (len(self.test), len(FEATURE_COLUMNS)), "SHAP dimensions/feature allowlist")
        prediction = self.predictions["logistic_regression"].set_index("inspection_id").loc[frame["inspection_id"]]
        require(np.array_equal(prediction["target"].to_numpy(), frame["target"].to_numpy()), "SHAP outcome alignment")
        reconstructed = expit(cache["base_value"] + values.sum(axis=1))
        max_error = float(np.abs(reconstructed - prediction["calibrated_probability"].to_numpy()).max())
        require(max_error < 1e-12, "SHAP probability reconstruction")
        require(cache["reference_n"] == len(self.splits["train"]), "SHAP training reference size")
        importance = pd.read_csv(METRICS / "shap_global_importance.csv").set_index("feature")
        require(set(importance.index) == set(FEATURE_COLUMNS), "Global SHAP feature coverage")
        sample = values[cache["sample_indices"]]
        for index, feature in enumerate(FEATURE_COLUMNS):
            close(importance.loc[feature, "mean_abs_shap"], np.abs(sample[:, index]).mean(), f"SHAP mean absolute contribution: {feature}")
            close(importance.loc[feature, "mean_shap"], sample[:, index].mean(), f"SHAP mean contribution: {feature}")
        errors = read_json(METRICS / "error_analysis.json")
        require(errors["local_shap_source_sha256"] == sha256_file(cache_path), "Error/SHAP source hash")
        y, p = frame["target"].to_numpy(), cache["calibrated_probability"]
        weights = capacity_weights(p, int(np.ceil(len(frame) * 0.1)))
        actual = confusion(y, weights)
        for field, value in actual.items():
            close(value, errors["primary_operating_point"]["confusion"][field], f"Capacity confusion: {field}")
        fixed = confusion(y, (p >= 0.5).astype(float))
        for field, value in fixed.items():
            close(value, errors["separate_fixed_probability_operating_point"]["confusion"][field], f"Fixed-threshold confusion: {field}")
        groups = pd.read_csv(METRICS / "error_group_summary.csv").set_index("group")
        for label in ("TP", "FP", "FN", "TN"):
            close(groups.loc[label, "expected_n"], actual[label.lower()], f"Error profile count: {label}")
        subgroups = pd.read_csv(METRICS / "error_subgroup_metrics.csv")
        for feature, group in subgroups.groupby("feature"):
            close(group["n"].sum(), len(frame), f"Subgroup row coverage: {feature}")
            close(group["failures"].sum(), y.sum(), f"Subgroup failure coverage: {feature}")
            for label in ("tp", "fp", "fn", "tn"):
                close(group[f"global_top10_{label}"].sum(), actual[label], f"Subgroup confusion partition: {feature}/{label}")
        cases = pd.read_csv(METRICS / "error_case_summary.csv")
        require(len(cases) == 9 and cases["case_id"].nunique() == 9, "Local case coverage")
        close(expit(cases["reconstructed_calibrated_logit"]), cases["probability"], "Local case probability reconstruction", atol=1e-12)
        return {"global_features": len(importance), "global_plot_sample": len(sample), "full_test_explanations": len(frame),
                "maximum_probability_reconstruction_error": max_error, "local_cases": len(cases), "subgroup_rows": len(subgroups)}

    def lag_sensitivity(self) -> dict:
        summary = read_json(METRICS / "lag_sensitivity.json")
        table = pd.read_csv(METRICS / "lag_sensitivity.csv")
        require(not summary["fit_performed"] and not summary["model_selection_performed"], "Lag experiment fitted/selected a model")
        require(summary["aggregate_csv_sha256"] == sha256_file(METRICS / "lag_sensitivity.csv"), "Lag CSV hash")
        base_weights = {}
        for lag in (1, 7, 30):
            frame = self.test if lag == 1 else pd.read_parquet(LOCAL / f"lag_{lag}_features.parquet")
            frame = frame.sort_values("inspection_id").reset_index(drop=True)
            require(frame[ALIGNMENT].equals(self.test[ALIGNMENT]), f"Lag {lag} target alignment")
            changes = changed_features(self.test, frame)
            close(changes["row_feature_change_fraction"], summary["variants"][str(lag)]["feature_changes_vs_lag_1"]["row_feature_change_fraction"], f"Lag {lag} feature-change rate")
            for family in ("logistic_regression", "lightgbm"):
                path = LOCAL / "models" / f"{family}.joblib"
                require(sha256_file(path) == summary["model_sha256"][family], f"Lag frozen model hash: {family}")
                bundle = joblib.load(path)
                probabilities = bundle["calibrator"].predict(bundle["pipeline"].predict_proba(frame)[:, 1])
                computed = evaluate_probabilities(frame["target"], probabilities, dates=frame["inspection_date"], top_fractions=self.config["top_fractions"])
                row = table.loc[table["model"].eq(family) & table["lag_days"].eq(lag)].iloc[0]
                for metric, value in computed.items():
                    close(row[metric], value, f"Lag {lag}/{family}/{metric}")
                baseline = self.predictions[family]["calibrated_probability"].to_numpy()
                close(row["prediction_mean_absolute_change_vs_lag_1"], np.abs(probabilities - baseline).mean(), f"Lag MAD: {lag}/{family}")
                weights = capacity_weights(probabilities, int(np.ceil(len(frame) * 0.1)))
                if lag == 1:
                    base_weights[family] = weights
                close(row["top_10pct_member_retention_vs_lag_1"], np.minimum(weights, base_weights[family]).sum() / weights.sum(), f"Lag allocation overlap: {lag}/{family}")
        return {"rows": len(table), "lags_days": [1, 7, 30], "identical_targets": len(self.test), "refitting_performed": False}

    def figures(self) -> dict:
        specs = []
        for name in ("figure_performance_manifest.json", "shap_summary.json", "error_analysis.json"):
            specs.extend(read_json(METRICS / name)["figures"])
        for name in ("annual_drift_figure.json", "calibration_summary.json"):
            data = read_json(METRICS / name)
            specs.append({"path": data["figure"], "sha256": data["figure_sha256"], "bytes": data["figure_bytes"],
                          "width_px": data["pixel_dimensions"][0], "height_px": data["pixel_dimensions"][1], "dpi": data["dpi"]})
        actual_paths = {str(path.relative_to(ROOT)) for path in (ROOT / "reports/figures").glob("*.png")}
        require(len(specs) == 11 and actual_paths == {item["path"] for item in specs}, "Expected exactly eleven documented PNG figures")
        results = []
        for item in specs:
            path = ROOT / item["path"]
            require(sha256_file(path) == item["sha256"] and path.stat().st_size == item["bytes"], f"Figure hash/bytes: {item['path']}")
            with Image.open(path) as picture:
                require(picture.size == (item["width_px"], item["height_px"]), f"Figure dimensions: {item['path']}")
                require(all(abs(value - 300) < 0.1 for value in picture.info.get("dpi", (0, 0))), f"Figure DPI: {item['path']}")
                picture.verify()
            results.append(item)
        return {"count": len(results), "all_dpi": 300, "files": results}

    def documents(self) -> dict:
        checked, claims = [], 0
        for name in ("README.md", "reports/topic1_report.md"):
            path = ROOT / name
            text = path.read_text()
            links = [url for url in re.findall(r"\]\(([^)]+)\)", text) if not url.startswith(("https://", "http://", "#"))]
            for url in links:
                require((path.parent / url.split("#")[0]).exists(), f"Broken relative link in {name}: {url}")
            headers = None
            for line in text.splitlines():
                if not line.startswith("|"):
                    headers = None
                    continue
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if "ROC-AUC" in line and ("AP" in line or "PR-AUC" in line):
                    headers = cells
                    continue
                if headers is None or not cells:
                    continue
                family = next((family for token, family in (("LightGBM", "lightgbm"), ("LR", "logistic_regression"), ("基率", "prevalence")) if token in cells[0]), None)
                if family is None:
                    continue
                for header, cell in zip(headers[1:], cells[1:]):
                    metric = next((field for token, field in (("ROC-AUC", "roc_auc"), ("AP", "average_precision"), ("Brier", "brier_score"), ("LogLoss", "log_loss"), ("Precision", "top_10pct_precision")) if token in header), None)
                    if metric is None:
                        continue
                    numbers = re.findall(r"\d+(?:\.\d+)?", cell)
                    keys = [metric, f"{metric}_ci_lower", f"{metric}_ci_upper"]
                    for value, key in zip(numbers, keys):
                        expected = self.comparison.loc[(family, "calibrated"), key] * (100 if "%" in cell else 1)
                        digits = len(value.split(".")[1]) if "." in value else 0
                        require(abs(float(value) - expected) <= 0.5 * 10 ** (-digits) + 1e-10, f"Rounded model-table value in {name}: {family}/{key}")
                        claims += 1
            checked.append({"path": name, "sha256": sha256_file(path), "relative_links": len(links)})
        report = (ROOT / "reports/topic1_report.md").read_text()
        dictionary = report.split("### 4.1 数值特征", 1)[1].split("## 5.", 1)[0]
        fields = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", dictionary, re.M)
        require(len(fields) == len(FEATURE_COLUMNS) and set(fields) == set(FEATURE_COLUMNS), "Report feature dictionary must cover all 39 predictors once")
        require(claims > 0, "No model metric table was verified in the documents")
        return {"documents": checked, "rounded_metric_values_verified": claims, "feature_dictionary_entries": len(fields)}

    def git_hygiene(self) -> dict:
        tracked = git("ls-files", "-z").decode().split("\0")
        pending = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
        files = sorted({name for name in tracked + pending if name and (ROOT / name).is_file()})
        forbidden = {".parquet", ".joblib", ".pkl", ".pickle", ".npy", ".npz", ".gz", ".zip", ".bz2"}
        maximum = (0, "")
        for name in files:
            path = ROOT / name
            require(path.suffix.lower() not in forbidden, f"Data/model binary would be published: {name}")
            require(not name.startswith(("data/raw/", "data/processed/", ".local/", "artifacts/")), f"Local-only path would be published: {name}")
            require(path.suffix.lower() != ".csv" or name.startswith("reports/metrics/"), f"Non-summary CSV would be published: {name}")
            require(path.stat().st_size <= 1_500_000, f"File exceeds 1.5 MB delivery cap: {name}")
            maximum = max(maximum, (path.stat().st_size, name))
        ignored = {}
        for name in ("data/raw/_qa_probe.csv", "data/snapshots/inspections.parquet", "data/snapshots/licenses.parquet",
                     ".local/features.parquet", ".local/models/logistic_regression.joblib", ".local/lag_30_features.parquet"):
            result = git("check-ignore", "--no-index", "-v", name).decode().strip()
            require(result.startswith(".gitignore:"), f"Local dataset must be protected by committed .gitignore: {name}")
            ignored[name] = result.split("\t")[0]
        core = {}
        for filename in ("data.py", "features.py", "models.py", "metrics.py"):
            name = f"src/chicago_food/{filename}"
            starting_hash = hashlib.sha256(git("show", f"{START_COMMIT}:{name}")).hexdigest()
            require(starting_hash == sha256_file(ROOT / name), f"Reusable core module changed: {name}")
            core[name] = starting_hash
        return {"tracked_or_pending_files": len(files), "largest_file": {"path": maximum[1], "bytes": maximum[0]},
                "forbidden_files": [], "ignore_rules": ignored, "core_unchanged_since": START_COMMIT, "core_sha256": core}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pytest", action="store_true", help="Force pytest instead of reusing a matching test receipt")
    args = parser.parse_args()
    started = time.perf_counter()
    output = METRICS / "qa_summary.json"
    output.write_text(json.dumps({"status": "running", "passed": False}) + "\n")
    results = []

    def run(name, callback):
        before = time.perf_counter()
        try:
            detail = callback()
            results.append({"name": name, "passed": True, "elapsed_seconds": time.perf_counter() - before, "details": detail})
            print(f"PASS {name}", flush=True)
        except Exception as error:
            results.append({"name": name, "passed": False, "elapsed_seconds": time.perf_counter() - before,
                            "error": f"{type(error).__name__}: {str(error)[:240]}"})
            print(f"FAIL {name}: {type(error).__name__}; see qa_summary.json", flush=True)

    run("full_pytest", lambda: pytest_check(args.run_pytest))
    try:
        qa = DeliveryQA()
    except Exception as error:
        results.append({"name": "verified_inputs", "passed": False, "error": f"{type(error).__name__}: {str(error)[:240]}"})
    else:
        for name in ("data_and_splits", "models_and_intervals", "calibration_and_capacity", "shap_and_errors", "lag_sensitivity", "figures", "documents", "git_hygiene"):
            run(name, getattr(qa, name))
    passed = all(item["passed"] for item in results)
    summary = {
        "schema_version": 1, "status": "passed" if passed else "failed", "passed": passed,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "source_commit_at_check": git("rev-parse", "HEAD").decode().strip(),
        "checks": results, "passed_checks": sum(item["passed"] for item in results), "total_checks": len(results),
        "scope": "Numerical artifact consistency, current documentation links/tables, PNG integrity, full tests and publication hygiene; no model training.",
        "limits": "Event-time tests and delay sensitivity do not prove historical publication-time availability or remove retrospective source corrections.",
    }
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"QA {'passed' if passed else 'failed'}: {summary['passed_checks']}/{len(results)} checks; {summary['elapsed_seconds']:.2f}s", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
