import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from chicago_food.models import (
    MonotoneSigmoidCalibrator,
    fit_pipeline,
    make_pipeline,
    temporal_splits,
    train_and_evaluate,
)


CONFIG = {
    "train_start": "2019-01-01", "train_end": "2022-12-31",
    "validation_start": "2023-01-01", "validation_end": "2023-12-31",
    "calibration_start": "2024-01-01", "calibration_end": "2024-12-31",
    "test_start": "2025-01-01", "test_end": "2025-12-31",
    "random_seed": 3, "top_fractions": [0.1], "bootstrap_replicates": 10, "n_jobs": 1,
}


def cohort():
    rng = np.random.default_rng(8)
    rows = []
    for year in (2020, 2023, 2024, 2025):
        for i in range(60):
            signal = rng.uniform()
            rows.append({
                "inspection_id": year * 1000 + i, "entity_id": f"site-{i}",
                "inspection_date": pd.Timestamp(year, 1, 1) + pd.Timedelta(days=i // 3),
                "target": int(signal > 0.6), "history_fail_rate": signal,
                "previous_type": "a" if i % 2 else "b",
            })
    return pd.DataFrame(rows)


class ModelTests(unittest.TestCase):
    def test_temporal_boundaries_and_overlap_rejection(self):
        frame = cohort()
        splits = temporal_splits(frame, CONFIG)
        self.assertEqual([len(value) for value in splits.values()], [60, 60, 60, 60])
        invalid = {**CONFIG, "validation_start": "2022-01-01"}
        with self.assertRaises(ValueError):
            temporal_splits(frame, invalid)

    def test_current_event_and_identifier_features_forbidden(self):
        for forbidden in ("target", "inspection_id", "license_number", "risk", "facility_type"):
            with self.assertRaises(ValueError):
                make_pipeline("logistic_regression", [forbidden], [], seed=1)

    def test_preprocessor_learns_only_training_categories_and_median(self):
        train = pd.DataFrame({"history_fail_rate": [0.1, 0.2, np.nan, 0.9] * 10,
                              "previous_type": ["a", "b", "a", "b"] * 10})
        y = [0, 0, 1, 1] * 10
        model = make_pipeline("logistic_regression", ["history_fail_rate"], ["previous_type"], seed=1)
        fit_pipeline(model, "logistic_regression", train, y,
                     ["history_fail_rate"], ["previous_type"])
        imputer = model["preprocess"].named_transformers_["num"]["impute"]
        encoder = model["preprocess"].named_transformers_["cat"]["encode"]
        self.assertAlmostEqual(imputer.statistics_[0], 0.2)
        heldout = pd.DataFrame({"history_fail_rate": [1000, np.nan],
                               "previous_type": ["FUTURE_ONLY", None]})
        result = model.predict_proba(heldout)
        self.assertTrue(np.isfinite(result).all())
        self.assertNotIn("FUTURE_ONLY", encoder.categories_[0])
        self.assertAlmostEqual(imputer.statistics_[0], 0.2)

    def test_monotone_calibrator_keeps_order_even_when_calibration_is_reversed(self):
        probabilities = np.linspace(0.05, 0.95, 20)
        labels = np.r_[np.ones(10), np.zeros(10)]
        calibrator = MonotoneSigmoidCalibrator().fit(probabilities, labels)
        calibrated = calibrator.predict(probabilities)
        self.assertGreater(calibrator.slope_, 0)
        self.assertTrue((np.diff(calibrated) > 0).all())
        self.assertTrue(((calibrated > 0) & (calibrated < 1)).all())

    def test_calibrator_rejects_invalid_inputs_and_unfitted_use(self):
        with self.assertRaises(ValueError):
            MonotoneSigmoidCalibrator().predict([0.2])
        with self.assertRaises(ValueError):
            MonotoneSigmoidCalibrator().fit([0.2, 0.4], [1, 1])
        with self.assertRaises(ValueError):
            MonotoneSigmoidCalibrator().fit([np.nan, 0.4], [0, 1])

    def test_test_outcomes_cannot_change_fitted_models_or_predictions(self):
        original = cohort()
        modified = original.copy()
        is_test = modified["inspection_date"].dt.year == 2025
        modified.loc[is_test, "target"] = 1 - modified.loc[is_test, "target"]
        numeric, categorical = ["history_fail_rate"], ["previous_type"]
        grid = [{"num_leaves": 7, "n_estimators": 20, "min_child_samples": 5,
                 "learning_rate": 0.1}]
        with TemporaryDirectory() as directory, patch(
            "chicago_food.models.LOGISTIC_CANDIDATES", [{"C": 1.0}]
        ), patch("chicago_food.models.LIGHTGBM_CANDIDATES", grid):
            first = train_and_evaluate(original, CONFIG, Path(directory) / "a",
                                       numeric_features=numeric, categorical_features=categorical)
            second = train_and_evaluate(modified, CONFIG, Path(directory) / "b",
                                        numeric_features=numeric, categorical_features=categorical)
            self.assertEqual(first["selected_name"], second["selected_name"])
            self.assertEqual(first["summary"]["selected_parameters"],
                             second["summary"]["selected_parameters"])
            self.assertEqual(first["summary"]["calibration"], second["summary"]["calibration"])
            np.testing.assert_array_equal(first["predictions"]["probability"],
                                          second["predictions"]["probability"])
            pd.testing.assert_frame_equal(first["validation_search"], second["validation_search"])
            self.assertEqual(first["calibrator"].n_samples_, 60)
            saved = json.loads((Path(directory) / "a" / "summary.json").read_text())
            self.assertEqual(saved["estimator_fit_period"], ["2019-01-01", "2022-12-31"])


if __name__ == "__main__":
    unittest.main()
