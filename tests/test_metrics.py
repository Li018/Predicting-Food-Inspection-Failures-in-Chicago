import unittest

import numpy as np

from chicago_food.metrics import capacity_weights, cluster_bootstrap_ci, evaluate_probabilities


class MetricsTests(unittest.TestCase):
    def test_constant_predictions_have_random_expected_lift(self):
        values = evaluate_probabilities([0, 0, 1, 1], [0.25] * 4, top_fractions=[0.25])
        self.assertEqual(values["average_precision"], 0.5)
        self.assertEqual(values["top_25pct_precision"], 0.5)
        self.assertEqual(values["top_25pct_recall"], 0.25)
        self.assertEqual(values["top_25pct_lift"], 1.0)

    def test_boundary_ties_do_not_depend_on_input_order(self):
        scores = np.array([0.9, 0.5, 0.5, 0.1])
        weights = capacity_weights(scores, 2)
        np.testing.assert_array_equal(weights, [1.0, 0.5, 0.5, 0.0])
        self.assertEqual(float(weights.sum()), 2)
        y = [0, 1, 0, 1]
        original = evaluate_probabilities(y, scores, top_fractions=[0.5])
        reversed_order = evaluate_probabilities(y[::-1], scores[::-1], top_fractions=[0.5])
        self.assertEqual(original, reversed_order)

    def test_daily_capacity_uses_ceiling_and_differs_from_pooled(self):
        result = evaluate_probabilities(
            [1, 0, 0, 1], [0.9, 0.8, 0.7, 0.1],
            dates=["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-02"],
            top_fractions=[0.25],
        )
        self.assertEqual(result["top_25pct_selected_n"], 1)
        self.assertEqual(result["daily_top_25pct_selected_n"], 2)
        self.assertEqual(result["daily_top_25pct_selected_fraction"], 0.5)
        self.assertEqual(result["daily_top_25pct_recall"], 1.0)

    def test_single_class_metric_is_explicitly_undefined(self):
        result = evaluate_probabilities([0, 0], [0.2, 0.4])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["top_10pct_recall"])
        self.assertIsNone(result["top_10pct_lift"])
        self.assertEqual(result["average_precision"], 0.0)

    def test_bootstrap_is_seeded_and_clustered(self):
        y = [0, 0, 1, 1, 0, 1]
        p = [0.2, 0.3, 0.8, 0.7, 0.3, 0.6]
        kwargs = {"n_replicates": 20, "seed": 42, "top_fractions": [0.5]}
        first = cluster_bootstrap_ci(y, p, ["a", "a", "b", "b", "c", "c"], **kwargs)
        second = cluster_bootstrap_ci(y, p, ["a", "a", "b", "b", "c", "c"], **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["n_entities"], 3)
        self.assertEqual(first["intervals"]["brier_score"]["valid_replicates"], 20)

    def test_invalid_probabilities_rejected(self):
        for p in ([0.2, float("nan")], [0.2, 1.2], [0.2, -0.1]):
            with self.assertRaises(ValueError):
                evaluate_probabilities([0, 1], p)


if __name__ == "__main__":
    unittest.main()
