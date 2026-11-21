"""Projection exports preserve signed explanations and original output paths."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/make_slide_figures.py"
SPEC = importlib.util.spec_from_file_location("slide_figures", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_top_five_waterfall_closes_signed_residual():
    explanation = {
        "base_value": -1.2,
        "shap_values": np.array([[3.0, -2.5, 1.5, -1.0, 0.8, 0.2, -0.1, -0.15]]),
        "feature_names": [f"feature_{i}" for i in range(8)],
    }
    labels, contributions, cumulative = module.reduced_waterfall(explanation, 0)
    assert len(labels) == len(contributions) == 6
    assert labels[-1] == "Other 3 features"
    assert contributions[-1] == pytest.approx(-0.05)
    assert cumulative[-1] == pytest.approx(0.55)
    assert cumulative[-1] == pytest.approx(explanation["base_value"] + explanation["shap_values"].sum())


def test_slide_output_cannot_target_original_figures(tmp_path):
    original = tmp_path / "reports/figures"
    original.mkdir(parents=True)
    for destination in (original, original / "nested"):
        with pytest.raises(ValueError, match="must not overwrite"):
            module.check_output_directory(tmp_path, destination)
    alias = tmp_path / "old-figure-alias"
    alias.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="must not overwrite"):
        module.check_output_directory(tmp_path, alias)
    module.check_output_directory(tmp_path, tmp_path / "reports/slide_figures")


def test_slide_numeric_serialization_is_finite_and_rounded():
    source = {"ap": np.float64(0.314566113), "n": np.int64(8363),
              "missing": float("nan"), "nested": [np.float64(-0.00062351), np.bool_(True)]}
    assert module.rounded(source) == {
        "ap": 0.3146, "n": 8363, "missing": None, "nested": [-0.0006, True],
    }


def test_projection_font_contract():
    assert module.FONT_SIZES["title"] >= 15
    assert min(module.FONT_SIZES.values()) >= 12
    assert len(module.FIGURE_NAMES) == 9
