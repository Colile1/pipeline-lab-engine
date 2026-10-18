"""Tests for the dataset characteristics copied onto every run row.

feature_scale_ratio was added after the meta-analysis found that the
study had no statistic measuring scale disparity, and had used
mean_abs_correlation as a proxy that measures something else entirely.
These tests pin the property it is supposed to capture.
"""
import numpy as np
import pandas as pd

from piplab.datasets.loaders import characterise


def _chars(frame):
    y = pd.Series([0, 1] * (len(frame) // 2) + [0] * (len(frame) % 2))
    return characterise(frame, y, "classification")


def test_scale_ratio_is_one_for_features_on_the_same_scale():
    rng = np.random.default_rng(671)
    frame = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(5)})
    assert _chars(frame)["feature_scale_ratio"] < 1.5


def test_scale_ratio_detects_one_dominant_feature():
    """The house_16H pathology: one feature vastly outscaling the rest."""
    rng = np.random.default_rng(671)
    frame = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(5)})
    frame["dominant"] = rng.normal(0, 66_000, 200)
    assert _chars(frame)["feature_scale_ratio"] > 10_000


def test_scale_ratio_uses_the_median_not_the_minimum():
    """A single near-constant column must not send the ratio to infinity.

    Using min(std) as the denominator would make this unbounded, which
    would flag ordinary datasets that merely carry one flat column.
    """
    rng = np.random.default_rng(671)
    frame = pd.DataFrame({f"f{i}": rng.normal(0, 1, 200) for i in range(5)})
    frame["almost_constant"] = 1e-12 * rng.normal(0, 1, 200)
    assert _chars(frame)["feature_scale_ratio"] < 10


def test_scale_ratio_is_one_when_there_is_nothing_to_compare():
    frame = pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0]})
    assert _chars(frame)["feature_scale_ratio"] == 1.0


def test_scale_ratio_ignores_categorical_columns():
    """Scale is a property of numeric features; text columns have none."""
    rng = np.random.default_rng(671)
    frame = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
        "label": ["x", "y"] * 50,
    })
    assert _chars(frame)["feature_scale_ratio"] < 1.5


def test_scale_ratio_survives_a_constant_column():
    """std of zero must not produce a nan or a division error."""
    rng = np.random.default_rng(671)
    frame = pd.DataFrame({
        "a": rng.normal(0, 1, 100),
        "b": rng.normal(0, 1, 100),
        "flat": np.ones(100),
    })
    value = _chars(frame)["feature_scale_ratio"]
    assert np.isfinite(value)
