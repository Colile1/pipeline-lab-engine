# Confirms analyse.py's reading of a recorded sweep, and in particular
# its power warning: the signed-rank test is discrete, so below a certain
# number of datasets no comparison can reach alpha however large the
# effect. Reporting "no difference" there would present an arithmetic
# limit as evidence.
import sys
from pathlib import Path

import pytest
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyse import _datasets_needed, complete_datasets, wilcoxon_floor


class Row(dict):
    """A stand-in for sqlite3.Row, which indexes by column name."""


@pytest.mark.parametrize("n_pairs,expected", [
    (4, 0.125),
    (5, 0.0625),
    (6, 0.03125),
    (7, 0.015625),
    (8, 0.0078125),
])
def test_the_floor_matches_what_scipy_actually_returns(n_pairs, expected):
    """Non-vacuous: the claimed floor must equal a real perfect-separation
    run through scipy, not just a formula asserted against itself."""
    actual = wilcoxon([1.0] * n_pairs).pvalue
    assert actual == pytest.approx(expected, abs=5e-5)
    assert wilcoxon_floor(n_pairs) == pytest.approx(actual, abs=5e-5)


def test_five_datasets_cannot_clear_the_conventional_alpha():
    """The case that prompted this: the regression arm had 5 datasets."""
    assert wilcoxon_floor(5) > 0.05


def test_six_datasets_clear_alpha_but_not_holm():
    """The subtlety that made the first version of this warning wrong.

    Holm tests the smallest of k comparisons at alpha/k, so clearing
    alpha is not enough. With 3 levels against a baseline the bar is
    0.0167, and n = 6 tops out at 0.03125: a perfect 6/6 sweep still
    cannot be rejected.
    """
    assert wilcoxon_floor(6) < 0.05
    assert wilcoxon_floor(6) > 0.05 / 3


def test_seven_datasets_are_the_real_minimum_under_holm():
    assert wilcoxon_floor(7) <= 0.05 / 3
    assert _datasets_needed(0.05 / 3) == 7


def test_datasets_needed_tracks_the_threshold():
    """A laxer threshold needs fewer datasets, a stricter one more."""
    assert _datasets_needed(0.05) == 6
    assert _datasets_needed(0.05 / 3) == 7
    assert _datasets_needed(0.05 / 10) == 9


def test_a_dataset_missing_one_level_is_excluded():
    """Friedman and Wilcoxon are paired across datasets, so an
    incomplete dataset would silently compare different sets."""
    levels = ["none", "standard", "minmax"]
    rows = [
        Row(dataset="a", model="svm", scaling=lv, accuracy_mean=0.9)
        for lv in levels
    ] + [
        # "b" never ran at minmax.
        Row(dataset="b", model="svm", scaling=lv, accuracy_mean=0.8)
        for lv in ["none", "standard"]
    ]
    keep = complete_datasets(
        rows, "scaling", levels, "svm", "accuracy_mean", ["a", "b"]
    )
    assert keep == ["a"]


def test_a_dataset_with_a_null_score_is_excluded():
    """A recorded row with no score is as unusable as a missing one:
    a skipped run writes its reason, not a number."""
    levels = ["none", "standard"]
    rows = [
        Row(dataset="a", model="svm", scaling="none", accuracy_mean=0.9),
        Row(dataset="a", model="svm", scaling="standard", accuracy_mean=0.95),
        Row(dataset="b", model="svm", scaling="none", accuracy_mean=0.8),
        Row(dataset="b", model="svm", scaling="standard", accuracy_mean=None),
    ]
    keep = complete_datasets(
        rows, "scaling", levels, "svm", "accuracy_mean", ["a", "b"]
    )
    assert keep == ["a"]
