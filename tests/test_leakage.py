# The leakage proofs. Feature selection and hyperparameter tuning are the
# two stages that silently inflate scores if fitted outside the fold, and
# they are exactly the two components neither B1 nor B2 implements. These
# tests assert the property structurally with a fit-spy, rather than
# trusting that the pipeline was assembled correctly -- the same approach
# prepfx uses to prove its own imputer/scaler are leakage-safe.
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from piplab.components import selection
from piplab.pipelines.spies import FitSpy

SEED = 671


@pytest.fixture
def data():
    """A small, deterministic classification frame with a known index."""
    rng = np.random.default_rng(SEED)
    n = 60
    X = pd.DataFrame(
        rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)], index=range(n)
    )
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int), index=X.index)
    return X, y


def _fit_across_folds(pipeline, X, y, k=5):
    """Fit a clone of `pipeline` on each training fold, returning test indices."""
    from sklearn.base import clone

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    test_sets = []
    for train_idx, test_idx in splitter.split(X, y):
        est = clone(pipeline)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        test_sets.append(frozenset(X.index[test_idx]))
    return test_sets


# --- feature selection ----------------------------------------------------

@pytest.mark.parametrize("level", ["chi2", "mutual_info", "rfe", "lasso"])
def test_selection_never_sees_a_test_fold(level, data):
    """The core proof: a selector must only ever be fitted on training rows."""
    X, y = data
    if level == "chi2":
        X = X.abs()  # chi2 requires non-negative features
    FitSpy.reset()

    selector = selection.make_selector(level, "classification", SEED, n_features=X.shape[1])
    pipeline = Pipeline([
        ("select", FitSpy(selector, "select")),
        ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    test_sets = _fit_across_folds(pipeline, X, y)

    records = FitSpy.records("select")
    assert len(records) == len(test_sets), "selector should be fitted once per fold"
    for record, held_out in zip(records, test_sets):
        overlap = record["row_ids"] & held_out
        assert not overlap, (
            f"{level} selector was fitted on {len(overlap)} held-out rows — leakage"
        )


def test_selection_is_refitted_per_fold_not_once_globally(data):
    """A selector fitted once on everything would show a single full-size fit."""
    X, y = data
    FitSpy.reset()
    pipeline = Pipeline([
        ("select", FitSpy(selection.make_selector("mutual_info", "classification", SEED, 6), "select")),
        ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    _fit_across_folds(pipeline, X, y, k=5)

    records = FitSpy.records("select")
    assert len(records) == 5
    # Every fit saw a strict subset of the data, never the whole frame.
    for record in records:
        assert record["n_rows"] < len(X)


# --- hyperparameter tuning ------------------------------------------------

def test_nested_tuning_never_lets_the_search_see_the_outer_test_fold(data):
    """Nested CV: the inner search must stay inside the outer training part.

    This is the subtler leak. Tuning on the folds that are also reported
    biases the score upward even with no per-fold leakage, because the
    hyperparameters were chosen to fit those folds.
    """
    X, y = data
    FitSpy.reset()

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    base = Pipeline([
        ("select", FitSpy(selection.make_selector("mutual_info", "classification", SEED, 6), "inner_select")),
        ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    search = GridSearchCV(
        base, param_grid={"model__C": [0.1, 1.0]}, cv=inner_cv, scoring="accuracy", n_jobs=1
    )
    test_sets = _fit_across_folds(search, X, y, k=5)

    records = FitSpy.records("inner_select")
    assert records, "the inner pipeline should have been fitted many times"
    # Fits are emitted per outer fold in sequence, so partition them and
    # assert each group avoids its own outer test set. (A global check
    # would be wrong: a fit from outer fold 2 legitimately contains rows
    # that are held out in outer fold 1.)
    per_outer = len(records) // len(test_sets)
    assert per_outer > 1, "nested search should fit more than once per outer fold"
    for i, held_out in enumerate(test_sets):
        group = records[i * per_outer:(i + 1) * per_outer]
        for record in group:
            overlap = record["row_ids"] & held_out
            assert not overlap, (
                f"inner search saw {len(overlap)} rows from outer fold {i}'s test set — leakage"
            )


def test_inner_fits_are_strictly_smaller_than_the_outer_training_part(data):
    """Inner folds subdivide the outer training part, so they must be smaller."""
    X, y = data
    FitSpy.reset()
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    base = Pipeline([
        ("select", FitSpy(selection.make_selector("mutual_info", "classification", SEED, 6), "inner")),
        ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    search = GridSearchCV(base, {"model__C": [1.0]}, cv=inner_cv, scoring="accuracy", n_jobs=1)
    _fit_across_folds(search, X, y, k=5)

    outer_train_size = len(X) * 4 // 5
    inner_sizes = [r["n_rows"] for r in FitSpy.records("inner")]
    # The refit on the full outer training part is the largest fit; every
    # search fit must be smaller than it.
    assert min(inner_sizes) < outer_train_size
    assert max(inner_sizes) <= outer_train_size


# --- the spy itself -------------------------------------------------------

def test_fitspy_records_what_it_was_fitted_on(data):
    X, y = data
    FitSpy.reset()
    spy = FitSpy(selection.make_selector("mutual_info", "classification", SEED, 6), "solo")
    spy.fit(X.iloc[:10], y.iloc[:10])
    records = FitSpy.records("solo")
    assert len(records) == 1
    assert records[0]["n_rows"] == 10
    assert records[0]["indexed"] is True
    assert records[0]["row_ids"] == frozenset(range(10))
    assert records[0]["y_fingerprint"] == frozenset(range(10))


def test_fitspy_reset_clears_records(data):
    X, y = data
    FitSpy.reset()
    FitSpy(selection.make_selector("mutual_info", "classification", SEED, 6), "x")
    FitSpy.reset()
    assert FitSpy.records("x") == []
