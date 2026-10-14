# Confirms each component factory returns an UNFITTED estimator for every
# Chapter 1 technique level, that row deletion behaves as a dataset-level
# baseline rather than a pipeline step, and that the tuning cost model
# matches what nested CV actually costs.
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from piplab.components import encoding, imputation, scaling, selection, tuning
from piplab.config import TECHNIQUES


def _is_unfitted(est) -> bool:
    """True when est is a real estimator that has not been fitted yet."""
    if est == "passthrough":
        return True
    try:
        check_is_fitted(est)
    except NotFittedError:
        return True
    return False


# --- every level of every component builds, unfitted ----------------------

@pytest.mark.parametrize("level", TECHNIQUES["scaling"])
def test_every_scaling_level_builds_unfitted(level):
    est = scaling.make_scaler(level, seed=671)
    assert _is_unfitted(est)


@pytest.mark.parametrize("level", TECHNIQUES["imputation"])
def test_every_imputation_level_builds_unfitted(level):
    est = imputation.make_imputer(level, seed=671)
    assert _is_unfitted(est)


@pytest.mark.parametrize("level", TECHNIQUES["selection"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_selection_level_builds_unfitted(level, task):
    est = selection.make_selector(level, task=task, seed=671, n_features=10)
    assert _is_unfitted(est)
    if est != "passthrough":
        assert isinstance(est, BaseEstimator)


@pytest.mark.parametrize("level", TECHNIQUES["encoding"])
def test_every_encoding_level_builds_unfitted(level):
    assert _is_unfitted(encoding.make_encoder(level, seed=671))


@pytest.mark.parametrize("component,fn", [
    ("scaling", lambda: scaling.make_scaler("bogus", 1)),
    ("imputation", lambda: imputation.make_imputer("bogus", 1)),
    ("encoding", lambda: encoding.make_encoder("bogus", 1)),
])
def test_unknown_level_is_rejected_by_name(component, fn):
    with pytest.raises(ValueError, match="bogus"):
        fn()


def test_unknown_selection_level_and_task_are_rejected():
    with pytest.raises(ValueError, match="bogus"):
        selection.make_selector("bogus", "classification", 1, 10)
    with pytest.raises(ValueError, match="unknown task"):
        selection.make_selector("none", "clustering", 1, 10)


# --- row deletion: Chapter 1's imputation baseline ------------------------

def test_drop_rows_is_flagged_for_the_evaluator_not_the_pipeline():
    # A transformer may not change sample count, so the evaluator has to
    # apply row deletion before folds are built.
    assert imputation.requires_row_deletion("drop_rows") is True
    assert imputation.requires_row_deletion("median") is False
    # ...and in-pipeline it must be a passthrough, never a silent impute.
    assert imputation.make_imputer("drop_rows", 671) == "passthrough"


def test_rows_to_keep_masks_only_complete_rows():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, np.nan, 4.0]})
    mask = imputation.rows_to_keep(X)
    assert list(mask) == [True, False, False, True]


def test_deletion_report_quantifies_what_the_baseline_discards():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, np.nan, 4.0]})
    report = imputation.deletion_report(X)
    assert report == {
        "rows_before": 4, "rows_after": 2, "rows_dropped": 2, "pct_dropped": 50.0,
    }


def test_deletion_report_on_complete_data_drops_nothing():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert imputation.deletion_report(X)["pct_dropped"] == 0.0


# --- selection: k resolution and task substitution ------------------------

@pytest.mark.parametrize("n_features,expected", [(10, 5), (1, 1), (3, 2), (0, 1)])
def test_resolve_k_never_exceeds_or_empties_the_feature_set(n_features, expected):
    assert selection.resolve_k(n_features) == expected


def test_chi2_substitutes_for_regression_and_says_so():
    # chi2 needs a classification target; the fallback must be recorded,
    # never silent.
    note = selection.substitution_note("chi2", "regression")
    assert note and "f_regression" in note
    assert selection.substitution_note("chi2", "classification") is None
    assert selection.substitution_note("rfe", "regression") is None


# --- tuning: nested-CV cost model -----------------------------------------

def test_no_tuning_costs_one_fit_per_outer_fold():
    assert tuning.fits_per_run("none", "logreg", outer_k=10, inner_k=3) == 10


def test_grid_cost_is_candidates_x_inner_x_outer_plus_refits():
    # logreg's grid has 4 candidates: 10 * (4*3 + 1) = 130 fits.
    assert tuning.fits_per_run("grid", "logreg", outer_k=10, inner_k=3) == 130
    # svm's grid is 3 * 2 = 6 candidates: 10 * (6*3 + 1) = 190.
    assert tuning.fits_per_run("grid", "svm", outer_k=10, inner_k=3) == 190


def test_random_and_bayesian_cost_scales_with_n_iter():
    expected = 10 * (tuning.DEFAULT_N_ITER * 3 + 1)
    assert tuning.fits_per_run("random", "logreg", 10, 3) == expected
    assert tuning.fits_per_run("bayesian", "svm", 10, 3) == expected


def test_tuning_is_far_more_expensive_than_none_which_is_why_cost_is_shown():
    cheap = tuning.fits_per_run("none", "svm", 10, 3)
    dear = tuning.fits_per_run("grid", "svm", 10, 3)
    assert dear > cheap * 15


def test_every_model_has_a_grid():
    from piplab.config import MODELS
    for model in MODELS:
        assert tuning.grid_for(model)


def test_unknown_model_grid_is_rejected():
    with pytest.raises(ValueError, match="no tuning grid"):
        tuning.grid_for("xgboost")


def test_tuning_none_returns_the_pipeline_untouched():
    sentinel = object()
    assert tuning.make_search("none", sentinel, "logreg", "accuracy", None, 671) is sentinel


def test_unknown_tuning_level_is_rejected():
    with pytest.raises(ValueError, match="bogus"):
        tuning.make_search("bogus", None, "logreg", "accuracy", None, 671)


# --- selection: the two failures the wide v2 sweep exposed ---------------
# Both were latent until the selection arm ran on 19 datasets instead of
# the handful the unit tests used. Neither is exotic: a dataset with
# negative values, and a target with three classes.


def test_chi2_falls_back_when_a_feature_is_negative():
    """chi2 is undefined on negative input, and the scaling baseline is
    'none', so an unscaled dataset with negative values reaches it raw.

    Shifting the data would be a scaling operation performed inside the
    selection component, contaminating the factor under test with the
    one held fixed. f_classif is used instead.
    """
    rng = np.random.default_rng(671)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, size=60))
    assert X.to_numpy().min() < 0, "fixture must actually contain negatives"

    selector = selection.make_selector("chi2", "classification", 671, X.shape[1])
    selector.fit(X, y)  # raised ValueError("Input X must be non-negative") before


def test_the_negative_feature_substitution_is_recorded():
    """A silent fallback would let a reader believe chi2 ran."""
    rng = np.random.default_rng(671)
    negative = pd.DataFrame(rng.normal(size=(20, 3)))
    positive = negative.abs()

    note = selection.substitution_note("chi2", "classification", negative)
    assert note and "f_classif" in note
    # Non-vacuous: no note when chi2 genuinely ran.
    assert selection.substitution_note("chi2", "classification", positive) is None


def test_chi2_accepts_sparse_input():
    """onehot keeps the ColumnTransformer output sparse once a dataset has
    enough categorical cardinality (adult, bank-marketing). np.asarray
    cannot cast a csr_matrix and raised "setting an array element with a
    sequence", failing chi2 on every model for adult. Found 2026-10-14
    when ofat_selection_large reached it; latent until a dataset this
    wide was run."""
    import scipy.sparse as sp

    rng = np.random.default_rng(671)
    dense = rng.integers(0, 2, size=(60, 8)).astype(float)
    X = sp.csr_matrix(dense)
    y = pd.Series(rng.integers(0, 2, size=60))

    selector = selection.make_selector("chi2", "classification", 671, X.shape[1])
    selector.fit(X, y)  # raised ValueError before

    # The substitution-note path hits the same cast and must not silently
    # swallow a genuine substitution on sparse input either.
    assert selection.substitution_note("chi2", "classification", X) is None


def test_lasso_selection_handles_a_multiclass_target():
    """liblinear cannot fit multiclass L1 and raised on any 3-class
    target. iris and wine are both 3-class, so this failed 8 runs."""
    rng = np.random.default_rng(671)
    X = pd.DataFrame(rng.normal(size=(90, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.integers(0, 3, size=90))
    assert y.nunique() == 3, "fixture must be genuinely multiclass"

    selector = selection.make_selector("lasso", "classification", 671, X.shape[1])
    selector.fit(X, y)  # raised ValueError("does not support multiclass") before


def test_lasso_still_works_on_a_binary_target():
    """The solver change must not break the case that already worked."""
    rng = np.random.default_rng(671)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, size=60))
    selector = selection.make_selector("lasso", "classification", 671, X.shape[1])
    selector.fit(X, y)


# --- selection: k is configurable, not hardcoded at half -----------------
# The first selection sweep found every technique cost accuracy, but could
# not say whether selection hurts or whether keeping HALF the features
# hurts, because resolve_k was fixed at 0.5 everywhere. k_fraction makes
# the two separable.


@pytest.mark.parametrize("fraction,expected", [(0.25, 15), (0.5, 30), (0.75, 45)])
def test_k_fraction_changes_how_many_features_survive(fraction, expected):
    rng = np.random.default_rng(671)
    X = pd.DataFrame(rng.random(size=(80, 60)), columns=[f"f{i}" for i in range(60)])
    y = pd.Series(rng.integers(0, 2, size=80))
    selector = selection.make_selector(
        "chi2", "classification", 671, X.shape[1], k_fraction=fraction
    )
    selector.fit(X, y)
    assert int(selector.get_support().sum()) == expected


def test_k_fraction_defaults_to_the_declared_half():
    """Omitting it must reproduce the behaviour every recorded run used."""
    rng = np.random.default_rng(671)
    X = pd.DataFrame(rng.random(size=(40, 20)), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(rng.integers(0, 2, size=40))
    default = selection.make_selector("chi2", "classification", 671, X.shape[1])
    explicit = selection.make_selector(
        "chi2", "classification", 671, X.shape[1], k_fraction=0.5
    )
    default.fit(X, y)
    explicit.fit(X, y)
    assert (default.get_support() == explicit.get_support()).all()


def test_the_k_default_has_one_source_of_truth():
    """config declares it; selection re-exports it. If they drifted, a
    sweep's recorded k_fraction would not match what actually ran."""
    from piplab.config import DEFAULT_K_FRACTION as declared

    assert selection.DEFAULT_K_FRACTION is declared


def test_lasso_ignores_k_fraction_because_it_chooses_its_own_count():
    """Embedded selection is data-driven; k does not apply to it. This is
    why lasso sat near zero in the sweep while the filters all lost."""
    rng = np.random.default_rng(671)
    X = pd.DataFrame(rng.normal(size=(80, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.integers(0, 2, size=80))
    quarter = selection.make_selector(
        "lasso", "classification", 671, X.shape[1], k_fraction=0.25
    )
    three_quarters = selection.make_selector(
        "lasso", "classification", 671, X.shape[1], k_fraction=0.75
    )
    quarter.fit(X, y)
    three_quarters.fit(X, y)
    assert (quarter.get_support() == three_quarters.get_support()).all()


# --- tuning: random and Bayesian search a continuous space ----------------
# In the first tuning sweep random search equalled grid search on all 56
# model-dataset pairs, because every grid was smaller than n_iter.


@pytest.mark.parametrize("model", ["logreg", "svm", "random_forest", "gradient_boosting"])
def test_every_search_space_is_larger_than_the_random_budget(model):
    """The condition that made random == grid must not be able to recur."""
    size = 1
    for spec in tuning.space_for(model).values():
        if spec[0] == "choice":
            size *= len(spec[1])
        elif spec[0] == "int":
            size *= spec[2] - spec[1] + 1
        else:
            size = float("inf")  # a continuous range cannot be exhausted
    assert size > tuning.DEFAULT_N_ITER


def test_random_search_no_longer_draws_from_the_grid():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("model", LogisticRegression(max_iter=200))])
    search = tuning.make_search("random", pipe, "logreg", "accuracy", 3, 671)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    search.fit(X, y)
    tried = {round(float(p["model__C"]), 10) for p in search.cv_results_["params"]}
    assert len(tried) == tuning.DEFAULT_N_ITER, "ten distinct draws, not a recycled grid"
    assert not tried <= set(tuning.grid_for("logreg")["model__C"])


def test_grid_search_still_uses_the_small_grid():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("model", LogisticRegression(max_iter=200))])
    search = tuning.make_search("grid", pipe, "logreg", "accuracy", 3, 671)
    assert search.param_grid == tuning.grid_for("logreg")


def test_searched_parameters_follow_the_level():
    assert tuning.searched_parameters("none", "random_forest") == []
    assert "model__min_samples_leaf" not in tuning.searched_parameters("grid", "random_forest")
    assert "model__min_samples_leaf" in tuning.searched_parameters("random", "random_forest")
    assert (tuning.searched_parameters("random", "svm")
            == tuning.searched_parameters("bayesian", "svm"))
