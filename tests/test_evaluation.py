# Confirms the evaluation protocols: k=10 stratified as the primary
# path, nested CV whenever tuning is on, row deletion applied before
# folds are built, and repeated holdout REFUSING a dataset where a 15%
# test set would be too small to mean anything.
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from piplab.evaluation import (
    EvaluationResult,
    ProtocolError,
    apply_row_deletion,
    check_holdout_eligible,
    evaluate,
    fold_hash,
    fold_seed,
    make_splitter,
    metrics_for,
    scoring_for,
    summarise,
)
from piplab.config import HOLDOUT_MIN_ROWS

SEED = 671
BASE = {"encoding": "onehot", "imputation": "median", "scaling": "standard",
        "selection": "none", "tuning": "none"}


@pytest.fixture
def wine():
    d = load_wine(as_frame=True)
    return d.data, d.target


@pytest.fixture
def small_reg():
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X["f0"] * 2 + rng.normal(scale=0.2, size=60))
    return X, y


# --- metrics: methodology section 9's set, nothing more -------------------

def test_classification_metrics_are_accuracy_and_f1_only():
    assert set(metrics_for("classification")) == {"accuracy", "f1"}


def test_regression_metric_is_rmse_only():
    assert set(metrics_for("regression")) == {"rmse"}


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="unknown task"):
        metrics_for("clustering")
    with pytest.raises(ValueError, match="unknown task"):
        scoring_for("clustering")


def test_f1_is_macro_averaged_matching_prepfx():
    """Comparability with the B1 results already reported."""
    from sklearn.metrics import f1_score
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    assert metrics_for("classification")["f1"](y_true, y_pred) == pytest.approx(
        f1_score(y_true, y_pred, average="macro")
    )


def test_summarise_gives_mean_and_sd_because_sq1_asks_about_consistency():
    out = summarise({"accuracy": [0.9, 0.8, 0.85]})
    assert out["accuracy_mean"] == pytest.approx(0.85, abs=1e-6)
    assert out["accuracy_sd"] > 0


def test_a_single_fold_has_zero_spread_not_nan():
    """One observation has no spread, rather than an unknown one."""
    assert summarise({"rmse": [3.0]})["rmse_sd"] == 0.0


# --- the primary protocol -------------------------------------------------

def test_kfold_runs_ten_stratified_folds_by_default(wine):
    X, y = wine
    result = evaluate(X=X, y=y, task="classification", levels=BASE,
                      model="logreg", seed=SEED, cv_k=10, dataset="wine")
    assert result.n_splits == 10
    assert result.cv_strategy == "stratified_kfold"
    assert len(result.per_fold["accuracy"]) == 10
    assert 0.0 <= result.summary["accuracy_mean"] <= 1.0
    assert result.summary["accuracy_sd"] >= 0


def test_regression_uses_unstratified_kfold_and_rmse(small_reg):
    X, y = small_reg
    result = evaluate(X=X, y=y, task="regression", levels=BASE,
                      model="logreg", seed=SEED, cv_k=5, dataset="synthetic")
    assert result.cv_strategy == "kfold"
    assert set(result.per_fold) == {"rmse"}
    assert result.summary["rmse_mean"] > 0


def test_identical_seeds_give_identical_folds(wine):
    """The fold hash is what proves two runs are comparable."""
    X, y = wine
    kwargs = dict(X=X, y=y, task="classification", levels=BASE, model="logreg",
                  cv_k=5, dataset="wine")
    first = evaluate(seed=SEED, **kwargs)
    second = evaluate(seed=SEED, **kwargs)
    assert first.fold_assignment_hash == second.fold_assignment_hash


def test_different_seeds_give_different_folds(wine):
    X, y = wine
    kwargs = dict(X=X, y=y, task="classification", levels=BASE, model="logreg",
                  cv_k=5, dataset="wine")
    first = evaluate(seed=SEED, **kwargs).fold_assignment_hash
    other = evaluate(seed=SEED + 1, **kwargs).fold_assignment_hash
    assert first != other


def test_fold_hash_is_order_independent_but_content_sensitive():
    assert fold_hash([[3, 1, 2]]) == fold_hash([[1, 2, 3]])
    assert fold_hash([[1, 2, 3]]) != fold_hash([[1, 2, 4]])


def test_fold_seed_is_deterministic_and_purpose_separated():
    assert fold_seed(671) == fold_seed(671)
    assert fold_seed(671, "folds") != fold_seed(671, "tuning")


def test_feature_selection_effect_is_measured_not_inferred(wine):
    X, y = wine
    result = evaluate(X=X, y=y, task="classification",
                      levels={**BASE, "selection": "mutual_info"},
                      model="logreg", seed=SEED, cv_k=5, dataset="wine")
    assert result.n_features_in == 13
    assert result.n_features_selected is not None
    assert result.n_features_selected < result.n_features_in


# --- row deletion: Chapter 1's imputation baseline ------------------------

def test_row_deletion_removes_incomplete_rows_before_folds():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    Xd, yd, report = apply_row_deletion(X, y, "drop_rows")
    assert len(Xd) == 3 and len(yd) == 3
    assert report["n_rows_before_deletion"] == 4
    assert report["rows_dropped"] == 1
    assert report["pct_dropped"] == 25.0


def test_other_imputation_levels_delete_nothing():
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})
    y = pd.Series([0, 1])
    Xd, _, report = apply_row_deletion(X, y, "median")
    assert len(Xd) == 2
    assert report["rows_dropped"] == 0


def test_deletion_that_empties_a_dataset_is_refused_not_scored():
    """Row deletion can discard nearly everything; that must be reported
    as unevaluable rather than scored on two rows."""
    X = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ProtocolError, match="row deletion left"):
        apply_row_deletion(X, y, "drop_rows")


def test_deletion_cost_reaches_the_result(wine):
    X, y = wine
    X = X.copy()
    X.iloc[0, 0] = np.nan
    result = evaluate(X=X, y=y, task="classification",
                      levels={**BASE, "imputation": "drop_rows"},
                      model="logreg", seed=SEED, cv_k=5, dataset="wine")
    assert result.deletion["n_rows_before_deletion"] == 178
    assert result.deletion["rows_dropped"] == 1


# --- nested CV for tuning -------------------------------------------------

def test_tuning_records_its_budget_and_chosen_params(wine):
    X, y = wine
    result = evaluate(X=X, y=y, task="classification",
                      levels={**BASE, "tuning": "grid"},
                      model="logreg", seed=SEED, cv_k=3, inner_k=2, dataset="wine")
    # logreg's grid has 4 candidates.
    assert result.tuning_budget == 4
    assert result.tuning_best_params is not None
    assert any("model__" in k for k in result.tuning_best_params)


def test_no_tuning_records_no_budget_rather_than_zero(wine):
    X, y = wine
    result = evaluate(X=X, y=y, task="classification", levels=BASE,
                      model="logreg", seed=SEED, cv_k=3, dataset="wine")
    # None distinguishes "not tuned" from "tuned with zero candidates".
    assert result.tuning_budget is None
    assert result.tuning_best_params is None


def test_tuning_still_produces_one_score_per_outer_fold(wine):
    """Nesting must not change the reported fold count."""
    X, y = wine
    result = evaluate(X=X, y=y, task="classification",
                      levels={**BASE, "tuning": "random"},
                      model="logreg", seed=SEED, cv_k=3, inner_k=2, dataset="wine")
    assert len(result.per_fold["accuracy"]) == 3


# --- the gated secondary protocol ----------------------------------------

def test_repeated_holdout_refuses_a_small_dataset_with_the_arithmetic():
    """iris-sized data must be refused, and the refusal must explain why."""
    with pytest.raises(ProtocolError) as excinfo:
        check_holdout_eligible(150, 0.15, "iris")
    message = str(excinfo.value)
    assert "iris" in message
    assert "22 rows" in message          # 15% of 150
    assert "4.55 percentage points" in message
    assert str(HOLDOUT_MIN_ROWS) in message


def test_repeated_holdout_accepts_a_dataset_at_the_gate():
    check_holdout_eligible(HOLDOUT_MIN_ROWS, 0.15, "abalone")  # must not raise


@pytest.mark.parametrize("n_rows", [4177, 22784, 45211, 48842])
def test_the_four_eligible_b2_datasets_pass_the_gate(n_rows):
    """abalone, house_16H, bank-marketing, adult."""
    check_holdout_eligible(n_rows, 0.15, "eligible")


@pytest.mark.parametrize("n_rows", [150, 178, 303, 569, 690, 1000])
def test_the_small_b2_and_prepfx_datasets_are_refused(n_rows):
    with pytest.raises(ProtocolError):
        check_holdout_eligible(n_rows, 0.15, "small")


def test_evaluate_refuses_repeated_holdout_on_a_small_dataset(wine):
    X, y = wine  # 178 rows
    with pytest.raises(ProtocolError, match="repeated holdout refused"):
        evaluate(X=X, y=y, task="classification", levels=BASE, model="logreg",
                 seed=SEED, protocol="repeated_holdout", holdout_repeats=5,
                 dataset="wine")


def test_repeated_holdout_runs_on_a_large_enough_dataset():
    rng = np.random.default_rng(SEED)
    n = HOLDOUT_MIN_ROWS + 100
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] > 0).astype(int))
    result = evaluate(X=X, y=y, task="classification", levels=BASE, model="logreg",
                      seed=SEED, protocol="repeated_holdout", holdout_repeats=3,
                      dataset="synthetic_large")
    assert result.cv_strategy == "stratified_shuffle_split"
    assert result.n_splits == 3
    assert len(result.per_fold["accuracy"]) == 3


# --- splitters and refusals ----------------------------------------------

def test_classification_splitters_are_always_stratified():
    assert "Stratified" in type(make_splitter("classification", "kfold", n_splits=5, seed=1)).__name__
    assert "Stratified" in type(
        make_splitter("classification", "repeated_holdout", n_splits=5, seed=1, test_size=0.15)
    ).__name__


def test_regression_splitters_are_not_stratified():
    assert "Stratified" not in type(make_splitter("regression", "kfold", n_splits=5, seed=1)).__name__


def test_unknown_protocol_is_rejected():
    with pytest.raises(ProtocolError, match="unknown protocol"):
        make_splitter("classification", "bootstrap", n_splits=5, seed=1)


def test_more_folds_than_rows_is_refused(small_reg):
    X, y = small_reg
    with pytest.raises(ProtocolError, match="fewer than"):
        evaluate(X=X.head(3), y=y.head(3), task="regression", levels=BASE,
                 model="logreg", seed=SEED, cv_k=10, dataset="tiny")


def test_chi2_substitution_on_regression_is_recorded(small_reg):
    X, y = small_reg
    result = evaluate(X=X.abs(), y=y, task="regression",
                      levels={**BASE, "selection": "chi2", "scaling": "minmax"},
                      model="logreg", seed=SEED, cv_k=3, dataset="synthetic")
    assert result.substitution_note and "f_regression" in result.substitution_note


def test_model_substitution_on_regression_is_recorded(small_reg):
    """A regression row says "logreg" but LinearRegression actually ran.

    Without the note a reader of lab.db cannot tell, and would read that
    model's necessarily-null scaling contrast as a finding about the data
    rather than a property of the estimator.
    """
    X, y = small_reg
    result = evaluate(X=X, y=y, task="regression", levels=BASE,
                      model="logreg", seed=SEED, cv_k=3, dataset="synthetic")
    assert result.substitution_note
    assert "LinearRegression" in result.substitution_note


def test_a_classification_run_records_no_substitution(wine):
    """Non-vacuous: the note must be absent when nothing was swapped."""
    X, y = wine
    result = evaluate(X=X, y=y, task="classification", levels=BASE,
                      model="logreg", seed=SEED, cv_k=3, dataset="wine")
    assert result.substitution_note is None


def test_both_substitutions_are_recorded_together(small_reg):
    """Selection and model can substitute on one run; keep both notes."""
    X, y = small_reg
    result = evaluate(X=X.abs(), y=y, task="regression",
                      levels={**BASE, "selection": "chi2", "scaling": "minmax"},
                      model="logreg", seed=SEED, cv_k=3, dataset="synthetic")
    assert "f_regression" in result.substitution_note
    assert "LinearRegression" in result.substitution_note


def test_result_is_a_dataclass_with_the_run_record_fields():
    fields = set(EvaluationResult.__dataclass_fields__)
    assert {"per_fold", "summary", "fold_assignment_hash", "n_splits",
            "cv_strategy", "n_features_in", "n_features_selected",
            "tuning_budget", "tuning_best_params", "substitution_note",
            "deletion"} <= fields


# --- tuning where the named model has nothing to tune ---------------------


@pytest.mark.parametrize("level", ["grid", "random"])
def test_tuning_linear_regression_is_skipped_with_a_reason(small_reg, level):
    """The grid is keyed by model name, but "logreg" on a regression
    target is LinearRegression, which has no C. The first tuning sweep
    failed 12 runs on this. It must be a ProtocolError (recorded as
    skipped, with the reason), not a crash and not a silent model swap.
    """
    X, y = small_reg
    with pytest.raises(ProtocolError, match="not applicable") as excinfo:
        evaluate(X=X, y=y, task="regression", levels={**BASE, "tuning": level},
                 model="logreg", seed=SEED, cv_k=3, dataset="synthetic")
    assert "LinearRegression" in str(excinfo.value)


def test_the_untuned_linear_regression_baseline_still_runs(small_reg):
    """Non-vacuous: only the tuned levels are refused."""
    X, y = small_reg
    result = evaluate(X=X, y=y, task="regression", levels=BASE,
                      model="logreg", seed=SEED, cv_k=3, dataset="synthetic")
    assert result.summary


def test_tuning_still_applies_to_a_regression_model_that_has_a_grid(small_reg):
    """The refusal must not spread to models whose grid is valid."""
    X, y = small_reg
    result = evaluate(X=X, y=y, task="regression", levels={**BASE, "tuning": "grid"},
                      model="svm", seed=SEED, cv_k=3, inner_k=2, dataset="synthetic")
    assert result.tuning_best_params


def test_a_class_too_small_to_stratify_is_skipped_with_a_reason():
    """Enough rows overall, but one class below k. sklearn raises a bare
    ValueError inside the splitter; the first injected sweep recorded
    that as 4 failures on sonar. It is the same situation as too few
    rows and must be a ProtocolError, so the run is skipped."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 27 + [1] * 3)
    with pytest.raises(ProtocolError, match="class with only 3"):
        evaluate(X=X, y=y, task="classification", levels=BASE,
                 model="logreg", seed=SEED, cv_k=10, dataset="lopsided")


def test_a_small_class_that_still_fits_the_folds_is_evaluated():
    """Non-vacuous: the check must not refuse a dataset that can be split."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 27 + [1] * 3)
    result = evaluate(X=X, y=y, task="classification", levels=BASE,
                      model="logreg", seed=SEED, cv_k=3, dataset="lopsided")
    assert result.summary
