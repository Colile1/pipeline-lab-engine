# Confirms the composed pipeline fits and predicts for every component
# level on both tasks and on mixed numeric/categorical data, that k is
# resolved AFTER one-hot expansion, and that the leakage proofs still
# hold for the composed pipeline rather than only hand-built ones.
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from piplab.components import tuning
from piplab.config import MODELS, TECHNIQUES
from piplab.pipelines.compose import (
    ColumnRoles,
    build_pipeline,
    column_roles,
    resolve_levels,
    selected_feature_counts,
)
from piplab.pipelines.spies import FitSpy

SEED = 671
BASE = {"encoding": "onehot", "imputation": "median", "scaling": "standard", "selection": "none"}


@pytest.fixture
def numeric_clf():
    """A purely numeric classification frame."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    return X, y


@pytest.fixture
def mixed_clf():
    """Classification data with categorical columns, so one-hot expands width."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({
        "num1": rng.normal(size=80),
        "num2": rng.normal(size=80),
        "cat1": rng.choice(["a", "b", "c"], size=80),
        "cat2": rng.choice(["x", "y"], size=80),
    })
    y = pd.Series((X["num1"] > 0).astype(int))
    return X, y


@pytest.fixture
def numeric_reg():
    """A purely numeric regression frame."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(X["f0"] * 2 + rng.normal(scale=0.1, size=80))
    return X, y


# --- column roles ---------------------------------------------------------

def test_column_roles_splits_numeric_from_categorical(mixed_clf):
    X, _ = mixed_clf
    roles = column_roles(X)
    assert roles.numeric == ["num1", "num2"]
    assert roles.categorical == ["cat1", "cat2"]


def test_column_roles_on_all_numeric_has_no_categoricals(numeric_clf):
    X, _ = numeric_clf
    assert column_roles(X).categorical == []


# --- the pipeline fits and predicts for every level -----------------------

@pytest.mark.parametrize("level", TECHNIQUES["scaling"])
def test_every_scaling_level_composes_and_fits(level, numeric_clf):
    X, y = numeric_clf
    pipe = build_pipeline(levels={**BASE, "scaling": level}, model="logreg",
                          roles=column_roles(X), task="classification", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


@pytest.mark.parametrize("level", [l for l in TECHNIQUES["imputation"] if l != "drop_rows"])
def test_every_in_pipeline_imputation_level_composes_and_fits(level, numeric_clf):
    X, y = numeric_clf
    X = X.copy()
    X.iloc[0, 0] = np.nan  # something to impute
    pipe = build_pipeline(levels={**BASE, "imputation": level}, model="logreg",
                          roles=column_roles(X), task="classification", seed=SEED)
    pipe.fit(X, y)
    assert not np.isnan(pipe.predict(X)).any()


@pytest.mark.parametrize("level", TECHNIQUES["selection"])
def test_every_selection_level_composes_and_fits_for_classification(level, numeric_clf):
    X, y = numeric_clf
    if level == "chi2":
        X = X.abs()  # chi2 needs non-negative features
    pipe = build_pipeline(levels={**BASE, "selection": level, "scaling": "minmax"},
                          model="logreg", roles=column_roles(X),
                          task="classification", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


@pytest.mark.parametrize("level", TECHNIQUES["selection"])
def test_every_selection_level_composes_and_fits_for_regression(level, numeric_reg):
    X, y = numeric_reg
    if level == "chi2":
        X = X.abs()
    pipe = build_pipeline(levels={**BASE, "selection": level, "scaling": "minmax"},
                          model="logreg", roles=column_roles(X),
                          task="regression", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


@pytest.mark.parametrize("model", MODELS)
def test_every_model_composes_and_fits(model, numeric_clf):
    X, y = numeric_clf
    pipe = build_pipeline(levels=BASE, model=model, roles=column_roles(X),
                          task="classification", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


def test_mixed_numeric_and_categorical_data_composes_and_fits(mixed_clf):
    X, y = mixed_clf
    pipe = build_pipeline(levels=BASE, model="random_forest", roles=column_roles(X),
                          task="classification", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


# --- k is resolved AFTER one-hot expansion (the real gotcha) --------------

def test_selector_k_is_resolved_from_the_expanded_width_not_the_raw_frame(mixed_clf):
    X, y = mixed_clf
    # Raw frame: 4 columns. After one-hot (cat1->3, cat2->2): 2 + 3 + 2 = 7.
    assert X.shape[1] == 4
    pipe = build_pipeline(levels={**BASE, "selection": "mutual_info"}, model="logreg",
                          roles=column_roles(X), task="classification", seed=SEED)
    pipe.fit(X, y)
    counts = selected_feature_counts(pipe)
    assert counts["n_features_in"] == 7, "k must be resolved post-expansion"
    # DEFAULT_K_FRACTION is 0.5, so round(7*0.5) = 4 (not 2 from the raw frame).
    assert counts["n_features_selected"] == 4


def test_selection_none_keeps_every_feature(mixed_clf):
    X, y = mixed_clf
    pipe = build_pipeline(levels=BASE, model="logreg", roles=column_roles(X),
                          task="classification", seed=SEED)
    pipe.fit(X, y)
    counts = selected_feature_counts(pipe)
    assert counts["n_features_in"] == counts["n_features_selected"] == 7


def test_lasso_selection_count_is_data_dependent_and_recorded(numeric_clf):
    X, y = numeric_clf
    pipe = build_pipeline(levels={**BASE, "selection": "lasso"}, model="logreg",
                          roles=column_roles(X), task="classification", seed=SEED)
    pipe.fit(X, y)
    counts = selected_feature_counts(pipe)
    # LASSO drops zero-coefficient features, so the count is <= input and
    # must be recorded rather than assumed.
    assert counts["n_features_selected"] is not None
    assert 1 <= counts["n_features_selected"] <= counts["n_features_in"]


# --- OFAT level resolution -------------------------------------------------

def test_resolve_levels_merges_baseline_with_the_varied_level():
    baseline = {"encoding": "onehot", "imputation": "drop_rows", "scaling": "none", "tuning": "none"}
    levels = resolve_levels(baseline, vary="selection", level="rfe")
    assert levels["selection"] == "rfe"
    assert levels["scaling"] == "none"  # untouched baseline


def test_resolve_levels_refuses_a_baseline_that_pins_the_varied_component():
    with pytest.raises(ValueError, match="must not contain"):
        resolve_levels({"scaling": "robust"}, vary="scaling", level="minmax")


def test_build_pipeline_requires_every_component_level(numeric_clf):
    X, _ = numeric_clf
    with pytest.raises(ValueError, match="missing the 'selection'"):
        build_pipeline(levels={"encoding": "onehot", "imputation": "mean", "scaling": "none"},
                       model="logreg", roles=column_roles(X), task="classification", seed=SEED)


def test_build_pipeline_refuses_empty_column_roles():
    with pytest.raises(ValueError, match="no numeric or categorical"):
        build_pipeline(levels=BASE, model="logreg", roles=ColumnRoles(),
                       task="classification", seed=SEED)


# --- leakage, now for the COMPOSED pipeline -------------------------------

@pytest.mark.parametrize("sel_level", ["chi2", "mutual_info", "rfe", "lasso"])
def test_composed_pipeline_never_fits_any_stage_on_a_test_fold(sel_level, numeric_clf):
    """The step-2 proof, extended to the real composed pipeline."""
    X, y = numeric_clf
    if sel_level == "chi2":
        X = X.abs()
    FitSpy.reset()
    pipe = build_pipeline(levels={**BASE, "selection": sel_level, "scaling": "minmax"},
                          model="logreg", roles=column_roles(X),
                          task="classification", seed=SEED, instrument=True)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_sets = []
    for train_idx, test_idx in splitter.split(X, y):
        clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        test_sets.append(frozenset(X.index[test_idx]))

    for spy_name in ("imputer", "scaler", "selector"):
        records = FitSpy.records(spy_name)
        assert records, f"{spy_name} should have been fitted once per fold"
        assert len(records) == len(test_sets), f"{spy_name} should fit once per fold"
        for record, held_out in zip(records, test_sets):
            # Row identity survives only while X is still a DataFrame. Once
            # an upstream transformer hands on a numpy array the index is
            # genuinely gone, so fall back to y's index, which a Pipeline
            # does preserve. Asserting on invented positional ids would be
            # a false failure, not a real leak.
            ids = record["row_ids"] if record["indexed"] else record["y_fingerprint"]
            assert ids is not None, f"{spy_name} fit had no recoverable row identity"
            overlap = ids & held_out
            assert not overlap, f"{spy_name} saw {len(overlap)} held-out rows — leakage"


def test_nested_tuning_over_the_composed_pipeline_stays_inside_the_outer_fold(numeric_clf):
    """Tuning wraps the whole composed pipeline, so selection is re-fitted
    inside the inner search rather than fixed before it."""
    X, y = numeric_clf
    FitSpy.reset()
    pipe = build_pipeline(levels={**BASE, "selection": "mutual_info"}, model="logreg",
                          roles=column_roles(X), task="classification",
                          seed=SEED, instrument=True)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = tuning.make_search("grid", pipe, "logreg", "accuracy", inner_cv, SEED)

    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    test_sets, groups = [], []
    for train_idx, test_idx in outer.split(X, y):
        FitSpy.reset()
        clone(search).fit(X.iloc[train_idx], y.iloc[train_idx])
        groups.append(list(FitSpy.records("selector")))
        test_sets.append(frozenset(X.index[test_idx]))

    for i, (records, held_out) in enumerate(zip(groups, test_sets)):
        assert len(records) > 1, "the inner search should refit the selector many times"
        for record in records:
            ids = record["row_ids"] if record["indexed"] else record["y_fingerprint"]
            assert ids is not None
            overlap = ids & held_out
            assert not overlap, (
                f"inner search saw {len(overlap)} rows of outer fold {i}'s test set — leakage"
            )


def test_instrument_flag_is_off_by_default(numeric_clf):
    """Production runs must not carry spies: they add bookkeeping per fit."""
    X, y = numeric_clf
    FitSpy.reset()
    build_pipeline(levels=BASE, model="logreg", roles=column_roles(X),
                   task="classification", seed=SEED).fit(X, y)
    assert FitSpy.records("selector") == []


def test_lasso_never_leaves_the_model_with_zero_features():
    """Regression guard for a real bug found in step 3.

    L1 at a strong penalty relative to the signal can zero EVERY
    coefficient. Before the _MinFeatures wrapper that raised
    "Found array with 0 feature(s)" deep inside the pipeline and lost the
    whole run. The technique's behaviour is preserved (the kept count
    stays data-dependent and small); it simply cannot reach zero.
    """
    rng = np.random.default_rng(SEED)
    # Pure noise against a noise target: nothing for L1 to hold on to.
    X = pd.DataFrame(rng.normal(size=(80, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.normal(size=80))

    pipe = build_pipeline(levels={**BASE, "selection": "lasso"}, model="logreg",
                          roles=column_roles(X), task="regression", seed=SEED)
    pipe.fit(X, y)  # must not raise
    counts = selected_feature_counts(pipe)
    assert counts["n_features_selected"] >= 1
    assert len(pipe.predict(X)) == len(y)
