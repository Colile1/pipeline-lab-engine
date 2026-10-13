# compose.py: build the leakage-safe pipeline for one configuration.
#
# Stage order (Chapter 1 section 1.2.1):
#   ColumnTransformer( numeric:     impute -> scale
#                      categorical: impute -> encode )
#     -> select
#     -> model
#
# Every transform sits INSIDE the Pipeline, so sklearn fits it on the
# training fold only and the leakage proofs in tests/test_leakage.py
# hold for the composed pipeline, not just hand-built ones.
#
# Two things are deliberately NOT composed here:
#   * drop_rows   - a transformer may not change sample count, so the
#                   evaluator applies it before folds are built.
#   * tuning      - wraps the WHOLE composed pipeline, so selection is
#                   re-fitted inside the inner search rather than fixed
#                   before it. See evaluation/protocols.py.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from piplab.components import encoding, imputation, models, scaling, selection
from piplab.pipelines.spies import FitSpy


@dataclass
class ColumnRoles:
    """Which feature columns are numeric and which are categorical."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)


def column_roles(X: pd.DataFrame) -> ColumnRoles:
    """Infer numeric vs categorical feature columns.

    Inputs: X - the feature frame.
    Output: a ColumnRoles. Anything not a numpy number is treated as
    categorical, matching how both existing engines split columns.
    """
    numeric = list(X.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnRoles(numeric=numeric, categorical=categorical)


class LateResolvedSelector(BaseEstimator, TransformerMixin):
    """A selector whose k is resolved from the data it is fitted on.

    The k-based selectors (chi2, mutual_info, rfe) need a feature count,
    but the count is only known after the ColumnTransformer has expanded
    categoricals into one-hot columns -- which happens inside the
    pipeline, per fold. Guessing from the raw frame would be wrong
    whenever a dataset has categorical features.

    So this wrapper builds the real selector at fit time, from the width
    of the matrix it actually receives, and records the resolved counts
    for the run record (n_features_in_, n_features_selected_).
    """

    def __init__(self, level: str, task: str, seed: int):
        self.level = level
        self.task = task
        self.seed = seed

    def fit(self, X, y=None):
        """Resolve k from X's real width, then fit the underlying selector."""
        n_features = X.shape[1]
        self.inner_ = selection.make_selector(
            self.level, task=self.task, seed=self.seed, n_features=n_features
        )
        self.n_features_in_ = int(n_features)
        if self.inner_ == "passthrough":
            self.n_features_selected_ = int(n_features)
            return self
        self.inner_.fit(X, y)
        self.n_features_selected_ = int(self.inner_.transform(X[:1]).shape[1])
        return self

    def transform(self, X):
        """Apply the fitted selector (identity for the `none` baseline)."""
        if self.inner_ == "passthrough":
            return X
        return self.inner_.transform(X)


def build_pipeline(
    *,
    levels: dict[str, str],
    model: str,
    roles: ColumnRoles,
    task: str,
    seed: int,
    instrument: bool = False,
) -> Pipeline:
    """Compose the pipeline for one fully-resolved configuration.

    Inputs: levels - one technique per component, i.e. the config's
    baseline merged with the varied component's current level, keyed
    encoding/imputation/scaling/selection (tuning is applied by the
    evaluator, not here); model - a model key; roles - the column split;
    task - "classification" or "regression"; seed - the global seed;
    instrument - wrap the imputer, scaler and selector in FitSpy so the
    leakage proofs can inspect what each was fitted on.
    Output: an unfitted Pipeline.
    Raises: ValueError if a level or the column split is unusable.
    """
    for component in ("encoding", "imputation", "scaling", "selection"):
        if component not in levels:
            raise ValueError(f"levels is missing the {component!r} component")

    numeric_imputer = imputation.make_imputer(levels["imputation"], seed)
    numeric_scaler = scaling.make_scaler(levels["scaling"], seed)
    if instrument:
        if numeric_imputer != "passthrough":
            numeric_imputer = FitSpy(numeric_imputer, "imputer")
        if numeric_scaler != "passthrough":
            numeric_scaler = FitSpy(numeric_scaler, "scaler")

    encoder = encoding.make_encoder(levels["encoding"], seed, task)
    if instrument:
        encoder = FitSpy(encoder, "encoder")

    transformers = []
    if roles.numeric:
        transformers.append((
            "numeric",
            Pipeline([("impute", numeric_imputer), ("scale", numeric_scaler)]),
            roles.numeric,
        ))
    if roles.categorical:
        # Categorical imputation is held at most_frequent throughout:
        # Chapter 1 varies imputation on NUMERIC features, and varying
        # both at once would confound the attribution.
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", encoder),
            ]),
            roles.categorical,
        ))
    if not transformers:
        raise ValueError("ColumnRoles has no numeric or categorical columns")

    pre = ColumnTransformer(transformers, remainder="drop")

    selector = LateResolvedSelector(levels["selection"], task=task, seed=seed)
    if instrument:
        selector = FitSpy(selector, "selector")

    return Pipeline([
        ("pre", pre),
        ("select", selector),
        ("model", models.make_model(model, task, seed)),
    ])


def resolve_levels(baseline: dict[str, str], vary: str, level: str) -> dict[str, str]:
    """Merge a config's baseline with the varied component's current level.

    This is what makes a run OFAT: exactly one component takes `level`
    and every other sits at its baseline.
    Inputs: baseline - the config's held-fixed components; vary - the
    component being swept; level - its current technique.
    Output: a complete level mapping for build_pipeline.
    Raises: ValueError if the baseline already pins the varied component.
    """
    if vary in baseline:
        raise ValueError(
            f"baseline must not contain the varied component {vary!r}; "
            "its levels belong in the sweep"
        )
    return {**baseline, vary: level}


def selected_feature_counts(fitted: Pipeline) -> dict:
    """Read back how many features the selector kept, for the run record.

    Inputs: fitted - a Pipeline already fitted on a training fold.
    Output: {"n_features_in": int|None, "n_features_selected": int|None};
    None when the pipeline has no resolvable selector (never guessed).
    """
    step = fitted.named_steps.get("select")
    inner = getattr(step, "inner", step)  # unwrap FitSpy if instrumented
    return {
        "n_features_in": getattr(inner, "n_features_in_", None),
        "n_features_selected": getattr(inner, "n_features_selected_", None),
    }
