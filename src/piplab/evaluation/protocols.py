# protocols.py: how one configuration is actually scored.
#
# Primary: stratified k-fold at k=10 (methodology section 6.3).
# Tuning:  nested CV -- the inner loop selects hyperparameters inside
#          each outer training fold, the outer fold scores the winner on
#          rows the inner search never saw. Without nesting, tuning on
#          the reported folds biases the score upward even though no
#          single fold leaked.
# Secondary: repeated holdout (30 x 70/15/15), gated to datasets with
#          >= 4000 rows. It REFUSES a smaller dataset rather than
#          producing a misleading number: 15% of iris is 22 rows, where
#          one misclassification moves accuracy 4.55 points while the
#          real effect to detect is ~7.3 points (95% CI 2.5-12.1).
#
# Row deletion (Chapter 1's imputation baseline) is applied HERE, not in
# the pipeline: a transformer may not change sample count. It is applied
# ONCE to the whole dataset before folds are built, because deleting per
# fold would change the test set between configurations and make scores
# incomparable.
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit

from piplab.components import imputation, models, selection, tuning
from piplab.config import HOLDOUT_MIN_ROWS
from piplab.evaluation.metrics import metrics_for, scoring_for, summarise
from piplab.pipelines.compose import (
    build_pipeline,
    column_roles,
    selected_feature_counts,
)


class ProtocolError(ValueError):
    """Raised when a protocol cannot honestly be applied to a dataset."""


@dataclass
class EvaluationResult:
    """Everything one evaluated configuration produced."""

    per_fold: dict[str, list[float]] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    fold_assignment_hash: str = ""
    n_splits: int = 0
    cv_strategy: str = ""
    n_features_in: int | None = None
    n_features_selected: int | None = None
    tuning_budget: int | None = None
    tuning_best_params: dict | None = None
    substitution_note: str | None = None
    deletion: dict = field(default_factory=dict)


def fold_seed(seed: int, purpose: str = "folds") -> int:
    """Derive a stable sub-seed, so fold assignment is reproducible.

    Inputs: seed - the config's global seed; purpose - a label, so fold
    assignment and tuning draws do not share a stream.
    Output: a deterministic integer seed.
    """
    digest = hashlib.sha256(f"{seed}:{purpose}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def fold_hash(test_index_lists: list[list[int]]) -> str:
    """Fingerprint a fold assignment, so two runs can be proven identical.

    Inputs: test_index_lists - the held-out positions per split.
    Output: a short stable hash; identical seeds give identical hashes.
    """
    payload = "|".join(",".join(map(str, sorted(idx))) for idx in test_index_lists)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_splitter(task: str, protocol: str, *, n_splits: int, seed: int,
                  test_size: float | None = None):
    """Build the splitter for one protocol.

    Inputs: task - "classification" or "regression"; protocol - "kfold"
    or "repeated_holdout"; n_splits - folds or repeats; seed - the
    config's seed; test_size - required for repeated_holdout.
    Output: an sklearn splitter. Classification is always stratified, so
    a fold cannot miss a class.
    Raises: ProtocolError for an unknown protocol.
    """
    state = fold_seed(seed)
    if protocol == "kfold":
        if task == "classification":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)
        return KFold(n_splits=n_splits, shuffle=True, random_state=state)
    if protocol == "repeated_holdout":
        if task == "classification":
            return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=state)
        return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=state)
    raise ProtocolError(f"unknown protocol {protocol!r}; valid: kfold, repeated_holdout")


def check_holdout_eligible(n_rows: int, test_size: float, dataset: str = "") -> None:
    """Refuse repeated holdout where a test set would be too small to mean anything.

    Inputs: n_rows - usable rows; test_size - the test fraction;
    dataset - name, for the message.
    Output: None when eligible.
    Raises: ProtocolError explaining the arithmetic, so the refusal is
    auditable rather than a bare rule.
    """
    if n_rows >= HOLDOUT_MIN_ROWS:
        return
    test_rows = max(1, int(round(n_rows * test_size)))
    resolution = 100.0 / test_rows
    raise ProtocolError(
        f"repeated holdout refused for {dataset or 'this dataset'}: {n_rows} rows gives a "
        f"{test_size:.0%} test set of {test_rows} rows, where one misclassification moves "
        f"accuracy {resolution:.2f} percentage points -- coarser than the effects this study "
        f"detects (~7.3 points, 95% CI 2.5-12.1). The gate is {HOLDOUT_MIN_ROWS} rows; use "
        "k-fold, which tests every row exactly once."
    )


def apply_row_deletion(X: pd.DataFrame, y: pd.Series, level: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Apply Chapter 1's row-deletion imputation baseline, if that is the level.

    Done once on the whole dataset before folds are built: deleting per
    fold would change the test set between configurations and make the
    comparison meaningless. The cost is reported rather than hidden,
    because row deletion can discard most of a dataset and that is itself
    a finding.
    Inputs: X, y - the full dataset; level - the imputation level.
    Output: (X, y, report) with rows removed when level is drop_rows.
    Raises: ProtocolError if deletion would leave too little data to
    cross-validate.
    """
    report = {
        "n_rows_before_deletion": int(len(X)),
        "rows_dropped": 0,
        "pct_dropped": 0.0,
    }
    if not imputation.requires_row_deletion(level):
        return X, y, report

    detail = imputation.deletion_report(X)
    mask = imputation.rows_to_keep(X)
    kept = int(mask.sum())
    if kept < 2:
        raise ProtocolError(
            f"row deletion left {kept} complete row(s) of {len(X)}: this dataset cannot be "
            "evaluated under the drop_rows baseline. Recorded as skipped rather than scored."
        )
    report.update({
        "rows_dropped": detail["rows_dropped"],
        "pct_dropped": detail["pct_dropped"],
    })
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True), report


def evaluate(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    levels: dict[str, str],
    model: str,
    seed: int,
    protocol: str = "kfold",
    cv_k: int = 10,
    inner_k: int = 3,
    holdout_repeats: int = 30,
    test_size: float = 0.15,
    dataset: str = "",
    k_fraction: float = selection.DEFAULT_K_FRACTION,
) -> EvaluationResult:
    """Score one fully-resolved configuration.

    Inputs: X, y - the dataset; task; levels - one technique per
    component; model - a model key; seed; protocol - "kfold" or
    "repeated_holdout"; cv_k/inner_k/holdout_repeats/test_size - protocol
    parameters; dataset - name, for error messages; k_fraction - the
    share of features the k-based selectors keep, recorded per run.
    Output: an EvaluationResult with per-fold scores, their mean and sd,
    the fold hash, and the feature/tuning details the run record needs.
    Raises: ProtocolError when a protocol cannot honestly be applied.
    """
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X, y, deletion = apply_row_deletion(X, y, levels["imputation"])

    n_splits = holdout_repeats if protocol == "repeated_holdout" else cv_k
    if protocol == "repeated_holdout":
        check_holdout_eligible(len(X), test_size, dataset)
    elif len(X) < n_splits:
        raise ProtocolError(
            f"{dataset or 'dataset'} has {len(X)} rows, fewer than the {n_splits} folds "
            "requested; cross-validation cannot be performed"
        )
    elif task == "classification":
        # Enough rows overall is not enough: stratified k-fold needs every
        # class to have at least k members. Row deletion on a wide dataset
        # can leave the row count above k while one class falls below it,
        # and sklearn then raises deep inside the splitter. Found
        # 2026-10-13 on sonar at 5% injected MCAR, where drop_rows left
        # too few of one class. That is the same situation as too few
        # rows, so it gets the same treatment: skipped, with the reason.
        #
        # The surviving-class count is checked FIRST and separately.
        # value_counts() only counts classes that are still present, so
        # when deletion removes a class outright it does not appear and
        # min() reports the smallest SURVIVING class, which can sit well
        # above n_splits. The guard then passes and sklearn raises three
        # different single-class errors further in. Found 2026-10-19 on
        # climate_sim at 15% injected MAR, where the 46-member minority
        # class (of 540 rows, 8.5%) was deleted entirely: min() returned
        # 70 against n_splits of 10 and the check did nothing. A rare
        # class and an absent one are different failures and need
        # different tests.
        present = int(y.nunique())
        if present < 2:
            raise ProtocolError(
                f"{dataset or 'dataset'} has only {present} class among its "
                f"{len(X)} remaining rows: row deletion removed every member "
                "of the other class(es), so there is nothing to classify. "
                "Recorded as skipped rather than scored."
            )
        smallest = int(y.value_counts().min())
        if smallest < n_splits:
            raise ProtocolError(
                f"{dataset or 'dataset'} has a class with only {smallest} "
                f"member(s) among its {len(X)} rows, fewer than the {n_splits} "
                "folds requested; stratified cross-validation cannot be performed"
            )

    roles = column_roles(X)
    metrics = metrics_for(task)
    splitter = make_splitter(task, protocol, n_splits=n_splits, seed=seed, test_size=test_size)
    base = build_pipeline(levels=levels, model=model, roles=roles,
                          task=task, seed=seed, k_fraction=k_fraction)

    tuning_level = levels.get("tuning", "none")
    inner_cv = None
    if tuning_level != "none":
        # The grid is keyed by model NAME, but the estimator behind a name
        # depends on the task: "logreg" is LinearRegression on a regression
        # target, which has no C to search. Found 2026-10-13 when the
        # tuning sweep failed 12 runs with "Invalid parameter 'C'".
        # Such a run is SKIPPED with the reason, not failed and not
        # quietly given a different model: swapping in Ridge would compare
        # a tuned model against a different untuned one.
        available = base.get_params()
        missing = [p for p in tuning.searched_parameters(tuning_level, model)
                   if p not in available]
        if missing:
            estimator_name = type(base.named_steps["model"]).__name__
            raise ProtocolError(
                f"tuning is not applicable to {model} on a {task} target: "
                f"the estimator is {estimator_name}, which has no "
                f"{', '.join(m.split('__')[-1] for m in missing)} to search. "
                f"The untuned baseline for this model still runs."
            )
        inner_cv = make_splitter(task, "kfold", n_splits=inner_k, seed=seed + 1)

    per_fold: dict[str, list[float]] = {name: [] for name in metrics}
    test_indices: list[list[int]] = []
    feature_counts: dict = {}
    best_params: dict | None = None

    for train_idx, test_idx in splitter.split(X, y):
        estimator = clone(base)
        if tuning_level != "none":
            estimator = tuning.make_search(
                tuning_level, estimator, model, scoring_for(task), inner_cv, seed
            )
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = estimator.predict(X.iloc[test_idx])
        truth = y.iloc[test_idx]
        for name, fn in metrics.items():
            per_fold[name].append(fn(truth, predictions))
        test_indices.append([int(i) for i in test_idx])

        fitted = getattr(estimator, "best_estimator_", estimator)
        feature_counts = selected_feature_counts(fitted)
        if hasattr(estimator, "best_params_"):
            best_params = dict(estimator.best_params_)

    return EvaluationResult(
        per_fold=per_fold,
        summary=summarise(per_fold),
        fold_assignment_hash=fold_hash(test_indices),
        n_splits=n_splits,
        cv_strategy=_strategy_name(task, protocol),
        n_features_in=feature_counts.get("n_features_in"),
        n_features_selected=feature_counts.get("n_features_selected"),
        tuning_budget=_tuning_budget(tuning_level, model),
        tuning_best_params=best_params,
        substitution_note=_substitution_note(levels, model, task, X),
        deletion=deletion,
    )


def _substitution_note(levels: dict, model: str, task: str, X=None) -> str | None:
    """Join every substitution this run made into one recorded string.

    Three substitutions are possible. Two depend only on the task: the
    selection filter (chi2 -> f_regression) and the model itself (logreg
    -> LinearRegression). The third depends on the DATA: chi2 falls back
    to f_classif when any feature is negative, which is why X is passed
    through. Recording only some would leave the others silent.

    Inputs: levels - the resolved component levels; model - the model
    key; task - the task type; X - the feature matrix.
    Output: the joined explanation, or None when nothing was substituted.
    """
    notes = [
        selection.substitution_note(levels.get("selection", "none"), task, X),
        models.substitution_note(model, task),
    ]
    present = [note for note in notes if note]
    return "; ".join(present) if present else None


def _strategy_name(task: str, protocol: str) -> str:
    """Name the splitter actually used, for the run record."""
    if protocol == "repeated_holdout":
        return "stratified_shuffle_split" if task == "classification" else "shuffle_split"
    return "stratified_kfold" if task == "classification" else "kfold"


def _tuning_budget(level: str, model: str) -> int | None:
    """Return how many hyperparameter candidates the search considered.

    Inputs: level - the tuning level; model - the model key.
    Output: the candidate count, or None when tuning is off, so the run
    record distinguishes "no tuning" from "tuned with zero candidates".
    """
    if level == "none":
        return None
    if level == "grid":
        candidates = 1
        for values in tuning.grid_for(model).values():
            candidates *= len(values)
        return candidates
    return tuning.DEFAULT_N_ITER
