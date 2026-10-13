# imputation.py: the missing-value component. Chapter 1's baseline is
# ROW DELETION, which neither B1 nor B2 implements.
#
# Row deletion cannot be an ordinary pipeline step: a transformer may not
# change the number of samples (sklearn drops y alignment), so dropping
# rows inside a Pipeline would desynchronise X and y. It is therefore
# applied by the evaluator BEFORE the fold is built -- see
# `rows_to_keep` and evaluation/protocols.py. The other three levels are
# ordinary in-pipeline transformers fitted per fold.
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

# Marks the level that the evaluator, not the pipeline, has to handle.
DROP_ROWS = "drop_rows"


def _mean(seed: int):
    """Return an unfitted mean imputer."""
    return SimpleImputer(strategy="mean")


def _median(seed: int):
    """Return an unfitted median imputer."""
    return SimpleImputer(strategy="median")


def _knn(seed: int):
    """Return an unfitted KNN imputer (k=5, sklearn's default).

    KNN imputation is fitted on the training fold only, so a test row's
    neighbours are drawn from training data -- never from its own fold.
    """
    return KNNImputer(n_neighbors=5)


def _drop_rows(seed: int):
    """Return the in-pipeline step for the row-deletion baseline.

    Rows with missing values are removed by the evaluator before folds
    are built, so by the time the pipeline runs there is nothing left to
    impute. This returns a passthrough rather than silently imputing,
    which would misrepresent the baseline.
    """
    return "passthrough"


IMPUTERS = {
    DROP_ROWS: _drop_rows,
    "mean": _mean,
    "median": _median,
    "knn": _knn,
}


def make_imputer(level: str, seed: int):
    """Build the numeric imputer for one technique level.

    Inputs: level - a key of IMPUTERS; seed - the global seed.
    Output: an unfitted transformer, or "passthrough" for drop_rows.
    Raises: ValueError naming the valid levels.
    """
    if level not in IMPUTERS:
        raise ValueError(f"unknown imputation level {level!r}; valid: {sorted(IMPUTERS)}")
    return IMPUTERS[level](seed)


def requires_row_deletion(level: str) -> bool:
    """Say whether this level needs rows dropped before folds are built.

    Inputs: level - an imputation level.
    Output: True only for the drop_rows baseline.
    """
    return level == DROP_ROWS


def rows_to_keep(X: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask of complete rows, for the drop_rows baseline.

    Applied to the whole dataset before cross-validation, because
    deleting rows is a property of the dataset under this baseline, not
    a transformation learned from a training fold. Doing it per fold
    would change the test set between configurations and make scores
    incomparable.
    Inputs: X - the feature frame.
    Output: a boolean mask, True where the row has no missing value.
    """
    return X.notna().all(axis=1).to_numpy()


def deletion_report(X: pd.DataFrame) -> dict:
    """Describe what row deletion would remove, for the run record.

    Row deletion can discard most of a dataset, which is itself a
    finding, so the cost is recorded rather than applied silently.
    Inputs: X - the feature frame.
    Output: dict with rows_before, rows_after, rows_dropped, pct_dropped.
    """
    mask = rows_to_keep(X)
    before = int(len(X))
    after = int(mask.sum())
    return {
        "rows_before": before,
        "rows_after": after,
        "rows_dropped": before - after,
        "pct_dropped": round(100.0 * (before - after) / before, 4) if before else 0.0,
    }
