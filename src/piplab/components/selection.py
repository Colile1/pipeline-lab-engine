# selection.py: the feature-selection component. Absent from both B1 and
# B2; new here. Chapter 1 baseline is "none".
#
# LEAKAGE: selection is one of the two stages that silently inflates
# scores if fitted on the whole dataset -- the selector would have seen
# the test fold's targets when choosing features. Every factory here
# returns an UNFITTED selector that the pipeline fits inside the
# training fold only. tests/test_leakage.py proves this with a fit-spy.
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class _MinFeatures(BaseEstimator, TransformerMixin):
    """Wrap a selector so it can never return fewer than `minimum` features.

    An embedded L1 selector can legitimately zero every coefficient when
    the penalty is strong relative to the signal. That is a real property
    of the technique, but it leaves the downstream model with a
    zero-column matrix and raises deep in the pipeline, failing the whole
    run. This keeps the strongest-scoring features instead, so the sweep
    completes and the (small) kept count is reported honestly rather than
    the run being lost.
    Inputs (constructor): inner - the selector; minimum - the floor.
    """

    def __init__(self, inner, minimum: int = 1):
        self.inner = inner
        self.minimum = minimum

    def fit(self, X, y=None):
        """Fit the inner selector, then compute a fallback mask if it kept nothing."""
        self.inner.fit(X, y)
        support = self.inner.get_support()
        self.fallback_ = None
        if support.sum() < self.minimum:
            scores = np.abs(np.ravel(self.inner.estimator_.coef_))
            keep = np.argsort(scores)[::-1][: max(self.minimum, 1)]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[keep] = True
            self.fallback_ = mask
        return self

    def transform(self, X):
        """Apply the inner selector, or the fallback mask when it kept nothing."""
        if self.fallback_ is not None:
            return np.asarray(X)[:, self.fallback_]
        return self.inner.transform(X)

    def get_support(self):
        """Return the effective support mask, fallback included."""
        return self.fallback_ if self.fallback_ is not None else self.inner.get_support()

# Chapter 1 does not fix k, so a proportion of the input features is
# used and the resolved count is recorded per run (n_features_selected).
DEFAULT_K_FRACTION = 0.5


def resolve_k(n_features: int, fraction: float = DEFAULT_K_FRACTION) -> int:
    """Return how many features to keep for a k-based selector.

    Inputs: n_features - input feature count; fraction - share to keep.
    Output: at least 1, at most n_features, so a selector can never ask
    for more features than exist (sklearn raises) or for none.
    """
    return max(1, min(n_features, int(round(n_features * fraction))))


def _none(task: str, seed: int, n_features: int):
    """Return the no-selection baseline (sklearn's identity step)."""
    return "passthrough"


def _chi2_or_f_classif(X, y):
    """chi2 where it is defined, f_classif where it is not.

    chi2 requires non-negative features. The Chapter 1 baseline for
    scaling is "none", so a dataset with naturally negative values
    (ionosphere, for one) reaches chi2 unscaled and raises "Input X must
    be non-negative", failing the run. Found 2026-10-12 when the
    selection sweep reached ionosphere.

    Shifting the data to make it non-negative was rejected: that is a
    scaling operation, and performing one inside the SELECTION component
    would contaminate the factor being measured with the factor being
    held fixed. f_classif is the same univariate-filter idea, defined on
    the real line, and the substitution is recorded on the run.
    """
    import numpy as np

    values = np.asarray(X, dtype=float)
    if np.nanmin(values) < 0:
        return f_classif(X, y)
    return chi2(X, y)


def _chi2(task: str, seed: int, n_features: int):
    """Return a chi-square selector.

    chi2 requires non-negative features and a classification target.
    For regression this falls back to f_regression; for classification
    with negative features it falls back to f_classif. Both are the same
    univariate-filter idea, and both substitutions are recorded per run
    rather than failing the whole sweep.
    """
    score_func = _chi2_or_f_classif if task == "classification" else f_regression
    return SelectKBest(score_func=score_func, k=resolve_k(n_features))


def _mutual_info(task: str, seed: int, n_features: int):
    """Return a mutual-information selector, task-appropriate."""
    if task == "classification":
        score_func = lambda X, y: mutual_info_classif(X, y, random_state=seed)
    else:
        score_func = lambda X, y: mutual_info_regression(X, y, random_state=seed)
    return SelectKBest(score_func=score_func, k=resolve_k(n_features))


def _rfe(task: str, seed: int, n_features: int):
    """Return recursive feature elimination around a shallow tree.

    A decision tree is used as the RFE estimator because it is cheap and
    provides feature_importances_ for both tasks; RFE around the study's
    own model would conflate the selection effect with the model effect.
    """
    base = (DecisionTreeClassifier(random_state=seed) if task == "classification"
            else DecisionTreeRegressor(random_state=seed))
    return RFE(estimator=base, n_features_to_select=resolve_k(n_features))


def _lasso(task: str, seed: int, n_features: int):
    """Return LASSO-based (L1) embedded selection.

    Classification uses L1-penalised logistic regression; regression uses
    Lasso. Features with a zero coefficient are dropped, so the kept
    count is data-dependent rather than fixed -- recorded per run.

    `max_features` is NOT set, so the count stays genuinely
    data-dependent, but `threshold=-inf` with `max_features` would be the
    wrong fix for the real hazard: L1 at a strong penalty can zero EVERY
    coefficient, leaving the model no features at all and raising deep
    inside the pipeline. `min_features` below guarantees at least one
    feature survives by keeping the largest-magnitude coefficient, and
    the run record still reports how many were kept.

    `saga`, not `liblinear`: liblinear cannot fit a multiclass L1 model
    and raises "does not support multiclass classification" on any
    target with three or more classes. Found 2026-10-12 when the
    selection sweep reached iris and wine, both 3-class, and failed 8
    runs. saga supports L1 for both binary and multiclass, at the cost
    of needing more iterations to converge.
    """
    if task == "classification":
        base = LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, random_state=seed
        )
    else:
        base = Lasso(random_state=seed)
    return _MinFeatures(SelectFromModel(estimator=base), minimum=1)


SELECTORS = {
    "none": _none,
    "chi2": _chi2,
    "mutual_info": _mutual_info,
    "rfe": _rfe,
    "lasso": _lasso,
}

# chi2 cannot run on a regression target; the fallback is recorded.
SUBSTITUTIONS = {
    ("chi2", "regression"): "f_regression (chi2 requires a classification target)",
}


def make_selector(level: str, task: str, seed: int, n_features: int):
    """Build the feature selector for one technique level.

    Inputs: level - a key of SELECTORS; task - "classification" or
    "regression"; seed - the global seed; n_features - input feature
    count, used to resolve k for the k-based selectors.
    Output: an unfitted selector, or "passthrough" for the baseline.
    Raises: ValueError naming the valid levels, or an unknown task.
    """
    if level not in SELECTORS:
        raise ValueError(f"unknown selection level {level!r}; valid: {sorted(SELECTORS)}")
    if task not in ("classification", "regression"):
        raise ValueError(f"unknown task {task!r}; valid: classification, regression")
    return SELECTORS[level](task, seed, n_features)


def substitution_note(level: str, task: str, X=None) -> str | None:
    """Return why a technique was substituted, if it was.

    Inputs: level - the selection level; task - the task type; X - the
    feature matrix, needed only for the chi2 case, where the
    substitution depends on the DATA rather than on the task alone.
    Output: the explanation string, or None when the requested technique
    ran as asked. Recorded on the run so a substitution is never silent.
    """
    fixed = SUBSTITUTIONS.get((level, task))
    if fixed:
        return fixed
    # chi2 on a classification target still falls back when any feature
    # is negative, which cannot be known from (level, task) alone.
    if level == "chi2" and task == "classification" and X is not None:
        import numpy as np

        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            return None  # non-numeric columns are encoded before chi2 sees them
        if values.size and np.nanmin(values) < 0:
            return ("f_classif (chi2 requires non-negative features and this "
                    "dataset has negative values under the 'none' scaling "
                    "baseline)")
    return None
