# models.py: Chapter 1's four classical model types, at library defaults.
# Tuning is a SEPARATE component here: a model built by this module is
# always untuned, and components/tuning.py wraps it when the tuning level
# asks for it. That separation is what lets the study attribute an effect
# to tuning rather than to the model.
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR


def _logreg(task: str, seed: int):
    """Logistic regression, or linear regression for a continuous target.

    Linear regression is the regression counterpart of logistic
    regression and has no randomness, so seed is unused there.
    """
    if task == "regression":
        return LinearRegression()
    return LogisticRegression(max_iter=1000, random_state=seed)


def _svm(task: str, seed: int):
    """Support vector machine, RBF kernel at library defaults."""
    return SVR() if task == "regression" else SVC(random_state=seed)


def _random_forest(task: str, seed: int):
    """Random forest at library defaults."""
    cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
    return cls(random_state=seed)


def _gradient_boosting(task: str, seed: int):
    """Gradient boosting at library defaults."""
    cls = GradientBoostingRegressor if task == "regression" else GradientBoostingClassifier
    return cls(random_state=seed)


MODEL_FACTORIES = {
    "logreg": _logreg,
    "svm": _svm,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
}

# Where the model key names a classifier but a regression target forces a
# different estimator. The run record stores "logreg" either way, so
# without this note a reader of lab.db could not tell that a regression
# row was produced by LinearRegression. It matters for interpretation:
# LinearRegression is invariant to affine rescaling in exact arithmetic,
# so its scaling contrast is null on well-conditioned data, and that
# null is a property of the estimator rather than a finding. It is NOT
# guaranteed in practice: on house_16H scaling lowered RMSE by 7.7% in
# two separate sweeps. The cause was checked directly on 2026-10-13 and
# is the solver's rank cutoff, not rounding error. One house_16H feature
# has a standard deviation near 66,000 and the other fifteen are below
# 0.34, so 7 of the 16 singular values fall under tol (1e-6) times the
# largest and the least-squares solver discards them: sklearn reports
# rank 9 of 16 on the raw data and 16 of 16 once scaled. Refitting the
# raw data with tol=1e-9 restores rank 16 and gives exactly the scaled
# score. The note below says so rather than overclaiming.
MODEL_SUBSTITUTIONS = {
    ("logreg", "regression"):
        "logreg -> LinearRegression on a continuous target "
        "(scale-invariant in exact arithmetic; can still shift on "
        "badly conditioned data)",
    ("svm", "regression"): "svm -> SVR on a continuous target",
    ("random_forest", "regression"):
        "random_forest -> RandomForestRegressor on a continuous target",
    ("gradient_boosting", "regression"):
        "gradient_boosting -> GradientBoostingRegressor on a continuous target",
}


def substitution_note(name: str, task: str) -> str | None:
    """Return why a different estimator ran for this task, if one did.

    Inputs: name - the model key; task - the task type.
    Output: the explanation string, or None when the named model ran as
    asked. Recorded on the run so a substitution is never silent.
    """
    return MODEL_SUBSTITUTIONS.get((name, task))


def make_model(name: str, task: str, seed: int):
    """Build one untuned estimator.

    Inputs: name - a key of MODEL_FACTORIES; task - "classification" or
    "regression"; seed - the global seed.
    Output: an unfitted estimator at library defaults.
    Raises: ValueError naming the valid models or tasks.
    """
    if name not in MODEL_FACTORIES:
        raise ValueError(f"unknown model {name!r}; valid: {sorted(MODEL_FACTORIES)}")
    if task not in ("classification", "regression"):
        raise ValueError(f"unknown task {task!r}; valid: classification, regression")
    return MODEL_FACTORIES[name](task, seed)
