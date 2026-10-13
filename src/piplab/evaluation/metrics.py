# metrics.py: exactly the metric set methodology_analysis_Colile.docx
# section 9 selected -- accuracy and F1 for classification, RMSE for
# regression. Nothing else.
#
# Adding ROC-AUC, precision or recall would be an AMENDMENT to that
# section, not a code improvement: section 9 argues this set deliberately
# (accuracy alone misleads under class imbalance; F1 addresses it; RMSE
# for regression). See pipeline-lab/docs/SCOPE-DECISIONS.md Decision 2.
#
# F1 uses average="macro" and RMSE uses sqrt(MSE), matching prepfx's own
# metrics module exactly, so figures stay comparable with the B1 results
# already reported.
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Whether a higher value is better, which the statistics step needs for
# ranking (Friedman ranks configurations, so direction matters).
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "rmse": False}


def _accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def _f1_macro(y_true, y_pred) -> float:
    """Macro-averaged F1, matching prepfx so results stay comparable."""
    return float(f1_score(y_true, y_pred, average="macro"))


def _rmse(y_true, y_pred) -> float:
    """Root mean squared error, matching prepfx's sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_for(task: str) -> dict:
    """Return the metric callables for one task.

    Inputs: task - "classification" or "regression".
    Output: ordered dict of metric name -> callable(y_true, y_pred).
    Raises: ValueError naming the valid tasks.
    """
    if task == "classification":
        return {"accuracy": _accuracy, "f1": _f1_macro}
    if task == "regression":
        return {"rmse": _rmse}
    raise ValueError(f"unknown task {task!r}; valid: 'classification', 'regression'")


def scoring_for(task: str) -> str:
    """Return the sklearn scoring string a hyperparameter search should use.

    The inner search optimises the task's primary metric, so tuning is
    not judged on a different quantity from the one reported.
    Inputs: task - "classification" or "regression".
    Output: an sklearn scoring name.
    """
    if task == "classification":
        return "accuracy"
    if task == "regression":
        return "neg_root_mean_squared_error"
    raise ValueError(f"unknown task {task!r}; valid: 'classification', 'regression'")


def summarise(per_fold: dict[str, list[float]]) -> dict:
    """Reduce per-fold scores to the mean and sd columns of the run record.

    The sd is not decoration: SQ1 asks which component has the "most
    consistent" effect, which is a claim about spread, and the
    Friedman/Wilcoxon tests need the per-fold structure. Both are kept.
    Inputs: per_fold - {metric: [score per fold]}.
    Output: {"<metric>_mean": float, "<metric>_sd": float} for each
    metric present; sd is 0.0 for a single fold rather than NaN, since
    one observation has no spread rather than an unknown one.
    """
    out: dict[str, float] = {}
    for metric, scores in per_fold.items():
        values = np.asarray([float(s) for s in scores], dtype=float)
        if values.size == 0:
            continue
        out[f"{metric}_mean"] = round(float(values.mean()), 6)
        out[f"{metric}_sd"] = round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6)
    return out
