# tuning.py: the hyperparameter-tuning component. Absent from both B1 and
# B2; new here. Chapter 1 baseline is "none".
#
# LEAKAGE: tuning is the subtler of the two risky stages. Selecting
# hyperparameters on the same folds that are reported biases the score
# upward even when no single fold leaked, because the hyperparameters
# were chosen to fit those folds. Every search here is therefore given
# its own INNER cv, and the evaluator nests it inside each outer
# training fold (evaluation/protocols.py). The outer fold scores the
# winner on data the inner search never saw.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# The grids and the cost arithmetic live in budgets.py, which imports
# nothing: the dashboard's composer needs fits_per_run() to cost a sweep
# and runs under its own interpreter, which deliberately lacks sklearn.
# Re-exported here so existing callers are unaffected.
from piplab.components.budgets import (  # noqa: F401
    DEFAULT_N_ITER,
    GRIDS,
    candidate_count,
    fits_per_run,
    grid_for,
)


def make_search(level: str, pipeline, model: str, scoring: str, inner_cv, seed: int):
    """Wrap a pipeline in the requested hyperparameter search.

    Inputs: level - "none", "grid", "random" or "bayesian"; pipeline -
    the unfitted Pipeline to tune; model - the model key, selecting the
    grid; scoring - the sklearn scoring string; inner_cv - the INNER
    splitter, supplied by the evaluator so the search never sees the
    outer test fold; seed - the global seed.
    Output: the pipeline unchanged for the "none" baseline, otherwise a
    search object that is itself fitted per outer fold.
    Raises: ValueError for an unknown level; ImportError with an
    actionable message if Bayesian is requested without optuna.
    """
    if level == "none":
        return pipeline
    if level == "grid":
        return GridSearchCV(
            pipeline, param_grid=grid_for(model), scoring=scoring,
            cv=inner_cv, n_jobs=1, refit=True,
        )
    if level == "random":
        return RandomizedSearchCV(
            pipeline, param_distributions=grid_for(model), scoring=scoring,
            cv=inner_cv, n_iter=DEFAULT_N_ITER, random_state=seed, n_jobs=1, refit=True,
        )
    if level == "bayesian":
        return _make_optuna_search(pipeline, model, scoring, inner_cv, seed)
    raise ValueError(
        f"unknown tuning level {level!r}; valid: none, grid, random, bayesian"
    )


def _make_optuna_search(pipeline, model: str, scoring: str, inner_cv, seed: int):
    """Build an Optuna-backed search over the same grid.

    Kept behind a lazy import so the other three levels run without
    optuna installed, and so a missing dependency produces a clear
    message rather than an import error at module load.
    """
    try:
        from optuna.integration import OptunaSearchCV
        from optuna.distributions import CategoricalDistribution
    except ImportError as exc:  # pragma: no cover - environment-dependent
        # OptunaSearchCV is NOT in the optuna package from 4.x onward; it
        # lives in optuna-integration. Saying "pip install optuna" sends
        # the reader to check a package that is already installed, which
        # is exactly what happened on 2026-10-12.
        raise ImportError(
            "Bayesian tuning needs optuna-integration in this engine's own "
            ".venv, which is a SEPARATE package from optuna: "
            "pip install 'optuna-integration[sklearn]'. "
            "Having optuna alone is not enough, because OptunaSearchCV "
            "moved out of it in optuna 4.x. "
            "Grid and random search need nothing extra."
        ) from exc

    distributions = {
        key: CategoricalDistribution(values)
        for key, values in grid_for(model).items()
    }
    return OptunaSearchCV(
        pipeline, param_distributions=distributions, scoring=scoring,
        cv=inner_cv, n_trials=DEFAULT_N_ITER, random_state=seed, refit=True,
    )
