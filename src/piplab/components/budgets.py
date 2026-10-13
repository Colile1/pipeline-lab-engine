# budgets.py: the hyperparameter grids and the cost arithmetic, kept
# free of sklearn ON PURPOSE.
#
# The dashboard's composer needs fits_per_run() to cost a sweep before
# launching it, and the dashboard runs under its OWN interpreter, which
# deliberately does not have the engines' dependencies. Importing
# components/tuning.py for this pulled in sklearn and broke a fresh
# dashboard install -- caught by the step-13 fresh-environment check.
#
# So the grids and the arithmetic live here (pure data, no imports),
# tuning.py imports them for the real searches, and the dashboard can
# cost a sweep with nothing but the standard library.

# Small, defensible grids. Chapter 1 does not fix these, so they are
# kept deliberately modest: the study compares tuning STRATEGIES, not
# who can search the largest space, and nested CV already makes each
# run ~30x the cost of an untuned one.
GRIDS: dict[str, dict] = {
    "logreg": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "svm": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
    "random_forest": {"model__n_estimators": [100, 300], "model__max_depth": [None, 10, 20]},
    "gradient_boosting": {"model__learning_rate": [0.05, 0.1, 0.2],
                          "model__n_estimators": [100, 200]},
}

# Random search draws this many candidates; Bayesian runs this many trials.
DEFAULT_N_ITER = 10

# Continuous search spaces for the RANDOM and BAYESIAN levels.
#
# Until 2026-10-13 both levels drew from GRIDS above. Every grid has 4 to
# 6 candidates and both levels draw 10, so random search exhausted the
# grid and was identical to grid search on all 56 model-dataset pairs of
# the first tuning sweep, and Bayesian search was close to exhaustive
# too. The factor had three tuned levels in name and one in fact.
#
# Random and Bayesian now share one space per model at the same budget,
# which makes them genuinely different from grid and makes random
# against Bayesian a fair comparison. Grid keeps GRIDS.
#
# Pure data, no imports, like everything else in this module:
#   ("log", low, high)    log-uniform float
#   ("int", low, high)    uniform integer, both ends inclusive
#   ("choice", [values])  one of a fixed list
SPACES: dict[str, dict] = {
    "logreg": {"model__C": ("log", 1e-3, 1e2)},
    "svm": {"model__C": ("log", 1e-2, 1e2),
            "model__gamma": ("log", 1e-4, 1e0)},
    "random_forest": {"model__n_estimators": ("int", 100, 300),
                      "model__max_depth": ("choice", [None, 5, 10, 20, 30]),
                      "model__min_samples_leaf": ("int", 1, 5)},
    "gradient_boosting": {"model__learning_rate": ("log", 0.01, 0.3),
                          "model__n_estimators": ("int", 100, 300),
                          "model__max_depth": ("int", 2, 5)},
}


def space_for(model: str) -> dict:
    """Return the continuous search space for one model.

    Inputs: model - a model key.
    Output: the space dict, keyed for a Pipeline step named "model".
    Raises: ValueError naming the models that have spaces.
    """
    if model not in SPACES:
        raise ValueError(f"no search space for model {model!r}; have: {sorted(SPACES)}")
    return SPACES[model]


def searched_parameters(level: str, model: str) -> list[str]:
    """Return the parameter names a level searches for a model.

    Grid searches GRIDS; random and Bayesian search SPACES. The evaluator
    uses this to refuse a search whose parameters the real estimator does
    not have, whichever level is asked for.
    """
    if level == "none":
        return []
    source = grid_for(model) if level == "grid" else space_for(model)
    return sorted(source)


def grid_for(model: str) -> dict:
    """Return the hyperparameter grid for one model.

    Inputs: model - a model key.
    Output: the grid dict, keyed for a Pipeline step named "model".
    Raises: ValueError naming the models that have grids.
    """
    if model not in GRIDS:
        raise ValueError(f"no tuning grid for model {model!r}; have: {sorted(GRIDS)}")
    return GRIDS[model]


def candidate_count(level: str, model: str) -> int:
    """Return how many hyperparameter candidates a level searches.

    Inputs: level - the tuning level; model - the model key.
    Output: 0 for "none" (nothing is searched), the grid's product for
    "grid", else DEFAULT_N_ITER.
    """
    if level == "none":
        return 0
    if level == "grid":
        candidates = 1
        for values in grid_for(model).values():
            candidates *= len(values)
        return candidates
    return DEFAULT_N_ITER


def fits_per_run(level: str, model: str, outer_k: int, inner_k: int) -> int:
    """Return how many model fits one run costs.

    This is the number the composer shows before launching: tuning turns
    10 fits into hundreds, and that has to be visible up front.
    Inputs: level - the tuning level; model - the model key; outer_k and
    inner_k - the nested CV fold counts.
    Output: total model fits for one (dataset, model, technique) run.
    """
    if level == "none":
        return outer_k
    # Per outer fold: every candidate is fitted on every inner fold,
    # then the winner is refitted once on the full outer training part.
    return outer_k * (candidate_count(level, model) * inner_k + 1)
