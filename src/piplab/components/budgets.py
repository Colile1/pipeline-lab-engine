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
