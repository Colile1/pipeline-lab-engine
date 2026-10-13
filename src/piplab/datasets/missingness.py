# missingness.py: controlled MCAR and MAR injection.
#
# WHY THIS EXISTS. The imputation arm of the OFAT programme is
# structurally untestable on this dataset collection: only 4 of 19
# datasets have any naturally missing values, so 15 of them return
# identical scores for every imputation level and Wilcoxon discards
# them as ties. The effective n is 4 against a Holm bar needing 7, so
# no imputation contrast can be rejected whatever the effect.
# PROCESS-LOG.md section 4c has the full reasoning.
#
# Injecting missingness into complete datasets makes every dataset
# informative and lifts the arm to n = 19.
#
# PROVENANCE. This is a deliberate port of the B2 artefact's
# `data/missingness.py`, written by Colile Sibanda for the
# Research_Impact_of_Preprocess study. It is ported rather than imported
# because B2 is a FROZEN research record: importing across would couple
# this engine to a repository that must not change, and copying keeps
# the two studies comparable while leaving B2 untouched.
#
# The algorithm is unchanged, so lab results stay directly comparable
# with B2's. A test asserts the two produce identical masks for the same
# seed, which is what makes that claim checkable rather than asserted.
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_MECHANISMS = ("mcar", "mar")


def inject_missing(X: pd.DataFrame, rate: float, mechanism: str,
                   seed: int) -> pd.DataFrame:
    """Inject controlled missingness into a complete frame's numeric columns.

    The rate is the fraction of numeric cells set to NaN, hit exactly
    (within rounding) by count-based sampling rather than per-cell
    Bernoulli draws, so the realised rate does not drift on small data.

    "mcar" removes cells uniformly at random: missingness depends on
    nothing, which is the assumption most imputation methods are
    justified under.

    "mar" makes missingness depend on one fully observed driver column
    (the numeric column with the most distinct values): rows with higher
    driver values are proportionally more likely to lose values. The
    driver itself receives no NaN, and the per-column rate is scaled up
    so the overall numeric rate still equals `rate`. This is MAR, not
    MNAR, because the probability is computed from the driver and never
    from the target's own values.

    Categorical columns are never touched.

    Inputs: X - a complete feature frame; rate - target fraction of
    numeric cells, strictly in (0, 1); mechanism - "mcar" or "mar";
    seed - seeds a local Generator, so the mask is identical for a given
    seed regardless of global random state.
    Output: a NEW frame; the input is never mutated.
    Raises: ValueError naming the offending input for every contract
    violation, including a rate that rounds to zero cells.
    """
    numeric_cols = _validate(X, rate, mechanism)
    rng = np.random.default_rng(seed)
    if mechanism == "mcar":
        mask = _mcar_mask(len(X), len(numeric_cols), rate, rng)
    else:
        mask = _mar_mask(X[numeric_cols], rate, rng)

    injected = X.copy()
    mask_frame = pd.DataFrame(mask, index=X.index, columns=numeric_cols)
    injected[numeric_cols] = X[numeric_cols].mask(mask_frame)
    return injected


def _validate(X: pd.DataFrame, rate: float, mechanism: str) -> list[str]:
    """Check every input and return the numeric column names."""
    if mechanism not in VALID_MECHANISMS:
        raise ValueError(
            f"unknown mechanism {mechanism!r}; valid: {list(VALID_MECHANISMS)}"
        )
    if not 0 < rate < 1:
        raise ValueError(f"rate {rate!r} must be strictly between 0 and 1")

    numeric_cols = [
        column for column in X.columns
        if not (
            pd.api.types.is_object_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        )
    ]
    if not numeric_cols:
        raise ValueError("X has no numeric columns; nothing to inject into")
    if X[numeric_cols].isna().any().any():
        # Injecting on top of natural missingness would make the realised
        # rate unknowable, and the recorded rate a lie.
        raise ValueError(
            "the numeric columns of X already contain NaN. Injection is "
            "defined for complete data only, so that the recorded rate is "
            "the real one. Use a dataset without natural missingness."
        )
    if mechanism == "mar" and len(numeric_cols) < 2:
        raise ValueError(
            "MAR needs at least 2 numeric columns (a fully observed driver "
            f"plus targets); X has {len(numeric_cols)}"
        )
    return numeric_cols


def _mcar_mask(n_rows: int, n_cols: int, rate: float,
               rng: np.random.Generator) -> np.ndarray:
    """A mask with exactly round(rate * cells) uniformly chosen True cells."""
    n_cells = n_rows * n_cols
    k = round(rate * n_cells)
    if k < 1:
        raise ValueError(
            f"rate={rate} over {n_cells} numeric cells rounds to zero cells; "
            "increase the rate or use more data"
        )
    flat = rng.choice(n_cells, size=k, replace=False)
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[np.unravel_index(flat, mask.shape)] = True
    return mask


def _mar_mask(numeric: pd.DataFrame, rate: float,
              rng: np.random.Generator) -> np.ndarray:
    """A mask whose row probabilities follow the driver column's rank.

    Each target column independently loses exactly
    round(adjusted rate * rows) rows, sampled without replacement with
    probability proportional to the driver's rank: strictly increasing
    in the driver, so the dependence is strong right-tail MAR.
    """
    n_rows, n_cols = numeric.shape
    driver = _select_driver(numeric)
    # The driver stays complete, so the remaining columns must carry the
    # whole budget for the overall rate to come out right.
    rate_per_column = rate * n_cols / (n_cols - 1)
    if rate_per_column >= 1:
        raise ValueError(
            f"rate={rate} is unreachable under MAR with {n_cols} numeric "
            f"columns: the driver stays complete, so each target column "
            f"would need rate {rate_per_column:.3f} >= 1. Use a lower rate."
        )
    k_per_column = round(rate_per_column * n_rows)
    if k_per_column < 1:
        raise ValueError(
            f"rate={rate} over {n_rows} rows rounds to zero rows per column; "
            "increase the rate or use more data"
        )

    ranks = numeric[driver].rank(method="average").to_numpy()
    probabilities = ranks / ranks.sum()
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for position, column in enumerate(numeric.columns):
        if column == driver:
            continue
        rows = rng.choice(
            n_rows, size=k_per_column, replace=False, p=probabilities
        )
        mask[rows, position] = True
    return mask


def _select_driver(numeric: pd.DataFrame) -> str:
    """The numeric column with the most distinct values.

    High cardinality keeps the rank weighting fine-grained, so the MAR
    dependence cannot degenerate towards MCAR. Ties break on column
    order, which keeps the choice deterministic.
    """
    distinct = numeric.nunique()
    driver = distinct.idxmax()
    if distinct[driver] < 2:
        raise ValueError(
            "every numeric column is constant; MAR needs a driver with at "
            "least 2 distinct values to define missingness probabilities"
        )
    return str(driver)
