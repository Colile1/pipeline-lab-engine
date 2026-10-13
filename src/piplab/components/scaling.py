# scaling.py: the feature-scaling component. Chapter 1 baseline is "none".
# Every factory returns an UNFITTED transformer, so the pipeline fits it
# inside each CV fold and no scaling statistic ever crosses a fold boundary.
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def _none(seed: int):
    """Return the no-scaling baseline.

    Inputs: seed - unused (scaling has no randomness).
    Output: "passthrough", sklearn's identity step, so the pipeline
    shape stays the same whether or not scaling is applied.
    """
    return "passthrough"


def _standard(seed: int):
    """Return an unfitted StandardScaler (zero mean, unit variance)."""
    return StandardScaler()


def _minmax(seed: int):
    """Return an unfitted MinMaxScaler (range [0, 1])."""
    return MinMaxScaler()


def _robust(seed: int):
    """Return an unfitted RobustScaler (median and IQR, outlier-tolerant)."""
    return RobustScaler()


SCALERS = {
    "none": _none,
    "standard": _standard,
    "minmax": _minmax,
    "robust": _robust,
}


def make_scaler(level: str, seed: int):
    """Build the scaler for one technique level.

    Inputs: level - a key of SCALERS; seed - the global seed.
    Output: an unfitted transformer, or "passthrough" for the baseline.
    Raises: ValueError naming the valid levels.
    """
    if level not in SCALERS:
        raise ValueError(f"unknown scaling level {level!r}; valid: {sorted(SCALERS)}")
    return SCALERS[level](seed)
