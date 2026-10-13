# spies.py: a transparent wrapper that records what a step was fitted on,
# so the leakage proofs can assert that selection and tuning never saw a
# test fold. Mirrors prepfx's own FitSpy, which proves the same property
# for its imputer and scaler.
from sklearn.base import BaseEstimator, TransformerMixin


class FitSpy(BaseEstimator, TransformerMixin):
    """Delegates to an inner transformer and logs every fit call.

    Inputs (constructor): inner - the transformer to wrap; spy_name -
    the key its records are filed under.
    Behaviour: fit/transform pass straight through; each fit appends
    {n_rows, row_ids} to FitSpy.RECORDS[spy_name]. Cloneable, so it
    survives sklearn's per-fold clone().
    """

    RECORDS: dict[str, list[dict]] = {}

    def __init__(self, inner, spy_name: str):
        self.inner = inner
        self.spy_name = spy_name

    @classmethod
    def reset(cls) -> None:
        """Clear all recorded fits. Call before each leakage assertion."""
        cls.RECORDS = {}

    @classmethod
    def records(cls, spy_name: str) -> list[dict]:
        """Return the fit records for one spy name (empty if never fitted)."""
        return cls.RECORDS.get(spy_name, [])

    def fit(self, X, y=None, **kwargs):
        """Record what this fit saw, then delegate to the inner step.

        `row_ids` is only meaningful when X still carries a pandas index.
        Once an upstream transformer has converted X to a numpy array the
        original row identity is genuinely gone, so `indexed` is False
        and a leakage assertion must use `y_fingerprint` instead of
        inventing positional ids that would collide with real index
        values.
        """
        index = getattr(X, "index", None)
        indexed = index is not None
        ids = list(index) if indexed else []
        FitSpy.RECORDS.setdefault(self.spy_name, []).append({
            "n_rows": int(len(X)),
            "row_ids": frozenset(ids),
            "indexed": indexed,
            "y_fingerprint": _y_fingerprint(y),
        })
        self.inner.fit(X, y, **kwargs)
        return self

    def transform(self, X):
        """Delegate transform unchanged."""
        return self.inner.transform(X)

    def get_params(self, deep=True):
        return {"inner": self.inner, "spy_name": self.spy_name}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _y_fingerprint(y):
    """Return a stable fingerprint of the target this fit saw.

    y keeps its pandas index through a Pipeline even when X is converted
    to a numpy array, so it is the reliable way to identify which rows a
    deep pipeline stage was fitted on.
    Inputs: y - the target passed to fit, or None.
    Output: a frozenset of y's index values, or None when y is absent or
    carries no index.
    """
    index = getattr(y, "index", None)
    return frozenset(index) if index is not None else None
