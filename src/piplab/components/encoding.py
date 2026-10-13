# encoding.py: the categorical-encoding component.
#
# AMENDMENT 2026-10-09: encoding is a sixth experimental factor here.
# The March sources (Chapter_1_Colile.docx, methodology_analysis_Colile.docx)
# name encoding only as a necessary pipeline STAGE in prose, never in the
# components-to-compare table, and no encoding technique is named
# anywhere in them. Promoting it is a deliberate amendment, recorded in
# pipeline-lab/docs/PLAN.md section 2b so the write-up can say so.
#
# LEAKAGE: target encoding builds the feature FROM y, so fitted naively
# it leaks the target straight into the training matrix. sklearn's
# TargetEncoder cross-fits internally for exactly this reason, and it
# sits inside the Pipeline so it is also refitted per outer fold.
# tests/test_encoding.py asserts that target encoding on a pure-noise
# feature does not beat chance.
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

# TargetEncoder's internal cross-fitting folds. Explicit rather than
# relying on the library default, because this value is what makes the
# technique leakage-safe and it belongs in the run record.
TARGET_ENCODER_CV = 5


def _onehot(seed: int, task: str):
    """Return an unfitted one-hot encoder (the baseline).

    handle_unknown="ignore" so a category seen only in a test fold
    encodes as all-zeros rather than raising -- the standard
    leakage-safe treatment, and current behaviour in all three engines.
    """
    return OneHotEncoder(handle_unknown="ignore")


def _ordinal(seed: int, task: str):
    """Return an unfitted ordinal encoder.

    This imposes a false ordering on unordered categories. That is the
    point of including it: the study measures how much that costs, and
    the cost is expected to differ sharply by model family (tree
    ensembles can recover the partition by splitting repeatedly; linear
    models and SVMs read the integer codes as a magnitude). Unseen
    categories map to -1 rather than raising.
    """
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _target(seed: int, task: str):
    """Return an unfitted target encoder, cross-fitted against leakage.

    Replaces a category with a statistic of the target within it.
    Because it reads y, `cv` cross-fitting is what keeps it honest: each
    row's encoding comes from folds that exclude that row.
    """
    # sklearn 1.9 deprecates TargetEncoder(shuffle=, random_state=) and
    # removes them at 1.11: pass a seeded splitter as `cv` instead. That
    # also makes the cross-fitting explicit, which is the property that
    # keeps this technique from leaking y.
    return TargetEncoder(
        cv=KFold(n_splits=TARGET_ENCODER_CV, shuffle=True, random_state=seed)
    )


ENCODERS = {
    "onehot": _onehot,
    "ordinal": _ordinal,
    "target": _target,
}

# Levels that read y while fitting, and therefore depend on the pipeline
# keeping them inside the training fold.
USES_TARGET = frozenset({"target"})


def make_encoder(level: str, seed: int, task: str = "classification"):
    """Build the categorical encoder for one technique level.

    Inputs: level - a key of ENCODERS; seed - the global seed; task -
    "classification" or "regression", accepted for symmetry with the
    other components and for the run record (sklearn's TargetEncoder
    infers its statistic from y itself).
    Output: an unfitted transformer.
    Raises: ValueError naming the valid levels.
    """
    if level not in ENCODERS:
        raise ValueError(f"unknown encoding level {level!r}; valid: {sorted(ENCODERS)}")
    return ENCODERS[level](seed, task)


def uses_target(level: str) -> bool:
    """Say whether this level reads y while fitting.

    Inputs: level - an encoding level.
    Output: True for target encoding, which is why it must never be
    fitted outside a training fold.
    """
    return level in USES_TARGET


def is_noop_for(n_categorical: int) -> bool:
    """Say whether varying encoding is meaningless for a dataset.

    An all-numeric dataset has nothing to encode, so every level builds
    an identical pipeline and the sweep would report a guaranteed null
    that says nothing about encoding. The composer uses this to flag
    such a sweep rather than burn compute on it. Of B2's 12 catalogued
    datasets, 4 are all-numeric (concrete, energy-efficiency, house_16H,
    wdbc).
    Inputs: n_categorical - the dataset's categorical column count.
    Output: True when the dataset has no categorical features.
    """
    return n_categorical == 0
