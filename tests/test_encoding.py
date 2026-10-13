# Encoding as a sixth factor (AMENDMENT, PLAN.md section 2b).
#
# The important test here is the target-encoding leakage proof. Target
# encoding builds the feature FROM y, so fitted without cross-fitting it
# leaks the target directly and a pure-noise feature will appear
# predictive. That failure mode is silent -- the scores just come out
# high -- so it is asserted explicitly.
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from piplab.components import encoding
from piplab.config import TECHNIQUES
from piplab.pipelines.compose import build_pipeline, column_roles
from piplab.pipelines.spies import FitSpy

SEED = 671
BASE = {"encoding": "onehot", "imputation": "median", "scaling": "standard", "selection": "none"}


@pytest.fixture
def categorical_clf():
    """Classification data where the signal lives in a categorical column."""
    rng = np.random.default_rng(SEED)
    n = 200
    cat = rng.choice(["a", "b", "c", "d"], size=n)
    # "a" and "b" are mostly class 1; "c" and "d" mostly class 0.
    p = np.where(np.isin(cat, ["a", "b"]), 0.85, 0.15)
    y = pd.Series(rng.binomial(1, p))
    X = pd.DataFrame({"num": rng.normal(size=n), "cat": cat})
    return X, y


@pytest.fixture
def noise_categorical_clf():
    """A high-cardinality categorical that carries NO signal at all.

    This is the leakage canary: with naive target encoding, each category
    is nearly unique, so its target mean *is* the label, and accuracy
    shoots to ~1.0. With correct cross-fitting it must stay near chance.
    """
    rng = np.random.default_rng(SEED)
    n = 200
    X = pd.DataFrame({
        "num": rng.normal(size=n),
        # 100 categories across 200 rows: ~2 rows each.
        "cat": [f"c{i}" for i in rng.integers(0, 100, size=n)],
    })
    y = pd.Series(rng.binomial(1, 0.5, size=n))  # pure coin flip
    return X, y


# --- the amendment is wired through ---------------------------------------

def test_encoding_now_offers_three_levels_with_onehot_as_baseline():
    assert TECHNIQUES["encoding"] == ("onehot", "ordinal", "target")
    # onehot stays first, so it remains the baseline and existing configs
    # behave exactly as before the amendment.
    assert TECHNIQUES["encoding"][0] == "onehot"


@pytest.mark.parametrize("level", TECHNIQUES["encoding"])
def test_every_encoding_level_builds(level):
    enc = encoding.make_encoder(level, SEED, "classification")
    assert hasattr(enc, "fit")


def test_unknown_encoding_level_is_rejected():
    with pytest.raises(ValueError, match="bogus"):
        encoding.make_encoder("bogus", SEED)


def test_only_target_encoding_reads_y():
    assert encoding.uses_target("target") is True
    assert encoding.uses_target("onehot") is False
    assert encoding.uses_target("ordinal") is False


def test_noop_detection_for_all_numeric_datasets():
    # 4 of B2's 12 datasets are all-numeric, where varying encoding gives
    # an identical pipeline and a guaranteed, meaningless null.
    assert encoding.is_noop_for(0) is True
    assert encoding.is_noop_for(1) is False


# --- every level composes and fits ----------------------------------------

@pytest.mark.parametrize("level", TECHNIQUES["encoding"])
def test_every_encoding_level_composes_and_fits(level, categorical_clf):
    X, y = categorical_clf
    pipe = build_pipeline(levels={**BASE, "encoding": level}, model="logreg",
                          roles=column_roles(X), task="classification", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


@pytest.mark.parametrize("level", TECHNIQUES["encoding"])
def test_every_encoding_level_works_for_regression(level):
    rng = np.random.default_rng(SEED)
    n = 200
    cat = rng.choice(["a", "b", "c"], size=n)
    X = pd.DataFrame({"num": rng.normal(size=n), "cat": cat})
    y = pd.Series(np.where(cat == "a", 5.0, 1.0) + rng.normal(scale=0.3, size=n))
    pipe = build_pipeline(levels={**BASE, "encoding": level}, model="logreg",
                          roles=column_roles(X), task="regression", seed=SEED)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)


def test_ordinal_produces_one_column_per_categorical_not_many(categorical_clf):
    """Ordinal keeps width; one-hot expands it. That difference is the
    mechanism the amendment exists to measure."""
    X, y = categorical_clf
    widths = {}
    for level in ("onehot", "ordinal"):
        pipe = build_pipeline(levels={**BASE, "encoding": level}, model="logreg",
                              roles=column_roles(X), task="classification", seed=SEED)
        pipe.fit(X, y)
        widths[level] = pipe.named_steps["pre"].transform(X).shape[1]
    # 1 numeric + 4 one-hot columns vs 1 numeric + 1 ordinal column.
    assert widths["onehot"] == 5
    assert widths["ordinal"] == 2


# --- THE LEAKAGE PROOF ----------------------------------------------------

def test_target_encoding_does_not_make_pure_noise_predictive(noise_categorical_clf):
    """The canary. If y leaked through the encoder, accuracy would be ~1.0.

    With correct cross-fitting, a high-cardinality categorical that
    carries no signal must score near chance (0.5).
    """
    X, y = noise_categorical_clf
    pipe = build_pipeline(levels={**BASE, "encoding": "target"}, model="logreg",
                          roles=column_roles(X), task="classification", seed=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    mean = float(scores.mean())
    assert mean < 0.65, (
        f"target encoding scored {mean:.3f} on pure noise — y has leaked "
        "into the features (expected ~0.5)"
    )


def test_naive_target_encoding_WOULD_leak_proving_the_test_can_fail():
    """Shows the canary is not vacuous.

    Hand-built naive target encoding (category -> target mean over the
    WHOLE dataset, no cross-fitting) on the same noise data should score
    far above chance. This is the mistake sklearn's cv= prevents.
    """
    rng = np.random.default_rng(SEED)
    n = 200
    cats = [f"c{i}" for i in rng.integers(0, 100, size=n)]
    y = pd.Series(rng.binomial(1, 0.5, size=n))
    frame = pd.DataFrame({"cat": cats, "y": y})
    # The leak: each category's encoding includes its own rows' labels.
    means = frame.groupby("cat")["y"].transform("mean")
    # A trivial threshold on the leaked feature recovers the label well.
    leaked_accuracy = float(((means > 0.5).astype(int) == y).mean())
    # Measured at 0.75 on this seed, vs 0.5 chance. It is not higher only
    # because 56 of the 200 rows fall in 2-row categories that split 1/1,
    # giving a mean of exactly 0.5 where the `> 0.5` rule coin-flips. The
    # leak is still unmistakable: 0.75 vs the ~0.5 the cross-fitted
    # encoder scores on the same data in the test above.
    assert leaked_accuracy >= 0.70, (
        "naive target encoding should visibly leak; if this fails the "
        "canary above proves nothing"
    )


def test_target_encoder_is_refitted_per_fold_on_training_rows_only(categorical_clf):
    """Extends the step-3 fit-spy proof to the encoder."""
    X, y = categorical_clf
    FitSpy.reset()
    pipe = build_pipeline(levels={**BASE, "encoding": "target"}, model="logreg",
                          roles=column_roles(X), task="classification",
                          seed=SEED, instrument=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_sets = []
    for train_idx, test_idx in splitter.split(X, y):
        clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        test_sets.append(frozenset(X.index[test_idx]))

    records = FitSpy.records("encoder")
    assert len(records) == len(test_sets), "encoder should fit once per fold"
    for record, held_out in zip(records, test_sets):
        ids = record["row_ids"] if record["indexed"] else record["y_fingerprint"]
        assert ids is not None
        overlap = ids & held_out
        assert not overlap, f"encoder saw {len(overlap)} held-out rows — leakage"


def test_target_encoder_cross_fitting_is_explicit_and_seeded():
    """Cross-fitting is what makes this technique safe, so it is pinned.

    Passed as a seeded splitter rather than cv=int + random_state, which
    sklearn 1.9 deprecates and 1.11 removes.
    """
    enc = encoding.make_encoder("target", SEED)
    assert enc.cv.get_n_splits() == encoding.TARGET_ENCODER_CV == 5
    assert enc.cv.random_state == SEED
    assert enc.cv.shuffle is True
