# Confirms controlled missingness injection: the rates are hit exactly,
# MAR really depends on the driver, MCAR really does not, and the port
# reproduces B2's masks bit for bit so the two studies stay comparable.
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piplab.datasets.missingness import VALID_MECHANISMS, inject_missing

SEED = 671

# B2 is a frozen research artefact, so it is loaded by path rather than
# imported as a package: nothing here may depend on B2 being installed,
# and nothing here may modify it.
B2_MISSINGNESS = (
    Path(__file__).resolve().parents[2]
    / "671 Research June" / "Artefact" / "Research_Impact_of_Preprocess"
    / "src" / "research_impact_of_preprocess" / "data" / "missingness.py"
)


@pytest.fixture
def frame():
    """A complete frame with numeric and categorical columns."""
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        rng.normal(size=(120, 6)), columns=[f"f{i}" for i in range(6)]
    )
    data["cat"] = pd.Categorical(rng.choice(list("abc"), 120))
    return data


@pytest.mark.parametrize("mechanism", VALID_MECHANISMS)
@pytest.mark.parametrize("rate", [0.05, 0.15, 0.30])
def test_the_realised_rate_matches_the_requested_rate(frame, mechanism, rate):
    """Count-based sampling, so the rate does not drift on small data."""
    out = inject_missing(frame, rate, mechanism, SEED)
    numeric = [c for c in frame.columns if c != "cat"]
    realised = out[numeric].isna().sum().sum() / out[numeric].size
    assert realised == pytest.approx(rate, abs=0.01)


@pytest.mark.parametrize("mechanism", VALID_MECHANISMS)
def test_categorical_columns_are_never_touched(frame, mechanism):
    out = inject_missing(frame, 0.3, mechanism, SEED)
    assert not out["cat"].isna().any()


@pytest.mark.parametrize("mechanism", VALID_MECHANISMS)
def test_the_input_frame_is_not_mutated(frame, mechanism):
    before = frame.copy()
    inject_missing(frame, 0.2, mechanism, SEED)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("mechanism", VALID_MECHANISMS)
def test_the_same_seed_gives_the_same_mask(frame, mechanism):
    first = inject_missing(frame, 0.15, mechanism, SEED)
    second = inject_missing(frame, 0.15, mechanism, SEED)
    assert first.isna().equals(second.isna())


@pytest.mark.parametrize("mechanism", VALID_MECHANISMS)
def test_a_different_seed_gives_a_different_mask(frame, mechanism):
    """Non-vacuous: the seed must actually drive the choice."""
    first = inject_missing(frame, 0.15, mechanism, SEED)
    second = inject_missing(frame, 0.15, mechanism, SEED + 1)
    assert not first.isna().equals(second.isna())


def test_mar_missingness_depends_on_the_driver(frame):
    """The defining property: rows with higher driver values lose more.

    Without this the mechanism would be MCAR wearing a MAR label, and
    every conclusion drawn about mechanism sensitivity would be void.
    """
    out = inject_missing(frame, 0.3, "mar", SEED)
    numeric = [c for c in frame.columns if c != "cat"]
    # The driver is the most distinct column and stays complete.
    driver = frame[numeric].nunique().idxmax()
    assert not out[driver].isna().any()

    targets = [c for c in numeric if c != driver]
    missing_per_row = out[targets].isna().sum(axis=1)
    high = frame[driver] > frame[driver].median()
    assert missing_per_row[high].mean() > missing_per_row[~high].mean()


def test_mcar_missingness_does_not_depend_on_any_column(frame):
    """The contrast that makes the MAR test meaningful."""
    out = inject_missing(frame, 0.3, "mcar", SEED)
    numeric = [c for c in frame.columns if c != "cat"]
    missing_per_row = out[numeric].isna().sum(axis=1)
    high = frame["f0"] > frame["f0"].median()
    difference = abs(missing_per_row[high].mean() - missing_per_row[~high].mean())
    assert difference < 0.5, "MCAR should not track any column"


@pytest.mark.parametrize("rate", [0, 1, -0.1, 1.5])
def test_an_impossible_rate_is_refused(frame, rate):
    with pytest.raises(ValueError, match="between 0 and 1"):
        inject_missing(frame, rate, "mcar", SEED)


def test_an_unknown_mechanism_is_refused(frame):
    with pytest.raises(ValueError, match="unknown mechanism"):
        inject_missing(frame, 0.1, "mnar", SEED)


def test_injecting_on_top_of_natural_missingness_is_refused(frame):
    """Otherwise the recorded rate would not be the realised rate."""
    dirty = frame.copy()
    dirty.loc[0, "f0"] = np.nan
    with pytest.raises(ValueError, match="already contain NaN"):
        inject_missing(dirty, 0.1, "mcar", SEED)


def test_mar_needs_more_than_one_numeric_column():
    single = pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="at least 2 numeric columns"):
        inject_missing(single, 0.25, "mar", SEED)


def test_a_rate_rounding_to_zero_cells_is_refused():
    tiny = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="rounds to zero"):
        inject_missing(tiny, 0.001, "mcar", SEED)


@pytest.mark.skipif(
    not B2_MISSINGNESS.exists(), reason="B2 artefact not present on this machine"
)
@pytest.mark.parametrize("mechanism", VALID_MECHANISMS)
@pytest.mark.parametrize("rate", [0.05, 0.15, 0.30])
def test_the_port_reproduces_b2_exactly(frame, mechanism, rate):
    """The claim that makes lab and B2 imputation results comparable.

    This module is a port of B2's, so the two must produce the IDENTICAL
    mask for the same seed. If they ever diverge, the two studies stop
    being comparable and the port has silently become a different
    experiment. B2 is frozen, so it is loaded by path and never
    modified.
    """
    spec = importlib.util.spec_from_file_location("b2_missingness", B2_MISSINGNESS)
    b2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b2)

    ported = inject_missing(frame, rate, mechanism, SEED)
    original = b2.inject_missing(frame, rate, mechanism, SEED)
    assert ported.isna().equals(original.isna())
