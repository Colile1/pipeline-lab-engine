# Confirms methodology section 7.2's framework: Friedman globally, then
# baseline-referenced Wilcoxon with Holm, and no p-value without an
# effect size.
import math

import numpy as np
import pytest

from piplab.statistics import (
    ALPHA,
    Comparison,
    EffectSize,
    StatisticsError,
    compare_levels,
    effect_size,
    friedman_test,
    holm_adjust,
    wilcoxon_against_baseline,
)

SEED = 671


def strong_scores() -> dict[str, list[float]]:
    """Three levels where two clearly beat the baseline, across 12 datasets."""
    base = [0.80, 0.78, 0.82, 0.76, 0.81, 0.79, 0.77, 0.83, 0.75, 0.80, 0.78, 0.81]
    return {
        "none": base,
        "standard": [b + 0.05 for b in base],
        "robust": [b + 0.04 for b in base],
    }


def null_scores() -> dict[str, list[float]]:
    """Three levels that genuinely do not differ."""
    rng = np.random.default_rng(SEED)
    base = rng.uniform(0.75, 0.85, size=12)
    return {
        "none": list(base),
        "standard": list(base + rng.normal(0, 0.002, size=12)),
        "robust": list(base + rng.normal(0, 0.002, size=12)),
    }


# --- Holm, and why it is not Bonferroni ----------------------------------

def test_holm_matches_a_hand_checkable_example():
    # m=4: smallest p x4, next x3, next x2, largest x1, made monotone.
    assert [round(x, 4) for x in holm_adjust([0.01, 0.04, 0.03, 0.005])] == [0.03, 0.06, 0.06, 0.02]


def test_holm_preserves_input_order():
    adjusted = holm_adjust([0.5, 0.001, 0.2])
    assert adjusted[1] < adjusted[2] < adjusted[0]


def test_holm_is_never_more_conservative_than_bonferroni():
    """The guarantee that makes Holm a free upgrade."""
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        m = int(rng.integers(2, 11))
        ps = list(rng.uniform(0, 1, size=m))
        holm = holm_adjust(ps)
        bonferroni = [min(p * m, 1.0) for p in ps]
        for h, b in zip(holm, bonferroni):
            assert h <= b + 1e-12


def test_holm_rejects_a_superset_of_bonferroni_and_sometimes_more():
    """A concrete case where Bonferroni would lose a real finding.

    Raw p = 0.013 adjusts to 0.013 under Holm (rejected) but 0.052 under
    Bonferroni (not rejected) at alpha = 0.05. That is one finding
    discarded for nothing: both control family-wise error identically.
    """
    ps = [0.001, 0.001, 0.001, 0.013]
    holm = holm_adjust(ps)
    bonferroni = [min(p * 4, 1.0) for p in ps]
    assert sum(p < ALPHA for p in holm) == 4
    assert sum(p < ALPHA for p in bonferroni) == 3
    assert holm[3] == pytest.approx(0.013)
    assert bonferroni[3] == pytest.approx(0.052)


def test_holm_is_monotone_in_rank():
    """An adjusted value must never undercut a smaller raw p's adjustment."""
    adjusted = sorted(holm_adjust([0.001, 0.002, 0.003, 0.9]))
    assert adjusted == sorted(adjusted)


def test_holm_handles_the_empty_case():
    assert holm_adjust([]) == []


# --- Friedman -------------------------------------------------------------

def test_friedman_detects_a_real_difference():
    result = friedman_test(strong_scores())
    assert result.significant
    assert result.p_value < ALPHA
    assert result.n_levels == 3
    assert result.n_datasets == 12


def test_friedman_reports_mean_ranks_and_kendalls_w():
    result = friedman_test(strong_scores())
    assert set(result.mean_ranks) == {"none", "standard", "robust"}
    # The baseline should rank worst when the others genuinely beat it.
    assert result.mean_ranks["none"] < result.mean_ranks["standard"]
    assert 0.0 <= result.kendalls_w <= 1.0


def test_friedman_finds_nothing_when_there_is_nothing():
    """A null must be reported as a null — nulls are findings."""
    assert not friedman_test(null_scores()).significant


def test_friedman_needs_three_levels_and_says_so():
    with pytest.raises(StatisticsError, match="at least 3 levels"):
        friedman_test({"none": [0.1, 0.2], "standard": [0.3, 0.4]})


def test_friedman_refuses_unpaired_data():
    with pytest.raises(StatisticsError, match="one score per dataset"):
        friedman_test({"a": [0.1, 0.2], "b": [0.3], "c": [0.4, 0.5]})


def test_friedman_refuses_a_single_dataset():
    with pytest.raises(StatisticsError, match="at least 2 datasets"):
        friedman_test({"a": [0.1], "b": [0.2], "c": [0.3]})


# --- effect sizes: mandatory, never optional -----------------------------

def test_effect_size_measures_magnitude_and_direction():
    effect = effect_size([0.85] * 10, [0.80] * 10, higher_is_better=True)
    assert effect.median_difference == pytest.approx(0.05)
    assert effect.n_improved == 10
    assert effect.n_worsened == 0


def test_effect_size_inverts_for_rmse_where_lower_is_better():
    """A smaller RMSE is an improvement, so the sign must flip."""
    effect = effect_size([3.0] * 10, [4.0] * 10, higher_is_better=False)
    assert effect.median_difference == pytest.approx(1.0)
    assert effect.n_improved == 10


def test_effect_size_confidence_interval_brackets_the_estimate():
    rng = np.random.default_rng(SEED)
    treatment = list(rng.normal(0.85, 0.02, size=20))
    baseline = list(rng.normal(0.80, 0.02, size=20))
    effect = effect_size(treatment, baseline, seed=SEED)
    assert effect.ci_low <= effect.median_difference <= effect.ci_high
    assert effect.ci_level == 0.95


def test_a_null_effect_has_a_ci_crossing_zero():
    """This is how a reader tells "no effect" from "unmeasured"."""
    rng = np.random.default_rng(SEED)
    values = list(rng.normal(0.80, 0.02, size=20))
    effect = effect_size(values, values, seed=SEED)
    assert effect.crosses_zero


def test_a_clear_effect_has_a_ci_excluding_zero():
    effect = effect_size([0.90] * 15, [0.80] * 15, seed=SEED)
    assert not effect.crosses_zero


def test_effect_size_refuses_mismatched_or_empty_pairs():
    with pytest.raises(StatisticsError, match="must match in length"):
        effect_size([0.1, 0.2], [0.3])
    with pytest.raises(StatisticsError, match="zero pairs"):
        effect_size([], [])


# --- baseline-referenced Wilcoxon ----------------------------------------

def test_comparisons_are_against_the_baseline_not_all_pairs():
    """OFAT asks "does this beat the baseline?", so 4 levels give 3
    comparisons, not 6 all-pairs. That more than halves the correction."""
    scores = {**strong_scores(), "minmax": [s + 0.03 for s in strong_scores()["none"]]}
    comparisons = wilcoxon_against_baseline(scores, "none")
    assert len(comparisons) == 3
    assert {c.level for c in comparisons} == {"standard", "robust", "minmax"}
    assert all(c.baseline == "none" for c in comparisons)


def test_every_comparison_carries_an_effect_size():
    """Section 7.2's requirement, enforced structurally."""
    for comparison in wilcoxon_against_baseline(strong_scores(), "none"):
        assert isinstance(comparison.effect, EffectSize)
        assert comparison.effect.n_pairs == 12


def test_a_comparison_cannot_be_built_without_an_effect_size():
    """The dataclass has no default for `effect`, so a bare p-value is
    not expressible in this API."""
    with pytest.raises(TypeError):
        Comparison(level="x", baseline="none", statistic=1.0,
                   p_value=0.01, p_adjusted=0.03, rejected=True)


def test_a_real_improvement_is_detected_and_adjusted():
    comparisons = {c.level: c for c in wilcoxon_against_baseline(strong_scores(), "none")}
    assert comparisons["standard"].rejected
    assert comparisons["standard"].p_adjusted >= comparisons["standard"].p_value


def test_identical_levels_are_reported_not_crashed():
    """Wilcoxon is undefined when every pair ties; that is a result."""
    base = [0.8, 0.79, 0.81, 0.78, 0.82, 0.80]
    scores = {"none": base, "standard": list(base), "robust": [b + 0.05 for b in base]}
    comparisons = {c.level: c for c in wilcoxon_against_baseline(scores, "none")}
    tied = comparisons["standard"]
    assert math.isnan(tied.statistic)
    assert not tied.rejected
    assert "identical to the baseline" in tied.note


def test_wilcoxon_refuses_a_missing_or_lonely_baseline():
    with pytest.raises(StatisticsError, match="not among the levels"):
        wilcoxon_against_baseline(strong_scores(), "bogus")
    with pytest.raises(StatisticsError, match="only the baseline"):
        wilcoxon_against_baseline({"none": [0.1, 0.2]}, "none")


def test_wilcoxon_refuses_unpaired_levels():
    scores = {"none": [0.1, 0.2, 0.3], "standard": [0.4, 0.5]}
    with pytest.raises(StatisticsError, match="but baseline has"):
        wilcoxon_against_baseline(scores, "none")


# --- the full framework ---------------------------------------------------

def test_post_hoc_runs_only_when_friedman_licenses_it():
    result = compare_levels(strong_scores(), "none")
    assert result.friedman.significant
    assert len(result.comparisons) == 2
    assert result.correction == "holm"
    assert result.skipped_reason == ""


def test_a_global_null_skips_post_hoc_and_explains_why():
    """Reporting pairwise results the global test does not license would
    inflate the error rate."""
    result = compare_levels(null_scores(), "none")
    assert not result.friedman.significant
    assert result.comparisons == []
    assert "not licensed" in result.skipped_reason
    assert "Friedman p" in result.skipped_reason


def test_rmse_direction_is_honoured_end_to_end():
    """Lower RMSE is better, so a reduction must read as an improvement."""
    base = [4.0, 4.2, 3.8, 4.1, 3.9, 4.3, 4.0, 3.7, 4.2, 4.1, 3.8, 4.0]
    scores = {
        "none": base,
        "standard": [b - 0.5 for b in base],
        "robust": [b - 0.4 for b in base],
    }
    result = compare_levels(scores, "none", higher_is_better=False)
    improved = {c.level: c for c in result.comparisons}
    assert improved["standard"].effect.median_difference > 0
    assert improved["standard"].effect.n_improved == 12


def test_the_selection_sweep_shape_gives_four_comparisons_not_ten():
    """5 selection levels: baseline-referenced is 4 tests, all-pairs 10."""
    base = [0.80, 0.78, 0.82, 0.76, 0.81, 0.79, 0.77, 0.83, 0.75, 0.80]
    scores = {
        "none": base,
        "chi2": [b + 0.012 for b in base],
        "mutual_info": [b + 0.021 for b in base],
        "rfe": [b + 0.018 for b in base],
        "lasso": [b + 0.009 for b in base],
    }
    result = compare_levels(scores, "none")
    assert len(result.comparisons) == 4
