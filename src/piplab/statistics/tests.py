# tests.py: the statistical framework methodology_analysis_Colile.docx
# section 7.2 selected -- Friedman for the global comparison, then
# Wilcoxon signed-rank post-hoc with Holm correction.
#
# Two deliberate departures from how the existing B1/B2 artefacts do it,
# both argued in pipeline-lab/docs/PLAN.md section 5:
#
#   * Wilcoxon, not Nemenyi. Nemenyi appears ZERO times in the March
#     methodology. It also compares all pairs, spending power on
#     comparisons this OFAT design never asks (minmax vs robust), and it
#     collapses to mean ranks, discarding the per-dataset pairing that
#     SQ4's mechanistic layer needs.
#   * Holm, not Bonferroni. Section 7.2's body permits either ("the
#     Bonferroni or Holm procedure"); Holm is uniformly more powerful at
#     the same family-wise error guarantee, is what Demsar (2006)
#     recommends, and is already what B2's reported results use.
#
# No p-value is returned without an effect size. Section 7.2 identifies
# the real weakness of rank tests -- "a 10-percentage-point difference
# and a 0.1-percentage-point difference contribute equally to the rank"
# -- and requires practical effect size as the remedy. The API enforces
# this structurally: a Comparison cannot exist without one.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

ALPHA = 0.05
# Bootstrap resamples for the effect-size confidence interval. The
# paired differences are rarely normal, so the CI is bootstrapped rather
# than assuming a t-distribution.
BOOTSTRAP_RESAMPLES = 10000


class StatisticsError(ValueError):
    """Raised when a test cannot honestly be run on the data given."""


@dataclass(frozen=True)
class EffectSize:
    """The practical magnitude of a difference, not just its significance."""

    median_difference: float
    mean_difference: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_pairs: int
    n_improved: int
    n_worsened: int
    n_tied: int

    @property
    def crosses_zero(self) -> bool:
        """True when the CI spans zero, i.e. the direction is not settled."""
        return self.ci_low <= 0.0 <= self.ci_high


@dataclass(frozen=True)
class Comparison:
    """One level against the baseline: p-value AND effect size, always."""

    level: str
    baseline: str
    statistic: float
    p_value: float
    p_adjusted: float
    rejected: bool
    effect: EffectSize
    note: str = ""


@dataclass
class FriedmanResult:
    """The global test: do these levels differ at all?"""

    statistic: float
    p_value: float
    kendalls_w: float
    n_datasets: int
    n_levels: int
    mean_ranks: dict[str, float] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        """True when the global test licenses post-hoc comparisons."""
        return self.p_value < ALPHA


@dataclass
class PostHocResult:
    """Friedman plus, when licensed, the baseline-referenced comparisons."""

    friedman: FriedmanResult
    comparisons: list[Comparison] = field(default_factory=list)
    correction: str = "holm"
    alpha: float = ALPHA
    skipped_reason: str = ""


def friedman_test(scores: dict[str, list[float]]) -> FriedmanResult:
    """Run the global Friedman test across a component's levels.

    Inputs: scores - {level: [score per dataset]}; every level must have
    one score per dataset, in the same dataset order, because the test
    is on paired measurements.
    Output: a FriedmanResult including mean ranks and Kendall's W (the
    effect size for the global test).
    Raises: StatisticsError when the data cannot support the test.
    """
    levels = list(scores)
    if len(levels) < 3:
        raise StatisticsError(
            f"Friedman needs at least 3 levels, got {len(levels)}: {levels}. "
            "For exactly 2, use a single Wilcoxon signed-rank test."
        )
    lengths = {len(v) for v in scores.values()}
    if len(lengths) != 1:
        raise StatisticsError(
            f"every level must have one score per dataset; got lengths {sorted(lengths)}"
        )
    n = lengths.pop()
    if n < 2:
        raise StatisticsError(f"Friedman needs at least 2 datasets, got {n}")

    columns = [np.asarray(scores[level], dtype=float) for level in levels]
    statistic, p_value = stats.friedmanchisquare(*columns)

    matrix = np.column_stack(columns)
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    mean_ranks = {level: float(ranks[:, i].mean()) for i, level in enumerate(levels)}
    k = len(levels)
    # Kendall's W: chi2 / (n * (k - 1)), the standard normalisation.
    kendalls_w = float(statistic / (n * (k - 1))) if n * (k - 1) else 0.0

    return FriedmanResult(
        statistic=float(statistic),
        p_value=float(p_value),
        kendalls_w=kendalls_w,
        n_datasets=int(n),
        n_levels=k,
        mean_ranks=mean_ranks,
    )


def effect_size(treatment: list[float], baseline: list[float], *,
                higher_is_better: bool = True,
                resamples: int = BOOTSTRAP_RESAMPLES,
                seed: int = 671) -> EffectSize:
    """Measure how large a difference is, not merely whether it is significant.

    Inputs: treatment and baseline - paired per-dataset scores;
    higher_is_better - orients the difference so a positive value always
    means "treatment is better" (RMSE is inverted); resamples and seed -
    the bootstrap.
    Output: an EffectSize with the median difference and a bootstrapped
    95% CI.
    Raises: StatisticsError on mismatched or empty input.
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise StatisticsError(f"paired scores must match in length: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise StatisticsError("cannot measure an effect from zero pairs")

    differences = (a - b) if higher_is_better else (b - a)
    rng = np.random.default_rng(seed)
    if a.size > 1:
        draws = rng.choice(differences, size=(resamples, differences.size), replace=True)
        medians = np.median(draws, axis=1)
        ci_low, ci_high = (float(x) for x in np.percentile(medians, [2.5, 97.5]))
    else:
        ci_low = ci_high = float(differences[0])

    return EffectSize(
        median_difference=round(float(np.median(differences)), 6),
        mean_difference=round(float(differences.mean()), 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        ci_level=0.95,
        n_pairs=int(differences.size),
        n_improved=int((differences > 0).sum()),
        n_worsened=int((differences < 0).sum()),
        n_tied=int((differences == 0).sum()),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply the Holm step-down correction.

    Uniformly more powerful than Bonferroni at the same family-wise
    error rate: the smallest p is tested against alpha/m, the next
    against alpha/(m-1), and so on. Implemented directly rather than
    adding statsmodels for six lines.
    Inputs: p_values - the raw p-values, in any order.
    Output: adjusted p-values in the SAME order as the input, each
    monotonically non-decreasing in rank so an adjusted value never
    undercuts a smaller one.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * p_values[index]
        running = max(running, min(value, 1.0))
        adjusted[index] = running
    return adjusted


def wilcoxon_against_baseline(
    scores: dict[str, list[float]],
    baseline: str,
    *,
    higher_is_better: bool = True,
    alpha: float = ALPHA,
    seed: int = 671,
) -> list[Comparison]:
    """Compare every level against the baseline, Holm-corrected.

    Baseline-referenced rather than all-pairs, because that is what OFAT
    asks: "does standard scaling beat NO scaling?", not "does minmax beat
    robust?". For 5 selection levels this is 4 comparisons rather than
    10, which more than halves the correction burden.

    Inputs: scores - {level: [score per dataset]}; baseline - the level
    to compare against; higher_is_better; alpha; seed for the bootstrap.
    Output: one Comparison per non-baseline level, each carrying an
    effect size. Never a bare p-value.
    Raises: StatisticsError when the baseline is absent or unpaired.
    """
    if baseline not in scores:
        raise StatisticsError(
            f"baseline {baseline!r} is not among the levels {sorted(scores)}"
        )
    reference = scores[baseline]
    levels = [level for level in scores if level != baseline]
    if not levels:
        raise StatisticsError("nothing to compare: only the baseline was supplied")

    raw: list[tuple[str, float, float, str]] = []
    for level in levels:
        values = scores[level]
        if len(values) != len(reference):
            raise StatisticsError(
                f"level {level!r} has {len(values)} scores but baseline has {len(reference)}"
            )
        differences = np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)
        if np.all(differences == 0):
            # Wilcoxon is undefined when every pair is tied; that is a
            # real result (identical performance), not an error.
            raw.append((level, float("nan"), 1.0, "identical to the baseline on every dataset"))
            continue
        statistic, p_value = stats.wilcoxon(values, reference, zero_method="wilcox")
        raw.append((level, float(statistic), float(p_value), ""))

    adjusted = holm_adjust([p for _, _, p, _ in raw])
    comparisons = []
    for (level, statistic, p_value, note), p_adj in zip(raw, adjusted):
        comparisons.append(Comparison(
            level=level,
            baseline=baseline,
            statistic=statistic,
            p_value=p_value,
            p_adjusted=p_adj,
            rejected=bool(p_adj < alpha),
            effect=effect_size(scores[level], reference,
                               higher_is_better=higher_is_better, seed=seed),
            note=note,
        ))
    return comparisons


def compare_levels(
    scores: dict[str, list[float]],
    baseline: str,
    *,
    higher_is_better: bool = True,
    alpha: float = ALPHA,
    seed: int = 671,
) -> PostHocResult:
    """Run the full framework: Friedman, then post-hoc only if licensed.

    Post-hoc comparisons are only made when the global test is
    significant, which is what controls the error rate across the family
    of comparisons (methodology section 7.2, following Demsar 2006).

    Inputs: scores - {level: [score per dataset]}; baseline - the
    Chapter 1 baseline for this component; higher_is_better - False for
    RMSE; alpha; seed.
    Output: a PostHocResult. When Friedman is not significant,
    `comparisons` is empty and `skipped_reason` says why -- rather than
    reporting pairwise results the global test does not license.
    """
    friedman = friedman_test(scores)
    if not friedman.significant:
        return PostHocResult(
            friedman=friedman,
            comparisons=[],
            alpha=alpha,
            skipped_reason=(
                f"Friedman p = {friedman.p_value:.4f} >= {alpha}: no global difference, "
                "so pairwise comparisons are not licensed."
            ),
        )
    return PostHocResult(
        friedman=friedman,
        comparisons=wilcoxon_against_baseline(
            scores, baseline, higher_is_better=higher_is_better, alpha=alpha, seed=seed
        ),
        alpha=alpha,
    )
