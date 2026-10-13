# piplab.statistics: Friedman + baseline-referenced Wilcoxon with Holm.
#
# Methodology section 7.2's selected framework. No p-value is ever
# returned without an effect size: a Comparison cannot be constructed
# without one, so the section 7.2 requirement is structural rather than
# a reporting convention.
from piplab.statistics.tests import (
    ALPHA,
    Comparison,
    EffectSize,
    FriedmanResult,
    PostHocResult,
    StatisticsError,
    compare_levels,
    effect_size,
    friedman_test,
    holm_adjust,
    wilcoxon_against_baseline,
)

__all__ = [
    "ALPHA", "Comparison", "EffectSize", "FriedmanResult", "PostHocResult",
    "StatisticsError", "compare_levels", "effect_size", "friedman_test",
    "holm_adjust", "wilcoxon_against_baseline",
]
