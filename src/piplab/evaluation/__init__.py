# piplab.evaluation: how a configuration is scored.
#
# Stratified k-fold at k=10 is the primary protocol (methodology 6.3).
# Tuning nests an inner CV inside each outer training fold. Repeated
# holdout exists as a gated secondary protocol and refuses datasets
# where a 15% test set would be too small to mean anything.
from piplab.evaluation.metrics import (
    HIGHER_IS_BETTER,
    metrics_for,
    scoring_for,
    summarise,
)
from piplab.evaluation.protocols import (
    EvaluationResult,
    ProtocolError,
    apply_row_deletion,
    check_holdout_eligible,
    evaluate,
    fold_hash,
    fold_seed,
    make_splitter,
)

__all__ = [
    "HIGHER_IS_BETTER", "metrics_for", "scoring_for", "summarise",
    "EvaluationResult", "ProtocolError", "apply_row_deletion",
    "check_holdout_eligible", "evaluate", "fold_hash", "fold_seed",
    "make_splitter",
]
