# config.py: parse and validate a Pipeline Lab experiment YAML.
# Configs are contracts: unknown fields, missing required fields and bad
# values are all errors that name the offending field. The five varied
# components and their baselines come from Chapter_1_Colile.docx (31 Mar
# 2026) Table "Pipeline Component / Candidate Techniques"; the evaluation
# protocol comes from methodology_analysis_Colile.docx section 6.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Chapter 1 Table 1. The FIRST entry of each list is that component's
# baseline, which OFAT holds the other components at.
TECHNIQUES: dict[str, tuple[str, ...]] = {
    # AMENDMENT 2026-10-09: encoding promoted to a sixth factor
    # (PLAN.md section 2b). onehot stays the baseline, so existing
    # configs behave identically.
    "encoding": ("onehot", "ordinal", "target"),
    "imputation": ("drop_rows", "mean", "median", "knn"),
    "scaling": ("none", "standard", "minmax", "robust"),
    "selection": ("none", "chi2", "mutual_info", "rfe", "lasso"),
    "tuning": ("none", "random", "grid", "bayesian"),
}
# Model type has no baseline: Chapter 1 compares all four throughout.
MODELS: tuple[str, ...] = ("logreg", "random_forest", "svm", "gradient_boosting")

COMPONENTS: tuple[str, ...] = tuple(TECHNIQUES)
BASELINES: dict[str, str] = {name: opts[0] for name, opts in TECHNIQUES.items()}

# methodology_analysis section 6.3 selects stratified k-fold at k = 10;
# section 6.1 rejects the single train/test split.
DEFAULT_CV_K = 10
# Nested CV for tuning (PLAN.md section 5): the inner loop selects
# hyperparameters, the outer loop scores the winner on unseen data.
DEFAULT_INNER_K = 3
PROTOCOLS = ("kfold", "repeated_holdout")
# Repeated holdout is only defensible where 15% of the data is a
# meaningful test set (PLAN.md section 5b).
HOLDOUT_MIN_ROWS = 4000
DEFAULT_HOLDOUT_REPEATS = 30

# Share of features the k-based selectors (chi2, mutual_info, rfe) keep.
# Declared here, not imported from components.selection, because this
# module must stay free of sklearn: the dashboard reads it under an
# interpreter that deliberately lacks the engine's dependencies.
# components.selection re-exports this value and a test pins them equal.
DEFAULT_K_FRACTION = 0.5

# Synthetic missingness mechanisms, matching the B2 artefact so the two
# studies stay comparable. Declared here, not imported from
# datasets.missingness, to keep this module free of pandas/numpy.
MECHANISMS = ("mcar", "mar")


class ConfigError(ValueError):
    """Raised when an experiment config is malformed; message names the field."""


@dataclass(frozen=True)
class SplitConfig:
    """The declared train/val/test ratios.

    Recorded on every run because the dashboard exposes them, but only
    *applied* under the repeated_holdout protocol. Under kfold they are
    metadata and `applied` is False, with the reason in `note`.
    """

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    applied: bool = False
    note: str = (
        "Not applied: this run used stratified k-fold cross-validation. "
        "methodology_analysis_Colile.docx section 6.1 rejects the single "
        "train/test split as insufficient for statistically reliable "
        "comparisons (Lones et al. 2024; Arlot and Celisse 2010)."
    )


@dataclass(frozen=True)
class EvaluationConfig:
    """How one configuration is scored."""

    protocol: str = "kfold"
    cv_k: int = DEFAULT_CV_K
    inner_k: int = DEFAULT_INNER_K
    holdout_repeats: int = DEFAULT_HOLDOUT_REPEATS
    # Share of features the k-based selectors keep. Recorded per run so
    # "selection hurts" can be told apart from "keeping half hurts".
    k_fraction: float = DEFAULT_K_FRACTION
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True)
class ExperimentConfig:
    """One validated Pipeline Lab experiment.

    `vary` names the component being swept and `techniques` its levels;
    every other component sits at `baseline`. That is the OFAT design
    Chapter 1 section 1.7.1 specifies.
    """

    name: str
    vary: str
    techniques: tuple[str, ...]
    baseline: dict[str, str]
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    seed: int
    evaluation: EvaluationConfig
    # Synthetic missingness, for the imputation arm. None means the
    # dataset is used exactly as it arrived.
    inject_rate: float | None = None
    inject_mechanism: str | None = None
    outputs_dir: str = "outputs/"

    def run_count(self) -> int:
        """Return how many (dataset, model, technique) runs this config expands to.

        Inputs: none.
        Output: the number of individual runs, so a caller can show the
        cost before launching.
        """
        return len(self.datasets) * len(self.models) * len(self.techniques)

    def is_ofat(self) -> bool:
        """Say whether this config keeps the OFAT guarantee.

        Inputs: none.
        Output: True when exactly one component varies, which is what
        makes an observed effect attributable to that component.
        """
        return len(self.techniques) >= 1 and self.vary in COMPONENTS


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML.

    Inputs: path - filesystem path to the experiment YAML.
    Output: a validated, frozen ExperimentConfig.
    Raises: ConfigError naming the offending field on any violation.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return parse_config(raw)


def parse_config(raw: dict) -> ExperimentConfig:
    """Validate an already-parsed config mapping.

    Inputs: raw - the mapping from YAML (or from the dashboard's API).
    Output: a validated, frozen ExperimentConfig.
    Raises: ConfigError naming the offending field.
    """
    _reject_unknown(raw, {
        "name", "vary", "techniques", "baseline", "datasets", "models",
        "seed", "evaluation", "outputs_dir", "inject",
    }, "")

    name = _require_text(raw, "name")
    vary = _require_choice(raw, "vary", COMPONENTS)
    techniques = _require_techniques(raw, vary)
    baseline = _require_baseline(raw, vary)
    datasets = _require_str_tuple(raw, "datasets")
    models = _require_models(raw)
    seed = _require_int(raw, "seed", minimum=0)
    evaluation = _parse_evaluation(raw.get("evaluation") or {})
    outputs_dir = _require_text(raw, "outputs_dir")
    inject_rate, inject_mechanism = _parse_inject(raw.get("inject") or {})

    return ExperimentConfig(
        name=name, vary=vary, techniques=techniques, baseline=baseline,
        datasets=datasets, models=models, seed=seed,
        evaluation=evaluation, outputs_dir=outputs_dir,
        inject_rate=inject_rate, inject_mechanism=inject_mechanism,
    )


def _parse_inject(raw: dict) -> tuple[float | None, str | None]:
    """Validate the optional `inject` block.

    Synthetic missingness exists because the imputation arm cannot be
    tested on naturally missing data alone: only 4 of 19 datasets have
    any, so the rest tie exactly and the effective n is 4 against a Holm
    bar needing 7. Injecting makes every dataset informative.

    Inputs: raw - the `inject` mapping, possibly empty.
    Output: (rate, mechanism), both None when no injection is asked for.
    Raises: ConfigError naming the offending field. Giving one of the
    pair without the other is an error rather than a default, because a
    guessed rate or mechanism would silently change what the experiment
    measures.
    """
    if not raw:
        return None, None
    _reject_unknown(raw, {"rate", "mechanism"}, "inject.")

    if "rate" not in raw or "mechanism" not in raw:
        raise ConfigError(
            "config block inject needs BOTH rate and mechanism; got "
            f"{sorted(raw)}. Neither has a safe default: a guessed rate or "
            "mechanism would change what the experiment measures."
        )
    rate = raw["rate"]
    is_number = isinstance(rate, (int, float)) and not isinstance(rate, bool)
    if not is_number or not 0 < float(rate) < 1:
        raise ConfigError(
            "config field inject.rate must be a number strictly between 0 "
            f"and 1, got {rate!r}"
        )
    mechanism = raw["mechanism"]
    if mechanism not in MECHANISMS:
        raise ConfigError(
            f"config field inject.mechanism must be one of {list(MECHANISMS)}, "
            f"got {mechanism!r}"
        )
    return float(rate), mechanism


def _parse_evaluation(raw: dict) -> EvaluationConfig:
    """Validate the evaluation block.

    Inputs: raw - the `evaluation` mapping, possibly empty.
    Output: an EvaluationConfig with documented defaults filled in.
    Raises: ConfigError naming the offending field.
    """
    if not isinstance(raw, dict):
        raise ConfigError("config field evaluation must be a mapping")
    _reject_unknown(
        raw,
        {"protocol", "cv_k", "inner_k", "holdout_repeats", "k_fraction", "split"},
        "evaluation.",
    )

    protocol = raw.get("protocol", "kfold")
    if protocol not in PROTOCOLS:
        raise ConfigError(
            f"config field evaluation.protocol must be one of {PROTOCOLS}, got {protocol!r}"
        )
    cv_k = _require_int(raw, "cv_k", minimum=2, default=DEFAULT_CV_K, where="evaluation.")
    inner_k = _require_int(raw, "inner_k", minimum=2, default=DEFAULT_INNER_K, where="evaluation.")
    repeats = _require_int(
        raw, "holdout_repeats", minimum=2, default=DEFAULT_HOLDOUT_REPEATS, where="evaluation."
    )
    k_fraction = raw.get("k_fraction", DEFAULT_K_FRACTION)
    valid_number = (
        isinstance(k_fraction, (int, float)) and not isinstance(k_fraction, bool)
    )
    if not valid_number or not 0 < float(k_fraction) <= 1:
        raise ConfigError(
            "config field evaluation.k_fraction must be a number in (0, 1], "
            f"got {k_fraction!r}"
        )
    split = _parse_split(raw.get("split") or {}, protocol)
    return EvaluationConfig(
        protocol=protocol, cv_k=cv_k, inner_k=inner_k,
        holdout_repeats=repeats, k_fraction=float(k_fraction), split=split,
    )


def _parse_split(raw: dict, protocol: str) -> SplitConfig:
    """Validate the declared split ratios.

    Inputs: raw - the `split` mapping, possibly empty; protocol - the
    validated protocol, which decides whether the split is applied.
    Output: a SplitConfig whose `applied` flag matches the protocol, so
    no caller can present a ratio as having produced a metric when the
    run used k-fold instead.
    Raises: ConfigError if the three ratios do not sum to 1.
    """
    if not isinstance(raw, dict):
        raise ConfigError("config field evaluation.split must be a mapping")
    _reject_unknown(raw, {"train", "val", "test"}, "evaluation.split.")

    defaults = SplitConfig()
    train = _require_fraction(raw, "train", defaults.train)
    val = _require_fraction(raw, "val", defaults.val)
    test = _require_fraction(raw, "test", defaults.test)
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ConfigError(
            "config fields evaluation.split.train/val/test must sum to 1.0, "
            f"got {train} + {val} + {test} = {train + val + test}"
        )
    if protocol == "repeated_holdout":
        return SplitConfig(
            train=train, val=val, test=test, applied=True,
            note=(
                "Applied: repeated holdout. Restricted to datasets with at "
                f"least {HOLDOUT_MIN_ROWS} rows, where a {test:.0%} test set "
                "is large enough to be meaningful; repeated so a variance "
                "estimate is still available."
            ),
        )
    return SplitConfig(train=train, val=val, test=test)


def _require_techniques(raw: dict, vary: str) -> tuple[str, ...]:
    """Validate the swept technique list against the varied component."""
    value = raw.get("techniques")
    if not isinstance(value, list) or not value:
        raise ConfigError("config field techniques must be a non-empty list")
    allowed = TECHNIQUES[vary]
    bad = [t for t in value if t not in allowed]
    if bad:
        raise ConfigError(
            f"config field techniques has values invalid for vary={vary!r}: {bad}; "
            f"allowed: {list(allowed)}"
        )
    if len(set(value)) != len(value):
        raise ConfigError("config field techniques must not repeat a technique")
    return tuple(value)


def _require_baseline(raw: dict, vary: str) -> dict[str, str]:
    """Validate the held-fixed baseline for every component except the varied one.

    Inputs: raw - the config mapping; vary - the component being swept.
    Output: a complete baseline dict, Chapter 1's defaults filling any
    component the config leaves out.
    Raises: ConfigError on an unknown component or technique, or if the
    baseline tries to pin the component that is being varied.
    """
    value = raw.get("baseline") or {}
    if not isinstance(value, dict):
        raise ConfigError("config field baseline must be a mapping")
    unknown = sorted(set(value) - set(COMPONENTS))
    if unknown:
        raise ConfigError(f"config field baseline has unknown component(s): {unknown}")
    if vary in value:
        raise ConfigError(
            f"config field baseline must not set {vary!r}: that component is the one "
            "being varied, so its levels belong in `techniques`"
        )
    resolved = {}
    for component in COMPONENTS:
        if component == vary:
            continue
        chosen = value.get(component, BASELINES[component])
        if chosen not in TECHNIQUES[component]:
            raise ConfigError(
                f"config field baseline.{component} must be one of "
                f"{list(TECHNIQUES[component])}, got {chosen!r}"
            )
        resolved[component] = chosen
    return resolved


def _require_models(raw: dict) -> tuple[str, ...]:
    """Validate the model list against Chapter 1's four model types."""
    value = raw.get("models")
    if not isinstance(value, list) or not value:
        raise ConfigError("config field models must be a non-empty list")
    bad = [m for m in value if m not in MODELS]
    if bad:
        raise ConfigError(f"config field models has unknown model(s): {bad}; allowed: {list(MODELS)}")
    if len(set(value)) != len(value):
        raise ConfigError("config field models must not repeat a model")
    return tuple(value)


def _require_str_tuple(raw: dict, field_name: str) -> tuple[str, ...]:
    value = raw.get(field_name)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"config field {field_name} must be a non-empty list")
    if not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"config field {field_name} must contain non-empty strings")
    return tuple(v.strip() for v in value)


def _require_text(raw: dict, field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"config field {field_name} must be a non-empty string")
    return value.strip()


def _require_choice(raw: dict, field_name: str, allowed) -> str:
    value = raw.get(field_name)
    if value not in allowed:
        raise ConfigError(
            f"config field {field_name} must be one of {list(allowed)}, got {value!r}"
        )
    return value


def _require_int(raw: dict, field_name: str, minimum: int, default=None, where: str = "") -> int:
    if field_name not in raw:
        if default is None:
            raise ConfigError(f"config field {where}{field_name} missing")
        return default
    value = raw[field_name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"config field {where}{field_name} must be an integer")
    if value < minimum:
        raise ConfigError(f"config field {where}{field_name} must be >= {minimum}, got {value}")
    return value


def _require_fraction(raw: dict, field_name: str, default: float) -> float:
    if field_name not in raw:
        return default
    value = raw[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"config field evaluation.split.{field_name} must be a number")
    if not 0.0 < float(value) < 1.0:
        raise ConfigError(
            f"config field evaluation.split.{field_name} must be in (0, 1), got {value}"
        )
    return float(value)


def _reject_unknown(mapping: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        joined = ", ".join(f"{where}{u}" for u in unknown)
        raise ConfigError(f"unknown config field(s): {joined}")
