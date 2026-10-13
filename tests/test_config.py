# Confirms the config contract: Chapter 1's components/baselines/models,
# the methodology's k=10 default, and that a declared 70/15/15 split is
# never marked applied under k-fold.
import pytest
import yaml

from piplab.config import (
    BASELINES,
    DEFAULT_CV_K,
    DEFAULT_INNER_K,
    MODELS,
    TECHNIQUES,
    ConfigError,
    load_config,
    parse_config,
)

GOOD = {
    "name": "t", "vary": "scaling", "techniques": ["none", "standard"],
    "datasets": ["iris", "wine"], "models": ["logreg"], "seed": 671,
    "outputs_dir": "outputs/",
}


def test_chapter1_baselines_are_the_first_option_of_each_component():
    # Chapter 1 names these as the baselines OFAT varies against.
    assert BASELINES == {
        "encoding": "onehot",
        "imputation": "drop_rows",
        "scaling": "none",
        "selection": "none",
        "tuning": "none",
    }


def test_the_five_chapter1_components_and_four_models_are_present():
    assert set(TECHNIQUES) == {"encoding", "imputation", "scaling", "selection", "tuning"}
    assert MODELS == ("logreg", "random_forest", "svm", "gradient_boosting")
    # Row deletion is Chapter 1's imputation baseline and is absent from
    # both existing engines; it must exist here.
    assert "drop_rows" in TECHNIQUES["imputation"]
    # Selection and tuning are the two components neither engine has.
    assert set(TECHNIQUES["selection"]) == {"none", "chi2", "mutual_info", "rfe", "lasso"}
    assert set(TECHNIQUES["tuning"]) == {"none", "random", "grid", "bayesian"}


def test_defaults_match_the_methodology_analysis():
    cfg = parse_config(dict(GOOD))
    # methodology section 6.3 selects stratified k-fold at k = 10.
    assert cfg.evaluation.cv_k == DEFAULT_CV_K == 10
    assert cfg.evaluation.protocol == "kfold"
    assert cfg.evaluation.inner_k == DEFAULT_INNER_K == 3


def test_unvaried_components_fall_back_to_chapter1_baselines():
    cfg = parse_config(dict(GOOD))
    # scaling is varied, so it must not appear in the baseline...
    assert "scaling" not in cfg.baseline
    # ...and every other component sits at its Chapter 1 baseline.
    assert cfg.baseline == {
        "encoding": "onehot", "imputation": "drop_rows",
        "selection": "none", "tuning": "none",
    }


def test_baseline_may_be_overridden_per_component():
    cfg = parse_config(dict(GOOD, baseline={"imputation": "median"}))
    assert cfg.baseline["imputation"] == "median"
    assert cfg.baseline["selection"] == "none"


def test_baseline_cannot_pin_the_component_being_varied():
    with pytest.raises(ConfigError, match="being varied"):
        parse_config(dict(GOOD, baseline={"scaling": "robust"}))


def test_run_count_is_datasets_x_models_x_techniques():
    cfg = parse_config(dict(GOOD, datasets=["a", "b", "c"], models=["logreg", "svm"],
                            techniques=["none", "standard", "robust"]))
    assert cfg.run_count() == 3 * 2 * 3 == 18


# --- the split, per PLAN.md section 5 -------------------------------------

def test_a_declared_split_is_NOT_applied_under_kfold():
    cfg = parse_config(dict(GOOD))
    split = cfg.evaluation.split
    # The ratios are recorded because the dashboard exposes them...
    assert (split.train, split.val, split.test) == (0.70, 0.15, 0.15)
    # ...but k-fold is what ran, so nothing may present them as applied.
    assert split.applied is False
    assert "section 6.1" in split.note


def test_split_is_applied_under_repeated_holdout():
    cfg = parse_config(dict(GOOD, evaluation={"protocol": "repeated_holdout"}))
    assert cfg.evaluation.split.applied is True
    assert cfg.evaluation.holdout_repeats == 30
    assert "Applied" in cfg.evaluation.split.note


def test_custom_split_ratios_are_accepted_and_recorded():
    cfg = parse_config(dict(GOOD, evaluation={"split": {"train": 0.6, "val": 0.2, "test": 0.2}}))
    assert (cfg.evaluation.split.train, cfg.evaluation.split.val) == (0.6, 0.2)


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ConfigError, match="sum to 1.0"):
        parse_config(dict(GOOD, evaluation={"split": {"train": 0.7, "val": 0.2, "test": 0.2}}))


# --- rejections -----------------------------------------------------------

@pytest.mark.parametrize("bad,match", [
    ({"vary": "encodings"}, "vary"),
    ({"vary": "scaling", "techniques": ["mean"]}, "invalid for vary"),
    ({"models": ["xgboost"]}, "unknown model"),
    ({"models": []}, "non-empty"),
    ({"datasets": []}, "non-empty"),
    ({"seed": -1}, ">= 0"),
    ({"name": ""}, "non-empty"),
])
def test_rejects_invalid_values(bad, match):
    with pytest.raises(ConfigError, match=match):
        parse_config(dict(GOOD, **bad))


def test_rejects_an_unknown_top_level_field():
    with pytest.raises(ConfigError, match="metric"):
        parse_config(dict(GOOD, metric="roc_auc"))


def test_rejects_an_unknown_evaluation_field():
    with pytest.raises(ConfigError, match="evaluation.folds"):
        parse_config(dict(GOOD, evaluation={"folds": 5}))


def test_rejects_a_repeated_technique():
    with pytest.raises(ConfigError, match="repeat"):
        parse_config(dict(GOOD, techniques=["none", "none"]))


def test_rejects_cv_k_below_two():
    with pytest.raises(ConfigError, match="cv_k"):
        parse_config(dict(GOOD, evaluation={"cv_k": 1}))


# --- the real example configs --------------------------------------------

@pytest.mark.parametrize("filename,expected_runs", [
    ("smoke.yaml", 2 * 2 * 1),
    ("ofat_selection.yaml", 6 * 4 * 5),
    ("ofat_tuning.yaml", 3 * 2 * 3),
    ("holdout_large.yaml", 4 * 2 * 4),
])
def test_shipped_configs_load_and_expand_correctly(filename, expected_runs):
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    assert cfg.run_count() == expected_runs
    assert cfg.is_ofat()


def test_the_selection_sweep_uses_chapter1_baselines_for_everything_else():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ofat_selection.yaml")
    assert cfg.vary == "selection"
    assert cfg.baseline == {
        "encoding": "onehot", "imputation": "drop_rows",
        "scaling": "none", "tuning": "none",
    }
    assert cfg.evaluation.cv_k == 10


def test_missing_config_file_names_the_path():
    with pytest.raises(ConfigError, match="not found"):
        load_config("no_such_config.yaml")


def test_the_encoding_sweep_config_loads_and_expands_correctly():
    """AMENDMENT (PLAN.md 2b): encoding as a sixth factor."""
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "ofat_encoding.yaml")
    assert cfg.vary == "encoding"
    assert cfg.techniques == ("onehot", "ordinal", "target")
    # 8 datasets with categorical features x 4 models x 3 levels.
    assert cfg.run_count() == 96
    # Everything else sits at its Chapter 1 baseline.
    assert cfg.baseline == {
        "imputation": "drop_rows", "scaling": "none",
        "selection": "none", "tuning": "none",
    }


def test_smoke_config_is_offline_only_so_tests_never_need_a_network():
    """configs/smoke.yaml must use only sklearn-bundled datasets."""
    from pathlib import Path
    from piplab.datasets import offline_names
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    assert set(cfg.datasets) <= set(offline_names())


# --- synthetic missingness: the inject block -----------------------------


def test_a_config_without_inject_requests_no_injection():
    """The default must stay "use the data exactly as it arrived"."""
    config = parse_config(dict(GOOD))
    assert config.inject_rate is None
    assert config.inject_mechanism is None


def test_inject_is_parsed_when_given():
    config = parse_config({**GOOD, "inject": {"rate": 0.15, "mechanism": "mcar"}})
    assert config.inject_rate == 0.15
    assert config.inject_mechanism == "mcar"


@pytest.mark.parametrize("partial", [{"rate": 0.15}, {"mechanism": "mcar"}])
def test_half_an_inject_block_is_refused(partial):
    """Defaulting the missing half would silently change the experiment:
    neither a rate nor a mechanism has a safe assumption."""
    with pytest.raises(ConfigError, match="BOTH rate and mechanism"):
        parse_config({**GOOD, "inject": partial})


@pytest.mark.parametrize("rate", [0, 1, 1.5, -0.2, "0.15", True])
def test_an_impossible_inject_rate_is_refused(rate):
    with pytest.raises(ConfigError, match="inject.rate"):
        parse_config({**GOOD, "inject": {"rate": rate, "mechanism": "mcar"}})


def test_an_unknown_mechanism_is_refused():
    with pytest.raises(ConfigError, match="inject.mechanism"):
        parse_config({**GOOD, "inject": {"rate": 0.15, "mechanism": "mnar"}})


def test_an_unknown_inject_field_is_refused():
    """A typo must fail loudly rather than be silently ignored."""
    with pytest.raises(ConfigError, match="inject."):
        parse_config({
            **GOOD,
            "inject": {"rate": 0.15, "mechanism": "mcar", "ratio": 0.3},
        })
