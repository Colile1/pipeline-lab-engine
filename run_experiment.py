# run_experiment.py: the CLI that runs one config.
#
# Deliberately the same contract B1 and B2 use --
# `python run_experiment.py --config <yaml>` -- so the pipeline-lab
# dashboard's existing runner, run-status parser and output isolation
# work unchanged against this engine.
#
# Log lines follow the `Description: value` convention the dashboard's
# run-status parser reads, and the markers below are what its engine
# adapter will key on.
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from piplab.config import ConfigError, load_config  # noqa: E402
from piplab.datasets import load_dataset  # noqa: E402
from piplab.datasets.loaders import characterise  # noqa: E402
from piplab.datasets.missingness import inject_missing  # noqa: E402
from piplab.evaluation import ProtocolError, evaluate  # noqa: E402
from piplab.pipelines.compose import resolve_levels  # noqa: E402
from piplab.provenance import (  # noqa: E402
    ResourceMonitor,
    capture_environment,
    machine_fingerprint,
)
from piplab.storage import (  # noqa: E402
    encode_fold_scores,
    new_batch_id,
    new_run_id,
    open_db,
    utc_now,
    write_run,
)


def main(argv=None) -> int:
    """Run every combination in one config and record each result.

    Inputs: argv - command-line arguments (None reads sys.argv).
    Output: process exit code; 0 only when every run was recorded.
    """
    parser = argparse.ArgumentParser(description="Run one Pipeline Lab experiment config.")
    parser.add_argument("--config", required=True, help="path to an experiment YAML")
    parser.add_argument("--db", default=None, help="run-record database (default: <outputs_dir>/lab.db)")
    parser.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2

    environment = capture_environment()
    fingerprint = machine_fingerprint(environment)
    batch_id = new_batch_id()
    total = config.run_count()

    print(f"Config: {config.name}")
    print(f"Varying: {config.vary}")
    print(f"Techniques: {len(config.techniques)} ({', '.join(config.techniques)})")
    print(f"Datasets: {len(config.datasets)}")
    print(f"Models: {len(config.models)}")
    print(f"Protocol: {config.evaluation.protocol}")
    print(f"CV folds: {config.evaluation.cv_k}")
    print(f"Total runs: {total}")
    print(f"Batch: {batch_id}")
    print(f"Machine: {fingerprint}")
    if args.dry_run:
        print("Dry run: no execution.")
        return 0

    db_path = Path(args.db) if args.db else Path(config.outputs_dir) / "lab.db"
    config_text = Path(args.config).read_text(encoding="utf-8")
    completed = failed = skipped = 0

    with open_db(db_path) as db:
        for dataset in config.datasets:
            try:
                X, y, characteristics = load_dataset(dataset)
                if config.inject_rate is not None:
                    # Blank cells on purpose BEFORE anything else sees the
                    # data, so every component downstream meets the same
                    # frame it would have met had the gaps been natural.
                    # characteristics is re-derived so missing_pct reports
                    # what the run actually used.
                    X = inject_missing(
                        X, config.inject_rate, config.inject_mechanism,
                        config.seed,
                    )
                    characteristics = characterise(X, y, characteristics["task_type"])
            except Exception as exc:  # dataset-level failure affects every run on it
                for _ in config.models:
                    for _ in config.techniques:
                        _record_failure(db, batch_id, config, environment, fingerprint,
                                        dataset, "?", "?", exc, config_text)
                        failed += 1
                print(f"Dataset failed: {dataset} ({type(exc).__name__}: {exc})")
                continue

            for technique in config.techniques:
                levels = resolve_levels(config.baseline, config.vary, technique)
                for model in config.models:
                    outcome = _run_one(
                        db, batch_id, config, environment, fingerprint,
                        X, y, characteristics, dataset, model, technique,
                        levels, config_text,
                    )
                    completed += outcome == "succeeded"
                    failed += outcome == "failed"
                    skipped += outcome == "skipped"

    print(f"Runs completed: {completed}")
    print(f"Runs failed: {failed}")
    print(f"Runs skipped: {skipped}")
    print(f"Record written: {db_path.resolve()}")
    return 0 if failed == 0 else 1


def _run_one(db, batch_id, config, environment, fingerprint, X, y,
             characteristics, dataset, model, technique, levels, config_text) -> str:
    """Evaluate and record one combination. Returns its status."""
    started = utc_now()
    try:
        with ResourceMonitor() as monitor:
            result = evaluate(
                X=X, y=y, task=characteristics["task_type"], levels=levels,
                model=model, seed=config.seed,
                protocol=config.evaluation.protocol,
                cv_k=config.evaluation.cv_k,
                inner_k=config.evaluation.inner_k,
                holdout_repeats=config.evaluation.holdout_repeats,
                test_size=config.evaluation.split.test,
                k_fraction=config.evaluation.k_fraction,
                dataset=dataset,
            )
        usage = monitor.usage()
    except ProtocolError as exc:
        # A protocol that cannot honestly be applied is SKIPPED, not
        # failed and not silently scored.
        _record_failure(db, batch_id, config, environment, fingerprint,
                        dataset, model, technique, exc, config_text,
                        status="skipped", levels=levels)
        print(f"Run skipped: dataset={dataset}, {config.vary}={technique}, model={model} ({exc})")
        return "skipped"
    except Exception as exc:
        _record_failure(db, batch_id, config, environment, fingerprint,
                        dataset, model, technique, exc, config_text, levels=levels)
        print(f"Run failed: dataset={dataset}, {config.vary}={technique}, model={model} "
              f"({type(exc).__name__}: {exc})")
        return "failed"

    row = _base_row(batch_id, config, environment, fingerprint, dataset, model,
                    levels, config_text, started)
    row.update({
        "status": "succeeded",
        "ended_at": utc_now(),
        "parallel_efficiency": usage.parallel_efficiency,
        **{k: v for k, v in usage.as_dict().items() if k != "n_samples"},
        **result.summary,
        "cv_k": config.evaluation.cv_k if config.evaluation.protocol == "kfold" else None,
        "cv_strategy": result.cv_strategy,
        "inner_k": config.evaluation.inner_k if levels.get("tuning") != "none" else None,
        "holdout_repeats": (config.evaluation.holdout_repeats
                            if config.evaluation.protocol == "repeated_holdout" else None),
        "fold_assignment_hash": result.fold_assignment_hash,
        "fold_scores_json": encode_fold_scores(result.per_fold),
        "n_features_in": result.n_features_in,
        "n_features_selected": result.n_features_selected,
        "k_fraction": config.evaluation.k_fraction,
        "injected_rate": config.inject_rate,
        "injected_mechanism": config.inject_mechanism,
        "tuning_budget": result.tuning_budget,
        "tuning_best_params": (str(result.tuning_best_params)
                               if result.tuning_best_params else None),
        "substitution_note": result.substitution_note,
        "n_rows": int(len(X)) - result.deletion.get("rows_dropped", 0),
        **result.deletion,
        **_characteristic_columns(characteristics),
    })
    write_run(db, row)
    print(f"Run complete: dataset={dataset}, {config.vary}={technique}, model={model}")
    return "succeeded"


def _base_row(batch_id, config, environment, fingerprint, dataset, model,
              levels, config_text, started) -> dict:
    """Build the fields every row carries, whatever its outcome."""
    split = config.evaluation.split
    return {
        "run_id": new_run_id(),
        "batch_id": batch_id,
        "config_name": config.name,
        "vary": config.vary,
        "started_at": started,
        "machine_fingerprint": fingerprint,
        **{k: v for k, v in environment.items()},
        "encoding": levels.get("encoding", "onehot"),
        "imputation": levels.get("imputation", "drop_rows"),
        "scaling": levels.get("scaling", "none"),
        "selection": levels.get("selection", "none"),
        "tuning": levels.get("tuning", "none"),
        "model": model,
        "dataset": dataset,
        "seed": config.seed,
        "protocol": config.evaluation.protocol,
        "split_train": split.train,
        "split_val": split.val,
        "split_test": split.test,
        "split_applied": int(split.applied),
        "split_note": split.note,
        "config_yaml": config_text,
    }


def _record_failure(db, batch_id, config, environment, fingerprint, dataset,
                    model, technique, exc, config_text, *, status="failed",
                    levels=None) -> None:
    """Record a failed or skipped run, with the engine's own error text.

    A failure that is not recorded is a silent gap in the sweep, so it
    gets a row like any other outcome.
    """
    row = _base_row(batch_id, config, environment, fingerprint, dataset,
                    model, levels or {}, config_text, utc_now())
    row.update({
        "status": status,
        "ended_at": utc_now(),
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:2000],
    })
    if levels is None:
        row[config.vary] = technique
    write_run(db, row)


def _characteristic_columns(characteristics: dict) -> dict:
    """Map a dataset's characteristics onto the run record's columns."""
    return {
        "task_type": characteristics.get("task_type"),
        "n_features_raw": characteristics.get("n_features"),
        "n_numeric": characteristics.get("n_numeric"),
        "n_categorical": characteristics.get("n_categorical"),
        "missing_pct": characteristics.get("missing_pct"),
        "class_balance": characteristics.get("class_balance"),
        "mean_abs_correlation": characteristics.get("mean_abs_correlation"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
