# Confirms the run record matches PLAN.md section 4, that re-running
# appends rather than overwrites, that every column is a type SQLite
# stores natively, and that a typo'd column is refused rather than
# silently dropped.
import json
import sqlite3

import pytest

from piplab.provenance import ResourceMonitor, capture_environment, machine_fingerprint
from piplab.storage import (
    COLUMN_NAMES,
    STATUSES,
    RecordError,
    count_runs,
    decode_fold_scores,
    encode_fold_scores,
    new_batch_id,
    new_run_id,
    open_db,
    read_runs,
    utc_now,
    write_run,
)
from piplab.storage.schema import COLUMNS, INDEXED, column_notes


def minimal_row(**overrides) -> dict:
    """The smallest valid row: every required field, nothing else."""
    row = {
        "run_id": new_run_id(),
        "batch_id": new_batch_id(),
        "config_name": "test_config",
        "vary": "scaling",
        "started_at": utc_now(),
        "status": "succeeded",
        "machine_fingerprint": "test|AMD64|8c/12t|15.75GB",
        "encoding": "onehot", "imputation": "drop_rows", "scaling": "none",
        "selection": "none", "tuning": "none", "model": "logreg",
        "dataset": "iris", "seed": 671, "protocol": "kfold",
    }
    row.update(overrides)
    return row


# --- schema matches the plan ----------------------------------------------

def test_every_block_from_plan_section_4_is_present():
    names = set(COLUMN_NAMES)
    # identity & timing
    assert {"run_id", "batch_id", "config_name", "started_at", "ended_at",
            "wall_seconds", "cpu_seconds_user", "cpu_seconds_system",
            "queue_seconds"} <= names
    # resources
    assert {"peak_rss_mb", "cpu_percent_mean", "n_threads_peak",
            "disk_read_mb", "disk_write_mb"} <= names
    # machine & environment
    assert {"hostname", "cpu_model", "cpu_cores_physical", "cpu_cores_logical",
            "ram_total_gb", "os_name", "os_version", "python_version",
            "version_scikit_learn", "version_numpy", "version_pandas",
            "version_scipy", "engine_git_sha"} <= names
    # one column per pipeline component, so each is directly groupable
    assert {"encoding", "imputation", "scaling", "selection", "tuning",
            "model", "seed", "cv_k"} <= names
    # evaluation
    assert {"cv_strategy", "fold_assignment_hash", "split_train", "split_val",
            "split_test", "split_applied", "split_note"} <= names
    # metrics: methodology section 9's set, each as mean AND sd
    assert {"accuracy_mean", "accuracy_sd", "f1_mean", "f1_sd",
            "rmse_mean", "rmse_sd", "n_features_in", "n_features_selected"} <= names
    # dataset characteristics, copied so trends need no join
    assert {"n_rows", "n_numeric", "n_categorical", "missing_pct",
            "class_balance", "mean_abs_correlation", "task_type"} <= names
    # outcome
    assert {"status", "error_type", "error_message", "log_path", "config_yaml"} <= names


def test_no_metric_beyond_the_methodology_set_is_recorded():
    """Methodology section 9 selected accuracy + F1 + RMSE deliberately.

    Adding ROC-AUC or precision/recall would be an amendment to that
    section, not a silent schema addition.
    """
    names = set(COLUMN_NAMES)
    for forbidden in ("roc_auc", "precision", "recall", "balanced_accuracy", "r2", "mae"):
        assert not any(forbidden in n for n in names), f"{forbidden} is beyond the registered set"


def test_schema_has_no_duplicate_columns():
    assert len(COLUMN_NAMES) == len(set(COLUMN_NAMES))


def test_trend_query_columns_are_indexed():
    names = set(COLUMN_NAMES)
    for column in INDEXED:
        assert column in names


def test_columns_carry_notes_where_they_need_explaining():
    notes = column_notes()
    # The non-obvious ones must explain themselves to a UI reader.
    assert "peak_rss_method" in notes
    assert "split_applied" in notes and "recorded, not executed" in notes["split_applied"]
    assert "fold_scores_json" in notes


# --- writing and reading --------------------------------------------------

def test_a_minimal_row_round_trips(tmp_path):
    with open_db(tmp_path / "lab.db") as db:
        run_id = write_run(db, minimal_row())
        rows = read_runs(db)
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "succeeded"
    # Unsupplied columns are NULL, not absent.
    assert set(rows[0]) == set(COLUMN_NAMES)
    assert rows[0]["accuracy_mean"] is None


def test_rerunning_appends_and_never_overwrites(tmp_path):
    """A sweep is expensive and often interrupted here, so a second
    attempt must sit beside the first, not replace it."""
    with open_db(tmp_path / "lab.db") as db:
        first = write_run(db, minimal_row(config_name="same", accuracy_mean=0.80))
        second = write_run(db, minimal_row(config_name="same", accuracy_mean=0.91))
        rows = read_runs(db, config_name="same")
    assert first != second
    assert len(rows) == 2
    assert {r["accuracy_mean"] for r in rows} == {0.80, 0.91}


def test_reopening_an_existing_database_keeps_its_rows(tmp_path):
    path = tmp_path / "lab.db"
    with open_db(path) as db:
        write_run(db, minimal_row())
    with open_db(path) as db:  # CREATE TABLE IF NOT EXISTS must not wipe it
        assert count_runs(db) == 1
        write_run(db, minimal_row())
        assert count_runs(db) == 2


def test_rows_come_back_newest_first(tmp_path):
    with open_db(tmp_path / "lab.db") as db:
        write_run(db, minimal_row(started_at="2026-01-01T00:00:00+00:00", dataset="old"))
        write_run(db, minimal_row(started_at="2026-06-01T00:00:00+00:00", dataset="new"))
        rows = read_runs(db)
    assert [r["dataset"] for r in rows] == ["new", "old"]


def test_filters_and_counts_work_for_trend_queries(tmp_path):
    batch = new_batch_id()
    with open_db(tmp_path / "lab.db") as db:
        write_run(db, minimal_row(batch_id=batch, model="logreg", status="succeeded"))
        write_run(db, minimal_row(batch_id=batch, model="svm", status="failed"))
        write_run(db, minimal_row(model="svm", status="succeeded"))
        assert count_runs(db) == 3
        assert count_runs(db, batch_id=batch) == 2
        assert count_runs(db, status="failed") == 1
        assert len(read_runs(db, batch_id=batch, status="succeeded")) == 1


def test_group_by_works_on_component_columns(tmp_path):
    """The whole point of one column per component: trend mining."""
    with open_db(tmp_path / "lab.db") as db:
        for level, score in (("none", 0.80), ("standard", 0.88), ("standard", 0.90)):
            write_run(db, minimal_row(scaling=level, accuracy_mean=score))
        result = dict(db.execute(
            "SELECT scaling, AVG(accuracy_mean) FROM runs GROUP BY scaling"
        ).fetchall())
    assert result["none"] == pytest.approx(0.80)
    assert result["standard"] == pytest.approx(0.89)


# --- refusals: a silent loss would be worse than a failure ----------------

def test_an_unknown_column_is_refused_not_silently_dropped(tmp_path):
    with open_db(tmp_path / "lab.db") as db:
        with pytest.raises(RecordError, match="roc_auc"):
            write_run(db, minimal_row(roc_auc=0.93))


@pytest.mark.parametrize("missing", ["run_id", "batch_id", "config_name", "status",
                                     "machine_fingerprint"])
def test_a_missing_required_field_is_refused(tmp_path, missing):
    row = minimal_row()
    row[missing] = None
    with open_db(tmp_path / "lab.db") as db:
        with pytest.raises(RecordError, match=missing):
            write_run(db, row)


def test_an_invalid_status_is_refused(tmp_path):
    with open_db(tmp_path / "lab.db") as db:
        with pytest.raises(RecordError, match="status"):
            write_run(db, minimal_row(status="finished"))
    assert set(STATUSES) == {"succeeded", "failed", "skipped"}


def test_an_unstorable_type_is_refused_with_a_usable_message(tmp_path):
    """A list would otherwise be stringified into something unqueryable."""
    with open_db(tmp_path / "lab.db") as db:
        with pytest.raises(RecordError, match="fold_scores_json"):
            write_run(db, minimal_row(accuracy_mean=[0.8, 0.9]))


def test_a_duplicate_run_id_is_refused(tmp_path):
    row = minimal_row()
    with open_db(tmp_path / "lab.db") as db:
        write_run(db, row)
        with pytest.raises(RecordError, match="could not store"):
            write_run(db, row)


# --- fold arrays ----------------------------------------------------------

def test_fold_scores_round_trip_without_loss():
    per_fold = {"accuracy": [0.91, 0.88, 0.93], "f1": [0.90, 0.87, 0.92]}
    restored = decode_fold_scores(encode_fold_scores(per_fold))
    assert restored == per_fold


def test_fold_scores_survive_the_database(tmp_path):
    per_fold = {"rmse": [3.2, 3.9, 4.1]}
    with open_db(tmp_path / "lab.db") as db:
        write_run(db, minimal_row(fold_scores_json=encode_fold_scores(per_fold)))
        stored = read_runs(db)[0]
    assert decode_fold_scores(stored["fold_scores_json"]) == per_fold


def test_decoding_nothing_gives_an_empty_dict_not_an_error():
    assert decode_fold_scores(None) == {}
    assert decode_fold_scores("") == {}


# --- integration with step 4's provenance ---------------------------------

def test_a_real_provenance_capture_stores_cleanly(tmp_path):
    """Step 4's output must drop straight into step 5's columns."""
    import time

    env = capture_environment()
    with ResourceMonitor(sample_interval=0.01) as monitor:
        time.sleep(0.05)
    usage = monitor.usage()

    row = minimal_row(
        machine_fingerprint=machine_fingerprint(env),
        parallel_efficiency=usage.parallel_efficiency,
        **{k: v for k, v in env.items() if k in set(COLUMN_NAMES)},
        **{k: v for k, v in usage.as_dict().items() if k in set(COLUMN_NAMES)},
    )
    with open_db(tmp_path / "lab.db") as db:
        write_run(db, row)
        stored = read_runs(db)[0]

    assert stored["hostname"] == env["hostname"]
    assert stored["cpu_cores_physical"] == env["cpu_cores_physical"]
    assert stored["version_scikit_learn"] == env["version_scikit_learn"]
    assert stored["wall_seconds"] == usage.wall_seconds
    assert stored["peak_rss_method"] == usage.peak_rss_method
    assert stored["peak_rss_mb"] > 0


def test_every_provenance_field_has_a_home_in_the_schema():
    """A captured field with no column would be measured then thrown away."""
    env = capture_environment()
    names = set(COLUMN_NAMES)
    unhoused = [k for k in env if k not in names]
    assert not unhoused, f"captured but not stored: {unhoused}"


def test_a_new_schema_column_is_added_to_an_existing_database(tmp_path):
    """CREATE TABLE IF NOT EXISTS does nothing to a table that already
    exists, so adding a schema column would break every write against an
    older record. k_fraction was added with 1,433 rows already stored.

    The migration must ADD only: existing rows keep their values and the
    new column reads NULL, which is the honest value for runs that
    predate the field.
    """
    import sqlite3

    from piplab.storage.record import _add_missing_columns

    path = tmp_path / "old.db"
    with open_db(path) as connection:
        write_run(connection, minimal_row())

    # Simulate an older database by dropping the newest column.
    with sqlite3.connect(path) as raw:
        raw.execute("alter table runs drop column k_fraction")
        before = raw.execute("select count(*) from runs").fetchone()[0]
    assert before == 1

    with open_db(path) as connection:
        columns = {r[1] for r in connection.execute("pragma table_info(runs)")}
        after = connection.execute("select count(*) from runs").fetchone()[0]
        nulls = connection.execute(
            "select count(*) from runs where k_fraction is null"
        ).fetchone()[0]

    assert "k_fraction" in columns, "the column should have been re-added"
    assert after == before, "no row may be lost to a migration"
    assert nulls == 1, "a pre-existing row should read NULL, not a guess"


def test_the_migration_is_a_no_op_on_a_current_database(tmp_path):
    """Non-vacuous: it must not churn a database that is already right."""
    from piplab.storage.record import _add_missing_columns

    path = tmp_path / "current.db"
    with open_db(path) as connection:
        added = _add_missing_columns(connection)
    assert added == []
