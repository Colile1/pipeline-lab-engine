# Confirms run_experiment.py works end to end: it loads a config, runs
# every combination, and records one row per run -- including rows for
# runs that fail or are skipped, since an unrecorded failure is a silent
# gap in the sweep.
import subprocess
import sys
from pathlib import Path

import pytest

from piplab.storage import open_db, read_runs

ENGINE_ROOT = Path(__file__).resolve().parents[1]
CLI = ENGINE_ROOT / "run_experiment.py"


def run_cli(*args, cwd=None):
    """Invoke the CLI the way the dashboard's runner will."""
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True, cwd=str(cwd or ENGINE_ROOT), timeout=600,
    )


def test_dry_run_reports_the_grid_without_executing(tmp_path):
    result = run_cli("--config", "configs/smoke.yaml", "--dry-run",
                     "--db", str(tmp_path / "lab.db"))
    assert result.returncode == 0, result.stderr
    assert "Total runs: 4" in result.stdout
    assert "Dry run: no execution." in result.stdout
    assert not (tmp_path / "lab.db").exists()


def test_a_real_smoke_run_records_one_row_per_combination(tmp_path):
    db = tmp_path / "lab.db"
    result = run_cli("--config", "configs/smoke.yaml", "--db", str(db))
    assert result.returncode == 0, result.stderr
    assert "Runs completed: 4" in result.stdout
    assert "Runs failed: 0" in result.stdout

    with open_db(db) as connection:
        rows = read_runs(connection)
    assert len(rows) == 4
    assert {r["dataset"] for r in rows} == {"iris", "wine"}
    assert {r["scaling"] for r in rows} == {"none", "standard"}
    assert all(r["status"] == "succeeded" for r in rows)


def test_every_recorded_row_carries_provenance_and_a_real_score(tmp_path):
    db = tmp_path / "lab.db"
    assert run_cli("--config", "configs/smoke.yaml", "--db", str(db)).returncode == 0
    with open_db(db) as connection:
        rows = read_runs(connection)
    for row in rows:
        assert row["hostname"]
        assert row["version_scikit_learn"]
        assert row["machine_fingerprint"]
        assert row["wall_seconds"] > 0
        assert row["peak_rss_mb"] > 0
        assert 0.0 <= row["accuracy_mean"] <= 1.0
        assert row["accuracy_sd"] >= 0
        assert row["fold_assignment_hash"]
        assert row["config_yaml"]


def test_the_declared_split_is_recorded_but_marked_not_applied(tmp_path):
    db = tmp_path / "lab.db"
    assert run_cli("--config", "configs/smoke.yaml", "--db", str(db)).returncode == 0
    with open_db(db) as connection:
        row = read_runs(connection)[0]
    assert (row["split_train"], row["split_val"], row["split_test"]) == (0.7, 0.15, 0.15)
    assert row["split_applied"] == 0
    assert "6.1" in row["split_note"]


def test_one_batch_id_groups_the_whole_sweep(tmp_path):
    db = tmp_path / "lab.db"
    assert run_cli("--config", "configs/smoke.yaml", "--db", str(db)).returncode == 0
    with open_db(db) as connection:
        rows = read_runs(connection)
    assert len({r["batch_id"] for r in rows}) == 1


def test_rerunning_appends_a_second_batch(tmp_path):
    db = tmp_path / "lab.db"
    run_cli("--config", "configs/smoke.yaml", "--db", str(db))
    run_cli("--config", "configs/smoke.yaml", "--db", str(db))
    with open_db(db) as connection:
        rows = read_runs(connection)
    assert len(rows) == 8
    assert len({r["batch_id"] for r in rows}) == 2


def test_an_unknown_dataset_is_recorded_as_failed_not_silently_dropped(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "name: bad\nvary: scaling\ntechniques: [none]\n"
        "datasets: [no_such_dataset]\nmodels: [logreg]\nseed: 671\n"
        "evaluation:\n  cv_k: 3\noutputs_dir: outputs/\n",
        encoding="utf-8",
    )
    db = tmp_path / "lab.db"
    result = run_cli("--config", str(config), "--db", str(db))
    assert result.returncode == 1, "a failed sweep must not report success"
    with open_db(db) as connection:
        rows = read_runs(connection)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error_type"]
    assert "no_such_dataset" in rows[0]["error_message"]


def test_an_invalid_config_exits_without_a_traceback(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("name: x\nvary: encodings\n", encoding="utf-8")
    result = run_cli("--config", str(config), "--db", str(tmp_path / "lab.db"))
    assert result.returncode == 2
    assert "Config error:" in result.stdout
    assert "Traceback" not in result.stdout


def test_scaling_effect_on_wine_is_visible_in_the_record(tmp_path):
    """A real result, not just a passing pipeline: unscaled logistic
    regression on wine does not converge, so standardisation both
    improves accuracy and is dramatically faster."""
    db = tmp_path / "lab.db"
    assert run_cli("--config", "configs/smoke.yaml", "--db", str(db)).returncode == 0
    with open_db(db) as connection:
        rows = {(r["dataset"], r["scaling"]): r for r in read_runs(connection)}
    none = rows[("wine", "none")]
    standard = rows[("wine", "standard")]
    assert standard["accuracy_mean"] > none["accuracy_mean"]
    assert standard["wall_seconds"] < none["wall_seconds"]


def test_a_skipped_run_still_records_the_settings_it_was_asked_for(tmp_path):
    """cv_k, k_fraction and the injection fields are known before a run
    starts. They used to be written only when a run succeeded, so a
    skipped row could not say at what missing rate it was skipped."""
    config = tmp_path / "skip.yaml"
    config.write_text(
        "name: skipcheck\n"
        "vary: imputation\n"
        "techniques: [drop_rows, mean]\n"
        "datasets: [wine]\n"
        "models: [logreg]\n"
        "seed: 671\n"
        "inject:\n"
        "  rate: 0.3\n"
        "  mechanism: mcar\n"
        "evaluation:\n"
        "  protocol: kfold\n"
        "  cv_k: 10\n"
        f"outputs_dir: {tmp_path.as_posix()}/\n",
        encoding="utf-8",
    )
    db = tmp_path / "lab.db"
    result = run_cli("--config", str(config), "--db", str(db))
    assert result.returncode == 0, result.stderr

    with open_db(db) as connection:
        rows = {r["imputation"]: r for r in read_runs(connection)}
    # 13 columns at 30% missing leave wine almost no complete rows.
    assert rows["drop_rows"]["status"] == "skipped"
    assert rows["mean"]["status"] == "succeeded"
    for row in rows.values():
        assert row["cv_k"] == 10
        assert row["k_fraction"] == 0.5
        assert row["injected_rate"] == 0.3
        assert row["injected_mechanism"] == "mcar"
