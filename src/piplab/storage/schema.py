# schema.py: the run record's column definitions.
#
# One row per (config x dataset x model x technique), designed for
# GROUP BY trend mining: every field that a trend might group or filter
# by is its own typed column, never buried in JSON.
#
# Two fields are deliberately serialised text rather than columns:
#   * per-fold score arrays - SQLite has no array type, and a fold count
#     varies by protocol. Stored as JSON so the raw per-fold values are
#     never lost, with mean and sd available as real columns for
#     grouping.
#   * config_yaml - kept verbatim for reproducibility.
# Everything else is INTEGER, REAL or TEXT.
from __future__ import annotations

TABLE = "runs"

# (column, sqlite type, note). Order is the physical column order, which
# follows PLAN.md section 4's blocks so the schema reads like the spec.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    # --- identity & timing ---
    ("run_id", "TEXT PRIMARY KEY", "uuid4 hex for this single run"),
    ("batch_id", "TEXT NOT NULL", "one 'Run all' press; groups a sweep"),
    ("config_name", "TEXT NOT NULL", "the config's `name`"),
    ("vary", "TEXT NOT NULL", "the component this sweep varies"),
    ("started_at", "TEXT NOT NULL", "ISO-8601 UTC"),
    ("ended_at", "TEXT", "ISO-8601 UTC; NULL while running"),
    ("queue_seconds", "REAL", "waiting between batch launch and this run starting"),

    # --- resources (provenance.ResourceUsage) ---
    ("wall_seconds", "REAL", ""),
    ("cpu_seconds_user", "REAL", ""),
    ("cpu_seconds_system", "REAL", ""),
    ("cpu_percent_mean", "REAL", ""),
    ("peak_rss_mb", "REAL", ""),
    ("peak_rss_method", "TEXT", "windows_peak_wset | posix_ru_maxrss | sampled_rss_approximate"),
    ("rss_start_mb", "REAL", ""),
    ("rss_end_mb", "REAL", ""),
    ("n_threads_peak", "INTEGER", ""),
    ("disk_read_mb", "REAL", "NULL where the OS does not permit reading it"),
    ("disk_write_mb", "REAL", ""),
    ("parallel_efficiency", "REAL", "cpu/wall; ~1 computing, far below means blocked"),

    # --- machine & environment (provenance.capture_environment) ---
    ("machine_fingerprint", "TEXT NOT NULL", "resource figures only compare within this"),
    ("hostname", "TEXT", ""),
    ("cpu_model", "TEXT", ""),
    ("cpu_arch", "TEXT", ""),
    ("cpu_cores_physical", "INTEGER", ""),
    ("cpu_cores_logical", "INTEGER", ""),
    ("cpu_freq_max_mhz", "REAL", ""),
    ("ram_total_gb", "REAL", ""),
    ("os_name", "TEXT", ""),
    ("os_version", "TEXT", ""),
    ("os_release", "TEXT", ""),
    ("python_version", "TEXT", ""),
    ("python_implementation", "TEXT", ""),
    ("version_scikit_learn", "TEXT", ""),
    ("version_numpy", "TEXT", ""),
    ("version_pandas", "TEXT", ""),
    ("version_scipy", "TEXT", ""),
    ("version_optuna", "TEXT", "NULL unless Bayesian tuning is installed"),
    ("version_psutil", "TEXT", ""),
    ("engine_git_sha", "TEXT", "NULL outside a repo, never a placeholder"),

    # --- pipeline configuration: one column per component ---
    ("encoding", "TEXT NOT NULL", ""),
    ("imputation", "TEXT NOT NULL", ""),
    ("scaling", "TEXT NOT NULL", ""),
    ("selection", "TEXT NOT NULL", ""),
    ("tuning", "TEXT NOT NULL", ""),
    ("model", "TEXT NOT NULL", ""),
    ("dataset", "TEXT NOT NULL", ""),
    ("seed", "INTEGER NOT NULL", ""),
    ("tuning_budget", "INTEGER", "candidates searched; NULL when tuning=none"),
    ("tuning_best_params", "TEXT", "JSON of the winning hyperparameters"),
    ("substitution_note", "TEXT", "e.g. chi2 -> f_regression on a regression target"),

    # --- evaluation ---
    ("protocol", "TEXT NOT NULL", "kfold | repeated_holdout"),
    ("cv_k", "INTEGER", "10 per methodology section 6.3"),
    ("cv_strategy", "TEXT", "stratified_kfold | kfold"),
    ("inner_k", "INTEGER", "nested CV inner folds; NULL when tuning=none"),
    ("holdout_repeats", "INTEGER", "NULL under kfold"),
    ("fold_assignment_hash", "TEXT", "proves two runs used identical folds"),
    ("split_train", "REAL", ""),
    ("split_val", "REAL", ""),
    ("split_test", "REAL", ""),
    ("split_applied", "INTEGER", "0 under kfold: recorded, not executed"),
    ("split_note", "TEXT", "why, citing methodology section 6.1"),

    # --- metrics: methodology section 9's set only ---
    ("accuracy_mean", "REAL", "classification"),
    ("accuracy_sd", "REAL", ""),
    ("f1_mean", "REAL", "classification"),
    ("f1_sd", "REAL", ""),
    ("rmse_mean", "REAL", "regression"),
    ("rmse_sd", "REAL", ""),
    ("fold_scores_json", "TEXT", "the raw per-fold arrays; SQLite has no array type"),
    ("n_features_in", "INTEGER", "after encoding expansion"),
    ("n_features_selected", "INTEGER", "so selection's effect is measured, not inferred"),
    ("k_fraction", "REAL",
     "share of features the k-based selectors keep. Its own column rather "
     "than folded into the selection level, so analysis can group by "
     "technique and by k independently"),

    # --- dataset characteristics, copied so trends need no join ---
    ("task_type", "TEXT", "classification | regression"),
    ("n_rows", "INTEGER", "rows actually used (after drop_rows, if applied)"),
    ("n_rows_before_deletion", "INTEGER", "equals n_rows unless imputation=drop_rows"),
    ("rows_dropped", "INTEGER", "row deletion can discard most of a dataset"),
    ("pct_dropped", "REAL", ""),
    ("n_features_raw", "INTEGER", "columns in the source frame"),
    ("n_numeric", "INTEGER", ""),
    ("n_categorical", "INTEGER", ""),
    ("missing_pct", "REAL", "canonical name; b2 calls this missing_pct, prepfx missing_proportion"),
    ("class_balance", "REAL", "minority/majority; NULL for regression"),
    ("mean_abs_correlation", "REAL", "canonical; b2 calls this mean_abs_corr"),

    # --- outcome ---
    ("status", "TEXT NOT NULL", "succeeded | failed | skipped"),
    ("error_type", "TEXT", ""),
    ("error_message", "TEXT", ""),
    ("log_path", "TEXT", ""),
    ("config_yaml", "TEXT", "verbatim, for reproducibility"),
)

COLUMN_NAMES: tuple[str, ...] = tuple(name for name, _, _ in COLUMNS)

# Columns a trend query is most likely to filter or group by.
INDEXED = (
    "batch_id", "config_name", "vary", "dataset", "model",
    "encoding", "imputation", "scaling", "selection", "tuning",
    "status", "machine_fingerprint", "task_type",
)

STATUSES = ("succeeded", "failed", "skipped")


def create_table_sql() -> str:
    """Return the CREATE TABLE statement for the run record.

    Inputs: none.
    Output: SQL text. `IF NOT EXISTS`, so opening an existing database
    never destroys rows.
    """
    body = ",\n    ".join(f"{name} {sql_type}" for name, sql_type, _ in COLUMNS)
    return f"CREATE TABLE IF NOT EXISTS {TABLE} (\n    {body}\n)"


def create_index_sql() -> list[str]:
    """Return the CREATE INDEX statements for the trend-query columns."""
    return [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_{column} ON {TABLE}({column})"
        for column in INDEXED
    ]


def insert_sql() -> str:
    """Return a parameterised INSERT covering every column, in order."""
    placeholders = ", ".join("?" for _ in COLUMN_NAMES)
    columns = ", ".join(COLUMN_NAMES)
    return f"INSERT INTO {TABLE} ({columns}) VALUES ({placeholders})"


def column_notes() -> dict[str, str]:
    """Return {column: note} for the columns that carry an explanation.

    Used by the UI to explain a field rather than making the reader guess
    what, say, `peak_rss_method` means.
    """
    return {name: note for name, _, note in COLUMNS if note}
