# record.py: write and read the run record.
#
# Append-only by design. A sweep is expensive and often interrupted on
# this hardware, so re-running a config must never overwrite earlier
# rows: each run gets a fresh run_id and lands beside its predecessors.
# Comparing two attempts is then a query, not an act of recovery.
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from piplab.storage.schema import (
    COLUMN_NAMES,
    STATUSES,
    TABLE,
    create_index_sql,
    create_table_sql,
    insert_sql,
)

DEFAULT_DB_PATH = Path("data") / "lab.db"


class RecordError(RuntimeError):
    """Raised when a run row cannot be stored; message names the problem."""


def new_run_id() -> str:
    """Return a fresh run identifier."""
    return uuid.uuid4().hex


def new_batch_id() -> str:
    """Return a fresh batch identifier, shared by one sweep's runs."""
    return uuid.uuid4().hex


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Stored as text because SQLite has no datetime type, and ISO-8601
    sorts lexicographically, so ORDER BY started_at works.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@contextmanager
def open_db(path: str | Path = DEFAULT_DB_PATH):
    """Open the run database, creating the table and indexes if absent.

    Inputs: path - the SQLite file; parent directories are created.
    Output: a context-managed sqlite3.Connection with row access by name.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(create_table_sql())
        for statement in create_index_sql():
            connection.execute(statement)
        connection.commit()
        yield connection
    finally:
        connection.close()


def write_run(connection: sqlite3.Connection, row: dict) -> str:
    """Append one run row.

    Inputs: connection - an open connection; row - a dict whose keys are
    a subset of the schema's columns. Missing columns are stored NULL;
    an unknown key is an error rather than being silently dropped, since
    a typo'd metric name would otherwise vanish without trace.
    Output: the stored run_id.
    Raises: RecordError on an unknown column, a missing required field,
    an invalid status, or a value SQLite cannot store.
    """
    unknown = sorted(set(row) - set(COLUMN_NAMES))
    if unknown:
        raise RecordError(f"unknown run-record column(s): {', '.join(unknown)}")
    for required in ("run_id", "batch_id", "config_name", "status", "machine_fingerprint"):
        if not row.get(required):
            raise RecordError(f"run record is missing required field {required!r}")
    if row["status"] not in STATUSES:
        raise RecordError(f"status must be one of {STATUSES}, got {row['status']!r}")

    values = [_storable(row.get(name), name) for name in COLUMN_NAMES]
    try:
        connection.execute(insert_sql(), values)
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise RecordError(f"could not store run {row['run_id']}: {exc}") from exc
    return row["run_id"]


def _storable(value, column: str):
    """Coerce a value to something SQLite stores, or refuse it.

    Inputs: value - the field value; column - its name, for the error.
    Output: None, int, float or str.
    Raises: RecordError naming the column when a value is a type that
    would otherwise be stringified into something unqueryable.
    """
    if value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, bool):  # pragma: no cover - bool is an int subclass
        return int(value)
    raise RecordError(
        f"column {column!r} got {type(value).__name__}, which SQLite cannot store; "
        "serialise it explicitly (fold arrays belong in fold_scores_json)"
    )


def encode_fold_scores(per_fold: dict[str, list[float]]) -> str:
    """Serialise the per-fold score arrays for storage.

    SQLite has no array type and the fold count varies by protocol, so
    the raw arrays are kept as JSON while mean and sd live in real
    columns for grouping. Nothing is discarded.
    Inputs: per_fold - {metric: [score per fold]}.
    Output: compact JSON text.
    """
    return json.dumps({k: [float(v) for v in vals] for k, vals in per_fold.items()},
                      separators=(",", ":"))


def decode_fold_scores(text: str | None) -> dict[str, list[float]]:
    """Read per-fold arrays back out of storage.

    Inputs: text - the stored JSON, or None.
    Output: {metric: [scores]}; empty dict when nothing was stored.
    """
    if not text:
        return {}
    return json.loads(text)


def read_runs(
    connection: sqlite3.Connection,
    *,
    batch_id: str | None = None,
    config_name: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read run rows back, newest first.

    Inputs: connection - an open connection; batch_id/config_name/status
    - optional equality filters; limit - optional row cap.
    Output: list of dicts, one per row, every schema column present.
    """
    clauses, params = [], []
    for column, value in (("batch_id", batch_id), ("config_name", config_name), ("status", status)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    tail = f" LIMIT {int(limit)}" if limit else ""
    sql = f"SELECT * FROM {TABLE}{where} ORDER BY started_at DESC{tail}"
    return [dict(row) for row in connection.execute(sql, params)]


def count_runs(connection: sqlite3.Connection, **filters) -> int:
    """Return how many rows match the given equality filters."""
    clauses, params = [], []
    for column, value in filters.items():
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor = connection.execute(f"SELECT COUNT(*) FROM {TABLE}{where}", params)
    return int(cursor.fetchone()[0])
