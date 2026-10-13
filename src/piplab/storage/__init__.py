# piplab.storage: the run record (SQLite).
#
# Append-only: a sweep is expensive and often interrupted on this
# hardware, so re-running never overwrites an earlier attempt.
from piplab.storage.record import (
    DEFAULT_DB_PATH,
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
from piplab.storage.schema import COLUMN_NAMES, COLUMNS, STATUSES, TABLE, column_notes

__all__ = [
    "DEFAULT_DB_PATH", "RecordError", "count_runs", "decode_fold_scores",
    "encode_fold_scores", "new_batch_id", "new_run_id", "open_db",
    "read_runs", "utc_now", "write_run",
    "COLUMN_NAMES", "COLUMNS", "STATUSES", "TABLE", "column_notes",
]
