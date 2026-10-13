# analyse.py: read a completed sweep out of lab.db and run the statistical
# framework over it.
#
# Deliberately a reader, never a writer: it opens the record read-only and
# recomputes nothing that the sweep already stored. Every number it prints
# traces to a row in lab.db, so a result here can be checked with a SQL
# query rather than taken on trust.
#
# The analysis is per model, because the design is OFAT against a named
# baseline and the research question ("does scaling help THIS model?") is
# asked separately of each. Pooling models would answer a question the
# design does not pose.
#
# Usage:
#   python analyse.py                      # the most recent config in the db
#   python analyse.py --config <name>      # one named config
#   python analyse.py --db outputs/lab.db
import argparse
import sqlite3
import sys
from pathlib import Path

from piplab.config import TECHNIQUES
from piplab.statistics.tests import StatisticsError, compare_levels

# RMSE is the only metric here where a smaller number is better; the
# statistics need to know, or every regression effect comes out inverted.
METRICS = {
    "classification": [("accuracy_mean", True), ("f1_mean", True)],
    "regression": [("rmse_mean", False)],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analyse a sweep recorded in lab.db")
    p.add_argument("--db", default="outputs/lab.db", help="path to the run record")
    p.add_argument("--config", default=None, help="config_name to analyse")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--baseline", default=None,
        help="compare against this level instead of the component's "
             "Chapter 1 baseline. Needed where the declared baseline cannot "
             "serve as a reference: under injected missingness drop_rows is "
             "skipped on most datasets and, where it runs, is scored on a "
             "different set of rows from the imputers.",
    )
    p.add_argument(
        "--exclude", action="append", default=[],
        help="leave a level out of the comparison entirely (repeatable)",
    )
    p.add_argument(
        "--complete-only", action="store_true",
        help="drop datasets that lack a score for every level, instead of "
             "reporting nothing. Useful mid-sweep; the printed dataset "
             "count always says what was actually tested.",
    )
    return p.parse_args(argv)


def complete_datasets(rows, component, levels, model, metric, datasets):
    """Datasets that have a score at every level for this model.

    Friedman and Wilcoxon are paired across datasets, so a dataset
    missing one level cannot take part without silently comparing
    different dataset sets to each other.
    """
    keep = []
    for dataset in datasets:
        have = {
            r[component] for r in rows
            if r["dataset"] == dataset and r["model"] == model
            and r[metric] is not None
        }
        if all(level in have for level in levels):
            keep.append(dataset)
    return keep


def open_record(path):
    """Open lab.db read-only, so analysis can never alter the record."""
    db = Path(path)
    if not db.exists():
        raise SystemExit(f"no run record at {db}. Run a sweep first.")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def latest_config(conn):
    row = conn.execute(
        "select config_name from runs order by started_at desc limit 1"
    ).fetchone()
    if row is None:
        raise SystemExit("the run record is empty.")
    return row["config_name"]


def load_rows(conn, config_name):
    """Succeeded rows only: a failed run has no score to compare."""
    return conn.execute(
        "select * from runs where config_name = ? and status = 'succeeded'",
        (config_name,),
    ).fetchall()


def build_scores(rows, component, levels, model, metric, datasets):
    """Collect one score per dataset per level, as the tests expect.

    A level is dropped unless it has a score for EVERY dataset: Friedman
    and Wilcoxon are paired tests, so an incomplete row would silently
    compare different dataset sets against each other.
    """
    scores = {}
    for level in levels:
        column = []
        for dataset in datasets:
            match = [
                r for r in rows
                if r[component] == level and r["model"] == model
                and r["dataset"] == dataset and r[metric] is not None
            ]
            if not match:
                column = None
                break
            column.append(match[0][metric])
        if column is not None:
            scores[level] = column
    return scores


def wilcoxon_floor(n_pairs):
    """Smallest two-sided Wilcoxon p-value attainable with n pairs.

    The signed-rank test is discrete: with every pair pointing the same
    way the p-value still cannot go below 2 / 2**n. At n = 5 that floor
    is 0.0625, so no comparison can clear alpha = 0.05 however large the
    effect. Saying "no difference" there would report an arithmetic
    limit as evidence, which is the opposite of what happened.
    """
    if n_pairs < 1:
        return 1.0
    return min(1.0, 2.0 / (2 ** n_pairs))


def _datasets_needed(threshold):
    """Smallest n whose Wilcoxon floor clears the given threshold.

    Answers "how many datasets would this comparison actually need?",
    which is the useful thing to know when a sweep comes back
    underpowered.
    """
    for n in range(2, 64):
        if wilcoxon_floor(n) <= threshold:
            return n
    return None


def report(component, model, metric, task, scores, datasets, baseline, alpha):
    higher_is_better = dict(METRICS[task])[metric]
    try:
        result = compare_levels(
            scores, baseline=baseline, higher_is_better=higher_is_better, alpha=alpha
        )
    except StatisticsError as exc:
        # The framework refuses a test it cannot run (one dataset, say).
        # That refusal is the right answer, so report it rather than
        # letting it end the whole analysis.
        print(f"\n--- {model} | {metric} | {len(datasets)} datasets")
        print(f"    not tested: {exc}")
        return
    friedman = result.friedman
    print(f"\n--- {model} | {metric} | {len(datasets)} datasets")
    print(f"    Friedman chi2 = {friedman.statistic:.4f}  p = {friedman.p_value:.4g}"
          f"  significant = {friedman.significant}")
    if result.skipped_reason:
        print(f"    {result.skipped_reason}")
        return

    # Holm tests the smallest p at alpha/k, so the bar the floor has to
    # clear is not alpha but alpha/k. Comparing against alpha alone
    # understates how many datasets a rejection actually needs.
    floor = wilcoxon_floor(len(datasets))
    k = len(result.comparisons)
    strictest = alpha / k if k else alpha
    if floor > strictest:
        print(f"    UNDERPOWERED: with {len(datasets)} datasets the smallest "
              f"possible Wilcoxon p is {floor:.5f}.")
        print(f"    Holm tests the smallest of {k} comparisons at "
              f"alpha/{k} = {strictest:.5f}, so no comparison below can be "
              f"rejected whatever the effect size.")
        print(f"    {_datasets_needed(strictest)} datasets would be the "
              f"minimum. Read the effects, not the verdicts.")
    for c in sorted(result.comparisons, key=lambda c: c.p_adjusted):
        verdict = "DIFFERS from baseline" if c.rejected else "no difference"
        print(f"    {c.level:9} vs {baseline:9} "
              f"p = {c.p_value:.4g}  holm = {c.p_adjusted:.4g}  {verdict}")
        e = c.effect
        # The win/loss split matters as much as the median: a technique
        # can win on median while losing on most datasets.
        print(f"      median {e.median_difference:+.4f}  "
              f"{int(e.ci_level * 100)}% CI "
              f"[{e.ci_low:+.4f}, {e.ci_high:+.4f}]  "
              f"better on {e.n_improved}/{e.n_pairs}, "
              f"worse on {e.n_worsened}, tied {e.n_tied}")
        if c.note:
            print(f"      note: {c.note}")


def main(argv=None):
    args = parse_args(argv)
    conn = open_record(args.db)
    config_name = args.config or latest_config(conn)
    rows = load_rows(conn, config_name)
    if not rows:
        raise SystemExit(f"no succeeded runs for config {config_name!r}.")

    component = rows[0]["vary"]
    levels = [lv for lv in TECHNIQUES[component] if lv not in args.exclude]
    baseline = args.baseline or levels[0]
    if baseline not in levels:
        raise SystemExit(
            f"baseline {baseline!r} is not among the compared levels {levels}"
        )
    # compare_levels reads the baseline by name; keep it first for display.
    levels = [baseline] + [lv for lv in levels if lv != baseline]
    print(f"config    : {config_name}")
    print(f"varying   : {component}  (baseline: {baseline})")
    print(f"runs      : {len(rows)} succeeded")

    for task in ("classification", "regression"):
        task_rows = [r for r in rows if r["task_type"] == task]
        if not task_rows:
            continue
        datasets = sorted({r["dataset"] for r in task_rows})
        models = sorted({r["model"] for r in task_rows})
        print(f"\n=== {task}: {len(datasets)} datasets, {len(models)} models")
        print(f"    {', '.join(datasets)}")
        for model in models:
            for metric, _ in METRICS[task]:
                used = datasets
                if args.complete_only:
                    used = complete_datasets(
                        task_rows, component, levels, model, metric, datasets
                    )
                    dropped = sorted(set(datasets) - set(used))
                    if dropped:
                        print(f"\n    [complete-only] dropped "
                              f"{len(dropped)}: {', '.join(dropped)}")
                if len(used) < 2:
                    print(f"\n--- {model} | {metric}: fewer than two complete "
                          f"datasets, nothing to compare")
                    continue
                scores = build_scores(
                    task_rows, component, levels, model, metric, used
                )
                if len(scores) < 2:
                    print(f"\n--- {model} | {metric}: fewer than two complete "
                          f"levels, nothing to compare")
                    continue
                report(component, model, metric, task, scores, used,
                       baseline, args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
