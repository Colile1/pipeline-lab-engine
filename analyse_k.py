# analyse_k.py: answer the question the first selection sweep could not.
#
# The v2 selection arm found that every k-based technique cost accuracy,
# but k was fixed at half the features everywhere. So "selection hurts"
# and "keeping half the features hurts" were indistinguishable.
#
# With k = 0.25, 0.50 and 0.75 recorded, they separate:
#
#   if accuracy RISES with k          -> the loss is about how much is
#                                        discarded, not about selecting
#   if accuracy is FLAT across k      -> the act of selecting is the cost
#   if there is an interior optimum   -> selection genuinely helps, at
#                                        the right k
#
# Read-only, like analyse.py: opens the record mode=ro and recomputes
# nothing that the sweep already stored.
import argparse
import sqlite3
import statistics
import sys
from pathlib import Path

# The three sweeps that share a selection design but differ in k.
SOURCES = {
    0.25: "ofat_selection_k25",
    0.50: "ofat_selection_v2",
    0.75: "ofat_selection_k75",
}
# Only the k-based filters: "none" is the baseline and "lasso" is
# embedded, so neither responds to k.
K_LEVELS = ("chi2", "mutual_info", "rfe")
# A verdict needs a real dataset set behind every k. Below this, a
# partially-finished sweep would otherwise yield a confident-looking
# median computed from two datasets.
MIN_DATASETS = 8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare feature selection across k")
    p.add_argument("--db", default="outputs/lab.db")
    p.add_argument("--metric", default="accuracy_mean")
    return p.parse_args(argv)


def connect(path):
    db = Path(path)
    if not db.exists():
        raise SystemExit(f"no run record at {db}")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def baseline_scores(conn, metric):
    """The no-selection score per (model, dataset), from the k=0.5 sweep.

    "none" is recorded once, in the v2 arm, because k does not apply to
    it. Every k is compared against that same baseline.
    """
    rows = conn.execute(
        f"select model, dataset, {metric} as score from runs "
        f"where config_name = ? and selection = 'none' "
        f"and status = 'succeeded' and {metric} is not null",
        (SOURCES[0.50],),
    ).fetchall()
    return {(r["model"], r["dataset"]): r["score"] for r in rows}


def scores_at(conn, config_name, metric):
    rows = conn.execute(
        f"select model, dataset, selection, {metric} as score, "
        f"n_features_in, n_features_selected from runs "
        f"where config_name = ? and status = 'succeeded' "
        f"and {metric} is not null",
        (config_name,),
    ).fetchall()
    return {(r["model"], r["dataset"], r["selection"]): r for r in rows}


def main(argv=None):
    args = parse_args(argv)
    conn = connect(args.db)
    base = baseline_scores(conn, args.metric)
    if not base:
        raise SystemExit("no baseline rows; has the v2 selection arm run?")

    tables = {k: scores_at(conn, name, args.metric) for k, name in SOURCES.items()}
    missing = [name for k, name in SOURCES.items() if not tables[k]]
    if missing:
        print(f"not yet recorded: {', '.join(missing)}")
        print("Run the k sweeps first; showing what is available.\n")

    print(f"metric: {args.metric}  (higher is better)")
    print("change against NO selection, median over datasets\n")
    header = f"{'model':20}{'technique':14}" + "".join(
        f"{f'k={k:.2f}':>12}" for k in sorted(SOURCES)
    )
    print(header)
    print("-" * len(header))

    models = sorted({m for m, _ in base})
    verdicts = []
    for model in models:
        for level in K_LEVELS:
            # Compare on the SAME datasets at every k. A k whose sweep is
            # still running covers only the datasets it has reached, and
            # a median over a different set is not comparable: it would
            # read as an effect of k when it is an effect of which
            # datasets happened to finish.
            common = None
            for k in SOURCES:
                have = {
                    dataset for (m, dataset), _ in base.items()
                    if m == model and (model, dataset, level) in tables[k]
                }
                common = have if common is None else (common & have)
            common = common or set()

            cells, medians = [], {}
            for k in sorted(SOURCES):
                diffs = [
                    tables[k][(model, dataset, level)]["score"] - base[(model, dataset)]
                    for dataset in sorted(common)
                ]
                if diffs:
                    med = statistics.median(diffs)
                    medians[k] = med
                    cells.append(f"{med:+12.4f}")
                else:
                    cells.append(f"{'-':>12}")
            print(f"{model:20}{level:14}" + "".join(cells) + f"{len(common):>7} ds")
            # Only interpret where every k has a real dataset set behind
            # it; MIN_DATASETS keeps a two-dataset partial sweep from
            # producing a confident-looking verdict.
            if len(medians) == len(SOURCES) and len(common) >= MIN_DATASETS:
                verdicts.append((model, level, medians))
    print()
    if not verdicts:
        covered = {k: len({d for (_, d, lv) in tables[k] if lv in K_LEVELS})
                   for k in sorted(SOURCES)}
        print("No model/technique yet has all three k values over at least "
              f"{MIN_DATASETS} shared datasets.")
        print(f"datasets reached so far: "
              + ", ".join(f"k={k:.2f}: {n}" for k, n in covered.items()))
        print("Re-run once the sweeps finish.")
        return 0
    interpret(verdicts)
    return 0


def interpret(verdicts):
    """State which explanation the numbers support, per model/technique."""
    if not verdicts:
        print("Not enough k values recorded yet to interpret.")
        return
    print("what this says:")
    rising = flat = interior = 0
    for model, level, medians in verdicts:
        ks = sorted(medians)
        low, mid, high = (medians[k] for k in ks)
        spread = max(medians.values()) - min(medians.values())
        if spread < 0.002:
            verdict = "flat across k: the ACT of selecting is the cost"
            flat += 1
        elif high > low and high >= mid:
            verdict = "rises with k: the cost is DISCARDING, not selecting"
            rising += 1
        elif mid > low and mid > high:
            verdict = "interior optimum: selection helps at the right k"
            interior += 1
        else:
            verdict = "falls with k: keeping more features hurts"
        print(f"   {model:20}{level:14}{verdict}")
    print()
    print(f"summary: {rising} rising, {flat} flat, {interior} interior optimum, "
          f"of {len(verdicts)} model-technique pairs")
    if rising > max(flat, interior):
        print("-> The dominant pattern is that accuracy recovers as k grows, so")
        print("   the v2 result was mostly 'keeping half the features hurts'")
        print("   rather than 'selection hurts'.")
    elif flat > max(rising, interior):
        print("-> The dominant pattern is flatness, so the cost is in selecting")
        print("   at all rather than in how much is discarded.")


if __name__ == "__main__":
    sys.exit(main())
