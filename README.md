# pipeline-lab-engine (piplab)

The experimental engine for the **original five-component** research
design in `31 Maarch analysis/Chapter_1_Colile.docx` §1.7 — the scope
the two simplified artefacts never implemented.

This is a **third, separate engine**. Neither
`preprocessing-effects-framework/` (B1, pre-registered, seed 20260702)
nor `Artefact/Research_Impact_of_Preprocess/` (B2, 672 completed runs)
is modified by anything here. Those are frozen research records; adding
feature selection or tuning to them would alter a pre-registered design
and break comparability with results already written into chapters.

## What this adds that B1/B2 do not have

| Component | Baseline (Ch.1) | Other levels | In B1/B2? |
|---|---|---|---|
| Encoding | onehot | ordinal, target | hardcoded, never varied — **now a factor (amendment, PLAN.md §2b)** |
| Imputation | **drop_rows** | mean, median, knn | **drop_rows absent** |
| Scaling | none | standard, minmax, robust | yes |
| **Selection** | none | chi2, mutual_info, rfe, lasso | **absent from both** |
| Model | (all four compared) | logreg, random_forest, svm, gradient_boosting | yes |
| **Tuning** | none | random, grid, bayesian | **absent from both** |

Verified 2026-10-09: `grep -rln "SelectKBest|chi2|mutual_info|RFE|Lasso"`
and `grep -rln "GridSearchCV|RandomizedSearchCV|optuna"` across both
existing `src/` trees return only `friedman.py` false positives.

## Experimental design

**OFAT** (`methodology_analysis_Colile.docx` §2, §4): one component
varies across its levels while every other sits at its Chapter 1
baseline, so an observed effect is attributable to that component. A
config therefore names `vary` plus `techniques`, and `baseline` fills in
the rest. Full factorial would be ~21,760 runs on 17 datasets; an OFAT
sweep is ~120–300.

## Evaluation protocol

**Primary — stratified k-fold, k = 10** (methodology §6.3). This is what
runs. Per-fold arrays plus mean and **sd** for accuracy/F1
(classification) or RMSE (regression) — the metric set §9 selected.

**Tuning uses nested CV** (inner k = 3). Tuning on the folds that are
also reported would select hyperparameters to fit those folds —
optimistically biased even with no per-fold leakage. The inner loop
picks hyperparameters inside each outer training fold; the outer fold
scores the winner on data the inner loop never saw. Cost: 30 fits per
run instead of 10, which is why `configs/ofat_tuning.yaml` is 18 runs
but 540 model fits.

**The declared 70/15/15 split** is recorded on every run
(`split.train/val/test`) because the dashboard exposes it, but under
k-fold `split.applied` is `False` with a note citing methodology §6.1,
which rejects the single train/test split (Lones et al. 2024; Arlot and
Celisse 2010). No metric is ever presented as having come from a split
that did not run.

**Secondary — repeated holdout** (`protocol: repeated_holdout`, 30 ×
70/15/15), gated to datasets with **n ≥ 4,000** where a 15% test set is
large enough to be meaningful. 30 repeats answer §6.1's actual objection
(which was to a *single* split) and still give a variance estimate. Never
pooled with k-fold runs in the same comparison.

Why the gate matters: 15% of iris is 22 rows, where one misclassification
moves accuracy by 4.55 points — while the real effect to detect is +7.3
points (95% CI 2.5–12.1, B1's confirmatory scaling result). The
instrument would be coarser than the thing measured.

## Commands

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pytest                                            # engine tests
python run_experiment.py --config configs/smoke.yaml
```

`run_experiment.py --config <yaml>` is deliberately the same CLI contract
B1 and B2 use, so the `pipeline-lab/` dashboard's existing runner,
run-status parser and output-isolation machinery work unchanged.

## Layout

| Path | Role |
|---|---|
| `src/piplab/config.py` | Config contract: components, baselines, protocols, validation |
| `src/piplab/components/` | One factory per technique (scaling, imputation, encoding, selection, tuning) |
| `src/piplab/pipelines/` | Composes a leakage-safe 6-stage sklearn Pipeline from a config |
| `src/piplab/evaluation/` | Stratified k-fold, nested CV for tuning, repeated holdout |
| `src/piplab/provenance/` | Resource + environment capture (psutil) |
| `src/piplab/storage/` | The run record (SQLite) |
| `src/piplab/datasets/` | Dataset loading + characterisation |
| `configs/` | Example configs: smoke, the three new OFAT sweeps (selection, tuning, encoding), the holdout secondary |

## Status

Steps 1–3 of 13 complete (see `pipeline-lab/docs/PLAN.md` §6):
config contract, component factories, and the leakage-safe composer.
**136 tests passing**, including leakage proofs for all four selectors,
nested tuning, and target encoding. Next is step 4 (`provenance/`).
See `PROGRESS.md`.

Three components here do not exist in B1 or B2 at all: **feature
selection**, **hyperparameter tuning**, and **row deletion** as the
imputation baseline. **Encoding** is a sixth factor by amendment.
