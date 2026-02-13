# Code Description

This file explains each major part of the project and how data flows through the system.

## 1) High-level flow

Primary runtime path:
1. `src/pipeline.py` orchestrates all stages.
2. Stage 1 calls `src/run_all.py::save_fronts`, which calls `src/nsga2.py::run_nsga2` per seed.
3. `run_nsga2` evaluates chromosomes by building models (`src/model.py`), training (`src/train.py`), and measuring metrics (`src/metrics.py`).
4. Stages 2-4 compute stability, baseline, and transfer results.
5. Stages 5-8 generate figures and report artifacts.

## 2) Module-by-module details

### `src/chromosome.py`
Defines the search space and GA operators.
- Search genes: number of conv blocks, filter sizes, kernel sizes, batch norm flag, dropout, activation, optimizer, learning rate.
- `random_chromosome()`: samples a candidate.
- `repair()`: enforces valid bounds and consistent lengths after random ops.
- `crossover()` and `mutate()`: GA variation operators, both followed by `repair()`.

### `src/model.py`
Converts chromosome dictionaries into executable PyTorch models.
- `make_activation()`: maps activation name to module.
- `CNNFromChromosome`: stacks conv blocks, optional BN, periodic max-pooling, global average pooling, dropout, linear classifier.

### `src/data.py`
CIFAR-10 loading for GA fitness evaluations.
- `cifar10_loaders()`: applies train/val transforms, split, dataloaders, and cache reuse keyed by loader settings.

### `src/train.py`
Model optimization and evaluation.
- `make_optimizer()`: picks SGD/Adam/RMSprop from chromosome.
- `train_epochs()`: supervised training loop with optional CUDA AMP.
- `accuracy()`: validation accuracy computation.

### `src/metrics.py`
Complexity metrics.
- `count_params()`: total trainable parameter count.
- `compute_flops()`: FLOPs estimate via `thop.profile` on a synthetic input.

### `src/nsga2.py`
Core NSGA-II engine.
- Uses DEAP multi-objective fitness with objective vector `(-acc, params, flops)`.
- `evaluate_individual()`: full fitness path (load data, build model, count params/FLOPs, train proxy epochs, return objectives).
- `run_nsga2()`: initializes population, evaluates initial and offspring populations, applies NSGA-II selection, logs progress, writes JSONL convergence logs and optional checkpoints.
- Includes evaluation caching to avoid retraining duplicate chromosomes.

### `src/experiments.py`
Utilities for transforming fronts.
- `front_to_df()`: converts DEAP front to tabular records with chromosome JSON.
- `run_multi_seeds()`: convenience wrapper for multiple seeds.

### `src/mo_utils.py`
Multi-objective analysis helpers.
- `pareto_filter()`: non-dominated filtering.
- `normalize_points()`: min-max normalization.
- `hypervolume_2d()`: 2D hypervolume for projected fronts.
- `igd()`: inverted generational distance.

### `src/baselines.py`
Manual baseline path for comparison.
- `manual_cnn_chrom()`: fixed architecture baseline.
- `grid_tune_training()`: activation/optimizer/lr grid over the baseline architecture using the same evaluator.

### `src/transfer.py`
Generalization check on a new dataset.
- `make_loaders()`: CIFAR-100 or FashionMNIST loaders with transforms/splits.
- `retrain_and_eval()`: retrains selected chromosome model on target dataset and returns target validation accuracy.

### `src/run_all.py`
Stage-level computational routines used by pipeline.
- `save_fronts()`: runs GA across seeds and saves `pareto_seed*.csv` + `pareto_all.csv`.
- `stability_metrics()` and `hv_projections()`: produce IGD/HV stability tables.
- `run_manual_baseline()`: runs baseline grid and writes `manual_baseline.csv`.
- `retrain_selected()`: picks representative models from the joint front and writes `retrain_<dataset>.csv`.

### `src/plots.py`
Main figures/tables for analysis.
- Pareto projections, seed overlays, joint-vs-baseline plot, dominant pair table, Spearman analysis, stability bars, retrain error bars, representative model table.

### `src/plots_plus.py`
Convergence diagnostics from runtime logs.
- Reads per-generation `log.jsonl` and generates convergence charts + `convergence_logs.csv`.

### `src/arch_diagrams.py`
Architecture visualizations.
- Draws schematic diagrams for representative Pareto solutions (`min_params`, `min_flops`, `max_acc`, `knee`).

### `src/report_gen.py`
Auto-generates LaTeX fragment files from results for report assembly.

### `src/report_fill.py`
Auto-generates markdown summary report (`report/REPORT_FINAL.md`).

### `src/pipeline.py`
Top-level orchestrator and CLI.
- Defines stage execution order.
- Supports resumable operation with `pipeline_state.json`.
- Supports partial stage ranges and skip-existing behavior.
- Handles report asset copy and report generation stage calls.

## 3) Run configs (what each is for)

Defined in `Makefile`:

- `smoke`:
  - Goal: verify end-to-end wiring quickly.
  - Characteristics: very small population/generation and 1 epoch budgets; reduced baseline combinations and retrain count.
  - Use when: validating code changes, CI/local sanity checks.

- `quick`:
  - Goal: iterate on behavior faster than full run while still producing meaningful outputs.
  - Characteristics: small search budget, single seed, short retraining, includes copy-to-report.
  - Use when: development loops and preliminary plots.

- `full`:
  - Goal: primary experiment configuration for stronger evidence.
  - Characteristics: larger population/generations, multi-seed stability, longer eval/retrain budgets, transfer on CIFAR-100.
  - Use when: final experiments and report-ready artifacts.

- `budget` variants (`budget5`, `budget15`):
  - Goal: budget sensitivity analysis for GA fitness evaluation cost.
  - Characteristics: same structure as `full`, only evaluation epoch budget differs (`--eval-epochs 5` vs `15`).
  - Use when: answering budget-sensitivity questions and comparing front quality vs cost.

## 4) How to run

### Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

### Preset runs
```bash
make smoke
make quick
make full
make budget5
make budget15
```

### Direct pipeline run
```bash
python -m src.pipeline --results-dir results --auto-device
```

### Resume / stage slicing examples
```bash
# Run only stage 1
python -m src.pipeline --results-dir results --auto-device --start-stage 1 --end-stage 1

# Continue later, skipping completed and existing outputs
python -m src.pipeline --results-dir results --auto-device --resume --skip-existing --start-stage 2 --end-stage 8
```

### Tests
```bash
make test
```

## 5) Result artifacts you should expect

- Pareto fronts: `pareto_seed*.csv`, `pareto_all.csv`
- Stability: `stability_hv2d.csv`, `stability_igd.csv`
- Baseline: `manual_baseline.csv`
- Transfer: `retrain_<dataset>.csv`
- Figures/tables: `results*/figures/*`
- Logs/checkpoints: `results*/logs/<run_id>/...`
- Pipeline state: `results*/pipeline_state.json`
