# Multi-Objective GA CNN Project (NSGA-II)

This project uses NSGA-II to jointly optimize CNN architecture and training strategy on CIFAR-10 with three objectives:
- maximize validation accuracy
- minimize parameter count
- minimize FLOPs

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline
Main entry point:
```bash
python -m src.pipeline --results-dir results --auto-device
```

Pipeline stages (1-8):
1. NSGA-II search
2. Stability metrics (HV, IGD)
3. Manual baseline grid
4. Transfer retraining
5. Plots
6. Architecture diagrams
7. Copy report assets (`--copy-to-report`)
8. Report fragments + markdown

Useful controls:
- `--resume`: skip completed stages listed in `pipeline_state.json`
- `--skip-existing`: skip stages when expected output already exists
- `--start-stage N --end-stage M`: run only a stage range

## Run configs
`Makefile` defines standard presets:

- `make smoke`: minimal sanity test for wiring and outputs. Fastest run, tiny population, 1 generation, 1 epoch, reduced baseline and retrain work.
- `make quick`: short experimental run for iteration. Small population/generation counts, cheap eval budget, includes report asset copy.
- `make full`: main report-quality run on CPU defaults (larger search, multi-seed stability, longer retraining on CIFAR-100).
- `make budget5`: same structure as `full` but with `--eval-epochs 5` to test lower fitness-evaluation budget sensitivity.
- `make budget15`: same structure as `full` but with `--eval-epochs 15` to test higher fitness-evaluation budget sensitivity.

CUDA variants are also provided (`smoke-cuda`, `quick-cuda`, `full-cuda`, `budget5-cuda`, `budget15-cuda`).

## Outputs
Key files under `results*/`:
- `pareto_seed*.csv`, `pareto_all.csv`
- `stability_hv2d.csv`, `stability_igd.csv`
- `manual_baseline.csv`
- `retrain_<dataset>.csv`
- `figures/*.png`, `figures/*.csv`, `figures/*.txt`
- `logs/<run_id>/run_meta.json`, `logs/<run_id>/log.jsonl`, `logs/<run_id>/checkpoints/gen_XXXX.pkl`

## Tests
```bash
make test
```

## Report helpers
```bash
make report-tex
make report-pdf
```

For detailed code walkthrough, see `DESCRIPTION.md`.
