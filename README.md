# Multi-objective GA for CNN Architecture + Training Strategy (NSGA-II)

Course project implementing NSGA-II to jointly optimize:
- CIFAR-10 validation accuracy (maximize)
- Parameter count (minimize)
- FLOPs (minimize)

The chromosome encodes both architecture and training strategy.

## Quick start
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

## Fast sanity runs
```bash
make smoke
make quick
```

## Full run
```bash
make full
```

## Pipeline stages and resume support
`src/pipeline.py` now supports stage-by-stage execution and resume-safe reruns.

Stages:
1. NSGA-II search
2. Stability metrics
3. Manual baseline
4. Transfer retraining
5. Plots
6. Architecture diagrams
7. Copy report assets (optional, requires `--copy-to-report`)
8. Report fragments and markdown

State is saved in:
- `results*/pipeline_state.json`

Use:
- `--resume` to skip completed stages from `pipeline_state.json`
- `--skip-existing` to skip heavy stages when output files already exist
- `--start-stage N --end-stage M` to run a subset of stages

### Colab-friendly step execution
```bash
# Stage 1 only (longest stage)
python -m src.pipeline --results-dir results_colab --auto-device --start-stage 1 --end-stage 1

# Continue later
python -m src.pipeline --results-dir results_colab --auto-device --resume --skip-existing --start-stage 2 --end-stage 4

# Finish plots/report
python -m src.pipeline --results-dir results_colab --auto-device --resume --skip-existing --start-stage 5 --end-stage 8 --copy-to-report
```

## Performance tuning options
Main runtime-related flags:
- `--auto-device` (use CUDA automatically when available)
- `--batch-size` (default: 128)
- `--num-workers` (default: 2)
- `--val-size` (default: 5000)
- `--checkpoint-every` (NSGA-II checkpoint interval, default: 1 generation)

Example:
```bash
python -m src.pipeline \
  --results-dir results_fast \
  --auto-device \
  --batch-size 256 \
  --num-workers 2 \
  --eval-epochs 4 \
  --baseline-epochs 4 \
  --retrain-epochs 20
```

## Outputs (in `results*/`)
- `pareto_seed*.csv`, `pareto_all.csv`
- `stability_hv2d.csv`, `stability_igd.csv`
- `manual_baseline.csv`
- `retrain_cifar100.csv` or `retrain_fashionmnist.csv`
- `figures/*.png`
- `logs/<run_id>/run_meta.json`
- `logs/<run_id>/log.jsonl`
- `logs/<run_id>/checkpoints/gen_XXXX.pkl`

## Report assets
With `--copy-to-report`, figures/tables are copied to:
- `report/figures/`

Generate report fragments:
```bash
make report-tex
```

Compile PDF (requires `pdflatex`):
```bash
make report-pdf
```

## Budget sensitivity (Q7)
```bash
make budget5
make budget15
```

## Tests
```bash
make test
```
