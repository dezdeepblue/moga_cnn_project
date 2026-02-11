
# Multi-objective GA for CNN Architecture + Training Strategy (NSGA-II) — FULL Extended

This is a complete, report-ready GA course project implementing NSGA-II
to jointly optimize:

- CIFAR-10 validation accuracy (maximize)
- parameter count (minimize)
- FLOPs (minimize)

The chromosome encodes both architecture and training strategy.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make smoke
make quick
```

## Smoke test (very fast)
Runs a tiny end-to-end pipeline for sanity checks with minimal compute:
```bash
make smoke
```

## Full run (CPU)
```bash
make full
```

## Budget sensitivity (Q7)
```bash
make budget5
make budget15
```

## Outputs (in results*/)
- `pareto_seed*.csv`, `pareto_all.csv`
- `stability_hv2d.csv`, `stability_igd.csv`
- `manual_baseline.csv`
- `retrain_cifar100.csv` or `retrain_fashionmnist.csv`
- `figures/*.png` + appendix tables (`appendix_representative_models.csv`)

## Report assets
With `--copy-to-report`, figures/tables go to `report/figures/`.

## Tests
```bash
make test
```

## A+ features
### Automatic LaTeX report generation
After `make full`, generate LaTeX fragments + an auto-filled report text:
```bash
make report-tex
```
Compile PDF (requires `pdflatex` installed):
```bash
make report-pdf
```

### Explicit HV/IGD and convergence plots
Generated into `report/figures/`:
- `fig_stability_hv_acc_params.png`, `fig_stability_hv_acc_flops.png`
- `fig_stability_igd.png`
- `fig_convergence_best_acc.png`, `fig_convergence_front_size.png`

### Runtime logging + checkpoints
Per-seed logs/checkpoints:
- `results/logs/seed*/log.jsonl`
- `results/logs/seed*/checkpoints/gen_XXXX.pkl`

### Architecture diagrams
Representative Pareto models:
- `fig_arch_min_params.png`, `fig_arch_min_flops.png`, `fig_arch_max_acc.png`, `fig_arch_knee.png`
