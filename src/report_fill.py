import os
import argparse
import pandas as pd

def _fmt_int(x):
    return f"{int(x):,}"

def generate_markdown(results_dir="results", outpath="report/REPORT_FINAL.md", transfer_dataset="cifar100"):
    pareto = pd.read_csv(os.path.join(results_dir, "pareto_all.csv"))
    best = pareto.sort_values("acc", ascending=False).iloc[0]
    minp = pareto.sort_values("params").iloc[0]
    minf = pareto.sort_values("flops").iloc[0]

    hv_path = os.path.join(results_dir, "stability_hv2d.csv")
    igd_path = os.path.join(results_dir, "stability_igd.csv")
    retr_path = os.path.join(results_dir, f"retrain_{transfer_dataset}.csv")
    reps_path = os.path.join(results_dir, "figures", "appendix_representative_models.csv")

    hv = pd.read_csv(hv_path) if os.path.exists(hv_path) else None
    igd = pd.read_csv(igd_path) if os.path.exists(igd_path) else None
    retr = pd.read_csv(retr_path) if os.path.exists(retr_path) else None
    reps = pd.read_csv(reps_path) if os.path.exists(reps_path) else None

    lines = []
    lines.append("# A+ Final Report Text (Auto-filled)\n")

    lines.append("## Key outcomes")
    lines.append(f"- Best proxy CIFAR-10 validation accuracy: **{best['acc']:.4f}**")
    lines.append(f"- Smallest model (params): **{_fmt_int(minp['params'])}** (acc={minp['acc']:.4f})")
    lines.append(f"- Lowest FLOPs model: **{_fmt_int(minf['flops'])}** (acc={minf['acc']:.4f})\n")

    lines.append("## Stability (Q6)")
    if hv is not None and len(hv) > 0:
        lines.append(f"- HV (Acc-Params) mean±std: **{hv['HV_acc_params'].mean():.4f} ± {hv['HV_acc_params'].std(ddof=1):.4f}**")
        lines.append(f"- HV (Acc-FLOPs) mean±std: **{hv['HV_acc_flops'].mean():.4f} ± {hv['HV_acc_flops'].std(ddof=1):.4f}**")
    if igd is not None and len(igd) > 0:
        lines.append(f"- IGD mean±std: **{igd['IGD'].mean():.4f} ± {igd['IGD'].std(ddof=1):.4f}**")
    lines.append("")

    lines.append("## Transfer / Generalization (Q4)")
    if retr is not None and len(retr) > 0:
        lines.append(f"Dataset: **{transfer_dataset.upper()}**")
        lines.append(f"- Mean target accuracy across selected models: **{retr['target_acc_mean'].mean():.4f}**")
        lines.append(f"- Mean retrain std across models: **{retr['target_acc_std'].mean():.4f}**")
    else:
        lines.append("Retrain results not found; run `make full`.")
    lines.append("")

    lines.append("## Representative Pareto models")
    if reps is not None and len(reps) > 0:
        lines.append(reps[['rep_tag','acc','params','flops','activation','optimizer','lr']].to_markdown(index=False))
    else:
        lines.append("Representative table not found.")
    lines.append("")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return outpath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--transfer-dataset", default="cifar100", choices=["cifar100","fashionmnist"])
    ap.add_argument("--out", default="report/REPORT_FINAL.md")
    args = ap.parse_args()
    generate_markdown(args.results_dir, args.out, args.transfer_dataset)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
