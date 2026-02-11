import os
import shutil
import argparse

from .run_all import save_fronts, stability_metrics, hv_projections, run_manual_baseline, retrain_selected
from .plots import main as plots_main
from .plots_plus import main as plots_plus_main
from .arch_diagrams import generate_arch_diagrams
from .report_gen import generate as latex_generate
from .report_fill import generate_markdown

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_report_assets(results_dir, report_dir="report"):
    figs_dir = os.path.join(results_dir, "figures")
    if not os.path.isdir(figs_dir):
        return
    ensure_dir(report_dir)
    out_figs = os.path.join(report_dir, "figures")
    ensure_dir(out_figs)
    for name in os.listdir(figs_dir):
        if name.lower().endswith((".png",".csv",".txt")):
            shutil.copy2(os.path.join(figs_dir, name), os.path.join(out_figs, name))

def run_pipeline(args):
    ensure_dir(args.results_dir)
    log_dir = os.path.join(args.results_dir, "logs")
    ensure_dir(log_dir)

    seeds = list(range(args.num_seeds))
    print(
        f"[PIPELINE] Starting run | seeds={seeds} | pop={args.pop_size} | ngen={args.ngen} | eval_epochs={args.eval_epochs}",
        flush=True,
    )

    # 1) GA with runtime logging + checkpoints
    print("[PIPELINE] Stage 1/7: Running NSGA-II search", flush=True)
    all_df, run_ids = save_fronts(
        seeds=seeds,
        outdir=args.results_dir,
        log_dir=log_dir,
        pop_size=args.pop_size,
        ngen=args.ngen,
        epochs=args.eval_epochs,
        device=args.device,
    )
    print(f"[PIPELINE] Stage 1/7 complete | runs={run_ids}", flush=True)

    # 2) Stability
    print("[PIPELINE] Stage 2/7: Computing stability metrics", flush=True)
    igd_df, _ = stability_metrics(all_df)
    igd_df.to_csv(os.path.join(args.results_dir, "stability_igd.csv"), index=False)

    hv_df = hv_projections(all_df)
    hv_df.to_csv(os.path.join(args.results_dir, "stability_hv2d.csv"), index=False)
    print("[PIPELINE] Stage 2/7 complete", flush=True)

    # 3) Baseline + retrain
    print("[PIPELINE] Stage 3/7: Running manual baseline", flush=True)
    max_combos = args.baseline_max_combos if args.baseline_max_combos > 0 else None
    run_manual_baseline(
        outdir=args.results_dir,
        device=args.device,
        seed=0,
        epochs=args.baseline_epochs,
        max_combos=max_combos,
    )
    print("[PIPELINE] Stage 3/7 complete", flush=True)

    print("[PIPELINE] Stage 4/7: Retraining selected models on transfer dataset", flush=True)
    retrain_selected(
        all_df=all_df,
        outdir=args.results_dir,
        dataset=args.transfer_dataset,
        retrain_epochs=args.retrain_epochs,
        device=args.device,
        retrain_seeds=tuple(range(args.retrain_seeds)),
        k_each=args.retrain_k_each,
    )
    print("[PIPELINE] Stage 4/7 complete", flush=True)

    # 4) Plots (includes HV/IGD bars from src/plots.py and convergence from logs)
    print("[PIPELINE] Stage 5/7: Generating plots", flush=True)
    plots_main(results_dir=args.results_dir, transfer_dataset=args.transfer_dataset)
    plots_plus_main(results_dir=args.results_dir, log_dir=log_dir, run_ids=run_ids)
    print("[PIPELINE] Stage 5/7 complete", flush=True)

    # 5) Architecture diagrams
    print("[PIPELINE] Stage 6/7: Drawing architecture diagrams", flush=True)
    generate_arch_diagrams(results_dir=args.results_dir)
    print("[PIPELINE] Stage 6/7 complete", flush=True)

    # 6) Copy assets into report/figures
    if args.copy_to_report:
        print("[PIPELINE] Stage 7/7: Copying report assets", flush=True)
        copy_report_assets(args.results_dir, report_dir=args.report_dir)
        print("[PIPELINE] Stage 7/7 complete", flush=True)

    # 7) Auto LaTeX fragments + auto-filled report text
    print("[PIPELINE] Writing report fragments and markdown", flush=True)
    latex_generate(report_dir=args.report_dir, results_dir=args.results_dir, transfer_dataset=args.transfer_dataset)
    generate_markdown(results_dir=args.results_dir, outpath=os.path.join(args.report_dir, "REPORT_FINAL.md"),
                      transfer_dataset=args.transfer_dataset)

    print("\nDONE ✅ (A+ pipeline)")
    print("Results:", args.results_dir)
    print("Report folder:", args.report_dir)

def build_argparser():
    p = argparse.ArgumentParser(description="A+ pipeline: GA -> stability -> baseline -> retrain -> plots -> diagrams -> LaTeX")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pop-size", type=int, default=24)
    p.add_argument("--ngen", type=int, default=8)
    p.add_argument("--eval-epochs", type=int, default=6)
    p.add_argument("--num-seeds", type=int, default=3)

    p.add_argument("--transfer-dataset", default="cifar100", choices=["cifar100","fashionmnist"])
    p.add_argument("--baseline-epochs", type=int, default=6)
    p.add_argument("--baseline-max-combos", type=int, default=0,
                   help="Limit baseline grid combinations; 0 means full grid.")
    p.add_argument("--retrain-epochs", type=int, default=50)
    p.add_argument("--retrain-seeds", type=int, default=3)
    p.add_argument("--retrain-k-each", type=int, default=3)

    p.add_argument("--copy-to-report", action="store_true")
    p.add_argument("--report-dir", default="report")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(args)
