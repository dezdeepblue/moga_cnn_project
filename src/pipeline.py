import os
import json
import shutil
import argparse
import pandas as pd
import torch

from .run_all import save_fronts, stability_metrics, hv_projections, run_manual_baseline, retrain_selected
from .plots import main as plots_main
from .plots_plus import main as plots_plus_main
from .arch_diagrams import generate_arch_diagrams
from .report_gen import generate as latex_generate
from .report_fill import generate_markdown

TOTAL_STAGES = 8


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def pipeline_state_path(results_dir):
    return os.path.join(results_dir, "pipeline_state.json")


def load_pipeline_state(results_dir):
    path = pipeline_state_path(results_dir)
    if not os.path.isfile(path):
        return {"completed": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_stage_complete(results_dir, stage_idx):
    state = load_pipeline_state(results_dir)
    completed = set(state.get("completed", []))
    completed.add(int(stage_idx))
    state["completed"] = sorted(completed)
    with open(pipeline_state_path(results_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def should_run_stage(args, stage_idx):
    return args.start_stage <= stage_idx <= args.end_stage


def discover_run_ids(log_dir):
    if not os.path.isdir(log_dir):
        return []
    run_ids = []
    for name in sorted(os.listdir(log_dir)):
        if os.path.isfile(os.path.join(log_dir, name, "run_meta.json")):
            run_ids.append(name)
    return run_ids


def load_pareto_df(results_dir):
    pareto_path = os.path.join(results_dir, "pareto_all.csv")
    if not os.path.isfile(pareto_path):
        raise FileNotFoundError(
            f"Missing required file '{pareto_path}'. Run stage 1 first or disable skipping."
        )
    return pd.read_csv(pareto_path)


def copy_report_assets(results_dir, report_dir="report"):
    figs_dir = os.path.join(results_dir, "figures")
    if not os.path.isdir(figs_dir):
        return
    ensure_dir(report_dir)
    out_figs = os.path.join(report_dir, "figures")
    ensure_dir(out_figs)
    for name in os.listdir(figs_dir):
        if name.lower().endswith((".png", ".csv", ".txt")):
            shutil.copy2(os.path.join(figs_dir, name), os.path.join(out_figs, name))


def run_pipeline(args):
    ensure_dir(args.results_dir)
    log_dir = os.path.join(args.results_dir, "logs")
    ensure_dir(log_dir)

    if args.auto_device and args.device == "cpu":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = list(range(args.num_seeds))
    completed_stages = set(load_pipeline_state(args.results_dir).get("completed", []))
    all_df = None
    run_ids = discover_run_ids(log_dir)

    print(
        f"[PIPELINE] Starting run | device={args.device} | seeds={seeds} | pop={args.pop_size} | "
        f"ngen={args.ngen} | eval_epochs={args.eval_epochs} | batch={args.batch_size} | workers={args.num_workers}",
        flush=True,
    )

    # 1) GA with runtime logging + checkpoints
    stage = 1
    if should_run_stage(args, stage):
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        elif args.skip_existing and os.path.isfile(os.path.join(args.results_dir, "pareto_all.csv")):
            all_df = load_pareto_df(args.results_dir)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (existing pareto_all.csv)", flush=True)
            mark_stage_complete(args.results_dir, stage)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Running NSGA-II search", flush=True)
            all_df, run_ids = save_fronts(
                seeds=seeds,
                outdir=args.results_dir,
                log_dir=log_dir,
                pop_size=args.pop_size,
                ngen=args.ngen,
                epochs=args.eval_epochs,
                batch_size=args.batch_size,
                val_size=args.val_size,
                num_workers=args.num_workers,
                checkpoint_every=args.checkpoint_every,
                device=args.device,
            )
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete | runs={run_ids}", flush=True)

    # 2) Stability
    stage = 2
    if should_run_stage(args, stage):
        if all_df is None:
            all_df = load_pareto_df(args.results_dir)
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        elif args.skip_existing and all(
            os.path.isfile(os.path.join(args.results_dir, name))
            for name in ("stability_igd.csv", "stability_hv2d.csv")
        ):
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (existing stability files)", flush=True)
            mark_stage_complete(args.results_dir, stage)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Computing stability metrics", flush=True)
            igd_df, _ = stability_metrics(all_df)
            igd_df.to_csv(os.path.join(args.results_dir, "stability_igd.csv"), index=False)
            hv_df = hv_projections(all_df)
            hv_df.to_csv(os.path.join(args.results_dir, "stability_hv2d.csv"), index=False)
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    # 3) Baseline
    stage = 3
    if should_run_stage(args, stage):
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        elif args.skip_existing and os.path.isfile(os.path.join(args.results_dir, "manual_baseline.csv")):
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (existing manual_baseline.csv)", flush=True)
            mark_stage_complete(args.results_dir, stage)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Running manual baseline", flush=True)
            max_combos = args.baseline_max_combos if args.baseline_max_combos > 0 else None
            run_manual_baseline(
                outdir=args.results_dir,
                device=args.device,
                seed=0,
                epochs=args.baseline_epochs,
                batch_size=args.batch_size,
                val_size=args.val_size,
                num_workers=args.num_workers,
                max_combos=max_combos,
            )
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    # 4) Retrain
    stage = 4
    if should_run_stage(args, stage):
        if all_df is None:
            all_df = load_pareto_df(args.results_dir)
        retrain_path = os.path.join(args.results_dir, f"retrain_{args.transfer_dataset}.csv")
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        elif args.skip_existing and os.path.isfile(retrain_path):
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (existing retrain file)", flush=True)
            mark_stage_complete(args.results_dir, stage)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Retraining selected models", flush=True)
            retrain_selected(
                all_df=all_df,
                outdir=args.results_dir,
                dataset=args.transfer_dataset,
                retrain_epochs=args.retrain_epochs,
                batch_size=args.batch_size,
                val_size=args.val_size,
                num_workers=args.num_workers,
                device=args.device,
                retrain_seeds=tuple(range(args.retrain_seeds)),
                k_each=args.retrain_k_each,
            )
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    # 5) Plots
    stage = 5
    if should_run_stage(args, stage):
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        else:
            if not run_ids:
                run_ids = discover_run_ids(log_dir)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Generating plots", flush=True)
            plots_main(results_dir=args.results_dir, transfer_dataset=args.transfer_dataset)
            plots_plus_main(results_dir=args.results_dir, log_dir=log_dir, run_ids=run_ids)
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    # 6) Architecture diagrams
    stage = 6
    if should_run_stage(args, stage):
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Drawing architecture diagrams", flush=True)
            generate_arch_diagrams(results_dir=args.results_dir)
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    # 7) Copy assets into report/figures
    stage = 7
    if should_run_stage(args, stage):
        if not args.copy_to_report:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (--copy-to-report not set)", flush=True)
        elif args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Copying report assets", flush=True)
            copy_report_assets(args.results_dir, report_dir=args.report_dir)
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    # 8) Auto LaTeX fragments + auto-filled report text
    stage = 8
    if should_run_stage(args, stage):
        if args.resume and stage in completed_stages:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} skipped (already completed)", flush=True)
        else:
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES}: Writing report fragments and markdown", flush=True)
            latex_generate(report_dir=args.report_dir, results_dir=args.results_dir, transfer_dataset=args.transfer_dataset)
            generate_markdown(
                results_dir=args.results_dir,
                outpath=os.path.join(args.report_dir, "REPORT_FINAL.md"),
                transfer_dataset=args.transfer_dataset,
            )
            mark_stage_complete(args.results_dir, stage)
            print(f"[PIPELINE] Stage {stage}/{TOTAL_STAGES} complete", flush=True)

    print("\nDONE (A+ pipeline)")
    print("Results:", args.results_dir)
    print("Report folder:", args.report_dir)


def build_argparser():
    p = argparse.ArgumentParser(description="A+ pipeline: GA -> stability -> baseline -> retrain -> plots -> diagrams -> LaTeX")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--auto-device", action="store_true", help="Use CUDA automatically when available.")
    p.add_argument("--pop-size", type=int, default=24)
    p.add_argument("--ngen", type=int, default=8)
    p.add_argument("--eval-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--num-seeds", type=int, default=3)

    p.add_argument("--transfer-dataset", default="cifar100", choices=["cifar100", "fashionmnist"])
    p.add_argument("--baseline-epochs", type=int, default=6)
    p.add_argument(
        "--baseline-max-combos",
        type=int,
        default=0,
        help="Limit baseline grid combinations; 0 means full grid.",
    )
    p.add_argument("--retrain-epochs", type=int, default=50)
    p.add_argument("--retrain-seeds", type=int, default=3)
    p.add_argument("--retrain-k-each", type=int, default=3)

    p.add_argument("--copy-to-report", action="store_true")
    p.add_argument("--report-dir", default="report")
    p.add_argument("--resume", action="store_true", help="Skip stages marked complete in results_dir/pipeline_state.json.")
    p.add_argument("--skip-existing", action="store_true", help="Skip heavy stages when expected output files already exist.")
    p.add_argument("--start-stage", type=int, default=1)
    p.add_argument("--end-stage", type=int, default=TOTAL_STAGES)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(args)
