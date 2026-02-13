
import os
import json
import numpy as np
import pandas as pd

from .nsga2 import run_nsga2
from .experiments import front_to_df
from .mo_utils import pareto_filter, normalize_points, hypervolume_2d, igd
from .baselines import manual_cnn_chrom, grid_tune_training
from .transfer import retrain_and_eval

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_fronts(seeds, outdir, **ga_kwargs):
    ensure_dir(outdir)
    fronts = []
    run_ids = []
    for s in seeds:
        print(f"[RUN_ALL] Running NSGA-II for seed {s}", flush=True)
        _pop, front, run_id = run_nsga2(seed=s, **ga_kwargs)
        df = front_to_df(front)
        df["seed"] = s
        df.to_csv(os.path.join(outdir, f"pareto_seed{s}.csv"), index=False)
        fronts.append(df)
        print(f"[RUN_ALL] Saved Pareto front for seed {s} ({len(df)} rows)", flush=True)
        run_ids.append(run_id)
    all_df = pd.concat(fronts, ignore_index=True)
    all_df.to_csv(os.path.join(outdir, "pareto_all.csv"), index=False)
    print(f"[RUN_ALL] Saved merged Pareto fronts ({len(all_df)} rows)", flush=True)
    return all_df, run_ids

def stability_metrics(all_df):
    df = all_df.copy()
    df["neg_acc"] = -df["acc"]
    points = df[["neg_acc","params","flops"]].to_numpy()
    ref = pareto_filter(points)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ref_n = normalize_points(ref, mins, maxs)

    rows = []
    for seed, sdf in df.groupby("seed"):
        A = sdf[["neg_acc","params","flops"]].to_numpy()
        A = pareto_filter(A)
        A_n = normalize_points(A, mins, maxs)
        rows.append({"seed": seed, "IGD": igd(A_n, ref_n)})
    return pd.DataFrame(rows), ref

def hv_projections(all_df):
    df = all_df.copy()
    df["neg_acc"] = -df["acc"]
    pts = df[["neg_acc","params","flops"]].to_numpy()
    mins = pts.min(axis=0); maxs = pts.max(axis=0)
    ref2 = np.array([1.0, 1.0])

    rows = []
    for seed, sdf in df.groupby("seed"):
        A = sdf[["neg_acc","params","flops"]].to_numpy()
        A = pareto_filter(A)
        A_n = normalize_points(A, mins, maxs)

        p1 = pareto_filter(A_n[:, [0,1]])
        hv1 = hypervolume_2d(p1, ref2)

        p2 = pareto_filter(A_n[:, [0,2]])
        hv2 = hypervolume_2d(p2, ref2)

        rows.append({"seed": seed, "HV_acc_params": hv1, "HV_acc_flops": hv2})
    return pd.DataFrame(rows)

def run_manual_baseline(
    outdir,
    device="cpu",
    seed=0,
    epochs=6,
    batch_size=128,
    val_size=5000,
    num_workers=2,
    max_combos=None,
):
    ensure_dir(outdir)
    base = manual_cnn_chrom()
    activations = ["relu","elu","gelu"]
    optimizers = ["sgd","adam","rmsprop"]
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    print("[RUN_ALL] Baseline grid search started", flush=True)
    res = grid_tune_training(
        base,
        activations,
        optimizers,
        lrs,
        epochs=epochs,
        batch_size=batch_size,
        val_size=val_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
        max_combos=max_combos,
    )
    rows = []
    for chrom, acc, params, flops in res:
        rows.append({"acc": float(acc), "params": int(params), "flops": int(flops),
                     "activation": chrom["activation"], "optimizer": chrom["optimizer"], "lr": float(chrom["lr"])})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "manual_baseline.csv"), index=False)
    print(f"[RUN_ALL] Baseline grid search complete ({len(rows)} combinations)", flush=True)

def retrain_selected(all_df, outdir, dataset="cifar100", retrain_epochs=50,
                    batch_size=128, val_size=5000, num_workers=2,
                    device="cpu", retrain_seeds=(0,1,2), k_each=3):
    ensure_dir(outdir)
    df = all_df.copy()
    picks = pd.concat([
        df.sort_values("acc", ascending=False).head(k_each),
        df.sort_values("params", ascending=True).head(k_each),
        df.sort_values("flops", ascending=True).head(k_each),
    ]).drop_duplicates(subset=["chrom_json"]).reset_index(drop=True)

    rows = []
    total_jobs = len(picks) * len(retrain_seeds)
    done_jobs = 0
    print(
        f"[RUN_ALL] Retraining started | picks={len(picks)} | seeds={list(retrain_seeds)} | total runs={total_jobs}",
        flush=True,
    )
    for i, r in picks.iterrows():
        chrom = json.loads(r["chrom_json"])
        scores = []
        for s in retrain_seeds:
            print(
                f"[RUN_ALL] Retrain pick {int(i)+1}/{len(picks)} on {dataset} | seed={s} | {done_jobs+1}/{total_jobs}",
                flush=True,
            )
            scores.append(
                retrain_and_eval(
                    chrom,
                    dataset=dataset,
                    epochs=retrain_epochs,
                    batch_size=batch_size,
                    val_size=val_size,
                    num_workers=num_workers,
                    device=device,
                    seed=s,
                )
            )
            done_jobs += 1
        rows.append({
            "pick_id": int(i),
            "region_tag": "selected",
            "cifar10_val_acc_proxy": float(r["acc"]),
            "params": int(r["params"]),
            "flops": int(r["flops"]),
            "target_dataset": dataset,
            "target_acc_mean": float(np.mean(scores)),
            "target_acc_std": float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0),
            "activation": r["activation"],
            "optimizer": r["optimizer"],
            "lr": float(r["lr"]),
            "chrom_json": r["chrom_json"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, f"retrain_{dataset}.csv"), index=False)
    print(f"[RUN_ALL] Retraining complete. Saved {len(out)} rows", flush=True)
    return out
