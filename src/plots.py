
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def spearman_corr(x, y):
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = (np.sqrt((xr**2).sum()) * np.sqrt((yr**2).sum()))
    return float((xr * yr).sum() / denom) if denom > 0 else 0.0

def dominant_pairs_table(pareto_df, out_csv):
    tab = (pareto_df.groupby(["activation","optimizer"]).size()
           .reset_index(name="count").sort_values("count", ascending=False))
    tab["share"] = tab["count"] / max(1, tab["count"].sum())
    tab.to_csv(out_csv, index=False)
    return tab

def plot_scatter(x, y, outpath, title, xlabel, ylabel, alpha=0.8):
    fig = plt.figure()
    plt.scatter(x, y, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_overlay_fronts(all_df, outpath):
    fig = plt.figure()
    for seed, sdf in all_df.groupby("seed"):
        plt.scatter(np.log10(sdf["params"]), sdf["acc"], alpha=0.8, label=f"seed={seed}")
    plt.xlabel("log10(Params)")
    plt.ylabel("CIFAR-10 Val Accuracy (proxy)")
    plt.title("Overlay: Pareto fronts across GA seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_bar(df, xcol, ycol, outpath, title, xlabel, ylabel):
    fig = plt.figure(figsize=(8,4))
    plt.bar(df[xcol].astype(str), df[ycol].to_numpy())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_retrain(retrain_df, outpath):
    fig = plt.figure(figsize=(9,4))
    x = np.arange(len(retrain_df))
    y = retrain_df["target_acc_mean"].to_numpy()
    yerr = retrain_df["target_acc_std"].to_numpy()
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.xlabel("Selected Model Index")
    plt.ylabel("Target Val Accuracy")
    plt.title(f"Generalization on {retrain_df['target_dataset'].iloc[0]} (mean±std)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def select_representatives(df):
    df = df.copy()
    df["score_knee"] = df["acc"] / (np.log10(df["params"]) + np.log10(df["flops"]) + 1e-9)
    reps = pd.concat([
        df.sort_values("params").head(1).assign(rep_tag="min_params"),
        df.sort_values("flops").head(1).assign(rep_tag="min_flops"),
        df.sort_values("acc", ascending=False).head(1).assign(rep_tag="max_acc"),
        df.sort_values("score_knee", ascending=False).head(1).assign(rep_tag="knee"),
    ]).drop_duplicates(subset=["chrom_json","rep_tag"])
    return reps

def main(results_dir="results", transfer_dataset="cifar100"):
    outdir = os.path.join(results_dir, "figures")
    ensure_dir(outdir)

    all_df = pd.read_csv(os.path.join(results_dir, "pareto_all.csv"))

    # Pareto projections
    plot_scatter(np.log10(all_df["params"]), all_df["acc"], os.path.join(outdir, "fig_acc_params.png"),
                 "Pareto: Accuracy vs Params", "log10(Params)", "CIFAR-10 Val Accuracy (proxy)")
    plot_scatter(np.log10(all_df["flops"]), all_df["acc"], os.path.join(outdir, "fig_acc_flops.png"),
                 "Pareto: Accuracy vs FLOPs", "log10(FLOPs)", "CIFAR-10 Val Accuracy (proxy)")
    plot_overlay_fronts(all_df, os.path.join(outdir, "fig_overlay_fronts.png"))

    # Manual baseline overlay if available
    mb = os.path.join(results_dir, "manual_baseline.csv")
    if os.path.exists(mb):
        manual = pd.read_csv(mb)
        fig = plt.figure()
        plt.scatter(np.log10(all_df["params"]), all_df["acc"], alpha=0.9, label="Joint GA Pareto")
        plt.scatter(np.log10(manual["params"]), manual["acc"], alpha=0.6, label="Manual baseline")
        plt.xlabel("log10(Params)")
        plt.ylabel("CIFAR-10 Val Accuracy (proxy)")
        plt.title("Joint GA Pareto vs Manual Baseline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fig_joint_vs_manual.png"), dpi=200)
        plt.close(fig)

    # Q2 dominant pairs
    tab = dominant_pairs_table(all_df, os.path.join(outdir, "q2_pairs_overall.csv"))
    tab_top = tab.head(10).copy()
    tab_top["pair"] = tab_top["activation"] + "/" + tab_top["optimizer"]
    plot_bar(tab_top, "pair", "share", os.path.join(outdir, "fig_q2_pairs.png"),
             "Dominant (Activation/Optimizer) on Pareto", "pair", "share")

    # Q3 Spearman and plot
    rho = spearman_corr(np.log10(all_df["params"]), all_df["acc"])
    with open(os.path.join(outdir, "q3_spearman.txt"), "w") as f:
        f.write(f"Spearman rho(log10(params), acc) = {rho:.4f}\n")
    plot_scatter(np.log10(all_df["params"]), all_df["acc"], os.path.join(outdir, "fig_q3_acc_vs_logparams.png"),
                 "Q3: Acc vs log10(Params)", "log10(Params)", "Accuracy", alpha=0.6)

    # Q6 stability plots if available
    hv_path = os.path.join(results_dir, "stability_hv2d.csv")
    igd_path = os.path.join(results_dir, "stability_igd.csv")
    if os.path.exists(hv_path):
        hv = pd.read_csv(hv_path)
        plot_bar(hv, "seed", "HV_acc_params", os.path.join(outdir, "fig_stability_hv_acc_params.png"),
                 "Stability: HV (Acc-Params projection)", "seed", "HV")
        plot_bar(hv, "seed", "HV_acc_flops", os.path.join(outdir, "fig_stability_hv_acc_flops.png"),
                 "Stability: HV (Acc-FLOPs projection)", "seed", "HV")
        hv.describe().to_csv(os.path.join(outdir, "q6_hv_summary.csv"))
    if os.path.exists(igd_path):
        igd = pd.read_csv(igd_path)
        plot_bar(igd, "seed", "IGD", os.path.join(outdir, "fig_stability_igd.png"),
                 "Stability: IGD (normalized 3D, lower better)", "seed", "IGD")
        igd.describe().to_csv(os.path.join(outdir, "q6_igd_summary.csv"))

    # Retraining plot if exists
    retr_path = os.path.join(results_dir, f"retrain_{transfer_dataset}.csv")
    if os.path.exists(retr_path):
        retr = pd.read_csv(retr_path)
        plot_retrain(retr, os.path.join(outdir, f"fig_retrain_{transfer_dataset}.png"))

    # Appendix representative table
    reps = select_representatives(all_df)
    reps[["rep_tag","acc","params","flops","n_conv","use_bn","dropout","activation","optimizer","lr"]].to_csv(
        os.path.join(outdir, "appendix_representative_models.csv"), index=False
    )

    print("Saved figures/tables to:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--transfer-dataset", default="cifar100", choices=["cifar100","fashionmnist"])
    args = ap.parse_args()
    main(args.results_dir, args.transfer_dataset)
