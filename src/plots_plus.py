import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else None

def plot_convergence(log_dir, run_ids, outdir):
    ensure_dir(outdir)
    all_logs = []
    for rid in run_ids:
        df = read_jsonl(os.path.join(log_dir, rid, "log.jsonl"))
        if df is not None:
            df["run_id"] = rid
            all_logs.append(df)
    if not all_logs:
        return None
    conv = pd.concat(all_logs, ignore_index=True)
    conv.to_csv(os.path.join(outdir, "convergence_logs.csv"), index=False)

    fig = plt.figure()
    for rid, sdf in conv.groupby("run_id"):
        plt.plot(sdf["gen"], sdf["best_acc"], marker="o", label=rid)
    plt.xlabel("Generation")
    plt.ylabel("Best Pareto-front accuracy (proxy)")
    plt.title("Convergence: best front accuracy vs generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_convergence_best_acc.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure()
    for rid, sdf in conv.groupby("run_id"):
        plt.plot(sdf["gen"], sdf["front_size"], marker="o", label=rid)
    plt.xlabel("Generation")
    plt.ylabel("Front size")
    plt.title("Convergence: Pareto front size vs generation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_convergence_front_size.png"), dpi=200)
    plt.close(fig)

    return conv

def main(results_dir="results", log_dir=None, run_ids=None):
    outdir = os.path.join(results_dir, "figures")
    ensure_dir(outdir)
    if log_dir and os.path.isdir(log_dir):
        if run_ids is None:
            run_ids = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if run_ids:
            plot_convergence(log_dir, run_ids, outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()
    main(args.results_dir, args.log_dir)
