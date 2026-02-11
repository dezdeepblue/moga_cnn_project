import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _draw_arch(chrom, outpath, title=None):
    n = chrom["n_conv"]
    filters = chrom["filters"]
    kernels = chrom["kernels"]
    use_bn = chrom["use_bn"]
    act = chrom["activation"]
    drop = chrom["dropout"]

    fig = plt.figure(figsize=(12, 2.8))
    ax = plt.gca()
    ax.axis("off")

    x = 0.5
    y = 0.7
    w = 1.2
    h = 0.6
    dx = 1.35

    for i in range(n):
        label = f"Conv{i+1}\n{filters[i]}@{kernels[i]}x{kernels[i]}"
        if use_bn:
            label += "\nBN"
        label += f"\n{act}"
        rect = plt.Rectangle((x, y), w, h, fill=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=9)
        if (i+1) % 2 == 0:
            px = x + dx
            prect = plt.Rectangle((px, y), w*0.9, h, fill=False, linestyle="--")
            ax.add_patch(prect)
            ax.text(px + (w*0.9)/2, y + h/2, "MaxPool", ha="center", va="center", fontsize=9)
            x = px + dx
        else:
            x += dx

    rect = plt.Rectangle((x, y), w, h, fill=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f"GAP\nDrop={drop}", ha="center", va="center", fontsize=9)
    x += dx

    rect = plt.Rectangle((x, y), w, h, fill=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, "Linear\nClassifier", ha="center", va="center", fontsize=9)

    if title:
        plt.title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def generate_arch_diagrams(results_dir="results", outdir=None):
    if outdir is None:
        outdir = os.path.join(results_dir, "figures")
    ensure_dir(outdir)

    df = pd.read_csv(os.path.join(results_dir, "pareto_all.csv"))
    df = df.copy()
    df["score_knee"] = df["acc"] / (np.log10(df["params"]) + np.log10(df["flops"]) + 1e-9)
    reps = pd.concat([
        df.sort_values("params").head(1).assign(rep_tag="min_params"),
        df.sort_values("flops").head(1).assign(rep_tag="min_flops"),
        df.sort_values("acc", ascending=False).head(1).assign(rep_tag="max_acc"),
        df.sort_values("score_knee", ascending=False).head(1).assign(rep_tag="knee"),
    ]).drop_duplicates(subset=["chrom_json","rep_tag"])

    rows = []
    for _, r in reps.iterrows():
        chrom = json.loads(r["chrom_json"])
        fn = f"fig_arch_{r['rep_tag']}.png"
        title = f"{r['rep_tag']} | acc={r['acc']:.4f} | params={int(r['params'])} | flops={int(r['flops'])}"
        _draw_arch(chrom, os.path.join(outdir, fn), title=title)
        rows.append({"rep_tag": r["rep_tag"], "file": fn})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "arch_diagrams_index.csv"), index=False)
    return rows

if __name__ == "__main__":
    generate_arch_diagrams()
