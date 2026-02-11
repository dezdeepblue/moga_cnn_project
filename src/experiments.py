
import json
import pandas as pd
from .nsga2 import run_nsga2

def front_to_df(front):
    rows = []
    for ind in front:
        neg_acc, params, flops = ind.fitness.values
        rows.append({
            "acc": -neg_acc,
            "params": int(params),
            "flops": int(flops),
            "n_conv": int(ind["n_conv"]),
            "use_bn": bool(ind["use_bn"]),
            "dropout": float(ind["dropout"]),
            "activation": ind["activation"],
            "optimizer": ind["optimizer"],
            "lr": float(ind["lr"]),
            "chrom_json": json.dumps(dict(ind)),
        })
    return pd.DataFrame(rows).sort_values(["acc"], ascending=False)

def run_multi_seeds(seeds=(0,1,2), **ga_kwargs):
    dfs = []
    for s in seeds:
        _pop, front, _run_id = run_nsga2(seed=s, **ga_kwargs)
        df = front_to_df(front)
        df["seed"] = s
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
