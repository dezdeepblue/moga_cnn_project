
import itertools
from .chromosome import repair
from .nsga2 import evaluate_individual

def manual_cnn_chrom():
    return repair({
        "n_conv": 4,
        "filters": [32, 64, 64, 128],
        "kernels": [3, 3, 3, 3],
        "use_bn": True,
        "dropout": 0.2,
        "activation": "relu",
        "optimizer": "adam",
        "lr": 1e-3,
    })

def grid_tune_training(
    base_chrom,
    activations,
    optimizers,
    lrs,
    epochs=6,
    batch_size=128,
    val_size=5000,
    num_workers=2,
    device="cpu",
    seed=0,
    max_combos=None,
):
    results = []
    combos = list(itertools.product(activations, optimizers, lrs))
    if max_combos is not None:
        combos = combos[:max(0, int(max_combos))]
    total = len(combos)
    if total == 0:
        return results
    for idx, (act, opt, lr) in enumerate(combos, start=1):
        print(f"[BASELINE] Training combo {idx}/{total}: act={act}, opt={opt}, lr={lr}", flush=True)
        chrom = dict(base_chrom)
        chrom["activation"] = act
        chrom["optimizer"] = opt
        chrom["lr"] = float(lr)
        chrom = repair(chrom)
        neg_acc, params, flops = evaluate_individual(
            chrom,
            epochs=epochs,
            batch_size=batch_size,
            val_size=val_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
            progress_label=f"baseline combo {idx}/{total}",
        )
        results.append((chrom, -neg_acc, params, flops))
    return results
