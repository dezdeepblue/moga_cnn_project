
import random
import math
from typing import Dict

FILTERS = [16, 32, 48, 64, 96, 128]
KERNELS = [3, 5]
ACTIVATIONS = ["relu", "leaky_relu", "elu", "gelu"]
OPTIMIZERS = ["sgd", "adam", "rmsprop"]

def log_uniform(a=1e-4, b=1e-1) -> float:
    return 10 ** random.uniform(math.log10(a), math.log10(b))

def random_chromosome() -> Dict:
    n_conv = random.randint(2, 6)
    return {
        "n_conv": n_conv,
        "filters": [random.choice(FILTERS) for _ in range(n_conv)],
        "kernels": [random.choice(KERNELS) for _ in range(n_conv)],
        "use_bn": random.choice([True, False]),
        "dropout": random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "activation": random.choice(ACTIVATIONS),
        "optimizer": random.choice(OPTIMIZERS),
        "lr": log_uniform(1e-4, 1e-1),
    }

def repair(chrom: Dict) -> Dict:
    n = int(chrom.get("n_conv", 4))
    chrom["n_conv"] = max(2, min(6, n))
    n = chrom["n_conv"]

    chrom["filters"] = (chrom.get("filters", [])[:n] +
                        [random.choice(FILTERS) for _ in range(max(0, n - len(chrom.get("filters", []))))])
    chrom["kernels"] = (chrom.get("kernels", [])[:n] +
                        [random.choice(KERNELS) for _ in range(max(0, n - len(chrom.get("kernels", []))))])

    chrom["filters"] = [f if f in FILTERS else random.choice(FILTERS) for f in chrom["filters"]]
    chrom["kernels"] = [k if k in KERNELS else random.choice(KERNELS) for k in chrom["kernels"]]

    chrom["use_bn"] = bool(chrom.get("use_bn", True))
    chrom["dropout"] = float(chrom.get("dropout", 0.0))
    chrom["dropout"] = min(0.5, max(0.0, chrom["dropout"]))

    chrom["activation"] = chrom.get("activation", "relu")
    if chrom["activation"] not in ACTIVATIONS:
        chrom["activation"] = random.choice(ACTIVATIONS)

    chrom["optimizer"] = chrom.get("optimizer", "adam")
    if chrom["optimizer"] not in OPTIMIZERS:
        chrom["optimizer"] = random.choice(OPTIMIZERS)

    chrom["lr"] = float(chrom.get("lr", 1e-3))
    chrom["lr"] = min(1e-1, max(1e-4, chrom["lr"]))
    return chrom

def crossover(a: Dict, b: Dict, p=0.5):
    for k in list(a.keys()):
        if random.random() < p:
            a[k], b[k] = b[k], a[k]
    return repair(a), repair(b)

def mutate(chrom: Dict, p=0.2) -> Dict:
    if random.random() < p: chrom["n_conv"] = random.randint(2, 6)
    n = chrom["n_conv"]
    if random.random() < p: chrom["filters"] = [random.choice(FILTERS) for _ in range(n)]
    if random.random() < p: chrom["kernels"] = [random.choice(KERNELS) for _ in range(n)]
    if random.random() < p: chrom["use_bn"] = random.choice([True, False])
    if random.random() < p: chrom["dropout"] = random.choice([0.0,0.1,0.2,0.3,0.4,0.5])
    if random.random() < p: chrom["activation"] = random.choice(ACTIVATIONS)
    if random.random() < p: chrom["optimizer"] = random.choice(OPTIMIZERS)
    if random.random() < p: chrom["lr"] = log_uniform(1e-4, 1e-1)
    return repair(chrom)
