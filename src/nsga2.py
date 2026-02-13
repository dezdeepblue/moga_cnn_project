import os
import json
import time
import pickle
import random
import hashlib
from typing import Optional, Tuple

import numpy as np
from deap import base, creator, tools

from .chromosome import random_chromosome, crossover, mutate, repair
from .model import CNNFromChromosome
from .metrics import count_params, compute_flops
from .data import cifar10_loaders
from .train import train_epochs, accuracy

# Fitness: minimize (-acc, params, flops)
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", dict, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: creator.Individual(repair(random_chromosome())))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selNSGA2)


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[no-work]"
    ratio = min(1.0, max(0.0, done / total))
    filled = int(width * ratio)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total} ({ratio * 100:5.1f}%)"


def _chrom_key(ind) -> str:
    return hashlib.md5(json.dumps(dict(ind), sort_keys=True).encode("utf-8")).hexdigest()


def evaluate_individual(
    ind,
    epochs=6,
    batch_size=128,
    val_size=5000,
    device="cpu",
    seed=0,
    progress_label=None,
    train_loader=None,
    val_loader=None,
    cache=None,
    flops_cache=None,
    batch_progress=False,
):
    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    key = _chrom_key(ind)
    if cache is not None and key in cache:
        return cache[key]

    if train_loader is None or val_loader is None:
        train_loader, val_loader = cifar10_loaders(
            batch_size=batch_size,
            val_size=val_size,
            seed=seed,
            device=device,
        )
    model = CNNFromChromosome(ind, num_classes=10)
    params = count_params(model)
    if flops_cache is not None and key in flops_cache:
        flops = flops_cache[key]
    else:
        flops = compute_flops(model, device=device)
        if flops_cache is not None:
            flops_cache[key] = flops

    if progress_label:
        print(f"[NSGA2] {progress_label} | training started", flush=True)
    train_epochs(
        model,
        train_loader,
        ind,
        epochs=epochs,
        device=device,
        progress_label=progress_label if batch_progress else None,
    )
    val_acc = accuracy(model, val_loader, device=device)
    if progress_label:
        print(f"[NSGA2] {progress_label} | validation acc={val_acc:.4f}", flush=True)

    out = (-val_acc, params, flops)
    if cache is not None:
        cache[key] = out
    return out

def _ensure_dir(p: Optional[str]):
    if p:
        os.makedirs(p, exist_ok=True)

def _save_checkpoint(log_dir: str, run_id: str, gen: int, pop, front):
    ckpt_dir = os.path.join(log_dir, run_id, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {"gen": gen, "timestamp": time.time(), "pop": pop, "front": front}
    with open(os.path.join(ckpt_dir, f"gen_{gen:04d}.pkl"), "wb") as f:
        pickle.dump(payload, f)

def _append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def run_nsga2(
    pop_size=24,
    ngen=8,
    cx_prob=0.9,
    mut_prob=0.2,
    epochs=6,
    batch_size=128,
    val_size=5000,
    device="cpu",
    seed=0,
    log_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    checkpoint_every: int = 1,
    batch_progress: bool = False,
) -> Tuple[list, list, str]:
    """NSGA-II with optional runtime logging + checkpoints."""
    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    if run_id is None:
        run_id = f"nsga2_seed{seed}_{int(time.time())}"

    log_jsonl = None
    if log_dir:
        _ensure_dir(os.path.join(log_dir, run_id))
        meta = {
            "run_id": run_id,
            "seed": seed,
            "pop_size": pop_size,
            "ngen": ngen,
            "cx_prob": cx_prob,
            "mut_prob": mut_prob,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_size": val_size,
            "device": device,
            "started_at": time.time(),
        }
        with open(os.path.join(log_dir, run_id, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log_jsonl = os.path.join(log_dir, run_id, "log.jsonl")

    pop = toolbox.population(n=pop_size)
    train_loader, val_loader = cifar10_loaders(
        batch_size=batch_size,
        val_size=val_size,
        seed=seed,
        device=device,
    )
    eval_cache = {}
    flops_cache = {}
    total_evals = pop_size * (ngen + 1)
    eval_done = 0
    print(f"[NSGA2] Seed={seed} | population={pop_size} | generations={ngen} | total evaluations={total_evals}")
    print(f"[NSGA2] Initial population evaluation: {_progress_bar(eval_done, total_evals)}", flush=True)
    t0 = time.time()
    for idx, ind in enumerate(pop, start=1):
        ind.fitness.values = evaluate_individual(
            ind,
            epochs,
            batch_size,
            val_size,
            device,
            seed,
            progress_label=f"seed{seed} init ind {idx}/{pop_size}",
            train_loader=train_loader,
            val_loader=val_loader,
            cache=eval_cache,
            flops_cache=flops_cache,
            batch_progress=batch_progress,
        )
        eval_done += 1
        print(
            f"\r[NSGA2] Initial population evaluation: {_progress_bar(eval_done, total_evals)}",
            end="",
            flush=True,
        )
    print("", flush=True)
    pop = toolbox.select(pop, len(pop))
    front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

    if log_dir and checkpoint_every and 0 % checkpoint_every == 0:
        _save_checkpoint(log_dir, run_id, 0, pop, front)

    if log_jsonl:
        best_acc = float(np.max([-i.fitness.values[0] for i in front]))
        _append_jsonl(log_jsonl, {
            "gen": 0,
            "elapsed_s": time.time() - t0,
            "front_size": len(front),
            "best_acc": best_acc,
            "mean_acc": float(np.mean([-i.fitness.values[0] for i in pop])),
        })

    for gen in range(1, ngen + 1):
        g0 = time.time()
        print(f"[NSGA2] Generation {gen}/{ngen} started", flush=True)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [creator.Individual(ind.copy()) for ind in offspring]

        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring) and random.random() < cx_prob:
                a, b = crossover(dict(offspring[i]), dict(offspring[i+1]))
                offspring[i].update(a)
                offspring[i+1].update(b)

            offspring[i].update(mutate(dict(offspring[i]), p=mut_prob))
            if i + 1 < len(offspring):
                offspring[i+1].update(mutate(dict(offspring[i+1]), p=mut_prob))

        for idx, ind in enumerate(offspring, start=1):
            ind.fitness.values = evaluate_individual(
                ind,
                epochs,
                batch_size,
                val_size,
                device,
                seed,
                progress_label=f"seed{seed} gen{gen} ind {idx}/{len(offspring)}",
                train_loader=train_loader,
                val_loader=val_loader,
                cache=eval_cache,
                flops_cache=flops_cache,
                batch_progress=batch_progress,
            )
            eval_done += 1
            print(
                f"\r[NSGA2] Overall evaluations: {_progress_bar(eval_done, total_evals)}",
                end="",
                flush=True,
            )
        print("", flush=True)

        pop = toolbox.select(pop + offspring, pop_size)
        front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        gen_best_acc = float(np.max([-i.fitness.values[0] for i in front]))
        print(
            f"[NSGA2] Generation {gen}/{ngen} done | front_size={len(front)} | best_acc={gen_best_acc:.4f} | gen_time={time.time() - g0:.1f}s",
            flush=True,
        )

        if log_dir and checkpoint_every and gen % checkpoint_every == 0:
            _save_checkpoint(log_dir, run_id, gen, pop, front)

        if log_jsonl:
            _append_jsonl(log_jsonl, {
                "gen": gen,
                "elapsed_s": time.time() - g0,
                "front_size": len(front),
                "best_acc": gen_best_acc,
                "mean_acc": float(np.mean([-i.fitness.values[0] for i in pop])),
            })

    if log_dir:
        meta_path = os.path.join(log_dir, run_id, "run_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["finished_at"] = time.time()
        meta["duration_s"] = meta["finished_at"] - meta["started_at"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"[NSGA2] Seed={seed} completed in {time.time() - t0:.1f}s", flush=True)

    return pop, front, run_id
