
from src.chromosome import random_chromosome, repair, crossover, mutate, FILTERS, KERNELS

def test_random_chromosome_valid():
    c = repair(random_chromosome())
    assert 2 <= c["n_conv"] <= 6
    assert len(c["filters"]) == c["n_conv"]
    assert len(c["kernels"]) == c["n_conv"]
    assert all(f in FILTERS for f in c["filters"])
    assert all(k in KERNELS for k in c["kernels"])
    assert 1e-4 <= c["lr"] <= 1e-1

def test_crossover_repairs_lengths():
    a = repair(random_chromosome()); b = repair(random_chromosome())
    a["n_conv"] = 6; a["filters"] = a["filters"][:2]
    a, b = crossover(a, b)
    assert len(a["filters"]) == a["n_conv"]
    assert len(a["kernels"]) == a["n_conv"]

def test_mutation_keeps_valid():
    c = repair(random_chromosome())
    c2 = mutate(dict(c), p=1.0)
    assert 2 <= c2["n_conv"] <= 6
    assert len(c2["filters"]) == c2["n_conv"]
