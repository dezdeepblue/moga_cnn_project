
import torch
from src.chromosome import repair, random_chromosome
from src.model import CNNFromChromosome
from src.metrics import count_params, compute_flops

def test_model_forward_shape():
    chrom = repair(random_chromosome())
    m = CNNFromChromosome(chrom, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10)

def test_metrics_positive():
    chrom = repair(random_chromosome())
    m = CNNFromChromosome(chrom, num_classes=10)
    assert count_params(m) > 0
    assert compute_flops(m, device="cpu") > 0
