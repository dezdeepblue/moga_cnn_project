
import torch
from thop import profile

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def compute_flops(model, input_shape=(1, 3, 32, 32), device="cpu") -> int:
    model = model.to(device)
    x = torch.randn(*input_shape).to(device)
    flops, _params = profile(model, inputs=(x,), verbose=False)
    return int(flops)
