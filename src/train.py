
import torch
import torch.nn.functional as F


def make_optimizer(model, chrom):
    lr = chrom["lr"]
    if chrom["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if chrom["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if chrom["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
    raise ValueError(f"Unknown optimizer: {chrom['optimizer']}")


def train_epochs(
    model,
    train_loader,
    chrom,
    epochs,
    device="cpu",
    progress_label=None,
    progress_every=50,
    amp=True,
):
    model.to(device)
    opt = make_optimizer(model, chrom)
    model.train()
    total_batches = len(train_loader)
    use_amp = bool(amp and str(device).startswith("cuda") and torch.cuda.is_available())
    if use_amp:
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch_idx in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=use_amp)
            y = y.to(device, non_blocking=use_amp)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if progress_label and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == total_batches):
                print(
                    f"\r[TRAIN] {progress_label} | epoch {epoch_idx+1}/{epochs} | batch {batch_idx}/{total_batches}",
                    end="",
                    flush=True,
                )
        if progress_label:
            print("", flush=True)

@torch.no_grad()
def accuracy(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    use_amp = bool(str(device).startswith("cuda") and torch.cuda.is_available())
    for x, y in loader:
        x = x.to(device, non_blocking=use_amp)
        y = y.to(device, non_blocking=use_amp)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)
