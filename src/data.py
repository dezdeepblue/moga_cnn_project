
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


_CIFAR10_CACHE = {}


def cifar10_loaders(
    batch_size=128,
    val_size=5000,
    num_workers=2,
    seed=0,
    data_root="./data",
    pin_memory=None,
    persistent_workers=True,
    use_cache=True,
    device=None,
):
    if pin_memory is None:
        if device is not None:
            pin_memory = str(device).startswith("cuda")
        else:
            pin_memory = torch.cuda.is_available()
    key = (
        int(batch_size),
        int(val_size),
        int(num_workers),
        int(seed),
        str(data_root),
        bool(pin_memory),
        bool(persistent_workers),
    )
    if use_cache and key in _CIFAR10_CACHE:
        return _CIFAR10_CACHE[key]

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261)),
    ])

    print("[DATA] Preparing CIFAR-10 dataset (download may take time on first run)...", flush=True)
    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
    train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))
    val_ds.dataset.transform = tfm_test

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    print(
        f"[DATA] CIFAR-10 ready | train={len(train_ds)} | val={len(val_ds)} | batch_size={batch_size}",
        flush=True,
    )
    if use_cache:
        _CIFAR10_CACHE[key] = (train_loader, val_loader)
    return train_loader, val_loader
