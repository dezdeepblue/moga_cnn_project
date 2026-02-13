import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .model import CNNFromChromosome
from .train import train_epochs, accuracy

_TRANSFER_LOADERS_CACHE = {}


def make_loaders(
    dataset="cifar100",
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
    dataset = dataset.lower()
    if pin_memory is None:
        if device is not None:
            pin_memory = str(device).startswith("cuda")
        else:
            pin_memory = torch.cuda.is_available()

    key = (
        dataset,
        int(batch_size),
        int(val_size),
        int(num_workers),
        int(seed),
        str(data_root),
        bool(pin_memory),
        bool(persistent_workers),
    )
    if use_cache and key in _TRANSFER_LOADERS_CACHE:
        return _TRANSFER_LOADERS_CACHE[key]

    if dataset == "cifar100":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        print("[TRANSFER] Preparing CIFAR-100 dataset (download may take time on first run)...", flush=True)
        ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tfm_train)
        num_classes = 100
    elif dataset == "fashionmnist":
        tfm_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        tfm_test = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print("[TRANSFER] Preparing FashionMNIST dataset (download may take time on first run)...", flush=True)
        ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm_train)
        num_classes = 10
    else:
        raise ValueError("dataset must be 'cifar100' or 'fashionmnist'")

    train_ds, val_ds = random_split(
        ds,
        [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
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
        f"[TRANSFER] {dataset} ready | train={len(train_ds)} | val={len(val_ds)} | batch_size={batch_size}",
        flush=True,
    )

    out = (train_loader, val_loader, num_classes)
    if use_cache:
        _TRANSFER_LOADERS_CACHE[key] = out
    return out


def retrain_and_eval(
    chrom,
    dataset="cifar100",
    epochs=50,
    batch_size=128,
    val_size=5000,
    num_workers=2,
    device="cpu",
    seed=0,
):
    torch.manual_seed(seed)
    train_loader, val_loader, num_classes = make_loaders(
        dataset=dataset,
        batch_size=batch_size,
        val_size=val_size,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    model = CNNFromChromosome(chrom, in_ch=3, num_classes=num_classes).to(device)
    train_epochs(
        model,
        train_loader,
        chrom,
        epochs=epochs,
        device=device,
        progress_label=f"transfer {dataset} seed={seed}",
    )
    return accuracy(model, val_loader, device=device)
