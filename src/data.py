
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def cifar10_loaders(batch_size=128, val_size=5000, num_workers=2, seed=0):
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
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))
    val_ds.dataset.transform = tfm_test

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(
        f"[DATA] CIFAR-10 ready | train={len(train_ds)} | val={len(val_ds)} | batch_size={batch_size}",
        flush=True,
    )
    return train_loader, val_loader
