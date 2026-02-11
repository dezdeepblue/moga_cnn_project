
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .model import CNNFromChromosome
from .train import train_epochs, accuracy

def make_loaders(dataset="cifar100", batch_size=128, val_size=5000, num_workers=2, seed=0):
    dataset = dataset.lower()
    if dataset == "cifar100":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
        ])
        print("[TRANSFER] Preparing CIFAR-100 dataset (download may take time on first run)...", flush=True)
        ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=tfm_train)
        num_classes = 100
    elif dataset == "fashionmnist":
        tfm_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        tfm_test = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        print("[TRANSFER] Preparing FashionMNIST dataset (download may take time on first run)...", flush=True)
        ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm_train)
        num_classes = 10
    else:
        raise ValueError("dataset must be 'cifar100' or 'fashionmnist'")

    train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))
    val_ds.dataset.transform = tfm_test
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(
        f"[TRANSFER] {dataset} ready | train={len(train_ds)} | val={len(val_ds)} | batch_size={batch_size}",
        flush=True,
    )
    return train_loader, val_loader, num_classes

def retrain_and_eval(chrom, dataset="cifar100", epochs=50, batch_size=128, device="cpu", seed=0):
    torch.manual_seed(seed)
    train_loader, val_loader, num_classes = make_loaders(dataset, batch_size=batch_size, seed=seed)
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
