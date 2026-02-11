
import torch.nn as nn

def make_activation(name: str) -> nn.Module:
    if name == "relu": return nn.ReLU()
    if name == "leaky_relu": return nn.LeakyReLU(0.1)
    if name == "elu": return nn.ELU()
    if name == "gelu": return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

class CNNFromChromosome(nn.Module):
    def __init__(self, chrom, in_ch=3, num_classes=10):
        super().__init__()
        layers = []
        ch = in_ch
        use_bn = chrom["use_bn"]
        act = chrom["activation"]

        for i in range(chrom["n_conv"]):
            out_ch = chrom["filters"][i]
            k = chrom["kernels"][i]
            pad = k // 2
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=k, padding=pad, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(make_activation(act))
            if (i + 1) % 2 == 0:
                layers.append(nn.MaxPool2d(2))
            ch = out_ch

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(chrom["dropout"])
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2,3))
        x = self.dropout(x)
        return self.classifier(x)
