"""Corruption classifier: image in [0,1] -> 4 logits (clean, salt, blur, occlusion).

Self-check: python -m task2_hard_routing.classifier
"""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

N_CLASSES = 4


class CorruptionCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), dropout=0.2, pool="max"):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            stride = 2 if pool == "stride" else 1
            layers += [nn.Conv2d(c_in, c, 3, stride, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            if pool == "max":
                layers.append(nn.MaxPool2d(2))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, N_CLASSES))

    def forward(self, x):
        return self.head(self.features(x))


class ResNet18Classifier(nn.Module):
    """ImageNet weights expect ImageNet-normalised inputs, so the normalisation lives inside the model
    (the app and ONNX then feed [0,1] images to every classifier alike)."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, N_CLASSES))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build(cfg: dict) -> nn.Module:
    if cfg["arch"] == "resnet18":
        return ResNet18Classifier(cfg["dropout"])
    return CorruptionCNN(tuple(int(c) for c in cfg["channels"].split("-")), cfg["dropout"], cfg["pool"])


if __name__ == "__main__":
    x = torch.rand(2, 3, 128, 128)
    for channels in ("16-32-64-128", "32-64-128-256", "32-64-128-128-256"):
        for pool in ("max", "stride"):
            m = build(dict(arch="cnn", channels=channels, dropout=0.2, pool=pool))
            assert m(x).shape == (2, N_CLASSES)
            print(f"cnn {channels:18s} pool={pool:6s} params={sum(p.numel() for p in m.parameters()):>10,}")
    m = build(dict(arch="resnet18", dropout=0.2))  # downloads ImageNet weights (~45 MB) on first use
    assert m(x).shape == (2, N_CLASSES)
    print(f"resnet18{'':29s} params={sum(p.numel() for p in m.parameters()):>10,}")
    print("classifier self-check OK")
