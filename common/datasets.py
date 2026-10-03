"""PyTorch datasets: runtime corruption (train), manifest replay (val/test), balanced batches (Tasks 2-3)."""
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from common import corruptions as C
from common.data import MANIFESTS, get_images


def _rng() -> np.random.Generator:
    # Draw from torch's RNG: DataLoader seeds it differently per worker and per epoch,
    # and torch.manual_seed(...) makes the whole run reproducible.
    return np.random.default_rng(int(torch.randint(2**62, (1,))))


class CleanDataset(Dataset):
    """Clean float images in [0,1]. Use with balanced_collate."""

    def __init__(self, split: str = "train"):
        self.images = get_images(split)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i].float() / 255


class RuntimeCorruptionDataset(CleanDataset):
    """New corruption type + severity every time an image is loaded.
    labels: restrict to some conditions, e.g. (1,) to train the salt-and-pepper specialist.
    Returns (corrupted, clean, label)."""

    def __init__(self, split: str = "train", labels=(0, 1, 2, 3)):
        super().__init__(split)
        self.labels = labels

    def __getitem__(self, i):
        x = super().__getitem__(i)
        rng = _rng()
        params = C.make_params(int(rng.choice(self.labels)), rng)
        return C.apply(x, params), x, params["label"]


class ManifestDataset(CleanDataset):
    """Replays manifests/{val,test}.json exactly. Returns (corrupted, clean, label, row_index)."""

    def __init__(self, name: str, labels=(0, 1, 2, 3)):
        super().__init__(name)
        rows = json.loads((MANIFESTS / f"{name}.json").read_text())
        self.rows = [r for r in rows if r["label"] in labels]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        r = self.rows[j]
        x = super().__getitem__(r["image"])
        return C.apply(x, r), x, r["label"], j


def balanced_collate(batch: list) -> tuple:
    """Exactly B/4 images per condition in every batch. Corruption still happens in the loader."""
    b = len(batch)
    assert b % 4 == 0, "batch size must be a multiple of 4 for balanced batches"
    labels = torch.arange(4).repeat(b // 4)[torch.randperm(b)]
    rng = _rng()
    clean = torch.stack(batch)
    corrupted = torch.stack([C.apply(x, C.make_params(int(y), rng)) for x, y in zip(batch, labels)])
    return corrupted, clean, labels
