"""Oxford-IIIT Pet loading, 128x128 cache, and the fixed 80/20 split."""
import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
MANIFESTS = ROOT / "manifests"
SIZE = 128
SEED = 42
HF_URL = "https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data"


def _official_images(split: str):
    """Yield PIL images of the official split, ordered by image id (e.g. 'Abyssinian_1').
    Uses the Hugging Face mirror (timm/oxford-iiit-pet: train = official trainval 3,680, test = 3,669)
    because robots.ox.ac.uk served at ~30 KB/s; falls back to torchvision's official download."""
    hf = DATA / "oxford-iiit-pet-hf" / f"{'train' if split == 'trainval' else 'test'}.parquet"
    try:
        if not hf.exists():
            hf.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{HF_URL}/{hf.name.replace('.parquet', '')}-00000-of-00001.parquet", hf)
        rows = sorted(pq.read_table(hf, columns=["image", "image_id"]).to_pylist(), key=lambda r: r["image_id"])
        for r in rows:
            yield Image.open(io.BytesIO(r["image"]["bytes"]))
    except OSError:
        hf.unlink(missing_ok=True)
        for img, _ in OxfordIIITPet(DATA, split=split, download=True):
            yield img


def _load_official(split: str) -> torch.Tensor:
    """split = 'trainval' | 'test' -> uint8 [N,3,128,128]. Decoded + resized once, then cached (clean images only)."""
    path = CACHE / f"pets{SIZE}_{split}.pt"
    if path.exists():
        return torch.load(path)
    # Some files are grayscale/RGBA -> convert("RGB"). Direct resize; BICUBIC downscaling in PIL is antialiased.
    imgs = torch.stack([pil_to_tensor(img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
                        for img in _official_images(split)])
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(imgs, path)
    return imgs


def split_indices() -> dict:
    """80/20 train/val split of official trainval, seed 42. Saved to manifests/split.json and reused by Tasks 1-3."""
    path = MANIFESTS / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    n = len(_load_official("trainval"))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    k = int(0.8 * n)
    split = {"seed": SEED, "train": perm[:k], "val": perm[k:]}
    MANIFESTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(split))
    return split


def get_images(split: str) -> torch.Tensor:
    """'train' | 'val' | 'test' -> uint8 [N,3,128,128]."""
    if split == "test":
        return _load_official("test")
    return _load_official("trainval")[split_indices()[split]]
