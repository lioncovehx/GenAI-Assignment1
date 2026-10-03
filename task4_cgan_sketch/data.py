"""FS2K photo/sketch pairs: style labels, train/val split, 128x128 caches and paired augmentation.

  python -m task4_cgan_sketch.data    # builds the split and caches
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, resize

from common.data import CACHE, DATA, MANIFESTS, SEED

FS2K = DATA / "FS2K"
SIZE = 128
JITTER = 143  # resize to 143, random-crop 128: pix2pix's 286 -> 256 jitter scaled to our size [Isola17]
EXPECTED = {"train": 1058, "test": 1046}
OUT = Path(__file__).parent / "outputs"


def pairs(split: str) -> list[dict]:
    """split = 'train' | 'test' -> one dict per pair, in annotation order."""
    files = {p.relative_to(FS2K).with_suffix("").as_posix().lower(): p
             for p in FS2K.glob("*/*/*") if p.suffix.lower() in (".jpg", ".png")}
    rows = []
    for a in json.loads((FS2K / f"anno_{split}.json").read_text()):
        folder, name = a["image_name"].split("/")  # e.g. photo1/image0110
        photo_key = f"photo/{folder}/{name}".lower()
        sketch_key = f"sketch/{folder.replace('photo', 'sketch')}/{name.replace('image', 'sketch')}".lower()
        assert photo_key in files and sketch_key in files, f"missing pair for {a['image_name']}"
        rows.append({"photo": str(files[photo_key]), "sketch": str(files[sketch_key]), "style": a["style"],
                     "source": folder, "frontal_face": a["frontal_face"], "hair": a["hair"]})
    assert len(rows) == EXPECTED[split], (split, len(rows))
    return rows


def split_indices() -> dict:
    """85/15 train/val split of official train, stratified by style (style counts must stay balanced)."""
    path = MANIFESTS / "fs2k_split.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = pairs("train")
    tr, va = train_test_split(range(len(rows)), test_size=0.15, stratify=[r["style"] for r in rows], random_state=SEED)
    split = {"seed": SEED, "train": sorted(tr), "val": sorted(va)}
    path.write_text(json.dumps(split))
    return split


def _load(row: dict, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """-> photo uint8 [3,128,128], sketch uint8 [1,128,128]."""
    photo = pil_to_tensor(Image.open(row["photo"]).convert("RGB"))
    sk = Image.open(row["sketch"])
    if sk.mode == "RGBA":  # 8 sketches; nearly opaque, composite over white anyway
        sk = Image.alpha_composite(Image.new("RGBA", sk.size, "white"), sk)
    sketch = pil_to_tensor(sk.convert("L"))
    assert photo.shape[1:] == sketch.shape[1:], row  # each photo and its sketch have the same size
    if mode == "pad":
        h, w = photo.shape[1:]
        d = abs(h - w)
        pad = (d // 2, d - d // 2, 0, 0) if h > w else (0, 0, d // 2, d - d // 2)  # (left, right, top, bottom)
        photo = F.pad(photo[None].float(), pad, mode="replicate")[0].to(torch.uint8)
        sketch = F.pad(sketch, pad, value=255)  # white paper
    return (resize(photo, [SIZE, SIZE], antialias=True), resize(sketch, [SIZE, SIZE], antialias=True))


def cached(split: str, mode: str = "pad") -> dict:
    """split = 'train' | 'val' | 'test'; mode = 'pad' | 'resize'. Decoded once, then cached in data/cache/."""
    official = "test" if split == "test" else "train"
    path = CACHE / f"fs2k{SIZE}_{mode}_{official}.pt"
    if not path.exists():
        rows = pairs(official)
        photos, sketches = zip(*(_load(r, mode) for r in rows))
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save({"photo": torch.stack(photos), "sketch": torch.stack(sketches),
                    "style": torch.tensor([r["style"] for r in rows]), "rows": rows}, path)
    d = torch.load(path)
    if split == "test":
        return d
    idx = split_indices()[split]
    return {"photo": d["photo"][idx], "sketch": d["sketch"][idx], "style": d["style"][idx],
            "rows": [d["rows"][i] for i in idx]}


def paired_augment(photo: torch.Tensor, sketch: torch.Tensor):
    """One random flip + jitter crop, drawn ONCE and applied to both. Float tensors [C,128,128]."""
    i, j = torch.randint(0, JITTER - SIZE + 1, (2,)).tolist()
    flip = bool(torch.rand(()) < 0.5)
    out = []
    for t in (photo, sketch):
        t = F.interpolate(t[None], size=JITTER, mode="bilinear", align_corners=False)[0, :, i:i + SIZE, j:j + SIZE]
        out.append(t.flip(-1) if flip else t)
    return out[0], out[1]


class FS2KDataset(Dataset):
    """Returns (photo [3,128,128], sketch [1,128,128], style) in [-1, 1] (tanh range, pix2pix convention)."""

    def __init__(self, split: str, mode: str = "pad", augment: bool = False):
        d = cached(split, mode)
        self.photo, self.sketch, self.style, self.rows = d["photo"], d["sketch"], d["style"], d["rows"]
        self.augment = augment

    def __len__(self):
        return len(self.style)

    def __getitem__(self, i):
        photo, sketch = self.photo[i].float() / 127.5 - 1, self.sketch[i].float() / 127.5 - 1
        if self.augment:
            photo, sketch = paired_augment(photo, sketch)
        return photo, sketch, self.style[i]


def _show(t):  # [-1,1] tensor -> image for imshow
    return ((t + 1) / 2).clamp(0, 1).permute(1, 2, 0).squeeze(-1).numpy()


def _figures():
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    # 1) paired augmentation check: sketch strokes (dark pixels) overlaid in red on the augmented photo
    ds = FS2KDataset("train", augment=True)
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    for c, i in enumerate((0, 300, 600, 850)):
        torch.manual_seed(c)
        photo, sketch, style = ds[i]
        overlay = _show(photo).copy()
        overlay[_show(sketch[0:1]).squeeze() < 0.5] = (1, 0, 0)
        ax[0, c].imshow(_show(sketch), cmap="gray")
        ax[0, c].set_title(f"augmented sketch (style {int(style) + 1})", fontsize=7)
        ax[1, c].imshow(overlay)
        ax[1, c].set_title("strokes on augmented photo", fontsize=7)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "paired_augment_check.png", dpi=130)
    plt.close(fig)
    # 2) one non-square photo of each source, direct resize vs pad-then-resize
    rows = pairs("train")
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    for c, src in enumerate(("photo2", "photo3")):
        row = next(r for r in rows if r["source"] == src)
        for k, mode in enumerate(("resize", "pad")):
            photo, sketch = _load(row, mode)
            ax[k, 2 * c].imshow(photo.permute(1, 2, 0).numpy())
            ax[k, 2 * c + 1].imshow(sketch[0].numpy(), cmap="gray")
            ax[k, 2 * c].set_title(f"{src} photo, {mode}", fontsize=7)
            ax[k, 2 * c + 1].set_title(f"{src} sketch, {mode}", fontsize=7)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "resize_vs_pad.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for split in ("train", "test"):
        rows = pairs(split)
        print(f"{split}: {len(rows)} pairs, all files found")
        print("  style counts (style 1/2/3):", [Counter(r["style"] for r in rows)[s] for s in range(3)])
        print("  style x photo source:", dict(sorted(Counter((f"style{r['style'] + 1}", r["source"]) for r in rows).items())))
    sp = split_indices()
    train_rows = pairs("train")
    for name in ("train", "val"):
        print(f"{name} (after split): {len(sp[name])} pairs, styles",
              [Counter(train_rows[i]["style"] for i in sp[name])[s] for s in range(3)])
    for mode in ("pad", "resize"):
        for split in ("train", "test"):
            d = cached(split, mode)
            print(f"cache {mode}/{split}: photo {tuple(d['photo'].shape)} sketch {tuple(d['sketch'].shape)}")

    # self-check: augmenting a pair whose "sketch" IS the photo's grey version must keep them pixel-aligned
    x = torch.rand(3, SIZE, SIZE) * 2 - 1
    for s in range(20):
        torch.manual_seed(s)
        p, q = paired_augment(x, x.mean(0, keepdim=True))
        assert torch.allclose(p.mean(0, keepdim=True), q, atol=1e-5), "photo and sketch got different augmentations"
    print("paired augmentation self-check passed (20 random draws)")
    _figures()
    print("saved", OUT / "figures" / "paired_augment_check.png", "and", OUT / "figures" / "resize_vs_pad.png")
