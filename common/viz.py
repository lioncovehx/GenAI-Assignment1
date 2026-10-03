"""Figures: restoration grids with error maps, and the corruption overview for the report.

Run: python -m common.viz   -> common/outputs/corruption_grid.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import corruptions as C


def _img(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def restoration_grid(clean, corrupted, restored, path, titles=None, err_max=0.5):
    """Rows: clean | corrupted | restored | |error| map (mean over channels, shared colour scale)."""
    n = len(clean)
    fig, ax = plt.subplots(n, 4, figsize=(8, 2 * n), squeeze=False)
    for i in range(n):
        err = (restored[i] - clean[i]).abs().mean(0).cpu().numpy()
        for j, (im, name) in enumerate(((clean[i], "clean"), (corrupted[i], "input"), (restored[i], "output"))):
            ax[i, j].imshow(_img(im))
            ax[i, j].set_title(name if i == 0 else "", fontsize=8)
        ax[i, 3].imshow(err, cmap="inferno", vmin=0, vmax=err_max)
        ax[i, 3].set_title("|error|" if i == 0 else "", fontsize=8)
        if titles:
            ax[i, 0].set_ylabel(titles[i], fontsize=7)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def corruption_overview(images: torch.Tensor, path):
    """Rows = conditions, columns = the three fixed test levels (clean shown once)."""
    fig, ax = plt.subplots(4, 3, figsize=(6, 8))
    rng = np.random.default_rng(0)
    for col in range(3):
        x = images[col].float() / 255
        ax[0, col].imshow(_img(x))
        for label in (1, 2, 3):
            level = C.TEST_LEVELS[label][col]
            ax[label, col].imshow(_img(C.apply(x, C.make_params(label, rng, level))))
            ax[label, col].set_title(f"{C.LABELS[label]} {C.SEVERITIES[col]}\n{level}", fontsize=6)
    ax[0, 0].set_title("clean", fontsize=7)
    for a in ax.flat:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from common.data import get_images

    out = Path(__file__).parent / "outputs" / "corruption_grid.png"
    corruption_overview(get_images("val")[:3], out)
    print("saved", out)
