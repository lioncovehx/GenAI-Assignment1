"""Zero-z probe: zero the bottleneck and see how much of the image still gets through the skip connections.

  python -m task1_universal_autoencoder.probe baseline D-1.2_limited D-1.2_limited16 D-1.2_full
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import metrics
from common.datasets import ManifestDataset
from common.viz import _img
from task1_universal_autoencoder.model import ConvAE
from task1_universal_autoencoder.train import MODEL_KEYS, OUT


def load(run, dev):
    ck = torch.load(OUT / run / "best.pt", map_location=dev)
    model = ConvAE(**{k: ck["cfg"][k] for k in MODEL_KEYS}).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    return model


@torch.no_grad()
def outputs(model, xc, zero_z):
    # forward hook: replace the bottleneck's output (z) with zeros
    h = model.to_z.register_forward_hook(lambda m, i, o: torch.zeros_like(o)) if zero_z else None
    y = model(xc)
    if h:
        h.remove()
    return y


def main(runs):
    dev = "cuda"
    ds = ManifestDataset("val")
    rows, show = [], {}
    # one example per condition for the figure
    pick = [next(j for j, r in enumerate(ds.rows) if r["name"] == n) for n in ("clean", "salt", "blur", "occlusion")]
    for run in runs:
        model = load(run, dev)
        for xc, x, _, idx in DataLoader(ds, 128):
            xc, x = xc.to(dev), x.to(dev)
            normal, zero = outputs(model, xc, False), outputs(model, xc, True)
            m_n, m_z = metrics.per_image(normal, x), metrics.per_image(zero, x)
            for i, j in enumerate(idx.tolist()):
                rows.append({"run": run, "name": ds.rows[j]["name"], "psnr_normal": float(m_n["psnr"][i]),
                             "psnr_zero_z": float(m_z["psnr"][i]), "ssim_zero_z": float(m_z["ssim"][i])})
        xc = torch.stack([ds[j][0] for j in pick]).to(dev)
        show[run] = (outputs(model, xc, False).cpu(), outputs(model, xc, True).cpu())

    df = pd.DataFrame(rows)
    table = df.groupby("run", sort=False)[["psnr_normal", "psnr_zero_z", "ssim_zero_z"]].mean().round(3)
    print(table.to_string())
    print("\nzero-z PSNR per condition:")
    print(df.pivot_table("psnr_zero_z", "run", "name", sort=False).round(2).to_string())

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(fig_dir / "zero_z_probe.csv", index=False)
    cols = 2 + 2 * len(runs)
    fig, ax = plt.subplots(len(pick), cols, figsize=(1.6 * cols, 1.7 * len(pick)), squeeze=False)
    for r, j in enumerate(pick):
        xc, x, _, _ = ds[j]
        panels = [(x, "clean"), (xc, "input")]
        for run in runs:
            panels += [(show[run][0][r], f"{run}\nnormal"), (show[run][1][r], f"{run}\nz = 0")]
        for c, (im, title) in enumerate(panels):
            ax[r, c].imshow(_img(im))
            ax[r, c].set_title(title if r == 0 else "", fontsize=6)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel(ds.rows[j]["name"], fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "zero_z_probe.png", dpi=130)
    print("saved", fig_dir / "zero_z_probe.png")


if __name__ == "__main__":
    main(sys.argv[1:] or ["baseline", "D-1.2_limited", "D-1.2_full"])
