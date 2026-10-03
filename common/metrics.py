"""Per-image PSNR / SSIM / L1 and grouped result tables, shared by Tasks 1-3 so the numbers are comparable."""
import pandas as pd
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

PSNR_CAP = 100.0  # identical images (do-nothing on clean) give infinite PSNR; capped so means stay finite


def per_image(x_hat: torch.Tensor, x: torch.Tensor) -> dict:
    """[B,3,H,W] in [0,1] -> dict of [B] tensors."""
    x_hat, x = x_hat.float().clamp(0, 1), x.float()
    mse = (x_hat - x).pow(2).flatten(1).mean(1)
    return {
        "psnr": (10 * torch.log10(1 / mse)).clamp(max=PSNR_CAP),
        "ssim": ssim(x_hat, x, data_range=1.0, reduction="none"),
        "l1": (x_hat - x).abs().flatten(1).mean(1),
    }


def objective(df: pd.DataFrame) -> float:
    """Optuna score for restoration tasks: SSIM + PSNR/40, independent of the loss weights."""
    return float(df["ssim"].mean() + df["psnr"].mean() / 40)


def table(df: pd.DataFrame, by=("name", "severity")) -> pd.DataFrame:
    """df needs columns name, severity, psnr, ssim, l1 (one row per image)."""
    return df.groupby(list(by), dropna=False, sort=False)[["psnr", "ssim", "l1"]].mean().round(4)
