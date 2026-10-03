"""L = alpha*L1 + (1-alpha)*(1-SSIM). SSIM: Gaussian window 11, sigma 1.5 [Wang04]."""
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


def recon_loss(x_hat, x, alpha: float):
    return alpha * F.l1_loss(x_hat, x) + (1 - alpha) * (1 - ssim(x_hat.float(), x.float(), data_range=1.0))
