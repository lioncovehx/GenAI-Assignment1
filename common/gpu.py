"""GPU speed settings shared by all tasks (TF32 matmuls, cuDNN autotune). Training also uses bf16 autocast
and channels_last.
"""
import torch


def setup() -> str:
    torch.backends.cudnn.benchmark = True  # input sizes never change -> benchmark conv algorithms once, keep the fastest
    torch.set_float32_matmul_precision("high")  # TF32 for the remaining fp32 matmuls (e.g. SSIM, Linear layers)
    return "cuda"
