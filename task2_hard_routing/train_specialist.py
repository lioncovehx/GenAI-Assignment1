"""Train one specialist: Task 1's autoencoder on a single corruption type.

  python -m task2_hard_routing.train_specialist --type salt --epochs 2 --limit 256 --name smoke_salt
"""
import argparse
import json
from pathlib import Path

from task1_universal_autoencoder.train import train as train_ae

OUT = Path(__file__).parent / "outputs"
TYPES = {"salt": 1, "blur": 2, "occlusion": 3}  # label ids from common/corruptions.py
SPECIALIST = dict(lr=8.783164913704277e-04, batch_size=16, base_ch=64, latent_ch=32, dropout=0.04408982445617521,
                  alpha=0.5415273983442785)  # skip limited16, resize, cosine: Task 1 DEFAULT


def train(kind: str, cfg: dict, epochs: int, run_name: str | None = None, limit: int | None = None):
    """Generator yielding (epoch, objective), objective = SSIM + PSNR/40 on this type's val images only."""
    return train_ae({**SPECIALIST, **cfg, "labels": (TYPES[kind],)}, epochs, run_name, limit,
                    experiment="task2_routing", out=OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--type", required=True, choices=TYPES)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--cfg", type=json.loads, default={})
    p.add_argument("--name")
    p.add_argument("--limit", type=int, help="train on the first N images only (smoke test)")
    a = p.parse_args()
    for _ in train(a.type, a.cfg, a.epochs, a.name, a.limit):
        pass
