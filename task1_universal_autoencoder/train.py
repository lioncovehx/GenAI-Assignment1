"""Training loop for the universal autoencoder, logged to MLflow.

  python -m task1_universal_autoencoder.train --epochs 15 --cfg '{"skip": "limited"}' [--limit 512]
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common import gpu, metrics, viz
from common.data import ROOT
from common.datasets import ManifestDataset, RuntimeCorruptionDataset
from common.losses import recon_loss
from task1_universal_autoencoder.model import ConvAE

OUT = Path(__file__).parent / "outputs"
DEFAULT = dict(lr=1e-3, batch_size=32, base_ch=32, latent_ch=32, dense=0, dropout=0.0, alpha=0.8,
               skip="limited16", up="resize", seed=0,  # chosen by ablation
               loss="l1ssim",  # "l2" = plain MSE, only for the loss comparison
               cosine=True,  # cosine LR decay to ~0 over the run, chosen by ablation (Step-2 ablations used False)
               speedups=True,  # common/gpu.py (cudnn.benchmark, TF32) + channels_last; False = Steps 2-4 setup (bf16 only)
               compile=False,  # torch.compile: optional, measured in Step 5 before use
               labels=(0, 1, 2, 3))  # conditions to train and validate on; (1,) = Task 2 salt specialist
MODEL_KEYS = ("base_ch", "latent_ch", "dropout", "skip", "up", "dense")


@torch.no_grad()
def evaluate(model, loader, rows, dev, mf=torch.contiguous_format):
    """Per-image metrics on the fixed val manifest -> DataFrame (name, severity, psnr, ssim, l1)."""
    model.eval()
    out = []
    for xc, x, _, idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(xc.to(dev, memory_format=mf))
        m = metrics.per_image(y, x.to(dev))
        out += [{"name": rows[j]["name"], "severity": rows[j]["severity"], **{k: float(v[i]) for k, v in m.items()}}
                for i, j in enumerate(idx.tolist())]
    return pd.DataFrame(out)


def train(cfg: dict, epochs: int, run_name: str | None = None, limit: int | None = None,
          experiment: str = "task1_universal", out: Path = OUT):
    """experiment/out: Task 2 reuses this loop for its specialists (own MLflow experiment and output folder)."""
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"])
    dev = gpu.setup() if cfg["speedups"] else "cuda"
    mf = torch.channels_last if cfg["speedups"] else torch.contiguous_format  # memory layout of model + inputs
    labels = tuple(cfg["labels"])
    train_ds = RuntimeCorruptionDataset("train", labels)
    if limit:
        train_ds = Subset(train_ds, range(limit))
    val_ds = ManifestDataset("val", labels)  # specialists: only their own condition's val images
    # forkserver avoids workers inheriting a held lock (startup hang); timeout turns a stuck batch into an error
    workers = dict(persistent_workers=True, multiprocessing_context="forkserver", timeout=300)
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=8, drop_last=True,
                          pin_memory=True, **workers)
    val_dl = DataLoader(val_ds, 128, num_workers=4, **workers)
    # fixed grid: first 2 val images of each condition, same every epoch and every run
    grid_idx = [i for n in ("clean", "salt", "blur", "occlusion")
                for i in [j for j, r in enumerate(val_ds.rows) if r["name"] == n][:2]]
    grid = [val_ds[i] for i in grid_idx]
    grid_x, grid_c = torch.stack([g[1] for g in grid]), torch.stack([g[0] for g in grid])

    net = ConvAE(**{k: cfg[k] for k in MODEL_KEYS}).to(dev, memory_format=mf)
    model = torch.compile(net) if cfg["compile"] else net  # checkpoints always save `net` (plain key names)
    opt = torch.optim.Adam(net.parameters(), cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cfg["cosine"] else None
    n_params = sum(p.numel() for p in net.parameters())

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    out_dir = out / (run_name or "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.log_params({**cfg, "epochs": epochs, "limit": limit, "params": n_params,
                           "z_ratio": net.z_size() / 49152})
        print(f"params={n_params:,} |z|/49152={net.z_size() / 49152:.3f}")
        for epoch in range(1, epochs + 1):
            model.train()
            total, n, lr = 0.0, 0, opt.param_groups[0]["lr"]  # lr used during this epoch
            t0 = time.perf_counter()
            for xc, x, _ in train_dl:
                xc = xc.to(dev, non_blocking=True, memory_format=mf)
                x = x.to(dev, non_blocking=True, memory_format=mf)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(xc)
                y = y.float()  # SSIM in fp32
                loss = F.mse_loss(y, x) if cfg["loss"] == "l2" else recon_loss(y, x, cfg["alpha"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total, n = total + loss.item() * len(x), n + len(x)
            if sched:
                sched.step()
            torch.cuda.synchronize()
            train_sec = time.perf_counter() - t0  # training part of the epoch only (no validation)

            df = evaluate(model, val_dl, val_ds.rows, dev, mf)
            score = metrics.objective(df)
            per_type = df.groupby("name")["psnr"].mean()
            mlflow.log_metrics({"train_loss": total / n, "lr": lr, "train_sec": train_sec, "val_psnr": df["psnr"].mean(), "val_ssim": df["ssim"].mean(),
                                "val_l1": df["l1"].mean(), "objective": score,
                                **{f"val_psnr_{k}": v for k, v in per_type.items()}}, step=epoch)
            print(f"epoch {epoch}: loss={total / n:.4f} psnr={df['psnr'].mean():.2f} "
                  f"ssim={df['ssim'].mean():.4f} objective={score:.4f} train_time={train_sec:.1f}s")
            if score > best:
                best = score
                torch.save({"cfg": cfg, "epoch": epoch, "objective": score, "state_dict": net.state_dict()},
                           out_dir / "best.pt")
            if epoch % 5 == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    restored = model(grid_c.to(dev, memory_format=mf)).float().cpu()
                path = out_dir / f"grid_e{epoch:03d}.png"
                viz.restoration_grid(grid_x, grid_c, restored, path,
                                     titles=[val_ds.rows[i]["name"] for i in grid_idx])
                mlflow.log_artifact(str(path), "grids")
            try:
                yield epoch, score
            except GeneratorExit:  # caller stopped early (Optuna pruning): end the run as FINISHED + tagged
                mlflow.set_tags({"pruned": "true", "pruned_at_epoch": epoch})
                mlflow.log_metric("best_objective", best)
                return
        mlflow.log_metric("best_objective", best)
        mlflow.log_artifact(str(out_dir / "best.pt"), "checkpoint")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--cfg", type=json.loads, default={})
    p.add_argument("--name")
    p.add_argument("--limit", type=int, help="train on the first N images only (smoke test)")
    a = p.parse_args()
    for _ in train(a.cfg, a.epochs, a.name, a.limit):
        pass
