"""Joint training of the soft MoE: gate-only warm-up, then everything at a small learning rate.

  python -m task3_soft_moe.train --epochs 2 --name smoke --limit 256 --cfg '{"warmup": 1}'
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common import gpu, metrics
from common.corruptions import LABELS
from common.data import ROOT
from common.datasets import CleanDataset, ManifestDataset, balanced_collate
from common.losses import recon_loss
from task3_soft_moe.moe import build

OUT = Path(__file__).parent / "outputs"
BRANCHES = ("identity", "salt", "blur", "occlusion")
DEFAULT = dict(tau=1.0, lam_c=0.1, lam_b=0.01,
               alpha=0.497,  # L1 weight of the reconstruction loss: the specialists' alpha
               balance="sq",  # "sq" (PDF) | "entropy"
               warmup=2, lr_gate=1e-4, lr_joint=1e-5, batch_size=16, seed=0, speedups=True)


def balance_loss(w, kind):
    wbar = w.mean(0)  # batch-mean weight per branch; batches are balanced, so the ideal is 1/4 each
    if kind == "entropy":
        return np.log(len(wbar)) + (wbar * wbar.clamp_min(1e-8).log()).sum()  # log K - H(wbar) >= 0
    return ((wbar - 1 / len(wbar)) ** 2).sum()


@torch.no_grad()
def evaluate(model, loader, rows, dev, mf):
    """-> (DataFrame per image: name, severity, psnr, ssim, label, pred, w_0..w_3; val CE)."""
    model.eval()
    out, ce = [], 0.0
    for xc, x, y, idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_hat, w, logits = model(xc.to(dev, memory_format=mf))
        ce += F.cross_entropy(logits, y.to(dev), reduction="sum").item()
        m = metrics.per_image(x_hat, x.to(dev))
        pred = logits.argmax(1).cpu()
        for i, j in enumerate(idx.tolist()):
            out.append({"name": rows[j]["name"], "severity": rows[j]["severity"], "label": int(y[i]),
                        "pred": int(pred[i]), "psnr": float(m["psnr"][i]), "ssim": float(m["ssim"][i]),
                        **{f"w_{k}": float(w[i, k]) for k in range(len(BRANCHES))}})
    return pd.DataFrame(out), ce / len(rows)


PSNR_CAP = 40.0  # dB; above this differences are invisible. Only for the objective; reported PSNR stays uncapped


def objective(df):
    return float(df.ssim.mean() + df.psnr.clip(upper=PSNR_CAP).mean() / 40)


def weight_table(df):
    """Mean weight of each branch (columns) for each true class (rows): the gating heatmap's numbers."""
    t = df.groupby("name")[[f"w_{k}" for k in range(len(BRANCHES))]].mean().reindex(list(LABELS))
    t.columns = list(BRANCHES)
    return t


def save_heatmap(t, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.imshow(t.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for (r, c), v in np.ndenumerate(t.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(t.shape[1]), [f"{b}\nbranch" for b in t.columns], fontsize=8)
    ax.set_yticks(range(t.shape[0]), [f"true: {n}" for n in t.index], fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: dict, epochs: int, run_name: str | None = None, limit: int | None = None):
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"])
    dev = gpu.setup() if cfg["speedups"] else "cuda"
    mf = torch.channels_last if cfg["speedups"] else torch.contiguous_format

    train_ds = CleanDataset("train")
    if limit:
        train_ds = Subset(train_ds, range(limit))
    val_ds = ManifestDataset("val")
    workers = dict(persistent_workers=True, multiprocessing_context="forkserver", timeout=300)  # see Task 1 LOG
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=6, drop_last=True,
                          pin_memory=True, collate_fn=balanced_collate, **workers)  # exactly B/4 per class
    val_dl = DataLoader(val_ds, 64, num_workers=2, **workers)

    model = build(cfg["tau"], dev).to(memory_format=mf)
    experts = model.experts

    def phase(joint: bool):
        experts.requires_grad_(joint)
        params = model.parameters() if joint else model.gate.parameters()
        return torch.optim.Adam(params, cfg["lr_joint"] if joint else cfg["lr_gate"])

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task3_moe")
    out_dir = OUT / (run_name or "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_df = -float("inf"), None
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.log_params({**cfg, "epochs": epochs, "limit": limit})

        def log_val(epoch, train_terms):
            df, val_ce = evaluate(model, val_dl, val_ds.rows, dev, mf)
            score = objective(df)
            wt = weight_table(df)
            wbar = df[[f"w_{k}" for k in range(len(BRANCHES))]].mean().to_numpy()  # over the near-balanced val set
            mlflow.log_metrics({**train_terms, "val_psnr": df.psnr.mean(), "val_ssim": df.ssim.mean(),
                                "objective": score, "objective_uncapped": metrics.objective(df), "val_ce": val_ce, "gate_acc": float((df.label == df.pred).mean()),
                                **{f"val_psnr_{n}": v for n, v in df.groupby("name").psnr.mean().items()},
                                **{f"w_{n}_{b}": float(wt.loc[n, b]) for n in wt.index for b in wt.columns},
                                **{f"wbar_{b}": float(v) for b, v in zip(BRANCHES, wbar)}}, step=epoch)
            diag = " ".join(f"{n[:4]}:{wt.loc[n, BRANCHES[k]]:.2f}" for k, n in enumerate(LABELS))
            print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in train_terms.items())
                  + f" | val psnr={df.psnr.mean():.2f} ssim={df.ssim.mean():.4f} objective={score:.4f} "
                  f"gate_acc={(df.label == df.pred).mean():.4f} | own-branch weight {diag} | wbar {np.round(wbar, 3).tolist()}")
            info = {"wbar": wbar, "w_blur_identity": float(wt.loc["blur", "identity"]),
                    **{f"psnr_{n}": float(v) for n, v in df.groupby("name").psnr.mean().items()}}
            return score, df, info

        score0, df0, _ = log_val(0, {})  # the model exactly as initialised from Task 2
        save_heatmap(weight_table(df0), out_dir / "weights_e000.png", f"{run_name}: val mean weights at init (tau {cfg['tau']})")
        for epoch in range(1, epochs + 1):
            if epoch == 1 or epoch == cfg["warmup"] + 1:
                opt = phase(joint=epoch > cfg["warmup"])
            model.train()
            experts.eval()  # BN running statistics of the experts stay frozen (see docstring)
            sums, n = {k: 0.0 for k in ("loss", "rec", "ce", "bal")}, 0
            for xc, x, y in train_dl:
                xc = xc.to(dev, non_blocking=True, memory_format=mf)
                x = x.to(dev, non_blocking=True, memory_format=mf)
                y = y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x_hat, w, logits = model(xc)
                terms = {"rec": recon_loss(x_hat.float(), x, cfg["alpha"]), "ce": F.cross_entropy(logits.float(), y),
                         "bal": balance_loss(w.float(), cfg["balance"])}
                terms["loss"] = terms["rec"] + cfg["lam_c"] * terms["ce"] + cfg["lam_b"] * terms["bal"]
                opt.zero_grad(set_to_none=True)
                terms["loss"].backward()
                opt.step()
                for k, v in terms.items():
                    sums[k] += v.item() * len(y)
                n += len(y)
            phase_name = "joint" if epoch > cfg["warmup"] else "warmup"
            score, df, info = log_val(epoch, {f"train_{k}": v / n for k, v in sums.items()})
            mlflow.set_tag(f"phase_e{epoch:03d}", phase_name)
            if score > best:
                best, best_df = score, (epoch, df)
                torch.save({"cfg": cfg, "epoch": epoch, "objective": score, "state_dict": model.state_dict()},
                           out_dir / "best.pt")
            try:
                yield epoch, score, info
            except GeneratorExit:  # caller stopped early (Optuna pruning): end the run as FINISHED + tagged
                mlflow.set_tags({"pruned": "true", "pruned_at_epoch": epoch})
                mlflow.log_metric("best_objective", best)
                return
        e, df = best_df
        save_heatmap(weight_table(df), out_dir / "weights_best.png", f"{run_name}: val mean weights, best epoch {e}")
        mlflow.log_metrics({"best_objective": best, "objective_at_init": score0})
        mlflow.log_artifact(str(out_dir / "weights_e000.png"))
        mlflow.log_artifact(str(out_dir / "weights_best.png"))
        mlflow.log_artifact(str(out_dir / "best.pt"), "checkpoint")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--cfg", type=json.loads, default={})
    p.add_argument("--name")
    p.add_argument("--limit", type=int, help="train on the first N clean images only (smoke test)")
    a = p.parse_args()
    for _ in train(a.cfg, a.epochs, a.name, a.limit):
        pass
