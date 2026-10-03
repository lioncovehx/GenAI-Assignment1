"""Train the corruption classifier on balanced batches (a quarter of each batch per class).

  python -m task2_hard_routing.train_classifier --epochs 2 --name smoke --limit 256
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader, Subset

from common import gpu
from common.corruptions import LABELS, SEVERITIES
from common.data import ROOT
from common.datasets import CleanDataset, ManifestDataset, balanced_collate
from task2_hard_routing.classifier import build

OUT = Path(__file__).parent / "outputs"
DEFAULT = dict(arch="cnn",  # chosen by ablation; "resnet18" = ImageNet-pretrained, fine-tuned
               lr=1e-3, batch_size=32, channels="32-64-128-256", dropout=0.2, weight_decay=1e-4,
               pool="max",  # chosen by ablation; "stride" = strided conv instead of max-pool
               seed=0, speedups=True)


@torch.no_grad()
def evaluate(model, loader, rows, dev, mf):
    """Val metrics: loss, accuracy, macro-F1, recall per class and per (class, severity). Also returns the predictions."""
    model.eval()
    preds, loss = [], 0.0
    for xc, _, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xc.to(dev, memory_format=mf)).float()
        loss += F.cross_entropy(logits, y.to(dev), reduction="sum").item()
        preds.append(logits.argmax(1).cpu())
    model.train()
    y_pred = torch.cat(preds).numpy()
    y_true = np.array([r["label"] for r in rows])
    sev = np.array([r["severity"] for r in rows])
    m = {"loss": loss / len(rows), "acc": float((y_pred == y_true).mean()),
         "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}
    for k, name in enumerate(LABELS):  # recall = of the images truly k, the fraction predicted k
        m[f"recall_{name}"] = float((y_pred[y_true == k] == k).mean())
        for s in SEVERITIES if k else ():
            mask = (y_true == k) & (sev == s)
            m[f"recall_{name}_{s}"] = float((y_pred[mask] == k).mean())
    return m, y_true, y_pred


def save_confusion(y_true, y_pred, path, title):
    """Normalised 4x4 confusion matrix: row = true class, each row sums to 1, diagonal = recall."""
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=range(len(LABELS)), display_labels=LABELS,
                                                   normalize="true", values_format=".2f", cmap="Blues", colorbar=False)
    disp.ax_.set_title(title, fontsize=9)
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=120)
    plt.close(disp.figure_)


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
    # balanced_collate corrupts inside the worker processes: exactly B/4 images per class in every batch
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=6, drop_last=True,
                          pin_memory=True, collate_fn=balanced_collate, **workers)
    val_dl = DataLoader(val_ds, 128, num_workers=2, **workers)

    model = build(cfg).to(dev, memory_format=mf)
    opt = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)  # same schedule as Task 1
    n_params = sum(p.numel() for p in model.parameters())

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task2_routing")
    out_dir = OUT / (run_name or "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    best, best_pred = float("inf"), None
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        mlflow.log_params({**cfg, "epochs": epochs, "limit": limit, "params": n_params})
        print(f"params={n_params:,}")
        for epoch in range(1, epochs + 1):
            total, correct, n, lr = 0.0, 0, 0, opt.param_groups[0]["lr"]  # lr used during this epoch
            for xc, _, y in train_dl:
                if epoch == 1 and n == 0:  # evidence for "balanced batches"
                    counts = dict(zip(LABELS, torch.bincount(y, minlength=len(LABELS)).tolist()))
                    mlflow.log_param("first_batch_counts", counts)
                    print("first batch class counts:", counts)
                xc = xc.to(dev, non_blocking=True, memory_format=mf)
                y = y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xc)
                loss = F.cross_entropy(logits.float(), y)  # loss in fp32
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
            sched.step()

            val, y_true, y_pred = evaluate(model, val_dl, val_ds.rows, dev, mf)
            score = val["loss"]
            mlflow.log_metrics({"train_loss": total / n, "train_acc": correct / n, "lr": lr,
                                **{f"val_{k}": v for k, v in val.items()}, "objective": score}, step=epoch)
            print(f"epoch {epoch}: loss={total / n:.4f} train_acc={correct / n:.4f} | val loss={val['loss']:.4f} "
                  f"acc={val['acc']:.4f} macro_f1={val['macro_f1']:.4f} | recall clean={val['recall_clean']:.3f} "
                  f"salt={val['recall_salt']:.3f} blur={val['recall_blur']:.3f} occl={val['recall_occlusion']:.3f} "
                  f"(salt_low={val['recall_salt_low']:.3f} blur_low={val['recall_blur_low']:.3f})")
            if score < best:
                best, best_pred = score, (epoch, y_pred, val["macro_f1"])
                torch.save({"cfg": cfg, "epoch": epoch, "objective": score, "state_dict": model.state_dict()},
                           out_dir / "best.pt")
            try:
                yield epoch, score, val["macro_f1"]
            except GeneratorExit:  # caller stopped early (Optuna pruning): end the run as FINISHED + tagged
                mlflow.set_tags({"pruned": "true", "pruned_at_epoch": epoch})
                mlflow.log_metric("best_objective", best)
                return
        path = out_dir / "confusion_val.png"
        save_confusion(y_true, best_pred[1], path,
                       f"{run_name}: val, best epoch {best_pred[0]} (loss {best:.4f}, macro-F1 {best_pred[2]:.4f})")
        mlflow.log_metrics({"best_objective": best, "best_macro_f1": best_pred[2]})
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(out_dir / "best.pt"), "checkpoint")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--cfg", type=json.loads, default={})
    p.add_argument("--name")
    p.add_argument("--limit", type=int, help="train on the first N clean images only (smoke test)")
    a = p.parse_args()
    for _ in train(a.cfg, a.epochs, a.name, a.limit):
        pass
