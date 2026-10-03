"""Training loop for the sketch cGAN, logged to MLflow. Objective: val SSIM - LPIPS.

  python -m task4_cgan_sketch.train --epochs 2 --name smoke --limit 64
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from common import gpu
from common.data import ROOT
from task4_cgan_sketch.data import FS2KDataset
from task4_cgan_sketch.models import N_STYLES, PatchDiscriminator, UNetGenerator

OUT = Path(__file__).parent / "outputs"
DEFAULT = dict(lr_G=2e-4, lr_D=2e-4, batch_size=8, base_ch=64, emb_dim=16, dropout=0.5, lam_l1=100.0,  # pix2pix values
               lam_adv=1.0,  # 0 = L1-only generator, no GAN (the "why a GAN?" baseline)
               g_cond="film", d_cond="concat", n_layers=3, mode="pad", seed=0, speedups=True)


def to01(t):
    return (t.float() + 1) / 2


@torch.no_grad()
def evaluate(G, loader, rows, lpips, dev, mf):
    """Val metrics overall, per style (s1/s2/s3) and per photo shape (photo1 = square CASIA, nonsquare = photo2/3).
    ssim, l1 on [0,1]; lpips per image; style_sens = mean |G(x,s_i) - G(x,s_j)| over style pairs. loader: unshuffled."""
    G.eval()
    per = {k: [] for k in ("ssim", "l1", "lpips", "style_sens")}
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(dev, memory_format=mf), sketch.to(dev), style.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = [G(photo, torch.full_like(style, s)) for s in range(N_STYLES)]
            fake = G(photo, style)
        fake = fake.float()
        per["ssim"].append(ssim(to01(fake), to01(sketch), data_range=1.0, reduction="none"))
        per["l1"].append((to01(fake) - to01(sketch)).abs().flatten(1).mean(1))
        per["lpips"].append(lpips.net(fake.repeat(1, 3, 1, 1).clamp(-1, 1), sketch.repeat(1, 3, 1, 1)).flatten())
        pairs = [(to01(outs[i]) - to01(outs[j])).abs().flatten(1).mean(1) for i, j in combinations(range(N_STYLES), 2)]
        per["style_sens"].append(torch.stack(pairs).mean(0))
    G.train()
    per = {k: torch.cat(v).float().cpu() for k, v in per.items()}
    groups = {"": torch.ones(len(rows), dtype=torch.bool)}
    groups |= {f"_s{s + 1}": torch.tensor([r["style"] == s for r in rows]) for s in range(N_STYLES)}
    groups |= {"_photo1": torch.tensor([r["source"] == "photo1" for r in rows])}
    groups |= {"_nonsquare": ~groups["_photo1"]}
    return {f"{k}{g}": float(v[m].mean()) for g, m in groups.items() for k, v in per.items()
            if m.any() and not (g and k == "style_sens")}


@torch.no_grad()
def sample_grid(G, ds, idx, dev, mf, path):
    """Rows = fixed val photos; columns = photo | real sketch | G in style 1 | 2 | 3."""
    G.eval()
    photo = torch.stack([ds[i][0] for i in idx]).to(dev, memory_format=mf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outs = [G(photo, torch.full((len(idx),), s, device=dev)).float().cpu() for s in range(N_STYLES)]
    G.train()
    fig, ax = plt.subplots(len(idx), 2 + N_STYLES, figsize=(2 * (2 + N_STYLES), 2 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        p, sk, st = ds[i]
        panels = [(to01(p).permute(1, 2, 0), f"photo (style {int(st) + 1})"), (to01(sk)[0], "real sketch")]
        panels += [(to01(outs[s][r])[0], f"G, style {s + 1}") for s in range(N_STYLES)]
        for c, (im, title) in enumerate(panels):
            ax[r, c].imshow(im.numpy(), cmap="gray", vmin=0, vmax=1)
            ax[r, c].set_title(title if r == 0 or c == 0 else "", fontsize=7)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def train(cfg: dict, epochs: int, run_name: str | None = None, limit: int | None = None):
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"])
    dev = gpu.setup() if cfg["speedups"] else "cuda"
    mf = torch.channels_last if cfg["speedups"] else torch.contiguous_format

    train_ds = FS2KDataset("train", cfg["mode"], augment=True)
    val_ds = FS2KDataset("val", cfg["mode"])
    if limit:
        train_ds = Subset(train_ds, range(limit))
    workers = dict(persistent_workers=True, multiprocessing_context="forkserver", timeout=300)  # see Task 1 LOG
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True,
                          pin_memory=True, **workers)
    val_dl = DataLoader(val_ds, 32, num_workers=2, **workers)
    grid_idx = [i for s in range(N_STYLES) for i in (val_ds.style == s).nonzero().flatten()[:2].tolist()]

    G = UNetGenerator(cfg["base_ch"], cfg["emb_dim"], cfg["dropout"], cfg["g_cond"]).to(dev, memory_format=mf)
    D = PatchDiscriminator(64, cfg["emb_dim"], cfg["n_layers"], cfg["d_cond"]).to(dev, memory_format=mf)
    opt_G = torch.optim.Adam(G.parameters(), cfg["lr_G"], betas=(0.5, 0.999))  # beta1 0.5 [Isola17]
    opt_D = torch.optim.Adam(D.parameters(), cfg["lr_D"], betas=(0.5, 0.999))
    # pix2pix schedule: constant LR for the first half, then linear decay towards 0
    half = epochs // 2
    decay = lambda e: min(1.0, (epochs - e) / max(epochs - half, 1))
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, decay) for o in (opt_G, opt_D)]
    bce = torch.nn.BCEWithLogitsLoss()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(dev)  # expects 3-ch images in [-1, 1]

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task4_sketch")
    out_dir = OUT / (run_name or "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -float("inf")
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None):
        n_G, n_D = (sum(p.numel() for p in m.parameters()) for m in (G, D))
        mlflow.log_params({**cfg, "epochs": epochs, "limit": limit, "params_G": n_G, "params_D": n_D})
        print(f"params G={n_G:,} D={n_D:,}")
        for epoch in range(1, epochs + 1):
            gan = cfg["lam_adv"] > 0
            sums, n = (dict(D_real=0.0, D_fake=0.0, G_adv=0.0) if gan else {}) | {"G_L1": 0.0}, 0
            for photo, sketch, style in train_dl:
                photo = photo.to(dev, non_blocking=True, memory_format=mf)
                sketch = sketch.to(dev, non_blocking=True, memory_format=mf)
                style = style.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake = G(photo, style)
                losses = {}
                if gan:
                    # D step: real -> 1, fake -> 0; x0.5 slows D relative to G [Isola17]. Losses in fp32.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        d_real, d_fake = D(photo, sketch, style), D(photo, fake.detach(), style)
                    losses["D_real"] = bce(d_real.float(), torch.ones_like(d_real, dtype=torch.float))
                    losses["D_fake"] = bce(d_fake.float(), torch.zeros_like(d_fake, dtype=torch.float))
                    opt_D.zero_grad(set_to_none=True)
                    (0.5 * (losses["D_real"] + losses["D_fake"])).backward()
                    opt_D.step()
                    # G step, part 1: fool D (fake -> 1), with D frozen
                    D.requires_grad_(False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        d_fake_g = D(photo, fake, style)
                    losses["G_adv"] = bce(d_fake_g.float(), torch.ones_like(d_fake_g, dtype=torch.float))
                    D.requires_grad_(True)
                # G step, part 2: stay close to the real sketch (L1)
                losses["G_L1"] = F.l1_loss(fake.float(), sketch)
                loss_G = cfg["lam_l1"] * losses["G_L1"] + (cfg["lam_adv"] * losses["G_adv"] if gan else 0)
                opt_G.zero_grad(set_to_none=True)
                loss_G.backward()
                opt_G.step()
                b = len(style)
                for k, v in losses.items():
                    sums[k] += v.item() * b
                n += b
            lr_now = scheds[0].get_last_lr()[0]
            for s in scheds if gan else scheds[:1]:  # L1-only: D's optimiser never steps, so don't step its schedule
                s.step()

            val = evaluate(G, val_dl, val_ds.rows, lpips, dev, mf)
            score = val["ssim"] - val["lpips"]
            mlflow.log_metrics({**{k: v / n for k, v in sums.items()}, **{f"val_{k}": v for k, v in val.items()},
                                "objective": score, "lr_G": lr_now}, step=epoch)
            train_part = " ".join(f"{k}={v / n:.3f}" for k, v in sums.items())
            print(f"epoch {epoch}: {train_part} | val ssim={val['ssim']:.4f} lpips={val['lpips']:.4f} "
                  f"(nonsquare {val['lpips_nonsquare']:.4f}) style_sens={val['style_sens']:.4f} objective={score:.4f}")
            if score > best:
                best = score
                torch.save({"cfg": cfg, "epoch": epoch, "objective": score,
                            "G": G.state_dict(), "D": D.state_dict()}, out_dir / "best.pt")
            if epoch % 5 == 0 or epoch == epochs:
                path = out_dir / f"grid_e{epoch:03d}.png"
                sample_grid(G, val_ds, grid_idx, dev, mf, path)
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
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--cfg", type=json.loads, default={})
    p.add_argument("--name")
    p.add_argument("--limit", type=int, help="train on the first N pairs only (smoke test)")
    a = p.parse_args()
    for _ in train(a.cfg, a.epochs, a.name, a.limit):
        pass
