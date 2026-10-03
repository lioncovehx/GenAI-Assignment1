"""Test-set evaluation of the sketch generator against the pix2pix baseline, the L1-only model and Canny edges.

  python -m task4_cgan_sketch.evaluate
"""
from itertools import combinations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from common.data import ROOT
from task4_cgan_sketch.data import FS2KDataset
from task4_cgan_sketch.models import N_STYLES, UNetGenerator
from task4_cgan_sketch.train import OUT, sample_grid, to01

MODELS = {"final": "final", "pix2pix": "baseline", "l1only": "D-4.5_l1only"}  # method -> run folder
TEST_OUT = OUT / "test"


def load_G(run: str, dev) -> UNetGenerator:
    ck = torch.load(OUT / run / "best.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    G = UNetGenerator(c["base_ch"], c["emb_dim"], c["dropout"], c["g_cond"])
    G.load_state_dict(ck["G"])
    print(f"{run}: epoch {ck['epoch']}, val objective {ck['objective']:.4f}")
    return G.to(dev).eval()  # eval: dropout off, the same as validation and the ONNX export


def canny_sketch(photo: torch.Tensor) -> torch.Tensor:
    """photo [3,H,W] in [-1,1] -> dark edges on white [1,H,W] in [-1,1]. Thresholds from the median (0.66x, 1.33x)."""
    gray = cv2.cvtColor((to01(photo).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    m = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * m)), int(min(255, 1.33 * m)))
    return torch.from_numpy(255 - edges)[None].float() / 127.5 - 1


@torch.no_grad()
def scores(fake, sketch, lpips) -> dict:
    """Per-image metrics, all [-1,1] tensors [B,1,H,W]."""
    fake = fake.float()
    return {"ssim": ssim(to01(fake), to01(sketch), data_range=1.0, reduction="none"),
            "l1": (to01(fake) - to01(sketch)).abs().flatten(1).mean(1),
            "lpips": lpips.net(fake.repeat(1, 3, 1, 1).clamp(-1, 1), sketch.repeat(1, 3, 1, 1)).flatten()}


@torch.no_grad()
def run_all(ds, dev):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(dev)
    loader = DataLoader(ds, 32)  # unshuffled: batch order = row order
    gens = {m: load_G(run, dev) for m, run in MODELS.items()}
    per = {m: {k: [] for k in ("ssim", "l1", "lpips", "style_sens")} for m in [*MODELS, "canny"]}
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(dev), sketch.to(dev), style.to(dev)
        for m, G in gens.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = [G(photo, torch.full_like(style, s)).float() for s in range(N_STYLES)]
            fake = torch.stack(outs)[style, torch.arange(len(style))]  # each photo in its own style
            for k, v in scores(fake, sketch, lpips).items():
                per[m][k].append(v.cpu())
            pairs = [(to01(outs[i]) - to01(outs[j])).abs().flatten(1).mean(1) for i, j in combinations(range(N_STYLES), 2)]
            per[m]["style_sens"].append(torch.stack(pairs).mean(0).cpu())
        edges = torch.stack([canny_sketch(p) for p in photo.cpu()]).to(dev)
        for k, v in scores(edges, sketch, lpips).items():
            per["canny"][k].append(v.cpu())
        per["canny"]["style_sens"].append(torch.full((len(style),), float("nan")))  # no style input: n/a
    rows = []
    for m, d in per.items():
        d = {k: torch.cat(v).numpy() for k, v in d.items()}
        for i, r in enumerate(ds.rows):
            rows.append({"method": m, "idx": i, "style": r["style"] + 1, "source": r["source"],
                         "group": f"{r['source']}-S{r['style'] + 1}", "frontal_face": r["frontal_face"],
                         "hair": r["hair"], **{k: float(v[i]) for k, v in d.items()}})
    df = pd.DataFrame(rows)
    df["objective"] = df["ssim"] - df["lpips"]
    return df, gens


def table(df, by):
    t = df.groupby(by, sort=False).agg(n=("idx", "size"), ssim=("ssim", "mean"), lpips=("lpips", "mean"),
                                       l1=("l1", "mean"), style_sens=("style_sens", "mean"))
    t["objective"] = t["ssim"] - t["lpips"]
    return t.round(4)


@torch.no_grad()
def failures(df, ds, gens, dev, path):
    """Rule-based (not hand-picked): the 6 worst test images by the final model's LPIPS, then 3 at the median."""
    f = df[df.method == "final"].sort_values("lpips")
    picks = [(i, "worst") for i in f.idx.iloc[::-1][:6]] + [(i, "median") for i in f.idx.iloc[len(f) // 2 - 1:][:3]]
    cols = ["photo", "real sketch", "final", "l1only", "canny"]
    fig, ax = plt.subplots(len(picks), len(cols), figsize=(2 * len(cols), 2.1 * len(picks)), squeeze=False)
    for r, (i, kind) in enumerate(picks):
        photo, sketch, style = ds[i]
        p, s = photo[None].to(dev), style[None].to(dev)
        panels = [to01(photo).permute(1, 2, 0), to01(sketch)[0]]
        panels += [to01(gens[m](p, s).float().cpu()[0, 0]) for m in ("final", "l1only")] + [to01(canny_sketch(photo))[0]]
        row = f[f.idx == i].iloc[0]
        for c, im in enumerate(panels):
            ax[r, c].imshow(im.numpy(), cmap="gray", vmin=0, vmax=1)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=8)
        ax[r, 0].set_ylabel(f"{kind}\n{row.group}\nLPIPS {row.lpips:.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    dev = "cuda"
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    ds = FS2KDataset("test", "pad")
    df, gens = run_all(ds, dev)
    df.to_csv(TEST_OUT / "per_image.csv", index=False)
    overall = table(df, ["method"])
    by_group = table(df, ["method", "group"])
    by_style = table(df, ["method", "style"])
    by_attr = pd.concat({a: table(df[df.method == "final"], [a]) for a in ("frontal_face", "hair")})
    overall.to_csv(TEST_OUT / "table_overall.csv")
    pd.concat([by_style, by_group]).to_csv(TEST_OUT / "table_by_group.csv")
    by_attr.to_csv(TEST_OUT / "table_by_attribute.csv")
    for name, t in (("overall", overall), ("by style", by_style), ("by source x style", by_group),
                    ("final model by attribute", by_attr)):
        print(f"\n== {name} ==\n{t.to_string()}")

    # style swap: first 2 test photos of each source x style group (fixed rule), each drawn in all 3 styles
    idx = [i for g in df[df.method == "final"].group.unique()
           for i in df[(df.method == "final") & (df.group == g)].idx[:2]]
    sample_grid(gens["final"], ds, idx, dev, torch.contiguous_format, TEST_OUT / "style_swap.png")
    failures(df, ds, gens, dev, TEST_OUT / "failures.png")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task4_sketch")
    with mlflow.start_run(run_name="test_eval"):
        mlflow.log_params({"checkpoint": "outputs/final/best.pt", "test_pairs": len(ds), "mode": "pad"})
        mlflow.log_metrics({f"test_{k}_{m}": float(overall.loc[m, k]) for m in overall.index
                            for k in ("ssim", "lpips", "l1", "objective", "style_sens")})
        mlflow.log_artifacts(str(TEST_OUT), "test")
    print(f"\nsaved {TEST_OUT}")


if __name__ == "__main__":
    main()
