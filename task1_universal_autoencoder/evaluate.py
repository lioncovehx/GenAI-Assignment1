"""Test-set evaluation of the Task 1 model against doing nothing and a classical fix per corruption.

  python -m task1_universal_autoencoder.evaluate
"""
import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur

from common import gpu, metrics, viz
from common.corruptions import LABELS
from common.data import ROOT
from common.datasets import ManifestDataset
from task1_universal_autoencoder.train import MODEL_KEYS, OUT
from task1_universal_autoencoder.model import ConvAE

CKPT = OUT / "final" / "best.pt"
TEST_OUT = OUT / "test"


def classical(xc: torch.Tensor, row: dict) -> torch.Tensor:
    """The standard non-learned fix for this row's corruption type. xc: float [3,128,128] in [0,1]."""
    name = row["name"]
    if name == "salt":
        return torch.from_numpy(median_filter(xc.numpy(), size=(1, 3, 3), mode="reflect"))
    if name == "blur":  # generic unsharp mask: x + (x - blur(x)); does not know the true kernel
        return (2 * xc - gaussian_blur(xc, [5, 5], [1.5, 1.5])).clamp(0, 1)
    if name == "occlusion":
        img = (xc.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        mask = np.zeros(img.shape[:2], np.uint8)
        for x0, y0, x1, y1 in row["rects"]:
            mask[y0:y1, x0:x1] = 255
        out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return torch.from_numpy(out).permute(2, 0, 1).float() / 255
    return xc.clone()


class TestWithClassical(ManifestDataset):
    """Test manifest rows + the classical baseline, computed in the DataLoader workers."""

    def __getitem__(self, j):
        xc, x, label, j = super().__getitem__(j)
        return xc, x, label, j, classical(xc, self.rows[j])


def main():
    cv2.setNumThreads(0)  # the DataLoader workers already run in parallel
    dev = gpu.setup()
    ck = torch.load(CKPT, map_location=dev)
    model = ConvAE(**{k: ck["cfg"][k] for k in MODEL_KEYS}).to(dev, memory_format=torch.channels_last).eval()
    model.load_state_dict(ck["state_dict"])
    print(f"model: {CKPT} (epoch {ck['epoch']}, val objective {ck['objective']:.4f})")

    ds = TestWithClassical("test")
    rows = []
    with torch.no_grad():
        for xc, x, label, idx, cl in DataLoader(ds, 256, num_workers=8, multiprocessing_context="forkserver", timeout=300):
            xc_d, x_d = xc.to(dev, memory_format=torch.channels_last), x.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ae = model(xc_d).float()
            scores = {m: metrics.per_image(out, x_d) for m, out in (("input", xc_d), ("ae", ae), ("classical", cl.to(dev)))}
            for i, j in enumerate(idx.tolist()):
                r = ds.rows[j]
                for m, sc in scores.items():
                    rows.append({"row": j, "image": r["image"], "name": r["name"], "severity": r["severity"] or "-",
                                 "method": m, **{k: float(v[i]) for k, v in sc.items()}})
    df = pd.DataFrame(rows)
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEST_OUT / "per_image.csv", index=False)

    order = list(LABELS)
    by_type = df.pivot_table(index="name", columns="method", values=["psnr", "ssim", "l1"]).reindex(order).round(4)
    by_sev = df.pivot_table(index=["name", "severity"], columns="method", values=["psnr", "ssim"]).round(4)
    by_sev = by_sev.reindex(sorted(by_sev.index, key=lambda t: (order.index(t[0]), ["-", "low", "medium", "high"].index(t[1]))))
    by_type.to_csv(TEST_OUT / "table_by_type.csv")
    by_sev.to_csv(TEST_OUT / "table_by_severity.csv")
    pd.set_option("display.width", 200)
    print("\nTable A: by corruption type (PSNR capped at 100 for identical images)\n", by_type.to_string())
    print("\nTable B: by severity\n", by_sev.to_string())

    # gain over do-nothing per test row, for picking examples and failures
    wide = df.pivot_table(index=["row", "image", "name", "severity"], columns="method", values="psnr").reset_index()
    wide["gain"] = wide["ae"] - wide["input"]

    def grid(sel, path, label):
        xc, x, out = [], [], []
        for _, w in sel.iterrows():
            a, b = ds[int(w["row"])][:2]
            xc.append(a), x.append(b)
        xc, x = torch.stack(xc), torch.stack(x)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xc.to(dev, memory_format=torch.channels_last)).float().cpu()
        titles = [f"{w['name']} {w['severity']}\n{label(w)}" for _, w in sel.iterrows()]
        viz.restoration_grid(x, xc, out, path, titles=titles)

    # 12 examples: one image index (fixed seed) per condition x severity; clean: 3 different images
    rng = np.random.default_rng(42)
    ex = []
    for name in order:
        sub = wide[wide["name"] == name]
        if name == "clean":
            ex.append(sub.sample(3, random_state=42))
        else:
            for sev in ("low", "medium", "high"):
                ex.append(sub[sub["severity"] == sev].sample(1, random_state=int(rng.integers(10**6))))
    ex = pd.concat(ex)
    grid(ex, TEST_OUT / "examples.png", lambda w: f"in {w['input']:.1f} -> out {w['ae']:.1f} dB")

    # failure cases: per corrupted type the 2 rows where the AE helped least (lowest gain); clean: 2 lowest PSNR
    fails = pd.concat([wide[wide["name"] == n].nsmallest(2, "gain" if n != "clean" else "ae") for n in order])
    fails.to_csv(TEST_OUT / "failures.csv", index=False)
    grid(fails, TEST_OUT / "failures.png", lambda w: f"img {int(w['image'])}: in {w['input']:.1f} -> out {w['ae']:.1f} dB")
    print("\nFailure cases (lowest gain per type; clean: lowest PSNR)\n", fails.round(2).to_string(index=False))

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task1_universal")
    with mlflow.start_run(run_name="test_eval"):
        mlflow.log_params({"checkpoint": str(CKPT.relative_to(ROOT)), "checkpoint_epoch": ck["epoch"], "test_rows": len(ds)})
        ae = df[df["method"] == "ae"]
        mlflow.log_metrics({"test_psnr": ae["psnr"].mean(), "test_ssim": ae["ssim"].mean(), "test_l1": ae["l1"].mean(),
                            **{f"test_psnr_{n}": v for n, v in ae.groupby("name")["psnr"].mean().items()}})
        for f in ("per_image.csv", "table_by_type.csv", "table_by_severity.csv", "failures.csv", "examples.png", "failures.png"):
            mlflow.log_artifact(str(TEST_OUT / f), "test")
    print("\nsaved to", TEST_OUT)


if __name__ == "__main__":
    main()
