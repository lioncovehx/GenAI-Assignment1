"""Test-set check of the occlusion post-processing: keep the input outside the detected boxes.
Runs the served ONNX models on all of manifests/test.json, raw output vs kept-pixels output.
Run from the project root in the training environment: MODELS_DIR=models python app/backend/eval_occlusion.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("ORT_THREADS", "0")  # 0 = all cores; the app itself uses 4
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "app/backend")]
import main  # noqa: E402
import preprocess as P  # noqa: E402
from common.datasets import ManifestDataset  # noqa: E402
from common.metrics import per_image  # noqa: E402
from common.viz import restoration_grid  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
S = main.SESSIONS
ds = ManifestDataset("test")
rows, examples = [], []
for start in tqdm(range(0, len(ds), 64)):
    batch = [ds[j] for j in range(start, min(start + 64, len(ds)))]
    x = torch.stack([b[0] for b in batch]).numpy()
    clean = torch.stack([b[1] for b in batch])
    (logits,) = S["classifier"].run(None, {"input": x})
    (hard,) = S["expert_occlusion"].run(None, {"input": x})
    soft, w, _ = S["soft_moe"].run(None, {"input": x})
    for i, (_, _, label, j) in enumerate(batch):
        r = ds.rows[j]
        mask = P.find_boxes(x[i])
        true = np.zeros_like(mask)
        for x0, y0, x1, y1 in r.get("rects", []):
            true[y0:y1, x0:x1] = True
        hard_k, share = P.keep_known(x[i], hard[i])
        soft_k, _ = P.keep_known(x[i], soft[i])
        outs = {"hard_raw": hard[i], "hard_kept": hard_k, "soft_raw": soft[i], "soft_kept": soft_k}
        sc = {k: per_image(torch.from_numpy(v)[None], clean[i:i + 1]) for k, v in outs.items()}
        rows.append({"name": r["name"], "severity": r["severity"] or "-", "routed_occ": int(logits[i].argmax()) == 3,
                     "gate_occ": int(w[i].argmax()) == 3, "gate_occ_expert": int(w[i, 1:].argmax()) == 2,
                     "box_share": share, "iou": (mask & true).sum() / max((mask | true).sum(), 1) if true.any() else np.nan,
                     **{f"{k}_{m}": float(v[m]) for k, v in sc.items() for m in ("psnr", "ssim")}})
        if r["name"] == "occlusion" and len(examples) < 6 and start % 3200 == 0:
            examples.append((clean[i], torch.from_numpy(x[i]), torch.from_numpy(hard[i]), torch.from_numpy(hard_k)))

df = pd.DataFrame(rows)
df.to_csv(OUT / "occlusion_eval.csv", index=False)
cols = [f"{k}_{m}" for m in ("psnr", "ssim") for k in ("hard_raw", "hard_kept", "soft_raw", "soft_kept")]
occ = df[df.name == "occlusion"]
print("\n== occlusion test images, all of them (kept = model output inside the detected boxes only) ==")
print(occ.groupby("severity")[cols].mean().round(3).to_string())
print(f"\nbox detection vs the true boxes: mean IoU {occ.iou.mean():.3f}, images with no box found {(occ.box_share == 0).mean():.1%}")
print(f"rule fires on occlusion images: hard (classifier says occlusion) {occ.routed_occ.mean():.1%}, "
      f"soft (gate argmax occlusion) {occ.gate_occ.mean():.1%}, soft (occlusion is the largest expert) {occ.gate_occ_expert.mean():.1%}")
other = df[df.name != "occlusion"]
print("\n== other images: how often would a rule fire, and does a box get found ==")
print(other.groupby("name").agg(routed_occ=("routed_occ", "mean"), gate_occ=("gate_occ", "mean"),
                                box_found=("box_share", lambda s: (s > 0).mean())).round(4).to_string())
c, x, h, k = (torch.stack(t) for t in zip(*examples))
restoration_grid(c, x, h, OUT / "occlusion_raw.png")
restoration_grid(c, x, k, OUT / "occlusion_kept.png")
print(f"\nwrote {OUT}/occlusion_eval.csv, occlusion_raw.png, occlusion_kept.png")
