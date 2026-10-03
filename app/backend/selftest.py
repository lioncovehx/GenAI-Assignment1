"""Checks the backend against the training code (corruptions, face preprocessing, every endpoint).
Run from the project root in the training environment: MODELS_DIR=models python app/backend/selftest.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "app/backend")]
import preprocess as P  # noqa: E402
from common import corruptions as C  # noqa: E402
from common.datasets import CleanDataset  # noqa: E402

imgs = CleanDataset("val")
xs = [imgs[i].numpy() for i in range(0, 700, 35)]  # 20 val images, float [3,128,128]

# 1. corruptions
worst = 0.0
for sev, lv in P.LEVELS["blur"].items():
    for x in xs:
        ref = C.apply(torch.from_numpy(x), {"label": 2, **lv, "seed": 0}).numpy()
        worst = max(worst, float(np.abs(P.blur(x, lv["k"], lv["sigma"]) - ref).max()))
print(f"blur: max |numpy - torchvision| over 20 images x 3 levels = {worst:.2e}")
assert worst < 1e-5

for sev, lv in P.LEVELS["occlusion"].items():
    out, label = P.corrupt(xs[0], "occlusion", sev, np.random.default_rng(1))
    rects = P.place_rects(np.random.default_rng(1), lv["n"], lv["coverage"])
    ref = C.apply(torch.from_numpy(xs[0]), {"label": 3, "rects": rects}).numpy()
    assert np.array_equal(out, ref), sev
    print(f"occlusion {sev}: identical to training code ({label})")

for sev, lv in P.LEVELS["salt"].items():
    out, _ = P.corrupt(xs[0], "salt", sev, np.random.default_rng(2))
    changed = np.any(out != xs[0], axis=0)
    dots = out[:, changed]
    assert np.all((dots == 0) | (dots == 1))
    print(f"salt {sev}: {changed.mean():.3f} of pixels set to black/white (requested p = {lv['p']})")

# 2. face preprocessing
from task4_cgan_sketch.data import FS2KDataset, _load, pairs  # noqa: E402
from PIL import Image  # noqa: E402

rows = pairs("test")[:10]
diffs = []
for r in rows:
    ref = _load(r, "pad")[0].float().numpy() / 127.5 - 1
    ours = P.to_sketch_input(Image.open(r["photo"]).convert("RGB"))
    diffs.append(np.abs(ours - ref).mean())
print(f"face pad+resize: mean |ours - training| = {np.mean(diffs) * 127.5:.2f} grey levels of 255 (resize filters differ)")
assert np.mean(diffs) * 127.5 < 3

# 3. endpoints
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

c = TestClient(main.app)
h = c.get("/api/health").json()
print("health:", h)
assert h["status"] == "ok"
pets = c.get("/api/samples", params={"kind": "pets"}).json()
faces = c.get("/api/samples", params={"kind": "faces"}).json()
assert len(pets) == 6 and len(faces) == 6 and c.get(pets[0]["url"]).status_code == 200
for kind, expect in (("none", None), ("salt", "salt"), ("blur", "blur"), ("occlusion", "occlusion")):
    r = c.post("/api/restore/hard", data={"sample_id": pets[2]["id"], "corruption": kind, "severity": "high"}).json()
    print(f"hard routing, {kind} high: predicted {r['predicted']}, expert {r['expert']}, {r['inference_ms']['total']} ms")
    if expect:
        assert r["predicted"] == expect
u = c.post("/api/restore/universal", data={"sample_id": pets[0]["id"], "corruption": "salt"}).json()
s = c.post("/api/restore/soft", data={"sample_id": pets[0]["id"], "corruption": "blur", "severity": "low"}).json()
print(f"universal: {u['corruption']['label']}, {u['inference_ms']} ms")
print(f"soft: weights {s['weights']}, dominant {s['dominant']}, {s['inference_ms']} ms")
assert abs(sum(s["weights"].values()) - 1) < 1e-3 and set(s["branch_pngs"]) == set(main.BRANCHES)
k = c.post("/api/sketch", data={"sample_id": faces[0]["id"], "style": 2, "all_styles": "true"}).json()
assert k["style"] == 2 and len(k["all_styles"]) == 3
print(f"sketch: style 2 + all styles, {k['inference_ms']} ms per sketch")
bad = c.post("/api/restore/universal", files={"image": ("x.png", b"not an image", "image/png")})
assert bad.status_code == 400, bad.text
assert c.post("/api/restore/universal", data={}).status_code == 400
print("errors: non-image upload and missing input both rejected with 400")
assert c.get("/api/experiments").status_code == 200
print("backend self-test OK")
