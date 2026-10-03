"""The four input conditions from the assignment.

A corruption is fully described by a params dict, so `apply(x, params)` is deterministic:
runtime training samples new params every load; manifests store params and replay them.
"""
import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

LABELS = ("clean", "salt", "blur", "occlusion")  # class index = position; same order in Tasks 2-3
SEVERITIES = ("low", "medium", "high")

# Fixed test levels from the PDF (low, medium, high)
TEST_LEVELS = {
    1: [{"p": 0.03}, {"p": 0.08}, {"p": 0.15}],
    2: [{"k": 3, "sigma": 0.7}, {"k": 5, "sigma": 1.5}, {"k": 7, "sigma": 2.5}],
    3: [{"n": 1, "coverage": 0.10}, {"n": 2, "coverage": 0.20}, {"n": 3, "coverage": 0.35}],
}


def place_rects(rng: np.random.Generator, n: int, coverage: float, size: int = 128) -> list:
    """n non-overlapping rectangles whose union covers `coverage` of the image.
    Non-overlap makes union = sum, so the coverage target is met by construction."""
    total = coverage * size * size
    while True:  # restart if a layout can't be packed (rare)
        areas = rng.dirichlet([5.0] * n) * total  # alpha=5: similar-sized rects, not slivers
        rects = []
        for a in areas:
            r = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))  # aspect ratio h/w in [0.5, 2]
            h = int(round(min(np.sqrt(a * r), size)))
            w = int(round(min(a / max(h, 1), size)))
            for _ in range(200):
                y0, x0 = int(rng.integers(0, size - h + 1)), int(rng.integers(0, size - w + 1))
                c = [x0, y0, x0 + w, y0 + h]
                if all(c[2] <= o[0] or o[2] <= c[0] or c[3] <= o[1] or o[3] <= c[1] for o in rects):
                    rects.append(c)
                    break
            else:
                break
        if len(rects) == n:
            return rects


def make_params(label: int, rng: np.random.Generator, level: dict | None = None) -> dict:
    """Params for one corruption. level=None -> sample from the training ranges; else use a fixed test level."""
    seed = int(rng.integers(2**31))
    if label == 0:
        return {"label": 0, "seed": seed}
    if label == 1:
        p = level["p"] if level else float(rng.uniform(0.02, 0.15))
        return {"label": 1, "p": p, "seed": seed}
    if label == 2:
        if level:
            return {"label": 2, **level, "seed": seed}
        return {"label": 2, "k": int(rng.choice([3, 5, 7])), "sigma": float(rng.uniform(0.5, 2.5)), "seed": seed}
    n = level["n"] if level else int(rng.integers(1, 4))
    cov = level["coverage"] if level else float(rng.uniform(0.10, 0.35))
    rects = place_rects(rng, n, cov)
    actual = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects) / 128**2
    return {"label": 3, "n": n, "coverage": actual, "rects": rects, "seed": seed}


def sample_params(rng: np.random.Generator) -> dict:
    """Training sampler: each of the 4 conditions with probability 1/4, then a random severity."""
    return make_params(int(rng.integers(4)), rng)


def apply(x: torch.Tensor, params: dict) -> torch.Tensor:
    """x: float [3,H,W] in [0,1]. Returns a new corrupted tensor."""
    label = params["label"]
    if label == 0:
        return x.clone()
    if label == 1:  # per pixel location, all channels together -> black/white dots
        g = torch.Generator().manual_seed(params["seed"])
        h, w = x.shape[-2:]
        hit = torch.rand(h, w, generator=g) < params["p"]
        white = torch.rand(h, w, generator=g) < 0.5
        out = x.clone()
        out[:, hit] = white[hit].to(x.dtype)
        return out
    if label == 2:
        k, s = params["k"], params["sigma"]
        return F.gaussian_blur(x, [k, k], [s, s])
    out = x.clone()
    for x0, y0, x1, y1 in params["rects"]:
        out[:, y0:y1, x0:x1] = 0.0
    return out


def severity_bin(params: dict) -> str | None:
    """Low/medium/high bin for continuously sampled params: each range split into thirds."""
    v, lo, hi = {
        1: lambda: (params.get("p"), 0.02, 0.15),
        2: lambda: (params.get("sigma"), 0.5, 2.5),
        3: lambda: (params.get("coverage"), 0.10, 0.35),
    }.get(params["label"], lambda: (None, 0, 1))()
    if v is None:
        return None
    return SEVERITIES[min(int((v - lo) / (hi - lo) * 3), 2)]
