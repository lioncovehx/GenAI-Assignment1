"""Image preparation for the backend: a numpy port of the training code's resize and corruptions,
so the container doesn't need PyTorch. selftest.py checks it against the originals.
"""
import io

import numpy as np
from PIL import Image

SIZE = 128
LEVELS = {  # the assignment's fixed test levels, low / medium / high (common/corruptions.py TEST_LEVELS)
    "salt": {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    "blur": {"low": {"k": 3, "sigma": 0.7}, "medium": {"k": 5, "sigma": 1.5}, "high": {"k": 7, "sigma": 2.5}},
    "occlusion": {"low": {"n": 1, "coverage": 0.10}, "medium": {"n": 2, "coverage": 0.20},
                  "high": {"n": 3, "coverage": 0.35}},
}


def load_rgb(data: bytes) -> Image.Image:
    """Bytes of any image file -> RGB PIL image. Raises ValueError when the bytes are not an image."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # PIL raises several types for bad input
        raise ValueError("not an image") from e
    return img.convert("RGB")


def to_restoration_input(img: Image.Image) -> np.ndarray:
    """Pets pipeline: direct resize to 128 x 128 with bicubic (antialiased when shrinking) -> float32 [3,128,128] in [0,1]."""
    arr = np.asarray(img.resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).copy()


def to_sketch_input(img: Image.Image) -> np.ndarray:
    """Faces pipeline ("pad" mode): pad the shorter side by repeating edge pixels to a square, resize to
    128 x 128 -> float32 [3,128,128] in [-1,1]."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    d = abs(h - w)
    pad = ((0, 0), (d // 2, d - d // 2), (0, 0)) if h > w else ((d // 2, d - d // 2), (0, 0), (0, 0))
    sq = Image.fromarray(np.pad(a, pad, mode="edge"))
    arr = np.asarray(sq.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1).copy()


def gaussian_kernel1d(k: int, sigma: float) -> np.ndarray:
    """The same kernel torchvision's gaussian_blur builds: softmax(-(x / sigma)^2) over x in +-(k-1)/(2 sqrt 2)."""
    lim = (k - 1) / (2.0 * np.sqrt(2.0))
    x = np.linspace(-lim, lim, k, dtype=np.float64)
    e = np.exp(-((x / sigma) ** 2))
    return e / e.sum()


def blur(x: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding, as torchvision.transforms.v2.functional.gaussian_blur."""
    g = gaussian_kernel1d(k, sigma)
    r = k // 2
    p = np.pad(x.astype(np.float64), ((0, 0), (r, r), (r, r)), mode="reflect")
    rows = sum(g[i] * p[:, :, i:i + x.shape[2]] for i in range(k))  # along width
    out = sum(g[i] * rows[:, i:i + x.shape[1], :] for i in range(k))  # along height
    return out.astype(np.float32)


def place_rects(rng: np.random.Generator, n: int, coverage: float, size: int = SIZE) -> list:
    """n non-overlapping boxes covering `coverage` of the image (verbatim port of common/corruptions.place_rects)."""
    total = coverage * size * size
    while True:
        areas = rng.dirichlet([5.0] * n) * total
        rects = []
        for a in areas:
            r = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
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


def corrupt(x: np.ndarray, kind: str, severity: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """x float32 [3,128,128] in [0,1] -> (corrupted copy, human-readable label)."""
    if kind == "none":
        return x.copy(), "None (the image is used as given)"
    lv = LEVELS[kind][severity]
    out = x.copy()
    if kind == "salt":  # per pixel, all channels together: black or white dots
        hit = rng.random(x.shape[1:]) < lv["p"]
        white = rng.random(x.shape[1:]) < 0.5
        out[:, hit] = white[hit].astype(np.float32)
        return out, f"Salt-and-pepper, {severity}, p = {lv['p']}"
    if kind == "blur":
        return blur(x, lv["k"], lv["sigma"]), f"Gaussian blur, {severity}, kernel {lv['k']}, sigma {lv['sigma']}"
    rects = place_rects(rng, lv["n"], lv["coverage"])
    for x0, y0, x1, y1 in rects:
        out[:, y0:y1, x0:x1] = 0.0
    covered = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects) / SIZE**2
    boxes = "1 box" if lv["n"] == 1 else f"{lv['n']} boxes"
    return out, f"Occlusion, {severity}, {boxes}, {covered:.0%} covered"


def png_data_url(x: np.ndarray, value_range=(0.0, 1.0)) -> str:
    """float [C,H,W] (C = 1 or 3) -> PNG data URL."""
    import base64
    lo, hi = value_range
    a = np.clip((x - lo) / (hi - lo), 0, 1)
    a = (a * 255 + 0.5).astype(np.uint8)
    img = Image.fromarray(a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
