"""Loupe backend: serves the four assignment models (ONNX).

Run: uvicorn main:app --host 0.0.0.0 --port 8000   (from app/backend; models from $MODELS_DIR, default /models)
"""
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import preprocess as P

HERE = Path(__file__).parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
SAMPLES_DIR = HERE / "samples"
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5001")
MAX_UPLOAD = 15 * 1024 * 1024
CLASSES = ("clean", "salt", "blur", "occlusion")  # classifier output order = training label order
BRANCHES = ("identity", "salt", "blur", "occlusion")  # soft MoE branch order

FILES = {"universal": ["universal_ae"], "hard_routing": ["classifier", "expert_salt", "expert_blur", "expert_occlusion"],
         "soft_moe": ["soft_moe"], "sketch": ["sketch_generator"]}
SESSIONS: dict[str, ort.InferenceSession] = {}


def session_options() -> ort.SessionOptions:
    """Seven models share one container: small thread pools without busy-waiting stop them competing for the CPU."""
    o = ort.SessionOptions()
    o.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
    o.inter_op_num_threads = 1
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return o


for names in FILES.values():
    for n in names:
        try:
            SESSIONS[n] = ort.InferenceSession(str(MODELS_DIR / f"{n}.onnx"), sess_options=session_options(),
                                               providers=["CPUExecutionProvider"])
        except Exception as e:  # a missing model degrades its workspace, the rest keeps working
            print(f"could not load {n}: {e}")

# Warm-up: the first run of each session is 10-50x slower, so do it at startup.
for n, sess in SESSIONS.items():
    feeds = {i.name: (np.zeros((1, 3, P.SIZE, P.SIZE), np.float32) if i.name != "style" else np.zeros(1, np.int64))
             for i in sess.get_inputs()}
    sess.run(None, feeds)

app = FastAPI(title="Loupe API")


def run(name: str, feeds: dict) -> tuple[list, float]:
    if name not in SESSIONS:
        raise HTTPException(503, f"The {name} model is not loaded on the server.")
    t0 = time.perf_counter()
    out = SESSIONS[name].run(None, feeds)
    return out, (time.perf_counter() - t0) * 1000


async def read_input(image: UploadFile | None, sample_id: str | None):
    """Exactly one of an upload or a bundled sample -> RGB PIL image."""
    if (image is None) == (sample_id is None):
        raise HTTPException(400, "Send an image or a sample_id.")
    if sample_id is not None:
        path = SAMPLES_DIR / f"{sample_id}.png"
        if not sample_id.replace("-", "").isalnum() or not path.exists():
            raise HTTPException(404, f"Unknown sample '{sample_id}'.")
        data = path.read_bytes()
    else:
        data = await image.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "This image is larger than 15 MB. Choose a smaller file.")
    try:
        return P.load_rgb(data)
    except ValueError:
        raise HTTPException(400, "This file is not an image. Choose a PNG or JPG.")


async def prepare(image, sample_id, corruption, severity):
    img = await read_input(image, sample_id)
    clean = P.to_restoration_input(img)
    x, label = P.corrupt(clean, corruption, severity, np.random.default_rng())
    return x, {"type": corruption, "severity": severity, "label": label}


Corruption = Literal["none", "salt", "blur", "occlusion"]
Severity = Literal["low", "medium", "high"]


@app.get("/api/health")
def health():
    models = {k: all(n in SESSIONS for n in names) for k, names in FILES.items()}
    return {"status": "ok" if all(models.values()) else "degraded", "models": models}


@app.get("/api/samples")
def samples(kind: Literal["pets", "faces"]):
    noun = "Test pet" if kind == "pets" else "Test face"
    return [{"id": p.stem, "url": f"/api/samples/{p.name}", "label": f"{noun} {i + 1}"}
            for i, p in enumerate(sorted(SAMPLES_DIR.glob(f"{kind}-*.png")))]


@app.get("/api/samples/{name}")
def sample_file(name: str):
    path = SAMPLES_DIR / name
    if not name.endswith(".png") or "/" in name or not path.exists():
        raise HTTPException(404, "Unknown sample.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/restore/universal")
async def restore_universal(image: UploadFile | None = File(None), sample_id: str | None = Form(None),
                            corruption: Corruption = Form("none"), severity: Severity = Form("medium")):
    x, info = await prepare(image, sample_id, corruption, severity)
    (y,), ms = run("universal_ae", {"input": x[None]})
    return {"input_png": P.png_data_url(x), "output_png": P.png_data_url(y[0]), "corruption": info,
            "inference_ms": round(ms, 2)}


@app.post("/api/restore/hard")
async def restore_hard(image: UploadFile | None = File(None), sample_id: str | None = Form(None),
                       corruption: Corruption = Form("none"), severity: Severity = Form("medium")):
    x, info = await prepare(image, sample_id, corruption, severity)
    (logits,), ms_c = run("classifier", {"input": x[None]})
    z = logits[0] - logits[0].max()
    p = np.exp(z) / np.exp(z).sum()
    k = int(p.argmax())
    if k == 0:  # identity bypass: a clean image is returned unchanged
        y, ms_e, expert = x, 0.0, "identity"
    else:
        expert = CLASSES[k]
        (out,), ms_e = run(f"expert_{expert}", {"input": x[None]})
        y = out[0]
    return {"input_png": P.png_data_url(x), "output_png": P.png_data_url(y), "corruption": info,
            "probs": {c: round(float(v), 4) for c, v in zip(CLASSES, p)}, "predicted": CLASSES[k], "expert": expert,
            "inference_ms": {"classifier": round(ms_c, 2), "expert": round(ms_e, 2), "total": round(ms_c + ms_e, 2)}}


@app.post("/api/restore/soft")
async def restore_soft(image: UploadFile | None = File(None), sample_id: str | None = Form(None),
                       corruption: Corruption = Form("none"), severity: Severity = Form("medium")):
    x, info = await prepare(image, sample_id, corruption, severity)
    (y, w, branches), ms = run("soft_moe", {"input": x[None]})
    weights = {b: round(float(v), 4) for b, v in zip(BRANCHES, w[0])}
    return {"input_png": P.png_data_url(x), "output_png": P.png_data_url(y[0]), "corruption": info,
            "weights": weights, "dominant": max(weights, key=weights.get),
            "branch_pngs": {b: P.png_data_url(branches[0, i]) for i, b in enumerate(BRANCHES)},
            "inference_ms": round(ms, 2)}


@app.post("/api/sketch")
async def sketch(image: UploadFile | None = File(None), sample_id: str | None = Form(None),
                 style: int = Form(1, ge=1, le=3), all_styles: bool = Form(False)):
    x = P.to_sketch_input(await read_input(image, sample_id))
    styles = [1, 2, 3] if all_styles else [style]
    (out,), ms = run("sketch_generator", {"photo": np.repeat(x[None], len(styles), 0),
                                          "style": np.array([s - 1 for s in styles], dtype=np.int64)})
    by_style = {s: P.png_data_url(out[i], (-1.0, 1.0)) for i, s in enumerate(styles)}
    body = {"photo_png": P.png_data_url(x, (-1.0, 1.0)), "sketch_png": by_style[style], "style": style,
            "inference_ms": round(ms / len(styles), 2)}
    if all_styles:
        body["all_styles"] = [{"style": s, "sketch_png": by_style[s]} for s in styles]
    return body


@app.get("/api/experiments")
def experiments():
    return json.loads((HERE / "experiments.json").read_text())


@app.get("/api/experiments/meta")
def experiments_meta():
    return {"mlflow_url": MLFLOW_URL}
