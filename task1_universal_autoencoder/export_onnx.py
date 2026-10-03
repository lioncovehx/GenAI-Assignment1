"""Export the final Task 1 model to ONNX and check it matches PyTorch (atol 1e-4).

  python -m task1_universal_autoencoder.export_onnx
"""
import time

import mlflow
import numpy as np
import onnxruntime as ort
import torch

from common.data import ROOT
from common.datasets import ManifestDataset
from task1_universal_autoencoder.model import ConvAE
from task1_universal_autoencoder.train import MODEL_KEYS, OUT

CKPT = OUT / "final" / "best.pt"
ONNX = OUT / "universal_ae.onnx"


def main():
    ck = torch.load(CKPT, map_location="cpu")
    model = ConvAE(**{k: ck["cfg"][k] for k in MODEL_KEYS}).eval()  # eval: dropout off, BatchNorm uses stored stats
    model.load_state_dict(ck["state_dict"])

    ONNX.with_name(ONNX.name + ".data").unlink(missing_ok=True)  # stale weights file from an older export
    # external_data=False: weights inside the .onnx itself (one self-contained file for MLflow and the app)
    torch.onnx.export(model, (torch.rand(1, 3, 128, 128),), ONNX, input_names=["input"], output_names=["output"],
                      dynamo=True, dynamic_shapes={"x": {0: torch.export.Dim("batch")}},  # key = forward()'s arg name
                      external_data=False)
    size_mb = ONNX.stat().st_size / 1e6
    assert size_mb > 10, f"{size_mb:.1f} MB: weights missing from the .onnx file"
    print(f"exported {ONNX} ({size_mb:.1f} MB) from epoch {ck['epoch']}")

    # parity on real inputs: 32 corrupted test images spread over all 10 conditions
    ds = ManifestDataset("test")
    x = torch.stack([ds[j][0] for j in np.linspace(0, len(ds) - 1, 32).astype(int)])
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    out = sess.run(["output"], {"input": x.numpy()})[0]
    max_diff = float(np.abs(out - ref).max())
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)
    print(f"parity OK on 32 test images: max |onnx - torch| = {max_diff:.2e} (tolerance 1e-4)")

    # one image at a time, as the app will call it
    one = x[:1].numpy()
    assert sess.run(["output"], {"input": one})[0].shape == (1, 3, 128, 128)  # dynamic batch works
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(["output"], {"input": one})
        times.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(times[5:]))  # skip warm-up runs
    print(f"onnxruntime CPU: {ms:.1f} ms per image (median of 45 runs)")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task1_universal")
    with mlflow.start_run(run_name="onnx_export"):
        mlflow.log_params({"checkpoint": str(CKPT.relative_to(ROOT)), "checkpoint_epoch": ck["epoch"],
                           "onnx_inputs": "input [batch,3,128,128] float32 [0,1]", "parity_images": 32})
        mlflow.log_metrics({"parity_max_abs_diff": max_diff, "onnx_size_mb": size_mb, "cpu_ms_per_image": ms})
        mlflow.log_artifact(str(ONNX), "onnx")


if __name__ == "__main__":
    main()
