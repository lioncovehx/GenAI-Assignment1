"""Export the whole soft MoE (gate, branches, weighted sum) as one ONNX graph and check it matches PyTorch.
Outputs: output [B,3,128,128], weights [B,4], branches [B,4,3,128,128].

  python -m task3_soft_moe.export_onnx [--run final]
"""
import argparse
import time

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from common.data import ROOT
from common.datasets import ManifestDataset
from task3_soft_moe.moe import build
from task3_soft_moe.train import OUT

ONNX = OUT / "soft_moe.onnx"


class Export(nn.Module):
    """The same computation as SoftMoE.forward, also returning each branch's output; raw logits dropped."""

    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        w = (self.moe.gate(x) / self.moe.tau).softmax(1)
        branches = torch.stack([x] + [e(x) for e in self.moe.experts], 1)
        return (w[:, :, None, None, None] * branches).sum(1), w, branches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="final")
    a = p.parse_args()
    ck = torch.load(OUT / a.run / "best.pt", map_location="cpu", weights_only=False)
    moe = build(ck["cfg"]["tau"], "cpu")
    moe.load_state_dict(ck["state_dict"])
    model = Export(moe).eval()  # eval: BatchNorm uses stored statistics, dropout off

    ONNX.with_name(ONNX.name + ".data").unlink(missing_ok=True)  # stale weights file from an older export
    # external_data=False: one self-contained file
    torch.onnx.export(model, (torch.rand(2, 3, 128, 128),), ONNX, input_names=["input"],
                      output_names=["output", "weights", "branches"], dynamo=True,
                      dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, external_data=False)
    size_mb = ONNX.stat().st_size / 1e6
    assert size_mb > 50, f"{size_mb:.1f} MB: weights missing from the .onnx file (expected ~56 MB)"
    print(f"exported {ONNX} ({size_mb:.1f} MB) from epoch {ck['epoch']}, tau {ck['cfg']['tau']:.3f}")

    ds = ManifestDataset("test")
    x = torch.stack([ds[j][0] for j in np.linspace(0, len(ds) - 1, 32).astype(int)])
    with torch.no_grad():
        ref_out, ref_w, _ = (t.numpy() for t in model(x))
        ref_moe = moe(x)[0].numpy()
    assert np.allclose(ref_out, ref_moe, atol=1e-5), "export wrapper differs from SoftMoE.forward"
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    out, w = sess.run(["output", "weights"], {"input": x.numpy()})
    diff_out, diff_w = float(np.abs(out - ref_out).max()), float(np.abs(w - ref_w).max())
    np.testing.assert_allclose(out, ref_out, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(w, ref_w, atol=1e-4, rtol=1e-3)
    assert np.allclose(w.sum(1), 1, atol=1e-5)
    print(f"parity OK on 32 test images: output max diff {diff_out:.2e}, weights max diff {diff_w:.2e} (tolerance 1e-4)")

    one = x[:1].numpy()
    assert sess.run(["output"], {"input": one})[0].shape == (1, 3, 128, 128)  # dynamic batch works
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, {"input": one})
        times.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(times[5:]))
    print(f"onnxruntime CPU: {ms:.1f} ms per image (median of 45 runs; all branches run)")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task3_moe")
    with mlflow.start_run(run_name="onnx_export"):
        mlflow.log_params({"checkpoint": f"outputs/{a.run}/best.pt", "checkpoint_epoch": ck["epoch"],
                           "onnx_io": "input [b,3,128,128] [0,1] -> output [b,3,128,128], weights [b,4], branches [b,4,3,128,128]",
                           "parity_images": 32})
        mlflow.log_metrics({"parity_output_max_abs_diff": diff_out, "parity_weights_max_abs_diff": diff_w,
                            "onnx_size_mb": size_mb, "cpu_ms_per_image": ms})
        mlflow.log_artifact(str(ONNX), "onnx")


if __name__ == "__main__":
    main()
