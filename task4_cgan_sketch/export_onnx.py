"""Export the Task 4 generator to ONNX and check it matches PyTorch (atol 1e-4).
Inputs: photo [B,3,128,128] in [-1,1], style [B] int64 in {0,1,2}. Output: sketch [B,1,128,128] in [-1,1].

  python -m task4_cgan_sketch.export_onnx
"""
import time

import mlflow
import numpy as np
import onnxruntime as ort
import torch

from common.data import ROOT
from task4_cgan_sketch.data import FS2KDataset
from task4_cgan_sketch.models import UNetGenerator
from task4_cgan_sketch.train import OUT

CKPT = OUT / "final" / "best.pt"
ONNX = OUT / "sketch_generator.onnx"


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    G = UNetGenerator(c["base_ch"], c["emb_dim"], c["dropout"], c["g_cond"]).eval()  # eval: dropout off
    G.load_state_dict(ck["G"])

    ONNX.with_name(ONNX.name + ".data").unlink(missing_ok=True)  # stale weights file from an older export
    batch = torch.export.Dim("batch")
    # external_data=False: weights inside the .onnx itself (one self-contained file)
    torch.onnx.export(G, (torch.rand(2, 3, 128, 128) * 2 - 1, torch.tensor([0, 2])), ONNX,
                      input_names=["photo", "style"], output_names=["sketch"], dynamo=True,
                      dynamic_shapes={"photo": {0: batch}, "style": {0: batch}},  # keys = forward()'s arg names
                      external_data=False)
    size_mb = ONNX.stat().st_size / 1e6
    assert size_mb > 80, f"{size_mb:.1f} MB: weights missing from the .onnx file (expected ~94 MB)"
    print(f"exported {ONNX} ({size_mb:.1f} MB) from epoch {ck['epoch']}, base_ch {c['base_ch']}")

    # parity on real inputs: 32 test photos spread over the whole test set, each in its own style
    ds = FS2KDataset("test", "pad")
    idx = np.linspace(0, len(ds) - 1, 32).astype(int)
    photo = torch.stack([ds[i][0] for i in idx])
    style = torch.stack([ds[i][2] for i in idx]).long()
    with torch.no_grad():
        ref = G(photo, style).numpy()
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    out = sess.run(["sketch"], {"photo": photo.numpy(), "style": style.numpy()})[0]
    max_diff = float(np.abs(out - ref).max())
    np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)
    print(f"parity OK on 32 test photos: max |onnx - torch| = {max_diff:.2e} (tolerance 1e-4)")

    # one photo at a time, as the app will call it; also check the style input really changes the output
    one = photo[:1].numpy()
    outs = [sess.run(["sketch"], {"photo": one, "style": np.array([s])})[0] for s in range(3)]
    assert outs[0].shape == (1, 1, 128, 128)  # dynamic batch works
    style_diff = float(np.mean([np.abs(outs[i] - outs[j]).mean() for i, j in ((0, 1), (0, 2), (1, 2))])) / 2
    print(f"style check: mean |difference| between styles = {style_diff:.4f} on [0,1] scale (0 = style ignored)")
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(["sketch"], {"photo": one, "style": np.array([0])})
        times.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(times[5:]))  # skip warm-up runs
    print(f"onnxruntime CPU: {ms:.1f} ms per image (median of 45 runs)")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task4_sketch")
    with mlflow.start_run(run_name="onnx_export"):
        mlflow.log_params({"checkpoint": str(CKPT.relative_to(ROOT)), "checkpoint_epoch": ck["epoch"],
                           "onnx_inputs": "photo [batch,3,128,128] float32 [-1,1]; style [batch] int64 {0,1,2}",
                           "onnx_output": "sketch [batch,1,128,128] float32 [-1,1]", "parity_images": 32})
        mlflow.log_metrics({"parity_max_abs_diff": max_diff, "onnx_size_mb": size_mb, "cpu_ms_per_image": ms,
                            "onnx_style_diff": style_diff})
        mlflow.log_artifact(str(ONNX), "onnx")


if __name__ == "__main__":
    main()
