"""Export the classifier and the three specialists to ONNX, check each file and the full route against PyTorch.

  python -m task2_hard_routing.export_onnx
"""
import time

import mlflow
import numpy as np
import onnxruntime as ort
import torch

from common.data import ROOT
from common.datasets import ManifestDataset
from task2_hard_routing.router import EXPERTS, HardRouter, load_classifier, load_specialist
from task2_hard_routing.train_classifier import OUT

ONNX_DIR = OUT / "onnx"


def export(model, path, output_name, min_mb):
    path.with_name(path.name + ".data").unlink(missing_ok=True)  # stale weights file from an older export
    # external_data=False: weights inside the .onnx itself
    torch.onnx.export(model, (torch.rand(2, 3, 128, 128),), path, input_names=["input"], output_names=[output_name],
                      dynamo=True, dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, external_data=False)
    size_mb = path.stat().st_size / 1e6
    assert size_mb > min_mb, f"{path.name}: {size_mb:.1f} MB, weights missing from the .onnx file"
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), size_mb


def ms_per_image(sess, one):
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, {"input": one})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times[5:]))  # skip warm-up runs


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    ds = ManifestDataset("test")
    x = torch.stack([ds[j][0] for j in np.linspace(0, len(ds) - 1, 32).astype(int)])  # spread over all 10 conditions
    models = {"classifier": load_classifier(), **{f"expert_{k}": load_specialist(k) for k in EXPERTS}}
    sessions, log = {}, {}
    for name, model in models.items():
        is_clf = name == "classifier"
        sess, size_mb = export(model, ONNX_DIR / f"{name}.onnx", "logits" if is_clf else "output", 1.5 if is_clf else 10)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(out - ref).max())
        np.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-3)
        assert sess.run(None, {"input": x[:1].numpy()})[0].shape[0] == 1  # dynamic batch works
        ms = ms_per_image(sess, x[:1].numpy())
        print(f"{name}.onnx: {size_mb:.1f} MB, parity max |onnx - torch| = {diff:.2e}, {ms:.1f} ms/image (CPU)")
        sessions[name] = sess
        log |= {f"{name}_size_mb": size_mb, f"{name}_parity_max_abs_diff": diff, f"{name}_cpu_ms": ms}

    # end to end: route the 32 images with the ONNX files (as the backend will) and compare with router.py
    router = HardRouter(models["classifier"], [models[f"expert_{k}"] for k in EXPERTS]).eval()
    with torch.no_grad():
        ref_out, _, ref_route = router(x)
    route = sessions["classifier"].run(None, {"input": x.numpy()})[0].argmax(1)
    out = x.numpy().copy()  # identity for clean
    for k, name in enumerate(EXPERTS, start=1):
        m = route == k
        if m.any():
            out[m] = sessions[f"expert_{name}"].run(None, {"input": x.numpy()[m]})[0]
    assert (route == ref_route.numpy()).all(), "ONNX routing chose a different expert"
    e2e = float(np.abs(out - ref_out.numpy()).max())
    np.testing.assert_allclose(out, ref_out.numpy(), atol=1e-4, rtol=1e-3)
    print(f"end to end: routes identical on 32/32 images ({np.bincount(route, minlength=4).tolist()} per class), "
          f"max |onnx - torch| = {e2e:.2e}")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("task2_routing")
    with mlflow.start_run(run_name="onnx_export"):
        mlflow.log_params({"files": "classifier, expert_salt, expert_blur, expert_occlusion", "parity_images": 32,
                           "classifier_io": "input [b,3,128,128] [0,1] -> logits [b,4]",
                           "expert_io": "input [b,3,128,128] [0,1] -> output [b,3,128,128] [0,1]"})
        mlflow.log_metrics({**log, "end_to_end_max_abs_diff": e2e})
        mlflow.log_artifacts(str(ONNX_DIR), "onnx")


if __name__ == "__main__":
    main()
