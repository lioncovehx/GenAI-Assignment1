"""Soft mixture of experts: w = softmax(G(x) / tau), output = w0*x + sum of w_k * expert_k(x).
The gate and experts start from Task 2's classifier and specialists.

Self-check: python -m task3_soft_moe.moe
"""
import torch
from torch import nn

from task2_hard_routing.router import EXPERTS, load_classifier, load_specialist


class SoftMoE(nn.Module):
    def __init__(self, gate: nn.Module, experts: list[nn.Module], tau: float = 1.0):
        super().__init__()
        self.gate, self.experts, self.tau = gate, nn.ModuleList(experts), tau

    def forward(self, x):
        """-> (x_hat [B,3,H,W], weights [B,4], raw gate logits [B,4]). Every expert runs on every image (dense)."""
        logits = self.gate(x).float()
        w = (logits / self.tau).softmax(1)
        outs = torch.stack([x.float()] + [e(x).float() for e in self.experts], 1)  # [B,4,3,H,W]
        return (w[:, :, None, None, None] * outs).sum(1), w, logits


def build(tau: float = 1.0, dev="cpu") -> SoftMoE:
    """Initialised from Task 2 (the PDF forbids starting from scratch)."""
    return SoftMoE(load_classifier(dev), [load_specialist(k, dev) for k in EXPERTS], tau).to(dev)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from common.datasets import ManifestDataset
    from task2_hard_routing.train_classifier import evaluate

    dev = "cuda"
    moe = build(1.0, dev).eval()
    ds = ManifestDataset("val")
    loader = DataLoader(ds, 128)
    # class order check: the gate must classify exactly like Task 2's classifier did on val
    val, _, _ = evaluate(moe.gate, loader, ds.rows, dev, torch.contiguous_format)
    print(f"gate at init: val accuracy {val['acc']:.4f}, macro-F1 {val['macro_f1']:.4f}")
    assert val["acc"] > 0.998, "gate != Task 2 classifier: wrong checkpoint or class order"
    moe.eval()  # evaluate() leaves the gate in train mode
    with torch.no_grad():
        xc, x, y, _ = next(iter(loader))
        x_hat, w, _ = moe(xc.to(dev))
        assert x_hat.shape == xc.shape and 0 <= x_hat.min() and x_hat.max() <= 1
        assert torch.allclose(w.sum(1), torch.ones(len(w), device=dev), atol=1e-5)
        top = w.argmax(1).cpu()
        print(f"weights at tau=1: mean max weight {w.max(1).values.mean():.3f}; argmax == true label on "
              f"{(top == y).float().mean():.3f} of a val batch")
    print("soft MoE self-check OK")
