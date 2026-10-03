"""Hard routing: classifier argmax picks one specialist; "clean" returns the input unchanged.

Self-check: python -m task2_hard_routing.router
"""
import torch
from torch import nn

from task1_universal_autoencoder.model import ConvAE
from task1_universal_autoencoder.train import MODEL_KEYS
from task2_hard_routing.classifier import build
from task2_hard_routing.train_classifier import OUT

EXPERTS = ("salt", "blur", "occlusion")  # label 1, 2, 3 (common/corruptions.py LABELS order)


class HardRouter(nn.Module):
    def __init__(self, classifier: nn.Module, experts: list[nn.Module]):
        super().__init__()
        self.classifier, self.experts = classifier, nn.ModuleList(experts)

    def forward(self, x, label=None):
        """label given -> oracle mode. Returns (restored, probabilities [B,4], route [B])."""
        p = self.classifier(x).float().softmax(1)
        r = label if label is not None else p.argmax(1)
        out = x.clone()  # identity for r == 0 (clean)
        for k, expert in enumerate(self.experts, start=1):
            m = r == k
            if m.any():
                out[m] = expert(x[m]).to(out.dtype)
        return out, p, r


def load_specialist(kind: str, dev="cpu") -> ConvAE:
    ck = torch.load(OUT / f"final_{kind}" / "best.pt", map_location="cpu", weights_only=False)
    m = ConvAE(**{k: ck["cfg"][k] for k in MODEL_KEYS})
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval()


def load_classifier(dev="cpu") -> nn.Module:
    ck = torch.load(OUT / "final" / "best.pt", map_location="cpu", weights_only=False)
    m = build(ck["cfg"])
    m.load_state_dict(ck["state_dict"])
    return m.to(dev).eval()


def load(dev="cpu") -> HardRouter:
    return HardRouter(load_classifier(dev), [load_specialist(k, dev) for k in EXPERTS]).eval()


if __name__ == "__main__":
    from common.datasets import ManifestDataset
    router = load()
    ds = ManifestDataset("val")
    batch = [ds[j] for j in range(0, len(ds), 37)][:16]
    xc = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[2] for b in batch])
    with torch.no_grad():
        out, p, r = router(xc, y)  # oracle
        assert out.shape == xc.shape and torch.equal(r, y)
        assert torch.equal(out[y == 0], xc[y == 0]), "clean images must pass through unchanged"
        assert torch.allclose(p.sum(1), torch.ones(len(y)), atol=1e-5)
        _, _, r_pred = router(xc)  # predicted
    print(f"oracle routes {r.tolist()}\npredicted    {r_pred.tolist()}\nrouter self-check OK")
