"""Self-check for the shared pipeline. Run: python -m common.test_common"""
import collections

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import corruptions as C
from common.data import get_images, split_indices
from common.datasets import CleanDataset, ManifestDataset, RuntimeCorruptionDataset, balanced_collate


def main():
    s = split_indices()
    assert (len(s["train"]), len(s["val"])) == (2944, 736), (len(s["train"]), len(s["val"]))
    assert not set(s["train"]) & set(s["val"])
    assert get_images("test").shape == (3669, 3, 128, 128)

    # Manifest replay is deterministic, and covers what the PDF asks for
    val, test = ManifestDataset("val"), ManifestDataset("test")
    assert len(val) == 736 and len(test) == 3669 * 10
    for ds in (val, test):
        for j in (0, 5, 17):
            assert torch.equal(ds[j][0], ds[j][0])

    x = torch.rand(3, 128, 128)
    rng = np.random.default_rng(1)
    for p in (0.03, 0.08, 0.15):  # S&P: hit fraction ~ p, values only 0/1, all channels equal
        y = C.apply(x, C.make_params(1, rng, {"p": p}))
        hit = (y != x).any(0)
        assert abs(hit.float().mean() - p) < 0.01, p
        assert set(y[:, hit].unique().tolist()) <= {0.0, 1.0} and (y[0, hit] == y[1, hit]).all()
    for lv in C.TEST_LEVELS[3]:  # occlusion coverage within 1% of target, union of non-overlapping rects
        prm = C.make_params(3, rng, lv)
        y = C.apply(torch.ones(3, 128, 128), prm)
        assert abs((y[0] == 0).float().mean() - lv["coverage"]) < 0.01 and len(prm["rects"]) == lv["n"]
    for _ in range(500):  # training occlusion stays in 10-35%
        prm = C.make_params(3, rng)
        assert 0.09 < prm["coverage"] < 0.36 and 1 <= prm["n"] <= 3
    tiny = C.apply(x, {"label": 2, "k": 3, "sigma": 0.5})
    strong = C.apply(x, {"label": 2, "k": 7, "sigma": 2.5})
    assert (tiny - x).abs().mean() < (strong - x).abs().mean()

    # Runtime sampler: uniform labels, different corruption each load
    torch.manual_seed(0)
    ds = RuntimeCorruptionDataset("train")
    counts = collections.Counter(ds[i % 50][2] for i in range(4000))
    assert all(abs(counts[k] / 4000 - 0.25) < 0.02 for k in range(4)), counts
    loads = [ds[0][0] for _ in range(20)]
    assert sum(not torch.equal(loads[0], t) for t in loads) > 5  # same image, new corruption most loads

    # Balanced batches: exactly B/4 each
    dl = DataLoader(CleanDataset("train"), batch_size=32, shuffle=True, collate_fn=balanced_collate)
    xt, xc, y = next(iter(dl))
    assert xt.shape == xc.shape == (32, 3, 128, 128) and torch.bincount(y).tolist() == [8, 8, 8, 8]
    assert torch.equal(xt[y == 0], xc[y == 0])
    print("common: all checks passed")


if __name__ == "__main__":
    main()
