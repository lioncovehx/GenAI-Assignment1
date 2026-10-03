"""Convolutional autoencoder for universal restoration (128 -> 8x8 bottleneck -> 128)."""
import torch
from torch import nn


def block(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, (k - 1) // 2), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def up_block(cin, cout, up):
    if up == "deconv":  # learnable, may checkerboard [Odena16]
        return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), block(cin, cout))  # no checkerboard


class ConvAE(nn.Module):
    """latent_ch: C of the 8x8xC spatial z. dense > 0: flat z of size `dense` instead.
    skip: "none" | "limited" / "limited16" (32x32 / 16x16 encoder features -> 1x1 conv -> 8 ch) | "full" (U-Net reference only)."""

    def __init__(self, base_ch=32, latent_ch=32, dropout=0.0, skip="none", up="resize", dense=0):
        super().__init__()
        b = base_ch
        enc_ch = [b, b, 2 * b, 4 * b, 8 * b]  # at 128, 64, 32, 16, 8
        self.stem = block(3, b)
        self.downs = nn.ModuleList(block(enc_ch[i], enc_ch[i + 1], 4, 2) for i in range(4))
        self.dense = dense
        if dense:
            self.to_z = nn.Sequential(nn.Flatten(), nn.Linear(8 * b * 64, dense), nn.Dropout(dropout))
            self.from_z = nn.Sequential(nn.Linear(dense, 8 * b * 64), nn.Unflatten(1, (8 * b, 8, 8)), nn.ReLU(inplace=True))
        else:
            self.to_z = nn.Sequential(nn.Conv2d(8 * b, latent_ch, 1), nn.Dropout2d(dropout))
            self.from_z = block(latent_ch, 8 * b, 1)

        # skips: resolution -> channels added to the decoder at that resolution
        res = [128, 64, 32, 16]
        if skip == "full":
            self.skip_ch = {r: enc_ch[i] for i, r in enumerate(res)}
            self.skip_proj = nn.ModuleDict({str(r): nn.Identity() for r in res})
        elif skip.startswith("limited"):  # "limited" = at 32x32, "limited16" = at 16x16
            r = int(skip[7:] or 32)
            self.skip_ch = {r: 8}
            self.skip_proj = nn.ModuleDict({str(r): nn.Conv2d(enc_ch[res.index(r)], 8, 1)})
        else:
            self.skip_ch, self.skip_proj = {}, nn.ModuleDict()

        dec_ch = [8 * b, 4 * b, 2 * b, b, b]  # at 8, 16, 32, 64, 128
        self.ups = nn.ModuleList(
            up_block(dec_ch[i] + self.skip_ch.get(8 << i, 0), dec_ch[i + 1], up) for i in range(4))
        self.head = nn.Sequential(nn.Conv2d(b + self.skip_ch.get(128, 0), 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        feats = {128: self.stem(x)}
        h = feats[128]
        for d in self.downs:
            h = d(h)
            feats[h.shape[-1]] = h
        h = self.from_z(self.to_z(h))
        for i, u in enumerate(self.ups):
            h = u(self._cat(h, feats, 8 << i))
        return self.head(self._cat(h, feats, 128))

    def _cat(self, h, feats, r):
        if r not in self.skip_ch:
            return h
        return torch.cat([h, self.skip_proj[str(r)](feats[r])], 1)

    def z_size(self):
        return self.dense or self.to_z[0].out_channels * 64


if __name__ == "__main__":
    for kw in ({}, {"skip": "limited"}, {"skip": "limited16"}, {"skip": "full"}, {"up": "deconv"}, {"dense": 512}):
        m = ConvAE(**kw)
        assert m(torch.rand(2, 3, 128, 128)).shape == (2, 3, 128, 128), kw
        print(kw, f"params={sum(p.numel() for p in m.parameters()):,}", f"|z|/49152={m.z_size() / 49152:.3f}")
