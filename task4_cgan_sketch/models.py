"""Style-conditioned pix2pix: U-Net generator and PatchGAN discriminator, each with its own style embedding.

Self-check: python -m task4_cgan_sketch.models
"""
import torch
from torch import nn

N_STYLES = 3


def init_weights(m):
    """pix2pix init: conv/linear weights ~ N(0, 0.02); norm scales ~ N(1, 0.02) [Isola17]."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class FiLM(nn.Module):
    """h <- (1 + gamma(e)) * h + beta(e), per channel [Perez18]. Zero-initialised, so it starts as the identity."""

    def __init__(self, emb_dim, ch):
        super().__init__()
        self.lin = nn.Linear(emb_dim, 2 * ch)

    def forward(self, h, e):
        gamma, beta = self.lin(e)[:, :, None, None].chunk(2, dim=1)
        return (1 + gamma) * h + beta


class UNetGenerator(nn.Module):
    """7 down / 7 up U-Net with full skips (allowed in Task 4). cond="film": style modulates every decoder block;
    cond="input": style embedding broadcast and concatenated to the photo."""

    def __init__(self, base_ch=64, emb_dim=16, dropout=0.5, cond="film"):
        super().__init__()
        b, self.cond = base_ch, cond
        self.emb = nn.Embedding(N_STYLES, emb_dim)
        enc = [b, 2 * b, 4 * b, 8 * b, 8 * b, 8 * b, 8 * b]  # outputs at 64, 32, 16, 8, 4, 2, 1
        cin = 3 + (emb_dim if cond == "input" else 0)
        self.downs = nn.ModuleList()
        for i, cout in enumerate(enc):
            # no norm on the first block (pix2pix) and on the innermost 1x1 block (InstanceNorm of one pixel = 0)
            norm = 0 < i < len(enc) - 1
            self.downs.append(nn.Sequential(
                nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm),
                nn.InstanceNorm2d(cout, affine=True) if norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True) if i < len(enc) - 1 else nn.ReLU(inplace=True)))
            cin = cout
        dec = [8 * b, 8 * b, 8 * b, 4 * b, 2 * b, b]  # outputs at 2, 4, 8, 16, 32, 64
        self.ups, self.films, self.drops = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, cout in enumerate(dec):
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                                          nn.InstanceNorm2d(cout, affine=True)))
            self.films.append(FiLM(emb_dim, cout) if cond == "film" else nn.Identity())
            self.drops.append(nn.Dropout(dropout) if i < 3 else nn.Identity())  # pix2pix: dropout in first 3
            cin = 2 * cout  # after concatenating the encoder skip of the same resolution
        self.head = nn.Sequential(nn.ConvTranspose2d(cin, 1, 4, 2, 1), nn.Tanh())
        self.apply(init_weights)
        for f in self.films:
            if isinstance(f, FiLM):
                nn.init.zeros_(f.lin.weight), nn.init.zeros_(f.lin.bias)

    def forward(self, photo, style):
        e = self.emb(style)
        h = photo
        if self.cond == "input":
            h = torch.cat([h, e[:, :, None, None].expand(-1, -1, *photo.shape[2:])], 1)
        skips = []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        skips = skips[-2::-1]  # encoder outputs at 2, 4, ..., 64 (the innermost 1x1 is h itself)
        for up, film, drop, skip in zip(self.ups, self.films, self.drops, skips):
            h = up(h)
            h = film(h, e) if self.cond == "film" else h
            h = torch.cat([torch.relu(drop(h)), skip], 1)
        return self.head(h)


class PatchDiscriminator(nn.Module):
    """pix2pix PatchGAN. n_layers=3 -> 70x70 receptive field, n_layers=2 -> 34x34.
    cond="concat": style embedding broadcast as extra input channels.
    cond="proj": projection discriminator [Miyato18proj], logit += <embed(style), features> per patch."""

    def __init__(self, base_ch=64, emb_dim=16, n_layers=3, cond="concat"):
        super().__init__()
        b, self.cond = base_ch, cond
        self.emb = nn.Embedding(N_STYLES, emb_dim)
        cin = 3 + 1 + (emb_dim if cond == "concat" else 0)
        layers = [nn.Conv2d(cin, b, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]  # no norm on the first layer
        ch = b
        for n in list(range(1, n_layers)) + [n_layers]:  # n_layers-1 more stride-2 blocks, then one stride-1 block
            nxt = b * min(2 ** n, 8)
            layers += [nn.Conv2d(ch, nxt, 4, 2 if n < n_layers else 1, 1, bias=False),
                       nn.InstanceNorm2d(nxt, affine=True), nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(ch, 1, 4, 1, 1)
        if cond == "proj":
            self.proj = nn.Embedding(N_STYLES, ch)
        self.apply(init_weights)

    def forward(self, photo, sketch, style):
        x = torch.cat([photo, sketch], 1)
        if self.cond == "concat":
            x = torch.cat([x, self.emb(style)[:, :, None, None].expand(-1, -1, *x.shape[2:])], 1)
        h = self.body(x)
        logit = self.out(h)
        if self.cond == "proj":
            # inner product per location, averaged over 2x2 so it lines up with the 4x4/pad-1 output conv
            term = (h * self.proj(style)[:, :, None, None]).sum(1, keepdim=True)
            logit = logit + nn.functional.avg_pool2d(term, 2, stride=1)
        return logit


def receptive_field(n_layers: int) -> int:
    """Input pixels seen by one output logit: all 4x4 convs, n_layers with stride 2, then two with stride 1."""
    r = 1
    for s in [1, 1] + [2] * n_layers:  # walk back from the output
        r = r * s + (4 - s)
    return r


if __name__ == "__main__":
    photo, sketch, style = torch.rand(2, 3, 128, 128) * 2 - 1, torch.rand(2, 1, 128, 128) * 2 - 1, torch.tensor([0, 2])
    count = lambda m: f"{sum(p.numel() for p in m.parameters()):,}"
    for cond in ("film", "input"):
        g = UNetGenerator(cond=cond).eval()
        y = g(photo, style)
        assert y.shape == (2, 1, 128, 128) and y.abs().max() <= 1, cond
        print(f"G cond={cond}: params={count(g)}, out {tuple(y.shape)}")
    g = UNetGenerator(base_ch=32, emb_dim=8)
    assert g(photo, style).shape == (2, 1, 128, 128)
    print(f"G base_ch=32 emb_dim=8: params={count(g)}")
    for n_layers in (2, 3):
        for cond in ("concat", "proj"):
            d = PatchDiscriminator(n_layers=n_layers, cond=cond)
            out = d(photo, sketch, style)
            side = 128 // 2 ** n_layers - 2
            assert out.shape == (2, 1, side, side), (n_layers, cond, out.shape)
            print(f"D n_layers={n_layers} cond={cond}: params={count(d)}, logits {tuple(out.shape[2:])}, "
                  f"receptive field {receptive_field(n_layers)}x{receptive_field(n_layers)}")
    assert receptive_field(3) == 70 and receptive_field(2) == 34
    print("self-check passed")
