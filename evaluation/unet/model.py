"""A small CPU-trainable U-Net: grayscale page -> geometry-ink probability."""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Three down/up levels; ``base`` channels at full resolution."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(1, c[0]), _block(c[0], c[1]), _block(c[1], c[2])])
        self.mid = _block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.head = nn.Conv2d(c[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
