"""Small deterministic backbone contract for integration tests."""

from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(3, c, 1) for c in (8, 16, 24, 32))
        self.feature_info = SimpleNamespace(channels=lambda: [8, 16, 24, 32])
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return [
            layer(F.adaptive_avg_pool2d(x, (16 // 2**i,) * 2))
            for i, layer in enumerate(self.layers)
        ]
