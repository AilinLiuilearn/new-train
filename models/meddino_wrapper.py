import os

import torch
import torch.nn as nn

from models.build_mdt_seg import ConvBNAct, SimpleFeatureInfo


class _PlaceholderMedDINOv3(nn.Module):
    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super().__init__()
        self.feature_info = SimpleFeatureInfo(channels)
        c1, c2, c3, c4 = channels
        self.stem = ConvBNAct(in_channels, c1, kernel_size=3, stride=2)
        self.stage1 = ConvBNAct(c1, c1, kernel_size=3, stride=2)
        self.stage2 = ConvBNAct(c1, c2, kernel_size=3, stride=2)
        self.stage3 = ConvBNAct(c2, c3, kernel_size=3, stride=2)
        self.stage4 = ConvBNAct(c3, c4, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return [p1, p2, p3, p4]


class FrozenMedDINOv3Encoder(nn.Module):
    """Frozen MedDINOv3 anatomical prior wrapper.

    The class keeps a stable [P1, P2, P3, P4] interface. When a real MedDINOv3
    checkpoint/model is not available, it falls back to a frozen lightweight
    placeholder prior encoder so the rest of the pipeline can be developed and
    tested without changing downstream code.
    """

    def __init__(self, ckpt_path=None, use_placeholder_if_missing=True, out_channels=(64, 128, 256, 512)):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.use_placeholder_if_missing = bool(use_placeholder_if_missing)
        self.out_channels = list(out_channels)
        self.is_placeholder = True

        if ckpt_path and os.path.exists(str(ckpt_path)):
            # Real MedDINOv3 loading hook. Keep this location stable for future integration.
            print(f'[+] MedDINOv3 checkpoint path provided: {ckpt_path}. Placeholder wrapper is used until real model integration is added.')
        elif ckpt_path and not os.path.exists(str(ckpt_path)) and not self.use_placeholder_if_missing:
            raise FileNotFoundError(f'MedDINOv3 checkpoint not found: {ckpt_path}')

        self.encoder = _PlaceholderMedDINOv3(in_channels=3, channels=out_channels)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @staticmethod
    def _to_3ch(ct):
        if ct.shape[1] == 1:
            return ct.repeat(1, 3, 1, 1)
        return ct

    def forward(self, ct):
        with torch.no_grad():
            ct = self._to_3ch(ct)
            return self.encoder(ct)
