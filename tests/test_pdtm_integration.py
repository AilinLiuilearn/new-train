import copy
import json
import os
from pathlib import Path

import torch

from models.baseline_blocks import UNetStyleDecoder
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.dual_shared_add_pdtm import DualSharedAddPDTM
from models.pdtm_runtime import PDTMRuntime


class _TinyBackbone(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._channels = list(channels)
        self.feature_info = type('FI', (), {'channels': lambda self_: list(channels)})()

    def forward(self, x):
        b, _, h, w = x.shape
        feats = []
        for i, c in enumerate(self._channels):
            scale = 2 ** (i + 1)
            hh = max(1, h // scale)
            ww = max(1, w // scale)
            feats.append(torch.randn(b, c, hh, ww, device=x.device, dtype=x.dtype) * 0.1 + (i + 1))
        return feats


class _TinyBaseline(DualSharedAddPETCTBaseline):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.use_deep_supervision = False
        self.enc_ct = _TinyBackbone([8, 16, 32, 64])
        self.enc_pet = _TinyBackbone([8, 16, 32, 64])
        self.ct_align = torch.nn.Identity()
        self.fusion = torch.nn.Identity()
        self.decoder = UNetStyleDecoder((8, 16, 32, 64), (64, 32, 16, 8), out_channels=1, use_deep_supervision=False)

    def _to_3ch(self, x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        return self.enc_ct(ct)

    def _encode_pet(self, pet):
        return self.enc_pet(pet)

    def _forward_full(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused = [a + b for a, b in zip(ct_feats, pet_feats)]
        return self._decode(fused, target_size)

    def _forward_missing(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused = [a + torch.zeros_like(b) for a, b in zip(ct_feats, pet_feats)]
        return self._decode(fused, target_size)

    def _forward_auto(self, ct, pet, pet_available, target_size):
        return self._forward_full(ct, pet, target_size)



def test_decoder_intermediates():
    dec = UNetStyleDecoder((8, 16, 32, 64), (64, 32, 16, 8), out_channels=1, use_deep_supervision=False)
    feats = [torch.randn(2, 8, 32, 32), torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4)]
    out = dec(feats, (32, 32), return_intermediates=True)
    assert 'decoder_feature' in out and 'native_logits' in out
    assert out['logits'].shape == (2, 1, 32, 32)


def test_pdtm_runtime_empty_and_ready():
    pdtm = PDTMRuntime(channels=8, slots=2, eps=1e-4)
    feat = torch.randn(2, 8, 8, 8, requires_grad=True)
    out, info = pdtm(feat)
    assert torch.allclose(out, feat)
    assert info['pdtm_memory_ready'] is False
    means = torch.zeros(2, 8)
    cov = torch.eye(8).unsqueeze(0).repeat(2, 1, 1)
    delta = torch.zeros(2, 8)
    op = torch.eye(8).unsqueeze(0).repeat(2, 1, 1)
    pdtm.load_memory(means, cov, delta, op, torch.zeros(2), torch.ones(2, dtype=torch.long))
    out2, info2 = pdtm(feat)
    assert out2.shape == feat.shape
    assert info2['pdtm_memory_ready'] is True
    assert torch.isfinite(out2).all()


def test_memory_build_and_checkpoint(tmp_path):
    model = DualSharedAddPDTM(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(64, 32, 16, 8),
        pdtm_slots=2,
        pdtm_eps=1e-4,
        pdtm_max_pairs=4,
    )
    ct = torch.randn(2, 3, 32, 32)
    pet = torch.randn(2, 3, 32, 32)
    model.collect_pdtm_pairs(ct, pet, case_ids=['a', 'b'])
    status = model.finalize_pdtm_memory()
    assert status['valid_slots'] >= 1
    diag = model.pdtm_diagnostics()
    assert 'valid_slots' in diag
    path = tmp_path / 'ckpt.pt'
    torch.save({'model': model.state_dict()}, path)
    loaded = DualSharedAddPDTM(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(64, 32, 16, 8),
        pdtm_slots=2,
        pdtm_eps=1e-4,
        pdtm_max_pairs=4,
    )
    payload = torch.load(path, map_location='cpu')
    loaded.load_state_dict(payload['model'], strict=False)
    assert int(loaded.pdtm.valid_slots.item()) == int(model.pdtm.valid_slots.item())


def test_runtime_json(tmp_path):
    pdtm = PDTMRuntime(channels=4, slots=1, eps=1e-4)
    pdtm.export_json(tmp_path, 'x')
    assert (tmp_path / 'x.json').exists()
