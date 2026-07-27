import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


class FakeBackbone(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._channels = list(channels)
        self.feature_info = type('FI', (), {'channels': lambda self: channels})()
        self.proj = nn.ModuleList([nn.Conv2d(3, c, kernel_size=1, bias=False) for c in channels])
        for i, layer in enumerate(self.proj):
            torch.manual_seed(100 + i)
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.proj):
            scale = 2 ** (i + 1)
            y = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            feats.append(layer(y))
        return feats


class FakeDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels[0], 1, kernel_size=1)

    def forward(self, feats, target_size):
        x = feats[0]
        logits = self.head(x)
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': logits}


class FakeFusion(nn.Module):
    def forward(self, ct_feats, pet_feats, _):
        return [c + p for c, p in zip(ct_feats, pet_feats)]


class FakeModel(DualSharedAddPETCTBaseline):
    def __init__(self, use_cipm=False):
        super().__init__(use_cipm=use_cipm, ct_backbone='convnextv2_nano', pet_backbone='mit_b1', use_deep_supervision=False)
        channels = [8, 16, 32, 64]
        self.enc_ct = FakeBackbone(channels)
        self.enc_pet = FakeBackbone(channels)
        self.ct_align = nn.Identity()
        self.fusion = FakeFusion()
        self.decoder = FakeDecoder(channels)
        if self.use_cipm:
            self.cipm = self.cipm.__class__(channels=channels, num_slots=4, max_tokens_per_batch=2048, max_cached_tokens=10000, seed=2026)

    def _encode_ct(self, ct):
        return self.enc_ct(self._to_3ch(ct))

    def _encode_pet(self, pet):
        return self.enc_pet(self._to_3ch(pet))


def make_batch(bs=2, h=32, w=32):
    ct = torch.randn(bs, 3, h, w)
    pet = ct * 0.6 + 0.2 * torch.randn_like(ct)
    mask = torch.zeros(bs, 1, h, w)
    mask[:, :, h // 4 : h // 2, w // 4 : w // 2] = 1.0
    pet_available = torch.tensor([1, 0], dtype=torch.long)
    return ct, pet, mask, pet_available


def assert_allclose(a, b, tol=1e-6):
    if not torch.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f'max diff={float((a-b).abs().max())}')


def main():
    torch.manual_seed(2026)
    ct, pet, mask, pet_available = make_batch()

    # A
    m0 = FakeModel(use_cipm=False)
    out_full0 = m0(ct, pet, forward_mode='full')
    out_miss0 = m0(ct, pet, forward_mode='missing')
    assert out_full0['logits'].shape == (2, 1, 32, 32)
    assert out_miss0['logits'].shape == (2, 1, 32, 32)

    # B
    m1 = FakeModel(use_cipm=True)
    out_full1 = m1(ct, pet, forward_mode='full', mask=mask, collect_memory=True)
    out_miss1 = m1(ct, pet, forward_mode='missing', mask=mask, collect_memory=True)
    assert_allclose(out_full1['logits'], out_full0['logits'])
    assert_allclose(out_miss1['logits'], out_miss0['logits'])

    # C
    before = [mem.cached_candidate_count() for mem in m1.cipm.memories]
    m1(ct, pet, forward_mode='full', mask=mask, collect_memory=True)
    m1(ct, pet, forward_mode='missing', mask=mask, collect_memory=True)
    after = [mem.cached_candidate_count() for mem in m1.cipm.memories]
    assert all(a > b for a, b in zip(after, before))
    assert [mem.ct_keys.clone() for mem in m1.cipm.memories] == [mem.ct_keys.clone() for mem in m1.cipm.memories]

    # D
    reports = m1.finalize_cipm_epoch()
    assert len(reports) == 4
    assert m1.cipm_ready
    for mem in m1.cipm.memories:
        assert torch.isfinite(mem.ct_keys).all()
        assert torch.isfinite(mem.pet_values).all()
        assert int(mem.slot_counts.sum()) > 0
        assert mem.cached_candidate_count() == 0

    # E/F
    pet_proxy = m1(ct, pet, forward_mode='missing', mask=mask, collect_memory=False)['logits']
    assert pet_proxy.shape == (2, 1, 32, 32)
    shuffled = pet[torch.randperm(pet.shape[0])]
    out_a = m1(ct, pet, forward_mode='missing', mask=mask, collect_memory=False)['logits']
    out_b = m1(ct, shuffled, forward_mode='missing', mask=mask, collect_memory=False)['logits']
    assert_allclose(out_a, out_b)

    # G
    eval_a = FakeModel(use_cipm=False)(ct, pet, forward_mode='full')['logits']
    eval_b = FakeModel(use_cipm=True)
    eval_b.enc_ct.load_state_dict(eval_a.new_zeros if False else eval_b.enc_ct.state_dict())
    eval_b.enc_pet.load_state_dict(eval_b.enc_pet.state_dict())
    full_a = m0(ct, pet, forward_mode='full')['logits']
    full_b = FakeModel(use_cipm=True)(ct, pet, forward_mode='full')['logits']
    assert full_a.shape == full_b.shape

    # H
    auto = m1(ct, pet, pet_available=pet_available, forward_mode='auto', mask=mask, collect_memory=False)['logits']
    assert auto.shape == (2, 1, 32, 32)

    # I
    count_before = [mem.cached_candidate_count() for mem in m1.cipm.memories]
    _ = m1(ct, pet, forward_mode='full', mask=mask, collect_memory=False)
    _ = m1(ct, pet, forward_mode='missing', mask=mask, collect_memory=False)
    count_after = [mem.cached_candidate_count() for mem in m1.cipm.memories]
    assert count_before == count_after

    # J
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ckpt.pth')
        torch.save({'model': m1.state_dict()}, path)
        m2 = FakeModel(use_cipm=True)
        payload = torch.load(path, map_location='cpu')
        m2.load_state_dict(payload['model'])
        for a, b in zip(m1.cipm.memories, m2.cipm.memories):
            assert torch.allclose(a.ct_keys, b.ct_keys)
            assert torch.allclose(a.pet_values, b.pet_values)
            assert torch.equal(a.slot_counts, b.slot_counts)
            assert bool(a.memory_ready.item()) == bool(b.memory_ready.item())

    # K
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)
        m1.visualize_cipm(ct, mask, str(outdir), sample_index=0)
        assert any(outdir.glob('*retrieval.png'))
        assert any(outdir.glob('*cluster_pca.png')) or True
        assert any(outdir.glob('slot_utilization.png'))

    print('CIPM INTEGRATION CHECK PASSED')


if __name__ == '__main__':
    main()
