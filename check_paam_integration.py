import os
import tempfile

import torch
import torch.nn as nn

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.dual_shared_add_paam import DualSharedAddPAAMPETCT
from models.baseline_blocks import UNetStyleDecoder


class DummyFeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


class DummyBackbone(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.feature_info = DummyFeatureInfo(channels)
        self.proj = nn.ModuleList([nn.Conv2d(3, c, 1) for c in channels])

    def forward(self, x):
        outs = []
        cur = x
        for i, proj in enumerate(self.proj):
            cur = torch.nn.functional.interpolate(cur, scale_factor=0.5, mode='bilinear', align_corners=False) if i > 0 else cur
            outs.append(proj(cur))
        return outs


class DummyConfig:
    def __init__(self, model_arch='dual_shared_add_paam', paam_k=8):
        self.model_arch = model_arch
        self.paam_k = paam_k
        self.ct_backbone = 'convnextv2_nano'
        self.pet_backbone = 'mit_b1'
        self.ct_pretrained_path = None
        self.pet_pretrained_path = None
        self.decoder_channels = [32, 16, 8, 4]
        self.use_deep_supervision = False
        self.deep_supervision = False


def patch_light_model(model, ct_channels=(64, 128, 320, 512), pet_channels=(64, 128, 320, 512)):
    model.enc_ct = DummyBackbone(ct_channels)
    model.enc_pet = DummyBackbone(pet_channels)
    model.ct_align = nn.Identity()
    model.decoder = UNetStyleDecoder(encoder_channels=pet_channels, decoder_channels=(32, 16, 8, 4), out_channels=1, use_deep_supervision=False)
    return model


def make_inputs(batch=2, size=32):
    ct = torch.randn(batch, 1, size, size)
    pet = torch.randn(batch, 1, size, size)
    mask = torch.randint(0, 2, (batch, 1, size, size)).float()
    return ct, pet, mask


def test_build():
    cfg = DummyConfig()
    model = build_mdt_seg_teacher(cfg)['model']
    assert hasattr(model, 'paam')
    assert list(model.paam.channels) == [64, 128, 320, 512]
    assert model.paam.K == 8


def test_full_forward():
    model = patch_light_model(DualSharedAddPAAMPETCT(paam_k=8))
    model.train()
    ct, pet, _ = make_inputs()
    out = model(ct, pet=pet, forward_mode='full')
    assert out['logits'].shape[:2] == (2, 1)
    assert out['paam_info']['scales'][0]['used_affine_source'] == 'current_real_pet_affine'


def test_epoch1_missing():
    model = patch_light_model(DualSharedAddPAAMPETCT(paam_k=8))
    model.paam.begin_epoch(1)
    model.train()
    ct, pet, _ = make_inputs()
    out = model(ct, pet=pet, forward_mode='missing')
    assert torch.isfinite(out['logits']).all()
    assert out['paam_info']['scales'][0]['paired_pet_used_for_current_fusion'] is False
    assert out['paam_info']['scales'][0]['used_affine_source'] == 'delayed_memory_retrieval'


def test_delayed_update():
    model = patch_light_model(DualSharedAddPAAMPETCT(paam_k=8))
    model.paam.begin_epoch(1)
    ct, pet, _ = make_inputs()
    model(ct, pet=pet, forward_mode='full')
    model(ct, pet=pet, forward_mode='missing')
    report = model.finalize_epoch_memory()
    assert len(report['scales']) == 4
    assert all(scale['memory_ready'] for scale in report['scales'])
    assert all(len(scale['slot_counts']) == 8 for scale in report['scales'])


def test_real_missing_eval():
    model = patch_light_model(DualSharedAddPAAMPETCT(paam_k=8))
    model.eval()
    ct, _, _ = make_inputs()
    out = model(ct, pet=None, forward_mode='missing')
    assert torch.isfinite(out['logits']).all()
    assert 'PASS' in out['paam_info']['leakage_guard']


def test_checkpoint():
    model = patch_light_model(DualSharedAddPAAMPETCT(paam_k=8))
    model.paam.begin_epoch(1)
    ct, pet, _ = make_inputs()
    model(ct, pet=pet, forward_mode='full')
    model.finalize_epoch_memory()
    state = model.state_dict()
    model2 = patch_light_model(DualSharedAddPAAMPETCT(paam_k=8))
    model2.load_state_dict(state)
    assert torch.equal(model.paam.memories[0].memory_ready, model2.paam.memories[0].memory_ready)
    assert torch.allclose(model.paam.memories[0].keys, model2.paam.memories[0].keys)
    assert torch.allclose(model.paam.memories[0].gamma_proto, model2.paam.memories[0].gamma_proto)
    assert torch.allclose(model.paam.memories[0].beta_proto, model2.paam.memories[0].beta_proto)


def test_baseline_unchanged():
    model = patch_light_model(DualSharedAddPETCTBaseline())
    ct, pet, _ = make_inputs()
    out1 = model(ct, pet=pet, forward_mode='full')
    out2 = model(ct, pet=pet, forward_mode='missing')
    assert out1['logits'].shape == out2['logits'].shape
    assert not hasattr(model, 'paam')


def main():
    test_build()
    test_full_forward()
    test_epoch1_missing()
    test_delayed_update()
    test_real_missing_eval()
    test_checkpoint()
    test_baseline_unchanged()
    print('ALL PAAM INTEGRATION TESTS PASSED')


if __name__ == '__main__':
    main()
