import torch
from types import SimpleNamespace

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def _config():
    return SimpleNamespace(ct_backbone='convnextv2_nano', pet_backbone='mit_b1', ct_pretrained_path=None, pet_pretrained_path=None)


def test_full_and_missing_forward_shapes():
    model = DualSharedAddPETCTBaseline(_config())
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    full = model(ct, pet=pet, forward_mode='full')
    missing = model(ct, pet=None, forward_mode='missing')
    assert full.shape == missing.shape == (2, 1, 64, 64)


def test_shared_decoder_single_object():
    model = DualSharedAddPETCTBaseline(_config())
    assert hasattr(model, 'decoder') and not hasattr(model, 'decoder_full') and not hasattr(model, 'decoder_missing')


def test_missing_does_not_require_pet():
    model = DualSharedAddPETCTBaseline(_config())
    ct = torch.randn(1, 1, 64, 64)
    out = model(ct, pet=None, forward_mode='missing')
    assert out.shape[-2:] == (64, 64)
