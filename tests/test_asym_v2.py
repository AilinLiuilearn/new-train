# -*- coding: utf-8 -*-
"""Tests for the v2 asymmetric fusion (residual + competition + text mask)."""
import torch

from models.full_petct_asymmetric_fusion_v2 import FullPETCTAsymmetricFusionV2
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def _model(**over):
    kw = dict(channels=(16, 24, 32, 48), pet_dims=(16, 16, 24, 32), heads=4,
              grid_cap=16, text_dim=32, use_text=True,
              text_embeddings=torch.randn(2, 32))
    kw.update(over)
    return FullPETCTAsymmetricFusionV2(**kw)


def _feats(channels=(16, 24, 32, 48), sizes=(32, 16, 8, 4), n=2):
    return [torch.randn(n, c, s, s) for c, s in zip(channels, sizes)]


def test_v2_fused_is_ct_plus_delta():
    m = _model().eval()
    cf, pf = _feats(), _feats()
    with torch.no_grad():
        fused, deltas = m.forward_with_delta(cf, pf, state='full')
    assert len(fused) == 4 and len(deltas) == 4
    for f, d, c in zip(fused, deltas, cf):
        assert f.shape == c.shape and d.shape == c.shape
        assert torch.allclose(f, c + d, atol=1e-5)


def test_v2_delta_is_bounded_without_pet_signal():
    # Zero PET contribution path: gates still compete, residual keeps anchor.
    m = _model().eval()
    cf = _feats()
    pf = [torch.zeros_like(p) for p in _feats()]
    with torch.no_grad():
        fused, deltas = m.forward_with_delta(cf, pf, state='full')
    for f, c in zip(fused, cf):
        assert torch.isfinite(f).all()
        assert (f - c).pow(2).mean().sqrt() < c.pow(2).mean().sqrt() + 1.0


def test_v2_text_mask_is_image_dependent():
    torch.manual_seed(0)
    m = _model().eval()
    mod = m.scales[0].ct_mod
    t = torch.randn(32)
    x1 = torch.randn(2, 16, 16, 16)
    x2 = torch.randn(2, 16, 16, 16) + 3.0
    with torch.no_grad():
        r1 = (mod(x1, t) / (x1 + 1e-6)).mean((2, 3))
        r2 = (mod(x2, t) / (x2 + 1e-6)).mean((2, 3))
    assert not torch.allclose(r1, r2, atol=1e-4)


def test_v2_rejects_missing_and_bad_shapes():
    m = _model().eval()
    cf, pf = _feats(), _feats()
    try:
        m(cf, pf, state='missing')
    except ValueError:
        pass
    else:
        raise AssertionError('missing state must be rejected')
    bad = [p[:, :, 1:, :] for p in pf]
    try:
        m(cf, bad, state='full')
    except ValueError:
        pass
    else:
        raise AssertionError('shape mismatch must be rejected')


def test_model_wiring_defaults_v1_and_accepts_v2():
    m = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    assert m.fusion_version == 'v1'
    assert type(m.fusion).__name__ == 'AddFusion'
    m2 = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None,
        asym_fusion_enabled=True, asym_use_text=False, fusion_version='v2')
    assert type(m2.fusion).__name__ == 'FullPETCTAsymmetricFusionV2'
    import pytest
    with pytest.raises(ValueError):
        DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None,
                                   fusion_version='v3')
