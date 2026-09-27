# -*- coding: utf-8 -*-
"""Tests for the pgf (paired global fusion) wiring with CT residual."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline


def _model(**over):
    kw = dict(ct_pretrained_path=None, pet_pretrained_path=None,
              asym_fusion_enabled=True, asym_use_text=False, fusion_version='pgf')
    kw.update(over)
    return DualSharedAddPETCTBaseline(**kw)


def test_pgf_selected_and_residual_default_on():
    m = _model()
    assert type(m.fusion).__name__ == 'PETCTPairedGlobalFusion'
    assert m.pgf_residual is True


def test_pgf_forward_modes_and_missing_identity():
    torch.manual_seed(0)
    m = _model().eval()
    ct = torch.randn(4, 1, 64, 64)
    pet = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        full = m(ct, pet, forward_mode='full')['logits']
        missing = m(ct, pet, forward_mode='missing')['logits']
        assert full.shape == (4, 1, 64, 64)
        assert torch.isfinite(full).all()
        ct_only = m._decode(m._encode_ct(ct), (64, 64))['logits']
        assert torch.allclose(missing, ct_only, atol=1e-5)
        auto = m(ct, pet, pet_available=torch.tensor([1, 1, 0, 0]), forward_mode='auto')
        assert auto['num_full'] == 2 and auto['num_missing'] == 2
        assert torch.allclose(auto['logits'][:2], full[:2], atol=1e-5)
        assert torch.allclose(auto['logits'][2:], missing[2:], atol=1e-5)


def test_pgf_residual_off_runs_raw():
    torch.manual_seed(0)
    m = _model(pgf_residual=False).eval()
    ct = torch.randn(2, 1, 64, 64)
    pet = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = m(ct, pet, forward_mode='full')['logits']
    assert out.shape == (2, 1, 64, 64) and torch.isfinite(out).all()


def test_pgf_rejects_bad_version():
    import pytest
    with pytest.raises(ValueError):
        _model(fusion_version='pgx')
