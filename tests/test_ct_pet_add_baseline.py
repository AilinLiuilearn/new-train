# -*- coding: utf-8 -*-
"""Tests for the clean CT+PET addition baseline."""
import torch

from models.ct_pet_add_baseline import CTPETAddBaseline


def _model(**over):
    kw = dict(ct_pretrained_path=None, pet_pretrained_path=None)
    kw.update(over)
    return CTPETAddBaseline(**kw)


def test_forward_modes_and_missing_is_ct_identity():
    torch.manual_seed(0)
    m = _model().eval()
    ct = torch.randn(4, 1, 64, 64)
    pet = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        full = m(ct, pet, forward_mode='full')['logits']
        missing = m(ct, pet, forward_mode='missing')['logits']
        assert full.shape == (4, 1, 64, 64)
        assert missing.shape == (4, 1, 64, 64)
        # Missing (encode-then-zero + add) must equal the CT-only decode.
        ct_only = m._decode(m._encode_ct(ct), (64, 64))['logits']
        assert torch.allclose(missing, ct_only, atol=1e-5)
        state = torch.tensor([1, 1, 0, 0])
        auto = m(ct, pet, pet_available=state, forward_mode='auto')
        assert auto['num_full'] == 2 and auto['num_missing'] == 2
        assert torch.allclose(auto['logits'][:2], full[:2], atol=1e-5)
        assert torch.allclose(auto['logits'][2:], missing[2:], atol=1e-5)


def test_decoder_norm_switch():
    import torch.nn as nn
    m_bn = _model(decoder_norm='bn')
    assert any(isinstance(x, nn.BatchNorm2d) for x in m_bn.decoder.modules())
    m_g = _model(decoder_norm='group')
    assert not any(isinstance(x, nn.BatchNorm2d) for x in m_g.decoder.modules())
    assert any(isinstance(x, nn.GroupNorm) for x in m_g.decoder.modules())
    import pytest
    with pytest.raises(ValueError):
        _model(decoder_norm='layer')


def test_task_attributes_present():
    m = _model()
    for attr in ('enc_ct', 'ct_align', 'decoder'):
        assert isinstance(getattr(m, attr), torch.nn.Module)
