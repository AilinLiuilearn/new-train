# -*- coding: utf-8 -*-
"""Tests for the decoder norm switch (bn | group).

- Default (bn) keeps the baseline exactly: BatchNorm layers present, no new
  keys, strict state_dict compatibility with old checkpoints.
- group replaces every decoder BatchNorm with per-sample GroupNorm, so Full
  and Missing rows no longer couple through batch statistics.
"""
import torch

from models.baseline_blocks import UNetStyleDecoder
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.build_mdt_seg import build_mdt_seg_teacher


def _norms(decoder):
    bns = [m for m in decoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    gns = [m for m in decoder.modules() if isinstance(m, torch.nn.GroupNorm)]
    return bns, gns


def test_bn_default_has_batchnorm_only():
    dec = UNetStyleDecoder()
    bns, gns = _norms(dec)
    assert len(bns) > 0 and len(gns) == 0
    assert dec.norm_type == 'bn'
    # BN path keeps the original ConvBNAct (.block wrapper): old baseline
    # checkpoint keys like 'fuse1.0.block.0.weight' must exist unchanged.
    keys = set(dec.state_dict().keys())
    assert 'fuse1.0.block.0.weight' in keys
    assert 'proj4.block.1.weight' in keys  # BatchNorm weight inside .block
    assert not any(k.startswith('fuse1.0.0.') or k.startswith('proj4.0.') for k in keys)


def test_group_has_no_batchnorm():
    dec = UNetStyleDecoder(norm_type='group')
    bns, gns = _norms(dec)
    assert len(bns) == 0 and len(gns) > 0
    assert dec.norm_type == 'group'


def test_group_is_per_sample_independent():
    torch.manual_seed(0)
    dec = UNetStyleDecoder(norm_type='group').train()
    feats = [torch.randn(4, c, s, s) for c, s in zip((64, 128, 320, 512), (32, 16, 8, 4))]
    out_all = dec([f.clone() for f in feats], (32, 32))
    outs = []
    for i in range(4):
        outs.append(dec([[f[i:i + 1].clone() for f in feats][j] for j in range(4)], (32, 32)))
    stacked = torch.cat([o['logits'] for o in outs], dim=0)
    assert torch.allclose(out_all['logits'], stacked, atol=1e-5), \
        'GroupNorm decoder output depends on batch composition'
    # Same check must FAIL for BN in training mode (documents the coupling).
    torch.manual_seed(0)
    bn = UNetStyleDecoder(norm_type='bn').train()
    bn_all = bn([f.clone() for f in feats], (32, 32))
    bn_stacked = torch.cat(
        [bn([[f[i:i + 1].clone() for f in feats][j] for j in range(4)], (32, 32))['logits']
         for i in range(4)], dim=0)
    assert not torch.allclose(bn_all['logits'], bn_stacked, atol=1e-5), \
        'BN decoder unexpectedly batch-independent'


def test_model_default_bn_and_invalid_rejected():
    import pytest
    m = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    assert m.decoder_norm == 'bn'
    bns, gns = _norms(m.decoder)
    assert len(bns) > 0 and len(gns) == 0
    with pytest.raises(ValueError):
        DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None,
                                   decoder_norm='layer')


def test_build_from_config_without_field_defaults_bn():
    old_cfg = type('Old', (), {
        'ct_backbone': 'convnextv2_nano', 'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None, 'pet_pretrained_path': None,
        'decoder_channels': (512, 256, 128, 64), 'use_deep_supervision': False,
    })()
    assert not hasattr(old_cfg, 'decoder_norm')
    m = build_mdt_seg_teacher(old_cfg)['model']
    assert m.decoder_norm == 'bn'


def test_bn_state_dict_compatible_with_old_checkpoint():
    torch.manual_seed(0)
    old_like = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    sd = {k: v.clone() for k, v in old_like.state_dict().items()}
    torch.manual_seed(1)
    fresh = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    msg = fresh.load_state_dict(sd, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
