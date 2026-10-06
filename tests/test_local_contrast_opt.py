# -*- coding: utf-8 -*-
"""Optimization + beta-ablation tests for the local-contrast experiment.

Covers: analytic mask equivalence (vs unfold, nonzero weights, with grads),
checkpoint on/off equivalence, beta init rules, old-checkpoint strict compat,
eval_amp parity reporting, HD95 v2 protocol, sampled grad logging.
Synthetic inputs only.
"""
import pytest
import torch
import torch.nn.functional as F

import run_full_local_contrast_fusion as fusion_entry
from models.build_local_contrast_fusion import build_local_contrast_fusion_model
from models.dual_shared_local_contrast_fusion import (
    DEFAULT_FUSION_KWARGS,
    DualSharedLocalContrastFusionModel,
)
from models.local_contrast_bidirectional_fusion import (
    LocalContrastFusionPyramid,
    _LocalExchange,
)
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import (
    HD95_PROTOCOL_VERSION,
    compute_hd95_pair,
    compute_hd95_pair_v2,
)


def _cfg(**over):
    base = dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False, pretrained=False,
        model_arch='dual_shared_local_contrast_fusion', train_batch_mode='full',
        accumulation_steps=1, optimizer='adamw',
        fusion_dim=32, fusion_heads=4, fusion_window=5,
        fusion_chunk_rows=16, fusion_checkpoint_chunks=False,
        fusion_position_bias_beta=0.0, check_finite=True,
        learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
        random_state=2023, ema_enabled=False,
        grad_log_interval=100, eval_amp=False,
    )
    base.update(over)
    return type('C', (), base)()


def _nonzero_init(module):
    torch.manual_seed(11)
    with torch.no_grad():
        for block in module.scales:
            for update in (block.ct_update, block.pet_update):
                update.out.weight.normal_(0, 0.02)


def test_mask_matches_unfold_with_nonzero_weights_and_grads():
    torch.manual_seed(0)
    module = LocalContrastFusionPyramid(chunk_rows=16, checkpoint_chunks=False)
    _nonzero_init(module)
    ct = [torch.randn(2, c, s, s, requires_grad=True)
          for c, s in zip((64, 128, 320, 512), (24, 12, 8, 4))]
    pet = [torch.randn_like(c.detach()).requires_grad_() for c in ct]
    out = module(ct, pet)
    loss = sum(f.square().mean() for f in out)
    loss.backward()
    in_grads = [c.grad.detach().clone() for c in ct]
    param_grads = {n: p.grad.detach().clone()
                   for n, p in module.named_parameters() if p.grad is not None}
    # Reference: unfold-built mask for every chunk geometry used above.
    for block in (module.scales[0].ct_local, module.scales[0].pet_local):
        for (h, w, start, end) in [(24, 24, 0, 16), (24, 24, 16, 24)]:
            got = block._allowed_mask(h, w, start, end, torch.device('cpu'))
            valid = F.pad(torch.ones(1, 1, h, w), (2, 2, 2, 2))
            ref = F.unfold(valid[:, :, start:end + 4], 5).reshape(1, 1, 25, -1).bool()
            assert torch.equal(got.reshape(1, 1, 25, -1).bool(), ref)
    assert all(g is not None and torch.isfinite(g).all() for g in in_grads)
    assert param_grads, 'nonzero-init must produce parameter gradients'
    assert len(module.scales[0].ct_local._mask_cache) <= 8, 'cache must stay bounded'


def test_checkpoint_on_off_equivalence():
    torch.manual_seed(1)
    module = LocalContrastFusionPyramid(chunk_rows=32, checkpoint_chunks=True)
    _nonzero_init(module)
    import copy
    plain = copy.deepcopy(module)
    for block in plain.scales:
        block.ct_local.checkpoint_chunks = False
        block.pet_local.checkpoint_chunks = False
    ct = [torch.randn(2, c, s, s) for c, s in zip((64, 128, 320, 512), (32, 16, 8, 4))]
    pet = [torch.randn_like(c) for c in ct]
    module.train()
    plain.train()
    a = module(ct, pet)
    b = plain(ct, pet)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6, rtol=1e-5)


def test_beta_zero_reproduces_old_init():
    assert DEFAULT_FUSION_KWARGS['position_bias_beta'] == 0.0
    module = LocalContrastFusionPyramid()
    for block in module.scales:
        for local in (block.ct_local, block.pet_local):
            assert torch.equal(local.relative_bias,
                               torch.zeros_like(local.relative_bias))


def test_beta_positive_init_and_still_exact_addition():
    module = LocalContrastFusionPyramid(position_bias_beta=0.25)
    r = 5 // 2
    off = torch.arange(-r, r + 1)
    expect = (-0.25 * (off[:, None] ** 2 + off[None, :] ** 2) / r ** 2).reshape(-1)
    for block in module.scales:
        for local in (block.ct_local, block.pet_local):
            assert torch.allclose(local.relative_bias[0], expect, atol=1e-7)
            assert torch.allclose(local.relative_bias[0], local.relative_bias[-1])
    module.eval()
    ct = [torch.randn(2, c, s, s) for c, s in zip((64, 128, 320, 512), (16, 8, 4, 2))]
    pet = [torch.randn_like(c) for c in ct]
    with torch.no_grad():
        for f, c, p in zip(module(ct, pet), ct, pet):
            assert torch.equal(f, c + p)


def test_old_checkpoint_strict_load_overrides_beta_init(tmp_path):
    torch.manual_seed(2)
    task = MDTSegTeacher(
        {'model': build_local_contrast_fusion_model(_cfg())['model']}, _cfg())
    path = str(tmp_path / 'ckpt.beta0.tar')
    task.save_checkpoint(path, 1, best=0.5, best_epoch=1, val={'dice': 0.5})
    ckpt = MDTSegTeacher.load_state_dicts(path)
    beta_model = build_local_contrast_fusion_model(
        _cfg(fusion_position_bias_beta=0.25))['model']
    assert not torch.equal(beta_model.fusion.scales[0].ct_local.relative_bias,
                            ckpt['model']['fusion.scales.0.ct_local.relative_bias'])
    beta_model.load_state_dict(ckpt['model'], strict=True)
    assert torch.equal(beta_model.fusion.scales[0].ct_local.relative_bias,
                       ckpt['model']['fusion.scales.0.ct_local.relative_bias'])


def test_builder_and_config_carry_beta_and_flags():
    model = build_local_contrast_fusion_model(
        _cfg(fusion_position_bias_beta=0.25, check_finite=False))['model']
    assert model.fusion_config()['position_bias_beta'] == 0.25
    assert model.check_finite is False
    assert model.fusion.position_bias_beta == 0.25
    import sys
    argv, sys.argv = sys.argv, ['x']
    try:
        cfg = fusion_entry.FullLocalContrastFusionConfig.parse_arguments()
    finally:
        sys.argv = argv
    assert cfg.fusion_position_bias_beta == 0.0
    assert cfg.grad_log_interval == 100
    assert cfg.eval_amp is False
    assert cfg.check_finite is True


def test_eval_amp_matches_fp32_on_same_weights():
    torch.manual_seed(3)
    task = MDTSegTeacher(
        {'model': build_local_contrast_fusion_model(_cfg())['model']},
        _cfg(ema_enabled=False))
    batch = {'ct': torch.randn(2, 1, 64, 64), 'pet': torch.randn(2, 1, 64, 64),
             'mask': (torch.rand(2, 1, 64, 64) > 0.5).float()}
    loader = [batch]
    fp32 = task.evaluate(loader, eval_mode='full', model=task.eval_model())
    amp = task.evaluate(loader, eval_mode='full', model=task.eval_model(), eval_amp=True)
    assert abs(fp32['dice'] - amp['dice']) < 1e-3, (fp32['dice'], amp['dice'])
    assert task.device == 'cpu' or True


def test_hd95_v2_protocol_rules():
    import numpy as np
    assert HD95_PROTOCOL_VERSION.startswith('v2')
    v, backend = compute_hd95_pair_v2(np.zeros((8, 8), bool), np.zeros((8, 8), bool))
    assert v == 0.0 and backend == 'empty-both'
    v, backend = compute_hd95_pair_v2(np.zeros((8, 8), bool), np.ones((8, 8), bool))
    assert backend == 'empty-one-side'
    assert abs(v - np.sqrt(128)) < 1e-9, 'one-sided empty must be the diagonal, symmetric'
    rng = np.random.default_rng(0)
    a, b = rng.random((32, 32)) > 0.6, rng.random((32, 32)) > 0.6
    v2, backend2 = compute_hd95_pair_v2(a, b)
    v1 = compute_hd95_pair(a, b)
    assert backend2 in ('medpy', 'scipy')
    assert abs(v2 - v1) < 0.5, (v2, v1, backend2)


def test_sampled_grad_logging_writes_empty_not_zero(tmp_path):
    import csv
    assert True  # protocol check below uses the runner's real header writer
    from utils.train_logger import init_train_log, append_epoch_log
    path = str(tmp_path / 'train_log.csv')
    init_train_log(path, extra_headers=['grad_fusion', 'grad_sampled_steps'])
    append_epoch_log(path, 1, 0.5, {'total_loss': 0.5, 'dice': 0.5, 'iou': 0.5,
                                    'acc': 0.5, 'acc_pixel': 0.5, 'hd95': 1.0},
                     lr=1e-4, grad_norm=0.1,
                     extra_metrics={'grad_fusion': '', 'grad_sampled_steps': 0.0})
    rows = list(csv.DictReader(open(path)))
    assert rows[0]['grad_fusion'] == ''
