#!/usr/bin/env python
"""Smoke tests for frozen Stage-1 + S4 TaskMoE Stage-2 integration."""
from __future__ import annotations

import argparse
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from run_mdt_seg import _load_stage1_for_taskmoe, _sync_cppi_config_from_stage1
from tasks.mdt_seg import MDTSegTeacher

DEFAULT_STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained/ckpt.best_joint.pth.tar'
)


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 8e-5,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'ct_backbone': 'convnextv2_nano',
        'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None,
        'pet_pretrained_path': None,
        'no_encoder_pretrained': True,
        'decoder_channels': (512, 256, 128, 64),
        'use_deep_supervision': False,
        'deep_supervision': False,
        'cppi_num_clusters': 6,
        'cppi_build_stage': 4,
        'stage1_checkpoint': DEFAULT_STAGE1_CKPT,
        'checkpoint_dir': '/tmp/taskmoe_stage2_smoke',
    }
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _build_stage2(stage1_ckpt: str):
    cfg = _make_cfg(stage1_checkpoint=stage1_ckpt)
    _sync_cppi_config_from_stage1(cfg, stage1_ckpt)
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.no_encoder_pretrained = True
    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    _load_stage1_for_taskmoe(model, stage1_ckpt)
    model.enable_stage2_moe_only()
    task = MDTSegTeacher(networks, cfg)
    return model, task, cfg


def _fake_batch(device, size=288, batch=1):
    return {
        'ct': torch.randn(batch, 1, size, size, device=device),
        'pet': torch.randn(batch, 1, size, size, device=device),
        'mask': torch.zeros(batch, 1, size, size, device=device),
    }


def run_smoke(stage1_ckpt: str):
    assert os.path.isfile(stage1_ckpt), f'missing Stage-1 checkpoint: {stage1_ckpt}'
    model, task, cfg = _build_stage2(stage1_ckpt)
    device = task.device
    model.to(device)

    # 1) load: non-taskmoe missing already rejected inside _load_stage1_for_taskmoe
    print('[SMOKE] 1 load_ok')

    # 2) Stage1 frozen
    stage1_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith('taskmoe_s4.')
    ]
    assert not stage1_trainable, stage1_trainable
    print('[SMOKE] 2 stage1_frozen')

    # 3) TaskMoE trainable
    moe_frozen = [
        n for n, p in model.named_parameters()
        if n.startswith('taskmoe_s4.') and not p.requires_grad
    ]
    assert not moe_frozen, moe_frozen
    print('[SMOKE] 3 taskmoe_trainable')

    # 4) CPPI bank ready
    assert bool(model.prototype_memory.bank_ready)
    print('[SMOKE] 4 bank_ready')

    # 5) collect skipped
    called = {'collect': False}
    orig_collect = model.prototype_memory.collect

    def _collect_guard(*args, **kwargs):
        called['collect'] = True
        return orig_collect(*args, **kwargs)

    model.prototype_memory.collect = _collect_guard
    batch = _fake_batch(device)
    model.train()
    _ = model(batch['ct'], pet=batch['pet'], mask=batch['mask'], forward_mode='full')
    assert called['collect'] is False
    model.prototype_memory.collect = orig_collect
    print('[SMOKE] 5 collect_skipped')

    # 6) Missing path does not call PET encoder
    pet_calls = {'n': 0}
    orig_encode_pet = model._encode_pet

    def _encode_pet_guard(*args, **kwargs):
        pet_calls['n'] += 1
        return orig_encode_pet(*args, **kwargs)

    model._encode_pet = _encode_pet_guard
    _ = model(batch['ct'], pet=batch['pet'], mask=batch['mask'], forward_mode='missing')
    assert pet_calls['n'] == 0, pet_calls
    model._encode_pet = orig_encode_pet
    print('[SMOKE] 6 missing_skips_pet_encoder')

    # 7/8/9) S1-S3 unchanged; S4 shape; beta=0 identity
    model.eval()
    with torch.no_grad():
        ct_feats = model._encode_ct(batch['ct'])
        pet_feats = model._encode_pet(batch['pet'])
        pet_cal = model.pet_calibration(ct_feats, pet_feats, None, reference_valid=False)
        fused = model.fusion(ct_feats, pet_cal, mode='full')
        fused_list = [f.detach().clone() for f in fused]
        assert fused_list[3].shape[1] == 512
        assert fused_list[3].shape[-2:] == (batch['ct'].shape[-2] // 32, batch['ct'].shape[-1] // 32) or True
        # For 64 input, S4 is typically 2x2 with mit_b1; just check channel and spatial match
        f4_in = fused_list[3]
        assert f4_in.shape == (batch['ct'].shape[0], 512, batch['ct'].shape[-2] // 32, batch['ct'].shape[-1] // 32), f4_in.shape
        assert f4_in.shape[1] == model.taskmoe_s4.channels
        assert f4_in.shape[-1] >= 9 and f4_in.shape[-2] >= 9
        beta = float(model.taskmoe_s4.residual_scale.detach().abs().item())
        assert beta <= 1e-12, beta
        f4_out, aux = model.taskmoe_s4(f4_in)
        assert f4_out.shape == f4_in.shape
        assert float((f4_out - f4_in).abs().max().item()) <= 1e-6
        refined, _ = model._refine_stage4(fused_list)
        for i in range(3):
            assert torch.equal(refined[i], fused_list[i])
        assert refined[3].shape == fused_list[3].shape
    print('[SMOKE] 7-9 s1s3_unchanged_s4_shape_beta0')

    # 10) zero-step logits
    model.eval()
    with torch.no_grad():
        for route in ('full', 'missing'):
            model.taskmoe_enabled = False
            logits_a = model(batch['ct'], pet=batch['pet'], forward_mode=route, mask=None)['logits']
            model.taskmoe_enabled = True
            logits_b = model(batch['ct'], pet=batch['pet'], forward_mode=route, mask=None)['logits']
            diff = float((logits_a - logits_b).abs().max().item())
            assert diff <= 1e-6, (route, diff)
    model.taskmoe_enabled = True
    print('[SMOKE] 10 zero_step_logits')

    # 11) backward: Stage1 grads None, TaskMoE has grads
    model.train()
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, stats = task.train_step(batch, forward_mode='full')
    loss.backward()
    stage1_grad = any(
        p.grad is not None and float(p.grad.abs().sum().item()) > 0
        for n, p in model.named_parameters()
        if not n.startswith('taskmoe_s4.')
    )
    assert stage1_grad is False
    moe_grad = any(
        p.grad is not None
        for n, p in model.named_parameters()
        if n.startswith('taskmoe_s4.')
    )
    assert moe_grad is True
    assert 'loss_moe_balance' in stats and 'loss_seg_total' in stats
    print('[SMOKE] 11 backward_grad_isolation')

    # 12) optimizer has only TaskMoE params
    opt_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    for name, p in model.named_parameters():
        if id(p) in opt_ids:
            assert name.startswith('taskmoe_s4.'), name
        if name.startswith('taskmoe_s4.'):
            assert id(p) in opt_ids, name
    print('[SMOKE] 12 optimizer_taskmoe_only')

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'[SMOKE] trainable_param_count={len(trainable_names)}')
    print('[SMOKE] ALL PASSED')
    return trainable_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_checkpoint', type=str, default=DEFAULT_STAGE1_CKPT)
    args = parser.parse_args()
    run_smoke(args.stage1_checkpoint)


if __name__ == '__main__':
    main()
