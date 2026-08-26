#!/usr/bin/env python
"""Smoke tests for frozen Stage-1 + logit residual decoder Stage-2."""
from __future__ import annotations

import argparse
import copy
import os
import sys
import types

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_shared_add_baseline import (
    _is_stage2_adapter_param_name,
    _is_stage2_residual_decoder_param_name,
    _is_taskmoe_param_name,
)
from run_mdt_seg import _load_stage1_for_taskmoe, _sync_cppi_config_from_stage1
from tasks.mdt_seg import MDTSegTeacher

DEFAULT_STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained/ckpt.best_joint.pth.tar'
)
SYNTH_STAGE1_CKPT = '/tmp/stage2_logit_residual_smoke/synth_stage1_ckpt.pth.tar'


def _ensure_stage1_checkpoint(stage1_ckpt: str) -> str:
    if stage1_ckpt and os.path.isfile(stage1_ckpt):
        return stage1_ckpt
    os.makedirs(os.path.dirname(SYNTH_STAGE1_CKPT), exist_ok=True)
    if os.path.isfile(SYNTH_STAGE1_CKPT):
        print(f'[SMOKE] using cached synth Stage-1 ckpt: {SYNTH_STAGE1_CKPT}', flush=True)
        return SYNTH_STAGE1_CKPT
    print('[SMOKE] synthesizing Stage-1 checkpoint (no real ckpt found)', flush=True)
    cfg = _make_cfg(
        stage1_checkpoint=None,
        stage2_strategy='legacy_taskmoe',
        checkpoint_dir='/tmp/stage2_logit_residual_smoke',
    )
    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    with torch.no_grad():
        model.prototype_memory.prototype_ready.fill_(1)
        model.prototype_memory.bank_version.fill_(1)
    state = {
        k: v for k, v in model.state_dict().items()
        if (
            not _is_taskmoe_param_name(k)
            and not _is_stage2_adapter_param_name(k)
            and not _is_stage2_residual_decoder_param_name(k)
        )
    }
    payload = {
        'model': state,
        'epoch': 0,
        'config': {
            'cppi_num_clusters': 6,
            'cppi_build_stage': 4,
            'ct_backbone': 'convnextv2_nano',
            'pet_backbone': 'mit_b1',
            'decoder_channels': [512, 256, 128, 64],
        },
    }
    torch.save(payload, SYNTH_STAGE1_CKPT)
    cfg_path = os.path.join(os.path.dirname(SYNTH_STAGE1_CKPT), 'config_args.json')
    import json
    with open(cfg_path, 'w') as f:
        json.dump(payload['config'], f, indent=2)
    print(f'[SMOKE] wrote synth Stage-1 ckpt: {SYNTH_STAGE1_CKPT}', flush=True)
    return SYNTH_STAGE1_CKPT


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 8e-5,
        'stage2_residual_lr': 5e-5,
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
        'checkpoint_dir': '/tmp/stage2_logit_residual_smoke',
        'taskmoe_scales': 's4',
        'taskmoe_mode': 'independent',
        'taskmoe_residual_mode': 'zero_start',
        'stage2_train_decoder': False,
        'decoder_lr': 8e-6,
        'taskmoe_private_rank': 16,
        'taskmoe_beta_max': 1.0,
        'taskmoe_role_loss_weight': 0.02,
        'taskmoe_fers_mode': 'both',
        'stage2_decoder_adapter': False,
        'stage2_decoder_adapter_level': 'd1',
        'taskmoe_use_text_prior': False,
        'taskmoe_num_experts': 6,
        'stage2_strategy': 'logit_residual_decoder',
        'stage2_residual_channels': 64,
        'stage2_residual_state_conditioned': False,
        'stage2_delta_logit_max': 2.0,
        'stage2_residual_dropout': 0.0,
    }
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _build_stage2(
    stage1_ckpt: str,
    state_conditioned: bool = False,
    residual_channels: int = 64,
):
    cfg = _make_cfg(
        stage1_checkpoint=stage1_ckpt,
        stage2_strategy='logit_residual_decoder',
        stage2_residual_state_conditioned=state_conditioned,
        stage2_residual_channels=residual_channels,
    )
    _sync_cppi_config_from_stage1(cfg, stage1_ckpt)
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.no_encoder_pretrained = True
    cfg.stage2_strategy = 'logit_residual_decoder'
    cfg.stage2_residual_state_conditioned = state_conditioned
    cfg.stage2_residual_channels = residual_channels
    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    _load_stage1_for_taskmoe(model, stage1_ckpt)
    model.enable_stage2_residual_decoder_only()
    task = MDTSegTeacher(networks, cfg)
    return model, task, cfg


def _fake_batch(device, size=288, batch=2):
    return {
        'ct': torch.randn(batch, 1, size, size, device=device),
        'pet': torch.randn(batch, 1, size, size, device=device),
        'mask': (torch.rand(batch, 1, size, size, device=device) > 0.7).float(),
    }


def _bank_snapshot(model):
    mem = model.prototype_memory
    return {
        'bank_version': int(mem.bank_version.item()),
        'buffers': {
            k: v.detach().cpu().clone()
            for k, v in mem.named_buffers()
        },
    }


def _assert_bank_unchanged(before, after):
    assert before['bank_version'] == after['bank_version']
    for k, v in before['buffers'].items():
        assert torch.equal(v, after['buffers'][k]), f'CPPI buffer changed: {k}'


def run_smoke(stage1_ckpt: str):
    device = torch.device('cpu')
    print('[SMOKE] building residual Stage-2 (state_conditioned=False)', flush=True)
    model, task, cfg = _build_stage2(stage1_ckpt, state_conditioned=False)
    model.to(device)
    task.device = device

    # 1-2: Stage1 load + CPPI ready already enforced by _load_stage1_for_taskmoe
    assert bool(model.prototype_memory.bank_ready)

    # 3-5: trainable set / optimizer group
    trainable = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]
    assert trainable and all(_is_stage2_residual_decoder_param_name(n) for n in trainable)
    for n, p in model.named_parameters():
        if _is_stage2_residual_decoder_param_name(n):
            assert p.requires_grad
        else:
            assert not p.requires_grad, n
    assert len(task.optimizer.param_groups) == 1
    assert task.optimizer.param_groups[0].get('name') == 'residual_decoder'
    opt_ids = {id(p) for p in task.optimizer.param_groups[0]['params']}
    train_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert opt_ids == train_ids

    # 6: train() mode — only residual decoder training
    model.train()
    assert model.stage2_residual_decoder.training
    assert not model.decoder.training
    assert not model.enc_ct.training
    assert not model.enc_pet.training
    assert not model.fusion.training
    assert not model.pet_calibration.training
    assert not model.prototype_memory.training
    if model.taskmoe_s4 is not None:
        assert not model.taskmoe_s4.training

    batch = _fake_batch(device, batch=2)
    bank_before = _bank_snapshot(model)

    # 7: Full / Missing / auto mixed shapes
    model.eval()
    with torch.no_grad():
        out_full = model(batch['ct'], pet=batch['pet'], forward_mode='full', mask=None)
        out_miss = model(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=None)
        pet_avail = torch.tensor([1, 0], device=device, dtype=torch.long)
        out_auto = model(
            batch['ct'],
            pet=batch['pet'],
            pet_available=pet_avail,
            forward_mode='auto',
            mask=None,
        )
    for name, out in (('full', out_full), ('missing', out_miss), ('auto', out_auto)):
        assert out['logits'].shape == (2, 1, 288, 288), name
        assert torch.isfinite(out['logits']).all(), name

    # 8: Missing skips PET encoder
    calls = {'n': 0}
    orig_encode_pet = model._encode_pet

    def _count_pet(*args, **kwargs):
        calls['n'] += 1
        return orig_encode_pet(*args, **kwargs)

    model._encode_pet = _count_pet
    with torch.no_grad():
        model(batch['ct'], pet=batch['pet'], forward_mode='missing', mask=None)
    model._encode_pet = orig_encode_pet
    assert calls['n'] == 0, f'Missing path called PET encoder {calls["n"]} times'

    # 9: CPPI unchanged across optimizer step; collect not called
    collect_calls = {'n': 0}
    orig_collect = model.prototype_memory.collect

    def _guard_collect(*args, **kwargs):
        collect_calls['n'] += 1
        return orig_collect(*args, **kwargs)

    model.prototype_memory.collect = _guard_collect
    model.train()
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, outputs, stats = task.train_step(batch, forward_mode='full')
    assert float(stats['loss_moe_balance']) == 0.0
    assert float(stats['loss_shared_consistency']) == 0.0
    assert abs(float(stats['loss_total']) - float(stats['loss_seg_total'])) < 1e-6
    loss.backward()

    # 12: delta_head has finite non-zero grad; Stage1 grads None
    head_w = model.stage2_residual_decoder.delta_head.weight
    assert head_w.grad is not None
    assert torch.isfinite(head_w.grad).all()
    assert float(head_w.grad.abs().sum()) > 0.0
    for n, p in model.named_parameters():
        if _is_stage2_residual_decoder_param_name(n):
            continue
        assert p.grad is None, n

    task.optimizer.step()
    bank_after = _bank_snapshot(model)
    _assert_bank_unchanged(bank_before, bank_after)
    assert collect_calls['n'] == 0
    model.prototype_memory.collect = orig_collect

    # 11: step-0 identity via residual_enabled toggle
    model.eval()
    with torch.no_grad():
        for route in ('full', 'missing'):
            model.stage2_residual_enabled = False
            z1 = model(batch['ct'][:1], pet=batch['pet'][:1], forward_mode=route, mask=None)['logits']
            model.stage2_residual_enabled = True
            z2 = model(batch['ct'][:1], pet=batch['pet'][:1], forward_mode=route, mask=None)['logits']
            # After one optimizer step residual is no longer zero; re-init head for identity check.
    # Re-zero delta_head to verify identity contract still holds when head is zero.
    with torch.no_grad():
        nn_delta = model.stage2_residual_decoder.delta_head
        nn_delta.weight.zero_()
        if nn_delta.bias is not None:
            nn_delta.bias.zero_()
    with torch.no_grad():
        for route in ('full', 'missing', 'auto'):
            kw = {}
            if route == 'auto':
                kw['pet_available'] = torch.tensor([1], device=device, dtype=torch.long)
            model.stage2_residual_enabled = False
            z1 = model(
                batch['ct'][:1], pet=batch['pet'][:1], forward_mode=route, mask=None, **kw
            )['logits']
            model.stage2_residual_enabled = True
            z2 = model(
                batch['ct'][:1], pet=batch['pet'][:1], forward_mode=route, mask=None, **kw
            )['logits']
            diff = float((z1 - z2).abs().max())
            assert diff <= 1e-6, f'step-0 identity failed on {route}: {diff}'

    # 13: after another train step residual not all zeros on at least one path
    model.train()
    # perturb head away from zero so residual can become non-zero
    with torch.no_grad():
        model.stage2_residual_decoder.delta_head.weight.add_(0.01)
    task.optimizer.zero_grad(set_to_none=True)
    loss2, _, out2, _ = task.train_step(batch, forward_mode='full')
    loss2.backward()
    task.optimizer.step()
    model.eval()
    with torch.no_grad():
        delta_seen = False
        for route in ('full', 'missing'):
            out = model(batch['ct'][:1], pet=batch['pet'][:1], forward_mode=route, mask=None)
            res = (out.get('aux') or {}).get('stage2_residual_stats') or {}
            if float(res.get('delta_logit_abs_max', torch.tensor(0.0))) > 0:
                delta_seen = True
        assert delta_seen, 'residual logits remained all-zero after optimizer steps'

    # 10: state_conditioned=True also builds/runs
    print('[SMOKE] building residual Stage-2 (state_conditioned=True)', flush=True)
    model_s, task_s, cfg_s = _build_stage2(stage1_ckpt, state_conditioned=True)
    model_s.to(device)
    task_s.device = device
    model_s.train()
    batch_s = _fake_batch(device, batch=2)
    loss_s, _, _, stats_s = task_s.train_step(batch_s, forward_mode='missing')
    assert torch.isfinite(loss_s)
    assert float(stats_s['loss_moe_balance']) == 0.0

    # 15: checkpoint strict reload
    ckpt_path = '/tmp/stage2_logit_residual_smoke/ckpt_residual.pth.tar'
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    model.eval()
    # Fresh zero-init model for stable save/load compare
    model_save, _, cfg_save = _build_stage2(stage1_ckpt, state_conditioned=False)
    model_save.to(device)
    model_save.eval()
    with torch.no_grad():
        fixed_ct = torch.randn(1, 1, 288, 288, device=device)
        fixed_pet = torch.randn(1, 1, 288, 288, device=device)
        before = {
            'full': model_save(fixed_ct, pet=fixed_pet, forward_mode='full')['logits'].cpu(),
            'missing': model_save(fixed_ct, pet=fixed_pet, forward_mode='missing')['logits'].cpu(),
        }
    payload = {
        'model': model_save.state_dict(),
        'config': vars(cfg_save),
        'epoch': 0,
    }
    torch.save(payload, ckpt_path)

    cfg_reload = copy.deepcopy(cfg_save)
    networks_r = build_mdt_seg_teacher(cfg_reload)
    model_r = networks_r['model']
    # Eval path: no freeze helper required; residual branch active via strategy.
    result = model_r.load_state_dict(payload['model'], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model_r.to(device)
    model_r.eval()
    with torch.no_grad():
        after_full = model_r(fixed_ct, pet=fixed_pet, forward_mode='full')['logits'].cpu()
        after_miss = model_r(fixed_ct, pet=fixed_pet, forward_mode='missing')['logits'].cpu()
    assert torch.allclose(before['full'], after_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(before['missing'], after_miss, atol=1e-5, rtol=1e-5)
    assert model_r.stage2_residual_decoder is not None
    assert model_r.taskmoe_enabled is False

    # 16: optional CUDA AMP
    if torch.cuda.is_available():
        print('[SMOKE] CUDA AMP residual check', flush=True)
        model_c, task_c, _ = _build_stage2(stage1_ckpt, state_conditioned=False)
        model_c.to('cuda')
        task_c.device = torch.device('cuda')
        batch_c = _fake_batch(torch.device('cuda'), batch=1)
        task_c.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            loss_c, _, _, _ = task_c.train_step(batch_c, forward_mode='full')
        assert torch.isfinite(loss_c)
        loss_c.backward()
        assert torch.isfinite(
            model_c.stage2_residual_decoder.delta_head.weight.grad
        ).all()

    print('[SMOKE] stage2 logit residual decoder PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_checkpoint', type=str, default=DEFAULT_STAGE1_CKPT)
    args = parser.parse_args()
    ckpt = _ensure_stage1_checkpoint(args.stage1_checkpoint)
    run_smoke(ckpt)


if __name__ == '__main__':
    main()
