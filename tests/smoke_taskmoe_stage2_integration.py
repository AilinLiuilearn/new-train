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
from models.dual_shared_add_baseline import (
    DualSharedAddPETCTBaseline,
    _is_stage2_adapter_param_name,
    _is_taskmoe_param_name,
)
from run_mdt_seg import _load_stage1_for_taskmoe, _sync_cppi_config_from_stage1
from tasks.mdt_seg import MDTSegTeacher

DEFAULT_STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained/ckpt.best_joint.pth.tar'
)
SYNTH_STAGE1_CKPT = '/tmp/taskmoe_stage2_smoke/synth_stage1_ckpt.pth.tar'


def _ensure_stage1_checkpoint(stage1_ckpt: str) -> str:
    """Use real Stage-1 ckpt if present; otherwise synthesize a bank-ready stub."""
    if stage1_ckpt and os.path.isfile(stage1_ckpt):
        return stage1_ckpt
    os.makedirs(os.path.dirname(SYNTH_STAGE1_CKPT), exist_ok=True)
    if os.path.isfile(SYNTH_STAGE1_CKPT):
        print(f'[SMOKE] using cached synth Stage-1 ckpt: {SYNTH_STAGE1_CKPT}', flush=True)
        return SYNTH_STAGE1_CKPT
    print('[SMOKE] synthesizing Stage-1 checkpoint (no real ckpt found)', flush=True)
    cfg = _make_cfg(
        stage1_checkpoint=None,
        taskmoe_scales='s4',
        taskmoe_mode='independent',
        checkpoint_dir='/tmp/taskmoe_stage2_smoke',
    )
    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    # Mark CPPI bank ready without running collect/finalize.
    with torch.no_grad():
        model.prototype_memory.prototype_ready.fill_(1)
        model.prototype_memory.bank_version.fill_(1)
    # Drop TaskMoE keys so Stage-2 load reports them as allowed missing.
    state = {
        k: v for k, v in model.state_dict().items()
        if not _is_taskmoe_param_name(k) and not _is_stage2_adapter_param_name(k)
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
    # Also place config next to synth path expected by _sync if using that dir.
    print(f'[SMOKE] wrote synth Stage-1 ckpt: {SYNTH_STAGE1_CKPT}', flush=True)
    return SYNTH_STAGE1_CKPT


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
        'taskmoe_scales': 's4',
        'taskmoe_mode': 'independent',
        'taskmoe_residual_mode': 'zero_start',
        'stage2_train_decoder': False,
        'decoder_lr': 8e-6,
        'taskmoe_private_rank': 16,
        'taskmoe_beta_max': 1.0,
        'taskmoe_shared_consistency_weight': 0.01,
        'taskmoe_shared_consistency_interval': 1,
        'stage2_decoder_adapter': False,
        'stage2_decoder_adapter_level': 'd1',
        'taskmoe_use_text_prior': False,
        'taskmoe_num_experts': 6,
    }
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _build_stage2(
    stage1_ckpt: str,
    taskmoe_scales: str = 's4',
    taskmoe_mode: str = 'independent',
    taskmoe_residual_mode: str = 'zero_start',
    stage2_train_decoder: bool = False,
    learning_rate: float = 8e-5,
    decoder_lr: float = 8e-6,
    taskmoe_private_rank: int = 16,
    taskmoe_beta_max: float = 1.0,
    taskmoe_shared_consistency_weight: float = 0.01,
    taskmoe_shared_consistency_interval: int = 1,
    stage2_decoder_adapter: bool = False,
    stage2_decoder_adapter_level: str = 'd1',
):
    cfg = _make_cfg(
        stage1_checkpoint=stage1_ckpt,
        taskmoe_scales=taskmoe_scales,
        taskmoe_mode=taskmoe_mode,
        taskmoe_residual_mode=taskmoe_residual_mode,
        stage2_train_decoder=stage2_train_decoder,
        learning_rate=learning_rate,
        decoder_lr=decoder_lr,
        taskmoe_private_rank=taskmoe_private_rank,
        taskmoe_beta_max=taskmoe_beta_max,
        taskmoe_shared_consistency_weight=taskmoe_shared_consistency_weight,
        taskmoe_shared_consistency_interval=taskmoe_shared_consistency_interval,
        stage2_decoder_adapter=stage2_decoder_adapter,
        stage2_decoder_adapter_level=stage2_decoder_adapter_level,
    )
    _sync_cppi_config_from_stage1(cfg, stage1_ckpt)
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.no_encoder_pretrained = True
    cfg.taskmoe_scales = taskmoe_scales
    cfg.taskmoe_mode = taskmoe_mode
    cfg.taskmoe_residual_mode = taskmoe_residual_mode
    cfg.stage2_train_decoder = stage2_train_decoder
    cfg.learning_rate = learning_rate
    cfg.decoder_lr = decoder_lr
    cfg.taskmoe_private_rank = taskmoe_private_rank
    cfg.taskmoe_beta_max = taskmoe_beta_max
    cfg.taskmoe_shared_consistency_weight = taskmoe_shared_consistency_weight
    cfg.taskmoe_shared_consistency_interval = taskmoe_shared_consistency_interval
    cfg.stage2_decoder_adapter = stage2_decoder_adapter
    cfg.stage2_decoder_adapter_level = stage2_decoder_adapter_level
    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    _load_stage1_for_taskmoe(model, stage1_ckpt)
    model.enable_stage2_moe_only(train_decoder=bool(stage2_train_decoder))
    task = MDTSegTeacher(networks, cfg)
    return model, task, cfg


def _fake_batch(device, size=288, batch=1):
    return {
        'ct': torch.randn(batch, 1, size, size, device=device),
        'pet': torch.randn(batch, 1, size, size, device=device),
        'mask': torch.zeros(batch, 1, size, size, device=device),
    }


def run_smoke(stage1_ckpt: str):
    stage1_ckpt = _ensure_stage1_checkpoint(stage1_ckpt)
    assert os.path.isfile(stage1_ckpt), f'missing Stage-1 checkpoint: {stage1_ckpt}'
    model, task, cfg = _build_stage2(stage1_ckpt, taskmoe_scales='s4')
    device = task.device
    model.to(device)

    # 1) load: non-taskmoe missing already rejected inside _load_stage1_for_taskmoe
    print('[SMOKE] 1 load_ok')

    # 2) Stage1 frozen
    stage1_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and not _is_taskmoe_param_name(n)
    ]
    assert not stage1_trainable, stage1_trainable
    print('[SMOKE] 2 stage1_frozen')

    # 3) TaskMoE trainable
    moe_frozen = [
        n for n, p in model.named_parameters()
        if _is_taskmoe_param_name(n) and not p.requires_grad
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

    # 7/8/9) S1-S3 unchanged for s4-only; S4 shape; beta=0 identity
    model.eval()
    with torch.no_grad():
        ct_feats = model._encode_ct(batch['ct'])
        pet_feats = model._encode_pet(batch['pet'])
        pet_cal = model.pet_calibration(ct_feats, pet_feats, None, reference_valid=False)
        fused = model.fusion(ct_feats, pet_cal, mode='full')
        fused_list = [f.detach().clone() for f in fused]
        f4_in = fused_list[3]
        assert f4_in.shape == (batch['ct'].shape[0], 512, batch['ct'].shape[-2] // 32, batch['ct'].shape[-1] // 32), f4_in.shape
        assert f4_in.shape[1] == model.taskmoe_s4.channels
        assert f4_in.shape[-1] >= 9 and f4_in.shape[-2] >= 9
        beta = float(model.taskmoe_s4.residual_scale.detach().abs().item())
        assert beta <= 1e-12, beta
        f4_out, aux = model.taskmoe_s4(f4_in)
        assert f4_out.shape == f4_in.shape
        assert float((f4_out - f4_in).abs().max().item()) <= 1e-6
        refined, _, _ = model._refine_stage4(fused_list)
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
        if not _is_taskmoe_param_name(n)
    )
    assert stage1_grad is False
    moe_grad = any(
        p.grad is not None
        for n, p in model.named_parameters()
        if _is_taskmoe_param_name(n)
    )
    assert moe_grad is True
    assert 'loss_moe_balance' in stats and 'loss_seg_total' in stats
    assert stats['loss_moe_balance'].dtype == torch.float32
    print('[SMOKE] 11 backward_grad_isolation')

    # 12) optimizer has only TaskMoE params
    opt_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    for name, p in model.named_parameters():
        if id(p) in opt_ids:
            assert _is_taskmoe_param_name(name), name
        if _is_taskmoe_param_name(name):
            assert id(p) in opt_ids, name
    print('[SMOKE] 12 optimizer_taskmoe_only')

    # 13) multi-scale ablation paths: s3s4 and all scales
    for scales_opt, expect in (
        ('s3s4', ('s3', 's4')),
        ('s2s3s4', ('s2', 's3', 's4')),
        ('all', ('s1', 's2', 's3', 's4')),
    ):
        model_m, task_m, _ = _build_stage2(stage1_ckpt, taskmoe_scales=scales_opt)
        model_m.to(device)
        assert model_m.taskmoe_scales == expect, (scales_opt, model_m.taskmoe_scales)
        for scale_name in expect:
            assert getattr(model_m, f'taskmoe_{scale_name}') is not None
        batch_m = _fake_batch(device)
        model_m.eval()
        with torch.no_grad():
            model_m.taskmoe_enabled = False
            logits_a = model_m(batch_m['ct'], pet=batch_m['pet'], forward_mode='full', mask=None)['logits']
            model_m.taskmoe_enabled = True
            logits_b = model_m(batch_m['ct'], pet=batch_m['pet'], forward_mode='full', mask=None)['logits']
            assert float((logits_a - logits_b).abs().max().item()) <= 1e-6, scales_opt
            ct_feats = model_m._encode_ct(batch_m['ct'])
            pet_feats = model_m._encode_pet(batch_m['pet'])
            pet_cal = model_m.pet_calibration(ct_feats, pet_feats, None, reference_valid=False)
            fused = model_m.fusion(ct_feats, pet_cal, mode='full')
            fused_list = [f.detach().clone() for f in fused]
            refined, aux, _ = model_m._refine_stage4(fused_list)
            active_idx = {idx for _, idx, _ in model_m._iter_active_taskmoe()}
            for i in range(4):
                if i in active_idx:
                    assert refined[i].shape == fused_list[i].shape
                else:
                    assert torch.equal(refined[i], fused_list[i])
            assert aux.dtype == torch.float32
        model_m.train()
        task_m.optimizer.zero_grad(set_to_none=True)
        loss_m, _, _, _ = task_m.train_step(batch_m, forward_mode='missing')
        loss_m.backward()
        for scale_name in expect:
            assert any(
                p.grad is not None
                for n, p in model_m.named_parameters()
                if n.startswith(f'taskmoe_{scale_name}.')
            ), scale_name
        print(f'[SMOKE] 13 multi_scale_ok scales={scales_opt}')

    # 14) cross_scale_shared mode
    model_s, task_s, _ = _build_stage2(
        stage1_ckpt, taskmoe_scales='all', taskmoe_mode='cross_scale_shared'
    )
    model_s.to(device)
    assert model_s.taskmoe_mode == 'cross_scale_shared'
    assert model_s.cross_scale_taskmoe is not None
    assert model_s.taskmoe_s1 is None and model_s.taskmoe_s4 is None
    shared_names = [n for n, p in model_s.named_parameters() if p.requires_grad]
    assert shared_names and all(n.startswith('cross_scale_taskmoe.') for n in shared_names)
    batch_s = _fake_batch(device)
    model_s.eval()
    with torch.no_grad():
        model_s.taskmoe_enabled = False
        la = model_s(batch_s['ct'], pet=batch_s['pet'], forward_mode='full', mask=None)['logits']
        model_s.taskmoe_enabled = True
        lb = model_s(batch_s['ct'], pet=batch_s['pet'], forward_mode='full', mask=None)['logits']
        assert float((la - lb).abs().max().item()) <= 1e-6
        beta = model_s.cross_scale_taskmoe.beta.detach()
        assert float(beta.abs().max().item()) <= 1e-12
    model_s.train()
    task_s.optimizer.zero_grad(set_to_none=True)
    loss_s, _, _, stats_s = task_s.train_step(batch_s, forward_mode='missing')
    assert stats_s['loss_moe_balance'].dtype == torch.float32
    loss_s.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in model_s.named_parameters()
        if n.startswith('cross_scale_taskmoe.') and p.requires_grad
    )
    print('[SMOKE] 14 cross_scale_shared_ok')

    # 15) paper residual mode: no beta, first-step grads on MoE path
    model_p, task_p, _ = _build_stage2(
        stage1_ckpt,
        taskmoe_scales='all',
        taskmoe_mode='cross_scale_shared',
        taskmoe_residual_mode='paper',
    )
    model_p.to(device)
    assert model_p.cross_scale_taskmoe.beta is None
    assert model_p.cross_scale_taskmoe.residual_mode == 'paper'
    assert not any(n.endswith('.beta') for n, _ in model_p.named_parameters())
    batch_p = _fake_batch(device)
    model_p.eval()
    with torch.no_grad():
        out_f = model_p(batch_p['ct'], pet=batch_p['pet'], forward_mode='full', mask=None)
        out_m = model_p(batch_p['ct'], pet=batch_p['pet'], forward_mode='missing', mask=None)
        assert torch.isfinite(out_f['logits']).all() and torch.isfinite(out_m['logits']).all()
        assert torch.isfinite(out_f['aux']['taskmoe_balance_loss'])
        for i in range(1, 5):
            assert torch.isfinite(out_f['aux']['taskmoe_stats'][f's{i}_delta_feat_ratio'])
    model_p.train()
    task_p.optimizer.zero_grad(set_to_none=True)
    loss_p, _, _, _ = task_p.train_step(batch_p, forward_mode='full')
    loss_p.backward()
    def _has_finite(prefix):
        return any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for n, p in model_p.named_parameters()
            if n.startswith(prefix) and p.requires_grad
        )
    assert _has_finite('cross_scale_taskmoe.scale_adapters.0.router')
    assert _has_finite('cross_scale_taskmoe.shared_expert_bank.experts')
    assert _has_finite('cross_scale_taskmoe.scale_adapters.0.out_proj')
    assert _has_finite('cross_scale_taskmoe.scale_adapters.0.prompt')
    task_p.optimizer.step()
    print('[SMOKE] 15 paper_residual_ok')

    # 16) state_scale_factorized: freeze / identity / route / eligibility grads
    model_f, task_f, _ = _build_stage2(
        stage1_ckpt,
        taskmoe_scales='all',
        taskmoe_mode='state_scale_factorized',
        taskmoe_shared_consistency_weight=0.01,
        stage2_decoder_adapter=True,
    )
    model_f.to(device)
    assert model_f.taskmoe_mode == 'state_scale_factorized'
    assert model_f.state_scale_taskmoe is not None
    assert model_f.stage2_decoder_adapter is not None
    assert model_f.cross_scale_taskmoe is None
    trainable = [n for n, p in model_f.named_parameters() if p.requires_grad]
    assert trainable
    assert all(
        _is_taskmoe_param_name(n) or _is_stage2_adapter_param_name(n)
        for n in trainable
    )
    assert not any(n.startswith('decoder.') for n in trainable)
    batch_f = _fake_batch(device, batch=2)
    model_f.eval()
    with torch.no_grad():
        beta = model_f.state_scale_taskmoe.effective_beta().detach()
        assert float(beta.abs().max().item()) <= 1e-12
        for route in ('full', 'missing'):
            model_f.taskmoe_enabled = False
            la = model_f(batch_f['ct'], pet=batch_f['pet'], forward_mode=route, mask=None)['logits']
            model_f.taskmoe_enabled = True
            lb = model_f(batch_f['ct'], pet=batch_f['pet'], forward_mode=route, mask=None)['logits']
            diff = float((la - lb).abs().max().item())
            assert diff <= 1e-6, (route, diff)
    # Missing skips PET encoder
    pet_calls = {'n': 0}
    orig_encode_pet = model_f._encode_pet

    def _encode_pet_guard(*args, **kwargs):
        pet_calls['n'] += 1
        return orig_encode_pet(*args, **kwargs)

    model_f._encode_pet = _encode_pet_guard
    model_f.train()
    _ = model_f(batch_f['ct'], pet=batch_f['pet'], mask=batch_f['mask'], forward_mode='missing')
    assert pet_calls['n'] == 0
    model_f._encode_pet = orig_encode_pet

    # CPPI bank version unchanged across an optimizer step
    bank_before = int(model_f.prototype_memory.bank_version.item())
    task_f.optimizer.zero_grad(set_to_none=True)
    loss_f, _, _, stats_f = task_f.train_step(batch_f, forward_mode='full')
    assert 'loss_shared_consistency' in stats_f
    assert torch.isfinite(stats_f['loss_shared_consistency'])
    loss_f.backward()
    stage1_grad = any(
        p.grad is not None and float(p.grad.abs().sum().item()) > 0
        for n, p in model_f.named_parameters()
        if not (_is_taskmoe_param_name(n) or _is_stage2_adapter_param_name(n))
    )
    assert stage1_grad is False
    assert any(
        p.grad is not None
        for n, p in model_f.named_parameters()
        if n.startswith('state_scale_taskmoe.shared_expert.')
    )
    assert any(
        p.grad is not None
        for n, p in model_f.named_parameters()
        if n.startswith('stage2_decoder_adapter.')
    )
    task_f.optimizer.step()
    bank_after = int(model_f.prototype_memory.bank_version.item())
    assert bank_before == bank_after
    print('[SMOKE] 16 state_scale_factorized_freeze_identity_ok')

    # 17) eligibility: Full updates only full-state expert; S1 updates only scale-s1
    model_e, task_e, _ = _build_stage2(
        stage1_ckpt,
        taskmoe_scales='all',
        taskmoe_mode='state_scale_factorized',
        taskmoe_shared_consistency_weight=0.0,
        stage2_decoder_adapter=False,
    )
    model_e.to(device)
    batch_e = _fake_batch(device, batch=1)
    model_e.train()
    task_e.optimizer.zero_grad(set_to_none=True)
    out_full = model_e(batch_e['ct'], pet=batch_e['pet'], mask=batch_e['mask'], forward_mode='full')
    out_full['logits'].mean().backward()
    assert model_e.state_scale_taskmoe.state_experts[1].fc1.weight.grad is not None
    assert model_e.state_scale_taskmoe.state_experts[0].fc1.weight.grad is None
    task_e.optimizer.zero_grad(set_to_none=True)
    out_miss = model_e(batch_e['ct'], pet=batch_e['pet'], mask=batch_e['mask'], forward_mode='missing')
    out_miss['logits'].mean().backward()
    assert model_e.state_scale_taskmoe.state_experts[0].fc1.weight.grad is not None
    assert model_e.state_scale_taskmoe.state_experts[1].fc1.weight.grad is None
    # auto route mixed batch
    task_e.optimizer.zero_grad(set_to_none=True)
    batch_mix = _fake_batch(device, batch=2)
    pet_available = torch.tensor([1, 0], device=device, dtype=torch.long)
    out_auto = model_e(
        batch_mix['ct'],
        pet=batch_mix['pet'],
        pet_available=pet_available,
        mask=batch_mix['mask'],
        forward_mode='auto',
    )
    out_auto['logits'].mean().backward()
    assert model_e.state_scale_taskmoe.state_experts[0].fc1.weight.grad is not None
    assert model_e.state_scale_taskmoe.state_experts[1].fc1.weight.grad is not None
    assert model_e.state_scale_taskmoe.shared_expert.fc1.weight.grad is not None
    print('[SMOKE] 17 route_eligibility_ok')

    # 18) forbidden combos raise
    try:
        _build_stage2(
            stage1_ckpt,
            taskmoe_scales='all',
            taskmoe_mode='state_scale_factorized',
            taskmoe_residual_mode='paper',
        )
        raise AssertionError('expected ValueError for paper residual')
    except ValueError:
        pass
    try:
        DualSharedAddPETCTBaseline(
            ct_backbone='convnextv2_nano',
            pet_backbone='mit_b1',
            ct_pretrained_path=None,
            pet_pretrained_path=None,
            taskmoe_mode='state_scale_factorized',
            taskmoe_scales='all',
            taskmoe_use_text_prior=True,
        )
        raise AssertionError('expected ValueError for text prior')
    except ValueError:
        pass
    print('[SMOKE] 18 forbidden_combos_ok')

    # 19) numeric: gates in [0,1], beta bounded, finite AMP-ish fp32 path
    model_n, _, _ = _build_stage2(
        stage1_ckpt,
        taskmoe_scales='all',
        taskmoe_mode='state_scale_factorized',
        taskmoe_shared_consistency_weight=0.0,
        stage2_decoder_adapter=True,
    )
    model_n.to(device)
    model_n.train()
    batch_n = _fake_batch(device, batch=1)
    out_n = model_n(batch_n['ct'], pet=batch_n['pet'], mask=batch_n['mask'], forward_mode='full')
    assert torch.isfinite(out_n['logits']).all()
    stats_n = out_n['aux']['taskmoe_stats']
    for i in range(1, 5):
        g_s = float(stats_n[f's{i}_scale_gate_mean'])
        g_t = float(stats_n[f's{i}_state_gate_mean'])
        assert 0.0 <= g_s <= 1.0
        assert 0.0 <= g_t <= 1.0
        b = float(stats_n[f's{i}_beta'])
        assert abs(b) <= float(model_n.state_scale_taskmoe.beta_max) + 1e-6
    print('[SMOKE] 19 numerical_contract_ok')

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
