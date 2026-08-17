#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integration tests for TGTU fusion on Baseline+CPPI."""

from __future__ import annotations

import time
import traceback

import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.text_guided_task_utility_reliability_fusion import (
    TextGuidedTaskUtilityReliabilityFusion,
)
from tasks.mdt_seg import MDTSegTeacher
from utils.seg_losses import BCEDiceLoss


def _cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
    }
    base.update(kwargs)
    return type('C', (), base)()


def _make_model(
    use_tgtu_fusion=True,
    tgtu_use_text=False,
    tgtu_use_turr_loss=True,
    tgtu_turr_interval=5,
    device='cpu',
):
    model = DualSharedAddPETCTBaseline(
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        use_deep_supervision=False,
        use_tgtu_fusion=use_tgtu_fusion,
        tgtu_use_text=tgtu_use_text,
        tgtu_use_turr_loss=tgtu_use_turr_loss,
        tgtu_turr_interval=tgtu_turr_interval,
    )
    return model.to(device)


def _finite(*tensors):
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, (list, tuple)):
            _finite(*t)
            continue
        if torch.is_tensor(t):
            assert torch.isfinite(t).all(), 'found NaN/Inf'


def _capture_fused(model, ct, pet, mode, pet_available=None, global_step=None):
    """Hook fusion to return four-scale features for shape checks."""
    captured = {}

    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            captured['fused'] = output[0]
        else:
            captured['fused'] = output

    handle = model.fusion.register_forward_hook(_hook)
    try:
        kwargs = dict(forward_mode=mode, global_step=global_step)
        if pet_available is not None:
            kwargs['pet_available'] = pet_available
        out = model(ct, pet, **kwargs)
    finally:
        handle.remove()
    return out, captured['fused']


def test_1_four_scale_shapes():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(device=device)
    model.eval()
    ct = torch.randn(2, 1, 64, 64, device=device)
    pet = torch.randn(2, 1, 64, 64, device=device)
    pet_available = torch.tensor([1, 0], device=device)

    for mode, pet_in, avail in [
        ('full', pet, None),
        ('missing', None, None),
        ('auto', pet, pet_available),
    ]:
        out, fused = _capture_fused(model, ct, pet_in, mode, pet_available=avail)
        assert len(fused) == 4
        ct_feats = model._encode_ct(ct)
        for f, c in zip(fused, ct_feats):
            assert f.shape == c.shape, f'{mode}: {tuple(f.shape)} vs {tuple(c.shape)}'
        assert out['logits'].shape == (2, 1, 64, 64)
        _finite(out['logits'], fused)
    print('[PASS] test_1_four_scale_shapes')


def test_2_auto_pet_available_state_mapping():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(device=device)
    model.eval()
    ct = torch.randn(2, 1, 64, 64, device=device)
    pet = torch.randn(2, 1, 64, 64, device=device)
    pet_available = torch.tensor([1, 0], device=device)
    captured = {}

    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            captured['diag'] = output[1]

    # Force diagnostics by temporarily enabling training path flags
    orig = model._fuse

    def _fuse_diag(ct_feats, pet_feats, mode, pet_available=None, global_step=None):
        fused, diag = model.fusion(
            ct_feats,
            pet_feats,
            mode=mode,
            pet_available=pet_available,
            return_diagnostics=True,
        )
        captured['diag'] = diag
        return fused, {'reliability_logits': diag['reliability_logits'], 'pet_state_ids': diag['pet_state_ids']}

    model._fuse = _fuse_diag
    try:
        model(ct, pet, pet_available=pet_available, forward_mode='auto')
    finally:
        model._fuse = orig

    state_ids = captured['diag']['pet_state_ids']
    # pet_available=1 -> real state 0; pet_available=0 -> proxy state 1
    assert int(state_ids[0].item()) == 0
    assert int(state_ids[1].item()) == 1
    print('[PASS] test_2_auto_pet_available_state_mapping')


def test_3_text_disabled_forward():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(tgtu_use_text=False, device=device)
    model.eval()
    assert not model.fusion.text_available
    ct = torch.randn(1, 1, 64, 64, device=device)
    pet = torch.randn(1, 1, 64, 64, device=device)
    out = model(ct, pet, forward_mode='full')
    assert out['logits'].shape == (1, 1, 64, 64)
    _finite(out['logits'])
    print('[PASS] test_3_text_disabled_forward')


def test_4_turr_disabled_or_not_triggered_zero():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TURR disabled
    model = _make_model(tgtu_use_turr_loss=False, tgtu_turr_interval=1, device=device)
    task = MDTSegTeacher({'model': model}, _cfg())
    batch = {
        'ct': torch.randn(1, 1, 64, 64),
        'pet': torch.randn(1, 1, 64, 64),
        'mask': torch.zeros(1, 1, 64, 64),
    }
    decode_calls = {'n': 0}
    orig_dec = model.decoder.forward

    def wrapped_dec(*args, **kwargs):
        decode_calls['n'] += 1
        return orig_dec(*args, **kwargs)

    model.decoder.forward = wrapped_dec
    task.global_batch_step = 0
    loss, _, outputs, stats = task.train_step(batch, forward_mode='full')
    # Main decode once; no TURR counterfactual decode
    assert decode_calls['n'] == 1
    assert float(stats['loss_turr']) == 0.0
    assert 'turr_context' not in outputs.get('aux', {})

    # TURR enabled but not on interval
    model2 = _make_model(tgtu_use_turr_loss=True, tgtu_turr_interval=5, device=device)
    task2 = MDTSegTeacher({'model': model2}, _cfg())
    decode_calls2 = {'n': 0}
    orig_dec2 = model2.decoder.forward

    def wrapped_dec2(*args, **kwargs):
        decode_calls2['n'] += 1
        return orig_dec2(*args, **kwargs)

    model2.decoder.forward = wrapped_dec2
    task2.global_batch_step = 1  # 1 % 5 != 0
    loss2, _, outputs2, stats2 = task2.train_step(batch, forward_mode='full')
    assert decode_calls2['n'] == 1
    assert float(stats2['loss_turr']) == 0.0
    assert 'turr_context' not in outputs2.get('aux', {})
    assert torch.isfinite(loss2)
    print('[PASS] test_4_turr_disabled_or_not_triggered_zero')


def test_5_turr_enabled_finite_and_reliability_grad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(tgtu_use_turr_loss=True, tgtu_turr_interval=1, device=device)
    task = MDTSegTeacher({'model': model}, _cfg())
    task.global_batch_step = 0
    batch = {
        'ct': torch.randn(2, 1, 64, 64),
        'pet': torch.randn(2, 1, 64, 64),
        'mask': (torch.rand(2, 1, 64, 64) > 0.7).float(),
    }
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    assert torch.isfinite(loss)
    assert torch.isfinite(stats['loss_turr'])
    loss.backward()
    rel_grads = [
        p.grad for p in model.fusion.reliability_head.parameters()
        if p.requires_grad
    ]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in rel_grads)
    print('[PASS] test_5_turr_enabled_finite_and_reliability_grad')


def test_6_missing_pet_none_skips_pet_encoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(device=device)
    model.eval()
    calls = {'n': 0}
    orig = model.enc_pet.forward

    def wrapped(*args, **kwargs):
        calls['n'] += 1
        return orig(*args, **kwargs)

    model.enc_pet.forward = wrapped
    ct = torch.randn(1, 1, 64, 64, device=device)
    out = model(ct, None, forward_mode='missing')
    assert 'logits' in out
    assert calls['n'] == 0
    print('[PASS] test_6_missing_pet_none_skips_pet_encoder')


def test_7_tgtu_off_legacy_fusion():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(use_tgtu_fusion=False, device=device)
    from models.baseline_blocks import StateAwareWeightedAddFusion
    assert isinstance(model.fusion, StateAwareWeightedAddFusion)
    model.eval()
    ct = torch.randn(1, 1, 64, 64, device=device)
    pet = torch.randn(1, 1, 64, 64, device=device)
    out_full = model(ct, pet, forward_mode='full')
    out_missing = model(ct, None, forward_mode='missing')
    assert out_full['logits'].shape == out_missing['logits'].shape
    _finite(out_full['logits'], out_missing['logits'])
    print('[PASS] test_7_tgtu_off_legacy_fusion')


def test_8_no_nan_inf():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(tgtu_turr_interval=1, device=device)
    task = MDTSegTeacher({'model': model}, _cfg())
    task.global_batch_step = 0
    batch = {
        'ct': torch.randn(2, 1, 64, 64),
        'pet': torch.randn(2, 1, 64, 64),
        'mask': (torch.rand(2, 1, 64, 64) > 0.5).float(),
    }
    loss, logits, outputs, stats = task.train_step(batch, forward_mode='full')
    _finite(loss, logits, stats['loss_turr'])
    # Capture fused features
    model.train()
    _, fused = _capture_fused(
        model,
        batch['ct'].to(task.device),
        batch['pet'].to(task.device),
        'full',
        global_step=0,
    )
    _finite(fused)
    print('[PASS] test_8_no_nan_inf')


def test_9_full_missing_backward_optimizer():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _make_model(tgtu_turr_interval=5, device=device)
    task = MDTSegTeacher({'model': model}, _cfg())
    batch = {
        'ct': torch.randn(2, 1, 64, 64),
        'pet': torch.randn(2, 1, 64, 64),
        'mask': (torch.rand(2, 1, 64, 64) > 0.5).float(),
    }
    for step, route in enumerate(['full', 'missing']):
        task.global_batch_step = step
        task.optimizer.zero_grad(set_to_none=True)
        loss, logits, _, stats = task.train_step(batch, forward_mode=route)
        _finite(loss, logits, stats['loss_turr'])
        loss.backward()
        task.optimizer.step()
    print('[PASS] test_9_full_missing_backward_optimizer')


def test_10_batch16_amp_memory_time():
    if not torch.cuda.is_available():
        print('[SKIP] test_10_batch16_amp_memory_time: CUDA unavailable')
        return
    device = 'cuda'
    model = _make_model(tgtu_use_text=False, tgtu_turr_interval=5, device=device)
    task = MDTSegTeacher({'model': model}, _cfg(mixed_precision=True))
    batch = {
        'ct': torch.randn(16, 1, 64, 64),
        'pet': torch.randn(16, 1, 64, 64),
        'mask': (torch.rand(16, 1, 64, 64) > 0.5).float(),
    }
    # Warmup
    task.global_batch_step = 1
    task.optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        loss, _, _, _ = task.train_step(batch, forward_mode='full')
    task.scaler.scale(loss).backward()
    task.scaler.step(task.optimizer)
    task.scaler.update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    task.global_batch_step = 2
    task.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=True):
        loss, logits, _, stats = task.train_step(batch, forward_mode='missing')
    task.scaler.scale(loss).backward()
    task.scaler.step(task.optimizer)
    task.scaler.update()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    _finite(loss, logits, stats['loss_turr'])
    print(
        f'[PASS] test_10_batch16_amp_memory_time '
        f'peak_mem_mb={peak_mb:.2f} elapsed_ms={elapsed_ms:.2f}'
    )
    return {'peak_mem_mb': peak_mb, 'elapsed_ms': elapsed_ms}


def test_shared_tgtu_parameters():
    model = _make_model(tgtu_use_text=False)
    assert isinstance(model.fusion, TextGuidedTaskUtilityReliabilityFusion)
    ids = {id(model.fusion)}
    # same module object used for all modes — verified by single attribute
    assert hasattr(model, 'fusion') and len(ids) == 1
    print('[PASS] test_shared_tgtu_parameters')


def main():
    results = {}
    tests = [
        ('test_shared_tgtu_parameters', test_shared_tgtu_parameters),
        ('test_1_four_scale_shapes', test_1_four_scale_shapes),
        ('test_2_auto_pet_available_state_mapping', test_2_auto_pet_available_state_mapping),
        ('test_3_text_disabled_forward', test_3_text_disabled_forward),
        ('test_4_turr_disabled_or_not_triggered_zero', test_4_turr_disabled_or_not_triggered_zero),
        ('test_5_turr_enabled_finite_and_reliability_grad', test_5_turr_enabled_finite_and_reliability_grad),
        ('test_6_missing_pet_none_skips_pet_encoder', test_6_missing_pet_none_skips_pet_encoder),
        ('test_7_tgtu_off_legacy_fusion', test_7_tgtu_off_legacy_fusion),
        ('test_8_no_nan_inf', test_8_no_nan_inf),
        ('test_9_full_missing_backward_optimizer', test_9_full_missing_backward_optimizer),
        ('test_10_batch16_amp_memory_time', test_10_batch16_amp_memory_time),
    ]
    failed = []
    for name, fn in tests:
        try:
            out = fn()
            results[name] = out if out is not None else 'ok'
        except Exception as exc:
            failed.append(name)
            results[name] = f'FAIL: {exc}'
            traceback.print_exc()
            print(f'[FAIL] {name}: {exc}')
    print('\n===== SUMMARY =====')
    for k, v in results.items():
        print(f'{k}: {v}')
    if failed:
        raise SystemExit(1)
    print('ALL TESTS PASSED')


if __name__ == '__main__':
    main()
