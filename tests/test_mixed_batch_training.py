# -*- coding: utf-8 -*-
"""Tests for per-sample mixed Full/Missing batch training."""
import csv
import math

import pytest
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from run_mdt_seg import build_balanced_pet_available
from tasks.mdt_seg import MDTSegTeacher
from utils.seg_losses import BCEDiceLoss


def _make_cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'train_batch_mode': 'mixed',
    }
    base.update(kwargs)
    return type('C', (), base)()


def _make_task(batch_size=16, size=64):
    torch.manual_seed(0)
    cfg = _make_cfg()
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, cfg)
    task.global_batch_step = 0
    return task


def _make_batch(batch_size=16, size=64):
    return {
        'ct': torch.randn(batch_size, 1, size, size),
        'pet': torch.randn(batch_size, 1, size, size),
        'mask': (torch.rand(batch_size, 1, size, size) > 0.5).float(),
    }


def test_balanced_state_16_has_exact_half():
    state = build_balanced_pet_available(16, 0, 2023, torch.device('cpu'))
    assert state.shape == (16,)
    assert int(state.sum()) == 8
    assert int((state == 0).sum()) == 8


def test_balanced_state_reproducible_same_seed_and_step():
    a = build_balanced_pet_available(16, 7, 2023, torch.device('cpu'))
    b = build_balanced_pet_available(16, 7, 2023, torch.device('cpu'))
    assert torch.equal(a, b)


def test_balanced_state_varies_across_steps():
    a = build_balanced_pet_available(16, 0, 2023, torch.device('cpu'))
    b = build_balanced_pet_available(16, 1, 2023, torch.device('cpu'))
    assert not torch.equal(a, b)


def test_balanced_state_rejects_odd_or_small_batch():
    with pytest.raises(ValueError):
        build_balanced_pet_available(15, 0, 2023, torch.device('cpu'))
    with pytest.raises(ValueError):
        build_balanced_pet_available(1, 0, 2023, torch.device('cpu'))


def test_train_step_mixed_single_forward_and_auto_mode(monkeypatch):
    task = _make_task()
    calls = {'n': 0, 'modes': []}
    orig_forward = task.model.forward

    def wrapped(ct, pet=None, pet_available=None, target_size=None, forward_mode='auto'):
        calls['n'] += 1
        calls['modes'].append(forward_mode)
        return orig_forward(ct, pet=pet, pet_available=pet_available, target_size=target_size, forward_mode=forward_mode)

    monkeypatch.setattr(task.model, 'forward', wrapped)
    state = build_balanced_pet_available(16, 0, 2023, task.device)
    loss, logits, outputs, stats = task.train_step_mixed(_make_batch(16), pet_available=state)
    assert calls['n'] == 1
    assert calls['modes'] == ['auto']
    assert logits.shape == (16, 1, 64, 64)
    assert stats['num_full'] == 8 and stats['num_missing'] == 8
    assert math.isfinite(float(loss.detach()))


def test_mixed_loss_equals_half_full_plus_half_missing():
    task = _make_task()
    task.model.eval()
    batch = _make_batch(16)
    state = build_balanced_pet_available(16, 3, 2023, task.device)
    total, _, _, stats = task.train_step_mixed(batch, pet_available=state, missing_loss_weight=1.0)
    expected = 0.5 * stats['loss_full'] + 0.5 * stats['loss_missing']
    assert torch.allclose(stats['loss_total'], expected, atol=1e-6)


def test_mixed_loss_respects_missing_loss_weight():
    task = _make_task()
    task.model.eval()
    batch = _make_batch(16)
    state = build_balanced_pet_available(16, 3, 2023, task.device)
    _, _, _, stats = task.train_step_mixed(batch, pet_available=state, missing_loss_weight=3.0)
    assert stats['full_weight'] == pytest.approx(0.25)
    assert stats['missing_weight'] == pytest.approx(0.75)
    expected = 0.25 * stats['loss_full'] + 0.75 * stats['loss_missing']
    assert torch.allclose(stats['loss_total'], expected, atol=1e-6)


def test_mixed_gradient_equals_weighted_sum_of_subset_gradients():
    task = _make_task()
    batch = _make_batch(16)
    state = build_balanced_pet_available(16, 5, 2023, task.device)
    full_idx = state.eq(1)
    missing_idx = state.eq(0)

    params = [p for p in task.model.parameters() if p.requires_grad]

    task.model.zero_grad(set_to_none=True)
    loss, _, _, _ = task.train_step_mixed(batch, pet_available=state)
    loss.backward()
    g_mixed = [p.grad.detach().clone() for p in params]

    task.model.zero_grad(set_to_none=True)
    outputs = task.model(batch['ct'].to(task.device), pet=batch['pet'].to(task.device), pet_available=state.to(task.device), forward_mode='auto')
    logits = outputs['logits']
    mask = batch['mask'].to(task.device).float()
    full_loss, _ = task.criterion(logits[full_idx], mask[full_idx])
    full_loss.backward()
    g_full = [p.grad.detach().clone() for p in params]

    task.model.zero_grad(set_to_none=True)
    outputs = task.model(batch['ct'].to(task.device), pet=batch['pet'].to(task.device), pet_available=state.to(task.device), forward_mode='auto')
    logits = outputs['logits']
    missing_loss, _ = task.criterion(logits[missing_idx], mask[missing_idx])
    missing_loss.backward()
    g_missing = [p.grad.detach().clone() for p in params]

    for gm, gf, gmi in zip(g_mixed, g_full, g_missing):
        if gm is None:
            continue
        expected = 0.5 * gf + 0.5 * gmi
        assert torch.allclose(gm.float(), expected.float(), atol=5e-3, rtol=5e-2), f'gradient mismatch: max diff={(gm - expected).abs().max():.2e}'


def test_eval_mixed_rows_match_separate_full_and_missing():
    task = _make_task()
    task.model.eval()
    dev = task.device
    batch = {k: v.to(dev) for k, v in _make_batch(16).items()}
    state = build_balanced_pet_available(16, 9, 2023, dev)
    full_idx = state.eq(1)
    missing_idx = state.eq(0)
    with torch.no_grad():
        mixed = task.model(batch['ct'], pet=batch['pet'], pet_available=state, forward_mode='auto')['logits']
        full_only = task.model(batch['ct'][full_idx], pet=batch['pet'][full_idx], forward_mode='full')['logits']
        missing_only = task.model(batch['ct'][missing_idx], pet=batch['pet'][missing_idx], forward_mode='missing')['logits']
    assert torch.allclose(mixed[full_idx], full_only, atol=1e-5)
    assert torch.allclose(mixed[missing_idx], missing_only, atol=1e-5)


def test_missing_rows_logits_independent_of_pet_content():
    task = _make_task()
    task.model.eval()
    dev = task.device
    ct = torch.randn(16, 1, 64, 64).to(dev)
    state = build_balanced_pet_available(16, 11, 2023, dev)
    pet_a = torch.randn(16, 1, 64, 64).to(dev)
    pet_b = torch.randn(16, 1, 64, 64).to(dev)
    with torch.no_grad():
        logits_a = task.model(ct, pet=pet_a, pet_available=state, forward_mode='auto')['logits']
        logits_b = task.model(ct, pet=pet_b, pet_available=state, forward_mode='auto')['logits']
    missing_idx = state.eq(0)
    assert torch.allclose(logits_a[missing_idx], logits_b[missing_idx], atol=1e-5)


def test_full_rows_logits_depend_on_pet_content():
    task = _make_task()
    task.model.eval()
    dev = task.device
    ct = torch.randn(16, 1, 64, 64).to(dev)
    state = build_balanced_pet_available(16, 13, 2023, dev)
    pet_a = torch.randn(16, 1, 64, 64).to(dev)
    pet_b = torch.randn(16, 1, 64, 64).to(dev)
    with torch.no_grad():
        logits_a = task.model(ct, pet=pet_a, pet_available=state, forward_mode='auto')['logits']
        logits_b = task.model(ct, pet=pet_b, pet_available=state, forward_mode='auto')['logits']
    full_idx = state.eq(1)
    assert not torch.allclose(logits_a[full_idx], logits_b[full_idx], atol=1e-5)


def test_train_step_mixed_input_validation():
    task = _make_task()
    batch = _make_batch(16)
    with pytest.raises(ValueError):
        task.train_step_mixed(batch, pet_available=torch.ones(15, dtype=torch.long))
    with pytest.raises(ValueError):
        task.train_step_mixed(batch, pet_available=torch.full((16,), 2, dtype=torch.long))
    with pytest.raises(ValueError):
        task.train_step_mixed(batch, pet_available=torch.zeros(16, dtype=torch.long))
    with pytest.raises(ValueError):
        task.train_step_mixed(batch, pet_available=torch.ones(16, dtype=torch.long))


def test_mixed_loop_single_optimizer_and_scheduler_step(monkeypatch):
    task = _make_task()
    opt_calls = {'n': 0}
    sched_calls = {'n': 0}

    orig_opt_step = task.optimizer.step

    def opt_step(*args, **kwargs):
        opt_calls['n'] += 1
        return orig_opt_step(*args, **kwargs)

    sched = torch.optim.lr_scheduler.LambdaLR(task.optimizer, lr_lambda=lambda step: 1.0)
    task.scheduler = sched
    sched_step_orig = torch.optim.lr_scheduler.LambdaLR.step

    def counting_sched_step(self, *args, **kwargs):
        sched_calls['n'] += 1
        return sched_step_orig(self, *args, **kwargs)

    monkeypatch.setattr(task.optimizer, 'step', opt_step)
    monkeypatch.setattr(torch.optim.lr_scheduler.LambdaLR, 'step', counting_sched_step)

    state = build_balanced_pet_available(16, 0, 2023, task.device)
    for step in range(3):
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = task.train_step_mixed(_make_batch(16), pet_available=state)
        loss.backward()
        task.scaler.step(task.optimizer)
        task.scaler.update()
        task.scheduler.step()
        task.global_batch_step += 1

    assert opt_calls['n'] == 3
    assert sched_calls['n'] == 3
    assert task.global_batch_step == 3


def test_checkpoint_roundtrip_records_mode_and_step(tmp_path):
    task = _make_task()
    task.global_batch_step = 123
    path = str(tmp_path / 'ckpt.last.pth.tar')
    task.save_checkpoint(path, 2, best_joint=0.1, best_full=0.2, best_missing=0.3, best_joint_epoch=1, val_full={'dice': 0.2}, val_missing={'dice': 0.3}, joint_dice=0.25)
    ckpt = torch.load(path, map_location='cpu')
    assert ckpt['train_batch_mode'] == 'mixed'
    assert ckpt['global_batch_step'] == 123
    task.global_batch_step = 0
    torch.save(ckpt, path)
    ckpt2 = torch.load(path, map_location='cpu')
    task.global_batch_step = ckpt2['global_batch_step']
    assert task.global_batch_step == 123


def test_mixed_and_alternating_checkpoint_mode_recorded(tmp_path):
    cfg = _make_cfg(train_batch_mode='alternating')
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(use_deep_supervision=False)}, cfg)
    path = str(tmp_path / 'ckpt.alt.pth.tar')
    task.save_checkpoint(path, 1)
    ckpt = torch.load(path, map_location='cpu')
    assert ckpt['train_batch_mode'] == 'alternating'


def test_train_logger_header_order_and_strict_columns(tmp_path):
    from utils.train_logger import append_epoch_log, init_train_log
    log = str(tmp_path / 'train_log.csv')
    init_train_log(log, extra_headers=['train_batch_mode', 'train_mixed_loss'])
    append_epoch_log(
        log, 1, 0.5,
        {'total_loss': 0.4, 'dice': 0.3, 'iou': 0.2, 'acc': 0.9, 'acc_pixel': 0.8, 'hd95': 5.0},
        lr=1e-4, grad_norm=0.1,
        extra_metrics={'train_batch_mode': 'mixed', 'train_mixed_loss': 0.45},
    )
    with open(log, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['epoch', 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'val_acc', 'val_acc_pixel', 'val_hd95', 'lr', 'grad_norm', 'train_batch_mode', 'train_mixed_loss']
    assert rows[1][10] == 'mixed'

    with pytest.raises(ValueError):
        append_epoch_log(
            log, 2, 0.5,
            {'total_loss': 0.4, 'dice': 0.3, 'iou': 0.2, 'acc': 0.9, 'acc_pixel': 0.8, 'hd95': 5.0},
            lr=1e-4, grad_norm=0.1,
            extra_metrics={'train_mixed_loss': 0.45, 'undeclared_field': 1.0},
        )
    with pytest.raises(ValueError):
        append_epoch_log(
            log, 2, 0.5,
            {'total_loss': 0.4, 'dice': 0.3, 'iou': 0.2, 'acc': 0.9, 'acc_pixel': 0.8, 'hd95': 5.0},
            lr=1e-4, grad_norm=0.1,
            extra_metrics={'train_batch_mode': 'mixed'},
        )
