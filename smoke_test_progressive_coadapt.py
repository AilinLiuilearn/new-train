#!/usr/bin/env python3
"""Smoke tests for Progressive Co-Adaptation on FGMS Stage2."""

import csv
import tempfile
from types import SimpleNamespace

import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.train_logger import append_epoch_log, init_train_log

STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained-rerun-s3407/ckpt.best_joint.pth.tar'
)
SMOKE_SIZE = 288


def make_config(**overrides):
    cfg = SimpleNamespace(
        model_arch='dual_shared_add_fgms_stage2',
        stage1_checkpoint=STAGE1_CKPT,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano',
        pet_pretrained_path='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1',
        no_encoder_pretrained=False,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        deep_supervision=False,
        cppi_num_clusters=6,
        cppi_build_stage=4,
        checkpoint_dir='/tmp/progressive_smoke',
        fgms_expert_dim=128,
        fgms_num_experts=6,
        fgms_top_k=2,
        fgms_enable_balance_loss=True,
        fgms_balance_loss_weight=0.1,
        fgms_residual_mode='zero_start',
        fgms_progressive_coadapt=True,
        fgms_decoder_unfreeze_epoch=2,
        fgms_boundary_unfreeze_epoch=4,
        stage1_boundary_lr=5e-6,
        learning_rate=8e-5,
        decoder_lr=2e-5,
        weight_decay=1e-4,
        mixed_precision=True,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_task(cfg):
    return MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)


def _batch(task, b=1):
    h = w = SMOKE_SIZE
    device = task.device
    return {
        'ct': torch.randn(b, 1, h, w, device=device),
        'pet': torch.randn(b, 1, h, w, device=device),
        'mask': torch.randint(0, 2, (b, 1, h, w), device=device).float(),
    }


def _trainable_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _backward_step(task, batch, route):
    task.optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        loss, _, _, _ = task.train_step(batch, forward_mode=route)
    loss.backward()
    return loss


def _grad_count(params):
    return sum(1 for p in params if p.grad is not None and p.grad.abs().sum().item() > 0)


def test_epoch1(task):
    print('[TEST A] Epoch1 moe_warmup')
    task.model.configure_trainable_phase(1)
    names = _trainable_names(task.model)
    assert all(n.startswith('stage2_moe.') for n in names), names[:5]
    batch = _batch(task)
    _backward_step(task, batch, 'full')
    assert _grad_count(task.model.stage2_moe.parameters()) > 0
    assert _grad_count(task.model.stage2_decoder.parameters()) == 0
    assert task.model.count_boundary_nonzero_grads() == 0
    assert task.model.count_forbidden_stage1_nonzero_grads() == 0
    print('  PASS')


def test_epoch2(task):
    print('[TEST B] Epoch2 stage2_adapt')
    task.model.configure_trainable_phase(2)
    names = _trainable_names(task.model)
    assert any(n.startswith('stage2_moe.') for n in names)
    assert any(n.startswith('stage2_decoder.') for n in names)
    batch = _batch(task)
    _backward_step(task, batch, 'full')
    assert _grad_count(task.model.stage2_moe.parameters()) > 0
    assert _grad_count(task.model.stage2_decoder.parameters()) > 0
    assert task.model.count_boundary_nonzero_grads() == 0
    assert task.model.count_forbidden_stage1_nonzero_grads() == 0
    print('  PASS')


def test_epoch4(task):
    print('[TEST C] Epoch4 boundary_coadapt')
    task.model.configure_trainable_phase(4)
    batch = _batch(task)
    _backward_step(task, batch, 'full')
    assert _grad_count(task.model.stage1.pet_calibration.parameters()) > 0
    assert _grad_count(task.model.stage1.fusion.parameters()) > 0
    assert task.model.stage1.fusion.raw_alpha_full.grad is not None
    task.optimizer.zero_grad(set_to_none=True)
    _backward_step(task, batch, 'missing')
    assert _grad_count(task.model.stage1.pet_calibration.parameters()) > 0
    assert task.model.stage1.fusion.raw_alpha_missing.grad is not None
    assert task.model.count_forbidden_stage1_nonzero_grads() == 0
  # requires_grad boundary on outputs
    ct = batch['ct']
    pet = batch['pet']
    ct_feats, pet_cal, fbase = task.model._extract_full_features(ct, pet)
    assert not ct_feats[0].requires_grad
    assert pet_cal[0].requires_grad
    assert fbase[0].requires_grad
    print('  PASS')


def test_missing_pet_encoder(task):
    print('[TEST D] Missing PET encoder call count = 0')
    count = {'n': 0}
    handle = task.model.stage1.enc_pet.register_forward_hook(lambda *args, **kwargs: count.__setitem__('n', count['n'] + 1))
    try:
        for epoch in (1, 2, 4):
            task.model.configure_trainable_phase(epoch)
            batch = _batch(task)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                task.model(batch['ct'], pet=batch['pet'], forward_mode='missing')
            assert count['n'] == 0, f'epoch={epoch} pet encoder calls={count["n"]}'
    finally:
        handle.remove()
    print('  PASS')


def test_cppi_readonly(task):
    print('[TEST E] CPPI readonly in Phase C')
    from models.fgms_stage2_model import assert_cppi_unchanged
    task.model.configure_trainable_phase(4)
    before = task.model.get_cppi_fingerprint()
    batch = _batch(task)
    for route in ('full', 'missing'):
        task.optimizer.zero_grad(set_to_none=True)
        _backward_step(task, batch, route)
        task.optimizer.step()
    after = task.model.get_cppi_fingerprint()
    assert_cppi_unchanged(before, after)
    print('  PASS')


def test_optimizer_groups(task):
    print('[TEST] Optimizer groups')
    names = {g.get('name'): g['lr'] for g in task.optimizer.param_groups}
    assert names['stage2_moe'] == 8e-5
    assert names['stage2_decoder'] == 2e-5
    assert names['stage1_boundary'] == 5e-6
    print('  PASS')


def test_csv_logger():
    print('[TEST] CSV header/row alignment')
    from run_mdt_seg import _stage2_extra_headers
    headers = ['epoch', 'train_loss'] + _stage2_extra_headers()[:5]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = f'{tmp}/train_log.csv'
        init_train_log(log_path, extra_headers=headers[2:])
        extra = {h: 1.23 for h in headers[2:]}
        append_epoch_log(
            log_path, 1, 0.5,
            {'total_loss': 0.4, 'dice': 0.7, 'iou': 0.6, 'acc': 0.8, 'acc_pixel': 0.8, 'hd95': 1.0},
            lr=8e-5, grad_norm=0.1, extra_metrics=extra,
        )
        with open(log_path, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        assert len(rows[0]) == len(rows[1])
    print('  PASS')


def main():
    print('=' * 60)
    print('Progressive Co-Adaptation Smoke Tests')
    print('=' * 60)
    cfg = make_config()
    task = build_task(cfg)
    test_optimizer_groups(task)
    test_epoch1(task)
    test_epoch2(task)
    test_epoch4(task)
    test_missing_pet_encoder(task)
    test_cppi_readonly(task)
    test_csv_logger()
    print('=' * 60)
    print('ALL PROGRESSIVE SMOKE TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()
