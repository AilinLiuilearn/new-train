#!/usr/bin/env python3
"""Smoke tests for FGMS Stage2 integration."""

import copy
import sys
from types import SimpleNamespace

import torch

STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained-rerun-s3407/ckpt.best_joint.pth.tar'
)
SMOKE_SIZE = 288  # minimum input size so deepest FGMS feature map >= 9


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
        checkpoint_dir='/tmp/fgms_stage2_smoke',
        fgms_expert_dim=128,
        fgms_num_experts=6,
        fgms_top_k=2,
        fgms_enable_balance_loss=True,
        fgms_balance_loss_weight=0.1,
        fgms_residual_mode='zero_start',
        learning_rate=8e-5,
        decoder_lr=2e-5,
        weight_decay=1e-4,
        mixed_precision=False,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_model(cfg):
    from models.build_mdt_seg import build_mdt_seg_teacher
    from tasks.mdt_seg import MDTSegTeacher
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    return task


def test_stage1_loading(task):
    print('[TEST 1] Stage1 loading')
    stage1 = task.model.stage1
    assert not any(p.requires_grad for p in stage1.parameters())
    assert stage1.prototype_memory.bank_ready
    print('  PASS')


def test_decoder_copy(task):
    print('[TEST 2] Decoder copy')
    old_dec = task.model.stage1.decoder
    new_dec = task.model.stage2_decoder
    for old_p, new_p in zip(old_dec.parameters(), new_dec.parameters()):
        assert torch.equal(old_p.data, new_p.data)
        assert old_p.data_ptr() != new_p.data_ptr()
    print('  PASS')


def test_frozen_stage1(task):
    print('[TEST 3] Frozen Stage1')
    assert all(not p.requires_grad for p in task.model.stage1.parameters())
    print('  PASS')


def test_full_forward(task):
    print('[TEST 4] Full forward')
    model = task.model
    model.train()
    b, h, w = 2, SMOKE_SIZE, SMOKE_SIZE
    ct = torch.randn(b, 1, h, w, device=task.device)
    pet = torch.randn(b, 1, h, w, device=task.device)
    out = model(ct, pet=pet, forward_mode='full')
    logits = out['logits']
    assert logits.shape == (b, 1, h, w), f'got {logits.shape}'
    layout = model.stage2_moe.expert_layout()
    for scale in range(1, 5):
        imp = out['moe_stats'][f's{scale}_importance']
        assert imp[layout['proxy_pet']].abs().sum().item() == 0.0
    print(f'  output shape={tuple(logits.shape)}')
    print('  proxy importance=0 PASS')
    print('  PASS')


def test_missing_forward(task):
    print('[TEST 5] Missing forward (PET encoder must NOT run)')
    model = task.model
    model.train()
    pet_call_count = {'n': 0}

    def _hook(module, inp, out):
        pet_call_count['n'] += 1

    handle = model.stage1.enc_pet.register_forward_hook(_hook)
    try:
        b, h, w = 2, SMOKE_SIZE, SMOKE_SIZE
        ct = torch.randn(b, 1, h, w, device=task.device)
        pet = torch.randn(b, 1, h, w, device=task.device)
        out = model(ct, pet=pet, forward_mode='missing')
        logits = out['logits']
        assert logits.shape == (b, 1, h, w), f'got {logits.shape}'
        assert pet_call_count['n'] == 0, f'PET encoder called {pet_call_count["n"]} times'
        layout = model.stage2_moe.expert_layout()
        for scale in range(1, 5):
            imp = out['moe_stats'][f's{scale}_importance']
            assert imp[layout['real_pet']].abs().sum().item() == 0.0
        print(f'  PET encoder calls={pet_call_count["n"]}')
        print(f'  output shape={tuple(logits.shape)}')
        print('  real-pet importance=0 PASS')
        print('  PASS')
    finally:
        handle.remove()


def test_cppi_readonly(task):
    print('[TEST 6] CPPI readonly')
    from models.fgms_stage2_model import assert_cppi_unchanged
    model = task.model
    before = model.get_cppi_fingerprint()
    b, h, w = 1, SMOKE_SIZE, SMOKE_SIZE
    ct = torch.randn(b, 1, h, w, device=task.device)
    pet = torch.randn(b, 1, h, w, device=task.device)
    mask = torch.randint(0, 2, (b, 1, h, w), device=task.device).float()
    batch = {'ct': ct, 'pet': pet, 'mask': mask}
    task.optimizer.zero_grad(set_to_none=True)
    loss_full, _, _, _ = task.train_step(batch, forward_mode='full')
    loss_full.backward()
    task.optimizer.step()
    task.optimizer.zero_grad(set_to_none=True)
    loss_missing, _, _, _ = task.train_step(batch, forward_mode='missing')
    loss_missing.backward()
    task.optimizer.step()
    after = model.get_cppi_fingerprint()
    assert_cppi_unchanged(before, after)
    print('  CPPI unchanged PASS')
    print('  PASS')


def test_gradient(task):
    print('[TEST 7] Gradient flow')
    model = task.model
    model.configure_trainable_phase(2)
    model.train()
    b, h, w = 1, SMOKE_SIZE, SMOKE_SIZE
    ct = torch.randn(b, 1, h, w, device=task.device)
    pet = torch.randn(b, 1, h, w, device=task.device)
    mask = torch.randint(0, 2, (b, 1, h, w), device=task.device).float()
    batch = {'ct': ct, 'pet': pet, 'mask': mask}
    for step in range(3):
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, _, _ = task.train_step(batch, forward_mode='full')
        loss.backward()
        task.optimizer.step()
    stage1_grad = model.count_forbidden_stage1_nonzero_grads()
    moe_grad = sum(
        1 for p in model.stage2_moe.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    dec_grad = sum(
        1 for p in model.stage2_decoder.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert stage1_grad == 0, f'stage1 grad nonzero={stage1_grad}'
    assert dec_grad > 0, 'stage2_decoder has no gradient'
    print(f'  stage1_grad_nonzero={stage1_grad}')
    print(f'  stage2_moe params with grad={moe_grad}')
    print(f'  stage2_decoder params with grad={dec_grad}')
    print('  PASS')


def test_optimizer(task):
    print('[TEST 8] Optimizer param groups')
    names = []
    for group in task.optimizer.param_groups:
        for p in group['params']:
            for name, param in task.model.named_parameters():
                if param is p:
                    names.append(name)
                    break
    bad = [n for n in names if not (n.startswith('stage2_moe.') or n.startswith('stage2_decoder.'))]
    assert not bad, f'unexpected optimizer params in stage2 groups: {bad[:5]}'
    moe_lr = task.get_lr_by_name('stage2_moe')
    dec_lr = task.get_lr_by_name('stage2_decoder')
    boundary_lr = task.get_lr_by_name('stage1_boundary')
    print(f'  optimizer params count={len(names)}')
    print(f'  stage2 groups only PASS (boundary group exists separately)')
    print('  PASS')
    return moe_lr, dec_lr


def test_lr(task, moe_lr, dec_lr):
    print('[TEST 9] Learning rates')
    assert abs(moe_lr - 8e-5) < 1e-10, f'moe lr={moe_lr}'
    assert abs(dec_lr - 2e-5) < 1e-10, f'dec lr={dec_lr}'
    from utils.optimization import get_cosine_scheduler
    scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=15,
        warmup_steps=10,
        min_lr=1e-6,
        steps_per_epoch=100,
        flat_ratio=0.3,
    )
    for _ in range(10):
        scheduler.step()
    moe_lr2 = task.get_lr_by_name('stage2_moe')
    dec_lr2 = task.get_lr_by_name('stage2_decoder')
    ratio = moe_lr2 / (dec_lr2 + 1e-12)
    assert abs(ratio - 3.0) < 0.05, f'lr ratio={ratio}'
    print(f'  initial: moe={moe_lr:.8f} dec={dec_lr:.8f}')
    print(f'  after warmup: moe={moe_lr2:.8f} dec={dec_lr2:.8f} ratio={ratio:.2f}')
    task.scheduler = scheduler
    print('  PASS')


def test_9_experts(cfg):
    print('[TEST 10] 9 experts compatibility')
    cfg9 = make_config(fgms_num_experts=9)
    from models.build_mdt_seg import _build_fgms_stage2_model
    model = _build_fgms_stage2_model(cfg9)
    layout = model.stage2_moe.expert_layout()
    assert len(layout['ct']) == 3
    assert len(layout['real_pet']) == 3
    assert len(layout['proxy_pet']) == 3
    b, h, w = 1, SMOKE_SIZE, SMOKE_SIZE
    ct = torch.randn(b, 1, h, w)
    pet = torch.randn(b, 1, h, w)
    out = model(ct, pet=pet, forward_mode='full')
    assert out['logits'].shape == (b, 1, h, w)
    out = model(ct, pet=pet, forward_mode='missing')
    assert out['logits'].shape == (b, 1, h, w)
    print(f'  layout={layout}')
    print('  PASS')
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print('=' * 60)
    print('FGMS Stage2 Smoke Tests')
    print('=' * 60)
    cfg = make_config()
    task = build_model(cfg)
    test_stage1_loading(task)
    test_decoder_copy(task)
    test_frozen_stage1(task)
    test_full_forward(task)
    test_missing_forward(task)
    test_cppi_readonly(task)
    test_gradient(task)
    moe_lr, dec_lr = test_optimizer(task)
    test_lr(task, moe_lr, dec_lr)
    test_9_experts(cfg)
    print('=' * 60)
    print('ALL SMOKE TESTS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()
