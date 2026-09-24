# -*- coding: utf-8 -*-
"""Integration tests for the Full MFFA fusion inside the real baseline model.

Covers the OFF switch (exact AddFusion baseline, extra params absent, legacy
encode-then-zero Missing behaviour) and the ON switch (Full/Missing/mixed
routing, module parameters inside AdamW, one optimizer + one scheduler step,
checkpoint strict reload, old config without the new fields).
"""
import math

import pytest
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.baseline_blocks import AddFusion
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from run_mdt_seg import build_balanced_pet_available


def _cfg(**kwargs):
    base = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': False,
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'train_batch_mode': 'mixed',
        'mffa_enabled': True,
        'mffa_checkpoint_attention': False,
    }
    base.update(kwargs)
    return type('C', (), base)()


def _batch(batch_size=8, size=64):
    return {
        'ct': torch.randn(batch_size, 1, size, size),
        'pet': torch.randn(batch_size, 1, size, size),
        'mask': (torch.rand(batch_size, 1, size, size) > 0.5).float(),
    }


# --------------------------------------------------------------------- OFF switch

def test_off_switch_uses_plain_addfusion_without_new_params():
    torch.manual_seed(0)
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False, mffa_enabled=False)
    assert isinstance(model.fusion, AddFusion)
    assert sum(p.numel() for p in model.fusion.parameters()) == 0
    assert not any(k.startswith('fusion.') for k in model.state_dict())


def test_off_switch_matches_manual_add_fusion():
    torch.manual_seed(0)
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False, mffa_enabled=False).eval()
    ct, pet = _batch(4)['ct'], _batch(4)['pet']
    with torch.no_grad():
        ct_feats = model._encode_ct(ct)
        pet_feats = model._encode_pet(pet)
        expected_full = model._decode([c + p for c, p in zip(ct_feats, pet_feats)], ct.shape[-2:])['logits']
        expected_missing = model._decode(list(ct_feats), ct.shape[-2:])['logits']
        got_full = model(ct, pet, forward_mode='full')['logits']
        got_missing = model(ct, pet, forward_mode='missing')['logits']
    assert torch.allclose(got_full, expected_full, atol=1e-6, rtol=1e-5)
    assert torch.allclose(got_missing, expected_missing, atol=1e-6, rtol=1e-5)


def test_off_switch_missing_is_encoder_then_zero():
    """Legacy behaviour: real PET is encoded, then zeroed (encode-then-zero)."""
    torch.manual_seed(0)
    model = DualSharedAddPETCTBaseline(use_deep_supervision=False, mffa_enabled=False).eval()
    ct = torch.randn(4, 1, 64, 64)
    pet_a = torch.randn(4, 1, 64, 64)
    pet_b = torch.randn(4, 1, 64, 64)
    calls = {'n': 0}
    orig = model._encode_pet

    def counting(p):
        calls['n'] += 1
        return orig(p)

    model._encode_pet = counting
    with torch.no_grad():
        out_a = model(ct, pet_a, forward_mode='missing')['logits']
        out_b = model(ct, pet_b, forward_mode='missing')['logits']
    assert calls['n'] == 2
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_old_config_without_mffa_fields_builds_addfusion():
    old_cfg = type('Old', (), {
        'ct_backbone': 'convnextv2_nano', 'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None, 'pet_pretrained_path': None,
        'decoder_channels': (512, 256, 128, 64), 'use_deep_supervision': False,
    })()
    assert not hasattr(old_cfg, 'mffa_enabled')
    model = build_mdt_seg_teacher(old_cfg)['model']
    assert isinstance(model.fusion, AddFusion)
    assert model.mffa_enabled is False


# ---------------------------------------------------------------------- ON switch

def _mffa_model():
    torch.manual_seed(0)
    return DualSharedAddPETCTBaseline(use_deep_supervision=False, mffa_enabled=True)


def test_on_switch_builds_mffa_and_keeps_base_keys():
    model = _mffa_model()
    assert type(model.fusion).__name__ == 'PETCTFullMFFA'
    assert model.mffa_enabled is True
    assert any(k.startswith('fusion.') for k in model.state_dict())


def test_mixed_missing_rows_equal_missing_mode_and_ct_decode():
    model = _mffa_model().eval()
    batch = _batch(8)
    state = build_balanced_pet_available(8, 0, 2023, torch.device('cpu'))
    full_idx, missing_idx = state.eq(1), state.eq(0)
    with torch.no_grad():
        ct_feats = model._encode_ct(batch['ct'])
        ct_decode = model._decode(list(ct_feats), batch['ct'].shape[-2:])['logits']
        mixed = model(batch['ct'], batch['pet'], pet_available=state, forward_mode='auto')['logits']
        missing_only = model(batch['ct'], batch['pet'], forward_mode='missing')['logits']
    assert torch.allclose(mixed[missing_idx], ct_decode[missing_idx], atol=1e-5, rtol=1e-4)
    assert torch.allclose(missing_only, ct_decode, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(mixed[full_idx], ct_decode[full_idx], atol=1e-6)


def test_full_rows_depend_on_pet_missing_rows_do_not():
    model = _mffa_model().eval()
    ct = torch.randn(8, 1, 64, 64)
    state = build_balanced_pet_available(8, 1, 2023, torch.device('cpu'))
    pet_a, pet_b = torch.randn(8, 1, 64, 64), torch.randn(8, 1, 64, 64)
    full_idx, missing_idx = state.eq(1), state.eq(0)
    with torch.no_grad():
        a = model(ct, pet_a, pet_available=state, forward_mode='auto')['logits']
        b = model(ct, pet_b, pet_available=state, forward_mode='auto')['logits']
    assert torch.allclose(a[missing_idx], b[missing_idx], atol=1e-6)
    assert not torch.allclose(a[full_idx], b[full_idx], atol=1e-6)


def test_mffa_params_registered_in_adamw():
    task = MDTSegTeacher({'model': _mffa_model()}, _cfg())
    optimizer_ids = {id(p) for group in task.optimizer.param_groups for p in group['params']}
    for p in task.model.fusion.parameters():
        assert id(p) in optimizer_ids


def test_train_step_mixed_gradients_and_single_step(monkeypatch):
    task = MDTSegTeacher({'model': _mffa_model()}, _cfg())
    task.scheduler = torch.optim.lr_scheduler.LambdaLR(task.optimizer, lr_lambda=lambda s: 1.0)
    opt_calls, sched_calls = {'n': 0}, {'n': 0}
    orig_opt, orig_sched = task.optimizer.step, torch.optim.lr_scheduler.LambdaLR.step

    def opt_step(*a, **k):
        opt_calls['n'] += 1
        return orig_opt(*a, **k)

    def sched_step(self, *a, **k):
        sched_calls['n'] += 1
        return orig_sched(self, *a, **k)

    monkeypatch.setattr(task.optimizer, 'step', opt_step)
    monkeypatch.setattr(torch.optim.lr_scheduler.LambdaLR, 'step', sched_step)

    batch = _batch(8)
    state = build_balanced_pet_available(8, 0, 2023, task.device)
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, stats = task.train_step_mixed(batch, pet_available=state)
    assert math.isfinite(float(loss.detach()))
    assert stats['num_full'] == stats['num_missing'] == 4
    loss.backward()
    task.optimizer.step()
    task.scheduler.step()

    assert opt_calls['n'] == 1 and sched_calls['n'] == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.fusion.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.enc_pet.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.enc_ct.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.model.decoder.parameters())


def test_checkpoint_strict_reload_restores_eval_outputs():
    torch.manual_seed(0)
    model = _mffa_model()
    batch = _batch(8)
    state = build_balanced_pet_available(8, 2, 2023, torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        before = model(batch['ct'], batch['pet'], pet_available=state, forward_mode='auto')['logits']
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    torch.manual_seed(1)
    fresh = DualSharedAddPETCTBaseline(use_deep_supervision=False, mffa_enabled=True)
    msg = fresh.load_state_dict(sd, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
    fresh.eval()
    with torch.no_grad():
        after = fresh(batch['ct'], batch['pet'], pet_available=state, forward_mode='auto')['logits']
    assert torch.allclose(before, after, atol=1e-6)
