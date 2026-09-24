# -*- coding: utf-8 -*-
"""Tests for the optional weight-EMA used at evaluation time."""
import torch

from utils.ema import ModelEMA
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _cfg(**over):
    base = dict(learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
                loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
                ema_enabled=True, ema_decay=0.9, ema_warmup=False,
                mffa_enabled=False, mffa_checkpoint_attention=False)
    base.update(over)
    return type('C', (), base)()


def _batch(n=4, size=48):
    return {'ct': torch.randn(n, 1, size, size),
            'pet': torch.randn(n, 1, size, size),
            'mask': (torch.rand(n, 1, size, size) > 0.5).float()}


def test_ema_update_matches_formula():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 4)
    ema = ModelEMA(base, decay=0.9, warmup=False)
    before = {k: v.clone() for k, v in ema.model.state_dict().items()}
    with torch.no_grad():
        for p in base.parameters():
            p.add_(1.0)
    ema.update(base)
    for k, v in base.state_dict().items():
        assert torch.allclose(ema.model.state_dict()[k], 0.9 * before[k] + 0.1 * v, atol=1e-6)


def test_ema_warmup_ramps_decay():
    ema = ModelEMA(torch.nn.Linear(2, 2), decay=0.999, warmup=True)
    d1 = ema.update(ema.model)
    assert abs(d1 - 2.0 / 11.0) < 1e-6
    ema.updates = 99
    d2 = ema.update(ema.model)
    assert d2 > d1


def test_ema_disabled_teacher_has_no_ema():
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)},
                         _cfg(ema_enabled=False))
    assert task.ema is None
    assert task.eval_model() is task.model


def test_ema_enabled_teacher_uses_ema_for_eval():
    torch.manual_seed(0)
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)},
                         _cfg(ema_enabled=True, ema_decay=0.9, ema_warmup=False))
    assert task.ema is not None
    assert task.eval_model() is task.ema.model
    # Move the live weights, but the EMA should not follow until update_ema().
    ema_before = {k: v.clone() for k, v in task.ema.model.state_dict().items()}
    with torch.no_grad():
        for p in task.model.parameters():
            p.add_(0.5)
    for k, v in task.ema.model.state_dict().items():
        assert torch.allclose(v, ema_before[k], atol=1e-7)
    task.update_ema()
    changed = not all(torch.allclose(task.ema.model.state_dict()[k], ema_before[k], atol=1e-7)
                      for k in ema_before)
    assert changed


def test_ema_eval_runs_and_checkpoint_roundtrips():
    torch.manual_seed(0)
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)},
                         _cfg(ema_enabled=True))
    batch = _batch()
    state = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    task.optimizer.zero_grad(set_to_none=True)
    loss, _, _, _ = task.train_step_mixed(batch, pet_available=state)
    loss.backward()
    task.optimizer.step()
    task.update_ema()

    metrics = task.evaluate([batch], eval_mode='full', model=task.eval_model())
    assert 0.0 <= metrics['dice'] <= 1.0

    import os
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'ckpt.pth.tar')
    task.save_checkpoint(path, epoch=1)
    ckpt = torch.load(path, map_location='cpu')
    assert ckpt['model_ema'] is not None
    assert ckpt['ema_updates'] >= 1
    # EMA weights load strictly back into a fresh model.
    fresh = DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)
    msg = fresh.load_state_dict(ckpt['model_ema'], strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
