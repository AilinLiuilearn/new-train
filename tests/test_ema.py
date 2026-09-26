# -*- coding: utf-8 -*-
"""Tests for the optional weight-EMA used at evaluation time."""
import torch

from utils.ema import ModelEMA
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


def _cfg(**over):
    base = dict(learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
                loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
                ema_enabled=True, ema_decay=0.9, ema_decay_warmup=False, ema_start_epoch=0)
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


def test_ema_start_epoch_delays_activation():
    torch.manual_seed(0)
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)},
                         _cfg(ema_enabled=True, ema_start_epoch=3))
    assert task.ema is not None
    assert task.ema_active is False
    # Warmup epochs: eval uses the raw model and update_ema is a no-op.
    for ep in (1, 2, 3):
        task.begin_epoch(ep)
        assert task.ema_active is False
        assert task.eval_model() is task.model
        assert task.update_ema() is None
    assert task.ema.updates == 0
    # The EMA copy must still hold the *initial* weights during warmup.
    # First active epoch hard-syncs, then tracks.
    task.begin_epoch(4)
    assert task.ema_active is True
    assert task.eval_model() is task.ema.model
    with torch.no_grad():
        for p in task.model.parameters():
            p.add_(0.3)
    task.begin_epoch(4)  # idempotent once active
    task.update_ema()
    assert task.ema.updates == 1


def test_ema_warmup_epochs_do_not_pollute_ema():
    """After the delay the EMA is hard-synced, so it never blends the stale
    pre-warmup initialization into the average."""
    torch.manual_seed(0)
    task = MDTSegTeacher({'model': DualSharedAddPETCTBaseline(ct_pretrained_path=None, pet_pretrained_path=None)},
                         _cfg(ema_enabled=True, ema_decay=0.9, ema_decay_warmup=False, ema_start_epoch=2))
    # Move the live weights a lot during warmup (EMA must ignore this).
    with torch.no_grad():
        for p in task.model.parameters():
            p.add_(5.0)
    task.begin_epoch(3)
    assert task.ema_active is True
    task.begin_epoch(3)
    # reset() hard-copied the *current* live weights into the EMA.
    for k, v in task.model.state_dict().items():
        assert torch.allclose(task.ema.model.state_dict()[k], v, atol=1e-6)


def _asym_model(**over):
    kw = dict(ct_pretrained_path=None, pet_pretrained_path=None,
              asym_fusion_enabled=True, asym_use_text=True,
              asym_clip_path='pretrained/clip-vit-base-patch32')
    kw.update(over)
    return DualSharedAddPETCTBaseline(**kw)


def test_ema_asym_text_cache_exact_and_extra_state_contract():
    """Two EMA updates: frozen CLIP text cache copies exactly, the _extra_state
    contract stays consistent, and float weights follow the decay formula."""
    torch.manual_seed(0)
    model = _asym_model()
    task = MDTSegTeacher({'model': model},
                         _cfg(ema_enabled=True, ema_decay=0.9, ema_decay_warmup=False,
                              ema_start_epoch=0))
    assert task.ema is not None
    src = task.model
    ema_before = {k: (v.clone() if torch.is_tensor(v) else v)
                  for k, v in task.ema.model.state_dict().items()}
    with torch.no_grad():
        for p in src.parameters():
            if p.is_floating_point():
                p.add_(0.25)
    task.update_ema()
    task.update_ema()
    ema_state = task.ema.model.state_dict()
    src_state = src.state_dict()
    # 1. Frozen text cache is bit-exact after updates.
    assert torch.equal(ema_state['fusion.text_embeddings'], src_state['fusion.text_embeddings'])
    # 2. Extra-state contract consistent between source and EMA copy.
    assert src.fusion.get_extra_state() == task.ema.model.fusion.get_extra_state()
    # 3. Float weights follow theta = d^2*before + (1-d^2)*src for two updates.
    d2 = 0.9 ** 2
    checked = 0
    for k, v in src_state.items():
        if torch.is_tensor(v) and v.is_floating_point() and k != 'fusion.text_embeddings':
            expected = d2 * ema_before[k] + (1.0 - d2) * v
            assert torch.allclose(ema_state[k], expected, atol=1e-5), k
            checked += 1
    assert checked > 0


def test_ema_params_frozen_and_outside_optimizer():
    torch.manual_seed(0)
    task = MDTSegTeacher({'model': _asym_model()},
                         _cfg(ema_enabled=True, ema_decay_warmup=False))
    for p in task.ema.model.parameters():
        assert p.requires_grad is False
    ema_ids = {id(p) for p in task.ema.model.parameters()}
    for group in task.optimizer.param_groups:
        for p in group['params']:
            assert id(p) not in ema_ids


def test_optimizer_step_helper_success_and_overflow_skip():
    from run_mdt_seg import _optimizer_step_succeeded

    class _Scaler:
        def __init__(self, enabled, scales):
            self._enabled = enabled
            self._scales = list(scales)
            self.step_calls = 0

        def is_enabled(self):
            return self._enabled

        def get_scale(self):
            return self._scales[0]

        def step(self, optimizer):
            self.step_calls += 1

        def update(self):
            self._scales.pop(0)

    class _Task:
        def __init__(self, scaler):
            self.scaler = scaler
            self.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)

    # Non-AMP path always succeeds and steps the optimizer.
    t = _Task(_Scaler(False, [1.0]))
    assert _optimizer_step_succeeded(t) is True
    # AMP path with stable scale succeeds.
    t = _Task(_Scaler(True, [65536.0, 65536.0]))
    assert _optimizer_step_succeeded(t) is True
    assert t.scaler.step_calls == 1
    # AMP overflow (scale drops after update) reports a skip.
    t = _Task(_Scaler(True, [65536.0, 32768.0]))
    assert _optimizer_step_succeeded(t) is False
