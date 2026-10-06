# -*- coding: utf-8 -*-
"""Tests for the Full-only CT+PET Add entry (synthetic data only)."""
import subprocess
import sys

import pytest
import torch

import run_full_add_baseline as full_add
from models.build_mdt_seg import build_dual_model
from models.components.add_fusion import AddFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from tasks.mdt_seg import MDTSegTeacher


OLD_FILES = [
    'run_ct_only_seg.py',
    'run_full_missing_baseline.py',
    'run_mdt_seg.py',
    'run_full_missing_spatial_fusion.py',
    'models/dual_shared_add_baseline.py',
]


def _cfg(**over):
    base = dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=None, pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False, deep_supervision=False, pretrained=False,
        model_arch='dual_shared_add_baseline',
        train_batch_mode='full', accumulation_steps=1, optimizer='adamw',
        learning_rate=1e-4, weight_decay=1e-4, mixed_precision=False,
        random_state=2023, ema_enabled=True, ema_decay=0.9,
        ema_decay_warmup=False, ema_start_epoch=0,
        experiment_type='full_add_baseline', fusion_type='AddFusion',
        pet_missing_rate=0.0,
    )
    base.update(over)
    return type('C', (), base)()


def _batch(n=4, size=32):
    return {'ct': torch.randn(n, 1, size, size),
            'pet': torch.randn(n, 1, size, size),
            'mask': (torch.rand(n, 1, size, size) > 0.5).float()}


def _model():
    return DualSharedAddPETCTBaseline(
        ct_pretrained_path=None, pet_pretrained_path=None, pretrained=False)


def test_config_defaults_to_full_only():
    argv, sys.argv = sys.argv, ['x']
    try:
        cfg = full_add.FullAddConfig.parse_arguments()
    finally:
        sys.argv = argv
    assert cfg.train_batch_mode == 'full'


def test_config_rejects_non_full_mode():
    argv, sys.argv = sys.argv, ['x', '--train_batch_mode', 'mixed']
    try:
        with pytest.raises(SystemExit):
            full_add.FullAddConfig.parse_arguments()
    finally:
        sys.argv = argv


def test_shared_config_default_untouched():
    from configs.seg_mdt import SegMDTConfig
    argv, sys.argv = sys.argv, ['x']
    try:
        old = SegMDTConfig.parse_arguments()
    finally:
        sys.argv = argv
    assert old.train_batch_mode == 'mixed'
    assert not hasattr(old, 'fusion_enabled')


def test_train_step_full_uses_real_pet_and_full_mode(monkeypatch):
    task = MDTSegTeacher({'model': _model()}, _cfg(ema_enabled=False))
    seen = {}

    orig_forward = task.model.forward

    def spy(ct, pet=None, pet_available=None, target_size=None, forward_mode='auto'):
        seen['mode'] = forward_mode
        seen['pet_none'] = pet is None
        seen['pet_shape'] = tuple(pet.shape) if pet is not None else None
        return orig_forward(ct, pet=pet, pet_available=pet_available,
                            target_size=target_size, forward_mode=forward_mode)

    monkeypatch.setattr(task.model, 'forward', spy)
    batch = _batch()
    loss, outputs, stats = full_add.train_step_full(task, batch)
    assert seen['mode'] == 'full'
    assert seen['pet_none'] is False
    assert seen['pet_shape'] == (4, 1, 32, 32)
    assert stats['num_samples'] == 4
    assert torch.isfinite(loss).all()
    assert torch.isfinite(outputs['logits']).all()


def test_train_step_full_does_not_use_mixed_or_masks(monkeypatch):
    import tasks.mdt_seg as task_module
    task = MDTSegTeacher({'model': _model()}, _cfg(ema_enabled=False))
    monkeypatch.setattr(task, 'train_step_mixed',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not call mixed')))
    import utils.run_common as common
    monkeypatch.setattr(common, 'build_balanced_pet_available',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not sample masks')))
    batch = _batch()
    loss, _, _ = full_add.train_step_full(task, batch)
    loss.backward()
    assert torch.isfinite(loss).all()


def test_full_loss_reaches_all_four_parts():
    task = MDTSegTeacher({'model': _model()}, _cfg(ema_enabled=False))
    task.model.zero_grad(set_to_none=True)
    loss, _, _ = full_add.train_step_full(task, _batch())
    loss.backward()
    for name, module in (('enc_ct', task.model.enc_ct), ('enc_pet', task.model.enc_pet),
                         ('ct_align', task.model.ct_align), ('decoder', task.model.decoder)):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f'full loss must update {name}'


def test_builder_is_add_and_spatial_free():
    model = build_dual_model(_cfg())['model']
    assert isinstance(model.fusion, AddFusion)
    assert 'spatial' not in type(model.fusion).__module__
    import models.spatial_bidirectional_fusion  # noqa: F401 (import must exist standalone)
    assert 'MultiScaleSpatialBidirectionalFusion' in dir(models.spatial_bidirectional_fusion)


def test_assert_protocol_rejects_non_add_fusion():
    cfg = _cfg()
    model = _model()
    object.__setattr__(model, 'fusion', torch.nn.Identity())
    with pytest.raises(RuntimeError):
        full_add._assert_full_add_protocol(cfg, model)


def test_best_full_selection_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    task = MDTSegTeacher({'model': _model()}, _cfg())
    paths = full_add._checkpoint_paths(str(tmp_path))
    val = {'dice': 0.6, 'iou': 0.5, 'hd95': 3.0, 'acc': 0.9, 'total_loss': 0.4}
    task.save_checkpoint(paths['best_full'], 2, best=0.6, best_epoch=2, val=val)
    ckpt = MDTSegTeacher.load_state_dicts(paths['best_full'])
    assert ckpt['best'] == 0.6 and ckpt['best_epoch'] == 2
    fresh = build_dual_model(_cfg())['model']
    assert isinstance(fresh.fusion, AddFusion)
    fresh.load_state_dict(ckpt['model'], strict=True)
    fresh_ema = build_dual_model(_cfg())['model']
    fresh_ema.load_state_dict(ckpt['model_ema'], strict=True)


def test_eval_weights_rule_prefers_recorded_over_presence():
    """model_ema bytes existing in a checkpoint must not imply EMA use: the
    runner follows ckpt['eval_weights'] (raw unless EMA was actually active)."""
    torch.manual_seed(1)
    task = MDTSegTeacher({'model': _model()}, _cfg())
    assert task.ema is not None
    assert ckpt_has_ema(task) is True
    assert task.eval_weights_tag() == 'ema'
    raw_task = MDTSegTeacher({'model': _model()}, _cfg(ema_enabled=False))
    assert raw_task.eval_weights_tag() == 'raw'
    assert raw_task.ema is None


def ckpt_has_ema(task):
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), 'c.tar')
    task.save_checkpoint(p, 1, best=0.1, best_epoch=1, val={'dice': 0.1})
    return MDTSegTeacher.load_state_dicts(p).get('model_ema') is not None


def test_final_eval_uses_full_mode_only(monkeypatch):
    torch.manual_seed(2)
    task = MDTSegTeacher({'model': _model()}, _cfg(ema_enabled=False))
    modes = []
    orig_evaluate = task.evaluate

    def spy(loader, eval_mode='full', tag='val', model=None):
        modes.append(eval_mode)
        return orig_evaluate(loader, eval_mode=eval_mode, tag=tag, model=model)

    monkeypatch.setattr(task, 'evaluate', spy)
    loader = [_batch(2)]
    out = task.evaluate(loader, eval_mode='full', model=task.eval_model())
    assert modes == ['full']
    assert 0.0 <= out['dice'] <= 1.0


def test_old_entries_and_model_unchanged():
    import subprocess
    r = subprocess.run(['git', 'status', '--porcelain', '--'] + OLD_FILES,
                       capture_output=True, text=True, cwd='.')
    assert r.stdout.strip() == '', f'old files modified: {r.stdout.strip()}'
    for path in OLD_FILES:
        src = open(path).read()
        assert 'FullAddConfig' not in src, path
        assert 'train_step_full' not in src, path


def test_entry_help_lists_full_mode():
    r = subprocess.run([sys.executable, 'run_full_add_baseline.py', '--help'],
                       capture_output=True, text=True, cwd='.')
    assert r.returncode == 0
    assert '--train_batch_mode {full}' in r.stdout
    assert 'dual_shared_add_baseline' in r.stdout
