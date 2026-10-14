"""Real ConvNeXtV2/MiT training and checkpoint integration (synthetic data)."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

import run_full_add_baseline as runner
from configs.seg_mdt import SegMDTConfig
from eval_full_contrinet_fusion import load_full_checkpoint
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_contrinet_fusion import (
    CONFIG_FIELDS, FullConTriNetPETCT, build_full_contrinet_model)
from run_full_contrinet_fusion import FullConTriNetConfig, main
from tasks.mdt_seg import MDTSegTeacher

ARMS = [(False, False, 'residual'), (True, False, 'residual'),
        (True, True, 'residual'), (True, True, 'replace')]


def config(monkeypatch, tmp_path, enabled=True, cross=True, mode='residual'):
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', f'mfm-{enabled}-{cross}-{mode}',
        '--mfm_enabled', str(enabled), '--mfm_cross_scale', str(cross),
        '--mfm_output_mode', mode, '--mixed_precision', 'false',
        '--hd95_mode', 'surface', '--batch_size', '2', '--epochs', '1',
        '--cosine_warmup', '0'])
    cfg = FullConTriNetConfig.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = cfg.model_arch = 'full_contrinet_fusion'
    cfg.fusion_type = 'ConTriNetMFMFusion'
    return cfg


def batch():
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1
    return dict(ct=torch.randn(2, 3, 64, 64),
                pet=torch.randn(2, 3, 64, 64), mask=mask)


def test_disabled_strict_baseline_state_and_exact_logits():
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    torch.manual_seed(2023)
    model = FullConTriNetPETCT(mfm_enabled=False, pretrained=False).eval()
    assert model.state_dict().keys() == base.state_dict().keys()
    for k, v in base.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    model.load_state_dict(base.state_dict(), strict=True)
    data = batch()
    with torch.no_grad():
        a = base(data['ct'], pet=data['pet'], forward_mode='full')['logits']
        b = model(data['ct'], pet=data['pet'])['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_enabled_preserves_shared_initialization_and_step_zero_logits():
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    torch.manual_seed(2023)
    model = FullConTriNetPETCT(pretrained=False).eval()
    for k, v in base.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    data = batch()
    with torch.no_grad():
        a = base(data['ct'], pet=data['pet'], forward_mode='full')['logits']
        b = model(data['ct'], pet=data['pet'])['logits']
        changed = model(data['ct'], pet=torch.zeros_like(data['pet']))['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert (b - changed).abs().max() > 0


@pytest.mark.parametrize('enabled,cross,mode', ARMS)
def test_actual_optimizer_gradients_and_checkpoint_roundtrip(
        monkeypatch, tmp_path, enabled, cross, mode):
    cfg = config(monkeypatch, tmp_path, enabled, cross, mode)
    task = MDTSegTeacher(build_full_contrinet_model(cfg), cfg)
    optimizer_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    assert all(id(p) in optimizer_ids for p in task.model.fusion.parameters())
    loss, out, _ = runner.train_step_full(task, batch())
    assert out['logits'].shape == (2, 1, 64, 64)
    loss.backward()
    for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
        grads = [p.grad for p in getattr(task.model, name).parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(float(g.abs().sum()) for g in grads) > 0, name
    task.optimizer.step()
    path = tmp_path / 'model.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    restored, saved, _, tag = load_full_checkpoint(str(path))
    assert (saved.mfm_enabled, saved.mfm_cross_scale, saved.mfm_output_mode) == (
        enabled, cross, mode)
    assert tag == 'raw'
    for k, v in task.model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    data = batch()
    task.model.eval()
    with torch.no_grad():
        a = task.model(data['ct'], pet=data['pet'])['logits']
        b = restored(data['ct'], pet=data['pet'])['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('enabled,cross,mode', ARMS)
def test_train_val_final_rebuild_uses_saved_arm(monkeypatch, tmp_path, enabled, cross, mode):
    cfg = config(monkeypatch, tmp_path, enabled, cross, mode)
    cfg.ema_enabled = True
    cfg.ema_start_epoch = 3  # EMA payload exists but is not selected yet.
    monkeypatch.setattr(FullConTriNetConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    data = batch()
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([data], [data], [data]))
    main()
    result = json.loads((Path(cfg.checkpoint_dir) / 'final_test_full.json').read_text())
    assert result['weights'] == 'raw' and result['best_epoch'] == 1
    assert all(result[name] == getattr(cfg, name) for name in CONFIG_FIELDS)
    assert result['experiment_type'] == result['model_arch'] == 'full_contrinet_fusion'
    with open(Path(cfg.checkpoint_dir) / 'train_log.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]['grad_fusion']) > 0 if enabled else float(rows[0]['grad_fusion']) == 0
    assert float(rows[0]['grad_pet_align']) > 0


def test_active_ema_loading_and_overrides(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    task = MDTSegTeacher(build_full_contrinet_model(cfg), cfg)
    with torch.no_grad():
        task.model.fusion.output_proj[0].weight.fill_(0.1)
    task.update_ema()
    path = tmp_path / 'ema.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    for override, tag in [(None, 'ema'), (True, 'ema'), (False, 'raw')]:
        loaded, _, _, actual = load_full_checkpoint(str(path), use_ema=override)
        assert actual == tag
        expected = task.ema.model if tag == 'ema' else task.model
        for k, v in expected.state_dict().items():
            assert torch.equal(v, loaded.state_dict()[k]), k


@pytest.mark.parametrize('mode,state,has_pet', [('missing', None, True),
    ('auto', [1, 0], True), ('full', None, False), ('full', [0, 1], True)])
def test_missing_rejected_before_encoder_calls(mode, state, has_pet):
    model = FullConTriNetPETCT(pretrained=False)
    model.enc_ct.register_forward_pre_hook(
        lambda *a: (_ for _ in ()).throw(AssertionError('encoder must not run')))
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError):
        model(x, pet=x if has_pet else None, pet_available=state, forward_mode=mode)


def test_config_rejects_mixed_and_existing_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['x', '--train_batch_mode', 'mixed'])
    with pytest.raises(SystemExit):
        FullConTriNetConfig.parse_arguments()
    cfg = config(monkeypatch, tmp_path)
    saved = Path(cfg.checkpoint_dir) / 'config_args.json'
    saved.write_text('{"existing": true}')
    with pytest.raises(FileExistsError):
        FullConTriNetConfig.parse_arguments()
    assert saved.read_text() == '{"existing": true}'


def test_builder_does_not_silently_supply_missing_checkpoint_flags():
    with pytest.raises(ValueError, match='Explicit saved'):
        build_full_contrinet_model(SegMDTConfig(args={'pretrained': False}))
