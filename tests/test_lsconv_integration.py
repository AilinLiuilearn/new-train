"""Real ConvNeXtV2/MiT, shared optimizer, EMA, train/val/final and reload."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

import run_full_add_baseline as runner
from configs.seg_mdt import SegMDTConfig
from eval_full_lsconv_fusion import load_full_checkpoint
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_lsconv_fusion import FullLSConvPETCT, build_full_lsconv_model
from run_full_lsconv_fusion import FullLSConvConfig, main
from tasks.mdt_seg import MDTSegTeacher


def config(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', f'lsconv-{enabled}',
        '--lsconv_enabled', str(enabled), '--lsconv_backend', 'torch',
        '--mixed_precision', 'false', '--hd95_mode', 'surface',
        '--batch_size', '2', '--epochs', '1', '--cosine_warmup', '0'])
    cfg = FullLSConvConfig.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = cfg.model_arch = 'full_lsconv_fusion'
    cfg.fusion_type = 'LSConvFusion'
    return cfg


def batch(size=64):
    mask = torch.zeros(2, 1, size, size)
    mask[:, :, size // 3:2 * size // 3, size // 3:2 * size // 3] = 1
    return dict(ct=torch.randn(2, 3, size, size),
                pet=torch.randn(2, 3, size, size), mask=mask)


@pytest.mark.parametrize('enabled', [False, True])
def test_shared_initialization_and_exact_step_zero_logits(enabled):
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullLSConvPETCT(lsconv_enabled=enabled, lsconv_backend='torch',
                           pretrained=False).eval()
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for k, v in base.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    if not enabled:
        assert model.state_dict().keys() == base.state_dict().keys()
        model.load_state_dict(base.state_dict(), strict=True)
    data = batch()
    with torch.no_grad():
        a = base(data['ct'], pet=data['pet'], forward_mode='full')['logits']
        b = model(data['ct'], pet=data['pet'])['logits']
        changed = model(data['ct'], pet=torch.zeros_like(data['pet']))['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert (b - changed).abs().max() > 0


@pytest.mark.parametrize('enabled', [False, True])
def test_real_optimizer_two_steps_and_strict_checkpoint(monkeypatch, tmp_path, enabled):
    cfg = config(monkeypatch, tmp_path, enabled)
    task = MDTSegTeacher(build_full_lsconv_model(cfg), cfg)
    optimizer_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    assert all(id(p) in optimizer_ids for p in task.model.fusion.parameters())
    data = batch()
    for step in range(2):
        task.optimizer.zero_grad(set_to_none=True)
        loss, out, _ = runner.train_step_full(task, data)
        assert out['logits'].shape == (2, 1, 64, 64)
        loss.backward()
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
            grads = [p.grad for p in getattr(task.model, name).parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        if enabled and step == 1:
            for block in task.model.fusion.blocks:
                assert block.lkp.cv1.c.weight.grad.abs().sum() > 0
        task.optimizer.step()
    path = tmp_path / 'checkpoint.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    restored, saved, _, tag = load_full_checkpoint(str(path), device=task.device)
    assert saved.lsconv_enabled == enabled and saved.lsconv_backend == 'torch'
    assert tag == 'raw'
    for k, v in task.model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    task.model.eval()
    data = {name: value.to(task.device) for name, value in data.items()}
    with torch.no_grad():
        a = task.model(data['ct'], pet=data['pet'])['logits']
        b = restored(data['ct'], pet=data['pet'])['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('enabled', [False, True])
def test_one_epoch_train_val_final_and_saved_flags(monkeypatch, tmp_path, enabled):
    cfg = config(monkeypatch, tmp_path, enabled)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 3
    monkeypatch.setattr(FullLSConvConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    data = batch()
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([data], [data], [data]))
    main()
    result = json.loads((Path(cfg.checkpoint_dir) / 'final_test_full.json').read_text())
    assert result['weights'] == 'raw' and result['best_epoch'] == 1
    assert result['lsconv_enabled'] == enabled
    assert result['experiment_type'] == result['model_arch'] == 'full_lsconv_fusion'
    with open(Path(cfg.checkpoint_dir) / 'train_log.csv') as f:
        row = list(csv.DictReader(f))[0]
    assert (float(row['grad_fusion']) > 0) == enabled
    assert float(row['grad_pet_align']) > 0


def test_active_ema_and_runtime_backend_override(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    task = MDTSegTeacher(build_full_lsconv_model(cfg), cfg)
    with torch.no_grad():
        task.model.fusion.blocks[0].bn.weight.fill_(0.1)
    task.update_ema()
    path = tmp_path / 'ema.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    for override, tag in [(None, 'ema'), (True, 'ema'), (False, 'raw')]:
        loaded, _, _, actual = load_full_checkpoint(str(path), device=task.device,
                                                   use_ema=override, backend='torch')
        assert actual == tag
        expected = task.ema.model if tag == 'ema' else task.model
        for k, v in expected.state_dict().items():
            assert torch.equal(v, loaded.state_dict()[k]), k


@pytest.mark.parametrize('mode,state,has_pet', [('missing', None, True),
    ('auto', [1, 0], True), ('full', None, False), ('full', [0, 1], True)])
def test_missing_rejected_before_encoding(mode, state, has_pet):
    model = FullLSConvPETCT(pretrained=False)
    model.enc_ct.register_forward_pre_hook(
        lambda *a: (_ for _ in ()).throw(AssertionError('encoder must not run')))
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError):
        model(x, pet=x if has_pet else None, pet_available=state, forward_mode=mode)


def test_reject_mixed_config_and_existing_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['x', '--train_batch_mode', 'mixed'])
    with pytest.raises(SystemExit):
        FullLSConvConfig.parse_arguments()
    cfg = config(monkeypatch, tmp_path)
    saved = Path(cfg.checkpoint_dir) / 'config_args.json'
    saved.write_text('{"existing":true}')
    with pytest.raises(FileExistsError):
        FullLSConvConfig.parse_arguments()
    assert saved.read_text() == '{"existing":true}'


def test_reject_unrecorded_architecture():
    with pytest.raises(ValueError, match='Explicit saved'):
        build_full_lsconv_model(SegMDTConfig(args={'pretrained': False}))
