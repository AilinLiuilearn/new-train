"""Actual backbones and Full training/checkpoint/EMA lifecycle with official CMX."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

import run_full_cmx_fusion as runner
from configs.full_cmx import FullCMXConfig, prepare_run_directory
from eval_full_cmx_fusion import load_full_checkpoint
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_cmx_fusion import FullCMXPETCT, build_full_cmx_model
from tasks.mdt_seg import MDTSegTeacher

ARMS = {'add': (False, False), 'ffm': (True, False),
        'frm_add': (False, True), 'frm_ffm': (True, True)}


def config(monkeypatch, tmp_path, arm='ffm'):
    ffm, frm = ARMS[arm]
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false', '--require_cuda', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', 'cmx-' + arm,
        '--cmx_ffm_enabled', str(ffm), '--cmx_frm_enabled', str(frm),
        '--mixed_precision', 'false', '--hd95_mode', 'surface', '--batch_size', '2',
        '--epochs', '1', '--cosine_warmup', '0'])
    cfg = FullCMXConfig.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = cfg.model_arch = 'full_cmx_fusion'
    cfg.fusion_type = 'CMXFusion'
    return cfg


def batch(size=64):
    mask = torch.zeros(2, 1, size, size)
    mask[:, :, size // 3:size // 2, size // 3:size // 2] = 1
    return dict(ct=torch.randn(2, 3, size, size), pet=torch.randn(2, 3, size, size), mask=mask)


@pytest.mark.parametrize('enabled', [False, True])
def test_shared_baseline_init_rng_and_disabled_exact_equivalence(enabled):
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullCMXPETCT(pretrained=False, cmx_ffm_enabled=enabled).eval()
    assert torch.equal(expected_rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    if not enabled:
        model.load_state_dict(base.state_dict(), strict=True)
        assert base.state_dict().keys() == model.state_dict().keys()
    data = batch()
    with torch.no_grad():
        a = base(data['ct'], pet=data['pet'], forward_mode='full')['logits']
        b = model(data['ct'], pet=data['pet'])['logits']
    if enabled:
        assert not torch.equal(a, b)
    else:
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('arm', list(ARMS))
def test_two_steps_main_loss_only_and_all_enabled_modules_learn(monkeypatch, tmp_path, arm):
    cfg = config(monkeypatch, tmp_path, arm)
    task = MDTSegTeacher(build_full_cmx_model(cfg), cfg)
    ids = {id(p) for group in task.optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in task.model.parameters())
    data = batch()
    for _ in range(2):
        task.optimizer.zero_grad(set_to_none=True)
        loss, out, stats = runner.train_step_full(task, data)
        main, _ = task.criterion(out['logits'], data['mask'])
        torch.testing.assert_close(loss, main, atol=0, rtol=0)
        assert not out['aux'] and stats['num_samples'] == 2
        loss.backward()
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
            grads = [p.grad for p in getattr(task.model, name).parameters()]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        for blocks in (task.model.fusion.ffms, task.model.fusion.frms):
            for block in blocks:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
                assert sum(float(p.grad.abs().sum()) for p in block.parameters()) > 0
        task.optimizer.step()


@pytest.mark.parametrize('arm', ['add', 'ffm'])
def test_train_val_ema_checkpoint_final_evaluation(monkeypatch, tmp_path, arm):
    cfg = config(monkeypatch, tmp_path, arm)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    monkeypatch.setattr(FullCMXConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    data = batch()
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([data], [data], [data]))
    runner.main()
    root = Path(cfg.checkpoint_dir)
    result = json.loads((root / 'final_test_full.json').read_text())
    assert result['weights'] == 'ema' and result['best_epoch'] == 1
    assert result['cmx_ffm_enabled'] == cfg.cmx_ffm_enabled
    assert result['cmx_frm_enabled'] == cfg.cmx_frm_enabled
    assert result['experiment_type'] == result['model_arch'] == 'full_cmx_fusion'
    with (root / 'train_log.csv').open() as f:
        row = list(csv.DictReader(f))[0]
    assert (float(row['grad_fusion']) > 0) == cfg.cmx_ffm_enabled
    assert float(row['grad_pet_align']) > 0
    loaded, saved, ckpt, tag = load_full_checkpoint(str(root / 'ckpt.best_full.pth.tar'))
    assert tag == 'ema' and saved.cmx_ffm_enabled == cfg.cmx_ffm_enabled
    assert loaded.fusion.ffm_enabled == cfg.cmx_ffm_enabled
    assert saved.cmx_stage_heads == [1, 2, 5, 8]
    for key, value in ckpt['model_ema'].items():
        assert torch.equal(value, loaded.state_dict()[key])


@pytest.mark.parametrize('arm', ['frm_add', 'frm_ffm'])
def test_optional_frm_strict_raw_ema_reload(monkeypatch, tmp_path, arm):
    cfg = config(monkeypatch, tmp_path, arm)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    task = MDTSegTeacher(build_full_cmx_model(cfg), cfg)
    with torch.no_grad():
        task.model.fusion.frms[0].spatial_weights.mlp[0].weight.add_(.01)
    task.update_ema()
    path = tmp_path / 'both.pth.tar'
    task.save_checkpoint(str(path), 1, .5, 1, {'dice': .5})
    for use_ema in (None, False):
        loaded, saved, ckpt, tag = load_full_checkpoint(str(path), use_ema=use_ema)
        assert saved.cmx_frm_enabled is True and saved.cmx_ffm_enabled == cfg.cmx_ffm_enabled
        key = 'model_ema' if tag == 'ema' else 'model'
        for name, value in ckpt[key].items():
            assert torch.equal(value, loaded.state_dict()[name])


@pytest.mark.parametrize('mode,state,has_pet', [('missing', None, True),
    ('auto', [1, 0], True), ('full', [0, 1], True), ('full', None, False)])
def test_missing_rejected_before_any_encoder(mode, state, has_pet):
    model = FullCMXPETCT(pretrained=False)
    for encoder in (model.enc_ct, model.enc_pet):
        encoder.register_forward_pre_hook(lambda *a: (_ for _ in ()).throw(AssertionError('encoder must not run')))
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError):
        model(x, pet=x if has_pet else None, forward_mode=mode, pet_available=state)


def test_no_cuda_and_invalid_modes_do_not_create_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['x', '--checkpoint_root', str(tmp_path), '--hash', 'blocked'])
    with pytest.raises(RuntimeError, match='No experiment files'): FullCMXConfig.parse_arguments()
    assert not (tmp_path / 'MDT').exists()
    monkeypatch.setattr(sys, 'argv', ['x', '--train_batch_mode', 'mixed'])
    with pytest.raises(SystemExit): FullCMXConfig.parse_arguments()
    monkeypatch.setattr(sys, 'argv', ['x', '--require_cuda', 'false', '--deep_supervision', 'true'])
    with pytest.raises(ValueError): FullCMXConfig.parse_arguments()


def test_safe_retry_and_checkpoint_protection(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path)
    folder = Path(cfg.checkpoint_dir)
    (folder / 'config_args.json').write_text(json.dumps(vars(cfg)))
    (folder / 'train_log.csv').write_text('epoch,train_loss\n')
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    cfg.retry_failed_start = True
    prepare_run_directory(cfg)
    backup, = list(folder.parent.glob(folder.name + '.failed-start-*'))
    assert before == {p.name: p.read_bytes() for p in backup.iterdir()}
    folder.mkdir()
    (folder / 'config_args.json').write_text(json.dumps(vars(cfg)))
    (folder / 'ckpt.last.pth.tar').write_text('preserve')
    with pytest.raises(FileExistsError): prepare_run_directory(cfg)
    assert (folder / 'ckpt.last.pth.tar').read_text() == 'preserve'


def test_checkpoint_rejects_wrong_architecture_metadata(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path)
    task = MDTSegTeacher(build_full_cmx_model(cfg), cfg)
    path = tmp_path / 'wrong.pth.tar'
    task.save_checkpoint(str(path), 1, .5, 1, {'dice': .5})
    ckpt = MDTSegTeacher.load_state_dicts(str(path))
    ckpt['config']['cmx_stage_heads'] = [1, 2, 4, 8]
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='architecture/source'): load_full_checkpoint(str(path))
