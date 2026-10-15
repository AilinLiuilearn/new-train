"""Real backbones, switches, optimizer, checkpoint/EMA and full training lifecycle."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

import run_full_lesion_guided_fusion as runner
from configs.full_lesion_guided import FullLesionGuidedConfig
from eval_full_lesion_guided_fusion import load_full_checkpoint
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_lesion_guided_fusion import FullLesionGuidedPETCT, build_full_lesion_guided_model
from tasks.mdt_seg import MDTSegTeacher

ARMS = {
    'add': (False, False, False, False), 'aux': (True, False, False, True),
    'residual': (True, True, False, False), 'full': (True, True, True, True),
    'no_aux': (True, True, True, False), 'aux_no_guidance': (True, True, False, True),
}


def config(monkeypatch, tmp_path, arm='full'):
    flags = list(zip(('fusion_enabled', 'correction_enabled', 'guidance_enabled', 'aux_loss_enabled'), ARMS[arm]))
    argv = ['x', '--pretrained', 'false', '--require_cuda', 'false',
            '--checkpoint_root', str(tmp_path), '--hash', 'lesion-' + arm,
            '--mixed_precision', 'false', '--hd95_mode', 'surface',
            '--batch_size', '2', '--epochs', '1', '--cosine_warmup', '0']
    for name, enabled in flags:
        argv.extend(['--lesion_' + name, str(enabled)])
    monkeypatch.setattr(sys, 'argv', argv)
    cfg = FullLesionGuidedConfig.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = cfg.model_arch = 'full_lesion_guided_fusion'
    cfg.fusion_type = 'LesionGuidedFusion'
    return cfg


def batch(size=64):
    mask = torch.zeros(2, 1, size, size)
    mask[:, :, size // 3:size // 2, size // 3:size // 2] = 1
    return dict(ct=torch.randn(2, 3, size, size), pet=torch.randn(2, 3, size, size), mask=mask)


@pytest.mark.parametrize('enabled', [False, True])
def test_initialization_rng_and_complete_logits_equal_add(enabled):
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullLesionGuidedPETCT(pretrained=False, lesion_fusion_enabled=enabled).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in base.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k
    if not enabled:
        model.load_state_dict(base.state_dict(), strict=True)
        assert base.state_dict().keys() == model.state_dict().keys()
    data = batch()
    with torch.no_grad():
        expected = base(data['ct'], pet=data['pet'], forward_mode='full')['logits']
        result = model(data['ct'], pet=data['pet'])['logits']
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize('arm', list(ARMS))
def test_loss_switch_gradients_and_two_optimizer_steps(monkeypatch, tmp_path, arm):
    cfg = config(monkeypatch, tmp_path, arm)
    task = MDTSegTeacher(build_full_lesion_guided_model(cfg), cfg)
    data = batch()
    optimizer_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    assert all(id(p) in optimizer_ids for p in task.model.parameters())
    for step in range(2):
        task.optimizer.zero_grad(set_to_none=True)
        loss, out, stats = runner.train_step_full(task, data)
        main, _ = task.criterion(out['logits'], data['mask'])
        torch.testing.assert_close(loss, main + stats['aux_weighted'])
        assert bool(stats['aux_weighted'] > 0) == cfg.lesion_aux_loss_enabled
        if not cfg.lesion_aux_loss_enabled:
            torch.testing.assert_close(loss, main, atol=0, rtol=0)
            assert not out['aux']
        loss.backward()
        for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
            grads = [p.grad for p in getattr(task.model, name).parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            assert sum(float(g.abs().sum()) for g in grads) > 0, name
        if cfg.lesion_fusion_enabled:
            for block in task.model.fusion.blocks:
                if cfg.lesion_aux_loss_enabled:
                    assert block.mask_head[-1].weight.grad.abs().sum() > 0
                elif not cfg.lesion_guidance_enabled:
                    assert block.mask_head[-1].weight.grad is None
                if cfg.lesion_correction_enabled and step == 1:
                    assert block.gate[-1].weight.grad.abs().sum() > 0
                if not cfg.lesion_correction_enabled:
                    assert block.out_ct.weight.grad is None
        task.optimizer.step()


@pytest.mark.parametrize('arm', ['add', 'aux', 'full', 'no_aux'])
def test_runner_train_validation_final_and_checkpoint_flags(monkeypatch, tmp_path, arm):
    cfg = config(monkeypatch, tmp_path, arm)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    monkeypatch.setattr(FullLesionGuidedConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    data = batch()
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([data], [data], [data]))
    runner.main()
    root = Path(cfg.checkpoint_dir)
    result = json.loads((root / 'final_test_full.json').read_text())
    assert result['weights'] == 'ema' and result['best_epoch'] == 1
    assert result['lesion_aux_loss_enabled'] == cfg.lesion_aux_loss_enabled
    assert result['model_arch'] == result['experiment_type'] == 'full_lesion_guided_fusion'
    with (root / 'train_log.csv').open() as f:
        row = list(csv.DictReader(f))[0]
    assert bool(float(row['train_aux_weighted']) > 0) == cfg.lesion_aux_loss_enabled
    assert float(row['train_loss_main']) > 0
    assert float(row['grad_pet_align']) > 0
    loaded, saved, ckpt, tag = load_full_checkpoint(str(root / 'ckpt.best_full.pth.tar'))
    assert tag == 'ema' and saved.lesion_aux_loss_enabled == cfg.lesion_aux_loss_enabled
    assert loaded.fusion.correction_enabled == cfg.lesion_correction_enabled
    assert loaded.fusion.guidance_enabled == cfg.lesion_guidance_enabled
    for override in (True, False):
        restored, _, _, actual_tag = load_full_checkpoint(str(root / 'ckpt.best_full.pth.tar'), use_ema=override)
        key = 'model_ema' if override else 'model'
        assert actual_tag == ('ema' if override else 'raw')
        for k, v in ckpt[key].items():
            assert torch.equal(v, restored.state_dict()[k]), k


@pytest.mark.parametrize('mode,state,has_pet', [('missing', None, True),
    ('auto', [1, 0], True), ('full', [0, 1], True), ('full', None, False)])
def test_missing_rejected_before_any_encoder(mode, state, has_pet):
    model = FullLesionGuidedPETCT(pretrained=False)
    model.enc_ct.register_forward_pre_hook(lambda *a: (_ for _ in ()).throw(AssertionError('CT must not run')))
    model.enc_pet.register_forward_pre_hook(lambda *a: (_ for _ in ()).throw(AssertionError('PET must not run')))
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError):
        model(x, pet=x if has_pet else None, forward_mode=mode, pet_available=state)


def test_no_cuda_does_not_create_run(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['x', '--checkpoint_root', str(tmp_path), '--hash', 'no-cuda'])
    with pytest.raises(RuntimeError, match='No experiment files'):
        FullLesionGuidedConfig.parse_arguments()
    assert not (tmp_path / 'MDT').exists()


def test_invalid_config_and_existing_results_are_rejected(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path)
    folder = Path(cfg.checkpoint_dir)
    (folder / 'config_args.json').write_text(json.dumps(vars(cfg)))
    (folder / 'ckpt.last.pth.tar').write_text('preserve')
    from configs.full_lesion_guided import prepare_run_directory
    cfg.retry_failed_start = True
    with pytest.raises(FileExistsError): prepare_run_directory(cfg)
    assert (folder / 'ckpt.last.pth.tar').read_text() == 'preserve'
    argv = ['x', '--require_cuda', 'false', '--lesion_aux_loss_weight', '-1']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError): FullLesionGuidedConfig.parse_arguments()
    monkeypatch.setattr(sys, 'argv', ['x', '--train_batch_mode', 'mixed'])
    with pytest.raises(SystemExit): FullLesionGuidedConfig.parse_arguments()


def test_startup_only_retry_preserves_all_files(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path)
    folder = Path(cfg.checkpoint_dir)
    (folder / 'config_args.json').write_text(json.dumps(vars(cfg)))
    (folder / 'train_log.csv').write_text('epoch,train_loss\n')
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    cfg.retry_failed_start = True
    from configs.full_lesion_guided import prepare_run_directory
    prepare_run_directory(cfg)
    backup, = list(folder.parent.glob(folder.name + '.failed-start-*'))
    assert before == {p.name: p.read_bytes() for p in backup.iterdir()}
    assert not folder.exists()
