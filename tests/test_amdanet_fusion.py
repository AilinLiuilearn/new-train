"""Source provenance, true gradient routing and Full-only experiment lifecycle."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

import run_full_add_baseline as full_runner
from run_full_amdanet_fusion import FullAMDANetConfig, main
from eval_full_amdanet_fusion import load_full_checkpoint
from models.components.amdanet_fusion import AMDANetFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_amdanet_fusion import FullAMDANetPETCT, build_full_amdanet_model
from tasks.mdt_seg import MDTSegTeacher


def _features(batch=2):
    # Real channel contract; reduced spatial sizes only to keep unit tests fast.
    shapes = [(64, 16), (128, 8), (320, 4), (512, 2)]
    return [torch.randn(batch, c, s, s, requires_grad=True) for c, s in shapes]


def _model():
    return FullAMDANetPETCT(pretrained=False)


def _config(monkeypatch, tmp_path, **overrides):
    monkeypatch.setattr(sys, 'argv', ['run_full_amdanet_fusion.py',
                        '--pretrained', 'false', '--mixed_precision', 'false',
                        '--checkpoint_root', str(tmp_path), '--hash', 'synthetic',
                        '--hd95_mode', 'surface', '--epochs', '1',
                        '--batch_size', '2', '--num_workers', '0',
                        '--cosine_warmup', '0'])
    cfg = FullAMDANetConfig.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = 'full_amdanet_fusion'
    cfg.fusion_type = 'AMDANetFusion'
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _batch():
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1
    return dict(ct=torch.randn(2, 3, 64, 64),
                pet=torch.randn(2, 3, 64, 64), mask=mask)


def test_upstream_fdam_class_definitions_are_unchanged():
    source = Path('models/components/amdanet_fdam.py').read_text()
    body = source.split('class From_Channel', 1)[1].strip()
    assert hashlib.sha256(body.encode()).hexdigest() == (
        'c9dfcf1f8d1adb7c5581620e9088fba335c6b852703ee9ab460118c5d2ecf4be')


def test_four_scale_fusion_backward_and_both_inputs():
    torch.manual_seed(7)
    fusion = AMDANetFusion()
    ct, pet = _features(), _features()
    original = [x.detach().clone() for x in ct + pet]
    outputs = fusion(ct, pet)
    assert [x.shape for x in outputs] == [x.shape for x in ct]
    sum(x.square().mean() for x in outputs).backward()
    for x, old in zip(ct + pet, original):
        assert torch.equal(x.detach(), old), 'must not modify encoder features in place'
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in fusion.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_outer_bypass_and_exact_original_local_global_composition():
    torch.manual_seed(12)
    fusion = AMDANetFusion().eval()
    ct, pet = _features(), _features()
    with torch.no_grad():
        output = fusion(ct, pet)
        for i, (c, p) in enumerate(zip(ct, pet)):
            delta = fusion.global_[i](*fusion.local[i](c, p))
            torch.testing.assert_close(output[i], c + p + delta, rtol=0, atol=0)
            assert not torch.equal(output[i], c + p), 'original FDAM is active at initialization'


@pytest.mark.parametrize('failure', ['scale_count', 'shape', 'channel', 'small', 'nan', 'dtype'])
def test_bad_feature_contract_fails(failure):
    ct, pet = _features(), _features()
    if failure == 'scale_count':
        pet = pet[:3]
    elif failure == 'shape':
        pet[0] = pet[0][:, :, :-1]
    elif failure == 'channel':
        ct[0] = pet[0] = torch.randn(2, 32, 16, 16)
    elif failure == 'small':
        ct[0] = pet[0] = torch.randn(2, 64, 4, 4)
    elif failure == 'nan':
        pet[0] = torch.full_like(pet[0], float('nan'))
    elif failure == 'dtype':
        pet[0] = pet[0].double()
    with pytest.raises((ValueError, RuntimeError)):
        AMDANetFusion()(ct, pet)


def test_same_seed_preserves_common_baseline_initialization():
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False)
    torch.manual_seed(2023)
    experiment = _model()
    for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder'):
        a = getattr(baseline, name).state_dict()
        b = getattr(experiment, name).state_dict()
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), f'{name}.{key}'


def test_full_segmentation_loss_updates_encoders_align_decoder_and_fusion(monkeypatch, tmp_path):
    cfg = _config(monkeypatch, tmp_path)
    task = MDTSegTeacher(build_full_amdanet_model(cfg), cfg)
    optimizer_ids = {id(p) for g in task.optimizer.param_groups for p in g['params']}
    assert all(id(p) in optimizer_ids for p in task.model.fusion.parameters())
    loss, outputs, _ = full_runner.train_step_full(task, _batch())
    assert outputs['logits'].shape == (2, 1, 64, 64)
    loss.backward()
    for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder', 'fusion'):
        grads = [p.grad for p in getattr(task.model, name).parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(float(g.abs().sum()) for g in grads) > 0, name
    before = task.model.fusion.global_[3].proj.weight.detach().clone()
    task.optimizer.step()
    assert not torch.equal(before, task.model.fusion.global_[3].proj.weight)


@pytest.mark.parametrize('mode,state,pet_present', [
    ('missing', None, True), ('auto', [1, 0], True), ('full', [1, 0], True),
    ('full', None, False), ('auto', [1., 1.], True),
])
def test_missing_and_invalid_states_fail_before_any_encoder(mode, state, pet_present):
    model = _model()
    model.enc_ct.register_forward_pre_hook(
        lambda *args: (_ for _ in ()).throw(AssertionError('must reject before encoding')))
    ct = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError):
        model(ct, pet=ct if pet_present else None, pet_available=state, forward_mode=mode)


def test_real_pet_affects_full_prediction():
    torch.manual_seed(99)
    model = _model().eval()
    ct, pet = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = model(ct, pet=pet)['logits']
        b = model(ct, pet=-pet, pet_available=[1], forward_mode='auto')['logits']
    assert not torch.allclose(a, b)


@pytest.mark.parametrize('ema_active', [False, True])
def test_checkpoint_rebuild_honors_recorded_raw_or_ema(monkeypatch, tmp_path, ema_active):
    cfg = _config(monkeypatch, tmp_path, ema_enabled=True,
                  ema_start_epoch=0 if ema_active else 3)
    task = MDTSegTeacher(build_full_amdanet_model(cfg), cfg)
    with torch.no_grad():
        task.ema.model.fusion.global_[3].proj.weight.add_(0.1)
    path = tmp_path / 'full.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    model, saved_cfg, _, tag = load_full_checkpoint(str(path))
    assert saved_cfg.norm_mode == cfg.norm_mode
    assert saved_cfg.hd95_mode == cfg.hd95_mode
    assert tag == ('ema' if ema_active else 'raw')
    expected = task.ema.model if ema_active else task.model
    for k, v in expected.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), k


def test_full_runner_train_validate_save_rebuild_test(monkeypatch, tmp_path):
    cfg = _config(monkeypatch, tmp_path, ema_enabled=True, ema_start_epoch=3)
    monkeypatch.setattr(FullAMDANetConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    batch = _batch()
    monkeypatch.setattr(full_runner, '_loaders', lambda cfg: ([batch], [batch], [batch]))
    seen = []
    real_evaluate = MDTSegTeacher.evaluate

    def evaluate(self, *args, **kwargs):
        seen.append((kwargs['eval_mode'], type(kwargs['model'].fusion).__name__))
        return real_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(MDTSegTeacher, 'evaluate', evaluate)
    main()
    out = Path(cfg.checkpoint_dir)
    result = json.loads((out / 'final_test_full.json').read_text())
    assert result['experiment_type'] == 'full_amdanet_fusion'
    assert result['fusion_type'] == 'AMDANetFusion'
    assert result['weights'] == 'raw', 'inactive EMA must not be selected just because bytes exist'
    assert result['best_epoch'] == 1
    assert seen == [('full', 'AMDANetFusion'), ('full', 'AMDANetFusion')]
    assert (out / 'ckpt.best_full.pth.tar').exists()
    assert not (out / 'ckpt.best_joint.pth.tar').exists()


def test_config_uses_full_only_and_rejects_mixed(monkeypatch, tmp_path):
    cfg = _config(monkeypatch, tmp_path)
    assert cfg.model_arch == 'full_amdanet_fusion'
    assert cfg.train_batch_mode == 'full'
    monkeypatch.setattr(sys, 'argv', ['x', '--train_batch_mode', 'mixed'])
    with pytest.raises(SystemExit):
        FullAMDANetConfig.parse_arguments()
