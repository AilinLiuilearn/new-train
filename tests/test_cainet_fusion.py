"""Upstream provenance, four ablations, equivalence and checkpoint lifecycle."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

import run_full_add_baseline as runner
from run_full_cainet_fusion import FullCAINetConfig, main
from eval_full_cainet_fusion import load_full_checkpoint
from models.components.add_fusion import AddFusion
from models.components.cainet_fusion import CAINetFusion
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_cainet_fusion import FullCAINetPETCT, build_full_cainet_model
from tasks.mdt_seg import MDTSegTeacher

COMBINATIONS = [(False, False), (True, False), (False, True), (True, True)]


def features():
    return [torch.randn(2, c, s, s, requires_grad=True)
            for c, s in zip((64, 128, 320, 512), (16, 8, 4, 2))]


def config(monkeypatch, tmp_path, cacr, gcm):
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', f'c{int(cacr)}g{int(gcm)}',
        '--cacr_enabled', str(cacr), '--gcm_enabled', str(gcm),
        '--mixed_precision', 'false', '--hd95_mode', 'surface',
        '--batch_size', '2', '--epochs', '1', '--cosine_warmup', '0'])
    cfg = FullCAINetConfig.parse_arguments()
    cfg.ct_pretrained_path = cfg.pet_pretrained_path = None
    cfg.experiment_type = 'full_cainet_fusion'
    cfg.fusion_type = 'CAINetFusion'
    return cfg


def batch():
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1
    return {'ct': torch.randn(2, 3, 64, 64),
            'pet': torch.randn(2, 3, 64, 64), 'mask': target}


def test_author_classes_are_unchanged():
    s = Path('models/components/cainet_source.py').read_text().split('class Interv2', 1)[1].strip()
    assert hashlib.sha256(s.encode()).hexdigest() == (
        'a56bda7c22a0a6fa9d647225f9385874f5edec0aeb85a6063db6fc6acf0fc8ee')


@pytest.mark.parametrize('cacr,gcm', COMBINATIONS)
def test_all_switches_forward_backward_and_disabled_modules_absent(cacr, gcm):
    torch.manual_seed(32)
    fusion = CAINetFusion(cacr, gcm)
    assert hasattr(fusion, 'cacr') == cacr
    assert hasattr(fusion, 'gcm') == gcm
    expected = (2406400 if cacr else 0) + (835584 if gcm else 0)
    assert sum(p.numel() for p in fusion.parameters()) == expected
    ct, pet = features(), features()
    old = [x.detach().clone() for x in ct + pet]
    out = fusion(ct, pet)
    assert [x.shape for x in out] == [x.shape for x in ct]
    if not cacr:
        torch.testing.assert_close(out[0], ct[0] + pet[0], rtol=0, atol=0)
    sum(x.square().mean() for x in out).backward()
    for x, saved in zip(ct + pet, old):
        assert torch.equal(x.detach(), saved)
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in fusion.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize('cacr', [False, True])
def test_gcm_zero_init_matches_same_model_without_gcm(cacr):
    torch.manual_seed(12)
    base = CAINetFusion(cacr, False)
    torch.manual_seed(12)
    guided = CAINetFusion(cacr, True)
    ct, pet = features(), features()
    with torch.no_grad():
        for a, b in zip(base(ct, pet), guided(ct, pet)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_native_cacr_and_gcm_composition_after_projection_is_active():
    fusion = CAINetFusion(True, True)
    torch.nn.init.normal_(fusion.gcm_to_s4.weight, std=0.01)
    ct, pet = features(), features()
    with torch.no_grad():
        out = fusion(ct, pet)
        joint = [fusion.cacr[i](torch.cat((ct[i], pet[i]), 1)) for i in range(4)]
        expected = [sum(x.chunk(2, 1)) for x in joint]
        grid = joint[2].shape[-2:]
        context = torch.cat([torch.nn.functional.interpolate(
            x, size=grid, mode='bilinear', align_corners=False) for x in reversed(joint[1:])], 1)
        delta = torch.nn.functional.interpolate(fusion.gcm_to_s4(fusion.gcm(context)),
            size=expected[3].shape[-2:], mode='bilinear', align_corners=False)
        expected[3] = expected[3] + delta
        for a, b in zip(out, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_gcm_gets_nonzero_gradients_after_projection_starts_learning():
    torch.manual_seed(15)
    fusion = CAINetFusion(True, True)
    opt = torch.optim.AdamW(fusion.parameters(), lr=1e-3)
    ct, pet = features(), features()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = fusion(ct, pet)
        sum(x.square().mean() for x in out).backward()
        if step == 0:
            assert fusion.gcm_to_s4.weight.grad.abs().sum() > 0
            assert all(p.grad is not None and p.grad.abs().sum() == 0
                       for p in fusion.gcm.parameters())
        else:
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in fusion.gcm.parameters())
            assert sum(float(p.grad.abs().sum()) for p in fusion.gcm.parameters()) > 0
        opt.step()
    assert fusion.gcm_to_s4.weight.abs().sum() > 0


def test_both_off_is_exact_baseline_model_and_strict_state_dict():
    torch.manual_seed(2023)
    original = DualSharedAddPETCTBaseline(pretrained=False).eval()
    torch.manual_seed(2023)
    candidate = FullCAINetPETCT(cacr_enabled=False, gcm_enabled=False, pretrained=False).eval()
    assert original.state_dict().keys() == candidate.state_dict().keys()
    for k, v in original.state_dict().items():
        assert torch.equal(v, candidate.state_dict()[k]), k
    candidate.load_state_dict(original.state_dict(), strict=True)
    data = batch()
    with torch.no_grad():
        a = original(data['ct'], pet=data['pet'], forward_mode='full')['logits']
        b = candidate(data['ct'], pet=data['pet'])['logits']
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('cacr,gcm', COMBINATIONS)
def test_real_model_optimizer_loss_and_saved_switches(monkeypatch, tmp_path, cacr, gcm):
    cfg = config(monkeypatch, tmp_path, cacr, gcm)
    task = MDTSegTeacher(build_full_cainet_model(cfg), cfg)
    ids = {id(p) for group in task.optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in task.model.fusion.parameters())
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
    fresh, saved, _, tag = load_full_checkpoint(str(path))
    assert saved.cacr_enabled == cacr and saved.gcm_enabled == gcm
    assert saved.cacr_stages == [1, 2, 3, 4]
    assert fresh.fusion.cacr_stages == (1, 2, 3, 4)
    assert tag == 'raw'
    for k, v in task.model.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k


@pytest.mark.parametrize('mode,state,pet', [('missing', None, True),
    ('auto', [1, 0], True), ('full', None, False)])
def test_missing_fails_before_encoding(mode, state, pet):
    model = FullCAINetPETCT(pretrained=False)
    model.enc_ct.register_forward_pre_hook(
        lambda *args: (_ for _ in ()).throw(AssertionError('encoder must not run')))
    x = torch.randn(2, 3, 64, 64)
    with pytest.raises(ValueError):
        model(x, pet=x if pet else None, pet_available=state, forward_mode=mode)


@pytest.mark.parametrize('cacr,gcm', COMBINATIONS)
def test_training_final_rebuild_preserves_each_combination(monkeypatch, tmp_path, cacr, gcm):
    cfg = config(monkeypatch, tmp_path, cacr, gcm)
    cfg.ema_enabled = True
    cfg.ema_start_epoch = 3  # EMA bytes exist but selection must use raw.
    monkeypatch.setattr(FullCAINetConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    data = batch()
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([data], [data], [data]))
    main()
    result = json.loads((Path(cfg.checkpoint_dir) / 'final_test_full.json').read_text())
    assert result['cacr_enabled'] == cacr and result['gcm_enabled'] == gcm
    assert result['cacr_stages'] == [1, 2, 3, 4]
    assert result['weights'] == 'raw'
    assert result['model_arch'] == 'full_cainet_fusion'
    assert result['experiment_type'] == 'full_cainet_fusion'
    assert result['best_epoch'] == 1


def test_feature_contract_and_no_nan_masking():
    model = CAINetFusion(True, True)
    ct, pet = features(), features()
    with pytest.raises(ValueError):
        model(ct, pet[:3])
    pet[1] = torch.full_like(pet[1], float('nan'))
    with pytest.raises(RuntimeError):
        model(ct, pet)


def test_config_rejects_mixed(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['x', '--train_batch_mode', 'mixed'])
    with pytest.raises(SystemExit):
        FullCAINetConfig.parse_arguments()


def test_relation_blocks_use_float32_under_autocast():
    fusion = CAINetFusion(True, True)
    observed = []
    for module in list(fusion.cacr) + [fusion.gcm]:
        module.register_forward_pre_hook(lambda module, args: observed.append(args[0].dtype))
    ct = [x.detach().to(torch.bfloat16).requires_grad_() for x in features()]
    pet = [x.detach().to(torch.bfloat16).requires_grad_() for x in features()]
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = fusion(ct, pet)
        loss = sum(x.float().square().mean() for x in out)
    loss.backward()
    assert observed == [torch.float32] * 5
    assert all(x.dtype == torch.bfloat16 for x in out)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in fusion.parameters())


def test_active_ema_checkpoint_selection_and_explicit_override(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path, True, True)
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    task = MDTSegTeacher(build_full_cainet_model(cfg), cfg)
    with torch.no_grad():
        task.model.fusion.gcm_to_s4.weight.fill_(0.01)
    task.update_ema()
    path = tmp_path / 'ema.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    assert not torch.equal(task.model.fusion.gcm_to_s4.weight,
                           task.ema.model.fusion.gcm_to_s4.weight)
    for choice, expected in [(None, 'ema'), (True, 'ema'), (False, 'raw')]:
        restored, saved, _, tag = load_full_checkpoint(str(path), use_ema=choice)
        assert tag == expected and saved.cacr_enabled and saved.gcm_enabled
        original = task.ema.model if tag == 'ema' else task.model
        for k, v in original.state_dict().items():
            assert torch.equal(v, restored.state_dict()[k]), k


def test_pet_changes_prediction_and_common_initial_weights_are_preserved():
    torch.manual_seed(2023)
    original = DualSharedAddPETCTBaseline(pretrained=False)
    torch.manual_seed(2023)
    candidate = FullCAINetPETCT(pretrained=False).eval()
    for k, v in original.state_dict().items():
        assert torch.equal(v, candidate.state_dict()[k]), k
    data = batch()
    with torch.no_grad():
        a = candidate(data['ct'], pet=data['pet'])['logits']
        b = candidate(data['ct'], pet=torch.zeros_like(data['pet']))['logits']
    assert (a - b).abs().max() > 0


def test_existing_run_cannot_be_overwritten(monkeypatch, tmp_path):
    cfg = config(monkeypatch, tmp_path, True, True)
    path = Path(cfg.checkpoint_dir) / 'config_args.json'
    path.write_text('{"existing": true}')
    with pytest.raises(FileExistsError, match='fresh --hash'):
        FullCAINetConfig.parse_arguments()
    assert path.read_text() == '{"existing": true}'


@pytest.mark.parametrize('gcm', [False, True])
def test_s1_cacr_is_executed_and_learns_from_s1_loss(gcm):
    torch.manual_seed(7)
    fusion = CAINetFusion(True, gcm)
    assert len(fusion.cacr) == 4
    assert fusion.cacr[0].theta.in_channels == 128
    captured = []
    fusion.cacr[0].register_forward_pre_hook(
        lambda module, args: captured.append(tuple(args[0].shape)))
    opt = torch.optim.SGD(fusion.parameters(), lr=0.01)
    ct, pet = features(), features()
    out = fusion(ct, pet)
    assert captured == [(2, 128, 16, 16)]
    assert (out[0] - ct[0] - pet[0]).abs().max() > 1e-6
    before = [p.detach().clone() for p in fusion.cacr[0].parameters()]
    out[0].square().mean().backward()
    grads = [p.grad for p in fusion.cacr[0].parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    opt.step()
    assert any(not torch.equal(old, p) for old, p in zip(before, fusion.cacr[0].parameters()))


@pytest.mark.parametrize('gcm', [False, True])
def test_legacy_three_scale_checkpoint_restores_saved_layout(monkeypatch, tmp_path, gcm):
    cfg = config(monkeypatch, tmp_path, True, gcm)
    cfg.cacr_stages = [2, 3, 4]
    task = MDTSegTeacher(build_full_cainet_model(cfg), cfg)
    task.model.eval()
    assert len(task.model.fusion.cacr) == 3
    assert task.model.fusion.cacr[0].theta.in_channels == 256
    path = tmp_path / 'legacy.pth.tar'
    task.save_checkpoint(str(path), 1, 0.5, 1, {'dice': 0.5})
    restored, saved, _, _ = load_full_checkpoint(str(path))
    assert saved.cacr_stages == [2, 3, 4]
    assert restored.fusion.cacr_stages == (2, 3, 4)
    assert len(restored.fusion.cacr) == 3
    data = batch()
    with torch.no_grad():
        a = task.model(data['ct'], pet=data['pet'])['logits']
        b = restored(data['ct'], pet=data['pet'])['logits']
    torch.testing.assert_close(a, b, rtol=0, atol=0)
