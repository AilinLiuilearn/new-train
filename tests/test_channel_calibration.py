"""Contracts for the MMTM-inspired four-scale calibrated addition experiment."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

import run_full_cmx_fusion as runner
from configs.full_channel_calibration import FullChannelCalibrationConfig
from eval_full_channel_calibration import load_full_checkpoint
from models.components.channel_calibration import (
    ChannelCalibrationFusion, STAGE_CHANNELS, UPSTREAM_BLOB, LICENSE_BLOB)
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_channel_calibration import FullChannelCalibrationPETCT, build_full_channel_model


def features(actual=False):
    sizes = (128, 64, 32, 16) if actual else (8, 4, 2, 1)
    cs = [torch.randn(2, c, s, s, requires_grad=True) for c, s in zip(STAGE_CHANNELS, sizes)]
    return cs, [torch.randn_like(c, requires_grad=True) for c in cs]


def test_upstream_source_and_license_identity():
    root = Path(__file__).resolve().parents[1]
    for path, expected in {'third_party/MMTM/mmtm.py': UPSTREAM_BLOB,
                           'third_party/MMTM/LICENSE': LICENSE_BLOB}.items():
        raw = (root / path).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == expected


@pytest.mark.parametrize('enabled', [True, False])
def test_all_actual_scales_step_zero_exact_add_and_input_gradients(enabled):
    fusion = ChannelCalibrationFusion(enabled=enabled)
    cs, ps = features(actual=True)
    outputs = fusion(cs, ps)
    for out, c, p in zip(outputs, cs, ps):
        torch.testing.assert_close(out, c + p, rtol=0, atol=0)
    sum(x.square().mean() for x in outputs).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in cs + ps)
    assert sum(p.numel() for p in fusion.parameters()) == (196744 if enabled else 0)


def test_zero_heads_then_shared_hidden_layer_learns_on_second_step():
    fusion = ChannelCalibrationFusion()
    opt = torch.optim.AdamW(fusion.parameters(), lr=4e-5)
    cs, ps = features()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        sum(x.square().mean() for x in fusion(cs, ps)).backward()
        for block in fusion.blocks:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
            assert float(block.fc_ct.weight.grad.abs().sum()) > 0
            assert float(block.fc_pet.weight.grad.abs().sum()) > 0
            total = float(block.fc_squeeze.weight.grad.abs().sum())
            assert total == 0 if step == 0 else total > 0
        opt.step()


def test_bounded_independent_weights_and_source_preserving_descriptors():
    fusion = ChannelCalibrationFusion()
    for block in fusion.blocks:
        with torch.no_grad():
            block.fc_ct.bias.fill_(100)
            block.fc_pet.bias.fill_(-100)
    cs, ps = features()
    for out, c, p in zip(fusion(cs, ps), cs, ps):
        torch.testing.assert_close(out, 1.25*c + .75*p, rtol=0, atol=0)
    block = fusion.blocks[0]
    seen = []
    hook = block.fc_squeeze.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
    block(torch.ones_like(cs[0]), torch.full_like(ps[0], 3))
    block(torch.full_like(cs[0], 3), torch.ones_like(ps[0]))
    hook.remove()
    assert not torch.equal(seen[0], seen[1])


@pytest.mark.parametrize('bad', ['count', 'channels', 'dtype', 'empty', 'batch', 'nonfinite'])
def test_invalid_inputs(bad):
    fusion = ChannelCalibrationFusion()
    cs, ps = features()
    if bad == 'count': ps.pop()
    if bad == 'channels': ps[0] = ps[0][:, :63]
    if bad == 'dtype': ps[0] = ps[0].double()
    if bad == 'empty': ps[0] = ps[0][..., :0]
    if bad == 'batch': ps[0] = ps[0][:1]
    if bad == 'nonfinite': ps[0] = torch.full_like(ps[0], float('nan'))
    with pytest.raises((ValueError, RuntimeError)):
        fusion(cs, ps)


@pytest.mark.parametrize('kwargs', [{'reduction': 0}, {'reduction': 1.5}, {'strength': 0},
                                   {'strength': 1}, {'strength': float('nan')}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): ChannelCalibrationFusion(**kwargs)


def test_pet_spatial_alignment_and_state_roundtrip():
    fusion = ChannelCalibrationFusion()
    cs, ps = features()
    ps = [p[..., ::2, ::2] for p in ps]
    other = copy.deepcopy(fusion)
    other.load_state_dict(fusion.state_dict(), strict=True)
    for a, b in zip(fusion(cs, ps), other(cs, ps)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_real_model_shared_initialization_rng_and_baseline_equivalence():
    torch.manual_seed(2023)
    base = DualSharedAddPETCTBaseline(pretrained=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullChannelCalibrationPETCT(pretrained=False).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    ct, pet = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(ct, pet, forward_mode='full')['logits'],
                                   base(ct, pet, forward_mode='full')['logits'], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 32639385


def test_missing_and_mixed_rows_never_read_missing_pet():
    model = FullChannelCalibrationPETCT(pretrained=False).eval()
    ct, pet = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    def forbidden(*args): raise AssertionError('Missing must bypass PET encoder and fusion')
    hooks = [model.enc_pet.register_forward_pre_hook(forbidden),
             model.fusion.register_forward_pre_hook(forbidden)]
    with torch.no_grad():
        a = model(ct, pet=None, forward_mode='missing')['logits']
        b = model(ct, pet=torch.full_like(pet, float('nan')), forward_mode='missing')['logits']
    for h in hooks: h.remove()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        normal = model(ct, pet, pet_available=[1, 0], forward_mode='auto')['logits']
        pet[1].fill_(float('nan'))
        changed = model(ct, pet, pet_available=[1, 0], forward_mode='auto')['logits']
    torch.testing.assert_close(normal, changed, rtol=0, atol=0)


def make_cfg(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setattr(sys, 'argv', ['x', '--require_cuda', 'false', '--pretrained', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', 'calibration-test', '--batch_size', '2',
        '--epochs', '1', '--cosine_warmup', '0', '--mixed_precision', 'false',
        '--hd95_mode', 'surface', '--channel_calibration_enabled', str(enabled)])
    return FullChannelCalibrationConfig.parse_arguments()


@pytest.mark.parametrize('enabled', [True, False])
def test_training_ema_final_evaluation_strict_reload(monkeypatch, tmp_path, enabled):
    cfg = make_cfg(monkeypatch, tmp_path, enabled)
    assert cfg.learning_rate == 4e-5 and cfg.early_stop_patience == 60
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    monkeypatch.setattr(FullChannelCalibrationConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:32, 20:32] = 1
    batch = dict(ct=torch.randn(2, 3, 64, 64), pet=torch.randn(2, 3, 64, 64), mask=mask)
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([batch, batch], [batch], [batch]))
    runner.main(config_cls=FullChannelCalibrationConfig, model_builder=build_full_channel_model,
                expected_fusion=ChannelCalibrationFusion, experiment_type='full_channel_calibration',
                result_config_fields=('channel_calibration_enabled', 'channel_reduction', 'channel_strength'))
    folder = Path(cfg.checkpoint_dir)
    result = json.loads((folder / 'final_test_full.json').read_text())
    assert result['channel_calibration_enabled'] == enabled and result['weights'] == 'ema'
    path = folder / 'ckpt.best_full.pth.tar'
    for choice in (None, False):
        model, saved, checkpoint, tag = load_full_checkpoint(path, use_ema=choice)
        weights = checkpoint['model_ema'] if tag == 'ema' else checkpoint['model']
        for key, value in weights.items(): assert torch.equal(value, model.state_dict()[key])
        assert saved.channel_calibration_enabled == enabled
    checkpoint['config']['channel_initialization'] = 'random'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='architecture'):
        load_full_checkpoint(path)


def test_cli_rejects_existing_run_without_touching_it(monkeypatch, tmp_path):
    cfg = make_cfg(monkeypatch, tmp_path)
    folder = Path(cfg.checkpoint_dir)
    marker = folder / 'config_args.json'
    marker.write_text('original')
    with pytest.raises(FileExistsError): FullChannelCalibrationConfig.parse_arguments()
    assert marker.read_text() == 'original'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_amp_forward_backward():
    fusion = ChannelCalibrationFusion().cuda()
    cs, ps = features(actual=True)
    cs, ps = [x.detach().cuda().requires_grad_() for x in cs], [x.detach().cuda().requires_grad_() for x in ps]
    with torch.autocast('cuda', dtype=torch.float16):
        outputs = fusion(cs, ps)
        loss = sum(x.float().square().mean() for x in outputs)
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in cs + ps)
