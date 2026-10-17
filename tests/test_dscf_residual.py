"""Source parity, additive initialization, delayed gradients and runner contracts."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import einops
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

from configs.full_dscf import FullDSCFConfig, architecture_metadata
from eval_full_dscf import load_full_checkpoint
from models.components.dscf_residual import (
    DSCFResidualFusion, SparseCorrection, STAGE_CHANNELS, UPSTREAM_BLOB, LICENSE_BLOB)
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.full_dscf_residual import FullDSCFResidualPETCT, build_full_dscf_model
import run_full_dscf_residual as runner


def features(actual=False, batch=2):
    sizes = (128, 64, 32, 16) if actual else (16, 8, 4, 2)
    cs = [torch.randn(batch, c, s, s, requires_grad=True) for c, s in zip(STAGE_CHANNELS, sizes)]
    return cs, [torch.randn_like(c, requires_grad=True) for c in cs]


def archived_reference():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'third_party/KTB/swin.py').read_text())
    wanted = {'LayerNormProxy', 'conv_bn_relu', 'DAttentionMM', 'DeformMPGBlock'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    assert len(nodes) == len(wanted)
    scope = dict(torch=torch, nn=nn, F=F, einops=einops, np=np,
                 DropPath=DropPath, trunc_normal_=trunc_normal_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<archived KTB>', 'exec'), scope)
    return scope


def test_source_license_and_extracted_class_identity():
    root = Path(__file__).resolve().parents[1]
    for name, blob in {'third_party/KTB/swin.py': UPSTREAM_BLOB,
                       'third_party/KTB/LICENSE': LICENSE_BLOB}.items():
        raw = (root / name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob
    original = ast.parse((root / 'third_party/KTB/swin.py').read_text())
    used = ast.parse((root / 'models/components/dscf_source.py').read_text())
    definitions = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.ClassDef)}
    for node in used.body:
        if isinstance(node, ast.ClassDef): assert ast.dump(node) == definitions[node.name]


@pytest.mark.parametrize('stage', range(4))
def test_native_core_output_input_and_parameter_gradient_parity(stage):
    size = (128, 64, 32, 16)[stage]
    channels = STAGE_CHANNELS[stage]
    rank = max(16, channels // 8)
    stride = max(1, size // 8)
    actual = SparseCorrection(channels, rank, size, stride, stage).eval()
    source = archived_reference()
    reference = source['DeformMPGBlock'](channels, stride, 1, 2, 0., stage, rank/channels).eval()
    reference.deform_atten = source['DAttentionMM'](
        rank, q_size=(size, size), stride=stride, n_heads=2, n_groups=1, dpr=0., level=stage).eval()
    # Open only the residual output head to test the complete borrowed core.
    nn.init.normal_(actual.U_fc1.weight, std=.02)
    reference.load_state_dict(actual.state_dict(), strict=True)
    cs = torch.randn(2, channels, size, size, requires_grad=True)
    ps = torch.randn_like(cs, requires_grad=True)
    cr = cs.detach().clone().requires_grad_()
    pr = ps.detach().clone().requires_grad_()
    a = actual(cs, ps)
    b = reference(cr.flatten(2).transpose(1, 2), pr.flatten(2).transpose(1, 2), size, size, stage)
    b = b.transpose(1, 2).reshape_as(cs)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    probe = torch.randn_like(a)
    (a * probe).mean().backward()
    (b * probe).mean().backward()
    for x, y in ((cs, cr), (ps, pr)):
        torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
    for (name, x), (_, y) in zip(actual.named_parameters(), reference.named_parameters()):
        assert x.grad is not None and torch.isfinite(x.grad).all(), name
        torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)


@pytest.mark.parametrize('enabled', [True, False])
def test_four_scales_initially_exact_add(enabled):
    fusion = DSCFResidualFusion(enabled=enabled, image_size=64)
    cs, ps = features()
    for a, c, p in zip(fusion(cs, ps), cs, ps):
        torch.testing.assert_close(a, c+p, atol=0, rtol=0)
    if not enabled: assert sum(p.numel() for p in fusion.parameters()) == 0
    assert all(v == 0 for v in fusion.pop_diagnostics().values())


def test_zero_head_opens_then_offsets_and_both_projections_learn():
    torch.manual_seed(2023)
    fusion = DSCFResidualFusion(image_size=64)
    opt = torch.optim.AdamW(fusion.parameters(), lr=4e-5)
    cs, ps = features()
    target = [torch.randn_like(c) for c in cs]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        loss = sum((a-y).square().mean() for a, y in zip(fusion(cs, ps), target))
        loss.backward()
        for block in fusion.blocks:
            for name, p in block.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert block.U_fc1.weight.grad.abs().sum() > 0
            for module in (block.D_fc1, block.D_fc2,
                           block.deform_atten.conv_offset_x, block.deform_atten.conv_offset_y,
                           block.deform_atten.proj_k, block.deform_atten.proj_v,
                           block.deform_atten.get_sample_weight):
                total = sum(float(p.grad.abs().sum()) for p in module.parameters())
                assert (total == 0) if step == 0 else (total > 0)
        opt.step()
    ratios = fusion.pop_diagnostics()
    assert all(v > 0 for v in ratios.values())
    assert fusion.pop_diagnostics() == {f'dscf_delta_ratio_s{i+1}': 0. for i in range(4)}


def test_resize_signed_output_and_strict_roundtrip():
    fusion = DSCFResidualFusion(image_size=64).eval()
    for block in fusion.blocks:
        nn.init.normal_(block.U_fc1.weight, std=.02)
    cs, ps = features()
    ps = [p[..., ::2, ::2] for p in ps]
    other = copy.deepcopy(fusion)
    other.load_state_dict(fusion.state_dict(), strict=True)
    corrections = []
    for a, b, c, p in zip(fusion(cs, ps), other(cs, ps), cs, ps):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        corrections.append(a - c - F.interpolate(p, c.shape[-2:], mode='bilinear', align_corners=False))
    assert any((x < 0).any() for x in corrections)
    assert any((x > 0).any() for x in corrections)
    assert fusion.pop_diagnostics() == {f'dscf_delta_ratio_s{i+1}': 0. for i in range(4)}


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_features_use_float32_core(dtype):
    fusion = DSCFResidualFusion(image_size=64)
    cs, ps = features()
    cs = [x.detach().to(dtype).requires_grad_() for x in cs]
    ps = [x.detach().to(dtype).requires_grad_() for x in ps]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = fusion(cs, ps)
        for a, c, p in zip(out, cs, ps):
            assert a.dtype == dtype
            torch.testing.assert_close(a, c+p, atol=0, rtol=0)
        loss = sum(x.float().square().mean() for x in out)
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in cs+ps)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
    with pytest.raises(ValueError, match='master weights'):
        fusion.half()(cs, ps)


@pytest.mark.parametrize('bad', ['count', 'channel', 'batch', 'dtype', 'empty', 'nonfinite', 'too_small'])
def test_invalid_input_rejected(bad):
    fusion = DSCFResidualFusion(image_size=64)
    cs, ps = features()
    if bad == 'count': ps.pop()
    if bad == 'channel': ps[0] = ps[0][:, :63]
    if bad == 'batch': ps[0] = ps[0][:1]
    if bad == 'dtype': ps[0] = ps[0].double()
    if bad == 'empty': ps[0] = ps[0][..., :0]
    if bad == 'nonfinite': ps[0] = torch.full_like(ps[0], float('nan'))
    if bad == 'too_small': cs[0] = cs[0][..., :1, :1]
    with pytest.raises((ValueError, RuntimeError)): fusion(cs, ps)


@pytest.mark.parametrize('enabled', [True, False])
def test_model_shared_weights_rng_and_initial_logits_equal_add(enabled):
    torch.manual_seed(2023)
    baseline = DualSharedAddPETCTBaseline(pretrained=False).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    model = FullDSCFResidualPETCT(pretrained=False, dscf_enabled=enabled, image_size=64).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in baseline.state_dict().items(): assert torch.equal(v, model.state_dict()[k]), k
    ct, pet = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        actual = model(ct, pet, forward_mode='full')['logits']
        expected = baseline(ct, pet, forward_mode='full')['logits']
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_missing_and_auto_bypass_forbidden_pet_and_fusion_calls():
    model = FullDSCFResidualPETCT(pretrained=False, image_size=64).eval()
    ct, pet = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    def forbidden(*args): raise AssertionError('Missing must bypass PET/fusion')
    hooks = [m.register_forward_pre_hook(forbidden) for m in (model.enc_pet, model.fusion)]
    with torch.no_grad():
        a = model(ct, pet=None, forward_mode='missing')['logits']
        b = model(ct, pet=torch.full_like(pet, float('nan')), forward_mode='missing')['logits']
        auto_missing = model(ct, pet=None, pet_available=[0, 0], forward_mode='auto')['logits']
    for h in hooks: h.remove()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a, auto_missing, atol=0, rtol=0)
    with torch.no_grad():
        a = model(ct, pet, pet_available=[1, 0], forward_mode='auto')['logits']
        pet[1].fill_(float('nan'))
        b = model(ct, pet, pet_available=[1, 0], forward_mode='auto')['logits']
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def cfg_for(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setattr(sys, 'argv', ['x', '--pretrained', 'false', '--require_cuda', 'false',
        '--checkpoint_root', str(tmp_path), '--hash', 'dscf-test', '--dscf_enabled', str(enabled),
        '--image_size_2d', '64', '--batch_size', '2', '--epochs', '1', '--cosine_warmup', '0',
        '--mixed_precision', 'false', '--hd95_mode', 'surface'])
    return FullDSCFConfig.parse_arguments()


@pytest.mark.parametrize('enabled', [True, False])
def test_runner_two_steps_ema_final_eval_and_strict_checkpoint(monkeypatch, tmp_path, enabled):
    cfg = cfg_for(monkeypatch, tmp_path, enabled)
    assert cfg.learning_rate == 4e-5 and cfg.norm_mode == 'imagenet' and cfg.early_stop_patience == 60
    cfg.ema_enabled, cfg.ema_start_epoch = True, 0
    monkeypatch.setattr(FullDSCFConfig, 'parse_arguments', classmethod(lambda cls: cfg))
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:32, 20:32] = 1
    batch = dict(ct=torch.randn(2, 3, 64, 64), pet=torch.randn(2, 3, 64, 64), mask=mask)
    monkeypatch.setattr(runner, '_loaders', lambda cfg: ([batch, batch], [batch], [batch]))
    runner.main()
    folder = Path(cfg.checkpoint_dir)
    result = json.loads((folder / 'final_test_full.json').read_text())
    assert result['dscf_enabled'] == enabled and result['weights'] == 'ema'
    header = (folder / 'train_log.csv').read_text().splitlines()[0]
    assert 'dscf_delta_ratio_s4' in header
    path = folder / 'ckpt.best_full.pth.tar'
    for choice in (None, False):
        model, saved, checkpoint, tag = load_full_checkpoint(path, use_ema=choice)
        weights = checkpoint['model_ema'] if tag == 'ema' else checkpoint['model']
        for k, v in weights.items(): assert torch.equal(v, model.state_dict()[k])
        assert saved.dscf_enabled == enabled
    checkpoint['config']['dscf_output_mode'] = 'replace_add'
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='architecture'): load_full_checkpoint(path)


def test_main_bce_dice_only(monkeypatch, tmp_path):
    from tasks.mdt_seg import MDTSegTeacher
    cfg = cfg_for(monkeypatch, tmp_path)
    task = MDTSegTeacher(build_full_dscf_model(cfg), cfg)
    batch = dict(ct=torch.randn(2, 3, 64, 64), pet=torch.randn(2, 3, 64, 64), mask=torch.ones(2, 1, 64, 64))
    loss, out, _ = runner.train_step_full(task, batch)
    main, _ = task.criterion(out['logits'], batch['mask'])
    torch.testing.assert_close(loss, main, atol=0, rtol=0)
    assert not out['aux']
    loss.backward()
    for name in ('enc_ct', 'enc_pet', 'ct_align', 'pet_align', 'decoder', 'fusion'):
        grads = [p.grad for p in getattr(task.model, name).parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(float(g.abs().sum()) for g in grads) > 0, name


def test_fresh_run_guard_and_constructor_validation(monkeypatch, tmp_path):
    cfg = cfg_for(monkeypatch, tmp_path)
    path = Path(cfg.checkpoint_dir) / 'config_args.json'
    path.write_text('original')
    with pytest.raises(FileExistsError): FullDSCFConfig.parse_arguments()
    assert path.read_text() == 'original'
    for size in (0, 32, 65, 512.0, True):
        with pytest.raises(ValueError): DSCFResidualFusion(image_size=size)
    with pytest.raises(ValueError): DSCFResidualFusion(enabled='false')
    assert architecture_metadata(512)['dscf_stage_strides'] == [16, 8, 4, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_amp_two_steps_exact_add_and_finite_offsets():
    fusion = DSCFResidualFusion().cuda()
    cs, ps = features(actual=True)
    cs = [x.detach().cuda().half().requires_grad_() for x in cs]
    ps = [x.detach().cuda().half().requires_grad_() for x in ps]
    opt = torch.optim.AdamW(fusion.parameters(), lr=4e-5)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = fusion(cs, ps)
            if step == 0:
                for a, c, p in zip(out, cs, ps): torch.testing.assert_close(a, c+p, atol=0, rtol=0)
            loss = sum(x.float().square().mean() for x in out)
        loss.backward()
        assert all(torch.isfinite(x.grad).all() for x in cs + ps)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters())
        if step:
            for block in fusion.blocks:
                assert sum(p.grad.abs().sum() for p in block.deform_atten.conv_offset_x.parameters()) > 0
                assert sum(p.grad.abs().sum() for p in block.deform_atten.conv_offset_y.parameters()) > 0
        opt.step()
