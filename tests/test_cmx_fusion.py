"""Official-source parity and four-scale fusion seam contracts."""
import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from models.components.cmx_fusion import CMXFusion, STAGE_CHANNELS, STAGE_HEADS
from models.components.cmx_source import FeatureFusionModule, FeatureRectifyModule


def features(actual=False):
    sizes = (128, 64, 32, 16) if actual else (16, 8, 4, 2)
    return [torch.randn(2, c, s, s, requires_grad=True) for c, s in zip(STAGE_CHANNELS, sizes)]


def test_vendored_source_and_license_are_byte_identical():
    for name, sha in [('models/components/cmx_source.py', 'ebe09316222c871a2335adb804576a472b5a1013'),
                      ('third_party/CMX_LICENSE', '4c65cea1d1a536da33bd79a1f1e7a86988a63203')]:
        raw = Path(name).read_bytes()
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == sha


@pytest.mark.parametrize('frm,ffm', [(False, False), (False, True), (True, False), (True, True)])
def test_actual_scales_shapes_and_original_constructor_configuration(frm, ffm):
    module = CMXFusion(ffm_enabled=ffm, frm_enabled=frm).eval()
    ct, pet = features(True), features(True)
    with torch.no_grad(): out = module(ct, pet)
    assert len(out) == 4
    for c, p, y in zip(ct, pet, out):
        assert y.shape == c.shape and torch.isfinite(y).all()
        if not frm and not ffm:
            torch.testing.assert_close(y, c + p, atol=0, rtol=0)
    assert len(module.ffms) == (4 if ffm else 0)
    assert len(module.frms) == (4 if frm else 0)
    for block, channels, heads in zip(module.ffms, STAGE_CHANNELS, STAGE_HEADS):
        assert type(block) is FeatureFusionModule
        assert block.cross.cross_attn.num_heads == heads
        assert block.cross.cross_attn.scale == (channels // heads) ** -.5
        assert isinstance(block.cross.norm1, torch.nn.LayerNorm)
        assert isinstance(block.channel_emb.norm, torch.nn.BatchNorm2d)
        assert torch.all(block.channel_emb.norm.weight == 1)  # never zero-init original FFM
        assert block.channel_emb.residual.weight.abs().sum() > 0
    for block in module.frms:
        assert type(block) is FeatureRectifyModule
        assert block.lambda_c == block.lambda_s == .5
    if not frm and not ffm:
        assert not module.state_dict()
    elif ffm:
        assert any(not torch.equal(y, c + p) for y, c, p in zip(out, ct, pet))


@pytest.mark.parametrize('frm,ffm', [(False, True), (True, False), (True, True)])
def test_wrapper_value_and_gradient_parity_with_direct_official_modules(frm, ffm):
    module = CMXFusion(ffm_enabled=ffm, frm_enabled=frm).train()
    reference_ffms, reference_frms = copy.deepcopy(module.ffms), copy.deepcopy(module.frms)
    ct, pet = features(), features()
    ct_ref = [x.detach().clone().requires_grad_() for x in ct]
    pet_ref = [x.detach().clone().requires_grad_() for x in pet]
    actual = module(ct, pet)
    expected = []
    for i, (c, p) in enumerate(zip(ct_ref, pet_ref)):
        if frm: c, p = reference_frms[i](c, p)
        expected.append(reference_ffms[i](c, p) if ffm else c + p)
    probes = [torch.randn_like(x) for x in actual]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    ga = torch.autograd.grad(sum((x * z).sum() for x, z in zip(actual, probes)), ct + pet)
    gb = torch.autograd.grad(sum((x * z).sum() for x, z in zip(expected, probes)), ct_ref + pet_ref)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_all_enabled_parameters_learn_without_delayed_zero_initialization():
    module = CMXFusion(ffm_enabled=True, frm_enabled=True).train()
    ct, pet = features(), features()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    out = module(ct, pet)
    sum((x * torch.randn_like(x)).mean() for x in out).backward()
    for blocks in (module.ffms, module.frms):
        for block in blocks:
            grads = [p.grad for p in block.parameters()]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads) > 0
    before = [b.cross.cross_attn.kv1.weight.detach().clone() for b in module.ffms]
    optimizer.step()
    assert all(not torch.equal(w, b.cross.cross_attn.kv1.weight) for w, b in zip(before, module.ffms))


def test_frm_switch_does_not_change_initial_ffm_weights():
    torch.manual_seed(2023)
    a = CMXFusion(ffm_enabled=True, frm_enabled=False)
    torch.manual_seed(2023)
    b = CMXFusion(ffm_enabled=True, frm_enabled=True)
    for key, value in a.ffms.state_dict().items():
        assert torch.equal(value, b.ffms.state_dict()[key])


def test_pet_alignment_matches_baseline_and_source():
    module = CMXFusion().eval()
    ct, pet = features(), features()
    pet[0] = torch.randn(2, 64, 8, 8)
    actual = module(ct, pet)[0]
    aligned = F.interpolate(pet[0], ct[0].shape[-2:], mode='bilinear', align_corners=False)
    expected = module.ffms[0](ct[0], aligned)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_inputs_and_switches():
    ct, pet = features(), features()
    with pytest.raises(TypeError): CMXFusion(ffm_enabled='false')
    with pytest.raises(ValueError): CMXFusion()(ct[:3], pet)
    pet[0] = torch.randn(1, 64, 16, 16)
    with pytest.raises(ValueError): CMXFusion()(ct, pet)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('frm', [False, True])
def test_cuda_amp_forward_backward(frm):
    module = CMXFusion(frm_enabled=frm).cuda().train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    ct, pet = [[x.detach().cuda().requires_grad_() for x in features(True)] for _ in range(2)]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out = module(ct, pet)
            loss = sum((x.float() * torch.randn_like(x.float())).mean() for x in out)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        scaler.step(optimizer)
        scaler.update()
