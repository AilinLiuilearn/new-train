"""Source fidelity, scale causality, and isolated fusion contract tests."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from models.components.contrinet_fusion import ConTriNetMFMFusion, STAGE_CHANNELS

ROOT = Path(__file__).resolve().parents[1]


def features(sizes=(16, 8, 4, 2), batch_size=2):
    return [torch.randn(batch_size, c, s, s, requires_grad=True)
            for c, s in zip(STAGE_CHANNELS, sizes)]


def test_official_source_provenance_and_unchanged_class_bodies():
    archived = ROOT / 'third_party/contrinet/ConTriNet.py'
    runtime = ROOT / 'models/components/contrinet_source.py'
    manifest = json.loads((ROOT / 'third_party/contrinet/SOURCE.json').read_text())
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == manifest['archived_sha256']
    original_nodes = {n.name: n for n in ast.parse(archived.read_text()).body
                      if isinstance(n, ast.ClassDef)}
    extracted_nodes = {n.name: n for n in ast.parse(runtime.read_text()).body
                       if isinstance(n, ast.ClassDef)}
    assert set(extracted_nodes) == {'MFM_0', 'MFM'}
    for name, node in extracted_nodes.items():
        assert ast.dump(node) == ast.dump(original_nodes[name]), name
        assert ast.get_source_segment(runtime.read_text(), node) == ast.get_source_segment(
            archived.read_text(), original_nodes[name]), name


@pytest.mark.parametrize('cross_scale', [False, True])
@pytest.mark.parametrize('mode', ['residual', 'replace'])
def test_forward_backward_no_input_mutation(cross_scale, mode):
    fusion = ConTriNetMFMFusion(cross_scale=cross_scale, output_mode=mode)
    ct, pet = features(), features()
    saved = [x.detach().clone() for x in ct + pet]
    out = fusion(ct, pet)
    assert [x.shape for x in out] == [x.shape for x in ct]
    if mode == 'residual':
        for x, c, p in zip(out, ct, pet):
            torch.testing.assert_close(x, c + p, atol=0, rtol=0)
    sum(x.square().mean() for x in out).backward()
    for x, original in zip(ct + pet, saved):
        assert torch.equal(x.detach(), original)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
    for name, param in fusion.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_disabled_is_exact_add_and_has_no_parameters():
    fusion = ConTriNetMFMFusion(enabled=False)
    assert list(fusion.parameters()) == []
    assert not fusion.state_dict()
    ct, pet = features(), features()
    for a, c, p in zip(fusion(ct, pet), ct, pet):
        torch.testing.assert_close(a, c + p, atol=0, rtol=0)


def test_source_module_learns_after_zero_projection_first_step():
    torch.manual_seed(8)
    fusion = ConTriNetMFMFusion()
    opt = torch.optim.AdamW(fusion.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in fusion.first.parameters()]
    for step in range(2):
        ct, pet = features(), features()
        opt.zero_grad(set_to_none=True)
        loss = sum(x.square().mean() for x in fusion(ct, pet))
        loss.backward()
        source_grads = [p.grad for p in fusion.first.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in source_grads)
        if step == 0:
            assert sum(float(g.abs().sum()) for g in source_grads) == 0
            assert all(p.weight.grad.abs().sum() > 0 for p in fusion.output_proj)
        else:
            assert sum(float(g.abs().sum()) for g in source_grads) > 0
        opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, fusion.first.parameters()))


@pytest.mark.parametrize('cross_scale', [False, True])
def test_s4_depends_on_s1_only_when_cross_scale_enabled(cross_scale):
    torch.manual_seed(5)
    fusion = ConTriNetMFMFusion(cross_scale=cross_scale, output_mode='replace').eval()
    ct, pet = features(), features()
    out = fusion(ct, pet)
    changed = list(ct)
    changed[0] = ct[0] + 3.0
    altered = fusion(changed, pet)
    if cross_scale:
        assert (out[-1] - altered[-1]).abs().max() > 1e-8
    else:
        torch.testing.assert_close(out[-1], altered[-1], rtol=0, atol=0)
    out[-1].square().mean().backward()
    if cross_scale:
        assert ct[0].grad is not None and ct[0].grad.abs().sum() > 0
        assert pet[0].grad is not None and pet[0].grad.abs().sum() > 0
    else:
        assert ct[0].grad is None and pet[0].grad is None


def test_full_resolution_and_odd_size_pyramids():
    fusion = ConTriNetMFMFusion().eval()
    for sizes in [(128, 64, 32, 16), (17, 9, 5, 3)]:
        ct, pet = features(sizes, batch_size=1), features(sizes, batch_size=1)
        with torch.no_grad():
            out = fusion(ct, pet)
        assert [x.shape for x in out] == [x.shape for x in ct]
        assert all(torch.isfinite(x).all() for x in out)


def test_bfloat16_autocast_forward_backward():
    fusion = ConTriNetMFMFusion(output_mode='replace')
    ct = [x.detach().bfloat16().requires_grad_() for x in features()]
    pet = [x.detach().bfloat16().requires_grad_() for x in features()]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = fusion(ct, pet)
        loss = sum(x.float().square().mean() for x in out)
    loss.backward()
    assert all(x.dtype == torch.bfloat16 for x in out)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in ct + pet)


def test_invalid_inputs_raise_before_source_batchnorm_updates():
    fusion = ConTriNetMFMFusion()
    ct, pet = features(), features()
    with pytest.raises(ValueError):
        fusion(ct, pet[:3])
    pet[3] = torch.full_like(pet[3], float('nan'))
    before = fusion.first.layer_11[1].num_batches_tracked.clone()
    with pytest.raises(RuntimeError, match='input S4'):
        fusion(ct, pet)
    assert torch.equal(before, fusion.first.layer_11[1].num_batches_tracked)


def test_source_pooling_and_external_input_alignment():
    fusion = ConTriNetMFMFusion(output_mode='replace').eval()
    ct, pet = features(), features()
    pet[0] = torch.randn(2, 64, 15, 15)
    with torch.no_grad():
        out = fusion(ct, pet)
    assert [x.shape for x in out] == [x.shape for x in ct]
