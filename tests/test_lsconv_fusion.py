"""LSConv contract and independent SKA value/gradient oracle."""
import ast
import hashlib
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from models.components.lsconv_fusion import LSConvFusion, STAGE_CHANNELS
from models.components.lsconv_ska import SKA, ska_torch
from models.components.lsconv_source import LSConv


def oracle(x, w):
    b, c, h, width = x.shape
    patches = F.unfold(x, 3, padding=1).reshape(b, c, 9, h, width)
    # Match official ci % wc mapping rather than repeating adjacent channels.
    kernels = w[:, torch.arange(c, device=x.device) % w.shape[1]]
    return (patches * kernels).sum(2)


def features(actual=False):
    sizes = (128, 64, 32, 16) if actual else (16, 8, 4, 2)
    return [torch.randn(2, c, s, s, requires_grad=True)
            for c, s in zip(STAGE_CHANNELS, sizes)]


def test_official_triton_file_is_byte_identical():
    data = Path('models/components/lsconv_ska_triton.py').read_bytes()
    git_blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    assert git_blob == 'ee0ef7ac261ec8821827f781c712431a68ae7b46'


def test_extracted_class_bodies_match_pinned_official_source():
    expected = {
        'Conv2d_BN': 'e6fcb3d35271dc6286f958e66b3c11a244df445d77cfbbf80ab041492b1afd77',
        'LKP': '0d4ca810bdfb77b9dada35d0f1e9c4c2d2ec439a528b961863b8e64e04b3c05e',
        'LSConv': 'ab82444ffd65c42d5cf78e2aceff06dc7e82247edd8d3f4a28b9c1407583e0ca',
    }
    # Digests were checked against the source blob at the pinned commit.
    tree = ast.parse(Path('models/components/lsconv_source.py').read_text())
    # Ignore Python 3.12's extra empty type_params field for 3.9–3.11 users.
    actual = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False)
              .replace(', type_params=[]', '').encode()).hexdigest()
              for node in tree.body if isinstance(node, ast.ClassDef)}
    assert actual == expected


def test_ska_matches_independent_unfold_forward_and_both_gradients():
    torch.manual_seed(19)
    x = torch.randn(2, 16, 5, 7, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 2, 9, 5, 7, dtype=torch.float64, requires_grad=True)
    probe = torch.randn_like(x)
    a, b = ska_torch(x, w), oracle(x, w)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad((a * probe).sum(), (x, w))
    gb = torch.autograd.grad((b * probe).sum(), (x, w))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_actual_all_scale_shapes_and_exact_add_initialization(training, enabled):
    model = LSConvFusion(enabled, 'torch').train(training)
    ct, pet = features(actual=True), features(actual=True)
    with torch.no_grad():
        out = model(ct, pet)
    for a, b, result in zip(ct, pet, out):
        assert result.shape == a.shape
        torch.testing.assert_close(result, a + b, atol=0, rtol=0)
    if enabled:
        assert len(model.blocks) == 4
        assert len({id(block) for block in model.blocks}) == 4
        for block, c in zip(model.blocks, STAGE_CHANNELS):
            assert isinstance(block, LSConv)
            assert block.lkp.cv2.c.kernel_size == (7, 7)
            assert block.lkp.cv4.out_channels == 9 * c // 8
    else:
        assert not model.state_dict()


def test_zero_bn_learns_then_unblocks_lkp_and_both_modalities():
    model = LSConvFusion(backend='torch').train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ct, pet = features(), features()
    out = model(ct, pet)
    sum(y.square().mean() for y in out).backward()
    for block in model.blocks:
        assert block.bn.weight.grad.abs().sum() > 0
        # Zero final BN blocks LKP gradient on the first backward only.
        assert torch.count_nonzero(block.lkp.cv1.c.weight.grad) == 0
    for a, b in zip(ct, pet):
        assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    out = model(ct, pet)
    sum(y.square().mean() for y in out).backward()
    for block in model.blocks:
        grads = [p.grad for p in block.lkp.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert block.lkp.cv1.c.weight.grad.abs().sum() > 0
    assert any(not torch.equal(y, a + b) for y, a, b in zip(out, ct, pet))


def test_pet_spatial_alignment_is_the_baseline_add_operation():
    ct, pet = features(), features()
    pet[0] = torch.randn(2, 64, 8, 8)
    out = LSConvFusion(backend='torch')(ct, pet)
    expected = ct[0] + F.interpolate(pet[0], ct[0].shape[-2:], mode='bilinear', align_corners=False)
    torch.testing.assert_close(out[0], expected, atol=0, rtol=0)


def test_invalid_inputs_and_no_silent_triton_cpu_fallback():
    ct, pet = features(), features()
    with pytest.raises(ValueError):
        LSConvFusion()(ct[:3], pet)
    pet[0] = torch.randn(1, 64, 16, 16)
    with pytest.raises(ValueError):
        LSConvFusion()(ct, pet)
    with pytest.raises(ValueError):
        LSConvFusion(backend='unknown')
    with pytest.raises(RuntimeError, match='requires CUDA'):
        SKA('triton')(torch.randn(1, 8, 3, 3), torch.randn(1, 1, 9, 3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_official_triton_matches_reference_forward_and_backward(dtype):
    x = torch.randn(2, 64, 13, 17, device='cuda', dtype=dtype, requires_grad=True)
    w = torch.randn(2, 8, 9, 13, 17, device='cuda', dtype=dtype, requires_grad=True)
    a, b = SKA('triton')(x, w), ska_torch(x, w)
    tol = 2e-2 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    probe = torch.randn_like(x)
    ga = torch.autograd.grad((a * probe).sum(), (x, w))
    gb = torch.autograd.grad((b * probe).sum(), (x, w))
    for left, right in zip(ga, gb):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, atol=tol, rtol=tol)
