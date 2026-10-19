"""Contracts: actual sparse selection, edge cues, safe residuals and state isolation."""
import copy

import pytest
import torch

from models.components.asymmetric_fusion import (
    AsymmetricFusion, SparsePETAttention, laplace, ct_to_gray,
)


def features(size=32, channels=(64, 128, 320, 512)):
    return [torch.randn(2, c, max(1, size // 2**i), max(1, size // 2**i))
            for i, c in enumerate(channels)]


def test_all_scales_zero_init_and_disabled_are_exact_add():
    c, p = features(), features()
    module = AsymmetricFusion().eval()
    enabled, regions = module(c, p, torch.randn(2, 3, 128, 128))
    assert len(regions) == 4
    for f, x, y, region in zip(enabled, c, p, regions):
        assert torch.equal(f, x + y)
        assert region.shape == (2, 1, *x.shape[-2:])
    module.enabled = False
    # Disabling must bypass all fusion computation and require no image.
    def forbidden(*args):
        raise AssertionError('disabled fusion ran a trainable branch')
    for stage in module.stages:
        stage.forward = forbidden
    disabled, regions = module(c, p, None)
    assert regions == []
    assert all(torch.equal(f, x+y) for f, x, y in zip(disabled, c, p))


@pytest.mark.parametrize('shape', [(8, 8), (7, 11), (1, 1), (2, 3)])
def test_sparse_attention_padding_tiny_maps_and_gradients(shape):
    module = SparsePETAttention(8, heads=2, n_win=4, topk=4)
    x = torch.randn(2, 8, *shape, requires_grad=True)
    out = module(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert module.qkv.weight.grad.abs().sum() > 0


def test_two_steps_open_both_corrections_then_internal_paths():
    torch.manual_seed(4)
    module = AsymmetricFusion(channels=(8, 8, 8, 8), ranks=(8, 8, 8, 8))
    c, p = features(16, (8,)*4), features(16, (8,)*4)
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
    for step in range(2):
        opt.zero_grad()
        out, _ = module(c, p, torch.randn(2, 3, 64, 64))
        loss = sum(x.square().mean() for x in out)
        loss.backward()
        for name, param in module.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        for stage in module.stages:
            assert stage.up_ct.weight.grad.abs().sum() > 0
            assert stage.up_pet.weight.grad.abs().sum() > 0
            if step:
                for layer in (stage.ct_project, stage.pet_project,
                              stage.region_head, stage.attention.qkv,
                              stage.ega.fusion_conv[0]):
                    assert layer.weight.grad.abs().sum() > 0
        opt.step()
    assert any(not torch.equal(f, x+y) for f, x, y in zip(out, c, p))


def test_ct_image_details_and_pet_region_reach_ct_correction():
    torch.manual_seed(5)
    module = AsymmetricFusion(channels=(8,)*4, ranks=(8,)*4).eval()
    for stage in module.stages:
        torch.nn.init.normal_(stage.up_ct.weight, std=.1)
    c, p = features(16, (8,)*4), features(16, (8,)*4)
    a, _ = module(c, p, torch.zeros(2, 3, 64, 64))
    b, _ = module(c, p, torch.randn(2, 3, 64, 64))
    assert any(not torch.allclose(x,y) for x,y in zip(a,b))
    # With PET residual still zero, PET changes must affect CT correction.
    changed = [x + torch.randn_like(x) for x in p]
    d, _ = module(c, changed, torch.zeros(2, 3, 64, 64))
    assert any(not torch.allclose(x-ci-pi, y-ci-pj)
               for x,y,ci,pi,pj in zip(a,d,c,p,changed))


def test_laplacian_and_imagenet_inverse():
    gray = torch.rand(2, 1, 32, 32)
    mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
    std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
    normalized = (gray.expand(-1,3,-1,-1)-mean)/std
    assert torch.allclose(ct_to_gray(normalized, 'imagenet'), gray, atol=1e-6)
    assert laplace(torch.zeros(2,1,32,32)).abs().max() == 0
    assert torch.isfinite(laplace(torch.ones(2,1,1,1))).all()


def test_invalid_feature_contract_raises():
    module = AsymmetricFusion()
    c,p = features(),features()
    with pytest.raises(ValueError):
        module(c[:3],p,None)
    p[0] = p[0][:,:4]
    with pytest.raises(ValueError):
        module(c,p,None)


def test_state_roundtrip_and_no_forward_tensor_cache():
    m = AsymmetricFusion(channels=(8,)*4, ranks=(8,)*4).eval()
    c,p=features(16,(8,)*4),features(16,(8,)*4)
    image=torch.randn(2,3,64,64)
    out,_=m(c,p,image)
    fresh=copy.deepcopy(m)
    fresh.load_state_dict(m.state_dict(), strict=True)
    again,_=fresh(c,p,image)
    assert all(torch.equal(a,b) for a,b in zip(out,again))
