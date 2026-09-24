# -*- coding: utf-8 -*-
"""Unit tests for PETCTFullMFFA and its Haar helpers.

These tests describe the module contract only. They do not exercise the real
backbones or the whole network; see tests/test_mffa_baseline_integration.py and
tools/smoke_petct_full_mffa.py for the integrated checks.

Optional source cross-check (only when a WFANet checkout is provided):

    WFANET_REFERENCE_PATH=/path/to/WFANet/net_torch.py python -m pytest -q tests/test_petct_full_mffa.py
"""
import importlib.util
import os

import pytest
import torch

from models.petct_full_mffa import (
    PETCTFullMFFA,
    MFFAScale,
    GlobalFrequencyAttention,
    haar_dwt2,
    haar_ll,
    haar_idwt2,
)


def _tiny_model(**kwargs):
    return PETCTFullMFFA(channels=(8, 16, 24, 32), embed_dim=8, num_heads=2, **kwargs)


def _feats(batch=4, channels=(8, 16, 24, 32), sizes=(16, 8, 4, 2), requires_grad=False):
    return [torch.randn(batch, c, s, s, requires_grad=requires_grad) for c, s in zip(channels, sizes)]


# ---------------------------------------------------------------- Haar operators

def test_haar_idwt_exact_inverse():
    x = torch.randn(2, 3, 16, 16)
    assert torch.allclose(haar_idwt2(*haar_dwt2(x)), x, atol=1e-6)


def test_haar_ll_matches_dwt_ll():
    x = torch.randn(2, 5, 8, 8)
    assert torch.allclose(haar_ll(x), haar_dwt2(x)[0], atol=1e-7)


def test_haar_idwt_accepts_odd_band_shapes():
    ll = torch.randn(1, 2, 3, 5)
    lh, hl, hh = (torch.randn_like(ll) for _ in range(3))
    out = haar_idwt2(ll, lh, hl, hh)
    assert out.shape[-2:] == (6, 10)


def test_haar_rejects_odd_input():
    with pytest.raises(ValueError):
        haar_dwt2(torch.randn(1, 1, 7, 8))


# ---------------------------------------------------------------- attention unit

def test_global_attention_is_global_not_windowed():
    torch.manual_seed(0)
    attn = GlobalFrequencyAttention(dim=8, num_heads=2)
    base = torch.randn(1, 8, 4, 4)
    perturbed_v = base.clone()
    # Perturb the Value of a single token with a random vector (avoids
    # LayerNorm shift-invariance, which would cancel a constant offset).
    perturbed_v[..., 0, 0] += torch.randn(8) * 5.0
    with torch.no_grad():
        a = attn(base, base, base)
        b = attn(base, base, perturbed_v)
    # Every output token attends to every input token, so a single Value token
    # perturbed changes the far corner too. A small-window/local attention
    # would leave the far token untouched.
    assert not torch.allclose(a, b, atol=1e-6)
    assert (a - b).abs()[..., 3, 3].sum() > 0


def test_attention_rejects_shape_mismatch():
    attn = GlobalFrequencyAttention(dim=8, num_heads=2)
    with pytest.raises(ValueError):
        attn(torch.randn(1, 8, 4, 4), torch.randn(1, 8, 3, 3), torch.randn(1, 8, 4, 4))


# ---------------------------------------------------------------- module contract

def test_default_parameter_count_is_510720():
    model = PETCTFullMFFA()
    assert model.parameter_report()["total"] == 510720


def test_full_returns_ct_plus_nonzero_delta():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    ct = _feats()
    pet = _feats()
    state = torch.ones(ct[0].shape[0], dtype=torch.long)
    with torch.no_grad():
        fused, delta = model(ct, pet, state, return_delta=True)
    for f, c, d in zip(fused, ct, delta):
        assert torch.allclose(f, c + d, atol=1e-6)
        assert d.abs().sum() > 0  # standard init, not zero-output


def test_missing_is_exact_ct_identity():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    ct = _feats()
    with torch.no_grad():
        fused = model.forward_missing(ct)
        fused_delta, delta = model.forward_missing(ct, return_delta=True)
    for f, f2, c, d in zip(fused, fused_delta, ct, delta):
        assert torch.equal(f, c)
        assert torch.equal(f2, c)
        assert torch.count_nonzero(d) == 0


def test_missing_ignores_supplied_pet():
    model = _tiny_model().eval()
    ct = _feats()
    pet_a = _feats()
    pet_b = [t * 10.0 for t in pet_a]
    state = torch.zeros(ct[0].shape[0], dtype=torch.long)
    with torch.no_grad():
        out_a = model(ct, pet_a, state)
        out_b = model(ct, pet_b, state)
    for a, b, c in zip(out_a, out_b, ct):
        assert torch.equal(a, c)
        assert torch.equal(b, c)


def test_mixed_routes_full_and_missing_rows():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = 4
    ct = _feats(batch=batch)
    pet = _feats(batch=batch)
    state = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    with torch.no_grad():
        fused = model(ct, pet, state)
    for f, c in zip(fused, ct):
        assert torch.allclose(f[state == 0], c[state == 0], atol=1e-6)
        assert not torch.allclose(f[state == 1], c[state == 1], atol=1e-6)


def test_full_rows_receive_pet_gradient_missing_rows_do_not():
    torch.manual_seed(0)
    model = _tiny_model().train()
    ct = _feats(requires_grad=False)
    pet = _feats(requires_grad=True)
    state = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    fused = model(ct, pet, state)
    sum(t.pow(2).mean() for t in fused).backward()
    for p in pet:
        assert p.grad is not None
        assert torch.count_nonzero(p.grad[state == 0]) == 0
        assert torch.count_nonzero(p.grad[state == 1]) > 0
    assert all(param.grad is not None for param in model.parameters())


def test_checkpoint_attention_matches_plain_in_eval():
    torch.manual_seed(0)
    a = _tiny_model(checkpoint_attention=False).eval()
    b = _tiny_model(checkpoint_attention=True).eval()
    b.load_state_dict(a.state_dict())
    ct, pet = _feats(requires_grad=True), _feats(requires_grad=True)
    state = torch.ones(ct[0].shape[0], dtype=torch.long)
    with torch.no_grad():
        fa = a(ct, pet, state)
        fb = b(ct, pet, state)
    for x, y in zip(fa, fb):
        assert torch.allclose(x, y, atol=1e-6)


def test_all_missing_returns_identity_without_touching_pet():
    model = _tiny_model().eval()
    ct = _feats()
    state = torch.zeros(ct[0].shape[0], dtype=torch.long)
    # pet_feats=None is allowed only when every row is Missing.
    with torch.no_grad():
        fused, delta = model(ct, None, state, return_delta=True)
    for f, c, d in zip(fused, ct, delta):
        assert torch.equal(f, c)
        assert torch.count_nonzero(d) == 0


# ---------------------------------------------------------------- validation guards

def test_rejects_floating_state():
    model = _tiny_model()
    ct, pet = _feats(), _feats()
    with pytest.raises(ValueError):
        model(ct, pet, torch.ones(ct[0].shape[0]))


def test_rejects_state_out_of_range():
    model = _tiny_model()
    ct, pet = _feats(), _feats()
    with pytest.raises(ValueError):
        model(ct, pet, torch.full((ct[0].shape[0],), 2, dtype=torch.long))


def test_rejects_channel_mismatch():
    model = _tiny_model()
    ct = _feats()
    pet = [t[:, :1] for t in ct]  # wrong channels
    with pytest.raises(ValueError):
        model(ct, pet, torch.ones(ct[0].shape[0], dtype=torch.long))


def test_requires_pet_when_any_full_row():
    model = _tiny_model()
    ct = _feats()
    with pytest.raises(ValueError):
        model(ct, None, torch.ones(ct[0].shape[0], dtype=torch.long))


def test_mffa_scale_rejects_pet_shape_mismatch():
    scale = MFFAScale(channels=8, embed_dim=8, num_heads=2)
    with pytest.raises(ValueError):
        scale(torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 5))


# ---------------------------------------------------------------- optional source check

def test_reference_haar_matches_source_when_available():
    path = os.getenv("WFANET_REFERENCE_PATH")
    if not path or not os.path.exists(path):
        pytest.skip("WFANET_REFERENCE_PATH not set")
    spec = importlib.util.spec_from_file_location("wfanet_net_torch", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    if not hasattr(ref, "DWT"):
        pytest.skip("reference module has no DWT entry point")
    x = torch.randn(1, 2, 8, 8)
    bands = haar_dwt2(x)
    ref_out = ref.DWT(x)
    ref_bands = ref_out if isinstance(ref_out, (list, tuple)) else list(ref_out)
    assert len(ref_bands) == 4
    for mine, theirs in zip(bands, ref_bands):
        assert torch.allclose(mine, theirs, atol=1e-6)
