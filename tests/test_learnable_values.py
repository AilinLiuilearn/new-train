"""Gradient and lifecycle contracts for the learnable affine dictionary."""

import copy

import pytest
import torch

from models.spatial_modulation_bank import SpatialModulationScale, UnifiedModulationBank


def test_keys_count_all_positions_and_never_overwrite_values():
    bank = UnifiedModulationBank(2, clusters=2, chunk_size=3)
    x = torch.cat([torch.eye(2).repeat(5, 1), torch.zeros(2, 2)])
    fg = torch.arange(12).float() / 12
    bank.add_all(x, fg, first_epoch=True)
    assert int(bank.accepted) == 12 and bank.count.sum() == 12
    assert bank.fg_mass.sum() == fg.double().sum()
    assert isinstance(bank.gamma, torch.nn.Parameter)
    assert isinstance(bank.beta, torch.nn.Parameter)
    assert "keys" not in dict(bank.named_parameters())
    with torch.no_grad():
        bank.gamma.fill_(0.3)
        bank.beta.fill_(0.4)
    before = copy.deepcopy(bank.state_dict())
    unit = torch.nn.functional.normalize(x + 0.2, dim=1)
    ids = (unit @ bank.keys.T).argmax(1)
    expected = bank.keys.clone()
    for j in range(2):
        mean = unit[ids == j].mean(0)
        expected[j] = torch.nn.functional.normalize(
            0.99 * expected[j] + 0.01 * mean, dim=0
        )
    bank.add_all(x + 0.2, fg, first_epoch=False)
    assert torch.allclose(bank.keys, expected)
    assert bank.count.sum() == 24
    assert torch.equal(bank.gamma, before["gamma"])
    assert torch.equal(bank.beta, before["beta"])
    assert bank.keys.grad is None
    other = UnifiedModulationBank(2, clusters=2, chunk_size=3)
    other.add_all(x, 1 - fg, first_epoch=True)
    assert torch.equal(other.keys, before["keys"])
    assert torch.equal(other.count, before["count"])


def test_warmup_and_auxiliary_gradients():
    scale = SpatialModulationScale(4, clusters=2)
    ct = torch.randn(2, 4, 4, 4, requires_grad=True)
    full = (ct.detach() + 1).requires_grad_()
    mask = torch.ones(2, 1, 16, 16)
    loss, pending = scale.supervise(ct, full, mask, True)
    assert not loss.requires_grad and loss == 0
    assert torch.equal(scale(ct), ct)
    scale.bank.add_all(*pending, first_epoch=True)
    assert scale.bank.gamma.grad is None and torch.count_nonzero(scale.bank.gamma) == 0
    scale.epoch.fill_(2)
    assert torch.equal(scale(ct), ct)
    loss, _ = scale.supervise(ct, full, mask, True)
    expected = (
        (ct.detach() - full.detach()).abs().mean((1, 2, 3))
        / (full.detach().square().mean((1, 2, 3)).sqrt() + 1e-6)
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert ct.grad is None and full.grad is None and scale.bank.keys.grad is None
    assert scale.bank.beta.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(scale.parameters(), lr=0.05)
    optimizer.step()
    improved, _ = scale.supervise(ct, full, mask, True)
    assert improved < loss.detach()
    scale.zero_grad(set_to_none=True)
    off, _ = scale.supervise(ct, full, mask, False)
    assert not off.requires_grad and off == 0
    scale(ct).square().mean().backward()
    assert scale.bank.gamma.grad.abs().sum() > 0
    assert scale.bank.beta.grad.abs().sum() > 0
    assert ct.grad is not None


def test_spatial_affine_and_differentiable_query():
    scale = SpatialModulationScale(2, clusters=2, temperature=0.5)
    scale.bank.add_all(torch.eye(2), torch.zeros(2), first_epoch=True)
    scale.epoch.fill_(2)
    with torch.no_grad():
        scale.bank.keys.copy_(torch.eye(2))
        scale.bank.gamma.copy_(torch.tensor([[0.1, 0.2], [-0.2, -0.1]]))
        scale.bank.beta.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    ct = torch.randn(1, 2, 5, 5, requires_grad=True)
    query = torch.nn.functional.normalize(
        scale.describe(ct).flatten(2).transpose(1, 2), dim=-1
    )
    query.retain_grad()
    weights = (query @ scale.bank.keys.T / 0.5).softmax(-1)
    g = (weights @ scale.bank.gamma).transpose(1, 2).reshape_as(ct)
    b = (weights @ scale.bank.beta).transpose(1, 2).reshape_as(ct)
    expected = ct * (1 + g) + b
    diagnostics = {}
    actual = scale(ct, diagnostics)
    assert torch.allclose(actual, expected, atol=1e-6)
    expected.sum().backward()
    assert query.grad.abs().sum() > 0 and scale.bank.keys.grad is None
    assert b.std(dim=(-2, -1)).sum() > 0
    assert 0 <= diagnostics["attention_entropy_normalized"] <= 1.000001
    assert torch.equal(scale(ct, read_mode="off"), ct)
    assert not torch.allclose(scale(ct, read_mode="shuffle"), actual)


def test_validation_and_parameters():
    with pytest.raises(ValueError):
        SpatialModulationScale(2, temperature=0)
    bank = UnifiedModulationBank(2, clusters=2)
    with pytest.raises(ValueError, match="finite"):
        bank.add_all(torch.full((3, 2), float("nan")), torch.zeros(3), True)
    assert not bool(bank.initialized)
    from models.spatial_modulation_bank import SpatialModulationMemory

    memory = SpatialModulationMemory()
    assert sum(p.numel() for p in memory.parameters()) == 16384
