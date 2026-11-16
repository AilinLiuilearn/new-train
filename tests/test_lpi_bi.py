"""Independent coordinate reference, soft targets and delayed task gradients."""
import math
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from variants.lpi_bi.front import Direction, Scale, LPIBI, spatial_descriptor, spatial_bias, valid_candidates
from variants.lpi_bi.losses import localization_loss, squared_dice_loss

torch.set_num_threads(2)


def test_parameter_count_and_uniform_or_small_foreground_pooling():
    assert sum(p.numel() for p in LPIBI().parameters()) == 654608
    x = torch.randn(2, 8, 5, 7)
    for value in [0., 1e-8, .5, 1.]:
        v, _ = spatial_descriptor(x, torch.full((2, 1, 5, 7), value))
        torch.testing.assert_close(v, x.mean((-2, -1)), atol=1e-6, rtol=1e-6)
    for side in [4, 16, 64]:
        x = torch.zeros(1, 1, side, side)
        x[:, :, 0, 0] = 1
        v, _ = spatial_descriptor(x, x)
        assert float(v) > .999


@pytest.mark.parametrize('shape', [(1, 1), (2, 3), (5, 7)])
def test_valid_candidates_and_uniform_bias(shape):
    h, w = shape
    valid = valid_candidates(h, w, 'cpu')
    assert valid[:, :, 4].all()
    assert int(valid[:, :, :, 0].sum()) == min(2, h) * min(2, w)
    for value in [0., 1e-8, .5, 1.]:
        bias = spatial_bias(torch.full((2, 1, h, w), value), valid)
        assert torch.isfinite(bias).all() and float(bias.abs().max()) < 2e-6


def test_direction_matches_independent_coordinate_loop():
    torch.manual_seed(22)
    layer = Direction(4)
    torch.nn.init.normal_(layer.out.weight)
    torch.nn.init.normal_(layer.out.bias)
    c, p = torch.randn(2, 4, 2, 3), torch.randn(2, 4, 2, 3)
    gc, gp = torch.rand(2, 4) + .5, torch.rand(2, 4) + .5
    s = torch.rand(2, 1, 2, 3)
    actual, _ = layer(c, p, gc, gp, s, valid_candidates(2, 3, 'cpu'))
    q = layer.q(c * gc[:, :, None, None]).reshape(2, 4, 8, 2, 3)
    k = layer.k(p * gp[:, :, None, None]).reshape(2, 4, 8, 2, 3)
    v = layer.v(p).reshape(2, 4, 8, 2, 3)
    message = torch.zeros_like(q)
    for y in range(2):
        for x in range(3):
            coordinates = [(yy, xx) for yy in range(max(0, y-1), min(2, y+2))
                           for xx in range(max(0, x-1), min(3, x+2))]
            keys = torch.stack([k[:, :, :, yy, xx] for yy, xx in coordinates], -1)
            values = torch.stack([v[:, :, :, yy, xx] for yy, xx in coordinates], -1)
            responses = torch.stack([s[:, 0, yy, xx] for yy, xx in coordinates], -1)
            logs = (responses + 1e-6).log()
            prior = responses.amax(-1, keepdim=True) * (logs - logs.mean(-1, keepdim=True)).tanh()
            scores = (q[:, :, :, y, x, None] * keys).sum(2) / math.sqrt(8)
            weights = (scores + layer.bias_logit.sigmoid() * prior[:, None]).softmax(-1)
            message[:, :, :, y, x] = (weights[:, :, None] * values).sum(-1)
    expected = layer.out(message.reshape(2, 32, 2, 3))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_soft_occupancy_and_empty_loss_policy():
    p = torch.tensor([[[[.25, .75]]]], requires_grad=True)
    dice = squared_dice_loss(p, p.detach(), torch.tensor([True]))
    assert float(dice) == 0
    assert torch.autograd.grad(dice.sum(), p)[0].abs().max() < 1e-6
    target = torch.zeros(2, 1, 4, 4)
    target[1, 0, 0, 0] = 1
    heads = {m: [torch.zeros(2, 1, 1, 1, requires_grad=True) for _ in range(4)]
             for m in ['ct', 'pet']}
    loss, details = localization_loss(heads, target)
    occupancy = 1/16
    expected_dice = 1-(2*.5*occupancy+1e-6)/(.25+occupancy**2+1e-6)
    assert abs(float(loss) - (math.log(2)+expected_dice/2)) < 1e-6
    loss.backward()
    assert all(z.grad[0].item() > 0 for z in heads['ct'])
    assert float(details['loc_nonempty_fraction']) == .5


def test_identity_and_task_gradients_after_multiple_real_updates():
    torch.manual_seed(2023)
    block = Scale(16)
    c, p = torch.randn(2, 16, 5, 7), torch.randn(2, 16, 5, 7)
    goal_c, goal_p = torch.randn_like(c), torch.randn_like(p)
    opt = torch.optim.AdamW(block.parameters(), lr=8e-5, weight_decay=0)
    seen = {}
    for step in range(6):
        opt.zero_grad(set_to_none=True)
        cc, pp, zc, zp, _ = block(c, p)
        if step == 0:
            assert torch.equal(c, cc) and torch.equal(p, pp)
        # Task only: distinguishes routing gradients from auxiliary head gradients.
        loss = (cc-goal_c).square().mean() + (pp-goal_p).square().mean()
        loss.backward()
        for name, param in block.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                norm = float(param.grad.abs().sum())
                seen[name] = max(seen.get(name, 0), norm)
                if step == 0 and '.out.' not in name:
                    assert norm == 0
        opt.step()
    assert all(seen.get(name, 0) > 0 for name, _ in block.named_parameters()), seen
