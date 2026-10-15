"""Design contract: four scales, independent switches, signed zero-init residuals."""
import copy

import pytest
import torch
from torch.nn import functional as F

from models.components.lesion_guided_fusion import (
    LesionGuidedFusion, auxiliary_lesion_loss, STAGE_CHANNELS)
from utils.seg_losses import BCEDiceLoss


def features(actual=False):
    sizes = (128, 64, 32, 16) if actual else (16, 8, 4, 2)
    return [torch.randn(2, c, s, s, requires_grad=True)
            for c, s in zip(STAGE_CHANNELS, sizes)]


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('enabled', [True, False])
def test_actual_four_scale_shapes_exact_add_and_aux(training, enabled):
    module = LesionGuidedFusion(enabled=enabled).train(training)
    ct, pet = features(True), features(True)
    with torch.no_grad():
        out, aux = module.forward_with_aux(ct, pet, return_aux=True)
    assert len(out) == 4 and len(aux) == (4 if enabled else 0)
    for i, (c, p, y) in enumerate(zip(ct, pet, out)):
        torch.testing.assert_close(y, c + p, atol=0, rtol=0)
        if enabled:
            assert aux[i].shape == (2, 1, *c.shape[-2:])
    if enabled:
        assert len({id(m) for m in module.blocks}) == 4
    else:
        assert not module.state_dict()


def test_two_steps_unblock_gates_and_both_corrections():
    module = LesionGuidedFusion()
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    ct, pet = features(), features()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = module(ct, pet)
        sum(x.square().mean() for x in out).backward()
        for block in module.blocks:
            for proj in (block.out_ct, block.out_pet):
                assert torch.isfinite(proj.weight.grad).all()
                assert proj.weight.grad.abs().sum() > 0
            if step == 1:
                assert block.gate[-1].weight.grad.abs().sum() > 0
                assert block.mask_head[-1].weight.grad.abs().sum() > 0
                assert block.ct_context.project[0].weight.grad.abs().sum() > 0
                assert block.pet_context.project[0].weight.grad.abs().sum() > 0
        optimizer.step()
    for c, p in zip(ct, pet):
        assert torch.isfinite(c.grad).all() and c.grad.abs().sum() > 0
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_auxiliary_only_updates_evidence_but_not_residual_or_gate():
    module = LesionGuidedFusion(correction_enabled=False)
    ct, pet = features(), features()
    out, aux = module.forward_with_aux(ct, pet, return_aux=True)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 17, 19] = 1  # retain a small foreground at original resolution
    loss, stats = auxiliary_lesion_loss(aux, target, weight=.1)
    loss.backward()
    for c, p, y, block in zip(ct, pet, out, module.blocks):
        torch.testing.assert_close(y, c + p, atol=0, rtol=0)
        assert block.mask_head[-1].weight.grad.abs().sum() > 0
        assert block.out_ct.weight.grad is None and block.out_pet.weight.grad is None
        assert all(p.grad is None for p in block.gate.parameters())
    assert stats['aux_raw'] > 0 and stats['aux_weighted'] > 0


def test_guidance_switch_removes_prediction_from_correction_computation():
    ct, pet = features(), features()
    for guidance in (False, True):
        module = LesionGuidedFusion(guidance_enabled=guidance).eval()
        for block in module.blocks:
            torch.nn.init.normal_(block.out_ct.weight, std=.1)
            torch.nn.init.normal_(block.out_pet.weight, std=.1)
        before = module(ct, pet)
        for block in module.blocks:
            with torch.no_grad():
                block.mask_head[-1].weight.zero_()
                block.mask_head[-1].bias.fill_(15.)
        after = module(ct, pet)
        if guidance:
            assert any(not torch.equal(a, b) for a, b in zip(before, after))
        else:
            for a, b in zip(before, after):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_four_scale_aux_loss_matches_full_resolution_baseline_criterion():
    aux = [torch.randn(2, 1, s, s, requires_grad=True) for s in (16, 8, 4, 2)]
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    expected = torch.stack([BCEDiceLoss()(F.interpolate(x, (64, 64),
        mode='bilinear', align_corners=False), target)[0] for x in aux]).mean()
    loss, stats = auxiliary_lesion_loss(aux, target, weight=.1)
    torch.testing.assert_close(loss, expected * .1)
    torch.testing.assert_close(stats['aux_raw'], expected.detach())
    loss.backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in aux)


@pytest.mark.parametrize('enabled,weight', [(False, .1), (True, 0.)])
def test_loss_switch_has_exactly_zero_contribution(enabled, weight):
    target = torch.zeros(2, 1, 64, 64)
    loss, stats = auxiliary_lesion_loss([], target, enabled=enabled, weight=weight)
    assert loss.item() == 0 and not loss.requires_grad
    assert stats['aux_raw'].item() == stats['aux_weighted'].item() == 0


@pytest.mark.parametrize('foreground', [False, True])
def test_all_background_or_foreground_aux_loss_is_finite(foreground):
    aux = [torch.full((2, 1, s, s), -100., requires_grad=True) for s in (16, 8, 4, 2)]
    loss, _ = auxiliary_lesion_loss(aux, torch.full((2, 1, 64, 64), float(foreground)))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in aux)


def test_spatial_alignment_and_no_stateful_graph_cache():
    module = LesionGuidedFusion()
    ct, pet = features(), features()
    pet[0] = torch.randn(2, 64, 8, 8)
    expected = ct[0] + F.interpolate(pet[0], ct[0].shape[-2:], mode='bilinear', align_corners=False)
    result = module(ct, pet)
    torch.testing.assert_close(result[0], expected, atol=0, rtol=0)
    assert not any(torch.is_tensor(v) for v in vars(module).values())
    copied = copy.deepcopy(module)
    copied.load_state_dict(module.state_dict(), strict=True)


def test_input_validation():
    ct, pet = features(), features()
    with pytest.raises(ValueError): LesionGuidedFusion(widths=(16, 32))
    with pytest.raises(ValueError): LesionGuidedFusion(widths=(16, 32, 0, 64))
    with pytest.raises(TypeError): LesionGuidedFusion(enabled='false')
    with pytest.raises(ValueError): LesionGuidedFusion()(ct[:3], pet)
    pet[0] = torch.randn(1, 64, 16, 16)
    with pytest.raises(ValueError): LesionGuidedFusion()(ct, pet)
    with pytest.raises(ValueError): auxiliary_lesion_loss([], torch.zeros(2, 1, 64, 64))
    with pytest.raises(ValueError): auxiliary_lesion_loss([], torch.zeros(2, 1, 64, 64), weight=float('nan'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_amp_forward_and_two_step_backward():
    module = LesionGuidedFusion().cuda()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    ct, pet = [[x.detach().cuda().requires_grad_() for x in features()] for _ in range(2)]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            out, aux = module.forward_with_aux(ct, pet, return_aux=True)
            extra, _ = auxiliary_lesion_loss(aux, torch.rand(2, 1, 64, 64, device='cuda'))
            loss = sum(x.float().square().mean() for x in out) + extra
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
        scaler.step(optimizer)
        scaler.update()
