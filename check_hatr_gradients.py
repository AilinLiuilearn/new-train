import copy

import torch
import torch.nn.functional as F

from models.dual_decoder_hatr_task_residual import DualDecoderHATRTaskResidual


def bce_dice_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probs = torch.sigmoid(logits)
    inter = (probs * mask).sum()
    dice = 1 - (2 * inter + 1.0) / (probs.sum() + mask.sum() + 1.0)
    return bce + dice


def named_param_clone(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def changed(before, model, prefix):
    return any(not torch.allclose(before[n], p.detach()) for n, p in model.named_parameters() if n.startswith(prefix))


def main():
    torch.manual_seed(0)
    model = DualDecoderHATRTaskResidual(use_deep_supervision=False)
    batch = {'ct': torch.randn(2, 3, 128, 128), 'pet': torch.randn(2, 3, 128, 128), 'mask': torch.randint(0, 2, (2, 1, 128, 128)).float()}

    model.zero_grad(set_to_none=True)
    out = model(batch['ct'], batch['pet'], target_size=batch['mask'].shape[-2:], forward_mode='full')
    loss = bce_dice_loss(out['logits'], batch['mask'])
    loss.backward()
    assert any(p.grad is not None for p in model.enc_ct.parameters())
    assert any(p.grad is not None for p in model.enc_pet.parameters())
    assert any(p.grad is not None for p in model.full_decoder.parameters())
    assert all(p.grad is None for p in model.hatr_recovery.parameters())

    model.zero_grad(set_to_none=True)
    cf_loss = bce_dice_loss(out['hatr_counterfactual_logits'], batch['mask'])
    cf_loss.backward()
    assert any(p.grad is not None for p in model.full_decoder.parameters())
    assert all(p.grad is None for p in model.enc_ct.parameters())
    assert all(p.grad is None for p in model.enc_pet.parameters())
    assert all(p.grad is None for p in model.hatr_recovery.parameters())
    assert all(p.grad is None for p in model.missing_decoder.parameters())

    model.zero_grad(set_to_none=True)
    p_f = torch.sigmoid(out['hatr_teacher_full_logits'].detach())
    p_c = torch.sigmoid(out['hatr_counterfactual_logits'].detach())
    y = batch['mask']
    advantage = F.relu((p_c - y).pow(2) - (p_f - y).pow(2)) / ((p_c - y).pow(2) + (p_f - y).pow(2) + 1e-6)
    hatr_loss = 0
    for i, (ts, cs, pr) in enumerate(zip(out['hatr_teacher_full_states'], out['hatr_counterfactual_states'], out['hatr_pred_residuals'])):
        target = F.interpolate(advantage, size=ts.shape[-2:], mode='bilinear', align_corners=False) * (ts.detach() - cs.detach())
        hatr_loss = hatr_loss + F.smooth_l1_loss(pr.float(), target.float())
    hatr_loss = hatr_loss / 4.0
    hatr_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.hatr_recovery.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.enc_ct.parameters())
    assert all(p.grad is None for p in model.full_decoder.parameters())

    model.zero_grad(set_to_none=True)
    miss_out = model(batch['ct'], None, target_size=batch['mask'].shape[-2:], forward_mode='missing')
    miss_loss = bce_dice_loss(miss_out['logits'], batch['mask'])
    miss_loss.backward()
    assert any(p.grad is not None for p in model.enc_ct.parameters())
    assert any(p.grad is not None for p in model.ct_align.parameters())
    assert any(p.grad is not None for p in model.missing_decoder.parameters())
    assert any(p.grad is not None for p in model.correction_adapter1.parameters())
    assert all(p.grad is None for p in model.hatr_recovery.parameters())
    assert all(p.grad is None for p in model.full_decoder.parameters())

    before_full = named_param_clone(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    after_full = named_param_clone(model)
    assert changed(before_full, model, 'full_decoder.') is False

    print('gradient checks passed')


if __name__ == '__main__':
    main()
