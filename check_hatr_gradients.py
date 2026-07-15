import copy

import torch
import torch.nn.functional as F

from models.dual_decoder_hatr_task_residual import DualDecoderHATRTaskResidual


def param_max_update(before, after):
    return max((before[k] - after[k]).abs().max().item() for k in before)


def bce_dice_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    probs = torch.sigmoid(logits)
    inter = (probs * mask).sum()
    dice = 1 - (2 * inter + 1.0) / (probs.sum() + mask.sum() + 1.0)
    return bce + dice


def main():
    torch.manual_seed(0)
    model = DualDecoderHATRTaskResidual(use_deep_supervision=False)
    batch = {'ct': torch.randn(2, 3, 128, 128), 'pet': torch.randn(2, 3, 128, 128), 'mask': torch.randint(0, 2, (2, 1, 128, 128)).float()}

    outputs = model(batch['ct'], batch['pet'], target_size=batch['mask'].shape[-2:], forward_mode='full')
    loss_seg = bce_dice_loss(outputs['logits'], batch['mask'])
    model.zero_grad(); loss_seg.backward()
    assert any(p.grad is not None for p in model.enc_ct.parameters())
    assert any(p.grad is not None for p in model.enc_pet.parameters())
    assert any(p.grad is not None for p in model.full_decoder.parameters())
    assert all((p.grad is None) for p in model.hatr_recovery.parameters())

    model.zero_grad()
    p_f = torch.sigmoid(outputs['logits'].detach())
    p_c = torch.sigmoid(outputs['hatr_counterfactual_logits'].detach())
    y = batch['mask']
    e_f = (p_f - y).pow(2)
    e_c = (p_c - y).pow(2)
    advantage = F.relu(e_c - e_f) / (e_c + e_f + 1e-6)
    hatr_loss = 0
    for full_state, ct_state, pred in zip(outputs['hatr_full_states'], outputs['hatr_counterfactual_states'], outputs['hatr_pred_residuals']):
        adv_s = F.interpolate(advantage, size=full_state.shape[-2:], mode='bilinear', align_corners=False)
        target_s = adv_s * (full_state.detach() - ct_state.detach())
        hatr_loss = hatr_loss + F.smooth_l1_loss(pred.float(), target_s.float())
    hatr_loss = hatr_loss / 4.0
    hatr_loss.backward()
    assert any(p.grad is not None for p in model.hatr_recovery.parameters())
    assert all((p.grad is None) for p in model.enc_ct.parameters())
    assert all((p.grad is None) for p in model.full_decoder.parameters())

    model.zero_grad()
    miss_out = model(batch['ct'], None, target_size=batch['mask'].shape[-2:], forward_mode='missing')
    miss_loss = bce_dice_loss(miss_out['logits'], batch['mask'])
    miss_loss.backward()
    assert any(p.grad is not None for p in model.enc_ct.parameters())
    assert any(p.grad is not None for p in model.missing_decoder.parameters())
    assert any(p.grad is not None for p in model.correction_adapter4.parameters())

    before = {k: v.detach().clone() for k, v in model.hatr_recovery.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step(); opt.zero_grad()
    miss_out2 = model(batch['ct'], None, target_size=batch['mask'].shape[-2:], forward_mode='missing')
    miss_loss2 = bce_dice_loss(miss_out2['logits'], batch['mask'])
    miss_loss2.backward()
    assert any((p.grad is not None and p.grad.abs().sum().item() > 0) for p in model.hatr_recovery.parameters())

    before_full = {k: v.detach().clone() for k, v in model.full_decoder.state_dict().items()}
    before_ct = {k: v.detach().clone() for k, v in model.enc_ct.state_dict().items()}
    before_pet = {k: v.detach().clone() for k, v in model.enc_pet.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(); hatr_loss.backward(); opt.step()
    after = model.full_decoder.state_dict(); after_ct = model.enc_ct.state_dict(); after_pet = model.enc_pet.state_dict()
    assert param_max_update(before_full, after) == 0
    assert param_max_update(before_ct, after_ct) == 0
    assert param_max_update(before_pet, after_pet) == 0
    assert param_max_update(before, model.hatr_recovery.state_dict()) > 0
    print('gradient checks passed')


if __name__ == '__main__':
    main()
