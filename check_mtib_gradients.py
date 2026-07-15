# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from models.dual_decoder_multiscale_task_increment_bank import DualDecoderMultiScaleTaskIncrementBank


def _make_batch(device):
    ct = torch.randn(2, 1, 128, 128, device=device)
    pet = torch.randn(2, 1, 128, 128, device=device)
    mask = (torch.rand(2, 1, 128, 128, device=device) > 0.5).float()
    return ct, pet, mask


def _named_param(model, name):
    return dict(model.named_parameters())[name]


def _print_param_status(model, names):
    for name in names:
        p = _named_param(model, name)
        grad = p.grad
        has_grad = grad is not None
        grad_finite = bool(torch.isfinite(grad).all()) if has_grad else False
        grad_norm = float(grad.float().norm().detach().cpu()) if has_grad else 0.0
        print(name, 'requires_grad=', p.requires_grad, 'has_grad=', has_grad, 'grad_finite=', grad_finite, 'grad_norm=', grad_norm)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualDecoderMultiScaleTaskIncrementBank(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        mtib_stages='all',
        mtib_num_tokens=8,
        mtib_temperature=0.07,
    ).to(device)
    ct, pet, mask = _make_batch(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # A
    model.zero_grad(set_to_none=True)
    out = model(ct, pet, forward_mode='full')
    loss = loss_fn(out['logits'], mask)
    loss.backward()
    print('Test A')
    _print_param_status(model, ['enc_ct.model.patch_embeddings.projection.weight'] if False else [n for n, _ in model.named_parameters()][:3])

    # B
    model.zero_grad(set_to_none=True)
    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    full_feats, true_inc = model._true_increment(ct_feats, pet_feats, None)
    _, bank_loss, _ = model.mtib.forward_full(full_feats, true_inc)
    bank_loss.backward()
    print('Test B bank_loss', float(bank_loss))

    # C
    model.zero_grad(set_to_none=True)
    _, comp_loss, _ = model.mtib.forward_ct_comp(ct_feats, true_inc)
    comp_loss.backward()
    print('Test C comp_loss', float(comp_loss))

    # D
    model.zero_grad(set_to_none=True)
    out = model(ct, pet, forward_mode='missing')
    miss_loss = loss_fn(out['logits'], mask)
    miss_loss.backward()
    print('Test D miss_loss', float(miss_loss))
    print('gradient_tests_done')


if __name__ == '__main__':
    main()
