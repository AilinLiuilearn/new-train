# -*- coding: utf-8 -*-

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher


def _print_status(model, names):
    for name in names:
        obj = model
        for part in name.split('.'):
            obj = getattr(obj, part)
        has_grad = False
        grad_norm = 0.0
        finite = True
        for p in obj.parameters() if hasattr(obj, 'parameters') else []:
            if p.grad is None:
                continue
            has_grad = True
            g = p.grad.detach().float()
            finite = finite and torch.isfinite(g).all().item()
            grad_norm += float((g ** 2).sum().item())
        print(name, 'has_grad=', has_grad, 'grad_finite=', finite, 'grad_norm=', grad_norm ** 0.5)


def main():
    cfg = SegMDTConfig.parse_arguments()
    cfg.model_arch = 'dual_decoder_pg_mtr_c2maot_ot'
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None
    cfg.use_deep_supervision = False
    cfg.deep_supervision = False
    nets = build_mdt_seg_teacher(cfg)
    model = nets['model']
    model.train()
    ct = torch.randn(2, 3, 128, 128, device='cuda' if torch.cuda.is_available() else 'cpu')
    pet = torch.randn_like(ct)
    mask = torch.randint(0, 2, (2, 1, 128, 128), device=ct.device).float()
    model.zero_grad(set_to_none=True)
    out = model(ct=ct, pet=pet, forward_mode='full')
    loss_seg, _ = model.full_decoder(out['logits'], ct.shape[-2:]), None
    loss = out['aux_losses']['pg_mtr_ot_loss']
    loss = loss + out['logits'].mean() * 0.0
    loss.backward()
    _print_status(model, ['enc_ct', 'ct_align', 'enc_pet', 'full_decoder', 'pg_mtr.shared_memory_tokens', 'pg_mtr.shared_token_key', 'pg_mtr.shared_token_value'])
    model.zero_grad(set_to_none=True)
    out = model(ct=ct, pet=None, forward_mode='missing')
    out['logits'].mean().backward()
    _print_status(model, ['enc_ct', 'ct_align', 'missing_decoder', 'pg_mtr.shared_memory_tokens', 'pg_mtr.shared_token_key', 'pg_mtr.shared_token_value'])


if __name__ == '__main__':
    main()
