# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from models.dual_decoder_multiscale_task_increment_bank import DualDecoderMultiScaleTaskIncrementBank


def _make_model():
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
    )
    return model


def main():
    torch.manual_seed(0)
    model = _make_model().eval()
    ct = torch.randn(2, 1, 128, 128)
    pet = torch.randn(2, 1, 128, 128)
    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    print('latent_dim', model.mtib.latent_dim)
    for s in (1, 2, 3, 4):
        i = s - 1
        c = ct_feats[i]
        p = pet_feats[i]
        j = c + p
        r = model.task_refine[str(s)](j)
        z = j + r
        d = z - c
        print(f's{s} C shape', tuple(c.shape))
        print(f's{s} P shape', tuple(p.shape))
        print(f's{s} J shape', tuple(j.shape))
        print(f's{s} H(J) shape', tuple(r.shape), 'max_abs', float(r.abs().max()))
        print(f's{s} D* shape', tuple(d.shape))
        print(f's{s} Zfull shape', tuple(z.shape))
        assert torch.allclose(d, p + r, atol=1e-6, rtol=1e-5)
        assert torch.allclose(z, c + d, atol=1e-6, rtol=1e-5)
        assert float(r.abs().max()) < 1e-6
    full_feats, true_inc = model._true_increment(ct_feats, pet_feats, None)
    bank_out, bank_loss, bank_diag = model.mtib.forward_full(full_feats, true_inc)
    comp_out, comp_loss, comp_diag = model.mtib.forward_ct_comp(ct_feats, true_inc)
    miss_out, miss_diag = model.mtib.forward_missing(ct_feats)
    for s in (1, 2, 3, 4):
        print(f's{s} full retrieved shape', tuple(bank_out[s]['retrieved'].shape))
        print(f's{s} ct retrieved shape', tuple(comp_out[s]['retrieved'].shape))
        print(f's{s} full delta shape', tuple(bank_out[s]['delta'].shape))
        print(f's{s} ct delta shape', tuple(comp_out[s]['delta'].shape))
        print(f's{s} missing delta shape', tuple(miss_out[s].shape))
    print('bank_loss', float(bank_loss))
    print('comp_loss', float(comp_loss))
    print('bank_diag_keys', sorted(k for k in bank_diag if 'mtib_' in k)[:10])
    print('comp_diag_keys', sorted(k for k in comp_diag if 'mtib_' in k)[:10])
    print('missing_diag_keys', sorted(k for k in miss_diag if 'mtib_' in k)[:10])
    print('dataflow_ok')


if __name__ == '__main__':
    main()
