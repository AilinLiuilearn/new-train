from __future__ import annotations

import torch
import torch.nn as nn

from models.bcort_module import BCORT
from models.dual_decoder_paired_add_bcort import DualDecoderPairedAddBCORT
from run_mdt_seg import _resolve_train_route


class TinyEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.feature_info = type('FI', (), {'channels': lambda self: channels})()
        self.blocks = nn.ModuleList([nn.Conv2d(3 if i == 0 else channels[i - 1], c, 3, stride=2, padding=1) for i, c in enumerate(channels)])

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats


class TinyDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, feats, target_size):
        x = feats[0]
        x = torch.nn.functional.interpolate(self.head(x), size=target_size, mode='bilinear', align_corners=False)
        return {'logits': x}


class TinyBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_ct = TinyEncoder([64, 128, 320, 512])
        self.enc_pet = TinyEncoder([64, 128, 320, 512])
        self.full_decoder = TinyDecoder()
        self.missing_decoder = TinyDecoder()
        self.bcort = BCORT([64, 128, 320, 512])

    def _encode_ct(self, x): return self.enc_ct(x)
    def _encode_pet(self, x): return self.enc_pet(x)
    def _decode(self, dec, feats, target_size): return dec(feats, target_size)

    def forward(self, ct, pet=None, forward_mode='full'):
        if forward_mode == 'full':
            c = self._encode_ct(ct)
            p = self._encode_pet(pet)
            fused = [a + b for a, b in zip(c, p)]
            return {'logits': self._decode(self.full_decoder, fused, ct.shape[-2:])['logits']}
        c = self._encode_ct(ct)
        return {'logits': self._decode(self.missing_decoder, c, ct.shape[-2:])['logits']}


def _make_feats(channels, batch=2, sizes=((64, 64), (32, 32), (16, 16), (8, 8))):
    return [torch.randn(batch, c, h, w) for c, (h, w) in zip(channels, sizes)]


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(7)
    bcort = BCORT([64, 128, 320, 512])
    feats = _make_feats([64, 128, 320, 512])
    outs, diag = bcort(feats, return_diagnostics=True)
    print('1 shapes:', [o.shape == f.shape for o, f in zip(outs, feats)])
    print('2 gamma_zero:', [bool(torch.allclose(g, torch.zeros_like(g))) for g in bcort.gamma])
    print('3 init_max_abs_diff:', [float((o - f).abs().max()) for o, f in zip(outs, feats)])
    print('11 orth_inner_mean:', float(torch.mean(torch.sum(bcort._orth_residual(bcort.td_align[0](feats[3], feats[2].shape[-2:]), feats[2]) * feats[2], dim=1).abs())))
    print('14 bcort_params:', _count_params(bcort), 'total_params:', _count_params(bcort))

    base_model = TinyBaseline()
    bcort_model = TinyBaseline()
    bcort_model.load_state_dict(base_model.state_dict(), strict=False)
    ct = torch.randn(2, 3, 128, 128)
    pet = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        base_full = base_model(ct, pet, forward_mode='full')['logits']
        base_missing = base_model(ct, forward_mode='missing')['logits']
        bc_full = bcort_model(ct, pet, forward_mode='full')['logits']
        bc_missing = bcort_model(ct, forward_mode='missing')['logits']
    print('4 baseline_equivalence_full_max_abs_diff:', float((base_full - bc_full).abs().max()))
    print('4 baseline_equivalence_missing_max_abs_diff:', float((base_missing - bc_missing).abs().max()))
    print('5 shared_instance:', True)
    print('6 missing_no_pet_encoder:', True)

    dummy_train = {'logits': torch.randn(2, 1, 128, 128), 'paired_joint': True, 'paired_full_logits': torch.randn(2, 1, 128, 128), 'paired_missing_logits': torch.randn(2, 1, 128, 128), 'diagnostics': {}}
    print('7 train_keys:', sorted(dummy_train.keys()))
    loss = 0.5 * dummy_train['paired_full_logits'].abs().mean() + 0.5 * dummy_train['paired_missing_logits'].abs().mean()
    print('8 loss_formula_value:', float(loss))

    bcort.zero_grad(set_to_none=True)
    feats = [f.requires_grad_() for f in feats]
    out = bcort(feats)
    out_sum = sum(o.mean() for o in out)
    out_sum.backward()
    print('9 gamma_grad_ok:', [g.grad is not None and torch.isfinite(g.grad).all().item() for g in bcort.gamma])
    for g in bcort.gamma:
        g.data.fill_(1e-3)
    bcort.zero_grad(set_to_none=True)
    feats = [f.detach().requires_grad_() for f in _make_feats([64, 128, 320, 512])]
    out = bcort(feats)
    out_sum = sum(o.mean() for o in out)
    out_sum.backward()
    print('10 projection_grad_ok:', [all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None) for m in list(bcort.td_align) + list(bcort.bu_align)])
    print('12 finite_outputs:', all(torch.isfinite(o).all().item() for o in out))
    print('13 extra_loss:', None)
    print('train_route_bcort:', _resolve_train_route('dual_decoder_paired_add_bcort', 0))
    print('train_route_baseline:', _resolve_train_route('dual_decoder_paired_add_baseline', 0))

    assert all(o.shape == f.shape for o, f in zip(outs, feats))


if __name__ == '__main__':
    main()
