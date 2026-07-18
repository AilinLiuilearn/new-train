from __future__ import annotations

import torch
import torch.nn as nn

from models.bcort_module import BCORT
from models.dual_decoder_paired_add_bcort import DualDecoderPairedAddBCORT
from run_mdt_seg import _resolve_train_route




def _make_feats(channels, batch=2, sizes=((64, 64), (32, 32), (16, 16), (8, 8))):
    return [torch.randn(batch, c, h, w) for c, (h, w) in zip(channels, sizes)]


def main():
    torch.manual_seed(7)
    bcort = BCORT([64, 128, 320, 512])
    feats = _make_feats([64, 128, 320, 512])
    outs, diag = bcort(feats, return_diagnostics=True)
    assert all(o.shape == f.shape for o, f in zip(outs, feats))
    assert all(torch.allclose(g.detach(), torch.zeros_like(g)) for g in bcort.gamma)
    assert max(float((o - f).abs().max()) for o, f in zip(outs, feats)) == 0.0
    orth_err = float(torch.mean(torch.sum(bcort._orth_residual(bcort.td_align[0](feats[3], feats[2].shape[-2:]), feats[2]) * feats[2], dim=1).abs()))
    assert orth_err < 1e-4
    bcort_param_count = sum(p.numel() for p in bcort.parameters())
    assert bcort_param_count == 427008

    model = DualDecoderPairedAddBCORT()
    assert model.bcort is model.bcort
    ct = torch.randn(2, 3, 128, 128)
    pet = torch.randn(2, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out = model(ct, pet, forward_mode='full')
        miss = model(ct, None, forward_mode='missing')
    assert out['logits'].shape == miss['logits'].shape
    assert 'diagnostics' in out and 'diagnostics' in miss

    counts = {'enc_ct': 0, 'enc_pet': 0, 'bcort': 0, 'full_decoder': 0, 'missing_decoder': 0}

    def make_counter(name):
        def hook(module, inputs, outputs):
            counts[name] += 1
        return hook

    handles = [
        model.enc_ct.register_forward_hook(make_counter('enc_ct')),
        model.enc_pet.register_forward_hook(make_counter('enc_pet')),
        model.bcort.register_forward_hook(make_counter('bcort')),
        model.full_decoder.register_forward_hook(make_counter('full_decoder')),
        model.missing_decoder.register_forward_hook(make_counter('missing_decoder')),
    ]

    model.train()
    train_out = model(ct, pet, forward_mode='full')
    assert counts == {'enc_ct': 1, 'enc_pet': 1, 'bcort': 2, 'full_decoder': 1, 'missing_decoder': 1}, counts
    assert train_out['paired_joint'] is True
    assert train_out['paired_full_logits'].shape == train_out['paired_missing_logits'].shape
    assert 'aux_losses' not in train_out
    assert torch.isfinite(train_out['paired_full_logits']).all()
    assert torch.isfinite(train_out['paired_missing_logits']).all()
    assert torch.isfinite(train_out['logits']).all()
    loss_full = train_out['paired_full_logits'].mean()
    loss_missing = train_out['paired_missing_logits'].mean()
    loss = 0.5 * loss_full + 0.5 * loss_missing
    assert torch.isfinite(loss)
    assert _resolve_train_route('dual_decoder_paired_add_bcort', 0) == 'full'
    assert _resolve_train_route('dual_decoder_paired_add_baseline', 0) == 'full'

    for handle in handles:
        handle.remove()

    print('BCORT params:', bcort_param_count)
    print('init error:', max(float((o - f).abs().max()) for o, f in zip(outs, feats)))
    print('orth error:', orth_err)
    print('diagnostics keys:', sorted(diag.keys())[:6], '...')
    print('checks passed')


if __name__ == '__main__':
    main()
