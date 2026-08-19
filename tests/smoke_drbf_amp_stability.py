#!/usr/bin/env python
"""AMP Full/Missing smoke test for DualSharedAddPETCTBaseline + DRBF."""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _finite(name, x):
    ok = bool(torch.isfinite(x).all().item())
    print(f'  {name}: shape={tuple(x.shape)} finite={ok}', flush=True)
    assert ok, f'{name} non-finite'


def _module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
        if not torch.isfinite(p.grad).all():
            return float('nan')
    if total is None:
        return 0.0
    return float(total.sqrt().item())


def main():
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'device={device} amp={use_amp} batch={args.batch_size} size={args.size}', flush=True)

    model = DualSharedAddPETCTBaseline(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        drbf_use_text_prior=False,
        cppi_build_stage=4,
    ).to(device)
    model.train()

    with torch.no_grad():
        model.prototype_memory.prototype_ready.fill_(True)
        for s in range(len(model.fusion.channels)):
            getattr(model.prototype_memory, f'ct_keys_s{s + 1}').normal_()
            getattr(model.prototype_memory, f'pet_values_s{s + 1}').normal_()
        model.prototype_memory.bank_version.fill_(1)

    batch = args.batch_size
    size = args.size
    ct = torch.randn(batch, 1, size, size, device=device)
    pet = torch.randn(batch, 1, size, size, device=device)
    mask = (torch.rand(batch, 1, size, size, device=device) > 0.8).float()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5)

    results = {}
    for mode in ('full', 'missing'):
        print(f'=== AMP main-model {mode} ===', flush=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(ct, pet, forward_mode=mode, mask=mask, target_size=(size, size))
            logits = out['logits']
            loss = logits.float().square().mean()
        _finite(f'{mode}.logits', logits)
        assert torch.isfinite(loss), f'{mode} loss non-finite'
        print(f'  loss={float(loss.detach()):.6f}', flush=True)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        enc_ct_g = _module_grad_norm(model.enc_ct)
        enc_pet_g = _module_grad_norm(model.enc_pet)
        ct_align_g = _module_grad_norm(model.ct_align)
        fusion_g = _module_grad_norm(model.fusion)
        decoder_g = _module_grad_norm(model.decoder)
        print(
            f'  enc_ct_grad={enc_ct_g:.6f} enc_pet_grad={enc_pet_g:.6f} '
            f'ct_align_grad={ct_align_g:.6f} fusion_grad={fusion_g:.6f} decoder_grad={decoder_g:.6f}',
            flush=True,
        )
        for name, g in [
            ('enc_ct', enc_ct_g),
            ('ct_align', ct_align_g),
            ('fusion', fusion_g),
            ('decoder', decoder_g),
        ]:
            assert g == g and g != float('inf'), f'{mode} {name} grad non-finite'
            assert g > 0, f'{mode} {name} grad is zero'
        if mode == 'full':
            assert enc_pet_g > 0, f'{mode} enc_pet grad is zero'
        else:
            # Missing fusion path uses proxy PET; real PET encoder may have zero grad
            # from the segmentation loss even though forward still runs enc_pet.
            print(f'  missing enc_pet_grad={enc_pet_g:.6f} (zero expected for fusion loss)', flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        results[mode] = {
            'enc_ct': enc_ct_g,
            'enc_pet': enc_pet_g,
            'ct_align': ct_align_g,
            'fusion': fusion_g,
            'decoder': decoder_g,
            'loss': float(loss.detach()),
        }

    if device.type == 'cuda':
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f'peak_allocated_MB={peak_alloc:.1f}', flush=True)
        print(f'peak_reserved_MB={peak_reserved:.1f}', flush=True)

    print('MAIN MODEL AMP SMOKE PASSED', flush=True)
    print(results, flush=True)


if __name__ == '__main__':
    main()
