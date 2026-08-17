#!/usr/bin/env python
"""Minimal smoke tests for TRDF integration into DualSharedAddPETCTBaseline."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _seed_cppi(model):
    with torch.no_grad():
        model.prototype_memory.prototype_ready.fill_(True)
        for s in range(len(model.fusion.channels)):
            getattr(model.prototype_memory, f'ct_keys_s{s + 1}').normal_()
            getattr(model.prototype_memory, f'pet_values_s{s + 1}').normal_()
        model.prototype_memory.bank_version.fill_(1)


def _fusion_grad_sum(model) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name.startswith('fusion.') and param.grad is not None:
            total += float(param.grad.abs().sum())
    return total


def _check_finite(name, tensor):
    ok = bool(torch.isfinite(tensor).all().item())
    print(f'  {name}: shape={tuple(tensor.shape)} finite={ok}', flush=True)
    assert ok, f'{name} has non-finite values'


def build_model(**kwargs):
    from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline

    defaults = dict(
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        trdf_use_text_prior=False,
    )
    defaults.update(kwargs)
    return DualSharedAddPETCTBaseline(**defaults)


def test_replace_and_text_off(device, batch_size=1, size=64):
    from models.baseline_blocks import StateAwareWeightedAddFusion
    from models.trdf_fusion import TRDFFusion

    print('=' * 72, flush=True)
    print('[Test2] build model Text OFF', flush=True)
    model = build_model(trdf_use_text_prior=False).to(device)
    model.train()
    _seed_cppi(model)

    print('fusion_type=', type(model.fusion).__name__, flush=True)
    assert isinstance(model.fusion, TRDFFusion)
    assert not isinstance(model.fusion, StateAwareWeightedAddFusion)
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith('fusion.raw_alpha_') for n in names)
    assert any(n.startswith('fusion.scales.') for n in names)
    print('pet_channels=', model.fusion.channels, flush=True)

    ct = torch.randn(batch_size, 1, size, size, device=device)
    pet = torch.randn(B := batch_size, 1, size, size, device=device)
    del B
    mask = (torch.rand(batch_size, 1, size, size, device=device) > 0.8).float()

    print('--- Full ---', flush=True)
    out = model(ct, pet, forward_mode='full', mask=mask, target_size=(size, size))
    _check_finite('logits_full', out['logits'])
    loss = out['logits'].float().square().mean()
    loss.backward()
    g = _fusion_grad_sum(model)
    print(f'  loss={float(loss):.6f} fusion_grad={g:.6f}', flush=True)
    assert g > 0
    model.zero_grad(set_to_none=True)

    print('--- Missing ---', flush=True)
    out = model(ct, pet, forward_mode='missing', mask=mask, target_size=(size, size))
    _check_finite('logits_missing', out['logits'])
    loss = out['logits'].float().square().mean()
    loss.backward()
    g = _fusion_grad_sum(model)
    print(f'  loss={float(loss):.6f} fusion_grad={g:.6f}', flush=True)
    assert g > 0
    model.zero_grad(set_to_none=True)

    print('--- Four-scale shapes ---', flush=True)
    ct_feats = model._encode_ct(ct)
    pet_feats = model._encode_pet(pet)
    pet_cal = model.pet_calibration(ct_feats, pet_feats, None, reference_valid=False)
    fused = model.fusion(ct_feats, pet_cal, mode='full')
    for i, (c, p, f) in enumerate(zip(ct_feats, pet_cal, fused), 1):
        print(f'  S{i}: CT={tuple(c.shape)} PET_cal={tuple(p.shape)} FUSED={tuple(f.shape)}', flush=True)
        assert c.shape == p.shape == f.shape

    print('--- Auto ---', flush=True)
    pet_available = torch.zeros(batch_size, device=device, dtype=torch.long)
    if batch_size >= 2:
        pet_available[0] = 1
    elif batch_size == 1:
        # still exercise auto path; alternate via a second sample when possible
        pass
    if batch_size == 1:
        # expand to 2 for mixed auto check if memory allows
        ct2 = torch.cat([ct, ct], dim=0)
        pet2 = torch.cat([pet, pet], dim=0)
        mask2 = torch.cat([mask, mask], dim=0)
        pet_available = torch.tensor([1, 0], device=device, dtype=torch.long)
        out = model(ct2, pet2, pet_available=pet_available, forward_mode='auto', mask=mask2, target_size=(size, size))
    else:
        out = model(ct, pet, pet_available=pet_available, forward_mode='auto', mask=mask, target_size=(size, size))
    _check_finite('logits_auto', out['logits'])
    loss = out['logits'].float().square().mean()
    loss.backward()
    g = _fusion_grad_sum(model)
    print(f'  loss={float(loss):.6f} fusion_grad={g:.6f}', flush=True)
    assert g > 0

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    fusion_ids = {id(p) for n, p in model.named_parameters() if n.startswith('fusion.')}
    opt_ids = {id(p) for g in opt.param_groups for p in g['params']}
    print('fusion_in_optimizer=', fusion_ids.issubset(opt_ids), 'n=', len(fusion_ids), flush=True)
    assert fusion_ids.issubset(opt_ids)
    return model


def test_text_on(device, size=64):
    print('=' * 72, flush=True)
    print('[Test3] Text ON precomputed', flush=True)
    emb_path = os.path.join(tempfile.gettempdir(), 'trdf_smoke_embeddings.pt')
    torch.save({'real': torch.randn(768), 'proxy': torch.randn(768) + 1.0}, emb_path)

    model = build_model(
        trdf_use_text_prior=True,
        trdf_text_backend='precomputed',
        trdf_text_embedding_path=emb_path,
    ).to(device)
    model.train()
    _seed_cppi(model)

    prior = model.fusion.text_prior
    real = prior.real_embedding.to(device=device, dtype=torch.float32)
    proxy = prior.proxy_embedding.to(device=device, dtype=torch.float32)
    emb_full = prior.get(2, 'full', device, torch.float32)
    emb_miss = prior.get(2, 'missing', device, torch.float32)
    emb_auto = prior.get(4, 'auto', device, torch.float32, pet_available=torch.tensor([1, 0, 1, 0], device=device))
    assert torch.allclose(emb_full[0], real)
    assert torch.allclose(emb_miss[0], proxy)
    assert torch.allclose(emb_auto[0], real) and torch.allclose(emb_auto[1], proxy)
    assert torch.allclose(emb_auto[2], real) and torch.allclose(emb_auto[3], proxy)
    print('  text selection full/missing/auto OK', flush=True)

    ct = torch.randn(1, 1, size, size, device=device)
    pet = torch.randn(1, 1, size, size, device=device)
    mask = (torch.rand(1, 1, size, size, device=device) > 0.8).float()
    for mode in ('full', 'missing'):
        out = model(ct, pet, forward_mode=mode, mask=mask, target_size=(size, size))
        _check_finite(f'text_{mode}', out['logits'])
    ct2 = torch.cat([ct, ct], dim=0)
    pet2 = torch.cat([pet, pet], dim=0)
    mask2 = torch.cat([mask, mask], dim=0)
    out = model(
        ct2, pet2,
        pet_available=torch.tensor([1, 0], device=device),
        forward_mode='auto',
        mask=mask2,
        target_size=(size, size),
    )
    _check_finite('text_auto', out['logits'])
    loss = out['logits'].float().square().mean()
    loss.backward()
    g = _fusion_grad_sum(model)
    print(f'  text_auto fusion_grad={g:.6f}', flush=True)
    assert g > 0


def test_builder_and_resume():
    from types import SimpleNamespace

    from models.build_mdt_seg import build_mdt_seg_teacher
    from run_mdt_seg import _load_state_dict_with_report

    print('=' * 72, flush=True)
    print('[Builder] build_mdt_seg_teacher + resume message', flush=True)
    cfg = SimpleNamespace(
        checkpoint_dir=tempfile.mkdtemp(),
        cppi_num_clusters=6,
        cppi_build_stage=3,
        no_encoder_pretrained=True,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        deep_supervision=False,
        trdf_use_text_prior=False,
        trdf_text_backend='precomputed',
        trdf_text_embedding_path=None,
        trdf_text_model_path=None,
        trdf_text_hidden_dim=128,
    )
    built = build_mdt_seg_teacher(cfg)
    print('built fusion=', type(built['model'].fusion).__name__, flush=True)
    ckpt_path = os.path.join(tempfile.gettempdir(), 'fake_old_fusion.pth')
    torch.save(
        {
            'model': {
                'fusion.raw_alpha_full': torch.zeros(4),
                'fusion.raw_alpha_missing': torch.zeros(4),
            }
        },
        ckpt_path,
    )
    _load_state_dict_with_report(built['model'], ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'text_off', 'text_on', 'builder'), default='all')
    parser.add_argument('--size', type=int, default=64)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    if args.stage in ('all', 'text_off'):
        test_replace_and_text_off(device, batch_size=1, size=args.size)
    if args.stage in ('all', 'text_on'):
        test_text_on(device, size=args.size)
    if args.stage in ('all', 'builder'):
        test_builder_and_resume()
    print('\nALL REQUESTED SMOKE TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
