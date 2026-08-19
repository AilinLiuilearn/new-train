#!/usr/bin/env python
"""Smoke tests for DRBF Stage-2 integration with preserved Stage-1 alphas."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained/ckpt.best_joint.pth.tar'
)


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
        drbf_use_text_prior=False,
        cppi_build_stage=4,
        cppi_num_clusters=6,
    )
    defaults.update(kwargs)
    return DualSharedAddPETCTBaseline(**defaults)


def _prepare_calibrated(model, ct, pet, mode='full'):
    ct_feats = model._encode_ct(ct)
    pet_feats_real = model._encode_pet(pet)
    if mode == 'full':
        if model.prototype_memory.bank_ready:
            _, ct_ref, _ = model._retrieve_cppi(
                ct_feats, return_ct_reference=True
            )
            ct_ref = [x.detach() for x in ct_ref]
            pet_cal = model.pet_calibration(
                ct_feats, pet_feats_real, ct_ref, reference_valid=True
            )
        else:
            pet_cal = model.pet_calibration(
                ct_feats, pet_feats_real, None, reference_valid=False
            )
    elif mode == 'missing':
        pet_proxy, ct_ref, _ = model._retrieve_cppi(
            ct_feats, return_ct_reference=True
        )
        ct_ref = [x.detach() for x in ct_ref]
        pet_cal = model.pet_calibration(
            ct_feats,
            pet_proxy,
            ct_ref,
            reference_valid=model.prototype_memory.bank_ready,
        )
    else:
        raise ValueError(mode)
    return ct_feats, pet_cal


def test_replace_and_text_off(device, batch_size=1, size=64):
    from models.baseline_blocks import StateAwarePETEvidenceScaler, StateAwareWeightedAddFusion
    from models.drbf_fusion import DRBFFusion

    print('=' * 72, flush=True)
    print('[Test] build model Text OFF + alpha evidence path', flush=True)
    model = build_model(drbf_use_text_prior=False).to(device)
    model.train()
    _seed_cppi(model)

    print('fusion_type=', type(model.fusion).__name__, flush=True)
    print('scaler_type=', type(model.pet_evidence_scaler).__name__, flush=True)
    assert isinstance(model.fusion, DRBFFusion)
    assert isinstance(model.pet_evidence_scaler, StateAwarePETEvidenceScaler)
    assert not isinstance(model.fusion, StateAwareWeightedAddFusion)
    assert not any(
        isinstance(m, StateAwareWeightedAddFusion) for m in model.modules()
    )

    alpha_full = model.pet_evidence_scaler.alpha_full.detach().cpu().tolist()
    alpha_missing = model.pet_evidence_scaler.alpha_missing.detach().cpu().tolist()
    print(f'  alpha_full = {[round(x, 6) for x in alpha_full]}', flush=True)
    print(f'  alpha_missing = {[round(x, 6) for x in alpha_missing]}', flush=True)
    assert all(torch.isfinite(model.pet_evidence_scaler.alpha_full))
    assert all(torch.isfinite(model.pet_evidence_scaler.alpha_missing))

    ct = torch.randn(batch_size, 1, size, size, device=device)
    pet = torch.randn(batch_size, 1, size, size, device=device)
    mask = (torch.rand(batch_size, 1, size, size, device=device) > 0.8).float()

    print('--- Full / Missing forward ---', flush=True)
    for mode in ('full', 'missing'):
        out = model(ct, pet, forward_mode=mode, mask=mask, target_size=(size, size))
        _check_finite(f'logits_{mode}', out['logits'])
        loss = out['logits'].float().square().mean()
        loss.backward()
        g = _fusion_grad_sum(model)
        print(f'  {mode}: loss={float(loss):.6f} fusion_grad={g:.6f}', flush=True)
        assert g > 0
        model.zero_grad(set_to_none=True)

    print('--- Evidence shapes + zero-step equivalence ---', flush=True)
    for mode in ('full', 'missing'):
        ct_feats, pet_cal = _prepare_calibrated(model, ct, pet, mode=mode)
        evidence = model.pet_evidence_scaler(pet_cal, mode=mode)
        alpha = (
            model.pet_evidence_scaler.alpha_full
            if mode == 'full'
            else model.pet_evidence_scaler.alpha_missing
        )
        # Fresh zero-init DRBF must equal C + E.
        fresh = DRBFFusion(channels=model.fusion.channels, use_text_prior=False).to(device)
        fused0, aux = fresh(ct_feats, evidence, mode=mode, return_aux=True)
        max_err = 0.0
        for i, (c, p, e, f, a) in enumerate(
            zip(ct_feats, pet_cal, evidence, fused0, aux), 1
        ):
            expected = c + e
            err = (f - expected).abs().max().item()
            max_err = max(max_err, err)
            e_from_alpha = (alpha[i - 1].to(p) * p)
            e_err = (e - e_from_alpha).abs().max().item()
            d_sum_err = (a['d_ct'] + a['d_pet'] - 1.0).abs().max().item()
            print(
                f'  {mode} S{i}: CT={tuple(c.shape)} P_cal={tuple(p.shape)} '
                f'E={tuple(e.shape)} F={tuple(f.shape)} '
                f'|F-(C+E)|={err:.3e} |E-alpha*P|={e_err:.3e} '
                f'|Dsum-1|={d_sum_err:.3e} attn={tuple(a["modality_attention"].shape)}',
                flush=True,
            )
            assert c.shape == p.shape == e.shape == f.shape
            assert e_err <= 1e-6
            assert err <= 1e-6
            assert d_sum_err < 1e-6
            assert a['modality_attention'].shape[-2:] == (2, 2)
            assert a['d_ct'].shape[1:] == (1, c.shape[-2], c.shape[-1])
        print(f'  {mode} zero-step max_err={max_err:.3e}', flush=True)

    print('--- Auto route consistency ---', flush=True)
    ct2 = torch.cat([ct, ct], dim=0)
    pet2 = torch.cat([pet, pet], dim=0)
    mask2 = torch.cat([mask, mask], dim=0)
    pet_available = torch.tensor([1, 0], device=device, dtype=torch.long)
    out = model(
        ct2, pet2,
        pet_available=pet_available,
        forward_mode='auto',
        mask=mask2,
        target_size=(size, size),
    )
    _check_finite('logits_auto', out['logits'])

    # Verify per-sample alpha selection on evidence.
    ct_feats = model._encode_ct(ct2)
    pet_real = model._encode_pet(pet2)
    pet_proxy, ct_ref, _ = model._retrieve_cppi(ct_feats, return_ct_reference=True)
    ct_ref = [x.detach() for x in ct_ref]
    avail = pet_available.view(-1, 1, 1, 1).to(dtype=pet_real[0].dtype)
    pet_sel = [
        r * avail + p * (1.0 - avail)
        for r, p in zip(pet_real, pet_proxy)
    ]
    pet_cal = model.pet_calibration(
        ct_feats, pet_sel, ct_ref, reference_valid=True
    )
    evidence_auto = model.pet_evidence_scaler(
        pet_cal, mode='auto', pet_available=pet_available
    )
    for i, (p, e) in enumerate(zip(pet_cal, evidence_auto), 1):
        a_full = model.pet_evidence_scaler.alpha_full[i - 1].to(p)
        a_miss = model.pet_evidence_scaler.alpha_missing[i - 1].to(p)
        expected0 = a_full * p[0:1]
        expected1 = a_miss * p[1:2]
        err0 = (e[0:1] - expected0).abs().max().item()
        err1 = (e[1:2] - expected1).abs().max().item()
        print(f'  auto S{i}: sample0|E-a_full*P|={err0:.3e} sample1|E-a_miss*P|={err1:.3e}', flush=True)
        assert err0 <= 1e-6 and err1 <= 1e-6

    return model


def test_text_on(device, size=64):
    print('=' * 72, flush=True)
    print('[Test] Text ON precomputed', flush=True)
    emb_path = os.path.join(tempfile.gettempdir(), 'drbf_smoke_embeddings.pt')
    torch.save({'real': torch.randn(128), 'proxy': torch.randn(128) + 1.0}, emb_path)

    model = build_model(
        drbf_use_text_prior=True,
        drbf_text_embedding_path=emb_path,
    ).to(device)
    model.train()
    _seed_cppi(model)

    real = model.fusion.real_text_embedding.to(device=device, dtype=torch.float32)
    proxy = model.fusion.proxy_text_embedding.to(device=device, dtype=torch.float32)
    emb_full = model.fusion._resolve_text(2, device, torch.float32, 'full', None, None)
    emb_miss = model.fusion._resolve_text(2, device, torch.float32, 'missing', None, None)
    emb_auto = model.fusion._resolve_text(
        4, device, torch.float32, 'auto', None, torch.tensor([1, 0, 1, 0], device=device)
    )
    assert torch.allclose(emb_full[0], real)
    assert torch.allclose(emb_miss[0], proxy)
    assert torch.allclose(emb_auto[0], real) and torch.allclose(emb_auto[1], proxy)
    assert torch.allclose(emb_auto[2], real) and torch.allclose(emb_auto[3], proxy)
    print('  text selection full/missing/auto OK', flush=True)

    # text_proj zero-init must preserve Stage-1 baseline at step 0.
    ct = torch.randn(1, 1, size, size, device=device)
    pet = torch.randn(1, 1, size, size, device=device)
    ct_feats, pet_cal = _prepare_calibrated(model, ct, pet, mode='full')
    evidence = model.pet_evidence_scaler(pet_cal, mode='full')
    fused, aux = model.fusion(ct_feats, evidence, mode='full', return_aux=True)
    for i, (c, e, f) in enumerate(zip(ct_feats, evidence, fused), 1):
        err = (f - (c + e)).abs().max().item()
        print(f'  text-on zero-step S{i} |F-(C+E)|={err:.3e}', flush=True)
        assert err <= 1e-6

    for mode in ('full', 'missing'):
        out = model(ct, pet, forward_mode=mode, target_size=(size, size))
        _check_finite(f'text_{mode}', out['logits'])
    ct2 = torch.cat([ct, ct], dim=0)
    pet2 = torch.cat([pet, pet], dim=0)
    out = model(
        ct2, pet2,
        pet_available=torch.tensor([1, 0], device=device),
        forward_mode='auto',
        target_size=(size, size),
    )
    _check_finite('text_auto', out['logits'])
    loss = out['logits'].float().square().mean()
    loss.backward()
    g = _fusion_grad_sum(model)
    print(f'  text_auto fusion_grad={g:.6f}', flush=True)
    assert g > 0


def test_optimizer_groups():
    from types import SimpleNamespace
    from tasks.mdt_seg import MDTSegTeacher

    print('=' * 72, flush=True)
    print('[Test] optimizer groups', flush=True)
    model = build_model(drbf_use_text_prior=False)
    cfg = SimpleNamespace(
        mixed_precision=False,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        old_module_lr=2e-5,
        learning_rate=8e-5,
        weight_decay=1e-4,
    )
    task = MDTSegTeacher({'model': model}, cfg)
    groups = {g['name']: g for g in task.optimizer.param_groups}
    assert set(groups) == {'stage1_modules', 'drbf'}
    assert abs(groups['stage1_modules']['lr'] - 2e-5) < 1e-12
    assert abs(groups['drbf']['lr'] - 8e-5) < 1e-12
    scaler_ids = {id(p) for p in model.pet_evidence_scaler.parameters()}
    old_ids = {id(p) for p in groups['stage1_modules']['params']}
    drbf_ids = {id(p) for p in groups['drbf']['params']}
    assert scaler_ids.issubset(old_ids)
    assert not (scaler_ids & drbf_ids)
    print('  groups OK: stage1=2e-5 (includes evidence scaler), drbf=8e-5', flush=True)


def test_stage1_equivalence(device, stage1_ckpt, size=64):
    from models.baseline_blocks import StateAwareWeightedAddFusion
    from run_mdt_seg import _load_stage1_warmstart, _sync_cppi_config_from_stage1
    from types import SimpleNamespace

    print('=' * 72, flush=True)
    print('[Test] Stage1 -> Stage2 zero-step equivalence', flush=True)
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f'Stage-1 checkpoint not found: {stage1_ckpt}')

    cfg = SimpleNamespace(
        checkpoint_dir=tempfile.mkdtemp(),
        cppi_num_clusters=6,
        cppi_build_stage=4,
        no_encoder_pretrained=True,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        deep_supervision=False,
        drbf_use_text_prior=False,
        drbf_text_embedding_path=None,
        drbf_text_dim=128,
    )
    _sync_cppi_config_from_stage1(cfg, stage1_ckpt)
    print(
        f'  cppi_build_stage={cfg.cppi_build_stage} cppi_num_clusters={cfg.cppi_num_clusters}',
        flush=True,
    )

    model = build_model(
        cppi_build_stage=cfg.cppi_build_stage,
        cppi_num_clusters=cfg.cppi_num_clusters,
        cppi_output_dir=os.path.join(cfg.checkpoint_dir, 'cppi'),
    ).to(device)
    model.eval()
    _load_stage1_warmstart(model, stage1_ckpt)

    ckpt = torch.load(stage1_ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    raw_full = state['fusion.raw_alpha_full'].to(device)
    raw_miss = state['fusion.raw_alpha_missing'].to(device)
    assert torch.allclose(model.pet_evidence_scaler.raw_alpha_full, raw_full)
    assert torch.allclose(model.pet_evidence_scaler.raw_alpha_missing, raw_miss)
    alpha_full = model.pet_evidence_scaler.alpha_full.detach().cpu().tolist()
    alpha_missing = model.pet_evidence_scaler.alpha_missing.detach().cpu().tolist()
    print(f'  restored alpha_full={[round(x, 6) for x in alpha_full]}', flush=True)
    print(f'  restored alpha_missing={[round(x, 6) for x in alpha_missing]}', flush=True)

    # Stage-1 final fusion reconstructed from restored alphas.
    stage1_fusion = StateAwareWeightedAddFusion(num_scales=len(model.fusion.channels)).to(device)
    stage1_fusion.raw_alpha_full.data.copy_(raw_full)
    stage1_fusion.raw_alpha_missing.data.copy_(raw_miss)

    ct = torch.randn(1, 1, size, size, device=device)
    pet = torch.randn(1, 1, size, size, device=device)

    report = {}
    with torch.no_grad():
        for mode in ('full', 'missing'):
            ct_feats, pet_cal = _prepare_calibrated(model, ct, pet, mode=mode)
            f_stage1 = stage1_fusion(ct_feats, pet_cal, mode=mode)
            evidence = model.pet_evidence_scaler(pet_cal, mode=mode)
            f_stage2 = model.fusion(ct_feats, evidence, mode=mode)

            scale_errs = []
            for i, (a, b) in enumerate(zip(f_stage1, f_stage2), 1):
                err = (a - b).abs().max().item()
                scale_errs.append(err)
                print(f'  {mode} S{i} max|F_stage1-F_stage2|={err:.3e}', flush=True)
                assert err <= 1e-6

            logits1 = model.decoder(f_stage1, (size, size))['logits']
            logits2 = model.decoder(f_stage2, (size, size))['logits']
            logit_err = (logits1 - logits2).abs().max().item()
            print(f'  {mode} max|logits_stage1-logits_stage2|={logit_err:.3e}', flush=True)
            assert logit_err <= 1e-6
            report[mode] = {
                'feat_max_err': max(scale_errs),
                'logit_max_err': logit_err,
            }

    # Also verify end-to-end model logits match Stage1-path logits.
    with torch.no_grad():
        for mode in ('full', 'missing'):
            out2 = model(ct, pet, forward_mode=mode, target_size=(size, size))
            ct_feats, pet_cal = _prepare_calibrated(model, ct, pet, mode=mode)
            f_stage1 = stage1_fusion(ct_feats, pet_cal, mode=mode)
            logits1 = model.decoder(f_stage1, (size, size))['logits']
            e2e_err = (out2['logits'] - logits1).abs().max().item()
            print(f'  {mode} e2e max|logits_model-logits_stage1path|={e2e_err:.3e}', flush=True)
            assert e2e_err <= 1e-5
            report[mode]['e2e_logit_err'] = e2e_err

    print('  Stage1 equivalence PASSED', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        choices=('all', 'text_off', 'text_on', 'optimizer', 'equivalence'),
        default='all',
    )
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--stage1_checkpoint', type=str, default=DEFAULT_STAGE1_CKPT)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)
    if args.stage in ('all', 'text_off'):
        test_replace_and_text_off(device, batch_size=1, size=args.size)
    if args.stage in ('all', 'text_on'):
        test_text_on(device, size=args.size)
    if args.stage in ('all', 'optimizer'):
        test_optimizer_groups()
    if args.stage in ('all', 'equivalence'):
        test_stage1_equivalence(device, args.stage1_checkpoint, size=args.size)
    print('\nALL REQUESTED SMOKE TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
