#!/usr/bin/env python
"""Smoke tests for DRBF Stage-2 integration with preserved Stage-1 alphas."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_STAGE1_CKPT = (
    '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/'
    'e1-api-masked-baseline-CPPI-k6-c4-affinecalib-pretrained/ckpt.best_joint.pth.tar'
)
DEFAULT_BIOMEDCLIP = '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model'
DEFAULT_TEXT_TOWER = '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower'


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
            _, ct_ref, _ = model._retrieve_cppi(ct_feats, return_ct_reference=True)
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


def test_pet_encoder_and_cppi_collect(device, batch_size=1, size=64):
    print('=' * 72, flush=True)
    print('[Test A] PET encoder + CPPI collect in Full/Missing', flush=True)
    model = build_model(drbf_use_text_prior=False).to(device)
    model.train()
    _seed_cppi(model)

    ct = torch.randn(batch_size, 1, size, size, device=device)
    pet = torch.randn(batch_size, 1, size, size, device=device)
    mask = (torch.rand(batch_size, 1, size, size, device=device) > 0.8).float()

    enc_calls = {'full': 0, 'missing': 0}
    collect_calls = {'full': 0, 'missing': 0}
    orig_encode = model._encode_pet
    orig_collect = model._collect_cppi

    def encode_hook(p):
        enc_calls[model._current_route] += 1
        return orig_encode(p)

    def collect_hook(ct_feats, pet_feats, m):
        collect_calls[model._current_route] += 1
        return orig_collect(ct_feats, pet_feats, m)

    model._encode_pet = encode_hook
    model._collect_cppi = collect_hook

    for mode in ('full', 'missing'):
        model._current_route = mode
        out = model(ct, pet, forward_mode=mode, mask=mask, target_size=(size, size))
        _check_finite(f'logits_{mode}', out['logits'])
        assert enc_calls[mode] >= 1, f'PET encoder not called in {mode}'
        assert collect_calls[mode] >= 1, f'CPPI collect not called in {mode}'
        print(f'  {mode}: enc_pet_calls={enc_calls[mode]} collect_calls={collect_calls[mode]}', flush=True)


def test_missing_proxy_evidence(device, size=64):
    print('=' * 72, flush=True)
    print('[Test B/C] Missing DRBF evidence from proxy PET + alpha_missing', flush=True)
    model = build_model(drbf_use_text_prior=False).to(device)
    model.eval()
    _seed_cppi(model)

    ct = torch.randn(1, 1, size, size, device=device)
    pet_a = torch.randn(1, 1, size, size, device=device)
    pet_b = torch.randn(1, 1, size, size, device=device) * 5.0 + 3.0

    ct_feats = model._encode_ct(ct)
    pet_real_a = model._encode_pet(pet_a)
    pet_real_b = model._encode_pet(pet_b)
    pet_proxy, ct_ref, _ = model._retrieve_cppi(ct_feats, return_ct_reference=True)
    ct_ref = [x.detach() for x in ct_ref]

    pet_cal_proxy = model.pet_calibration(
        ct_feats, pet_proxy, ct_ref, reference_valid=True
    )
    pet_cal_real_a = model.pet_calibration(
        ct_feats, pet_real_a, ct_ref, reference_valid=True
    )
    pet_cal_real_b = model.pet_calibration(
        ct_feats, pet_real_b, ct_ref, reference_valid=True
    )

    evidence_proxy = model.pet_evidence_scaler(pet_cal_proxy, mode='missing')
    evidence_wrong_real = model.pet_evidence_scaler(pet_cal_real_a, mode='missing')

    alpha_miss = model.pet_evidence_scaler.alpha_missing
    for i, (p, e) in enumerate(zip(pet_cal_proxy, evidence_proxy), 1):
        err = (e - alpha_miss[i - 1].to(p) * p).abs().max().item()
        print(f'  S{i} |E - alpha_missing*P_proxy_cal|={err:.3e}', flush=True)
        assert err <= 1e-6

    fused_correct = model.fusion(ct_feats, evidence_proxy, mode='missing')
    fused_wrong = model.fusion(ct_feats, evidence_wrong_real, mode='missing')

    err_proxy_vs_real = max((a - b).abs().max().item() for a, b in zip(fused_correct, fused_wrong))
    print(f'  proxy vs real evidence max diff={err_proxy_vs_real:.3e}', flush=True)
    assert err_proxy_vs_real > 1e-3, 'Missing fusion should differ when using real vs proxy evidence'
    print('  Missing path uses proxy-calibrated evidence (not real PET)', flush=True)


def test_replace_and_text_off(device, batch_size=1, size=64):
    from models.baseline_blocks import StateAwarePETEvidenceScaler, StateAwareWeightedAddFusion
    from models.drbf_fusion import DRBFFusion

    print('=' * 72, flush=True)
    print('[Test] build model Text OFF + alpha evidence path', flush=True)
    model = build_model(drbf_use_text_prior=False).to(device)
    model.train()
    _seed_cppi(model)

    assert model.text_encoder is None
    assert isinstance(model.fusion, DRBFFusion)
    assert isinstance(model.pet_evidence_scaler, StateAwarePETEvidenceScaler)
    assert not isinstance(model.fusion, StateAwareWeightedAddFusion)

    ct = torch.randn(batch_size, 1, size, size, device=device)
    pet = torch.randn(batch_size, 1, size, size, device=device)
    mask = (torch.rand(batch_size, 1, size, size, device=device) > 0.8).float()

    for mode in ('full', 'missing'):
        out = model(ct, pet, forward_mode=mode, mask=mask, target_size=(size, size))
        _check_finite(f'logits_{mode}', out['logits'])
        loss = out['logits'].float().square().mean()
        loss.backward()
        g = _fusion_grad_sum(model)
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
        fresh = DRBFFusion(channels=model.fusion.channels, use_text_prior=False).to(device)
        fused0, aux = fresh(ct_feats, evidence, mode=mode, return_aux=True)
        for i, (c, p, e, f, a) in enumerate(
            zip(ct_feats, pet_cal, evidence, fused0, aux), 1
        ):
            expected = c + e
            err = (f - expected).abs().max().item()
            e_err = (e - alpha[i - 1].to(p) * p).abs().max().item()
            d_sum_err = (a['d_ct'] + a['d_pet'] - 1.0).abs().max().item()
            print(
                f'  {mode} S{i}: |F-(C+E)|={err:.3e} |E-alpha*P|={e_err:.3e} '
                f'|Dsum-1|={d_sum_err:.3e} attn={tuple(a["modality_attention"].shape)}',
                flush=True,
            )
            assert err <= 1e-6
            assert e_err <= 1e-6
            assert d_sum_err < 1e-6
            assert a['modality_attention'].shape[-2:] == (2, 2)

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

    ct_feats = model._encode_ct(ct2)
    pet_real = model._encode_pet(pet2)
    pet_proxy, ct_ref, _ = model._retrieve_cppi(ct_feats, return_ct_reference=True)
    ct_ref = [x.detach() for x in ct_ref]
    avail = pet_available.view(-1, 1, 1, 1).to(dtype=pet_real[0].dtype)
    pet_sel = [r * avail + p * (1.0 - avail) for r, p in zip(pet_real, pet_proxy)]
    pet_cal = model.pet_calibration(ct_feats, pet_sel, ct_ref, reference_valid=True)
    evidence_auto = model.pet_evidence_scaler(
        pet_cal, mode='auto', pet_available=pet_available
    )
    for i, (p, e) in enumerate(zip(pet_cal, evidence_auto), 1):
        a_full = model.pet_evidence_scaler.alpha_full[i - 1].to(p)
        a_miss = model.pet_evidence_scaler.alpha_missing[i - 1].to(p)
        err0 = (e[0:1] - a_full * p[0:1]).abs().max().item()
        err1 = (e[1:2] - a_miss * p[1:2]).abs().max().item()
        assert err0 <= 1e-6 and err1 <= 1e-6


def test_text_on_frozen(device, size=64):
    print('=' * 72, flush=True)
    print('[Test F/G/H] Text ON frozen BioMedCLIP internal load', flush=True)
    if not os.path.isdir(DEFAULT_BIOMEDCLIP):
        print(f'  SKIP: BioMedCLIP not found at {DEFAULT_BIOMEDCLIP}', flush=True)
        return

    model = build_model(
        drbf_use_text_prior=True,
        drbf_text_encoder_path=DEFAULT_BIOMEDCLIP,
        drbf_text_tower_path=DEFAULT_TEXT_TOWER,
        drbf_text_encoder_trainable=False,
    ).to(device)
    model.train()
    _seed_cppi(model)

    assert model.text_encoder is not None
    assert model.text_encoder.real_prompt_ready
    assert model.text_encoder.proxy_prompt_ready
    assert model.fusion.real_text_embedding.numel() > 0
    assert model.fusion.proxy_text_embedding.numel() > 0

    real = model.fusion.real_text_embedding.to(device=device, dtype=torch.float32)
    proxy = model.fusion.proxy_text_embedding.to(device=device, dtype=torch.float32)
    emb_full = model.fusion._resolve_text(2, device, torch.float32, 'full', None, None)
    emb_miss = model.fusion._resolve_text(2, device, torch.float32, 'missing', None, None)
    emb_auto = model.fusion._resolve_text(
        4, device, torch.float32, 'auto', None,
        torch.tensor([1, 0, 1, 0], device=device),
    )
    assert torch.allclose(emb_full[0], real)
    assert torch.allclose(emb_miss[0], proxy)
    assert torch.allclose(emb_auto[0], real) and torch.allclose(emb_auto[1], proxy)

    ct = torch.randn(1, 1, size, size, device=device)
    pet = torch.randn(1, 1, size, size, device=device)
    ct_feats, pet_cal = _prepare_calibrated(model, ct, pet, mode='full')
    evidence = model.pet_evidence_scaler(pet_cal, mode='full')
    fused, _ = model.fusion(ct_feats, evidence, mode='full', return_aux=True)
    for i, (c, e, f) in enumerate(zip(ct_feats, evidence, fused), 1):
        err = (f - (c + e)).abs().max().item()
        print(f'  text-on zero-step S{i} |F-(C+E)|={err:.3e}', flush=True)
        assert err <= 1e-6

    for mode in ('full', 'missing'):
        out = model(ct, pet, forward_mode=mode, target_size=(size, size))
        _check_finite(f'text_{mode}', out['logits'])


def test_optimizer_groups():
    from tasks.mdt_seg import MDTSegTeacher

    print('=' * 72, flush=True)
    print('[Test I] optimizer groups scratch + warmstart', flush=True)
    model = build_model(drbf_use_text_prior=False)

    cfg_scratch = SimpleNamespace(
        train_mode='scratch',
        mixed_precision=False,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        old_module_lr=2e-5,
        learning_rate=8e-5,
        weight_decay=1e-4,
    )
    task_scratch = MDTSegTeacher({'model': model}, cfg_scratch)
    groups = {g['name']: g for g in task_scratch.optimizer.param_groups}
    assert abs(groups['stage1_modules']['lr'] - 8e-5) < 1e-12
    assert abs(groups['drbf']['lr'] - 8e-5) < 1e-12
    print('  scratch: stage1=8e-5 drbf=8e-5 OK', flush=True)

    cfg_warm = SimpleNamespace(
        train_mode='stage1_warmstart',
        mixed_precision=False,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        old_module_lr=2e-5,
        learning_rate=8e-5,
        weight_decay=1e-4,
    )
    task_warm = MDTSegTeacher({'model': build_model()}, cfg_warm)
    groups = {g['name']: g for g in task_warm.optimizer.param_groups}
    assert abs(groups['stage1_modules']['lr'] - 2e-5) < 1e-12
    assert abs(groups['drbf']['lr'] - 8e-5) < 1e-12
    print('  stage1_warmstart: stage1=2e-5 drbf=8e-5 OK', flush=True)


def test_stage1_equivalence(device, stage1_ckpt, size=64):
    from models.baseline_blocks import StateAwareWeightedAddFusion
    from run_mdt_seg import _load_stage1_warmstart, _sync_cppi_config_from_stage1

    print('=' * 72, flush=True)
    print('[Test J] Stage1 -> Stage2 zero-step equivalence', flush=True)
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f'Stage-1 checkpoint not found: {stage1_ckpt}')

    cfg = SimpleNamespace(
        checkpoint_dir=tempfile.mkdtemp(),
        cppi_num_clusters=6,
        cppi_build_stage=4,
        no_encoder_pretrained=True,
    )
    _sync_cppi_config_from_stage1(cfg, stage1_ckpt)

    model = build_model(
        cppi_build_stage=cfg.cppi_build_stage,
        cppi_num_clusters=cfg.cppi_num_clusters,
        cppi_output_dir=os.path.join(cfg.checkpoint_dir, 'cppi'),
    ).to(device)
    model.eval()
    _load_stage1_warmstart(model, stage1_ckpt)

    ckpt = torch.load(stage1_ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    assert torch.allclose(
        model.pet_evidence_scaler.raw_alpha_full,
        state['fusion.raw_alpha_full'].to(device),
    )
    assert torch.allclose(
        model.pet_evidence_scaler.raw_alpha_missing,
        state['fusion.raw_alpha_missing'].to(device),
    )

    stage1_fusion = StateAwareWeightedAddFusion(num_scales=len(model.fusion.channels)).to(device)
    stage1_fusion.raw_alpha_full.data.copy_(model.pet_evidence_scaler.raw_alpha_full)
    stage1_fusion.raw_alpha_missing.data.copy_(model.pet_evidence_scaler.raw_alpha_missing)

    ct = torch.randn(1, 1, size, size, device=device)
    pet = torch.randn(1, 1, size, size, device=device)
    report = {}
    with torch.no_grad():
        for mode in ('full', 'missing'):
            ct_feats, pet_cal = _prepare_calibrated(model, ct, pet, mode=mode)
            f_stage1 = stage1_fusion(ct_feats, pet_cal, mode=mode)
            evidence = model.pet_evidence_scaler(pet_cal, mode=mode)
            f_stage2 = model.fusion(ct_feats, evidence, mode=mode)
            feat_err = max((a - b).abs().max().item() for a, b in zip(f_stage1, f_stage2))
            logits1 = model.decoder(f_stage1, (size, size))['logits']
            logits2 = model.decoder(f_stage2, (size, size))['logits']
            logit_err = (logits1 - logits2).abs().max().item()
            report[mode] = {'feat_max_err': feat_err, 'logit_max_err': logit_err}
            print(f'  {mode} feat_err={feat_err:.3e} logit_err={logit_err:.3e}', flush=True)
            assert feat_err <= 1e-6 and logit_err <= 1e-6
    return report


def test_resume_roundtrip(device, size=64):
    from tasks.mdt_seg import MDTSegTeacher

    print('=' * 72, flush=True)
    print('[Test K] resume full training state', flush=True)
    model = build_model(drbf_use_text_prior=False).to(device)
    cfg = SimpleNamespace(
        train_mode='scratch',
        mixed_precision=False,
        loss_smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        old_module_lr=2e-5,
        learning_rate=8e-5,
        weight_decay=1e-4,
        random_state=2023,
    )
    task = MDTSegTeacher({'model': model}, cfg)
    task.global_batch_step = 42
    task.scheduler = torch.optim.lr_scheduler.ConstantLR(task.optimizer, factor=1.0)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, 'ckpt.last.pth.tar')
        task.save_checkpoint(
            ckpt_path,
            epoch=3,
            best_joint=0.75,
            best_full=0.80,
            best_missing=0.70,
            best_joint_epoch=2,
        )

        model2 = build_model(drbf_use_text_prior=False).to(device)
        task2 = MDTSegTeacher({'model': model2}, cfg)
        task2.scheduler = torch.optim.lr_scheduler.ConstantLR(task2.optimizer, factor=1.0)
        state = task2.load_training_checkpoint(ckpt_path)

        assert state['epoch'] == 3
        assert state['global_batch_step'] == 42
        assert abs(state['best_joint'] - 0.75) < 1e-6
        assert abs(state['best_full'] - 0.80) < 1e-6
        assert abs(state['best_missing'] - 0.70) < 1e-6
        assert state['best_joint_epoch'] == 2
        for k, v in model.state_dict().items():
            assert torch.allclose(v.cpu(), model2.state_dict()[k].cpu())
        print('  resume roundtrip OK', flush=True)


def test_train_mode_resolution():
    from run_mdt_seg import _resolve_train_mode

    print('=' * 72, flush=True)
    print('[Test] train_mode resolution', flush=True)

    cfg = SimpleNamespace(
        train_mode='scratch',
        stage1_checkpoint=None,
        resume_checkpoint=None,
    )
    mode, s1, rs = _resolve_train_mode(cfg)
    assert mode == 'scratch'

    with tempfile.NamedTemporaryFile(suffix='.pth.tar') as f:
        cfg = SimpleNamespace(
            train_mode='stage1_warmstart',
            stage1_checkpoint=f.name,
            resume_checkpoint=None,
        )
        mode, s1, rs = _resolve_train_mode(cfg)
        assert mode == 'stage1_warmstart'

    with tempfile.NamedTemporaryFile(suffix='.pth.tar') as f:
        cfg = SimpleNamespace(
            train_mode='resume',
            stage1_checkpoint=None,
            resume_checkpoint=f.name,
        )
        mode, s1, rs = _resolve_train_mode(cfg)
        assert mode == 'resume'
    print('  train_mode resolution OK', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        choices=('all', 'core', 'text', 'optimizer', 'equivalence', 'resume'),
        default='all',
    )
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--stage1_checkpoint', type=str, default=DEFAULT_STAGE1_CKPT)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}', flush=True)

    if args.stage in ('all', 'core'):
        test_pet_encoder_and_cppi_collect(device, size=args.size)
        test_missing_proxy_evidence(device, size=args.size)
        test_replace_and_text_off(device, size=args.size)
        test_train_mode_resolution()
    if args.stage in ('all', 'text'):
        test_text_on_frozen(device, size=args.size)
    if args.stage in ('all', 'optimizer'):
        test_optimizer_groups()
    if args.stage in ('all', 'equivalence'):
        test_stage1_equivalence(device, args.stage1_checkpoint, size=args.size)
    if args.stage in ('all', 'resume'):
        test_resume_roundtrip(device, size=args.size)

    print('\nALL REQUESTED SMOKE TESTS PASSED', flush=True)


if __name__ == '__main__':
    main()
