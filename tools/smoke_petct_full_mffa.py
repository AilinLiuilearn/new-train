# -*- coding: utf-8 -*-
"""Integrated smoke test for the Full MFFA fusion inside the REAL baseline.

Two modes:

    python tools/smoke_petct_full_mffa.py --integrated --device cpu --image-size 128 --batch-size 2 --amp

builds the actual ConvNeXtV2-Nano CT encoder, MiT-B1 PET encoder, the shared
UNetStyleDecoder and BCEDiceLoss, then checks: mixed routing, real PET encoded
only for Full rows at the fusion level, one optimizer + one scheduler step,
Missing rows producing the exact CT decode, NaN-safe Missing PET isolation,
and output shapes.

Without ``--integrated`` the tool delegates to the standalone module smoke
(``python models/petct_full_mffa.py --smoke-test``).

Baseline contract kept on purpose: the PET encoder still encodes the whole
batch and Missing rows are zeroed afterwards (encode-then-zero). Only Full rows
take the MFFA residual; Missing rows decode the CT identity.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.build_mdt_seg import build_mdt_seg_teacher, FallbackFeatureBackbone
from tasks.mdt_seg import MDTSegTeacher
from run_mdt_seg import build_balanced_pet_available


def _cfg(mffa_enabled=True, checkpoint_attention=False, amp=False):
    base = {
        'ct_backbone': 'convnextv2_nano',
        'pet_backbone': 'mit_b1',
        'ct_pretrained_path': None,
        'pet_pretrained_path': None,
        'decoder_channels': (512, 256, 128, 64),
        'use_deep_supervision': False,
        'deep_supervision': False,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'mixed_precision': bool(amp),
        'loss_smooth': 1.0,
        'bce_weight': 1.0,
        'dice_weight': 1.0,
        'random_state': 2023,
        'train_batch_mode': 'mixed',
        'mffa_enabled': bool(mffa_enabled),
        'mffa_checkpoint_attention': bool(checkpoint_attention),
    }
    return type('C', (), base)()


def _batch(batch_size, size, device, dtype=torch.float32):
    return {
        'ct': torch.randn(batch_size, 1, size, size, device=device, dtype=dtype),
        'pet': torch.randn(batch_size, 1, size, size, device=device, dtype=dtype),
        'mask': (torch.rand(batch_size, 1, size, size, device=device) > 0.5).to(dtype),
    }


def _count_calls(module):
    box = {'n': 0}

    def hook(*_args, **_kwargs):
        box['n'] += 1

    h = module.register_forward_pre_hook(hook)
    return box, h


def run_integrated(args):
    torch.manual_seed(2023)
    device = torch.device(args.device)
    report = {'mode': 'integrated', 'device': str(device), 'torch_version': torch.__version__,
              'scope': 'real backbones + shared decoder + BCE/Dice'}

    task = MDTSegTeacher(build_mdt_seg_teacher(_cfg(True, args.checkpoint_attention, args.amp)), _cfg(True, args.checkpoint_attention, args.amp))
    # Respect the requested device even though MDTSegTeacher defaults to CUDA.
    task.device = device
    task.model.to(device)
    model = task.model
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Real backbones, no fallback CNN pretending to be the encoders.
    assert not isinstance(model.enc_ct, FallbackFeatureBackbone), 'CT backbone fell back to CNN'
    assert not isinstance(model.enc_pet, FallbackFeatureBackbone), 'PET backbone fell back to CNN'
    assert model.mffa_enabled and type(model.fusion).__name__ == 'PETCTFullMFFA'
    report['ct_encoder'] = type(model.enc_ct).__name__
    report['pet_encoder'] = type(model.enc_pet).__name__
    report['fusion'] = type(model.fusion).__name__
    report['mffa_params'] = model.fusion.parameter_report()['total']

    batch = _batch(args.batch_size, args.image_size, device)
    state = build_balanced_pet_available(args.batch_size, 0, 2023, device)

    use_amp = bool(args.amp or args.flash_only)
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    fused_backend, backend_name = None, 'disabled (auto SDPA)'
    if args.flash_only:
        if device.type != 'cuda':
            raise SystemExit('flash-only requires CUDA')
        fused_backend, backend_name = _flash_context()
    report['attention_backend'] = backend_name
    report['amp_dtype'] = str(amp_dtype) if use_amp else 'disabled'
    flash_ctx = fused_backend if fused_backend is not None else _null()

    with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp), flash_ctx:
        # --- one mixed step with optimizer + scheduler ---
        task.scheduler = torch.optim.lr_scheduler.LambdaLR(task.optimizer, lr_lambda=lambda s: 1.0)
        model.train()
        task.optimizer.zero_grad(set_to_none=True)
        loss, _, _, stats = task.train_step_mixed(batch, pet_available=state, missing_loss_weight=1.0)
        assert torch.isfinite(loss), 'non-finite mixed loss'
        loss.backward()
        task.optimizer.step()
        task.scheduler.step()
        report['num_full'] = int(stats['num_full'])
        report['num_missing'] = int(stats['num_missing'])
        report['mixed_loss'] = float(loss.detach())

        full_idx = state.eq(1)
        missing_idx = state.eq(0)
        mffa_grad = sum(float(p.grad.abs().sum()) for p in model.fusion.parameters() if p.grad is not None)
        ct_grad = sum(float(p.grad.abs().sum()) for p in model.enc_ct.parameters() if p.grad is not None)
        dec_grad = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)
        pet_grad = sum(float(p.grad.abs().sum()) for p in model.enc_pet.parameters() if p.grad is not None)
        assert mffa_grad > 0 and ct_grad > 0 and dec_grad > 0, 'missing gradient on CT/MFFA/decoder'
        report['grad_norms'] = {'mffa': mffa_grad, 'ct': ct_grad, 'decoder': dec_grad, 'pet': pet_grad}

        # --- routing: PET encoder called once per forward; MFFA called once ---
        model.eval()
        pet_calls, h1 = _count_calls(model.enc_pet)
        mffa_calls, h2 = _count_calls(model.fusion)
        with torch.no_grad():
            out_auto = model(batch['ct'], batch['pet'], pet_available=state, forward_mode='auto')['logits']
        h1.remove(); h2.remove()
        assert pet_calls['n'] == 1, f'PET encoder calls in auto={pet_calls["n"]}'
        assert mffa_calls['n'] == 1, f'MFFA calls in auto={mffa_calls["n"]}'
        report['pet_encoder_calls_auto'] = pet_calls['n']
        report['mffa_calls_auto'] = mffa_calls['n']

        # --- Missing rows equal the CT-only decode in eval (exact identity) ---
        with torch.no_grad():
            ct_feats = model._encode_ct(batch['ct'])
            ref_auto = model._decode(list(ct_feats), batch['ct'].shape[-2:])['logits']
            ref_missing = model(batch['ct'], batch['pet'], forward_mode='missing')['logits']
        assert torch.allclose(out_auto[missing_idx], ref_auto[missing_idx], atol=1e-4, rtol=1e-3), 'auto Missing != CT decode'
        assert torch.allclose(ref_missing, ref_auto, atol=1e-4, rtol=1e-3), 'missing-mode logits != CT decode'
        assert not torch.allclose(out_auto[full_idx], ref_auto[full_idx], atol=1e-5), 'Full rows did not use MFFA'
        report['missing_is_ct_identity'] = True
        # --- PET encoded for the pure-missing path (baseline encode-then-zero) ---
        pet_calls2, h3 = _count_calls(model.enc_pet)
        with torch.no_grad():
            model(batch['ct'], batch['pet'], forward_mode='missing')
        h3.remove()
        assert pet_calls2['n'] == 1, 'pure-missing path must still encode PET (encode-then-zero)'
        report['pet_encoder_calls_missing'] = pet_calls2['n']

        # --- Missing PET isolation; Full rows depend on PET ---
        pet_alt = batch['pet'].clone()
        pet_alt[full_idx] = torch.randn_like(pet_alt[full_idx]) * 5.0
        with torch.no_grad():
            out_alt = model(batch['ct'], pet_alt, pet_available=state, forward_mode='auto')['logits']
        assert not torch.allclose(out_alt[full_idx], out_auto[full_idx], atol=1e-6), 'Full rows ignored PET'
        assert torch.allclose(out_alt[missing_idx], out_auto[missing_idx], atol=1e-6), 'Full-row PET leaked into Missing rows'

        pet_alt2 = batch['pet'].clone()
        pet_alt2[missing_idx] = torch.randn_like(pet_alt2[missing_idx]) * 5.0
        with torch.no_grad():
            out_alt2 = model(batch['ct'], pet_alt2, pet_available=state, forward_mode='auto')['logits']
        assert torch.allclose(out_alt2[missing_idx], out_auto[missing_idx], atol=1e-6), 'Missing rows depend on their PET'
        report['missing_pet_isolated'] = True

        report['output_shape'] = list(out_auto.shape)

    # --- checkpoint strict round-trip ---
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fresh = MDTSegTeacher(build_mdt_seg_teacher(_cfg(True, args.checkpoint_attention, False)), _cfg(True, args.checkpoint_attention, False))
    msg = fresh.model.load_state_dict(sd, strict=True)
    assert not msg.missing_keys and not msg.unexpected_keys
    report['strict_reload'] = True

    report['parameters_total'] = sum(p.numel() for p in model.parameters())
    if device.type == 'cuda':
        report['peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        report['peak_reserved_MiB'] = torch.cuda.max_memory_reserved(device) / 2 ** 20
        report['gpu_name'] = torch.cuda.get_device_name(device)
    print(json.dumps(report, indent=2))
    return report


def _flash_context():
    """Force the Flash SDPA backend on the current PyTorch/CUDA build.

    Returns (context_manager, human_readable_backend_name). Raises if this
    build cannot force Flash so the caller never silently falls back to a
    different attention algorithm.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION), 'sdpa_kernel(FLASH_ATTENTION)'
    except Exception:
        pass
    if hasattr(torch.backends.cuda, 'sdp_kernel'):
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        return ctx, 'torch.backends.cuda.sdp_kernel(enable_flash=True, math/mem_efficient=False)'
    raise SystemExit('this PyTorch build cannot force Flash attention (report plainly, do not silently change the algorithm)')


def _null():
    from contextlib import nullcontext
    return nullcontext()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--integrated', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--checkpoint-attention', action='store_true')
    p.add_argument('--flash-only', action='store_true')
    args = p.parse_args()
    if args.batch_size < 2 or args.batch_size % 2 != 0:
        p.error('integrated smoke requires an even batch-size >= 2')
    if not args.integrated:
        print('[INFO] --integrated not set; delegating to module smoke.')
        from models.petct_full_mffa import _main as module_main
        sys.argv = [sys.argv[0], '--smoke-test', '--device', args.device, '--batch-size', str(args.batch_size)]
        if args.amp:
            sys.argv.append('--amp')
        if args.checkpoint_attention:
            sys.argv.append('--checkpoint-attention')
        module_main()
        return
    start = time.perf_counter()
    report = run_integrated(args)
    report['seconds'] = time.perf_counter() - start
    if torch.cuda.is_available() and args.device.startswith('cuda'):
        report['peak_allocated_MiB'] = torch.cuda.max_memory_allocated() / 2 ** 20
    print(json.dumps({'integrated_seconds': report['seconds']}, indent=2))


if __name__ == '__main__':
    main()
