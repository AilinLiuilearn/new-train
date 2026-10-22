"""Real encoder forward/backward/AdamW+EMA preflight, no real data required."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from run_staged import forward_loss, import_stage1, parser, build_fresh_model, normalize_fusion_config
from datasets.staged import split_signature
from utils.ema import ModelEMA
from utils.amp import make_grad_scaler, autocast_context
from utils.run_common import module_grad_norm
from utils.seg_losses import BCEDiceLoss


def check_finite_state(model):
    bad = [name for name, value in model.state_dict().items()
           if value.is_floating_point() and not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f'Nonfinite model parameters/buffers: {bad[:8]}')


def preflight_update(model, batch, stage, device, optimizer, scaler, ema, max_attempts=8):
    """Require one real finite update; AMP overflow skips never count as success.

    GradScaler must see unscaled Inf/NaN gradients before it can skip an update
    and reduce its scale. A nonfinite forward, FP32 gradient, or finite-gradient
    norm overflow is an error, not a reason to disable strict clipping.
    """
    check_finite_state(model)
    criterion = BCEDiceLoss()
    skipped = 0
    for attempt in range(1, max_attempts+1):
        optimizer.zero_grad(set_to_none=True)
        before = scaler.get_scale()
        with autocast_context(device, enabled=scaler.is_enabled()):
            loss, stats = forward_loss(model, batch, stage, device, criterion)
        print(f'[CHECK STEP {attempt}/{max_attempts}] loss={float(loss.detach()):.6f} '
              f'ct={stats["ct_loss"]:.6f} full={stats["full_loss"]:.6f} scale={before:g}', flush=True)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite forward loss; not a recoverable gradient-scale overflow. '
                               'Use --no-mixed_precision to diagnose FP32.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            if not scaler.is_enabled():
                raise RuntimeError(f'Nonfinite FP32 gradients: {bad[:8]}')
            # unscale_ recorded actual nonfinite elements. step MUST skip here;
            # never clip them or count a skipped step as a successful preflight.
            scaler.step(optimizer)
            scaler.update()
            after = scaler.get_scale()
            if after >= before:
                raise RuntimeError('GradScaler did not back off after nonfinite gradients')
            skipped += 1
            print(f'[CHECK AMP] skipped update; scale={before:g}->{after:g}; '
                  f'nonfinite_gradients={bad[:6]}', flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        # Keep strict clipping: finite elements can still have an overflowing norm.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        grads = {name: module_grad_norm(m) for name, m in model.named_children() if list(m.parameters())}
        if not all(math.isfinite(v) and v > 0 for v in grads.values()):
            raise AssertionError(f'Missing/nonfinite gradient: {grads}')
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            raise RuntimeError('Unexpected GradScaler skip after finite gradient validation')
        check_finite_state(model)
        ema.update(model)
        return loss, stats, grads, {'attempts': attempt, 'skipped_updates': skipped,
                                   'amp_scale': scaler.get_scale(), 'grad_norm_before_clip': float(norm)}
    raise RuntimeError(f'No finite optimizer update after {max_attempts} AMP attempts; '
                       f'last scale={scaler.get_scale():g}. '
                       'Use --no-mixed_precision to diagnose FP32; preflight has NOT passed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=int, choices=[1, 2], default=2)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--mixed_precision', action=argparse.BooleanOptionalAction, default=True,
                   help='Preflight precision; --no-mixed_precision diagnoses FP32 without changing training')
    init = p.add_mutually_exclusive_group()
    init.add_argument('--stage1_checkpoint')
    init.add_argument('--joint_from_pretrained', action='store_true')
    defaults = parser().parse_args([])
    p.add_argument('--ct_pretrained_path', default=defaults.ct_pretrained_path)
    p.add_argument('--pet_pretrained_path', default=defaults.pet_pretrained_path)
    p.add_argument('--no_pretrained', action='store_true', help='Random encoder smoke check only')
    p.add_argument('--decoder_norm', choices=['GN','BN'], default='GN')
    p.add_argument('--fusion_type', choices=['add', 'local_region_residual'], default='add')
    p.add_argument('--fusion_gaussian', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stage1_weights', default='auto', choices=['auto', 'raw', 'ema'])
    p.add_argument('--root', default='/root/autodl-tmp/data/PCLT20K')
    a = p.parse_args()
    if a.batch_size < 1 or a.image_size < 64 or a.image_size % 32:
        p.error('batch_size >= 1; image_size must be >=64 and a multiple of 32')
    if (a.joint_from_pretrained or a.stage1_checkpoint) and a.stage != 2:
        p.error('joint/task checkpoint initialization requires stage 2')
    torch.set_num_threads(1)
    device = torch.device(a.device)
    print(f'[CHECK] building real ConvNeXtV2-Nano / MiT-B1 + {a.decoder_norm} decoders', flush=True)
    cfg = parser().parse_args(['--stage', str(a.stage), '--root', a.root,
        '--stage1_weights', a.stage1_weights, '--image_size_2d', str(a.image_size),
        '--ct_pretrained_path', a.ct_pretrained_path, '--pet_pretrained_path', a.pet_pretrained_path,
        '--val_split_file', 'test.txt', '--test_split_file', 'test.txt'])
    cfg.joint_from_pretrained = a.joint_from_pretrained
    cfg.decoder_norm = a.decoder_norm
    cfg.fusion_type = a.fusion_type
    cfg.fusion_gaussian = a.fusion_gaussian
    normalize_fusion_config(cfg)
    cfg.no_pretrained = a.no_pretrained or not a.joint_from_pretrained
    model = build_fresh_model(cfg)
    if a.joint_from_pretrained:
        print('[CHECK INIT] direct joint training; no CT-only task checkpoint; '
              f'generic_pretrained={not cfg.no_pretrained}', flush=True)
    if a.stage1_checkpoint:
        import_stage1(model, a.stage1_checkpoint, cfg, split_signature(cfg))
    model.to(device).train()
    ema = ModelEMA(model, decay=.999, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scaler = make_grad_scaler(device, enabled=a.mixed_precision)
    shape = (a.batch_size, 3, a.image_size, a.image_size)
    batch = {'ct': torch.randn(shape), 'mask': (torch.rand(a.batch_size, 1, a.image_size, a.image_size) > .98).float()}
    if a.stage == 2:
        batch.update(pet=torch.randn(shape), is_full=torch.ones(a.batch_size, dtype=torch.bool))
    calls = {'ct': [], 'pet': []}
    handles = [model.enc_ct.register_forward_pre_hook(lambda m, args: calls['ct'].append(len(args[0])))]
    if a.stage == 2:
        handles.append(model.enc_pet.register_forward_pre_hook(lambda m, args: calls['pet'].append(len(args[0]))))
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    begin = time.time()
    loss, stats, grads, update = preflight_update(model, batch, a.stage, device, optimizer, scaler, ema)
    expected_calls = [a.batch_size]*update['attempts']
    if calls != {'ct': expected_calls, 'pet': expected_calls if a.stage == 2 else []}:
        raise AssertionError(f'Unexpected encoder coverage: {calls}')
    for handle in handles: handle.remove()
    optimizer.zero_grad(set_to_none=True)
    def forbidden(*unused): raise AssertionError('Missing entered a PET/Full-only module')
    hooks = []
    if a.stage == 2:
        hooks = [m.register_forward_pre_hook(forbidden) for m in
                 [model.enc_pet, model.pet_align, model.fusion, model.decoder_full]]
    with torch.inference_mode():
        model.eval()
        ct = batch['ct'][:1].to(device)
        first = model(ct, mode='missing')['logits']
        second = model(ct, torch.full_like(ct, float('nan')), mode='missing')['logits']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for hook in hooks: hook.remove()
    print(json.dumps({'stage': a.stage, 'batch': a.batch_size, 'size': a.image_size,
        'params': sum(p.numel() for p in model.parameters()), 'loss': float(loss.detach()),
        **stats, **update, 'gradients': grads, 'seconds': time.time()-begin,
        'encoder_calls': calls,
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else None,
        'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/1024**3 if device.type == 'cuda' else None,
        'missing_isolated': True}, indent=2), flush=True)


if __name__ == '__main__': main()
