# -*- coding: utf-8 -*-
"""Reproducible benchmark for the local-contrast fusion experiment.

Measures, with real four-scale sizes and real batch sizes:
  1. four-scale fusion module forward (+backward);
  2. full model pure forward;
  3. full training step (forward + BCE+Dice + backward + optimizer step);
  4. dataloader batch time, full evaluate() time, CPU HD95 time.

CUDA timing uses explicit synchronization; warmup/compile time reported
separately. No per-batch sync is added to training code by this script.
Writes a JSON report with env, precision, peak memory, median/P95 and
profiler top operators.
"""
import argparse
import json
import os
import statistics
import time

import torch

from models.dual_shared_local_contrast_fusion import DualSharedLocalContrastFusionModel
from models.local_contrast_bidirectional_fusion import LocalContrastFusionPyramid
from utils.metrics_seg import compute_hd95_pair
from utils.seg_losses import BCEDiceLoss

CHANNELS = (64, 128, 320, 512)
SIZES = (128, 64, 32, 16)


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def _timed(fn, repeats, device, warmup=5):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mid = times[len(times) // 2]
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    return {'median_ms': mid, 'p95_ms': p95, 'n': len(times)}


def _profile_top_ops(fn, device, top=12):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA] if device == 'cuda'
            else [torch.profiler.ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            fn()
    _sync(device)
    table = prof.key_averages().table(sort_by='cuda_time_total' if device == 'cuda'
                                      else 'cpu_time_total', row_limit=top)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--chunk-rows', type=int, default=16)
    ap.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--descriptor-type', default='contrast',
                    choices=('contrast', 'rasfe_fixed', 'rasfe_learnable'))
    ap.add_argument('--position-bias-beta', type=float, default=0.0)
    ap.add_argument('--out', default='bench_local_contrast.json')
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(2023)
    report = {
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'batch_size': args.batch_size,
        'chunk_rows': args.chunk_rows,
        'checkpoint': args.checkpoint,
        'amp': args.amp,
        'descriptor_type': args.descriptor_type,
        'position_bias_beta': args.position_bias_beta,
    }

    module = LocalContrastFusionPyramid(
        chunk_rows=args.chunk_rows, checkpoint_chunks=args.checkpoint,
        position_bias_beta=args.position_bias_beta,
        descriptor_type=args.descriptor_type).to(device)
    ct = [torch.randn(args.batch_size, c, s, s, device=device) for c, s in zip(CHANNELS, SIZES)]
    pet = [torch.randn_like(t) for t in ct]

    t0 = time.perf_counter()
    module.train()
    report['fusion_fwd'] = _timed(lambda: module(ct, pet), args.repeats, device)
    loss_fn = BCEDiceLoss()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def fwd_bwd():
        module.zero_grad(set_to_none=True)
        out = module(ct, pet)
        sum(f.square().mean() for f in out).backward()

    report['fusion_fwd_bwd'] = _timed(fwd_bwd, max(5, args.repeats // 4), device)
    report['fusion_peak_MiB'] = (torch.cuda.max_memory_allocated() / 2 ** 20
                                 if device == 'cuda' else 0.0)
    report['fusion_params'] = sum(p.numel() for p in module.parameters())

    model = DualSharedLocalContrastFusionModel(
        ct_pretrained_path=None, pet_pretrained_path=None, pretrained=False,
        fusion_kwargs={'descriptor_type': args.descriptor_type,
                       'position_bias_beta': args.position_bias_beta}).to(device)
    model.eval()
    ct_img = torch.randn(args.batch_size, 1, 512, 512, device=device)
    pet_img = torch.randn(args.batch_size, 1, 512, 512, device=device)
    # NOTE (fixed 2026-10-07): amp_ctx was previously constructed but never
    # entered, so model_fwd always timed FP32 even with --amp. Now the
    # autocast context is actually applied; report['amp'] reflects reality.
    # The old 195ms model_fwd number was FP32, not AMP.
    use_amp = bool(args.amp and device == 'cuda')
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type='cuda', enabled=True):
                report['model_fwd'] = _timed(
                    lambda: model(ct_img, pet=pet_img, forward_mode='full'),
                    args.repeats, device)
        else:
            report['model_fwd'] = _timed(
                lambda: model(ct_img, pet=pet_img, forward_mode='full'), args.repeats, device)
    report['model_fwd_precision'] = 'amp' if use_amp else 'fp32'

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mask = (torch.rand(args.batch_size, 1, 512, 512, device=device) > 0.5).float()

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=(args.amp and device == 'cuda')):
            logits = model(ct_img, pet=pet_img, forward_mode='full')['logits']
            loss, _ = loss_fn(logits, mask)
        loss.backward()
        opt.step()

    report['train_step'] = _timed(train_step, max(5, args.repeats // 4), device)
    if device == 'cuda':
        report['train_peak_MiB'] = torch.cuda.max_memory_allocated() / 2 ** 20
    report['profiler_top_ops'] = _profile_top_ops(
        lambda: model(ct_img, pet=pet_img, forward_mode='full'), device)

    import numpy as np
    rng = np.random.default_rng(0)
    masks = [(rng.random((512, 512)) > 0.7) for _ in range(20)]
    pairs = [(masks[i], masks[(i + 1) % 20]) for i in range(20)]
    t0 = time.perf_counter()
    for a, b in pairs:
        compute_hd95_pair(a, b)
    report['hd95_cpu_20pairs_ms'] = (time.perf_counter() - t0) * 1000.0

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != 'profiler_top_ops'}, indent=2))
    print('saved', os.path.abspath(args.out))


if __name__ == '__main__':
    main()