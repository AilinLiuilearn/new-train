# -*- coding: utf-8 -*-
"""Minimal inc stage-1 smoke: real encoders + decoder, 512 input, batch=1, AMP.

One Full forward/backward/optimizer-step/EMA-update with the increment
fusion; reports peak CUDA memory, finite loss and output shapes. Uses only
local offline pretrained weights; downloads nothing.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base import str2bool
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from run_mdt_seg import _optimizer_step_succeeded


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--amp', type=str2bool, default=True)
    p.add_argument('--decoder-norm', type=str, default='group', choices=('bn', 'group'))
    p.add_argument('--ct-pretrained-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    p.add_argument('--pet-pretrained-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'inc full smoke requires CUDA for the AMP + memory report'
    torch.manual_seed(0)
    device = torch.device(args.device)
    size = int(args.image_size)

    cfg = type('C', (), dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=args.ct_pretrained_path,
        pet_pretrained_path=args.pet_pretrained_path,
        decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
        asym_fusion_enabled=True, asym_use_text=False, fusion_version='inc',
        inc_region_grids=(16, 16, 16, 16), inc_num_points=4, inc_offset_radius=0.5,
        inc_query_chunk_size=512, inc_checkpoint_attention=True, inc_missing_policy='error',
        decoder_norm=args.decoder_norm, train_batch_mode='full',
        learning_rate=1e-4, weight_decay=1e-4,
        mixed_precision=bool(args.amp),
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
        ema_enabled=True, ema_decay=0.999, ema_decay_warmup=True, ema_start_epoch=0,
    ))()
    built = build_mdt_seg_teacher(cfg)
    task = MDTSegTeacher(built, cfg)
    task.scheduler = get_cosine_scheduler(task.optimizer, epochs=1, warmup_steps=0, steps_per_epoch=1)
    task.begin_epoch(1)
    print(f'[inc-smoke] fusion={type(task.model.fusion).__name__} '
          f'fusion_params={sum(p.numel() for p in task.model.fusion.parameters())} '
          f'norm={task.model.decoder_norm} policy={task.model.inc_missing_policy}', flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    batch = {'ct': torch.randn(1, 1, size, size, generator=g).to(device),
             'pet': torch.randn(1, 1, size, size, generator=g).to(device),
             'mask': (torch.rand(1, 1, size, size, generator=g) > 0.9).float().to(device)}
    task.optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=bool(args.amp)):
        loss, logits, _, stats = task.train_step(batch, forward_mode='full')
    assert torch.isfinite(loss), 'non-finite loss'
    if task.scaler.is_enabled():
        task.scaler.scale(loss).backward()
        task.scaler.unscale_(task.optimizer)
    else:
        loss.backward()
    assert any(p.grad is not None for p in task.model.fusion.parameters()), 'fusion got no grad'
    assert _optimizer_step_succeeded(task)
    task.scheduler.step()
    task.update_ema()
    task.model.eval()
    with torch.no_grad():
        eval_logits = task.eval_model()(batch['ct'], pet=batch['pet'], forward_mode='full')['logits']
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f'[inc-smoke] loss={float(loss):.4f} logits={tuple(logits.shape)} '
          f'eval={tuple(eval_logits.shape)} finite={bool(torch.isfinite(eval_logits).all())} '
          f'ema_updates={task.ema.updates} peak_cuda_MiB={peak:.1f} elapsed_s={time.time() - t0:.1f}',
          flush=True)
    print('[inc-smoke] PASS', flush=True)


if __name__ == '__main__':
    main()
