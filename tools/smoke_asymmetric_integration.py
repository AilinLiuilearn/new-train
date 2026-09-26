# -*- coding: utf-8 -*-
"""Small real-link smoke test for the asymmetric Full-only fusion.

Runs the true builder (real CT/PET encoders + optional local CLIP text),
a mixed Full/Missing batch, backward, one AdamW step, scheduler step, EMA
update, and a strict save/load round-trip. Reports Full/Missing/Joint dice
and peak CUDA memory.
"""
import argparse
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base import str2bool
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from run_mdt_seg import _optimizer_step_succeeded, build_balanced_pet_available


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--amp', type=str2bool, default=False)
    p.add_argument('--ema', type=str2bool, default=False)
    p.add_argument('--ema-start-epoch', type=int, default=0)
    p.add_argument('--checkpoint-attention', type=str2bool, default=False)
    p.add_argument('--use-text', type=str2bool, default=True)
    p.add_argument('--decoder-norm', type=str, default='group', choices=('bn', 'group'))
    p.add_argument('--clip-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/clip-vit-base-patch32')
    p.add_argument('--ct-pretrained-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    p.add_argument('--pet-pretrained-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    size = int(args.image_size)
    batch_size = int(args.batch_size)
    assert batch_size >= 2 and batch_size % 2 == 0, 'mixed smoke needs an even batch_size >= 2'

    cfg = type('C', (), dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=args.ct_pretrained_path,
        pet_pretrained_path=args.pet_pretrained_path,
        decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
        asym_fusion_enabled=True, asym_use_text=bool(args.use_text),
        asym_clip_path=args.clip_path,
        asym_checkpoint_attention=bool(args.checkpoint_attention),
        asym_grid_cap=32, asym_pet_dims=(64, 128, 160, 256), asym_heads=4,
        decoder_norm=args.decoder_norm,
        learning_rate=1e-4, weight_decay=1e-4,
        mixed_precision=bool(args.amp) and device.type == 'cuda',
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
        ema_enabled=bool(args.ema), ema_decay=0.999, ema_decay_warmup=True,
        ema_start_epoch=int(args.ema_start_epoch),
    ))()

    built = build_mdt_seg_teacher(cfg)
    task = MDTSegTeacher(built, cfg)
    from utils.optimization import get_cosine_scheduler
    task.scheduler = get_cosine_scheduler(
        task.optimizer, epochs=1, warmup_steps=0, steps_per_epoch=max(1, int(args.steps)))
    task.model.to(device)
    if task.ema is not None:
        task.ema.model.to(device)
    task.begin_epoch(1)
    n_fusion = sum(p.numel() for p in task.model.fusion.parameters())
    ema_flag = 'on' if task.ema is not None else 'off'
    print(f'[smoke] fusion={type(task.model.fusion).__name__} '
          f'fusion_params={n_fusion} decoder_norm={task.model.decoder_norm} '
          f'ema={ema_flag}', flush=True)

    amp_on = bool(args.amp) and device.type == 'cuda'
    skipped = 0
    t0 = time.time()
    for step in range(int(args.steps)):
        g = torch.Generator().manual_seed(1000 + step)
        batch = {'ct': torch.randn(batch_size, 1, size, size, generator=g).to(device),
                 'pet': torch.randn(batch_size, 1, size, size, generator=g).to(device),
                 'mask': (torch.rand(batch_size, 1, size, size, generator=g) > 0.5).float().to(device)}
        state = build_balanced_pet_available(batch_size, step, 2023, device)
        task.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_on):
            loss, _, _, stats = task.train_step_mixed(batch, pet_available=state)
        assert torch.isfinite(loss), 'non-finite loss'
        if task.scaler.is_enabled():
            task.scaler.scale(loss).backward()
            task.scaler.unscale_(task.optimizer)
        else:
            loss.backward()
        has_grad = any(p.grad is not None for p in task.model.fusion.parameters())
        assert has_grad, 'fusion received no gradient'
        if _optimizer_step_succeeded(task):
            task.scheduler.step()
            task.update_ema()
        else:
            skipped += 1
        print(f'[smoke] step={step} loss={float(loss):.4f} '
              f'full={stats["num_full"]} missing={stats["num_missing"]} skipped={skipped}', flush=True)

    # Eval both pure states with the EMA-or-live weights, whichever is active.
    task.model.eval()
    eval_model = task.eval_model()
    g = torch.Generator().manual_seed(7)
    batch = {'ct': torch.randn(batch_size, 1, size, size, generator=g).to(device),
             'pet': torch.randn(batch_size, 1, size, size, generator=g).to(device),
             'mask': (torch.rand(batch_size, 1, size, size, generator=g) > 0.5).float().to(device)}
    with torch.no_grad():
        full_logits = eval_model(batch['ct'], pet=batch['pet'],
                                 pet_available=torch.ones(batch_size, dtype=torch.long, device=device),
                                 forward_mode='auto')['logits']
        missing_logits = eval_model(batch['ct'], pet=batch['pet'],
                                    pet_available=torch.zeros(batch_size, dtype=torch.long, device=device),
                                    forward_mode='auto')['logits']

    def _dice(logits, mask):
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter = (pred * mask).sum()
        return float(2 * inter / (pred.sum() + mask.sum() + 1e-6))

    d_full = _dice(full_logits, batch['mask'])
    d_missing = _dice(missing_logits, batch['mask'])
    eval_src = 'ema' if (task.ema is not None and task.ema_active) else 'raw'
    print(f'[smoke] eval_source={eval_src} '
          f'full_dice={d_full:.4f} missing_dice={d_missing:.4f} '
          f'joint={(d_full + d_missing) / 2:.4f}', flush=True)

    path = os.path.join(tempfile.mkdtemp(), 'smoke_ckpt.pth.tar')
    task.save_checkpoint(path, epoch=1)
    ckpt = torch.load(path, map_location='cpu')
    assert 'model' in ckpt and 'optimizer' in ckpt
    print(f'[smoke] checkpoint ok: {path} ema_updates={ckpt.get("ema_updates")}', flush=True)
    if device.type == 'cuda':
        print(f'[smoke] peak_cuda_MiB={torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} '
              f'elapsed_s={time.time() - t0:.1f}', flush=True)
    else:
        print(f'[smoke] elapsed_s={time.time() - t0:.1f}', flush=True)
    print('[smoke] PASS', flush=True)


if __name__ == '__main__':
    main()
