# -*- coding: utf-8 -*-
"""Bounded dcim stage-1 smoke: real encoders + offline PET text, 512, b1, AMP.

One Full forward/backward/optimizer-step/EMA-update, then a checkpoint
round-trip that rebuilds from the checkpoint text buffer while the CLIP
directory is pointed at a nonexistent path (proves resume needs no CLIP).
Raw and EMA eval outputs must match after strict reload. One batch only.
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
from utils.optimization import get_cosine_scheduler
from run_mdt_seg import _optimizer_step_succeeded


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--amp', type=str2bool, default=True)
    p.add_argument('--decoder-norm', type=str, default='group', choices=('bn', 'group'))
    p.add_argument('--clip-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/clip-vit-base-patch32')
    p.add_argument('--ct-pretrained-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
    p.add_argument('--pet-pretrained-path', type=str,
                   default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
    return p.parse_args()


def _cfg(args):
    return type('C', (), dict(
        ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
        ct_pretrained_path=args.ct_pretrained_path,
        pet_pretrained_path=args.pet_pretrained_path,
        decoder_channels=(512, 256, 128, 64), use_deep_supervision=False,
        asym_fusion_enabled=True, asym_use_text=True, asym_clip_path=args.clip_path,
        text_encoder='clip', text_encoder_path='', text_encoder_vocab='',
        asym_checkpoint_attention=False, asym_grid_cap=32,
        asym_pet_dims=(64, 128, 160, 256), asym_heads=4, fusion_version='dcim',
        dcim_region_chunk_size=128, dcim_checkpoint_attention=True,
        decoder_norm=args.decoder_norm, train_batch_mode='full',
        learning_rate=1e-4, weight_decay=1e-4,
        mixed_precision=bool(args.amp),
        loss_smooth=1.0, bce_weight=1.0, dice_weight=1.0, random_state=2023,
        ema_enabled=True, ema_decay=0.999, ema_decay_warmup=True, ema_start_epoch=0,
    ))()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'dcim smoke requires CUDA'
    torch.manual_seed(0)
    device = torch.device(args.device)
    size = int(args.image_size)

    task = MDTSegTeacher(build_mdt_seg_teacher(_cfg(args)), _cfg(args))
    print(f'[dcim-smoke] encoders=convnextv2_nano/mit_b1 '
          f'fusion={type(task.model.fusion).__name__} '
          f'text={tuple(task.model.fusion.text_embeddings.shape)} '
          f'norm={task.model.decoder_norm}', flush=True)
    assert type(task.model.enc_ct).__name__ != 'FallbackFeatureBackbone', 'CT fallback used'
    assert type(task.model.enc_pet).__name__ != 'FallbackFeatureBackbone', 'PET fallback used'
    task.scheduler = get_cosine_scheduler(task.optimizer, epochs=1, warmup_steps=0, steps_per_epoch=1)
    task.begin_epoch(1)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    g = torch.Generator().manual_seed(0)
    batch = {'ct': torch.randn(1, 1, size, size, generator=g).to(device),
             'pet': torch.randn(1, 1, size, size, generator=g).to(device),
             'mask': (torch.rand(1, 1, size, size, generator=g) > 0.9).float().to(device)}
    task.optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=bool(args.amp)):
        loss, logits, _, _ = task.train_step(batch, forward_mode='full')
    assert torch.isfinite(loss), 'non-finite loss'
    if task.scaler.is_enabled():
        task.scaler.scale(loss).backward()
        task.scaler.unscale_(task.optimizer)
    else:
        loss.backward()
    assert any(p.grad is not None for p in task.model.fusion.parameters()), 'fusion got no grad'
    assert any(p.grad is not None for p in task.model.enc_ct.parameters()), 'CT encoder got no grad'
    assert any(p.grad is not None for p in task.model.enc_pet.parameters()), 'PET encoder got no grad'
    assert _optimizer_step_succeeded(task)
    task.scheduler.step()
    task.update_ema()
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f'[dcim-smoke] loss={float(loss):.4f} logits={tuple(logits.shape)} '
          f'ema_updates={task.ema.updates} peak_cuda_MiB={peak:.1f} elapsed_s={time.time() - t0:.1f}',
          flush=True)

    # Checkpoint round-trip with CLIP made unavailable.
    path = os.path.join(tempfile.mkdtemp(), 'dcim_smoke_ckpt.pth.tar')
    task.save_checkpoint(path, epoch=1)
    ckpt = torch.load(path, map_location='cpu')
    assert 'model' in ckpt and 'model_ema' in ckpt and 'optimizer' in ckpt
    assert 'fusion.text_embeddings' in ckpt['model'], 'text buffer missing from checkpoint'
    cfg2 = _cfg(args)
    cfg2.asym_clip_path = '/nonexistent/clip_dir_for_resume_test'
    rebuilt = MDTSegTeacher(
        build_mdt_seg_teacher(cfg2,
                              fusion_text_embeddings=ckpt['model']['fusion.text_embeddings']),
        cfg2)
    rebuilt.model.load_state_dict(ckpt['model'], strict=True)
    rebuilt.model.eval()
    task.model.eval()
    with torch.no_grad():
        ref = task.model(batch['ct'], pet=batch['pet'], forward_mode='full')['logits']
        got = rebuilt.model(batch['ct'], pet=batch['pet'], forward_mode='full')['logits']
    assert torch.allclose(ref.cpu(), got.cpu(), rtol=0, atol=0), 'raw rebuild mismatch'
    rebuilt_ema_sd = ckpt['model_ema']
    rebuilt.model.load_state_dict(rebuilt_ema_sd, strict=True)
    with torch.no_grad():
        got_ema = rebuilt.model(batch['ct'], pet=batch['pet'], forward_mode='full')['logits']
        ref_ema = task.eval_model()(batch['ct'], pet=batch['pet'], forward_mode='full')['logits']
    assert torch.allclose(ref_ema.cpu(), got_ema.cpu(), rtol=0, atol=0), 'EMA rebuild mismatch'
    print('[dcim-smoke] resume-without-CLIP strict roundtrip OK (raw + EMA)', flush=True)
    print('[dcim-smoke] PASS', flush=True)


if __name__ == '__main__':
    main()
