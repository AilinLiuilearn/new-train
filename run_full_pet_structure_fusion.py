# -*- coding: utf-8 -*-
"""Full-only PET-guided structure fusion experiment (Stage-1 control).

Same Full training protocol as run_full_add_baseline.py, but the Full path
fuses with MultiScalePETGuidedStructureFusion instead of AddFusion:

  CT (ConvNeXtV2-Nano) -> ct_align, PET (MiT-B1) -> PET features,
  four-scale PET-guided structure fusion -> shared GroupNorm
  UNetStyleDecoder, plain BCE+Dice loss, model selection by val Full Dice.

No missing simulation, no Full/Missing split, no extra losses, no retrieval,
no generation, no compensation. Missing routing stays inherited (PET never
encoded, fusion never called); this entry only ever runs Full.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.build_pet_structure_fusion import build_pet_structure_model
from models.components.add_fusion import AddFusion
from models.dual_shared_pet_structure_fusion import MODEL_ARCH
from models.pet_guided_structure_fusion import MultiScalePETGuidedStructureFusion
from run_full_add_baseline import FullAddConfig, _loaders, train_step_full
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (count_parameters, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log


class FullPETStructureConfig(FullAddConfig):
    """Full-only; the new arch plus the module parameter table."""

    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        for action in p._actions:
            if action.dest == 'model_arch':
                action.default = MODEL_ARCH
                action.choices = (MODEL_ARCH,)
        p.add_argument('--fusion_type', type=str, default='pet_structure',
                       choices=('pet_structure', 'add'))
        p.add_argument('--fusion_inner_channels', type=int, default=32)
        p.add_argument('--fusion_heads', type=int, default=4)
        p.add_argument('--fusion_kernel_size', type=int, default=5)
        p.add_argument('--fusion_chunk_rows', type=int, default=16)
        p.add_argument('--fusion_checkpoint_chunks', type=str2bool, default=True)
        p.add_argument('--fusion_beta_init', type=float, default=0.0)
        p.add_argument('--fusion_structure_strength_init', type=float, default=0.1)
        p.add_argument('--fusion_use_pet_guidance', type=str2bool, default=True)
        p.add_argument('--fusion_use_structure_constraint', type=str2bool, default=True)
        p.add_argument('--fusion_ct_update_type', type=str, default='difference',
                       choices=('difference', 'content'))
        p.add_argument('--fusion_check_finite', type=str2bool, default=True)
        return p


def _assert_pet_structure_protocol(cfg, model):
    assert str(cfg.train_batch_mode) == 'full'
    assert str(cfg.model_arch) == MODEL_ARCH
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'
    fusion_type = str(cfg.fusion_type)
    if fusion_type == 'pet_structure':
        expected = MultiScalePETGuidedStructureFusion
    elif fusion_type == 'add':
        expected = AddFusion
    else:
        raise ValueError(f'Unknown fusion_type: {fusion_type!r}')
    if not isinstance(model.fusion, expected):
        raise RuntimeError(
            f'full_pet_structure_fusion with fusion_type={fusion_type!r} requires '
            f'{expected.__name__}, got {type(model.fusion).__name__}')


def _assert_checkpoint_fusion_type(ckpt, cfg):
    """Refuse cross-type restores: strict=True cannot tell buffer/param apart,
    and the module contract signature must match the recorded config."""
    requested = str(cfg.fusion_type)
    if requested not in ('pet_structure', 'add'):
        raise ValueError(f'Unknown requested fusion_type: {requested!r}')
    saved_cfg = ckpt.get('config', {})
    if not isinstance(saved_cfg, dict):
        raise TypeError('Checkpoint config must be a dict')
    saved_type = str(saved_cfg.get('fusion_type', 'pet_structure'
                                   if requested == 'pet_structure' else 'add'))
    if saved_type != requested:
        raise ValueError(
            f'Fusion mismatch: checkpoint={saved_type!r}, requested={requested!r}; '
            'do not cross-load fusion checkpoints')


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main():
    print('[INFO] starting Full-only PET-guided structure fusion experiment', flush=True)
    cfg = FullPETStructureConfig.parse_arguments()
    cfg.experiment_type = 'full_pet_structure_fusion'
    cfg.train_batch_mode = 'full'
    cfg.pet_missing_rate = 0.0
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_pet_structure_model(cfg), cfg)
    _assert_pet_structure_protocol(cfg, task.model)
    with open(os.path.join(cfg.checkpoint_dir, 'fusion_config.json'), 'w') as f:
        json.dump(task.model.fusion_config(), f, indent=2)
    total_params, trainable_params = count_parameters(task.model)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)
    if task.ema is not None:
        print(f'[INFO] ema_enabled=True decay={task.ema.decay} warmup={task.ema.warmup} '
              f'start_epoch={task.ema_start_epoch}', flush=True)
    else:
        print('[INFO] ema_enabled=False eval_uses_raw=True', flush=True)
    task.scheduler = get_cosine_scheduler(
        task.optimizer, epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr, steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    extra_headers = [
        'train_full_loss', 'train_samples', 'train_batches',
        'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc',
        'val_full_acc_pixel', 'val_full_hd95',
        'best_full', 'best_full_epoch',
        'grad_enc_ct', 'grad_enc_pet', 'grad_ct_align', 'grad_fusion', 'grad_decoder',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
                   extra_headers=extra_headers)

    best_full = -1.0
    best_full_epoch = 0
    no_improve = 0
    patience = int(cfg.early_stop_patience)
    amp_enabled = bool(cfg.mixed_precision)
    global_batch_step = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        train_loss_sum = 0.0
        train_sample_count = 0
        train_n = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        grads = {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'fusion': [], 'decoder': []}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, train_stats = train_step_full(task, batch)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grads['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads['enc_pet'].append(module_grad_norm(task.model.enc_pet))
            grads['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads['fusion'].append(module_grad_norm(task.model.fusion))
            grads['decoder'].append(module_grad_norm(task.model.decoder))
            total_grad_norm = (torch.nn.utils.clip_grad_norm_(
                task.trainable_parameters(), float(cfg.grad_clip))
                if float(cfg.grad_clip) > 0 else 0.0)
            grad_norm_accum += float(total_grad_norm)
            grad_norm_steps += 1
            if optimizer_step_succeeded(task):
                task.scheduler.step()
                task.update_ema()
            else:
                skipped_update_count += 1
            if (batch_idx + 1) % 100 == 0:
                print(f'[BATCH {batch_idx + 1}] route=full_fusion '
                      f'loss={float(loss.detach()):.6f}', flush=True)
            train_loss_sum += float(loss.detach()) * int(train_stats['num_samples'])
            train_sample_count += int(train_stats['num_samples'])
            train_n += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step

        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full',
                                 model=task.eval_model())
        improved = val_full['dice'] > best_full
        if improved:
            best_full = val_full['dice']
            best_full_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if improved:
            task.save_checkpoint(paths['best_full'], epoch, best_full,
                                 best_full_epoch, val_full)
        task.save_checkpoint(paths['last'], epoch, best_full, best_full_epoch, val_full)

        train_loss = train_loss_sum / max(1, train_sample_count)
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'), epoch, train_loss,
            {'total_loss': val_full['total_loss'], 'dice': val_full['dice'],
             'iou': val_full['iou'], 'acc': val_full['acc'],
             'acc_pixel': val_full.get('acc_pixel', 0.0), 'hd95': val_full['hd95']},
            lr=task.optimizer.param_groups[0]['lr'], grad_norm=avg_grad_norm,
            extra_metrics={
                'train_full_loss': train_loss,
                'train_samples': float(train_sample_count),
                'train_batches': float(train_n),
                'val_full_loss': val_full['total_loss'],
                'val_full_dice': val_full['dice'],
                'val_full_iou': val_full['iou'],
                'val_full_acc': val_full['acc'],
                'val_full_acc_pixel': val_full.get('acc_pixel', 0.0),
                'val_full_hd95': val_full['hd95'],
                'best_full': best_full,
                'best_full_epoch': best_full_epoch,
                'grad_enc_ct': float(np.mean(grads['enc_ct'])) if grads['enc_ct'] else 0.0,
                'grad_enc_pet': float(np.mean(grads['enc_pet'])) if grads['enc_pet'] else 0.0,
                'grad_ct_align': float(np.mean(grads['ct_align'])) if grads['ct_align'] else 0.0,
                'grad_fusion': float(np.mean(grads['fusion'])) if grads['fusion'] else 0.0,
                'grad_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else 0.0,
                'ema_enabled': 1.0 if task.ema is not None else 0.0,
                'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
                'skipped_updates': float(skipped_update_count),
                'epoch_time': time.time() - epoch_start,
            },
        )
        print(f'[EPOCH {epoch}] val_full_dice={val_full["dice"]:.4f} '
              f'best_full={best_full:.4f} '
              f'lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    if not os.path.isfile(paths['best_full']):
        raise RuntimeError('best_full checkpoint was not created')
    ckpt = MDTSegTeacher.load_state_dicts(paths['best_full'])
    _assert_checkpoint_fusion_type(ckpt, cfg)
    recorded = ckpt.get('eval_weights', 'raw')
    use_ema = (task.ema is not None and recorded == 'ema'
               and ckpt.get('model_ema') is not None)
    weights_tag = 'ema' if use_ema else 'raw'
    eval_model = build_pet_structure_model(cfg)['model'].to(task.device)
    _assert_pet_structure_protocol(cfg, eval_model)
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'],
                               strict=True)
    eval_model.eval()
    print(f'[FINAL] evaluating best_full checkpoint at epoch {ckpt["best_epoch"]} '
          f'with weights={weights_tag} (recorded={recorded}) '
          f'fusion={type(eval_model.fusion).__name__}', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full',
                              model=eval_model)
    result = {
        'experiment_type': 'full_pet_structure_fusion',
        'fusion_type': str(cfg.fusion_type),
        'dice': float(test_full['dice']),
        'iou': float(test_full['iou']),
        'hd95': float(test_full['hd95']),
        'loss': float(test_full['total_loss']),
        'weights': weights_tag,
        'best_epoch': ckpt['best_epoch'],
        'checkpoint': os.path.abspath(paths['best_full']),
    }
    with open(os.path.join(cfg.checkpoint_dir, 'final_test_full.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('[FINAL TEST FULL]', json.dumps(result, indent=2), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
