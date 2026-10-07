# -*- coding: utf-8 -*-
"""Full-only CT+PET local-contrast-fusion experiment.

Same Full training protocol as run_full_add_baseline.py, but the Full path
fuses with LocalContrastFusionPyramid instead of AddFusion:

  CT (ConvNeXtV2-Nano) -> ct_align, PET (MiT-B1) -> PET features,
  four-scale LocalContrastFusionPyramid -> shared GroupNorm UNetStyleDecoder,
  plain BCE+Dice loss, model selection by val Full Dice.

No missing simulation, no Full/Missing split, no extra losses.
Missing routing stays inherited (PET never encoded, fusion never called);
this entry only ever runs Full.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.build_local_contrast_fusion import build_local_contrast_fusion_model
from models.dual_shared_local_contrast_fusion import MODEL_ARCH
from models.gaussian_response_descriptor import sync_fixed_gaussian_buffers
from models.local_contrast_bidirectional_fusion import LocalContrastFusionPyramid
from run_full_add_baseline import train_step_full
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import HD95_PROTOCOL_VERSION
from utils.optimization import get_cosine_scheduler
from utils.run_common import (count_parameters, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log


class FullLocalContrastFusionConfig(SegMDTConfig):
    """Shared data/training params; full mode and the new arch only."""

    @staticmethod
    def train_parser():
        p = SegMDTConfig.train_parser()
        for action in p._actions:
            if action.dest == 'train_batch_mode':
                action.default = 'full'
                action.choices = ('full',)
        p.add_argument('--grad_log_interval', type=int, default=100,
                       help='Compute module grad-norm diagnostics every N steps; '
                            'other steps log empty values, clipping/step/EMA unchanged.')
        p.add_argument('--eval_amp', type=str2bool, default=False,
                       help='Run validation/final-test forward under AMP autocast; '
                            'default False preserves the FP32 protocol.')
        return p

    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        for action in p._actions:
            if action.dest == 'model_arch':
                action.default = MODEL_ARCH
                action.choices = (MODEL_ARCH,)
        p.add_argument('--fusion_dim', type=int, default=32)
        p.add_argument('--fusion_heads', type=int, default=4)
        p.add_argument('--fusion_window', type=int, default=5)
        p.add_argument('--fusion_chunk_rows', type=int, default=16)
        p.add_argument('--fusion_checkpoint_chunks', type=str2bool, default=True)
        p.add_argument('--fusion_position_bias_beta', type=float, default=0.0,
                       help='Spatial-distance init for relative_bias: '
                            '-beta*(dy^2+dx^2)/radius^2; 0.0 reproduces old init.')
        p.add_argument('--fusion_descriptor_type', type=str, default='contrast',
                       choices=('contrast', 'rasfe_fixed', 'rasfe_learnable'),
                       help='Descriptor used by both modalities at all four scales.')
        p.add_argument('--check_finite', type=str2bool, default=True,
                       help='Intermediate NaN/Inf checks; logits/loss/update '
                            'checks always stay on.')
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(),
                   cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        config._ensure_hash()
        return config


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
        allow_val_equals_test=bool(getattr(cfg, 'allow_val_equals_test', False)),
        ct_only=False,
    )


def _assert_fusion_protocol(cfg, model):
    assert str(cfg.train_batch_mode) == 'full'
    assert str(cfg.model_arch) == MODEL_ARCH
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'
    if not isinstance(model.fusion, LocalContrastFusionPyramid):
        raise RuntimeError(
            'full_local_contrast_fusion requires LocalContrastFusionPyramid, '
            f'got {type(model.fusion).__name__}')
    requested = str(cfg.fusion_descriptor_type)
    if model.fusion.descriptor_type != requested:
        raise RuntimeError(
            f'Fusion descriptor mismatch: model={model.fusion.descriptor_type!r}, '
            f'config={requested!r}')
    for i, block in enumerate(model.fusion.scales):
        if block.descriptor_type != requested:
            raise RuntimeError(
                f'Scale {i} descriptor mismatch: {block.descriptor_type!r} '
                f'!= {requested!r}')


def _assert_checkpoint_descriptor_type(ckpt, requested_type):
    valid = ('contrast', 'rasfe_fixed', 'rasfe_learnable')
    if requested_type not in valid:
        raise ValueError(f'Unknown requested descriptor: {requested_type!r}')
    saved_cfg = ckpt.get('config', {})
    if not isinstance(saved_cfg, dict):
        raise TypeError('Checkpoint config must be a dict')
    saved_type = saved_cfg.get('fusion_descriptor_type', 'contrast')
    if saved_type not in valid:
        raise ValueError(f'Unknown checkpoint descriptor: {saved_type!r}')
    if saved_type != requested_type:
        raise ValueError(
            f'Descriptor mismatch: checkpoint={saved_type!r}, '
            f'requested={requested_type!r}; do not cross-load fusion checkpoints')


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main():
    print('[INFO] starting Full-only local-contrast fusion experiment', flush=True)
    cfg = FullLocalContrastFusionConfig.parse_arguments()
    cfg.experiment_type = 'full_local_contrast_fusion'
    cfg.train_batch_mode = 'full'
    cfg.fusion_type = 'LocalContrastFusionPyramid'
    cfg.pet_missing_rate = 0.0
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_local_contrast_fusion_model(cfg), cfg)
    _assert_fusion_protocol(cfg, task.model)
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
        'grad_sampled_steps',
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
    eval_amp = bool(cfg.eval_amp)
    grad_log_interval = max(1, int(cfg.grad_log_interval))
    print(f'[INFO] eval_precision={"amp" if eval_amp else "fp32"} '
          f'grad_log_interval={grad_log_interval} (sampled grad norms; '
          f'clipping/step/EMA unchanged)', flush=True)
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
            log_grads = (global_batch_step % grad_log_interval == 0)
            if log_grads:
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
                if cfg.fusion_descriptor_type == 'rasfe_fixed' and task.ema is not None:
                    sync_fixed_gaussian_buffers(task.ema.model, task.model)
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
                                 model=task.eval_model(), eval_amp=eval_amp)
        val_full['eval_precision'] = 'amp' if eval_amp else 'fp32'
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
                'grad_enc_ct': float(np.mean(grads['enc_ct'])) if grads['enc_ct'] else '',
                'grad_enc_pet': float(np.mean(grads['enc_pet'])) if grads['enc_pet'] else '',
                'grad_ct_align': float(np.mean(grads['ct_align'])) if grads['ct_align'] else '',
                'grad_fusion': float(np.mean(grads['fusion'])) if grads['fusion'] else '',
                'grad_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else '',
                'grad_sampled_steps': float(len(grads['fusion'])),
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
    _assert_checkpoint_descriptor_type(ckpt, cfg.fusion_descriptor_type)
    recorded = ckpt.get('eval_weights', 'raw')
    use_ema = (task.ema is not None and recorded == 'ema'
               and ckpt.get('model_ema') is not None)
    weights_tag = 'ema' if use_ema else 'raw'
    eval_model = build_local_contrast_fusion_model(cfg)['model'].to(task.device)
    if not isinstance(eval_model.fusion, LocalContrastFusionPyramid):
        raise RuntimeError('final test must rebuild the fusion model, '
                           f'got {type(eval_model.fusion).__name__}')
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'],
                               strict=True)
    eval_model.eval()
    print(f'[FINAL] evaluating best_full checkpoint at epoch {ckpt["best_epoch"]} '
          f'with weights={weights_tag} (recorded={recorded})', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full',
                              model=eval_model, eval_amp=eval_amp)
    result = {
        'experiment_type': 'full_local_contrast_fusion',
        'dice': float(test_full['dice']),
        'iou': float(test_full['iou']),
        'hd95': float(test_full['hd95']),
        'loss': float(test_full['total_loss']),
        'weights': weights_tag,
        'eval_precision': 'amp' if eval_amp else 'fp32',
        'hd95_protocol': HD95_PROTOCOL_VERSION,
        'descriptor_type': str(cfg.fusion_descriptor_type),
        'best_epoch': ckpt['best_epoch'],
        'checkpoint': os.path.abspath(paths['best_full']),
    }
    with open(os.path.join(cfg.checkpoint_dir, 'final_test_full.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('[FINAL TEST FULL]', json.dumps(result, indent=2), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()