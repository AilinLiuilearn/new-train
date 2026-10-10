# -*- coding: utf-8 -*-
"""Full-only background-referenced region fusion experiment (stage-1).

Same Full training protocol as run_full_add_baseline.py, but the Full path
fuses with BackgroundReferencedRegionFusion instead of AddFusion:

  CT (ConvNeXtV2-Nano) -> ct_align, PET (MiT-B1) -> pet_align,
  four-scale background-referenced region fusion -> shared GroupNorm
  UNetStyleDecoder, BCE+Dice plus a small Gaussian-NLL background term.

Background supervision region: non-foreground pixels (no independent tissue
ROI exists in this dataset, so black canvas is included; see startup log).
Model selection by val Full Dice; final test rebuilds the same model.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.background_referenced_region_fusion import (
    BackgroundReferencedRegionFusion,
    FusionResult,
    make_background_train_mask,
    save_fusion_diagnostics,
)
from models.full_background_referenced_model import (
    MODEL_ARCH,
    build_full_background_referenced_model,
)
from run_full_add_baseline import FullAddConfig, _loaders
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (count_parameters, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log


class FullBackgroundReferencedConfig(FullAddConfig):
    """Full-only; the new arch plus background-loss / visualization options."""

    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        for action in p._actions:
            if action.dest == 'model_arch':
                action.default = MODEL_ARCH
                action.choices = (MODEL_ARCH,)
        p.add_argument('--brlc_bg_loss_weight', type=float, default=0.01)
        p.add_argument('--brlc_bg_exclusion_radius', type=int, default=2)
        p.add_argument('--brlc_vis_every', type=int, default=0,
                       help='0 disables; otherwise export one fixed val sample every N epochs')
        p.add_argument('--brlc_descriptor_dim', type=int, default=32)
        p.add_argument('--brlc_region_size', type=int, default=8)
        p.add_argument('--brlc_region_stride', type=int, default=4)
        p.add_argument('--brlc_window_chunk', type=int, default=256)
        p.add_argument('--brlc_evidence_pool', type=str, default='signed_peak',
                       choices=('signed_peak', 'area'))
        p.add_argument('--brlc_bg_kernels', type=int, nargs='+', default=[9, 17])
        p.add_argument('--brlc_bg_hole', type=int, default=5)
        p.add_argument('--brlc_bg_ring_channels', type=int, default=4)
        p.add_argument('--brlc_bg_hidden', type=int, default=16)
        p.add_argument('--brlc_sigma_min', type=float, default=0.01)
        p.add_argument('--brlc_sigma_max', type=float, default=0.5)
        p.add_argument('--brlc_temperature_min', type=float, default=0.1)
        p.add_argument('--brlc_temperature_max', type=float, default=10.0)
        p.add_argument('--brlc_eta_max', type=float, default=4.0)
        p.add_argument('--brlc_prior_logit_limit', type=float, default=6.0)
        p.add_argument('--brlc_loo_min_mass', type=float, default=1e-4)
        p.add_argument('--brlc_checkpoint_windows', type=str2bool, default=True)
        p.add_argument('--brlc_checkpoint_background', type=str2bool, default=True)
        p.add_argument('--brlc_validate_values', type=str2bool, default=True)
        return p


def train_step_full_brlc(task, batch, cfg):
    """Full step: BCE+Dice on logits plus the Gaussian-NLL background term."""
    ct = batch['ct'].to(task.device, non_blocking=True)
    pet = batch['pet'].to(task.device, non_blocking=True)
    mask = batch['mask'].to(task.device, non_blocking=True).float()
    background_train_mask = make_background_train_mask(
        foreground=mask, valid_mask=None,
        exclusion_radius=int(cfg.brlc_bg_exclusion_radius))
    outputs = task.model(ct, pet=pet, forward_mode='full',
                         background_train_mask=background_train_mask)
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    if not torch.isfinite(logits).all():
        raise RuntimeError('[NaN/Inf] logits at train_step_full_brlc')
    seg_loss, loss_stats = task.criterion(logits, mask)
    bg_loss = outputs.get('background_loss', None)
    if bg_loss is None:
        raise RuntimeError('background_loss missing in training mode')
    if not torch.isfinite(bg_loss).all():
        raise RuntimeError('[NaN/Inf] background_loss at train_step_full_brlc')
    total_loss = seg_loss + float(cfg.brlc_bg_loss_weight) * bg_loss
    if not torch.isfinite(total_loss).all():
        raise RuntimeError('[NaN/Inf] total_loss at train_step_full_brlc')
    stats = {
        'loss_total': total_loss.detach(),
        'loss_seg': seg_loss.detach(),
        'loss_bg': bg_loss.detach(),
        'num_samples': int(ct.shape[0]),
    }
    return total_loss, outputs, stats


def _assert_brlc_protocol(cfg, model):
    assert str(cfg.train_batch_mode) == 'full'
    assert str(cfg.model_arch) == MODEL_ARCH
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'
    if not isinstance(model.fusion, BackgroundReferencedRegionFusion):
        raise RuntimeError('full_background_referenced requires '
                           'BackgroundReferencedRegionFusion, '
                           f'got {type(model.fusion).__name__}')


def _assert_checkpoint_fusion(cfg, ckpt):
    saved = ckpt.get('config', {})
    if not isinstance(saved, dict):
        raise TypeError('Checkpoint config must be a dict')
    if str(saved.get('model_arch', MODEL_ARCH)) != MODEL_ARCH:
        raise ValueError('Checkpoint model_arch mismatch: refusing cross-model load')
    saved_fusion = saved.get('fusion', {})
    live_fusion = getattr(cfg, 'fusion', None)
    if isinstance(saved_fusion, dict) and isinstance(live_fusion, dict) \
            and saved_fusion != live_fusion:
        raise ValueError('Checkpoint fusion config mismatch: refusing cross-config load')


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def _visualize_once(model, loader, device, out_path):
    """Single fixed val sample, eval/no_grad, current raw/EMA weights."""
    was_training = model.training
    model.eval()
    try:
        first = next(iter(loader))
        ct = first['ct'][:1].to(device)
        pet = first['pet'][:1].to(device)
        with torch.no_grad():
            outputs = model(ct, pet=pet, forward_mode='full', return_diagnostics=True)
        diag = outputs.get('fusion_diagnostics', {})
        if not diag or 'scales' not in diag:
            raise RuntimeError('diagnostics missing for visualization')
        save_fusion_diagnostics(
            FusionResult(diag.get('fused', []), None, diag),
            out_path, full_row=0)
    finally:
        model.train(was_training)


def main():
    print('[INFO] starting Full-only background-referenced region fusion', flush=True)
    cfg = FullBackgroundReferencedConfig.parse_arguments()
    cfg.experiment_type = 'full_background_referenced'
    cfg.train_batch_mode = 'full'
    cfg.pet_missing_rate = 0.0
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    print('[INFO] background supervision region = all non-foreground pixels '
          '(no independent tissue ROI in this dataset; black canvas included)',
          flush=True)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_full_background_referenced_model(cfg), cfg)
    _assert_brlc_protocol(cfg, task.model)
    fusion_cfg = task.model.fusion_config()
    cfg.fusion = fusion_cfg
    with open(os.path.join(cfg.checkpoint_dir, 'fusion_config.json'), 'w') as f:
        json.dump(fusion_cfg, f, indent=2)
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
        'train_full_loss', 'train_seg_loss', 'train_bg_loss',
        'train_samples', 'train_batches',
        'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc',
        'val_full_acc_pixel', 'val_full_hd95',
        'best_full', 'best_full_epoch',
        'grad_enc_ct', 'grad_enc_pet', 'grad_ct_align', 'grad_pet_align',
        'grad_fusion', 'grad_decoder',
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
        train_total_sum = 0.0
        train_seg_sum = 0.0
        train_bg_sum = 0.0
        train_sample_count = 0
        train_n = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        grads = {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'pet_align': [],
                 'fusion': [], 'decoder': []}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, train_stats = train_step_full_brlc(task, batch, cfg)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grads['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads['enc_pet'].append(module_grad_norm(task.model.enc_pet))
            grads['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads['pet_align'].append(module_grad_norm(task.model.pet_align))
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
                print(f'[BATCH {batch_idx + 1}] route=full_brlc '
                      f'loss={float(loss.detach()):.6f}', flush=True)
            n = int(train_stats['num_samples'])
            train_total_sum += float(train_stats['loss_total']) * n
            train_seg_sum += float(train_stats['loss_seg']) * n
            train_bg_sum += float(train_stats['loss_bg']) * n
            train_sample_count += n
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
        if int(cfg.brlc_vis_every) > 0 and epoch % int(cfg.brlc_vis_every) == 0:
            _visualize_once(task.eval_model(), val_loader, task.device,
                            os.path.join(cfg.checkpoint_dir, 'brlc_vis',
                                         f'epoch{epoch:03d}_{task.eval_weights_tag()}.png'))

        train_total = train_total_sum / max(1, train_sample_count)
        train_seg = train_seg_sum / max(1, train_sample_count)
        train_bg = train_bg_sum / max(1, train_sample_count)
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'), epoch, train_total,
            {'total_loss': val_full['total_loss'], 'dice': val_full['dice'],
             'iou': val_full['iou'], 'acc': val_full['acc'],
             'acc_pixel': val_full.get('acc_pixel', 0.0), 'hd95': val_full['hd95']},
            lr=task.optimizer.param_groups[0]['lr'], grad_norm=avg_grad_norm,
            extra_metrics={
                'train_full_loss': train_total,
                'train_seg_loss': train_seg,
                'train_bg_loss': train_bg,
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
                'grad_pet_align': float(np.mean(grads['pet_align'])) if grads['pet_align'] else 0.0,
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
    _assert_checkpoint_fusion(cfg, ckpt)
    recorded = ckpt.get('eval_weights', 'raw')
    if recorded not in ('raw', 'ema'):
        raise ValueError(f'Unknown recorded evaluation weights: {recorded}')
    if recorded == 'ema' and ckpt.get('model_ema') is None:
        raise ValueError('Best checkpoint recorded EMA but EMA weights are missing')
    use_ema = recorded == 'ema'
    weights_tag = 'ema' if use_ema else 'raw'
    cfg.pretrained = False  # restore complete trained weights, not initialization again
    eval_model = build_full_background_referenced_model(cfg)['model'].to(task.device)
    _assert_brlc_protocol(cfg, eval_model)
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'],
                               strict=True)
    eval_model.eval()
    print(f'[FINAL] evaluating best_full checkpoint at epoch {ckpt["best_epoch"]} '
          f'with weights={weights_tag} (recorded={recorded}) '
          f'fusion={type(eval_model.fusion).__name__}', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full',
                              model=eval_model)
    result = {
        'experiment_type': 'full_background_referenced',
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
