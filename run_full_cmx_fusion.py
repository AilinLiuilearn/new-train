# -*- coding: utf-8 -*-
"""Full runner: official CMX or SKFF modules, baseline BCE+Dice only.

Baseline data/encoders/aligners/decoder/optimizer/scheduler/EMA/metrics unchanged.
No lesion prediction, auxiliary supervision, prototype losses or extra residual.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from configs.full_cmx import FullCMXConfig
from models.full_cmx_fusion import CONFIG_FIELDS, build_full_cmx_model, fusion_class
from models.components.cmx_fusion import CMXFusion
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (count_parameters, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log


def train_step_full(task, batch):
    """Full-modality step: all rows train the selected CT+PET path with BCE+Dice."""
    ct = batch['ct'].to(task.device, non_blocking=True)
    pet = batch['pet'].to(task.device, non_blocking=True)
    mask = batch['mask'].to(task.device, non_blocking=True).float()
    outputs = task.model(ct, pet=pet, forward_mode='full')
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    if not torch.isfinite(logits).all():
        raise RuntimeError('[NaN/Inf] logits at train_step_full')
    loss, loss_stats = task.criterion(logits, mask)
    if not torch.isfinite(loss).all():
        raise RuntimeError('[NaN/Inf] loss at train_step_full')
    stats = {
        'loss_total': loss.detach(),
        'loss_seg': loss_stats.get('loss_dice', loss.detach()),
        'num_samples': int(ct.shape[0]),
    }
    return loss, outputs, stats


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


def _assert_full_add_protocol(cfg, model, expected_fusion=CMXFusion):
    assert str(cfg.train_batch_mode) == 'full'
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'
    if not isinstance(model.fusion, expected_fusion):
        raise RuntimeError(
            f'Full runner requires {expected_fusion.__name__}, got {type(model.fusion).__name__}')


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main(config_cls=FullCMXConfig, model_builder=build_full_cmx_model,
         expected_fusion=None, experiment_type='full_cmx_fusion',
         result_config_fields=CONFIG_FIELDS, extra_grad_modules=('pet_align', 'fusion')):
    cfg = config_cls.parse_arguments()
    if expected_fusion is None:
        expected_fusion = fusion_class(getattr(cfg, 'fusion_backend', 'cmx'))
    print(f'[INFO] starting Full-only CT+PET {expected_fusion.__name__}', flush=True)
    cfg.experiment_type = experiment_type
    cfg.train_batch_mode = 'full'
    cfg.fusion_type = expected_fusion.__name__
    cfg.pet_missing_rate = 0.0
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(model_builder(cfg), cfg)
    _assert_full_add_protocol(cfg, task.model, expected_fusion=expected_fusion)
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
        'grad_enc_ct', 'grad_enc_pet', 'grad_ct_align', 'grad_decoder',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    extra_headers.extend(f'grad_{name}' for name in extra_grad_modules)
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
        grads = {name: [] for name in
                 ('enc_ct', 'enc_pet', 'ct_align', 'decoder') + tuple(extra_grad_modules)}
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
            grads['decoder'].append(module_grad_norm(task.model.decoder))
            for name in extra_grad_modules:
                grads[name].append(module_grad_norm(getattr(task.model, name)))
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
                print(f'[BATCH {batch_idx + 1}] route=full '
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
                'grad_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else 0.0,
                'ema_enabled': 1.0 if task.ema is not None else 0.0,
                'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
                'skipped_updates': float(skipped_update_count),
                'epoch_time': time.time() - epoch_start,
                **{f'grad_{name}': float(np.mean(grads[name])) if grads[name] else 0.0
                   for name in extra_grad_modules},
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
    recorded = ckpt.get('eval_weights', 'raw')
    use_ema = (task.ema is not None and recorded == 'ema'
               and ckpt.get('model_ema') is not None)
    weights_tag = 'ema' if use_ema else 'raw'
    eval_model = model_builder(cfg)['model'].to(task.device)
    if not isinstance(eval_model.fusion, expected_fusion):
        raise RuntimeError(f'final test must rebuild {expected_fusion.__name__}, '
                           f'got {type(eval_model.fusion).__name__}')
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'],
                               strict=True)
    eval_model.eval()
    print(f'[FINAL] evaluating best_full checkpoint at epoch {ckpt["best_epoch"]} '
          f'with weights={weights_tag} (recorded={recorded})', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full',
                              model=eval_model)
    result = {
        'experiment_type': experiment_type,
        'dice': float(test_full['dice']),
        'iou': float(test_full['iou']),
        'hd95': float(test_full['hd95']),
        'hd95_mode': task.metrics.hd95_mode,
        'hd95_unit': 'pixel',
        'loss': float(test_full['total_loss']),
        'weights': weights_tag,
        'best_epoch': ckpt['best_epoch'],
        'checkpoint': os.path.abspath(paths['best_full']),
    }
    if result_config_fields:
        result.update(fusion_type=expected_fusion.__name__, model_arch=cfg.model_arch)
        result.update({name: getattr(cfg, name) for name in result_config_fields})
    with open(os.path.join(cfg.checkpoint_dir, 'final_test_full.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('[FINAL TEST FULL]', json.dumps(result, indent=2), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
