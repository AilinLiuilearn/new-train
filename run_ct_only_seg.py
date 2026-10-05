# -*- coding: utf-8 -*-
"""CT-only baseline: train on all CT slices, select by val CT Dice.

Train/val/test use the same patient-disjoint splits, CT weights, aug, norm,
batch size, optimizer, LR schedule and budget as the Full/Missing baseline.
Final test runs on the best_ct checkpoint with the recorded eval weights.
"""
import json
import os
import time

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_ct_only_model
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (build_balanced_pet_available, count_parameters,
                              eval_missing_rates, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir, ct_only=True,
    )


def _assert_ct_only_protocol(cfg):
    assert str(cfg.ct_backbone).lower().replace('-', '_') == 'convnextv2_nano'
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_ct': os.path.join(checkpoint_dir, 'ckpt.best_ct.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main():
    print('[INFO] starting CT-only baseline (all CT slices, fixed protocol)', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_ct_only_protocol(cfg)
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_ct_only_model(cfg), cfg)
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
        'train_ct_loss', 'train_batches',
        'val_ct_loss', 'val_ct_dice', 'val_ct_iou', 'val_ct_acc',
        'val_ct_acc_pixel', 'val_ct_hd95',
        'best_ct', 'best_ct_epoch',
        'grad_enc_ct', 'grad_ct_align', 'grad_decoder',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
                   extra_headers=extra_headers)

    best_ct = -1.0
    best_ct_epoch = 0
    no_improve = 0
    patience = int(cfg.early_stop_patience)
    amp_enabled = bool(cfg.mixed_precision)
    global_batch_step = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        train_loss_sum = 0.0
        train_n = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        grads = {'enc_ct': [], 'ct_align': [], 'decoder': []}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, _ = task.train_step_ct(batch)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grads['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads['ct_align'].append(module_grad_norm(task.model.ct_align))
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
                print(f'[BATCH {batch_idx + 1}] route=ct_only loss={float(loss.detach()):.6f}',
                      flush=True)
            train_loss_sum += float(loss.detach())
            train_n += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step

        val_ct = task.evaluate(val_loader, eval_mode='ct', tag='val_ct',
                               model=task.eval_model())
        improved = val_ct['dice'] > best_ct
        if improved:
            best_ct = val_ct['dice']
            best_ct_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if improved:
            task.save_checkpoint(paths['best_ct'], epoch, best_ct, best_ct_epoch, val_ct)
        task.save_checkpoint(paths['last'], epoch, best_ct, best_ct_epoch, val_ct)

        train_loss = train_loss_sum / max(1, train_n)
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'), epoch, train_loss,
            {'total_loss': val_ct['total_loss'], 'dice': val_ct['dice'],
             'iou': val_ct['iou'], 'acc': val_ct['acc'],
             'acc_pixel': val_ct.get('acc_pixel', 0.0), 'hd95': val_ct['hd95']},
            lr=task.optimizer.param_groups[0]['lr'], grad_norm=avg_grad_norm,
            extra_metrics={
                'train_ct_loss': train_loss,
                'train_batches': train_n,
                'val_ct_loss': val_ct['total_loss'],
                'val_ct_dice': val_ct['dice'],
                'val_ct_iou': val_ct['iou'],
                'val_ct_acc': val_ct['acc'],
                'val_ct_acc_pixel': val_ct.get('acc_pixel', 0.0),
                'val_ct_hd95': val_ct['hd95'],
                'best_ct': best_ct,
                'best_ct_epoch': best_ct_epoch,
                'grad_enc_ct': float(np.mean(grads['enc_ct'])) if grads['enc_ct'] else 0.0,
                'grad_ct_align': float(np.mean(grads['ct_align'])) if grads['ct_align'] else 0.0,
                'grad_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else 0.0,
                'ema_enabled': 1.0 if task.ema is not None else 0.0,
                'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
                'skipped_updates': float(skipped_update_count),
                'epoch_time': time.time() - epoch_start,
            },
        )
        print(f'[EPOCH {epoch}] val_ct_dice={val_ct["dice"]:.4f} best_ct={best_ct:.4f} '
              f'lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    if not os.path.isfile(paths['best_ct']):
        raise RuntimeError('best CT-only checkpoint was not created')
    ckpt = MDTSegTeacher.load_state_dicts(paths['best_ct'])
    use_ema = task.ema is not None and ckpt.get('model_ema') is not None
    weights_tag = 'ema' if use_ema else 'raw'
    eval_model = build_ct_only_model(cfg)['model'].to(task.device)
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'], strict=True)
    eval_model.eval()
    eval_missing_rates(task, eval_model, test_loader, cfg.random_state,
                       cfg.checkpoint_dir, 'final_test_ct',
                       rates=list(cfg.final_test_missing_rates), weights_tag=weights_tag)
    print('done', flush=True)


if __name__ == '__main__':
    main()