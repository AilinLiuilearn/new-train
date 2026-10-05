# -*- coding: utf-8 -*-
"""Full/Missing mixed baseline entry point.

Fixed to the mixed protocol:
- even batch_size >= 2, DataLoader drop_last=True;
- each batch is exact-half Full (1) / Missing (0), reshuffled per step
  reproducibly via build_balanced_pet_available;
- loss = 0.5 * full_subset + 0.5 * missing_subset, one backward/step;
- val selection by Joint = 0.5 * full_dice + 0.5 * missing_dice;
- final Full/Missing/Joint all come from the same best_joint checkpoint.

Shared config/bbox_loss/metrics/logging/EMA with run_ct_only_seg; this is a
separate entry point, not a duplicated copy of the common implementation.
"""
import json
import os
import time

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_dual_model
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
        checkpoint_dir=cfg.checkpoint_dir, ct_only=False,
    )


def _assert_mixed(cfg, train_loader):
    batch_size = int(cfg.batch_size)
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(f'mixed mode requires an even batch_size >= 2, got {batch_size}')
    if not bool(getattr(train_loader, 'drop_last', False)):
        raise ValueError('mixed mode requires DataLoader drop_last=True')
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_joint': os.path.join(checkpoint_dir, 'ckpt.best_joint.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main():
    print('[INFO] starting Full/Missing mixed baseline', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    _assert_mixed(cfg, train_loader)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_dual_model(cfg), cfg)
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
        'train_full_loss', 'train_missing_loss', 'train_mixed_loss',
        'train_full_samples', 'train_missing_samples', 'mixed_train_batches',
        'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc',
        'val_full_acc_pixel', 'val_full_hd95',
        'val_missing_loss', 'val_missing_dice', 'val_missing_iou', 'val_missing_acc',
        'val_missing_acc_pixel', 'val_missing_hd95',
        'joint_dice', 'best_joint', 'best_joint_epoch',
        'grad_mixed_enc_ct', 'grad_mixed_enc_pet', 'grad_mixed_ct_align', 'grad_mixed_decoder',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
                   extra_headers=extra_headers)

    best_joint = -1.0
    best_joint_epoch = 0
    no_improve = 0
    patience = int(cfg.early_stop_patience)
    amp_enabled = bool(cfg.mixed_precision)
    global_batch_step = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        full_loss_sum = missing_loss_sum = mixed_loss_sum = 0.0
        full_sample_count = missing_sample_count = 0
        mixed_n = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        grads = {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'decoder': []}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            actual_batch_size = batch['ct'].shape[0]
            if actual_batch_size != int(cfg.batch_size):
                raise ValueError(
                    f'mixed mode requires fixed batch size {int(cfg.batch_size)} '
                    f'(drop_last=True), got {actual_batch_size}')
            pet_available = build_balanced_pet_available(
                int(cfg.batch_size), global_batch_step, cfg.random_state, task.device,
            )
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, train_stats = task.train_step_mixed(batch, pet_available=pet_available)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grads['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads['enc_pet'].append(module_grad_norm(task.model.enc_pet))
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
                print(f'[BATCH {batch_idx + 1}] mode=mixed '
                      f'num_full={int(train_stats["num_full"])} '
                      f'num_missing={int(train_stats["num_missing"])} '
                      f'loss={float(loss.detach()):.6f}', flush=True)
            full_loss_sum += float(train_stats['loss_full']) * int(train_stats['num_full'])
            missing_loss_sum += float(train_stats['loss_missing']) * int(train_stats['num_missing'])
            full_sample_count += int(train_stats['num_full'])
            missing_sample_count += int(train_stats['num_missing'])
            mixed_loss_sum += float(loss.detach())
            mixed_n += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step

        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full',
                                 model=task.eval_model())
        val_missing = task.evaluate(val_loader, eval_mode='missing', tag='val_missing',
                                    model=task.eval_model())
        joint_dice = 0.5 * val_full['dice'] + 0.5 * val_missing['dice']

        if joint_dice > best_joint:
            best_joint = joint_dice
            best_joint_epoch = epoch
            no_improve = 0
            improved = True
        else:
            no_improve += 1
            improved = False
        if improved:
            task.save_checkpoint(paths['best_joint'], epoch, best_joint,
                                 best_joint_epoch, {'full': {k: val_full[k]
                                                              for k in ('dice', 'iou', 'hd95', 'acc')},
                                                    'missing': {k: val_missing[k]
                                                                for k in ('dice', 'iou', 'hd95', 'acc')},
                                                    'joint_dice': joint_dice})
        task.save_checkpoint(paths['last'], epoch, best_joint, best_joint_epoch,
                             {'full': val_full['dice'], 'missing': val_missing['dice'],
                              'joint_dice': joint_dice})

        epoch_full_loss = full_loss_sum / max(1, full_sample_count)
        epoch_missing_loss = missing_loss_sum / max(1, missing_sample_count)
        train_mixed_loss = mixed_loss_sum / max(1, mixed_n)
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        val_joint = {
            'total_loss': 0.5 * val_full['total_loss'] + 0.5 * val_missing['total_loss'],
            'dice': joint_dice,
            'iou': 0.5 * val_full['iou'] + 0.5 * val_missing['iou'],
            'acc': 0.5 * val_full['acc'] + 0.5 * val_missing['acc'],
            'acc_pixel': 0.5 * val_full.get('acc_pixel', 0.0) + 0.5 * val_missing.get('acc_pixel', 0.0),
            'hd95': 0.5 * val_full['hd95'] + 0.5 * val_missing['hd95'],
        }
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'), epoch, train_mixed_loss,
            val_joint,
            lr=task.optimizer.param_groups[0]['lr'], grad_norm=avg_grad_norm,
            extra_metrics={
                'train_full_loss': epoch_full_loss,
                'train_missing_loss': epoch_missing_loss,
                'train_mixed_loss': train_mixed_loss,
                'train_full_samples': float(full_sample_count),
                'train_missing_samples': float(missing_sample_count),
                'mixed_train_batches': float(mixed_n),
                'val_full_loss': val_full['total_loss'],
                'val_full_dice': val_full['dice'],
                'val_full_iou': val_full['iou'],
                'val_full_acc': val_full['acc'],
                'val_full_acc_pixel': val_full.get('acc_pixel', 0.0),
                'val_full_hd95': val_full['hd95'],
                'val_missing_loss': val_missing['total_loss'],
                'val_missing_dice': val_missing['dice'],
                'val_missing_iou': val_missing['iou'],
                'val_missing_acc': val_missing['acc'],
                'val_missing_acc_pixel': val_missing.get('acc_pixel', 0.0),
                'val_missing_hd95': val_missing['hd95'],
                'joint_dice': joint_dice,
                'best_joint': best_joint,
                'best_joint_epoch': best_joint_epoch,
                'grad_mixed_enc_ct': float(np.mean(grads['enc_ct'])) if grads['enc_ct'] else 0.0,
                'grad_mixed_enc_pet': float(np.mean(grads['enc_pet'])) if grads['enc_pet'] else 0.0,
                'grad_mixed_ct_align': float(np.mean(grads['ct_align'])) if grads['ct_align'] else 0.0,
                'grad_mixed_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else 0.0,
                'ema_enabled': 1.0 if task.ema is not None else 0.0,
                'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
                'skipped_updates': float(skipped_update_count),
                'epoch_time': time.time() - epoch_start,
            },
        )
        print(f'[EPOCH {epoch}] full_dice={val_full["dice"]:.4f} '
              f'missing_dice={val_missing["dice"]:.4f} joint_dice={joint_dice:.4f} '
              f'best_joint={best_joint:.4f} '
              f'lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    if not os.path.isfile(paths['best_joint']):
        raise RuntimeError('best_joint checkpoint was not created')
    ckpt = MDTSegTeacher.load_state_dicts(paths['best_joint'])
    use_ema = task.ema is not None and ckpt.get('model_ema') is not None
    weights_tag = 'ema' if use_ema else 'raw'
    eval_model = build_dual_model(cfg)['model'].to(task.device)
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'], strict=True)
    eval_model.eval()
    print(f'[FINAL] evaluating best_joint checkpoint at epoch {ckpt["best_epoch"]} '
          f'with weights={weights_tag}', flush=True)
    eval_missing_rates(task, eval_model, test_loader, cfg.random_state,
                       cfg.checkpoint_dir, 'final_test',
                       rates=list(cfg.final_test_missing_rates), weights_tag=weights_tag)
    print('done', flush=True)


if __name__ == '__main__':
    main()