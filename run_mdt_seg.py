# -*- coding: utf-8 -*-
import json
import os
import random
import time

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        value = p.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    return float(total.sqrt().item()) if total is not None else 0.0


def _seed(cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root,
        cfg.image_size_2d,
        cfg.batch_size,
        cfg.num_workers,
        cfg.random_state,
        cfg.pin_memory,
        cfg.aug_mode,
        cfg.norm_mode,
        cfg.train_split_file,
        cfg.val_split_file,
        cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
    )


def _assert_baseline(cfg):
    assert cfg.accumulation_steps == 1


def _optimizer_step_succeeded(task):
    if not task.scaler.is_enabled():
        task.optimizer.step()
        return True
    before = task.scaler.get_scale()
    task.scaler.step(task.optimizer)
    task.scaler.update()
    after = task.scaler.get_scale()
    return bool(after >= before)


def _run_final_test(cfg, paths, task, test_loader):
    """Evaluate the best checkpoint on the held-out test split once."""
    best_path = paths['best'] if os.path.exists(paths['best']) else paths['last']
    ckpt = torch.load(best_path, map_location='cpu')
    use_ema = bool(task.ema_active) and ckpt.get('model_ema') is not None
    state_dict = ckpt['model_ema'] if use_ema else ckpt['model']
    model = build_mdt_seg_teacher(cfg)['model'].to(task.device)
    model.load_state_dict(state_dict, strict=True)
    out = task.evaluate(test_loader, tag='test', model=model)
    result = {
        'checkpoint': os.path.basename(best_path),
        'weights': 'ema' if use_ema else 'raw',
        'best_val_dice': ckpt.get('dice'),
        'best_epoch': ckpt.get('best_epoch'),
        'test_loss': out['total_loss'],
        'test_dice': out['dice'],
        'test_iou': out['iou'],
        'test_acc': out['acc'],
        'test_acc_pixel': out.get('acc_pixel', 0.0),
        'test_hd95': out['hd95'],
    }
    with open(os.path.join(cfg.checkpoint_dir, 'final_test_metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[FINAL TEST] weights={result['weights']} dice={result['test_dice']:.4f} "
          f"iou={result['test_iou']:.4f} hd95={result['test_hd95']:.4f}", flush=True)
    return result
    if not task.scaler.is_enabled():
        task.optimizer.step()
        return True
    before = task.scaler.get_scale()
    task.scaler.step(task.optimizer)
    task.scaler.update()
    after = task.scaler.get_scale()
    return bool(after >= before)


def _checkpoint_paths(checkpoint_dir):
    return {
        'best': os.path.join(checkpoint_dir, 'ckpt.best.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    total_params, trainable_params = _count_parameters(task.model)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)
    if task.ema is not None:
        print(
            f'[INFO] ema_enabled=True decay={task.ema.decay} '
            f'decay_warmup={task.ema.warmup} start_epoch={task.ema_start_epoch} '
            f'(EMA active from epoch {task.ema_start_epoch + 1}) eval_uses_ema_after_start=True',
            flush=True,
        )
    else:
        print('[INFO] ema_enabled=False eval_uses_ema=False', flush=True)
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    extra_headers = [
        'train_full_loss', 'full_train_batches',
        'val_loss', 'val_dice', 'val_iou', 'val_acc', 'val_acc_pixel', 'val_hd95',
        'best', 'best_epoch',
        'grad_full_enc_ct', 'grad_full_ct_align', 'grad_full_decoder',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'), extra_headers=extra_headers)

    best = -1.0
    best_epoch = 0
    global_batch_step = 0
    amp_enabled = bool(cfg.mixed_precision)
    patience = int(getattr(cfg, 'early_stop_patience', 10))
    no_improve = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        epoch_start = time.time()
        fixed_diag_batch = None
        diag_stats = {}
        full_loss_sum = 0.0
        full_n = 0
        grads = {'enc_ct': [], 'ct_align': [], 'decoder': []}

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, _ = task.train_step(batch)
            if not torch.isfinite(loss):
                raise RuntimeError('loss became non-finite')

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            grads['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads['decoder'].append(module_grad_norm(task.model.decoder))
            total_grad_norm = torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip)) if float(cfg.grad_clip) > 0 else 0.0
            grad_norm_accum += float(total_grad_norm)
            grad_norm_steps += 1

            step_succeeded = _optimizer_step_succeeded(task)
            if step_succeeded:
                task.scheduler.step()
                task.update_ema()
            else:
                skipped_update_count += 1

            if (batch_idx + 1) % 100 == 0:
                print(f'[BATCH {batch_idx + 1}] loss={float(loss.detach()):.6f}', flush=True)

            full_loss_sum += float(loss.detach())
            full_n += 1

            global_batch_step += 1
            task.global_batch_step = global_batch_step
            if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is None:
                fixed_diag_batch = {
                    'ct': batch['ct'][:1].detach().cpu(),
                    'pet': batch['pet'][:1].detach().cpu(),
                    'mask': batch['mask'][:1].detach().cpu(),
                }

        if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is not None and epoch % int(cfg.gradient_diagnostics_interval) == 0:
            diag_stats = task.gradient_diagnostics(fixed_diag_batch, max_samples=min(1, int(cfg.gradient_diagnostics_num_samples))) or {}

        val = task.evaluate(val_loader, tag='val', model=task.eval_model())
        dice = float(val['dice'])

        if dice > best:
            best = dice
            best_epoch = epoch
            no_improve = 0
            improved = True
        else:
            no_improve += 1
            improved = False

        if improved:
            task.save_checkpoint(paths['best'], epoch, best, best_epoch, val, dice)
        task.save_checkpoint(paths['last'], epoch, best, best_epoch, val, dice)

        train_loss = full_loss_sum / max(1, full_n)
        val_loss = val['total_loss']
        val_dice = val['dice']
        val_iou = val['iou']
        val_acc = val['acc']
        val_acc_pixel = val.get('acc_pixel', 0.0)
        val_hd95 = val['hd95']
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        extra = {
            'train_full_loss': train_loss,
            'full_train_batches': full_n,
            'grad_full_enc_ct': float(np.mean(grads['enc_ct'])) if grads['enc_ct'] else 0.0,
            'grad_full_ct_align': float(np.mean(grads['ct_align'])) if grads['ct_align'] else 0.0,
            'grad_full_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else 0.0,
            'ema_enabled': 1.0 if task.ema is not None else 0.0,
            'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
            'skipped_updates': float(skipped_update_count),
            'epoch_time': time.time() - epoch_start,
            **{f'diag_{k}': v for k, v in diag_stats.items()},
        }
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=avg_grad_norm,
            extra_metrics={
                **extra,
                'val_loss': val['total_loss'],
                'val_dice': val['dice'],
                'val_iou': val['iou'],
                'val_acc': val['acc'],
                'val_acc_pixel': val.get('acc_pixel', 0.0),
                'val_hd95': val['hd95'],
                'best': best,
                'best_epoch': best_epoch,
            },
        )

        print(f'[EPOCH {epoch}] full_dice={val["dice"]:.4f} best={best:.4f} lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    _run_final_test(cfg, paths, task, test_loader)
    print('done', flush=True)


if __name__ == '__main__':
    main()
