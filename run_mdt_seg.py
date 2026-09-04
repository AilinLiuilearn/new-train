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


def _seed(cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned, get_pclt20k_prototype_loader
    train_loader, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
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
    prototype_loader = get_pclt20k_prototype_loader(
        cfg.root,
        image_size=cfg.image_size_2d,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        random_state=cfg.random_state,
        pin_memory=cfg.pin_memory,
        norm_mode=cfg.norm_mode,
        train_split_file=cfg.train_split_file,
    )
    return train_loader, val_loader, test_loader, prototype_loader


def _assert_baseline(cfg):
    assert cfg.accumulation_steps == 1
    assert float(cfg.train_pet_drop_prob) == 0.0
    assert float(cfg.missing_loss_weight) == 1.0
    assert float(cfg.joint_full_weight) == 0.5
    assert float(cfg.joint_missing_weight) == 0.5
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert float(cfg.boundary_loss_weight) == 0.0


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
    return float(total.sqrt().item()) if total is not None else 0.0


@torch.no_grad()
def rebuild_cppi_bank_from_snapshot(task, prototype_loader, epoch, cfg):
    """
    Epoch-end snapshot-consistent CPPI bank rebuild.

    Uses a frozen encoder snapshot (eval + no_grad + FP32, no augmentation).
    Does not clear the committed retrieval bank; only the candidate cache.
    """
    model = task.model
    was_training = model.training
    model.prototype_memory.reset_epoch_cache()
    model.eval()
    snapshot_start = time.time()
    for batch in prototype_loader:
        ct = batch['ct'].to(task.device, non_blocking=True).float()
        pet = batch['pet'].to(task.device, non_blocking=True).float()
        mask = batch['mask'].to(task.device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=False):
            model.collect_cppi_snapshot(ct=ct, pet=pet, mask=mask)
    snapshot_time = time.time() - snapshot_start
    cppi_report = model.finalize_cppi_epoch(
        epoch=epoch,
        save_json=True,
        save_visualizations=(
            epoch == 1
            or epoch % 5 == 0
            or epoch == cfg.epochs
        ),
        print_info=True,
        ema_momentum=float(getattr(cfg, 'cppi_bank_ema', 0.9)),
    )
    cppi_report['cppi_snapshot_time'] = float(snapshot_time)
    model.train(was_training)
    return cppi_report


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_joint': os.path.join(checkpoint_dir, 'ckpt.best_joint.pth.tar'),
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'best_missing': os.path.join(checkpoint_dir, 'ckpt.best_missing.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def _load_state_dict_with_report(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    print(f'[RESUME] checkpoint={checkpoint_path}', flush=True)
    print(f'[RESUME] missing_keys={missing_keys}', flush=True)
    print(f'[RESUME] unexpected_keys={unexpected_keys}', flush=True)
    return checkpoint


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

    train_loader, val_loader, _, prototype_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} proto_batches={len(prototype_loader)}', flush=True)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    if getattr(cfg, 'resume_checkpoint', None):
        _load_state_dict_with_report(task.model, cfg.resume_checkpoint)
    total_params, trainable_params = _count_parameters(task.model)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    extra_headers = [
        'train_full_loss', 'train_missing_loss', 'train_overall_loss',
        'full_train_batches', 'missing_train_batches',
        'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc', 'val_full_acc_pixel', 'val_full_hd95',
        'val_missing_loss', 'val_missing_dice', 'val_missing_iou', 'val_missing_acc', 'val_missing_acc_pixel', 'val_missing_hd95',
        'joint_dice', 'best_joint', 'best_joint_epoch',
        'grad_full_enc_ct', 'grad_missing_enc_ct', 'grad_full_ct_align', 'grad_missing_ct_align', 'grad_full_decoder', 'grad_missing_decoder',
        'epoch_time',
        'cppi_bank_version', 'cppi_ready_slots', 'cppi_bg_candidates', 'cppi_fg_candidates',
        'cppi_bg_match_mean_cos', 'cppi_fg_match_mean_cos',
        'cppi_bg_match_min_cos', 'cppi_fg_match_min_cos',
        'cppi_bank_drift_before_ema', 'cppi_bank_drift_after_ema',
        'cppi_snapshot_candidates_bg', 'cppi_snapshot_candidates_fg',
        'cppi_snapshot_time',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'), extra_headers=extra_headers)

    best_joint = -1.0
    best_full = -1.0
    best_missing = -1.0
    best_joint_epoch = 0
    global_batch_step = 0
    amp_enabled = bool(cfg.mixed_precision)
    patience = int(getattr(cfg, 'early_stop_patience', 10))
    no_improve = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        full_n = missing_n = 0
        full_loss = missing_loss = 0.0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        grads = {
            'full': {'enc_ct': [], 'ct_align': [], 'decoder': []},
            'missing': {'enc_ct': [], 'ct_align': [], 'decoder': []},
        }
        epoch_start = time.time()
        fixed_diag_batch = None
        diag_stats = {}

        for batch_idx, batch in enumerate(train_loader):
            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, _ = task.train_step(batch, forward_mode=route)
            if not torch.isfinite(loss):
                raise RuntimeError('loss became non-finite')

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            grads[route]['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads[route]['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads[route]['decoder'].append(module_grad_norm(task.model.decoder))
            total_grad_norm = torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip)) if float(cfg.grad_clip) > 0 else 0.0
            grad_norm_accum += float(total_grad_norm)
            grad_norm_steps += 1

            if task.scaler.is_enabled():
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                task.optimizer.step()

            task.scheduler.step()

            if (batch_idx + 1) % 100 == 0:
                print(f'[BATCH {batch_idx + 1}] route={route} loss={float(loss.detach()):.6f}', flush=True)

            if route == 'full':
                full_n += 1
                full_loss += float(loss.detach())
            else:
                missing_n += 1
                missing_loss += float(loss.detach())

            global_batch_step += 1
            task.global_batch_step = global_batch_step
            if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is None:
                fixed_diag_batch = {
                    'ct': batch['ct'][:1].detach().cpu(),
                    'pet': batch['pet'][:1].detach().cpu(),
                    'mask': batch['mask'][:1].detach().cpu(),
                }

        cppi_report = rebuild_cppi_bank_from_snapshot(
            task=task,
            prototype_loader=prototype_loader,
            epoch=epoch,
            cfg=cfg,
        )
        def _cppi_num(key, default=0.0):
            val = cppi_report.get(key, default)
            return float(val) if val is not None else float(default)

        print(
            f"[CPPI EPOCH {epoch}]\n"
            f"bank_version={cppi_report.get('bank_version_after', cppi_report.get('bank_version_before', 0))}\n"
            f"ready_slots={cppi_report.get('ready_count', cppi_report.get('ready_slots', 0))}\n"
            f"bg_candidates={cppi_report.get('classes', {}).get('background', {}).get('num_candidates', 0)}\n"
            f"fg_candidates={cppi_report.get('classes', {}).get('foreground', {}).get('num_candidates', 0)}\n"
            f"bg_match_mean_cos={cppi_report.get('cppi_bg_match_mean_cos')}\n"
            f"fg_match_mean_cos={cppi_report.get('cppi_fg_match_mean_cos')}\n"
            f"bg_match_min_cos={cppi_report.get('cppi_bg_match_min_cos')}\n"
            f"fg_match_min_cos={cppi_report.get('cppi_fg_match_min_cos')}\n"
            f"drift_before_ema={cppi_report.get('cppi_bank_drift_before_ema')}\n"
            f"drift_after_ema={cppi_report.get('cppi_bank_drift_after_ema')}\n"
            f"snapshot_time={cppi_report.get('cppi_snapshot_time')}",
            flush=True,
        )
        if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is not None and epoch % int(cfg.gradient_diagnostics_interval) == 0:
            diag_stats = task.gradient_diagnostics(fixed_diag_batch, max_samples=min(1, int(cfg.gradient_diagnostics_num_samples))) or {}

        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        val_missing = task.evaluate(val_loader, eval_mode='fixed_missing', tag='val_missing')
        joint_dice = float(cfg.joint_full_weight) * val_full['dice'] + float(cfg.joint_missing_weight) * val_missing['dice']

        joint_improved = joint_dice > best_joint
        full_improved = val_full['dice'] > best_full
        missing_improved = val_missing['dice'] > best_missing
        if joint_improved:
            best_joint = joint_dice
            best_joint_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if full_improved:
            best_full = val_full['dice']
        if missing_improved:
            best_missing = val_missing['dice']

        if joint_improved:
            task.save_checkpoint(paths['best_joint'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        if full_improved:
            task.save_checkpoint(paths['best_full'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        if missing_improved:
            task.save_checkpoint(paths['best_missing'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        task.save_checkpoint(paths['last'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)

        train_loss = (full_loss + missing_loss) / max(1, full_n + missing_n)
        val_loss = 0.5 * val_full['total_loss'] + 0.5 * val_missing['total_loss']
        val_dice = joint_dice
        val_iou = 0.5 * val_full['iou'] + 0.5 * val_missing['iou']
        val_acc = 0.5 * val_full['acc'] + 0.5 * val_missing['acc']
        val_acc_pixel = 0.5 * val_full.get('acc_pixel', 0.0) + 0.5 * val_missing.get('acc_pixel', 0.0)
        val_hd95 = 0.5 * val_full['hd95'] + 0.5 * val_missing['hd95']
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        extra_metrics = {
            'train_full_loss': full_loss / max(1, full_n),
            'train_missing_loss': missing_loss / max(1, missing_n),
            'train_overall_loss': train_loss,
            'full_train_batches': full_n,
            'missing_train_batches': missing_n,
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
            'grad_full_enc_ct': float(np.mean(grads['full']['enc_ct'])) if grads['full']['enc_ct'] else 0.0,
            'grad_missing_enc_ct': float(np.mean(grads['missing']['enc_ct'])) if grads['missing']['enc_ct'] else 0.0,
            'grad_full_ct_align': float(np.mean(grads['full']['ct_align'])) if grads['full']['ct_align'] else 0.0,
            'grad_missing_ct_align': float(np.mean(grads['missing']['ct_align'])) if grads['missing']['ct_align'] else 0.0,
            'grad_full_decoder': float(np.mean(grads['full']['decoder'])) if grads['full']['decoder'] else 0.0,
            'grad_missing_decoder': float(np.mean(grads['missing']['decoder'])) if grads['missing']['decoder'] else 0.0,
            'epoch_time': time.time() - epoch_start,
            'cppi_bank_version': int(cppi_report.get('bank_version_after', cppi_report.get('bank_version_before', 0))),
            'cppi_ready_slots': int(cppi_report.get('ready_count', cppi_report.get('ready_slots', 0))),
            'cppi_bg_candidates': int(cppi_report.get('classes', {}).get('background', {}).get('num_candidates', 0)),
            'cppi_fg_candidates': int(cppi_report.get('classes', {}).get('foreground', {}).get('num_candidates', 0)),
            'cppi_bg_match_mean_cos': _cppi_num('cppi_bg_match_mean_cos'),
            'cppi_fg_match_mean_cos': _cppi_num('cppi_fg_match_mean_cos'),
            'cppi_bg_match_min_cos': _cppi_num('cppi_bg_match_min_cos'),
            'cppi_fg_match_min_cos': _cppi_num('cppi_fg_match_min_cos'),
            'cppi_bank_drift_before_ema': _cppi_num('cppi_bank_drift_before_ema'),
            'cppi_bank_drift_after_ema': _cppi_num('cppi_bank_drift_after_ema'),
            'cppi_snapshot_candidates_bg': int(cppi_report.get('cppi_snapshot_candidates_bg', cppi_report.get('classes', {}).get('background', {}).get('num_candidates', 0))),
            'cppi_snapshot_candidates_fg': int(cppi_report.get('cppi_snapshot_candidates_fg', cppi_report.get('classes', {}).get('foreground', {}).get('num_candidates', 0))),
            'cppi_snapshot_time': _cppi_num('cppi_snapshot_time'),
            **{f'diag_{k}': v for k, v in diag_stats.items()},
        }
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=avg_grad_norm,
            extra_metrics=extra_metrics,
        )

        print(f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


if __name__ == '__main__':
    main()
