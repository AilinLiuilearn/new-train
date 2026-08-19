# -*- coding: utf-8 -*-
import json
import math
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


def _nonfinite_grad_report(model, max_names=12):
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
            if len(bad) >= max_names:
                break
    return bad


_STAGE1_MODULE_PREFIXES = (
    'enc_ct.',
    'enc_pet.',
    'ct_align.',
    'pet_calibration.',
    'decoder.',
    'prototype_memory.',
)


def _count_prefix_matches(keys, prefix):
    return sum(1 for k in keys if k.startswith(prefix))


def _load_stage1_warmstart(model, checkpoint_path):
    """
    Warm-start Stage-2 TRDF training from a Stage-1 CPPI+Calibration checkpoint.

    Loads: encoders, align, CPPI bank, PET calibration, decoder.
    Skips: old StateAwareWeightedAddFusion (and any fusion.* keys).
    TRDF remains randomly initialized.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f'Invalid Stage-1 checkpoint model state: {checkpoint_path}')

    filtered = {}
    skipped_fusion = []
    skipped_other = []
    for key, value in state_dict.items():
        if key.startswith('fusion.'):
            skipped_fusion.append(key)
            continue
        if any(key.startswith(prefix) for prefix in _STAGE1_MODULE_PREFIXES):
            filtered[key] = value
        else:
            skipped_other.append(key)

    result = model.load_state_dict(filtered, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    loaded_keys = sorted(filtered.keys())
    print(f'[WARMSTART] stage1_checkpoint={checkpoint_path}', flush=True)
    print(
        f'[WARMSTART] loaded_tensors={len(loaded_keys)} '
        f'skipped_fusion={len(skipped_fusion)} skipped_other={len(skipped_other)}',
        flush=True,
    )
    for prefix, label in [
        ('enc_ct.', 'CT encoder'),
        ('enc_pet.', 'PET encoder'),
        ('ct_align.', 'CT align'),
        ('prototype_memory.', 'CPPI prototype bank'),
        ('pet_calibration.', 'PET affine calibration'),
        ('decoder.', 'shared decoder'),
    ]:
        n = _count_prefix_matches(loaded_keys, prefix)
        print(f'[WARMSTART] loaded {label}: tensors={n}', flush=True)

    if skipped_fusion:
        print(
            f'[WARMSTART] ignored old fusion weights ({len(skipped_fusion)}): '
            f'{skipped_fusion[:8]}{"..." if len(skipped_fusion) > 8 else ""}',
            flush=True,
        )
    else:
        print('[WARMSTART] no fusion.* keys found in Stage-1 checkpoint', flush=True)

    trdf_missing = [k for k in missing_keys if k.startswith('fusion.')]
    other_missing = [k for k in missing_keys if not k.startswith('fusion.')]
    print(f'[WARMSTART] TRDF initialized from scratch (missing_fusion_keys={len(trdf_missing)})', flush=True)
    if other_missing:
        print(f'[WARMSTART] other_missing_keys={other_missing[:20]}', flush=True)
    if unexpected_keys:
        print(f'[WARMSTART] unexpected_keys={unexpected_keys[:20]}', flush=True)

    pm = model.prototype_memory
    ready_slots = int(pm.prototype_ready.sum().item())
    total_slots = int(pm.prototype_ready.numel())
    bank_version = int(pm.bank_version.item())
    bank_ready = bool(pm.bank_ready)
    print(
        f'[WARMSTART] Stage1 prototype bank restored: '
        f'ready={bank_ready} version={bank_version} '
        f'ready_slots={ready_slots}/{total_slots}',
        flush=True,
    )
    if not bank_ready:
        print(
            '[WARMSTART][WARN] prototype bank is NOT ready after Stage-1 load; '
            'Missing path will be weak until first finalize.',
            flush=True,
        )
    return checkpoint


def _load_state_dict_with_report(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    print(f'[RESUME] checkpoint={checkpoint_path}', flush=True)
    print(f'[RESUME] missing_keys={missing_keys}', flush=True)
    print(f'[RESUME] unexpected_keys={unexpected_keys}', flush=True)
    old_fusion_ignored = any(k.startswith('fusion.raw_alpha_') for k in unexpected_keys)
    trdf_from_scratch = any(k.startswith('fusion.scales.') for k in missing_keys)
    if old_fusion_ignored or trdf_from_scratch:
        print('old fusion weights ignored', flush=True)
        print('TRDF initialized from scratch', flush=True)
    return checkpoint


def _print_optimizer_groups(optimizer, warmup_epochs):
    print('[OPT] two learning-rate groups:', flush=True)
    for group in optimizer.param_groups:
        name = group.get('name', 'unnamed')
        n_params = sum(p.numel() for p in group['params'])
        print(f"[OPT] group={name} lr={group['lr']:.8f} n_params={n_params}", flush=True)
    print(f'[OPT] cosine_warmup_epochs={warmup_epochs}', flush=True)


def _optimizer_lrs(optimizer):
    lrs = {}
    for group in optimizer.param_groups:
        name = group.get('name', f"group{len(lrs)}")
        lrs[name] = float(group['lr'])
    return lrs


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_joint': os.path.join(checkpoint_dir, 'ckpt.best_joint.pth.tar'),
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'best_missing': os.path.join(checkpoint_dir, 'ckpt.best_missing.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, _ = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)

    stage1_ckpt = getattr(cfg, 'stage1_checkpoint', None)
    resume_ckpt = getattr(cfg, 'resume_checkpoint', None)
    if stage1_ckpt and resume_ckpt:
        raise ValueError(
            'Use either --stage1_checkpoint (TRDF warm-start) or --resume_checkpoint '
            '(continue a Stage-2 TRDF run), not both.'
        )

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    if stage1_ckpt:
        _load_stage1_warmstart(task.model, stage1_ckpt)
    elif resume_ckpt:
        print(
            '[RESUME][WARN] Resuming a Stage-2 checkpoint. '
            'Do not resume NaN-polluted TRDF runs.',
            flush=True,
        )
        _load_state_dict_with_report(task.model, resume_ckpt)

    total_params, trainable_params = _count_parameters(task.model)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)
    _print_optimizer_groups(task.optimizer, warmup_epochs=int(cfg.cosine_warmup))

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
            # AMP may produce rare Inf grads; GradScaler should skip the step and
            # lower the loss scale. Never optimizer.step() with non-finite grads.
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                task.trainable_parameters(),
                float(cfg.grad_clip),
                error_if_nonfinite=False,
            ) if float(cfg.grad_clip) > 0 else torch.tensor(0.0)
            total_grad_norm = float(total_grad_norm)

            if not math.isfinite(total_grad_norm):
                bad = _nonfinite_grad_report(task.model)
                print(
                    f'[WARN] non-finite grads at epoch={epoch} batch={batch_idx + 1} '
                    f'route={route} loss={float(loss.detach()):.6f}; '
                    f'skip optimizer.step; bad_params={bad}',
                    flush=True,
                )
                if task.scaler.is_enabled():
                    task.scaler.update()
                task.optimizer.zero_grad(set_to_none=True)
                global_batch_step += 1
                task.global_batch_step = global_batch_step
                continue

            grad_norm_accum += total_grad_norm
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

        cppi_report = task.model.finalize_cppi_epoch(
            epoch=epoch,
            save_json=True,
            save_visualizations=(
                epoch == 1
                or epoch % 5 == 0
                or epoch == cfg.epochs
            ),
            print_info=True,
        )
        print(
            f"[CPPI EPOCH {epoch}]\n"
            f"status={cppi_report.get('status', 'unknown')}\n"
            f"bank_version_before={cppi_report.get('bank_version_before', 0)}\n"
            f"bank_version_after={cppi_report.get('bank_version_after', cppi_report.get('bank_version_before', 0))}\n"
            f"ready_slots={cppi_report.get('ready_count', cppi_report.get('ready_slots', 0))}\n"
            f"bg_candidates={cppi_report.get('classes', {}).get('background', {}).get('num_candidates', 0)}\n"
            f"fg_candidates={cppi_report.get('classes', {}).get('foreground', {}).get('num_candidates', 0)}\n"
            f"bank_updated={cppi_report.get('status') == 'bank_updated'}",
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
            **{f'diag_{k}': v for k, v in diag_stats.items()},
        }
        lrs = _optimizer_lrs(task.optimizer)
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=lrs.get('trdf', task.optimizer.param_groups[-1]['lr']),
            grad_norm=avg_grad_norm,
            extra_metrics=extra_metrics,
        )

        print(
            f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} '
            f'lr_old={lrs.get("stage1_modules", float("nan")):.8f} '
            f'lr_trdf={lrs.get("trdf", float("nan")):.8f}',
            flush=True,
        )
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


if __name__ == '__main__':
    main()
