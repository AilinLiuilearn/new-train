# -*- coding: utf-8 -*-
import json
import os
import random
import subprocess
import sys
import time
import warnings

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
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = False


def _silence_known_warnings():
    warnings.filterwarnings(
        'ignore',
        message='Deterministic behavior was enabled with either `torch.use_deterministic_algorithms`*',
        category=UserWarning,
    )


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
    assert float(cfg.joint_full_weight) == 0.5
    assert float(cfg.joint_missing_weight) == 0.5


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
    return float(total.sqrt().item()) if total is not None else 0.0


def _check_model_finite(model):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            return f'parameter:{name}'
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point() and not torch.isfinite(buffer).all():
            return f'buffer:{name}'
    return None


def _tensor_finite_stats(x):
    x = x.detach().float()
    finite = torch.isfinite(x)
    if finite.any():
        vals = x[finite]
        return {
            'finite': True,
            'min': float(vals.min().item()),
            'max': float(vals.max().item()),
            'mean': float(vals.mean().item()),
        }
    return {'finite': False, 'min': None, 'max': None, 'mean': None}


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_joint': os.path.join(checkpoint_dir, 'ckpt.best_joint.pth.tar'),
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'best_missing': os.path.join(checkpoint_dir, 'ckpt.best_missing.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _git_info(repo_dir):
    def _run(args):
        return subprocess.check_output(args, cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
    try:
        return {
            'git_commit_sha': _run(['git', 'rev-parse', 'HEAD']),
            'git_branch': _run(['git', 'branch', '--show-current']),
        }
    except Exception:
        return {'git_commit_sha': None, 'git_branch': None}


def _write_reproducibility(cfg):
    payload = {
        **_git_info('/root/autodl-tmp/mkd-main/new-train'),
        'sys_argv': sys.argv,
        'random_state': cfg.random_state,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'fusion_module': 'APSF',
        'missing_compensation': 'MPPC',
        'apsf_auxiliary_loss': False,
        'amp_init_scale': getattr(cfg, 'amp_init_scale', 4096.0),
        'mppc_params': {'num_slots': 3, 'momentum': 0.9, 'temperature': 0.1, 'gate_init_logit': -6.0},
        'ct_backbone': getattr(cfg, 'ct_backbone', 'convnextv2_nano'),
        'pet_backbone': getattr(cfg, 'pet_backbone', 'mit_b1'),
        'train_split_file': getattr(cfg, 'train_split_file', None),
        'val_split_file': getattr(cfg, 'val_split_file', None),
        'test_split_file': getattr(cfg, 'test_split_file', None),
        'norm_mode': getattr(cfg, 'norm_mode', None),
        'aug_mode': getattr(cfg, 'aug_mode', None),
    }
    with open(os.path.join(cfg.checkpoint_dir, 'reproducibility.json'), 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    _silence_known_warnings()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    _write_reproducibility(cfg)

    train_loader, val_loader, _ = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    total_params, trainable_params = _count_parameters(task.model)
    mppc_params = sum(p.numel() for p in task.model.mppc.parameters()) if getattr(task.model, 'mppc', None) is not None else 0
    apsf_params = sum(p.numel() for p in task.model.apsf.parameters()) if getattr(task.model, 'apsf', None) is not None else 0
    print(f'[INFO] random_state={cfg.random_state} params_total={total_params} params_trainable={trainable_params} mppc_params={mppc_params} apsf_params={apsf_params}', flush=True)
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
        skip_counts = {'full': 0, 'missing': 0}
        consecutive_skips = 0
        epoch_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, loss_stats = task.train_step(batch, forward_mode=route)
            bad_state_after_forward = _check_model_finite(task.model)
            if bad_state_after_forward is not None:
                raise FloatingPointError(
                    f'non-finite model state after forward: {bad_state_after_forward} '
                    f'(step={global_batch_step}, batch={batch_idx}, route={route})'
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'non-finite loss: step={global_batch_step}, '
                    f'batch={batch_idx}, route={route}, value={loss.detach()}'
                )

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            bad_grad = None
            for name, parameter in task.model.named_parameters():
                grad = parameter.grad
                if grad is not None and not torch.isfinite(grad).all():
                    finite = grad.detach()[torch.isfinite(grad.detach())]
                    stats = {'min': None, 'max': None}
                    if finite.numel() > 0:
                        stats = {'min': float(finite.min().item()), 'max': float(finite.max().item())}
                    bad_grad = (name, grad, stats)
                    break
            if bad_grad is not None:
                name, grad, stats = bad_grad
                if task.scaler.is_enabled():
                    scale_before = float(task.scaler.get_scale())
                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                    scale_after = float(task.scaler.get_scale())
                    print(f'[SKIP] non-finite grad at step={global_batch_step} batch={batch_idx} route={route} name={name} dtype={grad.dtype} shape={tuple(grad.shape)} nan={int(torch.isnan(grad).sum().item())} posinf={int(torch.isposinf(grad).sum().item())} neginf={int(torch.isneginf(grad).sum().item())} finite_min={stats["min"]} finite_max={stats["max"]} scale_before={scale_before} scale_after={scale_after}', flush=True)
                    skip_counts[route] += 1
                    consecutive_skips += 1
                    if consecutive_skips > 20:
                        raise FloatingPointError(f'too many consecutive skipped AMP steps: {consecutive_skips}')
                else:
                    raise FloatingPointError(
                        f'non-finite gradient: step={global_batch_step}, batch={batch_idx}, route={route}, '
                        f'name={name}, dtype={grad.dtype}, shape={tuple(grad.shape)}'
                    )
                task.optimizer.zero_grad(set_to_none=True)
                global_batch_step += 1
                task.global_batch_step = global_batch_step
                continue

            total_grad_norm = torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip), error_if_nonfinite=True) if float(cfg.grad_clip) > 0 else 0.0
            grad_norm_accum += float(total_grad_norm)
            grad_norm_steps += 1

            if task.scaler.is_enabled():
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                task.optimizer.step()
            consecutive_skips = 0
            skip_counts[route] += 0

            bad_state = _check_model_finite(task.model)
            if bad_state is not None:
                raise FloatingPointError(
                    f'non-finite model state after optimizer step: {bad_state} '
                    f'(step={global_batch_step}, batch={batch_idx}, route={route})'
                )

            task.scheduler.step()

            if (batch_idx + 1) % 100 == 0:
                print(f'[BATCH {batch_idx + 1}] route={route} total={float(loss.detach()):.6f} seg={float(loss_stats["loss_seg"].detach()):.6f}', flush=True)

            if route == 'full':
                full_n += 1
                full_loss += float(loss.detach())
            else:
                missing_n += 1
                missing_loss += float(loss.detach())

            global_batch_step += 1
            task.global_batch_step = global_batch_step

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
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=avg_grad_norm,
            extra_metrics={
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
            },
        )

        print(f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


if __name__ == '__main__':
    main()
