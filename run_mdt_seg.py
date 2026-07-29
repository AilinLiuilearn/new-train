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
    from datasets.pclt20k_seg import PCLT20KSegDataset, get_pclt20k_loaders_cipa_aligned, _make_loader
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
    memory_loader = None
    if cfg.model_arch == 'dual_shared_add_cpbdm':
        memory_ds = PCLT20KSegDataset(
            records=train_loader.dataset.records,
            image_size=cfg.image_size_2d,
            train=False,
            random_state=cfg.random_state,
            aug_mode='none',
            norm_mode=cfg.norm_mode,
        )
        memory_loader = _make_loader(memory_ds, cfg.batch_size, cfg.num_workers, False, False, cfg.random_state + 97, cfg.pin_memory)
    return train_loader, val_loader, test_loader, memory_loader


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


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, _, memory_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)
    assert train_loader.batch_size == 16
    assert val_loader.batch_size == 16
    if memory_loader is not None:
        assert memory_loader.batch_size == 16
        print('[CPBDM] train_batch_size=16')
        print('[CPBDM] memory_batch_size=16')
        print('[CPBDM] val_batch_size=16')

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
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
        'train_full_loss',
        'train_missing_loss',
        'train_overall_loss',
        'full_train_batches',
        'missing_train_batches',
        'val_full_loss',
        'val_full_dice',
        'val_full_iou',
        'val_full_acc',
        'val_full_acc_pixel',
        'val_full_hd95',
        'val_missing_loss',
        'val_missing_dice',
        'val_missing_iou',
        'val_missing_acc',
        'val_missing_acc_pixel',
        'val_missing_hd95',
        'joint_dice',
        'best_joint',
        'best_joint_epoch',
        'grad_full_enc_ct',
        'grad_missing_enc_ct',
        'grad_full_ct_align',
        'grad_missing_ct_align',
        'grad_full_decoder',
        'grad_missing_decoder',
        'cpbdm_memory_ready',
        'cpbdm_key_pair_cos',
        'cpbdm_raw_retrieval_entropy',
        'cpbdm_retrieval_entropy',
        'cpbdm_raw_effective_slots',
        'cpbdm_effective_slots',
        'cpbdm_raw_top1_weight',
        'cpbdm_top1_weight',
        'cpbdm_max_similarity',
        'cpbdm_raw_delta_abs_mean',
        'cpbdm_delta_abs_mean',
        'cpbdm_consensus_weight_change',
        'cpbdm_raw_delta_tv',
        'cpbdm_coherent_delta_tv',
        'cpbdm_positive_ratio',
        'cpbdm_negative_ratio',
        'cpbdm_zero_ratio',
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

        if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is not None and epoch % int(cfg.gradient_diagnostics_interval) == 0:
            diag_stats = task.gradient_diagnostics(fixed_diag_batch, max_samples=min(1, int(cfg.gradient_diagnostics_num_samples))) or {}

        if cfg.model_arch == 'dual_shared_add_cpbdm' and memory_loader is not None:
            build_report = task.rebuild_cpbdm_memory(memory_loader, epoch)
            task.model.cpbdm.reset_retrieval_stats()
        else:
            build_report = {}
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
        cpbdm_diag = task.model.cpbdm_diagnostics() if cfg.model_arch == 'dual_shared_add_cpbdm' else {}
        weighted_ratio = cpbdm_diag.get('global_weighted_event_ratio', {'positive': 0.0, 'negative': 0.0, 'zero': 0.0}) if cpbdm_diag else {'positive': 0.0, 'negative': 0.0, 'zero': 0.0}
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
                'cpbdm_memory_ready': float(cpbdm_diag.get('memory_ready', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_key_pair_cos': float(cpbdm_diag.get('key_pairwise_cosine_mean', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_raw_retrieval_entropy': float(cpbdm_diag.get('retrieval_running', {}).get('raw_entropy', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_retrieval_entropy': float(cpbdm_diag.get('retrieval_running', {}).get('entropy', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_raw_effective_slots': float(cpbdm_diag.get('retrieval_running', {}).get('raw_effective_slots', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_effective_slots': float(cpbdm_diag.get('retrieval_running', {}).get('effective_slots', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_raw_top1_weight': float(cpbdm_diag.get('retrieval_running', {}).get('raw_top1_weight', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_top1_weight': float(cpbdm_diag.get('retrieval_running', {}).get('top1_weight', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_max_similarity': float(cpbdm_diag.get('retrieval_running', {}).get('max_similarity', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_raw_delta_abs_mean': float(cpbdm_diag.get('retrieval_running', {}).get('raw_delta_abs_mean', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_delta_abs_mean': float(cpbdm_diag.get('retrieval_running', {}).get('delta_abs_mean', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_consensus_weight_change': float(cpbdm_diag.get('retrieval_running', {}).get('consensus_weight_change', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_raw_delta_tv': float(cpbdm_diag.get('retrieval_running', {}).get('raw_delta_tv', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_coherent_delta_tv': float(cpbdm_diag.get('retrieval_running', {}).get('coherent_delta_tv', 0.0)) if cpbdm_diag else 0.0,
                'cpbdm_positive_ratio': float(weighted_ratio.get('positive', 0.0)),
                'cpbdm_negative_ratio': float(weighted_ratio.get('negative', 0.0)),
                'cpbdm_zero_ratio': float(weighted_ratio.get('zero', 0.0)),
                'epoch_time': time.time() - epoch_start,
                **{f'diag_{k}': v for k, v in diag_stats.items()},
            },
        )

        print(f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


if __name__ == '__main__':
    main()
