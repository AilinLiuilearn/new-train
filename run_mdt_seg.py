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
from utils.train_logger import append_epoch_log, init_train_log


def _seed(cfg):
    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_state)


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    return get_pclt20k_loaders_cipa_aligned(cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers, cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file)


def _assert_baseline(cfg):
    assert cfg.accumulation_steps == 1
    assert float(cfg.train_pet_drop_prob) == 0.0
    assert float(cfg.missing_loss_weight) == 1.0
    assert float(cfg.joint_full_weight) == 0.5
    assert float(cfg.joint_missing_weight) == 0.5
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert float(cfg.boundary_loss_weight) == 0.0


def main():
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    train_loader, val_loader, _ = _loaders(cfg)
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    extra_headers = ['train_full_loss', 'train_missing_loss', 'train_overall_loss', 'full_train_batches', 'missing_train_batches', 'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc', 'val_full_acc_pixel', 'val_full_hd95', 'val_missing_loss', 'val_missing_dice', 'val_missing_iou', 'val_missing_acc', 'val_missing_acc_pixel', 'val_missing_hd95', 'joint_dice', 'best_joint', 'best_joint_epoch', 'grad_full_enc_ct', 'grad_missing_enc_ct', 'grad_full_ct_align', 'grad_missing_ct_align', 'grad_full_decoder', 'grad_missing_decoder', 'epoch_time']
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'), extra_headers=extra_headers)
    best_joint = best_full = best_missing = -1.0
    best_joint_epoch = 0
    global_batch_step = 0
    amp_enabled = bool(cfg.mixed_precision)
    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        full_n = missing_n = 0
        full_loss = missing_loss = 0.0
        epoch_start = time.time()
        for batch in train_loader:
            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, _, _ = task.train_step(batch, forward_mode=route)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip))) if float(cfg.grad_clip) > 0 else 0.0
            if task.scaler.is_enabled():
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                task.optimizer.step()
            if route == 'full':
                full_n += 1; full_loss += float(loss.detach())
            else:
                missing_n += 1; missing_loss += float(loss.detach())
            global_batch_step += 1
            task.global_batch_step = global_batch_step
        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        val_missing = task.evaluate(val_loader, eval_mode='fixed_missing', tag='val_missing')
        joint_dice = float(cfg.joint_full_weight) * val_full['dice'] + float(cfg.joint_missing_weight) * val_missing['dice']
        if joint_dice > best_joint:
            best_joint = joint_dice; best_joint_epoch = epoch
            task.save_checkpoint(os.path.join(cfg.checkpoint_dir, 'ckpt.best_joint.pth.tar'), epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
            task.save_checkpoint(os.path.join(cfg.checkpoint_dir, 'ckpt.best.pth.tar'), epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        if val_full['dice'] > best_full:
            best_full = val_full['dice']
            task.save_checkpoint(os.path.join(cfg.checkpoint_dir, 'ckpt.best_full.pth.tar'), epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        if val_missing['dice'] > best_missing:
            best_missing = val_missing['dice']
            task.save_checkpoint(os.path.join(cfg.checkpoint_dir, 'ckpt.best_missing.pth.tar'), epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        task.save_checkpoint(os.path.join(cfg.checkpoint_dir, 'ckpt.last.pth.tar'), epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            (full_loss + missing_loss) / max(1, full_n + missing_n),
            {'total_loss': val_full['total_loss'], 'dice': joint_dice, 'iou': 0.0, 'acc': 0.0, 'acc_pixel': 0.0, 'hd95': 0.0},
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=grad_norm,
            extra_metrics={
                'train_full_loss': full_loss / max(1, full_n),
                'train_missing_loss': missing_loss / max(1, missing_n),
                'train_overall_loss': (full_loss + missing_loss) / max(1, full_n + missing_n),
                'full_train_batches': full_n,
                'missing_train_batches': missing_n,
                'val_full_loss': val_full['total_loss'],
                'val_full_dice': val_full['dice'], 'val_full_iou': val_full['iou'], 'val_full_acc': val_full['acc'], 'val_full_acc_pixel': val_full.get('acc_pixel', 0.0), 'val_full_hd95': val_full['hd95'],
                'val_missing_loss': val_missing['total_loss'], 'val_missing_dice': val_missing['dice'], 'val_missing_iou': val_missing['iou'], 'val_missing_acc': val_missing['acc'], 'val_missing_acc_pixel': val_missing.get('acc_pixel', 0.0), 'val_missing_hd95': val_missing['hd95'],
                'joint_dice': joint_dice, 'best_joint': best_joint, 'best_joint_epoch': best_joint_epoch,
                'grad_full_enc_ct': 0.0, 'grad_missing_enc_ct': 0.0, 'grad_full_ct_align': 0.0, 'grad_missing_ct_align': 0.0, 'grad_full_decoder': 0.0, 'grad_missing_decoder': 0.0,
                'epoch_time': time.time() - epoch_start,
            },
        )
    print('done')


if __name__ == '__main__':
    main()
