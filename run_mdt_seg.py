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
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned, get_pclt20k_memory_loader_cipa_aligned
    train_loader, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers, cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file, checkpoint_dir=cfg.checkpoint_dir)
    memory_loader = get_pclt20k_memory_loader_cipa_aligned(cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers, cfg.random_state, cfg.pin_memory, 'none', cfg.norm_mode, cfg.train_split_file)
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


def _checkpoint_paths(checkpoint_dir):
    return {k: os.path.join(checkpoint_dir, f'ckpt.{k}.pth.tar') for k in ('best_joint', 'best_full', 'best_missing', 'last')}


def module_grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def main():
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    train_loader, val_loader, _, memory_loader = _loaders(cfg)
    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.scheduler = get_cosine_scheduler(task.optimizer, epochs=cfg.epochs, warmup_steps=cfg.cosine_warmup * len(train_loader), min_lr=cfg.cosine_min_lr, steps_per_epoch=len(train_loader), flat_ratio=cfg.lr_flat_ratio)
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'))
    best_joint = best_full = best_missing = -1.0
    best_joint_epoch = 0
    global_batch_step = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)
    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        for batch in train_loader:
            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.mixed_precision) and torch.cuda.is_available()):
                loss, _, _, _ = task.train_step(batch, forward_mode=route)
            (task.scaler.scale(loss) if task.scaler.is_enabled() else loss).backward()
            if task.scaler.is_enabled():
                task.scaler.unscale_(task.optimizer)
            torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip))
            if task.scaler.is_enabled():
                task.scaler.step(task.optimizer); task.scaler.update()
            else:
                task.optimizer.step()
            task.scheduler.step(); global_batch_step += 1
        if getattr(cfg, 'model_arch', 'dual_shared_add_baseline') == 'dual_shared_add_pdtm' and epoch >= int(cfg.pdtm_start_epoch) and ((epoch - int(cfg.pdtm_start_epoch)) % int(cfg.pdtm_rebuild_interval) == 0):
            was_training = task.model.training
            task.model.eval()
            task.model.clear_pdtm_cache()
            pair_count = 0
            for batch in memory_loader:
                task.model.collect_pdtm_pairs(batch['ct'].to(task.device), batch['pet'].to(task.device), case_ids=batch.get('case_id'))
                pair_count = len(task.model._pdtm_pairs)
                if pair_count >= int(cfg.pdtm_max_pairs):
                    break
            task.model.finalize_pdtm_memory()
            pdtm_dir = os.path.join(cfg.checkpoint_dir, 'pdtm')
            build_json = task.model.export_pdtm_json(pdtm_dir, f'epoch_{epoch:03d}_build')
            retrieval_json = task.model.export_pdtm_json(pdtm_dir, f'epoch_{epoch:03d}_retrieval')
            task.model.train(was_training)
        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        val_missing = task.evaluate(val_loader, eval_mode='fixed_missing', tag='val_missing')
        joint_dice = 0.5 * val_full['dice'] + 0.5 * val_missing['dice']
        if joint_dice > best_joint:
            best_joint = joint_dice; best_joint_epoch = epoch
        best_full = max(best_full, val_full['dice'])
        best_missing = max(best_missing, val_missing['dice'])
        task.save_checkpoint(paths['last'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        append_epoch_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'), epoch, 0.0, {'total_loss': 0.0, 'dice': joint_dice, 'iou': 0.0, 'acc': 0.0, 'acc_pixel': 0.0, 'hd95': 0.0}, lr=task.optimizer.param_groups[0]['lr'], grad_norm=0.0, extra_metrics={})


if __name__ == '__main__':
    main()
