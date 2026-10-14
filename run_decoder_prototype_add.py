"""Exact-half mixed Add experiment; only new model/task/config are selected.

Loader, BCE+Dice, seeded masks, scheduler, metrics and test-rate evaluation
are reused from the clean baseline. Original baseline runners are untouched.
"""
import argparse
import copy
import csv
import json
import os
import time

import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.prototype_add_model import build_prototype_model
from tasks.prototype_seg import PrototypeSegTask
from run_full_missing_baseline import _loaders, _assert_mixed, _checkpoint_paths
from utils.optimization import get_cosine_scheduler
from utils.run_common import (seed_everything, build_balanced_pet_available,
                              optimizer_step_succeeded, eval_missing_rates)
from utils.train_logger import init_train_log, append_epoch_log, CSV_HEADER


class PrototypeConfig(SegMDTConfig):
    @staticmethod
    def task_specific_parser():
        p = argparse.ArgumentParser('Decoder prototypes', add_help=False)
        p.add_argument('--prototype_enabled', type=str2bool, default=True)
        p.add_argument('--prototype_spatial', type=str2bool, default=True)
        p.add_argument('--prototype_channel', type=str2bool, default=True)
        p.add_argument('--prototype_dim', type=int, default=32)
        p.add_argument('--prototype_clusters', type=int, default=6)
        p.add_argument('--prototype_capacity', type=int, default=256)
        p.add_argument('--prototype_min_candidates', type=int, default=24)
        p.add_argument('--prototype_decay', type=float, default=.999)
        p.add_argument('--prototype_refresh_steps', type=int, default=20)
        p.add_argument('--prototype_warmup_steps', type=int, default=100)
        p.add_argument('--prototype_foreground_weight', type=float, default=2.)
        p.add_argument('--prototype_background_weight', type=float, default=1.)
        p.add_argument('--prototype_min_gain', type=float, default=.01)
        p.add_argument('--prototype_ct_aux_weight', type=float, default=.1)
        p.add_argument('--resume', type=str, default=None)
        return p


def main():
    cfg = PrototypeConfig.parse_arguments()
    if cfg.train_batch_mode != 'mixed' or cfg.optimizer != 'adamw':
        raise ValueError('prototype experiment requires --train_batch_mode mixed --optimizer adamw')
    cfg.experiment_type = 'decoder_dual_prototype_add'
    cfg.model_arch = 'prototype_add_model'
    cfg.fusion_type, cfg.pet_missing_rate = 'AddFusion', .5
    if not {0., 1.}.issubset(cfg.final_test_missing_rates) or any(
            rate < 0 or rate > 1 for rate in cfg.final_test_missing_rates):
        raise ValueError('final_test_missing_rates must be in [0,1] and include 0 and 1 for Joint')
    seed_everything(cfg.random_state)
    train_loader, val_loader, test_loader = _loaders(cfg)
    _assert_mixed(cfg, train_loader)
    cfg.steps_per_epoch = len(train_loader)
    # Restore complete checkpoint weights without needing original pretrained files.
    build_cfg = copy.deepcopy(cfg)
    if cfg.resume:
        build_cfg.pretrained = False
        build_cfg.ct_pretrained_path = build_cfg.pet_pretrained_path = None
    task = PrototypeSegTask(build_prototype_model(build_cfg), cfg)
    task.loader_generators = {name: loader.generator for name, loader in
                              (('train', train_loader), ('val', val_loader), ('test', test_loader))
                              if getattr(loader, 'generator', None) is not None}
    task.scheduler = get_cosine_scheduler(
        task.optimizer, epochs=cfg.epochs, warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr, steps_per_epoch=len(train_loader), flat_ratio=cfg.lr_flat_ratio)
    paths = _checkpoint_paths(cfg.checkpoint_dir)
    start_epoch, best, best_epoch = 1, -1., 0
    if cfg.resume:
        ckpt = task.resume(cfg.resume)
        start_epoch, best, best_epoch = int(ckpt['epoch']) + 1, float(ckpt['best']), int(ckpt['best_epoch'])
        # A resumed run must keep the original best candidate for final selection.
        old_best = os.path.join(os.path.dirname(os.path.abspath(cfg.resume)), 'ckpt.best_joint.pth.tar')
        if not os.path.isfile(paths['best_joint']):
            if not os.path.isfile(old_best):
                raise ValueError('resume requires original ckpt.best_joint.pth.tar alongside last checkpoint')
            import shutil
            shutil.copy2(old_best, paths['best_joint'])
    elif os.path.exists(paths['last']):
        raise ValueError('checkpoint directory already contains a run; use a new --hash or --resume')

    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)
    extra_headers = ['train_full_loss', 'train_missing_loss', 'train_ct_aux_loss',
                     'train_full_samples', 'train_missing_samples', 'joint_dice',
                     'val_full_dice', 'val_missing_dice', 'best_joint',
                     'prototype_spatial_valid', 'prototype_channel_valid',
                     'skipped_updates', 'epoch_seconds']
    log_path = os.path.join(cfg.checkpoint_dir, 'train_log.csv')
    if cfg.resume and os.path.exists(log_path):
        with open(log_path, newline='') as f:
            if next(csv.reader(f)) != CSV_HEADER + extra_headers:
                raise ValueError('resume log schema does not match this runner')
    else:
        init_train_log(log_path, extra_headers)

    for epoch in range(start_epoch, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        sums = dict(full=0., missing=0., ct=0., total=0., grad=0.)
        batches, samples, skipped = 0, 0, 0
        begin = time.time()
        for batch in train_loader:
            if batch['ct'].shape[0] != cfg.batch_size:
                raise ValueError('expected fixed full-size batches with drop_last=True')
            state = build_balanced_pet_available(cfg.batch_size, task.global_batch_step,
                                                 cfg.random_state, task.device)
            task.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=task.device.type,
                                enabled=bool(cfg.mixed_precision) and task.device.type == 'cuda'):
                loss, _, _, stats = task.train_step_mixed(batch, state)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grad = (torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), cfg.grad_clip)
                    if cfg.grad_clip > 0 else 0.)
            flags = [torch.isfinite(p.grad).all() for p in task.model.parameters() if p.grad is not None]
            finite_gradients = bool(torch.stack(flags).all()) if flags else False
            if task.scaler.is_enabled():
                # GradScaler owns overflow detection/update of its scale.
                succeeded = optimizer_step_succeeded(task)
            else:
                succeeded = finite_gradients and optimizer_step_succeeded(task)
            task.after_optimizer_step(succeeded)
            if succeeded:
                task.scheduler.step()
                task.update_ema()
            else:
                skipped += 1
            for key, stat in (('full', 'loss_full'), ('missing', 'loss_missing'), ('ct', 'loss_ct_aux')):
                sums[key] += float(stats[stat])
            sums['total'] += float(loss.detach())
            sums['grad'] += float(grad)
            batches += 1
            samples += cfg.batch_size // 2
            task.global_batch_step += 1
        if batches == 0:
            raise ValueError('training loader has no complete mixed batch')
        active = task.eval_model()
        full = task.evaluate(val_loader, eval_mode='full', model=active)
        missing = task.evaluate(val_loader, eval_mode='missing', model=active)
        joint = .5 * (full['dice'] + missing['dice'])
        improved = joint > best
        if improved:
            best, best_epoch, task.no_improve = joint, epoch, 0
        else:
            task.no_improve += 1
        val = {'full': full, 'missing': missing, 'joint_dice': joint}
        if improved:
            task.save_checkpoint(paths['best_joint'], epoch, best, best_epoch, val)
        task.save_checkpoint(paths['last'], epoch, best, best_epoch, val)
        diag = task.model.prototype.diagnostics()
        with open(os.path.join(cfg.checkpoint_dir, 'prototype_log.jsonl'), 'a') as f:
            f.write(json.dumps(dict(epoch=epoch, **diag)) + '\n')
        sv = sum(diag[f'd{l}_spatial_valid'] for l in (4, 3, 2, 1))
        cv = sum(diag[f'd{l}_channel_valid'] for l in (4, 3, 2, 1))
        joint_metrics = {key: .5 * (full[key] + missing[key])
                         for key in ('total_loss', 'dice', 'iou', 'acc', 'acc_pixel', 'hd95')}
        append_epoch_log(log_path, epoch, sums['total'] / batches, joint_metrics,
                         lr=task.optimizer.param_groups[0]['lr'], grad_norm=sums['grad'] / batches,
                         extra_metrics=dict(
                             train_full_loss=sums['full'] / batches,
                             train_missing_loss=sums['missing'] / batches,
                             train_ct_aux_loss=sums['ct'] / batches,
                             train_full_samples=samples, train_missing_samples=samples,
                             joint_dice=joint, val_full_dice=full['dice'], val_missing_dice=missing['dice'],
                             best_joint=best, prototype_spatial_valid=sv, prototype_channel_valid=cv,
                             skipped_updates=skipped, epoch_seconds=time.time() - begin))
        print(f'[epoch {epoch}] Full={full["dice"]:.4f} Missing={missing["dice"]:.4f} '
              f'Joint={joint:.4f} spatial_valid={sv}/4 channel_valid={cv}/4', flush=True)
        if task.no_improve >= cfg.early_stop_patience:
            break

    ckpt = task.load_state_dicts(paths['best_joint'])
    weights_tag = ckpt['eval_weights']
    # Honor the weights actually used to select this checkpoint, including
    # checkpoints selected before model EMA activation.
    state = ckpt['model_ema'] if weights_tag == 'ema' else ckpt['model']
    task.model.load_state_dict(state, strict=True)
    task.model.eval()
    eval_missing_rates(task, task.model, test_loader, cfg.random_state, cfg.checkpoint_dir,
                       'final_test', rates=list(cfg.final_test_missing_rates), weights_tag=weights_tag)


if __name__ == '__main__':
    main()
