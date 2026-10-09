# -*- coding: utf-8 -*-
"""Full-only PET-prompt deformable fusion candidate.
Training loop copied from the inspected Full-only reference; original file untouched.

Every train/val/test sample uses real CT and real PET:
  CT (ConvNeXtV2-Nano) -> ct_align, PET (MiT-B1) -> PET features,
  four-scale candidate fusion -> shared GroupNorm UNetStyleDecoder.

No missing simulation, no Full/Missing split, plain BCE+Dice loss.
Model selection by val Full Dice -> ckpt.best_full.pth.tar.
Final test rebuilds the same configured model and runs full only.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from configs.base import str2bool
from models.build_pet_prompt_deformable import build_pet_prompt_model
from models.pet_prompt_deformable_fusion import MultiScalePETPromptDeformableFusion, export_soft_prompts
from models.components.add_fusion import AddFusion
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (count_parameters, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log


class FullPETPromptConfig(SegMDTConfig):
    """Same shared config, but this entry only allows full mode."""

    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        for action in p._actions:
            if action.dest == 'model_arch':
                action.default = 'dual_shared_pet_prompt_deformable'
                action.choices = ('dual_shared_pet_prompt_deformable',)
        p.add_argument('--fusion_type', choices=('pet_prompt','add'), default='pet_prompt')
        p.add_argument('--fusion_inner_channels', type=int, default=32)
        p.add_argument('--fusion_heads', type=int, default=4)
        p.add_argument('--fusion_offset_radius', type=float, default=2.)
        p.add_argument('--fusion_chunk_rows', type=int, default=16)
        for name in ('checkpoint_chunks','use_gaussian','use_pet_prompt','use_deformable','use_ct_correction','check_finite'):
            p.add_argument('--fusion_'+name, type=str2bool, default=True)
        p.add_argument('--visualize_every', type=int, default=0,
                       help='0 disables; otherwise export a few fixed val samples every N epochs')
        p.add_argument('--visualize_samples', type=int, default=1)
        p.add_argument('--visualize_scales', type=int, nargs='+', default=[1])
        p.add_argument('--visualize_only_checkpoint', default=None)
        return p

    @staticmethod
    def train_parser():
        p = SegMDTConfig.train_parser()
        for action in p._actions:
            if action.dest == 'train_batch_mode':
                action.default = 'full'
                action.choices = ('full',)
                action.help = 'All train/val/test samples use real CT and PET; Full only.'
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(),
                   cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        config._ensure_hash()
        return config


def train_step_full(task, batch):
    """All rows train the configured Full fusion path with original BCE+Dice."""
    ct = batch['ct'].to(task.device, non_blocking=True)
    pet = batch['pet'].to(task.device, non_blocking=True)
    mask = batch['mask'].to(task.device, non_blocking=True).float()
    outputs = task.model(ct, pet=pet, forward_mode='full')
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    if not torch.isfinite(logits).all():
        raise RuntimeError('[NaN/Inf] logits at train_step_full')
    loss, loss_stats = task.criterion(logits, mask)
    if not torch.isfinite(loss).all():
        raise RuntimeError('[NaN/Inf] loss at train_step_full')
    stats = {
        'loss_total': loss.detach(),
        'loss_seg': loss_stats.get('loss_dice', loss.detach()),
        'num_samples': int(ct.shape[0]),
    }
    return loss, outputs, stats


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, cfg.aug_mode, cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
        allow_val_equals_test=bool(getattr(cfg, 'allow_val_equals_test', False)),
        ct_only=False,
    )


def _assert_full_prompt_protocol(cfg, model):
    assert str(cfg.train_batch_mode) == 'full'
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'
    expected = AddFusion if cfg.fusion_type == 'add' else MultiScalePETPromptDeformableFusion
    if not isinstance(model.fusion, expected):
        raise RuntimeError(f'Wrong fusion type: {type(model.fusion).__name__}')


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


FUSION_CONFIG_FIELDS = ('fusion_type','fusion_inner_channels','fusion_heads','fusion_offset_radius',
    'fusion_use_gaussian','fusion_use_pet_prompt','fusion_use_deformable','fusion_use_ct_correction')


@torch.no_grad()
def visualize_model(model, loader, device, cfg, prefix):
    if cfg.fusion_type == 'add':
        print('[VIS] AddFusion has no soft prompts; skipped', flush=True)
        return
    was_training = model.training
    model.eval()
    try:
        first = next(iter(loader))
        n = min(cfg.visualize_samples, first['ct'].shape[0])
        # One bounded diagnostic pass, no training hooks or persistent caches.
        with torch.cuda.amp.autocast(enabled=bool(cfg.mixed_precision) and torch.cuda.is_available()):
            _, maps = model.forward_with_fusion_diagnostics(
                first['ct'][:n].to(device), first['pet'][:n].to(device))
        for i in range(n):
            for scale in cfg.visualize_scales:
                export_soft_prompts(maps[scale-1], os.path.join(cfg.checkpoint_dir,'soft_prompts'),
                    prefix=f'{prefix}_sample{i}_scale{scale}', sample_index=i,
                    ct_image=first['ct'][:n], pet_image=first['pet'][:n], mask=first.get('mask'))
    finally:
        model.train(was_training)


def visualize_checkpoint(cli_cfg):
    checkpoint_path = cli_cfg.visualize_only_checkpoint
    ckpt = MDTSegTeacher.load_state_dicts(checkpoint_path)
    if not isinstance(ckpt.get('config'), dict):
        raise ValueError('Visualization requires recorded checkpoint config')
    cfg = FullPETPromptConfig(args=ckpt['config'])
    # Always preserve trained structure; CLI only controls display/export count.
    cfg.visualize_samples = cli_cfg.visualize_samples
    cfg.visualize_scales = cli_cfg.visualize_scales
    cfg.hash = cli_cfg.hash
    cfg.checkpoint_root = cli_cfg.checkpoint_root
    cfg.pretrained = False  # exact encoder weights are restored below; no downloads
    seed_everything(cfg.random_state)
    _, val_loader, _ = _loaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_pet_prompt_model(cfg)['model'].to(device)
    _assert_full_prompt_protocol(cfg, model)
    tag = ckpt.get('eval_weights','raw')
    if tag not in ('raw','ema'):
        raise ValueError(f'Unknown recorded evaluation weights: {tag}')
    key = 'model_ema' if tag == 'ema' else 'model'
    if ckpt.get(key) is None:
        raise ValueError(f'Recorded {tag} weights missing')
    model.load_state_dict(ckpt[key], strict=True)
    visualize_model(model, val_loader, device, cfg, f'best_epoch{ckpt.get("best_epoch",0)}_{tag}')
    print('[VIS] exported stored validation samples; no training', flush=True)


def main():
    print('[INFO] starting Full-only PET-prompt deformable fusion', flush=True)
    cfg = FullPETPromptConfig.parse_arguments()
    cfg.experiment_type = 'full_pet_prompt_deformable'
    cfg.train_batch_mode = 'full'
    cfg.fusion_module_version = 1
    if cfg.visualize_every < 0 or cfg.visualize_samples < 1 or any(s not in (1,2,3,4) for s in cfg.visualize_scales):
        raise ValueError('Invalid visualization settings')
    if cfg.visualize_only_checkpoint:
        visualize_checkpoint(cfg)
        return
    cfg.pet_missing_rate = 0.0
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_pet_prompt_model(cfg), cfg)
    _assert_full_prompt_protocol(cfg, task.model)
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
        'train_full_loss', 'train_samples', 'train_batches',
        'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc',
        'val_full_acc_pixel', 'val_full_hd95',
        'best_full', 'best_full_epoch',
        'grad_enc_ct', 'grad_enc_pet', 'grad_ct_align', 'grad_decoder',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
                   extra_headers=extra_headers)

    best_full = -1.0
    best_full_epoch = 0
    no_improve = 0
    patience = int(cfg.early_stop_patience)
    amp_enabled = bool(cfg.mixed_precision)
    global_batch_step = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        task.begin_epoch(epoch)
        train_loss_sum = 0.0
        train_sample_count = 0
        train_n = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        skipped_update_count = 0
        grads = {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'decoder': []}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, train_stats = train_step_full(task, batch)
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
                print(f'[BATCH {batch_idx + 1}] route=full '
                      f'loss={float(loss.detach()):.6f}', flush=True)
            train_loss_sum += float(loss.detach()) * int(train_stats['num_samples'])
            train_sample_count += int(train_stats['num_samples'])
            train_n += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step

        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full',
                                 model=task.eval_model())
        if cfg.visualize_every and epoch % cfg.visualize_every == 0:
            visualize_model(task.eval_model(), val_loader, task.device, cfg, f'epoch{epoch:03d}_{task.eval_weights_tag()}')
        improved = val_full['dice'] > best_full
        if improved:
            best_full = val_full['dice']
            best_full_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if improved:
            task.save_checkpoint(paths['best_full'], epoch, best_full,
                                 best_full_epoch, val_full)
        task.save_checkpoint(paths['last'], epoch, best_full, best_full_epoch, val_full)

        train_loss = train_loss_sum / max(1, train_sample_count)
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'), epoch, train_loss,
            {'total_loss': val_full['total_loss'], 'dice': val_full['dice'],
             'iou': val_full['iou'], 'acc': val_full['acc'],
             'acc_pixel': val_full.get('acc_pixel', 0.0), 'hd95': val_full['hd95']},
            lr=task.optimizer.param_groups[0]['lr'], grad_norm=avg_grad_norm,
            extra_metrics={
                'train_full_loss': train_loss,
                'train_samples': float(train_sample_count),
                'train_batches': float(train_n),
                'val_full_loss': val_full['total_loss'],
                'val_full_dice': val_full['dice'],
                'val_full_iou': val_full['iou'],
                'val_full_acc': val_full['acc'],
                'val_full_acc_pixel': val_full.get('acc_pixel', 0.0),
                'val_full_hd95': val_full['hd95'],
                'best_full': best_full,
                'best_full_epoch': best_full_epoch,
                'grad_enc_ct': float(np.mean(grads['enc_ct'])) if grads['enc_ct'] else 0.0,
                'grad_enc_pet': float(np.mean(grads['enc_pet'])) if grads['enc_pet'] else 0.0,
                'grad_ct_align': float(np.mean(grads['ct_align'])) if grads['ct_align'] else 0.0,
                'grad_decoder': float(np.mean(grads['decoder'])) if grads['decoder'] else 0.0,
                'ema_enabled': 1.0 if task.ema is not None else 0.0,
                'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
                'skipped_updates': float(skipped_update_count),
                'epoch_time': time.time() - epoch_start,
            },
        )
        print(f'[EPOCH {epoch}] val_full_dice={val_full["dice"]:.4f} '
              f'best_full={best_full:.4f} '
              f'lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    if not os.path.isfile(paths['best_full']):
        raise RuntimeError('best_full checkpoint was not created')
    ckpt = MDTSegTeacher.load_state_dicts(paths['best_full'])
    recorded = ckpt.get('eval_weights', 'raw')
    if recorded not in ('raw', 'ema'):
        raise ValueError(f'Unknown recorded evaluation weights: {recorded}')
    if recorded == 'ema' and ckpt.get('model_ema') is None:
        raise ValueError('Best checkpoint recorded EMA but EMA weights are missing')
    use_ema = recorded == 'ema'
    weights_tag = 'ema' if use_ema else 'raw'
    recorded_cfg = ckpt.get('config', {})
    for name in FUSION_CONFIG_FIELDS:
        if name in recorded_cfg and getattr(cfg, name) != recorded_cfg[name]:
            raise RuntimeError(f'Checkpoint/config mismatch: {name}')
    cfg.pretrained = False  # restore complete trained weights, not initialization again
    eval_model = build_pet_prompt_model(cfg)['model'].to(task.device)
    _assert_full_prompt_protocol(cfg, eval_model)
    eval_model.load_state_dict(ckpt['model_ema'] if use_ema else ckpt['model'],
                               strict=True)
    eval_model.eval()
    print(f'[FINAL] evaluating best_full checkpoint at epoch {ckpt["best_epoch"]} '
          f'with weights={weights_tag} (recorded={recorded})', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full',
                              model=eval_model)
    result = {
        'experiment_type': 'full_pet_prompt_deformable',
        'dice': float(test_full['dice']),
        'iou': float(test_full['iou']),
        'hd95': float(test_full['hd95']),
        'loss': float(test_full['total_loss']),
        'weights': weights_tag,
        'best_epoch': ckpt['best_epoch'],
        'checkpoint': os.path.abspath(paths['best_full']),
    }
    with open(os.path.join(cfg.checkpoint_dir, 'final_test_full.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('[FINAL TEST FULL]', json.dumps(result, indent=2), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
