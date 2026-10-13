# -*- coding: utf-8 -*-
"""Independent Full-only RoadFormer+ GFRM submodule experiment.

Every sample uses real CT and PET. Existing encoders, alignments, decoder,
BCE+Dice, optimizer, scheduler, evaluation and EMA policies are retained.
Only the fusion seam changes; --gfrm_enabled False is the genuine Add control.
"""
import json
import os
import time

import numpy as np
import torch

from models.full_roadformer_gfrm import build_gfrm_model
from models.roadformer_gfrm import RoadFormerGFRMFusion
from configs.roadformer_gfrm import FullGFRMConfig
from utils.roadformer_gfrm_visualization import visualize_gfrm
from models.components.add_fusion import AddFusion
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler
from utils.run_common import (count_parameters, module_grad_norm,
                              optimizer_step_succeeded, seed_everything)
from utils.train_logger import append_epoch_log, init_train_log



DIAGNOSTIC_KEYS = ('gamma_ct','gamma_pet','delta_ct_ratio','delta_pet_ratio',
                   'attention_ct_entropy','attention_pet_entropy',
                   'channel_weight_mean','fused_vs_add_ratio')


def assert_core_gradients(model):
    for name in ('enc_ct','enc_pet','ct_align','pet_align','decoder','fusion'):
        module = getattr(model,name)
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            continue
        if any(p.grad is None for p in params):
            raise RuntimeError(f'[GRAD CHECK] disconnected trainable parameters: {name}')
        if not all(torch.isfinite(p.grad).all() for p in params):
            print(f'[GRAD CHECK] {name}: AMP overflow; scaler skip policy stays active', flush=True)
    print('[GRAD CHECK] all enabled trainable modules connected', flush=True)


def load_selected_checkpoint(model, ckpt, cfg):
    saved = ckpt.get('config', {})
    for name in ('experiment_type','gfrm_enabled','image_size_2d','ct_backbone',
                 'pet_backbone','decoder_channels'):
        actual, expected = saved.get(name), getattr(cfg,name,None)
        if name == 'decoder_channels':
            actual, expected = tuple(actual or ()), tuple(expected or ())
        if actual != expected:
            raise ValueError(f'Checkpoint {name} mismatch: {actual!r} vs {expected!r}')
    tag = ckpt.get('eval_weights')
    if tag not in ('raw','ema'):
        raise ValueError('Checkpoint must record eval_weights=raw or ema')
    state = ckpt.get('model_ema' if tag == 'ema' else 'model')
    if state is None:
        raise ValueError(f'Checkpoint has no recorded {tag} weights')
    model.load_state_dict(state, strict=True)
    return tag


def train_step_full(task, batch, diagnostics=False):
    """Full-modality step: paired real CT/PET, unchanged BCE+Dice."""
    ct = batch['ct'].to(task.device, non_blocking=True)
    pet = batch['pet'].to(task.device, non_blocking=True)
    mask = batch['mask'].to(task.device, non_blocking=True).float()
    outputs = task.model(ct, pet=pet, forward_mode='full', return_diagnostics=diagnostics)
    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
    if not torch.isfinite(logits).all():
        raise RuntimeError('[NaN/Inf] logits at train_step_full')
    loss, loss_stats = task.criterion(logits, mask)
    if not torch.isfinite(loss).all():
        raise RuntimeError('[NaN/Inf] loss at train_step_full')
    stats = {
        'loss_total': loss.detach(),
        'loss_bce': loss_stats['loss_bce'],
        'loss_dice': loss_stats['loss_dice'],
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


def _assert_gfrm_protocol(cfg, model):
    assert str(cfg.train_batch_mode) == 'full'
    assert cfg.accumulation_steps == 1
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert str(cfg.optimizer).lower() == 'adamw'
    expected = RoadFormerGFRMFusion if cfg.gfrm_enabled else AddFusion
    if not isinstance(model.fusion, expected):
        raise RuntimeError(f'Expected {expected.__name__}, got {type(model.fusion).__name__}')


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def main():
    print('[INFO] starting Full-only RoadFormer+ GFRM transplant', flush=True)
    cfg = FullGFRMConfig.parse_arguments()
    if not torch.cuda.is_available() and not cfg.allow_cpu:
        raise RuntimeError('CUDA unavailable; check your GPU allocation before training.')
    if cfg.fusion_log_every < 1 or cfg.fusion_vis_every < 1 or cfg.fusion_vis_samples < 0:
        raise ValueError('Diagnostic intervals must be positive; samples >= 0')
    if cfg.ct_backbone != 'convnextv2_nano' or cfg.pet_backbone != 'mit_b1':
        raise ValueError('This comparison keeps CT ConvNeXtV2-Nano / PET MiT-B1')
    if cfg.image_size_2d < 32 or cfg.image_size_2d % 32:
        raise ValueError('image_size_2d must be a positive multiple of 32')
    cfg.experiment_type = 'full_roadformer_gfrm'
    cfg.model_arch = 'full_roadformer_gfrm'
    cfg.train_batch_mode = 'full'
    cfg.fusion_type = 'RoadFormerGFRM' if cfg.gfrm_enabled else 'AddFusion'
    cfg.pet_missing_rate = 0.0
    seed_everything(cfg.random_state)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, test_loader = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)} '
          f'test_batches={len(test_loader)}', flush=True)

    task = MDTSegTeacher(build_gfrm_model(cfg), cfg)
    _assert_gfrm_protocol(cfg, task.model)
    total_params, trainable_params = count_parameters(task.model)
    print(f'[MODEL] params_total={total_params} params_trainable={trainable_params} '
          f'fusion={cfg.fusion_type} fusion_params={sum(p.numel() for p in task.model.fusion.parameters())} '
          f'Full=real_CT+real_PET device={task.device}', flush=True)
    if cfg.gfrm_enabled:
        print('[GFRM] GFFM -> FFRM(GN32,ReLU) -> split-add; '
              'four independent scales; gamma_init=0; enabled_init_equals_raw_add=False; '
              'channel attention, FP32 matching, no extra loss', flush=True)
    else:
        print('[CONTROL] original AddFusion, no GFRM parameters or normalization', flush=True)
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
        'grad_enc_ct', 'grad_enc_pet', 'grad_ct_align', 'grad_pet_align', 'grad_fusion', 'grad_decoder',
        'eval_weights',
        'ema_enabled', 'ema_updates', 'skipped_updates',
        'epoch_time',
    ]
    for scale in range(1,5):
        extra_headers.extend(f'gfrm_s{scale}_{name}' for name in DIAGNOSTIC_KEYS)
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
        grads = {name:[] for name in ('enc_ct','enc_pet','ct_align','pet_align','fusion','decoder')}
        fusion_stats = {}
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                measure = batch_idx == 0 or (batch_idx+1) % cfg.fusion_log_every == 0
                loss, outputs, train_stats = train_step_full(task, batch, diagnostics=measure)
            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()
            grads['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads['enc_pet'].append(module_grad_norm(task.model.enc_pet))
            grads['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads['decoder'].append(module_grad_norm(task.model.decoder))
            if measure:
                grads['pet_align'].append(module_grad_norm(task.model.pet_align))
                grads['fusion'].append(module_grad_norm(task.model.fusion))
                if epoch == 1 and batch_idx == 0:
                    assert_core_gradients(task.model)
                fusion_stats = outputs.get('fusion_diagnostics', {})
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
            if measure:
                print(f'[BATCH {batch_idx+1}] Full={train_stats["num_samples"]} '
                      f'loss={float(loss.detach()):.4f} '
                      f'BCE={float(train_stats["loss_bce"]):.4f} '
                      f'DiceLoss={float(train_stats["loss_dice"]):.4f} '
                      f'EMAupdates={task.ema.updates if task.ema is not None else 0}', flush=True)
                for scale, values in fusion_stats.items():
                    print(f'[GFRM {scale.upper()}] ' + ' '.join(
                        f'{name}={values[name]:.4f}' for name in DIAGNOSTIC_KEYS), flush=True)
            train_loss_sum += float(loss.detach()) * int(train_stats['num_samples'])
            train_sample_count += int(train_stats['num_samples'])
            train_n += 1
            global_batch_step += 1
            task.global_batch_step = global_batch_step

        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full',
                                 model=task.eval_model())
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
                'grad_pet_align': float(np.mean(grads['pet_align'])) if grads['pet_align'] else 0.0,
                'grad_fusion': float(np.mean(grads['fusion'])) if grads['fusion'] else 0.0,
                'eval_weights': task.eval_weights_tag(),
                **{f'gfrm_s{scale}_{key}':fusion_stats.get(f's{scale}',{}).get(key,'disabled')
                   for scale in range(1,5) for key in DIAGNOSTIC_KEYS},
                'ema_enabled': 1.0 if task.ema is not None else 0.0,
                'ema_updates': float(task.ema.updates) if task.ema is not None else 0.0,
                'skipped_updates': float(skipped_update_count),
                'epoch_time': time.time() - epoch_start,
            },
        )
        print(f'[EPOCH {epoch}] val_full_dice={val_full["dice"]:.4f} '
              f'val_full_hd95={val_full["hd95"]:.4f} '
              f'weights={task.eval_weights_tag()} best_full={best_full:.4f} '
              f'lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if cfg.fusion_visualize and (epoch in (1,2) or epoch % cfg.fusion_vis_every == 0):
            vis_paths = visualize_gfrm(task.eval_model(), val_loader, cfg, task.device,
                                       epoch, task.eval_weights_tag())
            if vis_paths:
                print(f'[VIS] {os.path.dirname(vis_paths[0])}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    if not os.path.isfile(paths['best_full']):
        raise RuntimeError('best_full checkpoint was not created')
    ckpt = MDTSegTeacher.load_state_dicts(paths['best_full'])
    weights_tag = load_selected_checkpoint(task.model, ckpt, cfg)
    eval_model = task.model
    eval_model.eval()
    print(f'[FINAL] best_full epoch={ckpt["best_epoch"]} weights={weights_tag}', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full',
                              model=eval_model)
    result = {
        'experiment_type': cfg.experiment_type,
        'fusion_type': cfg.fusion_type,
        'dice': float(test_full['dice']),
        'iou': float(test_full['iou']),
        'hd95': float(test_full['hd95']),
        'hd95_mode': task.metrics.hd95_mode,
        'hd95_unit': 'pixel',
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
