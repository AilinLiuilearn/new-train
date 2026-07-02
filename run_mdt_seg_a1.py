# -*- coding: utf-8 -*-
"""A1 training: PET-Prompted CT Decoder (full PET, no dropout/proxy/DMoME/TPE)."""

import math
import os
import sys

import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from models.pet_prompted_ct_decoder import PET_PROMPT_LOG_KEYS
from run_mdt_seg import (
    _build_loaders,
    _collect_deep_supervision_metrics,
    _deep_supervision_log_headers,
    _grad_finite_and_norm,
    _prepare_env,
    _save_config,
)
from tasks.mdt_seg import MDTSegTeacher, _use_deep_supervision
from utils.model_profile import print_baseline_profile
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log
from utils.vis_teacher import save_segmentation_diagnostics


class A1SegConfig(SegMDTConfig):
    @staticmethod
    def model_parser():
        parser = SegMDTConfig.model_parser()
        parser.set_defaults(model_arch='a1_pet_prompt_ct')
        return parser


def _collect_prompt_metrics(loss_dict):
    metrics = {}
    for key in PET_PROMPT_LOG_KEYS:
        val = loss_dict.get(key)
        if torch.is_tensor(val):
            metrics[key] = float(val.detach().cpu())
        elif val is not None:
            metrics[key] = float(val)
    return metrics


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = A1SegConfig.parse_arguments()
    config.task = 'MDT_Teacher'
    config.model_arch = 'a1_pet_prompt_ct'
    config.train_pet_drop_prob = 0.0
    config.eval_full_pet = True
    config.eval_fixed_missing_pet = False
    config.eval_random_missing_pet = False
    g0 = _prepare_env(config)

    print(f'GPU={g0}')
    print(f'model_arch={config.model_arch}')
    print(f'ct_backbone={getattr(config, "ct_backbone", None)}')
    print(f'dinov3_model_name={getattr(config, "dinov3_model_name", None)}')
    print(f'dinov3_pretrained_path={getattr(config, "dinov3_pretrained_path", None)}')
    print(f'use_deep_supervision={_use_deep_supervision(config)}')
    print(f'image_size_2d={config.image_size_2d}')
    print(f'batch_size={config.batch_size}')
    print(f'lr={config.learning_rate} wd={config.weight_decay}')

    _save_config(config)
    train_loader, val_loader, test_loader = _build_loaders(config)
    networks = build_mdt_seg_teacher(config)

    print('\n' + '=' * 30 + ' MODEL PROFILE ' + '=' * 30)
    print_baseline_profile(networks, config)
    print('=' * 75 + '\n')

    task = MDTSegTeacher(networks, config)
    spe = len(train_loader)
    accum_iter = max(1, int(getattr(config, 'accumulation_steps', 1)))
    updates_per_epoch = math.ceil(spe / accum_iter)
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        config.epochs,
        warmup_steps=config.cosine_warmup * updates_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=updates_per_epoch,
        flat_ratio=getattr(config, 'lr_flat_ratio', 0.3),
    )

    log_path = os.path.join(config.checkpoint_dir, 'train_log.csv')
    extra_headers = _deep_supervision_log_headers(config) + PET_PROMPT_LOG_KEYS
    init_train_log(log_path, extra_headers=extra_headers)
    grad_clip = getattr(config, 'grad_clip', 5.0)
    clip_params = task.trainable_parameters()
    best_dice = -1.0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tn = 0.0, 0
        ds_metric_sum = {k: 0.0 for k in _deep_supervision_log_headers(config)}
        prompt_metric_sum = {k: 0.0 for k in PET_PROMPT_LOG_KEYS}
        metric_steps = 0
        task.set_epoch(epoch)
        task.optimizer.zero_grad()

        print(f'[Epoch {epoch}] start training: steps={spe}', flush=True)
        for i, batch in enumerate(train_loader):
            stepped = False
            with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                loss, _, _, loss_dict = task.train_step(batch)
                loss = loss / accum_iter

            if task.scaler:
                task.scaler.scale(loss).backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        task.scaler.unscale_(task.optimizer)
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                    task.optimizer.zero_grad()
                    stepped = True
            else:
                loss.backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.optimizer.step()
                    task.optimizer.zero_grad()
                    stepped = True

            if task.scheduler and stepped:
                task.scheduler.step()

            step_loss = loss.item() * accum_iter
            tloss += step_loss
            tn += 1
            batch_ds = _collect_deep_supervision_metrics(loss_dict)
            batch_prompt = _collect_prompt_metrics(loss_dict)
            if batch_ds or batch_prompt:
                metric_steps += 1
                for key, value in batch_ds.items():
                    ds_metric_sum[key] = ds_metric_sum.get(key, 0.0) + value
                for key, value in batch_prompt.items():
                    prompt_metric_sum[key] = prompt_metric_sum.get(key, 0.0) + value

            if (i + 1) % 20 == 0 or (i + 1) == 1 or (i + 1) == spe:
                seg_loss = float(loss_dict.get('loss_seg', torch.tensor(0.0)).detach().cpu())
                curr_lr = task.optimizer.param_groups[0]['lr']
                print(
                    f'  Ep{epoch}[{i + 1}/{spe}] loss={step_loss:.4f} seg={seg_loss:.4f} lr={curr_lr:.6g}',
                    flush=True,
                )

        print(f'[Epoch {epoch}] evaluating full PET...', flush=True)
        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        grad_finite, grad_norm = _grad_finite_and_norm(clip_params)
        extra_metrics = {}
        if metric_steps > 0:
            for key, total in ds_metric_sum.items():
                extra_metrics[key] = total / metric_steps
            for key, total in prompt_metric_sum.items():
                extra_metrics[key] = total / metric_steps
        for key in PET_PROMPT_LOG_KEYS:
            if key in val_full:
                extra_metrics[key] = val_full[key]
        extra_metrics['grad_norm'] = grad_norm
        append_epoch_log(
            log_path,
            epoch,
            tloss / max(tn, 1),
            val_full,
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=grad_norm,
            extra_metrics=extra_metrics,
        )
        print(
            f'Epoch {epoch}\n'
            f'train_loss={tloss / max(tn, 1):.4f}\n'
            f'val_full: Dice={val_full["dice"]:.4f} IoU={val_full["iou"]:.4f} HD95={val_full["hd95"]:.2f}',
            flush=True,
        )
        if extra_metrics:
            prompt_line = ' '.join(
                f'{k}={extra_metrics[k]:.4f}'
                for k in PET_PROMPT_LOG_KEYS
                if k in extra_metrics
            )
            if prompt_line:
                print(f'PET prompt maps: {prompt_line}', flush=True)

        if getattr(config, 'vis_every_epoch', False):
            save_segmentation_diagnostics(
                task=task,
                loader=val_loader,
                out_dir=os.path.join(config.checkpoint_dir, 'vis_val_full', f'epoch_{epoch:03d}'),
                num_samples=max(1, int(getattr(config, 'vis_epoch_samples', 2))),
                threshold=getattr(config, 'eval_threshold', 0.5),
                eval_mode='full',
                mode='full',
            )

        if val_full['dice'] > best_dice:
            best_dice = val_full['dice']
            no_improve = 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_dice.pth.tar'), epoch)
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print('[TEST] evaluating full PET...', flush=True)
    test_full = task.evaluate(test_loader, eval_mode='full', tag='test_full')
    print(f'[TEST full] {test_full}', flush=True)


if __name__ == '__main__':
    main()
