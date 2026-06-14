# -*- coding: utf-8 -*-
"""Teacher training entry for PVTv2-B1 lightweight UNet baseline."""

import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.model_profile import print_baseline_profile
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log
from utils.vis_teacher import save_segmentation_diagnostics

_torch_version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
if _torch_version >= (2, 1):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def _load_dataset_module():
    root = os.getcwd()
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_env(config):
    gpus = [int(g) for g in config.gpus] if config.gpus else [0]
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        gpus = [g for g in gpus if 0 <= g < visible]
        if not gpus:
            gpus = [0]
    config.gpus = gpus
    g0 = gpus[0]
    seed = int(config.random_state)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if torch.cuda.is_available():
        torch.cuda.set_device(g0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return g0


def _build_loaders(config):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'use_textproxy_loader', False):
        return dataset_mod.get_pclt20k_loaders_textproxy_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'cipa'),
            train_list=getattr(config, 'train_list', 'train_orgian.txt'),
            val_list=getattr(config, 'val_list', 'test.txt'),
            test_list=getattr(config, 'test_list', 'test.txt'),
            pet_drop_prob=getattr(config, 'pet_drop_prob', 0.4),
            eval_missing_pet=getattr(config, 'eval_missing_pet', False),
        )
    if getattr(config, 'cipa_aligned', False):
        return dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'imagenet'),
            train_split_file=getattr(config, 'train_split_file', 'train.txt'),
            val_split_file=getattr(config, 'val_split_file', 'val.txt'),
            test_split_file=getattr(config, 'test_split_file', 'test.txt'),
        )
    return dataset_mod.get_pclt20k_loaders(
        config.root,
        config.image_size_2d,
        config.batch_size,
        config.num_workers,
        val_ratio=config.val_ratio,
        random_state=config.random_state,
        use_case_split=getattr(config, 'use_case_split', True),
        pin_memory=getattr(config, 'pin_memory', True),
        aug_mode=getattr(config, 'aug_mode', 'cipa'),
        norm_mode=getattr(config, 'norm_mode', 'imagenet'),
    )


def _save_config(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _collect_adc_gamma(task):
    model = _unwrap_model(task.networks.get('model'))
    gammas = {}
    for branch_name in ('enc_ct', 'enc_pet'):
        encoder = getattr(model, branch_name, None)
        if encoder is None or not hasattr(encoder, 'adc_mac'):
            continue
        for stage_idx, mac in enumerate(encoder.adc_mac, start=1):
            gamma = getattr(mac, 'gamma', None)
            if gamma is not None:
                gammas[f'{branch_name}_s{stage_idx}_gamma'] = float(gamma.detach().cpu())
    return gammas


def _adc_gamma_headers(config):
    if not getattr(config, 'use_adc_mac', False):
        return []
    return [f'{branch}_s{stage}_gamma' for branch in ('enc_ct', 'enc_pet') for stage in range(1, 5)]


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = SegMDTConfig.parse_arguments()
    config.task = 'MDT_Teacher'
    g0 = _prepare_env(config)

    print(f'GPU={g0} backbone={config.backbone} single_modality={getattr(config, "single_modality", False)}')
    print(f'lr={config.learning_rate} wd={config.weight_decay} bs={config.batch_size} boundary_w={getattr(config, "boundary_loss_weight", 0.0)}')

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
    gamma_headers = _adc_gamma_headers(config)
    init_train_log(log_path, extra_headers=gamma_headers)

    grad_clip = getattr(config, 'grad_clip', 0.5)
    clip_params = [p for net in task.networks.values() for p in net.parameters()]
    best_dice, best_dice_epoch = -1.0, 0
    best_hd95, best_hd95_epoch = float('inf'), 0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tseg, tboundary, tn = 0.0, 0.0, 0.0, 0
        grad_norm_sum, grad_norm_steps = 0.0, 0
        model_for_epoch = _unwrap_model(task.networks.get('model'))
        if hasattr(model_for_epoch, 'set_epoch'):
            model_for_epoch.set_epoch(epoch)
        task.set_epoch(epoch)
        task.optimizer.zero_grad()

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
                        total_norm = torch.nn.utils.clip_grad_norm_(clip_params, float('inf'))
                        grad_norm_sum += float(total_norm)
                        grad_norm_steps += 1
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                    task.optimizer.zero_grad()
                    stepped = True
            else:
                loss.backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(clip_params, float('inf'))
                        grad_norm_sum += float(total_norm)
                        grad_norm_steps += 1
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.optimizer.step()
                    task.optimizer.zero_grad()
                    stepped = True

            if task.scheduler and stepped:
                task.scheduler.step()

            step_loss = loss.item() * accum_iter
            step_seg = float(loss_dict.get('loss_seg', torch.tensor(0.0)).item())
            step_boundary = float(loss_dict.get('loss_boundary', torch.tensor(0.0)).item())
            tloss += step_loss
            tseg += step_seg
            tboundary += step_boundary
            tn += 1
            if (i + 1) % 50 == 0:
                curr_lr = task.optimizer.param_groups[0]['lr']
                print(
                    f'  Ep{epoch}[{i + 1}/{spe}] '
                    f'loss={step_loss:.4f} '
                    f'seg={step_seg:.4f} '
                    f'boundary={step_boundary:.4f} '
                    f'total={float(loss_dict.get("loss_total", loss.detach()).item()):.4f} '
                    f'lr={curr_lr:.6f}'
                )

        if getattr(config, 'eval_missing_pet', False):
            val_full = task.evaluate(val_loader, tag='val_full', force_missing_pet=False)
            val_missing = task.evaluate(val_loader, tag='val_missing', force_missing_pet=True)
            val_m = val_full
        else:
            val_full = task.evaluate(val_loader, tag='val_full', force_missing_pet=False)
            val_missing = None
            val_m = val_full
        avg_grad_norm = grad_norm_sum / max(grad_norm_steps, 1)
        curr_lr = task.optimizer.param_groups[0]['lr']
        gamma_metrics = _collect_adc_gamma(task)
        append_epoch_log(
            log_path,
            epoch,
            tloss / max(tn, 1),
            val_m,
            lr=curr_lr,
            grad_norm=avg_grad_norm,
            extra_metrics=gamma_metrics,
        )
        gamma_text = ''
        if gamma_metrics:
            gamma_text = ' ' + ' '.join(f'{k}={v:.4f}' for k, v in gamma_metrics.items())
        gate_text = 'g_pet={:.4f} g_txt={:.4f} g_prior={:.4f}'.format(
            val_full.get('pet_gate_mean', 0.0), val_full.get('text_gate_mean', 0.0), val_full.get('prior_gate_mean', 0.0)
        )
        if val_missing is not None:
            print(
                f'Epoch {epoch}\n'
                f'train_loss={tloss / max(tn, 1):.4f} train_seg={tseg / max(tn, 1):.4f} train_boundary={tboundary / max(tn, 1):.4f}\n'
                f'val_full: Dice={val_full["dice"]:.4f} IoU={val_full["iou"]:.4f} Acc={val_full["acc"]:.4f} HD95={val_full["hd95"]:.2f}\n'
                f'val_missing: Dice={val_missing["dice"]:.4f} IoU={val_missing["iou"]:.4f} Acc={val_missing["acc"]:.4f} HD95={val_missing["hd95"]:.2f}\n'
                f'{gate_text} lr={curr_lr:.6f} grad={avg_grad_norm:.4f} best_dice={best_dice:.4f} best_hd95={best_hd95:.2f} boundary_w={getattr(config, "boundary_loss_weight", 0.0):.3f}{gamma_text}'
            )
        else:
            print(
                f'Epoch {epoch}\n'
                f'train_loss={tloss / max(tn, 1):.4f} train_seg={tseg / max(tn, 1):.4f} train_boundary={tboundary / max(tn, 1):.4f}\n'
                f'val_full: Dice={val_full["dice"]:.4f} IoU={val_full["iou"]:.4f} Acc={val_full["acc"]:.4f} HD95={val_full["hd95"]:.2f}\n'
                f'{gate_text} lr={curr_lr:.6f} grad={avg_grad_norm:.4f} best_dice={best_dice:.4f} best_hd95={best_hd95:.2f} boundary_w={getattr(config, "boundary_loss_weight", 0.0):.3f}{gamma_text}'
            )

        if getattr(config, 'vis_every_epoch', False):
            save_segmentation_diagnostics(
                task=task,
                loader=val_loader,
                out_dir=os.path.join(config.checkpoint_dir, 'vis_epochs', f'epoch_{epoch:03d}'),
                num_samples=max(1, int(getattr(config, 'vis_epoch_samples', 2))),
                threshold=getattr(config, 'eval_threshold', 0.5),
                force_missing_pet=False,
                mode='full',
            )
            if getattr(config, 'eval_missing_pet', False):
                save_segmentation_diagnostics(
                    task=task,
                    loader=val_loader,
                    out_dir=os.path.join(config.checkpoint_dir, 'vis_epochs', f'epoch_{epoch:03d}_missing'),
                    num_samples=max(1, int(getattr(config, 'vis_epoch_samples', 2))),
                    threshold=getattr(config, 'eval_threshold', 0.5),
                    force_missing_pet=True,
                    mode='missing',
                )

        if val_m['dice'] > best_dice:
            best_dice, best_dice_epoch, no_improve = val_m['dice'], epoch, 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_dice.pth.tar'), epoch)
        else:
            no_improve += 1

        if val_m['hd95'] < best_hd95:
            best_hd95, best_hd95_epoch = val_m['hd95'], epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_hd95.pth.tar'), epoch)

        if patience > 0 and no_improve >= patience:
            print('Early stop at epoch', epoch)
            break

    task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), epoch)

    def _load_checkpoint(path):
        ckpt = torch.load(path, map_location='cpu')
        for k, v in task.networks.items():
            if k in ckpt:
                task.load_model_state_dict(v, ckpt[k], strict=False)

    best_dice_path = os.path.join(config.checkpoint_dir, 'ckpt.best_dice.pth.tar')
    best_hd95_path = os.path.join(config.checkpoint_dir, 'ckpt.best_hd95.pth.tar')

    _load_checkpoint(best_dice_path)
    test_m_dice_full = task.evaluate(test_loader, tag='test_best_dice_full', force_missing_pet=False)
    test_m_dice_missing = task.evaluate(test_loader, tag='test_best_dice_missing', force_missing_pet=True)
    print('\n=== TEST(best_dice, full PET) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m_dice_full['dice'], test_m_dice_full['iou'], test_m_dice_full['acc'], test_m_dice_full['hd95']))
    print('=== TEST(best_dice, missing PET) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m_dice_missing['dice'], test_m_dice_missing['iou'], test_m_dice_missing['acc'], test_m_dice_missing['hd95']))

    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_best_dice_full'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
        force_missing_pet=False,
        mode='full',
    )
    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_best_dice_missing'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
        force_missing_pet=True,
        mode='missing',
    )

    _load_checkpoint(best_hd95_path)
    test_m_hd95_full = task.evaluate(test_loader, tag='test_best_hd95_full', force_missing_pet=False)
    test_m_hd95_missing = task.evaluate(test_loader, tag='test_best_hd95_missing', force_missing_pet=True)
    print('=== TEST(best_hd95, full PET) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m_hd95_full['dice'], test_m_hd95_full['iou'], test_m_hd95_full['acc'], test_m_hd95_full['hd95']))
    print('=== TEST(best_hd95, missing PET) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m_hd95_missing['dice'], test_m_hd95_missing['iou'], test_m_hd95_missing['acc'], test_m_hd95_missing['hd95']))

    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_best_hd95_full'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
        force_missing_pet=False,
        mode='full',
    )
    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_best_hd95_missing'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
        force_missing_pet=True,
        mode='missing',
    )


if __name__ == '__main__':
    main()
