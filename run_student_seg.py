# -*- coding: utf-8 -*-
"""Student baseline training entry — single-modality CT-only segmentation.

Usage examples:
    # ConvNeXt-Pico + AttentionUNet decoder (teacher-identical decoder)
    python run_student_seg.py --student_backbone convnext_pico --student_decoder_type attention

    # ConvNeXt-Atto + lightweight decoder
    python run_student_seg.py --student_backbone convnext_atto --student_decoder_type light

    # ConvNeXt-Nano + AttentionUNet (same scale as teacher encoder)
    python run_student_seg.py --student_backbone convnext_nano --student_decoder_type attention
"""

import importlib.util
import json
import math
import os
import random
import sys

import numpy as np
import torch

from configs.seg_student import SegStudentConfig
from models.build_student_seg import build_student_seg
from tasks.student_seg import StudentSegTask
from utils.model_profile import print_student_profile
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log
from utils.vis_student import save_student_diagnostics

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def _load_dataset_module():
    root = os.getcwd()
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_env(config):
    g0 = int(config.gpus[0]) if config.gpus else 0
    config.gpus = [g0]
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
    torch.use_deterministic_algorithms(True)
    return g0


def _build_loaders(config):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'cipa_aligned', False):
        return dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
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
    )


def _save_config(config):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = SegStudentConfig.parse_arguments()
    config.task = 'MDT_Student'
    g0 = _prepare_env(config)

    backbone = getattr(config, 'student_backbone', 'convnext_pico')
    dec_type = getattr(config, 'student_decoder_type', 'attention')

    if not getattr(config, 'no_pretrained', False) and not getattr(config, 'student_pretrained_path', None):
        auto_path = os.path.join('.', 'pretrained', backbone)
        if os.path.isdir(auto_path):
            config.student_pretrained_path = auto_path
            print(f'[+] Auto-resolved pretrained: {auto_path}')
    if getattr(config, 'no_pretrained', False):
        config.student_pretrained_path = None
        print('[!] No pretrained encoder weights (training from scratch)')

    print(f'GPU={g0}  backbone={backbone}  decoder={dec_type}')
    print(f'lr={config.learning_rate}  wd={config.weight_decay}  bs={config.batch_size}')

    _save_config(config)
    train_loader, val_loader, test_loader = _build_loaders(config)

    networks = build_student_seg(config)
    print('\n' + '=' * 30 + ' STUDENT MODEL PROFILE ' + '=' * 30)
    print_student_profile(networks, config, tag=f'student_{backbone}_{dec_type}')
    print('=' * 83 + '\n')

    task = StudentSegTask(networks, config)
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
    init_train_log(log_path)

    grad_clip = getattr(config, 'grad_clip', 0.5)
    clip_params = [p for net in task.networks.values() for p in net.parameters()]
    best_dice, best_dice_epoch = -1.0, 0
    best_hd95, best_hd95_epoch = float('inf'), 0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tn = 0.0, 0
        task.set_epoch(epoch)
        task.optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            stepped = False
            with torch.amp.autocast('cuda', enabled=config.mixed_precision):
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
                    task.update_ema()
                    stepped = True
            else:
                loss.backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.optimizer.step()
                    task.optimizer.zero_grad()
                    task.update_ema()
                    stepped = True

            if task.scheduler and stepped:
                task.scheduler.step()

            tloss += loss.item() * accum_iter
            tn += 1
            if (i + 1) % 50 == 0:
                curr_lr = task.optimizer.param_groups[0]['lr']
                print(
                    f'  Ep{epoch}[{i + 1}/{spe}] '
                    f'loss={loss.item() * accum_iter:.4f} '
                    f'seg={loss_dict["loss_seg"].item():.4f} '
                    f'lr={curr_lr:.6f}'
                )

        val_m = task.evaluate(val_loader)
        append_epoch_log(log_path, epoch, tloss / max(tn, 1), val_m)
        print('Epoch {} loss={:.4f} Dice={:.4f} IoU={:.4f} HD95={:.2f}'.format(
            epoch, tloss / max(tn, 1), val_m['dice'], val_m['iou'], val_m['hd95']))

        if getattr(config, 'vis_every_epoch', False):
            save_student_diagnostics(
                task=task,
                loader=val_loader,
                out_dir=os.path.join(config.checkpoint_dir, 'vis_epochs', f'epoch_{epoch:03d}'),
                num_samples=max(1, int(getattr(config, 'vis_epoch_samples', 2))),
                threshold=getattr(config, 'eval_threshold', 0.5),
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
                v.load_state_dict(ckpt[k], strict=False)
        if task.use_ema and 'ema_model' in ckpt:
            task.ema_model.load_state_dict(ckpt['ema_model'], strict=False)

    best_dice_path = os.path.join(config.checkpoint_dir, 'ckpt.best_dice.pth.tar')
    best_hd95_path = os.path.join(config.checkpoint_dir, 'ckpt.best_hd95.pth.tar')

    _load_checkpoint(best_dice_path)
    test_m_dice = task.evaluate(test_loader)
    print('\n=== TEST(best_dice) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m_dice['dice'], test_m_dice['iou'], test_m_dice['acc'], test_m_dice['hd95']))

    save_student_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_best_dice'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
    )

    _load_checkpoint(best_hd95_path)
    test_m_hd95 = task.evaluate(test_loader)
    print('=== TEST(best_hd95) Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m_hd95['dice'], test_m_hd95['iou'], test_m_hd95['acc'], test_m_hd95['hd95']))

    save_student_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_best_hd95'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
    )


if __name__ == '__main__':
    main()
