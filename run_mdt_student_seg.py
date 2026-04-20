# -*- coding: utf-8 -*-
"""Student training entry for PVTv2-EMCAD baseline."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_student
from tasks.mdt_student_seg import MDTSegStudent
from utils.model_profile import print_baseline_profile
from utils.optimization import get_cosine_scheduler
from utils.train_logger import append_epoch_log, init_train_log
from utils.vis_teacher import save_segmentation_diagnostics


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
    if torch.cuda.is_available():
        torch.cuda.set_device(g0)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    torch.cuda.manual_seed_all(config.random_state)
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
            getattr(config, 'aug_strong', False),
        )
    return dataset_mod.get_pclt20k_loaders(
        config.root,
        config.image_size_2d,
        config.batch_size,
        config.num_workers,
        val_ratio=config.val_ratio,
        random_state=config.random_state,
        use_case_split=getattr(config, 'use_case_split', True),
        aug_strong=getattr(config, 'aug_strong', False),
    )


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = SegMDTConfig.parse_arguments()
    config.task = 'MDT_Student'
    g0 = _prepare_env(config)

    print(f'GPU={g0} student_backbone={config.student_backbone} model=PVTv2-b0+EMCAD (single-branch CT-only)')
    print(f'lr={config.learning_rate} wd={config.weight_decay} bs={config.batch_size}')
    print(f'student_pretrained_path={getattr(config, "student_pretrained_path", None)}')

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    train_loader, val_loader, test_loader = _build_loaders(config)
    networks = build_mdt_seg_student(config)

    print('\n' + '=' * 30 + ' MODEL PROFILE (Student) ' + '=' * 30)
    print_baseline_profile(networks, config, tag='学生 baseline')
    print('=' * 86 + '\n')

    task = MDTSegStudent(networks, config)
    spe = len(train_loader)
    accum_iter = max(1, int(getattr(config, 'accumulation_steps', 1)))
    updates_per_epoch = math.ceil(spe / accum_iter)
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        config.epochs,
        warmup_steps=config.cosine_warmup * updates_per_epoch,
        min_lr=config.cosine_min_lr,
        steps_per_epoch=updates_per_epoch,
    )

    log_path = os.path.join(config.checkpoint_dir, 'train_log.csv')
    init_train_log(log_path)
    grad_clip = getattr(config, 'grad_clip', 0.5)
    clip_params = [p for net in task.networks.values() for p in net.parameters()]
    best_dice, best_epoch, no_improve = -1.0, 0, 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tn = 0.0, 0
        for i, batch in enumerate(train_loader):
            task.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config.mixed_precision):
                loss, _, _, loss_dict = task.train_step(batch)

            stepped = False
            if task.scaler:
                task.scaler.scale(loss).backward()
                if grad_clip > 0:
                    task.scaler.unscale_(task.optimizer)
                    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                task.scaler.step(task.optimizer)
                task.scaler.update()
                stepped = True
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                task.optimizer.step()
                stepped = True

            if task.scheduler and stepped:
                task.scheduler.step()

            tloss += loss.item()
            tn += 1
            if (i + 1) % 50 == 0:
                print(f'  Ep{epoch}[{i + 1}/{spe}] loss={loss.item():.4f} seg={loss_dict["loss_seg"].item():.4f}')

        val_m = task.evaluate(val_loader)
        append_epoch_log(log_path, epoch, tloss / max(tn, 1), val_m)
        print('Epoch {} loss={:.4f} Dice={:.4f} IoU={:.4f} HD95={:.2f}'.format(
            epoch, tloss / max(tn, 1), val_m['dice'], val_m['iou'], val_m['hd95']))

        if val_m['dice'] > best_dice:
            best_dice, best_epoch, no_improve = val_m['dice'], epoch, 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print('Early stop at epoch', epoch)
            break

    task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.last.pth.tar'), epoch)
    ckpt = torch.load(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)

    test_m = task.evaluate(test_loader)
    print('\n=== TEST Dice={:.4f} IoU={:.4f} Acc={:.4f} HD95={:.2f} ==='.format(
        test_m['dice'], test_m['iou'], test_m['acc'], test_m['hd95']))
    with open(os.path.join(config.checkpoint_dir, 'test_results.json'), 'w') as f:
        json.dump({k: float(v) for k, v in test_m.items()}, f, indent=2)

    save_segmentation_diagnostics(
        task=task,
        loader=test_loader,
        out_dir=os.path.join(config.checkpoint_dir, 'vis_diagnostic'),
        num_samples=min(8, config.batch_size),
        threshold=getattr(config, 'eval_threshold', 0.5),
    )
    print('Best epoch:', best_epoch, ' Val Dice:', round(best_dice, 4))


if __name__ == '__main__':
    main()
