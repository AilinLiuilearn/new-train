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
            pet_drop_prob=getattr(config, 'train_pet_drop_prob', None) if getattr(config, 'train_pet_drop_prob', None) is not None else getattr(config, 'pet_drop_prob', 0.4),
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

    print(f'GPU={g0}')
    print(f'model_arch={config.model_arch}')
    print(f'single_modality={getattr(config, "single_modality", False)}')
    print(f'ct_backbone={getattr(config, "ct_backbone", None)}')
    print(f'pet_backbone={getattr(config, "pet_backbone", None)}')
    print(f'fusion_type={getattr(config, "fusion_type", None)}')
    print(f'use_meddino={getattr(config, "use_meddino", False)}')
    print(f'use_lapa={getattr(config, "use_lapa", False)}')
    print(f'use_text_proxy={getattr(config, "use_text_proxy", False)}')
    print(f'text_in_full_mode={getattr(config, "text_in_full_mode", False)}')
    print(f'aug_mode={getattr(config, "aug_mode", None)}')
    print(f'norm_mode={getattr(config, "norm_mode", None)}')
    print(f'image_size_2d={config.image_size_2d}')
    print(f'batch_size={config.batch_size}')
    print(f'accumulation_steps={config.accumulation_steps}')
    print(f'mixed_precision={config.mixed_precision}')
    print(f'train_pet_drop_prob={getattr(config, "train_pet_drop_prob", None)}')
    print(f'lr={config.learning_rate} wd={config.weight_decay} boundary_w={getattr(config, "boundary_loss_weight", 0.0)}')
    if getattr(config, 'use_meddino', False) and getattr(config, 'use_lapa', False) and getattr(config, 'mixed_precision', False):
        print('[Hint] real MedDINO + LAPA may be numerically unstable in AMP mode.')
        print('Recommended first smoke run:')
        print('  --mixed_precision false --batch_size 2 --train_pet_drop_prob 0.0')

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

    grad_clip = getattr(config, 'grad_clip', 5.0)
    clip_params = [p for net in task.networks.values() for p in net.parameters() if p.requires_grad]
    best_dice, best_dice_epoch = -1.0, 0
    best_hd95, best_hd95_epoch = float('inf'), 0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tseg, tboundary, tn = 0.0, 0.0, 0.0, 0
        grad_norm_sum, grad_norm_steps, grad_clip_count = 0.0, 0, 0
        model_for_epoch = _unwrap_model(task.networks.get('model'))
        if hasattr(model_for_epoch, 'set_epoch'):
            model_for_epoch.set_epoch(epoch)
        task.set_epoch(epoch)
        task.optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            stepped = False
            try:
                with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                    loss, _, _, loss_dict = task.train_step(batch)
                    loss = loss / accum_iter
            except RuntimeError as exc:
                if '[NaN/Inf]' in str(exc) or 'nan' in str(exc).lower() or 'inf' in str(exc).lower():
                    print(f'[NaN Guard] skip epoch={epoch} batch={i + 1}/{spe}: {exc}')
                    task.optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            if task.scaler:
                task.scaler.scale(loss).backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    if grad_clip > 0:
                        task.scaler.unscale_(task.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(clip_params, float('inf'))
                        grad_norm_sum += float(total_norm)
                        grad_norm_steps += 1
                        if float(total_norm) > float(grad_clip):
                            grad_clip_count += 1
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

        val_results = {}
        if getattr(config, 'eval_full_pet', True):
            val_results['full'] = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        if getattr(config, 'eval_fixed_missing_pet', True):
            val_results['fixed_missing'] = task.evaluate(val_loader, eval_mode='fixed_missing', tag='val_fixed_missing')
        if getattr(config, 'eval_random_missing_pet', True):
            val_results['random_missing'] = task.evaluate(
                val_loader,
                eval_mode='random_missing',
                random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
                random_seed=getattr(config, 'eval_random_missing_seed', 2026),
                tag='val_random_missing',
            )
        if not val_results:
            val_results['full'] = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        val_full = val_results.get('full') or next(iter(val_results.values()))
        val_fixed_missing = val_results.get('fixed_missing')
        val_random_missing = val_results.get('random_missing')
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
        val_lines = []
        for mode_name, metrics in (('val_full', val_full), ('val_fixed_missing', val_fixed_missing), ('val_random_missing', val_random_missing)):
            if metrics is None:
                continue
            val_lines.append(f'{mode_name}: Dice={metrics["dice"]:.4f} IoU={metrics["iou"]:.4f} Acc={metrics["acc"]:.4f} HD95={metrics["hd95"]:.2f}')
        print(
            f'Epoch {epoch}\n'
            f'train_loss={tloss / max(tn, 1):.4f} train_seg={tseg / max(tn, 1):.4f} train_boundary={tboundary / max(tn, 1):.4f}\n'
            + '\n'.join(val_lines) + '\n'
            f'{gate_text} train_pet_drop_prob={getattr(config, "train_pet_drop_prob", 0.0):.3f} eval_random_pet_drop_prob={getattr(config, "eval_random_pet_drop_prob", 0.0):.3f} '
            f'lr={curr_lr:.6f} grad_norm={avg_grad_norm:.4f} grad_clipped_steps={grad_clip_count}/{grad_norm_steps} best_dice={best_dice:.4f} best_hd95={best_hd95:.2f} boundary_w={getattr(config, "boundary_loss_weight", 0.0):.3f}{gamma_text}'
        )

        if getattr(config, 'vis_every_epoch', False):
            for eval_mode in val_results.keys():
                save_segmentation_diagnostics(
                    task=task,
                    loader=val_loader,
                    out_dir=os.path.join(config.checkpoint_dir, f'vis_val_{eval_mode}', f'epoch_{epoch:03d}'),
                    num_samples=max(1, int(getattr(config, 'vis_epoch_samples', 2))),
                    threshold=getattr(config, 'eval_threshold', 0.5),
                    eval_mode=eval_mode,
                    random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
                    random_seed=getattr(config, 'eval_random_missing_seed', 2026),
                    mode=eval_mode,
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
    test_results = {}
    if getattr(config, 'eval_full_pet', True):
        test_results['full'] = task.evaluate(test_loader, eval_mode='full', tag='test_best_dice_full')
    if getattr(config, 'eval_fixed_missing_pet', True):
        test_results['fixed_missing'] = task.evaluate(test_loader, eval_mode='fixed_missing', tag='test_best_dice_fixed_missing')
    if getattr(config, 'eval_random_missing_pet', True):
        test_results['random_missing'] = task.evaluate(
            test_loader,
            eval_mode='random_missing',
            random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
            random_seed=getattr(config, 'eval_random_missing_seed', 2026),
            tag='test_best_dice_random_missing',
        )
    for mode_name, metrics in test_results.items():
        print(f'=== TEST(best_dice, {mode_name}) Dice={metrics["dice"]:.4f} IoU={metrics["iou"]:.4f} Acc={metrics["acc"]:.4f} HD95={metrics["hd95"]:.2f} ===')
    for mode_name in test_results.keys():
        save_segmentation_diagnostics(
            task=task,
            loader=test_loader,
            out_dir=os.path.join(config.checkpoint_dir, f'vis_test_{mode_name}'),
            num_samples=min(8, config.batch_size),
            threshold=getattr(config, 'eval_threshold', 0.5),
            eval_mode=mode_name,
            random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
            random_seed=getattr(config, 'eval_random_missing_seed', 2026),
            mode=mode_name,
        )

    _load_checkpoint(best_hd95_path)
    test_results = {}
    if getattr(config, 'eval_full_pet', True):
        test_results['full'] = task.evaluate(test_loader, eval_mode='full', tag='test_best_hd95_full')
    if getattr(config, 'eval_fixed_missing_pet', True):
        test_results['fixed_missing'] = task.evaluate(test_loader, eval_mode='fixed_missing', tag='test_best_hd95_fixed_missing')
    if getattr(config, 'eval_random_missing_pet', True):
        test_results['random_missing'] = task.evaluate(
            test_loader,
            eval_mode='random_missing',
            random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
            random_seed=getattr(config, 'eval_random_missing_seed', 2026),
            tag='test_best_hd95_random_missing',
        )
    for mode_name, metrics in test_results.items():
        print(f'=== TEST(best_hd95, {mode_name}) Dice={metrics["dice"]:.4f} IoU={metrics["iou"]:.4f} Acc={metrics["acc"]:.4f} HD95={metrics["hd95"]:.2f} ===')
    for mode_name in test_results.keys():
        save_segmentation_diagnostics(
            task=task,
            loader=test_loader,
            out_dir=os.path.join(config.checkpoint_dir, f'vis_test_{mode_name}'),
            num_samples=min(8, config.batch_size),
            threshold=getattr(config, 'eval_threshold', 0.5),
            eval_mode=mode_name,
            random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
            random_seed=getattr(config, 'eval_random_missing_seed', 2026),
            mode=mode_name,
        )


if __name__ == '__main__':
    main()
