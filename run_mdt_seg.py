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
from tasks.mdt_seg import MDTSegTeacher, _use_deep_supervision
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
    if getattr(config, 'use_aligned_loader', False):
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
            pet_drop_prob=getattr(config, 'train_pet_drop_prob', 0.0),
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


def _resolve_checkpoint_selection(checkpoint_select, val_full, val_fixed_missing, force_full_only=False):
    checkpoint_select = str(checkpoint_select)
    full_select_score = float(val_full['dice'])
    missing_select_score = float(val_fixed_missing['dice'])
    joint_select_score = 0.5 * full_select_score + 0.5 * missing_select_score
    if force_full_only:
        return full_select_score, 'full_dice', 'ckpt.best_full_dice.pth.tar'
    if checkpoint_select == 'full_dice':
        return full_select_score, 'full_dice', 'ckpt.best_full_dice.pth.tar'
    if checkpoint_select == 'missing_dice':
        return missing_select_score, 'missing_dice', 'ckpt.best_missing_dice.pth.tar'
    if checkpoint_select == 'joint_dice':
        return joint_select_score, 'joint_dice', 'ckpt.best_joint_dice.pth.tar'
    raise ValueError(f'Unsupported checkpoint_select={checkpoint_select!r}. Expected one of full_dice, missing_dice, joint_dice.')


def _deep_supervision_log_headers(config):
    if not _use_deep_supervision(config):
        return []
    return [
        'use_deep_supervision',
        'loss_main',
        'loss_aux_d2',
        'loss_aux_d3',
        'loss_aux_d4',
        'ds_weight_main',
        'ds_weight_d2',
        'ds_weight_d3',
        'ds_weight_d4',
    ]


def _pg_mtr_log_headers(model, prefix):
    headers = []
    stage_numbers = getattr(getattr(model, 'pg_mtr', None), 'active_stage_numbers', ())
    for stage in stage_numbers:
        if prefix == 'val_full':
            headers.extend([
                f'{prefix}_pg_mtr_s{stage}_ct_route_entropy',
                f'{prefix}_pg_mtr_s{stage}_ct_route_peak',
                f'{prefix}_pg_mtr_s{stage}_pet_route_entropy',
                f'{prefix}_pg_mtr_s{stage}_pet_route_peak',
                f'{prefix}_pg_mtr_s{stage}_token_key_cosine_offdiag',
                f'{prefix}_pg_mtr_s{stage}_token_value_cosine_offdiag',
                f'{prefix}_pg_mtr_s{stage}_pet_memory_rms',
                f'{prefix}_pg_mtr_s{stage}_route_loss',
                f'{prefix}_pg_mtr_s{stage}_mem_loss',
            ])
        else:
            headers.extend([
                f'{prefix}_pg_mtr_s{stage}_ct_route_entropy',
                f'{prefix}_pg_mtr_s{stage}_ct_route_peak',
                f'{prefix}_pg_mtr_s{stage}_token_key_cosine_offdiag',
                f'{prefix}_pg_mtr_s{stage}_token_value_cosine_offdiag',
                f'{prefix}_pg_mtr_s{stage}_retrieved_memory_rms',
                f'{prefix}_pg_mtr_s{stage}_ct_rms',
                f'{prefix}_pg_mtr_s{stage}_injection_rms',
                f'{prefix}_pg_mtr_s{stage}_injection_ct_ratio',
            ])
    return headers


_DEEP_SUPERVISION_LOG_KEYS = [
    'use_deep_supervision',
    'loss_main',
    'loss_aux_d2',
    'loss_aux_d3',
    'loss_aux_d4',
    'ds_weight_main',
    'ds_weight_d2',
    'ds_weight_d3',
    'ds_weight_d4',
]


def _collect_deep_supervision_metrics(loss_dict):
    metrics = {}
    for key in _DEEP_SUPERVISION_LOG_KEYS:
        val = loss_dict.get(key)
        if torch.is_tensor(val):
            metrics[key] = float(val.detach().cpu())
        elif val is not None:
            metrics[key] = float(val)
    return metrics


def _adc_gamma_headers(config):
    if not getattr(config, 'use_adc_mac', False):
        return []
    return [f'{branch}_s{stage}_gamma' for branch in ('enc_ct', 'enc_pet') for stage in range(1, 5)]


def _grad_finite_and_norm(trainable_params):
    all_finite = True
    sq_sum = 0.0
    for p in trainable_params:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        if not torch.isfinite(grad).all():
            all_finite = False
            break
        sq_sum += float(torch.sum(grad.float() ** 2).detach().cpu())
    return all_finite, math.sqrt(sq_sum)


def _module_grad_status(model):
    module_names = ('enc_ct', 'enc_pet', 'pet_proj', 'decoder', 'boundary_head')
    status = {}
    first_bad = None
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            status[name] = 'missing'
            continue
        has_grad = False
        finite = True
        max_abs = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            has_grad = True
            grad = p.grad.detach()
            if not torch.isfinite(grad).all():
                finite = False
                break
            max_abs = max(max_abs, float(grad.float().abs().max().detach().cpu()))
        if not has_grad:
            status[name] = 'no_grad'
        elif finite:
            status[name] = f'finite|max={max_abs:.4g}'
        else:
            status[name] = 'non_finite'
            if first_bad is None:
                first_bad = name
    return first_bad, status


def _curriculum_value(epoch, enabled, start, final, warmup_epochs):
    if not enabled:
        return float(final)
    warmup_epochs = max(1, int(warmup_epochs))
    if epoch >= warmup_epochs + 1:
        return float(final)
    progress = max(0.0, min(1.0, float(epoch - 1) / float(warmup_epochs)))
    return float(start) + (float(final) - float(start)) * progress


def _set_train_pet_drop_prob(train_loader, prob):
    dataset = getattr(train_loader, 'dataset', None)
    if hasattr(dataset, 'set_pet_drop_prob'):
        dataset.set_pet_drop_prob(prob)
    elif hasattr(dataset, 'pet_drop_prob'):
        dataset.pet_drop_prob = float(prob)


def _print_nan_diagnostics(exc, task, batch, epoch, batch_idx, spe, lr, grad_norm=None):
    print(f'[NaN Guard] module_hit={exc} epoch={epoch} batch={batch_idx + 1}/{spe} lr={lr:.8g} grad_norm={grad_norm if grad_norm is not None else "NA"}')
    if 'ct_feats[0]' in str(exc):
        ct = batch.get('ct')
        if torch.is_tensor(ct):
            ctf = ct.detach().float()
            print(f'[NaN Guard] input CT stats: min={float(ctf.min()):.6g} max={float(ctf.max()):.6g} mean={float(ctf.mean()):.6g}')
        model = _unwrap_model(task.networks.get('model'))
        enc_ct = getattr(model, 'enc_ct', None)
        first_param = next(enc_ct.parameters(), None) if enc_ct is not None else None
        if first_param is not None:
            print(f'[NaN Guard] enc_ct first param finite={bool(torch.isfinite(first_param.detach()).all())}')


def main():
    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = SegMDTConfig.parse_arguments()
    config.task = 'MDT_Teacher'
    if getattr(config, 'model_arch', '') == 'pet_mrp_gsa':
        config.eval_full_pet = True
        config.eval_fixed_missing_pet = False
        config.eval_random_missing_pet = False
        print('[pet_mrp_gsa] validation restricted to full PET only')
    g0 = _prepare_env(config)

    print(f'GPU={g0}')
    print(f'model_arch={config.model_arch}')
    print(f'single_modality={getattr(config, "single_modality", False)}')
    print(f'ct_backbone={getattr(config, "ct_backbone", None)}')
    print(f'pet_backbone={getattr(config, "pet_backbone", None)}')
    print(f'fusion_type={getattr(config, "fusion_type", None)}')
    if getattr(config, 'fusion_type', '') == 'hybrid_concat_dmome':
        print(f'hybrid_concat_stages={getattr(config, "hybrid_concat_stages", None)}')
        print(f'hybrid_dmome_stages={getattr(config, "hybrid_dmome_stages", None)}')
    if getattr(config, 'fusion_type', '') == 'dmome_channel_prior_gate' or getattr(config, 'use_channel_prior_gate', False):
        print(f'prior_gate_stages={getattr(config, "prior_gate_stages", None)}')
    if getattr(config, 'model_arch', '') == 'pet_mrp_gsa':
        print(f'use_pet_mrp_gsa={getattr(config, "use_pet_mrp_gsa", True)}')
        print(f'pet_mrp_stages={getattr(config, "pet_mrp_stages", "all")}')
        print(f'pet_mrp_prior_mode={getattr(config, "pet_mrp_prior_mode", "minmax")}')
    if getattr(config, 'model_arch', '') == 'mafd_net':
        print(f'freq_method={getattr(config, "freq_method", "fft")}')
        print(f'use_pet_proxy={getattr(config, "use_pet_proxy", True)}')
        print(f'proxy_loss_weight={getattr(config, "proxy_loss_weight", 0.05)}')
    print(f'use_deep_supervision={_use_deep_supervision(config)}')
    print(f'use_fprm={getattr(config, "use_fprm", False)}')
    print(f'fprm_slots={getattr(config, "fprm_slots", 32)} fprm_dim={getattr(config, "fprm_dim", 0)} fprm_beta={getattr(config, "fprm_beta", 0.1)} fprm_gamma={getattr(config, "fprm_gamma", 0.1)}')
    print(f'fprm_mem_loss_weight={getattr(config, "fprm_mem_loss_weight", 0.05)} fprm_use_memory={getattr(config, "fprm_use_memory", True)} fprm_use_shape={getattr(config, "fprm_use_shape", True)}')
    print(f'train_pet_drop_prob={getattr(config, "train_pet_drop_prob", 0.0)}')
    if config.model_arch in {'pet_contribution_ct_only', 'pet_contribution_full'}:
        config.eval_full_pet = True
        config.eval_fixed_missing_pet = False
        config.eval_random_missing_pet = False
        if getattr(config, 'mixed_precision', False):
            print('[pet_contribution] mixed_precision is disabled automatically for stability')
            config.mixed_precision = False
    print(f'eval_full_pet={getattr(config, "eval_full_pet", True)} eval_fixed_missing_pet={getattr(config, "eval_fixed_missing_pet", False)} eval_random_missing_pet={getattr(config, "eval_random_missing_pet", False)}')
    print(f'aug_mode={getattr(config, "aug_mode", None)}')
    print(f'norm_mode={getattr(config, "norm_mode", None)}')
    print(f'image_size_2d={config.image_size_2d}')
    print(f'batch_size={config.batch_size}')
    print(f'accumulation_steps={config.accumulation_steps}')
    print(f'mixed_precision={config.mixed_precision}')
    print(f'lr={config.learning_rate}')
    print(f'grad_clip={getattr(config, "grad_clip", 5.0)}')
    print(f'wd={config.weight_decay} boundary_w={getattr(config, "boundary_loss_weight", 0.0)}')
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
    ds_headers = _deep_supervision_log_headers(config)
    model_for_logging = task._unwrap(task.networks['model'])
    pg_headers = _pg_mtr_log_headers(model_for_logging, 'val_full') + _pg_mtr_log_headers(model_for_logging, 'val_missing') + ['val_full_dice', 'val_missing_dice', 'val_joint_dice', 'selected_score', 'best_selected_score', 'best_selected_epoch']
    init_train_log(log_path, extra_headers=gamma_headers + ds_headers + pg_headers)

    grad_clip = getattr(config, 'grad_clip', 5.0)
    clip_params = task.trainable_parameters()
    checkpoint_select = getattr(config, 'checkpoint_select', 'joint_dice')
    best_full_dice, best_full_dice_epoch = -1.0, 0
    best_missing_dice, best_missing_dice_epoch = -1.0, 0
    best_joint_dice, best_joint_dice_epoch = -1.0, 0
    best_selected_score, best_selected_epoch = -float('inf'), 0
    selected_checkpoint_filename = 'ckpt.best.pth.tar'
    best_full_hd95, best_full_hd95_epoch = float('inf'), 0
    best_missing_hd95, best_missing_hd95_epoch = float('inf'), 0
    best_joint_hd95, best_joint_hd95_epoch = float('inf'), 0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)

    for epoch in range(1, config.epochs + 1):
        tloss, tseg, tboundary, tn = 0.0, 0.0, 0.0, 0
        ds_metric_sum = {k: 0.0 for k in _deep_supervision_log_headers(config)}
        ds_metric_steps = 0
        grad_norm_sum, grad_norm_steps, grad_clip_count = 0.0, 0, 0
        first_bad_module_counter = {k: 0 for k in ('enc_ct', 'enc_pet', 'pet_proj', 'decoder', 'boundary_head')}
        train_pet_drop_prob = float(getattr(config, 'train_pet_drop_prob', 0.0))
        _set_train_pet_drop_prob(train_loader, train_pet_drop_prob)
        model_for_epoch = _unwrap_model(task.networks.get('model'))
        if hasattr(model_for_epoch, 'set_epoch'):
            model_for_epoch.set_epoch(epoch)
        task.set_epoch(epoch)
        task.optimizer.zero_grad()
        print(f'[Epoch {epoch}] train PET dropout prob={train_pet_drop_prob:.3f}')

        full_train_steps = 0
        missing_train_steps = 0
        full_loss_sum = 0.0
        missing_loss_sum = 0.0
        for i, batch in enumerate(train_loader):
            stepped = False
            global_batch_step = (epoch - 1) * spe + i
            pet_contribution_archs = {'pet_contribution_ct_only', 'pet_contribution_full'}
            if config.model_arch in pet_contribution_archs:
                train_route = 'full'
            else:
                train_route = 'full' if global_batch_step % 2 == 0 else 'missing'
            try:
                use_amp = bool(config.mixed_precision) and train_route == 'full'
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, _, _, loss_dict = task.train_step(batch, forward_mode=train_route)
                    loss = loss / accum_iter
            except RuntimeError as exc:
                if '[NaN/Inf]' in str(exc) or 'nan' in str(exc).lower() or 'inf' in str(exc).lower():
                    _print_nan_diagnostics(exc, task, batch, epoch, i, spe, task.optimizer.param_groups[0]['lr'])
                    task.optimizer.zero_grad(set_to_none=True)
                    continue
                raise

            if task.scaler:
                task.scaler.scale(loss).backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    task.scaler.unscale_(task.optimizer)
                    all_finite, total_norm = _grad_finite_and_norm(clip_params)
                    if not all_finite:
                        first_bad_module, module_status = _module_grad_status(model_for_epoch)
                        if first_bad_module in first_bad_module_counter:
                            first_bad_module_counter[first_bad_module] += 1
                        print(f'[Grad Guard] epoch={epoch} batch={i + 1}/{spe}')
                        print(f'[Grad Guard] first bad module: {first_bad_module or "unknown"}')
                        print(f'[Grad Guard] module_grad_status={module_status}')
                        task.optimizer.zero_grad(set_to_none=True)
                        task.scaler.update()
                        continue
                    grad_norm_sum += float(total_norm)
                    grad_norm_steps += 1
                    if grad_clip > 0:
                        if total_norm > grad_clip:
                            grad_clip_count += 1
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                    task.optimizer.zero_grad(set_to_none=True)
                    stepped = True
            else:
                loss.backward()
                if (i + 1) % accum_iter == 0 or (i + 1) == spe:
                    all_finite, total_norm = _grad_finite_and_norm(clip_params)
                    if not all_finite:
                        first_bad_module, module_status = _module_grad_status(model_for_epoch)
                        if first_bad_module in first_bad_module_counter:
                            first_bad_module_counter[first_bad_module] += 1
                        print(f'[Grad Guard] epoch={epoch} batch={i + 1}/{spe}')
                        print(f'[Grad Guard] first bad module: {first_bad_module or "unknown"}')
                        print(f'[Grad Guard] module_grad_status={module_status}')
                        task.optimizer.zero_grad(set_to_none=True)
                        continue
                    grad_norm_sum += float(total_norm)
                    grad_norm_steps += 1
                    if grad_clip > 0:
                        if total_norm > grad_clip:
                            grad_clip_count += 1
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                    task.optimizer.step()
                    task.optimizer.zero_grad(set_to_none=True)
                    stepped = True

            if task.scheduler and stepped:
                task.scheduler.step()

            step_loss = loss.item() * accum_iter
            step_seg = float(loss_dict.get('loss_seg', torch.tensor(0.0)).item())
            step_boundary = float(loss_dict.get('loss_boundary', torch.tensor(0.0)).item())
            curr_lr = task.optimizer.param_groups[0]['lr']
            if (not torch.isfinite(torch.tensor(step_loss))) or (not torch.isfinite(torch.tensor(step_seg))) or (not torch.isfinite(torch.tensor(step_boundary))):
                _print_nan_diagnostics(RuntimeError('non-finite loss stats'), task, batch, epoch, i, spe, curr_lr, grad_norm=grad_norm_sum / max(grad_norm_steps, 1) if grad_norm_steps > 0 else None)
                task.optimizer.zero_grad(set_to_none=True)
                continue
            tloss += step_loss
            tseg += step_seg
            tboundary += step_boundary
            tn += 1
            if train_route == 'full':
                full_train_steps += 1
                full_loss_sum += float(loss_dict.get('loss_total', loss.detach()).item())
            else:
                missing_train_steps += 1
                missing_loss_sum += float(loss_dict.get('loss_total', loss.detach()).item())
            if ds_metric_sum:
                batch_ds = _collect_deep_supervision_metrics(loss_dict)
                for key in ds_metric_sum:
                    if key in batch_ds:
                        ds_metric_sum[key] += batch_ds[key]
                ds_metric_steps += 1
            if (i + 1) % 50 == 0:
                curr_lr = task.optimizer.param_groups[0]['lr']
                print(
                    f'  Ep{epoch}[{i + 1}/{spe}] route={train_route} '
                    f'loss={step_loss:.4f} '
                    f'seg={step_seg:.4f} '
                    f'boundary={step_boundary:.4f} '
                    f'total={float(loss_dict.get("loss_total", loss.detach()).item()):.4f} '
                    f'lr={curr_lr:.6f}'
                )

        val_results = {}
        if getattr(config, 'eval_full_pet', True):
            val_results['full'] = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        if getattr(config, 'eval_fixed_missing_pet', False) and config.model_arch not in {'pet_contribution_ct_only', 'pet_contribution_full'}:
            val_results['fixed_missing'] = task.evaluate(val_loader, eval_mode='fixed_missing', tag='val_fixed_missing')
        if getattr(config, 'eval_random_missing_pet', False) and config.model_arch not in {'pet_contribution_ct_only', 'pet_contribution_full'}:
            val_results['random_missing'] = task.evaluate(
                val_loader,
                eval_mode='random_missing',
                random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
                random_seed=getattr(config, 'eval_random_seed', 2026),
                tag='val_random_missing',
            )
        if not val_results:
            val_results['full'] = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        val_full = val_results.get('full', next(iter(val_results.values())))
        val_fixed_missing = val_results.get('fixed_missing', val_full) if config.model_arch not in {'pet_contribution_ct_only', 'pet_contribution_full'} else val_full
        joint_dice = float(val_full['dice'])
        joint_hd95 = float(val_full['hd95'])
        force_full_only = config.model_arch in {'pet_contribution_ct_only', 'pet_contribution_full'}
        selected_score, selected_name, selected_checkpoint_filename = _resolve_checkpoint_selection(checkpoint_select, val_full, val_fixed_missing, force_full_only=force_full_only)
        val_m = val_full
        def _collect_pg_metrics(metrics, prefix):
            collected = {}
            for key, value in metrics.items():
                if not key.startswith('pg_mtr_'):
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 0:
                        continue
                    scalar = float(value.detach().float().mean().cpu())
                else:
                    try:
                        scalar = float(value)
                    except (TypeError, ValueError):
                        continue
                if math.isfinite(scalar):
                    collected[f'{prefix}_{key}'] = scalar
            return collected
        pg_full_metrics = _collect_pg_metrics(val_full, 'val_full')
        pg_missing_metrics = _collect_pg_metrics(val_fixed_missing, 'val_missing')
        avg_grad_norm = grad_norm_sum / max(grad_norm_steps, 1)
        curr_lr = task.optimizer.param_groups[0]['lr']
        gamma_metrics = _collect_adc_gamma(task)
        ds_metrics = {}
        if ds_metric_sum and ds_metric_steps > 0:
            ds_metrics = {k: v / ds_metric_steps for k, v in ds_metric_sum.items()}
        append_epoch_log(
            log_path,
            epoch,
            tloss / max(tn, 1),
            val_m,
            lr=curr_lr,
            grad_norm=avg_grad_norm,
            extra_metrics={
                **gamma_metrics,
                **ds_metrics,
                **pg_full_metrics,
                **pg_missing_metrics,
                'val_full_dice': float(val_full['dice']),
                'val_missing_dice': float(val_fixed_missing['dice']),
                'val_joint_dice': float(joint_dice),
                'selected_score': float(selected_score),
                'best_selected_score': float(best_selected_score if best_selected_epoch > 0 else selected_score),
                'best_selected_epoch': float(best_selected_epoch),
            },
        )
        gamma_text = ''
        if gamma_metrics:
            gamma_text = ' ' + ' '.join(f'{k}={v:.4f}' for k, v in gamma_metrics.items())
        val_lines = []
        for mode_name, metrics in val_results.items():
            val_lines.append(
                f'val_{mode_name}: Dice={metrics["dice"]:.4f} IoU={metrics["iou"]:.4f} Acc={metrics["acc"]:.4f} HD95={metrics["hd95"]:.2f}'
            )
        print(
            f'Epoch {epoch}\n'
            f'train_loss={tloss / max(tn, 1):.4f} train_seg={tseg / max(tn, 1):.4f} train_boundary={tboundary / max(tn, 1):.4f}\n'
            + '\n'.join(val_lines) + '\n'
            f'joint_metrics dice={joint_dice:.4f} hd95={joint_hd95:.2f}\n'
            f'checkpoint_select={checkpoint_select} selected_score={selected_score:.4f} best_selected_score={best_selected_score:.4f} best_selected_epoch={best_selected_epoch}\n'
            f'route_stats full_steps={full_train_steps} missing_steps={missing_train_steps} '
            f'avg_full_loss={full_loss_sum / max(full_train_steps, 1):.4f} '
            f'avg_missing_loss={missing_loss_sum / max(missing_train_steps, 1):.4f}\n'
            f'full PET-CT training/evaluation '
            f'lr={curr_lr:.6f} grad_norm={avg_grad_norm:.4f} grad_clipped_steps={grad_clip_count}/{grad_norm_steps} '
            f'grad_guard_skipped_steps={sum(first_bad_module_counter.values())} first_bad_module_counter={first_bad_module_counter} '
            f'best_full_dice={best_full_dice:.4f} best_missing_dice={best_missing_dice:.4f} best_joint_dice={best_joint_dice:.4f} '
            f'best_full_hd95={best_full_hd95:.2f} best_missing_hd95={best_missing_hd95:.2f} best_joint_hd95={best_joint_hd95:.2f} '
            f'boundary_w={getattr(config, "boundary_loss_weight", 0.0):.3f}{gamma_text}'
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
                    random_pet_drop_prob=0.0,
                    random_seed=2026,
                    mode=eval_mode,
                )

        if val_full['dice'] > best_full_dice:
            best_full_dice, best_full_dice_epoch = val_full['dice'], epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_full_dice.pth.tar'), epoch)
        if val_fixed_missing['dice'] > best_missing_dice:
            best_missing_dice, best_missing_dice_epoch = val_fixed_missing['dice'], epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_missing_dice.pth.tar'), epoch)
        if joint_dice > best_joint_dice:
            best_joint_dice, best_joint_dice_epoch = joint_dice, epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_joint_dice.pth.tar'), epoch)
        if selected_score > best_selected_score:
            best_selected_score, best_selected_epoch, no_improve = selected_score, epoch, 0
            task.save_checkpoint(os.path.join(config.checkpoint_dir, selected_checkpoint_filename), epoch)
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar'), epoch)
        else:
            no_improve += 1
        if val_full['hd95'] < best_full_hd95:
            best_full_hd95, best_full_hd95_epoch = val_full['hd95'], epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_full_hd95.pth.tar'), epoch)
        if val_fixed_missing['hd95'] < best_missing_hd95:
            best_missing_hd95, best_missing_hd95_epoch = val_fixed_missing['hd95'], epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_missing_hd95.pth.tar'), epoch)
        if joint_hd95 < best_joint_hd95:
            best_joint_hd95, best_joint_hd95_epoch = joint_hd95, epoch
            task.save_checkpoint(os.path.join(config.checkpoint_dir, 'ckpt.best_joint_hd95.pth.tar'), epoch)
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

    def _evaluate_enabled_modes(loader, prefix):
        results = {}
        if getattr(config, 'eval_full_pet', True):
            results['full'] = task.evaluate(loader, eval_mode='full', tag=f'{prefix}_full')
        if getattr(config, 'eval_fixed_missing_pet', False) and config.model_arch not in {'pet_contribution_ct_only', 'pet_contribution_full'}:
            results['fixed_missing'] = task.evaluate(loader, eval_mode='fixed_missing', tag=f'{prefix}_fixed_missing')
        if getattr(config, 'eval_random_missing_pet', False) and config.model_arch not in {'pet_contribution_ct_only', 'pet_contribution_full'}:
            results['random_missing'] = task.evaluate(
                loader,
                eval_mode='random_missing',
                random_pet_drop_prob=getattr(config, 'eval_random_pet_drop_prob', 0.4),
                random_seed=getattr(config, 'eval_random_seed', 2026),
                tag=f'{prefix}_random_missing',
            )
        if not results:
            results['full'] = task.evaluate(loader, eval_mode='full', tag=f'{prefix}_full')
        return results

    best_dice_path = os.path.join(config.checkpoint_dir, 'ckpt.best.pth.tar')
    best_hd95_path = os.path.join(config.checkpoint_dir, 'ckpt.best_joint_hd95.pth.tar')

    print(f'checkpoint_select={checkpoint_select}')
    print(f'selected_checkpoint={os.path.basename(best_dice_path)}')
    print(f'best_selected_score={best_selected_score:.4f}')
    print(f'best_selected_epoch={best_selected_epoch}')

    _load_checkpoint(best_dice_path)
    test_results = _evaluate_enabled_modes(test_loader, 'test_best_dice')
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
            random_pet_drop_prob=0.0,
            random_seed=2026,
            mode=mode_name,
        )

    _load_checkpoint(best_hd95_path)
    test_results = _evaluate_enabled_modes(test_loader, 'test_best_hd95')
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
            random_pet_drop_prob=0.0,
            random_seed=2026,
            mode=mode_name,
        )


if __name__ == '__main__':
    main()
