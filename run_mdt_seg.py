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
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    return get_pclt20k_loaders_cipa_aligned(
        cfg.root,
        cfg.image_size_2d,
        cfg.batch_size,
        cfg.num_workers,
        cfg.random_state,
        cfg.pin_memory,
        cfg.aug_mode,
        cfg.norm_mode,
        cfg.train_split_file,
        cfg.val_split_file,
        cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
    )


def _assert_baseline(cfg):
    assert cfg.accumulation_steps == 1
    assert float(cfg.train_pet_drop_prob) == 0.0
    assert float(cfg.missing_loss_weight) == 1.0
    assert float(cfg.joint_full_weight) == 0.5
    assert float(cfg.joint_missing_weight) == 0.5
    assert bool(cfg.use_deep_supervision) is False
    assert bool(cfg.deep_supervision) is False
    assert float(cfg.boundary_loss_weight) == 0.0


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
    return float(total.sqrt().item()) if total is not None else 0.0


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_joint': os.path.join(checkpoint_dir, 'ckpt.best_joint.pth.tar'),
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'best_missing': os.path.join(checkpoint_dir, 'ckpt.best_missing.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def _load_state_dict_with_report(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    print(f'[RESUME] checkpoint={checkpoint_path}', flush=True)
    print(f'[RESUME] missing_keys={missing_keys}', flush=True)
    print(f'[RESUME] unexpected_keys={unexpected_keys}', flush=True)
    return checkpoint


def _sync_cppi_config_from_stage1(cfg, checkpoint_path):
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), 'config_args.json')
    if not os.path.isfile(cfg_path):
        print(f'[STAGE2 LOAD][WARN] Stage-1 config not found: {cfg_path}', flush=True)
        return
    with open(cfg_path, 'r') as f:
        stage1_cfg = json.load(f)
    for key in ('cppi_num_clusters', 'cppi_build_stage', 'ct_backbone', 'pet_backbone', 'decoder_channels'):
        if key in stage1_cfg:
            setattr(cfg, key, stage1_cfg[key])
            print(f'[STAGE2 LOAD] sync {key}={stage1_cfg[key]} from Stage-1 config', flush=True)


def _load_stage1_for_taskmoe(model, checkpoint_path):
    from models.dual_shared_add_baseline import _is_stage2_new_param_name

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    missing_taskmoe = [k for k in missing_keys if _is_stage2_new_param_name(k)]
    missing_other = [k for k in missing_keys if not _is_stage2_new_param_name(k)]
    if missing_other:
        raise RuntimeError(
            'Stage-1 load has non-TaskMoE missing keys: '
            f'{missing_other}'
        )
    if unexpected_keys:
        raise RuntimeError(
            f'Stage-1 load has unexpected keys: {unexpected_keys}'
        )
    if not bool(model.prototype_memory.bank_ready):
        raise RuntimeError(
            'Stage-1 checkpoint CPPI bank is not ready; Stage-2 requires a mature bank'
        )

    print('[STAGE2 LOAD]', flush=True)
    print(f'checkpoint={checkpoint_path}', flush=True)
    print(f'taskmoe_mode={getattr(model, "taskmoe_mode", "independent")}', flush=True)
    print(f'taskmoe_scales={getattr(model, "taskmoe_scales", ())}', flush=True)
    print(f'missing_taskmoe_keys={missing_taskmoe}', flush=True)
    print(f'unexpected_keys={unexpected_keys}', flush=True)
    print(f'cppi_bank_ready={bool(model.prototype_memory.bank_ready)}', flush=True)
    return checkpoint


@torch.no_grad()
def _verify_stage2_zero_step(task, val_loader):
    model = task.model
    mode = getattr(model, 'taskmoe_mode', 'independent')
    residual_mode = getattr(model, 'taskmoe_residual_mode', 'zero_start')
    if mode == 'cross_scale_shared':
        residual_mode = getattr(model.cross_scale_taskmoe, 'residual_mode', residual_mode)
    elif mode == 'state_scale_factorized':
        residual_mode = 'zero_start'

    batch = next(iter(val_loader))
    ct = batch['ct'][:1].to(task.device, non_blocking=True)
    pet = batch['pet'][:1].to(task.device, non_blocking=True)
    was_training = model.training
    model.eval()

    # paper residual: step-0 is NOT Stage-1 identity; run finite/shape checks only.
    if mode == 'cross_scale_shared' and residual_mode == 'paper':
        model.taskmoe_enabled = True
        route_ok = {}
        for route in ('full', 'missing'):
            out = model(ct, pet=pet, forward_mode=route, mask=None)
            logits = out['logits']
            moe_loss = out.get('aux', {}).get('taskmoe_balance_loss')
            moe_stats = out.get('aux', {}).get('taskmoe_stats', {}) or {}
            if not torch.isfinite(logits).all():
                model.train(was_training)
                raise RuntimeError(f'paper residual: non-finite logits on {route}')
            if moe_loss is None or not torch.isfinite(moe_loss):
                model.train(was_training)
                raise RuntimeError(f'paper residual: non-finite balance loss on {route}')
            for s in (1, 2, 3, 4):
                key = f's{s}_delta_feat_ratio'
                if key not in moe_stats or not torch.isfinite(moe_stats[key]):
                    model.train(was_training)
                    raise RuntimeError(f'paper residual: bad residual stats {key} on {route}')
            route_ok[route] = True
        model.train(was_training)
        print('[INITIAL RESIDUAL CHECK]', flush=True)
        print('residual_mode=paper', flush=True)
        print(f'full_forward_finite={route_ok["full"]}', flush=True)
        print(f'missing_forward_finite={route_ok["missing"]}', flush=True)
        print('passed=True', flush=True)
        return {'full': None, 'missing': None, 'residual_mode': 'paper'}

    if mode == 'cross_scale_shared':
        if model.cross_scale_taskmoe.beta is None:
            raise RuntimeError('zero_start shared TaskMoE requires learnable beta')
        beta = model.cross_scale_taskmoe.beta.detach().float()
        if float(beta.abs().max().item()) > 1e-12:
            raise RuntimeError(
                f'TaskMoE shared beta must be ~0 at step-0, got {beta.tolist()}'
            )
        beta_vals = [float(beta[i].item()) for i in range(4)]
    elif mode == 'state_scale_factorized':
        moe = model.state_scale_taskmoe
        if moe is None:
            raise RuntimeError('state_scale_factorized requires state_scale_taskmoe')
        beta = moe.effective_beta().detach().float()
        if float(beta.abs().max().item()) > 1e-12:
            raise RuntimeError(
                f'TaskMoE factorized effective beta must be ~0 at step-0, got {beta.tolist()}'
            )
        if float(moe.raw_beta.detach().abs().max().item()) > 1e-12:
            raise RuntimeError('TaskMoE factorized raw_beta must be ~0 at step-0')
        if model.stage2_decoder_adapter is not None:
            # Adapter residual must be identity at init (zero-init up conv + FiLM).
            dummy = torch.zeros(1, 64, 8, 8, device=task.device)
            delta = model.stage2_decoder_adapter(dummy, None)
            if float(delta.abs().max().item()) > 1e-12:
                raise RuntimeError('stage2_decoder_adapter must be zero at step-0')
        beta_vals = [float(beta[i].item()) for i in range(4)]
    else:
        for scale_name, _, module in model._iter_active_taskmoe():
            b = float(module.residual_scale.detach().abs().item())
            if b > 1e-12:
                raise RuntimeError(
                    f'TaskMoE {scale_name} residual_scale beta must be ~0 at step-0, got {b}'
                )
        beta_vals = [0.0, 0.0, 0.0, 0.0]
        for scale_name, idx, module in model._iter_active_taskmoe():
            beta_vals[idx] = float(module.residual_scale.detach().item())

    diffs = {}
    for route in ('full', 'missing'):
        model.taskmoe_enabled = False
        out_stage1 = model(ct, pet=pet, forward_mode=route, mask=None)
        logits_stage1 = out_stage1['logits']

        model.taskmoe_enabled = True
        out_stage2 = model(ct, pet=pet, forward_mode=route, mask=None)
        logits_stage2 = out_stage2['logits']

        max_abs_diff = float((logits_stage1 - logits_stage2).abs().max().item())
        diffs[route] = max_abs_diff
        if max_abs_diff > 1e-6:
            model.taskmoe_enabled = True
            model.train(was_training)
            raise RuntimeError(
                f'Zero-step equivalence failed for {route}: '
                f'max_abs_diff={max_abs_diff} > 1e-6'
            )

    model.taskmoe_enabled = True
    model.train(was_training)
    print('[ZERO-STEP]', flush=True)
    print(f'mode={mode}', flush=True)
    print(f'residual_mode={residual_mode}', flush=True)
    print(f'beta_s1={beta_vals[0]}', flush=True)
    print(f'beta_s2={beta_vals[1]}', flush=True)
    print(f'beta_s3={beta_vals[2]}', flush=True)
    print(f'beta_s4={beta_vals[3]}', flush=True)
    print(f'full_max_abs_diff={diffs["full"]}', flush=True)
    print(f'missing_max_abs_diff={diffs["missing"]}', flush=True)
    print('passed=True', flush=True)
    return diffs


def _shared_taskmoe_beta_log_values(model):
    """CSV beta columns for shared / factorized TaskMoE.

    zero_start: real learnable (or effective bounded) beta.
    paper: display 1.0 meaning F + 1*DeltaF (NOT a learnable parameter).
    """
    mode = getattr(model, 'taskmoe_mode', 'independent')
    if mode == 'state_scale_factorized':
        moe = getattr(model, 'state_scale_taskmoe', None)
        if moe is None:
            return None
        beta = moe.effective_beta().detach().float()
        return (
            float(beta[3].item()),
            float(beta[0].item()),
            float(beta[1].item()),
            float(beta[2].item()),
            float(beta[3].item()),
        )
    moe = getattr(model, 'cross_scale_taskmoe', None)
    if moe is None:
        return None
    if getattr(moe, 'residual_mode', 'zero_start') == 'paper' or moe.beta is None:
        # Display-only identity coefficient for F_base + DeltaF logging.
        return (1.0, 1.0, 1.0, 1.0, 1.0)  # beta, s1, s2, s3, s4
    beta = moe.beta.detach().float()
    return (
        float(beta[3].item()),
        float(beta[0].item()),
        float(beta[1].item()),
        float(beta[2].item()),
        float(beta[3].item()),
    )


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _optimizer_group_lr(optimizer, name, default=0.0):
    for group in optimizer.param_groups:
        if group.get('name') == name:
            return float(group['lr'])
    if name == 'taskmoe' and optimizer.param_groups:
        # Legacy single-group Stage2 MoE-only optimizer, or factorized alias.
        for group in optimizer.param_groups:
            if group.get('name') in ('factorized_taskmoe', 'taskmoe', None):
                return float(group['lr'])
        return float(optimizer.param_groups[0]['lr'])
    if name == 'decoder_adapter':
        for group in optimizer.param_groups:
            if group.get('name') == 'decoder_adapter':
                return float(group['lr'])
    return float(default)


def _print_stage2_startup(model, cfg, total_params, trainable_params, optimizer=None):
    from models.dual_shared_add_baseline import (
        _is_stage2_adapter_param_name,
        _is_taskmoe_param_name,
    )

    taskmoe_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and _is_taskmoe_param_name(n)
    )
    adapter_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and _is_stage2_adapter_param_name(n)
    )
    decoder_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith('decoder.')
    )
    stage1_core_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad
        and not _is_taskmoe_param_name(n)
        and not _is_stage2_adapter_param_name(n)
        and not n.startswith('decoder.')
    )
    train_decoder = bool(getattr(model, 'stage2_train_decoder', False))
    mode = getattr(model, 'taskmoe_mode', 'independent')
    print('[TRAIN MODE]', flush=True)
    print('mode=stage2_taskmoe', flush=True)
    print('[STAGE1]', flush=True)
    print(f'checkpoint={cfg.stage1_checkpoint}', flush=True)
    print('loaded=True', flush=True)
    print(f'frozen={bool(model.stage2_moe_only)}', flush=True)
    print('[CPPI]', flush=True)
    print(f'bank_ready={bool(model.prototype_memory.bank_ready)}', flush=True)
    print('frozen=True', flush=True)
    print('collect=False', flush=True)
    print('finalize=False', flush=True)
    print('retrieve=True', flush=True)
    print('[TASKMOE]', flush=True)
    if mode == 'cross_scale_shared':
        moe = model.cross_scale_taskmoe
        residual_mode = getattr(moe, 'residual_mode', 'zero_start')
        print('mode=cross_scale_shared', flush=True)
        print(f'residual_mode={residual_mode}', flush=True)
        print('scales=S1+S2+S3+S4', flush=True)
        print(f'expert_dim={moe.expert_dim}', flush=True)
        print(f'num_experts={moe.num_experts}', flush=True)
        print(f'top_k={moe.top_k}', flush=True)
        activation_ratio = float(moe.top_k) / float(moe.num_experts)
        print(f'expert_activation_ratio={activation_ratio:.4f}', flush=True)
        print('shared_expert_bank=True', flush=True)
        print('scale_specific_prompt=True', flush=True)
        print('scale_specific_router=True', flush=True)
        print(f'balance_loss_weight={moe.balance_loss_weight}', flush=True)
        if residual_mode == 'paper':
            print('learnable_beta=False', flush=True)
            print('residual_formula=F_base+DeltaF', flush=True)
            print(f'has_beta_parameter={moe.beta is not None}', flush=True)
        else:
            beta = moe.beta.detach().float()
            print(f'beta_s1={float(beta[0].item())}', flush=True)
            print(f'beta_s2={float(beta[1].item())}', flush=True)
            print(f'beta_s3={float(beta[2].item())}', flush=True)
            print(f'beta_s4={float(beta[3].item())}', flush=True)
        use_text = bool(getattr(moe, 'use_text_prior', False))
        print('[TEXT PRIOR]', flush=True)
        print(f'enabled={use_text}', flush=True)
        if use_text and getattr(moe, 'text_prior', None) is not None:
            tp = moe.text_prior
            print(f'biomedclip_model_path={tp.biomedclip_model_path}', flush=True)
            print(f'text_tower_path={tp.text_tower_path}', flush=True)
            print(f'backend={tp.backend}', flush=True)
            print(f'local_only={tp.local_only}', flush=True)
            print(f'text_encoder_trainable={tp.text_encoder_trainable}', flush=True)
            print(f'text_encoder_retained={tp.text_encoder_retained}', flush=True)
            print(f'text_embedding_dim={tp.text_embedding_dim}', flush=True)
            print(f'Full text={tp.full_text}', flush=True)
            print(f'Missing text={tp.missing_text}', flush=True)
            print(f'Full/Missing embedding cosine={tp.embedding_cosine():.6f}', flush=True)
            print('text_to_expert_shared_across_scales=True', flush=True)
            print('text_encoder_frozen=True', flush=True)
            print('text_encoder_runtime=False', flush=True)
            print('text_prior_target=expert_router', flush=True)
            print(f'num_text_expert_logits={tp.num_experts}', flush=True)
    elif mode == 'state_scale_factorized':
        moe = model.state_scale_taskmoe
        print('mode=state_scale_factorized', flush=True)
        print('residual_mode=zero_start_bounded', flush=True)
        print('scales=S1+S2+S3+S4', flush=True)
        print(f'expert_dim={moe.expert_dim}', flush=True)
        print('experts=E_shared+E_scale_s1..s4+E_state_full+E_state_missing', flush=True)
        print(f'private_rank={moe.private_rank}', flush=True)
        print(f'beta_max={moe.beta_max}', flush=True)
        beta = moe.effective_beta().detach().float()
        print(f'beta_s1={float(beta[0].item())}', flush=True)
        print(f'beta_s2={float(beta[1].item())}', flush=True)
        print(f'beta_s3={float(beta[2].item())}', flush=True)
        print(f'beta_s4={float(beta[3].item())}', flush=True)
        print(f'role_loss_weight={moe.role_loss_weight}', flush=True)
        print(f'fers_mode={moe.fers_mode}', flush=True)
        print('shared_consistency=removed', flush=True)
        print('noisy_topk=False', flush=True)
        print('balance_loss=False', flush=True)
        print('[TEXT PRIOR]', flush=True)
        print('enabled=False', flush=True)
    else:
        print('mode=independent', flush=True)
        print(f'scales={"+".join(model.taskmoe_scales).upper()}', flush=True)
        for scale_name, _, module in model._iter_active_taskmoe():
            print(
                f'{scale_name}: channels={module.channels} '
                f'num_experts={module.num_experts} top_k={module.top_k} '
                f'residual_mode={module.residual_mode} '
                f'beta_init={float(module.residual_scale.detach().item())} '
                f'balance_loss_weight={module.balance_loss_weight}',
                flush=True,
            )
        print('[TEXT PRIOR]', flush=True)
        print('enabled=False', flush=True)
    print('[DECODER ADAPTER]', flush=True)
    print(f'enabled={model.stage2_decoder_adapter is not None}', flush=True)
    if model.stage2_decoder_adapter is not None:
        print(f'level={model.stage2_decoder_adapter.level}', flush=True)
        print(f'trainable={adapter_trainable}', flush=True)
    text_to_expert_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and 'text_to_expert' in n
    )
    moe_lr = float(getattr(cfg, 'learning_rate', 0.0))
    dec_lr = float(getattr(cfg, 'decoder_lr', 0.0)) if train_decoder else 0.0
    adapter_lr = 0.0
    if optimizer is not None:
        moe_lr = _optimizer_group_lr(optimizer, 'taskmoe', default=moe_lr)
        if any(g.get('name') == 'factorized_taskmoe' for g in optimizer.param_groups):
            moe_lr = _optimizer_group_lr(optimizer, 'factorized_taskmoe', default=moe_lr)
        dec_lr = _optimizer_group_lr(optimizer, 'decoder', default=0.0) if train_decoder else 0.0
        adapter_lr = _optimizer_group_lr(optimizer, 'decoder_adapter', default=0.0)
    ratio = (dec_lr / moe_lr) if moe_lr > 0 else 0.0
    print('[STAGE2 TRAINABLE]', flush=True)
    print(f'taskmoe_trainable={taskmoe_trainable}', flush=True)
    print(f'text_to_expert_trainable={text_to_expert_trainable}', flush=True)
    print(f'decoder_adapter_trainable={adapter_trainable}', flush=True)
    print(f'decoder_enabled={train_decoder}', flush=True)
    print(f'decoder_trainable={decoder_trainable}', flush=True)
    print(f'stage1_core_trainable={stage1_core_trainable}', flush=True)
    print('[LEARNING RATE]', flush=True)
    print(f'taskmoe_lr={moe_lr}', flush=True)
    print(f'decoder_adapter_lr={adapter_lr}', flush=True)
    print(f'decoder_lr={dec_lr}', flush=True)
    print(f'decoder_lr_ratio={ratio}', flush=True)
    print('[PARAMS]', flush=True)
    # Keep legacy key for older log parsers: stage1_core only.
    print(f'stage1_trainable={stage1_core_trainable}', flush=True)
    print(f'taskmoe_trainable={taskmoe_trainable}', flush=True)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)
    print('[EXPERT ABLATION]', flush=True)
    if mode == 'cross_scale_shared' and getattr(model, 'cross_scale_taskmoe', None) is not None:
        moe = model.cross_scale_taskmoe
        print(f'num_experts={moe.num_experts}', flush=True)
        print(f'top_k={moe.top_k}', flush=True)
        print('shared_expert_bank=True', flush=True)
    elif mode == 'state_scale_factorized':
        print('num_experts=7', flush=True)
        print('top_k=N/A', flush=True)
        print('shared_expert_bank=factorized_shared_private', flush=True)
    else:
        print(f'num_experts={getattr(model, "taskmoe_num_experts", 6)}', flush=True)
        print('top_k=2', flush=True)
        print('shared_expert_bank=False', flush=True)
    print(f'taskmoe_trainable={taskmoe_trainable}', flush=True)
    print(f'total_trainable={trainable_params}', flush=True)


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    if not getattr(cfg, 'stage1_checkpoint', None):
        raise ValueError('Stage-2 TaskMoE training requires --stage1_checkpoint')
    from models.dual_shared_add_baseline import _parse_taskmoe_scales
    mode = str(getattr(cfg, 'taskmoe_mode', 'independent')).lower()
    if mode in ('cross_scale_shared', 'state_scale_factorized'):
        if _parse_taskmoe_scales(getattr(cfg, 'taskmoe_scales', 's4')) != ('s1', 's2', 's3', 's4'):
            raise ValueError(
                f'{mode} TaskMoE is an all-scale module; use --taskmoe_scales all'
            )
    if mode == 'state_scale_factorized':
        if bool(getattr(cfg, 'taskmoe_use_text_prior', False)):
            raise ValueError('state_scale_factorized forbids --taskmoe_use_text_prior True')
        if bool(getattr(cfg, 'stage2_train_decoder', False)):
            raise ValueError(
                'state_scale_factorized forbids --stage2_train_decoder True; '
                'use --stage2_decoder_adapter'
            )
        if str(getattr(cfg, 'taskmoe_residual_mode', 'zero_start')).lower() == 'paper':
            raise ValueError('state_scale_factorized forbids paper residual mode')
    if bool(getattr(cfg, 'stage2_decoder_adapter', False)) and bool(
        getattr(cfg, 'stage2_train_decoder', False)
    ):
        raise ValueError('Cannot combine --stage2_decoder_adapter with --stage2_train_decoder')
    _sync_cppi_config_from_stage1(cfg, cfg.stage1_checkpoint)

    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, _ = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)

    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    _load_stage1_for_taskmoe(model, cfg.stage1_checkpoint)
    model.enable_stage2_moe_only(
        train_decoder=bool(getattr(cfg, 'stage2_train_decoder', False))
    )
    task = MDTSegTeacher(networks, cfg)

    total_params, trainable_params = _count_parameters(task.model)
    _print_stage2_startup(
        task.model, cfg, total_params, trainable_params, optimizer=task.optimizer
    )
    _verify_stage2_zero_step(task, val_loader)

    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    extra_headers = [
        'train_full_loss', 'train_missing_loss', 'train_overall_loss',
        'full_train_batches', 'missing_train_batches',
        'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc', 'val_full_acc_pixel', 'val_full_hd95',
        'val_missing_loss', 'val_missing_dice', 'val_missing_iou', 'val_missing_acc', 'val_missing_acc_pixel', 'val_missing_hd95',
        'joint_dice', 'best_joint', 'best_joint_epoch',
        'grad_full_enc_ct', 'grad_missing_enc_ct', 'grad_full_ct_align', 'grad_missing_ct_align', 'grad_full_decoder', 'grad_missing_decoder',
        'epoch_time',
        'cppi_bank_version', 'cppi_ready_slots', 'cppi_bg_candidates', 'cppi_fg_candidates',
        'nonfinite_grad_steps', 'skipped_optimizer_steps',
        'nonfinite_full_steps', 'nonfinite_missing_steps',
            'train_moe_balance_loss', 'taskmoe_beta',
        'taskmoe_beta_s1', 'taskmoe_beta_s2', 'taskmoe_beta_s3', 'taskmoe_beta_s4',
        'taskmoe_delta_ratio_s1', 'taskmoe_delta_ratio_s2',
        'taskmoe_delta_ratio_s3', 'taskmoe_delta_ratio_s4',
        'taskmoe_delta_l2_ratio_s1', 'taskmoe_delta_l2_ratio_s2',
        'taskmoe_delta_l2_ratio_s3', 'taskmoe_delta_l2_ratio_s4',
        'train_fers_loss', 'train_fers_scale_loss', 'train_fers_state_loss',
        'scale_role_acc', 'state_role_acc',
        'lr_taskmoe', 'lr_decoder', 'lr_decoder_adapter',
    ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'), extra_headers=extra_headers)

    best_joint = -1.0
    best_full = -1.0
    best_missing = -1.0
    best_joint_epoch = 0
    global_batch_step = 0
    amp_enabled = bool(cfg.mixed_precision)
    patience = int(getattr(cfg, 'early_stop_patience', 10))
    no_improve = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        full_n = missing_n = 0
        full_loss = missing_loss = 0.0
        moe_balance_accum = 0.0
        moe_balance_steps = 0
        fers_accum = 0.0
        fers_scale_accum = 0.0
        fers_state_accum = 0.0
        scale_role_acc_accum = 0.0
        state_role_acc_accum = 0.0
        fers_steps = 0
        delta_ratio_accum = {f's{i}': 0.0 for i in range(1, 5)}
        delta_l2_ratio_accum = {f's{i}': 0.0 for i in range(1, 5)}
        delta_ratio_steps = 0
        nonfinite_grad_steps = 0
        skipped_optimizer_steps = 0
        nonfinite_full_steps = 0
        nonfinite_missing_steps = 0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        grads = {
            'full': {'enc_ct': [], 'ct_align': [], 'decoder': []},
            'missing': {'enc_ct': [], 'ct_align': [], 'decoder': []},
        }
        epoch_start = time.time()
        fixed_diag_batch = None
        diag_stats = {}

        for batch_idx, batch in enumerate(train_loader):
            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, outputs, stats = task.train_step(batch, forward_mode=route)
            if not torch.isfinite(loss):
                raise RuntimeError('loss became non-finite')

            moe_stats = (outputs.get('aux', {}) or {}).get('taskmoe_stats', {}) or {}
            if moe_stats:
                for i in range(1, 5):
                    rkey = f's{i}_delta_feat_ratio'
                    lkey = f's{i}_delta_feat_l2_ratio'
                    if rkey in moe_stats:
                        delta_ratio_accum[f's{i}'] += float(moe_stats[rkey].detach())
                    if lkey in moe_stats:
                        delta_l2_ratio_accum[f's{i}'] += float(moe_stats[lkey].detach())
                delta_ratio_steps += 1

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            has_nonfinite_grad = False
            for name, p in task.model.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    has_nonfinite_grad = True
                    print(
                        '[NONFINITE GRAD]',
                        epoch,
                        batch_idx,
                        route,
                        name,
                        'nan=',
                        int(torch.isnan(p.grad).sum().item()),
                        'inf=',
                        int(torch.isinf(p.grad).sum().item()),
                        flush=True,
                    )

            if has_nonfinite_grad:
                nonfinite_grad_steps += 1
                skipped_optimizer_steps += 1
                if route == 'full':
                    nonfinite_full_steps += 1
                else:
                    nonfinite_missing_steps += 1
                task.optimizer.zero_grad(set_to_none=True)
                if task.scaler.is_enabled():
                    task.scaler.update()
                print(
                    '[SKIP NONFINITE STEP]\n'
                    f'epoch={epoch}\n'
                    f'batch={batch_idx}\n'
                    f'route={route}',
                    flush=True,
                )
            else:
                grads[route]['enc_ct'].append(module_grad_norm(task.model.enc_ct))
                grads[route]['ct_align'].append(module_grad_norm(task.model.ct_align))
                grads[route]['decoder'].append(module_grad_norm(task.model.decoder))
                total_grad_norm = torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip)) if float(cfg.grad_clip) > 0 else 0.0
                grad_norm_accum += float(total_grad_norm)
                grad_norm_steps += 1

                if task.scaler.is_enabled():
                    task.scaler.step(task.optimizer)
                    task.scaler.update()
                else:
                    task.optimizer.step()

                task.scheduler.step()

            if (batch_idx + 1) % 100 == 0:
                print(f'[BATCH {batch_idx + 1}] route={route} loss={float(loss.detach()):.6f}', flush=True)

            moe_balance_accum += float(stats['loss_moe_balance'].detach())
            moe_balance_steps += 1
            if 'loss_fers' in stats:
                fers_accum += float(stats['loss_fers'].detach())
                fers_scale_accum += float(stats['loss_fers_scale'].detach())
                fers_state_accum += float(stats['loss_fers_state'].detach())
                scale_role_acc_accum += float(stats['scale_role_acc'].detach())
                state_role_acc_accum += float(stats['state_role_acc'].detach())
                fers_steps += 1
            if route == 'full':
                full_n += 1
                full_loss += float(loss.detach())
            else:
                missing_n += 1
                missing_loss += float(loss.detach())

            global_batch_step += 1
            task.global_batch_step = global_batch_step
            if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is None:
                fixed_diag_batch = {
                    'ct': batch['ct'][:1].detach().cpu(),
                    'pet': batch['pet'][:1].detach().cpu(),
                    'mask': batch['mask'][:1].detach().cpu(),
                }

        if getattr(task.model, 'stage2_moe_only', False):
            cppi_report = {
                'bank_version_after': int(task.model.prototype_memory.bank_version.item()),
                'ready_count': int(task.model.prototype_memory.prototype_ready.sum().item()),
                'classes': {
                    'background': {'num_candidates': 0},
                    'foreground': {'num_candidates': 0},
                },
            }
            print(
                '[CPPI STAGE2]\n'
                'frozen=True\n'
                f'bank_ready={bool(task.model.prototype_memory.bank_ready)}\n'
                'collect=False\n'
                'finalize=False\n'
                'retrieve=True',
                flush=True,
            )
        else:
            cppi_report = task.model.finalize_cppi_epoch(
                epoch=epoch,
                save_json=True,
                save_visualizations=(
                    epoch == 1
                    or epoch % 5 == 0
                    or epoch == cfg.epochs
                ),
                print_info=True,
            )
            print(
                f"[CPPI EPOCH {epoch}]\n"
                f"bank_version={cppi_report.get('bank_version_after', cppi_report.get('bank_version_before', 0))}\n"
                f"ready_slots={cppi_report.get('ready_count', cppi_report.get('ready_slots', 0))}\n"
                f"bg_candidates={cppi_report.get('classes', {}).get('background', {}).get('num_candidates', 0)}\n"
                f"fg_candidates={cppi_report.get('classes', {}).get('foreground', {}).get('num_candidates', 0)}",
                flush=True,
            )
        if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is not None and epoch % int(cfg.gradient_diagnostics_interval) == 0:
            diag_stats = task.gradient_diagnostics(fixed_diag_batch, max_samples=min(1, int(cfg.gradient_diagnostics_num_samples))) or {}

        val_full = task.evaluate(val_loader, eval_mode='full', tag='val_full')
        val_missing = task.evaluate(val_loader, eval_mode='fixed_missing', tag='val_missing')
        joint_dice = float(cfg.joint_full_weight) * val_full['dice'] + float(cfg.joint_missing_weight) * val_missing['dice']

        joint_improved = joint_dice > best_joint
        full_improved = val_full['dice'] > best_full
        missing_improved = val_missing['dice'] > best_missing
        if joint_improved:
            best_joint = joint_dice
            best_joint_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if full_improved:
            best_full = val_full['dice']
        if missing_improved:
            best_missing = val_missing['dice']

        if joint_improved:
            task.save_checkpoint(paths['best_joint'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        if full_improved:
            task.save_checkpoint(paths['best_full'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        if missing_improved:
            task.save_checkpoint(paths['best_missing'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)
        task.save_checkpoint(paths['last'], epoch, best_joint, best_full, best_missing, best_joint_epoch, val_full, val_missing, joint_dice)

        train_loss = (full_loss + missing_loss) / max(1, full_n + missing_n)
        val_loss = 0.5 * val_full['total_loss'] + 0.5 * val_missing['total_loss']
        val_dice = joint_dice
        val_iou = 0.5 * val_full['iou'] + 0.5 * val_missing['iou']
        val_acc = 0.5 * val_full['acc'] + 0.5 * val_missing['acc']
        val_acc_pixel = 0.5 * val_full.get('acc_pixel', 0.0) + 0.5 * val_missing.get('acc_pixel', 0.0)
        val_hd95 = 0.5 * val_full['hd95'] + 0.5 * val_missing['hd95']
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        extra_metrics = {
            'train_full_loss': full_loss / max(1, full_n),
            'train_missing_loss': missing_loss / max(1, missing_n),
            'train_overall_loss': train_loss,
            'full_train_batches': full_n,
            'missing_train_batches': missing_n,
            'val_full_loss': val_full['total_loss'],
            'val_full_dice': val_full['dice'],
            'val_full_iou': val_full['iou'],
            'val_full_acc': val_full['acc'],
            'val_full_acc_pixel': val_full.get('acc_pixel', 0.0),
            'val_full_hd95': val_full['hd95'],
            'val_missing_loss': val_missing['total_loss'],
            'val_missing_dice': val_missing['dice'],
            'val_missing_iou': val_missing['iou'],
            'val_missing_acc': val_missing['acc'],
            'val_missing_acc_pixel': val_missing.get('acc_pixel', 0.0),
            'val_missing_hd95': val_missing['hd95'],
            'joint_dice': joint_dice,
            'best_joint': best_joint,
            'best_joint_epoch': best_joint_epoch,
            'grad_full_enc_ct': float(np.mean(grads['full']['enc_ct'])) if grads['full']['enc_ct'] else 0.0,
            'grad_missing_enc_ct': float(np.mean(grads['missing']['enc_ct'])) if grads['missing']['enc_ct'] else 0.0,
            'grad_full_ct_align': float(np.mean(grads['full']['ct_align'])) if grads['full']['ct_align'] else 0.0,
            'grad_missing_ct_align': float(np.mean(grads['missing']['ct_align'])) if grads['missing']['ct_align'] else 0.0,
            'grad_full_decoder': float(np.mean(grads['full']['decoder'])) if grads['full']['decoder'] else 0.0,
            'grad_missing_decoder': float(np.mean(grads['missing']['decoder'])) if grads['missing']['decoder'] else 0.0,
            'epoch_time': time.time() - epoch_start,
            'cppi_bank_version': int(cppi_report.get('bank_version_after', cppi_report.get('bank_version_before', 0))),
            'cppi_ready_slots': int(cppi_report.get('ready_count', cppi_report.get('ready_slots', 0))),
            'cppi_bg_candidates': int(cppi_report.get('classes', {}).get('background', {}).get('num_candidates', 0)),
            'cppi_fg_candidates': int(cppi_report.get('classes', {}).get('foreground', {}).get('num_candidates', 0)),
            'nonfinite_grad_steps': nonfinite_grad_steps,
            'skipped_optimizer_steps': skipped_optimizer_steps,
            'nonfinite_full_steps': nonfinite_full_steps,
            'nonfinite_missing_steps': nonfinite_missing_steps,
            'train_moe_balance_loss': moe_balance_accum / max(1, moe_balance_steps),
            'taskmoe_beta': (
                _shared_taskmoe_beta_log_values(task.model)[0]
                if getattr(task.model, 'taskmoe_mode', 'independent') in (
                    'cross_scale_shared', 'state_scale_factorized'
                )
                and _shared_taskmoe_beta_log_values(task.model) is not None
                else (
                    float(task.model.taskmoe_s4.residual_scale.detach().item())
                    if getattr(task.model, 'taskmoe_s4', None) is not None else 0.0
                )
            ),
            'taskmoe_beta_s1': (
                _shared_taskmoe_beta_log_values(task.model)[1]
                if getattr(task.model, 'taskmoe_mode', 'independent') in (
                    'cross_scale_shared', 'state_scale_factorized'
                )
                and _shared_taskmoe_beta_log_values(task.model) is not None
                else (
                    float(task.model.taskmoe_s1.residual_scale.detach().item())
                    if getattr(task.model, 'taskmoe_s1', None) is not None else 0.0
                )
            ),
            'taskmoe_beta_s2': (
                _shared_taskmoe_beta_log_values(task.model)[2]
                if getattr(task.model, 'taskmoe_mode', 'independent') in (
                    'cross_scale_shared', 'state_scale_factorized'
                )
                and _shared_taskmoe_beta_log_values(task.model) is not None
                else (
                    float(task.model.taskmoe_s2.residual_scale.detach().item())
                    if getattr(task.model, 'taskmoe_s2', None) is not None else 0.0
                )
            ),
            'taskmoe_beta_s3': (
                _shared_taskmoe_beta_log_values(task.model)[3]
                if getattr(task.model, 'taskmoe_mode', 'independent') in (
                    'cross_scale_shared', 'state_scale_factorized'
                )
                and _shared_taskmoe_beta_log_values(task.model) is not None
                else (
                    float(task.model.taskmoe_s3.residual_scale.detach().item())
                    if getattr(task.model, 'taskmoe_s3', None) is not None else 0.0
                )
            ),
            'taskmoe_beta_s4': (
                _shared_taskmoe_beta_log_values(task.model)[4]
                if getattr(task.model, 'taskmoe_mode', 'independent') in (
                    'cross_scale_shared', 'state_scale_factorized'
                )
                and _shared_taskmoe_beta_log_values(task.model) is not None
                else (
                    float(task.model.taskmoe_s4.residual_scale.detach().item())
                    if getattr(task.model, 'taskmoe_s4', None) is not None else 0.0
                )
            ),
            'taskmoe_delta_ratio_s1': delta_ratio_accum['s1'] / max(1, delta_ratio_steps),
            'taskmoe_delta_ratio_s2': delta_ratio_accum['s2'] / max(1, delta_ratio_steps),
            'taskmoe_delta_ratio_s3': delta_ratio_accum['s3'] / max(1, delta_ratio_steps),
            'taskmoe_delta_ratio_s4': delta_ratio_accum['s4'] / max(1, delta_ratio_steps),
            'taskmoe_delta_l2_ratio_s1': delta_l2_ratio_accum['s1'] / max(1, delta_ratio_steps),
            'taskmoe_delta_l2_ratio_s2': delta_l2_ratio_accum['s2'] / max(1, delta_ratio_steps),
            'taskmoe_delta_l2_ratio_s3': delta_l2_ratio_accum['s3'] / max(1, delta_ratio_steps),
            'taskmoe_delta_l2_ratio_s4': delta_l2_ratio_accum['s4'] / max(1, delta_ratio_steps),
            'train_fers_loss': fers_accum / max(1, fers_steps),
            'train_fers_scale_loss': fers_scale_accum / max(1, fers_steps),
            'train_fers_state_loss': fers_state_accum / max(1, fers_steps),
            'scale_role_acc': scale_role_acc_accum / max(1, fers_steps),
            'state_role_acc': state_role_acc_accum / max(1, fers_steps),
            'lr_taskmoe': _optimizer_group_lr(task.optimizer, 'taskmoe'),
            'lr_decoder': (
                _optimizer_group_lr(task.optimizer, 'decoder', default=0.0)
                if bool(getattr(task.model, 'stage2_train_decoder', False))
                else 0.0
            ),
            'lr_decoder_adapter': _optimizer_group_lr(
                task.optimizer, 'decoder_adapter', default=0.0
            ),
            **{f'diag_{k}': v for k, v in diag_stats.items()},
        }
        lr_taskmoe = extra_metrics['lr_taskmoe']
        lr_decoder = extra_metrics['lr_decoder']
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=lr_taskmoe,
            grad_norm=avg_grad_norm,
            extra_metrics=extra_metrics,
        )

        print(
            f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} '
            f'lr_taskmoe={lr_taskmoe:.8f} lr_decoder={lr_decoder:.8f}',
            flush=True,
        )
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


if __name__ == '__main__':
    main()
