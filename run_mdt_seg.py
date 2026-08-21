# -*- coding: utf-8 -*-
import json
import os
import random
import time

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from models.dual_prompt_pgfa import FULL_TEXT, MISSING_TEXT
from models.dual_shared_add_baseline import DP_SCALE_NAMES
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
    if module is None:
        return 0.0
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
    return float(total.sqrt().item()) if total is not None else 0.0


def named_param_grad_norm(model, predicate):
    total = None
    for name, p in model.named_parameters():
        if not predicate(name) or p.grad is None:
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


def _is_allowed_dp_missing_key(key):
    return key.startswith('dp_pgfa_')


def _load_stage1_for_dp(model, checkpoint_path):
    if not checkpoint_path:
        raise RuntimeError('stage1_checkpoint is required when dp_pgfa_enabled=True')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'stage1_checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    allowed_missing = [k for k in missing_keys if _is_allowed_dp_missing_key(k)]
    disallowed_missing = [k for k in missing_keys if not _is_allowed_dp_missing_key(k)]
    if disallowed_missing:
        raise RuntimeError(
            'Stage1 checkpoint is missing required Stage1 keys: '
            + ', '.join(disallowed_missing[:40])
        )
    if unexpected_keys:
        raise RuntimeError(
            'Stage1 checkpoint has unexplained unexpected keys: '
            + ', '.join(unexpected_keys[:40])
        )

    required_prefixes = (
        'enc_ct.',
        'enc_pet.',
        'ct_align.',
        'pet_calibration.',
        'fusion.',
        'decoder.',
        'prototype_memory.',
    )
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    for prefix in required_prefixes:
        present = [k for k in model_keys if k.startswith(prefix)]
        if not present:
            raise RuntimeError(f'Stage1 restore check failed: model has no keys for {prefix}')
        restored = [k for k in present if k in ckpt_keys]
        if not restored:
            raise RuntimeError(f'Stage1 restore check failed: checkpoint has no keys for {prefix}')

    if not hasattr(model.fusion, 'raw_alpha_full') or not hasattr(model.fusion, 'raw_alpha_missing'):
        raise RuntimeError('Stage1 fusion alphas were not restored')
    if 'fusion.raw_alpha_full' not in ckpt_keys or 'fusion.raw_alpha_missing' not in ckpt_keys:
        raise RuntimeError('Stage1 checkpoint missing fusion.raw_alpha_full/raw_alpha_missing')

    cppi = model.cppi_status_snapshot()
    alpha_full = model.fusion.alpha_full.detach().float().cpu().tolist()
    alpha_missing = model.fusion.alpha_missing.detach().float().cpu().tolist()
    print('[STAGE1 LOAD]', flush=True)
    print(f'checkpoint={checkpoint_path}', flush=True)
    print('success=True', flush=True)
    print(f'allowed_missing_dp_keys={allowed_missing}', flush=True)
    print(f'unexpected_keys={unexpected_keys}', flush=True)
    print(f'fusion_alpha_full={alpha_full}', flush=True)
    print(f'fusion_alpha_missing={alpha_missing}', flush=True)
    print(f'cppi_bank_ready={cppi["bank_ready"]}', flush=True)
    print(f'cppi_bank_version={cppi["bank_version"]}', flush=True)
    print(f'cppi_ready_slots={cppi["ready_slots"]}', flush=True)
    if not cppi['bank_ready']:
        raise RuntimeError('Stage1 CPPI bank is not ready after checkpoint load')
    return {
        'checkpoint': checkpoint_path,
        'allowed_missing': allowed_missing,
        'unexpected_keys': unexpected_keys,
        'alpha_full': alpha_full,
        'alpha_missing': alpha_missing,
        'cppi': cppi,
    }


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _count_group_trainable(model, prefixes):
    total = 0
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes) and p.requires_grad:
            total += p.numel()
    return total


def _print_stage2_startup(model, cfg, stage1_info):
    scales = list(model.dp_pgfa_scales)
    print('[STAGE2 DP-PGFA]', flush=True)
    print(f'enabled={bool(model.dp_pgfa_enabled)}', flush=True)
    print(f'scales={scales}', flush=True)
    print('stage1_frozen=True', flush=True)
    print('decoder_frozen=True', flush=True)
    print('cppi_mode=retrieve_only', flush=True)
    print('missing_real_pet_encoder=False', flush=True)

    print('[DP DESIGN]', flush=True)
    print(f'window_size={cfg.dp_window_size}', flush=True)
    print(f'depth={cfg.dp_depth}', flush=True)
    print(f'prompt_len={cfg.dp_prompt_len}', flush=True)
    print(f'compress_ratio={cfg.dp_compress_ratio}', flush=True)
    print(f'task_prompt={bool(cfg.dp_use_task_prompt)}', flush=True)
    print(f'text_prompt={bool(cfg.dp_use_text_prompt)}', flush=True)
    print('zero_init_output=True', flush=True)
    print('outer_beta=False', flush=True)
    print('moe=False', flush=True)

    print('[TEXT]', flush=True)
    print(f'full_text={FULL_TEXT}', flush=True)
    print(f'missing_text={MISSING_TEXT}', flush=True)
    print(f'text_tower={cfg.dp_text_tower_path}', flush=True)
    print(f'biomedclip_root={cfg.dp_biomedclip_model_path}', flush=True)
    print('local_files_only=True', flush=True)
    print('text_encoder_retained=False', flush=True)
    print(f'text_embedding_dim={model.dp_text_embedding_dim}', flush=True)
    print(f'full_missing_text_cosine={model.dp_full_missing_text_cosine}', flush=True)

    for scale_name, adapter in model._iter_dp_adapters(active_only=True):
        w_max = float(adapter.out_proj.weight.detach().abs().max().item())
        b_max = float(adapter.out_proj.bias.detach().abs().max().item())
        if w_max != 0.0 or b_max != 0.0:
            raise RuntimeError(f'{scale_name} out_proj is not zero-initialized: w={w_max} b={b_max}')
        print(f'[ZERO INIT] {scale_name} out_proj weight_max={w_max} bias_max={b_max}', flush=True)

    total_params, trainable_params = _count_parameters(model)
    dp_trainable = _count_group_trainable(model, ('dp_pgfa_',))
    stage1_trainable = _count_group_trainable(
        model,
        ('enc_ct.', 'enc_pet.', 'ct_align.', 'pet_calibration.', 'fusion.', 'prototype_memory.'),
    )
    decoder_trainable = _count_group_trainable(model, ('decoder.',))
    print('[TRAINABLE]', flush=True)
    print(f'total_params={total_params}', flush=True)
    print(f'trainable_params={trainable_params}', flush=True)
    print(f'dp_pgfa_trainable={dp_trainable}', flush=True)
    print(f'stage1_trainable={stage1_trainable}', flush=True)
    print(f'decoder_trainable={decoder_trainable}', flush=True)
    if dp_trainable <= 0 or stage1_trainable != 0 or decoder_trainable != 0:
        raise RuntimeError('Stage2 trainable parameter safety check failed')
    print(
        f'[STAGE1 LOAD SUMMARY] alpha_full={stage1_info["alpha_full"]} '
        f'alpha_missing={stage1_info["alpha_missing"]} '
        f'cppi_bank_version={stage1_info["cppi"]["bank_version"]}',
        flush=True,
    )


def _scalarize_dp_stats(dp_stats):
    out = {}
    if not isinstance(dp_stats, dict):
        return out
    for key, value in dp_stats.items():
        if key.endswith('_prompt_weights_mean') or key.endswith('_task_atom_weights_mean'):
            continue
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = float(value.detach().float().cpu().item())
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _dp_csv_headers():
    headers = []
    for scale in DP_SCALE_NAMES:
        headers.extend([
            f'dp_raw_residual_l2_ratio_{scale}',
            f'dp_delta_l2_ratio_{scale}',
            f'dp_prompt_entropy_{scale}',
        ])
    headers.extend([
        'grad_full_dp_pgfa', 'grad_missing_dp_pgfa',
        'grad_full_dp_out_proj', 'grad_missing_dp_out_proj',
        'grad_full_dp_core', 'grad_missing_dp_core',
        'nonfinite_grad_steps', 'skipped_optimizer_steps',
        'nonfinite_full_steps', 'nonfinite_missing_steps',
    ])
    return headers


def _active_dp_module(model):
    adapters = [adapter for _, adapter in model._iter_dp_adapters(active_only=True)]
    if not adapters:
        return None
    return torch.nn.ModuleList(adapters)


@torch.no_grad()
def _zero_step_identity_check(model, batch, device, amp_enabled):
    """Compare Stage1-only vs Stage1+DP before any optimizer step."""
    model.eval()
    ct = batch['ct'].to(device)
    pet = batch['pet'].to(device)
    was_enabled = bool(model.dp_pgfa_enabled)

    def _run(with_dp):
        model.dp_pgfa_enabled = bool(with_dp)
        with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
            out_full = model(ct, pet=pet, forward_mode='full', mask=None)
            out_missing = model(ct, pet=pet, forward_mode='missing', mask=None)
        return out_full['logits'].float(), out_missing['logits'].float()

    # Feature-level check on active scales via Stage1 fused feats + adapter.
    ct_feats = model._encode_ct(ct)
    pet_feats_real = model._encode_pet(pet)
    _, ct_reference_feats, _ = model._retrieve_cppi(
        ct_feats, return_ct_reference=True
    )
    ct_reference_feats = [x.detach() for x in ct_reference_feats]
    pet_feats_cal = model.pet_calibration(
        ct_feats, pet_feats_real, ct_reference_feats, reference_valid=True
    )
    fused_full = model.fusion(ct_feats, pet_feats_cal, mode='full')
    feature_diffs = {}
    for scale_name, adapter in model._iter_dp_adapters(active_only=True):
        idx = DP_SCALE_NAMES.index(scale_name)
        base = fused_full[idx]
        refined = adapter(base, route='full').feature
        feature_diffs[scale_name] = float((refined - base).abs().max().item())

    logits_a_full, logits_a_missing = _run(with_dp=False)
    logits_b_full, logits_b_missing = _run(with_dp=True)
    model.dp_pgfa_enabled = was_enabled

    full_diff = float((logits_a_full - logits_b_full).abs().max().item())
    missing_diff = float((logits_a_missing - logits_b_missing).abs().max().item())
    print('[ZERO STEP]', flush=True)
    for scale_name, diff in feature_diffs.items():
        print(f'{scale_name} feature diff={diff:.8e}', flush=True)
        if diff > 1e-7:
            raise RuntimeError(f'Zero-step feature identity failed at {scale_name}: {diff}')
    print(f'Full logits diff={full_diff:.8e}', flush=True)
    print(f'Missing logits diff={missing_diff:.8e}', flush=True)
    if full_diff > 1e-6 or missing_diff > 1e-6:
        raise RuntimeError(
            f'Zero-step logits identity failed: full={full_diff} missing={missing_diff}'
        )
    print('PASS=True', flush=True)
    model.train()
    return {
        'feature_diffs': feature_diffs,
        'full_logits_diff': full_diff,
        'missing_logits_diff': missing_diff,
    }


def _dp_out_proj_grad_norm(model):
    return named_param_grad_norm(model, lambda n: n.startswith('dp_pgfa_') and '.out_proj.' in n)


def _dp_core_grad_norm(model):
    return named_param_grad_norm(
        model,
        lambda n: n.startswith('dp_pgfa_') and ('.core.' in n or '.task_prompt.' in n),
    )


def _trainable_grads_finite(model):
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False, name
    return True, None


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    _seed(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, _ = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)

    networks = build_mdt_seg_teacher(cfg)
    model = networks['model']
    stage2_enabled = bool(getattr(cfg, 'dp_pgfa_enabled', False))
    stage1_info = None
    cppi_bank_version_before = None

    if stage2_enabled:
        if int(getattr(cfg, 'cppi_build_stage', 3)) != 4 or int(getattr(cfg, 'cppi_num_clusters', 6)) != 6:
            print(
                '[WARN] Stage2 DP formal experiments should use '
                '--cppi_num_clusters 6 --cppi_build_stage 4 to match Stage1 best checkpoint',
                flush=True,
            )
        stage1_info = _load_stage1_for_dp(model, getattr(cfg, 'stage1_checkpoint', None))
        model.enable_stage2_dp_only()
        cppi_bank_version_before = int(stage1_info['cppi']['bank_version'])
        _print_stage2_startup(model, cfg, stage1_info)

    task = MDTSegTeacher(networks, cfg)
    if getattr(cfg, 'resume_checkpoint', None):
        if stage2_enabled:
            print('[WARN] resume_checkpoint is for Stage2 continuation; stage1_checkpoint already applied', flush=True)
        _load_state_dict_with_report(task.model, cfg.resume_checkpoint)
        if stage2_enabled:
            task.model.enable_stage2_dp_only()

    total_params, trainable_params = _count_parameters(task.model)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)

    if stage2_enabled:
        first_batch = next(iter(train_loader))
        _zero_step_identity_check(
            task.model,
            first_batch,
            task.device,
            amp_enabled=bool(cfg.mixed_precision),
        )

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
    ]
    if stage2_enabled:
        extra_headers.extend(_dp_csv_headers())
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
    nonfinite_grad_steps = 0
    skipped_optimizer_steps = 0
    nonfinite_full_steps = 0
    nonfinite_missing_steps = 0

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        full_n = missing_n = 0
        full_loss = missing_loss = 0.0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        grads = {
            'full': {
                'enc_ct': [], 'ct_align': [], 'decoder': [],
                'dp_pgfa': [], 'dp_out_proj': [], 'dp_core': [],
            },
            'missing': {
                'enc_ct': [], 'ct_align': [], 'decoder': [],
                'dp_pgfa': [], 'dp_out_proj': [], 'dp_core': [],
            },
        }
        dp_stat_accum = {h: [] for h in _dp_csv_headers() if h.startswith('dp_')}
        epoch_start = time.time()
        fixed_diag_batch = None
        diag_stats = {}

        for batch_idx, batch in enumerate(train_loader):
            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, outputs, _ = task.train_step(batch, forward_mode=route)

            if not torch.isfinite(loss):
                nonfinite_grad_steps += 1
                skipped_optimizer_steps += 1
                if route == 'full':
                    nonfinite_full_steps += 1
                else:
                    nonfinite_missing_steps += 1
                print(
                    f'[NONFINITE LOSS] epoch={epoch} batch={batch_idx} route={route} loss={loss}',
                    flush=True,
                )
                global_batch_step += 1
                task.global_batch_step = global_batch_step
                continue

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            grads_ok, bad_name = _trainable_grads_finite(task.model)
            if not grads_ok:
                nonfinite_grad_steps += 1
                skipped_optimizer_steps += 1
                if route == 'full':
                    nonfinite_full_steps += 1
                else:
                    nonfinite_missing_steps += 1
                dp_stats = {}
                if isinstance(outputs, dict):
                    dp_stats = _scalarize_dp_stats((outputs.get('aux') or {}).get('dp_pgfa_stats') or {})
                print(
                    f'[NONFINITE GRAD] epoch={epoch} batch={batch_idx} route={route} '
                    f'bad_param={bad_name} loss={float(loss.detach())} dp_stats={dp_stats}',
                    flush=True,
                )
                task.optimizer.zero_grad(set_to_none=True)
                if task.scaler.is_enabled():
                    task.scaler.update()
                global_batch_step += 1
                task.global_batch_step = global_batch_step
                continue

            grads[route]['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads[route]['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads[route]['decoder'].append(module_grad_norm(task.model.decoder))
            if stage2_enabled:
                grads[route]['dp_pgfa'].append(module_grad_norm(_active_dp_module(task.model)))
                grads[route]['dp_out_proj'].append(_dp_out_proj_grad_norm(task.model))
                grads[route]['dp_core'].append(_dp_core_grad_norm(task.model))
                dp_stats = {}
                if isinstance(outputs, dict):
                    dp_stats = _scalarize_dp_stats((outputs.get('aux') or {}).get('dp_pgfa_stats') or {})
                for scale in DP_SCALE_NAMES:
                    raw_key = f'dp_{scale}_raw_residual_l2_ratio'
                    delta_key = f'dp_{scale}_delta_l2_ratio'
                    ent_key = f'dp_{scale}_prompt_weights_entropy'
                    dp_stat_accum[f'dp_raw_residual_l2_ratio_{scale}'].append(dp_stats.get(raw_key, 0.0))
                    dp_stat_accum[f'dp_delta_l2_ratio_{scale}'].append(dp_stats.get(delta_key, 0.0))
                    dp_stat_accum[f'dp_prompt_entropy_{scale}'].append(dp_stats.get(ent_key, 0.0))

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

        if stage2_enabled:
            status = task.model.cppi_status_snapshot()
            if cppi_bank_version_before is not None and int(status['bank_version']) != int(cppi_bank_version_before):
                raise RuntimeError(
                    f'Stage2 CPPI bank_version drifted: before={cppi_bank_version_before} after={status["bank_version"]}'
                )
            cppi_report = {
                'bank_version_before': status['bank_version'],
                'bank_version_after': status['bank_version'],
                'ready_count': status['ready_slots'],
                'ready_slots': status['ready_slots'],
                'classes': {
                    'background': {'num_candidates': 0},
                    'foreground': {'num_candidates': 0},
                },
            }
            print(
                f"[CPPI EPOCH {epoch}]\n"
                f"bank_version={status['bank_version']} (retrieve_only)\n"
                f"ready_slots={status['ready_slots']}/{status['total_slots']}\n"
                f"bg_candidates=0\n"
                f"fg_candidates=0",
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
            **{f'diag_{k}': v for k, v in diag_stats.items()},
        }
        if stage2_enabled:
            for key, values in dp_stat_accum.items():
                extra_metrics[key] = float(np.mean(values)) if values else 0.0
            extra_metrics['grad_full_dp_pgfa'] = float(np.mean(grads['full']['dp_pgfa'])) if grads['full']['dp_pgfa'] else 0.0
            extra_metrics['grad_missing_dp_pgfa'] = float(np.mean(grads['missing']['dp_pgfa'])) if grads['missing']['dp_pgfa'] else 0.0
            extra_metrics['grad_full_dp_out_proj'] = float(np.mean(grads['full']['dp_out_proj'])) if grads['full']['dp_out_proj'] else 0.0
            extra_metrics['grad_missing_dp_out_proj'] = float(np.mean(grads['missing']['dp_out_proj'])) if grads['missing']['dp_out_proj'] else 0.0
            extra_metrics['grad_full_dp_core'] = float(np.mean(grads['full']['dp_core'])) if grads['full']['dp_core'] else 0.0
            extra_metrics['grad_missing_dp_core'] = float(np.mean(grads['missing']['dp_core'])) if grads['missing']['dp_core'] else 0.0
            extra_metrics['nonfinite_grad_steps'] = nonfinite_grad_steps
            extra_metrics['skipped_optimizer_steps'] = skipped_optimizer_steps
            extra_metrics['nonfinite_full_steps'] = nonfinite_full_steps
            extra_metrics['nonfinite_missing_steps'] = nonfinite_missing_steps

        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=avg_grad_norm,
            extra_metrics=extra_metrics,
        )

        print(f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


if __name__ == '__main__':
    main()
