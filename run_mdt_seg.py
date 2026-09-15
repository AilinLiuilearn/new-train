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
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


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


def _validate_train_mode(cfg):
    train_mode = str(getattr(cfg, 'train_mode', 'within_batch_alternating'))
    if train_mode not in ('within_batch_alternating', 'legacy_batch_alternating'):
        raise ValueError(f'unsupported train_mode={train_mode!r}')
    if train_mode == 'within_batch_alternating':
        full_w = float(getattr(cfg, 'within_batch_full_weight', 0.5))
        missing_w = float(getattr(cfg, 'within_batch_missing_weight', 0.5))
        if full_w <= 0.0 or missing_w <= 0.0:
            raise ValueError(f'within-batch weights must be > 0, got {full_w}, {missing_w}')
        if abs(full_w + missing_w - 1.0) > 1e-6:
            raise ValueError(f'within-batch weights must sum to 1.0, got {full_w} + {missing_w}')
    return train_mode


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
    return float(total.sqrt().item()) if total is not None else 0.0


def _retrieval_grad_norm(model):
    if not getattr(model, 'pspi_enabled', False) or getattr(model, 'module1', None) is None:
        return 0.0
    total = 0.0
    for mod in model.module1.attention:
        total += sum(p.grad.detach().float().pow(2).sum().item() if p.grad is not None else 0 for p in mod.parameters())
    return float(total ** 0.5) if total > 0 else 0.0


def _prior_scale_grad_norm(model):
    g = getattr(model, 'missing_prior_logits', None)
    if g is None or getattr(g, 'grad', None) is None:
        return 0.0
    return float(g.grad.detach().float().pow(2).sum().sqrt().item())


def _checkpoint_paths(checkpoint_dir):
    return {
        'best_joint': os.path.join(checkpoint_dir, 'ckpt.best_joint.pth.tar'),
        'best_full': os.path.join(checkpoint_dir, 'ckpt.best_full.pth.tar'),
        'best_missing': os.path.join(checkpoint_dir, 'ckpt.best_missing.pth.tar'),
        'last': os.path.join(checkpoint_dir, 'ckpt.last.pth.tar'),
    }


def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _has_batchnorm(model):
    return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def main():
    print('[INFO] starting baseline training', flush=True)
    cfg = SegMDTConfig.parse_arguments()
    _assert_baseline(cfg)
    train_mode = _validate_train_mode(cfg)
    full_weight = float(getattr(cfg, 'within_batch_full_weight', 0.5))
    missing_weight = float(getattr(cfg, 'within_batch_missing_weight', 0.5))
    cosine_diag_enabled = bool(getattr(cfg, 'within_batch_cosine_diagnostics', False))
    cosine_every = max(1, int(getattr(cfg, 'within_batch_cosine_every', 100)))
    cosine_samples = max(1, int(getattr(cfg, 'within_batch_cosine_samples', 2)))

    def _cosine_due(step):
        return bool(cosine_diag_enabled) and (int(step) % cosine_every == 0)
    _seed(cfg)
    print('[REPRO] mode=seeded_stochastic', flush=True)
    print('[REPRO] seed={}'.format(cfg.random_state), flush=True)
    print('[REPRO] deterministic_algorithms=False', flush=True)
    print('[REPRO] cudnn_deterministic=True', flush=True)
    print('[REPRO] cudnn_benchmark=False', flush=True)
    print('[REPRO] CUBLAS_WORKSPACE_CONFIG=unset', flush=True)
    print('[REPRO] TF32=False', flush=True)
    print(f'[TRAIN] mode={train_mode}', flush=True)
    if train_mode == 'within_batch_alternating':
        print(f'[TRAIN] full_weight={full_weight} missing_weight={missing_weight}', flush=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, 'config_args.json'), 'w') as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    train_loader, val_loader, _ = _loaders(cfg)
    print(f'[INFO] train_batches={len(train_loader)} val_batches={len(val_loader)}', flush=True)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    total_params, trainable_params = _count_parameters(task.model)
    print(f'[INFO] params_total={total_params} params_trainable={trainable_params}', flush=True)
    print(f'[INFO] batchnorm_present={_has_batchnorm(task.model)} bn_forward_updates_per_batch=2 bn_behaviour_unchanged=True', flush=True)
    # No Stage-1.5 bootstrap: epoch-1 cold start, bank_version=0, ready=False

    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )

    within_batch = (train_mode == 'within_batch_alternating')
    if within_batch:
        extra_headers = [
            'train_mode', 'train_full_loss', 'train_missing_loss', 'train_combined_loss',
            'full_train_batches', 'missing_train_batches',
            'train_full_proto_loss', 'train_missing_proto_loss',
            'train_full_proto_loss_weighted', 'train_missing_proto_loss_weighted',
            'full_loss_weight', 'missing_loss_weight',
            'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc', 'val_full_acc_pixel', 'val_full_hd95',
            'val_missing_loss', 'val_missing_dice', 'val_missing_iou', 'val_missing_acc', 'val_missing_acc_pixel', 'val_missing_hd95',
            'joint_dice', 'best_joint', 'best_joint_epoch',
            'grad_full_enc_ct', 'grad_missing_enc_ct', 'grad_combined_enc_ct',
            'grad_full_enc_pet', 'grad_missing_enc_pet', 'grad_combined_enc_pet',
            'grad_full_ct_align', 'grad_missing_ct_align', 'grad_combined_ct_align',
            'grad_full_decoder', 'grad_missing_decoder', 'grad_combined_decoder',
            'grad_full_module1_retrieval', 'grad_missing_module1_retrieval', 'grad_combined_module1_retrieval',
            'grad_full_prior_scale', 'grad_missing_prior_scale', 'grad_combined_prior_scale',
            'within_batch_grad_cosine_shared', 'within_batch_grad_cosine_ct', 'within_batch_grad_cosine_decoder',
            'optimizer_steps', 'full_forward_count', 'missing_forward_count', 'scheduler_step_count',
            'module1_collection_calls',
            'attention_entropy_s1', 'attention_entropy_s2', 'attention_entropy_s3', 'attention_entropy_s4',
            'normalized_attention_entropy_s1', 'normalized_attention_entropy_s2', 'normalized_attention_entropy_s3', 'normalized_attention_entropy_s4',
            'pet_prior_norm',
            'bank_ready', 'bank_version',
            'prototype_diversity_background', 'prototype_diversity_foreground',
            'mean_matching_cosine_distance', 'max_matching_cosine_distance',
            'duplicate_current_match_count', 'ct_key_update_norm', 'pet_value_update_norm',
            'bank_update_mode', 'bank_update_detail_mode',
            'missing_prior_alpha_s1', 'missing_prior_alpha_s2', 'missing_prior_alpha_s3', 'missing_prior_alpha_s4',
            'epoch_time',
        ]
    else:
        extra_headers = [
            'train_full_loss', 'train_missing_loss', 'train_overall_loss',
            'full_train_batches', 'missing_train_batches',
            'train_full_proto_loss', 'train_missing_proto_loss',
            'train_full_proto_loss_weighted', 'train_missing_proto_loss_weighted',
            'val_full_loss', 'val_full_dice', 'val_full_iou', 'val_full_acc', 'val_full_acc_pixel', 'val_full_hd95',
            'val_missing_loss', 'val_missing_dice', 'val_missing_iou', 'val_missing_acc', 'val_missing_acc_pixel', 'val_missing_hd95',
            'joint_dice', 'best_joint', 'best_joint_epoch',
            'grad_full_enc_ct', 'grad_missing_enc_ct',
            'grad_full_enc_pet', 'grad_missing_enc_pet',
            'grad_full_ct_align', 'grad_missing_ct_align',
            'grad_full_decoder', 'grad_missing_decoder',
            'grad_full_module1_retrieval', 'grad_missing_module1_retrieval',
            'grad_full_prior_scale', 'grad_missing_prior_scale',
            'attention_entropy_s1', 'attention_entropy_s2', 'attention_entropy_s3', 'attention_entropy_s4',
            'normalized_attention_entropy_s1', 'normalized_attention_entropy_s2', 'normalized_attention_entropy_s3', 'normalized_attention_entropy_s4',
            'pet_prior_norm',
            'bank_ready', 'bank_version',
            'prototype_diversity_background', 'prototype_diversity_foreground',
            'mean_matching_cosine_distance', 'max_matching_cosine_distance',
            'duplicate_current_match_count', 'ct_key_update_norm', 'pet_value_update_norm',
            'bank_update_mode', 'bank_update_detail_mode',
            'missing_prior_alpha_s1', 'missing_prior_alpha_s2', 'missing_prior_alpha_s3', 'missing_prior_alpha_s4',
            'epoch_time',
        ]
    init_train_log(os.path.join(cfg.checkpoint_dir, 'train_log.csv'), extra_headers=extra_headers)

    best_joint = -1.0
    best_full = -1.0
    best_missing = -1.0
    best_joint_epoch = 0
    global_batch_step = 0
    scheduler_step_count = 0
    amp_enabled = bool(cfg.mixed_precision)
    patience = int(getattr(cfg, 'early_stop_patience', 10))
    no_improve = 0
    paths = _checkpoint_paths(cfg.checkpoint_dir)

    for epoch in range(1, cfg.epochs + 1):
        task.model.train()
        full_n = missing_n = 0
        full_loss = missing_loss = combined_loss = 0.0
        full_proto = 0.0
        missing_proto = 0.0
        full_proto_w = 0.0
        missing_proto_w = 0.0
        grad_norm_accum = 0.0
        grad_norm_steps = 0
        grads = {
            'full': {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'decoder': [], 'retrieval': [], 'prior_scale': []},
            'missing': {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'decoder': [], 'retrieval': [], 'prior_scale': []},
            'combined': {'enc_ct': [], 'enc_pet': [], 'ct_align': [], 'decoder': [], 'retrieval': [], 'prior_scale': []},
        }
        cosine_shared_vals = []
        cosine_ct_vals = []
        cosine_dec_vals = []
        epoch_start = time.time()
        epoch_opt_steps = 0
        epoch_sched_steps = 0
        epoch_full_fwd = 0
        epoch_missing_fwd = 0
        epoch_collect_calls = 0
        fixed_diag_batch = None
        diag_stats = {}
        attn_ent_accum = {f's{i}': [] for i in range(1, 5)}
        nattn_ent_accum = {f's{i}': [] for i in range(1, 5)}
        prior_norm_vals = []
        prior_alpha_accum = {f's{i}': [] for i in range(1, 5)}

        for batch_idx, batch in enumerate(train_loader):
            if within_batch:
                combined, full_logits, missing_logits, step_stats = task.train_batch_full_missing(
                    batch, full_weight=full_weight, missing_weight=missing_weight,
                )
                if not torch.isfinite(combined):
                    raise RuntimeError('combined loss became non-finite')
                task.scheduler.step()
                scheduler_step_count += 1

                epoch_opt_steps += 1
                epoch_sched_steps += 1
                epoch_full_fwd += 1
                epoch_missing_fwd += 1
                collect_calls = int(getattr(task.model.module1, '_collect_calls', 0)) if getattr(task.model, 'module1', None) is not None else 0
                if task.model.training:
                    epoch_collect_calls += 1
                full_outputs = step_stats['full_outputs']
                missing_outputs = step_stats['missing_outputs']

                grad_norm_accum += float(step_stats['total_grad_norm'])
                grad_norm_steps += 1
                grads['combined']['enc_ct'].append(float(step_stats['grad_combined_enc_ct']))
                grads['combined']['enc_pet'].append(float(step_stats['grad_combined_enc_pet']))
                grads['combined']['ct_align'].append(float(step_stats['grad_combined_ct_align']))
                grads['combined']['decoder'].append(float(step_stats['grad_combined_decoder']))
                grads['combined']['retrieval'].append(float(step_stats['grad_combined_module1_retrieval']))
                grads['combined']['prior_scale'].append(float(step_stats['grad_combined_prior_scale']))

                cosine_shared_vals.append(
                    _branch_cosine(
                        task, batch, params='shared',
                        full_weight=full_weight, missing_weight=missing_weight,
                        max_samples=cosine_samples,
                    ) if _cosine_due(global_batch_step) else 0.0
                )
                cosine_ct_vals.append(
                    _branch_cosine(
                        task, batch, params='enc_ct',
                        full_weight=full_weight, missing_weight=missing_weight,
                        max_samples=cosine_samples,
                    ) if _cosine_due(global_batch_step) else 0.0
                )
                cosine_dec_vals.append(
                    _branch_cosine(
                        task, batch, params='decoder',
                        full_weight=full_weight, missing_weight=missing_weight,
                        max_samples=cosine_samples,
                    ) if _cosine_due(global_batch_step) else 0.0
                )

                full_n += 1
                missing_n += 1
                full_loss += float(step_stats['full_loss'])
                missing_loss += float(step_stats['missing_loss'])
                combined_loss += float(step_stats['combined_loss'])
                full_proto += float(step_stats['full_proto_loss'])
                missing_proto += float(step_stats['missing_proto_loss'])
                full_proto_w += float(step_stats['full_proto_loss_weighted'])
                missing_proto_w += float(step_stats['missing_proto_loss_weighted'])

                if (batch_idx + 1) % 100 == 0:
                    print(f'[BATCH {batch_idx + 1}] mode=within_batch full_loss={float(step_stats["full_loss"]):.6f} missing_loss={float(step_stats["missing_loss"]):.6f} combined={float(step_stats["combined_loss"]):.6f}', flush=True)

                for i in range(1, 5):
                    for outputs, key in ((full_outputs, f'attention_entropy_s{i}'),):
                        if key in outputs:
                            attn_ent_accum[f's{i}'].append(float(outputs[key]))
                    k2 = f'normalized_attention_entropy_s{i}'
                    for outputs in (full_outputs, missing_outputs):
                        if k2 in outputs:
                            nattn_ent_accum[f's{i}'].append(float(outputs[k2]))
                for outputs in (full_outputs, missing_outputs):
                    if 'pet_prior_norm' in outputs:
                        prior_norm_vals.append(float(outputs['pet_prior_norm']))
                for i in range(1, 5):
                    k = f'missing_prior_alpha_s{i}'
                    for outputs in (full_outputs, missing_outputs):
                        if k in outputs:
                            prior_alpha_accum[f's{i}'].append(float(outputs[k]))

                global_batch_step += 1
                task.global_batch_step = global_batch_step
                if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is None:
                    fixed_diag_batch = {
                        'ct': batch['ct'][:1].detach().cpu(),
                        'pet': batch['pet'][:1].detach().cpu(),
                        'mask': batch['mask'][:1].detach().cpu(),
                    }
                continue

            route = 'full' if global_batch_step % 2 == 0 else 'missing'
            task.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
                loss, _, outputs, step_stats = task.train_step(batch, forward_mode=route)
            if not torch.isfinite(loss):
                raise RuntimeError('loss became non-finite')

            if task.scaler.is_enabled():
                task.scaler.scale(loss).backward()
                task.scaler.unscale_(task.optimizer)
            else:
                loss.backward()

            grads[route]['enc_ct'].append(module_grad_norm(task.model.enc_ct))
            grads[route]['enc_pet'].append(module_grad_norm(task.model.enc_pet))
            grads[route]['ct_align'].append(module_grad_norm(task.model.ct_align))
            grads[route]['decoder'].append(module_grad_norm(task.model.decoder))
            if task.model.pspi_enabled and task.model.module1 is not None:
                ret_norm = 0.0
                for mod in task.model.module1.attention:
                    ret_norm += sum(p.grad.detach().float().pow(2).sum().item() if p.grad is not None else 0 for p in mod.parameters())
                ret_norm = float(ret_norm ** 0.5) if ret_norm > 0 else 0.0
                grads[route]['retrieval'].append(ret_norm)
                if getattr(task.model, 'missing_prior_logits', None) is not None:
                    g = task.model.missing_prior_logits.grad
                    ps_norm = float(g.detach().float().pow(2).sum().sqrt().item()) if g is not None else 0.0
                else:
                    ps_norm = 0.0
                grads[route]['prior_scale'].append(ps_norm)
            else:
                grads[route]['retrieval'].append(0.0)
                grads[route]['prior_scale'].append(0.0)

            total_grad_norm = torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip)) if float(cfg.grad_clip) > 0 else 0.0
            grad_norm_accum += float(total_grad_norm)
            grad_norm_steps += 1

            if task.scaler.is_enabled():
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                task.optimizer.step()

            task.scheduler.step()
            scheduler_step_count += 1

            if (batch_idx + 1) % 100 == 0:
                print(f'[BATCH {batch_idx + 1}] route={route} loss={float(loss.detach()):.6f}', flush=True)

            if route == 'full':
                full_n += 1
                full_loss += float(loss.detach())
                full_proto += float(step_stats['loss_proto'].detach())
                full_proto_w += float(step_stats['loss_proto_weighted'].detach())
            else:
                missing_n += 1
                missing_loss += float(loss.detach())
                missing_proto += float(step_stats['loss_proto'].detach())
                missing_proto_w += float(step_stats['loss_proto_weighted'].detach())

            if outputs is not None and isinstance(outputs, dict):
                for i in range(1, 5):
                    k = f'attention_entropy_s{i}'
                    if k in outputs:
                        attn_ent_accum[f's{i}'].append(float(outputs[k]))
                    k2 = f'normalized_attention_entropy_s{i}'
                    if k2 in outputs:
                        nattn_ent_accum[f's{i}'].append(float(outputs[k2]))
                if 'pet_prior_norm' in outputs:
                    prior_norm_vals.append(float(outputs['pet_prior_norm']))
                for i in range(1, 5):
                    k = f'missing_prior_alpha_s{i}'
                    if k in outputs:
                        prior_alpha_accum[f's{i}'].append(float(outputs[k]))

            global_batch_step += 1
            task.global_batch_step = global_batch_step
            if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is None:
                fixed_diag_batch = {
                    'ct': batch['ct'][:1].detach().cpu(),
                    'pet': batch['pet'][:1].detach().cpu(),
                    'mask': batch['mask'][:1].detach().cpu(),
                }

        if within_batch:
            if not (epoch_full_fwd == epoch_opt_steps == epoch_missing_fwd == len(train_loader)):
                raise RuntimeError(
                    f'within-batch sanity failed: full_fwd={epoch_full_fwd} missing_fwd={epoch_missing_fwd} '
                    f'opt_steps={epoch_opt_steps} batches={len(train_loader)}'
                )
            if epoch_sched_steps != epoch_opt_steps:
                raise RuntimeError(
                    f'scheduler/optimizer mismatch: sched={epoch_sched_steps} opt={epoch_opt_steps}'
                )
            if epoch_collect_calls != len(train_loader):
                raise RuntimeError(
                    f'module1 collection count wrong: got {epoch_collect_calls}, expected {len(train_loader)}'
                )
            print(f'[TRAIN] epoch={epoch} optimizer_steps={epoch_opt_steps} full_forwards={epoch_full_fwd} missing_forwards={epoch_missing_fwd} scheduler_steps={epoch_sched_steps} module1_collections={epoch_collect_calls}', flush=True)

        if getattr(cfg, 'enable_gradient_diagnostics', False) and fixed_diag_batch is not None and epoch % int(cfg.gradient_diagnostics_interval) == 0:
            diag_stats = task.gradient_diagnostics(fixed_diag_batch, max_samples=min(1, int(cfg.gradient_diagnostics_num_samples))) or {}

        module1_report = None
        bank_update_detail_mode = ""
        prototype_diversity_background = 0.0
        prototype_diversity_foreground = 0.0
        mean_matching_cosine_distance = 0.0
        max_matching_cosine_distance = 0.0
        duplicate_current_match_count = 0
        ct_key_update_norm = 0.0
        pet_value_update_norm = 0.0
        if hasattr(task.model, 'finalize_module1_epoch'):
            module1_report = task.model.finalize_module1_epoch(epoch)
        if module1_report is not None:
            update = module1_report.get("update") or {}
            bank_update_detail_mode = str(update.get("mode", ""))
            prototype_diversity_background = float(module1_report.get("prototype_diversity_background", update.get("prototype_diversity_background", 0.0)))
            prototype_diversity_foreground = float(module1_report.get("prototype_diversity_foreground", update.get("prototype_diversity_foreground", 0.0)))
            mean_matching_cosine_distance = float(module1_report.get("mean_matching_cosine_distance", update.get("mean_matching_cosine_distance", 0.0)))
            max_matching_cosine_distance = float(module1_report.get("max_matching_cosine_distance", update.get("max_matching_cosine_distance", 0.0)))
            duplicate_current_match_count = int(module1_report.get("duplicate_current_match_count", update.get("duplicate_current_match_count", 0)))
            ct_key_update_norm = float(module1_report.get("ct_key_update_norm", update.get("ct_key_update_norm", 0.0)))
            pet_value_update_norm = float(module1_report.get("pet_value_update_norm", update.get("pet_value_update_norm", 0.0)))
            extra = ""
            if bank_update_detail_mode in ("fedmepd_ema", "fedmepd_ema_init"):
                extra = (
                    f" diversity_bg={prototype_diversity_background:.4f}"
                    f" diversity_fg={prototype_diversity_foreground:.4f}"
                    f" mean_dist={mean_matching_cosine_distance:.4f}"
                    f" dup={duplicate_current_match_count}"
                    f" ct_norm={ct_key_update_norm:.6f}"
                    f" pet_norm={pet_value_update_norm:.6f}"
                )
            if getattr(task.model, 'missing_prior_logits', None) is not None:
                _alphas = [float(torch.sigmoid(v).item()) for v in task.model.missing_prior_logits.detach()]
            else:
                _alphas = [1.0, 1.0, 1.0, 1.0]
            print(
                f"[PSPI][BANK] epoch={module1_report.get('epoch', epoch)} "
                f"status={module1_report.get('status')} "
                f"mode={bank_update_detail_mode or getattr(cfg, 'pspi_bank_update_mode', '')} "
                f"bank_version={module1_report.get('bank_version_after', module1_report.get('bank_version_before', 0))} "
                f"ready_count={module1_report.get('ready_count', 0)} "
                f"total_slots={module1_report.get('total_slots', 0)}"
                f"{extra} "
                f"alpha=[{', '.join(f'{a:.4f}' for a in _alphas)}]",
                flush=True,
            )

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

        if within_batch:
            train_loss = combined_loss / max(1, missing_n)
        else:
            train_loss = (full_loss + missing_loss) / max(1, full_n + missing_n)
        val_loss = 0.5 * val_full['total_loss'] + 0.5 * val_missing['total_loss']
        val_dice = joint_dice
        val_iou = 0.5 * val_full['iou'] + 0.5 * val_missing['iou']
        val_acc = 0.5 * val_full['acc'] + 0.5 * val_missing['acc']
        val_acc_pixel = 0.5 * val_full.get('acc_pixel', 0.0) + 0.5 * val_missing.get('acc_pixel', 0.0)
        val_hd95 = 0.5 * val_full['hd95'] + 0.5 * val_missing['hd95']
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_steps)
        bank_ready_val = 0
        bank_version_val = 0
        if task.model.pspi_enabled and task.model.module1 is not None:
            bank_ready_val = 1 if task.model.module1.bank_ready else 0
            bank_version_val = int(task.model.module1.bank_version.item())
        grads_combined_keys = {'enc_ct': 'enc_ct', 'enc_pet': 'enc_pet', 'ct_align': 'ct_align', 'decoder': 'decoder', 'retrieval': 'module1_retrieval', 'prior_scale': 'prior_scale'}
        if within_batch:
            extra = {
                'train_mode': train_mode,
                'train_full_loss': full_loss / max(1, full_n),
                'train_missing_loss': missing_loss / max(1, missing_n),
                'train_combined_loss': combined_loss / max(1, missing_n),
                'full_train_batches': full_n,
                'missing_train_batches': missing_n,
                'train_full_proto_loss': full_proto / max(1, full_n),
                'train_missing_proto_loss': missing_proto / max(1, missing_n),
                'train_full_proto_loss_weighted': full_proto_w / max(1, full_n),
                'train_missing_proto_loss_weighted': missing_proto_w / max(1, missing_n),
                'full_loss_weight': full_weight,
                'missing_loss_weight': missing_weight,
                'within_batch_grad_cosine_shared': float(np.mean(cosine_shared_vals)) if cosine_shared_vals else 0.0,
                'within_batch_grad_cosine_ct': float(np.mean(cosine_ct_vals)) if cosine_ct_vals else 0.0,
                'within_batch_grad_cosine_decoder': float(np.mean(cosine_dec_vals)) if cosine_dec_vals else 0.0,
                'optimizer_steps': epoch_opt_steps,
                'full_forward_count': epoch_full_fwd,
                'missing_forward_count': epoch_missing_fwd,
                'scheduler_step_count': epoch_sched_steps,
                'module1_collection_calls': epoch_collect_calls,
                **{f'grad_full_{grads_combined_keys[k]}': float(np.mean(v)) if v else 0.0 for k, v in grads['full'].items()},
                **{f'grad_missing_{grads_combined_keys[k]}': float(np.mean(v)) if v else 0.0 for k, v in grads['missing'].items()},
                **{f'grad_combined_{grads_combined_keys[k]}': float(np.mean(v)) if v else 0.0 for k, v in grads['combined'].items()},
            }
        else:
            extra = {}
        append_epoch_log(
            os.path.join(cfg.checkpoint_dir, 'train_log.csv'),
            epoch,
            train_loss,
            {'total_loss': val_loss, 'dice': val_dice, 'iou': val_iou, 'acc': val_acc, 'acc_pixel': val_acc_pixel, 'hd95': val_hd95},
            lr=task.optimizer.param_groups[0]['lr'],
            grad_norm=avg_grad_norm,
            extra_metrics={
                **extra,
                'train_full_loss': full_loss / max(1, full_n) if not within_batch else extra['train_full_loss'],
                'train_missing_loss': missing_loss / max(1, missing_n) if not within_batch else extra['train_missing_loss'],
                **({} if within_batch else {'train_overall_loss': train_loss}),
                'full_train_batches': full_n,
                'missing_train_batches': missing_n,
                'train_full_proto_loss': full_proto / max(1, full_n),
                'train_missing_proto_loss': missing_proto / max(1, missing_n),
                'train_full_proto_loss_weighted': full_proto_w / max(1, full_n),
                'train_missing_proto_loss_weighted': missing_proto_w / max(1, missing_n),
                **({} if within_batch else {
                    'grad_full_enc_ct': float(np.mean(grads['full']['enc_ct'])) if grads['full']['enc_ct'] else 0.0,
                    'grad_missing_enc_ct': float(np.mean(grads['missing']['enc_ct'])) if grads['missing']['enc_ct'] else 0.0,
                    'grad_full_enc_pet': float(np.mean(grads['full']['enc_pet'])) if grads['full']['enc_pet'] else 0.0,
                    'grad_missing_enc_pet': float(np.mean(grads['missing']['enc_pet'])) if grads['missing']['enc_pet'] else 0.0,
                    'grad_full_ct_align': float(np.mean(grads['full']['ct_align'])) if grads['full']['ct_align'] else 0.0,
                    'grad_missing_ct_align': float(np.mean(grads['missing']['ct_align'])) if grads['missing']['ct_align'] else 0.0,
                    'grad_full_decoder': float(np.mean(grads['full']['decoder'])) if grads['full']['decoder'] else 0.0,
                    'grad_missing_decoder': float(np.mean(grads['missing']['decoder'])) if grads['missing']['decoder'] else 0.0,
                    'grad_full_module1_retrieval': float(np.mean(grads['full']['retrieval'])) if grads['full']['retrieval'] else 0.0,
                    'grad_missing_module1_retrieval': float(np.mean(grads['missing']['retrieval'])) if grads['missing']['retrieval'] else 0.0,
                    'grad_full_prior_scale': float(np.mean(grads['full']['prior_scale'])) if grads['full']['prior_scale'] else 0.0,
                    'grad_missing_prior_scale': float(np.mean(grads['missing']['prior_scale'])) if grads['missing']['prior_scale'] else 0.0,
                }),
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
                'attention_entropy_s1': float(np.mean(attn_ent_accum['s1'])) if attn_ent_accum['s1'] else 0.0,
                'attention_entropy_s2': float(np.mean(attn_ent_accum['s2'])) if attn_ent_accum['s2'] else 0.0,
                'attention_entropy_s3': float(np.mean(attn_ent_accum['s3'])) if attn_ent_accum['s3'] else 0.0,
                'attention_entropy_s4': float(np.mean(attn_ent_accum['s4'])) if attn_ent_accum['s4'] else 0.0,
                'normalized_attention_entropy_s1': float(np.mean(nattn_ent_accum['s1'])) if nattn_ent_accum['s1'] else 0.0,
                'normalized_attention_entropy_s2': float(np.mean(nattn_ent_accum['s2'])) if nattn_ent_accum['s2'] else 0.0,
                'normalized_attention_entropy_s3': float(np.mean(nattn_ent_accum['s3'])) if nattn_ent_accum['s3'] else 0.0,
                'normalized_attention_entropy_s4': float(np.mean(nattn_ent_accum['s4'])) if nattn_ent_accum['s4'] else 0.0,
                'pet_prior_norm': float(np.mean(prior_norm_vals)) if prior_norm_vals else 0.0,
                'bank_ready': bank_ready_val,
                'bank_version': bank_version_val,
                'prototype_diversity_background': prototype_diversity_background,
                'prototype_diversity_foreground': prototype_diversity_foreground,
                'mean_matching_cosine_distance': mean_matching_cosine_distance,
                'max_matching_cosine_distance': max_matching_cosine_distance,
                'duplicate_current_match_count': duplicate_current_match_count,
                'ct_key_update_norm': ct_key_update_norm,
                'pet_value_update_norm': pet_value_update_norm,
                'bank_update_mode': getattr(cfg, 'pspi_bank_update_mode', 'direct'),
                'bank_update_detail_mode': bank_update_detail_mode,
                'missing_prior_alpha_s1': float(np.mean(prior_alpha_accum['s1'])) if prior_alpha_accum['s1'] else 0.0,
                'missing_prior_alpha_s2': float(np.mean(prior_alpha_accum['s2'])) if prior_alpha_accum['s2'] else 0.0,
                'missing_prior_alpha_s3': float(np.mean(prior_alpha_accum['s3'])) if prior_alpha_accum['s3'] else 0.0,
                'missing_prior_alpha_s4': float(np.mean(prior_alpha_accum['s4'])) if prior_alpha_accum['s4'] else 0.0,
                'epoch_time': time.time() - epoch_start,
                **{f'diag_{k}': v for k, v in diag_stats.items()},
            },
        )

        print(f'[EPOCH {epoch}] joint_dice={joint_dice:.4f} best_joint={best_joint:.4f} lr={task.optimizer.param_groups[0]["lr"]:.8f}', flush=True)
        if no_improve >= patience:
            print(f'[EARLY STOP] no improvement for {patience} epochs', flush=True)
            break

    print('done', flush=True)


def _branch_cosine(task, batch, params='shared', full_weight=0.5, missing_weight=0.5, max_samples=2):
    """Full-vs-Missing branch cosine at the current parameters (diagnostic).

    Uses the same weighting as the combined objective, i.e. the gradient of
    the combined loss is exactly full_weight*grad_full + missing_weight*
    grad_missing; the cosine is measured between the two branch gradients.
    Read-only: parameters and .grad are untouched. Runs under torch.no_grad
    with grad enabled only for the two autograd.grad calls, on a small slice.
    """
    was_training = task.model.training
    task.model.eval()
    bn_states = []
    for m in task.model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            bn_states.append((m, m.track_running_stats))
            m.track_running_stats = False
    try:
        with torch.no_grad():
            ct = batch['ct'][:max_samples].to(task.device, non_blocking=True)
            pet = batch['pet'][:max_samples].to(task.device, non_blocking=True)
            mask = batch['mask'][:max_samples].to(task.device, non_blocking=True).float()
        proto_w = float(getattr(task.config, 'pspi_proto_contrastive_weight', 0.01))

        def _total(out):
            logits = out['logits'] if isinstance(out, dict) else out
            seg, _ = task.criterion(logits.float(), mask.float())
            proto = out.get('prototype_contrastive_loss', seg.new_zeros(())) if isinstance(out, dict) else seg.new_zeros(())
            if torch.is_tensor(proto) and proto.dim() > 0:
                proto = proto.reshape(())
            return seg + proto_w * proto

        out_f = task.model(ct, pet=pet, forward_mode='full', mask=mask, collect_module1_candidates=False)
        out_m = task.model(ct, pet=pet, forward_mode='missing', mask=mask, collect_module1_candidates=False)
        loss_f = full_weight * _total(out_f)
        loss_m = missing_weight * _total(out_m)
        if params == 'enc_ct':
            plist = list(task.model.enc_ct.parameters())
        elif params == 'decoder':
            plist = list(task.model.decoder.parameters())
        else:
            plist = (
                list(task.model.enc_ct.parameters())
                + list(task.model.ct_align.parameters())
                + list(task.model.decoder.parameters())
            )
        plist = [p for p in plist if p.requires_grad]
        gf = torch.autograd.grad(loss_f, plist, retain_graph=True, allow_unused=True)
        gm = torch.autograd.grad(loss_m, plist, retain_graph=True, allow_unused=True)
        af = torch.cat([g.reshape(-1) for g in gf if g is not None]) if any(g is not None for g in gf) else torch.zeros(1)
        am = torch.cat([g.reshape(-1) for g in gm if g is not None]) if any(g is not None for g in gm) else torch.zeros(1)
        af = af.float()
        am = am.float()
        return float(torch.dot(af, am) / (af.norm() * am.norm() + 1e-8))
    finally:
        for m, state in bn_states:
            m.track_running_stats = state
        task.model.train(was_training)


if __name__ == '__main__':
    main()
