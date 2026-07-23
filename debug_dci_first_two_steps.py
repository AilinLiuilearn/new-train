# -*- coding: utf-8 -*-
import copy
import json
import os
import random
from contextlib import contextmanager

import numpy as np
import torch

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.optimization import get_cosine_scheduler


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_stats(x):
    x = x.detach().float()
    finite = torch.isfinite(x)
    stats = {
        'dtype': str(x.dtype),
        'shape': tuple(x.shape),
        'device': str(x.device),
        'finite': bool(finite.all().item()),
        'numel': int(x.numel()),
        'nan': int(torch.isnan(x).sum().item()),
        'posinf': int(torch.isposinf(x).sum().item()),
        'neginf': int(torch.isneginf(x).sum().item()),
    }
    if finite.any():
        vals = x[finite]
        stats.update({
            'min': float(vals.min().item()),
            'max': float(vals.max().item()),
            'mean': float(vals.mean().item()),
            'std': float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        })
    else:
        stats.update({'min': None, 'max': None, 'mean': None, 'std': None})
    return stats


def _report_batch(prefix, batch, include_pet=True):
    report = {}
    for key in ('case_name', 'filename', 'path', 'image_path', 'index', 'sample_id'):
        if key in batch:
            val = batch[key]
            report[key] = val if isinstance(val, (str, int, float)) else str(val)
    report['ct'] = _tensor_stats(batch['ct'])
    report['mask'] = _tensor_stats(batch['mask'])
    if include_pet and 'pet' in batch and batch['pet'] is not None:
        report['pet'] = _tensor_stats(batch['pet'])
    print(f'[{prefix}] {json.dumps(report, indent=2, default=str)}', flush=True)


@contextmanager
def _restored_state(model, optimizer, scaler, state):
    model.load_state_dict(copy.deepcopy(state['model']))
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(copy.deepcopy(state['scaler']))
        if hasattr(scaler, '_per_optimizer_states'):
            scaler._per_optimizer_states.clear()
    random.setstate(copy.deepcopy(state['python_rng']))
    np.random.set_state(copy.deepcopy(state['numpy_rng']))
    torch.set_rng_state(copy.deepcopy(state['torch_rng']))
    if torch.cuda.is_available() and state.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all(copy.deepcopy(state['cuda_rng']))
    try:
        yield
    finally:
        pass


def _capture_state(model, optimizer, scaler):
    return {
        'model': copy.deepcopy(model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'scaler': copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        'python_rng': copy.deepcopy(random.getstate()),
        'numpy_rng': copy.deepcopy(np.random.get_state()),
        'torch_rng': copy.deepcopy(torch.get_rng_state()),
        'cuda_rng': copy.deepcopy(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
    }


def _check_finite_named(module, kind):
    for name, param in module.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f'[NaN/Inf] non-finite {kind} parameter: {name}')
    for name, buf in module.named_buffers():
        if buf.is_floating_point() and not torch.isfinite(buf).all():
            raise RuntimeError(f'[NaN/Inf] non-finite {kind} buffer: {name}')


def _first_two_batches(loader):
    it = iter(loader)
    return next(it), next(it)


def _make_input(batch, device, route):
    ct = batch['ct'].to(device, non_blocking=True)
    mask = batch['mask'].to(device, non_blocking=True).float()
    pet = batch['pet'].to(device, non_blocking=True) if route == 'full' else None
    return ct, pet, mask


def _grad_report(model):
    bad = []
    for n, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(n)
    return bad


def _step_diagnosis(task, cfg, state, batch0, batch1, name, amp_enabled, use_dci, dci_sample, dci_weight, do_step):
    task.model.use_dci = use_dci
    if getattr(task.model, 'dci_fusion', None) is not None:
        task.model.dci_fusion.sample_during_training = bool(dci_sample)
    with _restored_state(task.model, task.optimizer, task.scaler, state):
        task.model.train()
        task.optimizer.zero_grad(set_to_none=True)
        ct0, pet0, mask0 = _make_input(batch0, task.device, 'full')
        ct1, _, mask1 = _make_input(batch1, task.device, 'missing')
        _report_batch(f'{name}-batch0', batch0, include_pet=True)
        _report_batch(f'{name}-batch1', batch1, include_pet=False)
        result = {'exp': name}
        with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
            out0 = task.model(ct0, pet=pet0, forward_mode='full', mask=mask0)
        result['full_logits_finite'] = bool(torch.isfinite(out0['logits']).all())
        result['full_dci_finite'] = bool(torch.isfinite(out0.get('loss_dci_dist', torch.tensor(0., device=task.device))).all())
        loss0, _ = task.criterion(out0['logits'], mask0)
        loss0 = loss0 + float(dci_weight) * out0.get('loss_dci_dist', loss0.new_zeros(()))
        result['loss0_finite_before_backward'] = bool(torch.isfinite(loss0).all())
        if task.scaler.is_enabled() and amp_enabled:
            task.scaler.scale(loss0).backward()
            result['scaled_backward_done'] = True
            task.scaler.unscale_(task.optimizer)
            result['unscale_done'] = True
        else:
            loss0.backward()
            result['scaled_backward_done'] = False
            result['unscale_done'] = False
        bad_grads = _grad_report(task.model)
        result['first_bad_grad'] = bad_grads[0] if bad_grads else None
        result['grad_finite'] = not bad_grads
        try:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(task.trainable_parameters(), float(cfg.grad_clip), error_if_nonfinite=True)
            result['clip_ok'] = True
            result['total_grad_norm'] = float(total_grad_norm)
        except Exception as e:
            result['clip_ok'] = False
            result['clip_err'] = repr(e)
            raise
        if do_step:
            if task.scaler.is_enabled() and amp_enabled:
                task.scaler.step(task.optimizer)
                task.scaler.update()
            else:
                task.optimizer.step()
            result['post_step_params_ok'] = all(torch.isfinite(p).all() for p in task.model.parameters())
            result['post_step_buffers_ok'] = all((not b.is_floating_point()) or torch.isfinite(b).all() for b in task.model.buffers())
        with torch.cuda.amp.autocast(enabled=amp_enabled and torch.cuda.is_available()):
            out1 = task.model(ct1, pet=None, forward_mode='missing')
        result['missing_logits_finite'] = bool(torch.isfinite(out1['logits']).all())
        result['missing_dci_finite'] = bool(torch.isfinite(out1.get('loss_dci_dist', torch.tensor(0., device=task.device))).all())
        loss1, _ = task.criterion(out1['logits'], mask1)
        loss1 = loss1 + float(dci_weight) * out1.get('loss_dci_dist', loss1.new_zeros(()))
        result['missing_loss_finite'] = bool(torch.isfinite(loss1).all())
        return result


def main():
    cfg = SegMDTConfig.parse_arguments()
    cfg.random_state = 2023
    _seed_everything(cfg.random_state)
    from run_mdt_seg import _loaders, _assert_baseline
    _assert_baseline(cfg)
    train_loader, _, _ = _loaders(cfg)
    batch0, batch1 = _first_two_batches(train_loader)
    print(json.dumps({
        'root': cfg.root,
        'train_split_file': cfg.train_split_file,
        'image_size_2d': cfg.image_size_2d,
        'batch_size': cfg.batch_size,
        'mixed_precision': cfg.mixed_precision,
        'use_dci': cfg.use_dci,
        'dci_sample_during_training': cfg.dci_sample_during_training,
        'dci_dist_weight': cfg.dci_dist_weight,
        'checkpoint_dir': cfg.checkpoint_dir,
        'seed': cfg.random_state,
    }, indent=2), flush=True)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.scheduler = get_cosine_scheduler(
        task.optimizer,
        epochs=cfg.epochs,
        warmup_steps=cfg.cosine_warmup * len(train_loader),
        min_lr=cfg.cosine_min_lr,
        steps_per_epoch=len(train_loader),
        flat_ratio=cfg.lr_flat_ratio,
    )
    _check_finite_named(task.model, 'initial')
    state = _capture_state(task.model, task.optimizer, task.scaler)

    experiments = [
        ('T0-AMP', True, True, True, cfg.dci_dist_weight, False),
        ('T0-FP32', False, True, True, cfg.dci_dist_weight, False),
        ('T1', True, True, True, cfg.dci_dist_weight, False),
        ('T2', True, True, True, cfg.dci_dist_weight, False),
        ('T3', True, True, True, cfg.dci_dist_weight, True),
        ('T4', True, False, True, 0.0, True),
        ('T5', True, True, False, 0.0, True),
        ('T6', True, True, False, 0.001, True),
    ]

    results = []
    for exp in experiments:
        name, amp_enabled, use_dci, dci_sample, dci_weight, do_step = exp
        try:
            results.append(_step_diagnosis(task, cfg, state, batch0, batch1, name, amp_enabled, use_dci, dci_sample, dci_weight, do_step))
        except Exception as e:
            results.append({'exp': name, 'error': repr(e)})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
