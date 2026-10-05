# -*- coding: utf-8 -*-
"""Shared training utilities for the two clean baselines.

One implementation is shared by both runners; independence means separate
entry points and checkpoints, not duplicated logic.
"""
import csv
import json
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_grad_norm(module):
    total = None
    for p in module.parameters():
        if p.grad is None:
            continue
        value = p.grad.detach().float().pow(2).sum()
        total = value if total is None else total + value
    return float(total.sqrt().item()) if total is not None else 0.0


def optimizer_step_succeeded(task):
    """AMP-aware optimizer step. Returns False when the step was skipped."""
    if not task.scaler.is_enabled():
        task.optimizer.step()
        return True
    before = task.scaler.get_scale()
    task.scaler.step(task.optimizer)
    task.scaler.update()
    after = task.scaler.get_scale()
    return bool(after >= before)


def build_balanced_pet_available(batch_size, global_batch_step, random_state, device):
    """Reproducible exact-half Full/Missing state per batch.

    Each batch holds B/2 Full (1) and B/2 Missing (0) rows, reshuffled every
    batch with a step-dependent seed so the assignment is reproducible.
    """
    batch_size = int(batch_size)
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(f'mixed mode requires an even batch_size >= 2, got {batch_size}')
    state = torch.cat([
        torch.ones(batch_size // 2, dtype=torch.long),
        torch.zeros(batch_size // 2, dtype=torch.long),
    ])
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(random_state) + int(global_batch_step) * 1000003)
    state = state[torch.randperm(batch_size, generator=generator)]
    return state.to(device)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


FINAL_MISSING_RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_case_mask(case_ids, missing_rate, seed):
    """Patient-level missing assignment shared by all experiments.

    Same (case list, rate, seed) always yields the same assignment, so both
    baselines evaluate on identical missing allocations.
    """
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(case_ids)))
    cut = int(round(float(missing_rate) * len(case_ids)))
    missing = {case_ids[idx] for idx in perm[:cut]}
    return {cid: (1 if cid in missing else 0) for cid in case_ids}


@torch.inference_mode()
def run_missing_rate_eval(task, model, loader, case_mask, tag='test', ct_only=False):
    """Evaluate one missing-rate operating point; returns full metric dict.

    ``ct_only=True`` calls the model with CT alone (the result is
    PET-invariant by construction); otherwise routes per-sample state.
    """
    from utils.metrics_seg import SegmentationMetricsCIPA
    metric = SegmentationMetricsCIPA()
    total_loss = []
    total_case_ids = set()
    missing_case_ids = set()
    slice_count = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            ct = batch['ct'].to(task.device, non_blocking=True)
            pet = batch.get('pet')
            if pet is not None:
                pet = pet.to(task.device, non_blocking=True)
            mask = batch['mask'].to(task.device, non_blocking=True).float()
            case_ids = list(batch['case_id'])
            total_case_ids.update(case_ids)
            missing_case_ids.update([cid for cid in case_ids if case_mask.get(cid, 1) == 1])
            if ct_only:
                outputs = model(ct)
            else:
                state = torch.tensor([0 if case_mask.get(cid, 1) else 1 for cid in case_ids],
                                     device=task.device, dtype=torch.long)
                outputs = model(ct, pet=pet, pet_available=state, forward_mode='auto')
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss, _ = task.criterion(logits, mask)
            metric.update(logits, mask)
            total_loss.append(float(loss))
            slice_count += len(case_ids)
    finally:
        if was_training:
            model.train()
    out = metric.compute()
    out['loss'] = float(np.mean(total_loss)) if total_loss else 0.0
    out['missing_case_count'] = len(missing_case_ids)
    out['total_case_count'] = len(total_case_ids)
    out['slice_count'] = slice_count
    out['tag'] = tag
    return out


def collect_case_ids(loader):
    all_case_ids = []
    for batch in loader:
        all_case_ids.extend(list(batch['case_id']))
    return sorted(set(all_case_ids))


def eval_missing_rates(task, model, loader, seed, checkpoint_dir, stem,
                       rates=None, weights_tag='raw', ct_only=False):
    """Run all missing-rate operating points and write metrics + assignments.

    Returns (results, joint) with joint = 0.5 * (rate0 dice + rate1 dice).
    """
    rates = list(FINAL_MISSING_RATES if rates is None else rates)
    case_ids = collect_case_ids(loader)
    results = []
    assignments = {}
    for rate in rates:
        case_mask = build_case_mask(case_ids, rate, int(seed))
        assignments[str(rate)] = case_mask
        out = run_missing_rate_eval(task, model, loader, case_mask, tag=f'{stem}_{rate}',
                                    ct_only=ct_only)
        results.append({
            'missing_rate': rate,
            'dice': out['dice'],
            'iou': out['iou'],
            'acc': out['acc'],
            'acc_pixel': out['acc_pixel'],
            'hd95': out['hd95'],
            'loss': out['loss'],
            'missing_case_count': out['missing_case_count'],
            'total_case_count': out['total_case_count'],
            'slice_count': out['slice_count'],
            'weights': weights_tag,
        })
    by_rate = {r['missing_rate']: r for r in results}
    joint = 0.5 * (by_rate[0.0]['dice'] + by_rate[1.0]['dice'])
    csv_path = os.path.join(checkpoint_dir, f'{stem}_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(checkpoint_dir, f'{stem}_metrics.json'), 'w') as f:
        json.dump({'joint_dice': joint, 'results': results}, f, indent=2)
    with open(os.path.join(checkpoint_dir, f'{stem}_case_assignments.json'), 'w') as f:
        json.dump(assignments, f, indent=2)
    print(f'[FINAL] joint={joint:.4f} full={by_rate[0.0]["dice"]:.4f} '
          f'missing={by_rate[1.0]["dice"]:.4f} weights={weights_tag}', flush=True)
    return results, joint