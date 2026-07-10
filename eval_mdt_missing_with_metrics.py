# -*- coding: utf-8 -*-
"""Evaluate MDT under deterministic PET missing rates with explicit split routes.

This script fixes two issues:
1) The saved missing masks are exactly the masks used during inference.
2) Mixed batches do not rely on forward_mode='auto'; instead they are split into
   full and missing sub-batches explicitly so missing samples always use the
   PG-MTR missing route.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import SegmentationMetricsCIPA

DEFAULT_MISSING_RATES = [0.0, 0.3, 0.5, 0.7, 1.0]


def _set_seed(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(task, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'epoch' in ckpt:
        print(f'[checkpoint] epoch={ckpt["epoch"]}')
    for k, v in task.networks.items():
        if k in ckpt:
            task.load_model_state_dict(v, ckpt[k], strict=False)
        else:
            raise KeyError(f'Missing network key in checkpoint: {k}')


def _load_dataset_module():
    import importlib.util
    root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_test_loader(config):
    module = _load_dataset_module()
    if getattr(config, 'use_aligned_loader', False):
        loaders = module.get_pclt20k_loaders_textproxy_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'cipa'),
            train_list=getattr(config, 'train_list', 'train_original.txt'),
            val_list=getattr(config, 'val_list', 'test.txt'),
            test_list=getattr(config, 'test_list', 'test.txt'),
            pet_drop_prob=0.0,
        )
    else:
        loaders = module.get_pclt20k_loaders(
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
    return loaders[2]


def _dataset_meta(dataset, idx: int) -> Dict:
    record = dataset.records[idx]
    image_id = record.get('image_id', str(idx))
    parts = image_id.split('_')
    slice_id = parts[-1] if len(parts) > 1 else image_id
    return {
        'dataset_idx': int(idx),
        'case_id': record.get('case_id', parts[0] if parts else ''),
        'slice_id': slice_id,
        'image_id': image_id,
        'has_pet': 1 if record.get('pet_path') is not None else 0,
    }


def _make_nested_base_random(num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random(num_samples)


def _mask_from_base_random(base_random: np.ndarray, rate: float) -> np.ndarray:
    rate = float(rate)
    if rate <= 0.0:
        return np.ones_like(base_random, dtype=np.int64)
    if rate >= 1.0:
        return np.zeros_like(base_random, dtype=np.int64)
    return (base_random >= rate).astype(np.int64)


def _extract_logits(outputs):
    if isinstance(outputs, dict):
        if 'logits' in outputs:
            return outputs['logits']
        if 'pred' in outputs:
            return outputs['pred']
        if 'preds' in outputs and isinstance(outputs['preds'], (list, tuple)):
            return outputs['preds'][0]
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _forward_batch_explicit(model, ct, pet, pet_available, target_size):
    device = ct.device
    batch_size = ct.shape[0]
    h, w = target_size
    batch_logits = torch.empty((batch_size, 1, h, w), device=device, dtype=ct.dtype)

    full_idx = torch.nonzero(pet_available == 1, as_tuple=False).flatten()
    miss_idx = torch.nonzero(pet_available == 0, as_tuple=False).flatten()

    full_forward_count = 0
    miss_forward_count = 0

    if full_idx.numel() > 0:
        ct_full = ct.index_select(0, full_idx)
        pet_full = pet.index_select(0, full_idx)
        outputs_full = model(
            ct=ct_full,
            pet=pet_full,
            target_size=target_size,
            forward_mode='full',
        )
        logits_full = _extract_logits(outputs_full)
        batch_logits.index_copy_(0, full_idx, logits_full)
        full_forward_count = int(full_idx.numel())

    if miss_idx.numel() > 0:
        ct_missing = ct.index_select(0, miss_idx)
        outputs_missing = model(
            ct=ct_missing,
            pet=None,
            target_size=target_size,
            forward_mode='missing',
        )
        logits_missing = _extract_logits(outputs_missing)
        batch_logits.index_copy_(0, miss_idx, logits_missing)
        miss_forward_count = int(miss_idx.numel())

    return batch_logits, full_forward_count, miss_forward_count


def evaluate_with_explicit_availability_mask(task, loader, pet_available_mask, threshold=0.5):
    model = task._unwrap(task.networks['model'])
    model.eval()
    device = task.device
    metrics = SegmentationMetricsCIPA(threshold=threshold).to(device)
    metrics.reset()

    total_loss = 0.0
    total_n = 0
    sample_offset = 0
    full_forward_count = 0
    miss_forward_count = 0
    sample_records: List[Dict] = []

    with torch.inference_mode():
        for batch in loader:
            ct = batch['ct'].float().to(device)
            pet = batch['pet'].float().to(device)
            mask = batch['mask'].float().to(device)

            bs = ct.shape[0]
            batch_available = pet_available_mask[sample_offset: sample_offset + bs]
            batch_start = sample_offset
            sample_offset += bs
            pet_available = torch.tensor(batch_available, device=device, dtype=torch.long)

            batch_logits, full_count, miss_count = _forward_batch_explicit(
                model,
                ct,
                pet,
                pet_available,
                mask.shape[-2:],
            )
            full_forward_count += full_count
            miss_forward_count += miss_count

            loss_seg, _ = task.loss_seg(batch_logits, mask)
            total_loss += loss_seg.item() * bs
            total_n += bs
            metrics.update(batch_logits, mask)

            for i in range(bs):
                sample_records.append({
                    'dataset_idx': int(batch_start + i),
                    'pet_available': int(batch_available[i]),
                    'missing': int(1 - batch_available[i]),
                })

    assert sample_offset == len(pet_available_mask), 'Mask length does not match dataset size.'
    assert full_forward_count + miss_forward_count == len(pet_available_mask)
    assert miss_forward_count == int((pet_available_mask == 0).sum())

    out = metrics.compute()
    out['total_loss'] = total_loss / max(total_n, 1)
    out['full_forward_count'] = int(full_forward_count)
    out['missing_forward_count'] = int(miss_forward_count)
    out['dataset_size'] = int(len(pet_available_mask))
    return out, sample_records


def main():
    parents = [
        SegMDTConfig.ddp_parser(),
        SegMDTConfig.data_parser(),
        SegMDTConfig.model_parser(),
        SegMDTConfig.train_parser(),
        SegMDTConfig.logging_parser(),
        SegMDTConfig.task_specific_parser(),
    ]
    parser = argparse.ArgumentParser(
        description='Explicit split-route MDT evaluation with deterministic PET masks.',
        parents=parents,
        fromfile_prefix_chars='@',
    )
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--missing_rates', type=float, nargs='+', default=DEFAULT_MISSING_RATES)
    parser.add_argument('--export_only_missing_masks', type=str2bool, default=False)
    config = parser.parse_args()

    os.makedirs(config.output_dir, exist_ok=True)
    _set_seed(config.eval_random_seed)

    networks = build_mdt_seg_teacher(config)
    task = MDTSegTeacher(networks, config)
    _load_checkpoint(task, config.ckpt_path)

    loader = _build_test_loader(config)
    test_dataset = loader.dataset
    dataset_size = len(test_dataset)
    base_random = _make_nested_base_random(dataset_size, config.eval_random_seed)
    results = {}

    prev_missing_set = None
    for rate in config.missing_rates:
        rate = float(rate)
        mask = _mask_from_base_random(base_random, rate)
        missing_set = set(np.where(mask == 0)[0].tolist())
        if prev_missing_set is not None:
            assert prev_missing_set.issubset(missing_set), 'Missing masks are not nested.'
        prev_missing_set = missing_set

        rate_tag = f'random_missing_{int(round(rate * 100)):03d}pct'
        rate_dir = os.path.join(config.output_dir, rate_tag)
        os.makedirs(rate_dir, exist_ok=True)

        sample_records = []
        for i in range(dataset_size):
            sample_records.append({
                **_dataset_meta(test_dataset, i),
                'pet_available': int(mask[i]),
                'missing': int(1 - mask[i]),
                'missing_rate': rate,
                'seed': int(config.eval_random_seed),
            })

        metrics = None
        route_stats = {
            'requested_missing_rate': float(rate),
            'dataset_size': int(dataset_size),
            'missing_samples': int((mask == 0).sum()),
            'available_samples': int((mask == 1).sum()),
            'actual_missing_rate': float((mask == 0).mean()),
            'full_forward_count': 0,
            'missing_forward_count': 0,
            'nested_validation': True,
            'masked_by_same_seed': int(config.eval_random_seed),
        }

        if not config.export_only_missing_masks:
            metrics, sample_records = evaluate_with_explicit_availability_mask(
                task,
                loader,
                mask,
                threshold=0.5,
            )
            route_stats['full_forward_count'] = int(metrics['full_forward_count'])
            route_stats['missing_forward_count'] = int(metrics['missing_forward_count'])
            route_stats['actual_missing_rate'] = float(route_stats['missing_samples'] / max(dataset_size, 1))
            print('==================================================')
            print('Missing rate evaluation')
            print('==================================================')
            print(f'Requested rate: {rate:.4f}')
            print(f'Dataset samples: {dataset_size}')
            print(f"Missing samples: {route_stats['missing_samples']}")
            print(f"Available samples: {route_stats['available_samples']}")
            print(f"Actual missing rate: {route_stats['actual_missing_rate']:.4f}")
            print('')
            print('Route execution:')
            print(f"Full samples forwarded: {route_stats['full_forward_count']}")
            print(f"Missing samples forwarded: {route_stats['missing_forward_count']}")
            print('')
            assert route_stats['full_forward_count'] + route_stats['missing_forward_count'] == dataset_size
            assert route_stats['missing_forward_count'] == route_stats['missing_samples']
            if rate in (0.3, 0.5, 0.7):
                print('PG-MTR route verified for missing samples via forward_mode="missing".')
            # ensure saved samples come from the actual executed mask
            for i, rec in enumerate(sample_records):
                rec['pet_available'] = int(mask[i])
                rec['missing'] = int(1 - int(mask[i]))
                rec['missing_rate'] = rate
                rec['seed'] = int(config.eval_random_seed)
        else:
            route_stats['actual_missing_rate'] = float(route_stats['missing_samples'] / max(dataset_size, 1))

        payload = {
            'seed': int(config.eval_random_seed),
            'missing_rate': rate,
            'metrics': metrics,
            'samples': sample_records,
            'route_stats': route_stats,
        }

        with open(os.path.join(rate_dir, 'missing_samples.json'), 'w') as f:
            json.dump(sample_records, f, indent=2)
        with open(os.path.join(rate_dir, 'metrics.json'), 'w') as f:
            json.dump(payload, f, indent=2)
        results[f'{rate:.3f}'] = payload

    with open(os.path.join(config.output_dir, 'all_missing_rate_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n[done] Saved outputs to:', config.output_dir)


if __name__ == '__main__':
    main()
