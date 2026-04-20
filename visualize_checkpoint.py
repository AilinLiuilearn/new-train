# -*- coding: utf-8 -*-
"""Load a saved checkpoint and export diagnostic visualizations without retraining."""

import argparse
import csv
import importlib.util
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_student, build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher
from tasks.mdt_student_seg import MDTSegStudent
from utils.metrics_seg import compute_hd95_pair
from utils.vis_teacher import save_segmentation_diagnostics


def _load_dataset_module():
    dataset_path = os.path.join(os.getcwd(), 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(checkpoint_dir):
    cfg_path = os.path.join(checkpoint_dir, 'config_args.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f'未找到配置文件: {cfg_path}')
    with open(cfg_path, 'r') as f:
        raw = json.load(f)
    raw.pop('checkpoint_dir', None)
    return SegMDTConfig(args=raw)


def _resolve_mode(mode, checkpoint_dir, config):
    if mode != 'auto':
        return mode
    task = getattr(config, '_task', '') or getattr(config, 'task', '')
    if 'Student' in task or 'student' in checkpoint_dir.lower():
        return 'student'
    return 'teacher'


def _build_loaders(config, split):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'cipa_aligned', False):
        train_loader, val_loader, test_loader = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            getattr(config, 'aug_strong', False),
        )
    else:
        train_loader, val_loader, test_loader = dataset_mod.get_pclt20k_loaders(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            val_ratio=config.val_ratio,
            random_state=config.random_state,
            use_case_split=getattr(config, 'use_case_split', True),
            aug_strong=False,
        )
    if split == 'train':
        return train_loader
    if split == 'val':
        return val_loader
    return test_loader


def _build_task(config, mode):
    g0 = int(config.gpus[0]) if config.gpus else 0
    config.gpus = [g0]
    if torch.cuda.is_available():
        torch.cuda.set_device(g0)
    config.mixed_precision = False

    if mode == 'student':
        config.task = 'MDT_Student'
        task = MDTSegStudent(build_mdt_seg_student(config), config)
    else:
        config.task = 'MDT_Teacher'
        task = MDTSegTeacher(build_mdt_seg_teacher(config), config)
    return task


def _load_checkpoint(task, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for k, v in task.networks.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
    return ckpt


def _counts_to_metrics(tp, fp, fn, tn):
    denom_iou = tp + fp + fn
    denom_dice = 2 * tp + fp + fn
    iou = 1.0 if denom_iou == 0 else (tp / denom_iou)
    dice = 1.0 if denom_dice == 0 else ((2 * tp) / denom_dice)
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return {
        'dice': float(dice),
        'iou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
    }


def _safe_record(dataset, data_idx):
    if hasattr(dataset, 'records') and isinstance(dataset.records, list) and 0 <= data_idx < len(dataset.records):
        return dataset.records[data_idx]
    return {}


def _collect_case_scores(task, loader, threshold, compute_hd95):
    for v in task.networks.values():
        v.eval()

    rows = []
    dataset = loader.dataset

    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].float().to(task.device)
            pet = batch['pet'].float().to(task.device)
            mask = batch['mask'].float().to(task.device)
            outputs = task.networks['model'](ct, pet, target_size=mask.shape[-2:])
            preds = outputs['preds'] if isinstance(outputs, dict) else outputs
            logit = preds[0]
            prob = torch.sigmoid(logit)
            pred_bin = (prob > threshold).float()

            idxs = batch.get('idx')
            if idxs is None:
                idxs = torch.arange(ct.size(0))

            for b in range(ct.size(0)):
                data_idx = int(idxs[b])
                p = pred_bin[b, 0]
                g = mask[b, 0]
                tp = float((p * g).sum().item())
                fp = float((p * (1 - g)).sum().item())
                fn = float(((1 - p) * g).sum().item())
                tn = float(((1 - p) * (1 - g)).sum().item())
                m = _counts_to_metrics(tp, fp, fn, tn)
                rec = _safe_record(dataset, data_idx)
                row = {
                    'dataset_idx': data_idx,
                    'image_id': rec.get('image_id', str(data_idx)),
                    'case_id': rec.get('case_id', ''),
                    'gt_pixels': float(g.sum().item()),
                    'pred_pixels': float(p.sum().item()),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    **m,
                }
                if compute_hd95:
                    row['hd95'] = float(compute_hd95_pair((p > 0.5).cpu().numpy(), (g > 0.5).cpu().numpy()))
                rows.append(row)

    return rows


def _write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _rank_rows(rows, metric, descending=True):
    return sorted(rows, key=lambda x: x.get(metric, 0.0), reverse=descending)


def _subset_loader_from_indices(loader, selected_indices):
    subset = Subset(loader.dataset, selected_indices)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _export_ranked_visuals(task, loader, rows, out_dir, top_k, bottom_k, metric, threshold):
    if metric == 'hd95':
        sorted_rows = _rank_rows(rows, metric, descending=False)
        worst_rows = _rank_rows(rows, metric, descending=True)[:bottom_k]
        best_rows = sorted_rows[:top_k]
    else:
        sorted_rows = _rank_rows(rows, metric, descending=True)
        best_rows = sorted_rows[:top_k]
        worst_rows = list(reversed(sorted_rows))[:bottom_k]

    best_indices = [int(r['dataset_idx']) for r in best_rows]
    worst_indices = [int(r['dataset_idx']) for r in worst_rows]

    best_loader = _subset_loader_from_indices(loader, best_indices)
    worst_loader = _subset_loader_from_indices(loader, worst_indices)

    best_out = os.path.join(out_dir, f'best_{top_k}_{metric}')
    worst_out = os.path.join(out_dir, f'worst_{bottom_k}_{metric}')

    save_segmentation_diagnostics(task, best_loader, best_out, num_samples=len(best_indices), threshold=threshold)
    save_segmentation_diagnostics(task, worst_loader, worst_out, num_samples=len(worst_indices), threshold=threshold)

    _write_csv(best_rows, os.path.join(out_dir, f'best_{top_k}_{metric}.csv'))
    _write_csv(worst_rows, os.path.join(out_dir, f'worst_{bottom_k}_{metric}.csv'))


def _summarize_rows(rows):
    if not rows:
        return {}
    n = len(rows)

    def _avg(vals):
        return float(sum(vals) / max(len(vals), 1))

    dices = [float(r['dice']) for r in rows]
    ious = [float(r['iou']) for r in rows]
    sens = [float(r['sensitivity']) for r in rows]
    prec = [float(r['precision']) for r in rows]
    miss = [1 for r in rows if float(r['tp']) == 0.0 and float(r['gt_pixels']) > 0.0]

    return {
        'n': n,
        'mean_dice': _avg(dices),
        'mean_iou': _avg(ious),
        'mean_sensitivity': _avg(sens),
        'mean_precision': _avg(prec),
        'miss_count': int(sum(miss)),
        'miss_rate': float(sum(miss) / n),
    }


def _size_bucket(gt_pixels):
    g = float(gt_pixels)
    if g <= 50:
        return 'tiny(<=50)'
    if g <= 200:
        return 'small(51-200)'
    if g <= 1000:
        return 'medium(201-1000)'
    return 'large(>1000)'


def _write_size_stratified_summary(rows, out_dir):
    groups = {
        'tiny(<=50)': [],
        'small(51-200)': [],
        'medium(201-1000)': [],
        'large(>1000)': [],
    }
    for r in rows:
        groups[_size_bucket(r['gt_pixels'])].append(r)

    summary = {'overall': _summarize_rows(rows), 'by_size': {k: _summarize_rows(v) for k, v in groups.items()}}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'size_stratified_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    summary_rows = []
    for name in ('overall', 'tiny(<=50)', 'small(51-200)', 'medium(201-1000)', 'large(>1000)'):
        s = summary['overall'] if name == 'overall' else summary['by_size'][name]
        summary_rows.append({'group': name, **s})
    _write_csv(summary_rows, os.path.join(out_dir, 'size_stratified_summary.csv'))



def main():
    parser = argparse.ArgumentParser(description='加载 checkpoint 导出可视化与 best/worst 排名分析')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='训练输出目录')
    parser.add_argument('--ckpt', type=str, default='ckpt.best_dice.pth.tar', help='权重文件名')
    parser.add_argument('--mode', type=str, default='auto', choices=('auto', 'teacher', 'student'))
    parser.add_argument('--split', type=str, default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--num_samples', type=int, default=12)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out_dir', type=str, default=None)

    parser.add_argument('--analyze_cases', action='store_true', help='是否导出全量样本评分与 best/worst 可视化')
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--bottom_k', type=int, default=100)
    parser.add_argument('--rank_metric', type=str, default='dice', choices=('dice', 'iou', 'sensitivity', 'precision', 'hd95'))
    parser.add_argument('--compute_hd95', action='store_true', help='是否计算每张图 HD95（较慢）')
    args = parser.parse_args()

    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    config = _load_config(args.checkpoint_dir)
    mode = _resolve_mode(args.mode, args.checkpoint_dir, config)
    loader = _build_loaders(config, args.split)
    task = _build_task(config, mode)

    ckpt_path = os.path.join(args.checkpoint_dir, args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'未找到权重文件: {ckpt_path}')
    _load_checkpoint(task, ckpt_path)

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, f'vis_{args.split}_{mode}')

    if not args.analyze_cases:
        save_segmentation_diagnostics(
            task=task,
            loader=loader,
            out_dir=out_dir,
            num_samples=args.num_samples,
            threshold=args.threshold,
        )
        print(f'[visualize_checkpoint] mode={mode} split={args.split} out_dir={out_dir}')
        return

    rows = _collect_case_scores(task, loader, args.threshold, args.compute_hd95 or args.rank_metric == 'hd95')
    _write_csv(rows, os.path.join(out_dir, 'case_scores_all.csv'))
    _export_ranked_visuals(task, loader, rows, out_dir, args.top_k, args.bottom_k, args.rank_metric, args.threshold)
    _write_size_stratified_summary(rows, out_dir)
    print(f'[visualize_checkpoint] case analysis done. out_dir={out_dir}')


if __name__ == '__main__':
    main()
