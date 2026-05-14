# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.build_mdt_seg import build_mdt_seg_teacher
from models.build_student_seg import build_student_seg
from utils.metrics_seg import compute_hd95_pair


def parse_args():
    p = argparse.ArgumentParser(description='Compare teacher vs student on hard samples')
    p.add_argument('--teacher_dir', type=str, required=True)
    p.add_argument('--student_dir', type=str, required=True)
    p.add_argument('--split', type=str, default='test', choices=('val', 'test'))
    p.add_argument('--top_k', type=int, default=12)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--out_dir', type=str, default=None)
    return p.parse_args()


def _read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _to_namespace(d):
    return SimpleNamespace(**d)


def _resolve_ckpt(run_dir):
    for name in ('ckpt.best_dice.pth.tar', 'ckpt.best.pth.tar', 'ckpt.best_hd95.pth.tar', 'ckpt.last.pth.tar'):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'No checkpoint found in {run_dir}')


def _load_dataset_module():
    import importlib.util
    root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_loaders_from_teacher_cfg(cfg, batch_size, num_workers):
    dataset_mod = _load_dataset_module()
    if getattr(cfg, 'cipa_aligned', False):
        return dataset_mod.get_pclt20k_loaders_cipa_aligned(
            cfg.root, cfg.image_size_2d, batch_size, num_workers, cfg.random_state,
            pin_memory=getattr(cfg, 'pin_memory', True),
        )
    return dataset_mod.get_pclt20k_loaders(
        cfg.root, cfg.image_size_2d, batch_size, num_workers,
        val_ratio=cfg.val_ratio, random_state=cfg.random_state,
        use_case_split=getattr(cfg, 'use_case_split', True),
        pin_memory=getattr(cfg, 'pin_memory', True),
    )


def _load_teacher(run_dir, device):
    cfg = _to_namespace(_read_json(os.path.join(run_dir, 'config_args.json')))
    nets = build_mdt_seg_teacher(cfg)
    ckpt = torch.load(_resolve_ckpt(run_dir), map_location='cpu', weights_only=False)
    model = nets['model']
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    if 'ema_model' in ckpt:
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(ckpt['ema_model'], strict=False)
        ema_model.to(device).eval()
        model = ema_model
    return cfg, model


def _load_student(run_dir, device):
    cfg = _to_namespace(_read_json(os.path.join(run_dir, 'config_args.json')))
    nets = build_student_seg(cfg)
    ckpt = torch.load(_resolve_ckpt(run_dir), map_location='cpu', weights_only=False)
    model = nets['model']
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    if 'ema_model' in ckpt:
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(ckpt['ema_model'], strict=False)
        ema_model.to(device).eval()
        model = ema_model
    return cfg, model


def _safe_div(a, b):
    return 0.0 if b == 0 else a / b


def _binary_metrics(pred_bin, gt_bin):
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    dice = 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
    acc = 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp))
    hd95 = compute_hd95_pair(pred, gt)
    return float(dice), float(iou), float(acc), float(hd95)


def _to_display(img_3ch):
    img = np.transpose(img_3ch, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img.mean(axis=2)


def _overlay(gray_img, mask, color):
    rgb = np.stack([gray_img, gray_img, gray_img], axis=-1).astype(np.float32)
    mask = (mask > 0.5)[..., None].astype(np.float32)
    color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb * (1 - 0.85 * mask) + color * (0.85 * mask), 0, 1)


def _compare_overlay(gray_img, gt_mask, pred_mask):
    rgb = np.stack([gray_img, gray_img, gray_img], axis=-1).astype(np.float32)
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5
    tp = np.logical_and(gt_mask, pred_mask)
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask)
    rgb[tp] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    rgb[fn] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[fp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _save_case_figure(case, out_path, group_name=None):
    ct, pet, gt = case['ct_img'], case['pet_img'], case['gt']
    tp, sp = case['teacher_prob'], case['student_prob']
    tb, sb = case['teacher_bin'], case['student_bin']
    panels = [
        ('CT', ct, 'gray'),
        ('PET', pet, 'inferno'),
        ('GT overlay', _overlay(ct, gt, (1, 0, 0)), None),
        ('Teacher prob', tp, 'jet'),
        ('Student prob', sp, 'jet'),
        ('Teacher GT/pred', _compare_overlay(ct, gt, tb), None),
        ('Student GT/pred', _compare_overlay(ct, gt, sb), None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.reshape(2, 4)
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap else None)
        ax.set_title(title)
        ax.axis('off')
    axes.flat[-1].axis('off')
    prefix = f'group={group_name} | ' if group_name else ''
    fig.suptitle(
        prefix +
        f"sample={case['sample_id']} | T dice={case['teacher_dice']:.4f} | "
        f"S dice={case['student_dice']:.4f} | gap={case['dice_gap']:.4f} | "
        f"T hd95={case['teacher_hd95']:.2f} | S hd95={case['student_hd95']:.2f}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _assign_group_labels(rows):
    if not rows:
        return rows
    gaps = np.array([r['dice_gap'] for r in rows], dtype=np.float32)
    q1 = float(np.quantile(gaps, 1.0 / 3.0))
    q2 = float(np.quantile(gaps, 2.0 / 3.0))
    for r in rows:
        if r['dice_gap'] >= q2:
            r['group'] = 'hard'
        elif r['dice_gap'] >= q1:
            r['group'] = 'medium'
        else:
            r['group'] = 'easy'
    return rows


def _group_stats(rows, group_name):
    subset = [r for r in rows if r['group'] == group_name]
    if not subset:
        return None
    def mean(key):
        return float(np.mean([r[key] for r in subset]))
    return {
        'group': group_name,
        'count': len(subset),
        'teacher_dice_mean': mean('teacher_dice'),
        'student_dice_mean': mean('student_dice'),
        'dice_gap_mean': mean('dice_gap'),
        'teacher_hd95_mean': mean('teacher_hd95'),
        'student_hd95_mean': mean('student_hd95'),
        'hd95_gap_mean': mean('hd95_gap'),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    teacher_cfg, teacher_model = _load_teacher(args.teacher_dir, device)
    _, student_model = _load_student(args.student_dir, device)
    _, val_loader, test_loader = _build_loaders_from_teacher_cfg(teacher_cfg, args.batch_size, args.num_workers)
    loader = test_loader if args.split == 'test' else val_loader

    out_dir = args.out_dir or os.path.join(args.teacher_dir, f'compare_student_{os.path.basename(args.student_dir)}_{args.split}')
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    with torch.no_grad():
        for batch in loader:
            ct = batch['ct'].float().to(device)
            pet = batch['pet'].float().to(device)
            mask = batch['mask'].float().to(device)
            idxs = batch['idx'].cpu().numpy().tolist()
            t_out = teacher_model(ct, pet, target_size=mask.shape[-2:])
            s_out = student_model(ct, target_size=mask.shape[-2:])
            t_logit = t_out['pred'] if isinstance(t_out, dict) else t_out[0]
            s_logit = s_out['pred'] if isinstance(s_out, dict) else s_out[0]
            t_prob = torch.sigmoid(t_logit).squeeze(1).cpu().numpy()
            s_prob = torch.sigmoid(s_logit).squeeze(1).cpu().numpy()
            gt = mask.squeeze(1).cpu().numpy()
            ct_np = ct.cpu().numpy()
            pet_np = pet.cpu().numpy()
            for i, sample_id in enumerate(idxs):
                gt_bin = (gt[i] > 0.5).astype(np.uint8)
                teacher_bin = (t_prob[i] > args.threshold).astype(np.uint8)
                student_bin = (s_prob[i] > args.threshold).astype(np.uint8)
                td, ti, ta, th = _binary_metrics(teacher_bin, gt_bin)
                sd, si, sa, sh = _binary_metrics(student_bin, gt_bin)
                rows.append({
                    'sample_id': sample_id,
                    'teacher_dice': td, 'teacher_iou': ti, 'teacher_acc': ta, 'teacher_hd95': th,
                    'student_dice': sd, 'student_iou': si, 'student_acc': sa, 'student_hd95': sh,
                    'dice_gap': td - sd, 'hd95_gap': sh - th,
                    'teacher_prob': t_prob[i], 'student_prob': s_prob[i],
                    'teacher_bin': teacher_bin, 'student_bin': student_bin,
                    'gt': gt_bin, 'ct_img': _to_display(ct_np[i]), 'pet_img': _to_display(pet_np[i]),
                })

    rows.sort(key=lambda x: (x['dice_gap'], x['hd95_gap']), reverse=True)
    rows = _assign_group_labels(rows)

    all_csv_path = os.path.join(out_dir, 'all_samples_teacher_vs_student.csv')
    with open(all_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'sample_id', 'group', 'teacher_dice', 'student_dice', 'dice_gap', 'teacher_iou', 'student_iou', 'teacher_acc', 'student_acc', 'teacher_hd95', 'student_hd95', 'hd95_gap'])
        for rank, row in enumerate(rows, start=1):
            w.writerow([
                rank, row['sample_id'], row['group'],
                f"{row['teacher_dice']:.4f}", f"{row['student_dice']:.4f}", f"{row['dice_gap']:.4f}",
                f"{row['teacher_iou']:.4f}", f"{row['student_iou']:.4f}",
                f"{row['teacher_acc']:.4f}", f"{row['student_acc']:.4f}",
                f"{row['teacher_hd95']:.2f}", f"{row['student_hd95']:.2f}", f"{row['hd95_gap']:.2f}",
            ])

    hard_csv_path = os.path.join(out_dir, 'hard_samples_teacher_vs_student.csv')
    with open(hard_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'sample_id', 'group', 'teacher_dice', 'student_dice', 'dice_gap', 'teacher_hd95', 'student_hd95', 'hd95_gap'])
        for rank, row in enumerate(rows[:args.top_k], start=1):
            w.writerow([rank, row['sample_id'], row['group'], f"{row['teacher_dice']:.4f}", f"{row['student_dice']:.4f}", f"{row['dice_gap']:.4f}", f"{row['teacher_hd95']:.2f}", f"{row['student_hd95']:.2f}", f"{row['hd95_gap']:.2f}"])

    group_csv_path = os.path.join(out_dir, 'group_summary.csv')
    with open(group_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['group', 'count', 'teacher_dice_mean', 'student_dice_mean', 'dice_gap_mean', 'teacher_hd95_mean', 'student_hd95_mean', 'hd95_gap_mean'])
        for group_name in ('hard', 'medium', 'easy'):
            stats = _group_stats(rows, group_name)
            if stats is None:
                continue
            w.writerow([
                stats['group'], stats['count'],
                f"{stats['teacher_dice_mean']:.4f}", f"{stats['student_dice_mean']:.4f}", f"{stats['dice_gap_mean']:.4f}",
                f"{stats['teacher_hd95_mean']:.2f}", f"{stats['student_hd95_mean']:.2f}", f"{stats['hd95_gap_mean']:.2f}",
            ])

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for rank, row in enumerate(rows[:args.top_k], start=1):
        _save_case_figure(row, os.path.join(fig_dir, f'hard_{rank:02d}_sample_{row["sample_id"]}.png'), group_name=row['group'])

    group_fig_dir = os.path.join(out_dir, 'group_examples')
    os.makedirs(group_fig_dir, exist_ok=True)
    for group_name in ('hard', 'medium', 'easy'):
        subset = [r for r in rows if r['group'] == group_name]
        if not subset:
            continue
        rep = subset[0] if group_name == 'hard' else subset[len(subset) // 2]
        _save_case_figure(rep, os.path.join(group_fig_dir, f'{group_name}_representative_sample_{rep["sample_id"]}.png'), group_name=group_name)

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'split': args.split, 'top_k': args.top_k, 'num_samples': len(rows), 'groups': {'hard': len([r for r in rows if r['group'] == 'hard']), 'medium': len([r for r in rows if r['group'] == 'medium']), 'easy': len([r for r in rows if r['group'] == 'easy'])}}, f, indent=2)
    print(f'Saved all-sample CSV to: {all_csv_path}')
    print(f'Saved hard-sample CSV to: {hard_csv_path}')
    print(f'Saved group summary CSV to: {group_csv_path}')
    print(f'Saved hard figures to: {fig_dir}')
    print(f'Saved representative group figures to: {group_fig_dir}')


if __name__ == '__main__':
    main()


