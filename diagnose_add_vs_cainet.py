#!/usr/bin/env python
"""Paired Full-only checkpoint diagnosis; never trains or changes checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from datasets.pclt20k_seg import PCLT20KSegDataset, _read_list, _records_from_ids
from models.build_mdt_seg import build_dual_model
from models.full_cainet_fusion import build_full_cainet_model
from tasks.mdt_seg import MDTSegTeacher
from utils.metrics_seg import compute_hd95_pair


DEFAULT_BASE = '/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT'
COUNT_KEYS = ('tp', 'fp', 'fn', 'tn')
PROTOCOL_KEYS = ('norm_mode', 'image_size_2d', 'ct_backbone', 'pet_backbone',
                 'decoder_channels', 'train_split_file', 'val_split_file',
                 'test_split_file', 'aug_mode', 'random_state', 'learning_rate',
                 'weight_decay', 'epochs', 'cosine_warmup', 'lr_flat_ratio',
                 'early_stop_patience', 'ema_enabled', 'ema_decay', 'batch_size')


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    best = path / 'ckpt.best_full.pth.tar'
    if best.is_file():
        return best
    choices = sorted(p.name for p in path.glob('*') if p.is_file()) if path.is_dir() else []
    raise FileNotFoundError(f'{best} not found. Pass an exact checkpoint file. Found: {choices}')


def select_weights(ckpt, choice):
    if choice == 'auto':
        choice = ckpt.get('eval_weights')
        if choice is None:
            if ckpt.get('model_ema') is not None:
                raise ValueError('Ambiguous raw/EMA selection: specify --add_weights/--fusion_weights')
            choice = 'raw'
    if choice not in ('raw', 'ema'):
        raise ValueError(f'Invalid checkpoint eval_weights: {choice!r}')
    state = ckpt.get('model_ema' if choice == 'ema' else 'model')
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Checkpoint does not contain {choice} state dict')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state, choice


def checkpoint_config(ckpt, path):
    saved = ckpt.get('config')
    source = 'checkpoint.config'
    if saved is None:
        for name in ('config_args.json', 'configs.json'):
            candidate = path.parent / name
            if candidate.is_file():
                saved = json.loads(candidate.read_text())
                source = str(candidate)
                break
    if not isinstance(saved, dict):
        raise ValueError(f'{path}: missing config and adjacent config_args.json/configs.json')
    saved = dict(saved)
    required = ('norm_mode', 'image_size_2d', 'ct_backbone', 'pet_backbone', 'decoder_channels')
    missing = [k for k in required if k not in saved]
    if missing:
        raise ValueError(f'{path}: missing explicit inference settings: {missing}')
    if saved['norm_mode'] not in ('imagenet', 'cipa'):
        raise ValueError(f'Unknown saved norm_mode: {saved["norm_mode"]}')
    return saved, source


def load_model(path, kind, choice='auto', device='cpu'):
    path = resolve_checkpoint(path)
    ckpt = MDTSegTeacher.load_state_dicts(str(path))
    state, tag = select_weights(ckpt, choice)
    saved, source = checkpoint_config(ckpt, path)
    config = dict(saved, pretrained=False, ct_pretrained_path=None, pet_pretrained_path=None)
    has_alignment = any(k.startswith('pet_align.') for k in state)
    fusion_keys = any(k.startswith('fusion.') for k in state)
    if kind == 'add':
        if fusion_keys or saved.get('fusion_type') not in (None, 'AddFusion'):
            raise ValueError('The Add checkpoint contains a learned fusion or incompatible fusion_type')
        if saved.get('experiment_type') not in (None, 'full_add_baseline', 'full_cainet_fusion'):
            raise ValueError('Expected a Full-only Add checkpoint, not a CT-only or mixed checkpoint')
        model = build_dual_model(SimpleNamespace(**config))['model']
    else:
        if saved.get('fusion_type') != 'CAINetFusion':
            raise ValueError('Expected fusion_type=CAINetFusion in the fusion checkpoint')
        for key in ('cacr_enabled', 'gcm_enabled'):
            if not isinstance(saved.get(key), bool):
                raise ValueError(f'Explicit saved boolean {key} is required')
        if saved['cacr_enabled'] and 'cacr_stages' not in config:
            first = state.get('fusion.cacr.0.theta.weight')
            if first is None or first.shape[1] not in (128, 256):
                raise ValueError('Cannot infer legacy CACR stages; add saved cacr_stages to config')
            config['cacr_stages'] = (1, 2, 3, 4) if first.shape[1] == 128 else (2, 3, 4)
        model = build_full_cainet_model(SimpleNamespace(**config))['model']
    if not has_alignment:
        # Exact old clean-group forward: PET encoder features go straight to fusion.
        # Strict loading below still rejects any other missing/unexpected tensor.
        model.pet_align = torch.nn.Identity()
    model.load_state_dict(state, strict=True)
    for key, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite checkpoint tensor: {key}')
    model.requires_grad_(False).to(device).eval()
    metadata = dict(checkpoint=str(path), weights=tag, epoch=ckpt.get('epoch'),
                    best_epoch=ckpt.get('best_epoch'), recorded_val=ckpt.get('val'),
                    config_source=source, config=saved, pet_alignment=has_alignment,
                    checkpoint_bytes=path.stat().st_size)
    del ckpt, state
    return model, saved, metadata


def safe_div(a, b, empty=0.0):
    return float(a / b) if b else float(empty)


def counts(pred, gt):
    return dict(tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()),
                fn=int((~pred & gt).sum()), tn=int((~pred & ~gt).sum()))


def scores(c):
    return dict(dice=safe_div(2*c['tp'], 2*c['tp']+c['fp']+c['fn'], 1),
                iou=safe_div(c['tp'], c['tp']+c['fp']+c['fn'], 1),
                precision=safe_div(c['tp'], c['tp']+c['fp']),
                recall=safe_div(c['tp'], c['tp']+c['fn']))


def size_bin(area, limits):
    if area == 0:
        return 'empty'
    for label, limit in zip(('small', 'medium', 'large'), limits):
        if area <= limit:
            return label
    return 'very_large'


def boundary_f1(pred, gt, tolerance):
    pb = pred & ~ndi.binary_erosion(pred)
    gb = gt & ~ndi.binary_erosion(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    precision = (ndi.distance_transform_edt(~gb)[pb] <= tolerance).mean()
    recall = (ndi.distance_transform_edt(~pb)[gb] <= tolerance).mean()
    return safe_div(2*precision*recall, precision+recall)


def describe_mask(pred, gt, ring, outer, interior, hot, tolerance, hd95_mode):
    c = counts(pred, gt)
    out = dict(c, **scores(c), pred_area=int(pred.sum()),
               missed_positive=int(gt.any() and not pred.any()),
               boundary_f1=boundary_f1(pred, gt, tolerance),
               boundary_fp=int((pred & ~gt & ring).sum()),
               boundary_fn=int((~pred & gt & ring).sum()),
               far_fp=int((pred & ~outer).sum()),
               interior_fn=int((~pred & interior).sum()),
               pet_hot_fp=int((pred & ~gt & hot).sum()))
    # Official CIPA HD95 assumes positive GT for its one-empty fallback.
    # Keep negative-GT false positives evaluable with an explicit tagged penalty.
    if hd95_mode == 'cipa_official' and not gt.any() and pred.any():
        out['hd95'] = math.hypot(*gt.shape)
        out['hd95_negative_gt_extension'] = 1
    else:
        out['hd95'] = float(compute_hd95_pair(pred, gt, hd95_mode))
        out['hd95_negative_gt_extension'] = 0
    return out


def compare_masks(gt, add, fusion, cf, pet, radius=2, tolerance=2,
                  limits=(64, 256, 1024), lesion_overlap=0.1, hd95_mode='surface'):
    gt, add, fusion, cf = [np.asarray(x, dtype=bool) for x in (gt, add, fusion, cf)]
    outer = ndi.binary_dilation(gt, iterations=radius)
    interior = ndi.binary_erosion(gt, iterations=radius)
    ring = outer & ~interior
    pet = np.asarray(pet, dtype=np.float32)
    hot = pet > np.quantile(pet, .95)
    row = dict(gt_area=int(gt.sum()), area_group=size_bin(int(gt.sum()), limits))
    for name, pred in (('add', add), ('fusion', fusion), ('cf_add', cf)):
        for key, value in describe_mask(pred, gt, ring, outer, interior, hot,
                                        tolerance, hd95_mode).items():
            row[f'{name}_{key}'] = value
    row.update(delta_dice=row['fusion_dice']-row['add_dice'],
               delta_boundary_f1=row['fusion_boundary_f1']-row['add_boundary_f1'],
               cf_delta_dice=row['fusion_dice']-row['cf_add_dice'],
               disagreement_pixels=int((add != fusion).sum()),
               corrected_fn=int((gt & ~add & fusion).sum()),
               introduced_fn=int((gt & add & ~fusion).sum()),
               corrected_fp=int((~gt & add & ~fusion).sum()),
               introduced_fp=int((~gt & ~add & fusion).sum()))
    labels, n = ndi.label(gt)  # 2D, 4-connectivity; not 3D clinical lesion counting.
    lesions = []
    for label, box in enumerate(ndi.find_objects(labels), 1):
        component = labels[box] == label
        area = int(component.sum())
        item = dict(component=label, gt_area=area, area_group=size_bin(area, limits))
        for name, pred in (('add', add), ('fusion', fusion), ('cf_add', cf)):
            overlap = int((pred[box] & component).sum())
            item[f'{name}_coverage'] = overlap / area
            item[f'{name}_detected'] = int(overlap > 0 and overlap/area >= lesion_overlap)
        lesions.append(item)
    row['gt_components'] = n
    return row, lesions


def feature_statistics(ct, pet, fused, target, image_ids):
    rows = []
    for scale, (c, p, f) in enumerate(zip(ct, pet, fused), 1):
        base = (c + p).float()
        delta = f.float() - base
        occupancy = F.interpolate(target.float(), size=c.shape[-2:], mode='area')
        for b, image_id in enumerate(image_ids):
            item = dict(image_id=image_id, scale=scale)
            for name, weight in (('all', torch.ones_like(occupancy[b])),
                                 ('fg', occupancy[b]), ('bg', 1-occupancy[b])):
                numerator = (delta[b].square()*weight).sum()
                denominator = (base[b].square()*weight).sum()
                item[f'{name}_delta_relative_l2'] = (
                    float(torch.sqrt(numerator / denominator)) if denominator > 0 else None)
            item['pet_ct_l2_ratio'] = float(p[b].float().norm() / c[b].float().norm().clamp_min(1e-12))
            rows.append(item)
    return rows


def aggregate(rows, prefix):
    c = {key: sum(row[f'{prefix}_{key}'] for row in rows) for key in COUNT_KEYS}
    out = dict(c, **scores(c), samples=len(rows))
    for key in ('dice', 'boundary_f1', 'hd95'):
        out[f'mean_slice_{key}'] = float(np.mean([r[f'{prefix}_{key}'] for r in rows]))
    for key in ('missed_positive', 'boundary_fp', 'boundary_fn', 'far_fp',
                'interior_fn', 'pet_hot_fp', 'hd95_negative_gt_extension'):
        out[key] = sum(r[f'{prefix}_{key}'] for r in rows)
    return out


def summarize(rows, lesions, features, metadata, epsilon):
    out = dict(metadata=metadata, models={name: aggregate(rows, name)
                                         for name in ('add', 'fusion', 'cf_add')})
    cases = defaultdict(list)
    for row in rows:
        cases[row['case_id']].append(row)
    case_rows = []
    for case, group in sorted(cases.items()):
        item = dict(case_id=case, slices=len(group))
        for name in ('add', 'fusion', 'cf_add'):
            for key, value in aggregate(group, name).items():
                item[f'{name}_{key}'] = value
        item['delta_dice'] = item['fusion_dice'] - item['add_dice']
        case_rows.append(item)
    out['paired'] = dict(
        improved_slices=sum(r['delta_dice'] > epsilon for r in rows),
        harmed_slices=sum(r['delta_dice'] < -epsilon for r in rows),
        tied_slices=sum(abs(r['delta_dice']) <= epsilon for r in rows),
        improved_cases=sum(r['delta_dice'] > epsilon for r in case_rows),
        harmed_cases=sum(r['delta_dice'] < -epsilon for r in case_rows),
        mean_absolute_probability_change=float(np.mean([r['probability_mae'] for r in rows])),
        disagreement_fraction=safe_div(sum(r['disagreement_pixels'] for r in rows),
                                       sum(r['pixels'] for r in rows)),
        **{key: sum(r[key] for r in rows) for key in
           ('corrected_fn', 'introduced_fn', 'corrected_fp', 'introduced_fp')})
    out['slice_area_groups'] = {}
    for group in ('empty', 'small', 'medium', 'large', 'very_large'):
        subset = [r for r in rows if r['area_group'] == group]
        if subset:
            out['slice_area_groups'][group] = {name: aggregate(subset, name)
                                              for name in ('add', 'fusion', 'cf_add')}
    out['lesion_groups'] = {}
    for group in ('small', 'medium', 'large', 'very_large'):
        subset = [r for r in lesions if r['area_group'] == group]
        if subset:
            out['lesion_groups'][group] = dict(components=len(subset), **{
                name+'_detected_fraction': np.mean([r[name+'_detected'] for r in subset]).item()
                for name in ('add', 'fusion', 'cf_add')})
    out['feature_scales'] = {}
    for scale in range(1, 5):
        subset = [r for r in features if r['scale'] == scale]
        stats = {}
        for key in ('all_delta_relative_l2', 'fg_delta_relative_l2',
                    'bg_delta_relative_l2', 'pet_ct_l2_ratio'):
            values = [r[key] for r in subset if r[key] is not None]
            stats[key] = float(np.mean(values)) if values else None
        out['feature_scales'][str(scale)] = stats
    for name in ('add', 'fusion', 'cf_add'):
        out['models'][name]['mean_case_dice'] = float(np.mean([r[name+'_dice'] for r in case_rows]))
    return out, case_rows


class PairedDataset(Dataset):
    def __init__(self, records, size, add_norm, fusion_norm):
        self.add = PCLT20KSegDataset(records, size, train=False, aug_mode='none', norm_mode=add_norm)
        self.fusion = PCLT20KSegDataset(records, size, train=False, aug_mode='none', norm_mode=fusion_norm)

    def __len__(self):
        return len(self.add)

    def __getitem__(self, idx):
        a, f = self.add[idx], self.fusion[idx]
        if a['image_id'] != f['image_id'] or not torch.equal(a['mask'], f['mask']):
            raise RuntimeError('Paired sample/mask mismatch')
        return dict(add_ct=a['ct'], add_pet=a['pet'], fusion_ct=f['ct'], fusion_pet=f['pet'],
                    mask=a['mask'], image_id=a['image_id'], case_id=a['case_id'])


def denormalize(tensor, norm):
    array = tensor[0].cpu().numpy()
    array = array*.229+.485 if norm == 'imagenet' else (array+1.6)/3.2
    return (array.clip(0, 1)*255).round().astype(np.uint8)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_panel(path, sample):
    ct, pet, gt, a, f, cf, row = sample
    panels = []
    for label, gray, pred in [('CT', ct, None), ('PET', pet, None), ('GT', ct, gt),
                              ('Add', ct, a), ('CAINet', ct, f), ('Same-weights Add', ct, cf)]:
        rgb = np.repeat(gray[..., None], 3, -1).copy()
        if pred is not None:
            edge = pred & ~ndi.binary_erosion(pred)
            rgb[edge] = (0, 255, 0)
        tile = Image.fromarray(rgb).resize((256, 256))
        panel = Image.new('RGB', (256, 282), 'black')
        panel.paste(tile, (0, 26)); ImageDraw.Draw(panel).text((6, 6), label, fill='white')
        panels.append(panel)
    errors = np.repeat(ct[..., None], 3, -1).copy()
    errors[gt & ~a & f] = (0, 255, 0)
    errors[~gt & a & ~f] = (0, 220, 255)
    errors[gt & a & ~f] = (255, 0, 0)
    errors[~gt & ~a & f] = (255, 0, 255)
    tile = Image.new('RGB', (256, 282), 'black')
    tile.paste(Image.fromarray(errors).resize((256, 256)), (0, 26))
    ImageDraw.Draw(tile).text((6, 6), 'Changed errors (see report)', fill='white')
    panels.append(tile)
    canvas = Image.new('RGB', (256*4, 282*2+32), 'black')
    ImageDraw.Draw(canvas).text((6, 6),
        f'{row["image_id"]} Add={row["add_dice"]:.4f} Fusion={row["fusion_dice"]:.4f} delta={row["delta_dice"]:+.4f}', fill='white')
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % 4)*256, 32+(i//4)*282))
    canvas.save(path)


def report(summary, rows, output):
    models, paired = summary['models'], summary['paired']
    lines = ['# Add 与 CAINet checkpoint 成对诊断', '',
             '本报告描述已训练模型的差异；不据此宣称框架上限或确定训练因果。', '',
             '| 指标 | Add | CAINet | CAINet 同权重临时相加 |',
             '|---|---:|---:|---:|']
    for title, key in [('全局 Dice', 'dice'), ('病例平均 Dice（按病例汇总像素）', 'mean_case_dice'),
                       ('切片平均 Dice', 'mean_slice_dice'), ('边界 F1', 'mean_slice_boundary_f1'),
                       ('HD95 / pixel', 'mean_slice_hd95'), ('整张漏检的阳性切片', 'missed_positive'),
                       ('远离 GT 的假阳性像素', 'far_fp'), ('边界假阳性像素', 'boundary_fp'),
                       ('边界假阴性像素', 'boundary_fn')]:
        lines.append('| '+title+' | '+' | '.join(f'{models[n][key]:.6f}' for n in models)+' |')
    lines += ['', f'改善切片 {paired["improved_slices"]}，退化切片 {paired["harmed_slices"]}，'
              f'持平切片 {paired["tied_slices"]}；差值阈值见 summary.json。',
              f'修复 FN/FP：{paired["corrected_fn"]}/{paired["corrected_fp"]} 像素；'
              f'新增 FN/FP：{paired["introduced_fn"]}/{paired["introduced_fp"]} 像素。',
              '', '## 二维病灶连通域检出', '',
              '| 面积分组 | 连通域数量 | Add 检出率 | CAINet 检出率 | 同权重相加检出率 |',
              '|---|---:|---:|---:|---:|']
    for group, values in summary['lesion_groups'].items():
        lines.append(f'| {group} | {values["components"]} | '+
                     ' | '.join(f'{values[n+"_detected_fraction"]:.4f}' for n in models)+' |')
    lines += ['', '## 融合对特征的实际改变量', '',
              '相对 L2 为 ||融合输出−(CT+PET)|| / ||CT+PET||；前景按面积占比分配到各尺度。', '',
              '| 尺度 | 全图相对变化 | 前景相对变化 | 背景相对变化 | PET/CT L2 |',
              '|---|---:|---:|---:|---:|']
    for scale, values in summary['feature_scales'].items():
        cells = [f'{values[k]:.5f}' if values[k] is not None else 'NA' for k in
                 ('all_delta_relative_l2', 'fg_delta_relative_l2', 'bg_delta_relative_l2', 'pet_ct_l2_ratio')]
        lines.append('| S'+scale+' | '+' | '.join(cells)+' |')
    lines += ['', '## 配置与可比性', '']
    warnings = summary['metadata']['comparison_notes']
    lines += ['- '+note for note in warnings] or ['- 已检查的配置字段一致。']
    lines += ['', '## 如何据此选择下一步', '',
        '- 先确认全局 Dice 能复现原记录。不能复现时，先核对 split、权重选择和归一化，暂不解释模块优劣。',
        '- per_lesion.csv / lesion_groups：小连通域检出率下降时，检查浅层交互是否损害小目标；这是二维连通域，不是三维独立肿瘤。',
        '- boundary_fp / boundary_fn 与 boundary_f1：确定轮廓外扩或侵蚀是否集中发生。若证据集中在边界，再测试局部边界约束。',
        '- far_fp 与 pet_hot_fp：查看远处假阳性是否增加、是否与 PET 高亮区域重合。PET 高亮仅为输入最高 5% 灰度代理，不能直接解释为病理热点或配准错误。',
        '- 改善和退化病例同时较多：平均分可能掩盖收益抵消；优先检查退化病例，而不是继续增加模块复杂度。',
        '- feature_scales.csv：相对变化接近零只能说明该层改动小；改动大也不代表有效。结合前景/背景、实际预测和误差变化判断。',
        '- 同权重临时相加：只替换 CAINet 中间融合输出，保留其编码器、对齐和解码器。'
        '它不等于训练后的 Add，也不是正式重训消融；存在分布变化，只用于定位当前融合运算的即时影响。',
        '', '## 指标约定', '',
        '- 正式主指标为汇总 TP/FP/FN 的全局 Dice；不使用切片 Dice 平均替代它。空 GT 且空预测的切片 Dice 为 1。',
        '- 病例 Dice 按该病例全部二维切片汇总计数，不计算三维边界或体素间距。',
        '- 边界容差与区域带宽按像素计算；二维连通域检出阈值、大小分箱、阈值设置见 metadata。',
        '- CIPA official HD95 对空 GT 且非空预测未定义可靠分支；本脚本对该情形用图像对角线惩罚，'
        '在 hd95_negative_gt_extension 单独计数。其余调用仓库原指标。',
        '- 默认按每个 checkpoint 记录的 eval_weights 选择 raw/EMA，分别保留其训练时归一化，不搜索最优阈值。',
        '', '## 可视化', '',
        '绿色轮廓为各面板自身 mask。最后一面板：绿色=修复漏检，青色=修复假阳性，红色=新增漏检，紫色=新增假阳性。',
        'top_improved/ 与 top_harmed/ 保存差值排序前若干张切片；完整结果以 CSV 为准。',
        '', '本报告给出观察依据和待验证方向，不自动宣布某种融合设计一定提高性能。']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add_checkpoint', default=DEFAULT_BASE+'/full-add-baseline')
    parser.add_argument('--fusion_checkpoint', default=DEFAULT_BASE+'/full-cainet-s1s4-both-imagenet-ema-s2023')
    parser.add_argument('--root', default=None)
    parser.add_argument('--split_file', default=None, help='Default: fusion checkpoint test_split_file')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--add_weights', choices=('auto', 'raw', 'ema'), default='auto')
    parser.add_argument('--fusion_weights', choices=('auto', 'raw', 'ema'), default='auto')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--hd95_mode', choices=('cipa_official', 'surface', 'legacy'), default='cipa_official')
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--boundary_radius', type=int, default=2)
    parser.add_argument('--boundary_tolerance', type=float, default=2)
    parser.add_argument('--lesion_overlap', type=float, default=.1)
    parser.add_argument('--area_bins', nargs=3, type=int, default=(64, 256, 1024))
    parser.add_argument('--delta_epsilon', type=float, default=.001)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=None, help='Smoke only; report marks partial evaluation')
    parser.add_argument('--save_predictions', action='store_true', help='Save binary masks as compressed NPZ')
    args = parser.parse_args(argv)
    if not (0 < args.threshold < 1 and 0 < args.lesion_overlap <= 1):
        parser.error('threshold must be in (0,1), lesion_overlap in (0,1]')
    if (args.batch_size < 1 or args.num_workers < 0 or args.boundary_radius < 1
            or args.boundary_tolerance < 0 or args.delta_epsilon < 0 or args.top_k < 0
            or (args.max_samples is not None and args.max_samples < 1)
            or not (0 < args.area_bins[0] < args.area_bins[1] < args.area_bins[2])):
        parser.error('Invalid batch/worker/boundary/area/top_k/max_samples argument')
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be new or empty: {output}')
    device = torch.device(args.device)
    torch.manual_seed(2023); np.random.seed(2023)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    add, acfg, ameta = load_model(args.add_checkpoint, 'add', args.add_weights, device)
    fusion, fcfg, fmeta = load_model(args.fusion_checkpoint, 'fusion', args.fusion_weights, device)
    if acfg['image_size_2d'] != fcfg['image_size_2d']:
        raise ValueError('Different saved image sizes: cannot silently compare different geometry')
    size = int(fcfg['image_size_2d'])
    if args.hd95_mode == 'cipa_official':
        if size != 512:
            raise ValueError('cipa_official requires 512x512; use --hd95_mode surface for smoke data')
        from medpy.metric.binary import hd95  # fail before inference if dependency is missing
    root = Path(args.root or fcfg['root']).expanduser().resolve()
    if args.root is None and Path(acfg['root']).resolve() != root:
        raise ValueError('Saved dataset roots differ; pass --root explicitly')
    split_file = args.split_file or fcfg.get('test_split_file')
    if not split_file:
        raise ValueError('Pass --split_file; no saved test_split_file')
    split_path = root/split_file
    ids = _read_list(str(split_path))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Evaluation split is missing, empty, or contains duplicate IDs')
    all_ids = list(ids)
    if args.max_samples is not None:
        ids = ids[:args.max_samples]
    records = _records_from_ids(str(root), ids)
    notes = []
    differences = {k: dict(add=acfg.get(k), fusion=fcfg.get(k))
                   for k in PROTOCOL_KEYS if acfg.get(k) != fcfg.get(k)}
    if differences:
        notes.append('训练配置存在差异（见 summary.json metadata.protocol_differences）；不能将差异全部归因于融合。')
    if ameta['pet_alignment'] != fmeta['pet_alignment']:
        notes.append('Add 与 CAINet 的 PET 对齐结构不同：已按权重恢复旧/新结构，此对比含结构混杂因素。')
    if ameta['weights'] != fmeta['weights']:
        notes.append('两个实验记录的 raw/EMA 选择不同；当前分别遵循原选择。')
    for name, cfg in [('Add', acfg), ('CAINet', fcfg)]:
        for key in ('val_split_file', 'test_split_file'):
            if cfg.get(key) and (root/cfg[key]).is_file():
                prior_ids = _read_list(str(root/cfg[key]))
                if set(prior_ids) != set(all_ids):
                    notes.append(f'{name} 的 {key} 与当前评估样本不一致，不能直接要求复现对应原指标。')
        train_path = root/cfg.get('train_split_file', 'train_original.txt')
        train_ids = _read_list(str(train_path))
        if train_ids:
            overlap = {x.split('_')[0] for x in ids} & {x.split('_')[0] for x in train_ids}
            if overlap:
                notes.append(f'{name} 当前评估含 {len(overlap)} 个训练病例：只能作为训练集诊断。')
        else:
            notes.append(f'{name} 训练 split 不可读，未核实训练/评估病例重叠。')
    if len(ids) < len(all_ids):
        notes.append(f'仅评估 {len(ids)}/{len(all_ids)} 张切片，不代表完整 split 指标。')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                          stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = dict(add=ameta, fusion=fmeta, protocol_differences=differences,
                    comparison_notes=notes, root=str(root), split_file=str(split_path),
                    split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    sample_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                    samples=len(ids), full_split_samples=len(all_ids), arguments=vars(args),
                    image_size=size, git_commit=commit, torch=torch.__version__,
                    python=platform.python_version(), inference='float32, no autocast',
                    threshold_comparison='sigmoid(logits) > threshold',
                    counterfactual='CAINet own aligned features -> Add -> CAINet own decoder')
    output.mkdir(parents=True, exist_ok=True)
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    dataset = PairedDataset(records, size, acfg['norm_mode'], fcfg['norm_mode'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    captured = {}
    def hook(module, inputs, result):
        captured.update(ct=inputs[0], pet=inputs[1], fused=result)
    handle = fusion.fusion.register_forward_hook(hook)
    rows, lesions, features, bank = [], [], [], {}
    if args.save_predictions:
        (output/'predictions').mkdir()
    try:
        with torch.inference_mode():
            for batch in loader:
                target = batch['mask'].to(device)
                a = add(batch['add_ct'].to(device), batch['add_pet'].to(device),
                        forward_mode='full')['logits']
                f = fusion(batch['fusion_ct'].to(device), batch['fusion_pet'].to(device),
                           forward_mode='full')['logits']
                cf = fusion._decode([c+p for c, p in zip(captured['ct'], captured['pet'])],
                                    target.shape[-2:])['logits']
                features.extend(feature_statistics(captured['ct'], captured['pet'],
                                                    captured['fused'], target, batch['image_id']))
                captured.clear()
                if any(x.shape != target.shape or not torch.isfinite(x).all() for x in (a, f, cf)):
                    raise RuntimeError('Nonfinite prediction or prediction/GT shape mismatch')
                pa, pf, pc = [x.sigmoid().cpu().numpy()[:, 0] for x in (a, f, cf)]
                for i, image_id in enumerate(batch['image_id']):
                    gt = batch['mask'][i, 0].numpy().astype(bool)
                    ba, bf, bc = [p[i] > args.threshold for p in (pa, pf, pc)]
                    ct = denormalize(batch['fusion_ct'][i], fcfg['norm_mode'])
                    pet = denormalize(batch['fusion_pet'][i], fcfg['norm_mode'])
                    row, component_rows = compare_masks(gt, ba, bf, bc, pet,
                        args.boundary_radius, args.boundary_tolerance, args.area_bins,
                        args.lesion_overlap, args.hd95_mode)
                    row = dict(image_id=image_id, case_id=batch['case_id'][i], **row,
                               pixels=gt.size, probability_mae=float(np.abs(pa[i]-pf[i]).mean()),
                               cf_probability_mae=float(np.abs(pc[i]-pf[i]).mean()))
                    rows.append(row)
                    lesions.extend(dict(image_id=image_id, case_id=batch['case_id'][i], **r)
                                   for r in component_rows)
                    if args.save_predictions:
                        filename = hashlib.sha256(image_id.encode()).hexdigest()[:16]+'.npz'
                        np.savez_compressed(output/'predictions'/filename, image_id=image_id,
                                            gt=gt, add=ba, fusion=bf, cf_add=bc)
                    selected = sorted(rows, key=lambda r: r['delta_dice'])
                    keep = {r['image_id'] for r in selected[:args.top_k]+selected[-args.top_k:]} if args.top_k else set()
                    if image_id in keep:
                        bank[image_id] = (ct, pet, gt, ba, bf, bc, row)
                    for key in list(bank):
                        if key not in keep:
                            del bank[key]
                print(f'[DIAG] {len(rows)}/{len(ids)} samples', flush=True)
    finally:
        handle.remove()
    summary, cases = summarize(rows, lesions, features, metadata, args.delta_epsilon)
    write_csv(output/'per_slice.csv', rows)
    write_csv(output/'per_case.csv', cases)
    write_csv(output/'per_lesion.csv', lesions)
    write_csv(output/'feature_scales.csv', features)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                 allow_nan=False), encoding='utf-8')
    for name, selected in [('top_improved', sorted(rows, key=lambda r: -r['delta_dice'])),
                           ('top_harmed', sorted(rows, key=lambda r: r['delta_dice']))]:
        (output/name).mkdir()
        selected = [r for r in selected if (r['delta_dice'] > args.delta_epsilon if name == 'top_improved'
                                           else r['delta_dice'] < -args.delta_epsilon)]
        for rank, row in enumerate(selected[:args.top_k], 1):
            filename = f'{rank:02d}_'+hashlib.sha256(row['image_id'].encode()).hexdigest()[:12]+'.png'
            save_panel(output/name/filename, bank[row['image_id']])
    report(summary, rows, output)
    for name, metrics in summary['models'].items():
        print(f'{name}: Dice={metrics["dice"]:.6f} HD95={metrics["mean_slice_hd95"]:.4f}')
    print(f'Diagnostics saved to {output}')
    return summary


if __name__ == '__main__':
    main()
