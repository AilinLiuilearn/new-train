# -*- coding: utf-8 -*-
"""Visualize encoder fusion stages and light decoder feature stages for MDT baseline."""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher


def _load_dataset_module():
    root = os.getcwd()
    dataset_path = os.path.join(root, 'datasets', 'pclt20k_seg.py')
    spec = importlib.util.spec_from_file_location('local_pclt20k_seg', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_loader(config, split='val'):
    dataset_mod = _load_dataset_module()
    if getattr(config, 'cipa_aligned', False):
        loaders = dataset_mod.get_pclt20k_loaders_cipa_aligned(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            config.random_state,
            pin_memory=getattr(config, 'pin_memory', True),
            aug_mode=getattr(config, 'aug_mode', 'cipa'),
            norm_mode=getattr(config, 'norm_mode', 'imagenet'),
        )
    else:
        loaders = dataset_mod.get_pclt20k_loaders(
            config.root,
            config.image_size_2d,
            config.batch_size,
            config.num_workers,
            val_ratio=config.val_ratio,
            random_state=config.random_state,
            use_case_split=getattr(config, 'use_case_split', True),
            pin_memory=getattr(config, 'pin_memory', True),
        )
    mapping = {'train': 0, 'val': 1, 'test': 2}
    return loaders[mapping[split]]


def _load_config(path):
    with open(path, 'r') as f:
        data = json.load(f)
    data.pop('checkpoint_dir', None)
    cfg = SegMDTConfig(args=data)
    return cfg


def _load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)


def _to_3ch(x):
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def _norm01(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)


def _to_display(img_3ch, norm_mode='imagenet'):
    arr = img_3ch.detach().float().cpu().numpy()
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
        if norm_mode == 'cipa':
            arr = np.clip((arr + 1.6) / 3.2, 0, 1)
        else:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = np.clip(arr * std + mean, 0, 1)
        return arr.mean(axis=2)
    return _norm01(arr.squeeze())


def _feature_map(feat, sample_idx, target_size=None):
    x = feat[sample_idx:sample_idx + 1].detach().float()
    energy = x.abs().mean(dim=1, keepdim=True)
    if target_size is not None:
        energy = F.interpolate(energy, size=target_size, mode='bilinear', align_corners=False)
    arr = energy.squeeze().cpu().numpy()
    return _norm01(arr)


def _logit_map(logit, sample_idx, target_size=None):
    x = logit[sample_idx:sample_idx + 1].detach().float()
    if target_size is not None and x.shape[-2:] != target_size:
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    return torch.sigmoid(x).squeeze().cpu().numpy().astype(np.float32)


def _overlay_gt_pred(gray, gt, pred):
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    gt = gt > 0.5
    pred = pred > 0.5
    tp = np.logical_and(gt, pred)
    fn = np.logical_and(gt, np.logical_not(pred))
    fp = np.logical_and(np.logical_not(gt), pred)
    rgb[tp] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    rgb[fn] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[fp] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)


def _forward_with_features(model, ct, pet, target_size):
    ct3 = _to_3ch(ct)
    pet3 = _to_3ch(pet)
    ct_feats = model.enc_ct(ct3)
    pet_feats = model.enc_pet(pet3)
    if model.fusion_type == 'concat':
        fused_feats = [proj(torch.cat([c, p], dim=1)) for proj, c, p in zip(model.fusion, ct_feats, pet_feats)]
    elif model.fusion is not None:
        fusion_out = model.fusion(ct_feats, pet_feats)
        fused_feats = fusion_out[0] if isinstance(fusion_out, tuple) else fusion_out
    else:
        fused_feats = [c + p for c, p in zip(ct_feats, pet_feats)]

    decoder = model.decoder
    x1, x2, x3, x4 = fused_feats
    d4 = decoder.proj4(x4)
    s3 = decoder.proj3(x3)
    d3 = decoder.fuse3(torch.cat([decoder._upsample_to(d4, s3), s3], dim=1))
    s2 = decoder.proj2(x2)
    d2 = decoder.fuse2(torch.cat([decoder._upsample_to(d3, s2), s2], dim=1))
    s1 = decoder.proj1(x1)
    d1 = decoder.fuse1(torch.cat([decoder._upsample_to(d2, s1), s1], dim=1))
    p1 = decoder._upsample_size(decoder.head1(d1), target_size)
    p2 = decoder._upsample_size(decoder.head2(d2), target_size)
    p3 = decoder._upsample_size(decoder.head3(d3), target_size)
    p4 = decoder._upsample_size(decoder.head4(d4), target_size)
    outputs = {'preds': [p1, p2, p3, p4], 'pred': p1}
    feats = {
        'ct': ct_feats,
        'pet': pet_feats,
        'fused': fused_feats,
        'dec': [d1, d2, d3, d4],
        'preds': [p1, p2, p3, p4],
    }
    return outputs, feats


def _parse_indices(text):
    if not text:
        return None
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _save_feature_panels(batch, model, device, args, plt, saved):
    ct = batch['ct'].float().to(device)
    pet = batch['pet'].float().to(device)
    mask = batch['mask'].float().to(device)
    target_size = mask.shape[-2:]
    outputs, feats = _forward_with_features(model, ct, pet, target_size)
    prob = torch.sigmoid(outputs['pred']).squeeze(1)

    for i in range(ct.shape[0]):
        if saved >= args.num_samples:
            break
        ct_img = _to_display(ct[i], getattr(args, 'norm_mode', 'imagenet'))
        pet_img = _to_display(pet[i], getattr(args, 'norm_mode', 'imagenet'))
        gt = mask[i].squeeze().detach().cpu().numpy().astype(np.float32)
        pred = (prob[i].detach().cpu().numpy() > args.threshold).astype(np.float32)
        compare = _overlay_gt_pred(ct_img, gt, pred)

        panels = [
            ('CT', ct_img, 'gray'),
            ('PET', pet_img, 'inferno'),
            ('GT/PRED', compare, None),
            ('Prob p1', _logit_map(feats['preds'][0], i, target_size), 'jet'),
            ('GT', gt, 'gray'),
        ]
        for s in range(4):
            panels.extend([
                (f'CT enc{s + 1}', _feature_map(feats['ct'][s], i, target_size), 'viridis'),
                (f'PET enc{s + 1}', _feature_map(feats['pet'][s], i, target_size), 'inferno'),
                (f'SUM fused{s + 1}', _feature_map(feats['fused'][s], i, target_size), 'magma'),
                (f'Dec d{s + 1}', _feature_map(feats['dec'][s], i, target_size), 'cividis'),
                (f'Pred p{s + 1}', _logit_map(feats['preds'][s], i, target_size), 'jet'),
            ])

        ncols = 5
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows))
        axes = np.atleast_1d(axes).reshape(nrows, ncols)
        for ax, (title, img, cmap) in zip(axes.flat, panels):
            if img.ndim == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        for ax in axes.flat[len(panels):]:
            ax.axis('off')
        plt.tight_layout()
        idx_text = batch.get('idx')
        suffix = ''
        if idx_text is not None:
            try:
                suffix = f'_idx{int(idx_text[i])}'
            except Exception:
                suffix = ''
        save_path = os.path.join(args.out_dir, f'feature_stages_{saved:03d}{suffix}.png')
        plt.savefig(save_path, dpi=int(args.dpi))
        plt.close(fig)
        saved += 1
    return saved


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--num_samples', type=int, default=24)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--random_sample', action='store_true', help='Randomly sample slices from the selected split instead of taking the first batches.')
    parser.add_argument('--random_seed', type=int, default=2026)
    parser.add_argument('--one_per_case', action='store_true', help='When random_sample is enabled, sample at most one slice per case if case_id is available.')
    parser.add_argument('--indices', type=str, default='', help='Comma-separated dataset indices, e.g. 1092,1380,1297')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    sys.modules.pop('datasets', None)

    cfg = _load_config(args.config)
    args.norm_mode = getattr(cfg, 'norm_mode', 'imagenet')
    cfg.gpus = [args.gpu]
    cfg.batch_size = min(int(getattr(cfg, 'batch_size', 16)), 8)
    cfg.num_workers = 0
    cfg.pretrained_path = None
    cfg.ct_pretrained_path = None
    cfg.pet_pretrained_path = None

    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    networks = build_mdt_seg_teacher(cfg)
    model = networks['model'].to(device)
    _load_checkpoint(model, args.ckpt)
    model.eval()

    loader = _build_loader(cfg, split=args.split)
    os.makedirs(args.out_dir, exist_ok=True)
    saved = 0

    selected = _parse_indices(args.indices)
    if selected is not None:
        dataset = loader.dataset
        print(f'[vis_baseline_feature_stages] fixed indices: {selected}')
        for idx in selected:
            if saved >= args.num_samples:
                break
            if idx < 0 or idx >= len(dataset):
                print(f'[vis_baseline_feature_stages] skip invalid index: {idx}')
                continue
            batch = dataset[int(idx)]
            batch = {
                k: (v.unsqueeze(0) if torch.is_tensor(v) else torch.tensor([v]) if isinstance(v, (int, float, bool)) else v)
                for k, v in batch.items()
            }
            batch['idx'] = torch.tensor([idx])
            saved = _save_feature_panels(batch, model, device, args, plt, saved)
    elif args.random_sample:
        dataset = loader.dataset
        rng = np.random.default_rng(int(args.random_seed))
        if args.one_per_case and hasattr(dataset, 'records'):
            by_case = {}
            for idx, record in enumerate(dataset.records):
                by_case.setdefault(record.get('case_id', str(idx)), []).append(idx)
            case_ids = np.array(list(by_case.keys()), dtype=object)
            rng.shuffle(case_ids)
            selected = []
            for case_id in case_ids[:args.num_samples]:
                selected.append(int(rng.choice(by_case[case_id])))
        else:
            selected = rng.permutation(len(dataset))[:args.num_samples].tolist()
        print(f'[vis_baseline_feature_stages] random selected indices: {selected}')
        for idx in selected:
            batch = dataset[int(idx)]
            batch = {
                k: (v.unsqueeze(0) if torch.is_tensor(v) else torch.tensor([v]) if isinstance(v, (int, float, bool)) else v)
                for k, v in batch.items()
            }
            saved = _save_feature_panels(batch, model, device, args, plt, saved)
            if saved >= args.num_samples:
                break
    else:
        for batch in loader:
            saved = _save_feature_panels(batch, model, device, args, plt, saved)
            if saved >= args.num_samples:
                break
    print(f'[vis_baseline_feature_stages] saved {saved} figures to {args.out_dir}')


if __name__ == '__main__':
    main()
