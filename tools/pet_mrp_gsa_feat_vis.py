# -*- coding: utf-8 -*-
"""Visualize CT encoder features before/after PET-MRP-GSA guidance."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.baseline_petct_unet import SingleModalityBaselineUNet
from models.build_mdt_seg import build_mdt_seg_teacher
from models.ct_pet_mrp_gsa_seg import CTPETMRPGSASegmentation, STAGE_NAMES
from run_mdt_seg import _build_loaders, _prepare_env
from tasks.mdt_seg import MDTSegTeacher


def _load_config(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'config_args.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def _feat_map(feat: torch.Tensor) -> np.ndarray:
    """Channel-mean activation map, per-sample min-max to [0,1]."""
    x = feat.detach().float().mean(dim=1)[0]
    x = x - x.min()
    x = x / (x.max() + 1e-6)
    return x.cpu().numpy()


def _diff_map(before: torch.Tensor, after: torch.Tensor) -> np.ndarray:
    b = before.detach().float().mean(dim=1)[0]
    a = after.detach().float().mean(dim=1)[0]
    d = (a - b).abs()
    d = d / (d.max() + 1e-6)
    return d.cpu().numpy()


def _to_display_ct(ct: torch.Tensor) -> np.ndarray:
    x = ct.detach().float()[0, 0].cpu().numpy()
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip(x, lo, hi)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x


def _to_display_pet(pet: torch.Tensor) -> np.ndarray:
    x = pet.detach().float()[0, 0].cpu().numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x


def _extract_pet_mrp_feats(model: CTPETMRPGSASegmentation, ct, pet):
    ct_in = model._to_3ch(ct)
    before = model.enc_ct.forward_stages(ct_in)
    after = []
    for idx, feat in enumerate(before):
        block = model.pet_guides[idx]
        if hasattr(block, 'forward'):
            out = block(feat, pet, pet_available=True)
        else:
            out = feat
        after.append(out)
    return before, after


def _extract_ct_baseline_feats(model: SingleModalityBaselineUNet, ct):
    x = model._to_3ch(ct)
    return model.encoder(x)


def _load_pet_mrp_model(ckpt_path, device):
    cfg = _load_config(os.path.dirname(ckpt_path))
    from types import SimpleNamespace
    config = SimpleNamespace(**cfg)
    config.mixed_precision = False
    networks = build_mdt_seg_teacher(config)
    model = networks['model'].to(device).eval()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    print(f'[load PET-MRP-GSA] epoch={ckpt.get("epoch")} from {ckpt_path}')
    return model, config


def _load_ct_model(ckpt_path, device):
    cfg = _load_config(os.path.dirname(ckpt_path))
    model = SingleModalityBaselineUNet(
        backbone=cfg.get('ct_backbone', 'convnext_tiny'),
        pretrained_path=cfg.get('ct_pretrained_path'),
        modality='ct',
        out_channels=1,
        use_deep_supervision=cfg.get('use_deep_supervision', True),
    ).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    print(f'[load CT-only] epoch={ckpt.get("epoch")} from {ckpt_path}')
    return model


def _save_stage_panel(out_path, ct_np, pet_np, before_np, after_np, diff_np, title):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        ('CT', ct_np, 'gray'),
        ('PET', pet_np, 'hot'),
        ('Before guide', before_np, 'viridis'),
        ('After guide', after_np, 'viridis'),
        ('|After-Before|', diff_np, 'magma'),
    ]
    for ax, (name, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap)
        ax.set_title(name, fontsize=11)
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pet-ckpt', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/pet_mrp_gsa_all_v1/ckpt.best_dice.pth.tar')
    parser.add_argument('--ct-ckpt', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/checkpoints_new/MDT/ct_convnext_tiny_ds_v1/ckpt.best_dice.pth.tar')
    parser.add_argument('--out-dir', type=str,
                        default='/root/autodl-tmp/mkd-main/new-train/experiments/pet_mrp_gsa_feat_vis')
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--sample-indices', type=int, nargs='*', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    pet_model, config = _load_pet_mrp_model(args.pet_ckpt, device)
    ct_model = _load_ct_model(args.ct_ckpt, device)

    from types import SimpleNamespace
    config.gpus = [0]
    _prepare_env(config)
    _, _, test_loader = _build_loaders(config)

    indices = args.sample_indices or list(range(args.num_samples))
    summary = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > max(indices):
                break
            if batch_idx not in indices:
                continue

            ct = batch['ct'].float().to(device)
            pet = batch['pet'].float().to(device)
            mask = batch['mask'].float().to(device)
            case_id = batch.get('case_id', [f'sample{batch_idx}'])[0]
            slice_id = batch.get('slice_id', [batch_idx])[0]

            ct_np = _to_display_ct(ct)
            pet_np = _to_display_pet(pet)

            before_list, after_list = _extract_pet_mrp_feats(pet_model, ct, pet)
            ct_feats = _extract_ct_baseline_feats(ct_model, ct)

            sample_dir = os.path.join(args.out_dir, f'sample_{batch_idx:03d}_{case_id}_{slice_id}')
            os.makedirs(sample_dir, exist_ok=True)

            # overview: inputs + c1/c4 before-after-diff
            for stage_idx, stage_name in enumerate(STAGE_NAMES):
                if stage_idx not in (0, 3):
                    continue
                before_np = _feat_map(before_list[stage_idx])
                after_np = _feat_map(after_list[stage_idx])
                diff_np = _diff_map(before_list[stage_idx], after_list[stage_idx])
                h, w = before_np.shape
                before_up = F.interpolate(
                    torch.from_numpy(before_np)[None, None],
                    size=ct_np.shape, mode='bilinear', align_corners=False,
                )[0, 0].numpy()
                after_up = F.interpolate(
                    torch.from_numpy(after_np)[None, None],
                    size=ct_np.shape, mode='bilinear', align_corners=False,
                )[0, 0].numpy()
                diff_up = F.interpolate(
                    torch.from_numpy(diff_np)[None, None],
                    size=ct_np.shape, mode='bilinear', align_corners=False,
                )[0, 0].numpy()
                _save_stage_panel(
                    os.path.join(sample_dir, f'{stage_name}_before_after.png'),
                    ct_np, pet_np, before_up, after_up, diff_up,
                    f'{stage_name} ({h}x{w}) before vs after PET-MRP-GSA | case={case_id} slice={slice_id}',
                )

            # all stages grid for this sample
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            for si, stage_name in enumerate(STAGE_NAMES):
                b = _feat_map(before_list[si])
                a = _feat_map(after_list[si])
                d = _diff_map(before_list[si], after_list[si])
                c = _feat_map(ct_feats[si])
                axes[si, 0].imshow(b, cmap='viridis')
                axes[si, 0].set_title(f'{stage_name} before')
                axes[si, 1].imshow(a, cmap='viridis')
                axes[si, 1].set_title(f'{stage_name} after')
                axes[si, 2].imshow(d, cmap='magma')
                axes[si, 2].set_title(f'{stage_name} |diff|')
                axes[si, 3].imshow(c, cmap='viridis')
                axes[si, 3].set_title(f'{stage_name} CT-only')
                for j in range(4):
                    axes[si, j].axis('off')
            fig.suptitle(f'PET-MRP-GSA feature guide | case={case_id} slice={slice_id}', fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(sample_dir, 'all_stages_grid.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # numeric summary
            row = {'batch_idx': batch_idx, 'case_id': str(case_id), 'slice_id': int(slice_id)}
            for si, stage_name in enumerate(STAGE_NAMES):
                b = before_list[si]
                a = after_list[si]
                diff = (a - b).abs().mean().item()
                rel = diff / (b.abs().mean().item() + 1e-6)
                row[f'{stage_name}_abs_diff_mean'] = diff
                row[f'{stage_name}_rel_change'] = rel
            summary.append(row)
            print(
                f'[sample {batch_idx}] case={case_id} slice={slice_id} '
                + ' '.join(f'{STAGE_NAMES[i]}_rel={row[f"{STAGE_NAMES[i]}_rel_change"]:.4f}' for i in range(4))
            )

    with open(os.path.join(args.out_dir, 'feature_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[saved] figures -> {args.out_dir}')


if __name__ == '__main__':
    main()
