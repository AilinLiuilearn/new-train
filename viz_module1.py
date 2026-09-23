# -*- coding: utf-8 -*-
"""Visualise Missing-path stages for a trained Module-1 checkpoint.

For each slice we run the Missing path manually and dump a 7-panel grid:
  CT | PET_real(privileged) | P_prior(retrieved) | P_comp(after affine)
  | pred | GT | absdiff(PET_real, P_comp)

Runs on CPU by default (safe while a GPU training job is active).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.seg_mdt import SegMDTConfig
from models.build_mdt_seg import build_mdt_seg_teacher
from tasks.mdt_seg import MDTSegTeacher


def _norm01(x):
    x = x - x.min()
    d = x.max() - x.min()
    return x / d if d > 1e-8 else x * 0


def _to_np(t):
    return t.detach().float().cpu().numpy()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--root', type=str, default='/root/autodl-tmp/data/PCLT20K')
    p.add_argument('--num_slices', type=int, default=24)
    p.add_argument('--out', type=str, default=None)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'ckpt.best_joint.pth.tar'),
                      map_location='cpu', weights_only=False)
    saved = dict(ckpt['config'])
    saved.pop('checkpoint_dir', None)
    saved['root'] = args.root
    saved['ct_pretrained_path'] = None
    saved['pet_pretrained_path'] = None
    cfg = SegMDTConfig(args=saved)

    task = MDTSegTeacher(build_mdt_seg_teacher(cfg), cfg)
    task.device = torch.device(args.device)
    task.model.to(task.device)
    task.model.load_state_dict(ckpt['model'], strict=True)
    task.model.eval()
    m = task.model

    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, 4, 0, cfg.random_state, False, 'none',
        cfg.norm_mode, cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
    )

    out_dir = args.out or os.path.join(args.checkpoint_dir, 'viz')
    os.makedirs(out_dir, exist_ok=True)

    saved_n = 0
    best = {}  # case_id -> (fg_frac, ct, pet, mask, case_id)
    for batch in test_loader:
        ct = batch['ct'].float()
        pet = batch['pet'].float()
        mask = batch['mask'].float()
        case_ids = list(batch['case_id'])
        for i in range(ct.shape[0]):
            fg = float((mask[i] > 0.5).float().mean())
            cid = case_ids[i]
            if cid not in best or fg > best[cid][0]:
                best[cid] = (fg, ct[i:i + 1], pet[i:i + 1], mask[i:i + 1], cid)
    # pick first N distinct cases with the largest lesion area
    order = sorted(best.values(), key=lambda x: -x[0])[:args.num_slices]
    for (fg, ct_i, pet_i, mask_i, cid) in order:
        if saved_n >= args.num_slices:
            break
        ct_i = ct_i.to(task.device); pet_i = pet_i.to(task.device); mask_i = mask_i.to(task.device)
        if True:
            ct_feats = m._encode_ct(ct_i)
            pet_real = m._encode_pet(pet_i)
            pet_prior, aux = m.module1.retrieve_pet_prior(ct_feats, return_attention=False)
            pet_comp, gammas, betas = m.pet_affine(ct_feats, pet_prior)
            fused = m._fuse_features(ct_feats, pet_comp, False)
            out = m._decode(fused, ct_i.shape[-2:])
            pred = out['logits'] if isinstance(out, dict) else out
            pred_bin = (pred.sigmoid() > 0.5).float()

            # Full path reference (uses real PET) for context.
            fused_full = m._fuse_features(ct_feats, pet_real, True)
            out_full = m._decode(fused_full, ct_i.shape[-2:])
            pred_full = out_full['logits'] if isinstance(out_full, dict) else out_full
            pred_full_bin = (pred_full.sigmoid() > 0.5).float()

            # S4 maps (highest-res comparison of prior vs comp).
            prior_s4 = pet_prior[-1]
            comp_s4 = pet_comp[-1]
            real_s4 = pet_real[-1]

            def ch_mean(t):
                return _norm01(_to_np(t[0].mean(0)))

            fig, ax = plt.subplots(2, 5, figsize=(20, 8))
            panels = [
                (ch_mean(ct_i[0:1].repeat(1, 3, 1, 1) if ct_i.shape[1] == 1 else ct_i), 'CT'),
                (ch_mean(real_s4), 'PET_real S4 (privileged)'),
                (ch_mean(prior_s4), 'P_prior S4 (retrieved)'),
                (ch_mean(comp_s4), 'P_comp S4 (after affine)'),
                (ch_mean((comp_s4 - prior_s4).abs()), '|P_comp - P_prior|'),
                (_to_np(mask_i[0, 0]), 'GT mask'),
                (_to_np(pred_bin[0, 0]), 'pred Missing'),
                (_to_np(pred_full_bin[0, 0]), 'pred Full(ref real PET)'),
                (_to_np((pred_bin[0, 0] - mask_i[0, 0]).abs()), '|pred - GT|'),
                (ch_mean((comp_s4 - real_s4).abs()), '|P_comp - PET_real|'),
            ]
            for a, (img, title) in zip(ax.ravel(), panels):
                a.imshow(img, cmap='gray')
                a.set_title(title, fontsize=9)
                a.axis('off')
            fig.suptitle(f'case={cid} fg={fg:.3f}  '
                         f'gamma_mean_s4={float(gammas[-1].mean()):.3f} '
                         f'comp_rms_s4={float(pet_comp[-1].pow(2).mean().sqrt()):.3f} '
                         f'prior_rms_s4={float(pet_prior[-1].pow(2).mean().sqrt()):.3f}', fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'slice_{saved_n:02d}_{cid}.png'), dpi=90)
            plt.close(fig)
            saved_n += 1
    print(f'saved {saved_n} slices -> {out_dir}')


if __name__ == '__main__':
    main()
