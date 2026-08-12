# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
from models.build_mdt_seg import build_mdt_seg_teacher


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ('1', 'true', 'yes', 'y', 't'):
        return True
    if v in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {v}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize prototype-referenced PET calibration for a trained checkpoint.'
    )
    parser.add_argument('--checkpoint', required=True, help='Path to ckpt.best_joint.pth.tar')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--forward_mode', choices=('full', 'missing'), default='full')
    parser.add_argument('--num_cases', type=int, default=4, help='How many unique cases to visualize')
    parser.add_argument('--slices_per_case', type=int, default=1, help='How many slices to save per case')
    parser.add_argument('--random_case_order', type=str2bool, default=False, help='Shuffle case order before selection')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--slice_index', type=int, default=0, help='Skip this many matched samples before saving')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(checkpoint_config, checkpoint_path):
    config = dict(checkpoint_config)
    config['checkpoint_root'] = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
    config['hash'] = os.path.basename(os.path.dirname(checkpoint_path))
    config['checkpoint_dir'] = os.path.dirname(checkpoint_path)
    return SimpleNamespace(**config)


def _to_numpy_2d(x):
    x = x.detach().float().cpu().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(axis=0)
    return x


def _normalize_map(x):
    x = x.astype(np.float32)
    mn = np.percentile(x, 1)
    mx = np.percentile(x, 99)
    if mx <= mn:
        return np.zeros_like(x)
    x = np.clip((x - mn) / (mx - mn + 1e-8), 0.0, 1.0)
    return x


def _save_panel(path, ct_map, pet_before_map, pet_after_map, diff_map, title):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    items = [
        ('CT', ct_map, 'gray'),
        ('PET before', pet_before_map, 'magma'),
        ('PET after', pet_after_map, 'magma'),
        ('Delta', diff_map, 'coolwarm'),
    ]
    for ax, (name, img, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _save_scale_triplet(path, ct_map, pet_before_map, pet_after_map, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    items = [
        ('CT ref', ct_map, 'gray'),
        ('PET before', pet_before_map, 'magma'),
        ('PET after', pet_after_map, 'magma'),
    ]
    for ax, (name, img, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap)
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = build_config(ckpt['config'], checkpoint_path)
    seed_everything(int(getattr(config, 'random_state', 2023)))

    output_dir = args.output_dir or os.path.join(os.path.dirname(checkpoint_path), f'pet_calibration_vis_{args.forward_mode}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'npz'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)

    split_file = config.val_split_file if args.split == 'val' else config.test_split_file
    _, val_loader, test_loader = get_pclt20k_loaders_cipa_aligned(
        config.root,
        config.image_size_2d,
        args.batch_size,
        args.num_workers,
        config.random_state,
        config.pin_memory,
        config.aug_mode,
        config.norm_mode,
        config.train_split_file,
        config.val_split_file,
        config.test_split_file,
        checkpoint_dir=os.path.dirname(checkpoint_path),
    )
    loader = val_loader if args.split == 'val' else test_loader

    model = build_mdt_seg_teacher(config)['model']
    load_result = model.load_state_dict(ckpt['model'], strict=True)
    model.to(args.device)
    model.eval()

    meta = {
        'checkpoint': checkpoint_path,
        'split': args.split,
        'forward_mode': args.forward_mode,
        'load_result': str(load_result),
        'bank_ready': bool(model.prototype_memory.bank_ready),
        'bank_version': int(model.prototype_memory.bank_version.item()),
        'ready_slots': int(model.prototype_memory.prototype_ready.sum().item()),
        'output_dir': output_dir,
        'selected_samples': [],
    }

    selected_cases = {}
    ordered_case_ids = []
    all_batches = []

    for batch in loader:
        all_batches.append(batch)
        case_ids = batch['case_id']
        for idx, case_id in enumerate(case_ids):
            if case_id not in selected_cases:
                selected_cases[case_id] = []
                ordered_case_ids.append(case_id)
            selected_cases[case_id].append((len(all_batches) - 1, idx))

    if args.random_case_order:
        rng = random.Random(int(getattr(config, 'random_state', 2023)))
        rng.shuffle(ordered_case_ids)

    target_case_ids = ordered_case_ids[:args.num_cases]
    saved = 0
    matched_counter = 0

    for case_id in target_case_ids:
        if saved >= args.num_cases:
            break
        sample_indices = selected_cases[case_id][:args.slices_per_case]
        for batch_idx, item_idx in sample_indices:
            if saved >= args.num_cases:
                break
            batch = all_batches[batch_idx]
            ct = batch['ct'].to(args.device, non_blocking=True)
            pet = batch['pet'].to(args.device, non_blocking=True)
            mask = batch['mask'].to(args.device, non_blocking=True).float()
            image_ids = batch['image_id']
            case_ids_batch = batch['case_id']
            slice_ids = batch['slice_id']

            ct_feats = model._encode_ct(ct)
            pet_feats_real = model._encode_pet(pet)
            model._collect_cppi(ct_feats, pet_feats_real, mask)

            if args.forward_mode == 'full':
                if model.prototype_memory.bank_ready:
                    _, ct_reference_feats, _ = model._retrieve_cppi(
                        ct_feats,
                        compute_report=False,
                        save_diagnostics=False,
                        print_info=False,
                        return_ct_reference=True,
                    )
                    ct_reference_feats = [x.detach() for x in ct_reference_feats]
                    pet_feats_before = pet_feats_real
                    pet_feats_after = model.pet_calibration(
                        ct_feats,
                        pet_feats_before,
                        ct_reference_feats,
                        reference_valid=True,
                    )
                else:
                    ct_reference_feats = [torch.zeros_like(x) for x in ct_feats]
                    pet_feats_before = pet_feats_real
                    pet_feats_after = model.pet_calibration(
                        ct_feats,
                        pet_feats_before,
                        ct_reference_feats,
                        reference_valid=False,
                    )
            else:
                pet_feats_before, ct_reference_feats, _ = model._retrieve_cppi(
                    ct_feats,
                    compute_report=False,
                    save_diagnostics=False,
                    print_info=False,
                    return_ct_reference=True,
                )
                ct_reference_feats = [x.detach() for x in ct_reference_feats]
                pet_feats_after = model.pet_calibration(
                    ct_feats,
                    pet_feats_before,
                    ct_reference_feats,
                    reference_valid=model.prototype_memory.bank_ready,
                )

            b = item_idx
            sample_id = f"{case_ids_batch[b]}__{slice_ids[b]}__{args.forward_mode}"
            sample_dir = os.path.join(output_dir, 'png', sample_id)
            os.makedirs(sample_dir, exist_ok=True)

            ct_maps = []
            before_maps = []
            after_maps = []
            delta_ratios = []
            gamma_abs_means = []
            beta_abs_means = []

            for scale_idx, (ct_s, before_s, after_s, ref_s) in enumerate(zip(ct_feats, pet_feats_before, pet_feats_after, ct_reference_feats)):
                ct_s = ct_s[b:b+1]
                before_s = before_s[b:b+1]
                after_s = after_s[b:b+1]
                ref_s = ref_s[b:b+1]
                ct_tokens = torch.nn.functional.normalize(ct_s.flatten(2).transpose(1, 2), p=2, dim=-1, eps=1e-6)
                ref_tokens = torch.nn.functional.normalize(ref_s.flatten(2).transpose(1, 2), p=2, dim=-1, eps=1e-6)
                delta = (ct_tokens - ref_tokens).mean(dim=1)
                affine = model.pet_calibration.heads[scale_idx](delta)
                raw_gamma, raw_beta = affine.chunk(2, dim=-1)
                gamma = torch.tanh(raw_gamma).view(1, ct_s.shape[1], 1, 1)
                beta = torch.tanh(raw_beta).view(1, ct_s.shape[1], 1, 1)
                centered = before_s - before_s.mean(dim=(2, 3), keepdim=True)
                recon = before_s + gamma * centered + beta
                gamma_abs_means.append(float(gamma.abs().mean().item()))
                beta_abs_means.append(float(beta.abs().mean().item()))
                delta_ratios.append(float((recon - before_s).abs().mean().item() / (before_s.abs().mean().item() + 1e-6)))
                ct_maps.append(_normalize_map(_to_numpy_2d(ct_s)))
                before_maps.append(_normalize_map(_to_numpy_2d(before_s)))
                after_maps.append(_normalize_map(_to_numpy_2d(after_s)))

                scale_path = os.path.join(sample_dir, f'scale_{scale_idx + 1}.png')
                _save_scale_triplet(
                    scale_path,
                    ct_maps[-1],
                    before_maps[-1],
                    after_maps[-1],
                    title=f'{sample_id} | scale={scale_idx + 1} | gamma={gamma_abs_means[-1]:.4f} beta={beta_abs_means[-1]:.4f}',
                )

            ct_img = ct[b:b+1]
            before_img = pet_feats_before[-1][b:b+1]
            after_img = pet_feats_after[-1][b:b+1]
            _save_panel(
                os.path.join(sample_dir, 'summary.png'),
                _normalize_map(_to_numpy_2d(ct_img)),
                _normalize_map(_to_numpy_2d(before_img)),
                _normalize_map(_to_numpy_2d(after_img)),
                _normalize_map(_to_numpy_2d(after_img - before_img)),
                title=f'{sample_id} | mean_gamma={np.mean(gamma_abs_means):.4f} mean_beta={np.mean(beta_abs_means):.4f}',
            )

            np.savez_compressed(
                os.path.join(output_dir, 'npz', f'{sample_id}.npz'),
                ct=ct[b].detach().cpu().numpy(),
                pet=pet[b].detach().cpu().numpy(),
                pet_before=np.array([t[b].detach().cpu().numpy() for t in pet_feats_before], dtype=object),
                pet_after=np.array([t[b].detach().cpu().numpy() for t in pet_feats_after], dtype=object),
                ct_reference=np.array([t[b].detach().cpu().numpy() for t in ct_reference_feats], dtype=object),
                gamma_abs_mean=np.array(gamma_abs_means, dtype=np.float32),
                beta_abs_mean=np.array(beta_abs_means, dtype=np.float32),
                delta_ratio=np.array(delta_ratios, dtype=np.float32),
            )

            meta['selected_samples'].append({
                'sample_id': sample_id,
                'case_id': case_ids_batch[b],
                'image_id': image_ids[b],
                'slice_id': slice_ids[b],
                'gamma_abs_mean': gamma_abs_means,
                'beta_abs_mean': beta_abs_means,
                'delta_ratio': delta_ratios,
            })
            saved += 1
            matched_counter += 1
            print(f'[SAVE] {sample_id}', flush=True)

        if saved >= args.num_cases:
            break

    with open(os.path.join(output_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(f'[SAVED] {output_dir}', flush=True)


if __name__ == '__main__':
    main()
