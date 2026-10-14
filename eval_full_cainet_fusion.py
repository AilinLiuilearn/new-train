# -*- coding: utf-8 -*-
"""Rebuild and evaluate a CAINet Full checkpoint with its saved protocol.

Default raw/EMA choice follows the weight type used to select best_full.
Pretrained encoder files are unnecessary: the checkpoint contains all weights.
"""
import argparse
import json
import os

import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.full_cainet_fusion import build_full_cainet_model
from tasks.mdt_seg import MDTSegTeacher
from utils.run_common import seed_everything


def load_full_checkpoint(path, device='cpu', use_ema=None):
    ckpt = MDTSegTeacher.load_state_dicts(path)
    saved = dict(ckpt['config'])
    if (saved.get('experiment_type') != 'full_cainet_fusion'
            or saved.get('fusion_type') != 'CAINetFusion'):
        raise ValueError('Expected a Full CAINet checkpoint; Add/Missing checkpoints are incompatible')
    saved.pop('checkpoint_dir', None)
    saved.update(pretrained=False, ct_pretrained_path=None, pet_pretrained_path=None)
    cfg = SegMDTConfig(args=saved)
    model = build_full_cainet_model(cfg)['model'].to(device)
    if use_ema is None:
        use_ema = ckpt.get('eval_weights', 'raw') == 'ema'
    tag = 'ema' if use_ema else 'raw'
    weights = ckpt.get('model_ema') if use_ema else ckpt.get('model')
    if weights is None:
        raise ValueError(f'Checkpoint does not contain requested {tag} weights')
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, cfg, ckpt, tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--root', default=None, help='Default: root saved in checkpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--use_ema', type=str2bool, default=None,
                        help='Default: checkpoint recorded raw/EMA selection')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--hd95_mode', choices=('cipa_official', 'legacy', 'surface'), default=None)
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    result_path = os.path.join(output_dir, 'eval_full_cainet.json')
    if os.path.exists(result_path):
        raise FileExistsError(f'Refusing to overwrite {result_path}; use a new output directory')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, ckpt, tag = load_full_checkpoint(args.checkpoint, device, args.use_ema)
    for name in ('root', 'num_workers', 'batch_size', 'hd95_mode'):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    cfg.hd95_mode = getattr(cfg, 'hd95_mode', 'cipa_official')
    seed_everything(cfg.random_state)
    # The model being evaluated already contains the selected EMA/raw bytes.
    # Avoid constructing an unrelated EMA copy in the evaluation task.
    cfg.ema_enabled = False
    task = MDTSegTeacher({'model': model}, cfg)
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    os.makedirs(output_dir, exist_ok=True)
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, 'none', cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=output_dir,
        allow_val_equals_test=bool(getattr(cfg, 'allow_val_equals_test', False)), ct_only=False)
    metrics = task.evaluate(test_loader, eval_mode='full', tag='test_full', model=model)
    result = dict(metrics, weights=tag, experiment_type='full_cainet_fusion',
                  cacr_enabled=cfg.cacr_enabled, gcm_enabled=cfg.gcm_enabled,
                  checkpoint=os.path.abspath(args.checkpoint),
                  best_epoch=ckpt.get('best_epoch'),
                  hd95_mode=cfg.hd95_mode, hd95_unit='pixel', norm_mode=cfg.norm_mode)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
