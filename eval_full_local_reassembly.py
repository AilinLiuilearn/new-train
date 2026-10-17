"""Strict raw/EMA reconstruction and dataset evaluation of Full local reassembly."""
import argparse
import json
from pathlib import Path

import torch

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from configs.full_local_reassembly import architecture_metadata
from models.full_local_reassembly import build_full_local_model
from tasks.mdt_seg import MDTSegTeacher as SegmentationTask
from utils.run_common import seed_everything


def load_full_checkpoint(path, device='cpu', use_ema=None):
    checkpoint = SegmentationTask.load_state_dicts(path)
    saved = dict(checkpoint['config'])
    try:
        metadata = architecture_metadata(saved.get('image_size_2d'))
    except (ValueError, TypeError) as error:
        raise ValueError('Unsupported local reassembly checkpoint architecture/image_size') from error
    expected = dict(metadata, experiment_type='full_local_reassembly',
                    model_arch='full_local_reassembly', fusion_type='LocalReassemblyFusion')
    if (any(saved.get(k) != v for k, v in expected.items())
            or not isinstance(saved.get('local_enabled'), bool)
            or not isinstance(saved.get('protect_enabled'), bool)):
        raise ValueError('Unsupported local reassembly checkpoint architecture/source')
    saved.pop('checkpoint_dir', None)
    cfg = SegMDTConfig(args=saved)
    model = build_full_local_model(cfg)['model'].to(device)
    if use_ema is None:
        use_ema = checkpoint.get('eval_weights', 'raw') == 'ema'
    tag = 'ema' if use_ema else 'raw'
    weights = checkpoint.get('model_ema' if use_ema else 'model')
    if weights is None:
        raise ValueError(f'Checkpoint does not contain requested {tag} weights')
    model.load_state_dict(weights, strict=True)
    return model.eval(), cfg, checkpoint, tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--root')
    parser.add_argument('--use_ema', type=str2bool, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Use a new evaluation output directory')
    model, cfg, checkpoint, tag = load_full_checkpoint(args.checkpoint, args.device, args.use_ema)
    for name in ('root', 'batch_size', 'num_workers'):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    seed_everything(cfg.random_state)
    cfg.ema_enabled = cfg.mixed_precision = False
    task = SegmentationTask({'model': model}, cfg)
    task.device = torch.device(args.device)
    task.model.to(task.device)
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = get_pclt20k_loaders_cipa_aligned(
        cfg.root, cfg.image_size_2d, cfg.batch_size, cfg.num_workers,
        cfg.random_state, cfg.pin_memory, 'none', cfg.norm_mode,
        cfg.train_split_file, cfg.val_split_file, cfg.test_split_file,
        checkpoint_dir=str(output_dir), allow_val_equals_test=bool(cfg.allow_val_equals_test), ct_only=False)
    metrics = task.evaluate(test_loader, eval_mode='full', tag='test_full', model=model)
    result = dict(metrics, weights=tag, checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_epoch=checkpoint.get('epoch'), best_epoch=checkpoint.get('best_epoch'),
                  experiment_type='full_local_reassembly', local_enabled=cfg.local_enabled,
                  hd95_mode=cfg.hd95_mode, hd95_unit='pixel')
    (output_dir / 'eval_full_local_reassembly.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
