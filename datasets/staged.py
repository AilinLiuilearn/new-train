"""Existing PNG/augmentation protocol; every stage-2 sample is paired CT/PET."""
import hashlib
from pathlib import Path
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned


def split_signature(cfg):
    return {name: hashlib.sha256((Path(cfg.root)/getattr(cfg, name+'_split_file')).read_bytes()).hexdigest()
            for name in ('train', 'val', 'test')}


def loaders(cfg, output):
    return get_pclt20k_loaders_cipa_aligned(
        root=cfg.root, image_size=cfg.image_size_2d, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, random_state=cfg.random_state,
        pin_memory=cfg.device != 'cpu', aug_mode=cfg.aug_mode, norm_mode=cfg.norm_mode,
        train_split_file=cfg.train_split_file, val_split_file=cfg.val_split_file,
        test_split_file=cfg.test_split_file, checkpoint_dir=str(output),
        ct_only=cfg.stage == 1 or getattr(cfg, 'eval_mode', 'both') == 'missing',
        allow_val_equals_test=cfg.allow_val_equals_test)
