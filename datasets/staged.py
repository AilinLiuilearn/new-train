"""Existing PNG/augmentation protocol plus exact balanced modality assignment."""
import hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler
from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned, _seed_worker


class HalfMissingBatches(Sampler):
    """Exact half Full per batch. Adjacent epochs use the same permutation and
    complement states, so each retained slice is Full once and Missing once.
    The tail smaller than a batch is dropped, matching the existing baseline.
    """
    def __init__(self, size, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('stage 2 requires an even batch_size >= 2')
        self.size, self.batch_size, self.seed, self.epoch = size, batch_size, seed, 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + (self.epoch-1)//2)
        order = torch.randperm(self.size, generator=g).tolist()
        for start in range(0, len(self)*self.batch_size, self.batch_size):
            yield [(idx, (j < self.batch_size//2) == (self.epoch % 2 == 1))
                   for j, idx in enumerate(order[start:start+self.batch_size])]


def split_signature(cfg):
    return {name: hashlib.sha256((Path(cfg.root)/getattr(cfg, name+'_split_file')).read_bytes()).hexdigest()
            for name in ('train', 'val', 'test')}


def loaders(cfg, output):
    train, val, test = get_pclt20k_loaders_cipa_aligned(
        root=cfg.root, image_size=cfg.image_size_2d, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, random_state=cfg.random_state,
        pin_memory=cfg.device != 'cpu', aug_mode=cfg.aug_mode, norm_mode=cfg.norm_mode,
        train_split_file=cfg.train_split_file, val_split_file=cfg.val_split_file,
        test_split_file=cfg.test_split_file, checkpoint_dir=str(output),
        ct_only=cfg.stage == 1 or getattr(cfg, 'eval_mode', 'both') == 'missing',
        allow_val_equals_test=cfg.allow_val_equals_test)
    if cfg.stage == 2 and getattr(cfg, 'eval_mode', 'both') != 'missing':
        train = DataLoader(train.dataset,
            batch_sampler=HalfMissingBatches(len(train.dataset), cfg.batch_size, cfg.random_state+11),
            num_workers=cfg.num_workers, pin_memory=cfg.device != 'cpu',
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(cfg.random_state+31))
    return train, val, test
