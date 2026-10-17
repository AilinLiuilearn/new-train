"""Data and checkpoint paths shared by modulation training and evaluation."""

import os


def _loaders(cfg):
    from datasets.pclt20k_seg import get_pclt20k_loaders_cipa_aligned

    return get_pclt20k_loaders_cipa_aligned(
        cfg.root,
        cfg.image_size_2d,
        cfg.batch_size,
        cfg.num_workers,
        cfg.random_state,
        cfg.pin_memory,
        cfg.aug_mode,
        cfg.norm_mode,
        cfg.train_split_file,
        cfg.val_split_file,
        cfg.test_split_file,
        checkpoint_dir=cfg.checkpoint_dir,
        allow_val_equals_test=bool(getattr(cfg, "allow_val_equals_test", False)),
        ct_only=False,
    )


def _checkpoint_paths(checkpoint_dir):
    return {
        "best_joint": os.path.join(checkpoint_dir, "ckpt.best_joint.pth.tar"),
        "last": os.path.join(checkpoint_dir, "ckpt.last.pth.tar"),
    }
