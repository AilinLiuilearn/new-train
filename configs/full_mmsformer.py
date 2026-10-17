"""Isolated Stage-1 Full MMSFormer fusion experiment configuration."""
from pathlib import Path

import torch

from configs.base import str2bool
from run_full_add_baseline import FullAddConfig
from models.components.mmsformer_fusion import UPSTREAM_COMMIT, UPSTREAM_BLOB

ARCH_METADATA = {
    'mmsformer_schema_version': 1,
    'mmsformer_stage_channels': [64, 128, 320, 512],
    'mmsformer_stage_order': [1, 2, 3, 4],
    'mmsformer_num_modals': 2,
    'mmsformer_se_reduction': 16,
    'mmsformer_kernels': [3, 5, 7],
    'mmsformer_initialization': 'native_FusionBlock_constructor',
    'mmsformer_output_mode': 'replace_add',
    'mmsformer_upstream_commit': UPSTREAM_COMMIT,
    'mmsformer_source_blob': UPSTREAM_BLOB,
}


class FullMMSFormerConfig(FullAddConfig):
    @staticmethod
    def data_parser():
        parser = FullAddConfig.data_parser()
        parser.set_defaults(norm_mode='imagenet')
        return parser

    @staticmethod
    def model_parser():
        parser = FullAddConfig.model_parser()
        for action in parser._actions:
            if action.dest == 'model_arch':
                action.default = 'full_mmsformer_fusion'
                action.choices = ('full_mmsformer_fusion',)
        parser.add_argument('--mmsformer_enabled', type=str2bool, default=True)
        parser.add_argument('--require_cuda', type=str2bool, default=True)
        return parser

    @staticmethod
    def train_parser():
        parser = FullAddConfig.train_parser()
        parser.set_defaults(learning_rate=4e-5, early_stop_patience=60, ema_enabled=True)
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        if (cfg.accumulation_steps != 1 or cfg.optimizer != 'adamw'
                or cfg.use_deep_supervision or cfg.deep_supervision):
            raise ValueError('Full comparison requires AdamW, accumulation_steps=1, no deep supervision')
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA required but unavailable; CPU smoke: --require_cuda false')
        folder = Path(cfg.checkpoint_root) / 'MDT' / cfg.hash
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Existing run {folder}; use a fresh --hash (no resume support)')
        for key, value in ARCH_METADATA.items():
            setattr(cfg, key, value.copy() if isinstance(value, list) else value)
        return cfg
