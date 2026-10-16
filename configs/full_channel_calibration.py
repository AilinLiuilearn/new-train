"""Independent config; no changes to the existing Add/CMX/SKFF defaults."""
from pathlib import Path

import torch

from configs.base import str2bool
from run_full_add_baseline import FullAddConfig
from models.components.channel_calibration import UPSTREAM_COMMIT, UPSTREAM_BLOB, validate_settings

ARCH_METADATA = {
    'channel_schema_version': 1,
    'channel_stage_channels': [64, 128, 320, 512],
    'channel_stage_order': [1, 2, 3, 4],
    'channel_hidden_min': 16,
    'channel_descriptor': 'separate_gap_concat',
    'channel_activation': 'relu_then_bounded_tanh',
    'channel_initialization': 'zero_excitation_heads_exact_add',
    'channel_output': 'calibrated_add',
    'channel_upstream_commit': UPSTREAM_COMMIT,
    'channel_source_blob': UPSTREAM_BLOB,
}


class FullChannelCalibrationConfig(FullAddConfig):
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
                action.default = 'full_channel_calibration'
                action.choices = ('full_channel_calibration',)
        parser.add_argument('--channel_calibration_enabled', type=str2bool, default=True)
        parser.add_argument('--channel_reduction', type=int, default=8)
        parser.add_argument('--channel_strength', type=float, default=0.25)
        parser.add_argument('--require_cuda', type=str2bool, default=True)
        return parser

    @staticmethod
    def train_parser():
        parser = FullAddConfig.train_parser()
        parser.set_defaults(learning_rate=4e-5, early_stop_patience=60,
                            ema_enabled=True)
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        validate_settings(cfg.channel_reduction, cfg.channel_strength)
        if (cfg.accumulation_steps != 1 or cfg.optimizer != 'adamw'
                or cfg.use_deep_supervision or cfg.deep_supervision):
            raise ValueError('Full comparison requires AdamW, accumulation_steps=1, no deep supervision')
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU preflight: --require_cuda false')
        # Avoid checkpoint_dir's mkdir side effect before checking freshness.
        folder = Path(cfg.checkpoint_root) / 'MDT' / cfg.hash
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Existing run {folder}; use a fresh --hash (no resume support)')
        for name, value in ARCH_METADATA.items():
            setattr(cfg, name, value.copy() if isinstance(value, list) else value)
        return cfg
