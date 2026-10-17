"""Isolated Stage-1 Full Add + DSCF residual experiment configuration."""
from pathlib import Path

import torch

from configs.base import str2bool
from run_full_add_baseline import FullAddConfig
from models.components.dscf_residual import (
    UPSTREAM_COMMIT, UPSTREAM_BLOB, STAGE_CHANNELS, STAGE_RANKS, stage_geometry)


def architecture_metadata(image_size):
    sizes, strides = stage_geometry(image_size)
    return {
        'dscf_schema_version': 1,
        'dscf_stage_channels': list(STAGE_CHANNELS),
        'dscf_stage_ranks': list(STAGE_RANKS),
        'dscf_stage_order': [1, 2, 3, 4],
        'dscf_query_sizes': list(sizes),
        'dscf_stage_strides': list(strides),
        'dscf_heads': 2,
        'dscf_groups': 1,
        'dscf_kernel': 9,
        'dscf_initialization': 'native_core_zero_U_fc1',
        'dscf_output_mode': 'CT+PET+delta',
        'dscf_core_precision': 'float32',
        'dscf_position_encoding': 'native_sampled_relative_bias',
        'dscf_upstream_commit': UPSTREAM_COMMIT,
        'dscf_source_blob': UPSTREAM_BLOB,
    }


class FullDSCFConfig(FullAddConfig):
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
                action.default = 'full_dscf_residual'
                action.choices = ('full_dscf_residual',)
        parser.add_argument('--dscf_enabled', type=str2bool, default=True)
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
        metadata = architecture_metadata(cfg.image_size_2d)
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA required but unavailable; CPU smoke: --require_cuda false')
        folder = Path(cfg.checkpoint_root) / 'MDT' / cfg.hash
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Existing run {folder}; use a fresh --hash (no resume support)')
        for key, value in metadata.items(): setattr(cfg, key, value)
        return cfg
