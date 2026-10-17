"""Single-network fusion ablation with baseline encoder pretraining and BCE+Dice."""
import argparse
from pathlib import Path

import torch

from configs.base import str2bool
from run_full_add_baseline import FullAddConfig
from models.components.local_reassembly import STAGE_CHANNELS,STAGE_RANKS


def architecture_metadata(image_size):
    if type(image_size) is not int or image_size < 32 or image_size % 32:
        raise ValueError('image_size must be an integer >=32 divisible by 32')
    return dict(local_schema_version=3,local_stage_channels=list(STAGE_CHANNELS),
        local_stage_ranks=list(STAGE_RANKS),local_stage_order=[1,2,3,4],
        local_window=3,local_heads=2,local_null_value='fixed_zero',
        local_position_mode='centered_masked_3x3',local_output_mode='CT+PET+delta_CT+delta_PET',
        local_initialization='zero_two_output_projections',local_core_precision='float32',
        loss_method='BCE+Dice')


class FullLocalConfig(FullAddConfig):
    @staticmethod
    def data_parser():
        parser=FullAddConfig.data_parser(); parser.set_defaults(norm_mode='imagenet')
        return parser

    @staticmethod
    def model_parser():
        parser=argparse.ArgumentParser('Single PETCT model',add_help=False)
        parser.add_argument('--model_arch',default='full_local_reassembly',choices=('full_local_reassembly',))
        parser.add_argument('--ct_backbone',default='convnextv2_nano')
        parser.add_argument('--pet_backbone',default='mit_b1')
        parser.add_argument('--pretrained',type=str2bool,default=True,
                            help='Load CT/PET encoder pretraining only; false for random-init smoke tests')
        parser.add_argument('--ct_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
        parser.add_argument('--pet_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
        parser.add_argument('--decoder_channels',type=int,nargs=4,default=[512,256,128,64])
        parser.add_argument('--use_deep_supervision',type=str2bool,default=False)
        parser.add_argument('--deep_supervision',type=str2bool,default=False)
        parser.add_argument('--local_enabled',type=str2bool,default=True)
        parser.add_argument('--require_cuda',type=str2bool,default=True)
        return parser

    @staticmethod
    def train_parser():
        parser=FullAddConfig.train_parser()
        parser.set_defaults(learning_rate=4e-5,early_stop_patience=60,ema_enabled=True)
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg=super().parse_arguments()
        if cfg.accumulation_steps != 1 or cfg.optimizer != 'adamw' or cfg.use_deep_supervision or cfg.deep_supervision:
            raise ValueError('Use AdamW, accumulation_steps=1, no deep supervision')
        metadata=architecture_metadata(cfg.image_size_2d)
        if cfg.require_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA required but unavailable; CPU smoke: --require_cuda false')
        folder=Path(cfg.checkpoint_root)/'MDT'/cfg.hash
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Existing run {folder}; use a fresh --hash (no resume support)')
        for key,value in metadata.items(): setattr(cfg,key,value)
        cfg.initialization_mode=('encoder_pretrained_fresh_segmentation' if cfg.pretrained else 'all_random')
        return cfg
