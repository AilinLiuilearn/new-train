"""Separate Full-only parser; no baseline default changes."""
import argparse
from configs.base import str2bool
from run_full_add_baseline import FullAddConfig


class FullStructureConfig(FullAddConfig):
    @staticmethod
    def data_parser():
        p=FullAddConfig.data_parser();p.set_defaults(norm_mode='imagenet',num_workers=2);return p

    @staticmethod
    def model_parser():
        p=FullAddConfig.model_parser()
        for action in p._actions:
            if action.dest=='model_arch':action.choices=('full_structure_response',)
        p.set_defaults(model_arch='full_structure_response');return p

    @staticmethod
    def train_parser():
        p=FullAddConfig.train_parser();p.set_defaults(ema_enabled=True);return p

    @staticmethod
    def task_specific_parser():
        p=argparse.ArgumentParser('Structure/response fusion',add_help=False)
        p.add_argument('--fusion_enabled',type=str2bool,default=True)
        p.add_argument('--fusion_heads',type=int,default=4)
        p.add_argument('--sae_expansion',type=float,default=2.66)
        p.add_argument('--fusion_log_every',type=int,default=100)
        p.add_argument('--fusion_visualize',type=str2bool,default=False)
        p.add_argument('--fusion_vis_every_epochs',type=int,default=5)
        p.add_argument('--fusion_vis_samples',type=int,default=2)
        p.add_argument('--allow_cpu',type=str2bool,default=False,
                       help='Explicit CPU opt-in for tests; formal training requires visible CUDA by default.')
        p.add_argument('--resume',type=str,default=None)
        p.add_argument('--eval_checkpoint',type=str,default=None)
        p.add_argument('--smoke',type=str2bool,default=False)
        return p

    @classmethod
    def parse_arguments(cls):
        cfg=super().parse_arguments()
        if cfg.accumulation_steps!=1 or cfg.use_deep_supervision or cfg.deep_supervision or cfg.optimizer!='adamw':
            raise ValueError('Match Full Add: AdamW, accumulation=1, no extra deep supervision')
        if min(cfg.fusion_heads,cfg.fusion_log_every,cfg.fusion_vis_every_epochs,cfg.fusion_vis_samples)<1 or cfg.sae_expansion<=0:
            raise ValueError('Invalid fusion width/diagnostic settings')
        if any((c//2)%cfg.fusion_heads for c in (64,128,320,512)):
            raise ValueError('Heads must divide all half-widths')
        return cfg
