"""Independent config; baseline parsers/defaults remain untouched."""
import argparse
from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig


class PETCorrectionConfig(SegMDTConfig):
    @staticmethod
    def data_parser():
        p=SegMDTConfig.data_parser(); p.set_defaults(norm_mode='imagenet')
        return p

    @staticmethod
    def train_parser():
        p=SegMDTConfig.train_parser()
        p.set_defaults(ema_enabled=True,final_test_missing_rates=[0.,1.])
        return p

    @staticmethod
    def model_parser():
        p=SegMDTConfig.model_parser()
        for action in p._actions:
            if action.dest=='model_arch':
                action.choices=('pet_correction_memory',)
        p.set_defaults(model_arch='pet_correction_memory')
        return p

    @staticmethod
    def task_specific_parser():
        p=argparse.ArgumentParser('PET correction memory',add_help=False)
        p.add_argument('--full_ratio',type=float,default=.7)
        p.add_argument('--full_loss_weight',type=float,default=.7)
        p.add_argument('--missing_loss_weight',type=float,default=.3)
        p.add_argument('--ct_aux_weight',type=float,default=.1)
        p.add_argument('--ct_aux_enabled',type=str2bool,default=False,
                       help='Optional independent CT-only decoder/supervision; OFF does not remove the CT encoder or audit reference.')
        p.add_argument('--scale_loss_weight',type=float,default=.1)
        p.add_argument('--negative_region_weight',type=float,default=.5)
        p.add_argument('--memory_enabled',type=str2bool,default=True)
        p.add_argument('--bank_ema_enabled',type=str2bool,default=True,
                       help='Independent momentum source for bank writes/CT queries; not evaluation EMA.')
        p.add_argument('--bank_ema_start_epoch',type=int,default=0,
                       help='0 enables bank EMA immediately; N activates it after N epochs.')
        p.add_argument('--bank_ema_decay',type=float,default=.999)
        p.add_argument('--bank_ema_decay_warmup',type=str2bool,default=True)
        p.add_argument('--region_calibration',type=str2bool,default=True)
        p.add_argument('--memory_warmup_epochs',type=int,default=0)
        p.add_argument('--retrieval_start_epoch',type=int,default=2)
        p.add_argument('--memory_capacity',type=int,default=2048)
        p.add_argument('--memory_key_dim',type=int,default=32)
        p.add_argument('--memory_topk',type=int,default=4)
        p.add_argument('--memory_query_chunk',type=int,default=256)
        p.add_argument('--memory_min_entries',type=int,default=16)
        p.add_argument('--memory_max_age',type=int,default=128)
        p.add_argument('--memory_tokens_per_full',type=int,default=16)
        p.add_argument('--memory_temperature',type=float,default=.1)
        p.add_argument('--memory_gain_margin',type=float,default=.02)
        p.add_argument('--memory_min_support',type=float,default=.6)
        p.add_argument('--memory_max_negative_fraction',type=float,default=.1)
        p.add_argument('--memory_exclude_group',type=str2bool,default=False,
                       help='Enable only after confirming filename group meaning.')
        p.add_argument('--memory_diagnostics',type=str2bool,default=True)
        p.add_argument('--memory_log_every',type=int,default=100,
                       help='Sparse batch diagnostics, not extra validation passes.')
        p.add_argument('--memory_visualize',type=str2bool,default=False)
        p.add_argument('--memory_vis_every_epochs',type=int,default=5,
                       help='Also visualize epochs 1/2; only a few fixed validation slices.')
        p.add_argument('--memory_vis_samples',type=int,default=2)
        p.add_argument('--resume',type=str,default=None)
        p.add_argument('--init_checkpoint',type=str,default=None,
                       help='Optional aligned-add checkpoint; not required for joint training.')
        p.add_argument('--eval_checkpoint',type=str,default=None)
        p.add_argument('--smoke',type=str2bool,default=False,
                       help='Real encoder smoke on synthetic input; no pretrained download.')
        return p

    @classmethod
    def parse_arguments(cls):
        cfg=super().parse_arguments()
        if not 0 < cfg.full_ratio < 1 or cfg.batch_size < 2:
            raise ValueError('Need 0<full_ratio<1 and batch_size>=2')
        if cfg.batch_size*min(cfg.full_ratio,1-cfg.full_ratio) < 1:
            raise ValueError('batch_size too small to represent the requested ratio with both states')
        if min(cfg.full_loss_weight,cfg.missing_loss_weight)<=0:
            raise ValueError('Both main loss weights must be positive')
        if min(cfg.ct_aux_weight,cfg.scale_loss_weight,cfg.negative_region_weight)<0:
            raise ValueError('Auxiliary/calibration weights must be nonnegative')
        if cfg.scale_loss_weight<=0:
            raise ValueError('Scale heads need supervision to audit memory entries')
        if cfg.use_deep_supervision or cfg.deep_supervision or cfg.accumulation_steps!=1:
            raise ValueError('Use correction scale supervision and accumulation_steps=1')
        if cfg.retrieval_start_epoch <= cfg.memory_warmup_epochs:
            raise ValueError('Retrieval must start after the memory warmup period')
        if not (0<=cfg.memory_max_negative_fraction<=1 and .5<=cfg.memory_min_support<=1):
            raise ValueError('Invalid support/negative-fraction threshold')
        if cfg.memory_gain_margin<0 or cfg.memory_tokens_per_full<1:
            raise ValueError('Invalid gain margin/token quota')
        if cfg.bank_ema_start_epoch<0 or not 0<=cfg.bank_ema_decay<1:
            raise ValueError('Invalid bank EMA start epoch/decay')
        if min(cfg.memory_log_every,cfg.memory_vis_every_epochs,cfg.memory_vis_samples)<1:
            raise ValueError('Diagnostic/visualization intervals and sample count must be positive')
        return cfg
