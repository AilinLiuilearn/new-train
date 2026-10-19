"""Full-only protocol. Encoder pretraining is independent of task checkpoints."""
import argparse
from pathlib import Path

from configs.base import ConfigBase, str2bool


class FullConfig(ConfigBase):
    @classmethod
    def parser(cls, default_fusion='add'):
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('--root',default='/root/autodl-tmp/data/PCLT20K')
        p.add_argument('--train_split_file',default='train_original.txt')
        p.add_argument('--val_split_file',default='val.txt')
        p.add_argument('--test_split_file',default='test.txt')
        p.add_argument('--allow_val_equals_test',type=str2bool,default=False)
        p.add_argument('--image_size_2d',type=int,default=512)
        p.add_argument('--batch_size',type=int,default=16)
        p.add_argument('--num_workers',type=int,default=4)
        p.add_argument('--pin_memory',type=str2bool,default=True)
        p.add_argument('--aug_mode',choices=('cipa','light','none'),default='cipa')
        p.add_argument('--norm_mode',choices=('imagenet','cipa'),default='imagenet')
        p.add_argument('--ct_backbone',choices=('convnextv2_nano',),default='convnextv2_nano')
        p.add_argument('--pet_backbone',choices=('mit_b1',),default='mit_b1')
        p.add_argument('--pretrained',type=str2bool,default=True)
        p.add_argument('--ct_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/convnextv2_nano')
        p.add_argument('--pet_pretrained_path',default='/root/autodl-tmp/mkd-main/new-train/pretrained/mit-b1')
        p.add_argument('--decoder_channels',type=int,nargs=4,default=[512,256,128,64])
        p.add_argument('--fusion',choices=('add','asymmetric'),default=default_fusion)
        p.add_argument('--fusion_enabled',type=str2bool,default=True)
        p.add_argument('--fusion_ranks',type=int,nargs=4,default=[16,16,32,32])
        p.add_argument('--context_size',type=int,default=32)
        p.add_argument('--region_loss_weight',type=float,default=0.)
        p.add_argument('--epochs',type=int,default=60)
        p.add_argument('--learning_rate',type=float,default=4e-5)
        p.add_argument('--weight_decay',type=float,default=1e-4)
        p.add_argument('--cosine_warmup',type=int,default=3)
        p.add_argument('--cosine_min_lr',type=float,default=1e-6)
        p.add_argument('--lr_flat_ratio',type=float,default=.3)
        p.add_argument('--mixed_precision',type=str2bool,default=True)
        p.add_argument('--ema_enabled',type=str2bool,default=True)
        p.add_argument('--ema_decay',type=float,default=.999)
        p.add_argument('--ema_start_epoch',type=int,default=3)
        p.add_argument('--ema_decay_warmup',type=str2bool,default=True)
        p.add_argument('--grad_clip',type=float,default=5.)
        p.add_argument('--hd95_mode',choices=('legacy','surface','cipa_official'),default='cipa_official')
        p.add_argument('--early_stop_patience',type=int,default=60)
        p.add_argument('--random_state',type=int,default=2023)
        p.add_argument('--checkpoint_root',default='./checkpoints_new')
        p.add_argument('--hash',default=None)
        p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
        return p

    @classmethod
    def parse_arguments(cls, default_fusion='add', args=None):
        cfg=cls(args=vars(cls.parser(default_fusion).parse_args(args)))
        if cfg.image_size_2d<32 or cfg.image_size_2d%32 or cfg.batch_size<2:
            raise ValueError('Use image_size_2d divisible by 32 and batch_size>=2')
        if cfg.epochs<1 or cfg.learning_rate<=0 or cfg.context_size<1:
            raise ValueError('epochs, learning_rate and context_size must be positive')
        if cfg.region_loss_weight<0 or (cfg.region_loss_weight and
                (cfg.fusion!='asymmetric' or not cfg.fusion_enabled)):
            raise ValueError('Region loss requires enabled asymmetric fusion and nonnegative weight')
        if any(r<2 or r%2 for r in cfg.fusion_ranks):
            raise ValueError('fusion_ranks must be even positive integers')
        if not 0<cfg.ema_decay<1 or cfg.grad_clip<0 or not 0<=cfg.lr_flat_ratio<1:
            raise ValueError('Invalid EMA, gradient clipping or scheduler configuration')
        return cfg

    @property
    def checkpoint_dir(self):
        # Reading config must not create a run directory.
        return str(Path(self.checkpoint_root)/'MDT'/self.hash)
