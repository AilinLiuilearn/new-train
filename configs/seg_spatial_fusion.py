# -*- coding: utf-8 -*-
"""Independent config for the spatial-fusion experiment.

Subclasses :class:`SegMDTConfig` and only adds/overrides the fusion model
parser; the base config class is untouched.
"""
import argparse

from configs.base import str2bool
from configs.seg_mdt import SegMDTConfig
from models.dual_shared_spatial_fusion import MODEL_ARCH


class SegSpatialFusionConfig(SegMDTConfig):
    @staticmethod
    def model_parser():
        p = SegMDTConfig.model_parser()
        # Retarget --model_arch to the new experiment (argparse forbids adding
        # the same option twice, so patch the inherited action in place).
        for action in p._actions:
            if action.dest == 'model_arch':
                action.default = MODEL_ARCH
                action.choices = (MODEL_ARCH,)
        p.add_argument('--fusion_enabled', type=str2bool, default=True)
        p.add_argument('--fusion_mode', type=str, default='full',
                       choices=('full', 'local', 'axial', 'add'))
        p.add_argument('--fusion_attention_dim', type=int, default=64)
        p.add_argument('--fusion_num_heads', type=int, default=4)
        p.add_argument('--fusion_local_kernel_size', type=int, default=5)
        p.add_argument('--fusion_max_axis_length', type=int, default=128)
        p.add_argument('--fusion_axis_chunk_size', type=int, default=32)
        p.add_argument('--fusion_use_checkpoint', type=str2bool, default=True)
        return p

    @classmethod
    def parse_arguments(cls):
        parents = [cls.ddp_parser(), cls.data_parser(), cls.model_parser(),
                   cls.train_parser(), cls.logging_parser(), cls.task_specific_parser()]
        parser = argparse.ArgumentParser(add_help=True, parents=parents)
        config = cls()
        parser.parse_args(namespace=config)
        config._ensure_hash()
        return config