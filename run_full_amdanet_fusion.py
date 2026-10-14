# -*- coding: utf-8 -*-
"""Full-only AMDANet experiment on the aligned Add baseline training loop."""
from models.components.amdanet_fusion import AMDANetFusion
from models.full_amdanet_fusion import (
    BASELINE_COMMIT, UPSTREAM_COMMIT, build_full_amdanet_model,
)
from run_full_add_baseline import FullAddConfig, main as run_full


class FullAMDANetConfig(FullAddConfig):
    @staticmethod
    def train_parser():
        parser = FullAddConfig.train_parser()
        for action in parser._actions:
            if action.dest == 'train_batch_mode':
                action.help = 'Full only: every sample uses real CT and real PET.'
        return parser

    @staticmethod
    def model_parser():
        parser = FullAddConfig.model_parser()
        for action in parser._actions:
            if action.dest == 'model_arch':
                action.default = 'full_amdanet_fusion'
                action.choices = ('full_amdanet_fusion',)
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        cfg.baseline_commit = BASELINE_COMMIT
        cfg.amdanet_upstream_commit = UPSTREAM_COMMIT
        cfg.fusion_formula = 'CT + PET + Global_Eliminating(*Local_Eliminating(CT, PET))'
        cfg.amdanet_channels = [64, 128, 320, 512]
        cfg.amdanet_num_heads = [1, 2, 5, 8]
        cfg.amdanet_sr_ratios = [8, 4, 2, 1]
        return cfg


def main():
    return run_full(config_cls=FullAMDANetConfig,
                    model_builder=build_full_amdanet_model,
                    expected_fusion=AMDANetFusion,
                    experiment_type='full_amdanet_fusion')


if __name__ == '__main__':
    main()
