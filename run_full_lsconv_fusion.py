"""Full-only Add -> LSConv at all four scales; same baseline training loop."""
import os

from configs.base import str2bool
from models.components.lsconv_fusion import LSConvFusion, STAGE_CHANNELS
from models.components.lsconv_ska import BACKENDS
from models.full_lsconv_fusion import (
    BASELINE_COMMIT, CONFIG_FIELDS, UPSTREAM_COMMIT, build_full_lsconv_model)
from run_full_add_baseline import FullAddConfig, main as run_full


class FullLSConvConfig(FullAddConfig):
    @staticmethod
    def model_parser():
        parser = FullAddConfig.model_parser()
        for action in parser._actions:
            if action.dest == 'model_arch':
                action.default = 'full_lsconv_fusion'
                action.choices = ('full_lsconv_fusion',)
        parser.add_argument('--lsconv_enabled', type=str2bool, default=True)
        parser.add_argument('--lsconv_backend', choices=BACKENDS, default='auto',
                            help='auto: official Triton on CUDA, torch reference on CPU; '
                                 'CUDA errors never silently fall back')
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        cfg.baseline_commit = BASELINE_COMMIT
        cfg.lsnet_upstream_commit = UPSTREAM_COMMIT
        cfg.lsconv_stage_channels = list(STAGE_CHANNELS)
        cfg.lsconv_stage_order = [1, 2, 3, 4]
        cfg.lsconv_final_bn_init = 'zeros'
        cfg.lsconv_large_kernel, cfg.lsconv_small_kernel, cfg.lsconv_groups = 7, 3, 8
        if os.path.isfile(os.path.join(cfg.checkpoint_dir, 'config_args.json')):
            raise FileExistsError('This run already exists; use a fresh --hash (no resume support)')
        return cfg


def main():
    return run_full(config_cls=FullLSConvConfig, model_builder=build_full_lsconv_model,
                    expected_fusion=LSConvFusion, experiment_type='full_lsconv_fusion',
                    result_config_fields=CONFIG_FIELDS, extra_grad_modules=('pet_align', 'fusion'))


if __name__ == '__main__':
    main()
