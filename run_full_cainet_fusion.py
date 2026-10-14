"""Full-only source CAINet CACR/GCM, with independent ablation switches."""
import os
from configs.base import str2bool
from models.components.cainet_fusion import CAINetFusion
from models.full_cainet_fusion import (
    BASELINE_COMMIT, UPSTREAM_COMMIT, build_full_cainet_model)
from run_full_add_baseline import FullAddConfig, main as run_full


class FullCAINetConfig(FullAddConfig):
    @staticmethod
    def model_parser():
        parser = FullAddConfig.model_parser()
        for action in parser._actions:
            if action.dest == 'model_arch':
                action.default = 'full_cainet_fusion'
                action.choices = ('full_cainet_fusion',)
        parser.add_argument('--cacr_enabled', type=str2bool, default=True,
                            help='Author Interv2 on all four stages S1/S2/S3/S4')
        parser.add_argument('--gcm_enabled', type=str2bool, default=True,
                            help='Author CRRM across S2/S3/S4, residual guidance at S4')
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        cfg.baseline_commit = BASELINE_COMMIT
        cfg.cainet_upstream_commit = UPSTREAM_COMMIT
        cfg.cacr_stages = [1, 2, 3, 4]
        cfg.gcm_grid_stage = 3
        cfg.gcm_concat_order = [4, 3, 2]
        cfg.gcm_injection_stage = 4
        cfg.gcm_projection_init = 'zeros'
        cfg.relation_precision = 'float32'
        if os.path.isfile(os.path.join(cfg.checkpoint_dir, 'config_args.json')):
            raise FileExistsError('This run already exists; use a fresh --hash (no resume support)')
        return cfg


def main():
    return run_full(config_cls=FullCAINetConfig, model_builder=build_full_cainet_model,
                    expected_fusion=CAINetFusion, experiment_type='full_cainet_fusion')


if __name__ == '__main__':
    main()
