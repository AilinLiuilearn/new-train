"""Full-only official ConTriNet MFM with shallow-to-deep fusion."""
import os

from configs.base import str2bool
from models.components.contrinet_fusion import ConTriNetMFMFusion, OUTPUT_MODES
from models.full_contrinet_fusion import (
    BASELINE_COMMIT, CONFIG_FIELDS, UPSTREAM_COMMIT, build_full_contrinet_model)
from run_full_add_baseline import FullAddConfig, main as run_full


class FullConTriNetConfig(FullAddConfig):
    @staticmethod
    def model_parser():
        parser = FullAddConfig.model_parser()
        for action in parser._actions:
            if action.dest == 'model_arch':
                action.default = 'full_contrinet_fusion'
                action.choices = ('full_contrinet_fusion',)
        parser.add_argument('--mfm_enabled', type=str2bool, default=True)
        parser.add_argument('--mfm_cross_scale', type=str2bool, default=True,
                            help='Propagate author 64-channel fused features S1 -> S4')
        parser.add_argument('--mfm_output_mode', choices=OUTPUT_MODES, default='residual',
                            help='residual: CT+PET+zero-init projection(MFM); '
                                 'replace: projected MFM without outer CT+PET skip')
        return parser

    @classmethod
    def parse_arguments(cls):
        cfg = super().parse_arguments()
        cfg.baseline_commit = BASELINE_COMMIT
        cfg.contrinet_upstream_commit = UPSTREAM_COMMIT
        cfg.mfm_width = 64
        cfg.mfm_stage_order = [1, 2, 3, 4]
        cfg.mfm_projection_init = 'zeros' if cfg.mfm_output_mode == 'residual' else 'pytorch_default'
        if os.path.isfile(os.path.join(cfg.checkpoint_dir, 'config_args.json')):
            raise FileExistsError('This run already exists; use a fresh --hash (no resume support)')
        return cfg


def main():
    return run_full(config_cls=FullConTriNetConfig, model_builder=build_full_contrinet_model,
                    expected_fusion=ConTriNetMFMFusion, experiment_type='full_contrinet_fusion',
                    result_config_fields=CONFIG_FIELDS, extra_grad_modules=('pet_align', 'fusion'))


if __name__ == '__main__':
    main()
