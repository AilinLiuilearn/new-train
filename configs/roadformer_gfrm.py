"""Separate config; inherits every baseline training default unchanged."""
from configs.base import str2bool
from run_full_add_baseline import FullAddConfig


class FullGFRMConfig(FullAddConfig):
    @staticmethod
    def task_specific_parser():
        p = FullAddConfig.task_specific_parser()
        p.add_argument('--gfrm_enabled', type=str2bool, default=True)
        p.add_argument('--fusion_log_every', type=int, default=100)
        p.add_argument('--fusion_visualize', type=str2bool, default=True)
        p.add_argument('--fusion_vis_every', type=int, default=5)
        p.add_argument('--fusion_vis_samples', type=int, default=2)
        p.add_argument('--allow_cpu', type=str2bool, default=False,
                       help='Only for small synthetic smoke tests; formal training needs CUDA.')
        return p
