"""Standalone Full SKFF entry point; reuses the current BCE+Dice/EMA loop.

Same options as run_full_cmx_fusion.py, with SKFF selected by default.
Use scripts/run_full_skff_fusion.sh for the existing server protocol.
"""
from configs.full_cmx import FullCMXConfig
from run_full_cmx_fusion import main


class FullSKFFConfig(FullCMXConfig):
    @staticmethod
    def model_parser():
        parser = FullCMXConfig.model_parser()
        parser.set_defaults(fusion_backend='skff', cmx_ffm_enabled=False,
                            cmx_frm_enabled=False)
        return parser


if __name__ == '__main__':
    main(config_cls=FullSKFFConfig)
