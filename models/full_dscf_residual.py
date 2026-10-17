"""Stage-1 Full CT+PET with an Add-preserving four-scale DSCF correction."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.dscf_residual import DSCFResidualFusion

CONFIG_FIELDS = ('dscf_enabled',)


class FullDSCFResidualPETCT(DualSharedAddPETCTBaseline):
    """Inherited encoders/aligners/GN decoder and Missing/auto routing.

    Only Full fusion changes. Corrections go to decoder features and never
    feed back into subsequent encoder stages or build a prototype bank.
    """

    def __init__(self, dscf_enabled=True, image_size=512, **kwargs):
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('This Full comparison uses main BCE+Dice only')
        super().__init__(**kwargs)
        # New fusion initialization must not shift shared baseline weights
        # or the subsequent data/drop-path RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.fusion = DSCFResidualFusion(enabled=dscf_enabled, image_size=image_size)


def build_full_dscf_model(cfg):
    model = FullDSCFResidualPETCT(
        dscf_enabled=cfg.dscf_enabled, image_size=cfg.image_size_2d,
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=bool(cfg.pretrained), decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print(f'[fusion] backend=DSCFResidualFusion enabled={cfg.dscf_enabled} '
          f'scales=S1,S2,S3,S4 ranks=16,16,40,64 strides={model.fusion.strides} '
          'initialization=zero_output_projection output=CT+PET+delta core_precision=fp32 '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
