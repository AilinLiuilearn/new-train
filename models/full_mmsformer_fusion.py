"""MMSFormer fusion at the existing four-scale Full fusion boundary."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.mmsformer_fusion import MMSFormerFusion

CONFIG_FIELDS = ('mmsformer_enabled',)


class FullMMSFormerPETCT(DualSharedAddPETCTBaseline):
    """Same encoders/aligners/GN decoder; only Full fusion is replaced.

    The Stage-1 runner uses Full only. Inherited Missing and auto semantics
    are preserved: Missing does not encode PET or call this fusion module.
    Fused features feed only the decoder, never subsequent encoder stages.
    """

    def __init__(self, mmsformer_enabled=True, **kwargs):
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('This Full comparison uses main BCE+Dice only')
        super().__init__(**kwargs)
        # Keep all existing weights and subsequent CPU RNG identical to Add.
        # This does not zero/reinitialize any native FusionBlock parameter.
        with torch.random.fork_rng(devices=[]):
            self.fusion = MMSFormerFusion(enabled=mmsformer_enabled)


def build_full_mmsformer_model(cfg):
    model = FullMMSFormerPETCT(
        mmsformer_enabled=cfg.mmsformer_enabled,
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=bool(cfg.pretrained), decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print(f'[fusion] backend=MMSFormerFusion enabled={cfg.mmsformer_enabled} '
          'scales=S1,S2,S3,S4 channels=64,128,320,512 modalities=CT,PET '
          'initialization=native_FusionBlock kernels=3,5,7 SE_reduction=16 output=replace_add '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
