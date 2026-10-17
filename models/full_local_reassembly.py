"""Baseline encoder pretraining plus four-scale fusion, with BCE+Dice only."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.local_reassembly import LocalReassemblyFusion

CONFIG_FIELDS = ('local_enabled', 'initialization_mode', 'pretrained', 'loss_method')


class FullLocalReassemblyPETCT(DualSharedAddPETCTBaseline):
    """Preserve the original baseline constructor and Full/Missing routing."""
    def __init__(self, local_enabled=True, ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                 ct_pretrained_path=None, pet_pretrained_path=None, pretrained=True,
                 decoder_channels=(512,256,128,64), in_channels=3, out_channels=1,
                 use_deep_supervision=False):
        if use_deep_supervision:
            raise ValueError('This experiment uses main BCE+Dice only')
        # Same encoder factory, local weight loader, aligners and decoder as Add.
        # Explicitly suppress paths when opting out: optional loaders otherwise
        # still read a supplied existing path even if required=False.
        super().__init__(ct_backbone=ct_backbone, pet_backbone=pet_backbone,
            ct_pretrained_path=ct_pretrained_path if pretrained else None,
            pet_pretrained_path=pet_pretrained_path if pretrained else None,
            pretrained=pretrained, decoder_channels=tuple(decoder_channels),
            in_channels=in_channels, out_channels=out_channels, use_deep_supervision=False)
        # Adding fusion must not change shared initialization or subsequent RNG.
        with torch.random.fork_rng(devices=[]):
            self.fusion=LocalReassemblyFusion(enabled=local_enabled)


def build_full_local_model(cfg, *, initialize_encoders=True):
    """Load encoder sources at training init; full-checkpoint rebuild skips them."""
    load_encoders=bool(cfg.pretrained and initialize_encoders)
    model=FullLocalReassemblyPETCT(local_enabled=cfg.local_enabled,
        ct_backbone=cfg.ct_backbone,pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=getattr(cfg,'ct_pretrained_path',None),
        pet_pretrained_path=getattr(cfg,'pet_pretrained_path',None),
        pretrained=load_encoders,decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    mode=('encoder_pretrained_fresh_segmentation' if load_encoders else
          ('checkpoint_reconstruction' if not initialize_encoders else 'all_random'))
    print(f'[model] Single PETCT network; initialization={mode} '
          f'fusion={cfg.local_enabled} scales=S1..S4 ranks=16,16,40,64 '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}',flush=True)
    return {'model':model}
