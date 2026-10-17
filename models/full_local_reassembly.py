"""Full-only local-reassembly experiment; inherited Missing/auto seam is unchanged."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.local_reassembly import LocalReassemblyFusion

CONFIG_FIELDS = ('local_enabled','protect_enabled','protect_weight','baseline_weights_resolved',
                 'baseline_sha256','initialization_mode')


class FullLocalReassemblyPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, local_enabled=True, **kwargs):
        if kwargs.get('use_deep_supervision',False):
            raise ValueError('No deep supervision in this fusion comparison')
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.fusion = LocalReassemblyFusion(enabled=local_enabled)


def build_full_local_model(cfg):
    # Pure reconstruction only. Reference loading is training-runner-only,
    # so inference never needs the original checkpoint or pretrained files.
    model = FullLocalReassemblyPETCT(local_enabled=cfg.local_enabled,
        ct_backbone=cfg.ct_backbone,pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path if cfg.pretrained else None,
        pet_pretrained_path=cfg.pet_pretrained_path if cfg.pretrained else None,
        pretrained=bool(cfg.pretrained),decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print(f'[fusion] LocalReassembly enabled={cfg.local_enabled} scales=S1..S4 '
          'ranks=16,16,40,64 heads=2 window=3 null_value=0 core=fp32 '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}',flush=True)
    return {'model':model}
