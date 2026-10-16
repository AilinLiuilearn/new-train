"""Full-only CMX/SKFF choices with unchanged baseline backbones/decoder."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state
from models.components.cmx_fusion import CMXFusion
from models.components.skff_fusion import SKFFFusion

BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'
CONFIG_FIELDS = ('cmx_ffm_enabled', 'cmx_frm_enabled', 'fusion_backend')


def fusion_class(backend):
    if backend == 'cmx':
        return CMXFusion
    if backend == 'skff':
        return SKFFFusion
    raise ValueError(f'Unsupported fusion_backend: {backend!r}')


class FullCMXPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, cmx_ffm_enabled=True, cmx_frm_enabled=False,
                 fusion_backend='cmx', **kwargs):
        selected = fusion_class(fusion_backend)
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('This Full CMX experiment does not use decoder deep supervision')
        super().__init__(**kwargs)
        self.fusion_backend = fusion_backend
        # Preserve shared baseline initialization and subsequent CPU RNG.
        with torch.random.fork_rng(devices=[]):
            self.fusion = (selected(cmx_ffm_enabled, cmx_frm_enabled)
                           if fusion_backend == 'cmx' else selected())

    def forward(self, ct, pet=None, pet_available=None, target_size=None, forward_mode='full'):
        if forward_mode not in ('full', 'auto'):
            raise ValueError('CMX experiment is Full-only; Missing is not connected')
        if pet_available is not None:
            state = _validate_state(pet_available, ct.shape[0])
            if not bool(torch.all(state == 1)):
                raise ValueError('CMX Full experiment requires real PET for every row')
        if pet is None or pet.shape != ct.shape or pet.device != ct.device:
            raise ValueError('Matching real CT/PET tensors are required')
        return super().forward(ct, pet=pet, target_size=target_size, forward_mode='full')


def build_full_cmx_model(cfg):
    # Pre-SKFF CMX checkpoints have no backend field; their meaning is CMX.
    missing = [name for name in CONFIG_FIELDS[:2] if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f'Explicit saved CMX flags are required: {missing}')
    model = FullCMXPETCT(
        cmx_ffm_enabled=cfg.cmx_ffm_enabled, cmx_frm_enabled=cfg.cmx_frm_enabled,
        fusion_backend=getattr(cfg, 'fusion_backend', 'cmx'),
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=bool(cfg.pretrained), decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    details = ('heads=1,2,5,8 reduction=1 norm=official '
               f'ffm={cfg.cmx_ffm_enabled} frm={cfg.cmx_frm_enabled}'
               if model.fusion_backend == 'cmx' else
               'height=2 reduction=8 output_scale=1 norm=none CMX_modules=off')
    print(f'[fusion] backend={model.fusion_backend} '
          f'scales=S1,S2,S3,S4 channels=64,128,320,512 {details} '
          'initialization=official output=replace_add '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
