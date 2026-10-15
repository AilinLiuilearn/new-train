"""Full-only model integration; baseline encoders, aligners and decoder unchanged."""
import torch

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _validate_state
from models.components.lesion_guided_fusion import LesionGuidedFusion

BASELINE_COMMIT = '663f9786122365d8b2774e4497df92bc06b715d5'
CONFIG_FIELDS = ('lesion_fusion_enabled', 'lesion_correction_enabled',
                 'lesion_guidance_enabled', 'lesion_aux_loss_enabled',
                 'lesion_aux_loss_weight', 'lesion_widths')


class FullLesionGuidedPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, lesion_fusion_enabled=True, lesion_correction_enabled=True,
                 lesion_guidance_enabled=True, lesion_widths=(16, 32, 64, 64), **kwargs):
        if kwargs.get('use_deep_supervision', False):
            raise ValueError('Decoder deep supervision is not part of this Full experiment')
        super().__init__(**kwargs)
        # Draw module weights only after building the complete baseline, and
        # restore CPU RNG so its presence does not shift later training randomness.
        with torch.random.fork_rng(devices=[]):
            self.fusion = LesionGuidedFusion(lesion_fusion_enabled, lesion_correction_enabled,
                                             lesion_guidance_enabled, lesion_widths)

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='full', return_aux=False):
        if forward_mode not in ('full', 'auto'):
            raise ValueError('This experiment is Full-only; Missing is not connected')
        if pet_available is not None and not bool(torch.all(_validate_state(pet_available, ct.shape[0]) == 1)):
            raise ValueError('Full experiment requires real PET for every sample')
        if pet is None or pet.shape != ct.shape or pet.device != ct.device:
            raise ValueError('Matching real CT/PET tensors are required')
        ct_features, pet_features = self._encode_ct(ct), self._encode_pet(pet)
        fused, auxiliary = self.fusion.forward_with_aux(ct_features, pet_features, return_aux=return_aux)
        outputs = self._decode(fused, target_size or ct.shape[-2:])
        if return_aux:
            outputs['aux'] = {'lesion_logits': auxiliary}
        return outputs


def build_full_lesion_guided_model(cfg):
    missing = [name for name in CONFIG_FIELDS if not hasattr(cfg, name)]
    if missing:
        raise ValueError(f'Explicit saved lesion configuration required: {missing}')
    model = FullLesionGuidedPETCT(
        **{name: getattr(cfg, name) for name in CONFIG_FIELDS if name not in
           ('lesion_aux_loss_enabled', 'lesion_aux_loss_weight')},
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path, pet_pretrained_path=cfg.pet_pretrained_path,
        pretrained=bool(cfg.pretrained), decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print('[lesion_guided] ' + ' '.join(f'{name}={getattr(cfg, name)}' for name in CONFIG_FIELDS)
          + f' scales=S1,S2,S3,S4 params_fusion={sum(p.numel() for p in model.fusion.parameters())}'
          + f' params_total={sum(p.numel() for p in model.parameters())}', flush=True)
    return {'model': model}
