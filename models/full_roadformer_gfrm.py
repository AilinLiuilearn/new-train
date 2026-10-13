"""Full-only GFRM wrapper; baseline files and decoder remain unchanged."""
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _check_finite
from models.roadformer_gfrm import RoadFormerGFRMFusion


class FullRoadFormerGFRMModel(DualSharedAddPETCTBaseline):
    def __init__(self, gfrm_enabled=True, image_size_2d=512, **kwargs):
        super().__init__(**kwargs)
        if int(image_size_2d) < 32 or int(image_size_2d) % 32:
            raise ValueError('image_size_2d must be a positive multiple of 32')
        if gfrm_enabled:
            size = int(image_size_2d)
            self.fusion = RoadFormerGFRMFusion(
                feature_sizes=tuple((size//s, size//s) for s in (4,8,16,32)))

    def forward(self, ct, pet=None, pet_available=None, target_size=None,
                forward_mode='full', return_diagnostics=False, return_maps=False):
        # Reject before either encoder is invoked: no Missing simulation here.
        if forward_mode != 'full' or pet_available is not None:
            raise ValueError('GFRM runner is Full-only; no Missing/mixed-state route')
        if pet is None or ct.shape != pet.shape:
            raise ValueError('Full-only model requires paired real CT and PET images')
        c, p = self._encode_ct(ct), self._encode_pet(pet)
        diagnostics = {}
        if isinstance(self.fusion, RoadFormerGFRMFusion):
            result = self.fusion(c, p, return_diagnostics, return_maps)
            if return_diagnostics or return_maps:
                fused, diagnostics = result
            else:
                fused = result
        else:
            fused = self.fusion(c, p)
        _check_finite('gfrm_fused', fused)
        out = self._decode(fused, ct.shape[-2:] if target_size is None else target_size)
        if return_diagnostics or return_maps:
            out['fusion_diagnostics'] = diagnostics
        return out


def build_gfrm_model(cfg):
    return {'model': FullRoadFormerGFRMModel(
        gfrm_enabled=cfg.gfrm_enabled, image_size_2d=cfg.image_size_2d,
        ct_backbone=cfg.ct_backbone, pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path,
        pet_pretrained_path=cfg.pet_pretrained_path,
        decoder_channels=tuple(cfg.decoder_channels),
        pretrained=bool(cfg.pretrained), use_deep_supervision=False)}
