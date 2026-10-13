"""Independent Full-only wrapper; original Add model/entry stay untouched."""
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, _check_finite
from models.structure_response_fusion import StructureResponseFusion


class FullStructureResponseModel(DualSharedAddPETCTBaseline):
    def __init__(self,fusion_enabled=True,fusion_heads=4,sae_expansion=2.66,**kwargs):
        super().__init__(**kwargs)
        if fusion_enabled:
            self.fusion=StructureResponseFusion(num_heads=fusion_heads,expansion=sae_expansion)

    def forward(self,ct,pet=None,pet_available=None,target_size=None,forward_mode='full',
                return_diagnostics=False,return_maps=False,use_add=False):
        if forward_mode!='full' or pet_available is not None:
            raise ValueError('This experiment is Full-only; no Missing or mixed-state route')
        if pet is None or ct.shape!=pet.shape:
            raise ValueError('Full-only experiment requires matching real CT/PET images')
        c=self._encode_ct(ct);p=self._encode_pet(pet);diagnostics={}
        if use_add:
            fused=[x+y for x,y in zip(c,p)]
        elif isinstance(self.fusion,StructureResponseFusion):
            result=self.fusion(c,p,return_diagnostics,return_maps)
            if return_diagnostics or return_maps:fused,diagnostics=result
            else:fused=result
        else:fused=self.fusion(c,p)
        _check_finite('fused_features',fused)
        out=self._decode(fused,ct.shape[-2:] if target_size is None else target_size)
        if return_diagnostics or return_maps:out['fusion_diagnostics']=diagnostics
        return out


def build_structure_model(cfg):
    return {'model':FullStructureResponseModel(
        ct_backbone=cfg.ct_backbone,pet_backbone=cfg.pet_backbone,
        ct_pretrained_path=cfg.ct_pretrained_path,pet_pretrained_path=cfg.pet_pretrained_path,
        decoder_channels=tuple(cfg.decoder_channels),pretrained=cfg.pretrained,
        use_deep_supervision=False,fusion_enabled=cfg.fusion_enabled,
        fusion_heads=cfg.fusion_heads,sae_expansion=cfg.sae_expansion)}
