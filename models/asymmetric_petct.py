"""Heterogeneous baseline plus four-scale DSSA/EGA fusion; decoder unchanged."""
import torch
from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline
from models.components.asymmetric_fusion import AsymmetricFusion


class AsymmetricPETCT(DualSharedAddPETCTBaseline):
    def __init__(self, *, fusion_enabled=True, fusion_ranks=(16,16,32,32),
                 context_size=32, norm_mode='imagenet', **kwargs):
        super().__init__(**kwargs)
        # Added parameter initialization must not shift the baseline sampler /
        # encoder DropPath RNG stream when paired experiments share a seed.
        with torch.random.fork_rng(devices=[]):
            self.fusion = AsymmetricFusion(ranks=fusion_ranks,context_size=context_size,
                                          enabled=fusion_enabled,norm_mode=norm_mode)

    def _forward_full(self, ct, pet, target_size):
        if pet is None:
            raise ValueError('Full requires real PET')
        c,p=self._encode_ct(ct),self._encode_pet(pet)
        fused,regions=self.fusion(c,p,ct)
        out=self._decode(fused,target_size)
        out['aux']={'region_logits':regions}
        # Scalars only, detached: do not cache feature graphs on model/EMA.
        diagnostics={}
        for i,(f,x,y) in enumerate(zip(fused,c,p),1):
            diagnostics[f'delta_ratio_s{i}']=((f.float()-(x+y).float()).detach().norm()
                                             /(x.float()+y.float()).detach().norm().clamp_min(1e-8))
        out['diagnostics']=diagnostics
        return out

    def forward(self, ct, pet=None, target_size=None, forward_mode='full'):
        target_size=ct.shape[-2:] if target_size is None else target_size
        if forward_mode=='full':
            return self._forward_full(ct,pet,target_size)
        if forward_mode=='missing':
            # Explicit CT bypass: never invokes PET encoder, region head or EGA.
            return self._forward_missing(ct,None,target_size)
        raise ValueError('This clean Full experiment supports full or missing inference only')
