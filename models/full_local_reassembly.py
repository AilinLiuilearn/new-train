"""One randomly initialized PET-CT network; no pretrained or baseline loading.

The optional training loss compares fused and Add logits produced by the
SAME encoder features and SAME decoder. It creates no second model.
"""
import torch
from torch import nn
from torch.nn import functional as F

from models.dual_shared_add_baseline import DualSharedAddPETCTBaseline, DECODER_INPUT_CHANNELS
from models.components.backbones import create_feature_backbone
from models.components.channel_align import StageChannelAlign
from models.components.group_unet_decoder import UNetStyleDecoder
from models.components.local_reassembly import LocalReassemblyFusion

CONFIG_FIELDS = ('local_enabled','protect_enabled','protect_weight','initialization_mode','pretrained')


class FullLocalReassemblyPETCT(DualSharedAddPETCTBaseline):
    """Retain baseline Full/Missing routing, but never invoke its weight loader."""
    def __init__(self, local_enabled=True, ct_backbone='convnextv2_nano', pet_backbone='mit_b1',
                 decoder_channels=(512,256,128,64), in_channels=3, out_channels=1,
                 use_deep_supervision=False):
        if use_deep_supervision:
            raise ValueError('This experiment uses main BCE+Dice and optional internal Add correction only')
        nn.Module.__init__(self)
        self.use_deep_supervision=False
        # Factory constructors explicitly build random backbones, with no from_pretrained call.
        self.enc_ct=create_feature_backbone(ct_backbone,in_channels=in_channels)
        self.enc_pet=create_feature_backbone(pet_backbone,in_channels=in_channels)
        self.ct_align=StageChannelAlign(list(self.enc_ct.feature_info.channels()),list(DECODER_INPUT_CHANNELS))
        self.pet_align=StageChannelAlign(list(self.enc_pet.feature_info.channels()),list(DECODER_INPUT_CHANNELS))
        self.decoder=UNetStyleDecoder(list(DECODER_INPUT_CHANNELS),decoder_channels=tuple(decoder_channels),
                                     out_channels=out_channels,use_deep_supervision=False)
        with torch.random.fork_rng(devices=[]):
            self.fusion=LocalReassemblyFusion(enabled=local_enabled)

    def forward(self,ct,pet=None,pet_available=None,target_size=None,forward_mode='auto',return_add_logits=False):
        if type(return_add_logits) is not bool:
            raise ValueError('return_add_logits must be boolean')
        if not return_add_logits:
            return super().forward(ct,pet,pet_available,target_size,forward_mode)
        if forward_mode != 'full':
            raise ValueError('Internal Add correction is only supported for explicit Full training')
        if pet is None:
            raise ValueError('Full path requires PET')
        target_size=ct.shape[-2:] if target_size is None else target_size
        ct_feats=self._encode_ct(ct)
        pet_feats=self._encode_pet(pet)
        out=self._decode(self.fusion(ct_feats,pet_feats),target_size)
        # Decoder is stateless Conv+GroupNorm+ReLU (no BN/dropout). Same mode,
        # same AMP precision, no extra encoder pass, and no extra model/weights.
        with torch.no_grad():
            added=[]
            for c,p in zip(ct_feats,pet_feats):
                c,p=c.detach(),p.detach()
                if c.shape[-2:] != p.shape[-2:]:
                    p=F.interpolate(p,c.shape[-2:],mode='bilinear',align_corners=False)
                added.append(c+p)
            out['add_logits']=self._decode(added,target_size)['logits'].detach()
        return out


def build_full_local_model(cfg):
    model=FullLocalReassemblyPETCT(local_enabled=cfg.local_enabled,
        ct_backbone=cfg.ct_backbone,pet_backbone=cfg.pet_backbone,
        decoder_channels=tuple(cfg.decoder_channels),
        use_deep_supervision=bool(cfg.use_deep_supervision or cfg.deep_supervision))
    print(f'[model] Single PETCT network; initialization=all_random external_weights=none '
          f'fusion={cfg.local_enabled} scales=S1..S4 ranks=16,16,40,64 '
          f'params_fusion={sum(p.numel() for p in model.fusion.parameters())} '
          f'params_total={sum(p.numel() for p in model.parameters())}',flush=True)
    return {'model':model}
