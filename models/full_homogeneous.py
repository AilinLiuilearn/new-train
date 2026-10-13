"""Full-only native-channel addition: two independent copies of one backbone.

No pre-fusion channel alignment is instantiated. Decoder projections are
unchanged decoder operations, with input widths taken from the actual encoder.
"""
from torch import nn

from models.components.backbones import create_feature_backbone, load_local_weights_safe
from models.components.add_fusion import AddFusion
from models.components.group_unet_decoder import UNetStyleDecoder
from models.full_structure_response import FullStructureResponseModel
from models.structure_response_fusion import StructureResponseFusion


NATIVE_CHANNELS = {
    'mit_b1': (64, 128, 320, 512),
    'convnextv2_nano': (80, 160, 320, 640),
}


class FullHomogeneousModel(FullStructureResponseModel):
    def __init__(self, encoder_pair='mit_b1', ct_pretrained_path=None,
                 pet_pretrained_path=None, pretrained=True,
                 decoder_channels=(512, 256, 128, 64), fusion_enabled=False,
                 fusion_heads=4, sae_expansion=2.66):
        # Build native backbones directly: do not construct/discard alignment
        # modules or a heterogeneous backbone (which also consumes RNG).
        nn.Module.__init__(self)
        if encoder_pair not in NATIVE_CHANNELS:
            raise ValueError(f'Unsupported homogeneous encoder: {encoder_pair}')
        self.encoder_pair = encoder_pair
        self.use_deep_supervision = False
        self.enc_ct = create_feature_backbone(encoder_pair, in_channels=3)
        self.enc_pet = create_feature_backbone(encoder_pair, in_channels=3)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path,
                                name='CT_Encoder', required=bool(pretrained))
        load_local_weights_safe(self.enc_pet, pet_pretrained_path,
                                name='PET_Encoder', required=bool(pretrained))
        self.encoder_channels = tuple(self.enc_ct.feature_info.channels())
        pet_channels = tuple(self.enc_pet.feature_info.channels())
        if self.encoder_channels != NATIVE_CHANNELS[encoder_pair] or pet_channels != self.encoder_channels:
            raise ValueError(f'Unexpected native channels: CT={self.encoder_channels}, PET={pet_channels}')
        # Identity is only an API seam for inherited encoder methods/loggers:
        # no learned projection, BN, activation or other transform before Add.
        self.ct_align = nn.Identity()
        self.pet_align = nn.Identity()
        self.fusion = (StructureResponseFusion(channels=self.encoder_channels,
                         num_heads=fusion_heads, expansion=sae_expansion)
                       if fusion_enabled else AddFusion())
        self.decoder = UNetStyleDecoder(self.encoder_channels,
            decoder_channels=tuple(decoder_channels), out_channels=1,
            use_deep_supervision=False)


def build_homogeneous_model(cfg):
    if cfg.ct_backbone != cfg.encoder_pair or cfg.pet_backbone != cfg.encoder_pair:
        raise ValueError('Both saved backbone names must match encoder_pair')
    return {'model': FullHomogeneousModel(
        encoder_pair=cfg.encoder_pair, ct_pretrained_path=cfg.ct_pretrained_path,
        pet_pretrained_path=cfg.pet_pretrained_path, pretrained=cfg.pretrained,
        decoder_channels=cfg.decoder_channels, fusion_enabled=cfg.fusion_enabled,
        fusion_heads=cfg.fusion_heads, sae_expansion=cfg.sae_expansion)}
