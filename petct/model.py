"""Confirmed SCGR T network and strict loading of its original checkpoint."""
import torch
from torch import nn

from .backbones import create_feature_backbone, load_local_weights_safe
from .channel_align import StageChannelAlign
from .group_unet_decoder import UNetStyleDecoder
from .mlrf import DEFAULT_FUSION_CONFIG, MultiScaleLocalRegionResidualFusion
from .tcsr import MultiScaleRefiner


CHANNELS = (64, 128, 320, 512)


class Model(nn.Module):
    """Independent ConvNeXtV2-Nano CT and MiT-B1 PET branches."""

    def __init__(self, text_anchor, *, ct_pretrained=None, pet_pretrained=None):
        super().__init__()
        if (ct_pretrained is None) != (pet_pretrained is None):
            raise ValueError('Provide both encoder pretraining paths together')
        self.enc_ct = create_feature_backbone('convnextv2_nano', in_channels=3)
        self.ct_align = StageChannelAlign(self.enc_ct.feature_info.channels(), CHANNELS)
        self.decoder_full = UNetStyleDecoder(CHANNELS, norm='GN')
        self.enc_pet = create_feature_backbone('mit_b1', in_channels=3)
        self.pet_align = StageChannelAlign(self.enc_pet.feature_info.channels(), CHANNELS)
        if ct_pretrained is not None:
            load_local_weights_safe(self.enc_ct, ct_pretrained, name='CT_Encoder')
            load_local_weights_safe(self.enc_pet, pet_pretrained, name='PET_Encoder')
        self.fusion = MultiScaleLocalRegionResidualFusion(**DEFAULT_FUSION_CONFIG)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2203)
            self.refiner = MultiScaleRefiner(text_anchor)

    @staticmethod
    def _input(image):
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError('Expected Bx1xHxW or Bx3xHxW')
        return image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image

    def forward(self, ct, pet, capture=False):
        if ct.shape != pet.shape:
            raise ValueError('Paired CT and PET must share the same shape')
        c = self.ct_align(self.enc_ct(self._input(ct)))
        p = self.pet_align(self.enc_pet(self._input(pet)))
        fused = self.fusion(c, p)
        refined, scores, details = self.refiner(fused, capture)
        output = self.decoder_full(refined, ct.shape[-2:], return_native=True)
        output['support_logits'] = scores
        if capture:
            output['capture'] = {'fused': fused, 'refined': refined, 'refiner': details}
        return output


def load_model(checkpoint, device='cpu'):
    bundle = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    if bundle.get('format') != 'scgr_support_conditioned_v1':
        raise ValueError('Expected an original SCGR T checkpoint')
    cfg = bundle['config']
    if (cfg['recipe'] != 'T' or not cfg['text_enabled'] or cfg['score_enabled'] or
            cfg['decoder_norm'] != 'GN' or cfg['fusion_config'] != DEFAULT_FUSION_CONFIG or
            cfg['refiner_seed'] != 2203 or cfg['refiner_width'] != 32 or cfg['context_pool'] != 4 or
            cfg['norm_mode'] != 'imagenet' or cfg['image_size_2d'] != 512 or
            cfg.get('ct_backbone', 'convnextv2_nano') != 'convnextv2_nano' or
            cfg.get('pet_backbone', 'mit_b1') != 'mit_b1' or
            cfg.get('tcsr_ablation') or cfg.get('mlrf_ablation')):
        raise ValueError('The checkpoint configuration differs from the confirmed model')
    weights = bundle['weights']
    model = Model(weights['refiner.text_anchor'])
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval().requires_grad_(False)
