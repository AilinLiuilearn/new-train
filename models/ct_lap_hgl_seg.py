"""CT encoder-decoder with optional Laplacian HGL PET spatial modulation on C4."""

import torch
import torch.nn as nn

from models.baseline_petct_unet import PETCTBaselineUNet, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.pet_lap_hgl_prior import DeepPETSpatialModulation, PETIntensitySpatialPrior


class CTLapHGLSegmentation(nn.Module):
    """
    CT is the main segmentation path. PET only modulates deepest CT feature C4.

    pet_prior_type:
        none      -> CT-only baseline
        intensity -> resized PET intensity sanity check
        lap_hgl   -> Laplacian high-frequency guided localization prior
    """

    def __init__(
        self,
        ct_backbone='convnext_tiny',
        ct_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        pet_prior_type='lap_hgl',
        pet_prior_size='lite',
        pet_prior_c4_channels=64,
        pet_prior_mid_channels=32,
        pet_channels=(24, 32, 48, 64),
        pet_fuse_mid_channels=32,
        pet_gn_groups=8,
        **kwargs,
    ):
        super().__init__()
        self.pet_prior_type = str(pet_prior_type)
        if self.pet_prior_type not in ('none', 'intensity', 'lap_hgl'):
            raise ValueError(f'Unsupported pet_prior_type={self.pet_prior_type}')

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        encoder_channels = self.enc_ct.feature_info.channels()
        self.use_deep_supervision = bool(use_deep_supervision)

        self.pet_prior = None
        if self.pet_prior_type == 'intensity':
            self.pet_prior = PETIntensitySpatialPrior()
        elif self.pet_prior_type == 'lap_hgl':
            self.pet_prior = DeepPETSpatialModulation(
                c4_channels=pet_prior_c4_channels,
                pet_channels=pet_channels,
                fuse_mid_channels=pet_fuse_mid_channels,
                gn_groups=pet_gn_groups,
                pet_prior_size=pet_prior_size,
                pet_prior_mid_channels=pet_prior_mid_channels,
            )

        self.decoder = UNetStyleDecoder(
            encoder_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    @staticmethod
    def _to_1ch_pet(pet):
        if pet.shape[1] == 1:
            return pet
        return pet.mean(dim=1, keepdim=True)

    def forward(self, ct, pet=None, pet_available=None, target_size=None, return_aux=False):
        if target_size is None:
            target_size = ct.shape[-2:]

        ct_feats = self.enc_ct(self._to_3ch(ct))
        _check_tensor_list('ct_feats', ct_feats)
        c1, c2, c3, c4 = ct_feats

        aux = {}
        if self.pet_prior is not None:
            if pet is None:
                raise ValueError('pet must be provided when pet_prior_type is not none.')
            pet_1ch = self._to_1ch_pet(pet.float())
            c4, pet_aux = self.pet_prior(c4, pet_1ch)
            aux.update(pet_aux)
            ct_feats = [c1, c2, c3, _sanitize(c4)]

        dec_out = self.decoder(ct_feats, target_size)
        outputs = PETCTBaselineUNet._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = aux
        return outputs
