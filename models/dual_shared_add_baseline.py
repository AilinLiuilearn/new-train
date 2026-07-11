import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe
from models.pg_mtr import PETGroundedMetabolicTokenRetrieval


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList(
            [ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)]
        )

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualSharedAddPETCTBaseline(nn.Module):
    def __init__(
        self,
        ct_backbone='convnextv2_nano',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        missing_mode='ct',
        pg_mtr_num_tokens=8,
        pg_mtr_temperature=0.07,
        pg_mtr_stages='deep',
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.missing_mode = str(missing_mode)
        self.pg_mtr_stages = str(pg_mtr_stages)

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')

        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        if len(ct_channels) != 4 or len(pet_channels) != 4:
            raise ValueError('Both encoders must output 4 stage features.')

        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.fusion = AddFusion()
        self.pg_mtr = PETGroundedMetabolicTokenRetrieval(
            pet_channels,
            num_tokens=pg_mtr_num_tokens,
            temperature=pg_mtr_temperature,
            stage_mode=self.pg_mtr_stages,
        )
        self.decoder = UNetStyleDecoder(
            pet_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct = self._to_3ch(ct)
        ct_feats = self.enc_ct(ct)
        _check_tensor_list('ct_feats', ct_feats)
        return self.ct_align(ct_feats)

    def _encode_pet(self, pet):
        pet = self._to_3ch(pet)
        pet_feats = self.enc_pet(pet)
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode(self, fused_feats, target_size):
        dec_out = self.decoder(fused_feats, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs

    def _forward_full(self, ct, pet, target_size):
        aligned_ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        fused_feats = self.fusion(aligned_ct_feats, pet_feats, pet_available=None)
        outputs = self._decode(fused_feats, target_size)
        if self.missing_mode == 'pg_mtr':
            _, pg_aux, pg_diag = self.pg_mtr(
                aligned_ct_feats=[feat.detach() for feat in aligned_ct_feats],
                pet_feats=[feat.detach() for feat in pet_feats],
                mode='full',
            )
            outputs['aux_losses'] = pg_aux
            outputs['diagnostics'] = pg_diag
        return outputs

    def _forward_missing(self, ct, target_size):
        aligned_ct_feats = self._encode_ct(ct)
        if self.missing_mode == 'pg_mtr':
            missing_feats, pg_aux, pg_diag = self.pg_mtr(aligned_ct_feats, pet_feats=None, mode='missing')
            outputs = self._decode(missing_feats, target_size)
            outputs['aux_losses'] = pg_aux
            outputs['diagnostics'] = pg_diag
            return outputs
        return self._decode(aligned_ct_feats, target_size)

    def _forward_auto(self, ct, pet, pet_available, target_size):
        pet_available = pet_available.long().view(-1)

        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size)

        if torch.all(pet_available == 0):
            return self._forward_missing(ct, target_size)

        aligned_ct_feats = self._encode_ct(ct)
        fused_feats = [feat.clone() for feat in aligned_ct_feats]

        full_idx = torch.nonzero(
            pet_available == 1,
            as_tuple=False,
        ).flatten()

        miss_idx = torch.nonzero(
            pet_available == 0,
            as_tuple=False,
        ).flatten()

        if full_idx.numel() > 0:
            pet_full = pet.index_select(0, full_idx)
            pet_feats = self._encode_pet(pet_full)
            ct_full_feats = [feat.index_select(0, full_idx) for feat in aligned_ct_feats]
            fused_full = self.fusion(
                ct_full_feats,
                pet_feats,
                pet_available=None,
            )
            for stage_idx, feat in enumerate(fused_full):
                fused_feats[stage_idx].index_copy_(0, full_idx, feat)

        pg_diag_miss = {}

        if miss_idx.numel() > 0:
            missing_ct_feats = [feat.index_select(0, miss_idx) for feat in aligned_ct_feats]
            if self.missing_mode == 'pg_mtr':
                missing_feats, _, pg_diag_miss = self.pg_mtr(
                    aligned_ct_feats=missing_ct_feats,
                    pet_feats=None,
                    mode='missing',
                )
            elif self.missing_mode == 'ct':
                missing_feats = missing_ct_feats
            else:
                raise ValueError(f'Unsupported missing_mode={self.missing_mode!r}')
            for stage_idx, feat in enumerate(missing_feats):
                fused_feats[stage_idx].index_copy_(0, miss_idx, feat)

        outputs = self._decode(
            fused_feats,
            target_size,
        )

        if self.missing_mode == 'pg_mtr':
            outputs['diagnostics'] = {
                'missing': pg_diag_miss,
                'full_count': int(full_idx.numel()),
                'missing_count': int(miss_idx.numel()),
            }

        return outputs

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        forward_mode = str(forward_mode)
        if forward_mode == 'full':
            if pet is None:
                raise ValueError('forward_mode="full" requires pet input.')
            return self._forward_full(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet_available is None:
                if pet is None:
                    raise ValueError('forward_mode="auto" requires pet_available or pet input.')
                pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
            if pet is None:
                raise ValueError('forward_mode="auto" requires pet input when mixed/full samples exist.')
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')

    @staticmethod
    def _finalize_decoder_output(dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}
