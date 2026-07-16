import torch
import torch.nn as nn
import torch.nn.functional as F

from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe
from models.simmlm_dmome_fusion import (
    DEFAULT_HYBRID_CONCAT_STAGES,
    DEFAULT_HYBRID_DMOME_STAGES,
    DEFAULT_PRIOR_GATE_STAGES,
    StageWiseHybridConcatDMoMEFusion,
    StageWiseSimMLMAlignedDMoME,
    make_full_pet_available,
)


def _check_tensor(name, x):
    if torch.is_tensor(x):
        x_cpu = x.detach().float().cpu()
        if not torch.isfinite(x_cpu).all():
            raise RuntimeError(f'[NaN/Inf] {name} contains invalid values')


def _check_tensor_list(name, xs):
    for i, x in enumerate(xs):
        _check_tensor(f'{name}[{i}]', x)


def _sanitize(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class UNetStyleDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels=(64, 128, 320, 512),
        decoder_channels=(512, 256, 128, 64),
        out_channels=1,
        use_deep_supervision=False,
    ):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d4, d3, d2, d1 = decoder_channels
        self.use_deep_supervision = bool(use_deep_supervision)
        self.proj4 = ConvBNAct(c4, d4, kernel_size=1)
        self.proj3 = ConvBNAct(c3, d3, kernel_size=1)
        self.proj2 = ConvBNAct(c2, d2, kernel_size=1)
        self.proj1 = ConvBNAct(c1, d1, kernel_size=1)
        self.fuse3 = nn.Sequential(
            ConvBNAct(d4 + d3, d3, kernel_size=3),
            ConvBNAct(d3, d3, kernel_size=3),
        )
        self.fuse2 = nn.Sequential(
            ConvBNAct(d3 + d2, d2, kernel_size=3),
            ConvBNAct(d2, d2, kernel_size=3),
        )
        self.fuse1 = nn.Sequential(
            ConvBNAct(d2 + d1, d1, kernel_size=3),
            ConvBNAct(d1, d1, kernel_size=3),
        )
        self.seg_head = nn.Conv2d(d1, out_channels, kernel_size=1)
        if self.use_deep_supervision:
            self.aux_head_d2 = nn.Conv2d(d2, out_channels, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(d3, out_channels, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(d4, out_channels, kernel_size=1)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _upsample_size(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, target_size):
        x1, x2, x3, x4 = features
        d4 = self.proj4(x4)
        s3 = self.proj3(x3)
        d3 = self.fuse3(torch.cat([self._upsample_to(d4, s3), s3], dim=1))
        s2 = self.proj2(x2)
        d2 = self.fuse2(torch.cat([self._upsample_to(d3, s2), s2], dim=1))
        s1 = self.proj1(x1)
        d1 = self.fuse1(torch.cat([self._upsample_to(d2, s1), s1], dim=1))
        logits = self.seg_head(d1)
        final_logits = self._upsample_size(logits, target_size)

        if not self.use_deep_supervision:
            return {'logits': final_logits}

        logits_d2 = self.aux_head_d2(d2)
        logits_d3 = self.aux_head_d3(d3)
        logits_d4 = self.aux_head_d4(d4)
        return {
            'logits': final_logits,
            'aux_logits': [logits_d2, logits_d3, logits_d4],
        }


class AddFusion(nn.Module):
    """Element-wise addition fusion for aligned CT/PET encoder features."""

    def forward(self, ct_feats, pet_feats, pet_available=None):
        if pet_available is None:
            pet_available = make_full_pet_available(ct_feats[0].shape[0], ct_feats[0].device)
        else:
            pet_available = pet_available.long().view(-1)

        B = ct_feats[0].shape[0]
        pet_mask = pet_available.float().view(B, 1, 1, 1)

        fused = []
        for ct_feat, pet_feat in zip(ct_feats, pet_feats):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(
                    pet_feat,
                    size=ct_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            pet_feat = pet_feat * pet_mask
            fused.append(_sanitize(ct_feat + pet_feat))
        return fused


class ConcatConvFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels=None):
        super().__init__()
        if len(ct_channels) != len(pet_channels):
            raise ValueError('ct_channels and pet_channels must have same length')
        out_channels = list(out_channels or ct_channels)
        if len(out_channels) != len(ct_channels):
            raise ValueError('out_channels must have same length as encoder channels')
        self.fuse = nn.ModuleList([
            ConvBNAct(c_ct + c_pet, c_out, kernel_size=1)
            for c_ct, c_pet, c_out in zip(ct_channels, pet_channels, out_channels)
        ])

    def forward(self, ct_feats, pet_feats):
        fused = []
        for ct_feat, pet_feat, fuse in zip(ct_feats, pet_feats, self.fuse):
            if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
                pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:], mode='bilinear', align_corners=False)
            fused.append(_sanitize(fuse(torch.cat([ct_feat, pet_feat], dim=1))))
        return fused


class PETCTBaselineUNet(nn.Module):
    def __init__(
        self,
        ct_backbone='mit_b1',
        pet_backbone='mit_b1',
        ct_pretrained_path=None,
        pet_pretrained_path=None,
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        fusion_type='concat_conv',
        dmome_expert_reduction=4,
        dmome_use_status_token=True,
        dmome_temperature=1.0,
        dmome_init_ct_bias=0.0,
        dmome_output_proj=False,
        dmome_norm_groups=8,
        use_channel_prior_gate=False,
        prior_gate_stages=DEFAULT_PRIOR_GATE_STAGES,
        hybrid_concat_stages=DEFAULT_HYBRID_CONCAT_STAGES,
        hybrid_dmome_stages=DEFAULT_HYBRID_DMOME_STAGES,
        use_deep_supervision=False,
        **kwargs,
    ):
        super().__init__()
        self.fusion_type = str(fusion_type)
        if self.fusion_type not in (
            'concat_conv', 'dmome', 'dmome_channel_prior_gate', 'hybrid_concat_dmome', 'add',
        ):
            raise ValueError(
                f'Unsupported fusion_type={self.fusion_type}. '
                'Use concat_conv, add, dmome, dmome_channel_prior_gate, or hybrid_concat_dmome.'
            )
        self.use_hybrid_fusion = self.fusion_type == 'hybrid_concat_dmome'
        self.use_channel_prior_gate = (
            bool(use_channel_prior_gate)
            or self.fusion_type == 'dmome_channel_prior_gate'
        )
        self.use_dmome = self.fusion_type in (
            'dmome', 'dmome_channel_prior_gate', 'hybrid_concat_dmome',
        )
        self.use_deep_supervision = bool(use_deep_supervision)

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = self.enc_ct.feature_info.channels()
        pet_channels = self.enc_pet.feature_info.channels()

        self.fusion = None
        self.stage_fusion = None
        if self.use_hybrid_fusion:
            self.stage_fusion = StageWiseHybridConcatDMoMEFusion(
                ct_channels_list=tuple(ct_channels),
                pet_channels_list=tuple(pet_channels),
                concat_stages=tuple(int(s) for s in hybrid_concat_stages),
                dmome_stages=tuple(int(s) for s in hybrid_dmome_stages),
                expert_reduction=dmome_expert_reduction,
                use_status_token=dmome_use_status_token,
                temperature=dmome_temperature,
                init_ct_bias=dmome_init_ct_bias,
                output_proj=dmome_output_proj,
                norm_groups=dmome_norm_groups,
            )
        elif self.use_dmome:
            self.stage_fusion = StageWiseSimMLMAlignedDMoME(
                channels_list=tuple(ct_channels),
                expert_reduction=dmome_expert_reduction,
                use_status_token=dmome_use_status_token,
                temperature=dmome_temperature,
                init_ct_bias=dmome_init_ct_bias,
                output_proj=dmome_output_proj,
                norm_groups=dmome_norm_groups,
                use_channel_prior_gate=self.use_channel_prior_gate,
                prior_gate_stages=tuple(int(s) for s in prior_gate_stages),
            )
        elif self.fusion_type == 'add':
            self.fusion = AddFusion()
        else:
            self.fusion = ConcatConvFusion(ct_channels, pet_channels, out_channels=ct_channels)

        self.decoder = UNetStyleDecoder(
            ct_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=self.use_deep_supervision,
        )

    def set_modality_prior_text_embeds(self, text_embeds):
        if not self.use_channel_prior_gate or self.stage_fusion is None:
            return
        self.stage_fusion.set_modality_prior_text_embeds(text_embeds)

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, pet_available=None, target_size=None, return_aux=False):
        if target_size is None:
            target_size = ct.shape[-2:]
        ct = self._to_3ch(ct)
        pet = self._to_3ch(pet)
        ct_feats = self.enc_ct(ct)
        pet_feats = self.enc_pet(pet)
        _check_tensor_list('ct_feats', ct_feats)
        _check_tensor_list('pet_feats', pet_feats)

        if self.use_dmome:
            if pet_available is None:
                pet_available = make_full_pet_available(ct.shape[0], ct.device)
            else:
                pet_available = pet_available.long().view(-1)
            fused_feats, fusion_aux = self.stage_fusion(ct_feats, pet_feats, pet_available)
            fused_feats = [_sanitize(feat) for feat in fused_feats]
            dec_out = self.decoder(fused_feats, target_size)
            outputs = self._finalize_decoder_output(dec_out)
            _check_tensor('logits', outputs['logits'])
            if return_aux:
                outputs['fusion_aux'] = fusion_aux
            return outputs

        if pet_available is None:
            pet_available = make_full_pet_available(ct.shape[0], ct.device)
        else:
            pet_available = pet_available.long().view(-1)

        if self.fusion_type == 'add':
            fused_feats = self.fusion(ct_feats, pet_feats, pet_available=pet_available)
        else:
            fused_feats = self.fusion(ct_feats, pet_feats)

        dec_out = self.decoder(fused_feats, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs

    @staticmethod
    def _finalize_decoder_output(dec_out):
        if isinstance(dec_out, dict):
            out = {
                'logits': _sanitize(dec_out['logits']),
            }
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}


class SingleModalityBaselineUNet(nn.Module):
    def __init__(
        self,
        backbone='mit_b1',
        pretrained_path=None,
        modality='ct',
        in_channels=3,
        out_channels=1,
        decoder_channels=(512, 256, 128, 64),
        use_deep_supervision=False,
        **kwargs,
    ):
        super().__init__()
        if modality not in ('ct', 'pet'):
            raise ValueError(f'Unsupported modality: {modality}. Use ct or pet.')
        self.modality = modality
        self.encoder = create_feature_backbone(backbone, in_channels=in_channels)
        load_local_weights_safe(self.encoder, pretrained_path, name=f'{modality.upper()}_Encoder')
        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = UNetStyleDecoder(
            encoder_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            use_deep_supervision=use_deep_supervision,
        )

    @staticmethod
    def _to_3ch(x):
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(self, ct, pet, pet_available=None, target_size=None):
        x = ct if self.modality == 'ct' else pet
        if target_size is None:
            target_size = x.shape[-2:]
        x = self._to_3ch(x)
        feats = self.encoder(x)
        _check_tensor_list(f'{self.modality}_feats', feats)
        dec_out = self.decoder(feats, target_size)
        outputs = PETCTBaselineUNet._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs
