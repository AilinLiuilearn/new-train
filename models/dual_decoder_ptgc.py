import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline_petct_unet import UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import create_feature_backbone, load_local_weights_safe
from models.dual_decoder_add_baseline import StageChannelAlign
from models.ptgc_module import PatchTaskGainCompensator


class DualDecoderPTGC(nn.Module):
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
        ptgc_ablation_mode='ptgc_gpnd',
        ptgc_alpha=0.25,
        ptgc_loss_weight=0.2,
        gpnd_rank_weight=1.0,
        gpnd_support_weight=0.05,
        ptgc_delta_active_threshold=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.ptgc_ablation_mode = str(ptgc_ablation_mode)
        self.ptgc_alpha = float(ptgc_alpha)
        self.ptgc_loss_weight = float(ptgc_loss_weight)
        self.gpnd_rank_weight = float(gpnd_rank_weight)
        self.gpnd_support_weight = float(gpnd_support_weight)
        self.ptgc_delta_active_threshold = float(ptgc_delta_active_threshold)

        self.enc_ct = create_feature_backbone(ct_backbone, in_channels=in_channels)
        self.enc_pet = create_feature_backbone(pet_backbone, in_channels=in_channels)
        load_local_weights_safe(self.enc_ct, ct_pretrained_path, name='CT_Encoder')
        load_local_weights_safe(self.enc_pet, pet_pretrained_path, name='PET_Encoder')
        ct_channels = list(self.enc_ct.feature_info.channels())
        pet_channels = list(self.enc_pet.feature_info.channels())
        self.ct_align = StageChannelAlign(ct_channels, pet_channels)
        self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=False)
        self.missing_decoder = copy.deepcopy(self.full_decoder)
        self.ptgc = PatchTaskGainCompensator(task_channels=pet_channels[-1])
        self._configure_ablation_mode()

    def _configure_ablation_mode(self):
        if self.ptgc_ablation_mode == 'baseline':
            for p in self.ptgc.parameters():
                p.requires_grad = False
        elif self.ptgc_ablation_mode in ('ptgc', 'ptgc_gpnd'):
            for p in self.ptgc.parameters():
                p.requires_grad = True
        else:
            raise ValueError('ptgc_ablation_mode must be one of baseline, ptgc, ptgc_gpnd')

    @staticmethod
    def _to_3ch(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _encode_ct(self, ct):
        ct = self._to_3ch(ct)
        ct_feats_raw = self.enc_ct(ct)
        _check_tensor_list('ct_feats_raw', ct_feats_raw)
        return self.ct_align(ct_feats_raw)

    def _encode_pet(self, pet):
        pet = self._to_3ch(pet)
        pet_feats = self.enc_pet(pet)
        _check_tensor_list('pet_feats', pet_feats)
        return pet_feats

    def _decode_with_states(self, decoder, features, target_size):
        x1, x2, x3, x4 = features
        d4 = decoder.proj4(x4)
        s3 = decoder.proj3(x3)
        d3 = decoder.fuse3(torch.cat([F.interpolate(d4, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = decoder.proj2(x2)
        d2 = decoder.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = decoder.proj1(x1)
        d1 = decoder.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        logits = decoder.seg_head(d1)
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': _sanitize(logits)}, [d1, d2, d3, d4]

    def _decode_missing_with_delta(self, ct_feats, d4_base, delta_d4, target_size):
        x1, x2, x3, x4 = ct_feats
        d4_comp = d4_base + delta_d4
        s3 = self.missing_decoder.proj3(x3)
        d3 = self.missing_decoder.fuse3(torch.cat([F.interpolate(d4_comp, size=s3.shape[-2:], mode='bilinear', align_corners=False), s3], dim=1))
        s2 = self.missing_decoder.proj2(x2)
        d2 = self.missing_decoder.fuse2(torch.cat([F.interpolate(d3, size=s2.shape[-2:], mode='bilinear', align_corners=False), s2], dim=1))
        s1 = self.missing_decoder.proj1(x1)
        d1 = self.missing_decoder.fuse1(torch.cat([F.interpolate(d2, size=s1.shape[-2:], mode='bilinear', align_corners=False), s1], dim=1))
        comp_logits = self.missing_decoder.seg_head(d1)
        comp_logits = F.interpolate(comp_logits, size=target_size, mode='bilinear', align_corners=False)
        return {'logits': _sanitize(comp_logits)}, [d1, d2, d3, d4_comp]

    def _forward_train_joint(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        ct_out, ct_states = self._decode_with_states(self.missing_decoder, ct_feats, target_size)
        full_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
        full_out, full_states = self._decode_with_states(self.full_decoder, full_feats, target_size)

        if self.ptgc_ablation_mode == 'baseline':
            return {
                'logits': ct_out['logits'],
                'ptgc_ct_logits': ct_out['logits'],
                'ptgc_full_logits': full_out['logits'],
                'ptgc_ablation_mode': 'baseline',
            }

        ptgc_out = self.ptgc(ct_states[-1], ct_out['logits'])
        comp_out, _ = self._decode_missing_with_delta(ct_feats, ct_states[-1], ptgc_out['delta_d4'], target_size)
        return {
            'logits': comp_out['logits'],
            'ptgc_ct_logits': ct_out['logits'],
            'ptgc_full_logits': full_out['logits'],
            'ptgc_comp_logits': comp_out['logits'],
            'ptgc_gain_pred': ptgc_out['gain_pred_signed'],
            'ptgc_benefit_pred': ptgc_out['benefit_pred'],
            'ptgc_delta_d4': ptgc_out['delta_d4'],
            'ptgc_entropy_patch': ptgc_out['entropy_patch'],
            'ptgc_ablation_mode': self.ptgc_ablation_mode,
        }

    def _forward_full_eval(self, ct, pet, target_size):
        ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        full_feats = [c + p for c, p in zip(ct_feats, pet_feats)]
        full_out, _ = self._decode_with_states(self.full_decoder, full_feats, target_size)
        return {'logits': full_out['logits']}

    def _forward_missing(self, ct, target_size):
        ct_feats = self._encode_ct(ct)
        ct_out, ct_states = self._decode_with_states(self.missing_decoder, ct_feats, target_size)
        if self.ptgc_ablation_mode == 'baseline':
            return {'logits': ct_out['logits'], 'ptgc_ct_logits': ct_out['logits'], 'ptgc_ablation_mode': 'baseline'}
        ptgc_out = self.ptgc(ct_states[-1], ct_out['logits'])
        comp_out, _ = self._decode_missing_with_delta(ct_feats, ct_states[-1], ptgc_out['delta_d4'], target_size)
        return {'logits': comp_out['logits'], 'ptgc_ct_logits': ct_out['logits'], 'ptgc_gain_pred': ptgc_out['gain_pred_signed'], 'ptgc_benefit_pred': ptgc_out['benefit_pred'], 'ptgc_delta_d4': ptgc_out['delta_d4'], 'ptgc_ablation_mode': self.ptgc_ablation_mode}

    def forward(self, ct, pet, pet_available=None, target_size=None, forward_mode='auto'):
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == 'full':
            if pet is None:
                raise ValueError('forward_mode="full" requires pet input.')
            return self._forward_full_eval(ct, pet, target_size)
        if forward_mode == 'missing':
            return self._forward_missing(ct, target_size)
        if forward_mode == 'auto':
            if pet is None:
                raise ValueError('forward_mode="auto" requires pet input.')
            return self._forward_train_joint(ct, pet, target_size)
        raise ValueError(f'Unsupported forward_mode={forward_mode!r}')
