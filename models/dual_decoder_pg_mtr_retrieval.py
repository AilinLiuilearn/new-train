import copy

import torch
import torch.nn as nn

from models.baseline_petct_unet import AddFusion, UNetStyleDecoder, _check_tensor, _check_tensor_list, _sanitize
from models.build_mdt_seg import ConvBNAct, create_feature_backbone, load_local_weights_safe
from models.pg_mtr import PETGroundedMetabolicTokenRetrieval


class StageChannelAlign(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.proj = nn.ModuleList([
            ConvBNAct(c_in, c_out, kernel_size=1) for c_in, c_out in zip(in_channels_list, out_channels_list)
        ])

    def forward(self, feats):
        return [proj(feat) for proj, feat in zip(self.proj, feats)]


class DualDecoderPGMTRRetrieval(nn.Module):
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
        pg_mtr_stages='all',
        pg_mtr_num_tokens=8,
        pg_mtr_temperature=0.07,
        pg_mtr_detach_bank_missing=True,
    ):
        super().__init__()
        self.use_deep_supervision = bool(use_deep_supervision)
        self.pg_mtr_detach_bank_missing = bool(pg_mtr_detach_bank_missing)

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
        self.full_decoder = UNetStyleDecoder(pet_channels, decoder_channels=decoder_channels, out_channels=out_channels, use_deep_supervision=self.use_deep_supervision)
        self.missing_decoder = copy.deepcopy(self.full_decoder)
        self.pg_mtr = PETGroundedMetabolicTokenRetrieval(
            channels_list=pet_channels,
            num_tokens=pg_mtr_num_tokens,
            temperature=pg_mtr_temperature,
            stage_mode=pg_mtr_stages,
        )
        self.retrieval_projs = nn.ModuleDict()
        for stage_number in self.pg_mtr.active_stage_numbers:
            stage_channels = pet_channels[stage_number - 1]
            latent_dim = self.pg_mtr.stage_modules[str(stage_number)].latent_dim
            proj = nn.Conv2d(latent_dim, stage_channels, kernel_size=1, bias=False)
            nn.init.zeros_(proj.weight)
            self.retrieval_projs[str(stage_number)] = proj

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

    def _finalize_decoder_output(self, dec_out):
        if isinstance(dec_out, dict):
            out = {'logits': _sanitize(dec_out['logits'])}
            if 'aux_logits' in dec_out:
                out['aux_logits'] = [_sanitize(x) for x in dec_out['aux_logits']]
            return out
        return {'logits': _sanitize(dec_out)}

    def _decode_with(self, decoder, features, target_size):
        dec_out = decoder(features, target_size)
        outputs = self._finalize_decoder_output(dec_out)
        _check_tensor('logits', outputs['logits'])
        outputs['pred'] = outputs['logits']
        outputs['aux'] = {}
        return outputs

    def _forward_full(self, ct, pet, target_size):
        aligned_ct_feats = self._encode_ct(ct)
        pet_feats = self._encode_pet(pet)
        full_feats = self.fusion(aligned_ct_feats, pet_feats, pet_available=None)
        outputs = self._decode_with(self.full_decoder, full_feats, target_size)
        _, pg_aux, pg_diag = self.pg_mtr(aligned_ct_feats=aligned_ct_feats, pet_feats=pet_feats, mode='full')
        outputs['aux_losses'] = pg_aux
        outputs['diagnostics'] = pg_diag
        return outputs

    def _forward_missing(self, ct, target_size):
        aligned_ct_feats = self._encode_ct(ct)
        retrieved_memories, _, pg_diag = self.pg_mtr(
            aligned_ct_feats=aligned_ct_feats,
            pet_feats=None,
            mode='missing',
            detach_bank=self.pg_mtr_detach_bank_missing,
        )
        missing_feats = list(aligned_ct_feats)
        for stage_number in self.pg_mtr.active_stage_numbers:
            stage_index = stage_number - 1
            retrieved_feature = self.retrieval_projs[str(stage_number)](retrieved_memories[stage_number])
            missing_feats[stage_index] = aligned_ct_feats[stage_index] + retrieved_feature
        outputs = self._decode_with(self.missing_decoder, missing_feats, target_size)
        outputs['aux_losses'] = {}
        outputs['diagnostics'] = pg_diag
        return outputs

    @staticmethod
    def _merge_outputs(full_outputs, missing_outputs, full_idx, missing_idx, batch_size):
        merged = {}
        logits = full_outputs['logits'].new_zeros((batch_size,) + tuple(full_outputs['logits'].shape[1:]))
        if full_idx.numel() > 0:
            logits = logits.index_copy(0, full_idx, full_outputs['logits'])
        if missing_idx.numel() > 0:
            logits = logits.index_copy(0, missing_idx, missing_outputs['logits'])
        merged['logits'] = logits
        merged['pred'] = logits
        if 'aux_logits' in full_outputs or 'aux_logits' in missing_outputs:
            aux_logits = []
            full_aux = full_outputs.get('aux_logits', [])
            missing_aux = missing_outputs.get('aux_logits', [])
            aux_len = max(len(full_aux), len(missing_aux))
            for idx in range(aux_len):
                template = full_aux[idx] if idx < len(full_aux) else missing_aux[idx]
                aux = template.new_zeros((batch_size,) + tuple(template.shape[1:]))
                if idx < len(full_aux) and full_idx.numel() > 0:
                    aux = aux.index_copy(0, full_idx, full_aux[idx])
                if idx < len(missing_aux) and missing_idx.numel() > 0:
                    aux = aux.index_copy(0, missing_idx, missing_aux[idx])
                aux_logits.append(aux)
            merged['aux_logits'] = aux_logits
        merged['aux'] = {}
        _check_tensor('merged_logits', merged['logits'])
        return merged

    def _forward_auto(self, ct, pet, pet_available, target_size):
        pet_available = pet_available.long().view(-1)
        if torch.all(pet_available == 1):
            return self._forward_full(ct, pet, target_size)
        if torch.all(pet_available == 0):
            return self._forward_missing(ct, target_size)
        aligned_ct_feats = self._encode_ct(ct)
        full_idx = torch.nonzero(pet_available == 1, as_tuple=False).flatten()
        missing_idx = torch.nonzero(pet_available == 0, as_tuple=False).flatten()
        full_outputs = None
        missing_outputs = None
        if full_idx.numel() > 0:
            full_ct_feats = [feat.index_select(0, full_idx) for feat in aligned_ct_feats]
            pet_full = pet.index_select(0, full_idx)
            pet_feats = self._encode_pet(pet_full)
            fused_full_feats = self.fusion(full_ct_feats, pet_feats, pet_available=None)
            full_outputs = self._decode_with(self.full_decoder, fused_full_feats, target_size)
        if missing_idx.numel() > 0:
            missing_ct_feats = [feat.index_select(0, missing_idx) for feat in aligned_ct_feats]
            retrieved_memories, _, pg_diag = self.pg_mtr(
                aligned_ct_feats=missing_ct_feats,
                pet_feats=None,
                mode='missing',
                detach_bank=self.pg_mtr_detach_bank_missing,
            )
            routed_feats = [feat.index_select(0, missing_idx) for feat in aligned_ct_feats]
            for stage_number in self.pg_mtr.active_stage_numbers:
                stage_index = stage_number - 1
                routed_feats[stage_index] = routed_feats[stage_index] + self.retrieval_projs[str(stage_number)](retrieved_memories[stage_number])
            missing_outputs = self._decode_with(self.missing_decoder, routed_feats, target_size)
            missing_outputs['diagnostics'] = pg_diag
        if full_outputs is None:
            return missing_outputs
        if missing_outputs is None:
            return full_outputs
        return self._merge_outputs(full_outputs, missing_outputs, full_idx, missing_idx, ct.shape[0])

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
